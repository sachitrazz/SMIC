"""
model.py -- the referenced STLAT model and the band-tiled variant.

Two classes:

  ReferencedSTLAT  -- the referenced STLAT architecture as implemented,
                    including the shared input projections that
                    Proposition 2 identifies as the source of the symmetry.
                    It follows the referenced implementation exactly, apart
                    from batching, so that the collapse measurement is a
                    measurement of that model.  wire_fix=True computes the
                    equations as written instead (the corrected STLAT row).

  SpectralSTLAT  -- the band-tiled variant, reported in the paper as a
                    negative result: a band-tiled memory bank followed by a
                    Transformer encoder and a linear head, with switches for
                    each ablation.
"""

import torch
import torch.nn as nn

from contrastive import ProjectionHead
from spectral import SpectralMemoryFilterBank, forget_to_tau


# ======================================================================
# Shared spatial front-end (the "STF block" of the architecture figure)
# ======================================================================

class ConvFrontEnd(nn.Module):
    """Small convolutional encoder shared across sequence positions.

    Feeding raw pixel strips straight into the recurrence bottlenecks the
    whole model before the memory mechanism is exercised at all: on the
    ISL split that configuration tops out near 19% validation accuracy,
    which is too low for any comparison between memory designs to mean
    anything.  This front-end restores the spatial encoder that the
    architecture description assumes.

    The sequence axis is preserved: the input (B, T, strip) is reshaped
    back into the image it tiles, encoded, and read out as T tokens along
    the same scan axis, so the temporal/scan semantics are unchanged.
    """

    def __init__(self, img_size, seq_len, channels=1, width=32, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.seq_len = seq_len
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1), nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # H/2
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # H/4
            nn.Dropout2d(dropout),
        )
        self.out_dim = 2 * width * (img_size // 4)

    def forward(self, x):
        B, T, _ = x.shape
        H = W = self.img_size
        img = x.reshape(B, H, W, self.channels).permute(0, 3, 1, 2)
        fmap = self.net(img)                                  # (B, 2w, H/4, W/4)
        # rows of the feature map -> T tokens along the scan axis
        fmap = torch.nn.functional.adaptive_avg_pool2d(
            fmap, (self.seq_len, self.img_size // 4))
        return fmap.permute(0, 2, 1, 3).reshape(B, self.seq_len, -1)


# ======================================================================
# Baseline: the referenced STLAT architecture
# ======================================================================

class ReferencedSTLAT(nn.Module):
    """Reproduction of the referenced STLAT model (SpatioTemporalModel).

    Preserved faithfully, including:
      * W_xg / W_xi / W_xf shared between the C and M streams;
      * the static readout H_t = o_t * tanh(C_t + M_t);
      * C_t recomputed from the *final* LSTM cell state at every step.

    ``use_asfg=True`` adds the ASFG readout (alpha_t, Phi_t, O_t')
    exactly as written in the paper, so its contribution can be
    measured under identical conditions.
    """

    def __init__(self, input_size, hidden_size, num_classes, num_layers=4,
                 num_heads=8, transformer_layers=4, dim_feedforward=512,
                 dropout=0.3, use_asfg=False, cross_beta=0.5,
                 front_end=None, share_input_proj=True, cascade=False,
                 wire_fix=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_asfg = use_asfg
        self.cross_beta = cross_beta

        # Optional shared spatial encoder; when present it sets the width
        # of everything downstream.
        self.front_end = front_end
        if front_end is not None:
            input_size = front_end.out_dim

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        enc = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc, num_layers=transformer_layers)

        # wire_fix=True makes the model compute the equations as written:
        # the dual-memory loop consumes the Transformer's tokens instead of
        # the raw input, C_t recurs on C_{t-1} rather than on the LSTM's
        # final cell, and H_{t-1} is the previous step's H_t.  The referenced
        # implementation does none of these; the
        # default False reproduces it exactly.
        self.wire_fix = wire_fix
        x_in = hidden_size if wire_fix else input_size

        # Input projections for the C path.
        self.W_xg = nn.Linear(x_in, hidden_size)
        self.W_xi = nn.Linear(x_in, hidden_size)
        self.W_xf = nn.Linear(x_in, hidden_size)

        # Input projections for the M path.  share_input_proj=True (the
        # referenced behaviour) aliases them onto the C path, which is the
        # collapse condition of Proposition 2; False gives the M stream its
        # own, which is the minimal intervention the proposition suggests.
        # cascade=True routes the M stream off C_t instead of X_t, which
        # removes the exchange symmetry of Proposition 2 topologically.
        self.cascade = cascade
        self.share_input_proj = share_input_proj
        if share_input_proj:
            self.W_xg_m, self.W_xi_m, self.W_xf_m = self.W_xg, self.W_xi, self.W_xf
        else:
            m_in = hidden_size if cascade else input_size
            self.W_xg_m = nn.Linear(m_in, hidden_size)
            self.W_xi_m = nn.Linear(m_in, hidden_size)
            self.W_xf_m = nn.Linear(m_in, hidden_size)

        self.W_hg = nn.Linear(hidden_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)

        self.W_mg = nn.Linear(hidden_size, hidden_size)
        self.W_mi = nn.Linear(hidden_size, hidden_size)
        self.W_mf = nn.Linear(hidden_size, hidden_size)

        self.W_xo = nn.Linear(x_in, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)
        self.W_co = nn.Linear(hidden_size, hidden_size)
        self.W_mo = nn.Linear(hidden_size, hidden_size)

        if use_asfg:
            self.W_alpha = nn.Linear(x_in, hidden_size)
            self.U_alpha = nn.Linear(hidden_size, hidden_size, bias=False)
            self.V_alpha = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_c = nn.Linear(hidden_size, hidden_size)
            self.W_m = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_cm = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_phi_o = nn.Linear(hidden_size, hidden_size, bias=False)

        self.fc = nn.Linear(hidden_size, num_classes)
        self.proj = ProjectionHead(hidden_size, out_dim=64)

    def forward(self, x, return_diagnostics=False):
        if self.front_end is not None:
            x = self.front_end(x)
        B, T, _ = x.shape
        dev, dt = x.device, x.dtype

        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=dev, dtype=dt)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=dev, dtype=dt)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.transformer_encoder(out)

        M_prev = torch.zeros(B, self.hidden_size, device=dev, dtype=dt)
        H_prev = hn[-1]
        C_prev = cn[-1]
        H_t = H_prev
        C_trace, M_trace, f_trace, fp_trace = [], [], [], []

        for t in range(T):
            # Reference behaviour: raw input, and C/H read the LSTM's FINAL
            # states at every step (so neither recurs).  Corrected: tokens
            # from the Transformer, and genuine recurrence.
            X_t = out[:, t, :] if self.wire_fix else x[:, t, :]

            g_t = torch.tanh(self.W_xg(X_t) + self.W_hg(H_prev))
            i_t = torch.sigmoid(self.W_xi(X_t) + self.W_hi(H_prev))
            f_t = torch.sigmoid(self.W_xf(X_t) + self.W_hf(H_prev))
            C_t = f_t * (C_prev if self.wire_fix else cn[-1]) + i_t * g_t

            # Aliased to the C path when share_input_proj=True, matching the
            # referenced implementation exactly.  Under cascade=True the M
            # stream is driven by C_t rather than X_t, so the two streams
            # occupy different positions in the graph and cannot be
            # exchanged (Proposition 2 has no fixed point).
            drive = C_t if self.cascade else X_t
            g_p = torch.tanh(self.W_xg_m(drive) + self.W_mg(M_prev))
            i_p = torch.sigmoid(self.W_xi_m(drive) + self.W_mi(M_prev))
            f_p = torch.sigmoid(self.W_xf_m(drive) + self.W_mf(M_prev))
            M_t = f_p * M_prev + i_p * g_p

            if self.use_asfg:
                alpha = torch.sigmoid(self.W_alpha(X_t) + self.U_alpha(C_t) + self.V_alpha(M_t))
                phi = torch.tanh(self.W_c(C_t) + self.W_m(M_t) + self.W_cm(C_t * M_t))
                o_t = torch.sigmoid(self.W_xo(X_t) + self.W_ho(H_prev)
                                    + self.W_co(C_t) + self.W_mo(M_t)
                                    + self.W_phi_o(phi))
                H_t = o_t * (alpha * torch.tanh(C_t)
                             + (1 - alpha) * torch.tanh(M_t)
                             + self.cross_beta * phi)
            else:
                o_t = torch.sigmoid(self.W_xo(X_t) + self.W_ho(H_prev)
                                    + self.W_co(C_t) + self.W_mo(M_t))
                H_t = o_t * torch.tanh(C_t + M_t)

            M_prev = M_t
            if self.wire_fix:
                C_prev, H_prev = C_t, H_t
            if return_diagnostics:
                C_trace.append(C_t)
                M_trace.append(M_t)
                f_trace.append(f_t)
                fp_trace.append(f_p)

        logits = self.fc(H_t)
        if not return_diagnostics:
            return logits, {"feat": H_t}

        states = torch.stack([torch.stack(C_trace, 1), torch.stack(M_trace, 1)], dim=2)
        fs = torch.stack([torch.stack(f_trace, 1), torch.stack(fp_trace, 1)], dim=2)
        return logits, {"states": states, "tau": forget_to_tau(fs), "attn": None,
                        "band_edges": None, "feat": H_t}


# ======================================================================
# Proposed: STLAT-S
# ======================================================================

class SpectralSTLAT(nn.Module):
    """Band-tiled memory bank + Transformer encoder + linear head.

    Ablation switches map one-to-one onto the paper's ablation table:

      num_bands        K in the filter bank (K = 1 removes multi-timescale
                       memory entirely, K = 2 is the dual-memory case)
      free_gate        replace band gates with plain sigmoids -> removes
                       the disjointness certificate, keeps K streams
      share_input_proj reinstate the referenced shared W_x -> reproduces the
                       collapse condition of Proposition 2
      use_cross_term   the Phi_t generalisation
      use_transformer  the Transformer encoder
      lambda_dec       weight of the decorrelation penalty
    """

    def __init__(self, input_size, hidden_size, num_classes, seq_len,
                 num_bands=4, num_heads=8, transformer_layers=4,
                 dim_feedforward=512, dropout=0.3, tau_min=1.0, tau_max=None,
                 free_gate=False, share_input_proj=False, use_cross_term=True,
                 use_transformer=True, cross_beta=0.5, pool="last",
                 front_end=None, tied_gate=False, fusion="softmax",
                 spread_init=False, proj_dim=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bands = num_bands
        self.use_transformer = use_transformer
        self.pool = pool

        self.front_end = front_end
        if front_end is not None:
            input_size = front_end.out_dim

        self.bank = SpectralMemoryFilterBank(
            input_size=input_size, hidden_size=hidden_size, num_bands=num_bands,
            seq_len=seq_len, tau_min=tau_min, tau_max=tau_max,
            share_input_proj=share_input_proj, use_cross_term=use_cross_term,
            cross_beta=cross_beta, free_gate=free_gate, tied_gate=tied_gate,
            fusion=fusion, spread_init=spread_init,
        )

        if use_transformer:
            enc = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads,
                dim_feedforward=dim_feedforward, dropout=dropout,
                batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(
                enc, num_layers=transformer_layers)

        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        # Projection head for the contrastive term.  Applied to the pooled
        # feature, not the logits, so the encoder keeps information the
        # contrastive objective would otherwise discard.
        self.proj = ProjectionHead(hidden_size, out_dim=proj_dim)

    def forward(self, x, return_diagnostics=False):
        if self.front_end is not None:
            x = self.front_end(x)
        H, diag = self.bank(x, return_diagnostics=return_diagnostics)
        if self.use_transformer:
            H = self.transformer_encoder(H)
        H = self.norm(H)
        feat = H.mean(dim=1) if self.pool == "mean" else H[:, -1, :]
        logits = self.fc(self.drop(feat))
        if diag is None:
            diag = {}
        diag["feat"] = feat
        return logits, diag


# ======================================================================
# Helpers
# ======================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build(name, input_size, num_classes, seq_len, hidden_size=128, **kw):
    """Factory used by train.py and ablation.py.

    Names correspond directly to rows of the ablation table.
    """
    heads = kw.pop("num_heads", 8)
    tl = kw.pop("transformer_layers", 4)
    ff = kw.pop("dim_feedforward", 512)
    do = kw.pop("dropout", 0.3)
    K = kw.pop("num_bands", 4)

    # Shared spatial encoder, built once and given to whichever model is
    # constructed, so every configuration in an ablation sees an identical
    # front-end and only the memory design differs.
    fe_cfg = kw.pop("front_end", None)
    front_end = None
    if fe_cfg:
        front_end = ConvFrontEnd(img_size=fe_cfg["img_size"], seq_len=seq_len,
                                 channels=fe_cfg.get("channels", 1),
                                 width=fe_cfg.get("width", 32),
                                 dropout=fe_cfg.get("dropout", 0.1))

    common = dict(input_size=input_size, hidden_size=hidden_size,
                  num_classes=num_classes, num_heads=heads,
                  transformer_layers=tl, dim_feedforward=ff, dropout=do)

    if name == "original":
        return ReferencedSTLAT(use_asfg=False, front_end=front_end, **common)
    if name == "original_asfg":
        return ReferencedSTLAT(use_asfg=True, front_end=front_end, **common)
    if name == "original_unshared":
        return ReferencedSTLAT(use_asfg=False, front_end=front_end,
                             share_input_proj=False, **common)
    if name == "original_asfg_unshared":
        return ReferencedSTLAT(use_asfg=True, front_end=front_end,
                             share_input_proj=False, **common)
    if name == "original_cascade":
        return ReferencedSTLAT(use_asfg=False, front_end=front_end,
                             share_input_proj=False, cascade=True, **common)
    if name == "original_asfg_cascade":
        return ReferencedSTLAT(use_asfg=True, front_end=front_end,
                             share_input_proj=False, cascade=True, **common)
    if name == "original_asfg_fixed":
        # the referenced design computed as its equations state
        return ReferencedSTLAT(use_asfg=True, front_end=front_end,
                             wire_fix=True, **common)

    # Ablation flags keyed by model name.  K is overridden by a trailing
    # _K<n> so the band sweep and the component ablation share one factory.
    flags = {
        "spectral": {},
        "spectral_free_gate": {"free_gate": True},
        "spectral_shared_proj": {"share_input_proj": True},
        "spectral_no_cross": {"use_cross_term": False},
        "spectral_no_transformer": {"use_transformer": False},
        # Proposition 1's actual setting: one shared retention for the whole
        # layer.  Distinct from spectral_K1, which still has d per-unit gates.
        "spectral_tied": {"tied_gate": True},
        "spectral_concat": {"fusion": "concat"},
        "spectral_spread": {"free_gate": True, "spread_init": True},
        "spectral_gated": {"fusion": "gated"},
    }
    if name.startswith("spectral_K"):
        tail = name[len("spectral_K"):]
        extra = {}
        for suffix, flag in (("_tied", {"tied_gate": True}),
                             ("_gated", {"fusion": "gated"}),
                             ("_spread", {"free_gate": True, "spread_init": True}),
                             ("_free", {"free_gate": True}),
                             ("_concat", {"fusion": "concat"})):
            if tail.endswith(suffix):
                tail, extra = tail[:-len(suffix)], flag
                break
        K = int(tail)
    elif name in flags:
        extra = flags[name]
    else:
        raise ValueError("unknown model name: %s" % name)

    return SpectralSTLAT(seq_len=seq_len, num_bands=K, front_end=front_end,
                         **common, **extra, **kw)
