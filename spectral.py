"""
spectral.py -- Band-constrained log-retention gating.

Core operator of the Spectral Memory Filter Bank (SMFB).

A classical LSTM forget gate is f_t = sigmoid(a_t) in (0, 1), with an
*effective timescale* tau_eff = -1 / ln f.  Two facts make that
parameterisation a poor basis for multi-timescale memory:

  1. tau_eff is unbounded above (f -> 1 gives tau -> inf), so a memory
     stream can silently drift into a pure integrator.
  2. Nothing prevents two nominally independent streams from settling on
     the *same* tau.  See Proposition 2 in theory.py: the symmetric
     readout makes the collapsed configuration a critical point.

The band-constrained log-retention gate fixes both by construction.  For
band k with edges [tau_lo_k, tau_hi_k]:

    tau_t^(k) = tau_lo_k * (tau_hi_k / tau_lo_k) ** sigmoid(a_t^(k))
    f_t^(k)   = exp(-1 / tau_t^(k))

Because sigmoid(.) lies in (0, 1), tau_t^(k) lies in (tau_lo_k, tau_hi_k)
for every input, every timestep and every parameter value.  With
geometrically tiled, non-overlapping edges the bands are therefore
*certified disjoint*: band k can never represent the timescale of band
k+1.  Collapse is not discouraged, it is impossible.

Implementation note
-------------------
The band axis is vectorised.  Input projections for all T timesteps are
computed in a single matmul before the recurrence, and the per-band
recurrent projections are one batched einsum rather than K separate
Linear calls.  A loop-per-band implementation is roughly 40x slower
because it is kernel-launch bound, not FLOP bound.
"""

import math

import torch
import torch.nn as nn


def geometric_band_edges(num_bands, tau_min=1.0, tau_max=None, seq_len=None):
    """Geometrically spaced band edges tiling [tau_min, tau_max].

    Returns a tensor of shape (num_bands + 1,).  Band k occupies
    [edges[k], edges[k+1]].  Geometric rather than linear spacing is the
    right choice because timescale error is naturally measured in log
    space -- see Proposition 1.
    """
    if tau_max is None:
        if seq_len is None:
            raise ValueError("provide tau_max or seq_len")
        # Cover up to 2x the sequence length: a band with tau > 2T acts as
        # a pure integrator over the observable window, so there is
        # nothing left to gain by tiling beyond it.
        tau_max = 2.0 * seq_len
    log_lo, log_hi = math.log(tau_min), math.log(tau_max)
    return torch.exp(torch.linspace(log_lo, log_hi, num_bands + 1))


def tau_to_forget(tau):
    """f = exp(-1/tau).  Retention implied by an effective timescale."""
    return torch.exp(-1.0 / tau)


def forget_to_tau(f, eps=1e-6):
    """tau_eff = -1 / ln f.  Inverse of tau_to_forget."""
    f = f.clamp(eps, 1.0 - eps)
    return -1.0 / torch.log(f)


class BandGate(nn.Module):
    """Vectorised band-constrained log-retention gate for all K bands.

    Given pre-activations of shape (B, K, d), returns retentions f and
    timescales tau of the same shape, with tau[:, k, :] confined to
    (edges[k], edges[k+1]).

    Notes
    -----
    d ln(tau) / d a = ln(kappa_k) * sigmoid_prime(a), with kappa_k =
    tau_hi_k / tau_lo_k.  The gradient in *log-timescale* space is
    bounded by ln(kappa_k)/4 and never vanishes for finite a, so the gate
    moves at a uniform relative rate anywhere inside its band.
    """

    def __init__(self, edges):
        super().__init__()
        edges = torch.as_tensor(edges, dtype=torch.float32)
        assert torch.all(edges[1:] > edges[:-1]), "band edges must increase"
        self.register_buffer("tau_lo", edges[:-1].clone())
        self.register_buffer("log_kappa", torch.log(edges[1:] / edges[:-1]))

    def forward(self, a):
        # a: (B, K, d)
        lo = self.tau_lo.view(1, -1, 1).to(a.dtype)
        lk = self.log_kappa.view(1, -1, 1).to(a.dtype)
        tau = lo * torch.exp(lk * torch.sigmoid(a))
        return tau_to_forget(tau), tau

    def extra_repr(self):
        hi = self.tau_lo * torch.exp(self.log_kappa)
        return "bands=" + ", ".join("[%.2f,%.2f]" % (float(a), float(b))
                                    for a, b in zip(self.tau_lo, hi))


class SpectralMemoryFilterBank(nn.Module):
    """K memory streams, each pinned to a disjoint timescale band.

    Each band k runs its own leaky-memory recurrence

        m_t^(k) = f_t^(k) * m_{t-1}^(k) + (1 - f_t^(k)) * i_t^(k) * g_t^(k)

    with f_t^(k) produced by a BandGate, so band k's retention is
    confined to [edges[k], edges[k+1]].  Unlike the referenced dual-memory
    design, every band gets its *own* input projection: sharing W_x
    across streams is exactly the condition under which Proposition 2
    predicts collapse.

    Bands are fused by spectral attention -- a per-timestep softmax over
    bands, which generalises the scalar ASFG gate alpha_t (recovered
    exactly at K = 2).
    """

    def __init__(self, input_size, hidden_size, num_bands=4, seq_len=None,
                 tau_min=1.0, tau_max=None, share_input_proj=False,
                 use_cross_term=True, cross_beta=0.5, free_gate=False,
                 tied_gate=False, fusion="softmax", spread_init=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_bands = num_bands
        self.share_input_proj = share_input_proj
        self.use_cross_term = use_cross_term and num_bands > 1
        self.cross_beta = cross_beta
        # free_gate=True replaces the band gate with a plain sigmoid, i.e.
        # removes the disjointness certificate while keeping K streams.
        # Used by the ablation to isolate what the certificate is worth.
        self.free_gate = free_gate
        # tied_gate=True forces every hidden unit in a band to share ONE
        # retention, which is the setting Proposition 1 actually describes.
        # Without it, K=1 is not a single-timescale model at all: each of
        # the d units carries its own gate and can select any timescale in
        # the range independently, so a one-"band" layer already has d
        # timescales.  This control exists to keep that distinction honest.
        self.tied_gate = tied_gate
        # How band states are combined into the fused activation.
        #   "softmax" -- per-unit convex combination over bands.  This is the
        #       direct generalisation of the ASFG gate alpha_t, and it is
        #       LOSSY: a convex combination can interpolate between bands but
        #       cannot expose two of them at once, so a task needing two
        #       timescales simultaneously is unsolvable through it.
        #   "concat" -- keep every band and project.  Preserves all bands.
        #   "gated"  -- per-band sigmoid weights that need not sum to one.
        self.fusion = fusion

        d, K = hidden_size, num_bands
        edges = geometric_band_edges(num_bands, tau_min, tau_max, seq_len)
        self.register_buffer("band_edges", edges)
        self.gate = BandGate(edges)

        # --- input pathway (f, i, g for every band), one matmul ---------
        # share_input_proj=True reproduces the collapse-prone configuration
        # of the referenced STLAT model.
        n_proj = 1 if share_input_proj else K
        self.W_x = nn.Linear(input_size, 3 * n_proj * d)
        self.n_proj = n_proj

        # --- recurrent pathway, per band, one batched einsum ------------
        self.W_m = nn.Parameter(torch.empty(K, d, 3 * d))
        nn.init.xavier_uniform_(self.W_m)

        # --- spectral attention over bands ------------------------------
        self.attn_x = nn.Linear(input_size, K * d)
        self.attn_m = nn.Parameter(torch.empty(K, d, K * d))
        nn.init.xavier_uniform_(self.attn_m)
        if fusion == "concat":
            self.mix_proj = nn.Linear(K * d, d)

        # --- cross-band interaction (adjacent pairs only, O(K)) ---------
        if self.use_cross_term:
            self.W_cross = nn.Parameter(torch.empty(K - 1, d, d))
            nn.init.xavier_uniform_(self.W_cross)
            self.cross_bias = nn.Parameter(torch.zeros(d))
            self.W_po = nn.Parameter(torch.empty(d, d))
            nn.init.xavier_uniform_(self.W_po)

        # --- context-aware output gate ----------------------------------
        self.W_xo = nn.Linear(input_size, d)
        self.W_ho = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.W_ho)
        self.W_mo = nn.Parameter(torch.empty(K, d, d))
        nn.init.xavier_uniform_(self.W_mo)

        nn.init.zeros_(self.W_x.bias)

        # Soft timescale diversity.  Hard band tiling supplies diversity by
        # removing capacity, and empirically that costs more than it buys.
        # This instead *initialises* the forget biases so units start spread
        # log-uniformly over [tau_min, tau_max], while leaving the gate free
        # to move anywhere.  Diversity without constraint.
        self.spread_init = spread_init
        if spread_init:
            with torch.no_grad():
                lo = math.log(float(self.band_edges[0]))
                hi = math.log(float(self.band_edges[-1]))
                taus = torch.exp(torch.linspace(lo, hi, d))
                f = tau_to_forget(taus).clamp(1e-4, 1 - 1e-4)
                logit = torch.log(f / (1 - f))
                b = self.W_x.bias.view(n_proj, 3, d)
                for k in range(n_proj):
                    b[k, 0].copy_(logit)          # forget pre-activation

    def forward(self, x, return_diagnostics=False):
        """x: (B, T, input_size) -> H: (B, T, hidden_size)."""
        B, T, _ = x.shape
        d, K = self.hidden_size, self.num_bands
        dev, dt = x.device, x.dtype

        # Precompute every input-dependent projection for all timesteps.
        xp = self.W_x(x).view(B, T, self.n_proj, 3, d)
        if self.n_proj == 1:
            xp = xp.expand(B, T, K, 3, d)
        ax = self.attn_x(x)                        # (B, T, K*d)
        xo = self.W_xo(x)                          # (B, T, d)

        m = torch.zeros(B, K, d, device=dev, dtype=dt)
        h = torch.zeros(B, d, device=dev, dtype=dt)

        outputs = []
        tau_trace, attn_trace, state_trace = [], [], []

        for t in range(T):
            # Recurrent projections for all bands at once: (B,K,3d)
            mp = torch.einsum("bkd,kde->bke", m, self.W_m)
            a_f = xp[:, t, :, 0, :] + mp[:, :, 0:d]
            a_i = xp[:, t, :, 1, :] + mp[:, :, d:2 * d]
            a_g = xp[:, t, :, 2, :] + mp[:, :, 2 * d:3 * d]

            if self.tied_gate:
                # one retention per band, shared across all hidden units
                a_f = a_f.mean(dim=-1, keepdim=True).expand_as(a_f)
            if self.free_gate:
                f = torch.sigmoid(a_f)
                tau = forget_to_tau(f)
            else:
                f, tau = self.gate(a_f)

            i = torch.sigmoid(a_i)
            g = torch.tanh(a_g)
            # Write scaled by (1 - f): the leaky-integrator normalisation,
            # so a long-timescale band integrates slowly instead of
            # saturating.
            m = f * m + (1.0 - f) * i * g

            # Spectral attention: softmax over bands, per unit.
            logits = (ax[:, t, :]
                      + torch.einsum("bkd,kde->be", m, self.attn_m)).view(B, K, d)
            if self.fusion == "softmax":
                attn = torch.softmax(logits, dim=1)
                mixed = (attn * torch.tanh(m)).sum(dim=1)
            elif self.fusion == "gated":
                attn = torch.sigmoid(logits)
                mixed = (attn * torch.tanh(m)).sum(dim=1)
            else:                                    # "concat"
                attn = torch.softmax(logits, dim=1)  # kept for diagnostics only
                mixed = self.mix_proj(
                    (torch.sigmoid(logits) * torch.tanh(m)).reshape(B, K * d))

            if self.use_cross_term:
                prod = m[:, :-1, :] * m[:, 1:, :]                 # (B,K-1,d)
                phi = torch.tanh(
                    torch.einsum("bkd,kde->be", prod, self.W_cross) + self.cross_bias)
                mixed = mixed + self.cross_beta * phi

            o = xo[:, t, :] + h @ self.W_ho + torch.einsum("bkd,kde->be", m, self.W_mo)
            if self.use_cross_term:
                o = o + phi @ self.W_po
            h = torch.sigmoid(o) * mixed
            outputs.append(h)

            if return_diagnostics:
                tau_trace.append(tau)
                attn_trace.append(attn)
                state_trace.append(m)

        H = torch.stack(outputs, dim=1)
        if not return_diagnostics:
            return H, None
        diag = {
            "tau": torch.stack(tau_trace, dim=1),       # (B, T, K, d)
            "attn": torch.stack(attn_trace, dim=1),     # (B, T, K, d)
            "states": torch.stack(state_trace, dim=1),  # (B, T, K, d)
            "band_edges": self.band_edges,
        }
        return H, diag
