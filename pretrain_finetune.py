"""
pretrain_finetune.py -- the encoder, augmentation and contrastive loss shared
by the benchmark, plus a pretraining experiment on unlabelled CSL frames.

bench.py uses HandEncoder, gpu_aug and seed_all from here, and nt_xent through
pretrain_source_control.py, so these define the SMIC trunk and its
pretraining objective.

Run as a script, it tests whether contrastive pretraining on 10,304 unlabelled
CSL hand crops helps the small labelled ISL-IEEE and ASL-IEEE sets (an
additional experiment; the paper's SMIC pretrains on each corpus's own
training split, see bench.py):

  Stage 1 (unlabelled): SimCLR-style contrastive pretraining.  Two augmented
    views of the same crop are positives; other crops in the batch are
    negatives.  No labels are used.
  Stage 2 (labelled): fine-tune on the labelled split.
  Baseline: identical encoder and fine-tuning from random initialisation.

The comparison is paired over seeds, splits and schedule.  Run it on the
group-disjoint splits (isl_grouped, asl_grouped).  The final-epoch accuracy
is the primary figure; the best-epoch value is stored for reference only,
because it is selected on the validation set.

    python pretrain_finetune.py isl_grouped
    python pretrain_finetune.py asl_grouped
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

PREP = os.environ.get("SMIC_PREPARED", os.path.join("..", "prepared"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    torch.set_num_threads(int(os.environ.get("SMIC_THREADS", min(14, os.cpu_count() or 1))))

# Inputs are 64 px and the trunk is narrow so that every experiment runs on a
# single laptop GPU, or on a CPU.  All models share these settings, so the
# comparisons are unaffected; absolute accuracies would rise with more compute.
RES = 64
WIDTH = 24
SEEDS = [42, 123, 456, 789, 1024]


def seed_all(s):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    # Off by default, so the released code reproduces the paper's numbers
    # exactly. Set SMIC_DETERMINISTIC=1 for bit-identical GPU runs; this is
    # slower, and cuDNN may then pick different kernels than the published runs.
    if os.environ.get("SMIC_DETERMINISTIC") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# encoder
# ----------------------------------------------------------------------

class HandEncoder(nn.Module):
    """Small conv trunk. Identical in both arms so the comparison is fair."""

    def __init__(self, width=32, out_dim=256):
        super().__init__()
        w = width
        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                nn.MaxPool2d(2))
        self.net = nn.Sequential(blk(3, w), blk(w, 2 * w), blk(2 * w, 4 * w),
                                 blk(4 * w, 4 * w), nn.AdaptiveAvgPool2d(1))
        self.out_dim = 4 * w
        self.proj = nn.Sequential(nn.Linear(self.out_dim, 256), nn.ReLU(True),
                                  nn.Linear(256, out_dim))

    def forward(self, x, project=False):
        h = self.net(x).flatten(1)
        return F.normalize(self.proj(h), dim=-1) if project else h


def gpu_aug(x, strong=True):
    """Batched augmentation: affine jitter, plus photometric when strong."""
    B = x.shape[0]
    ang = (torch.rand(B, device=x.device) * 2 - 1) * (0.42 if strong else 0.20)
    sc = 1 + (torch.rand(B, device=x.device) * 2 - 1) * (0.28 if strong else 0.14)
    tx = (torch.rand(B, device=x.device) * 2 - 1) * (0.22 if strong else 0.11)
    ty = (torch.rand(B, device=x.device) * 2 - 1) * (0.22 if strong else 0.11)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    th = torch.zeros(B, 2, 3, device=x.device)
    th[:, 0, 0], th[:, 0, 1], th[:, 0, 2] = cos, -sin, tx
    th[:, 1, 0], th[:, 1, 1], th[:, 1, 2] = sin, cos, ty
    g = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, g, padding_mode="border", align_corners=False)
    if strong:
        b = 1 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * 0.35
        x = (x * b).clamp(0, 1)
        gray = x.mean(1, keepdim=True)
        m = (torch.rand(B, 1, 1, 1, device=x.device) < 0.25).float()
        x = m * gray.expand_as(x) + (1 - m) * x
    return x


def nt_xent(z1, z2, t=0.2):
    """SimCLR loss over a batch of positive pairs."""
    z = torch.cat([z1, z2], 0)
    N = z1.shape[0]
    sim = (z @ z.t()) / t
    sim.fill_diagonal_(-1e9)
    tgt = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, tgt)


# ----------------------------------------------------------------------

def pretrain(epochs=20, bs=128, lr=1e-3, seed=0, verbose=True):
    d = np.load(os.path.join(PREP, "csl_pretrain.npz"))
    X = torch.from_numpy(d["X"]).permute(0, 3, 1, 2).float().div_(255)
    X = F.interpolate(X, size=(RES, RES), mode="area")
    seed_all(seed)
    enc = HandEncoder(width=WIDTH).to(DEV)
    opt = torch.optim.Adam(enc.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(X), batch_size=bs, shuffle=True, drop_last=True)
    t0 = time.time()
    for ep in range(epochs):
        enc.train(); tot = n = 0
        for (xb,) in dl:
            xb = xb.to(DEV, non_blocking=True)
            z1 = enc(gpu_aug(xb), project=True)
            z2 = enc(gpu_aug(xb), project=True)
            loss = nt_xent(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * xb.size(0); n += xb.size(0)
        if verbose and (ep % max(1, epochs // 6) == 0 or ep == epochs - 1):
            print("    pretrain ep %3d  loss %.4f  (%.0fs)"
                  % (ep, tot / n, time.time() - t0), flush=True)
    return enc


def finetune(state, name, seed, epochs=40, bs=64, lr=1e-3):
    """Returns (final-epoch accuracy, best-epoch accuracy)."""
    d = np.load(os.path.join(PREP, name + ".npz"))
    X = torch.from_numpy(d["X"]).permute(0, 3, 1, 2).float().div_(255)
    X = F.interpolate(X, size=(RES, RES), mode="area")
    y = torch.from_numpy(d["y"]).long()
    tr, va = d["train_idx"], d["val_idx"]
    nc = int(y[np.concatenate([tr, va])].max()) + 1

    seed_all(seed)
    enc = HandEncoder(width=WIDTH).to(DEV)
    if state is not None:
        enc.load_state_dict({k: v.clone() for k, v in state.items()})
    head = nn.Linear(enc.out_dim, nc).to(DEV)

    Xtr, ytr = X[tr].to(DEV), y[tr].to(DEV)
    Xva, yva = X[va].to(DEV), y[va].to(DEV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()),
                           lr=lr, weight_decay=1e-3)
    # Cosine decay to zero.  Under a constant learning rate several runs
    # reached 0.99 mid-training and then fell to 0.51 by the last epoch, so
    # the final-epoch value was unstable.  Decaying makes the last epoch the
    # converged one, so it can be reported without selecting an epoch on the
    # validation set.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    acc = best = 0.0
    for ep in range(epochs):
        enc.train(); head.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = gpu_aug(Xtr[idx], strong=False)
            loss = crit(head(enc(xb)), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            acc = (head(enc(Xva)).argmax(1) == yva).float().mean().item()
        best = max(best, acc)
    return acc, best


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "isl_grouped"
    print("Does CSL pretraining help on %s?  device=%s, threads=%d"
          % (name.upper(), DEV, torch.get_num_threads()), flush=True)

    # The pretrained encoder does not depend on the downstream set, so it is
    # computed once and reused.  ISL and ASL fine-tune from the same weights,
    # which also makes the two datasets directly comparable.
    ckpt = os.path.join(PREP, "csl_encoder.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        print("  stage 1: reusing cached CSL encoder", flush=True)
    else:
        print("  stage 1: contrastive pretraining on CSL (no labels)", flush=True)
        state = {k: v.detach().cpu().clone()
                 for k, v in pretrain().state_dict().items()}
        torch.save(state, ckpt)
        print("  saved encoder", flush=True)

    fin = {"scratch": [], "pretrained": []}
    bst = {"scratch": [], "pretrained": []}
    t0 = time.time()
    for s in SEEDS:
        a0, b0 = finetune(None, name, s)
        a1, b1 = finetune(state, name, s)
        fin["scratch"].append(a0); fin["pretrained"].append(a1)
        bst["scratch"].append(b0); bst["pretrained"].append(b1)
        print("  seed %-5d scratch %.4f   pretrained %.4f   %+.4f   (%.0fs)"
              % (s, a0, a1, a1 - a0, time.time() - t0), flush=True)

    out = {"dataset": name, "seeds": SEEDS,
           "metric": "final-epoch validation accuracy"}
    for tag, r in (("", fin), ("_best", bst)):
        a = np.array(r["scratch"]); b = np.array(r["pretrained"])
        t, p = stats.ttest_rel(b, a)
        dd = b - a
        sd = dd.std(ddof=1)
        dz = float(dd.mean() / sd) if sd > 0 else float("nan")
        out["scratch" + tag] = a.tolist()
        out["pretrained" + tag] = b.tolist()
        out["gain" + tag] = float(b.mean() - a.mean())
        out["p" + tag] = float(p)
        out["dz" + tag] = dz
        label = "final epoch" if tag == "" else "best epoch (optimistic)"
        print("", flush=True)
        print("  %s" % label, flush=True)
        print("    scratch     %.4f +/- %.4f" % (a.mean(), a.std(ddof=1)), flush=True)
        print("    pretrained  %.4f +/- %.4f" % (b.mean(), b.std(ddof=1)), flush=True)
        print("    gain        %+.4f   p=%.4f   d_z=%+.2f   (%d/%d seeds)"
              % (b.mean() - a.mean(), p, dz, (dd > 0).sum(), len(dd)), flush=True)

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/pretrain_%s.json" % name, "w"), indent=2)
    print("", flush=True)
    print("wrote results/pretrain_%s.json" % name, flush=True)
