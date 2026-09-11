"""
bench.py -- the main benchmark: every model on every dataset, one protocol.

Datasets (all audited, all leakage-controlled):
  isl   ISL-IEEE, 35 classes, near-duplicate-group-disjoint split
  asl   ASL-IEEE, 24 classes, near-duplicate-group-disjoint split
  csl   CSL, 97 classes, one clip per class for training, tested on a clip
        of the same sign by a DIFFERENT signer (8 sign frames per clip)

Every input is a 64 px RGB hand crop.  A shared convolutional trunk turns
an image into 16 spatial tokens, or a clip into T=8 temporal tokens (one
pooled vector per sign frame).  The models differ only in what reads the
token sequence:

  cnn        mean over tokens, LayerNorm, linear             (no sequence model)
  cnn_tf     2-layer Transformer, then the same readout
  lstm       2-layer LSTM, last hidden state                  (recurrent baseline)
  stlat      referenced dual-memory STLAT with ASFG readout   (as implemented)
  proposed   cnn, with the trunk initialised by self-supervised contrastive
             pretraining (own training split for isl/asl; a sign-disjoint
             pool of CSL clips for csl, so no test sign is ever seen)

Protocol: 5 shared seeds, 40 epochs, Adam 1e-3, weight decay 1e-3, cosine
decay to zero, final-epoch reporting.  Metrics: accuracy, macro-F1,
macro one-vs-rest ROC-AUC, train and validation loss, per-epoch curves.

    python bench.py isl asl csl
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

import pretrain_finetune as pf
import pretrain_source_control as psc
from model import ReferencedSTLAT

PREP = pf.PREP
DEV = pf.DEV
SEEDS = [42, 123, 456, 789, 1024]
# BENCH_EPOCHS / BENCH_TAG allow a disclosed sensitivity run (e.g. CSL at
# the optimiser-step count ISL receives) without touching the main results
EPOCHS = int(os.environ.get("BENCH_EPOCHS", 40))
TAG = os.environ.get("BENCH_TAG", "")
MODELS = os.environ.get("BENCH_MODELS", "cnn,cnn_tf,lstm,stlat,proposed").split(",")
FILES = {"isl": "isl_grouped", "asl": "asl_grouped", "csl": "csl_signer"}


# ---------------------------------------------------------------- data

def load(name):
    d = np.load(os.path.join(PREP, FILES[name] + ".npz"))
    X = torch.from_numpy(d["X"]).float().div_(255)
    video = X.dim() == 5                                   # N,T,H,W,C
    if video:
        N, T = X.shape[:2]
        X = X.reshape(N * T, *X.shape[2:]).permute(0, 3, 1, 2)
        X = F.interpolate(X, size=(pf.RES, pf.RES), mode="area")
        X = X.reshape(N, T, 3, pf.RES, pf.RES)
    else:
        X = F.interpolate(X.permute(0, 3, 1, 2), size=(pf.RES, pf.RES), mode="area")
    y = torch.from_numpy(d["y"]).long()
    tr, va = d["train_idx"], d["val_idx"]
    classes = np.unique(np.concatenate([d["y"][tr], d["y"][va]]))
    remap = {int(c): i for i, c in enumerate(classes)}
    y = torch.tensor([remap.get(int(v), -1) for v in d["y"]])
    return X, y, tr, va, len(classes), video


def lookup_baseline(X, y, tr, va, video):
    """1-NN in the 32x32 grayscale descriptor used by the audit."""
    def desc(A):
        if video:
            A = A.mean(1)
        g = F.interpolate(A.mean(1, keepdim=True), size=(32, 32), mode="area")
        g = g.flatten(1)
        g = g - g.mean(1, keepdim=True)
        return g / g.norm(dim=1, keepdim=True).clamp_min(1e-8)
    Ft, Fv = desc(X[tr]), desc(X[va])
    j = (Fv @ Ft.T).argmax(1)
    return float((y[tr][j] == y[va]).float().mean())


# --------------------------------------------------------------- model

class Net(nn.Module):
    def __init__(self, head, nc, video, state=None):
        super().__init__()
        enc = pf.HandEncoder(width=pf.WIDTH)
        if state is not None:
            enc.load_state_dict({k: v.clone() for k, v in state.items()})
        self.trunk = enc.net[:-1]                           # drop global pool
        d = enc.out_dim
        self.video, self.head = video, head
        L = 8 if video else 16
        if head == "cnn_tf":
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True)
            self.tf = nn.TransformerEncoder(layer, 2)
            self.pos = nn.Parameter(torch.zeros(1, L, d))
        elif head == "lstm":
            self.rnn = nn.LSTM(d, 64, 2, batch_first=True, dropout=0.2)
            d = 64
        elif head in ("stlat", "stlat_fixed"):
            # referenced STLAT configuration
            self.st = ReferencedSTLAT(d, 64, nc, num_layers=4, num_heads=4,
                                    transformer_layers=2, dim_feedforward=128,
                                    dropout=0.2, use_asfg=True,
                                    wire_fix=(head == "stlat_fixed"))
        if not head.startswith("stlat"):    # STLAT carries its own classifier
            self.norm = nn.LayerNorm(d)
            self.fc = nn.Linear(d, nc)

    def tokens(self, x):
        if self.video:
            B, T = x.shape[:2]
            h = self.trunk(x.flatten(0, 1))                 # B*T,d,4,4
            return h.mean((2, 3)).reshape(B, T, -1)         # temporal tokens
        return self.trunk(x).flatten(2).transpose(1, 2)     # B,16,d spatial

    def forward(self, x):
        z = self.tokens(x)
        if self.head.startswith("stlat"):
            return self.st(z)[0]
        if self.head == "cnn_tf":
            z = self.tf(z + self.pos).mean(1)
        elif self.head == "lstm":
            z = self.rnn(z)[0][:, -1]
        else:
            z = z.mean(1)
        return self.fc(self.norm(z))


def augment(x, video):
    if not video:
        return pf.gpu_aug(x, strong=False)
    B, T = x.shape[:2]
    return pf.gpu_aug(x.flatten(0, 1), strong=False).reshape(x.shape)


# ------------------------------------------------------------- metrics

def macro_auc(prob, y, nc):
    try:
        from sklearn.metrics import roc_auc_score
        present = np.unique(y)
        if len(present) < 2:
            return float("nan")
        return float(roc_auc_score(y, prob[:, present] / prob[:, present].sum(1, keepdims=True),
                                   multi_class="ovr", average="macro", labels=present))
    except Exception:
        aucs = []
        for c in np.unique(y):
            s, pos = prob[:, c], (y == c)
            if pos.all() or (~pos).all():
                continue
            order = np.argsort(s)
            r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
            aucs.append((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))
        return float(np.mean(aucs))


def macro_f1(pred, y):
    f = []
    for c in np.unique(y):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


# ----------------------------------------------------------- training

def run(name, head, seed, data, state=None, keep_probs=False):
    X, y, tr, va, nc, video = data
    pf.seed_all(seed)
    net = Net("cnn" if head == "proposed" else head, nc, video, state).to(DEV)
    Xtr, ytr, Xva, yva = X[tr].to(DEV), y[tr].to(DEV), X[va].to(DEV), y[va].to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    bs = 32 if video else 64
    curve = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        tl = tc = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            out = net(augment(Xtr[idx], video))
            loss = crit(out, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += float(loss) * len(idx); tc += float((out.argmax(1) == ytr[idx]).sum())
        sched.step()
        net.eval()
        with torch.no_grad():
            ov = torch.cat([net(Xva[i:i + 128]) for i in range(0, len(Xva), 128)])
            vl = float(crit(ov, yva))
            va_acc = float((ov.argmax(1) == yva).float().mean())
        curve["train_loss"].append(tl / len(Xtr)); curve["train_acc"].append(tc / len(Xtr))
        curve["val_loss"].append(vl); curve["val_acc"].append(va_acc)
    prob = F.softmax(ov, 1).cpu().numpy()
    yv = yva.cpu().numpy()
    pred = prob.argmax(1)
    res = {"acc": float((pred == yv).mean()), "f1": macro_f1(pred, yv),
           "auc": macro_auc(prob, yv, nc), "val_loss": curve["val_loss"][-1],
           "train_loss": curve["train_loss"][-1], "train_acc": curve["train_acc"][-1],
           "curve": curve,
           "params": int(sum(p.numel() for p in net.parameters()))}
    if keep_probs:
        res["probs"] = prob.tolist(); res["labels"] = yv.tolist()
    return res


def contrastive_state(name, data):
    ck = os.path.join(PREP, "bench_ssl_%s.pt" % name)
    if os.path.exists(ck):
        return torch.load(ck, map_location="cpu")
    X, y, tr, va, nc, video = data
    if name == "csl":
        pool = np.load(os.path.join(PREP, "csl_pool.npz"))["X"]      # sign-disjoint
    else:
        d = np.load(os.path.join(PREP, FILES[name] + ".npz"))
        pool = d["X"][d["train_idx"]]                                 # own train split
    print("  contrastive pretraining on %d images" % len(pool), flush=True)
    st = psc.pretrain_on(pool, 10304 // 128)
    torch.save(st, ck)
    return st


def main(names):
    os.makedirs("results", exist_ok=True)
    for name in names:
        t0 = time.time()
        data = load(name)
        X, y, tr, va, nc, video = data
        lk = lookup_baseline(X, y, tr, va, video)
        print("\n== %s: %d classes, train %d, val %d, %s, lookup %.4f"
              % (name, nc, len(tr), len(va), "video T=%d" % X.shape[1] if video else "image", lk),
              flush=True)
        st = contrastive_state(name, data) if "proposed" in MODELS else None
        out = {"dataset": name, "n_classes": nc, "n_train": int(len(tr)),
               "n_val": int(len(va)), "lookup": lk, "seeds": SEEDS,
               "chance": 1.0 / nc, "models": {}}
        for m in MODELS:
            rs = [run(name, m, s, data, st if m == "proposed" else None,
                      keep_probs=(s == SEEDS[0])) for s in SEEDS]
            agg = {k: [r[k] for r in rs] for k in ("acc", "f1", "auc", "val_loss",
                                                    "train_loss", "train_acc")}
            out["models"][m] = {
                "runs": agg, "params": rs[0]["params"],
                "mean": {k: float(np.nanmean(v)) for k, v in agg.items()},
                "std": {k: float(np.nanstd(v, ddof=1)) for k, v in agg.items()},
                "curve_seed42": rs[0]["curve"],
                "probs_seed42": rs[0]["probs"], "labels_seed42": rs[0]["labels"]}
            mm = out["models"][m]["mean"]
            print("  %-9s acc %.4f  f1 %.4f  auc %.4f  vloss %.3f  train %.3f  (%.0fs)"
                  % (m, mm["acc"], mm["f1"], mm["auc"], mm["val_loss"], mm["train_acc"],
                     time.time() - t0), flush=True)
        out["tests_vs_stlat"] = {}
        base = (np.array(out["models"]["stlat"]["runs"]["acc"])
                if "stlat" in out["models"] else None)
        for m in MODELS:
            if m == "stlat" or base is None:
                continue
            a = np.array(out["models"][m]["runs"]["acc"])
            t, p = stats.ttest_rel(a, base)
            d = a - base
            out["tests_vs_stlat"][m] = {"gain": float(d.mean()), "p": float(p),
                                        "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"),
                                        "wins": int((d > 0).sum())}
        if "proposed" in out["models"] and "cnn" in out["models"]:
            a, b = (np.array(out["models"]["proposed"]["runs"]["acc"]),
                    np.array(out["models"]["cnn"]["runs"]["acc"]))
            t, p = stats.ttest_rel(a, b)
            out["proposed_vs_cnn"] = {"gain": float((a - b).mean()), "p": float(p)}
        out["epochs"] = EPOCHS
        json.dump(out, open("results/bench_%s%s.json" % (name, TAG), "w"))
        print("  wrote results/bench_%s%s.json" % (name, TAG), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["isl", "asl", "csl"])
