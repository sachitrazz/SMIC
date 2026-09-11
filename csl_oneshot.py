"""
csl_oneshot.py -- the standard one-shot protocol on the CSL task.

With one training clip per class, gradient training of a classifier head is
the non-standard choice; the few-shot literature classifies by nearest
prototype in a fixed embedding.  This script does that, with no gradient
step on the labelled clips at all:

  embed(clip) = mean over the 8 sign frames and the 4x4 spatial map of the
                trunk's final feature map, L2-normalised
  predict     = the class whose single training clip is most cosine-similar

Two trunks are compared:
  random       HandEncoder at its random initialisation, one per seed
  contrastive  the trunk pretrained on the sign-disjoint CSL pool
               (prepared/bench_ssl_csl.pt, the same one bench.py uses)

Reported: top-1 and top-5 accuracy, macro-F1, macro OvR ROC-AUC (scores are
softmax(cos/0.1)), beside the pixel lookup baseline and chance.

    python csl_oneshot.py
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import bench
import pretrain_finetune as pf


def embed(trunk, X, dev):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 16):
            x = X[i:i + 16].to(dev)
            B, T = x.shape[:2]
            h = trunk(x.flatten(0, 1)).mean((2, 3)).reshape(B, T, -1).mean(1)
            out.append(F.normalize(h, dim=1).cpu())
    return torch.cat(out)


def evaluate(trunk, data, dev):
    X, y, tr, va, nc, video = data
    trunk.eval()
    Et, Ev = embed(trunk, X[tr], dev), embed(trunk, X[va], dev)
    yt, yv = y[tr].numpy(), y[va].numpy()
    S = (Ev @ Et.T).numpy()                              # val x train
    # one training clip per class: map columns to classes
    score = np.full((len(yv), nc), -1.0)
    for j, c in enumerate(yt):
        score[:, c] = np.maximum(score[:, c], S[:, j])
    pred = score.argmax(1)
    top5 = (np.argsort(-score, 1)[:, :5] == yv[:, None]).any(1).mean()
    prob = np.exp(score / 0.1); prob /= prob.sum(1, keepdims=True)
    return {"acc": float((pred == yv).mean()), "top5": float(top5),
            "f1": bench.macro_f1(pred, yv), "auc": bench.macro_auc(prob, yv, nc)}


def main():
    dev = bench.DEV
    data = bench.load("csl")
    X, y, tr, va, nc, video = data
    lk = bench.lookup_baseline(X, y, tr, va, video)
    out = {"lookup": lk, "chance": 1.0 / nc, "n_classes": nc, "seeds": bench.SEEDS}

    rand = []
    for s in bench.SEEDS:
        pf.seed_all(s)
        enc = pf.HandEncoder(width=pf.WIDTH)
        rand.append(evaluate(enc.net[:-1].to(dev), data, dev))
    out["random"] = {"runs": rand,
                     "mean": {k: float(np.mean([r[k] for r in rand])) for k in rand[0]},
                     "std": {k: float(np.std([r[k] for r in rand], ddof=1)) for k in rand[0]}}

    st = torch.load(os.path.join(bench.PREP, "bench_ssl_csl.pt"), map_location="cpu")
    enc = pf.HandEncoder(width=pf.WIDTH); enc.load_state_dict(st)
    out["contrastive"] = evaluate(enc.net[:-1].to(dev), data, dev)

    print("lookup %.4f  chance %.4f" % (lk, 1.0 / nc))
    print("random trunk      ", {k: round(v, 4) for k, v in out["random"]["mean"].items()})
    print("contrastive trunk ", {k: round(v, 4) for k, v in out["contrastive"].items()})
    json.dump(out, open("results/csl_oneshot.json", "w"), indent=1)
    print("wrote results/csl_oneshot.json")


if __name__ == "__main__":
    main()
