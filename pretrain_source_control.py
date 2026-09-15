"""
pretrain_source_control.py -- is it the CSL data, or just any pretraining?

pretrain_finetune.py compares a CSL-pretrained encoder against a randomly
initialised one.  A gain there has two possible explanations, and they are
not the same claim:

  (i)  the encoder learned something from 10,304 extra hands, or
  (ii) contrastive pretraining is a useful initialisation regardless of
       what it was run on, so any images would have done.

This separates them by adding a third arm that pretrains on the target
set's OWN training images, self-supervised, with no labels and no extra
data.  It sees the same contrastive objective, the same schedule and the
same number of gradient steps as the CSL arm; the only difference is that
it has 661 images to look at instead of 10,304.

  random init      no pretraining
  self-pretrained  contrastive on the target training split (no new data)
  CSL-pretrained   contrastive on 10,304 unlabelled CSL hands

If CSL beats self, the extra data is doing the work.  If they tie, the
claim is about contrastive initialisation, not about CSL.

    python pretrain_source_control.py isl_grouped
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

import pretrain_finetune as pf

PREP = pf.PREP


def pretrain_on(X_uint8, steps_per_epoch, epochs=20, bs=128, lr=1e-3, seed=0):
    """Contrastive pretraining on an arbitrary image array.

    `steps_per_epoch` is fixed to the CSL arm's value so that both
    pretraining arms take the same number of optimiser steps.  Without
    that, pretraining on a 661-image set would get 16x fewer updates and
    the comparison would be about step count rather than about data.
    """
    X = torch.from_numpy(X_uint8).permute(0, 3, 1, 2).float().div_(255)
    X = F.interpolate(X, size=(pf.RES, pf.RES), mode="area")
    pf.seed_all(seed)
    enc = pf.HandEncoder(width=pf.WIDTH).to(pf.DEV)
    opt = torch.optim.Adam(enc.parameters(), lr=lr, weight_decay=1e-4)
    n = len(X)
    t0 = time.time()
    for ep in range(epochs):
        enc.train()
        tot = seen = 0
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, n, (min(bs, n),))
            xb = X[idx].to(pf.DEV)
            z1 = enc(pf.gpu_aug(xb), project=True)
            z2 = enc(pf.gpu_aug(xb), project=True)
            loss = pf.nt_xent(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * xb.size(0); seen += xb.size(0)
        if ep % 5 == 0 or ep == epochs - 1:
            print("    ep %3d  loss %.4f  (%.0fs)"
                  % (ep, tot / seen, time.time() - t0), flush=True)
    return {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}


def main(name="isl_grouped"):
    print("Is the gain from CSL, or from contrastive pretraining as such?  "
          "device=%s" % pf.DEV, flush=True)

    csl = np.load(os.path.join(PREP, "csl_pretrain.npz"))["X"]
    steps = len(csl) // 128
    print("  matching both arms to %d steps/epoch (CSL's value)" % steps,
          flush=True)

    d = np.load(os.path.join(PREP, name + ".npz"))
    own = d["X"][d["train_idx"]]
    print("  self arm sees %d images; CSL arm sees %d" % (len(own), len(csl)),
          flush=True)

    ckpt = os.path.join(PREP, "csl_encoder.pt")
    if os.path.exists(ckpt):
        st_csl = torch.load(ckpt, map_location="cpu")
        print("  CSL encoder: cached", flush=True)
    else:
        print("  CSL encoder: training", flush=True)
        st_csl = pretrain_on(csl, steps)
        torch.save(st_csl, ckpt)

    print("  self-supervised encoder on %s train split:" % name, flush=True)
    st_own = pretrain_on(own, steps)

    arms = {"random": None, "self": st_own, "csl": st_csl}
    accs = {k: [] for k in arms}
    for s in pf.SEEDS:
        line = []
        for k, st in arms.items():
            a, _ = pf.finetune(st, name, s)
            accs[k].append(a)
            line.append("%s %.4f" % (k, a))
        print("  seed %-5d %s" % (s, "   ".join(line)), flush=True)

    out = {"dataset": name, "seeds": pf.SEEDS, "steps_per_epoch": steps,
           "n_self": int(len(own)), "n_csl": int(len(csl)),
           "accs": {k: v for k, v in accs.items()}}
    print("", flush=True)
    for k in arms:
        a = np.array(accs[k])
        print("  %-8s %.4f +/- %.4f" % (k, a.mean(), a.std(ddof=1)), flush=True)
    for a_, b_ in (("random", "self"), ("random", "csl"), ("self", "csl")):
        x, y = np.array(accs[a_]), np.array(accs[b_])
        t, p = stats.ttest_rel(y, x)
        out["%s_vs_%s" % (a_, b_)] = {"gain": float(y.mean() - x.mean()),
                                      "p": float(p)}
        print("  %-8s -> %-8s %+.4f   p=%.4f" % (a_, b_, y.mean() - x.mean(), p),
              flush=True)

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/pretrain_source_%s.json" % name, "w"), indent=2)
    print("", flush=True)
    print("wrote results/pretrain_source_%s.json" % name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "isl_grouped")
