"""
threshold_sweep.py -- is the grouping threshold doing the work, or is it
just a knob?

regroup_split.py joins two images into the same near-duplicate group when
their descriptors exceed cosine 0.98.  That number needs justifying: too
loose and the split still leaks, too tight and the dataset dissolves into
a handful of groups and the experiment is about nothing.

The lookup baseline answers this without training anything.  For each
candidate threshold we rebuild the group-disjoint split and ask how much
of the validation set a nearest-neighbour lookup can still recover.  A
sound threshold is one past which further tightening stops buying
reductions in lookup accuracy -- at that point what remains is genuine
between-photograph similarity within a sign class, which is the signal a
recogniser is supposed to use.

    python threshold_sweep.py isl asl
"""

import json
import os
import sys

import numpy as np

import dupgroups

PREP = os.path.join("..", "prepared")
THRESHOLDS = [0.999, 0.99, 0.98, 0.95, 0.92, 0.90]
VAL_FRAC = 0.30


def grouped_split(y, grp, seed=0, val_frac=VAL_FRAC):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        gs = np.unique(grp[ci])
        if len(gs) < 2:                    # unsplittable class
            continue
        rng.shuffle(gs)
        k = min(max(1, int(round(val_frac * len(gs)))), len(gs) - 1)
        vg = set(gs[:k].tolist())
        for i in ci:
            (va if grp[i] in vg else tr).append(int(i))
    return np.array(sorted(tr)), np.array(sorted(va))


def run(name):
    d = np.load(os.path.join(PREP, name + ".npz"))
    X, y = d["X"], d["y"]
    F = dupgroups.descriptors(X)
    S = F @ F.T
    nc = int(y.max()) + 1

    print("")
    print("%s   %d images, %d classes, chance %.3f"
          % (name.upper(), len(X), nc, 1.0 / nc))
    print("  %-11s %-9s %-9s %-9s %-9s %s"
          % ("threshold", "groups", "train", "val", "largest", "lookup acc"))
    rows = []
    for th in THRESHOLDS:
        grp = dupgroups.components(S, th)
        tr, va = grouped_split(y, grp)
        if len(tr) == 0 or len(va) == 0:
            print("  %-11.3f  no splittable classes remain" % th)
            continue
        Sv = S[np.ix_(va, tr)]
        acc = float((y[tr][Sv.argmax(1)] == y[va]).mean())
        # the largest component says when single-linkage starts to chain:
        # once one group swallows a large slice of a class, a loose threshold
        # is merging distinct photographs rather than duplicates
        largest = int(np.bincount(grp).max())
        rows.append({"threshold": th, "n_groups": int(grp.max() + 1),
                     "n_train": int(len(tr)), "n_val": int(len(va)),
                     "lookup_acc": acc, "largest_group": largest,
                     "n_classes": int(len(np.unique(y[va])))})
        print("  %-11.3f %-9d %-9d %-9d %-9d %.4f"
              % (th, grp.max() + 1, len(tr), len(va), largest, acc))
    return {"dataset": name, "chance": 1.0 / nc, "rows": rows}


if __name__ == "__main__":
    print("Lookup accuracy under a group-disjoint split, as the "
          "near-duplicate threshold tightens.")
    out = [run(n) for n in (sys.argv[1:] or ["isl", "asl"])]
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/threshold_sweep.json", "w"), indent=2)
    print("")
    print("wrote results/threshold_sweep.json")
