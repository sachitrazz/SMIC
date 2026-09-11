"""
regroup_split.py -- rebuild the splits so validation is not a lookup task.

Established by nn_leak_test.py: after MD5 deduplication the ISL validation
set is still 99.2% answerable by nearest-neighbour lookup, because 77% of
validation images have a near-identical twin in training.  dupgroups.py
shows why -- the 1006 MD5-unique ISL images form only ~216 independent
near-duplicate groups.

Fix: split by GROUP, never by image.  Every near-duplicate group lands
entirely in train or entirely in validation, so no validation image has a
twin on the other side.  Classes with a single group are dropped, since
such a class cannot appear on both sides of any honest split.

Writes {name}_grouped.npz alongside the originals; the originals are kept
so the two protocols can be compared directly in the paper.

    python regroup_split.py isl asl
"""

import json
import os
import sys

import numpy as np

import dupgroups

PREP = os.path.join("..", "prepared")
TH = 0.98            # see dupgroups.py: the knee of the group-count curve
VAL_FRAC = 0.30


def build(name, th=TH, seed=0):
    d = np.load(os.path.join(PREP, name + ".npz"))
    X, y = d["X"], d["y"]
    F = dupgroups.descriptors(X)
    grp = dupgroups.components(F @ F.T, th)

    # A group must belong to one class.  The 2 ISL groups that span classes
    # are assigned to their majority class, and the minority members are
    # dropped -- keeping them would put the same picture under two labels.
    drop = np.zeros(len(y), bool)
    for g in range(grp.max() + 1):
        m = grp == g
        vals, cnt = np.unique(y[m], return_counts=True)
        if len(vals) > 1:
            keep_c = vals[cnt.argmax()]
            drop |= m & (y != keep_c)

    keep = ~drop
    # classes need >= 2 groups to be splittable at all
    ok = np.zeros(len(y), bool)
    for c in np.unique(y[keep]):
        m = keep & (y == c)
        if len(np.unique(grp[m])) >= 2:
            ok |= m
    dropped_classes = len(np.unique(y[keep])) - len(np.unique(y[ok]))

    idx = np.where(ok)[0]
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c]
        gs = np.unique(grp[ci])
        rng.shuffle(gs)
        k = max(1, int(round(VAL_FRAC * len(gs))))
        k = min(k, len(gs) - 1)
        vg = set(gs[:k].tolist())
        for i in ci:
            (va if grp[i] in vg else tr).append(int(i))

    tr, va = np.array(sorted(tr)), np.array(sorted(va))
    # relabel classes contiguously
    classes = np.unique(y[np.concatenate([tr, va])])
    remap = {int(c): i for i, c in enumerate(classes)}
    y2 = np.array([remap.get(int(v), -1) for v in y])

    np.savez_compressed(os.path.join(PREP, name + "_grouped.npz"),
                        X=X, y=y2, train_idx=tr, val_idx=va, group=grp)

    meta_src = json.load(open(os.path.join(PREP, name + "_meta.json")))
    names = [meta_src["classes"][int(c)] for c in classes]
    meta = {"name": name + "_grouped", "threshold": th,
            "n_images": int(len(X)), "n_groups": int(grp.max() + 1),
            "n_train": int(len(tr)), "n_val": int(len(va)),
            "n_train_groups": int(len(np.unique(grp[tr]))),
            "n_val_groups": int(len(np.unique(grp[va]))),
            "classes": names, "n_classes": len(names),
            "dropped_single_group_classes": int(dropped_classes),
            "split": "group-disjoint: near-duplicate groups are never split "
                     "across train and validation",
            "why": "a random image split leaves 77%% of ISL validation images "
                   "with a near-identical training twin, making the task "
                   "answerable by lookup"}
    json.dump(meta, open(os.path.join(PREP, name + "_grouped_meta.json"), "w"),
              indent=2)
    print("%s: %d images -> %d groups | train %d img/%d grp, val %d img/%d grp"
          " | %d classes (%d dropped: single group)"
          % (name.upper(), len(X), grp.max() + 1, len(tr),
             len(np.unique(grp[tr])), len(va), len(np.unique(grp[va])),
             len(names), dropped_classes))
    return meta


if __name__ == "__main__":
    for n in (sys.argv[1:] or ["isl", "asl"]):
        build(n)
