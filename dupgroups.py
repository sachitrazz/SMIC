"""
dupgroups.py -- how much INDEPENDENT data is actually in these sets?

MD5 dedup left the ISL validation set answerable by nearest-neighbour
lookup (99.2%), so the images are not independent: many are the same
photograph re-saved, or consecutive frames of one held pose.

This groups images into near-duplicate components -- an edge whenever two
images exceed a cosine threshold on a 32x32 grayscale descriptor, then
connected components -- and reports how many components survive.  The
component count, not the file count, is the real sample size.
"""

import os
import sys

import cv2
import numpy as np

PREP = os.environ.get("SMIC_PREPARED", os.path.join("..", "prepared"))


def descriptors(X):
    F = np.stack([cv2.resize(a.astype(np.float32).mean(2) / 255.0, (32, 32),
                             interpolation=cv2.INTER_AREA) for a in X])
    F = F.reshape(len(F), -1)
    F -= F.mean(1, keepdims=True)
    return F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)


def components(S, th):
    n = len(S)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    ii, jj = np.where(np.triu(S, 1) > th)
    for a, b in zip(ii, jj):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    lab = np.array([find(i) for i in range(n)])
    _, lab = np.unique(lab, return_inverse=True)
    return lab


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["isl", "asl"]):
        d = np.load(os.path.join(PREP, name + ".npz"))
        X, y = d["X"], d["y"]
        F = descriptors(X)
        S = F @ F.T
        print("\n%s: %d files, %d MD5-unique images, %d classes"
              % (name.upper(), len(X), len(X), int(y.max()) + 1))
        print("  %-11s %-11s %-13s %s" % ("threshold", "groups", "per class",
                                          "largest group"))
        for th in (0.999, 0.99, 0.98, 0.95, 0.92, 0.90):
            lab = components(S, th)
            g = lab.max() + 1
            sizes = np.bincount(lab)
            # groups that stay inside one class (a group spanning classes
            # would mean two different signs are pixel-identical)
            print("  %-11.3f %-11d %-13.1f %d"
                  % (th, g, g / (y.max() + 1), sizes.max()))
        lab = components(S, 0.95)
        cross = 0
        for gid in range(lab.max() + 1):
            if len(np.unique(y[lab == gid])) > 1:
                cross += 1
        print("  at 0.95: %d groups span more than one class" % cross)
