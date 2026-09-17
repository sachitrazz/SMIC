"""
nn_leak_test.py -- can a split be answered by lookup?

MD5 deduplication removes byte-identical files.  It does not remove near
duplicates: the same photograph re-saved at a different JPEG quality, or two
consecutive frames of the same held pose.  Those are different files with the
same content, and they defeat a random split as thoroughly as exact copies.

The test is a baseline with no learning and no parameters.  For every
validation image, find its nearest training image in raw pixel space and copy
that image's label.  If this scores close to a trained network, the network's
accuracy on that split reflects retrieval rather than recognition.

    python nn_leak_test.py isl asl
"""

import json
import os
import sys

import numpy as np

PREP = os.environ.get("SMIC_PREPARED", os.path.join("..", "prepared"))


def run(name):
    d = np.load(os.path.join(PREP, name + ".npz"))
    X = d["X"].astype(np.float32) / 255.0
    y = d["y"]
    tr, va = d["train_idx"], d["val_idx"]

    # 32x32 grayscale, unit-normalised: coarse enough to ignore JPEG noise,
    # fine enough that genuinely different photographs are far apart.
    import cv2
    def feat(A):
        F = np.stack([cv2.resize(a.mean(2), (32, 32),
                                 interpolation=cv2.INTER_AREA) for a in A])
        F = F.reshape(len(F), -1)
        F -= F.mean(1, keepdims=True)
        n = np.linalg.norm(F, axis=1, keepdims=True)
        return F / np.maximum(n, 1e-8)

    Ftr, Fva = feat(X[tr]), feat(X[va])
    S = Fva @ Ftr.T                      # cosine similarity
    j = S.argmax(1)
    sim = S[np.arange(len(j)), j]
    pred = y[tr][j]
    acc = float((pred == y[va]).mean())

    print("\n%s   %d train / %d val, %d classes" %
          (name.upper(), len(tr), len(va), int(y.max()) + 1))
    print("  1-NN on raw pixels          %.4f   (chance %.4f)"
          % (acc, 1.0 / (int(y.max()) + 1)))
    print("  nearest-train similarity    median %.4f   mean %.4f"
          % (float(np.median(sim)), float(sim.mean())))
    for th in (0.99, 0.98, 0.95, 0.90):
        f = float((sim > th).mean())
        print("    val images with a train twin above %.2f : %5.1f%%" % (th, 100 * f))
    return {"dataset": name, "n_train": int(len(tr)), "n_val": int(len(va)),
            "nn_acc": acc, "chance": 1.0 / (int(y.max()) + 1),
            "median_sim": float(np.median(sim)),
            "frac_above_0.98": float((sim > 0.98).mean()),
            "frac_above_0.95": float((sim > 0.95).mean())}


if __name__ == "__main__":
    names = sys.argv[1:] or ["isl", "asl"]
    print("Nearest-neighbour leakage probe: can the validation set be "
          "answered by lookup?")
    out = [run(n) for n in names]
    # merge rather than overwrite: running the probe on one dataset should not
    # silently drop the others' entries, which the tables read from
    path = "results/nn_leak_test.json"
    prev = json.load(open(path)) if os.path.exists(path) else []
    by = {e["dataset"]: e for e in prev}
    by.update({e["dataset"]: e for e in out})
    json.dump(sorted(by.values(), key=lambda e: e["dataset"]),
              open(path, "w"), indent=2)
    print("\nwrote results/nn_leak_test.json")
