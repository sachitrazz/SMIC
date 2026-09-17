"""
protocol_test.py -- how much of a reported accuracy is the split?

Three evaluation protocols over the same images, the same model and the same
schedule.  Only the train/validation partition changes.

  A  supplied     the _TRAIN / _TEST folders as distributed
  B  deduplicated MD5-unique images, stratified random image split
  C  group-disjoint near-duplicate groups kept intact across the split

Only in C does no validation image have a near-copy in training.

Stage 0 builds the three index sets over one cropped array so that nothing
but the partition differs.

    python protocol_test.py build     # crop + index (slow, once)
    python protocol_test.py run
"""

import hashlib
import json
import os
import sys
import time

import numpy as np

NEW = os.environ.get("SMIC_DATA", os.path.join("..", "new"))
PREP = os.environ.get("SMIC_PREPARED", os.path.join("..", "prepared"))
CACHE = os.path.join(PREP, "isl_protocols.npz")
IMG = 96
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ----------------------------------------------------------------- build

def build():
    import cv2
    import handcrop as hc
    import dupgroups

    rows = []
    for split, root in (("train", "ISL_DATA_IEEE_TRAIN"),
                        ("test", "ISL_DATA_IEEE_TEST")):
        d0 = os.path.join(NEW, root)
        for cls in sorted(os.listdir(d0)):
            d = os.path.join(d0, cls)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(EXTS):
                    p = os.path.join(d, f)
                    rows.append({"path": p, "cls": cls, "supplied": split,
                                 "md5": hashlib.md5(open(p, "rb").read()).hexdigest()})

    classes = sorted({r["cls"] for r in rows})
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[r["cls"]] for r in rows])
    print("  %d files, %d classes" % (len(rows), len(classes)))

    X = np.zeros((len(rows), IMG, IMG, 3), np.uint8)
    t0 = time.time()
    for i, r in enumerate(rows):
        im = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if im is None:
            continue
        X[i] = hc.crop_hand(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), out_size=IMG)[0]
        if (i + 1) % 300 == 0:
            print("    cropped %d/%d (%.0fs)" % (i + 1, len(rows), time.time() - t0),
                  flush=True)

    # --- A: the supplied partition, verbatim
    A_tr = np.array([i for i, r in enumerate(rows) if r["supplied"] == "train"])
    A_va = np.array([i for i, r in enumerate(rows) if r["supplied"] == "test"])

    # --- B: MD5-unique, stratified random image split
    seen, uniq = set(), []
    for i, r in enumerate(rows):
        if r["md5"] not in seen:
            seen.add(r["md5"]); uniq.append(i)
    uniq = np.array(uniq)
    rng = np.random.default_rng(0)
    B_tr, B_va = [], []
    for c in np.unique(y[uniq]):
        ci = uniq[y[uniq] == c]; rng.shuffle(ci)
        k = min(max(1, int(round(.30 * len(ci)))), len(ci) - 1)
        B_va += ci[:k].tolist(); B_tr += ci[k:].tolist()

    # --- C: group-disjoint over the same MD5-unique images
    F = dupgroups.descriptors(X[uniq])
    grp_u = dupgroups.components(F @ F.T, 0.98)
    grp = -np.ones(len(rows), int); grp[uniq] = grp_u
    rng = np.random.default_rng(0)
    C_tr, C_va = [], []
    for c in np.unique(y[uniq]):
        ci = uniq[y[uniq] == c]
        gs = np.unique(grp[ci]); rng.shuffle(gs)
        k = min(max(1, int(round(.30 * len(gs)))), len(gs) - 1)
        vg = set(gs[:k].tolist())
        for i in ci:
            (C_va if grp[i] in vg else C_tr).append(int(i))

    np.savez_compressed(
        CACHE, X=X, y=y, group=grp,
        A_tr=A_tr, A_va=A_va,
        B_tr=np.array(sorted(B_tr)), B_va=np.array(sorted(B_va)),
        C_tr=np.array(sorted(C_tr)), C_va=np.array(sorted(C_va)))
    for n, tr, va in (("A supplied", A_tr, A_va),
                      ("B dedup", B_tr, B_va), ("C grouped", C_tr, C_va)):
        print("  %-11s train %4d / val %4d" % (n, len(tr), len(va)))
    print("  wrote", CACHE)


# ------------------------------------------------------------------- run

def run():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy import stats
    import pretrain_finetune as pf

    d = np.load(CACHE)
    X = torch.from_numpy(d["X"]).permute(0, 3, 1, 2).float().div_(255)
    X = F.interpolate(X, size=(pf.RES, pf.RES), mode="area")
    y = torch.from_numpy(d["y"]).long()
    nc = int(y.max()) + 1

    def one(tr, va, seed):
        pf.seed_all(seed)
        enc = pf.HandEncoder(width=pf.WIDTH).to(pf.DEV)
        head = nn.Linear(enc.out_dim, nc).to(pf.DEV)
        Xtr, ytr = X[tr].to(pf.DEV), y[tr].to(pf.DEV)
        Xva, yva = X[va].to(pf.DEV), y[va].to(pf.DEV)
        opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()),
                               lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
        crit = nn.CrossEntropyLoss()
        acc = best = 0.0
        for ep in range(40):
            enc.train(); head.train()
            perm = torch.randperm(len(Xtr), device=pf.DEV)
            for i in range(0, len(perm), 64):
                idx = perm[i:i + 64]
                loss = crit(head(enc(pf.gpu_aug(Xtr[idx], strong=False))), ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            enc.eval(); head.eval()
            with torch.no_grad():
                acc = (head(enc(Xva)).argmax(1) == yva).float().mean().item()
            best = max(best, acc)
        # final epoch is the primary figure; best-epoch is selected on the
        # validation set and reported on it, which is optimistic
        return acc, best

    def nn_lookup(tr, va):
        import dupgroups
        Fd = dupgroups.descriptors(d["X"])
        S = Fd[va] @ Fd[tr].T
        return float((d["y"][tr][S.argmax(1)] == d["y"][va]).mean())

    print("Same images, same model, same schedule -- only the split changes.\n")
    print("%-28s %-8s %-8s %-16s %s"
          % ("protocol", "n_train", "n_val", "CNN accuracy", "1-NN lookup"))
    print("-" * 78)
    out = {"seeds": pf.SEEDS, "rows": []}
    for key, label in (("A", "A  supplied split"),
                       ("B", "B  MD5-dedup, random"),
                       ("C", "C  group-disjoint")):
        tr, va = d[key + "_tr"], d[key + "_va"]
        got = [one(tr, va, s) for s in pf.SEEDS]
        accs = [g[0] for g in got]
        bests = [g[1] for g in got]
        lk = nn_lookup(tr, va)
        out["rows"].append({"protocol": label, "n_train": int(len(tr)),
                            "n_val": int(len(va)), "accs": accs,
                            "mean": float(np.mean(accs)),
                            "std": float(np.std(accs, ddof=1)),
                            "best": bests, "best_mean": float(np.mean(bests)),
                            "nn_lookup": lk})
        print("%-28s %-8d %-8d %.4f +/- %.4f   %.4f"
              % (label, len(tr), len(va), np.mean(accs),
                 np.std(accs, ddof=1), lk), flush=True)

    a = np.array(out["rows"][1]["accs"]); c = np.array(out["rows"][2]["accs"])
    t, p = stats.ttest_rel(a, c)
    out["B_vs_C"] = {"drop": float(a.mean() - c.mean()), "p": float(p)}
    print("\nB -> C costs %.1f accuracy points (p=%.2g)."
          % (100 * (a.mean() - c.mean()), p))
    print("That difference is the split, not the model.")
    json.dump(out, open("results/protocol_test.json", "w"), indent=2)
    print("\nwrote results/protocol_test.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "build":
        build()
    else:
        if not os.path.exists(CACHE):
            build()
        run()
