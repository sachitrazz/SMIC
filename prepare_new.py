"""
prepare_new.py -- deduplicated, hand-cropped arrays for ISL-IEEE and ASL-IEEE.

The distributed train and test folders cannot be used as a split.  Verified by
MD5:

  ISL_DATA_IEEE   175/175 test images are byte-identical to training images
  ASL_DATA_IEEE    99/ 99 unique test images likewise
  CSL            1302/1302 test clips are the same clips as train

There are also duplicates within the training folders (ASL: 720 files, 419
unique).  This script therefore ignores the supplied split:

  1. Deduplicate by content hash, keeping one copy of each image.
  2. Split.  ISL/ASL get a stratified image split here, which
     regroup_split.py then replaces with the group-disjoint split used in
     the paper.  The `csl` mode builds a signer-independent split
     (persons 0000-0007 train, 0008-0009 test); the one-shot CSL task in the
     paper is built by prepare_csl_signer.py instead.
  3. Crop to the hand (handcrop.py).  For clips the box is propagated across
     frames when the detector misses, so motion-blurred frames are kept.
  4. Cache to .npz so training does not re-decode the images every epoch.

Usage
-----
    python prepare_new.py isl
    python prepare_new.py asl
    python prepare_new.py csl        # slow: ~52k frames
    python prepare_new.py all
"""

import hashlib
import json
import os
import re
import sys
import time

import cv2
import numpy as np

import handcrop as hc

NEW = os.environ.get("SMIC_DATA", os.path.join("..", "new"))
OUT = os.environ.get("SMIC_PREPARED", os.path.join("..", "prepared"))
IMG = 96
CLIP_T = 12                       # frames sampled per CSL clip
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _read(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _hash(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


# ======================================================================
# still-image datasets (ISL, ASL)
# ======================================================================

def prepare_images(name, roots, val_frac=0.25, seed=0):
    """Dedup across ALL supplied folders, then make one clean split."""
    seen, items = {}, []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for cls in sorted(os.listdir(root)):
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not f.lower().endswith(EXTS):
                    continue
                p = os.path.join(d, f)
                h = _hash(p)
                if h in seen:                      # duplicate, keep first
                    continue
                seen[h] = p
                items.append((p, cls))

    classes = sorted({c for _, c in items})
    c2i = {c: i for i, c in enumerate(classes)}
    labels = np.array([c2i[c] for _, c in items])
    print("  %d unique images, %d classes (from %d files)"
          % (len(items), len(classes), sum(
              len(os.listdir(os.path.join(r, c)))
              for r in roots if os.path.isdir(r)
              for c in os.listdir(r) if os.path.isdir(os.path.join(r, c)))))

    rng = np.random.default_rng(seed)
    tr_i, va_i = [], []
    for c in np.unique(labels):                    # stratified
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(val_frac * len(idx))))
        k = min(k, len(idx) - 1) if len(idx) > 1 else 0
        va_i += idx[:k].tolist()
        tr_i += idx[k:].tolist()

    meth = {}
    X = np.zeros((len(items), IMG, IMG, 3), np.uint8)
    t0 = time.time()
    for i, (p, _) in enumerate(items):
        im = _read(p)
        if im is None:
            continue
        crop, m = hc.crop_hand(im, out_size=IMG)
        X[i] = crop
        meth[m] = meth.get(m, 0) + 1
        if (i + 1) % 250 == 0:
            print("    cropped %d/%d  (%.0fs)" % (i + 1, len(items), time.time() - t0),
                  flush=True)

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, name + ".npz"),
                        X=X, y=labels,
                        train_idx=np.array(sorted(tr_i)),
                        val_idx=np.array(sorted(va_i)))
    meta = {"name": name, "classes": classes, "n": len(items),
            "n_train": len(tr_i), "n_val": len(va_i),
            "img": IMG, "crop_methods": meth,
            "split": "stratified image split after MD5 dedup",
            "leakage_note": "supplied _TEST folders were subsets of _TRAIN; "
                            "the supplied split was discarded"}
    json.dump(meta, open(os.path.join(OUT, name + "_meta.json"), "w"), indent=2)
    print("  train %d / val %d   crop methods %s" % (len(tr_i), len(va_i), meth))
    return meta


# ======================================================================
# CSL video, signer-independent
# ======================================================================

def prepare_csl(train_persons=("0000", "0001", "0002", "0003", "0004",
                              "0005", "0006", "0007"),
                test_persons=("0008", "0009"), min_clips_per_sign=2):
    """Signer-independent split over CSL clips.

    Only signs that appear on BOTH sides are kept -- otherwise the test set
    would contain classes the model has never seen, which is a different
    (zero-shot) problem.
    """
    pat = re.compile(r"S(\d+)_P(\d+)_T(\d+)")
    clips = []
    for root in [os.path.join(NEW, "CSL_Train"), os.path.join(NEW, "CSL_Test")]:
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            m = pat.match(d)
            if not m:
                continue
            full = os.path.join(root, d)
            fr = sorted(f for f in os.listdir(full) if f.lower().endswith(EXTS))
            if len(fr) >= 4:
                clips.append({"sign": m.group(1), "person": m.group(2),
                              "dir": full, "frames": fr})

    # deduplicate clips that appear in both supplied folders
    best = {}
    for c in clips:
        k = (c["sign"], c["person"])
        if k not in best or len(c["frames"]) > len(best[k]["frames"]):
            best[k] = c
    clips = list(best.values())

    tr = [c for c in clips if c["person"] in train_persons]
    te = [c for c in clips if c["person"] in test_persons]
    tr_signs = {c["sign"] for c in tr}
    te_signs = {c["sign"] for c in te}
    keep = tr_signs & te_signs
    tr = [c for c in tr if c["sign"] in keep]
    te = [c for c in te if c["sign"] in keep]

    print("  clips after dedup: %d" % len(clips))
    print("  signer-independent: train persons %s, test persons %s"
          % (list(train_persons), list(test_persons)))
    print("  signs on both sides: %d  ->  train %d clips / test %d clips"
          % (len(keep), len(tr), len(te)))
    if not keep:
        print("  NO SHARED SIGNS between signer groups -- CSL cannot be used")
        print("  for signer-independent classification with this labelling.")
        json.dump({"name": "csl", "usable": False,
                   "reason": "no sign appears for both train and test signers",
                   "n_clips": len(clips), "n_signs": len(tr_signs | te_signs)},
                  open(os.path.join(OUT, "csl_meta.json"), "w"), indent=2)
        return None

    signs = sorted(keep)
    s2i = {s: i for i, s in enumerate(signs)}
    meth = {}

    def encode(group, tag):
        X = np.zeros((len(group), CLIP_T, IMG, IMG, 3), np.uint8)
        y = np.zeros(len(group), np.int64)
        t0 = time.time()
        for i, c in enumerate(group):
            idx = np.linspace(0, len(c["frames"]) - 1, CLIP_T).round().astype(int)
            frames = [_read(os.path.join(c["dir"], c["frames"][j])) for j in idx]
            frames = [f for f in frames if f is not None]
            if not frames:
                continue
            mo = hc.clip_motion(frames)
            last = None
            for t, (f, m) in enumerate(zip(frames, mo)):
                box = hc.hand_box_landmarks(f, motion=m)
                if box is None:
                    box = hc.hand_box_skin(f, motion=m, exclude_top=0.28)
                if box is None:
                    box, mname = (last, "propagated") if last is not None else (None, "center")
                else:
                    last, mname = box, "detected"
                if box is None:
                    H, W = f.shape[:2]
                    s = int(min(H, W) * .7)
                    box = ((W - s) / 2, (H - s) / 2, (W + s) / 2, (H + s) / 2)
                meth[mname] = meth.get(mname, 0) + 1
                H, W = f.shape[:2]
                x0, y0, x1, y1 = hc.expand_square(box, W, H, .22)
                cr = f[y0:y1, x0:x1]
                if cr.size:
                    X[i, t] = cv2.resize(cr, (IMG, IMG), interpolation=cv2.INTER_AREA)
            y[i] = s2i[c["sign"]]
            if (i + 1) % 25 == 0:
                print("    %s %d/%d  (%.0fs)" % (tag, i + 1, len(group),
                                                 time.time() - t0), flush=True)
        return X, y

    Xtr, ytr = encode(tr, "train")
    Xte, yte = encode(te, "test")
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "csl.npz"),
                        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    meta = {"name": "csl", "usable": True, "classes": signs,
            "n_train": len(tr), "n_val": len(te), "T": CLIP_T, "img": IMG,
            "crop_methods": meth,
            "split": "signer-independent: P0000-0007 train, P0008-0009 test",
            "leakage_note": "supplied CSL_Test duplicated CSL_Train; discarded"}
    json.dump(meta, open(os.path.join(OUT, "csl_meta.json"), "w"), indent=2)
    print("  crop methods:", meth)
    return meta


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    if which in ("isl", "all"):
        print("ISL:")
        prepare_images("isl", [os.path.join(NEW, "ISL_DATA_IEEE_TRAIN"),
                               os.path.join(NEW, "ISL_DATA_IEEE_TEST")])
    if which in ("asl", "all"):
        print("ASL:")
        prepare_images("asl", [os.path.join(NEW, "ASL_DATA_IEEE_TRAIN"),
                               os.path.join(NEW, "ASL_DATA_IEEE_TEST")])
    if which in ("csl", "all"):
        print("CSL:")
        prepare_csl()
    print("\ndone ->", OUT)
