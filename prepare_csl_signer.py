"""
prepare_csl_signer.py -- CSL as a signer-disjoint, one-shot video task.

What the corpus supports
------------------------
1302 unique clips cover 1197 signs; 1098 signs have exactly one clip.  Only
99 signs have two or more clips, and in every such case the clips come from
different signers.  The only classification task the corpus supports is
therefore 99-way recognition with one training clip per sign, tested on a
clip of the same sign performed by a DIFFERENT signer.  That is hard, and it
is the signer-independent condition evaluations usually cannot offer.

Sign-frame extraction
---------------------
A CSL clip contains idle frames before and after the sign.  Per frame we
compute motion energy (mean absolute deviation from the clip's median
frame, restricted to the lower 72% of the image so the face does not
dominate) and keep the T frames with the highest energy, in temporal order.
Those are the frames in which the hands are moving, i.e. the sign itself.
Each kept frame is cropped to the signing hand (MediaPipe landmarks, then
skin-and-motion, then the previous frame's box).

Pretraining pool
----------------
Clips of the 1098 single-clip signs are sign-disjoint from the evaluation
set, so their frames can be used for self-supervised pretraining without
any overlap with test signs.

    python prepare_csl_signer.py
"""

import json
import os
import re
import time
from collections import defaultdict

import cv2
import numpy as np

import handcrop as hc

NEW = os.environ.get("SMIC_DATA", os.path.join("..", "new"))
OUT = os.environ.get("SMIC_PREPARED", os.path.join("..", "prepared"))
IMG = 96
T = 8                 # sign frames kept per clip
POOL_PER_CLIP = 6     # frames per clip for the unlabelled pool


def list_clips():
    pat = re.compile(r"S(\d+)_P(\d+)_T(\d+)")
    best = {}
    for root in (os.path.join(NEW, "CSL_Train"), os.path.join(NEW, "CSL_Test")):
        for d in sorted(os.listdir(root)):
            m = pat.match(d)
            if not m:
                continue
            full = os.path.join(root, d)
            fr = sorted(f for f in os.listdir(full) if f.lower().endswith((".jpg", ".png")))
            k = (m.group(1), m.group(2), m.group(3))
            # prefer the longest copy of each clip (train copies hold 30 frames)
            if k not in best or len(fr) > len(best[k]["frames"]):
                best[k] = {"sign": k[0], "person": k[1], "take": k[2],
                           "dir": full, "frames": fr}
    return list(best.values())


def read(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is not None:
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    try:                                   # some frames defeat OpenCV's decoder
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


def load_frames(c):
    fr = [read(os.path.join(c["dir"], f)) for f in c["frames"]]
    return [f for f in fr if f is not None]


def sign_frames(frames, k):
    """Indices of the k highest-motion frames, in temporal order."""
    small = [cv2.resize(f, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
             for f in frames]
    med = np.median(np.stack(small), axis=0)
    h0 = int(0.28 * 128)                     # ignore the top band (face)
    energy = np.array([np.abs(s[h0:] - med[h0:]).mean() for s in small])
    if len(frames) <= k:
        idx = np.arange(len(frames))
        idx = np.concatenate([idx, np.full(k - len(idx), idx[-1])])
    else:
        idx = np.sort(np.argsort(energy)[-k:])
    return idx, energy


def crop_frames(frames, idx, stats):
    mo = hc.clip_motion([frames[i] for i in idx])
    out, last = [], None
    for f, m in zip([frames[i] for i in idx], mo):
        box = hc.hand_box_landmarks(f, motion=m)
        how = "landmark"
        if box is None:
            box = hc.hand_box_skin(f, motion=m, exclude_top=0.28)
            how = "skin"
        if box is None and last is not None:
            box, how = last, "propagated"
        if box is None:
            H, W = f.shape[:2]
            s = int(min(H, W) * .7)
            box, how = ((W - s) / 2, (H - s) / 2, (W + s) / 2, (H + s) / 2), "center"
        else:
            last = box
        stats[how] = stats.get(how, 0) + 1
        H, W = f.shape[:2]
        x0, y0, x1, y1 = hc.expand_square(box, W, H, .22)
        cr = f[y0:y1, x0:x1]
        out.append(cv2.resize(cr if cr.size else f, (IMG, IMG), interpolation=cv2.INTER_AREA))
    return np.stack(out)


def main():
    t0 = time.time()
    clips = list_clips()
    by_sign = defaultdict(list)
    for c in clips:
        by_sign[c["sign"]].append(c)
    eval_signs = sorted(s for s, cs in by_sign.items() if len(cs) >= 2)
    pool_signs = sorted(s for s, cs in by_sign.items() if len(cs) == 1)
    print("clips %d | eval signs %d | pool signs %d" % (len(clips), len(eval_signs), len(pool_signs)))

    # ---- labelled, signer-disjoint within each sign -------------------
    # Process every clip first; unreadable clips are skipped.  The split is
    # assigned only afterwards, over the clips that survived, and a sign is
    # kept only if it still has a training clip and a test clip by a
    # different signer.
    stats, done, skipped = {}, defaultdict(list), 0
    for s in eval_signs:
        for c in sorted(by_sign[s], key=lambda c: (c["person"], c["take"])):
            frames = load_frames(c)
            if len(frames) < 2:
                skipped += 1
                continue
            idx, _ = sign_frames(frames, T)
            done[s].append((c["person"], crop_frames(frames, idx, stats)))
    kept = [s for s in eval_signs
            if len({p for p, _ in done[s]}) >= 2]
    s2i = {s: i for i, s in enumerate(kept)}
    X, y, person, split = [], [], [], []
    for s in kept:
        tp, seen_train = done[s][0][0], False
        for p, clip in done[s]:
            X.append(clip); y.append(s2i[s]); person.append(int(p))
            if p == tp and not seen_train:
                split.append("train"); seen_train = True
            elif p == tp:
                split.append("drop")        # same signer, second take: unused
            else:
                split.append("val")
    eval_signs = kept
    print("  skipped %d unreadable clips; %d signs keep a cross-signer pair"
          % (skipped, len(kept)), flush=True)
    X = np.stack(X).astype(np.uint8)
    y, person, split = np.array(y), np.array(person), np.array(split)
    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    # verify the signer-disjoint property per test clip
    for i in va:
        tr_people = set(person[tr][y[tr] == y[i]])
        assert person[i] not in tr_people, "signer overlap"
    np.savez_compressed(os.path.join(OUT, "csl_signer.npz"),
                        X=X, y=y, person=person, train_idx=tr, val_idx=va)
    print("labelled: %d clips x %d frames | train %d | val %d | classes %d | crop %s"
          % (len(X), T, len(tr), len(va), len(eval_signs), stats))

    # ---- sign-disjoint unlabelled pool for self-supervised pretraining --
    pool, pstats = [], {}
    for n, s in enumerate(pool_signs):
        c = by_sign[s][0]
        frames = load_frames(c)
        if len(frames) < 2:
            continue
        idx, _ = sign_frames(frames, POOL_PER_CLIP)
        pool.append(crop_frames(frames, idx, pstats))
        if (n + 1) % 200 == 0:
            print("  pool %d/%d (%.0fs)" % (n + 1, len(pool_signs), time.time() - t0), flush=True)
    P = np.concatenate(pool).astype(np.uint8)
    np.savez_compressed(os.path.join(OUT, "csl_pool.npz"), X=P)
    print("pool: %d frames from %d sign-disjoint clips | crop %s" % (len(P), len(pool), pstats))

    json.dump({"name": "csl_signer", "task": "one-shot, signer-disjoint within class",
               "n_classes": len(eval_signs), "T": T, "img": IMG,
               "n_train": int(len(tr)), "n_val": int(len(va)),
               "n_clips_total": len(clips), "n_signs_total": len(by_sign),
               "single_clip_signs": len(pool_signs),
               "pool_frames": int(len(P)), "crop_eval": stats, "crop_pool": pstats,
               "sign_frame_selection": "top-%d motion-energy frames, face band excluded" % T},
              open(os.path.join(OUT, "csl_signer_meta.json"), "w"), indent=2)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
