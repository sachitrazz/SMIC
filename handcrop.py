"""
handcrop.py -- crop the signing hand(s) out of each frame.

The CSL clips are upper-body shots in which the hand is a small fraction of
the pixels.  A classifier trained on the full frame can learn signer identity,
clothing or background instead of the sign, so every frame is cropped to the
signing hand before it is encoded.

Two detectors, in order of preference:

  1. MediaPipe HandLandmarker (models/hand_landmarker.task): 21 landmarks per
     hand, handles two hands.
  2. A skin-and-motion fallback, used when the model file is absent or the
     detector finds nothing.  Skin is segmented in YCrCb, which is more robust
     to illumination than RGB thresholds; for clips, the per-frame difference
     from the clip median isolates the moving hand from the static face and
     torso.

The fallback matters because hand detectors fail on motion blur, which occurs
when the hand moves fastest; dropping those frames would bias the data toward
held poses.
"""

import os

import cv2
import numpy as np

_LANDMARKER = None
_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "models", "hand_landmarker.task")


def _get_landmarker(max_hands=2):
    """Lazily build the MediaPipe detector; None if unavailable."""
    global _LANDMARKER
    if _LANDMARKER == "unavailable":
        return None
    if _LANDMARKER is not None:
        return _LANDMARKER
    try:
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        base = mpp.BaseOptions(model_asset_path=_MODEL)
        opts = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=0.25,
            min_hand_presence_confidence=0.25)
        _LANDMARKER = vision.HandLandmarker.create_from_options(opts)
    except Exception:
        _LANDMARKER = "unavailable"
        return None
    return _LANDMARKER


# ----------------------------------------------------------------------
# detector 1: landmarks
# ----------------------------------------------------------------------

def hand_box_landmarks(rgb, motion=None):
    """Box around ONE hand, or None.

    Taking the union over all detected hands was a mistake: in the CSL
    upper-body shots both hands are visible and the union spans the whole
    torso, which defeats the point of cropping.  We pick a single hand --
    the one overlapping the most motion when a motion map is supplied,
    otherwise the largest -- so the crop stays tight on the signing hand.
    """
    lm = _get_landmarker()
    if lm is None:
        return None
    try:
        import mediapipe as mp
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=np.ascontiguousarray(rgb))
        res = lm.detect(img)
    except Exception:
        return None
    hands = getattr(res, "hand_landmarks", None)
    if not hands:
        return None

    H, W = rgb.shape[:2]
    boxes = []
    for hand in hands:
        xs = [p.x * W for p in hand]
        ys = [p.y * H for p in hand]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))

    if len(boxes) == 1:
        return boxes[0]

    def score(b):
        x0, y0, x1, y1 = b
        area = max(1.0, (x1 - x0) * (y1 - y0))
        if motion is None:
            return area
        xi0, yi0 = int(max(0, x0)), int(max(0, y0))
        xi1, yi1 = int(min(W, x1)), int(min(H, y1))
        if xi1 <= xi0 or yi1 <= yi0:
            return 0.0
        return float(motion[yi0:yi1, xi0:xi1].mean()) * area

    return max(boxes, key=score)


# ----------------------------------------------------------------------
# detector 2: skin + motion fallback
# ----------------------------------------------------------------------

def skin_mask(rgb):
    """YCrCb skin segmentation -- illumination-robust, no training needed."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    m = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))


def hand_box_skin(rgb, motion=None, exclude_top=0.0):
    """Largest skin blob, optionally weighted by a motion map.

    `exclude_top` drops that fraction of the image height, which suppresses
    the face when the hand is known to sign lower in the frame.
    """
    m = skin_mask(rgb)
    H, W = m.shape
    if exclude_top > 0:
        m[:int(exclude_top * H), :] = 0
    if motion is not None:
        mv = cv2.normalize(motion, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mv = cv2.dilate(mv, np.ones((15, 15), np.uint8))
        m = cv2.bitwise_and(m, cv2.inRange(mv, 18, 255))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    # A hand occupies a modest slice of the frame.  Without an upper bound
    # the largest skin blob is the face-plus-torso, which is what the first
    # version of this fallback returned on the CSL clips.
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    i = None
    for j in order:
        a = areas[j]
        if 0.0015 * H * W <= a <= 0.10 * H * W:
            i = 1 + int(j)
            break
    if i is None:
        return None
    x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                  stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
    return (float(x), float(y), float(x + w), float(y + h))


# ----------------------------------------------------------------------

def expand_square(box, W, H, pad=0.22):
    """Pad a box and make it square, clamped to the image."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0) * (1 + pad)
    s = max(s, 24.0)
    x0, y0 = cx - s / 2, cy - s / 2
    x0 = max(0, min(x0, W - s))
    y0 = max(0, min(y0, H - s))
    s = min(s, min(W, H))
    return int(x0), int(y0), int(x0 + s), int(y0 + s)


def crop_hand(rgb, motion=None, out_size=96, pad=0.22, exclude_top=0.0,
              fallback="full"):
    """Return (crop, method).

    `fallback` decides what to do when neither detector fires.  "full" keeps
    the whole frame (square-padded); "center" takes the middle 70%.

    "full" is the right default for still-image alphabet sets.  Those images
    are often already tight shots of a hand, so the detector fails simply
    because there is no wrist or forearm for it to anchor on -- and a 70%
    centre crop of an already-tight image zooms into a knuckle and upsamples
    it.  That is what happened to 270 of ASL's 419 images.  For upper-body
    video frames, where a failure means the hand really was not located, a
    centre crop is the better guess and the callers there pass "center".
    """
    H, W = rgb.shape[:2]
    box = hand_box_landmarks(rgb, motion=motion)
    method = "landmark"
    if box is None:
        box = hand_box_skin(rgb, motion=motion, exclude_top=exclude_top)
        method = "skin"
    if box is None:
        if fallback == "full":
            side = min(H, W)
            box = ((W - side) / 2, (H - side) / 2, (W + side) / 2, (H + side) / 2)
            method = "full"
        else:
            side = int(min(H, W) * 0.7)
            box = ((W - side) / 2, (H - side) / 2, (W + side) / 2, (H + side) / 2)
            method = "center"
    x0, y0, x1, y1 = expand_square(box, W, H, 0.0 if method == "full" else pad)
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        crop = rgb
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA), method


def clip_motion(frames):
    """Per-frame absolute deviation from the clip's median frame."""
    med = np.median(np.stack(frames), axis=0).astype(np.float32)
    return [np.abs(f.astype(np.float32) - med).mean(axis=2) for f in frames]
