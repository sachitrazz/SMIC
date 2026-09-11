"""
data.py -- datasets for the Spectral Memory Filter Bank experiments.

Three sources:

1.  SyntheticMultiTimescale -- a controlled task whose required
    timescales are known by construction.  This is what lets us *test*
    Propositions 1 and 3 rather than merely assert them.

2.  FrameSequenceDataset -- genuine temporal sequences from a directory
    of per-gesture frame folders (the CSL layout).

3.  ScanSequenceDataset -- static sign images turned into sequences by a
    spatial raster scan (the ISL / ASL layout).

For (3) the sequence axis is *spatial*, not temporal: the model reads the
image as a sequence of horizontal strips.  This is the standard
sequential-image protocol and a legitimate test of multi-timescale
integration, but it is not video.  Feeding single frames through an LSTM
with T = 1 would make the entire recurrent apparatus inert.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm")


# ======================================================================
# 1.  Synthetic multi-timescale benchmark
# ======================================================================

class SyntheticMultiTimescale(Dataset):
    """Sequence classification requiring two *specified* timescales.

    Each sequence has length T and carries two cues:

      * a SHORT cue: a marker placed at lag ``tau_s`` before the end;
      * a LONG  cue: a marker placed at lag ``tau_l`` before the end.

    The label is the XOR-style combination of the two cue values, so the
    model must retain *both* to be correct: a model that resolves only
    one timescale is capped at 50% accuracy (or at 1/n_classes when
    ``n_sym`` > 2).

    This is the cleanest possible probe of Proposition 1.  Sweeping the
    ratio tau_l / tau_s sweeps the quantity D = ln(tau_l/tau_s) that the
    proposition says drives the shared-gate penalty, so we can check the
    predicted quadratic growth directly.

    Parameters
    ----------
    n : int              number of sequences
    T : int              sequence length
    tau_s, tau_l : int   lags (from the end) of the short and long cues
    n_sym : int          alphabet size for the cues (labels = n_sym)
    dim : int            observation dimension
    noise : float        std of the distractor noise filling all other steps
    seed : int
    """

    def __init__(self, n=4000, T=64, tau_s=2, tau_l=48, n_sym=4, dim=16,
                 noise=1.0, seed=0):
        assert 0 < tau_s < tau_l < T, "need 0 < tau_s < tau_l < T"
        self.T, self.dim, self.n_sym = T, dim, n_sym
        self.tau_s, self.tau_l = tau_s, tau_l

        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, noise, size=(n, T, dim)).astype(np.float32)

        sym_s = rng.integers(0, n_sym, size=n)
        sym_l = rng.integers(0, n_sym, size=n)

        # Cue markers live in dedicated channels so the task is about
        # *retention*, not about disentangling the observation.
        s_idx, l_idx = T - tau_s, T - tau_l
        x[:, s_idx, :] = 0.0
        x[:, l_idx, :] = 0.0
        for i in range(n):
            x[i, s_idx, sym_s[i] % dim] = 4.0
            x[i, l_idx, (n_sym + sym_l[i]) % dim] = 4.0

        y = (sym_s + sym_l) % n_sym

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# ======================================================================
# 2.  Real frame sequences (CSL layout)
# ======================================================================

def _list_images(d):
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith(IMG_EXT) and not f.startswith("."))


class FrameSequenceDataset(Dataset):
    """Per-class folders of ordered frames -> fixed-length sequences.

    Directory layout::

        root/
          gesture_type_00/000027.jpg 000028.jpg ...
          gesture_type_01/...

    Frames inside a class folder are sorted by filename and split into
    contiguous clips of ``seq_len`` frames.  Clips shorter than
    ``seq_len`` are padded by repeating the last frame, which is the
    standard convention and is recorded in ``self.padded``.
    """

    def __init__(self, root, seq_len=8, img_size=32, grayscale=True,
                 stride=None, min_frames=2):
        self.root = root
        self.seq_len = seq_len
        self.img_size = img_size
        self.grayscale = grayscale
        stride = stride or seq_len

        classes = sorted(d for d in os.listdir(root)
                         if os.path.isdir(os.path.join(root, d))
                         and not d.startswith("."))
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.clips, self.labels, self.padded = [], [], []
        for c in classes:
            cdir = os.path.join(root, c)
            frames = [os.path.join(cdir, f) for f in _list_images(cdir)]
            if len(frames) < min_frames:
                continue
            for start in range(0, max(1, len(frames) - seq_len + 1), stride):
                clip = frames[start:start + seq_len]
                pad = seq_len - len(clip)
                if pad > 0:
                    clip = clip + [clip[-1]] * pad
                self.clips.append(clip)
                self.labels.append(self.class_to_idx[c])
                self.padded.append(pad)

        self.input_size = img_size * img_size * (1 if grayscale else 3)

    def __len__(self):
        return len(self.clips)

    def _load(self, path):
        im = Image.open(path)
        im = im.convert("L" if self.grayscale else "RGB")
        im = im.resize((self.img_size, self.img_size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        if a.ndim == 2:
            a = a[:, :, None]
        return a.reshape(-1)

    def __getitem__(self, i):
        seq = np.stack([self._load(p) for p in self.clips[i]], axis=0)
        seq = (seq - seq.mean()) / (seq.std() + 1e-6)
        return torch.from_numpy(seq.astype(np.float32)), self.labels[i]


# ======================================================================
# 3.  Static sign images as spatial scan sequences (ISL / ASL layout)
# ======================================================================

class ScanSequenceDataset(Dataset):
    """Class-per-folder still images -> sequences of horizontal strips.

    Directory layout::

        root/A/A (1).jpg ...
        root/B/...

    Each image is resized to ``img_size`` x ``img_size`` and split into
    ``seq_len`` horizontal strips read top to bottom, giving a sequence
    of length ``seq_len`` whose elements have dimension
    (img_size / seq_len) * img_size * channels.

    The scan axis is spatial.  Multi-timescale integration is still
    meaningful -- adjacent strips are short-range structure, the whole
    hand shape is long-range -- but this is not video.
    """

    def __init__(self, root, seq_len=8, img_size=32, grayscale=True, cache=True,
                 augment=False, aug_deg=12.0, aug_shift=0.12, aug_scale=0.12,
                 aug_seed=0):
        self.root = root
        self.seq_len = seq_len
        self.img_size = img_size
        self.grayscale = grayscale
        # Augmentation is essential at this dataset size: without it the
        # model reaches >0.94 train accuracy and ~0.16 validation accuracy,
        # i.e. it memorises the ~23 training images per class outright.
        self.augment = augment
        self.aug_deg, self.aug_shift, self.aug_scale = aug_deg, aug_shift, aug_scale
        self._rng = np.random.default_rng(aug_seed)
        assert img_size % seq_len == 0, "img_size must be divisible by seq_len"

        classes = sorted(d for d in os.listdir(root)
                         if os.path.isdir(os.path.join(root, d))
                         and not d.startswith("."))
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.files, self.labels = [], []
        for c in classes:
            cdir = os.path.join(root, c)
            for f in _list_images(cdir):
                self.files.append(os.path.join(cdir, f))
                self.labels.append(self.class_to_idx[c])

        ch = 1 if grayscale else 3
        self.strip_h = img_size // seq_len
        self.input_size = self.strip_h * img_size * ch

        # Decode once and keep the PIL images in memory.  These datasets are
        # small, and re-decoding every JPEG on every epoch dominated the
        # runtime (roughly 4x the cost of the forward and backward passes).
        self._imgs = [self._open(i) for i in range(len(self.files))] if cache else None

    def __len__(self):
        return len(self.files)

    def _open(self, i):
        im = Image.open(self.files[i])
        return im.convert("L" if self.grayscale else "RGB")

    def _to_seq(self, im):
        im = im.resize((self.img_size, self.img_size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        if a.ndim == 2:
            a = a[:, :, None]
        a = (a - a.mean()) / (a.std() + 1e-6)
        return torch.from_numpy(a.reshape(self.seq_len, -1).astype(np.float32))

    def _augment(self, im):
        """Random affine jitter.  No horizontal flip: many sign pairs are
        distinguished by handedness, so mirroring changes the label."""
        r = self._rng
        deg = float(r.uniform(-self.aug_deg, self.aug_deg))
        sc = float(r.uniform(1.0 - self.aug_scale, 1.0 + self.aug_scale))
        w, h = im.size
        dx = float(r.uniform(-self.aug_shift, self.aug_shift)) * w
        dy = float(r.uniform(-self.aug_shift, self.aug_shift)) * h
        im = im.rotate(deg, resample=Image.BILINEAR, fillcolor=None)
        nw, nh = max(4, int(w * sc)), max(4, int(h * sc))
        im = im.resize((nw, nh), Image.BILINEAR)
        out = Image.new(im.mode, (w, h), color=int(np.asarray(im).mean()))
        out.paste(im, (int((w - nw) / 2 + dx), int((h - nh) / 2 + dy)))
        return out

    def __getitem__(self, i):
        im = self._imgs[i] if self._imgs is not None else self._open(i)
        if self.augment:
            im = self._augment(im)
        return self._to_seq(im), self.labels[i]


# ======================================================================
# Splitting helper
# ======================================================================

def stratified_split(labels, val_frac=0.25, seed=0):
    """Indices for a class-balanced train/val split.

    Stratification matters on datasets this small: a random split can
    leave a class with zero training examples.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * len(idx))))
        n_val = min(n_val, len(idx) - 1) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def describe(ds, name=""):
    labels = np.asarray(ds.labels) if hasattr(ds, "labels") else ds.y.numpy()
    counts = np.bincount(labels)
    x, _ = ds[0]
    return ("%-28s n=%-6d classes=%-4d seq_len=%-4d input=%-6d "
            "min/class=%d max/class=%d"
            % (name, len(ds), len(counts), x.shape[0], x.shape[1],
               counts.min(), counts.max()))
