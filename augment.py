"""
augment.py -- batched affine jitter on the GPU.

Per-sample PIL transforms cost roughly 25 s/epoch on the ISL split, several
times the cost of the forward and backward passes combined.  The same
jitter applied as one batched ``grid_sample`` is essentially free.

The sequence layout is preserved: a batch of shape (B, T, strip) is
reshaped into the image it tiles, transformed, and reshaped back, so the
scan semantics the model sees are unchanged.

No horizontal flip: many sign pairs are distinguished by handedness, so
mirroring would change the label.
"""

import math

import torch
import torch.nn.functional as F


def affine_jitter(x, img_size, channels=1, deg=12.0, shift=0.12, scale=0.12,
                  generator=None):
    """Random rotation / translation / scale applied per sample.

    Parameters
    ----------
    x : (B, T, strip) tensor whose rows tile an img_size x img_size image.

    Returns
    -------
    tensor of the same shape as ``x``.
    """
    B, T, _ = x.shape
    H = W = img_size
    dev, dt = x.device, x.dtype

    img = x.reshape(B, H, W, channels).permute(0, 3, 1, 2)

    def r(lo, hi):
        u = torch.rand(B, device=dev, dtype=dt, generator=generator)
        return lo + (hi - lo) * u

    ang = r(-deg, deg) * math.pi / 180.0
    sc = r(1.0 - scale, 1.0 + scale)
    tx = r(-shift, shift) * 2.0          # grid coords span [-1, 1]
    ty = r(-shift, shift) * 2.0

    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(B, 2, 3, device=dev, dtype=dt)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty

    grid = F.affine_grid(theta, img.shape, align_corners=False)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out.permute(0, 2, 3, 1).reshape(B, T, -1)
