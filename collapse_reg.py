"""
collapse_reg.py -- a regulariser that actively repels the collapse critical set.

Motivation
----------
Proposition 2 establishes that {C == M, alpha == 1/2} is a critical
submanifold of the training objective: the gradient component that would
separate the two memory streams vanishes identically on it.

Separating the input projections moves the *initialisation* off that set,
which is why it helps.  But the set itself is still there in the loss
landscape, and nothing in the objective pushes away from it.  Whether
training drifts back toward it is left to chance.

The principled fix is a term that makes collapse explicitly costly.  This
is a solved problem in self-supervised learning, where the identical
failure mode -- two branches converging to the same constant -- is
prevented by variance and covariance penalties (Barlow Twins, VICReg).
That machinery has not, to our knowledge, been applied to the memory
streams of a recurrent network, where Proposition 2 says the same
degeneracy is not merely possible but structurally favoured.

Three terms, each targeting a distinct failure
----------------------------------------------
1. VARIANCE.  Per-dimension standard deviation of each stream is pushed
   above a floor.  This prevents *dimensional* collapse -- entire units
   going constant -- which is what drives the effective rank down. It is
   the term that directly targets the statistic we measured at 0.18.

2. COVARIANCE.  Off-diagonal covariance within a stream is penalised, so
   units inside one memory do not duplicate each other.

3. CROSS-STREAM.  Correlation between the two streams is penalised, which
   is the collapse of Proposition 2 proper.

Only (3) targets the theorem directly; (1) and (2) matter because a stream
that has itself collapsed to a low-rank subspace cannot be decorrelated
from anything in a meaningful way.

Cost: no new parameters, one pass over the state tensor.
"""

import torch
import torch.nn.functional as F


def _flatten(states):
    """(B, T, K, d) -> list of K tensors, each (B*T, d)."""
    B, T, K, d = states.shape
    z = states.reshape(B * T, K, d)
    return [z[:, k, :] for k in range(K)], B * T, d


def variance_term(zs, gamma=1.0, eps=1e-4):
    """Hinge each dimension's std above `gamma`.

    Zero once every unit varies enough; grows as units go constant.
    """
    out = 0.0
    for z in zs:
        std = torch.sqrt(z.var(dim=0) + eps)
        out = out + F.relu(gamma - std).mean()
    return out / max(len(zs), 1)


def covariance_term(zs):
    """Penalise off-diagonal covariance inside each stream."""
    out = 0.0
    for z in zs:
        N, d = z.shape
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(N - 1, 1)
        off = cov - torch.diag_embed(torch.diagonal(cov))
        out = out + (off ** 2).sum() / d
    return out / max(len(zs), 1)


def cross_stream_term(zs, eps=1e-5):
    """Penalise correlation between distinct streams (Proposition 2)."""
    K = len(zs)
    if K < 2:
        return zs[0].new_zeros(())
    norm = []
    for z in zs:
        zc = z - z.mean(dim=0, keepdim=True)
        norm.append(zc / zc.std(dim=0, keepdim=True).clamp_min(eps))
    N, d = norm[0].shape
    out, pairs = 0.0, 0
    for j in range(K):
        for k in range(j + 1, K):
            c = (norm[j].T @ norm[k]) / N
            out = out + (c ** 2).sum() / d
            pairs += 1
    return out / max(pairs, 1)


def collapse_regulariser(states, w_var=1.0, w_cov=0.04, w_cross=1.0, gamma=1.0):
    """Combined penalty.  `states` is (B, T, K, d).

    Default weights follow the VICReg convention of a small covariance
    weight relative to variance; w_cross is ours and is set equal to the
    variance weight because it is the term the theorem points at.
    """
    zs, N, d = _flatten(states)
    v = variance_term(zs, gamma=gamma)
    c = covariance_term(zs)
    x = cross_stream_term(zs)
    return w_var * v + w_cov * c + w_cross * x, {
        "var": float(v.detach()) if torch.is_tensor(v) else float(v),
        "cov": float(c.detach()) if torch.is_tensor(c) else float(c),
        "cross": float(x.detach()) if torch.is_tensor(x) else float(x),
    }
