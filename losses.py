"""
losses.py -- the spectral decorrelation objective and representation
diagnostics.

Proposition 2 says the symmetric readout makes stream collapse a
critical point.  Band disjointness removes the *retention* degeneracy,
but two bands could still carry the same information at different
timescales.  The decorrelation penalty targets the remaining redundancy
directly.

For band states M^(1) .. M^(K), each (B*T, d), let Z^(k) be M^(k)
standardised over the batch-time axis.  Define the cross-band coherence

    L_dec = (1 / (K(K-1)/2)) * sum_{j<k} || Z^(j)^T Z^(k) / N ||_F^2 / d

This is the mean squared cross-correlation between units of different
bands.  It is zero exactly when the bands are mutually uncorrelated and
one when they are identical, so it doubles as an interpretable collapse
metric -- the number reported for the referenced STLAT model.

Effective rank
--------------
We report the participation ratio of the eigenspectrum of the fused
covariance,

    r_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2

which is 1 for a rank-one representation and Kd for an isotropic one.
"""

import torch


def _standardise(z, eps=1e-5):
    """Zero-mean, unit-variance per feature, over the leading axis."""
    mu = z.mean(dim=0, keepdim=True)
    sd = z.std(dim=0, keepdim=True).clamp_min(eps)
    return (z - mu) / sd


def cross_band_coherence(states, eps=1e-5, bias_correct=True):
    """Mean squared cross-correlation between distinct bands.

    Parameters
    ----------
    states : (B, T, K, d) tensor of band states.
    bias_correct : bool
        Under independence each of the d^2 sample correlations has
        E[r^2] ~ 1/N, so the raw statistic sits at d/N rather than 0.
        With ``bias_correct`` that chance level is subtracted, making 0
        mean "no more coherent than independent streams" and 1 mean
        "identical streams".  Report the corrected value; the raw value
        differs by a constant and is fine as a training penalty.

    Returns
    -------
    scalar tensor, 0 = mutually uncorrelated bands, 1 = identical bands.
    """
    B, T, K, d = states.shape
    if K < 2:
        return states.new_zeros(())
    z = states.reshape(B * T, K, d)
    N = z.shape[0]
    total, pairs = states.new_zeros(()), 0
    for j in range(K):
        zj = _standardise(z[:, j, :], eps)
        for k in range(j + 1, K):
            zk = _standardise(z[:, k, :], eps)
            c = (zj.transpose(0, 1) @ zk) / N          # (d, d)
            total = total + (c ** 2).sum() / d
            pairs += 1
    val = total / max(pairs, 1)
    if bias_correct:
        val = (val - d / max(N, 1)) / max(1.0 - d / max(N, 1), 1e-6)
        val = val.clamp_min(0.0)
    return val


def decorrelation_loss(states, eps=1e-5):
    """Training penalty: the raw (uncorrected) coherence.

    The bias correction is an affine map, so it changes the value but not
    the gradient direction; the raw form is used here to keep the penalty
    strictly non-negative and cheap.
    """
    return cross_band_coherence(states, eps, bias_correct=False)


@torch.no_grad()
def effective_rank(states, eps=1e-8):
    """Participation ratio of the fused representation's eigenspectrum.

    Parameters
    ----------
    states : (B, T, K, d) -> flattened to (B*T, K*d).

    Returns
    -------
    float in [1, K*d].
    """
    B, T, K, d = states.shape
    z = states.reshape(B * T, K * d).float()
    z = z - z.mean(dim=0, keepdim=True)
    # Covariance eigenvalues via singular values (numerically safer).
    s = torch.linalg.svdvals(z)
    lam = s ** 2
    num = lam.sum() ** 2
    den = (lam ** 2).sum().clamp_min(eps)
    return float(num / den)


@torch.no_grad()
def pairwise_state_correlation(states):
    """Mean absolute correlation between matched units of band pairs.

    Reported for the K = 2 case as the direct analogue of rho(C, M) --
    the collapse statistic for the referenced dual-memory model.
    """
    B, T, K, d = states.shape
    if K < 2:
        return 0.0
    z = states.reshape(B * T, K, d)
    out, pairs = 0.0, 0
    for j in range(K):
        zj = _standardise(z[:, j, :])
        for k in range(j + 1, K):
            zk = _standardise(z[:, k, :])
            # correlation of unit i in band j with unit i in band k
            r = (zj * zk).mean(dim=0)
            out += float(r.abs().mean())
            pairs += 1
    return out / max(pairs, 1)


@torch.no_grad()
def band_timescale_overlap(tau, band_edges=None, n_bins=64):
    """Overlap coefficient between the realised timescale distributions.

    Parameters
    ----------
    tau : (B, T, K, d) realised effective timescales.

    Returns
    -------
    float in [0, 1].  0 means the bands occupy disjoint timescale ranges;
    1 means they are indistinguishable.  For the band-constrained gate
    this is 0 by construction -- that is the certificate.  For free
    sigmoid gates it is measured, and is the evidence that unconstrained
    dual memory collapses.
    """
    B, T, K, d = tau.shape
    if K < 2:
        return 0.0
    lt = torch.log(tau.reshape(-1, K, d).clamp_min(1e-6))
    lo, hi = float(lt.min()), float(lt.max())
    if hi - lo < 1e-9:
        return 1.0
    hists = []
    for k in range(K):
        h = torch.histc(lt[:, k, :].reshape(-1).float(), bins=n_bins, min=lo, max=hi)
        hists.append(h / h.sum().clamp_min(1e-9))
    out, pairs = 0.0, 0
    for j in range(K):
        for k in range(j + 1, K):
            out += float(torch.minimum(hists[j], hists[k]).sum())
            pairs += 1
    return out / max(pairs, 1)
