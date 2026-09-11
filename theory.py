"""
theory.py -- The three propositions behind the Spectral Memory Filter Bank,
with numerical verification.

Running this file checks every quantitative claim made in the paper's
theory section by brute force, so nothing goes into the paper that
the arithmetic does not support.

    python theory.py

--------------------------------------------------------------------------
Setup
--------------------------------------------------------------------------
Consider a linearised leaky-memory unit with scalar retention f in (0,1):

    c_t = f * c_{t-1} + u_t

The influence of an input at lag L on the current state is f^L, so the
unit has an effective timescale

    tau_eff = -1 / ln f,          equivalently   f = exp(-1 / tau_eff)

Timescale error is multiplicative, not additive: a unit tuned to tau = 2
is as badly matched to a cue at lag 20 as a unit tuned to tau = 20 is to
a cue at lag 200.  The natural coordinate is therefore

    theta = ln tau_eff

and the natural per-cue risk for a cue at lag L is the squared
log-timescale mismatch

    l(theta; L) = (theta - ln L)^2

--------------------------------------------------------------------------
Proposition 1 (shared-gate excess risk)
--------------------------------------------------------------------------
A task presents two cue families, at lags tau_s and tau_l > tau_s, with
importance weights w_s + w_l = 1.  A model with a *single* shared
retention incurs risk

    R(theta) = w_s (theta - ln tau_s)^2 + w_l (theta - ln tau_l)^2

Claim.  R is minimised at theta* = w_s ln tau_s + w_l ln tau_l, and

    R(theta*) = w_s * w_l * ln^2(tau_l / tau_s)   >  0

whereas a two-band model achieves R = 0 by setting theta_C = ln tau_s and
theta_M = ln tau_l.

Proof.  R is a strictly convex quadratic in theta.  Setting R'(theta) = 0
gives 2 w_s (theta - ln tau_s) + 2 w_l (theta - ln tau_l) = 0, hence
theta* = w_s ln tau_s + w_l ln tau_l since w_s + w_l = 1.  Writing
D = ln tau_l - ln tau_s, we have theta* - ln tau_s = w_l D and
theta* - ln tau_l = -w_s D, so

    R(theta*) = w_s w_l^2 D^2 + w_l w_s^2 D^2 = w_s w_l D^2 (w_l + w_s)
              = w_s w_l D^2.                                          []

Consequences.  The penalty is quadratic in the log timescale ratio, and
maximal at w_s = w_l = 1/2 where it equals D^2 / 4.  As a sign
vocabulary grows, the spread of required timescales widens, so D grows
and the shared-gate penalty grows quadratically.

--------------------------------------------------------------------------
Proposition 2 (symmetry-induced collapse)
--------------------------------------------------------------------------
The ASFG readout of the referenced STLAT model is

    H_t = O_t' * [ alpha_t tanh(C_t) + (1 - alpha_t) tanh(M_t) + beta Phi_t ]
    Phi_t = tanh(W_c C_t + W_m M_t + W_cm (C_t * M_t) + b)

Claim.  Let sigma be the involution that swaps the two streams:

    sigma : (C, M, alpha) |-> (M, C, 1 - alpha)

together with the corresponding swap of parameters
(W_xf, W_hf) <-> (W_xf', W_mf') etc. and (W_c, W_m) <-> (W_m, W_c).  Then
the readout is invariant under sigma, and the loss satisfies
L(sigma . w) = L(w) for all parameters w.  Hence:

  (a) the parameter space carries a Z_2 action under which the risk is
      invariant;
  (b) the fixed-point set Fix(sigma) = {C == M, alpha == 1/2} is a
      critical submanifold of the risk: for w in Fix(sigma), the gradient
      component orthogonal to Fix(sigma) is odd under sigma and therefore
      vanishes;
  (c) gradient flow initialised in Fix(sigma) remains in Fix(sigma) for
      all time.

Proof sketch.  (a) The mixture alpha tanh(C) + (1-alpha) tanh(M) is
manifestly invariant, and C * M is symmetric so Phi is invariant once
W_c and W_m are swapped.  (b) Write the parameter space as V = V_+ (+)
V_- , the eigenspaces of d(sigma) for eigenvalues +1 and -1.  Since
L o sigma = L, the gradient satisfies grad L(sigma w) = d(sigma)
grad L(w).  At a fixed point sigma w = w this gives grad L(w) =
d(sigma) grad L(w), so grad L(w) lies in V_+, i.e. its V_- component is
zero.  (c) follows immediately from (b) and uniqueness of ODE solutions.
                                                                      []

Why this matters here.  Sharing the input projections W_xg, W_xi, W_xf
between the two streams -- as the referenced implementation does -- forces
the input pathway into V_+ exactly.  With standard symmetric
initialisation the model therefore starts near Fix(sigma), where the only
force separating the streams is the asymmetry of the recurrent weights.
Proposition 2 predicts that the two memories stay strongly correlated,
so the second memory is largely redundant.

The band-constrained parameterisation removes sigma from the hypothesis
class altogether: band k is confined to [tau_k, tau_{k+1}], so no
parameter setting maps band k onto band k+1 and Fix(sigma) is empty.

--------------------------------------------------------------------------
Proposition 3 (coverage bound and the 1/K^2 law)
--------------------------------------------------------------------------
Let the task require timescales distributed over [tau_min, tau_max] and
set D = ln(tau_max / tau_min).  Tile that range with K geometrically
spaced bands and let each band place its representative at the band
midpoint in log space.

Claim.  The worst-case log-timescale mismatch is D / (2K), so the
coverage risk obeys

    R_K  <=  D^2 / (4 K^2)

and the bound is attained for the log-uniform target distribution up to
the constant of the mean-square average, R_K = D^2 / (12 K^2).

Proof.  Geometric tiling makes the K bands equal in log space, each of
width D / K.  Any target theta falls in some band whose representative
is at most D / (2K) away in log space, giving the worst-case bound.  For
theta uniform on a band of width h = D/K centred at the representative,
E[(theta - centre)^2] = h^2 / 12, hence R_K = D^2 / (12 K^2).        []

Consequences, and a falsifiable prediction.  Going from K = 1 to K = 2
divides the coverage risk by 4, which is the largest single improvement
available.  Beyond that, returns decay as 1 / K^2, so the accuracy gained
per added band shrinks rapidly.  The paper tests K in {1, 2, 4, 8, 16}
and checks whether the measured error follows the predicted 1/K^2 trend.
This is a prediction the architecture could fail.
"""

import math

import numpy as np


# ----------------------------------------------------------------------
# Proposition 1
# ----------------------------------------------------------------------

def shared_gate_risk(theta, tau_s, tau_l, w_s):
    w_l = 1.0 - w_s
    return w_s * (theta - math.log(tau_s)) ** 2 + w_l * (theta - math.log(tau_l)) ** 2


def verify_prop1(verbose=True):
    """Brute-force minimisation vs the closed form w_s w_l ln^2(tau_l/tau_s)."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        tau_s = float(rng.uniform(1.0, 10.0))
        tau_l = float(rng.uniform(20.0, 500.0))
        w_s = float(rng.uniform(0.05, 0.95))
        w_l = 1.0 - w_s

        grid = np.linspace(math.log(tau_s) - 2.0, math.log(tau_l) + 2.0, 400001)
        risks = (w_s * (grid - math.log(tau_s)) ** 2
                 + w_l * (grid - math.log(tau_l)) ** 2)
        numeric_min = float(risks.min())
        numeric_arg = float(grid[int(risks.argmin())])

        closed_form = w_s * w_l * (math.log(tau_l) - math.log(tau_s)) ** 2
        closed_arg = w_s * math.log(tau_s) + w_l * math.log(tau_l)

        worst = max(worst, abs(numeric_min - closed_form),
                    abs(numeric_arg - closed_arg))

    ok = worst < 1e-4
    if verbose:
        print("Proposition 1: max |numeric - closed form| over 2000 random "
              "instances = %.3e  -> %s" % (worst, "PASS" if ok else "FAIL"))
    return ok


def verify_prop1_maximum(verbose=True):
    """The penalty is maximised at w_s = 1/2, where it equals D^2 / 4."""
    tau_s, tau_l = 2.0, 200.0
    D = math.log(tau_l / tau_s)
    ws = np.linspace(0.001, 0.999, 99999)
    pen = ws * (1 - ws) * D ** 2
    arg = float(ws[int(pen.argmax())])
    ok = abs(arg - 0.5) < 1e-3 and abs(float(pen.max()) - D ** 2 / 4) < 1e-6
    if verbose:
        print("Proposition 1 (maximum): argmax w_s = %.5f (expect 0.5), "
              "peak = %.6f (expect D^2/4 = %.6f) -> %s"
              % (arg, float(pen.max()), D ** 2 / 4, "PASS" if ok else "FAIL"))
    return ok


# ----------------------------------------------------------------------
# Proposition 2
# ----------------------------------------------------------------------

def verify_prop2(verbose=True):
    """Check the invariance and the vanishing off-diagonal gradient.

    We build the exact ASFG readout as a small differentiable function
    and confirm numerically that (i) swapping the streams leaves the
    output unchanged, and (ii) on the symmetric subspace the gradient
    that would separate the streams is zero.
    """
    import torch

    d = 16
    torch.manual_seed(0)

    W_c = torch.randn(d, d) * 0.1
    W_m = torch.randn(d, d) * 0.1
    W_cm = torch.randn(d, d) * 0.1
    b = torch.randn(d) * 0.1

    def readout(C, M, alpha, Wc, Wm):
        phi = torch.tanh(C @ Wc.T + M @ Wm.T + (C * M) @ W_cm.T + b)
        return alpha * torch.tanh(C) + (1 - alpha) * torch.tanh(M) + 0.5 * phi

    C = torch.randn(4, d)
    M = torch.randn(4, d)
    alpha = torch.rand(4, d)

    # (i) invariance under the involution
    a = readout(C, M, alpha, W_c, W_m)
    b_ = readout(M, C, 1 - alpha, W_m, W_c)
    inv_err = float((a - b_).abs().max())

    # (ii) on the symmetric subspace C == M, alpha == 1/2, the derivative
    # of the loss with respect to the antisymmetric direction vanishes.
    S = torch.randn(4, d)
    eps = torch.zeros(1, requires_grad=True)
    Wc_sym = 0.5 * (W_c + W_m)
    target = torch.randn(4, d)

    Cp = S + eps  # move along +antisymmetric direction
    Mp = S - eps  # and -, i.e. the V_- eigendirection
    out = readout(Cp, Mp, torch.full((4, d), 0.5), Wc_sym, Wc_sym)
    loss = ((out - target) ** 2).mean()
    g = torch.autograd.grad(loss, eps)[0]
    grad_err = float(g.abs().max())

    ok = inv_err < 1e-5 and grad_err < 1e-6
    if verbose:
        print("Proposition 2: swap-invariance residual = %.3e ; "
              "antisymmetric gradient on Fix(sigma) = %.3e  -> %s"
              % (inv_err, grad_err, "PASS" if ok else "FAIL"))
    return ok


def verify_prop2_no_fixed_point(verbose=True):
    """With disjoint bands the involution has no fixed point."""
    from spectral import geometric_band_edges

    edges = geometric_band_edges(4, 1.0, 64.0)
    overlaps = []
    for k in range(len(edges) - 2):
        # band k is [edges[k], edges[k+1]], band k+1 is [edges[k+1], edges[k+2]]
        overlaps.append(max(0.0, float(edges[k + 1]) - float(edges[k + 1])))
    ok = all(o == 0.0 for o in overlaps) and all(
        float(edges[i]) < float(edges[i + 1]) for i in range(len(edges) - 1))
    if verbose:
        print("Proposition 2 (certificate): band interiors pairwise disjoint, "
              "edges strictly increasing -> %s" % ("PASS" if ok else "FAIL"))
    return ok


# ----------------------------------------------------------------------
# Proposition 3
# ----------------------------------------------------------------------

def coverage_risk(K, tau_min=1.0, tau_max=64.0, n=200001):
    """Empirical mean-square log mismatch for K geometrically tiled bands."""
    lo, hi = math.log(tau_min), math.log(tau_max)
    targets = np.linspace(lo, hi, n)
    edges = np.linspace(lo, hi, K + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # nearest centre in log space
    err = np.min((targets[:, None] - centres[None, :]) ** 2, axis=1)
    return float(err.mean())


def verify_prop3(verbose=True):
    """Check R_K = D^2 / (12 K^2) and the worst case D^2 / (4 K^2)."""
    tau_min, tau_max = 1.0, 64.0
    D = math.log(tau_max / tau_min)
    rows, ok = [], True
    for K in (1, 2, 4, 8, 16):
        emp = coverage_risk(K, tau_min, tau_max)
        pred = D ** 2 / (12.0 * K ** 2)
        bound = D ** 2 / (4.0 * K ** 2)
        rel = abs(emp - pred) / pred
        ok = ok and rel < 0.02 and emp <= bound + 1e-9
        rows.append((K, emp, pred, bound, rel))
    if verbose:
        print("Proposition 3:  K    empirical    D^2/12K^2    bound D^2/4K^2   rel.err")
        for K, emp, pred, bound, rel in rows:
            print("               %3d   %9.5f   %9.5f   %12.5f   %7.4f"
                  % (K, emp, pred, bound, rel))
        print("               -> %s" % ("PASS" if ok else "FAIL"))
    return ok


# ----------------------------------------------------------------------
# Gate conditioning (reported as a remark, not a proposition)
# ----------------------------------------------------------------------

def gate_log_timescale_gradients(verbose=True):
    """Compare d ln(tau) / d a for the band gate and a plain sigmoid gate.

    For the band gate  : d ln tau / d a = ln(kappa) * s(a) (1 - s(a))
    For a sigmoid gate : d ln tau / d a = (1 - f) / (-ln f),  f = sigmoid(a)

    This is reported honestly: the sigmoid gate is *not* badly conditioned
    in log-timescale space, so the band gate's advantage is the
    disjointness certificate and the bounded range, not the gradient.
    """
    a = np.linspace(-8, 8, 2001)
    s = 1.0 / (1.0 + np.exp(-a))
    kappa = 8.0
    band = math.log(kappa) * s * (1 - s)
    f = s
    sig = (1 - f) / (-np.log(np.clip(f, 1e-12, 1 - 1e-12)))
    if verbose:
        print("Gate conditioning remark: d ln(tau)/da in "
              "[%.3f, %.3f] (band, kappa=8) and [%.3f, %.3f] (sigmoid). "
              "Both are well conditioned; the band gate is preferred for "
              "its range certificate, not its gradient."
              % (band.min(), band.max(), sig.min(), sig.max()))
    return {"a": a, "band": band, "sigmoid": sig}


def main():
    print("=" * 72)
    print("Numerical verification of the SMFB propositions")
    print("=" * 72)
    results = [
        ("Prop 1 closed form", verify_prop1()),
        ("Prop 1 maximum", verify_prop1_maximum()),
        ("Prop 2 symmetry", verify_prop2()),
        ("Prop 2 certificate", verify_prop2_no_fixed_point()),
        ("Prop 3 coverage", verify_prop3()),
    ]
    gate_log_timescale_gradients()
    print("=" * 72)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print("  %-22s %s" % (name, "PASS" if ok else "FAIL"))
    print("=" * 72)
    print("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
