"""
asfg_test.py -- the decisive test of the convex-fusion claim.

At K=2 the spectral fusion rule reduces exactly to the ASFG readout of the
referenced STLAT model:

    softmax over 2 bands   ==   alpha * tanh(C) + (1 - alpha) * tanh(M)

so comparing "softmax" against "gated" at K=2 is a direct test of the
referenced gate, not of a proxy for it.  The only difference between the two
arms is whether the two mixing weights are forced to sum to one.

If the convex constraint is what limits the gate, the non-convex arm should
win by a wide margin on a task that needs both memories at once, and the
two arms should be indistinguishable on a task that needs only one.  Both
conditions are run.
"""

import json

import numpy as np

import train

SEEDS = [42, 123, 456]
KW = dict(epochs=30, batch_size=256, lr=2e-3, hidden_size=48, num_heads=4,
          transformer_layers=2, dim_feedforward=128, dropout=0.1, verbose=False)

CONDITIONS = [
    ("needs both timescales", {"tau_s": 2, "tau_l": 16}),
    ("timescales close together", {"tau_s": 2, "tau_l": 4}),
]

ARMS = [
    ("spectral_K2", "softmax over 2 bands  == ASFG convex mixture", {}),
    ("spectral_K2_gated", "sigmoid weights, not forced to sum to 1", {}),
]

if __name__ == "__main__":
    out = {"seeds": SEEDS, "conditions": []}
    for label, lags in CONDITIONS:
        synth = dict({"n": 2500, "T": 32, "n_sym": 4, "dim": 16, "seed": 0}, **lags)
        print("\n=== %s (tau_s=%d, tau_l=%d) ==="
              % (label, lags["tau_s"], lags["tau_l"]), flush=True)
        cond = {"label": label, "lags": lags, "arms": {}}
        for nm, desc, extra in ARMS:
            rs = train.run_seeds(nm, "synthetic", seeds=SEEDS, synth_kw=synth,
                                 num_bands=2, **KW)
            a = [r["final_val_acc"] for r in rs]
            cond["arms"][nm] = {"desc": desc, "accs": a,
                                "mean": float(np.mean(a)),
                                "std": float(np.std(a, ddof=1))}
            print("  %-44s %.4f +/- %.4f"
                  % (desc, np.mean(a), np.std(a, ddof=1)), flush=True)
        m = cond["arms"]
        cond["delta"] = (m["spectral_K2_gated"]["mean"]
                         - m["spectral_K2"]["mean"])
        print("  -> non-convex minus convex: %+.4f" % cond["delta"], flush=True)
        out["conditions"].append(cond)

    with open("results/asfg_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/asfg_test.json")
