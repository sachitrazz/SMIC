"""
cascade_test.py -- removing the exchange symmetry by changing the topology.

Two repairs for the collapse described by Proposition 2, compared on the same
seeds:

  unshared   separate W_x per stream.  The initialisation is no longer
             symmetric, but the invariant set still exists in the landscape.
  cascade    M is driven by C_t rather than X_t.  The two streams occupy
             different positions in the computation graph, so the exchange
             is not expressible and the invariant set is empty.  No capacity
             is removed: every gate keeps full freedom.

Band tiling, the third repair, is run by positive_test.py and asfg_test.py.

Both the synthetic two-cue task and ISL-IEEE are supported, because the first
requires two timescales by construction and the second does not.

    python cascade_test.py synthetic
    python cascade_test.py isl
    python cascade_test.py both
"""

import json
import sys

import numpy as np
from scipy import stats

import train

SEEDS = [42, 123, 456, 789, 1024]

SYNTH_KW = dict(epochs=30, batch_size=256, lr=2e-3, hidden_size=48, num_heads=4,
                transformer_layers=2, dim_feedforward=128, dropout=0.1,
                verbose=False)
SYNTH = {"n": 2500, "T": 32, "tau_s": 2, "tau_l": 16,
         "n_sym": 4, "dim": 16, "seed": 0}

ISL_KW = dict(epochs=60, batch_size=32, lr=1e-3, hidden_size=64, num_heads=4,
              transformer_layers=2, dim_feedforward=128, dropout=0.2,
              seq_len=8, img_size=32, verbose=False)

ARMS = [
    ("original_asfg", "referenced (shared W_x)"),
    ("original_asfg_unshared", "separate W_x"),
    ("original_asfg_cascade", "CASCADE (M reads C)"),
]


def run(dataset):
    kw = dict(SYNTH_KW, synth_kw=SYNTH) if dataset == "synthetic" else dict(ISL_KW)
    print("\n" + "=" * 68)
    print("%s   %d shared seeds" % (dataset.upper(), len(SEEDS)))
    print("=" * 68, flush=True)
    print("%-24s %-9s %-9s %-9s %-9s" % ("arm", "acc", "sd", "overlap", "eff-rank"),
          flush=True)

    out = {"dataset": dataset, "seeds": SEEDS, "arms": {}}
    for name, desc in ARMS:
        rs = train.run_seeds(name, dataset, seeds=SEEDS, **kw)
        a = [r["final_val_acc"] for r in rs]
        ov = [r["tau_overlap"] for r in rs if r["tau_overlap"] is not None]
        er = [r["eff_rank_frac"] for r in rs if r["eff_rank_frac"] is not None]
        co = [r["coherence"] for r in rs if r["coherence"] is not None]
        out["arms"][name] = {
            "desc": desc, "accs": a,
            "mean": float(np.mean(a)), "std": float(np.std(a, ddof=1)),
            "tau_overlap": float(np.mean(ov)) if ov else None,
            "eff_rank_frac": float(np.mean(er)) if er else None,
            "coherence": float(np.mean(co)) if co else None,
            "params": rs[0]["params"],
        }
        r = out["arms"][name]
        print("%-24s %-9.4f %-9.4f %-9.3f %-9.3f"
              % (desc, r["mean"], r["std"], r["tau_overlap"], r["eff_rank_frac"]),
              flush=True)

    base = np.array(out["arms"]["original_asfg"]["accs"])
    print("\npaired t-tests vs referenced:", flush=True)
    out["tests"] = []
    for name, _ in ARMS[1:]:
        arr = np.array(out["arms"][name]["accs"])
        t, p = stats.ttest_rel(arr, base)
        d = arr - base
        dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
        out["tests"].append({"arm": name, "delta": float(arr.mean() - base.mean()),
                             "p": float(p), "dz": dz})
        print("  %-24s %+.4f   p=%.4f   d_z=%+.2f"
              % (out["arms"][name]["desc"], arr.mean() - base.mean(), p, dz),
              flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    todo = ["synthetic", "isl"] if which == "both" else [which]
    allout = {}
    for ds in todo:
        allout[ds] = run(ds)
        with open("results/cascade_test.json", "w") as f:
            json.dump(allout, f, indent=2)
    print("\nwrote results/cascade_test.json")
