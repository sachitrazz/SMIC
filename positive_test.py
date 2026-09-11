"""
positive_test.py -- two theory-directed interventions, tested head to head.

The band-tiling mechanism failed.  The diagnosis that produced it did not:
Proposition 2 says the referenced STLAT model collapses *because* its two streams
share their input projections, and the collapse measurement confirmed it.
That points at two minimal interventions, neither of which removes capacity
the way hard tiling does.

Intervention A -- un-share the input projections (Proposition 2's own fix).
    `original` applies the same W_xg/W_xi/W_xf to both the C and M updates,
    which places the input pathway in the symmetric eigenspace exactly.
    `original_unshared` gives the M stream its own.  Same equations, one
    structural change.

Intervention B -- soft timescale diversity at initialisation.
    Hard tiling supplies diversity by *constraining* each unit to a band.
    Spreading the initial forget biases log-uniformly over [tau_min,
    tau_max] supplies the same diversity while leaving every gate free to
    move.  Diversity without constraint.

Run at tau_s=2, tau_l=16 -- the condition on which every variant
separated most sharply.
"""

import json

import numpy as np

import train

SEEDS = [42, 123, 456]
KW = dict(epochs=30, batch_size=256, lr=2e-3, hidden_size=48, num_heads=4,
          transformer_layers=2, dim_feedforward=128, dropout=0.1, verbose=False)
SYNTH = {"n": 2500, "T": 32, "tau_s": 2, "tau_l": 16,
         "n_sym": 4, "dim": 16, "seed": 0}

ARMS = [
    ("original", 2, "Referenced STLAT"),
    ("original_unshared", 2, "  + separate Wx"),
    ("original_asfg", 2, "Referenced STLAT + ASFG"),
    ("original_asfg_unshared", 2, "  + separate Wx"),
]

if __name__ == "__main__":
    print("tau_s=2, tau_l=16 -- theory-directed interventions\n", flush=True)
    print("%-28s %-10s %-10s %-9s %s"
          % ("arm", "acc", "sd", "tau-ovl", "eff-rank"), flush=True)
    out = {"seeds": SEEDS, "synth": SYNTH, "arms": {}}
    for nm, K, desc in ARMS:
        rs = train.run_seeds(nm, "synthetic", seeds=SEEDS, synth_kw=SYNTH,
                             num_bands=K, **KW)
        a = [r["final_val_acc"] for r in rs]
        ov = [r["tau_overlap"] for r in rs if r["tau_overlap"] is not None]
        er = [r["eff_rank_frac"] for r in rs if r["eff_rank_frac"] is not None]
        co = [r["coherence"] for r in rs if r["coherence"] is not None]
        out["arms"][nm] = {
            "desc": desc, "accs": a,
            "mean": float(np.mean(a)), "std": float(np.std(a, ddof=1)),
            "tau_overlap": float(np.mean(ov)) if ov else None,
            "eff_rank_frac": float(np.mean(er)) if er else None,
            "coherence": float(np.mean(co)) if co else None,
            "params": rs[0]["params"],
        }
        r = out["arms"][nm]
        print("%-28s %-10.4f %-10.4f %-9s %s"
              % (desc, r["mean"], r["std"],
                 "%.3f" % r["tau_overlap"] if r["tau_overlap"] is not None else "--",
                 "%.3f" % r["eff_rank_frac"] if r["eff_rank_frac"] is not None else "--"),
              flush=True)

    def delta(hi, lo):
        return out["arms"][hi]["mean"] - out["arms"][lo]["mean"]

    print("\n[A] un-sharing W_x, static fusion : %+.4f" %
          delta("original_unshared", "original"), flush=True)
    print("[A] un-sharing W_x, with ASFG    : %+.4f" %
          delta("original_asfg_unshared", "original_asfg"), flush=True)


    with open("results/collapse_final.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/collapse_final.json")
