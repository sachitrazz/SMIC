"""
run_all.py -- runs the experiment stages in order.

The synthetic studies come first because they test the propositions and do
not depend on the sign language corpora.

    python run_all.py                # everything
    python run_all.py ratio ksweep   # a subset
    python run_all.py audit          # split integrity only (numpy, seconds)

The "audit" stage trains nothing.  It checks whether a split can be answered
by lookup, counts how many independent photographs a corpus contains, and
sweeps the near-duplicate threshold.  On a random image split of ISL-IEEE the
lookup baseline alone scores 0.992 (results/nn_leak_test.json), so the audit
should be run before any accuracy on a corpus is quoted.
"""

import json
import os
import sys
import time

import experiments as ex

RESULTS = "results"
SEEDS3 = [42, 123, 456]
SEEDS5 = [42, 123, 456, 789, 1024]

# Sized so the whole suite finishes in a few hours on one consumer GPU.
SYNTH_KW = dict(epochs=30, batch_size=256, lr=2e-3, hidden_size=48, num_heads=4,
                transformer_layers=2, dim_feedforward=128, dropout=0.1,
                verbose=False)
SYNTH_BASE = {"n": 2500, "T": 32, "n_sym": 4, "dim": 16, "seed": 0}


def banner(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70, flush=True)


def run_ratio():
    banner("Proposition 1: timescale-ratio sweep (synthetic)")
    t0 = time.time()
    out = ex.run_ratio(SEEDS3, ratios=(2, 4, 8, 14), T=32, tau_s=2,
                       n=2500, num_bands=4, **SYNTH_KW)
    out["elapsed"] = time.time() - t0
    ex._save("ratio_synthetic.json", out)
    print(json.dumps(out["prop1_fit"], indent=2))


def run_ksweep():
    banner("Proposition 3: band-count sweep (synthetic)")
    t0 = time.time()
    kw = dict(SYNTH_KW)
    out = ex.run_ksweep("synthetic", SEEDS3, ks=(1, 2, 4, 8, 16),
                        synth_kw=dict(SYNTH_BASE, tau_s=2, tau_l=24), **kw)
    out["elapsed"] = time.time() - t0
    ex._save("ksweep_synthetic.json", out)
    print(json.dumps(out["fits"], indent=2))


def run_collapse_synth():
    banner("Proposition 2: collapse measurement (synthetic)")
    t0 = time.time()
    out = ex.run_collapse("synthetic", SEEDS3, num_bands=4,
                          synth_kw=dict(SYNTH_BASE, tau_s=2, tau_l=24),
                          **SYNTH_KW)
    out["elapsed"] = time.time() - t0
    ex._save("collapse_synthetic.json", out)


def run_ablation_synth():
    banner("Component ablation (synthetic)")
    t0 = time.time()
    out = ex.run_ablation("synthetic", SEEDS5, num_bands=4,
                          synth_kw=dict(SYNTH_BASE, tau_s=2, tau_l=24),
                          **SYNTH_KW)
    out["elapsed"] = time.time() - t0
    ex._save("ablation_synthetic.json", out)
    for r in out["rows"]:
        print("  %-26s %.4f  d=%+.4f  p=%.2e" %
              (r["model"], r["acc_mean"], -r["delta_vs_ref"], r["p_bonferroni"]))






def run_audit():
    """The split-integrity audit.  No training: numpy only, seconds to run."""
    banner("SPLIT INTEGRITY AUDIT")
    import subprocess
    sets = ["isl", "asl"]
    for script, args in (("dupgroups.py", sets),
                         ("nn_leak_test.py", sets + [s + "_grouped" for s in sets]),
                         ("threshold_sweep.py", sets)):
        print("")
        print("$ python %s %s" % (script, " ".join(args)), flush=True)
        subprocess.run([sys.executable, script] + args, check=True)


STUDIES = {
    "ratio": run_ratio,
    "ksweep": run_ksweep,
    "collapse": run_collapse_synth,
    "ablation": run_ablation_synth,
    "audit": run_audit,
}

if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    which = sys.argv[1:] or ["audit", "ratio", "ksweep", "ablation"]
    t0 = time.time()
    for name in which:
        if name not in STUDIES:
            raise SystemExit("unknown study %r; choose from %s"
                             % (name, ", ".join(STUDIES)))
        STUDIES[name]()
    print("\nsuite finished in %.1f min" % ((time.time() - t0) / 60.0))
