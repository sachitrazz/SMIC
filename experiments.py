"""
experiments.py -- the experiment suite behind the paper's tables.

Four studies:

  collapse   Measure stream collapse in the *referenced STLAT* model and in
             the band-tiled one.  This is the empirical content of
             Proposition 2 and the paper's central diagnostic.

  ablation   Component ablation with paired t-tests over shared seeds and
             Bonferroni correction.  Paired, because every configuration
             sees the identical seed set and data split -- which is what
             makes the test legitimate.

  ksweep     Accuracy and coverage risk as a function of the number of
             bands K, testing the 1/K^2 law of Proposition 3.

  ratio      Synthetic sweep over the timescale ratio tau_l / tau_s,
             testing the ln^2(tau_l/tau_s) penalty of Proposition 1.

Usage
-----
    python experiments.py collapse --dataset synthetic
    python experiments.py ablation --dataset synthetic --epochs 40
    python experiments.py ksweep   --dataset synthetic
    python experiments.py ratio
"""

import argparse
import json
import os

import numpy as np
from scipy import stats

import train as train_mod

RESULTS = "results"


def _save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("wrote", path)
    return path


def _mean_std(xs):
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    if a.size == 0:
        return None, None
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


# ======================================================================
# 1.  Collapse measurement  (Proposition 2)
# ======================================================================

COLLAPSE_MODELS = [
    ("original", "Referenced STLAT (shared W_x, static fusion)"),
    ("original_asfg", "Referenced STLAT + ASFG"),
    ("spectral_free_gate", "K streams, free sigmoid gates (no certificate)"),
    ("spectral_shared_proj", "Band gates but shared W_x"),
    ("spectral", "Band-tiled memory, certified disjoint bands (rejected)"),
]


def run_collapse(dataset, seeds, **kw):
    rows = []
    for name, desc in COLLAPSE_MODELS:
        runs = train_mod.run_seeds(name, dataset, seeds=seeds, **kw)
        acc_m, acc_s = _mean_std([r["final_val_acc"] for r in runs])
        coh_m, coh_s = _mean_std([r["coherence"] for r in runs])
        er_m, er_s = _mean_std([r["eff_rank_frac"] for r in runs])
        pc_m, pc_s = _mean_std([r["pair_corr"] for r in runs])
        ov_m, ov_s = _mean_std([r["tau_overlap"] for r in runs])
        rows.append({
            "model": name, "description": desc,
            "acc_mean": acc_m, "acc_std": acc_s,
            "coherence_mean": coh_m, "coherence_std": coh_s,
            "eff_rank_frac_mean": er_m, "eff_rank_frac_std": er_s,
            "pair_corr_mean": pc_m, "pair_corr_std": pc_s,
            "tau_overlap_mean": ov_m, "tau_overlap_std": ov_s,
            "params": runs[0]["params"],
            "accs": [r["final_val_acc"] for r in runs],
        })
        print("  %-22s acc %.4f  coherence %.4f  eff-rank %.3f  tau-overlap %.3f"
              % (name, acc_m, coh_m or -1, er_m or -1, ov_m or -1))
    return {"dataset": dataset, "seeds": seeds, "rows": rows}


# ======================================================================
# 2.  Ablation with paired t-tests
# ======================================================================

ABLATION_MODELS = [
    ("spectral", "Band-tiled memory (K=4)"),
    ("spectral_no_cross", "- cross-band interaction Phi"),
    ("spectral_no_transformer", "- Transformer encoder"),
    ("spectral_free_gate", "- disjointness certificate (free gates)"),
    ("spectral_shared_proj", "- separate input projections"),
    ("spectral_K2", "K=2 (dual memory)"),
    ("spectral_K1", "K=1 (single memory)"),
    ("original_asfg", "Referenced STLAT + ASFG"),
    ("original", "Referenced STLAT"),
]


def run_ablation(dataset, seeds, reference="spectral", **kw):
    per_model = {}
    for name, desc in ABLATION_MODELS:
        runs = train_mod.run_seeds(name, dataset, seeds=seeds, **kw)
        per_model[name] = {"desc": desc,
                           "accs": [r["final_val_acc"] for r in runs],
                           "params": runs[0]["params"],
                           "gap_acc": _mean_std([r["gap_acc"] for r in runs])[0],
                           "gap_acc_std": _mean_std([r["gap_acc"] for r in runs])[1],
                           "coherence": _mean_std([r["coherence"] for r in runs])[0],
                           "coherence_std": _mean_std([r["coherence"] for r in runs])[1],
                           "eff_rank_frac": _mean_std([r["eff_rank_frac"] for r in runs])[0],
                           "eff_rank_frac_std": _mean_std([r["eff_rank_frac"] for r in runs])[1],
                           "pair_corr": _mean_std([r["pair_corr"] for r in runs])[0],
                           "pair_corr_std": _mean_std([r["pair_corr"] for r in runs])[1],
                           "tau_overlap": _mean_std([r["tau_overlap"] for r in runs])[0],
                           "tau_overlap_std": _mean_std([r["tau_overlap"] for r in runs])[1],
                           "seconds": _mean_std([r["seconds"] for r in runs])[0]}

    ref = np.asarray(per_model[reference]["accs"])
    comparisons = [n for n, _ in ABLATION_MODELS if n != reference]
    m = len(comparisons)  # Bonferroni family size

    rows = []
    for name in comparisons:
        a = np.asarray(per_model[name]["accs"])
        diff = ref - a
        if np.allclose(diff, 0):
            t, p = 0.0, 1.0
        else:
            t, p = stats.ttest_rel(ref, a)
        # Cohen's d for paired samples
        dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("inf")
        w_stat, w_p = (float("nan"), float("nan"))
        if len(diff) >= 5 and not np.allclose(diff, 0):
            try:
                w_stat, w_p = stats.wilcoxon(ref, a)
            except ValueError:
                pass
        rows.append({
            "model": name, "desc": per_model[name]["desc"],
            "acc_mean": float(a.mean()), "acc_std": float(a.std(ddof=1)),
            "delta_vs_ref": float(ref.mean() - a.mean()),
            "t": float(t), "p_raw": float(p),
            "p_bonferroni": float(min(1.0, p * m)),
            "cohens_dz": dz,
            "wilcoxon_p": float(w_p),
            "params": per_model[name]["params"],
            "gap_acc": per_model[name]["gap_acc"],
            "coherence": per_model[name]["coherence"],
            "eff_rank_frac": per_model[name]["eff_rank_frac"],
            "pair_corr": per_model[name]["pair_corr"],
            "tau_overlap": per_model[name]["tau_overlap"],
        })

    return {
        "dataset": dataset, "seeds": seeds, "reference": reference,
        "bonferroni_family_size": m,
        "reference_acc_mean": float(ref.mean()),
        "reference_acc_std": float(ref.std(ddof=1)),
        "reference_params": per_model[reference]["params"],
        "rows": rows,
        "raw": per_model,
    }


# ======================================================================
# 3.  K sweep  (Proposition 3: the 1/K^2 law)
# ======================================================================

def run_ksweep(dataset, seeds, ks=(1, 2, 4, 8, 16), **kw):
    rows = []
    for K in ks:
        runs = train_mod.run_seeds("spectral_K%d" % K, dataset, seeds=seeds, **kw)
        m, s = _mean_std([r["final_val_acc"] for r in runs])
        rows.append({"K": K, "acc_mean": m, "acc_std": s,
                     "params": runs[0]["params"],
                     "accs": [r["final_val_acc"] for r in runs],
                     "eff_rank_frac": _mean_std([r["eff_rank_frac"] for r in runs])[0],
                     "seconds": _mean_std([r["seconds"] for r in runs])[0]})
        print("  K=%-3d acc %.4f +/- %.4f  (%d params)" % (K, m, s, runs[0]["params"]))

    # Fit  err(K) = a / K^2 + c  and compare against  err(K) = a / K + c,
    # so the 1/K^2 law is tested against a plausible alternative rather
    # than against nothing.
    K = np.array([r["K"] for r in rows], dtype=float)
    err = 1.0 - np.array([r["acc_mean"] for r in rows], dtype=float)
    fits = {}
    for label, basis in (("1/K^2", 1.0 / K ** 2), ("1/K", 1.0 / K),
                         ("log", -np.log(K))):
        A = np.stack([basis, np.ones_like(basis)], axis=1)
        coef, res, *_ = np.linalg.lstsq(A, err, rcond=None)
        pred = A @ coef
        ss_res = float(((err - pred) ** 2).sum())
        ss_tot = float(((err - err.mean()) ** 2).sum())
        fits[label] = {"coef": coef.tolist(),
                       "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                       "rmse": float(np.sqrt(ss_res / len(err)))}
    return {"dataset": dataset, "seeds": seeds, "rows": rows, "fits": fits}


# ======================================================================
# 4.  Timescale-ratio sweep  (Proposition 1)
# ======================================================================

def run_ratio(seeds, ratios=(2, 4, 8, 16, 24), T=64, tau_s=2, n=4000, **kw):
    """Sweep tau_l/tau_s on the synthetic task.

    Proposition 1 predicts the single-memory penalty grows as
    ln^2(tau_l/tau_s) while a banked model stays flat.  ``spectral_K1``
    is the single-memory control; ``spectral`` (K=4) is the banked model.
    """
    rows = []
    for r in ratios:
        tau_l = int(tau_s * r)
        if tau_l >= T:
            continue
        synth = {"n": n, "T": T, "tau_s": tau_s, "tau_l": tau_l,
                 "n_sym": 4, "dim": 16, "seed": 0}
        entry = {"ratio": r, "tau_s": tau_s, "tau_l": tau_l,
                 "log_ratio_sq": float(np.log(r) ** 2)}
        for name in ("spectral_K1", "spectral"):
            runs = train_mod.run_seeds(name, "synthetic", seeds=seeds,
                                       synth_kw=synth, **kw)
            m, s = _mean_std([x["final_val_acc"] for x in runs])
            entry[name] = {"acc_mean": m, "acc_std": s,
                           "accs": [x["final_val_acc"] for x in runs]}
        entry["gap"] = entry["spectral"]["acc_mean"] - entry["spectral_K1"]["acc_mean"]
        rows.append(entry)
        print("  ratio %-4d (tau_l=%d): K=1 %.4f  K=4 %.4f  gap %+.4f"
              % (r, tau_l, entry["spectral_K1"]["acc_mean"],
                 entry["spectral"]["acc_mean"], entry["gap"]))

    # Does the single-memory error track ln^2(ratio)?
    x = np.array([r["log_ratio_sq"] for r in rows])
    y = np.array([1.0 - r["spectral_K1"]["acc_mean"] for r in rows])
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    slope, icept, rval, pval, se = (stats.linregress(x, y) if len(x) > 2
                                    else (np.nan,) * 5)
    return {"rows": rows, "seeds": seeds,
            "prop1_fit": {"pearson_r": corr, "slope": float(slope),
                          "intercept": float(icept), "r2": float(rval ** 2),
                          "p": float(pval)}}


# ======================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("study", choices=["collapse", "ablation", "ksweep", "ratio"])
    p.add_argument("--dataset", default="synthetic", choices=["isl", "csl", "synthetic"])
    p.add_argument("--seeds", type=int, nargs="*", default=train_mod.DEFAULT_SEEDS)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-bands", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--lambda-dec", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tag", default="")
    a = p.parse_args()

    kw = dict(epochs=a.epochs, batch_size=a.batch_size,
              hidden_size=a.hidden_size, num_bands=a.num_bands,
              seq_len=a.seq_len, img_size=a.img_size,
              lambda_dec=a.lambda_dec, lr=a.lr, verbose=False)

    tag = ("_" + a.tag) if a.tag else ""
    if a.study == "collapse":
        out = run_collapse(a.dataset, a.seeds, **kw)
        _save("collapse_%s%s.json" % (a.dataset, tag), out)
    elif a.study == "ablation":
        out = run_ablation(a.dataset, a.seeds, **kw)
        _save("ablation_%s%s.json" % (a.dataset, tag), out)
        print("\n%-26s %-8s %-8s %-10s %-10s" % ("config", "acc", "delta", "p(Bonf)", "d_z"))
        print("%-26s %.4f" % (out["reference"], out["reference_acc_mean"]))
        for r in out["rows"]:
            print("%-26s %.4f   %+.4f   %.2e   %.2f"
                  % (r["model"], r["acc_mean"], -r["delta_vs_ref"],
                     r["p_bonferroni"], r["cohens_dz"]))
    elif a.study == "ksweep":
        out = run_ksweep(a.dataset, a.seeds, **kw)
        _save("ksweep_%s%s.json" % (a.dataset, tag), out)
        print("\nfits:", json.dumps(out["fits"], indent=2))
    elif a.study == "ratio":
        kw.pop("seq_len"), kw.pop("img_size")
        out = run_ratio(a.seeds, **kw)
        _save("ratio_synthetic%s.json" % tag, out)
        print("\nProp 1 fit:", json.dumps(out["prop1_fit"], indent=2))


if __name__ == "__main__":
    main()
