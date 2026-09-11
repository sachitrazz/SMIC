r"""
make_bench_table.py -- the main results table: accuracy, F1, ROC-AUC, loss.

Reads results/bench_{isl,asl,csl}.json and writes tables/bench.tex.  One
block per dataset; the block header carries the dataset's lookup baseline
and chance level, so every accuracy is read against what a parameter-free
nearest-neighbour table achieves on the same partition.
"""

import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TABLES = os.path.join(HERE, "..", "out", "tables")
NL = r" \\"

NAMES = {"isl": "ISL-IEEE (%d classes, group-disjoint)",
         "asl": "ASL-IEEE (%d classes, group-disjoint)",
         "csl": "CSL (%d classes, one-shot, signer-disjoint)"}
MODEL = {"cnn": "CNN (mean-pooled tokens)",
         "cnn_tf": "CNN + Transformer",
         "lstm": "CNN + LSTM",
         "stlat": "STLAT (dual memory + ASFG)",
         "stlat_fixed": "STLAT, equations as written",
         "proposed": "SMIC (CNN, contrastive init.)"}
ORDER = ["stlat", "stlat_fixed", "lstm", "cnn_tf", "cnn", "proposed"]


def pm(m, s, k, scale=1.0, fmt="%.3f"):
    return (r"$" + fmt + r"\pm" + fmt + r"$") % (scale * m[k], scale * s[k])


def main():
    blocks = []
    for ds in ("isl", "asl", "csl"):
        f = os.path.join(RES, "bench_%s.json" % ds)
        if os.path.exists(f):
            d = json.load(open(f))
            fx = os.path.join(RES, "bench_%s_fixed.json" % ds)
            if os.path.exists(fx) and "stlat" in d["models"]:
                # corrected-wiring STLAT, run separately on the same seeds
                e = json.load(open(fx))["models"]["stlat_fixed"]
                d["models"]["stlat_fixed"] = e
                a = np.array(e["runs"]["acc"])
                b = np.array(d["models"]["stlat"]["runs"]["acc"])
                d["tests_vs_stlat"]["stlat_fixed"] = {"gain": float((a - b).mean()),
                                                      "p": float(stats.ttest_rel(a, b)[1])}
            blocks.append((ds, d))
    if not blocks:
        print("skip bench.tex (no results yet)")
        return
    n = len(blocks[0][1]["seeds"])
    L = [
        r"\begin{table*}[!t]",
        r"\caption{Main results. Every model reads the same hand-crop tokens "
        r"from the same convolutional trunk and differs only in the sequence "
        r"model applied to them; see Section~\ref{sec:proposed}. Mean $\pm$ "
        r"s.d. over %d shared seeds, final-epoch values under a cosine "
        r"schedule. Accuracy and macro-F1 are on the held-out partition; AUC "
        r"is macro one-vs-rest ROC-AUC. $p$ is a paired $t$-test against "
        r"STLAT over the shared seeds. Each block header gives the "
        r"no-learning lookup baseline and chance level on that partition.}" % n,
        r"\label{tab:bench}",
        r"\centering",
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccrc}",
        r"\toprule",
        r"Model & Accuracy & Macro-F1 & ROC-AUC & Train loss & Val.\ loss & "
        r"Params & $p$ vs STLAT" + NL,
        r"\midrule",
    ]
    for ds, d in blocks:
        name = NAMES[ds] % round(1.0 / d["chance"])
        L.append(r"\multicolumn{8}{l}{\textbf{%s}\quad lookup $%.3f$, chance $%.3f$}%s"
                 % (name, d["lookup"], d["chance"], NL))
        best = max(d["models"][m]["mean"]["acc"] for m in ORDER if m in d["models"])
        for m in ORDER:
            if m not in d["models"]:
                continue
            r = d["models"][m]
            mu, sd = r["mean"], r["std"]
            acc = pm(mu, sd, "acc")
            if abs(mu["acc"] - best) < 1e-9:
                acc = r"$\mathbf{%.3f}\pm%.3f$" % (mu["acc"], sd["acc"])
            t = d["tests_vs_stlat"].get(m)
            pv = "n/a" if t is None else ("%.4f" % t["p"] if t["p"] >= 1e-4 else r"$<10^{-4}$")
            L.append(r"\quad %s & %s & %s & %s & $%.3f$ & $%.3f$ & %s & %s%s"
                     % (MODEL[m], acc, pm(mu, sd, "f1"), pm(mu, sd, "auc"),
                        mu["train_loss"], mu["val_loss"],
                        "%d\\,K" % round(r["params"] / 1e3), pv, NL))
        L.append(r"\addlinespace[3pt]")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table*}", ""]
    os.makedirs(TABLES, exist_ok=True)
    open(os.path.join(TABLES, "bench.tex"), "w").write("\n".join(L))
    print("wrote tables/bench.tex (%s)" % ", ".join(ds for ds, _ in blocks))


if __name__ == "__main__":
    main()
