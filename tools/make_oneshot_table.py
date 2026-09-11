r"""
make_oneshot_table.py -- tables/oneshot.tex: the CSL task under the standard
one-shot protocol (nearest prototype, no gradient step on labelled clips),
beside the supervised heads of the main benchmark at 40 epochs and, where it
has been run, at the optimiser-step count ISL receives.

Reads results/csl_oneshot.json, results/bench_csl.json and, if present,
results/bench_csl_steps.json.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TABLES = os.path.join(HERE, "..", "out", "tables")
NL = r" \\"
LABEL = {"stlat": "STLAT", "lstm": "CNN + LSTM", "cnn_tf": "CNN + Transformer",
         "cnn": "CNN", "proposed": "SMIC"}


def J(n):
    p = os.path.join(RES, n)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    one, b40, bst = J("csl_oneshot.json"), J("bench_csl.json"), J("bench_csl_steps.json")
    if not one:
        print("skip oneshot.tex (no results yet)")
        return
    L = [
        r"\begin{table}[!t]",
        r"\caption{The CSL task (one training clip per sign, tested on other "
        r"signers) under three protocols. \emph{Supervised}: the heads of "
        r"Table~\ref{tab:bench}, trained for $40$ epochs, and for $%s$ "
        r"epochs, which gives CSL the same number of optimiser steps as "
        r"ISL-IEEE. \emph{Prototype}: no training on labelled clips; each test "
        r"clip takes the label of the most cosine-similar training clip in "
        r"the trunk's embedding. Mean $\pm$ s.d. over five seeds where seeds "
        r"apply; the contrastive trunk is a single pretraining run. Lookup "
        r"$%.3f$, chance $%.3f$.}" % (bst["epochs"] if bst else "--",
                                     one["lookup"], one["chance"]),
        r"\label{tab:oneshot}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Protocol & Model & Top-1 & ROC-AUC" + NL,
        r"\midrule",
    ]
    for tag, d in (("Supervised, 40 ep.", b40), ("Supervised, matched steps", bst)):
        if not d:
            continue
        for k, m in enumerate(("stlat", "lstm", "cnn_tf", "cnn", "proposed")):
            r = d["models"][m]
            L.append(r"%s & %s & $%.3f\pm%.3f$ & $%.3f$%s" % (
                tag if k == 0 else "", LABEL[m], r["mean"]["acc"], r["std"]["acc"],
                r["mean"]["auc"], NL))
        L.append(r"\midrule")
    rm, rs = one["random"]["mean"], one["random"]["std"]
    c = one["contrastive"]
    L.append(r"Prototype & random trunk & $%.3f\pm%.3f$ & $%.3f$%s" % (rm["acc"], rs["acc"], rm["auc"], NL))
    L.append(r" & contrastive trunk & $\mathbf{%.3f}$ & $%.3f$%s" % (c["acc"], c["auc"], NL))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(TABLES, exist_ok=True)
    open(os.path.join(TABLES, "oneshot.tex"), "w").write("\n".join(L))
    print("wrote tables/oneshot.tex")


if __name__ == "__main__":
    main()
