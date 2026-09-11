"""
make_bench_figures.py -- training curves and ROC curves from bench.py output.

  BENCH_CURVES  validation accuracy (top row) and validation loss (bottom
                row) per epoch, one column per dataset, one line per model
                (seed 42; the table reports all five seeds)
  BENCH_ROC     micro-averaged one-vs-rest ROC curve per model, one panel
                per dataset, with the macro AUC in the legend

Both are written as vector PDF and a 300 dpi JPEG and copied into the
out/figures/ directory when it exists.

    python figures/make_bench_figures.py
"""

import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DST = os.path.join(HERE, "..", "out", "figures")

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

TITLE = {"isl": "ISL-IEEE", "asl": "ASL-IEEE", "csl": "CSL (signer-disjoint)"}
STYLE = {"stlat":    ("#b4531f", "-",  "STLAT"),
         "lstm":     ("#9a7a3f", "--", "CNN + LSTM"),
         "cnn_tf":   ("#2a6f97", "-.", "CNN + Transformer"),
         "cnn":      ("#7a7a7a", ":",  "CNN"),
         "proposed": ("#3f7d5a", "-",  "SMIC")}
ORDER = ["stlat", "lstm", "cnn_tf", "cnn", "proposed"]


def load():
    out = []
    for ds in ("isl", "asl", "csl"):
        f = os.path.join(RES, "bench_%s.json" % ds)
        if os.path.exists(f):
            out.append((ds, json.load(open(f))))
    return out


def save(fig, name):
    for ext in ("pdf", "jpg"):
        p = os.path.join(HERE, "%s.%s" % (name, ext))
        fig.savefig(p, dpi=300 if ext == "jpg" else None, bbox_inches="tight")
        if os.path.isdir(DST):
            shutil.copyfile(p, os.path.join(DST, "%s.%s" % (name, ext)))
    print("wrote", name)


def curves(data):
    fig, axes = plt.subplots(2, len(data), figsize=(3.9 * len(data), 5.6), squeeze=False)
    for c, (ds, d) in enumerate(data):
        for m in ORDER:
            if m not in d["models"]:
                continue
            col, ls, lab = STYLE[m]
            cu = d["models"][m]["curve_seed42"]
            ep = np.arange(1, len(cu["val_acc"]) + 1)
            lw = 2.0 if m in ("proposed", "stlat") else 1.3
            axes[0, c].plot(ep, cu["val_acc"], color=col, ls=ls, lw=lw, label=lab)
            axes[1, c].plot(ep, cu["val_loss"], color=col, ls=ls, lw=lw, label=lab)
        axes[0, c].axhline(d["lookup"], color="black", lw=0.9, ls=(0, (4, 3)))
        axes[0, c].text(1, d["lookup"], " lookup", fontsize=7, va="bottom")
        axes[0, c].set_title(TITLE[ds], fontsize=9.5, fontweight="bold", loc="left")
        axes[1, c].set_xlabel("epoch", fontsize=8.5)
        for r in (0, 1):
            axes[r, c].tick_params(labelsize=7.5)
    axes[0, 0].set_ylabel("validation accuracy", fontsize=8.5)
    axes[1, 0].set_ylabel("validation loss", fontsize=8.5)
    # one shared legend below the panels, so no data line is covered
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8, frameon=False)
    save(fig, "BENCH_CURVES")


def roc(data):
    fig, axes = plt.subplots(1, len(data), figsize=(3.9 * len(data), 3.6), squeeze=False)
    for c, (ds, d) in enumerate(data):
        ax = axes[0, c]
        for m in ORDER:
            if m not in d["models"]:
                continue
            col, ls, lab = STYLE[m]
            P = np.array(d["models"][m]["probs_seed42"])
            y = np.array(d["models"][m]["labels_seed42"])
            Y = np.zeros_like(P); Y[np.arange(len(y)), y] = 1
            s, t = P.ravel(), Y.ravel()                 # micro-average
            o = np.argsort(-s)
            tp = np.cumsum(t[o]); fp = np.cumsum(1 - t[o])
            tpr = np.concatenate([[0], tp / max(t.sum(), 1)])
            fpr = np.concatenate([[0], fp / max((1 - t).sum(), 1)])
            auc = d["models"][m]["mean"]["auc"]
            ax.plot(fpr, tpr, color=col, ls=ls, lw=2.0 if m in ("proposed", "stlat") else 1.3,
                    label="%s (AUC %.3f)" % (lab, auc))
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8)
        ax.set_title(TITLE[ds], fontsize=9.5, fontweight="bold", loc="left")
        ax.set_xlabel("false positive rate", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=6.4, frameon=False, loc="lower right")
    axes[0, 0].set_ylabel("true positive rate", fontsize=8.5)
    fig.tight_layout()
    save(fig, "BENCH_ROC")


if __name__ == "__main__":
    data = load()
    if not data:
        print("no bench results yet")
    else:
        curves(data)
        roc(data)
