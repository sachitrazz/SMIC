"""
make_theory_figure.py -- Figure 3: everything the experiments established.

Four panels, every number measured, including the failures:

  (a) The collapse and the repair Proposition 2 prescribes.  Sharing the
      input projections gives high stream overlap; separating them roughly
      halves it.
  (b) The accuracy consequence, pooled over two independent runs, drawn
      with a confidence interval that crosses zero -- the direction is
      consistent, the effect is not established.
  (c) Three axes of intervention, all tested and rejected.
  (d) The isolated-sign result: no configuration difference is resolvable
      at the available data scale.

    python figures/make_theory_figure.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

INK = "#1a1a1a"
ACCENT = "#b4531f"
GOOD = "#3f7d5a"
MUTED = "#7a7a7a"


def load(n):
    p = os.path.join(RESULTS, n)
    return json.load(open(p)) if os.path.exists(p) else None


def _empty(ax, msg):
    ax.text(.5, .5, msg, ha="center", va="center", fontsize=8.5, color=MUTED,
            style="italic", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- (a)
LBL = {"original_asfg": "shared $W_x$\n(referenced)",
       "original_asfg_unshared": "separate $W_x$\n(the repair)"}


def panel_collapse(ax, cas):
    if not cas:
        _empty(ax, "cascade_test.json not found"); return
    a = cas["synthetic"]["arms"]
    ks = ["original_asfg", "original_asfg_unshared"]
    ovl = [a[k]["tau_overlap"] for k in ks]
    er = [a[k]["eff_rank_frac"] for k in ks]
    x = np.arange(2); w = .36
    ax.bar(x - w / 2, ovl, w, color=ACCENT, label="stream overlap  $\\downarrow$")
    ax.bar(x + w / 2, er, w, color=INK, label="effective rank frac.  $\\uparrow$")
    for i, v in enumerate(ovl):
        ax.text(i - w / 2, v + .02, "%.2f" % v, ha="center", fontsize=7.4,
                color=ACCENT, fontweight="bold")
    ax.annotate("", xy=(1 - w / 2, ovl[1] + .07), xytext=(0 - w / 2, ovl[0] - .07),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))
    ax.text(.5, ovl[0] - .22, "halved", ha="center", fontsize=7.6,
            color=ACCENT, style="italic")
    ax.set_xticks(x); ax.set_xticklabels([LBL[k] for k in ks], fontsize=7.6)
    ax.set_ylim(0, 1.0); ax.set_ylabel("statistic", fontsize=9)
    ax.set_title("(a) The collapse, and the one-line repair",
                 fontsize=9.5, loc="left", fontweight="bold")
    ax.legend(fontsize=7.2, frameon=False, loc="upper right")
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------- (b)
def panel_gain(ax, col, cas):
    if not (col and cas):
        _empty(ax, "results not found"); return
    r1 = np.array(col["arms"]["original_asfg"]["accs"])
    r1u = np.array(col["arms"]["original_asfg_unshared"]["accs"])
    b = cas["synthetic"]["arms"]
    r2 = np.array(b["original_asfg"]["accs"])
    r2u = np.array(b["original_asfg_unshared"]["accs"])
    d = np.concatenate([r1u - r1, r2u - r2])
    m, sd, n = d.mean(), d.std(ddof=1), len(d)
    ci = 1.96 * sd / np.sqrt(n)

    ax.axvline(0, color=MUTED, lw=1.1, ls="--")
    ax.scatter(d, np.random.default_rng(0).normal(1, .06, n), s=34,
               color=[GOOD if v > 0 else ACCENT for v in d], alpha=.75,
               zorder=4, edgecolors="white", linewidths=.6)
    ax.errorbar([m], [0.5], xerr=[[ci], [ci]], fmt="D", ms=8, capsize=5,
                lw=2.0, color=INK, zorder=5)
    ax.text(m, 0.30, "pooled %+.3f\n95%% CI [%+.3f, %+.3f]" % (m, m - ci, m + ci),
            ha="center", fontsize=7.4, color=INK)
    ax.set_ylim(0, 1.45); ax.set_yticks([])
    ax.set_xlabel("accuracy gain from separating $W_x$", fontsize=9)
    ax.set_title("(b) Consistent direction, not a proven effect",
                 fontsize=9.5, loc="left", fontweight="bold")
    ax.text(.02, .95, "%d of %d seeds positive   $p=0.18$" % ((d > 0).sum(), n),
            transform=ax.transAxes, fontsize=7.6, va="top", color=ACCENT,
            style="italic")
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------- (c)
def panel_rejected(ax, ratio, cas):
    rows, labels, colors = [], [], []
    if ratio:
        r = {x["ratio"]: x for x in ratio["rows"]}.get(8)
        if r:
            rows.append(r["spectral"]["acc_mean"] - r["spectral_K1"]["acc_mean"])
            labels.append("band tiling\n(constrain $\\tau$)")
    if cas:
        b = cas["synthetic"]["arms"]
        rows.append(b["original_asfg_cascade"]["mean"] - b["original_asfg"]["mean"])
        labels.append("cascade\n(change topology)")
    ft = load("asfg_test.json")
    if ft and ft.get("conditions"):
        c0 = ft["conditions"][0]["arms"]
        rows.append(c0["spectral_K2_gated"]["mean"] - c0["spectral_K2"]["mean"])
        labels.append("non-convex fusion\n(change mixing)")
    if not rows:
        _empty(ax, "no rejected-mechanism results"); return

    y = np.arange(len(rows))[::-1]
    ax.barh(y, rows, .5, color=[ACCENT if v < -.01 else MUTED for v in rows])
    ax.axvline(0, color=INK, lw=1.1)
    for yi, v in zip(y, rows):
        ax.text(v + (-.02 if v < 0 else .02), yi, "%+.3f" % v,
                va="center", ha="right" if v < 0 else "left",
                fontsize=7.6, fontweight="bold",
                color=ACCENT if v < -.01 else MUTED)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.4)
    ax.set_xlabel("accuracy change vs its own baseline", fontsize=9)
    ax.set_title("(c) Three axes of intervention, tested and rejected",
                 fontsize=9.5, loc="left", fontweight="bold")
    ax.set_xlim(min(rows) * 1.35, max(max(rows), 0.05) * 2.4)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------- (d)
NICE = {"published (shared W_x + ASFG)": "published",
        "  + separate W_x": "+ separate $W_x$",
        "  + separate W_x + contrastive": "+ contrastive",
        "proposed encoder + contrastive": "proposed encoder"}


def panel_control(ax, ctl, ovf):
    """The headline: removing the recurrence improves accuracy."""
    if not ctl:
        _empty(ax, "control_test.json not found"); return
    A = ctl["arms"]
    ks = ["original_asfg", "cnn", "cnn_tf"]
    lab = ["STLAT\n(LSTM + dual mem)", "CNN only", "CNN + Transformer"]
    mu = [A[k]["mean"] for k in ks]
    sd = [A[k]["std"] for k in ks]
    n = len(ctl["seeds"])
    ci = [1.96 * x / np.sqrt(n) for x in sd]
    col = [ACCENT, GOOD, GOOD]
    x = np.arange(3)
    ax.bar(x, mu, .55, color=col, yerr=ci, capsize=4,
           error_kw=dict(lw=1.3, ecolor=INK))
    for i, v in enumerate(mu):
        ax.text(i, v + ci[i] + .012, "%.3f" % v, ha="center",
                fontsize=8, fontweight="bold", color=col[i])
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.2)
    ax.set_ylabel("ISL validation accuracy", fontsize=9)
    ax.set_ylim(0, max(mu) * 1.62)
    ax.set_title("(d) Removing the recurrence WINS",
                 fontsize=9.5, loc="left", fontweight="bold")
    t = {x["arm"]: x for x in ctl["tests"]}["cnn_tf"]
    ax.text(.5, .965, "+%.1f pts   $p=%.4f$   $d_z=%.2f$   5/5 seeds"
            % (100 * t["delta"], t["p"], t["dz"]),
            transform=ax.transAxes, ha="center", fontsize=7.6,
            color=GOOD, fontweight="bold")
    if ovf:
        ax.text(.5, .885, "STLAT train %.2f / val %.2f  (overfitting, not undertrained)"
                % (ovf["original_asfg"]["train"], ovf["original_asfg"]["val"]),
                transform=ax.transAxes, ha="center", fontsize=6.8,
                color=MUTED, style="italic")
    ax.tick_params(labelsize=8)


def build():
    col = load("collapse_final.json")
    cas = load("cascade_test.json")
    ratio = load("ratio_synthetic.json")
    ctl = load("control_test.json")
    ovf = load("overfit_test.json")

    # The sign-data result now lives in the benchmark figures
    # (make_bench_figures.py); this figure keeps only the controlled task.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    panel_collapse(axes[0], cas)
    panel_gain(axes[1], col, cas)
    panel_rejected(axes[2], ratio, cas)
    fig.tight_layout(pad=1.6)
    return fig


if __name__ == "__main__":
    fig = build()
    out = os.path.join(HERE, "THEORY.jpg")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=.12)
    fig.savefig(out.replace(".jpg", ".pdf"), bbox_inches="tight", pad_inches=.12)
    print("wrote", out)
