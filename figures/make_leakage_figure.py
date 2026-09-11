"""
make_leakage_figure.py -- Figure 4: the benchmark, not the model.

  (a) File count is not sample size.  Each dataset collapses from files to
      byte-unique images to independent near-duplicate groups.
  (b) One model, one schedule, three splits.  Accuracy tracks the split.
  (c) The lookup baseline: nearest neighbour in raw pixels, no learning.
      Where it matches the network, the network is doing lookup too.
  (d) On the leakage-free split, does unlabelled CSL pretraining help?

    python figures/make_leakage_figure.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
PREP = os.path.join(HERE, "..", "..", "prepared")

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
PALE = "#c9c9c9"


def load(n, base=RESULTS):
    p = os.path.join(base, n)
    return json.load(open(p)) if os.path.exists(p) else None


def _empty(ax, msg):
    ax.text(.5, .5, msg, ha="center", va="center", fontsize=8.5, color=MUTED,
            style="italic", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- (a)
def panel_shrink(ax):
    rows = []
    sw ={e["dataset"]: e for e in (load("threshold_sweep.json") or [])}
    def grp(ds, default):
        for r in sw.get(ds, {}).get("rows", []):
            if abs(r["threshold"] - 0.98) < 1e-9:
                return r["n_groups"]
        return default
    rows.append(("ISL-IEEE", 1225, 1006, grp("isl", 216)))
    rows.append(("ASL-IEEE", 840, 419, grp("asl", 350)))

    x = np.arange(len(rows)); w = .26
    files = [r[1] for r in rows]; uniq = [r[2] for r in rows]; grp = [r[3] for r in rows]
    ax.bar(x - w, files, w, color=PALE, label="files as distributed")
    ax.bar(x, uniq, w, color=MUTED, label="byte-unique (MD5)")
    ax.bar(x + w, grp, w, color=ACCENT, label="independent groups")
    for i, (f, u, g) in enumerate(zip(files, uniq, grp)):
        ax.text(i + w, g + 26, "%d" % g, ha="center", fontsize=7.6,
                fontweight="bold", color=ACCENT)
        if f > 0:
            ax.text(i, u + 26, "%d" % u, ha="center", fontsize=7.0, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=7.8)
    ax.set_ylabel("images", fontsize=9)
    ax.set_title("(a) File count is not sample size",
                 fontsize=9.5, loc="left", fontweight="bold")
    ax.legend(fontsize=7.0, frameon=False, loc="upper right")
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------- (b)
def panel_protocol(ax):
    pt = load("protocol_test.json")
    if not pt:
        _empty(ax, "protocol_test.json not found"); return
    rows = pt["rows"]
    lab = ["A supplied\nsplit", "B MD5-dedup\nrandom", "C group-\ndisjoint"]
    mu = [r["mean"] for r in rows]
    sd = [r["std"] for r in rows]
    n = len(pt["seeds"])
    ci = [1.96 * s / np.sqrt(n) for s in sd]
    col = [ACCENT, ACCENT, GOOD]
    x = np.arange(3)
    ax.bar(x, mu, .55, color=col, yerr=ci, capsize=4,
           error_kw=dict(lw=1.2, ecolor=INK))
    for i, v in enumerate(mu):
        ax.text(i, v + ci[i] + .022, "%.3f" % v, ha="center", fontsize=8,
                fontweight="bold", color=col[i])
    drop = pt["B_vs_C"]["drop"]
    ax.annotate("", xy=(2, mu[2] + .04), xytext=(1.3, mu[1] + .04),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(1.5, mu[1] + .10, "$-$%.1f pts" % (100 * drop), ha="center",
            fontsize=8.2, fontweight="bold", color=INK)
    lk = [r["nn_lookup"] for r in rows]
    ax.plot(x, lk, marker="_", ms=26, mew=2.4, ls="none", color=INK)
    for i, v in enumerate(lk):
        ax.text(i + .30, v, "%.3f" % v, va="center", fontsize=7.0, color=INK)
    ax.annotate("1-NN lookup" + chr(10) + "(no learning)", xy=(0, lk[0]), xytext=(-0.36, 0.60),
                fontsize=7.0, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK, lw=.9,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.4)
    ax.set_ylabel("ISL validation accuracy", fontsize=9)
    ax.set_ylim(0, 1.22)
    ax.set_title("(b) Same images, same model: the split decides",
                 fontsize=9.5, loc="left", fontweight="bold")
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------- (c)
def panel_threshold(ax):
    """The lookup baseline as the grouping threshold tightens."""
    sw = load("threshold_sweep.json")
    if not sw:
        _empty(ax, "threshold_sweep.json not found"); return
    style = {"isl": (ACCENT, "o", "ISL-IEEE"),
             "asl": (GOOD, "s", "ASL-IEEE")}
    for e in sw:
        col, mk, lab = style.get(e["dataset"], (MUTED, "^", e["dataset"]))
        th = [r["threshold"] for r in e["rows"]]
        lk = [r["lookup_acc"] for r in e["rows"]]
        ax.plot(th, lk, marker=mk, ms=5, lw=1.7, color=col, label=lab)
        ax.axhline(e["chance"], color=col, lw=.9, ls=":", alpha=.6)
    ax.axvline(0.98, color=MUTED, lw=1.0, ls="-", alpha=.45)
    ax.text(0.978, 1.02, "used here", fontsize=6.9, color=MUTED,
            rotation=90, va="top", ha="right")
    ax.invert_xaxis()
    ax.set_xlabel("near-duplicate threshold  (tighter to the right)", fontsize=9)
    ax.set_ylabel("lookup accuracy", fontsize=9)
    ax.set_ylim(0, 1.06)
    ax.set_title("(c) Lookup accuracy against the grouping threshold",
                 fontsize=9.5, loc="left", fontweight="bold")
    ax.legend(fontsize=7.0, frameon=False, loc="lower left")
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------- (d)
def panel_pretrain(ax):
    got = [(n, load("pretrain_%s.json" % n)) for n in
           ("isl_grouped", "asl_grouped")]
    got = [(n, r) for n, r in got if r]
    if not got:
        _empty(ax, "pretraining results not found"); return
    x = np.arange(len(got)); w = .34
    a = [np.mean(r["scratch"]) for _, r in got]
    b = [np.mean(r["pretrained"]) for _, r in got]
    ea = [1.96 * np.std(r["scratch"], ddof=1) / np.sqrt(len(r["scratch"]))
          for _, r in got]
    eb = [1.96 * np.std(r["pretrained"], ddof=1) / np.sqrt(len(r["pretrained"]))
          for _, r in got]
    ax.bar(x - w / 2, a, w, color=MUTED, yerr=ea, capsize=3,
           error_kw=dict(lw=1.1, ecolor=INK), label="random init")
    ax.bar(x + w / 2, b, w, color=GOOD, yerr=eb, capsize=3,
           error_kw=dict(lw=1.1, ecolor=INK), label="CSL-pretrained")
    for i, (_, r) in enumerate(got):
        g = r["gain"]
        ax.text(i, max(a[i] + ea[i], b[i] + eb[i]) + .022,
                "%+.3f\n$p=%.2g$" % (g, r["p"]), ha="center", fontsize=7.4,
                color=GOOD if g > 0 else ACCENT, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.split("_")[0].upper() + "\n(group-disjoint)"
                        for n, _ in got], fontsize=7.6)
    ax.set_ylabel("validation accuracy", fontsize=9)
    ax.set_ylim(0, max(b + a) * 1.45)
    ax.set_title("(d) 10,304 unlabelled CSL hands, on the honest split",
                 fontsize=9.5, loc="left", fontweight="bold")
    ax.legend(fontsize=7.0, frameon=False, loc="upper right")
    ax.tick_params(labelsize=8)


if __name__ == "__main__":
    # the pretraining panel (d) is superseded by the benchmark (bench.py)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9))
    panel_shrink(axes[0])
    panel_protocol(axes[1])
    panel_threshold(axes[2])
    fig.tight_layout(pad=1.6)
    out = os.path.join(HERE, "LEAKAGE.jpg")
    fig.savefig(out, dpi=260, bbox_inches="tight")
    fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")   # vector copy for print
    print("wrote", out)
    # also copy into out/figures when that directory exists
    jpg = os.path.join(HERE, "..", "out", "figures", "LEAKAGE.jpg")
    if os.path.isdir(os.path.dirname(jpg)):
        import shutil
        shutil.copyfile(out, jpg); shutil.copyfile(out[:-4] + ".pdf", jpg[:-4] + ".pdf")
        print("copied to", os.path.relpath(jpg, HERE))
