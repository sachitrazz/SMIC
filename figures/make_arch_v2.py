"""
make_arch_v2.py -- the architecture figure.

Top: the referenced STLAT design, with the shared input projection that
feeds both memory streams and the measured collapse.  Bottom: the SMIC
model, which removes the recurrence, the second memory and the fusion gate.

    python figures/make_arch_v2.py
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

INK = "#1a1a1a"
RED = "#b4531f"
GREEN = "#3f7d5a"
BLUE = "#2a6f97"
MUTED = "#8a8a8a"
PALE = "#eef1f3"
PALE_RED = "#fbece7"
PALE_GREEN = "#eaf2ed"


def box(ax, x, y, w, h, label, sub=None, fc=PALE, ec=INK, lw=1.3, fs=8.6,
        tc=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.018",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2 + (0.018 if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=tc, zorder=3, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.030, sub, ha="center", va="center",
                fontsize=7.0, color=MUTED, zorder=3)


def arrow(ax, p0, p1, color=INK, lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color,
                                 linewidth=lw, linestyle=ls,
                                 mutation_scale=11, zorder=4,
                                 connectionstyle="arc3,rad=%.2f" % rad))


def referenced(ax):
    ax.set_title("(a)  The referenced STLAT design, and what it actually does",
                 fontsize=10, loc="left", fontweight="bold", color=INK, pad=8)
    y = 0.46
    box(ax, .015, y, .115, .17, "frames", "$X_t$", fc="white")
    box(ax, .155, y, .125, .17, "shared\n$W_{xf},W_{xi},W_{xg}$", fc=PALE_RED,
        ec=RED, tc=RED)
    # the two streams, driven by the same projection -- the defect
    box(ax, .315, y + .105, .135, .125, "$C_t$", "temporal", fc=PALE, ec=BLUE)
    box(ax, .315, y - .105, .135, .125, "$M_t$", "spatiotemporal",
        fc=PALE, ec=BLUE)
    arrow(ax, (.130, y + .085), (.155, y + .085))
    arrow(ax, (.280, y + .085), (.315, y + .168), color=RED)
    arrow(ax, (.280, y + .085), (.315, y - .043), color=RED)

    box(ax, .485, y, .115, .17, "ASFG", "$\\alpha_t$ gate", fc=PALE)
    arrow(ax, (.450, y + .168), (.485, y + .105))
    arrow(ax, (.450, y - .043), (.485, y + .065))
    box(ax, .635, y, .115, .17, "Transformer", "8 heads", fc=PALE)
    arrow(ax, (.600, y + .085), (.635, y + .085))
    box(ax, .785, y, .105, .17, "classifier", fc="white")
    arrow(ax, (.750, y + .085), (.785, y + .085))

    ax.text(.2175, y - .075, "one projection feeds both streams",
            ha="center", fontsize=7.4, color=RED, style="italic")
    ax.annotate("", xy=(.3825, y + .10), xytext=(.3825, y + .015),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(.395, y + .058, "collapse", fontsize=7.6, color=RED,
            fontweight="bold", va="center")

    ax.text(.015, .30,
            "Measured: realised-timescale overlap $0.767$, effective rank "
            "$0.183$ of available directions.\n"
            "Proposition 2: $\\{C_t\\equiv M_t,\\ \\alpha_t\\equiv 1/2\\}$ is a "
            "critical submanifold that gradient flow cannot leave.\n"
            "Three repairs tested: band tiling $-59.0$ pts, cascade $-1.8$ "
            "(n.s.), non-convex fusion $+1.5$ (n.s.).",
            fontsize=7.8, color=INK, va="top", linespacing=1.65)
    ax.set_xlim(0, 1); ax.set_ylim(.10, .76); ax.axis("off")


def smic(ax):
    ax.set_title("(b)  The SMIC model: what the evidence supports",
                 fontsize=10, loc="left", fontweight="bold", color=INK, pad=8)
    y = .50

    # stage 1
    ax.add_patch(FancyBboxPatch((.012, y - .075), .30, .29,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor=PALE_GREEN, edgecolor=GREEN,
                                linewidth=1.2, linestyle="--", zorder=1))
    ax.text(.162, y + .185, "STAGE 1  ·  no labels", ha="center", fontsize=7.2,
            color=GREEN, fontweight="bold")
    box(ax, .030, y + .045, .12, .105, "two views", "augment", fc="white",
        ec=GREEN, fs=8.0)
    box(ax, .030, y - .055, .12, .085, "training\nsplit", fc="white", ec=GREEN,
        fs=8.0)
    box(ax, .175, y - .020, .12, .175, "NT-Xent", "contrastive", fc="white",
        ec=GREEN, fs=8.4)
    arrow(ax, (.150, y + .098), (.175, y + .098), color=GREEN)
    arrow(ax, (.150, y - .012), (.175, y + .045), color=GREEN)

    # stage 2
    box(ax, .360, y - .020, .135, .175, "conv encoder", "initialised\nfrom stage 1",
        fc=PALE_GREEN, ec=GREEN)
    arrow(ax, (.295, y + .068), (.360, y + .068), color=GREEN, lw=1.8)
    ax.text(.3275, y + .112, "weights", ha="center", fontsize=7.0, color=GREEN,
            style="italic")

    box(ax, .530, y - .020, .135, .175, "mean-pool", "16 spatial tokens\n+ LayerNorm",
        fc=PALE, ec=INK)
    arrow(ax, (.495, y + .068), (.530, y + .068))
    box(ax, .700, y - .020, .115, .175, "classifier", fc="white")
    arrow(ax, (.665, y + .068), (.700, y + .068))

    ax.text(.5975, y + .185, "STAGE 2  ·  labelled", ha="center", fontsize=7.2,
            color=MUTED, fontweight="bold")

    # what is gone
    ax.text(.845, y + .068,
            "no recurrence\nno dual memory\nno fusion gate\nno Transformer head",
            fontsize=7.6, color=RED, va="center", fontweight="bold",
            linespacing=1.6)

    ax.text(.012, .275,
            "Removing the recurrence (scan pipeline, $175$K vs $616$K params): "
            "$+10.0$ pts, $p=0.0036$, $d_z=2.74$, $5/5$ seeds.\n"
            "Contrastive initialisation: $0.618\\rightarrow0.671$, $+0.053$, "
            "$p=0.0003$, $5/5$ seeds, on the corpus that passes the leakage audit.\n"
            "Stage 1 uses no data beyond the training split: $10{,}304$ auxiliary "
            "crops add nothing ($-0.011$, $p=0.54$).  A Transformer head over the "
            "tokens was tested and does not help ($-0.006$ / $-0.035$, n.s.).",
            fontsize=7.8, color=INK, va="top", linespacing=1.65)
    ax.set_xlim(0, 1); ax.set_ylim(.10, .76); ax.axis("off")


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.2))
    referenced(axes[0])
    smic(axes[1])
    fig.subplots_adjust(left=.01, right=.99, top=.94, bottom=.02, hspace=.26)
    out = os.path.join(HERE, "ARCH.jpg")
    fig.savefig(out, dpi=260, bbox_inches="tight")
    fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")   # vector copy for print
    print("wrote", out)
    jpg = os.path.join(HERE, "..", "out", "figures", "ARCH.jpg")
    if os.path.isdir(os.path.dirname(jpg)):
        import shutil
        shutil.copyfile(out, jpg); shutil.copyfile(out[:-4] + ".pdf", jpg[:-4] + ".pdf")
        print("copied to JPG/ARCH.jpg")
