"""
make_arch_v2.py -- the architecture figure, three panels, black and white.

  (a) the referenced STLAT design: one input projection feeds both memory
      streams, and the measured collapse
  (b) the collapse mechanism at one step (drawn by make_flow2.draw)
  (c) the SMIC model: no recurrence, no second memory, no fusion gate

    python figures/make_arch_v2.py
"""

import os
import shutil

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import make_flow2

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

INK = "#000000"
FILL = "white"


def box(ax, x, y, w, h, label, sub=None, lw=1.3, fs=8.6, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.018",
                                facecolor=FILL, edgecolor=INK, linewidth=lw,
                                linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2 + (0.018 if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=INK, zorder=3, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.030, sub, ha="center", va="center",
                fontsize=7.0, color=INK, zorder=3)


def arrow(ax, p0, p1, lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=INK,
                                 linewidth=lw, linestyle=ls,
                                 mutation_scale=11, zorder=4,
                                 connectionstyle="arc3,rad=%.2f" % rad))


def title(ax, s):
    ax.set_title(s, fontsize=10, loc="left", fontweight="bold", color=INK, pad=8)


def referenced(ax):
    title(ax, "(a)  The referenced STLAT design, and what it actually does")
    y = 0.46
    box(ax, .015, y, .115, .17, "frames", "$X_t$")
    box(ax, .155, y, .125, .17, "shared\n$W_{xf},W_{xi},W_{xg}$", lw=2.4)
    box(ax, .315, y + .105, .135, .125, "$C_t$", "temporal")
    box(ax, .315, y - .105, .135, .125, "$M_t$", "spatiotemporal")
    arrow(ax, (.130, y + .085), (.155, y + .085))
    arrow(ax, (.280, y + .085), (.315, y + .168), lw=2.0)
    arrow(ax, (.280, y + .085), (.315, y - .043), lw=2.0)
    box(ax, .485, y, .115, .17, "ASFG", "$\\alpha_t$ gate")
    arrow(ax, (.450, y + .168), (.485, y + .105))
    arrow(ax, (.450, y - .043), (.485, y + .065))
    box(ax, .635, y, .115, .17, "Transformer")
    arrow(ax, (.600, y + .085), (.635, y + .085))
    box(ax, .785, y, .105, .17, "classifier")
    arrow(ax, (.750, y + .085), (.785, y + .085))
    ax.text(.2175, y - .075, "one projection feeds both streams",
            ha="center", fontsize=7.4, color=INK, style="italic")
    ax.annotate("", xy=(.3825, y + .10), xytext=(.3825, y + .015),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.5))
    ax.text(.395, y + .058, "collapse", fontsize=7.6, color=INK,
            fontweight="bold", va="center")
    ax.text(.015, .30,
            "Measured: realised-timescale overlap $0.767$, effective rank "
            "$0.183$ of available directions.\n"
            "Proposition 2: $\\{C_t\\equiv M_t,\\ \\alpha_t\\equiv 1/2\\}$ is an "
            "invariant set: once the memories are identical, training cannot separate them.\n"
            "Three repairs tested: separate projections $+7.6$ pts (n.s.), "
            "band tiling $-59.0$ pts, cascade $-1.8$ (n.s.).",
            fontsize=7.8, color=INK, va="top", linespacing=1.65)
    ax.set_xlim(0, 1); ax.set_ylim(.10, .76); ax.axis("off")


def smic(ax):
    title(ax, "(c)  The SMIC model: what the evidence supports")
    y = .50
    ax.add_patch(FancyBboxPatch((.012, y - .075), .30, .29,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor=FILL, edgecolor=INK,
                                linewidth=1.2, linestyle="--", zorder=1))
    ax.text(.162, y + .185, "STAGE 1  ·  no labels", ha="center", fontsize=7.2,
            color=INK, fontweight="bold")
    box(ax, .030, y + .045, .12, .105, "two views", "augment", fs=8.0)
    box(ax, .030, y - .055, .12, .085, "unlabelled\nimages", fs=8.0)
    box(ax, .175, y - .020, .12, .175, "NT-Xent", "contrastive", fs=8.4)
    arrow(ax, (.150, y + .098), (.175, y + .098))
    arrow(ax, (.150, y - .012), (.175, y + .045))
    box(ax, .360, y - .020, .135, .175, "conv encoder", "initialised\nfrom stage 1", lw=2.0)
    arrow(ax, (.295, y + .068), (.360, y + .068), lw=1.8)
    ax.text(.3275, y + .112, "weights", ha="center", fontsize=7.0, color=INK, style="italic")
    box(ax, .530, y - .020, .135, .175, "mean-pool", "16 spatial / 8 temporal\ntokens + LayerNorm")
    arrow(ax, (.495, y + .068), (.530, y + .068))
    box(ax, .700, y - .020, .115, .175, "classifier")
    arrow(ax, (.665, y + .068), (.700, y + .068))
    ax.text(.5975, y + .185, "STAGE 2  ·  labelled", ha="center", fontsize=7.2,
            color=INK, fontweight="bold")
    ax.text(.845, y + .068,
            "no recurrence\nno dual memory\nno fusion gate\nno Transformer head",
            fontsize=7.6, color=INK, va="center", fontweight="bold", linespacing=1.6)
    ax.set_xlim(0, 1); ax.set_ylim(.36, .76); ax.axis("off")


if __name__ == "__main__":
    # axes sized so each panel keeps the proportions it was drawn at
    W = 11.0
    ha, hb, hc, gap, top = 2.52, 5.87, 1.53, 0.50, 0.40
    H = 0.05 + hc + gap + hb + gap + ha + top
    fig = plt.figure(figsize=(W, H))
    ax_c = fig.add_axes([0.1 / W, 0.05 / H, 10.8 / W, hc / H])
    ax_b = fig.add_axes([0.1 / W, (0.05 + hc + gap) / H, 10.8 / W, hb / H])
    ax_a = fig.add_axes([0.1 / W, (0.05 + hc + gap + hb + gap) / H, 10.8 / W, ha / H])
    referenced(ax_a)
    make_flow2.draw(ax_b)
    title(ax_b, "(b)  The collapse mechanism at one step")
    smic(ax_c)
    out = os.path.join(HERE, "ARCH.jpg")
    fig.savefig(out, dpi=260, bbox_inches="tight")
    fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")   # vector copy for print
    print("wrote", out)
    jpg = os.path.join(HERE, "..", "out", "figures", "ARCH.jpg")
    if os.path.isdir(os.path.dirname(jpg)):
        shutil.copyfile(out, jpg); shutil.copyfile(out[:-4] + ".pdf", jpg[:-4] + ".pdf")
        print("copied to out/figures/ARCH.jpg")
