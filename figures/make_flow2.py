"""
make_flow2.py -- the collapse mechanism of the referenced STLAT model.

One input projection feeds both memory paths, the two paths meet in a readout
that is symmetric in them, and exchanging the two branches leaves the readout
unchanged (Proposition 2).  Black and white only.  draw(ax) is used for panel
(b) of the architecture figure (make_arch_v2.py); run on its own, this script
writes FLOW.jpg and FLOW.pdf.

    python figures/make_flow2.py
"""

import json
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

INK = "#000000"
FILL = "white"
XLIM, YLIM = (0, 10.9), (0.02, 5.35)


def overlaps():
    f = os.path.join(HERE, "..", "results", "cascade_test.json")
    a = json.load(open(f))["synthetic"]["arms"]
    return (a["original_asfg"]["tau_overlap"],
            a["original_asfg_unshared"]["tau_overlap"],
            a["original_asfg"]["eff_rank_frac"])


def box(ax, x, y, w, h, title, sub=None, lw=1.25, fs=9.2, sfs=7.4, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.030",
        facecolor=FILL, edgecolor=INK, linewidth=lw, linestyle=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.055 if sub else 0), title, ha="center",
            va="center", fontsize=fs, color=INK, zorder=4, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.075, sub, ha="center", va="center",
                fontsize=sfs, color=INK, zorder=4, linespacing=1.5)


def group(ax, x, y, w, h, title, lw=1.1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.035",
        facecolor="none", edgecolor=INK, linewidth=lw,
        linestyle=(0, (5, 3)), zorder=1))
    if title:
        ax.text(x + w / 2, y + h + 0.075, title, ha="center", va="bottom",
                fontsize=8.0, color=INK, fontweight="bold", zorder=4)


def arrow(ax, p0, p1, lw=1.35, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=12, color=INK,
        linewidth=lw, linestyle=ls, zorder=5,
        connectionstyle="arc3,rad=%.2f" % rad, shrinkA=0, shrinkB=0))


def draw(ax):
    ov_s, ov_u, er = overlaps()

    # input and the shared projection (heavy outline: this is the defect)
    box(ax, 0.030, 3.52, 0.95, 0.52, "$X_t$", "input token")
    box(ax, 1.24, 3.44, 2.05, 0.68, "Shared input projection",
        "$W_{xf},\\,W_{xi},\\,W_{xg}$", lw=2.4, fs=8.8)
    arrow(ax, (0.98, 3.78), (1.24, 3.78))
    ax.text(2.265, 3.28,
            "one set of weights, so both paths\nreceive an identical driving signal",
            ha="center", va="top", fontsize=7.5, color=INK, style="italic",
            linespacing=1.5)

    # the two memory paths
    group(ax, 3.72, 2.60, 2.34, 2.42, "TWO MEMORY PATHS")
    box(ax, 3.86, 4.06, 2.06, 0.72, "Temporal  $C_t$",
        "$C_t=f_t\\odot C_{t-1}+i_t\\odot g_t$", fs=9.0)
    box(ax, 3.86, 2.80, 2.06, 0.72, "Spatiotemporal  $M_t$",
        "$M_t=f'_t\\odot M_{t-1}+i'_t\\odot g'_t$", fs=9.0)
    for ytgt in (4.42, 3.16):
        ax.plot([3.29, 3.66, 3.66, 3.86], [3.78, 3.78, ytgt, ytgt],
                color=INK, lw=2.0, zorder=2, solid_joinstyle="round")
    arrow(ax, (3.80, 4.42), (3.86, 4.42), lw=2.0)
    arrow(ax, (3.80, 3.16), (3.86, 3.16), lw=2.0)

    # the symmetric readout
    group(ax, 6.42, 2.60, 3.05, 2.42, "ASFG READOUT, SYMMETRIC IN $C_t, M_t$", lw=1.6)
    box(ax, 6.56, 4.06, 2.77, 0.72, "Fusion gate  $\\alpha_t$",
        "$\\sigma(W_\\alpha X_t+U_\\alpha C_t+V_\\alpha M_t)$", fs=9.0)
    box(ax, 6.56, 2.80, 2.77, 0.72, "Cross term  $\\Phi_t$",
        "$\\tanh(W_cC_t+W_mM_t+W_{cm}(C_t\\odot M_t))$", fs=9.0, sfs=6.9)
    arrow(ax, (5.92, 4.42), (6.56, 4.42))
    arrow(ax, (5.92, 3.16), (6.56, 3.16))
    box(ax, 6.56, 1.72, 2.77, 0.52,
        "$H_t=O'_t\\odot[\\,\\alpha_t\\tanh C_t+(1-\\alpha_t)\\tanh M_t"
        "+\\beta\\Phi_t\\,]$", None, lw=1.9, fs=8.0)
    arrow(ax, (7.94, 2.80), (7.94, 2.24))
    box(ax, 9.90, 3.52, 0.95, 0.52, "$H_t$", "to classifier")
    ax.plot([9.33, 9.62, 9.62], [1.98, 1.98, 3.78], color=INK, lw=1.2, zorder=2)
    arrow(ax, (9.62, 3.78), (9.90, 3.78))

    # the involution
    yb = 0.16
    ax.add_patch(FancyBboxPatch(
        (0.030, yb), 10.82, 1.30, boxstyle="round,pad=0.012,rounding_size=0.035",
        facecolor=FILL, edgecolor=INK, linewidth=1.5, zorder=1))
    ax.text(0.30, yb + 1.10,
            "Proposition 2    the readout is unchanged if the two memories are swapped",
            fontsize=9.2, color=INK, fontweight="bold", va="center")
    x0, x1, ya, yc = 1.32, 2.22, yb + 0.72, yb + 0.40
    for lab, y in (("$C_t$", ya), ("$M_t$", yc)):
        ax.text(x0 - 0.10, y, lab, fontsize=9.0, ha="right", va="center", color=INK)
    arrow(ax, (x0, ya), (x1, yc), lw=1.2)
    arrow(ax, (x0, yc), (x1, ya), lw=1.2)
    for lab, y in (("$M_t$", ya), ("$C_t$", yc)):
        ax.text(x1 + 0.10, y, lab, fontsize=9.0, ha="left", va="center", color=INK)
    ax.text(2.86, yb + 0.56, "with $\\alpha_t\\!\\mapsto\\!1-\\alpha_t$",
            fontsize=8.8, va="center", color=INK)
    ax.text(4.62, yb + 0.56, "$\\Longrightarrow$", fontsize=11.0, va="center", color=INK)
    ax.text(5.12, yb + 0.56, "$H_t$ unchanged", fontsize=9.2, va="center",
            color=INK, fontweight="bold")
    ax.text(0.30, yb + 0.16,
            "so $\\{C_t\\!\\equiv\\!M_t,\\ \\alpha_t\\!\\equiv\\!\\frac{1}{2}\\}$ "
            "is a critical set of the objective that gradient flow cannot leave, "
            "and sharing $W_x$ starts training inside it.",
            fontsize=8.0, va="center", color=INK)
    ax.text(10.62, yb + 0.62,
            "measured\nstream overlap %.3f shared, %.3f separated\n"
            "effective rank %.3f" % (ov_s, ov_u, er),
            fontsize=7.6, va="center", ha="right", color=INK, style="italic",
            linespacing=1.7)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.axis("off")


def build():
    fig, ax = plt.subplots(figsize=(11.6, 6.3))
    draw(ax)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    return fig


if __name__ == "__main__":
    fig = build()
    for ext in ("jpg", "pdf"):
        p = os.path.join(HERE, "FLOW." + ext)
        fig.savefig(p, dpi=300 if ext == "jpg" else None, bbox_inches="tight", pad_inches=0.06)
        print("wrote", p)
