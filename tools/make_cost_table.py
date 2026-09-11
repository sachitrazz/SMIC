r"""
make_cost_table.py -- tables/cost.tex from results/bench_cost.json.

Measured on this machine only: parameters, FLOPs per forward pass, and
single-thread CPU latency at batch 1.  No GPU number is invented.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "bench_cost.json")
TABLES = os.path.join(HERE, "..", "out", "tables")
NL = r" \\"

LABEL = {"stlat": "STLAT (dual memory + ASFG)",
         "lstm": "CNN + LSTM",
         "cnn_tf": "CNN + Transformer",
         "cnn": "CNN; also SMIC at inference"}
ORDER = ["stlat", "lstm", "cnn_tf", "cnn"]
INPUT = {"image": r"image, $3\times64\times64$",
         "video": r"clip, $8\times3\times64\times64$"}


def fmt(n):
    if n is None:
        return "n/a"
    return "%.0fK" % (n / 1e3) if n < 1e6 else ("%.1fM" % (n / 1e6) if n < 1e9 else "%.2fG" % (n / 1e9))


def main():
    if not os.path.exists(RES):
        print("skip cost.tex (no results yet)"); return
    d = json.load(open(RES))
    L = [
        r"\begin{table}[!t]",
        r"\caption{Measured cost of the benchmark models: parameters (for "
        r"the $35$-class image and $97$-class clip heads), FLOPs per forward "
        r"pass, and single-thread CPU latency at batch size 1 (median and "
        r"90th percentile of 50 passes) on the workstation of "
        r"Section~\ref{sec4}. The SMIC model is the CNN network with a "
        r"different initialisation, so its inference cost is the CNN row. No "
        r"GPU latency is reported, because none was measured.}",
        r"\label{tab:cost}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Input & Params & FLOPs & ms (med) & ms (p90)" + NL,
        r"\midrule",
    ]
    for kind in ("image", "video"):
        for m in ORDER:
            v = d[kind].get(m)
            if not v:
                continue
            L.append("%s & %s & %s & %s & %.2f & %.2f%s" % (
                LABEL[m], INPUT[kind], fmt(v["params"]), fmt(v["flops"]),
                v["latency_ms_median"], v["latency_ms_p90"], NL))
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(TABLES, exist_ok=True)
    open(os.path.join(TABLES, "cost.tex"), "w").write("\n".join(L))
    print("wrote tables/cost.tex")


if __name__ == "__main__":
    main()
