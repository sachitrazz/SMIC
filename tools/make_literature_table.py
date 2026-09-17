r"""
make_literature_table.py -- comparison with published spatiotemporal models.

Two kinds of row, never mixed statistically:

  literature-reported   the number stated in the cited paper, on that paper's
                        dataset and protocol.  Each figure below was checked
                        against the source before inclusion: the abstract
                        (arXiv API) where it states one, otherwise the
                        paper's own results table or text (velmathi2023,
                        Fig. 7.2; jing2025, Section 5.3).
  direct experimental   our own models on our own partitions, read from
                        results/bench_*.json.

No significance test is run across the two, and no cross-paper superiority
is claimed: the datasets, protocols and modalities differ.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TABLES = os.path.join(HERE, "..", "out", "tables")
NL = r" \\"

# (citation key, model, sequence model family, dataset, input, reported result)
LIT = [
    (r"alishzade2025", "ConvLSTM vs.~Vanilla Transformer", "recurrent vs.~attention",
     "WLASL; AzSLD", "RGB video", r"Transformer $88.3\%$ (WLASL), $76.8\%$ (AzSLD) Top-1"),
    (r"kautsar2024", "LSTM vs.~1D-CNN + Transformer", "recurrent vs.~attention",
     "BISINDO (word)", "sequence input", r"LSTM $94.67\%$; 1D-CNN+Transformer $96.12\%$"),
    (r"sandoval2023", "MaskFeat video Transformer", "attention, self-supervised",
     "WLASL2000", "RGB video, gloss-based", r"$79.02\%$ Top-1"),
    (r"velmathi2023", "CNN vs.~LSTM", "convolutional vs.~recurrent",
     "ISL: 26 letters; 5 phrases", "MediaPipe keypoints",
     r"letters: CNN $97.8\%$, LSTM $85.6\%$; phrases: LSTM $100\%$, CNN $90\%$"),
    (r"jing2025", "CNN + Transformer", "attention",
     "NationalCSL6707", "RGB video, two views", r"$69.61\%$ Top-1"),
    (r"nguyen2025", "SignBart", "attention (encoder-decoder)",
     "LSA-64", "skeleton sequence", r"$96.04\%$ ($749{,}888$ parameters)"),
]

OURS = {"stlat": "STLAT (dual memory + ASFG)", "cnn_tf": "CNN + Transformer",
        "lstm": "CNN + LSTM", "proposed": "SMIC (CNN, contrastive init.)"}
FAMILY = {"stlat": "recurrent, dual memory", "lstm": "recurrent",
          "cnn_tf": "attention", "proposed": "none (pooled)"}
DS = {"isl": "ISL-IEEE", "asl": "ASL-IEEE", "csl": "CSL, signer-disjoint"}


def main():
    L = [
        r"\begin{table*}[!t]",
        r"\caption{Positioning against published spatiotemporal sign recognition "
        r"models. \emph{Literature-reported} rows quote the figure stated in the "
        r"cited work, on its own dataset and protocol, and are not re-run here. "
        r"\emph{Direct experimental} rows are our own models on our partitions "
        r"(Table~\ref{tab:bench}), with every input a hand crop. Because datasets, "
        r"protocols and input modalities differ, no row is statistically compared "
        r"with any other across the two groups, and no superiority is claimed. What "
        r"the literature rows share with this study is a within-paper comparison of "
        r"sequence models (recurrent against convolutional or attention-based) over "
        r"a common front end.}",
        r"\label{tab:literature}",
        r"\centering",
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Model & Sequence model & Dataset & Reported result" + NL,
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Literature-reported}}" + NL,
    ]
    for key, model, fam, ds, _input, res in LIT:
        L.append(r"%s~\cite{%s} & %s & %s & %s%s" % (model, key, fam, ds, res, NL))
    L.append(r"\midrule")
    L.append(r"\multicolumn{4}{l}{\textit{Direct experimental (this study)}}" + NL)
    for ds in ("isl", "asl", "csl"):
        f = os.path.join(RES, "bench_%s.json" % ds)
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for m in ("stlat", "lstm", "cnn_tf", "proposed"):
            if m in d["models"]:
                L.append(r"%s & %s & %s & $%.1f\%%$ accuracy (lookup $%.1f\%%$)%s"
                         % (OURS[m], FAMILY[m], DS[ds], 100 * d["models"][m]["mean"]["acc"],
                            100 * d["lookup"], NL))
    one = os.path.join(RES, "csl_oneshot.json")
    if os.path.exists(one):
        o = json.load(open(one))
        L.append(r"SMIC trunk, nearest prototype & none (pooled) & CSL, signer-disjoint & "
                 r"$%.1f\%%$ accuracy (lookup $%.1f\%%$)%s"
                 % (100 * o["contrastive"]["acc"], 100 * o["lookup"], NL))
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    os.makedirs(TABLES, exist_ok=True)
    open(os.path.join(TABLES, "literature.tex"), "w").write("\n".join(L))
    print("wrote out/tables/literature.tex")


if __name__ == "__main__":
    main()
