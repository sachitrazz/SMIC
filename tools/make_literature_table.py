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

# (citation key, model, sequence model family, dataset, protocol note, reported)
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

NAMES = {'alishzade2025': 'Alishzade et~al.', 'kautsar2024': 'Kautsar et~al.', 'sandoval2023': 'Sandoval-Castaneda et~al.', 'velmathi2023': 'Velmathi \\& Goyal', 'jing2025': 'Jing et~al.', 'nguyen2025': 'Nguyen \\& Tran'}

OURS = {"stlat": "STLAT (dual memory + ASFG)", "cnn_tf": "CNN + Transformer",
        "lstm": "CNN + LSTM", "proposed": "SMIC (CNN, contrastive init.)"}
DS = {"isl": "ISL-IEEE", "asl": "ASL-IEEE", "csl": "CSL, signer-disjoint"}


def main():
    L = [
        r"\begin{table*}[!t]",
        r"\caption{Positioning against published spatiotemporal sign "
        r"recognition models. \emph{Literature-reported} rows quote the "
        r"figure stated in the cited work, on its own dataset and protocol, "
        r"and are not re-run here. \emph{Direct} rows are our own models on "
        r"our partitions (Table~\ref{tab:bench}). Because datasets, protocols "
        r"and input modalities differ, no row is statistically compared with "
        r"any other across the two groups, and no superiority is claimed. What "
        r"the literature rows share with this study is a within-paper "
        r"comparison of sequence models (recurrent against convolutional or "
        r"attention-based) over a common front end.}",
        r"\label{tab:literature}",
        r"\centering",
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Source & Model & Sequence model & Dataset & Input & Reported result" + NL,
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Literature-reported}}" + NL,
    ]
    for key, model, fam, ds, inp, res in LIT:
        L.append(r"%s\cite{%s} & %s & %s & %s & %s & %s%s" % (NAMES[key], key, model, fam, ds, inp, res, NL))
    L.append(r"\midrule")
    L.append(r"\multicolumn{6}{l}{\textit{Direct experimental (this study)}}" + NL)
    any_ours = False
    for ds in ("isl", "asl", "csl"):
        f = os.path.join(RES, "bench_%s.json" % ds)
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for m in ("stlat", "lstm", "cnn_tf", "proposed"):
            if m not in d["models"]:
                continue
            mu = d["models"][m]["mean"]
            fam = {"stlat": "recurrent, dual memory", "lstm": "recurrent",
                   "cnn_tf": "attention", "proposed": "none (pooled)"}[m]
            L.append(r"This study & %s & %s & %s & hand crop & $%.1f\%%$ accuracy (lookup $%.1f\%%$)%s"
                     % (OURS[m], fam, DS[ds], 100 * mu["acc"], 100 * d["lookup"], NL))
            any_ours = True
    one = os.path.join(RES, "csl_oneshot.json")
    if os.path.exists(one):
        o = json.load(open(one))
        L.append(r"This study & SMIC trunk, nearest prototype & none (pooled) & "
                 r"CSL, signer-disjoint & hand crop & $%.1f\%%$ accuracy (lookup $%.1f\%%$)%s"
                 % (100 * o["contrastive"]["acc"], 100 * o["lookup"], NL))
    if not any_ours:
        L.append(r"\multicolumn{6}{l}{(pending)}" + NL)
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    os.makedirs(TABLES, exist_ok=True)
    open(os.path.join(TABLES, "literature.tex"), "w").write("\n".join(L))
    print("wrote tables/literature.tex")


if __name__ == "__main__":
    main()
