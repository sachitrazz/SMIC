r"""
make_dataset_table.py -- the three corpora used, as audited and as split.

Every number is read from prepared/*_meta.json, prepared/*.npz or
results/*.json; nothing is typed in except the distributed file counts,
which the audit of Section 4 established by hashing.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = os.path.join(HERE, "..", "..", "prepared")
RES = os.path.join(HERE, "..", "results")
TABLES = os.path.join(HERE, "..", "out", "tables")
NL = r" \\"


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def thousands(n):
    return "{:,}".format(int(n)).replace(",", "{,}")


def main():
    sw = {e["dataset"]: e for e in (J(os.path.join(RES, "threshold_sweep.json")) or [])}

    def groups(ds):
        for r in sw.get(ds, {}).get("rows", []):
            if abs(r["threshold"] - 0.98) < 1e-9:
                return r["n_groups"]
        return None

    def lookup(ds):
        b = J(os.path.join(RES, "bench_%s.json" % ds))
        return b["lookup"] if b else None

    rows = []
    for key, label, files in (("isl", "ISL-IEEE", 1225), ("asl", "ASL-IEEE", 840)):
        m = J(os.path.join(PREP, key + "_meta.json")) or {}
        g = J(os.path.join(PREP, key + "_grouped_meta.json")) or {}
        rows.append((label, "still images", thousands(files), thousands(m.get("n", 0)),
                     len(m.get("classes", [])), groups(key),
                     "%d / %d, group-disjoint" % (g.get("n_train", 0), g.get("n_val", 0)),
                     lookup(key)))
    c = J(os.path.join(PREP, "csl_signer_meta.json")) or {}
    rows.append(("CSL", "video, %d sign frames" % c.get("T", 8),
                 thousands(c.get("n_clips_total", 1302)) + " clips", "n/a",
                 c.get("n_classes", 0), "by signer",
                 "%d / %d, signer-disjoint" % (c.get("n_train", 0), c.get("n_val", 0)),
                 lookup("csl")))
    pool = os.path.join(PREP, "csl_pool.npz")
    if os.path.exists(pool):
        npool = np.load(pool)["y" if "y" in np.load(pool).files else "X"].shape[0]
        rows.append((r"CSL pool$^\dagger$", "unlabelled frames", thousands(npool), "n/a",
                     "none", "n/a", "pretraining only", None))

    lines = [
        r"\begin{table*}[!t]",
        r"\caption{The three corpora, as audited and as split. \emph{Unique} "
        r"is the count after MD5 deduplication; \emph{Groups} is the number of "
        r"independent near-duplicate components (cosine $>0.98$), which is the "
        r"effective sample size of a still-image corpus; \emph{Lookup} is the "
        r"parameter-free nearest-neighbour baseline on the split used. CSL is "
        r"split by signer instead: each sign's training clip and test clips "
        r"come from different people. $^\dagger$Frames drawn from CSL signs "
        r"that do not occur in the evaluation task, used without labels to "
        r"initialise the CSL model. Every figure was computed from the "
        r"material itself.}",
        r"\label{tab:datasets}",
        r"\centering",
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrlc}",
        r"\toprule",
        r"Corpus & Input & Files & Unique & Classes & Groups & Train / test & Lookup" + NL,
        r"\midrule",
    ]
    for lab, inp, f, u, cl, g, sp, lk in rows:
        lines.append("%s & %s & %s & %s & %s & %s & %s & %s%s" % (
            lab, inp, f, u, cl, g if g is not None else "n/a", sp,
            ("%.3f" % lk) if isinstance(lk, float) else "n/a", NL))
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    os.makedirs(TABLES, exist_ok=True)
    open(os.path.join(TABLES, "datasets.tex"), "w").write("\n".join(lines))
    print("wrote tables/datasets.tex")


if __name__ == "__main__":
    main()
