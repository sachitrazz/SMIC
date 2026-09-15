r"""
make_leakage_tables.py -- LaTeX tables for the split-integrity results.

Writes into out/tables/:
  leakage.tex    dataset integrity audit (files -> unique -> groups -> lookup)
  protocol.tex   one model, three splits
"""

import json
import os


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
TABLES = os.path.join(HERE, "..", "out", "tables")


def load(n, base=RESULTS):
    p = os.path.join(base, n)
    return json.load(open(p)) if os.path.exists(p) else None


def write(name, lines):
    os.makedirs(TABLES, exist_ok=True)
    p = os.path.join(TABLES, name)
    with open(p, "w") as f:
        f.write("\n".join(lines))
    print("wrote", os.path.relpath(p, HERE))


NL = r" \\"


# ----------------------------------------------------------------------

def leakage():
    nn = load("nn_leak_test.json") or []
    by = {r["dataset"]: r for r in nn}

    # group counts come from the sweep at threshold 0.98 rather than being
    # written in by hand, so a re-preparation of the data cannot leave a
    # stale number in the table
    sw = {e["dataset"]: e for e in (load("threshold_sweep.json") or [])}
    def groups(ds, default=0):
        e = sw.get(ds)
        if not e:
            return default
        for r_ in e["rows"]:
            if abs(r_["threshold"] - 0.98) < 1e-9:
                return r_["n_groups"]
        return default

    rows = []
    if "isl" in by:
        b = by["isl"]
        rows.append(("ISL-IEEE", 1225, 1006, groups("isl", 216),
                     100 * b["frac_above_0.98"], b["nn_acc"], b["chance"]))
    if "asl" in by:
        b = by["asl"]
        rows.append(("ASL-IEEE", 840, 419, groups("asl", 357),
                     100 * b["frac_above_0.98"], b["nn_acc"], b["chance"]))

    lines = [
        r"\begin{table}[!t]",
        r"\caption{Dataset integrity audit.  \emph{Groups} counts connected "
        r"components of the near-duplicate graph (cosine $>0.98$ between "
        r"$32\times32$ grayscale descriptors) and is the number of genuinely "
        r"independent photographs the set contains.  \emph{Twin} is the "
        r"percentage of validation images having a training image above that "
        r"threshold.  \emph{Lookup} copies the label of the nearest training "
        r"image in pixel space, with no learning of any kind; where it "
        r"approaches a trained network's accuracy, the network is performing "
        r"the same lookup.}",
        r"\label{tab:leakage}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Files & MD5-uniq. & Groups & Twin (\%) & Lookup & Chance"
        + NL,
        r"\midrule",
    ]
    for n, f, u, g, tw, lk, ch in rows:
        lines.append("%s & %d & %d & %d & %.1f & %.3f & %.3f%s"
                     % (n, f, u, g, tw, lk, ch, NL))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("leakage.tex", lines)


def protocol():
    pt = load("protocol_test.json")
    if not pt:
        print("skip protocol.tex (no results yet)")
        return
    lines = [
        r"\begin{table}[!t]",
        r"\caption{One model, one schedule, three evaluation protocols over "
        r"the same ISL images; only the train/validation partition differs.  "
        r"Mean $\pm$ s.d. over %d seeds.  The last column is the no-learning "
        r"nearest-neighbour baseline on the same partition.}" % len(pt["seeds"]),
        r"\label{tab:protocol}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lrrcc}",
        r"\toprule",
        r"Protocol & Train & Val & Accuracy & Lookup" + NL,
        r"\midrule",
    ]
    for r_ in pt["rows"]:
        lines.append(r"%s & %d & %d & $%.3f \pm %.3f$ & %.3f%s"
                     % (r_["protocol"].replace("  ", r"\quad "), r_["n_train"],
                        r_["n_val"], r_["mean"], r_["std"], r_["nn_lookup"], NL))
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\footnotesize Removing near-duplicate leakage "
        r"(B$\rightarrow$C) costs %.1f accuracy points, $p=%.2g$.}%s"
        % (100 * pt["B_vs_C"]["drop"], pt["B_vs_C"]["p"], NL),
        r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("protocol.tex", lines)


def threshold():
    sw = load("threshold_sweep.json")
    if not sw:
        print("skip threshold.tex (no results yet)")
        return
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Sensitivity of the group-disjoint split to the "
        r"near-duplicate threshold.  \emph{Lookup} is the parameter-free "
        r"nearest-neighbour baseline on the split that threshold produces; "
        r"\emph{Largest} is the size of the biggest near-duplicate group, "
        r"which indicates when single-linkage components begin to chain.  "
        r"ISL retains a high lookup baseline at every threshold that keeps "
        r"its classes splittable.}",
        r"\label{tab:threshold}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Set & Thresh. & Groups & Classes & Val & Largest & Lookup" + NL,
        r"\midrule",
    ]
    for e in sw:
        for k, r_ in enumerate(e["rows"]):
            lines.append("%s & %.3f & %d & %d & %d & %d & %.3f%s"
                         % (e["dataset"].upper() if k == 0 else "",
                            r_["threshold"], r_["n_groups"], r_["n_classes"],
                            r_["n_val"], r_.get("largest_group", 0),
                            r_["lookup_acc"], NL))
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    write("threshold.tex", lines)


if __name__ == "__main__":
    leakage()
    threshold()
    protocol()
