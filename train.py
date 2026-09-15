"""
train.py -- training and evaluation loop.

Everything is seeded and every run records the diagnostics the paper
reports (collapse coherence, effective rank, timescale overlap), so the
ablation table and the representation analysis come from the same runs.

Usage
-----
    python train.py --dataset isl --model spectral --seeds 42 123 456
    python train.py --dataset synthetic --model original --epochs 40
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import augment as aug_mod
import collapse_reg as creg
import contrastive as con_mod
import data as data_mod
import losses as loss_mod
from model import build, count_parameters

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

ISL_ROOT = os.environ.get("ISL_ROOT", os.path.join("..", "data", "ISL"))
CSL_TRAIN = os.environ.get(
    "CSL_TRAIN", os.path.join("..", "CSL_train-20260824T063333Z-1-001", "CSL_train"))
CSL_TEST = os.environ.get(
    "CSL_TEST", os.path.join("..", "CSL_test-20260824T063301Z-1-001", "CSL_test"))


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # Off by default, so the released code reproduces the paper's numbers
    # exactly. Set SMIC_DETERMINISTIC=1 for bit-identical GPU runs; this is
    # slower, and cuDNN may then pick different kernels than the published runs.
    if os.environ.get("SMIC_DETERMINISTIC") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_dataset(name, seq_len=8, img_size=32, synth_kw=None, augment=True):
    """Returns (train_ds, val_ds, input_size, num_classes, seq_len)."""
    if name == "synthetic":
        kw = dict(n=4000, T=64, tau_s=2, tau_l=48, n_sym=4, dim=16, seed=0)
        kw.update(synth_kw or {})
        tr = data_mod.SyntheticMultiTimescale(**kw)
        va_kw = dict(kw)
        va_kw["n"] = max(500, kw["n"] // 4)
        va_kw["seed"] = kw["seed"] + 10000
        va = data_mod.SyntheticMultiTimescale(**va_kw)
        return tr, va, kw["dim"], kw["n_sym"], kw["T"]

    if name == "isl":
        # Two instances over the same file list: augmentation is applied to
        # the training view only.  The split seed is fixed at 0 for every
        # run, so all configurations see the identical split -- which is
        # what makes the paired significance tests valid.
        # Augmentation happens on the GPU inside the training loop, so both
        # views are clean and cached; see augment.affine_jitter.
        ds_tr = data_mod.ScanSequenceDataset(ISL_ROOT, seq_len=seq_len,
                                             img_size=img_size, augment=False)
        ds_va = ds_tr
        tr_i, va_i = data_mod.stratified_split(ds_tr.labels, val_frac=0.25, seed=0)
        return (Subset(ds_tr, tr_i), Subset(ds_va, va_i), ds_tr.input_size,
                len(ds_tr.classes), seq_len)

    if name == "csl":
        tr = data_mod.FrameSequenceDataset(CSL_TRAIN, seq_len=seq_len,
                                           img_size=img_size, stride=1)
        va = data_mod.FrameSequenceDataset(CSL_TEST, seq_len=seq_len,
                                           img_size=img_size, stride=1)
        # CSL test folders are a subset of the train classes; align the
        # label space so the two agree.
        va.class_to_idx = tr.class_to_idx
        va.labels = [tr.class_to_idx[os.path.basename(os.path.dirname(c[0]))]
                     for c in va.clips]
        return tr, va, tr.input_size, len(tr.classes), seq_len

    raise ValueError("unknown dataset: %s" % name)


@torch.no_grad()
def evaluate(model, loader, device, collect_diag=True, max_diag_batches=4):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    diag_states, diag_tau = [], []

    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        want = collect_diag and bi < max_diag_batches
        logits, diag = model(x, return_diagnostics=want)
        loss_sum += float(crit(logits, y).detach()) * y.size(0)
        correct += int((logits.argmax(1) == y).sum())
        total += y.size(0)
        if want and diag is not None and "states" in diag:
            diag_states.append(diag["states"].detach().float().cpu())
            diag_tau.append(diag["tau"].detach().float().cpu())

    out = {"acc": correct / max(total, 1), "loss": loss_sum / max(total, 1)}
    if diag_states:
        S = torch.cat(diag_states, dim=0)
        Tt = torch.cat(diag_tau, dim=0)
        out["coherence"] = float(loss_mod.cross_band_coherence(S, bias_correct=True))
        out["eff_rank"] = loss_mod.effective_rank(S)
        out["eff_rank_frac"] = out["eff_rank"] / (S.shape[2] * S.shape[3])
        out["pair_corr"] = loss_mod.pairwise_state_correlation(S)
        out["tau_overlap"] = loss_mod.band_timescale_overlap(Tt)
        out["tau_median"] = [float(Tt[:, :, k, :].median()) for k in range(Tt.shape[2])]
    return out


def train_one(model_name, dataset, seed, epochs=40, batch_size=32, lr=1e-3,
              weight_decay=1e-3, lambda_dec=0.0, hidden_size=128, num_bands=4,
              seq_len=8, img_size=32, num_heads=8, transformer_layers=4,
              dim_feedforward=512, dropout=0.3, device=None, synth_kw=None,
              verbose=True, grad_clip=1.0, augment=True, front_end=True,
              lambda_con=0.0, temperature=0.1, two_views=True,
              lambda_creg=0.0, creg_gamma=1.0):
    """One full training run.  Returns a result dict."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    tr, va, input_size, num_classes, T = make_dataset(
        dataset, seq_len=seq_len, img_size=img_size, synth_kw=synth_kw,
        augment=augment)

    g = torch.Generator()
    g.manual_seed(seed)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g,
                    drop_last=False)
    vl = DataLoader(va, batch_size=batch_size, shuffle=False)

    # The convolutional front-end applies to the image datasets only; the
    # synthetic benchmark is already a feature sequence.
    fe = None
    if front_end and dataset in ("isl", "csl"):
        fe = {"img_size": img_size, "channels": 1, "width": 32}

    model = build(model_name, input_size, num_classes, T,
                  hidden_size=hidden_size, num_bands=num_bands,
                  num_heads=num_heads, transformer_layers=transformer_layers,
                  dim_feedforward=dim_feedforward, dropout=dropout,
                  front_end=fe).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                           weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    best = {"acc": 0.0}
    history = []
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        run_loss = run_corr = run_n = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            img = dataset in ("isl", "csl")

            # Two augmented views per sample when the contrastive term is
            # active: this guarantees every anchor has at least one
            # same-class positive, which matters with 23 classes and a
            # modest batch.
            if lambda_con > 0 and two_views and augment and img:
                x = torch.cat([
                    aug_mod.affine_jitter(x, img_size=img_size, channels=1),
                    aug_mod.affine_jitter(x, img_size=img_size, channels=1)], 0)
                y = torch.cat([y, y], 0)
            elif augment and img:
                x = aug_mod.affine_jitter(x, img_size=img_size, channels=1)

            need_diag = lambda_dec > 0 or lambda_creg > 0
            logits, diag = model(x, return_diagnostics=need_diag)
            loss = crit(logits, y)
            if need_diag and diag is not None and "states" in diag:
                loss = loss + lambda_dec * loss_mod.decorrelation_loss(diag["states"])
            if lambda_creg > 0 and diag is not None and "states" in diag:
                reg, _ = creg.collapse_regulariser(diag["states"], gamma=creg_gamma)
                loss = loss + lambda_creg * reg
            if lambda_con > 0 and diag is not None and "feat" in diag:
                z = model.proj(diag["feat"])
                loss = loss + lambda_con * con_mod.supervised_contrastive_loss(
                    z, y, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            run_loss += float(loss.detach()) * y.size(0)
            run_corr += int((logits.argmax(1) == y).sum())
            run_n += y.size(0)

        ev = evaluate(model, vl, device, collect_diag=(ep == epochs - 1))
        history.append({"epoch": ep, "train_loss": run_loss / run_n,
                        "train_acc": run_corr / run_n,
                        "val_loss": ev["loss"], "val_acc": ev["acc"]})
        if ev["acc"] >= best.get("acc", 0.0):
            best = dict(ev)
            best["epoch"] = ep
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print("    ep %3d  train %.3f/%.3f  val %.3f/%.3f"
                  % (ep, run_loss / run_n, run_corr / run_n, ev["loss"], ev["acc"]))

    final = evaluate(model, vl, device, collect_diag=True)
    last = history[-1]
    return {
        "model": model_name, "dataset": dataset, "seed": seed,
        "params": count_parameters(model),
        "final_val_acc": final["acc"], "best_val_acc": best["acc"],
        "final_val_loss": final["loss"],
        "train_acc": last["train_acc"], "train_loss": last["train_loss"],
        "gap_acc": last["train_acc"] - final["acc"],
        "gap_loss": final["loss"] - last["train_loss"],
        "coherence": final.get("coherence"),
        "eff_rank": final.get("eff_rank"),
        "eff_rank_frac": final.get("eff_rank_frac"),
        "pair_corr": final.get("pair_corr"),
        "tau_overlap": final.get("tau_overlap"),
        "tau_median": final.get("tau_median"),
        "seconds": time.time() - t0,
        "history": history,
    }


def run_seeds(model_name, dataset, seeds=None, **kw):
    seeds = seeds or DEFAULT_SEEDS
    out = []
    for s in seeds:
        print("  [%s / %s] seed %d" % (model_name, dataset, s))
        out.append(train_one(model_name, dataset, s, **kw))
        print("      -> val acc %.4f" % out[-1]["final_val_acc"])
    return out


def summarise(runs):
    a = np.array([r["final_val_acc"] for r in runs])
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": len(a), "accs": a.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic", choices=["isl", "csl", "synthetic"])
    p.add_argument("--model", default="spectral")
    p.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--lambda-dec", type=float, default=0.0)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-bands", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    runs = run_seeds(a.model, a.dataset, seeds=a.seeds, epochs=a.epochs,
                     batch_size=a.batch_size, lr=a.lr,
                     weight_decay=a.weight_decay, lambda_dec=a.lambda_dec,
                     hidden_size=a.hidden_size, num_bands=a.num_bands,
                     seq_len=a.seq_len, img_size=a.img_size)
    s = summarise(runs)
    print("\n%s on %s: %.4f +/- %.4f over %d seeds"
          % (a.model, a.dataset, s["mean"], s["std"], s["n"]))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"summary": s, "runs": runs}, f, indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
