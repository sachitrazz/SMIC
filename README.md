# SMIC

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sachitrazz/SMIC/blob/main/SMIC.ipynb)

Code for *Benchmark Retrievability and Symmetry-Induced Memory Collapse in Sign
Language Recognition*. SMIC is the recurrence-free model in the paper: a
convolutional hand-crop encoder initialised by self-supervised contrastive
pretraining, read out by mean pooling and a linear classifier. The repository
also contains the referenced STLAT dual-memory model it is compared with
(`model.py`, class `ReferencedSTLAT`), the split-integrity audit, and every
script that produces a number, table or figure in the paper. Every number is
written to `results/*.json`; the tables and figures are generated from those
files.

Arm names beginning `original_` in the code and result files denote
configurations of the referenced STLAT model.

## What the paper finds

- **The benchmarks are largely retrievable.** The distributed test folders of
  all three corpora are subsets of their training folders. After content-hash
  deduplication, a parameter-free nearest-neighbour lookup still scores 0.992 on
  ISL-IEEE, and splitting by near-duplicate group instead of by image costs 20.9
  accuracy points with an identical model.
- **Dual-memory recurrence collapses.** The referenced STLAT fusion readout is
  invariant under exchanging its two memories, so the collapsed state is an
  invariant submanifold of the training objective. The two memories reach a
  realised-timescale overlap of 0.767, and three repairs fail.
- **Initialisation matters more than architecture.** SMIC is the most accurate
  model on every corpus, for example 0.848 against 0.747 for the referenced
  STLAT model on ASL-IEEE (p = 3.1e-5).

## Colab notebook

`SMIC.ipynb` reproduces the main benchmark (Table 1 of the paper) end to end on
a Colab GPU: data and lookup baseline, the five models, contrastive pretraining,
training over the five shared seeds, the results tables with paired t-tests, a
side-by-side comparison with the paper's numbers, training curves, and the
proposition checks and unit tests. It expects the prepared arrays described
under *Data* in `MyDrive/SMIC/data`. A full run takes about an hour on a T4.
GPU kernels are not bit-deterministic by default, so accuracies on other
hardware differ slightly from the paper's; the notebook prints both.

## Quick check, no data needed

The test suite checks what can be checked without the corpora: the Appendix
propositions verify numerically, the referenced STLAT model builds and runs,
seeding is reproducible, and the corpus paths are configurable.

```
python -m unittest discover -s tests -v
python theory.py
```

Both run in under a minute on a CPU.

## Data

No data is included. ISL-IEEE (doi:10.21227/796w-a432) and ASL-IEEE
(doi:10.21227/4dz0-xv55) are distributed on IEEE DataPort; the CSL corpus is
available from its authors. The preparation scripts read the raw corpora from
`$SMIC_DATA` (default `../new`), with the folders `ISL_DATA_IEEE_TRAIN`,
`ISL_DATA_IEEE_TEST`, `ASL_DATA_IEEE_TRAIN`, `ASL_DATA_IEEE_TEST`, `CSL_Train`
and `CSL_Test`, and write prepared arrays to `$SMIC_PREPARED` (default
`../prepared`).

The benchmark and the notebook use four prepared files:

| File | Content |
|---|---|
| `isl_grouped.npz` | ISL-IEEE, 35 classes, group-disjoint split |
| `asl_grouped.npz` | ASL-IEEE, 24 classes, group-disjoint split |
| `csl_signer.npz` | CSL, 97 classes, 8 sign frames per clip, signer-disjoint |
| `csl_pool.npz` | unlabelled frames of CSL signs outside the evaluation set |

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SMIC_DATA` | `../new` | raw corpora, read by the preparation scripts |
| `SMIC_PREPARED` | `../prepared` | prepared arrays and cached encoders |
| `ISL_ROOT` | `../data/ISL` | ISL images for `train.py` (controlled-task scripts) |
| `CSL_TRAIN`, `CSL_TEST` | a `gesture_type_*` frame export | CSL frame folders for `train.py` |
| `SMIC_THREADS` | up to 14 | CPU threads for the pretraining scripts |
| `SMIC_DETERMINISTIC` | unset | set to `1` for bit-identical GPU runs (see below) |
| `BENCH_EPOCHS`, `BENCH_TAG`, `BENCH_MODELS` | see `bench.py` | benchmark variants used in the paper |

## Repository layout

| Paper | Scripts | Results |
|---|---|---|
| Results 2.1, Figure 1, Supplementary Tables S4 to S6: split-integrity audit | `dupgroups.py`, `nn_leak_test.py`, `threshold_sweep.py`, `protocol_test.py`, `regroup_split.py` | `nn_leak_test.json`, `threshold_sweep.json`, `protocol_test.json` |
| Results 2.2 and 2.3, Figures 2 and 3, Supplementary Tables S2 and S3: collapse and the three repairs | `model.py`, `spectral.py`, `losses.py`, `cascade_test.py`, `positive_test.py`, `asfg_test.py`, `experiments.py` | `collapse_final.json`, `cascade_test.json`, `asfg_test.json`, `ratio_synthetic.json`, `diagnose.json` |
| Results 2.4, Table 1, Figures 4 and 5, Supplementary Tables S1 and S7: benchmark | `bench.py`, `csl_oneshot.py`, `bench_cost.py`, `pretrain_finetune.py`, `pretrain_source_control.py` | `bench_*.json`, `csl_oneshot.json`, `bench_cost.json` |
| Results 2.5, Table 2: comparison with published models | `tools/make_literature_table.py` | `bench_*.json`, `csl_oneshot.json` |
| Appendix: Propositions 1 to 3 | `theory.py` | printed checks |
| Methods 4.6: data preparation | `prepare_new.py`, `prepare_csl_signer.py`, `handcrop.py`, `data.py`, `augment.py` | prepared arrays (not included) |
| Tables and figures | `tools/`, `figures/` | written to `out/` and `figures/` |

`collapse_reg.py`, `contrastive.py` and `train.py` support additional
experiments on the controlled task that are not reported in the paper.

## Run order

1. Data preparation
   ```
   python prepare_new.py            # ISL-IEEE and ASL-IEEE: deduplication, hand crops
   python regroup_split.py          # near-duplicate-group-disjoint splits
   python prepare_csl_signer.py     # CSL: sign-frame extraction, signer-disjoint task
   ```
2. Split-integrity audit
   ```
   python run_all.py audit          # dupgroups, nn_leak_test, threshold_sweep
   python protocol_test.py
   ```
3. Controlled two-cue task (collapse and the three repairs)
   ```
   python cascade_test.py synthetic
   python positive_test.py
   python asfg_test.py
   python run_all.py ratio
   python theory.py                 # numerical check of the three propositions
   ```
4. Benchmark on ISL-IEEE, ASL-IEEE and CSL
   ```
   python bench.py isl asl csl
   BENCH_EPOCHS=110 BENCH_TAG=_steps python bench.py csl
   BENCH_MODELS=stlat_fixed BENCH_TAG=_fixed python bench.py isl asl csl
   python csl_oneshot.py
   python bench_cost.py && python bench_cost.py --flops-only
   ```
5. Tables and figures (written to `out/`)
   ```
   for f in make_bench_table make_literature_table make_dataset_table make_oneshot_table make_cost_table make_leakage_tables; do python tools/$f.py; done
   python tools/make_tables.py --results results --out out/tables
   for f in make_bench_figures make_leakage_figure make_theory_figure make_arch_v2 make_flow2; do python figures/$f.py; done
   ```

## Reproducibility

Every model in the benchmark shares the five seeds 42, 123, 456, 789 and 1024,
and every reported value is final-epoch under a cosine schedule, never a
best-epoch value selected on the validation set.

Seeding fixes the Python, NumPy and PyTorch generators. GPU convolutions are
still not bit-identical run to run by default, because cuDNN may choose
different kernels. Setting `SMIC_DETERMINISTIC=1` forces deterministic kernels.
It is off by default so that the released code reproduces the paper's numbers,
which were produced without it.

All results were produced on one workstation: Intel Core Ultra 9 185H, 32 GB RAM
and a GeForce RTX 4060 Laptop GPU, with PyTorch 2.13.0 and CUDA 12.6. The
controlled two-cue task ran on the CPU and the benchmark on the GPU.

## Requirements

Tested with Python 3.14.3 and the exact versions pinned in `requirements.txt`.
Install the CUDA build of PyTorch first if you have an NVIDIA GPU:

```
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Citation

If you use this code, please cite the paper. GitHub's "Cite this repository"
button reads `CITATION.cff`; the BibTeX entry is:

```bibtex
@article{smic2026,
  title   = {Benchmark Retrievability and Symmetry-Induced Memory Collapse in Sign Language Recognition},
  author  = {Kar, Harapriya and Chaudhary, Sachit Raj and Singh, Saurya Pratap and Viswanathan, Sushma and P, Viswanathan},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Submitted}
}
```

## Licence

MIT. See `LICENSE`.
