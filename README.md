# SMIC

Code for *Symmetry-Induced Memory Collapse and Benchmark Retrievability in Sign
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

## Data

No data is included. ISL-IEEE (doi:10.21227/796w-a432) and ASL-IEEE
(doi:10.21227/4dz0-xv55) are distributed on IEEE DataPort; the CSL corpus is
available from its authors. Set the corpus paths at the top of `prepare_new.py`
and `prepare_csl_signer.py`. Prepared arrays are written to `../prepared`.

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

## Requirements

Python 3.11 or later and the packages in `requirements.txt`.
