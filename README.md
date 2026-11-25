# Constrained Optimization for SVM Training

This repository compares four solvers for the binary soft-margin support vector machine (SVM) dual: Platt SMO, second-order working-set SMO, conjugate SMO (CSMO), and projected gradient descent (PGD). The main question is whether CSMO's lower update count translates into lower training time than a second-order SMO baseline on the same problem.

## Optimization problem

For labels `yᵢ ∈ {−1,+1}` and `Qᵢⱼ = yᵢyⱼK(xᵢ,xⱼ)`, the solvers maximize

`D(α) = 1ᵀα − ½αᵀQα`, subject to `0 ≤ αᵢ ≤ C` and `yᵀα = 0`.

The shared configuration uses an RBF kernel, `C = 1`, `γ = 1/22`, zero initialization and a full-set KKT tolerance of `0.001`. Data rows and feature values retain their supplied order.

## Data and experiment design

| Data | Size | Purpose |
|---|---:|---|
| `heart_scale` | 270 rows, 13 features | Check feasibility and solution quality across all four Python solvers |
| Fixed `ijcnn1` subset | 1,500 rows, 22 features | Compare Python second-order SMO and CSMO convergence on identical rows |
| Full `ijcnn1` | 49,990 training and 91,701 test rows, 22 features | Compare author-solver modes in five paired training runs |

The subset was selected by label-stratified sampling with seed 19 and contains 1,354 negative and 146 positive rows. Its ordered zero-based indices are in `B_second_order/original_reproduce/inputs/ijcnn1_seed19/indices.npy`; the extracted LIBSVM-format rows are in `data/ijcnn1_seed19.libsvm`. The full train/test files are available from the [LIBSVM binary classification collection](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html). The expected SHA-256 of the decompressed training file is `16506cad788cf7c9607454150ed1994788204bac2ff4c9cb3b320036b6950d3f`.

## Repository structure

| Path | Contents |
|---|---|
| `A_platt/` | Platt SMO source and small input |
| `B_second_order/original_reproduce/` | Original second-order SMO source, fixed inputs, saved endpoints, and a separate Linux author-solver benchmark archive |
| `B_second_order/second_order_smo.py` | Independent baseline module used by the common command-line entry point |
| `C_csmo/` | CSMO source |
| `D_pgd/` | PGD source, configuration, and shared data/metric utilities |
| `run_python_solver.py` | Common command-line entry point for the baseline module and CSMO |
| `data/` | Ordered fixed subset and indices |
| `results/` | Saved endpoints and full-data timing records |
| `extra_experiments/` | Parameter-sensitivity script, raw table, and environment metadata |
| `figures/` | Script and rendered paired-timing and sensitivity figures used in the report |
| `SHA256SUMS.txt` | Checksums of repository files |

## Requirements and quick start

Use Python 3.11+ with NumPy (`1.26` or newer, below `3`). Matplotlib is needed only for the optional convergence plotting script in the original second-order folder. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy>=1.26,<3' 'matplotlib>=3.6,<4'
```

Run the four methods on the bundled small dataset:

```bash
python A_platt/platt_smo.py A_platt/heart_scale --out results/platt_smo_new --C 1 --gamma 0.045454545454545456 --epsilon 0.001
python run_python_solver.py --algorithm second_order_smo --data A_platt/heart_scale --out results/smo2_heart_new
python run_python_solver.py --algorithm csmo --data A_platt/heart_scale --out results/csmo_heart_new
python D_pgd/run_pgd.py --data D_pgd/data/heart_scale --config D_pgd/configs/heart.json --out results/pgd_heart_new
```

Run the second-order baseline and CSMO on the same fixed subset:

```bash
python run_python_solver.py --algorithm second_order_smo --data data/ijcnn1_seed19.libsvm --out results/smo2_subset_new
python run_python_solver.py --algorithm csmo --data data/ijcnn1_seed19.libsvm --out results/csmo_subset_new
```

To reproduce the saved second-order results with the original baseline implementation, run `python run_python.py heart_scale ijcnn1_seed19` from `B_second_order/original_reproduce/`. Solver outputs include configuration, summary, final `alpha.npy`, and convergence history where recorded. The original CSMO subset endpoint was produced in a different numerical environment; the common entry point may take a different number of updates while still satisfying the same KKT criterion.

## Recorded results

### Solution quality and subset convergence

| Data | Solver | Accepted updates | KKT violation | Maximized dual objective |
|---|---|---:|---:|---:|
| `heart_scale` | Platt SMO | 632 | 0.000940105 | 106.5191661 |
| `heart_scale` | Second-order SMO | 120 | 0.000763615 | 106.5191673 |
| `heart_scale` | CSMO | 119 | 0.000931543 | 106.5191688 |
| `heart_scale` | PGD | 9,785 | 0.000999706 | 106.5191603 |
| Fixed `ijcnn1` subset | Second-order SMO | 467 | 0.000951301 | 288.6010353 |
| Fixed `ijcnn1` subset | CSMO | 424 | 0.000991340 | 288.6010698 |

All listed endpoints meet the shared KKT threshold and have negligible equality and box residuals. An update is a different operation in PGD and SMO, so these counts are not a four-way speed ranking.

### Full-data paired timing

The archived Apple M6 comparison used the same author-solver build for both modes, a 100 MB kernel cache, disabled shrinking, one CPU thread, and a timer around `svm_train()` only. Across five paired rounds, second-order SMO required 6,730 updates and a median 6.733 s; CSMO required 6,627 updates and a median 6.839 s. The median of the five **within-round** ratios `T_SMO/T_CSMO` was 0.944. Thus CSMO saved 103 updates but did not reduce elapsed training time in this recorded setting.

The five paired rows, mode summary and endpoint audit are in `results/apple_m6_recorded/`. Verify the archived arithmetic with:

```bash
python results/apple_m6_recorded/verify_recorded.py
```

The archived table records one Apple M6 session. To rerun the same **benchmark protocol** on your own machine, use the pinned author source and instrumentation script below. It downloads the official full dataset, validates both solver modes on the bundled small and fixed-subset inputs, then runs five paired full-data rounds and saves logs, model coefficients, build metadata, endpoint checks, and summary tables:

```bash
cd B_second_order/original_reproduce
python3 run_author.py --revision 432631da95845152ce24404fbecb93a5e546e586
```

The command needs Git, a C++ compiler, Python 3.11+, NumPy, and network access for the source and official train/test files. If the LIBSVM host is unavailable, place the verified decompressed `ijcnn1` and `ijcnn1.t` files in `B_second_order/original_reproduce/work/data/` and rerun; the script checks their hashes and split sizes. Outputs go to a timestamped `results/author/` directory. Timing values depend on hardware, compiler and system load, so compare the generated paired ratios and endpoint accuracy within the new run rather than expecting the archived seconds to match. The separately archived Linux run in this directory gives a complete example of that output. `run_full_pairs.py` is an optional process-level timer; its scope includes startup and file I/O and must be labeled separately from the `svm_train()`-only benchmark.

### Parameter sensitivity

`extra_experiments/run_sensitivity.py` varies one parameter at a time on the same fixed subset: `C ∈ {0.5, 1, 2}` at `γ=1/22`, and `γ ∈ {1/44, 1/22, 1/11}` at `C=1`. It records accepted updates, objective, KKT and feasibility for both original Python solvers. Run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python extra_experiments/run_sensitivity.py
```

In the saved run, all ten endpoints met KKT `≤0.001`, and CSMO used fewer updates in each of the five settings (4.46%–9.98% fewer). This is a subset update-count check; it makes no full-data timing claim. See `extra_experiments/sensitivity.csv`, `metadata.json`, and the folder's README for exact values and environment details. Regenerate the paired-timing and sensitivity figures with `python figures/make_figures.py`; the convergence figure can be regenerated with `B_second_order/original_reproduce/plot_convergence.py`.

## References

- J. C. Platt, *Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines*, 1998.
- R.-E. Fan, P.-H. Chen, and C.-J. Lin, “Working Set Selection Using Second Order Information for Training Support Vector Machines,” *JMLR*, 2005.
- A. Torres-Barrán, C. M. Alaíz, and J. R. Dorronsoro, “Faster SVM Training via Conjugate SMO,” *Pattern Recognition*, 2021.
- C.-C. Chang and C.-J. Lin, “LIBSVM: A Library for Support Vector Machines,” *ACM TIST*, 2011.
