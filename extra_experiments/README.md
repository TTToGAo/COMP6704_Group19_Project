# Fixed-subset parameter sensitivity

Run from the repository root with Python 3.11+ and NumPy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 extra_experiments/run_sensitivity.py
```

The script uses the bundled ordered 1,500-row `ijcnn1` subset and the original Python second-order SMO and CSMO implementations. It varies one parameter at a time: `C` in `{0.5, 1, 2}` at `gamma=1/22`, or `gamma` in `{1/44, 1/22, 1/11}` at `C=1`. All runs start at zero and target KKT violation at most `0.001`. It writes `sensitivity.csv` and `metadata.json` to this directory. The included records were produced with Python 3.9.6 and NumPy 1.26.4; results may differ slightly across numerical environments. The experiment compares accepted update counts and endpoint metrics, not wall-clock times.
