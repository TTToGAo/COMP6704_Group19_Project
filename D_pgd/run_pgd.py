"""Run the small-data PGD check and export the four Group 19 artifacts."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter


ROOT = Path(__file__).resolve().parent
SOURCES = ("common/data.py", "common/kernel.py", "common/metrics.py",
           "pgd.py", "run_pgd.py")


def source_version():
    digest = hashlib.sha256()
    for name in SOURCES:
        digest.update(name.encode() + b"\0" + (ROOT / name).read_bytes() + b"\0")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"implementation_commit": commit, "implementation_sha256": digest.hexdigest()}


def write_json(path, record):
    # JSON has no NaN literal. The stop reason distinguishes invalid diagnostics.
    record = {k: None if isinstance(v, float) and not math.isfinite(v) else v
              for k, v in record.items()}
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New or empty run directory")
    args = parser.parse_args()
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error(f"{args.out} already contains results. Choose a fresh --out directory.")

    # Set limits before importing NumPy. The same command works with the common BLAS backends.
    for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np
    from common.data import load_libsvm
    from pgd import DEFAULT_LOG_EVERY, DEFAULT_MAX_ITER, HISTORY_FIELDS, solve

    settings = json.loads(args.config.read_text(encoding="utf-8"))
    params = {key: settings[key] for key in ("C", "gamma", "epsilon")}
    params["max_iter"] = settings.get("max_iter", DEFAULT_MAX_ITER)
    params["log_every"] = settings.get("log_every", DEFAULT_LOG_EVERY)
    if (settings.get("kernel", "rbf") != "rbf" or settings.get("algorithm", "pgd") != "pgd"
            or settings.get("alpha0", "zeros") != "zeros"):
        parser.error("The PGD experiment expects an RBF kernel and a zero start.")
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if settings.get("dataset_hash", data_hash) != data_hash:
        parser.error("The dataset SHA-256 differs from the saved run configuration.")
    X, y = load_libsvm(args.data)
    config = {
        "dataset_id": args.data.name,
        "dataset_hash": data_hash,
        "sample_indices_file": None, "n_train": len(y), "n_features": X.shape[1],
        "kernel": "rbf", **params, "alpha0": "zeros", "algorithm": "pgd",
        **source_version(), "machine": platform.platform(), "threads": 1,
        "python": platform.python_version(), "numpy": np.__version__,
    }
    start = perf_counter()
    result = solve(X, y, **params)
    elapsed = perf_counter() - start
    config["step_size"] = result.step_size
    summary = {
        "algorithm": "pgd", "iterations": result.iterations, "stop_reason": result.stop_reason,
        "train_seconds": elapsed, **result.final_metrics,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as temp:
        run = Path(temp) / "run"
        run.mkdir()
        write_json(run / "config.json", config)
        write_json(run / "summary.json", summary)
        with (run / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(result.history)
        np.save(run / "alpha.npy", result.alpha, allow_pickle=False)
        run.rename(args.out)
    print(f"{result.stop_reason}: {result.iterations} updates, "
          f"D={summary['dual_objective']:.10g}, KKT={summary['kkt_violation']:.8g}, "
          f"{elapsed:.3f} s. Results: {args.out}")
    return 0 if result.stop_reason == "converged" else 1


if __name__ == "__main__":
    raise SystemExit(main())
