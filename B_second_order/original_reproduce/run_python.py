"""Run the self-written baseline on the fixed small and convergence datasets."""

import argparse
import os
import time

THREADS = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
os.environ.update(THREADS)

import numpy as np

from experiment_io import DATASETS, HISTORY_FIELDS, ROOT, load_fixed, machine, sha256, write_csv, write_json
from second_order_smo import solve


def run(dataset):
    shared, X, y = load_fixed(dataset)
    start = time.perf_counter()
    result = solve(
        X, y, C=shared["C"], gamma=shared["gamma"], epsilon=shared["epsilon"],
        max_iter=shared["max_iter"], log_every=shared["log_every"],
    )
    train_seconds = time.perf_counter() - start
    output = ROOT / "results" / "python" / "smo2" / dataset
    output.mkdir(parents=True, exist_ok=True)
    shared_fields = (
        "dataset_id", "dataset_url", "dataset_hash", "dataset_hash_scope", "subset_hash",
        "sample_indices_file", "indices_hash", "seed", "rng", "n_train", "n_features",
        "negative_count", "positive_count", "kernel", "C", "gamma", "epsilon", "alpha0",
        "max_iter", "log_every", "dtype", "shrinking", "feature_scaling",
    )
    config = {key: shared[key] for key in shared_fields if key in shared}
    config.update(
        algorithm="second_order_smo",
        input_file=f"inputs/{dataset}/subset.libsvm",
        implementation_sha256=sha256(ROOT / "second_order_smo.py"),
        working_set_reference_sha256=sha256(ROOT / "reference" / "csmo.py"),
        machine=machine(),
        threads=THREADS,
        timing_scope="Python diagnostic with kernel creation and history logging",
    )
    summary = {
        key: None if isinstance(value, float) and not np.isfinite(value) else value
        for key, value in result.summary.items()
    }
    summary["train_seconds"] = train_seconds
    np.save(output / "alpha.npy", result.alpha, allow_pickle=False)
    write_csv(output / "history.csv", result.history, HISTORY_FIELDS)
    write_json(output / "summary.json", summary)
    write_json(output / "config.json", config)
    if result.stop_reason != "converged":
        raise SystemExit(f"{dataset}: {result.stop_reason}: {result.summary['failure_detail']}")
    print(f"{dataset}: {result.iterations} updates, KKT={summary['kkt_violation']:.10g}, "
          f"D={summary['dual_objective']:.10f}, |y^T alpha|={summary['equality_residual']:.3g}, "
          f"box={summary['box_violation']:.3g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", metavar="DATASET", help=" / ".join(DATASETS))
    arguments = parser.parse_args()
    for name in arguments.datasets or DATASETS:
        if name not in DATASETS:
            parser.error(f"Unknown dataset {name!r}.")
        run(name)
