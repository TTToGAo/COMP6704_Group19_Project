"""Independently check fixed-input identity, feasibility, and convergence logs."""

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from experiment_io import DATASETS, HISTORY_FIELDS, ROOT, load_fixed, read_csv, read_json, sha256, write_csv


def recompute(X, y, alpha, C, gamma, *, qfloat=False):
    support = alpha != 0
    support_rows = X[support]
    signed_alpha = y[support] * alpha[support]
    squared_norms = np.einsum("ij,ij->i", X, X)
    support_norms = squared_norms[support]
    qa = np.empty(len(y), dtype=np.float64)
    for start in range(0, len(X), 256):
        section = slice(start, start + 256)
        distances = X[section] @ support_rows.T
        distances *= -2
        distances += squared_norms[section, None]
        distances += support_norms
        np.maximum(distances, 0, out=distances)
        distances *= -gamma
        kernel = np.exp(distances, out=distances)
        if qfloat:
            kernel = kernel.astype(np.float32).astype(np.float64)
        qa[section] = y[section] * (kernel @ signed_alpha)
    G = qa - 1
    tau = 1e-10 * max(1.0, C)
    up = ((y == 1) & (alpha < C - tau)) | ((y == -1) & (alpha > tau))
    low = ((y == 1) & (alpha > tau)) | ((y == -1) & (alpha < C - tau))
    gmax = float(np.max((-y * G)[up]))
    gmax2 = float(np.max((y * G)[low]))
    return {
        "dual_objective": float(alpha.sum() - 0.5 * alpha @ qa),
        "kkt_violation": max(0.0, gmax + gmax2),
        "equality_residual": float(abs(y @ alpha)),
        "box_violation": float(max(0.0, -alpha.min(), alpha.max() - C)),
        "gmax": gmax,
        "gmax2": gmax2,
    }


def check_run(folder, X, y, config):
    summary = read_json(folder / "summary.json")
    alpha = np.load(folder / "alpha.npy", allow_pickle=False)
    assert alpha.shape == y.shape and alpha.dtype == np.float64 and np.isfinite(alpha).all()
    assert summary["stop_reason"] == "converged"
    metrics = recompute(X, y, alpha, config["C"], config["gamma"])
    for field in metrics:
        assert abs(metrics[field] - summary[field]) < 1e-10, (folder, field, metrics[field], summary[field])
    assert metrics["box_violation"] <= 1e-9 and metrics["equality_residual"] <= 1e-9
    assert metrics["kkt_violation"] <= config["epsilon"]

    history = read_csv(folder / "history.csv")
    assert tuple(history[0]) == HISTORY_FIELDS
    last, interval = summary["iterations"], config["log_every"]
    expected_grid = list(range(0, last, interval)) + [last]
    assert [int(row["iteration"]) for row in history] == expected_grid
    assert all(abs(float(history[-1][field]) - metrics[field]) < 1e-10 for field in HISTORY_FIELDS[1:])
    assert float(history[0]["dual_objective"]) == 0.0 and float(history[0]["kkt_violation"]) == 2.0
    assert np.all(np.diff([float(row["dual_objective"]) for row in history]) >= -1e-9)
    return {
        "iterations": last,
        "stop_reason": summary["stop_reason"],
        **metrics,
        "validation": "independent float64 recomputation",
    }


def main():
    rows = []
    for dataset in DATASETS:
        config, X, y = load_fixed(dataset)
        assert sha256(ROOT / "reference" / "csmo.py") == config["source_sha256"]["csmo.py"]
        paths = {
            "Second-order SMO (B)": ROOT / "results" / "python" / "smo2" / dataset,
            "CSMO (C)": ROOT / "reference" / dataset,
        }
        for algorithm, folder in paths.items():
            metrics = check_run(folder, X, y, config)
            rows.append({"dataset": dataset, "algorithm": algorithm, **metrics})
    write_csv(ROOT / "results" / "verified_python_endpoints.csv", rows, tuple(rows[0]))
    print(f"Verified {len(rows)} B/C endpoints and histories.")


if __name__ == "__main__":
    main()
