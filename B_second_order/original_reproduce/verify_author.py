"""Recompute full-data endpoints and check the original five-round timing tables."""

import argparse
from math import isclose
from pathlib import Path
from statistics import median

from verify_results import recompute
from experiment_io import ROOT, load_libsvm, read_csv, read_json, sha256, write_csv
from run_author import N_ROUNDS, OPTIONS, TEST_SHA256

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Author result directory containing build.json and full/.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "work" / "data")
    args = parser.parse_args()

    data = args.data_dir / "ijcnn1"
    expected_train_sha = read_json(ROOT / "reference" / "ijcnn1_seed19" / "config.json")["dataset_hash"]
    if sha256(data) != expected_train_sha or sha256(args.data_dir / "ijcnn1.t") != TEST_SHA256:
        raise ValueError("The supplied split differs from the recorded ijcnn1 files.")
    X, y = load_libsvm(data, n_features=22)
    build = read_json(args.run / "build.json")
    full = args.run / "full"
    runs = read_csv(full / "runs.csv")
    pairs = read_csv(full / "paired_speedups.csv")
    summary = read_json(full / "comparison.json")
    runs_by_mode = {mode: [] for mode in ("smo2", "csmo")}
    results = []
    metrics_by_sha = {}
    ratios = []
    assert len(runs) == 2 * N_ROUNDS and len(pairs) == N_ROUNDS

    for round_number, pair in enumerate(pairs, start=1):
        order = ("smo2", "csmo") if round_number % 2 else ("csmo", "smo2")
        times = {}
        for position, mode in enumerate(order, start=1):
            run = runs[(round_number - 1) * 2 + position - 1]
            assert (int(run["round"]), int(run["order"]), run["algorithm"]) == (round_number, position, mode)
            folder = full / f"round_{round_number:02d}" / mode
            config = read_json(folder / "config.json")
            assert config["train_binary_sha256"] == build["train_binary_sha256"]
            assert config["training_sha256"] == expected_train_sha and config["test_sha256"] == TEST_SHA256
            assert config["options"] == OPTIONS and run["stop_reason"] == "converged"
            assert 0 <= float(run["kkt_violation"]) <= config["epsilon"]
            assert int(run["n_test"]) == 91_701
            time = float(run["train_seconds"])
            assert isclose(time / int(run["iterations"]), float(run["seconds_per_update"]))
            times[mode] = time
            runs_by_mode[mode].append(run)

            path = folder / "alpha.npy"
            alpha_sha = sha256(path)
            if alpha_sha not in metrics_by_sha:
                alpha = np.load(path, allow_pickle=False)
                assert alpha.shape == y.shape and alpha.dtype == np.float64 and np.isfinite(alpha).all()
                metrics_by_sha[alpha_sha] = recompute(X, y, alpha, config["C"], config["gamma"])
            metrics = metrics_by_sha[alpha_sha]
            assert metrics["box_violation"] <= 1e-9 and metrics["equality_residual"] <= 1e-8
            results.append({
                "round": round_number, "order": position, "algorithm": mode, "alpha_sha256": alpha_sha,
                "native_kkt": float(run["kkt_violation"]), "native_dual_objective": float(run["dual_objective"]),
                **metrics, "meets_float64_kkt_target": metrics["kkt_violation"] <= config["epsilon"],
            })
        ratio = times["smo2"] / times["csmo"]
        assert pair["run_order"] == " -> ".join(order)
        assert isclose(ratio, float(pair["speedup_smo2_over_csmo"]))
        ratios.append(ratio)

    assert isclose(median(ratios), summary["median_paired_speedup"])
    median_columns = ("iterations", "dual_objective", "kkt_violation", "train_seconds",
                      "seconds_per_update", "test_accuracy")
    for mode in read_csv(full / "mode_medians.csv"):
        rows = runs_by_mode[mode["algorithm"]]
        assert int(mode["n_runs"]) == len(rows)
        for field in median_columns:
            assert isclose(float(mode[field + "_median"]), median(float(row[field]) for row in rows))

    write_csv(full / "verified_full_endpoints.csv", results, tuple(results[0]))
    passed = sum(row["meets_float64_kkt_target"] for row in results)
    print(f"Checked {len(results)} runs and all timing aggregates. Float64 KKT passed in {passed}/{len(results)}.")


if __name__ == "__main__":
    main()
