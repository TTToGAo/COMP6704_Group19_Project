"""Validate the author's two modes and run five paired full-data timings."""

import argparse
import bz2
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import shutil
from statistics import median
import subprocess

for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import numpy as np

from author_source import build
from experiment_io import DATASETS, ROOT, download, load_fixed, machine, read_json, sha256, write_csv, write_json
from verify_results import recompute

BASE_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
TEST_SHA256 = "e7b0bbd518641980dc322746c6bc59d019d3153934e6258e48f9d8cfae2c4bb8"
MODES = {"smo2": 0, "csmo": 1}
OPTIONS = ["-s", "0", "-t", "2", "-c", "1", "-g", "0.045454545454545456",
           "-e", "0.001", "-m", "100", "-h", "0", "-b", "0"]
N_ROUNDS = 5
TIMING_FIELDS = (
    "round", "order", "algorithm", "mode", "iterations", "stop_reason", "dual_objective",
    "kkt_violation", "gmax", "gmax2", "equality_residual", "box_violation",
    "train_seconds", "seconds_per_update", "test_accuracy", "test_correct", "n_test",
    "n_clipped", "n_fallback",
)


def run_logged(args, output, *, env=None):
    with output.open("wb") as stream:
        process = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, env=env, check=False)
    log = output.read_text(encoding="utf-8", errors="replace")
    if process.returncode:
        raise RuntimeError(f"{Path(args[0]).name} returned {process.returncode}; see {output}.")
    return log


def alpha_from_tsv(file, n_train):
    alpha = np.zeros(n_train, dtype=np.float64)
    seen = set()
    with file.open(newline="", encoding="utf-8") as stream:
        for row, coefficient in csv.reader(stream, delimiter="\t"):
            index = int(row) - 1
            value = abs(float(coefficient))
            if index in seen or not 0 <= index < n_train or not math.isfinite(value):
                raise ValueError(f"Invalid support-vector index or coefficient in {file}.")
            seen.add(index)
            alpha[index] = value
    if not seen:
        raise ValueError(f"No support vectors in {file}.")
    return alpha


def optional_counter(log, name):
    matches = re.findall(rf"#{re.escape(name)}\s*[:=]\s*(\d+)", log)
    return int(matches[-1]) if matches else None


def parse_training(log):
    payload = re.findall(r"(?m)^GROUP19_END ([^\r\n]+)$", log)
    iterations = optional_counter(log, "iter")
    if len(payload) != 1 or iterations is None:
        raise ValueError("Expected one native training endpoint and one iteration count.")
    seconds, gmax, gmax2, objective, source = payload[0].split()
    seconds, gmax, gmax2, objective = map(float, (seconds, gmax, gmax2, objective))
    source = int(source)
    kkt = max(0.0, gmax + gmax2) if math.isfinite(gmax + gmax2) else None
    if "reaching max number of iterations" in log.lower():
        stop = "max_iter"
    elif source == 1 and kkt is not None and kkt <= 0.001:
        stop = "converged"
    else:
        stop = "numerical_failure"
    if not math.isfinite(seconds) or seconds <= 0 or iterations <= 0:
        raise ValueError("Invalid native time or update count.")
    return {
        "iterations": iterations,
        "stop_reason": stop,
        "dual_objective": -objective if math.isfinite(objective) else None,
        "kkt_violation": kkt,
        "gmax": gmax if math.isfinite(gmax) else None,
        "gmax2": gmax2 if math.isfinite(gmax2) else None,
        "native_kkt_source": {0: "missing", 1: "solver_full_stop_check", 2: "final_scan"}.get(source),
        "train_seconds": seconds,
        "seconds_per_update": seconds / iterations,
        "n_clipped": optional_counter(log, "iter_clip"),
        "n_fallback": optional_counter(log, "iter_smo"),
    }


def capture_run(train, predict, metadata, name, data, y, output, *, round_number=None,
                order=None, test=None, n_test=None, fixed=None):
    output.mkdir(parents=True)
    alpha_tsv = output / "alpha.tsv"
    model = output / "model.txt"
    env = dict(os.environ, GROUP19_ALPHA_PATH=str(alpha_tsv))
    config = {
        "algorithm": name,
        "implementation": "author_libsvm_cd",
        "mode": MODES[name],
        "round": round_number,
        "order": order,
        "n_train": len(y),
        "training_file": data.name,
        "training_sha256": sha256(data),
        "options": OPTIONS,
        "kernel": "rbf",
        "C": 1,
        "gamma": 1 / 22,
        "epsilon": 0.001,
        "alpha0": 0,
        "shrinking": False,
        "cache_mb": 100,
        "feature_scaling": "none",
        "n_test": n_test,
        "test_sha256": sha256(test) if test else None,
        "author_commit": metadata["commit"],
        "train_binary_sha256": metadata["train_binary_sha256"],
        "timing_scope": "std::chrono::steady_clock directly around svm_train()",
    }
    write_json(output / "config.json", config)
    log = run_logged([str(train), *OPTIONS, "-a", str(MODES[name]), str(data), str(model)],
                     output / "train.log", env=env)
    summary = {"algorithm": name, "mode": MODES[name], **parse_training(log)}
    alpha = alpha_from_tsv(alpha_tsv, len(y))
    np.save(output / "alpha.npy", alpha, allow_pickle=False)
    summary["equality_residual"] = float(abs(y @ alpha))
    summary["box_violation"] = float(max(0, -alpha.min(), alpha.max() - 1))
    if fixed is not None:
        X, settings = fixed
        summary["common_float64_metrics"] = recompute(X, y, alpha, settings["C"], settings["gamma"])
        summary["float32_kernel_metrics"] = recompute(X, y, alpha, settings["C"], settings["gamma"], qfloat=True)
    if test is not None:
        log = run_logged([str(predict), str(test), str(model), str(output / "predictions.txt")],
                         output / "predict.log")
        accuracy = re.search(r"Accuracy\s*=\s*[^\n(]+%\s*\((\d+)\s*/\s*(\d+)\)", log)
        if accuracy is None or int(accuracy[2]) != n_test:
            raise ValueError(f"Missing accuracy or wrong test-set denominator in {output}.")
        summary.update(test_correct=int(accuracy[1]), n_test=n_test, test_accuracy=int(accuracy[1]) / n_test)
    write_json(output / "summary.json", summary)
    if (summary["stop_reason"] != "converged" or summary["dual_objective"] is None
            or summary["box_violation"] > 1e-9 or summary["equality_residual"] > 1e-8):
        raise RuntimeError(f"Run failed the native KKT or feasibility check; see {output / 'summary.json'}.")
    if fixed is not None and summary["common_float64_metrics"]["kkt_violation"] > settings["epsilon"]:
        raise RuntimeError(f"Native convergence differs from the shared float64 criterion; see {output / 'summary.json'}.")
    print(f"{output.name}: {name}, {summary['iterations']} updates, KKT={summary['kkt_violation']:.8g}, "
          f"T={summary['train_seconds']:.6f}s", flush=True)
    return summary


def ensure_input(file):
    if file.is_file():
        return
    archive = file.with_name(file.name + ".bz2")
    download(f"{BASE_URL}/{archive.name}", archive)
    partial = file.with_name(file.name + ".partial")
    with bz2.open(archive, "rb") as stream, partial.open("wb") as output:
        shutil.copyfileobj(stream, output)
    partial.replace(file)


def load_full(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    train, test = data_dir / "ijcnn1", data_dir / "ijcnn1.t"
    ensure_input(train)
    ensure_input(test)
    subset = "ijcnn1_seed19"
    settings = read_json(ROOT / "reference" / subset / "config.json")
    if sha256(train) != settings["dataset_hash"]:
        raise ValueError("The full training bytes differ from the dataset used by C.")
    if sha256(test) != TEST_SHA256:
        raise ValueError("The full test bytes differ from the pinned ijcnn1.t split.")
    rows = train.read_bytes().splitlines()
    with test.open("rb") as stream:
        n_test = sum(1 for row in stream if row.strip())
    if len(rows) != 49_990 or n_test != 91_701:
        raise ValueError(f"Unexpected official split sizes: {len(rows)} and {n_test}.")
    indices = np.load(ROOT / "inputs" / subset / "indices.npy", allow_pickle=False)
    actual = (ROOT / "inputs" / subset / "subset.libsvm").read_bytes().splitlines()
    if [rows[i] for i in indices] != actual:
        raise ValueError("The saved indices do not recover C's subset rows in the official train file.")
    y = np.array([float(row.split(None, 1)[0]) for row in rows], dtype=np.float64)
    return train, y, test, n_test


def tables(rows, destination):
    write_csv(destination / "runs.csv", rows, TIMING_FIELDS)
    ratios = []
    for round_number in range(1, N_ROUNDS + 1):
        pair = [row for row in rows if row["round"] == round_number]
        speeds = {row["algorithm"]: row["train_seconds"] for row in pair}
        ratios.append({"round": round_number, "run_order": " -> ".join(row["algorithm"] for row in pair),
                       "smo2_seconds": speeds["smo2"], "csmo_seconds": speeds["csmo"],
                       "speedup_smo2_over_csmo": speeds["smo2"] / speeds["csmo"]})
    write_csv(destination / "paired_speedups.csv", ratios, tuple(ratios[0]))
    medians = []
    for mode in MODES:
        runs = [row for row in rows if row["algorithm"] == mode]
        columns = ("iterations", "dual_objective", "kkt_violation", "train_seconds",
                   "seconds_per_update", "test_accuracy")
        medians.append({"algorithm": mode, "n_runs": len(runs),
                        **{name + "_median": median(row[name] for row in runs) for name in columns},
                        "train_seconds_min": min(row["train_seconds"] for row in runs),
                        "train_seconds_max": max(row["train_seconds"] for row in runs)})
    write_csv(destination / "mode_medians.csv", medians, tuple(medians[0]))
    write_json(destination / "comparison.json", {
        "rounds": N_ROUNDS,
        "median_paired_speedup": median(row["speedup_smo2_over_csmo"] for row in ratios),
        "modes": medians,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "work" / "libsvm_cd")
    parser.add_argument("--cxx", default=os.environ.get("CXX"), help="Path to an existing C++ compiler.")
    parser.add_argument("--revision", help="Optional exact author commit to check out or verify.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "work" / "data")
    parser.add_argument("--validate-only", action="store_true", help="Only run the small and fixed-subset gates.")
    options = parser.parse_args()
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results" / "author" / session_id
    train, predict, metadata = build(options.source.resolve(), ROOT / "work" / "builds" / session_id,
                                     options.cxx, options.revision)
    destination.mkdir(parents=True)
    write_json(destination / "build.json", {**metadata, "machine": machine(), "session_id": session_id,
                                            "threads": 1})
    endpoints = []
    for dataset in DATASETS:
        settings, X, y = load_fixed(dataset)
        data = ROOT / "inputs" / dataset / "subset.libsvm"
        for mode in MODES:
            summary = capture_run(train, predict, metadata, mode, data, y,
                                  destination / "validation" / dataset / mode, fixed=(X, settings))
            shared = summary["common_float64_metrics"]
            endpoints.append({
                "dataset": dataset, "algorithm": mode, "iterations": summary["iterations"],
                "stop_reason": summary["stop_reason"], "native_kkt": summary["kkt_violation"],
                "native_dual_objective": summary["dual_objective"], **shared,
            })
    write_csv(destination / "validation_endpoints.csv", endpoints, tuple(endpoints[0]))
    if options.validate_only:
        return

    train_file, labels, test_file, n_test = load_full(options.data_dir.resolve())
    output = destination / "full"
    rows = []
    for round_number in range(1, N_ROUNDS + 1):
        modes = tuple(MODES) if round_number % 2 else tuple(reversed(MODES))
        for order, mode in enumerate(modes, 1):
            summary = capture_run(train, predict, metadata, mode, train_file, labels,
                                  output / f"round_{round_number:02d}" / mode, round_number=round_number,
                                  order=order, test=test_file, n_test=n_test)
            flat = {"round": round_number, "order": order, **summary}
            rows.append({field: flat[field] for field in TIMING_FIELDS})
    tables(rows, output)
    print(f"Saved five paired runs and aggregation in {output}.", flush=True)


if __name__ == "__main__":
    main()
