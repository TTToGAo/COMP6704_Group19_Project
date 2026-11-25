"""Original Platt-style SMO for the shared binary C-SVC dual."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from common import dual_q, evaluate_dual, load_libsvm, rbf_kernel_matrix


@dataclass
class SolverResult:
    alpha: list[float]
    iterations: int
    stop_reason: str
    history: list[dict[str, float]]
    bias: float


def solve(
    kernel: list[list[float]], y: list[int], *, c: float, epsilon: float = 0.001,
    alpha0: list[float] | None = None, max_iter: int = 200000,
    log_every: int = 100, seed: int = 19,
) -> SolverResult:
    """Train with Platt's first-order error heuristic and analytic pair updates.

    `iterations` counts successful pair updates; all samples retain input order.
    """
    n = len(y)
    if n < 2 or len(kernel) != n or any(len(row) != n for row in kernel):
        raise ValueError("invalid kernel shape")
    if c <= 0 or epsilon <= 0 or max_iter <= 0 or log_every <= 0:
        raise ValueError("C, epsilon, max_iter and log_every must be positive")
    if any(label not in (-1, 1) for label in y) or len(set(y)) != 2:
        raise ValueError("labels must contain both -1 and +1")
    alpha = list(alpha0) if alpha0 is not None else [0.0] * n
    if len(alpha) != n or any(a < 0 or a > c for a in alpha):
        raise ValueError("alpha0 violates box constraints")
    if abs(math.fsum(a * label for a, label in zip(alpha, y))) > 1e-9:
        raise ValueError("alpha0 violates the equality constraint")
    q = dual_q(kernel, y)
    rng = random.Random(seed)
    bias = 0.0
    error = [-float(label) for label in y]
    updates = 0
    history: list[dict[str, float]] = []

    def record() -> dict[str, float]:
        metrics = evaluate_dual(alpha, y, q, c)
        history.append({"iteration": updates, **metrics})
        return metrics

    def non_bound() -> list[int]:
        return [i for i, a in enumerate(alpha) if 1e-12 < a < c - 1e-12]

    def take_step(i: int, j: int) -> bool:
        nonlocal bias, updates
        if i == j:
            return False
        a1, a2 = alpha[i], alpha[j]
        y1, y2 = y[i], y[j]
        e1, e2 = error[i], error[j]
        s = y1 * y2
        if s == 1:
            lo, hi = max(0.0, a1 + a2 - c), min(c, a1 + a2)
        else:
            lo, hi = max(0.0, a2 - a1), min(c, c + a2 - a1)
        if hi - lo <= 1e-14:
            return False
        k11, k12, k22 = kernel[i][i], kernel[i][j], kernel[j][j]
        eta = k11 + k22 - 2.0 * k12
        if eta > 1e-12:
            new_a2 = min(hi, max(lo, a2 + y2 * (e1 - e2) / eta))
        else:
            # Platt's non-positive-curvature fallback: compare both endpoints.
            candidate_values = []
            for candidate_a2 in (lo, hi):
                candidate_a1 = a1 + s * (a2 - candidate_a2)
                delta1, delta2 = candidate_a1 - a1, candidate_a2 - a2
                gain = delta1 + delta2 - delta1 * (error[i] + y1 - bias) * y1
                gain -= delta2 * (error[j] + y2 - bias) * y2
                gain -= 0.5 * (k11 * delta1**2 + 2 * s * k12 * delta1 * delta2 + k22 * delta2**2)
                candidate_values.append(gain)
            new_a2 = (lo, hi)[candidate_values[1] > candidate_values[0]]
        if abs(new_a2 - a2) < 1e-12 * (new_a2 + a2 + 1.0):
            return False
        new_a1 = a1 + s * (a2 - new_a2)
        # One variable is clipped; calculate the other from equality to preserve feasibility.
        new_a1 = min(c, max(0.0, new_a1))
        new_a2 = a2 + s * (a1 - new_a1)
        delta1, delta2 = new_a1 - a1, new_a2 - a2
        old_bias = bias
        b1 = bias - e1 - y1 * delta1 * k11 - y2 * delta2 * k12
        b2 = bias - e2 - y1 * delta1 * k12 - y2 * delta2 * k22
        if 0 < new_a1 < c:
            bias = b1
        elif 0 < new_a2 < c:
            bias = b2
        else:
            bias = 0.5 * (b1 + b2)
        alpha[i], alpha[j] = new_a1, new_a2
        for k in range(n):
            error[k] += y1 * delta1 * kernel[k][i] + y2 * delta2 * kernel[k][j] + bias - old_bias
        updates += 1
        if updates % log_every == 0:
            record()
        return True

    def examine(i: int) -> bool:
        # Platt's outer test uses the margin error, followed by three pair-choice passes.
        yi_ei = y[i] * error[i]
        # Platt's pointwise test can stop with a global gap near 2*tolerance.
        # Use half the requested gap so the shared LIBSVM-style check meets epsilon.
        point_tolerance = epsilon / 2.0
        if not ((yi_ei < -point_tolerance and alpha[i] < c - 1e-12) or
                (yi_ei > point_tolerance and alpha[i] > 1e-12)):
            return False
        candidates = non_bound()
        if candidates:
            j = max((j for j in candidates if j != i),
                    key=lambda j: abs(error[i] - error[j]), default=None)
            if j is not None and take_step(i, j):
                return True
        rng.shuffle(candidates)
        for j in candidates:
            if take_step(i, j):
                return True
        all_indices = list(range(n))
        rng.shuffle(all_indices)
        return any(take_step(i, j) for j in all_indices)

    record()
    examine_all = True
    changed = 0
    while updates < max_iter and (examine_all or changed):
        changed = 0
        indices = range(n) if examine_all else non_bound()
        for i in indices:
            changed += int(examine(i))
            if updates >= max_iter:
                break
        examine_all = False if examine_all else (changed == 0)
    metrics = record() if history[-1]["iteration"] != updates else history[-1]
    feasible = metrics["equality_residual"] <= 1e-7 and metrics["box_violation"] <= 1e-10
    if updates >= max_iter:
        reason = "max_iter"
    elif feasible and metrics["kkt_violation"] <= epsilon:
        reason = "converged"
    else:
        reason = "stalled"
    return SolverResult(alpha, updates, reason, history, bias)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the original Platt SMO on LIBSVM data")
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1 / 22)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=200000)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    rows, y = load_libsvm(args.data)
    kernel = rbf_kernel_matrix(rows, args.gamma)
    start = time.perf_counter()
    result = solve(kernel, y, c=args.C, epsilon=args.epsilon,
                   max_iter=args.max_iter, log_every=args.log_every)
    train_seconds = time.perf_counter() - start
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {"algorithm": "platt_smo", "iterations": result.iterations,
               "stop_reason": result.stop_reason, "train_seconds": train_seconds,
               "bias": result.bias, **{k: v for k, v in result.history[-1].items() if k != "iteration"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "alpha.json").write_text(json.dumps(result.alpha) + "\n")
    config = {"dataset_id": args.data.name,
              "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "n_train": len(y), "kernel": "rbf", "C": args.C,
              "gamma": args.gamma, "epsilon": args.epsilon,
              "alpha0": "zeros", "max_iter": args.max_iter,
              "log_every": args.log_every, "seed": 19,
              "algorithm": "platt_smo"}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.history[0].keys())
        writer.writeheader()
        writer.writerows(result.history)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
