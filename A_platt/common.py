"""Shared C-SVC dual data loading, kernels, and solution diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence


def load_libsvm(path: str | Path) -> tuple[list[dict[int, float]], list[int]]:
    """Read LIBSVM sparse rows without changing their order or feature values."""
    rows: list[dict[int, float]] = []
    labels: list[int] = []
    for line_number, line in enumerate(Path(path).open(encoding="utf-8"), 1):
        fields = line.split()
        if not fields:
            continue
        label = float(fields[0])
        if label not in (-1.0, 1.0):
            raise ValueError(f"line {line_number}: label must be -1 or +1")
        row: dict[int, float] = {}
        for field in fields[1:]:
            index_text, value_text = field.split(":", 1)
            index = int(index_text)
            if index <= 0 or index in row:
                raise ValueError(f"line {line_number}: invalid or repeated feature index")
            row[index] = float(value_text)
        rows.append(row)
        labels.append(int(label))
    if not rows or len(set(labels)) != 2:
        raise ValueError("expected a nonempty binary data set with both labels")
    return rows, labels


def rbf_kernel_matrix(rows: Sequence[dict[int, float]], gamma: float) -> list[list[float]]:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    n = len(rows)
    norms = [sum(value * value for value in row.values()) for row in rows]
    kernel = [[0.0] * n for _ in range(n)]
    for i in range(n):
        kernel[i][i] = 1.0
        for j in range(i):
            left, right = rows[i], rows[j]
            if len(left) > len(right):
                left, right = right, left
            dot = sum(value * right.get(index, 0.0) for index, value in left.items())
            distance2 = max(0.0, norms[i] + norms[j] - 2.0 * dot)
            kernel[i][j] = kernel[j][i] = math.exp(-gamma * distance2)
    return kernel


def dual_q(kernel: Sequence[Sequence[float]], y: Sequence[int]) -> list[list[float]]:
    n = len(y)
    if len(kernel) != n or any(len(row) != n for row in kernel):
        raise ValueError("kernel shape does not match labels")
    return [[y[i] * y[j] * kernel[i][j] for j in range(n)] for i in range(n)]


def evaluate_dual(
    alpha: Sequence[float], y: Sequence[int], q: Sequence[Sequence[float]], c: float
) -> dict[str, float]:
    """Evaluate the maximization objective and LIBSVM style full-set KKT gap."""
    n = len(alpha)
    if c <= 0 or len(y) != n or len(q) != n or any(len(row) != n for row in q):
        raise ValueError("invalid C or array shapes")
    if any(label not in (-1, 1) for label in y):
        raise ValueError("labels must be -1 or +1")
    if not all(math.isfinite(value) for value in alpha):
        raise ValueError("alpha contains a nonfinite value")
    qa = [math.fsum(q[i][j] * alpha[j] for j in range(n)) for i in range(n)]
    gradient = [value - 1.0 for value in qa]
    up: list[float] = []
    low: list[float] = []
    boundary_tolerance = 1e-10 * max(1.0, c)
    for ai, yi, gi in zip(alpha, y, gradient):
        if (yi == 1 and ai < c - boundary_tolerance) or (yi == -1 and ai > boundary_tolerance):
            up.append(-yi * gi)
        if (yi == 1 and ai > boundary_tolerance) or (yi == -1 and ai < c - boundary_tolerance):
            low.append(yi * gi)
    gmax = max(up, default=math.nan)
    gmax2 = max(low, default=math.nan)
    return {
        "dual_objective": math.fsum(alpha) - 0.5 * math.fsum(alpha[i] * qa[i] for i in range(n)),
        "equality_residual": abs(math.fsum(y[i] * alpha[i] for i in range(n))),
        "box_violation": max((max(0.0, -ai, ai - c) for ai in alpha), default=0.0),
        "kkt_violation": max(0.0, gmax + gmax2) if up and low else math.nan,
        "gmax": gmax,
        "gmax2": gmax2,
    }
