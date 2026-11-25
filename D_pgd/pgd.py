"""Fixed-step projected gradient ascent on the concave SVM dual."""

from dataclasses import dataclass
from numbers import Integral
from typing import Optional

import numpy as np

from common.kernel import rbf_dual_matrix
from common.metrics import evaluate_dual


DEFAULT_MAX_ITER = 100_000
DEFAULT_LOG_EVERY = 100
HISTORY_FIELDS = ("iteration", "dual_objective", "kkt_violation",
                  "equality_residual", "box_violation")


@dataclass
class SolverResult:
    alpha: np.ndarray
    iterations: int
    stop_reason: str
    history: list
    final_metrics: dict
    step_size: Optional[float]


def project_feasible(v, y, C):
    """Project onto [0,C]^n intersected with y.T @ alpha = 0 for ±1 labels."""
    v = np.asarray(v, dtype=np.float64)
    y = np.asarray(y)
    # At the two outermost clipping breakpoints the residual brackets zero.
    starts = y * v
    ends = y * (v - C)
    lower = float(min(starts.min(), ends.min()))
    upper = float(max(starts.max(), ends.max()))
    tolerance = 1e-12 * max(1.0, C)
    for _ in range(128):
        multiplier = 0.5 * lower + 0.5 * upper
        alpha = np.clip(v - multiplier * y, 0.0, C)
        residual = float(y @ alpha)
        if abs(residual) <= tolerance:
            return alpha
        if multiplier == lower or multiplier == upper:
            break
        if residual > 0:
            lower = multiplier
        else:
            upper = multiplier
    raise FloatingPointError("Projection did not reach its equality tolerance")


def solve(X, y, *, C, gamma, epsilon, alpha0=None, max_iter=None, log_every=None):
    max_iter = DEFAULT_MAX_ITER if max_iter is None else max_iter
    log_every = DEFAULT_LOG_EVERY if log_every is None else log_every
    if not isinstance(max_iter, Integral) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    if not isinstance(log_every, Integral) or log_every < 1:
        raise ValueError("log_every must be a positive integer")
    if not np.isfinite(epsilon) or epsilon <= 0 or not np.isfinite(C) or C <= 0:
        raise ValueError("epsilon and C must be finite and positive")

    y = np.asarray(y)
    Q = rbf_dual_matrix(X, y, gamma)
    alpha = (np.zeros(len(y), dtype=np.float64) if alpha0 is None
             else np.array(alpha0, dtype=np.float64, copy=True))
    if alpha.shape != y.shape or not np.isfinite(alpha).all():
        raise ValueError("alpha0 must be a finite vector with one entry per sample")
    feasibility_tol = 1e-10 * max(1.0, C)
    if np.any(alpha < 0) or np.any(alpha > C) or abs(y @ alpha) > feasibility_tol:
        raise ValueError("alpha0 must satisfy the box and equality constraints")

    iterations = 0
    history = []
    step_size = None
    while True:
        metrics = evaluate_dual(alpha, y, Q, C)
        row = {"iteration": iterations, **{k: metrics[k] for k in HISTORY_FIELDS[1:]}}
        if iterations % log_every == 0:
            history.append(row)
        if not all(np.isfinite(value) for value in metrics.values()):
            stop_reason = "numerical_failure"
            break
        if (metrics["kkt_violation"] <= epsilon
                and metrics["equality_residual"] <= feasibility_tol
                and metrics["box_violation"] <= feasibility_tol):
            stop_reason = "converged"
            break
        if iterations == max_iter:
            stop_reason = "max_iter"
            break
        try:
            if step_size is None:
                largest_eigenvalue = np.linalg.eigvalsh(Q)[-1]
                if not np.isfinite(largest_eigenvalue) or largest_eigenvalue <= 0:
                    raise FloatingPointError("Invalid largest eigenvalue of Q")
                step_size = float(1.0 / largest_eigenvalue)
            v = alpha + step_size * (1.0 - Q @ alpha)
            alpha = project_feasible(v, y, C)
        except (FloatingPointError, np.linalg.LinAlgError):
            stop_reason = "numerical_failure"
            break
        iterations += 1
    if history[-1]["iteration"] != iterations:
        history.append(row)
    return SolverResult(alpha, iterations, stop_reason, history, metrics, step_size)
