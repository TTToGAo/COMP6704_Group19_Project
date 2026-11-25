"""Float64 binary C-SVC dual solver using Fan et al.'s second-order SMO."""

from dataclasses import dataclass

import numpy as np

from reference.csmo import evaluate_dual, kkt_values, rbf_kernel, select_working_set


@dataclass
class SolverResult:
    alpha: np.ndarray
    iterations: int
    stop_reason: str
    history: list[dict]
    summary: dict


def solve(X, y, *, C=1.0, gamma=1 / 22, epsilon=0.001,
          alpha0=None, max_iter=None, log_every=None):
    """Solve the shared dual and log the full-set KKT violation every log_every updates."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if (X.ndim != 2 or y.shape != (len(X),) or not np.isfinite(X).all()
            or set(np.unique(y)) != {-1.0, 1.0}):
        raise ValueError("Expected finite X and both labels -1/+1.")
    if not all(np.isfinite(value) and value > 0 for value in (C, gamma, epsilon)):
        raise ValueError("C, gamma and epsilon must be finite and positive.")
    max_iter = 1_000_000 if max_iter is None else max_iter
    log_every = 10 if log_every is None else log_every
    if (not isinstance(max_iter, int) or max_iter < 0
            or not isinstance(log_every, int) or log_every < 1):
        raise ValueError("Expected a nonnegative max_iter and positive log_every.")
    if alpha0 is not None and (np.shape(alpha0) != y.shape or np.any(np.asarray(alpha0) != 0)):
        raise ValueError("The shared experiment requires alpha0=0.")

    K = rbf_kernel(X, gamma)
    Q = y[:, None] * y[None, :] * K
    alpha = np.zeros(len(y), dtype=np.float64)
    G = np.full(len(y), -1.0, dtype=np.float64)
    feasibility_tol = 1e-9 * max(1.0, C)
    history = []
    iterations = gradient_refreshes = 0
    stop_reason, failure_detail = "max_iter", ""

    def record():
        metrics = evaluate_dual(alpha, y, Q, C)
        history.append({
            "iteration": iterations,
            **{key: metrics[key] for key in (
                "dual_objective", "kkt_violation", "equality_residual", "box_violation"
            )},
        })

    record()
    while True:
        internal_kkt = kkt_values(alpha, y, G, C)[0]
        if not np.isfinite(internal_kkt):
            stop_reason, failure_detail = "numerical_failure", "Invalid full-set KKT value."
            break
        if internal_kkt <= epsilon:
            exact_G = Q @ alpha - 1
            exact_kkt = kkt_values(alpha, y, exact_G, C)[0]
            endpoint = evaluate_dual(alpha, y, Q, C)
            if (not np.isfinite(exact_kkt)
                    or not np.isfinite(endpoint["kkt_violation"])
                    or endpoint["equality_residual"] > feasibility_tol
                    or endpoint["box_violation"] > feasibility_tol):
                stop_reason, failure_detail = "numerical_failure", "Invalid final KKT or feasibility."
                break
            if exact_kkt <= epsilon and endpoint["kkt_violation"] <= epsilon:
                stop_reason = "converged"
                break
            G = exact_G
            gradient_refreshes += 1
            if gradient_refreshes > 10:
                stop_reason, failure_detail = "numerical_failure", "Repeated stopping discrepancy."
                break
        if iterations >= max_iter:
            break
        pair = select_working_set(alpha, y, G, K, C)
        if pair is None:
            stop_reason, failure_detail = "numerical_failure", "No improving pair."
            break

        i, j = pair
        # d = y_i e_i - y_j e_j preserves the equality constraint.
        direction_i, direction_j = y[i], -y[j]
        gain = -direction_i * G[i] - direction_j * G[j]
        curvature = K[i, i] + K[j, j] - 2 * K[i, j]
        cap_i = C - alpha[i] if direction_i > 0 else alpha[i]
        cap_j = C - alpha[j] if direction_j > 0 else alpha[j]
        step = min(gain / (curvature if curvature > 0 else 1e-12), cap_i, cap_j)
        if not np.isfinite(step) or step <= 0:
            stop_reason, failure_detail = "numerical_failure", "No positive feasible pair step."
            break

        # The common feasible step enforces the box; this clamps only roundoff.
        new_i = min(C, max(0.0, alpha[i] + step * direction_i))
        new_j = min(C, max(0.0, alpha[j] + step * direction_j))
        delta_i, delta_j = new_i - alpha[i], new_j - alpha[j]
        if delta_i == 0 and delta_j == 0:
            stop_reason, failure_detail = "numerical_failure", "Pair step made no floating-point progress."
            break
        alpha[i], alpha[j] = new_i, new_j
        G += delta_i * Q[:, i] + delta_j * Q[:, j]
        iterations += 1
        if iterations % log_every == 0:
            record()

    if history[-1]["iteration"] != iterations:
        record()
    exact_G = Q @ alpha - 1
    summary = evaluate_dual(alpha, y, Q, C)
    summary.update(
        algorithm="second_order_smo",
        iterations=iterations,
        stop_reason=stop_reason,
        failure_detail=failure_detail,
        internal_kkt_violation=float(kkt_values(alpha, y, G, C)[0]),
        recomputed_exact_boundary_kkt=float(kkt_values(alpha, y, exact_G, C)[0]),
        gradient_max_error=float(np.max(np.abs(G - exact_G))),
        n_gradient_refresh=gradient_refreshes,
    )
    return SolverResult(alpha, iterations, stop_reason, history, summary)
