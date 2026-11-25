"""Float64 C-SVC CSMO, with full working sets and no shrinking.

Torres-Barran et al., arXiv:2003.08719, Algorithm 2.
All iteration updates are O(n), given the dense O(n^2) kernel matrix.
Exact diagnostic recomputation is separate and costs O(n^2).
"""
from dataclasses import dataclass

import numpy as np


def rbf_kernel(X, gamma):
    X = np.asarray(X, dtype=np.float64)
    norm = np.einsum('ij,ij->i', X, X)
    distances = np.maximum(norm[:, None] + norm[None, :] - 2 * X @ X.T, 0)
    K = np.exp(-gamma * distances)
    np.fill_diagonal(K, 1.0)
    return K


def eligible(alpha, y, C, tau=0.0):
    up = ((y == 1) & (alpha < C - tau)) | ((y == -1) & (alpha > tau))
    low = ((y == 1) & (alpha > tau)) | ((y == -1) & (alpha < C - tau))
    return up, low


def kkt_values(alpha, y, G, C, tau=0.0):
    up, low = eligible(alpha, y, C, tau)
    if not up.any() or not low.any():
        return float('nan'), float('nan'), float('nan')
    gmax = float(np.max((-y * G)[up]))
    gmax2 = float(np.max((y * G)[low]))
    return max(0.0, gmax + gmax2), gmax, gmax2


def evaluate_dual(alpha, y, Q, C):
    Qa = Q @ alpha
    violation, gmax, gmax2 = kkt_values(
        alpha, y, Qa - 1, C, tau=1e-10 * max(1.0, C))
    return dict(
        dual_objective=float(alpha.sum() - 0.5 * alpha @ Qa),
        equality_residual=float(abs(y @ alpha)),
        box_violation=float(max(0, np.max(-alpha), np.max(alpha - C))),
        kkt_violation=violation, gmax=gmax, gmax2=gmax2)


def select_working_set(alpha, y, G, K, C):
    """Exact box boundaries; np.argmax breaks ties by smallest input index."""
    up, low = eligible(alpha, y, C)
    if not up.any() or not low.any():
        return None
    i = int(np.argmax(np.where(up, -y * G, -np.inf)))
    b = -y[i] * G[i] + y * G
    candidates = low & (b > 0)
    candidates[i] = False
    if not candidates.any():
        return None
    eta = K[i, i] + np.diag(K) - 2 * K[i]
    eta = np.where(eta > 0, eta, 1e-12)
    j = int(np.argmax(np.where(candidates, b * b / eta, -np.inf)))
    return i, j


def feasible_step(alpha, p, C):
    """Maximum nonnegative step, checking EVERY nonzero direction entry."""
    bounds = np.full(alpha.size, np.inf)
    positive, negative = p > 0, p < 0
    bounds[positive] = (C - alpha[positive]) / p[positive]
    bounds[negative] = -alpha[negative] / p[negative]
    return max(0.0, float(np.min(bounds)))


@dataclass
class SolverResult:
    alpha: np.ndarray
    iterations: int
    stop_reason: str
    history: list
    steps: list
    summary: dict


def solve(X, y, *, C=1.0, gamma=1 / 22, epsilon=0.001,
          alpha0=None, max_iter=1_000_000, log_every=10):
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (X.ndim != 2 or y.shape != (len(X),) or not np.isfinite(X).all()
            or set(np.unique(y)) != {-1.0, 1.0}):
        raise ValueError('Expected finite X and both labels -1/+1.')
    if (not all(np.isfinite(v) and v > 0 for v in (C, gamma, epsilon))
            or not isinstance(max_iter, int) or max_iter < 0
            or not isinstance(log_every, int) or log_every < 1):
        raise ValueError('Invalid parameters.')
    if alpha0 is not None and (np.shape(alpha0) != y.shape or np.any(np.asarray(alpha0) != 0)):
        raise ValueError('The shared protocol requires alpha0=0.')
    K = rbf_kernel(X, gamma)
    Q = y[:, None] * y[None, :] * K
    alpha, G = np.zeros(len(y)), -np.ones(len(y))
    p_prev, q_prev, delta_prev = np.zeros(len(y)), np.zeros(len(y)), 1.0
    history, steps = [], []
    iterations = n_clipped = n_fallback = n_conjugate = n_repair = 0
    n_refresh = 0
    stop_reason, failure_detail = 'max_iter', ''
    feasibility_tol = 1e-9 * max(1.0, C)

    def record():
        metrics = evaluate_dual(alpha, y, Q, C)
        history.append(dict(iteration=iterations, **{k: metrics[k] for k in (
            'dual_objective', 'kkt_violation', 'equality_residual', 'box_violation')}))

    record()
    while True:
        internal_kkt = kkt_values(alpha, y, G, C)[0]
        if not np.isfinite(internal_kkt):
            stop_reason, failure_detail = 'numerical_failure', 'Invalid KKT sets/gradient.'
            break
        if internal_kkt <= epsilon:
            exact_G = Q @ alpha - 1
            exact_kkt = kkt_values(alpha, y, exact_G, C)[0]
            diagnostics = evaluate_dual(alpha, y, Q, C)
            if (np.isfinite(exact_kkt) and exact_kkt <= epsilon
                    and diagnostics['kkt_violation'] <= epsilon
                    and diagnostics['equality_residual'] <= feasibility_tol
                    and diagnostics['box_violation'] <= feasibility_tol):
                stop_reason = 'converged'
                break
            if not np.isfinite(exact_kkt) or diagnostics['equality_residual'] > feasibility_tol:
                stop_reason, failure_detail = 'numerical_failure', 'Final feasibility/KKT check failed.'
                break
            # Recompute once when incremental rounding changes the stopping decision.
            G = exact_G
            p_prev.fill(0); q_prev.fill(0); delta_prev = 1.0
            n_refresh += 1
            if n_refresh > 10:
                stop_reason, failure_detail = 'numerical_failure', 'Repeated stopping discrepancy.'
                break
            continue
        if iterations >= max_iter:
            break
        pair = select_working_set(alpha, y, G, K, C)
        if pair is None:
            stop_reason, failure_detail = 'numerical_failure', 'No improving pair.'
            break
        i, j = pair
        d = np.zeros(len(y)); d[i], d[j] = y[i], -y[j]
        Qd = y[i] * Q[:, i] - y[j] * Q[:, j]
        beta = -(y[i] * q_prev[i] - y[j] * q_prev[j]) / delta_prev
        p, q = d + beta * p_prev, Qd + beta * q_prev
        # Full dot products are still O(n); they avoid assuming G @ p_prev == 0
        # exactly in floating point. In exact arithmetic these are Eq. 26/27.
        curvature, slope = float(p @ q), float(G @ p)
        fallback_reason = ''
        clipped = False
        attempted_step = 0.0
        cap = feasible_step(alpha, p, C)
        if not np.isfinite(curvature) or not np.isfinite(slope) or slope >= 0:
            fallback_reason = 'invalid_direction'
        elif curvature <= 0 and np.any(p_prev):
            fallback_reason = 'nonpositive_curvature'
        else:
            attempted_step = -slope / (curvature if curvature > 0 else 1e-12)
            step = min(attempted_step, cap)
            clipped = step < attempted_step
            if step <= 0:
                fallback_reason = 'blocked_by_box'
        if fallback_reason:
            p, q = d, Qd
            curvature = float(d @ Qd)
            pair_step = -float(G @ d) / (curvature if curvature > 0 else 1e-12)
            step = min(pair_step, feasible_step(alpha, d, C))
        if not np.isfinite(step) or step <= 0:
            stop_reason, failure_detail = 'numerical_failure', 'No positive feasible step.'
            break
        candidate = alpha + step * p
        # Repair only actual floating-point overshoots, never project a CSMO step.
        repaired = np.clip(candidate, 0.0, C)
        correction = repaired - candidate
        if np.max(np.abs(correction)) > 32 * np.finfo(float).eps * max(1.0, C):
            stop_reason, failure_detail = 'numerical_failure', 'Step left the box.'
            break
        if np.array_equal(alpha, repaired):
            stop_reason, failure_detail = 'numerical_failure', 'Step made no floating-point progress.'
            break
        new_G = G + step * q
        changed = np.flatnonzero(correction)
        if len(changed):
            new_G += Q[:, changed] @ correction[changed]
        if (not np.isfinite(new_G).all()
                or abs(y @ repaired) > feasibility_tol):
            stop_reason, failure_detail = 'numerical_failure', 'Nonfinite update/equality violation.'
            break
        alpha, G = repaired, new_G
        iterations += 1
        n_clipped += int(clipped)
        n_fallback += int(bool(fallback_reason))
        conjugate = bool(beta != 0 and not fallback_reason)
        n_conjugate += int(conjugate)
        n_repair += int(bool(len(changed)))
        reset = clipped or bool(fallback_reason) or bool(len(changed)) or curvature <= 0
        steps.append(dict(
            iteration=iterations, i=i, j=j, beta=float(beta),
            attempted_step=float(attempted_step), feasible_cap=float(cap), step=float(step),
            clipped=int(clipped), fallback=int(bool(fallback_reason)),
            fallback_reason=fallback_reason, conjugate=int(conjugate), reset=int(reset),
            roundoff_repair=int(bool(len(changed)))))
        if reset:
            p_prev.fill(0); q_prev.fill(0); delta_prev = 1.0
        else:
            p_prev, q_prev, delta_prev = p, q, curvature
        if iterations % log_every == 0:
            record()
    if history[-1]['iteration'] != iterations:
        record()
    summary = evaluate_dual(alpha, y, Q, C)
    summary.update(
        algorithm='CSMO', iterations=iterations, stop_reason=stop_reason,
        failure_detail=failure_detail,
        internal_kkt_violation=float(kkt_values(alpha, y, G, C)[0]),
        recomputed_exact_boundary_kkt=float(kkt_values(alpha, y, Q @ alpha - 1, C)[0]),
        gradient_max_error=float(np.max(np.abs(G - (Q @ alpha - 1)))),
        n_clipped=n_clipped, n_fallback=n_fallback, n_conjugate=n_conjugate,
        n_roundoff_repair=n_repair, n_gradient_refresh=n_refresh,
        clipping_rate=n_clipped / iterations if iterations else 0.0,
        fallback_rate=n_fallback / iterations if iterations else 0.0)
    return SolverResult(alpha, iterations, stop_reason, history, steps, summary)
