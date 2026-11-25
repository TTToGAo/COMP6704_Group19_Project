"""Report the maximization objective and the LIBSVM working-set KKT gap."""

import numpy as np


def evaluate_dual(alpha, y, Q, C):
    alpha = np.asarray(alpha, dtype=np.float64)
    y = np.asarray(y)
    if alpha.ndim != 1 or not alpha.size or y.shape != alpha.shape:
        raise ValueError("Expected nonempty alpha and y vectors of equal length")
    if not np.isfinite(C) or C <= 0 or not np.isin(y, (-1, 1)).all():
        raise ValueError("Expected positive finite C and labels in {-1, +1}")

    q_alpha = Q @ alpha
    gradient = q_alpha - 1.0
    boundary_tol = 1e-10 * max(1.0, C)
    below_upper = alpha < C - boundary_tol
    above_lower = alpha > boundary_tol
    up = np.where(y == 1, below_upper, above_lower)
    low = np.where(y == 1, above_lower, below_upper)
    signed_gradient = y * gradient
    gmax = float(np.max(-signed_gradient[up])) if up.any() else np.nan
    gmax2 = float(np.max(signed_gradient[low])) if low.any() else np.nan
    return {
        "dual_objective": float(alpha.sum() - 0.5 * (alpha @ q_alpha)),
        "equality_residual": float(abs(y @ alpha)),
        "box_violation": float(np.maximum(0.0, np.maximum(-alpha, alpha - C)).max()),
        "kkt_violation": float(np.maximum(0.0, gmax + gmax2)),
        "gmax": gmax,
        "gmax2": gmax2,
    }
