"""Independent reconstruction of second-order working-set SMO.

This is newly written for the review package. It is separate from the original member implementation
implementation and must not be cited as the source of the archived B results.
"""
from dataclasses import dataclass
import numpy as np

from csmo import rbf_kernel, select_working_set, feasible_step, evaluate_dual, kkt_values


@dataclass
class SolverResult:
    alpha: np.ndarray
    iterations: int
    stop_reason: str
    history: list
    summary: dict


def solve(X, y, *, C=1.0, gamma=1/22, epsilon=0.001,
          max_iter=1_000_000, log_every=10):
    X=np.asarray(X,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    K=rbf_kernel(X,gamma)
    Q=y[:,None]*y[None,:]*K
    alpha=np.zeros(len(y),dtype=np.float64)
    G=-np.ones(len(y),dtype=np.float64)
    history=[]
    iterations=0
    def record():
        m=evaluate_dual(alpha,y,Q,C)
        history.append({'iteration':iterations,**{k:m[k] for k in ('dual_objective','kkt_violation','equality_residual','box_violation')}})
    record()
    reason='max_iter'
    while iterations<max_iter:
        gap=kkt_values(alpha,y,G,C)[0]
        if gap<=epsilon:
            exact=evaluate_dual(alpha,y,Q,C)
            if exact['kkt_violation']<=epsilon and exact['equality_residual']<=1e-8 and exact['box_violation']<=1e-10:
                reason='converged';break
            G=Q@alpha-1
        pair=select_working_set(alpha,y,G,K,C)
        if pair is None:
            reason='stalled';break
        i,j=pair
        d=np.zeros(len(y),dtype=np.float64)
        d[i],d[j]=y[i],-y[j]
        qd=y[i]*Q[:,i]-y[j]*Q[:,j]
        slope=-float(G@d)
        curvature=max(float(d@qd),1e-12)
        step=min(slope/curvature,feasible_step(alpha,d,C))
        if not np.isfinite(step) or step<=0:
            reason='stalled';break
        alpha+=step*d
        alpha=np.clip(alpha,0,C)
        G+=step*qd
        iterations+=1
        if iterations%log_every==0:record()
    if history[-1]['iteration']!=iterations:record()
    summary={'algorithm':'second_order_smo_reconstructed','iterations':iterations,'stop_reason':reason,**evaluate_dual(alpha,y,Q,C)}
    return SolverResult(alpha,iterations,reason,history,summary)
