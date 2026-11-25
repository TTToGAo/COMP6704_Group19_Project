"""Shared CLI for the C author handoff module and reconstructed B solver."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'C_csmo'),str(ROOT/'B_second_order'),str(ROOT/'D_pgd')]
from common.data import load_libsvm

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--algorithm',choices=['csmo','second_order_smo'],required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--C',type=float,default=1.0)
    p.add_argument('--gamma',type=float,default=1/22)
    p.add_argument('--epsilon',type=float,default=.001)
    p.add_argument('--max-iter',type=int,default=1_000_000)
    p.add_argument('--log-every',type=int,default=10)
    a=p.parse_args()
    from csmo import solve as csmo_solve
    from second_order_smo import solve as smo_solve
    X,y=load_libsvm(a.data)
    solver=csmo_solve if a.algorithm=='csmo' else smo_solve
    r=solver(X,y,C=a.C,gamma=a.gamma,epsilon=a.epsilon,max_iter=a.max_iter,log_every=a.log_every)
    a.out.mkdir(parents=True,exist_ok=True)
    np.save(a.out/'alpha.npy',r.alpha)
    summary=dict(r.summary)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    config={'algorithm':a.algorithm,'dataset_id':a.data.name,'dataset_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
            'n_train':len(y),'kernel':'rbf','C':a.C,'gamma':a.gamma,'epsilon':a.epsilon,
            'alpha0':0,'max_iter':a.max_iter,'log_every':a.log_every,'dtype':'float64','feature_scaling':'none'}
    (a.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    fields=['iteration','dual_objective','kkt_violation','equality_residual','box_violation']
    with (a.out/'history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(r.history)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
