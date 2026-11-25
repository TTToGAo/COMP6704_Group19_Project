import csv,hashlib,json,sys,platform
from pathlib import Path
if not hasattr(hashlib,'file_digest'):
 def file_digest(stream,name):
  h=hashlib.new(name)
  while True:
   b=stream.read(1<<20)
   if not b:break
   h.update(b)
  return h
 hashlib.file_digest=file_digest
repo=Path(__file__).resolve().parents[1]
if (repo / 'code').is_dir():
    repo=repo / 'code'
root=repo / 'B_second_order' / 'original_reproduce'
sys.path.insert(0,str(root))
from experiment_io import load_fixed
from second_order_smo import solve as smo
from reference.csmo import solve as csmo
import numpy as np
cfg,X,y=load_fixed('ijcnn1_seed19')
cases=[('base',1,1/22),('C_half',.5,1/22),('C_double',2,1/22),('gamma_half',1,1/44),('gamma_double',1,1/11)]
out=Path(__file__).resolve().parent;out.mkdir(exist_ok=True)
rows=[]
for label,C,gamma in cases:
 for name,solver in [('second_order_smo',smo),('csmo',csmo)]:
  r=solver(X,y,C=C,gamma=gamma,epsilon=.001,max_iter=100000,log_every=10)
  s=r.summary
  row={'case':label,'algorithm':name,'C':C,'gamma':gamma,'updates':r.iterations,'stop_reason':r.stop_reason,'dual_objective':s['dual_objective'],'kkt_violation':s['kkt_violation'],'equality_residual':s['equality_residual'],'box_violation':s['box_violation']}
  rows.append(row)
  print(label,name,r.iterations,s['kkt_violation'],r.stop_reason,flush=True)
with (out/'sensitivity.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(out/'metadata.json').write_text(json.dumps({'dataset':'bundled ijcnn1_seed19 subset','subset_sha256':hashlib.sha256((root/'inputs/ijcnn1_seed19/subset.libsvm').read_bytes()).hexdigest(),'python':platform.python_version(),'numpy':np.__version__,'blas_threads':1,'epsilon':.001,'scope':'Python accepted update counts and endpoint metrics; elapsed time intentionally excluded'},indent=2)+'\n')
