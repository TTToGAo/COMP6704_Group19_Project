"""Run paired author-mode experiments; produces NEW measurements only.

Requires a compiled libsvm_cd svm-train supporting `-a 0` and `-a 1`.
This harness times whole processes, not the archived Apple M6 svm_train() scope.
Use the original instrumented binary and logs for the published slide numbers.
"""
import argparse,csv,hashlib,json,platform,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=5)
    a=p.parse_args()
    if a.rounds<1: p.error('--rounds must be positive')
    a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for round_no in range(1,a.rounds+1):
        order=[0,1] if round_no%2 else [1,0]
        for position,mode in enumerate(order,1):
            command=[str(a.binary.resolve()),'-s','0','-t','2','-c','1',
                     '-g','0.045454545454545456','-e','0.001','-m','100',
                     '-h','0','-a',str(mode),str(a.data.resolve()),
                     str((a.out/f'round{round_no}_mode{mode}.model').resolve())]
            start=time.perf_counter()
            result=subprocess.run(command,text=True,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
            seconds=time.perf_counter()-start
            log=a.out/f'round{round_no}_mode{mode}.log'
            log.write_text(result.stdout)
            rows.append({'round':round_no,'position':position,'mode':mode,
                         'process_wall_seconds':seconds,'exit_code':result.returncode,
                         'log':log.name})
            if result.returncode: raise RuntimeError(f'{log}: solver exited {result.returncode}')
    with (a.out/'runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    meta={'timing_scope':'whole process wall time (includes startup, file loading and model writing)',
          'not_comparable_to_archived_Apple_M6_svm_train_only_time':True,
          'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),
          'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
          'platform':platform.platform(),'machine':platform.machine(),
          'command_template':'svm-train -s 0 -t 2 -c 1 -g 1/22 -e 0.001 -m 100 -h 0 -a MODE DATA MODEL',
          'rounds':a.rounds}
    (a.out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
