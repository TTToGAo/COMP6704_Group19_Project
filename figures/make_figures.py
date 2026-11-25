"""Generate report figures from archived CSV data."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TIMING = (ROOT / 'results/apple_m6_recorded/timing_pairs.csv') if (ROOT / 'results').is_dir() else (ROOT / 'code/results/apple_m6_recorded/timing_pairs.csv')
SENSITIVITY = ROOT / 'extra_experiments/sensitivity.csv'
OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

with (TIMING).open(newline='') as f:
    timing = list(csv.DictReader(f))
rounds = np.arange(1,6)
smo = np.array([float(r['second_order_smo_train_seconds']) for r in timing])
csmo = np.array([float(r['csmo_train_seconds']) for r in timing])
fig,ax=plt.subplots(figsize=(6.5,3.5),layout='constrained')
ax.plot(rounds,smo,'o-',color='#2364aa',lw=2,label='Second-order SMO')
ax.plot(rounds,csmo,'s-',color='#d1495b',lw=2,label='CSMO')
for x,a,b in zip(rounds,smo,csmo):
    ax.plot([x,x],[a,b],color='#a6adb4',lw=1,zorder=0)
ax.set(xlabel='Paired round',ylabel='Training time (s)',xticks=rounds,ylim=(5.95,7.45))
ax.grid(axis='y',alpha=.25)
ax.legend(frameon=False,ncol=2,loc='upper left')
fig.savefig(OUT/'paired_training_times.pdf',bbox_inches='tight')
fig.savefig(OUT/'paired_training_times.png',dpi=180,bbox_inches='tight')
plt.close(fig)

with (SENSITIVITY).open(newline='') as f:
    rows=list(csv.DictReader(f))
order=['C_half','base','C_double','gamma_half','gamma_double']
labels=['C = 0.5','Shared baseline','C = 2','γ = 1/44','γ = 1/11']
by={(r['case'],r['algorithm']):int(r['updates']) for r in rows}
a=np.array([by[k,'second_order_smo'] for k in order]);b=np.array([by[k,'csmo'] for k in order])
fig,ax=plt.subplots(figsize=(7.1,3.8),layout='constrained')
x=np.arange(len(order));w=.36
ax.bar(x-w/2,a,w,color='#2364aa',label='Second-order SMO')
ax.bar(x+w/2,b,w,color='#d1495b',label='CSMO')
for i,(u,v) in enumerate(zip(a,b)):
    ax.text(i-w/2,u+4,str(u),ha='center',va='bottom',fontsize=9)
    ax.text(i+w/2,v+4,str(v),ha='center',va='bottom',fontsize=9)
ax.set(xlabel='Parameter setting on fixed ijcnn1 subset',ylabel='Accepted updates',xticks=x,xticklabels=labels,ylim=(0,530))
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
ax.legend(frameon=False,ncol=2,loc='upper right')
fig.savefig(OUT/'parameter_sensitivity.pdf',bbox_inches='tight')
fig.savefig(OUT/'parameter_sensitivity.png',dpi=180,bbox_inches='tight')
plt.close(fig)
