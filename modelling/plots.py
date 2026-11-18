"""诊断图直接读取真实标签与保存的 OOF，不造模拟成绩。"""
from pathlib import Path
import numpy as np


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'font.size':9,'svg.fonttype':'none'})
    return plt


def label_plot(data,output):
    plt=setup();path=Path(output);path.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(13,4.5))
    y=data['Y']; order=np.argsort(y.std(axis=0))
    p05,p25,p50,p75,p95=np.quantile(y,[.05,.25,.5,.75,.95],axis=0)
    for i,j in enumerate(order):
        ax.plot([i,i],[p05[j],p95[j]],color='#A8BCCB',linewidth=1)
        ax.plot([i,i],[p25[j],p75[j]],color='#346888',linewidth=3)
    ax.scatter(np.arange(len(order)),p50[order],s=8,color='#20262C',zorder=3)
    ax.set_xticks(np.arange(len(order)));ax.set_xticklabels(data['targets'][order],rotation=90,fontsize=7)
    ax.set_ylabel('Supplied continuous sensory value')
    ax.set_title('%d beers: median, IQR and 5–95%% range; ordered by SD (not original 0–6 scores)'%len(y))
    fig.tight_layout();fig.savefig(path/'real_label_distribution.png',dpi=180);fig.savefig(path/'real_label_distribution.svg');plt.close(fig)


def result_plots(run,table,data,adaptive,folds):
    plt=setup();out=Path(run)/'figures';out.mkdir(exist_ok=True)
    label_plot(data,out)
    pivot=table.pivot(index='model',columns='target',values='r2')
    pivot=pivot.reindex(columns=data['targets'])
    fig,ax=plt.subplots(figsize=(16,6))
    image=ax.imshow(pivot.to_numpy(),aspect='auto',vmin=-.5,vmax=.6,cmap='RdBu')
    ax.set_yticks(np.arange(len(pivot)));ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)));ax.set_xticklabels(pivot.columns,rotation=90,fontsize=7)
    ax.set_title('Pooled outer OOF R2 per target; color clipped [-0.5, 0.6], exact values in CSV')
    fig.colorbar(image,ax=ax,label='OOF R2');fig.tight_layout();fig.savefig(out/'oof_r2_all_targets.png',dpi=180);plt.close(fig)
    hardest=table[table.model=='adaptive'].sort_values('r2').head(4)
    fig,axes=plt.subplots(2,2,figsize=(9,8))
    for ax,(_,row) in zip(axes.ravel(),hardest.iterrows()):
        j=list(data['targets']).index(row.target);actual=data['Y'][:,j];pred=adaptive[:,j]
        ax.scatter(actual,pred,c=folds,cmap='viridis',s=12,alpha=.65,vmin=0,vmax=4)
        lo=min(actual.min(),pred.min());hi=max(actual.max(),pred.max())
        ax.plot([lo,hi],[lo,hi],color='#6B7280',linewidth=1)
        ax.set_title('%s | R2=%.3f'%(row.target,row.r2));ax.set_xlabel('Observed continuous label');ax.set_ylabel('Outer OOF prediction')
    fig.suptitle('Four hardest targets: post-hoc diagnostics, not a model selection signal')
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/'hard_targets_oof.png',dpi=180);plt.close(fig)
