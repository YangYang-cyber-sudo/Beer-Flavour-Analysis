"""内层选参、外层评估与逐拟合缓存。外层标签不传入搜索函数。"""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import warnings
import numpy as np
from support.common import clean_json
from .models import SensoryRegressor


def atomic_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    def convert(v):
        if isinstance(v,np.ndarray): return convert(v.tolist())
        if isinstance(v,dict): return {str(k):convert(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [convert(x) for x in v]
        return clean_json(v)
    tmp.write_text(json.dumps(convert(payload),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def atomic_npz(path,**values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as handle: np.savez_compressed(handle,**values)
    tmp.replace(path)


def scales(y):
    return y.std(axis=0),np.quantile(y,.75,axis=0)-np.quantile(y,.25,axis=0)


def scaled(error,scale):
    return np.divide(error,scale,out=np.full_like(error,np.nan,dtype=float),where=np.asarray(scale)>1e-10)


def score_vector(actual,predicted,scale_rows):
    normalized=scaled(predicted-actual,scale_rows)
    rmse=np.sqrt(np.nanmean(normalized**2,axis=0))
    if not np.isfinite(rmse).all(): raise ValueError('内层目标零方差或结果无效，需审查评分定义')
    k=max(1,int(np.ceil(.2*len(rmse))))
    return float(rmse.mean()+.2*np.sort(rmse)[-k:].mean()),rmse


def metric_rows(y,p,targets,sd,iqr):
    error=p-y; mse=np.mean(error**2,axis=0); den=np.sum((y-y.mean(axis=0))**2,axis=0)
    r2=np.divide(np.sum(error**2,axis=0),den,out=np.full(len(targets),np.nan),where=den>1e-15)
    nr=np.sqrt(np.nanmean(scaled(error,sd)**2,axis=0))
    ni=np.sqrt(np.nanmean(scaled(error,iqr)**2,axis=0))
    return [{'target':str(t),'rmse':float(np.sqrt(mse[j])),'mae':float(np.abs(error[:,j]).mean()),
             'r2':float(1-r2[j]),'nrmse_sd':float(nr[j]),'nrmse_iqr':float(ni[j]),
             'truth_std':float(y[:,j].std()),'n':len(y)} for j,t in enumerate(targets)]


def candidate_oof(x,y,folds,spec,groups,logs,path,seed,threads,device):
    path=Path(path)/spec['id']; path.mkdir(parents=True,exist_ok=True)
    predictions=np.full_like(y,np.nan); normalized=np.full_like(y,np.nan); elapsed=0
    for fold in range(3):
        tr=np.flatnonzero(folds!=fold); va=np.flatnonzero(folds==fold)
        cache=path/('inner_%d.npz'%fold)
        if cache.exists():
            with np.load(cache,allow_pickle=False) as saved:
                if not np.array_equal(saved['validation_indices'],va): raise ValueError('缓存内层索引不一致')
                pred=saved['pred']; duration=float(saved['seconds'])
        else:
            begin=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model=SensoryRegressor(spec,groups,logs,seed+fold,threads,device).fit(x[tr],y[tr])
                pred=model.predict(x[va])
            duration=time.perf_counter()-begin
            atomic_json(path/('inner_%d_log.json'%fold),{'spec':spec,'seconds':duration,
                'warnings':[str(w.message) for w in caught], 'train_n':len(tr),'validation_n':len(va)})
            atomic_npz(cache,validation_indices=va,pred=pred,seconds=duration)
        if pred.shape!=y[va].shape or not np.isfinite(pred).all(): raise ValueError('无效缓存/预测')
        predictions[va]=pred; normalized[va]=y[tr].std(axis=0); elapsed+=duration
    value, per=score_vector(y,predictions,normalized)
    result={'id':spec['id'],'family':spec['family'],'stage':spec['stage'],'selection_score':value,
            'macro_nrmse_sd':float(per.mean()),'worst_nrmse_sd':float(per.max()),'seconds':elapsed}
    atomic_json(path/'score.json',result)
    return result,predictions,normalized


def policies(y,inners,scale):
    names=sorted(inners)
    scored={n:score_vector(y,inners[n],scale)[0] for n in names}
    best=min(names,key=lambda n:(scored[n],n))
    loss=np.stack([np.mean(scaled(inners[n]-y,scale)**2,axis=0) for n in names])
    weights={}
    global_w=np.zeros_like(loss);global_w[names.index(best),:]=1
    target_w=np.zeros_like(loss); target_w[np.argmin(loss,axis=0),np.arange(y.shape[1])]=1
    eligible=sorted(names,key=lambda n:(scored[n],n))[:3]
    soft=np.zeros_like(loss)
    for n in eligible: soft[names.index(n)]=1/np.maximum(loss[names.index(n)],1e-8)**2
    soft/=soft.sum(axis=0)
    soft=.5*soft+.5*global_w
    stack=np.stack([inners[n] for n in names])
    if 'mean' in names:
        mixed=np.einsum('mnt,mt->nt',stack,soft)
        degrade=np.mean(scaled(mixed-y,scale)**2,axis=0)>loss[names.index('mean')]
        soft[:,degrade]=0;soft[names.index('mean'),degrade]=1
    for key,w in [('global_best',global_w),('target_best',target_w),('soft_blend',soft)]:
        p=np.einsum('mnt,mt->nt',stack,w)
        weights[key]={'weights':w,'inner_score':score_vector(y,p,scale)[0]}
    winner=min(weights,key=lambda k:(weights[k]['inner_score'],k))
    return names,weights,winner


def search(x,y,folds,protocol,groups,logs,path,threads=4,device='CPU'):
    records=[]; results={}; specs={r['id']:r for r in protocol['candidates']}
    def evaluate(spec):
        print('INNER',Path(path).name,spec['id'],flush=True)
        try:
            result,pred,scale=candidate_oof(x,y,folds,spec,groups,logs,path,protocol['seed'],threads,device)
            records.append(result);results[spec['id']]=(pred,scale)
        except Exception as exc:
            # 失败保留原始错误，不以均值填充后伪装成成功模型。
            import traceback
            atomic_json(Path(path)/spec['id']/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'spec':spec})
            records.append({'id':spec['id'],'family':spec['family'],'stage':spec['stage'],'error':repr(exc)})
            print('CANDIDATE_FAILED',spec['id'],repr(exc),flush=True)
    for spec in protocol['candidates']:
        if spec['stage']=='coarse': evaluate(spec)
    families=sorted({s['family'] for s in protocol['candidates']})
    def winners():
        best={}
        for family in families:
            ok=[r for r in records if r['family']==family and 'selection_score' in r]
            if not ok: raise RuntimeError('整个家族没有可信基线: '+family+'；检查 failure.json，不应静默跳过')
            best[family]=min(ok,key=lambda r:(r['selection_score'],r['id']))
        return best
    initial=winners()
    focus=sorted([f for f in families if f!='mean'],key=lambda f:(initial[f]['selection_score'],f))[:protocol['focus_families']]
    atomic_json(Path(path)/'focus.json',{'families':focus,'basis':'outer-training inner CV only'})
    for spec in protocol['candidates']:
        if spec['stage']=='refine' and spec['family'] in focus: evaluate(spec)
    best=winners()
    predictions={f:results[r['id']][0] for f,r in best.items()}
    scale=next(iter(results.values()))[1]
    names,weightsets,policy=policies(y,predictions,scale)
    atomic_json(Path(path)/'selection.json',{'best':best,'focus':focus,'policy':policy,
        'family_order':names,'policies':weightsets,'all_candidates':records})
    return {'best':{f:specs[r['id']] for f,r in best.items()},'records':records,
            'family_order':names,'policies':weightsets,'policy':policy,'best_scores':best}


class PredictionBundle:
    def __init__(self,models,weights,features,targets,grade_reference):
        self.models,self.weights,self.features,self.targets,self.grade_reference=models,weights,features,targets,grade_reference
    def predict(self,x):
        stack=np.stack([m.predict(x) for m in self.models])
        return np.einsum('mnt,mt->nt',stack,self.weights)
