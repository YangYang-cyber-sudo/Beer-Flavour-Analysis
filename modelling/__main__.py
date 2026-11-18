"""Command-line entry point: ``python -m modelling --stage evaluate ...``."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import zipfile
import numpy as np
import pandas as pd
from support.common import clean_json, protect_output
from support.tabular import read_tables
from .data import HERE, AF_FIELDS, load_data, load_splits, indexed, numeric


def runtime(expected=None):
    if expected and platform.python_version()!=expected:
        raise RuntimeError('解释器版本应为 '+expected+'，实际 '+platform.python_version())
    result={'python':platform.python_version(),'executable':sys.executable,'platform':platform.platform()}
    for name in ['numpy','pandas','openpyxl','scipy','sklearn','joblib','matplotlib','catboost','lightgbm']:
        result[name]=importlib.import_module(name).__version__
    return result


def dataframe(path,rows):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    tmp=Path(str(path)+'.tmp');frame.to_csv(tmp,index=False,encoding='utf-8-sig');tmp.replace(path)


def freeze(run,d,protocol,versions,options,split_dir,resume):
    from .engine import atomic_json,atomic_npz
    state=run/'data_snapshot.npz'
    signature={'protocol':protocol,'runtime':versions,'options':options}
    if state.exists():
        if not resume: raise FileExistsError('已有运行，需要 --resume；不能覆盖旧实验')
        previous=json.loads((run/'run_definition.json').read_text(encoding='utf-8'))
        if previous!=signature: raise ValueError('协议、版本或计算选项变化；必须使用新运行目录')
        with np.load(state,allow_pickle=False) as saved:
            for key in ['X','Y','beers','features','targets','styles','groups','verified_log']:
                a,b=saved[key],d[key]
                same=np.array_equal(a,b) if a.dtype.kind in 'USb' else np.array_equal(a,b,equal_nan=True)
                if not same: raise ValueError('数据或元信息变化，不能复用旧预测: '+key)
        for source in sorted(HERE.glob('*.py')):
            if (run/'code'/source.name).read_bytes()!=source.read_bytes():
                raise ValueError('训练源码变化；请开新协议目录，不与旧缓存混用')
        for name in ['outer.csv','inner.csv','definition.json']:
            if (run/'splits'/name).read_bytes()!=(Path(split_dir)/name).read_bytes():
                raise ValueError('冻结折被修改')
    else:
        atomic_json(run/'run_definition.json',signature)
        (run/'code').mkdir(exist_ok=True);(run/'splits').mkdir(exist_ok=True)
        for source in HERE.glob('*.py'): shutil.copyfile(source,run/'code'/source.name)
        for name in ['outer.csv','inner.csv','definition.json']: shutil.copyfile(Path(split_dir)/name,run/'splits'/name)
        dataframe(run/'feature_metadata.csv',d['metadata'])
        dataframe(run/'target_scope.csv',[{'target':t,'scope':'Aroma_Flavour_30' if t in AF_FIELDS else 'Other_20',
             'raw_0_6_truth_available':False} for t in d['targets']])
        atomic_npz(state,**{k:d[k] for k in ['X','Y','beers','features','targets','styles','groups','verified_log']})


def save_model(path,model):
    import joblib
    tmp=Path(str(path)+'.tmp');joblib.dump(model,tmp,compress=3);tmp.replace(path)


def fit_scope(d,indices,folds,protocol,path,args):
    from .engine import search,atomic_json,atomic_npz
    from .models import SensoryRegressor
    import joblib
    selection=search(d['X'][indices],d['Y'][indices],folds,protocol,d['groups'],d['verified_log'],
                     path/'search',args.threads,args.catboost_device)
    models={};costs={}
    for family,spec in selection['best'].items():
        dest=path/(family+'.joblib')
        timing=path/(family+'_fit.json')
        if dest.exists() and timing.exists():
            model=joblib.load(dest);seconds=json.loads(timing.read_text(encoding='utf-8'))['seconds']
        else:
            begin=time.perf_counter()
            import warnings
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model=SensoryRegressor(spec,d['groups'],d['verified_log'],protocol['seed']+1000,args.threads,args.catboost_device)
                model.fit(d['X'][indices],d['Y'][indices])
            seconds=time.perf_counter()-begin
            save_model(dest,model)
            atomic_json(timing,{'spec':spec,'seconds':seconds,'warnings':[str(w.message) for w in caught]})
        models[family]=model
        costs[family]={'search_seconds':sum(r.get('seconds',0) for r in selection['records'] if r['family']==family),
                       'fit_seconds':seconds}
    return selection,models,costs


def evaluate(d,outer,inners,protocol,run,args):
    from .engine import atomic_npz,atomic_json,scales
    from .models import relative_grade
    for fold in range(5):
        path=run/('outer_%d'%fold); path.mkdir(exist_ok=True)
        result=path/'predictions.npz'
        if result.exists():
            print('RESUME_OUTER_DONE',fold,flush=True)
            continue
        tr,ifolds=inners[fold];va=np.flatnonzero(outer==fold)
        print('OUTER',fold,'TRAIN',len(tr),'TEST',len(va),flush=True)
        # 搜索接口没有外层测试 X/Y 参数，所有自适应决策在这里结束。
        selected,models,costs=fit_scope(d,tr,ifolds,protocol,path,args)
        preds={f:m.predict(d['X'][va]) for f,m in models.items()}
        stack=np.stack([preds[n] for n in selected['family_order']])
        for key,item in selected['policies'].items():
            preds[key]=np.einsum('mnt,mt->nt',stack,np.asarray(item['weights']))
        preds['adaptive']=preds[selected['policy']]
        grade_params={}
        for method in ['quantile','robust']:
            grade,edges,bad=relative_grade(preds['adaptive'],d['Y'][tr],method)
            preds['relative_'+method]=grade
            grade_params[method]={'edges':edges,'undefined_targets':d['targets'][bad].tolist(),
                                  'truth_grade_accuracy':'not_available'}
        sd,iqr=scales(d['Y'][tr])
        atomic_json(path/'selection_and_costs.json',{'selected':selected,'costs':costs,'outer_fold':fold})
        atomic_json(path/'relative_grade_parameters.json',grade_params)
        # 折预测最后写入：存在意味着该折的模型、配置和等级参数都已完整写入。
        atomic_npz(result,indices=va,sd=sd,iqr=iqr,**preds)
    summarize(d,outer,run)
    atomic_json(run/'evaluation_complete.json',{'status':'NESTED_CV_COMPLETE_NOT_EXTERNAL_TEST',
        'prior_seed_2026_development':True,'outer_labels_used_for_search':False,
        'next_action':'Scientific review before using outer diagnostics to change protocol'})


def summarize(d,outer,run):
    from .engine import metric_rows,atomic_json,scaled
    n,t=d['Y'].shape;preds={};sd=np.zeros((n,t));iqr=np.zeros((n,t));metric=[];foldrows=[];costs=[];searchrows=[]
    definition=json.loads((run/'run_definition.json').read_text(encoding='utf-8'))
    specs={r['id']:r for r in definition['protocol']['candidates']}
    for fold in range(5):
        path=run/('outer_%d'%fold)
        with np.load(path/'predictions.npz',allow_pickle=False) as saved:
            va=saved['indices']
            if not np.array_equal(va,np.flatnonzero(outer==fold)): raise ValueError('外层缓存索引不匹配')
            sd[va]=saved['sd'];iqr[va]=saved['iqr']
            for name in saved.files:
                if name in ['indices','sd','iqr']:continue
                if name not in preds:preds[name]=np.full((n,t),np.nan)
                preds[name][va]=saved[name]
                if name.startswith('relative_'):continue
                rows=metric_rows(d['Y'][va],saved[name],d['targets'],saved['sd'],saved['iqr'])
                foldrows.extend(dict(r,model=name,outer_fold=fold) for r in rows)
        info=json.loads((path/'selection_and_costs.json').read_text(encoding='utf-8'))
        costs.extend(dict(c,model=f,outer_fold=fold) for f,c in info['costs'].items())
        searchrows.extend(dict(r,outer_fold=fold,preprocessing=json.dumps(specs[r['id']]['pre'],sort_keys=True),
                               parameters=json.dumps(specs[r['id']]['params'],sort_keys=True)) for r in info['selected']['records'])
    oofdir=run/'oof';oofdir.mkdir(exist_ok=True)
    for name,p in preds.items():
        if not np.isfinite(p).all():raise ValueError('OOF 缺失或无效 '+name)
        frame=pd.DataFrame(p,columns=d['targets']);frame.insert(0,'outer_fold',outer);frame.insert(0,'beer',d['beers'])
        dataframe(oofdir/(name+'.csv'),frame)
        if not name.startswith('relative_'):metric.extend(dict(r,model=name) for r in metric_rows(d['Y'],p,d['targets'],sd,iqr))
    actual=pd.DataFrame(d['Y'],columns=d['targets']);actual.insert(0,'beer',d['beers']);dataframe(oofdir/'truth_continuous.csv',actual)
    table=pd.DataFrame(metric);foldtable=pd.DataFrame(foldrows)
    dataframe(run/'metrics_per_target.csv',table);dataframe(run/'metrics_per_target_per_fold.csv',foldtable)
    dataframe(run/'metrics_aroma_flavour_30.csv',table[table.target.isin(AF_FIELDS)])
    dataframe(run/'model_seconds.csv',costs)
    dataframe(run/'inner_search_results.csv',searchrows)
    summaries=[]
    for name in table.model.unique():
        for scope,fields in [('All_50',d['targets']),('Aroma_Flavour_30',AF_FIELDS)]:
            rows=table[(table.model==name)&table.target.isin(fields)]
            per=foldtable[(foldtable.model==name)&foldtable.target.isin(fields)].groupby('outer_fold')[['rmse','mae','r2','nrmse_sd']].mean()
            accounted=[c for c in costs if c['model']==name] if name in {c['model'] for c in costs} else costs
            summaries.append({'model':name,'scope':scope,'macro_rmse':rows.rmse.mean(),'macro_mae':rows.mae.mean(),
                'macro_r2':rows.r2.mean(),'macro_nrmse_sd':rows.nrmse_sd.mean(),'macro_nrmse_iqr':rows.nrmse_iqr.mean(),
                'positive_r2_targets':int((rows.r2>0).sum()),'worst_r2':rows.r2.min(),
                'r2_defined_targets':int(rows.r2.notna().sum()),'nrmse_sd_defined_targets':int(rows.nrmse_sd.notna().sum()),
                'nrmse_iqr_defined_targets':int(rows.nrmse_iqr.notna().sum()),
                'worst_target':rows.loc[rows.r2.idxmin(),'target'],'worst_nrmse_sd':rows.nrmse_sd.max(),
                'tail20_nrmse_sd':rows.nlargest(max(1,int(np.ceil(len(rows)*.2))),'nrmse_sd').nrmse_sd.mean(),
                'fold_macro_rmse_std':per.rmse.std(ddof=1),'fold_macro_r2_std':per.r2.std(ddof=1),
                'search_and_fit_seconds':sum(c['search_seconds']+c['fit_seconds'] for c in accounted),
                'evaluation_role':'internal_nested_cv_prior_exploration_same_cohort'})
    dataframe(run/'summary.csv',summaries)
    normalized=scaled(preds['adaptive']-d['Y'],sd)
    dataframe(run/'sample_errors_diagnostic.csv',pd.DataFrame({'beer':d['beers'],'style':d['styles'],
        'outer_fold':outer,'adaptive_mean_normalized_squared_error':np.mean(normalized**2,axis=1),
        'X_missing_count':np.isnan(d['X']).sum(axis=1)}).sort_values('adaptive_mean_normalized_squared_error',ascending=False))
    # 依赖诊断在各训练折计算，不能把全量相关矩阵接回特征筛选。
    dependencies=[]
    for fold in range(5):
        tr=np.flatnonzero(outer!=fold)
        combined=pd.DataFrame(np.column_stack([d['X'][tr],d['Y'][tr]]))
        correlations_all=combined.corr(method='spearman').to_numpy()[:d['X'].shape[1],d['X'].shape[1]:]
        for j,target in enumerate(d['targets']):
            correlations=np.nan_to_num(np.abs(correlations_all[:,j]),nan=0.)
            k=int(correlations.argmax())
            dependencies.append({'outer_fold':fold,'target':target,'train_label_std':d['Y'][tr,j].std(),
                'max_train_univariate_abs_rank_corr':correlations[k],'feature':d['features'][k],
                'role':'posthoc_diagnostic_not_feature_selection'})
    dataframe(run/'train_side_target_dependencies.csv',dependencies)
    hard=table[table.model=='adaptive'].sort_values('r2').head(10)
    dataframe(run/'hard_targets_diagnostic.csv',hard)
    from .plots import result_plots
    result_plots(run,table,d,preds['adaptive'],outer)
    rows=pd.DataFrame(summaries);main=rows[(rows.model=='adaptive')&(rows.scope=='All_50')].iloc[0]
    atomic_json(run/'result_summary.json',main.to_dict())
    (run/'RESULT_README.md').write_text(
        '# Q1 连续标签嵌套 CV\n\n状态：NESTED_CV_COMPLETE_NOT_EXTERNAL_TEST。\n\n'
        'adaptive 是预冻结的训练侧自适应算法；其它模型是同折对照。全部外折预测在 oof/。\n'
        '不可依据这里的外折结果挑最佳家族再将其成绩称为一次独立测试。该批 250 样本已有先前探索。\n'
        'summary.csv 同时报全部50维与Aroma/Flavour30维；最差目标不能从论文省略。\n'
        'relative_* 是两种未审核的相对分级备选，-1=训练量尺不可区分，不是官方0–6真值。\n'
        'hard_targets、sample_errors、train_side_target_dependencies 只能诊断；改方案后需标为开发迭代。\n'
        '各折模型/配置、内层逐配置缓存/警告/耗时在 outer_*/；断点续跑不重新计算成功拟合。\n',encoding='utf-8')


def final_fit(d,inners,protocol,run,args):
    from .engine import PredictionBundle,atomic_json
    path=run/'final';path.mkdir(exist_ok=True)
    if (path/'model.joblib').exists():
        print('FINAL_ALREADY_COMPLETE');return
    indices,folds=inners[-1]
    selected,models,costs=fit_scope(d,indices,folds,protocol,path,args)
    names=selected['family_order']; weights=np.asarray(selected['policies'][selected['policy']]['weights'])
    active=np.flatnonzero(np.any(weights>0,axis=1))
    bundle=PredictionBundle([models[names[i]] for i in active],weights[active],d['features'],d['targets'],d['Y'])
    save_model(path/'model.joblib',bundle)
    atomic_json(path/'deployment.json',{'policy':selected['policy'],'family_order':names,'weights':weights,
                                      'selection_basis':'full-data inner CV only; no outer ranking',
                                      'validation':'no in-sample accuracy is reported','costs':costs})


def predict(args,run):
    import joblib
    from .models import relative_grade
    from .engine import atomic_json
    model=joblib.load(run/'final'/'model.joblib')
    frame=indexed(next(read_tables(args.predict_input))[1])
    missing=set(model.features)-set(frame.columns)
    if missing:raise ValueError('预测输入缺少字段 '+str(sorted(missing)))
    x=numeric(frame[list(model.features)]).to_numpy()
    p=model.predict(x)
    output=args.prediction_dir.resolve();protect_output(output,[args.predict_input,args.data_dir])
    output.mkdir(parents=True,exist_ok=False)
    result=pd.DataFrame(p,columns=model.targets);result.insert(0,'beer',frame.index)
    dataframe(output/'continuous_predictions.csv',result)
    for method in ['quantile','robust']:
        grade,edges,bad=relative_grade(p,model.grade_reference,method)
        g=pd.DataFrame(grade,columns=model.targets);g.insert(0,'beer',frame.index)
        dataframe(output/('relative_'+method+'_NOT_ORIGINAL_0_6.csv'),g)
        atomic_json(output/('relative_'+method+'_definition.json'),{'edges':edges,'undefined_targets':model.targets[bad],
                                                                  'participant_approval':'pending'})
    atomic_json(output/'prediction_status.json',{'status':'DEPLOYMENT_PREDICTIONS_NOT_OOF',
        'raw_0_6_truth_available':False,'input':str(args.predict_input.resolve()),
        'warning':'If historical training beers were supplied, these are refit predictions; no validation score is computed.'})
    pack(output)


def pack(run):
    # 临时文件以原子替换收尾；中断重启不破坏之前可下载的结果包。
    dest=run.with_name(run.name+'.zip');tmp=dest.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(run.rglob('*')):
            if file.is_file() and not file.name.endswith('.tmp'): archive.write(file,file.relative_to(run).as_posix())
    tmp.replace(dest)
    print('RESULT_ZIP='+str(dest.resolve()),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['runtime','evaluate','fit-final','predict'],default='evaluate')
    parser.add_argument('--data-dir',type=Path,default=HERE.parent/'data'/'raw')
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--protocol',type=Path,default=HERE/'protocol.json')
    parser.add_argument('--split-dir',type=Path,default=HERE.parent/'validation'/'splits')
    parser.add_argument('--expected-python',default=None)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--catboost-device',choices=['CPU','GPU'],default='CPU')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--predict-input',type=Path)
    parser.add_argument('--prediction-dir',type=Path)
    args=parser.parse_args();run=args.run_dir.resolve();protect_output(run,[args.data_dir])
    from support.paths import configure_console
    configure_console()
    if args.threads<1:parser.error('--threads 必须为正')
    if run.exists() and not args.resume:parser.error('运行目录已存在；续跑需明确 --resume')
    if run.exists() and args.resume and args.stage!='runtime' and not (run/'run_definition.json').exists():
        if {p.name for p in run.iterdir()}-{'run.log'}:
            parser.error('该目录不是本训练程序的运行目录，拒绝写入已有审查成果')
    if args.stage=='predict' and (not args.predict_input or not args.prediction_dir):parser.error('predict 需要输入和新输出目录')
    run.mkdir(parents=True,exist_ok=True)
    handle=(run/'run.log').open('a',encoding='utf-8',buffering=1)
    class Tee:
        def __init__(self,stream):self.stream=stream
        def write(self,text):self.stream.write(text);handle.write(text)
        def flush(self):self.stream.flush();handle.flush()
    stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(stdout),Tee(stderr)
    exitcode=0
    try:
        versions=runtime(args.expected_python)
        print('RUNTIME',json.dumps(versions,ensure_ascii=False),flush=True)
        if args.stage=='runtime':
            (run/'runtime.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
            print('Q1_RUNTIME_PASS')
        elif args.stage=='predict': predict(args,run)
        else:
            d=load_data(args.data_dir);outer,inners=load_splits(d['beers'],args.split_dir)
            protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
            required={'mean','ridge','elastic','pls','krr','extra','catboost','lightgbm'}
            if {s['family'] for s in protocol['candidates']}!=required:
                raise ValueError('正式协议必须覆盖全部八个家族，不能静默删掉失败或未安装的家族')
            if protocol['outer_folds']!=5 or protocol['inner_folds']!=3 or protocol['focus_families']!=3:
                raise ValueError('正式协议固定为外五内三、追加三家族；改动需另立版本')
            split_definition=json.loads((args.split_dir/'definition.json').read_text(encoding='utf-8'))
            if split_definition['seed']!=protocol['seed']:raise ValueError('协议种子与冻结折定义不一致')
            options={'threads':args.threads,'catboost_device':args.catboost_device}
            freeze(run,d,protocol,versions,options,args.split_dir,args.resume)
            styles=pd.DataFrame({'outer_fold':outer,'style':d['styles']}).groupby(['outer_fold','style']).size().reset_index(name='n')
            dataframe(run/'outer_style_counts.csv',styles)
            if args.stage=='evaluate': evaluate(d,outer,inners,protocol,run,args)
            else:
                if not (run/'evaluation_complete.json').exists():raise ValueError('先完成 evaluate，再做无评分的全量部署拟合')
                final_fit(d,inners,protocol,run,args)
    except KeyboardInterrupt:
        print('INTERRUPTED; reuse same command with --resume',flush=True);exitcode=130
    except Exception:
        traceback.print_exc();exitcode=1
    finally:
        sys.stdout,sys.stderr=stdout,stderr;handle.close()
        pack(run)
    return exitcode


if __name__=='__main__':raise SystemExit(main())
