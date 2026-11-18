"""复用已验证读取器；冻结划分只看 beer，不看标签。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from support.audit import indexed, numeric
from support.tabular import read_tables

HERE = Path(__file__).resolve().parent
AF_FIELDS = [prefix+group+'_'+suffix for prefix in ['A_','F_']
             for group,suffixes in [('malt',['all','grain','bread','cara','burn']),
                                    ('hops',['all','citrus','tropical','noble','woody']),
                                    ('esters',['all','ethac','isoaa','flower','fruity'])]
             for suffix in suffixes]


def load_data(data_dir, metadata_path=None):
    xframe = indexed(next(read_tables(Path(data_dir)/'1.csv'))[1])
    yframe = indexed(next(read_tables(Path(data_dir)/'4.csv'))[1])
    if set(xframe.index)!=set(yframe.index):
        raise ValueError('X/Y beer 集合不同，拒绝静默丢弃或补造标签')
    beers = sorted(xframe.index, key=lambda v:int(v))
    xframe, yframe = xframe.loc[beers], yframe.loc[beers]
    if not (xframe['tasting_category_fine']==yframe['tasting_category_fine']).all():
        raise ValueError('按 beer 对齐后的风格不一致')
    x = numeric(xframe.drop(columns=['beer_id','tasting_category_fine']))
    y = numeric(yframe.drop(columns=['beer_id','tasting_category_fine']))
    if x.shape!=(250,231) or y.shape!=(250,50) or not set(AF_FIELDS)<=set(y):
        raise ValueError('官方数据结构与已确认版本不同，需审查后继续')
    if not np.isfinite(y.to_numpy()).all() or np.isinf(x.to_numpy()).any():
        raise ValueError('标签缺失/无穷值或特征无穷值；需明确处理，不能静默删除')
    metadata = pd.read_csv(metadata_path or HERE/'feature_metadata.csv',keep_default_na=False)
    # .loc with a plain Index drops the index name in recent pandas versions;
    # restore it so the per-run feature_metadata.csv keeps the field names.
    metadata = metadata.set_index('field').loc[x.columns].rename_axis('field').reset_index()
    metadata['current_missing_count'] = x.isna().sum().to_numpy()
    metadata['current_negative_count'] = (x<0).sum().to_numpy()
    groups = metadata['origin_metadata'].replace('', 'Unspecified').to_numpy(dtype=str)
    logs = metadata['transform_status'].str.startswith('log10').to_numpy()
    return {'X':x.to_numpy(),'Y':y.to_numpy(),'beers':np.array(beers),
            'features':np.array(x.columns),'targets':np.array(y.columns),
            'styles':xframe['tasting_category_fine'].to_numpy(dtype=str),
            'groups':groups,'verified_log':logs,'metadata':metadata}


def assignment(beers, nfold, seed):
    order = np.random.RandomState(seed).permutation(len(beers))
    result = np.empty(len(beers),dtype=int)
    for fold, indices in enumerate(np.array_split(order,nfold)):
        result[indices]=fold
    return result


def make_splits(beers, path, seed=2601731):
    path = Path(path)
    path.mkdir(parents=True,exist_ok=True)
    if (path/'outer.csv').exists() or (path/'inner.csv').exists():
        raise FileExistsError('冻结划分已存在，拒绝重新生成')
    outer = assignment(beers,5,seed)
    pd.DataFrame({'beer':beers,'outer_fold':outer}).to_csv(path/'outer.csv',index=False)
    records=[]
    for scope in list(range(5))+[-1]:
        train = np.flatnonzero(outer!=scope) if scope!=-1 else np.arange(len(beers))
        folds = assignment(np.asarray(beers)[train],3,seed+100+scope)
        records.extend({'scope':scope,'beer':str(beers[i]),'inner_fold':int(f)} for i,f in zip(train,folds))
    pd.DataFrame(records).to_csv(path/'inner.csv',index=False)
    (path/'definition.json').write_text(json.dumps({'seed':seed,'unit':'beer','outer':5,'inner':3,
        'scope_minus_one':'full-data inner folds for later deployment fit',
        'label_access_for_split_generation':False},indent=2),encoding='utf-8')


def load_splits(beers,path):
    outer=pd.read_csv(Path(path)/'outer.csv',dtype={'beer':str})
    inner=pd.read_csv(Path(path)/'inner.csv',dtype={'beer':str})
    if outer.beer.duplicated().any() or set(outer.beer)!=set(beers):
        raise ValueError('外层划分 beer 不唯一或不完整')
    outer=outer.set_index('beer').loc[beers,'outer_fold'].to_numpy(dtype=int)
    if set(outer)!=set(range(5)) or not (np.bincount(outer)==len(beers)//5).all():
        raise ValueError('外层必须是 5 个等大小互斥折')
    lookup={}
    if set(inner.scope)!=set(list(range(5))+[-1]): raise ValueError('内层 scope 缺失/多余')
    for scope in list(range(5))+[-1]:
        train=np.flatnonzero(outer!=scope) if scope!=-1 else np.arange(len(beers))
        frame=inner[inner.scope==scope]
        if frame.beer.duplicated().any() or set(frame.beer)!=set(np.asarray(beers)[train]):
            raise ValueError('内层包含外层测试样本或缺少训练样本')
        values=frame.set_index('beer').loc[np.asarray(beers)[train],'inner_fold'].to_numpy(dtype=int)
        if set(values)!={0,1,2}: raise ValueError('训练侧内层必须有三折')
        lookup[scope]=(train,values)
    return outer,lookup


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True,type=Path)
    p.add_argument('--split-dir',required=True,type=Path)
    a=p.parse_args()
    d=load_data(a.data_dir)
    make_splits(d['beers'],a.split_dir)
    load_splits(d['beers'],a.split_dir)
    print('FROZEN_SPLITS_READY',a.split_dir.resolve())
