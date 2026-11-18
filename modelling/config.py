"""在观察任何外层结果之前冻结的有限搜索空间。"""
import json
from pathlib import Path


def grid():
    plain={'shape':'identity','clip':0,'scale':'standard','indicators':True}
    clip=dict(plain,clip=.01)
    log=dict(clip,shape='signed_log',scale='robust')
    rank=dict(plain,shape='rank')
    group=dict(rank,group_balance=True)
    select=dict(clip,select=100)
    records=[]
    def add(family,stage,pre,**params):
        number=sum(r['family']==family for r in records)
        records.append({'id':family+'_%02d'%number,'family':family,'stage':stage,'pre':dict(pre),'params':params})
    add('mean','coarse',plain)
    for pre,alpha in [(plain,100),(clip,10),(log,100),(rank,1000)]: add('ridge','coarse',pre,alpha=alpha)
    for pre,alpha in [(clip,100),(log,10),(group,100),(select,1000)]: add('ridge','refine',pre,alpha=alpha)
    for pre,alpha,l1 in [(plain,.2,.2),(clip,.05,.5),(log,.02,.2),(rank,.01,.8)]:
        add('elastic','coarse',pre,alpha=alpha,l1_ratio=l1)
    for pre,alpha,l1 in [(clip,.01,.2),(rank,.05,.2),(group,.01,.5),(select,.1,.8)]:
        add('elastic','refine',pre,alpha=alpha,l1_ratio=l1)
    for pre,n in [(plain,2),(clip,5),(log,10),(rank,15)]: add('pls','coarse',pre,n_components=n)
    for pre,n in [(clip,2),(rank,5),(group,10),(select,20)]: add('pls','refine',pre,n_components=n)
    for pre,alpha,gamma in [(clip,.1,1),(rank,.01,1),(log,1,.5),(plain,.1,4)]:
        add('krr','coarse',pre,alpha=alpha,gamma_factor=gamma)
    for pre,alpha,gamma in [(rank,.1,4),(clip,.01,.5),(group,.1,1),(rank,1,1)]:
        add('krr','refine',pre,alpha=alpha,gamma_factor=gamma)
    for pre,leaf,feat in [(plain,1,1.),(plain,2,.7),(clip,4,.7),(log,2,.4)]:
        add('extra','coarse',pre,n_estimators=200,min_samples_leaf=leaf,max_features=feat,max_leaf_nodes=100)
    for pre,leaf,feat in [(plain,2,.5),(clip,3,1.),(select,2,.7),(rank,1,.7)]:
        add('extra','refine',pre,n_estimators=400,min_samples_leaf=leaf,max_features=feat,max_leaf_nodes=100)
    for pre,depth,l2 in [(plain,3,10),(clip,4,10),(rank,5,20),(log,4,3)]:
        add('catboost','coarse',pre,iterations=300,depth=depth,learning_rate=.04,l2_leaf_reg=l2)
    for pre,depth,l2 in [(plain,4,20),(clip,5,10),(select,4,5),(rank,3,3)]:
        add('catboost','refine',pre,iterations=600,depth=depth,learning_rate=.03,l2_leaf_reg=l2)
    for pre,leaves,child,l2 in [(plain,5,15,5),(clip,7,20,10),(rank,11,15,10),(log,7,25,1)]:
        add('lightgbm','coarse',pre,n_estimators=180,num_leaves=leaves,max_depth=4,
            min_child_samples=child,reg_lambda=l2,learning_rate=.04,colsample_bytree=.8)
    for pre,leaves,child,l2 in [(plain,7,12,10),(clip,11,20,5),(select,7,15,5),(rank,5,25,20)]:
        add('lightgbm','refine',pre,n_estimators=350,num_leaves=leaves,max_depth=4,
            min_child_samples=child,reg_lambda=l2,learning_rate=.03,colsample_bytree=.8)
    return {'version':'2026A_Q1_v1','selection':'macro_NRMSE_SD + 0.2 * worst_20pct_mean_NRMSE_SD',
            'focus_families':3,'seed':2601731,'outer_folds':5,'inner_folds':3,
            'exploration_disclosure':'Earlier seed-2026 CV informed candidate families; this cohort is not an independent external test.',
            'grade_status':'relative_only_not_original_0_6; two alternatives, participant decision pending',
            'candidates':records}


if __name__=='__main__':
    dest=Path(__file__).with_name('protocol.json')
    if dest.exists(): raise FileExistsError(dest)
    dest.write_text(json.dumps(grid(),ensure_ascii=False,indent=2),encoding='utf-8')
    print(dest)
