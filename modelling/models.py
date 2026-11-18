"""所有数据依赖的变换都封装在 fit 内。"""
import warnings
import numpy as np
from scipy.special import ndtri


class FoldTransform:
    def __init__(self, config, groups, verified_log):
        self.config, self.groups, self.verified_log = config, np.asarray(groups), np.asarray(verified_log)

    def _nonlinear(self,x):
        out=x.copy()
        if self.config.get('shape')=='signed_log':
            mask=~self.verified_log
            out[:,mask]=np.sign(out[:,mask])*np.log1p(np.abs(out[:,mask]))
        return out

    def fit(self,x,y=None):
        x=self._nonlinear(np.asarray(x,dtype=float))
        self.p=x.shape[1]
        clip=self.config.get('clip',0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            self.low=np.nanquantile(x,clip,axis=0) if clip else np.full(self.p,-np.inf)
            self.high=np.nanquantile(x,1-clip,axis=0) if clip else np.full(self.p,np.inf)
            x=np.clip(x,self.low,self.high)
            self.median=np.nanmedian(x,axis=0)
        self.median=np.where(np.isfinite(self.median),self.median,0.)
        x=np.where(np.isnan(x),self.median,x)
        self.ranks=[np.sort(x[:,j]) for j in range(self.p)]
        if self.config.get('shape')=='rank': x=self._rank(x)
        if self.config.get('scale','standard')=='robust':
            self.center=np.median(x,axis=0)
            self.scale=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)
        else:
            self.center=x.mean(axis=0); self.scale=x.std(axis=0)
        self.scale=np.where(self.scale>1e-12,self.scale,1.)
        self.keep=np.arange(self.p)
        top=self.config.get('select',0)
        if top and y is not None and top<self.p:
            xc=(x-x.mean(axis=0))/(x.std(axis=0)+1e-12)
            yc=(y-y.mean(axis=0))/(y.std(axis=0)+1e-12)
            importance=np.mean(np.abs(xc.T.dot(yc)/len(x)),axis=1)
            self.keep=np.sort(np.argsort(-importance,kind='mergesort')[:top])
        self.balance=np.ones(self.p)
        if self.config.get('group_balance'):
            for group in np.unique(self.groups):
                selected=(self.groups==group)&np.isin(np.arange(self.p),self.keep)
                self.balance[selected]=1/np.sqrt(max(int(selected.sum()),1))
        return self

    def _rank(self,x):
        out=np.empty_like(x)
        for j, reference in enumerate(self.ranks):
            # 平均秩和有界 CDF 都只取训练样本，外部极端值不会影响边界。
            rank=(np.searchsorted(reference,x[:,j],side='left')+np.searchsorted(reference,x[:,j],side='right'))/2
            out[:,j]=ndtri(np.clip((rank+.5)/(len(reference)+1),.5/(len(reference)+1),1-.5/(len(reference)+1)))
        return out

    def transform(self,x):
        x=np.asarray(x,dtype=float)
        if x.shape[1]!=self.p: raise ValueError('特征维数不一致')
        missing=np.isnan(x)
        x=np.clip(self._nonlinear(x),self.low,self.high)
        x=np.where(np.isnan(x),self.median,x)
        if self.config.get('shape')=='rank': x=self._rank(x)
        x=((x-self.center)/self.scale*self.balance)[:,self.keep]
        if self.config.get('indicators',True): x=np.column_stack([x,missing[:,self.keep].astype(float)])
        if not np.isfinite(x).all(): raise ValueError('变换后出现无穷或 NaN')
        return x


class SensoryRegressor:
    def __init__(self,spec,groups,verified_log,seed=2026,threads=4,device='CPU'):
        self.spec,self.groups,self.verified_log=spec,groups,verified_log
        self.seed,self.threads,self.device=seed,threads,device

    def fit(self,x,y):
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import Ridge,ElasticNet
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.multioutput import MultiOutputRegressor
        self.transformer=FoldTransform(self.spec['pre'],self.groups,self.verified_log).fit(x,y)
        z=self.transformer.transform(x)
        self.mean=y.mean(axis=0); self.scale=y.std(axis=0)
        self.scale=np.where(self.scale>1e-10,self.scale,1.)
        target=(y-self.mean)/self.scale
        family=self.spec['family']; params=dict(self.spec['params'])
        if family=='mean': self.model=DummyRegressor(strategy='mean')
        elif family=='ridge': self.model=Ridge(solver='svd',**params)
        elif family=='elastic':
            self.model=MultiOutputRegressor(ElasticNet(max_iter=4000,tol=1e-4,random_state=self.seed,**params),n_jobs=1)
        elif family=='pls':
            params['n_components']=min(params['n_components'],len(z)-1,z.shape[1])
            self.model=PLSRegression(scale=False,max_iter=1000,**params)
        elif family=='krr':
            lengths=(z*z).sum(axis=1)
            squared=np.maximum(lengths[:,None]+lengths[None,:]-2*z.dot(z.T),0)
            distances=squared[np.triu_indices(len(z),k=1)]
            bandwidth=np.median(distances[distances>1e-12]) if (distances>1e-12).any() else 1.
            self.gamma=params.pop('gamma_factor')/bandwidth
            self.model=KernelRidge(kernel='rbf',gamma=self.gamma,**params)
        elif family=='extra':
            self.model=ExtraTreesRegressor(random_state=self.seed,n_jobs=self.threads,**params)
        elif family=='catboost':
            from catboost import CatBoostRegressor
            self.model=CatBoostRegressor(loss_function='MultiRMSE',random_seed=self.seed,
                thread_count=self.threads,task_type=self.device,allow_writing_files=False,verbose=False,**params)
        elif family=='lightgbm':
            from lightgbm import LGBMRegressor
            self.model=MultiOutputRegressor(LGBMRegressor(objective='regression',random_state=self.seed,
                n_jobs=self.threads,verbosity=-1,deterministic=True,force_col_wise=True,**params),n_jobs=1)
        else: raise ValueError('未知模型家族 '+family)
        self.model.fit(z,target)
        return self

    def predict(self,x):
        out=np.asarray(self.model.predict(self.transformer.transform(x)))
        if out.ndim==1: out=out[:,None]
        out=out*self.scale+self.mean
        if not np.isfinite(out).all(): raise ValueError('模型产生无效预测')
        return out


def relative_grade(values,train_y,method='quantile'):
    """0–6 仅表示训练参考分布中的相对等级，不是原评分。"""
    if method=='quantile':
        edges=np.quantile(train_y,np.arange(1,7)/7,axis=0)
    elif method=='robust':
        med=np.median(train_y,axis=0)
        spread=np.quantile(train_y,.75,axis=0)-np.quantile(train_y,.25,axis=0)
        edges=med+np.array([-1.5,-1.,-.5,.5,1.,1.5])[:,None]*spread
    else: raise ValueError(method)
    grades=np.sum(values[:,None,:]>edges[None,:,:],axis=1).astype(int)
    degenerate=np.any(np.diff(edges,axis=0)<=0,axis=0)
    # 不可分辨的量尺不伪造等级：-1 表示映射未定义。
    grades[:,degenerate]=-1
    return grades,edges,degenerate
