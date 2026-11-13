# Copyright (c) 2026 Longqi Zhang

import numpy as np
import pandas as pd

ZONE = ['Z1','Z2','Z3','Z4','Z5']
GROUPS = ['WOODY','HERBACEOUS','SPARSE']

def bounds(draw):
    x=np.asarray(draw)
    ok=np.isfinite(x)
    return (*(np.quantile(x[ok],[.025,.975]) if ok.sum()>=950 else (np.nan,np.nan)),int(ok.sum()))

def max_family(point, draw):
    draw=np.asarray(draw)
    ok=np.isfinite(draw).all(axis=1)
    if ok.sum()<950:
        return np.nan,np.full(len(point),np.nan),np.full(len(point),np.nan),np.full(len(point),np.nan),int(ok.sum())
    b=draw[ok]
    scale=np.std(b,axis=0,ddof=1)
    nonzero=scale>1e-13
    if not nonzero.any():
        return 1.,point.copy(),point.copy(),np.ones(len(point)),int(ok.sum())
    null=np.max(np.abs((b[:,nonzero]-point[nonzero])/scale[nonzero]),axis=1)
    observed=np.divide(abs(point),scale,out=np.zeros_like(point),where=nonzero)
    critical=np.quantile(null,.95)
    p=(1+(null[:,None]>=observed[None,:]).sum(0))/(len(null)+1)
    return (1+(null>=observed.max()).sum())/(len(null)+1),point-critical*scale,point+critical*scale,p,int(ok.sum())

def holm(v):
    v=np.asarray(v)
    p=np.full(len(v),np.nan)
    order=np.argsort(v)
    good=order[np.isfinite(v[order])]
    p[good]=np.minimum(1,np.maximum.accumulate(v[good]*(len(v)-np.arange(len(good)))))
    return p

def scaled_condition(x):
    scale=np.sqrt(np.mean(x*x,axis=0))
    if np.any(scale==0):
        return np.inf
    return np.linalg.cond(x/scale)

def fit_boot(x,y,wb,wt):
    """Weighted refits. Cross the *same* spatial and calendar multiplicities."""
    scale=np.sqrt(np.mean(x*x,axis=0))
    z=x/scale
    beta=np.linalg.lstsq(z,y,rcond=None)[0]/scale
    draws=np.full((len(wb),x.shape[1]),np.nan)
    failures=[]
    for i in range(len(draws)):
        w=wb[i]*wt[i]
        gram=z.T@(z*w[:,None])
        try:
            if np.linalg.matrix_rank(gram)!=gram.shape[0]:
                raise np.linalg.LinAlgError('weighted rank deficiency')
            draws[i]=np.linalg.solve(gram,z.T@(w*y))/scale
        except np.linalg.LinAlgError as e:
            failures.append((i,str(e)))
    return beta,draws,failures

def projection(names,x,base_h,d,prod,response,mode):
    n_extra=4*len(mode)
    names=names.copy()
    parts=[x]
    ix={}
    for lag in mode:
        for zone in ZONE[1:]:
            ix[zone,lag]=len(names)
            names.append(f'{zone}:L{lag}')
            parts.append((d[f'{prod}_L{lag}'].to_numpy()*(d.Zone.to_numpy()==zone))[:,None])
    new_x=np.column_stack(parts)
    maps={}
    for zone in ZONE:
        counts=d.loc[d.Zone.eq(zone),['analysis_unit_id','functional_group']].drop_duplicates().functional_group.value_counts(normalize=True)
        base=sum(counts.get(g,0)*base_h[g] for g in GROUPS if g in base_h) if response=='Y' else base_h['REGION']
        full=np.pad(base,((0,0),(0,n_extra)))
        off=np.zeros_like(full)
        for lag in mode:
            if zone!='Z1':
                full[lag,ix[zone,lag]]=1
                off[lag,ix[zone,lag]]=1
        maps[zone]=(full,off)
    return new_x,names,maps

def vif(x,j):
    y=x[:,j]
    z=np.delete(x,j,axis=1)
    r=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
    return (np.sum((y-y.mean())**2)/np.sum(r*r)) if np.sum(r*r)>1e-15 else np.inf

def regression(X,y,draws,weights):
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    scale=np.sqrt(np.mean(X*X,axis=0))
    z=X/scale
    out=np.full((len(weights),X.shape[1],y.shape[1]),np.nan)
    issues=[]
    for i,w in enumerate(weights):
        if not np.isfinite(draws[i,w>0]).all():
            issues.append((i,'nonfinite bootstrap outcome among sampled units'))
            continue
        gram=z.T@(z*w[:,None])
        if np.linalg.matrix_rank(gram)!=len(gram):
            issues.append((i,'rank deficient'))
            continue
        out[i]=np.linalg.solve(gram,z.T@(np.nan_to_num(draws[i])*w[:,None]))/scale[:,None]
    return beta,out,issues

def environment_design(units):
    """Historical climate-only primary design with reference-coded cover and zone."""
    climate = (units.D_mm_season - units.D_mm_season.mean()).to_numpy() / 100
    cover = pd.get_dummies(units.functional_group).reindex(columns=['WOODY', 'SPARSE'], fill_value=0)
    zone = pd.get_dummies(units.Zone).reindex(columns=ZONE[1:], fill_value=0)
    return np.column_stack([np.ones(len(units)), climate, cover.to_numpy(float), zone.to_numpy(float)])
