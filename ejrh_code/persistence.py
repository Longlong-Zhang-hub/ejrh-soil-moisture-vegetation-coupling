# Copyright (c) 2026 Longqi Zhang

import numpy as np
import pandas as pd

from .coupling import weights

def cube(p, u, dates, field):
    s=p.set_index(['analysis_unit_id','ordinal'])[field]
    return s.reindex(pd.MultiIndex.from_product([u,dates])).to_numpy().reshape(len(u),len(dates))

def time_pearson(a,tidx,source_idx,tw):
    shape=(len(a),13,len(tidx))
    x=np.broadcast_to(a[:,tidx][:,None,:],shape).reshape(-1,len(tidx))
    y=a[:,source_idx].reshape(x.shape)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    xc=x-x.mean(1,keepdims=True)
    yc=y-y.mean(1,keepdims=True)
    pt=(xc*yc).sum(1)/np.sqrt((xc*xc).sum(1)*(yc*yc).sum(1))
    w=tw.T.astype(float)
    n=w.sum(0)
    sx=x@w; sy=y@w
    varx=(x*x)@w-sx*sx/n
    vary=(y*y)@w-sy*sy/n
    den=np.sqrt(np.maximum(varx*vary,0))
    cor=np.divide((x*y)@w-sx*sy/n,den,out=np.full_like(den,np.nan),where=den>1e-12)
    r=cor.T.reshape(len(tw),len(a),13)
    p=pt.reshape(len(a),13)
    p[:,0]=1
    r[:,:,0]=1
    assert np.isfinite(p).all()
    return p,r

def z(r):
    return np.arctanh(np.clip(r,-1+1e-10,1-1e-10))

def aggregate(p,d,area,sw,bidx,tw_n):
    w=sw[:,bidx]*area
    distinct=(sw[:,np.unique(bidx)]>0).sum(1)
    eligible=(distinct>=3)&(tw_n>=10)
    den=w.sum(1)
    lag_good=(np.isfinite(d)|(w[:,:,None]==0)).all(1)
    numer=np.einsum('ru,rul->rl',w,z(np.nan_to_num(d[:,:,1:])))
    with np.errstate(invalid='ignore',divide='ignore'):
        boot=np.column_stack([np.ones(len(sw)),np.tanh(numer/den[:,None])])
    boot[~(eligible[:,None]&lag_good)]=np.nan
    est=np.r_[1,np.tanh(np.average(z(p[:,1:]),axis=0,weights=area))]
    return est,boot,distinct

def calendar_persistence(panel, registry, variables, end_year=2014, replicates=999):
    """Pair May–September targets to actual antecedent months on fixed support."""
    units = registry.sort_values("analysis_unit_id").set_index("analysis_unit_id")
    dates = np.arange(2000 * 12, end_year * 12 + 12)
    tidx = np.flatnonzero((dates >= 2001 * 12) & np.isin(dates % 12 + 1, range(5, 10)))
    source = tidx[None, :] - np.arange(13)[:, None]
    time = dates[tidx]
    population = pd.DataFrame({
        "spatial_block_id": np.repeat(units.spatial_block_id.to_numpy(), len(time)),
        "ordinal": np.tile(time, len(units))
    })
    sw, tw, bi, ti = weights(population, replicates=replicates)
    time_counts = tw[:, ti[:len(time)]]
    block_index = bi.reshape(len(units), len(time))[:, 0]
    result = {}
    for variable in variables:
        matrix = cube(panel, units.index, dates, variable)
        result[variable] = time_pearson(matrix, tidx, source, time_counts)
    return result, units, sw, block_index, time_counts.sum(axis=1)


def regional_fisher(point, draws, area, block_counts, block_index):
    """Area-weighted Fisher-z pooling and paired regional refits from branch A."""
    sampled_area = block_counts[:, block_index] * area
    estimate = np.r_[1, np.tanh(np.average(z(point[:, 1:]), axis=0, weights=area))]
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.tanh(np.einsum("ru,rul->rl", sampled_area, z(draws[:, :, 1:]))
                         / sampled_area.sum(axis=1)[:, None])
    return estimate, np.column_stack([np.ones(len(draws)), pooled])
