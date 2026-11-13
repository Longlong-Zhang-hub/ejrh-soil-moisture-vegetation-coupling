# Copyright (c) 2026 Longqi Zhang

import numpy as np
import pandas as pd

def robust(p,col):
 out=np.full(len(p),np.nan);pars=[]
 for (u,m),idx in p.groupby(['analysis_unit_id','Month']).groups.items():
  idx=np.array(list(idx)); v=p.loc[idx,col].to_numpy(float);ref=v[(p.loc[idx,'Year'].to_numpy()>=2001)&(p.loc[idx,'Year'].to_numpy()<=2014)];ref=ref[np.isfinite(ref)]
  med=np.median(ref) if len(ref)>=5 else np.nan;s=1.4826*np.median(abs(ref-med)) if len(ref)>=5 else np.nan;fallback=False
  if len(ref)>=5 and (not np.isfinite(s) or s<=1e-8):s=np.std(ref,ddof=1);fallback=True
  if not np.isfinite(s) or s<=1e-8:s=np.nan
  out[idx]=(v-med)/s;pars.append(dict(unit_id=u,month=m,variable=col,reference_period='2001-2014',n_ref=len(ref),center=med,scale=s,sd_fallback=fallback,missing_reason='zero_or_insufficient_scale' if not np.isfinite(s) else ''))
 return out,pars

def detrend(p,col,end):
 out=np.full(len(p),np.nan);pars=[]
 for u,idx in p.groupby('analysis_unit_id').groups.items():
  idx=np.array(list(idx));t=p.loc[idx,'time_index'].to_numpy(float);v=p.loc[idx,col].to_numpy(float);ok=np.isfinite(v)&(p.loc[idx,'Year'].to_numpy()<=end)&(p.loc[idx,'Year'].to_numpy()>=2001)
  c=np.linalg.lstsq(np.column_stack([np.ones(ok.sum()),t[ok]]),v[ok],rcond=None)[0];out[idx]=v-c[0]-c[1]*t;pars.append(dict(unit_id=u,variable=col,period_end=end,intercept=c[0],slope_per_month=c[1]))
 return out,pars

def aggregate_frame(frame,mapping,variables,min_cells=5):
 """Input fields GridB_ID,year,month + variables. Values must already be at intended transform stage.
 Keeps fixed original identity/Area; returns full and stable. Does not learn transforms or change years.
 """
 assert not frame.duplicated(['GridB_ID','year','month']).any()
 d=frame.merge(mapping,on='GridB_ID',how='inner',validate='many_to_one');out=[]
 for name,sub in [('full',d),('stable',d[d.stable_flag])]:
  keys=['analysis_unit_id','year','month'];group=sub.groupby(keys,sort=False);base=group.size().rename('support_cell_count').to_frame()
  for v in variables:
   ok=np.isfinite(sub[v]);tmp=sub.assign(_num=np.where(ok,sub[v]*sub.Area,0),_den=np.where(ok,sub.Area,0),_n=ok.astype(int));z=tmp.groupby(keys).agg(_num=('_num','sum'),_den=('_den','sum'),_n=('_n','sum'));z=z.reindex(base.index);base[v]=(z._num/z._den).where(z._n>=min_cells)
  base=base.reset_index();base['support']=name;out.append(base)
 return pd.concat(out,ignore_index=True)

def monthly_lags(panel, variables, max_lag=12):
    """Require full, consecutive calendar rows; preserve gaps in values as NaN."""
    data = panel.sort_values(["analysis_unit_id", "ordinal"]).reset_index(drop=True).copy()
    if data.duplicated(["analysis_unit_id", "ordinal"]).any():
        raise ValueError("Duplicate unit-month keys")
    if not data.ordinal.eq(data.Year * 12 + data.Month - 1).all() or not data.Month.between(1, 12).all():
        raise ValueError("Invalid month or ordinal")
    if not data.groupby("analysis_unit_id").ordinal.diff().dropna().eq(1).all():
        raise ValueError("Materialize missing calendar months as rows before lagging")
    grouped = data.groupby("analysis_unit_id")
    lagged = {f"{var}_L{lag}": grouped[var].shift(lag)
              for var in variables for lag in range(max_lag + 1)}
    return pd.concat([data, pd.DataFrame(lagged, index=data.index)], axis=1)
