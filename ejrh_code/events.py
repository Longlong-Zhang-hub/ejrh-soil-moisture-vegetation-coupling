# Copyright (c) 2026 Longqi Zhang

import numpy as np
import pandas as pd

import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups

PRODUCTS = {'SMCI': 'SMCI_month_mean_z', 'GLEAM': 'GLEAM_SMrz_month_z'}
B = 999
RNG = np.random.default_rng(20261107)

def mon(t):return None if pd.isna(t) else str(pd.Period('2001-01',freq='M')+int(t))

def segments(x):
 met=np.isfinite(x)&(x<=-1);out=[];i=0
 while i<len(x):
  if not met[i]:i+=1;continue
  j=i
  while j+1<len(x) and (met[j+1] or (j+2<len(x) and np.isfinite(x[j+1]) and not met[j+1] and met[j+2])):j+=1
  if met[i:j+1].sum()>=2:
   confirmed=j+2<len(x) and np.isfinite(x[j+1:j+3]).all() and not met[j+1:j+3].any()
   out.append(dict(i=i,j=j,q=j+2 if confirmed else None,n=int(met[i:j+1].sum()),retained=not met[max(0,i-3):i].any(),minimum=i+int(np.argmin(x[i:j+1]))))
  i=j+1
 return out

def known_drought_age(x):
    start=None;normal=0
    for t,v in enumerate(x):
        if not np.isfinite(v):start=None;normal=0
        elif v<=-1:
            if start is None:start=t
            normal=0
        elif start is not None:
            normal+=1
            if normal>=2:start=None;normal=0
    return 0 if start is None else len(x)-start

def global_episodes(y):
 out=[];i=0
 while i<len(y):
  if not np.isfinite(y[i]) or y[i]>-1:i+=1;continue
  s=i;q=None;gap=None
  for t in range(s+1,len(y)):
   if not np.isfinite(y[t]):gap=t;break
   if t>=s+2 and y[t-1]>-.25 and y[t]>-.25:q=t;break
  out.append(dict(s=s,q=q,gap=gap,known=bool(s>0 and np.isfinite(y[s-1]))))
  if q is not None:i=q+1
  elif gap is not None:i=gap+1
  else:break
 return out

def endpoint(y,s,H,theta):
 end=min(H,len(y)-1);q=None;gap=None
 for t in range(s+1,end+1):
  if not np.isfinite(y[t]):gap=t;break
  if t>=s+2 and y[t-1]>theta and y[t]>theta:q=t;break
 if q is not None:return dict(q=q,r=q-1,t=q-s,state='recovered',reason='',censor=None,last=q)
 last=(gap-1 if gap is not None else end)
 reason='internal_missing' if gap is not None else ('record_end' if H>=len(y) else 'administrative')
 return dict(q=None,r=None,t=last-s,state='no_followup' if last==s else 'right_censored',reason=reason,censor=last,last=last)

def bootweights(labels,B=B):
 codes,levels=pd.factorize(labels,sort=True);nb=len(levels);counts=RNG.multinomial(nb,np.ones(nb)/nb,size=B);return counts[:,codes],nb

def curve(d,competing=False,max_t=None,weights=None):
 t=d.t.to_numpy(int);ev=d.recovered.to_numpy(int);co=d.competing.to_numpy(int) if 'competing' in d else np.zeros(len(d),int)
 w=np.ones((1,len(d))) if weights is None else np.asarray(weights,dtype=float);T=int(t.max()) if max_t is None else max_t
 nr=np.stack([(w*(t>=k)).sum(1) for k in range(T+1)],axis=1);de=np.stack([(w*((t==k)&(ev==1))).sum(1) for k in range(T+1)],axis=1);dc=np.stack([(w*((t==k)&(co==1))).sum(1) for k in range(T+1)],axis=1)
 hz=np.divide(de+dc,nr,out=np.zeros_like(nr),where=nr>0);surv=np.cumprod(1-hz,axis=1);prev=np.concatenate([np.ones((len(w),1)),surv[:,:-1]],axis=1)
 F=np.cumsum(prev*np.divide(de,nr,out=np.zeros_like(nr),where=nr>0),axis=1) if competing else 1-surv
 return F,surv,nr,de,dc

def make_risk(p, eps, end, n_months):
 end=end[(end.recovery_threshold==-.25)&(end.followup_rule=='met_end_plus12')];groups={u:g.sort_values('time_index').reset_index(drop=True) for u,g in p.groupby('unit_id')};rows=[];expo=[]
 for e in eps.to_dict('records'):
  g=groups[e['unit_id']];L=e['L'];en=end[end.veg_episode_id==e['veg_episode_id']].iloc[0]
  for product,col in PRODUCTS.items():
   for ver in ['pooled','calendar_month']:
    low=g[product+'_low_'+ver].to_numpy();v=g[col].to_numpy();k=L-1;duration=0
    while k>=0 and np.isfinite(v[k]) and low[k]:duration+=1;k-=1
    truncated=k<0 or not np.isfinite(v[k]);exc='left_truncated' if truncated else ('no_evaluable_possible_confirmation' if en.t<2 else '')
    x=dict(veg_episode_id=e['veg_episode_id'],unit_id=e['unit_id'],block_id=e['block_id'],product=product,threshold_version=ver,landmark_month=mon(L),L=L,exposure_definition='consecutive_low_SM_through_Lminus1',exposure_start_month=mon(L-duration) if duration else None,exposure_end_month=mon(L-1),exposure_last_observed_month=mon(L-1),D_pre_or_lagged_value=duration,SM_previous=v[L-1] if L>0 else np.nan,truncation_flag=truncated,covariate_last_months=mon(L),SPEI_L=e['SPEI_L'],elapsed_drought_L=e['elapsed_drought_L'],initial_y=e['initial_y'],month=e['month'],zone=e['zone'],stratum=e['stratum'],entry_year=e['entry_year'],potential_complete12=L+12<n_months,exclusion_reason=exc)
    expo.append(x)
    if not exc:
     for elapsed in range(2,int(en.t)+1):
      t=L+elapsed;tb=str(elapsed) if elapsed<=4 else ('5-6' if elapsed<=6 else ('7-9' if elapsed<=9 else '10+'))
      rows.append(dict(**x,model_id='landmark_'+ver+'_'+product,riskset_id='paired_landmark',risk_start=mon(t-1),risk_stop=mon(t),outcome_month=mon(t),outcome_i=t,elapsed=elapsed,time_bin=tb,recovered_indicator=int(en.recovered==1 and elapsed==en.t),response_definition='second_consecutive_calendar_month_above_minus0p25',D_pre=duration))
 expo=pd.DataFrame(expo);risk=pd.DataFrame(rows)
 common=set.intersection(*[set(risk[(risk['product']==pr)&(risk.threshold_version=='pooled')].veg_episode_id) for pr in PRODUCTS]);risk['common_sample']=risk.veg_episode_id.isin(common)
 assert (risk.outcome_i>=risk.L+2).all();assert (pd.to_datetime(risk.exposure_last_observed_month)<pd.to_datetime(risk.landmark_month)).all()
 return risk,expo,groups

def fit_landmark(risk):
    """Fit one common-support product/scenario stratum from the landmark table."""
    formula = ("recovered_indicator ~ D_pre + SM_previous + SPEI_L + elapsed_drought_L + "
               "initial_y + C(month) + C(stratum) + C(zone) + C(time_bin)")
    y, x = dmatrices(formula, risk, return_type="dataframe")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Rank-deficient landmark model")
    fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=150)
    if not fit.converged:
        raise RuntimeError("Landmark model did not converge")
    block = pd.factorize(risk.loc[x.index, "block_id"])[0]
    year = risk.loc[x.index, "entry_year"].to_numpy()
    covariance = cov_cluster(fit, block)
    two_way, _, _ = cov_cluster_2groups(fit, block, year)
    return fit.params, covariance, two_way


def identify_recovery(spell, vegetation):
    """Classify meteorological events and construct primary/threshold endpoint rows per unit.

    Arrays cover the same consecutive calendar months. All times are zero-based
    offsets. Eligibility follows the same earliest-assignment rule as the registry.
    """
    spell, vegetation = np.asarray(spell, float), np.asarray(vegetation, float)
    if spell.ndim != 1 or spell.shape != vegetation.shape:
        raise ValueError("SPEI3 and vegetation must have matching monthly shapes")
    meteorological = segments(spell)
    veg = global_episodes(vegetation)
    assigned = set()
    registry, endpoints = [], []
    for index, met in enumerate(meteorological):
        if not met["retained"]:
            continue
        i, j = met["i"], met["j"]
        horizon = j + 12
        match = next((ep for ep in veg if ep["s"] <= horizon and
                      (ep["q"] is None or ep["q"] >= i)), None)
        start = None if match is None else match["s"]
        if match is None:
            status = "confirmed_never_suppressed" if horizon < len(vegetation) else "insufficient_observation"
        elif not match["known"]:
            status = "suppression_onset_unknown"
        elif start in assigned:
            status = "linked_unrecovered_episode"
        elif start < i:
            status = "preexisting_suppression"
        else:
            status = "new_suppression"
            assigned.add(start)
        next_i = meteorological[index + 1]["i"] if index + 1 < len(meteorological) else None
        registry.append(dict(met_i=i, met_j=j, met_q=met["q"], next_i=next_i,
                             origin_i=start, eligibility_class=status))
        if status != "new_suppression":
            continue
        for theta in (-0.5, -0.25, 0.0):
            for rule, limit in (("met_end_plus12", horizon), ("origin_plus12", start + 12)):
                stop = endpoint(vegetation, start, limit, theta)
                endpoints.append(dict(origin_i=start, met_i=i, next_i=next_i,
                                      recovery_threshold=theta, followup_rule=rule,
                                      t=stop["t"], recovered=int(stop["q"] is not None),
                                      censor_reason=stop["reason"], endpoint_state=stop["state"]))
    return pd.DataFrame(registry), pd.DataFrame(endpoints)


def before_recurrence(ends, drought_first=False):
    """Convert new drought into a competing event, with explicit tie ordering."""
    result = ends[ends.t.gt(0)].copy()
    nd = result.next_i - result.origin_i
    result = result[nd.isna() | nd.gt(0)].copy()
    nd = result.next_i - result.origin_i
    choose = nd.notna() & nd.le(result.t) & (nd.lt(result.t) | result.recovered.eq(0) | drought_first)
    result["competing"] = choose.astype(int)
    result.loc[choose, "t"] = nd[choose].astype(int)
    result.loc[choose, "recovered"] = 0
    return result
