# Copyright (c) 2026 Longqi Zhang

import numpy as np
import pandas as pd

from pathlib import Path
import hashlib
import json
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

GROUPS = ['WOODY', 'HERBACEOUS', 'SPARSE']
RUN = 'historical'
SEED = 20261107
NBOOT = 999

def csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')

def js(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')

def weights(d,replicates=NBOOT,blockmonths=24):
 # Paired whole spatial blocks and synchronous non-circular calendar blocks.
 rng=np.random.default_rng(SEED);blocks,bi=np.unique(d.spatial_block_id.astype(str),return_inverse=True);lo=int(d.ordinal.min());hi=int(d.ordinal.max());nt=hi-lo+1;ti=d.ordinal.to_numpy()-lo
 sw=rng.multinomial(len(blocks),np.ones(len(blocks))/len(blocks),size=replicates)
 tw=np.zeros((replicates,nt),dtype='int16');nb=int(np.ceil(nt/blockmonths))
 for r in range(replicates):
  starts=rng.integers(0,max(nt-blockmonths+1,1),nb);idx=np.concatenate([np.arange(s,min(s+blockmonths,nt)) for s in starts])[:nt];tw[r]=np.bincount(idx,minlength=nt)
 return sw,tw,bi,ti

def residual_check(d,res):
 q=d[['analysis_unit_id','functional_group','ordinal']].copy();q['res']=res;rows=[]
 for group in ['REGION']+GROUPS:
  x=q if group=='REGION' else q[q.functional_group==group]
  for lag in [1,2,3]:
   prev=x[['analysis_unit_id','ordinal','res']].copy();prev.ordinal+=lag;z=x.merge(prev,on=['analysis_unit_id','ordinal'],suffixes=('','_prev'));r=np.corrcoef(z.res,z.res_prev)[0,1] if len(z)>5 else np.nan
   rows.append(dict(stratum=group,lag=lag,n_pairs=len(z),correlation=r,kind='within_unit_actual_month'))
 s=q.groupby('ordinal').res.mean().reset_index()
 for lag in [1,2,3]:
  prev=s.copy();prev.ordinal+=lag;z=s.merge(prev,on='ordinal',suffixes=('','_prev'));r=np.corrcoef(z.res,z.res_prev)[0,1] if len(z)>5 else np.nan;rows.append(dict(stratum='REGION',lag=lag,n_pairs=len(z),correlation=r,kind='synchronous_region_mean'))
 return rows

def lag_design(d,driver,L,basis):
 a=d[[f'{driver}_L{i}' for i in range(L+1)]].to_numpy(float)
 B=np.eye(L+1) if basis=='U' else np.polynomial.legendre.legvander(np.linspace(-1,1,L+1),2 if basis=='Q' else 3)
 return a@B,B

def design(d,driver,response,L,basis):
 a,B=lag_design(d,driver,L,basis);parts=[];names=[];hs={};inter=response!='FVC'
 for g in (GROUPS if inter else ['COMMON']):
  z=a*(d.functional_group.to_numpy()==g)[:,None] if inter else a;off=len(names);parts.append(z);names += [f'{driver}_{basis}{j}_{g}' for j in range(a.shape[1])];hs[g]=(off,B)
 if response in ['Y','YnonSTL','FVC']:
  ctl,_=lag_design(d,'SPEI3',6,'Q');ctl=np.column_stack([ctl,d.PET_z])
  for g in (GROUPS if inter else ['COMMON']):
   parts.append(ctl*(d.functional_group.to_numpy()==g)[:,None] if inter else ctl);names += [f'climate{j}_{g}' for j in range(4)]
 parts.append(np.ones((len(d),1)));names+=['Intercept']
 for col,cats in [('functional_group',GROUPS),('Zone',sorted(d.Zone.unique())),('Month',sorted(d.Month.unique()))]:
  for cat in cats[1:]:parts.append((d[col].to_numpy()==cat)[:,None].astype(float));names.append(f'{col}_{cat}')
 X=np.column_stack(parts);H={}
 for g,(off,B) in hs.items():
  h=np.zeros((L+1,X.shape[1]));h[:,off:off+B.shape[1]]=B;H[g]=h
 if inter:
  w=np.array([1/3]*3) if response in ['SMCI','GLEAM'] else np.array([17,111,15])/143
  H['REGION']=sum(wi*H[g] for wi,g in zip(w,GROUPS))
 else:H['REGION']=H['COMMON']
 return X,names,H

def curve_summary(beta,draws):
 L=len(beta)-1;mass=np.abs(beta).sum();cent=np.dot(np.arange(L+1),abs(beta))/mass if mass>1e-12 else np.nan
 cum=draws.sum(1);dm=np.abs(draws).sum(1);dc=np.divide(np.abs(draws)@np.arange(L+1),dm,out=np.full(len(draws),np.nan),where=dm>1e-12);peak=np.argmax(abs(draws),axis=1);lo,hi=np.quantile(draws,[.025,.975],axis=0);tail=beta[3:];v=np.cov(draws[:,3:],rowvar=False);v=np.atleast_2d(v);rank=np.linalg.matrix_rank(v);W=float(tail@np.linalg.pinv(v)@tail) if rank else np.nan
 return dict(cumulative_net=float(beta.sum()),cumulative_ci_low=float(np.quantile(cum,.025)),cumulative_ci_high=float(np.quantile(cum,.975)),abs_coefficient_mass=float(mass),centroid_lag=float(cent),centroid_ci_low=float(np.nanquantile(dc,.025)),centroid_ci_high=float(np.nanquantile(dc,.975)),centroid_identifiability_status='DESCRIPTIVE_RATIO_ONLY_NO_DURATION' if not(np.all(lo<=0)&np.all(hi>=0)) else 'UNSTABLE_WHOLE_CURVE_ZERO_COMPATIBLE',peak_lag=int(np.argmax(abs(beta))),tail_start_lag=3,tail_net=float(tail.sum()),tail_absolute_fraction=float(abs(tail).sum()/mass),tail_joint_wald=W,tail_joint_df=rank,tail_joint_p=float(chi2.sf(W,rank)) if rank else np.nan,tail_equivalence_status='NOT_TESTED_NO_MARGIN'),lo,hi,peak

def run_dlm(p,output,sid,driver,response,product,L=6,basis='Q',months=None,legacy=False,startyear=2001,mask="original2020"):
 if not sid or Path(sid).name != sid or sid in ('.', '..'):
  raise ValueError('Specify a simple output identifier')
 out=Path(output)/sid;out.mkdir(exist_ok=False)
 # Main new comparisons share longest available lag vectors and both products.
 supportL=6 if legacy else 12
 req=[f'{v}_L{i}' for v in ['SMCI','GLEAM','SPEI3','SPEI6','D'] for i in range(supportL+1)]+[response,'PET_z']
 d=p[(p.Year.between(startyear,2014))&p.Month.isin(months or [5,6,7,8,9])].dropna(subset=req).copy()
 X,names,H=design(d,driver,response,L,basis);y=d[response].to_numpy(float);fit=sm.OLS(y,X).fit();beta=np.asarray(fit.params);res=y-X@beta
 bi=pd.factorize(d.spatial_block_id)[0];ti=pd.factorize(d.ordinal)[0];V,_,_=cov_cluster_2groups(fit,bi,ti,use_correction=True);np.savez_compressed(out/'fit.npz',coefficients=beta,covariance_two_way=V,X_names=np.array(names),**{'projection_'+g:h for g,h in H.items()})
 rd=residual_check(d,res);csv(out/'residual_diagnostics.csv',rd)
 trigger=(d.Month.nunique()==1) or any(abs(r['correlation'])>=(.2 if r['kind']=='synchronous_region_mean' else .1) for r in rd if np.isfinite(r['correlation']))
 # Dependence evidence licenses the prespecified full pair-vector bootstrap.
 if trigger:
  sw,tw,bidx,tidx=weights(d);draws=[];fail=[];scale=np.sqrt(np.mean(X*X,axis=0));scale[scale==0]=1;Z=X/scale
  for rep in range(NBOOT):
   w=sw[rep,bidx]*tw[rep,tidx];A=Z.T@(Z*w[:,None]);rhs=Z.T@(y*w)
   try:
    rank=np.linalg.matrix_rank(A)
    if rank<A.shape[0]:raise ValueError(f'rank {rank}/{len(A)}')
    draws.append(np.linalg.solve(A,rhs)/scale)
   except Exception as e:fail.append(dict(rep=rep,error=str(e)));draws.append(np.full(len(beta),np.nan))
  draws=np.array(draws);csv(out/'bootstrap_failures.csv',pd.DataFrame(fail,columns=['rep','error']));method='PAIRED_BLOCK_TIME24_OLS_REFIT';valid=np.isfinite(draws).all(1)
  if valid.sum()<999:raise RuntimeError(f'{sid}: {valid.sum()} valid bootstrap <999; diagnose before reporting')
 else:
  eig,U=np.linalg.eigh((V+V.T)/2)
  if eig.min()<-1e-9:raise RuntimeError(f'{sid} indefinite covariance requires bootstrap')
  rng=np.random.default_rng(SEED);draws=beta+rng.standard_normal((100000,len(beta)))@(U*np.sqrt(np.maximum(eig,0))).T;method='TWO_WAY_CR1_JOINT_NORMAL';valid=np.ones(len(draws),bool)
 np.savez_compressed(out/'coefficient_draws.npz',draws=draws)
 pred=d[['analysis_unit_id','spatial_block_id','Zone','functional_group','Year','Month','ordinal']].copy();pred['observed']=y;pred['fitted']=X@beta;pred['residual']=res;pred.to_parquet(out/'fitted_rows.parquet',index=False)
 support_id=hashlib.sha256(pd.util.hash_pandas_object(pred[['analysis_unit_id','ordinal']],index=False).values.tobytes()).hexdigest()[:16]
 rows=[];reg={}
 for g,h in H.items():
  b=h@beta;dd=draws@h.T;s,lo,hi,pk=curve_summary(b,dd)
  if g=='REGION':reg=s
  for lag in range(L+1):rows.append(dict(spec_id=sid,product=product,pathway=f'{driver}->{response}',stratum_or_region=g,lag_months=lag,coefficient=b[lag],ci_low=lo[lag],ci_high=hi[lag],covariance_or_resample_id=sid+'_'+method,aggregation_weight_id='equal_strata' if response in ['SMCI','GLEAM'] else ('unit_shares' if response!='FVC' else 'pooled_common_slope'),coefficient_scale='robust_z response per predictor scale; SPEI original z',simultaneous_or_marginal_interval='marginal_percentile' if trigger else 'marginal_joint_normal',peak_probability=float(np.mean(pk==lag)),n_valid_draws=len(draws),common_support_id=support_id))
 csv(out/'lag_coefficients.csv',rows)
 # Preserve two-way analogue intervals at the matching OLS center separately.
 for g,h in H.items():
  vv=h@V@h.T;z=h@beta;se=np.sqrt(np.maximum(np.diag(vv),0));csv(out/f'two_way_{g}.csv',pd.DataFrame(dict(lag=range(L+1),coefficient=z,ci_low=z-1.96*se,ci_high=z+1.96*se,variance_diagonal=np.diag(vv))))
 summary=dict(run_id=RUN,input_version='original_plus_recovered2000' if not legacy else 'original_2001_2022',card_id='Q-A04' if response in ['SMCI','GLEAM'] else 'Q-A05',spec_id=sid,parent_spec_id='',product=product,response_product=response,pathway=f'{driver}->{response}',driver=driver,basis_type=basis,basis_degree=0 if basis=='U' else (2 if basis=='Q' else 3),lag_min=0,lag_max=L,response_transform_id='A_nonSTL_seasonal_linear' if response=='YnonSTL' else 'A_original_robust',control_terms_id='stratum month zone FE; stratum-specific SPEI3 Q6 and PET for kNDVI; common for FVC',period_id=f'{startyear}-2014',target_months=months or [5,6,7,8,9],mask_version=mask,estimator='OLS matched fixed-effect analogue',random_structure_id='none',aggregation_weight_id=rows[0]['aggregation_weight_id'],common_support_id=support_id,n_obs=len(d),n_units=d.analysis_unit_id.nunique(),n_blocks=d.spatial_block_id.nunique(),n_years=d.Year.nunique(),rank=int(np.linalg.matrix_rank(X)),n_parameters=X.shape[1],standardized_condition_number=float(np.linalg.cond(X/np.sqrt(np.mean(X*X,axis=0)))),convergence_status='SOLVED',inference_method_id=method,n_requested_replicates=len(draws),n_valid_replicates=len(draws),n_failed_replicates=0,covariance_two_way_min_eigenvalue=float(np.linalg.eigvalsh(V).min()),rmse_in_sample=float(np.sqrt(np.mean(res**2))),implementation_note='All pairs built on original calendar; no predictions/out-of-sample claim; own OLS center',**reg)
 js(out/'summary.json',summary);print('FIT',sid,summary['n_obs'],summary['centroid_lag'],method,flush=True)
 return summary

def conditional(p,output,product,model,free=False):
 if product not in ('SMCI', 'GLEAM') or model not in ('F0', 'F1', 'F2', 'F3'):
  raise ValueError('Unknown product or conditional-state specification')
 sid=f'CONDITIONAL_{product}_{model}'+('_Dfree' if free else '');out=Path(output)/sid;out.mkdir(exist_ok=False)
 req=[f'{v}_L{i}' for v in ['SMCI','GLEAM','D'] for i in range(13)]+['PET_z'];d=p[p.Year.between(2001,2014)&p.Month.isin(range(5,10))].dropna(subset=req).copy();cols=[d.D.to_numpy(),d.PET_z.to_numpy()];names=['D_current','PET_current']
 if model!='F0':
  a=d[[f'D_L{i}' for i in range(1,7)]].to_numpy();B=np.eye(6) if free else np.polynomial.legendre.legvander(np.linspace(-1,1,6),2);z=a@B
  for j in range(z.shape[1]):cols.append(z[:,j]);names.append(f'D_history_{j}')
 if model in ['F2','F3']:
  for lag in ([1,2,3] if model=='F3' else [1]):
   for g in GROUPS:cols.append(d[f'{product}_L{lag}'].to_numpy()*(d.functional_group.to_numpy()==g));names.append(f'{product}_lag{lag}_{g}')
 for mo in range(6,10):cols.append((d.Month.to_numpy()==mo).astype(float));names.append(f'month_{mo}')
 Z=np.column_stack(cols);y=d[product].to_numpy();u,ui=np.unique(d.analysis_unit_id,return_inverse=True);nu=len(u);K=Z.shape[1]
 def solve(w):
  count=np.bincount(ui,weights=w,minlength=nu);sZ=np.zeros((nu,K));sy=np.bincount(ui,weights=w*y,minlength=nu);np.add.at(sZ,ui,Z*w[:,None]);inv=np.divide(1,count,out=np.zeros(nu),where=count>0);A=Z.T@(Z*w[:,None])-sZ.T@(sZ*inv[:,None]);rhs=Z.T@(y*w)-sZ.T@(sy*inv);be=np.linalg.solve(A,rhs);alpha=(sy-sZ@be)*inv;return be,alpha
 beta,alpha=solve(np.ones(len(y)));res=y-Z@beta-alpha[ui];rd=residual_check(d,res);csv(out/'residual_diagnostics.csv',rd);sw,tw,bi,ti=weights(d);draws=[]
 for rep in range(999):
  w=sw[rep,bi]*tw[rep,ti];be,_=solve(w);draws.append(be)
 draws=np.array(draws);lo,hi=np.quantile(draws,[.025,.975],axis=0);np.savez_compressed(out/'coefficients_and_draws.npz',beta=beta,draws=draws,names=np.array(names));rows=[]
 rmse=float(np.sqrt(np.mean(res**2)));trigger=any(r['kind']=='within_unit_actual_month' and r['lag'] in [2,3] and abs(r['correlation'])>=.1 for r in rd)
 for j,name in enumerate(names):rows.append(dict(run_id=RUN,input_version='original_plus2000',spec_id=sid,parent_spec_id='',card_id='Q-A03',product=product,stratum=next((g for g in GROUPS if name.endswith(g)),'REGION'),response_transform_id='A_SM_robust2001_2014',forcing_terms='D0 PET0 '+('D1-6 free' if free else ('D1-6 quadratic' if model!='F0' else '')),soil_lag_terms='1,2,3 by stratum' if model=='F3' else ('1 by stratum' if model=='F2' else 'none'),estimator='OLS unit FE month FE; weighted within-unit refit bootstrap',n_obs=len(d),n_units=nu,n_blocks=d.spatial_block_id.nunique(),coefficient_term=name,estimate=beta[j],ci_low=lo[j],ci_high=hi[j],covariance_or_resample_id='paired_spaceblock_time24_refit999',residual_diagnostic_id=str(out/'residual_diagnostics.csv'),fit_metric_name='in_sample_RMSE',fit_metric_value=rmse,metric_is_out_of_sample=False,convergence_status='SOLVED',followup_trigger=trigger))
 csv(out/'coefficients.csv',rows);d=d[['analysis_unit_id','spatial_block_id','functional_group','Zone','Year','Month','ordinal']].copy();d['observed']=y;d['fitted']=y-res;d['residual']=res;d.to_parquet(out/'fitted_rows.parquet',index=False)
 s=dict(spec_id=sid,product=product,model=model,free_forcing=free,n_obs=len(d),n_units=nu,n_blocks=d.spatial_block_id.nunique(),rmse_in_sample=rmse,residual_lag2_3_trigger=trigger,condition_number=float(np.linalg.cond(Z)),estimator='unit_FE_OLS',n_bootstrap=999);js(out/'summary.json',s);print('CONDITIONAL',s,flush=True);return s
