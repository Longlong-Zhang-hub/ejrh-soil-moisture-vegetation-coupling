# Copyright (c) 2026 Longqi Zhang

import numpy as np
import pandas as pd

import hashlib
import json
from pathlib import Path
from scipy.optimize import minimize, minimize_scalar
from patsy import dmatrices, build_design_matrices

GROUPS = ['WOODY', 'HERBACEOUS', 'SPARSE']
BASE = ['spei_B0', 'spei_B1', 'spei_B2', 'pet_z']
MODELS = ['M0', 'M1_SMCI', 'M2_SMCI', 'M1_GLEAM', 'M2_GLEAM']
CONS = [('CURRENT','M0','M1_'), ('PAST_GIVEN_CURRENT','M1_','M2_'), ('TOTAL','M0','M2_')]
SEED = 20261107

def h(ids):return hashlib.sha256('\n'.join(sorted(map(str,ids))).encode()).hexdigest()

def tasks():
 out=[dict(task_id='TEMPORAL_2015_2022',kind='TEMPORAL',end=2014,start_eval=2015,end_eval=2022,fold=0)]
 out +=[dict(task_id=f'ROLL_{e+1}_{e+2}',kind='ROLLING',end=e,start_eval=e+1,end_eval=e+2,fold=0) for e in range(2010,2021,2)]
 for kind in ['SPATIAL_NEW','SPATIAL_BASELINE']:
  out +=[dict(task_id=f'{kind}_F{f:02d}',kind=kind,end=2022,start_eval=2001 if kind=='SPATIAL_NEW' else 2015,end_eval=2022,fold=f) for f in range(1,6)]
 return out

def group_transform(d,fitmask,variable,keys,baseline=False):
 out=np.full(len(d),np.nan);off=np.zeros(len(d));scaleout=np.ones(len(d));records=[];resids={};fits={}
 fit=d.loc[fitmask]
 for key,g in fit.groupby(keys,observed=True):
  key=key if isinstance(key,tuple) else (key,);ok=np.isfinite(g[variable]);gg=g.loc[ok];y=gg[variable].to_numpy();tt=gg.time_index.to_numpy()/12
  if baseline:
   X=np.column_stack([(gg.Month.to_numpy()==m).astype(float) for m in range(1,13)]+[tt]);coef=np.linalg.lstsq(X,y,rcond=None)[0];r=y-X@coef
  else:coef=np.zeros(13);r=y
  fits[key]=coef;resids[key]=(gg,r)
 pool={}
 for cover in GROUPS:
  for m in range(1,13):
   chunks=[r[(gg.functional_group.to_numpy()==cover)&(gg.Month.to_numpy()==m)] for gg,r in resids.values()];v=np.concatenate(chunks) if chunks else np.array([]);pool[(cover,m)]=np.std(v,ddof=1) if len(v)>1 else np.nan
 for key,idx in d.groupby(keys,observed=True).groups.items():
  key=key if isinstance(key,tuple) else (key,)
  if key not in fits:raise ValueError('Unsupported transform group '+str(key))
  g=d.loc[idx];coef=fits[key];gg,r=resids[key]
  for m in range(1,13):
   ix=g.index[g.Month==m];sel=gg.Month.to_numpy()==m;rv=r[sel];c=np.median(rv);s=1.4826*np.median(np.abs(rv-c));fallback='none'
   if not np.isfinite(s) or s<=1e-8:s=np.std(rv,ddof=1);fallback='training_group_SD'
   cover=g.functional_group.iloc[0]
   if not np.isfinite(s) or s<=1e-8:s=pool[(cover,m)];fallback='training_cover_month_pooled_SD'
   if not np.isfinite(s) or s<=1e-8:s=1.;fallback='constant_reference'
   gp=coef[m-1]+coef[-1]*d.loc[ix,'time_index'].to_numpy()/12 if baseline else np.zeros(len(ix))
   off[ix]=gp+c;scaleout[ix]=s;out[ix]=(d.loc[ix,variable].to_numpy()-off[ix])/s
   ids=gg.loc[sel,'row_id'];records.append(dict(variable=variable,group_key='|'.join(map(str,key)),month=m,center=c,scale=s,trend_coefficient=coef[-1] if baseline else 0.,month_coefficient=coef[m-1] if baseline else 0.,fallback_reason=fallback,fit_time_start=str(gg.date.min()),fit_time_end=str(gg.date.max()),fit_record_ids_hash=h(gg.row_id),scale_record_ids_hash=h(ids),fit_n=len(gg),scale_n=len(ids),applied_rows_hash=h(d.loc[ix,'row_id']),baseline_formula='month12 + time_year' if baseline else 'median/MAD'))
 return out,off,scaleout,records

def prepare(d0,climate,t):
 if d0.duplicated(['analysis_unit_id','date']).any():
  raise ValueError('Duplicate unit-month input')
 check=d0.sort_values(['analysis_unit_id','Year','Month']).copy()
 check['ordinal']=check.Year*12+check.Month-1
 if not check.Month.between(1,12).all() or not check.groupby('analysis_unit_id').ordinal.diff().dropna().eq(1).all():
  raise ValueError('Prediction preprocessing requires explicit consecutive calendar rows')
 d=d0.copy().sort_values(['analysis_unit_id','date']).reset_index(drop=True);sp=t['kind'].startswith('SPATIAL');tr=(d.spatial_fold!=t['fold']) if sp else (d.Year<=t['end']);ev=(d.spatial_fold==t['fold']) if sp else (d.Year>=t['start_eval']);ev=ev&(d.Year>=t['start_eval'])&(d.Year<=t['end_eval']);train_y=tr.copy();keys=['analysis_unit_id']
 if t['kind']=='SPATIAL_NEW':keys=['functional_group']
 elif t['kind']=='SPATIAL_BASELINE':train_y=d.Year<=2014
 y,o,s,recs=group_transform(d,train_y,'y_raw',keys,True);d['y_frozen']=y;d['y_offset']=o;d['y_scale']=s
 for r in recs:r.update(fit_role='local_calibration' if t['kind']=='SPATIAL_BASELINE' else 'training',fit_spatial_support='all_units_local_2001_2014' if t['kind']=='SPATIAL_BASELINE' else h(d.loc[tr,'spatial_block_id'].unique()))
 for var,name in [('PET','pet_z'),('SMCI_month_mean','smci_z'),('GLEAM_SMrz_month','gleam_z')]:
  z,_,_,rr=group_transform(d,tr,var,['functional_group'] if sp else ['analysis_unit_id'],False);d[name]=z
  for r in rr:r.update(fit_role='training',fit_spatial_support=h(d.loc[tr,'spatial_block_id'].unique()))
  recs+=rr
 ref=f"SPEI3_1960_{2000 if sp else t['end']}";cc=climate[climate.transform_id==ref][['analysis_unit_id','date','SPEI3_fold']];d=d.merge(cc,on=['analysis_unit_id','date'],how='left',validate='one_to_one',indicator=True)
 if not d['_merge'].eq('both').all():
  raise ValueError('The fold-specific SPEI3 table must cover every monthly row, including explicit missing values')
 d=d.drop(columns='_merge')
 for v,prefix in [('SPEI3_fold','spei'),('smci_z','smci'),('gleam_z','gleam')]:
  for lag in range(7):d[f'{prefix}_L{lag}']=d.groupby('analysis_unit_id',sort=False)[v].shift(lag)
  vals=d[[f'{prefix}_L{k}' for k in range(7)]].to_numpy()
  for j in range(3):d[f'{prefix}_B{j}']=vals@(np.arange(7,dtype=float)**j)
 d['Y_L1']=d.groupby('analysis_unit_id',sort=False).y_frozen.shift(1)
 d['train_role']=tr.to_numpy();d['eval_role']=ev.to_numpy();d['calibration_role']=(d.Year<=2014)&(d.spatial_fold==t['fold']) if t['kind']=='SPATIAL_BASELINE' else False
 d['transform_id']=t['task_id']+'_TRAIN_FROZEN_V1';d['target_definition_id']=d.transform_id;d['climate_reference_id']=ref
 for r in recs:r.update(task_id=t['task_id'],fold_id=t['fold'],transform_id=d.transform_id.iloc[0],source_version='original_frozen_monthly_inputs',source_sha256=h(d0.row_id))
 d['functional_group']=pd.Categorical(d.functional_group,categories=GROUPS);d['Month_cat']=pd.Categorical(d.Month,categories=[5,6,7,8,9])
 return d,pd.DataFrame(recs)

def profiled_solver(df,formula,initial):
 y0,X0=dmatrices(formula,df,return_type='dataframe');X=np.asarray(X0);y=np.asarray(y0).ravel();n=len(y);uid=pd.Categorical(df.analysis_unit_id).codes;zid=pd.Categorical(df.Zone).codes;nu=np.bincount(uid);nz=zid.max()+1;U=np.zeros((len(nu),X.shape[1]+1));A=np.column_stack([X,y]);np.add.at(U,uid,A);uz=np.zeros(len(nu),int)
 for u in range(len(nu)):uz[u]=zid[np.where(uid==u)[0][0]]
 def evaluate(loglam,details=False,gradient=False):
  lu,lz=np.expm1(loglam);f=lu/(1+lu*nu);VA=A-f[uid,None]*U[uid];SZ=np.zeros((nz,A.shape[1]));np.add.at(SZ,zid,VA);vone=1/(1+lu*nu[uid]);s=np.bincount(uz,weights=nu/(1+lu*nu),minlength=nz);zz=lz/(1+lz*s);VA-=vone[:,None]*zz[zid,None]*SZ[zid];Q=A.T@VA;xt=Q[:-1,:-1];xy=Q[:-1,-1];beta=np.linalg.solve(xt,xy);ss=Q[-1,-1]-xy@beta;ld=np.log1p(lu*nu).sum()+np.log1p(lz*s).sum();nll=.5*(n*(np.log(2*np.pi)+1+np.log(ss/n))+ld)
  if details:return nll,beta,ss/n,xt
  if gradient:
   vr=VA[:,-1]-VA[:,:-1]@beta;su=np.bincount(uid,weights=vr);sz=np.bincount(zid,weights=vr);ud=nu/(1+lu*nu)
   trace_u=np.sum(ud-zz[uz]*ud**2);trace_z=np.sum(s/(1+lz*s));grad=.5*np.array([trace_u-n/ss*np.sum(su**2),trace_z-n/ss*np.sum(sz**2)])*(1+np.array([lu,lz]));return nll,grad
  return nll
 starts=[np.zeros(2),np.log1p([.01,.01]),np.log1p([1,1]),np.log1p([initial['unit_variance']/initial['residual_variance'],initial['zone_variance']/initial['residual_variance']])]
 cand=[(evaluate(s),s,'start') for s in starts]
 for s in starts:
  opt=minimize(lambda x:evaluate(x,gradient=True),s,jac=True,method='L-BFGS-B',bounds=[(0,20),(0,20)],options={'ftol':1e-13,'gtol':1e-7,'maxiter':1000});cand.append((opt.fun,opt.x,'L-BFGS-B'))
 for j in [0,1]:
  def fun(v):
   q=np.zeros(2);q[j]=v;return evaluate(q)
  opt=minimize_scalar(fun,bounds=(0,10),method='bounded',options={'xatol':1e-10});q=np.zeros(2);q[j]=opt.x;cand.append((opt.fun,q,'boundary_scalar'))
 best=min(cand,key=lambda z:z[0]);nll,beta,sigma,xt=evaluate(best[1],True);llold=-evaluate(starts[-1]);delta_check=llold-initial['loglikelihood'] if initial.get('loglikelihood') is not None else np.nan
 if initial.get('loglikelihood') is not None:assert abs(delta_check)<1e-5,(delta_check,formula)
 # Exact profile score and projected KKT residual at variance boundaries.
 _,gradient=evaluate(best[1],gradient=True);kkt=[(abs(gradient[j]) if best[1][j]>1e-10 else max(0,-gradient[j]))/n for j in range(2)]
 return dict(beta=beta,sigma=sigma,unit_variance=float(np.expm1(best[1][0])*sigma),zone_variance=float(np.expm1(best[1][1])*sigma),ll=-nll,original_ll_check=delta_check,method=best[2],kkt=max(kkt),design=X0.design_info,terms=X0.columns.tolist(),se=np.sqrt(np.diag(np.linalg.inv(xt))*sigma))

def metrics(y,p):
 e=y-p;return dict(RMSE=float(np.sqrt(np.mean(e*e))),MAE=float(np.mean(abs(e))),bias=float(np.mean(e)),MSE=float(np.mean(e*e)),centered_error_variance=float(np.var(e)))

def wide(q,ymode=False):
 wanted=[x+'Y' for x in MODELS] if ymode else MODELS;q=q[q.model_id.isin(wanted)];meta=q[q.model_id==wanted[0]].copy().set_index('row_id');pv=q.pivot(index='row_id',columns='model_id',values='yhat_raw');assert pv.notna().all().all();assert len(meta)==len(pv)
 meta=meta.join(pv);return meta.reset_index()

def contrasts(w,task,ymode=False):
 rows=[]
 for cid,b,a in CONS:
  for prod in ['SMCI','GLEAM']:
   bm=b+prod if b.endswith('_') else b;am=a+prod;bm+=('Y' if ymode else '');am+=('Y' if ymode else '');y=w.y_raw.to_numpy();pb=w[bm].to_numpy();pa=w[am].to_numpy();mb=metrics(y,pb);ma=metrics(y,pa);e=y-pb;c=pa-pb
   row=dict(task_id=task,fold_id_or_pooled=task,sample_set_id='M_Y1' if ymode else 'M_ALL',estimand_id='SECONDARY_YLAG1' if ymode else 'PRIMARY',target_scale='raw_kNDVI',product=prod,contrast_id=cid,base_model=bm,aug_model=am,n_rows=len(w),n_blocks=w.spatial_block_id.nunique(),n_units=w.analysis_unit_id.nunique(),n_years=w.Year.nunique(),RMSE_base=mb['RMSE'],RMSE_aug=ma['RMSE'],MAE_base=mb['MAE'],MAE_aug=ma['MAE'],bias_base=mb['bias'],bias_aug=ma['bias'],delta_RMSE=mb['RMSE']-ma['RMSE'],relative_RMSE_reduction_pct=100*(mb['RMSE']-ma['RMSE'])/mb['RMSE'],delta_MSE=mb['MSE']-ma['MSE'],delta_MAE=mb['MAE']-ma['MAE'],mean_e0c=float(np.mean(e*c)),mean_c2=float(np.mean(c*c)),two_mean_e0c=float(2*np.mean(e*c)),delta_MSE_identity=float(2*np.mean(e*c)-np.mean(c*c)),centered_error_variance_base=mb['centered_error_variance'],centered_error_variance_aug=ma['centered_error_variance']);row['identity_residual']=row['delta_MSE_identity']-row['delta_MSE'];rows.append(row)
 return rows

def bootstrap(w,label,ymode=False,length=2,manifest_dir=None):
 mods=[x+'Y' for x in MODELS] if ymode else MODELS;years=np.arange(w.Year.min(),w.Year.max()+1);blocks=sorted(w.spatial_block_id.unique());bi=pd.Categorical(w.spatial_block_id,categories=blocks).codes;yi=w.Year.to_numpy()-years[0];nb=len(blocks);ny=len(years);e=w.y_raw.to_numpy()[:,None]-w[mods].to_numpy();counts=np.zeros((nb,ny));sse=np.zeros((nb,ny,len(mods)));sae=np.zeros_like(sse)
 np.add.at(counts,(bi,yi),1);np.add.at(sse,(bi,yi),e*e);np.add.at(sae,(bi,yi),abs(e));rng=np.random.default_rng(SEED);space_only=(ny<=2)
 out=[];sampling=[];points=contrasts(w,label,ymode)
 for draw in range(2000):
  bw=np.bincount(rng.integers(0,nb,nb),minlength=nb)
  if space_only:yw=np.ones(ny,dtype=int);starts=np.array([],dtype=int)
  else:
   starts=rng.integers(0,ny-length+1,int(np.ceil(ny/length)));inds=np.concatenate([np.arange(v,v+length) for v in starts])[:ny];yw=np.bincount(inds,minlength=ny)
  sampling.append({'draw':draw,'block_counts':bw.tolist(),'year_counts':yw.tolist(),'time_block_start_years':(starts+years[0]).tolist()});ww=bw[:,None]*yw[None,:];den=np.sum(ww*counts);ms=np.sum(ww[:,:,None]*sse,axis=(0,1))/den;rm=np.sqrt(ms);ma=np.sum(ww[:,:,None]*sae,axis=(0,1))/den
  vals={}
  for row in points:
   b=mods.index(row['base_model']);a=mods.index(row['aug_model']);pct=100*(rm[b]-rm[a])/rm[b];vals[(row['contrast_id'],row['product'])]=(pct,rm[b]-rm[a],ms[b]-ms[a]);out.append(dict(task_id=label,estimand_id=row['estimand_id'],contrast_id=row['contrast_id'],product=row['product'],draw=draw,relative_RMSE_reduction_pct=pct,delta_RMSE=rm[b]-rm[a],delta_MSE=ms[b]-ms[a],delta_MAE=ma[b]-ma[a],time_block_length=length))
  for cid,_,_ in CONS:
   sm=vals[(cid,'SMCI')];gl=vals[(cid,'GLEAM')];out.append(dict(task_id=label,estimand_id=points[0]['estimand_id'],contrast_id=cid,product='SMCI_MINUS_GLEAM',draw=draw,relative_RMSE_reduction_pct=sm[0]-gl[0],delta_RMSE=sm[1]-gl[1],delta_MSE=sm[2]-gl[2],time_block_length=length))
 if manifest_dir is not None:
  sd=Path(manifest_dir);sd.mkdir(exist_ok=True);sid=str(w.sample_set_id.iloc[0]) if 'sample_set_id' in w else 'M_ALL';key=label+'__'+sid+('__Y' if ymode else '')+'__L'+str(length);pd.DataFrame(sampling).to_parquet(sd/(key+'.parquet'),index=False);(sd/(key+'.json')).write_text(json.dumps({'seed':SEED,'ordered_blocks':blocks,'ordered_years':years.tolist(),'replicates':2000,'space_only':space_only,'length':length}),encoding='utf-8')
 draws=pd.DataFrame(out);summ=[]
 for row in points:
  x=draws[(draws.contrast_id==row['contrast_id'])&(draws['product']==row['product'])];row.update(CI_lower=x.relative_RMSE_reduction_pct.quantile(.025),CI_upper=x.relative_RMSE_reduction_pct.quantile(.975),delta_RMSE_CI_lower=x.delta_RMSE.quantile(.025),delta_RMSE_CI_upper=x.delta_RMSE.quantile(.975),delta_MSE_CI_lower=x.delta_MSE.quantile(.025),delta_MSE_CI_upper=x.delta_MSE.quantile(.975),CI_method='paired_space_cluster_conditional_period' if space_only else 'paired_crossed_space_and_synchronous_calendar_year_MBB',time_block_length=0 if space_only else length,bootstrap_n=2000,conditional_on_fitted_models=True,temporal_support_warning='Only two years; spatial conditional interval' if space_only else ('Only eight test years' if ny==8 else 'Training sets overlap; interval conditional on fitted models'));summ.append(row)
 for cid,_,_ in CONS:
  sm=next(r for r in points if r['contrast_id']==cid and r['product']=='SMCI');gl=next(r for r in points if r['contrast_id']==cid and r['product']=='GLEAM');x=draws[(draws.contrast_id==cid)&(draws['product']=='SMCI_MINUS_GLEAM')];r=sm.copy();r.update(paired_product_difference=True,product='SMCI_MINUS_GLEAM',relative_RMSE_reduction_pct=sm['relative_RMSE_reduction_pct']-gl['relative_RMSE_reduction_pct'],delta_RMSE=sm['delta_RMSE']-gl['delta_RMSE'],delta_MSE=sm['delta_MSE']-gl['delta_MSE'],CI_lower=x.relative_RMSE_reduction_pct.quantile(.025),CI_upper=x.relative_RMSE_reduction_pct.quantile(.975),delta_RMSE_CI_lower=x.delta_RMSE.quantile(.025),delta_RMSE_CI_upper=x.delta_RMSE.quantile(.975),delta_MSE_CI_lower=x.delta_MSE.quantile(.025),delta_MSE_CI_upper=x.delta_MSE.quantile(.975));summ.append(r)
 ss=pd.DataFrame(summ);ss['paired_product_difference']=ss['product'].eq('SMCI_MINUS_GLEAM');return ss,draws

def evaluate_fold(raw_panel, fold_climate, task):
    """Fit the primary five-model information ladder on common complete rows."""
    if not raw_panel.groupby("analysis_unit_id").Zone.nunique().eq(1).all():
        raise ValueError("Units must be nested within reporting zones")
    design, transforms = prepare(raw_panel, fold_climate, task)
    soil = {product: [f"{product.lower()}_L{k}" for k in range(7)]
            for product in ("SMCI", "GLEAM")}
    common = (np.isfinite(design[["y_frozen", *BASE, *soil["SMCI"], *soil["GLEAM"]]]).all(axis=1)
              & design.Month.isin(range(5, 10)))
    train = design.loc[common & design.train_role].copy()
    test = design.loc[common & design.eval_role].copy()
    if train.empty or test.empty:
        raise ValueError("Empty common training or evaluation sample")
    if task["kind"].startswith("SPATIAL") and set(train.spatial_block_id) & set(test.spatial_block_id):
        raise ValueError("Spatial leakage across train and evaluation blocks")
    output = []
    for name in MODELS:
        if name == "M0":
            terms = BASE
        else:
            prefix, product = name.split("_")
            terms = BASE + (soil[product][:1] if prefix == "M1" else soil[product])
        formula = "y_frozen ~ C(functional_group) + C(Month_cat) + " + " + ".join(terms)
        model = profiled_solver(train, formula, dict(unit_variance=1., zone_variance=1., residual_variance=1.))
        if model["kkt"] >= 1e-5:
            raise RuntimeError(f"Profile ML failed convergence for {name}")
        x = np.asarray(build_design_matrices([model["design"]], test)[0])
        frozen = x @ model["beta"]
        output.append(test[["row_id", "analysis_unit_id", "spatial_block_id", "Year",
                            "Month", "date", "Zone", "functional_group", "y_raw"]].assign(
            model_id=name, task_id=task["task_id"], fold_id=task["fold"], sample_set_id="M_ALL",
            yhat_frozen=frozen, yhat_raw=test.y_offset + test.y_scale * frozen))
    return pd.concat(output, ignore_index=True), transforms
