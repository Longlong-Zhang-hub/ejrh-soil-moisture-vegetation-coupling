# Copyright (c) 2026 Longqi Zhang


"""Deterministic synthetic tests. All observations and geometries are generated here."""
from pathlib import Path
import argparse
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from ejrh_code import coupling, events, exposure, figures, monthly, persistence, prediction, zonal

PRODUCTS = ["SMCI", "GLEAM"]
GROUPS = ["WOODY", "HERBACEOUS", "SPARSE"]
ZONES = [f"Z{i}" for i in range(1, 6)]


def sample_panel():
    rng = np.random.default_rng(4927)
    dates = pd.period_range("2000-01", "2018-12", freq="M")
    parts = []
    for i in range(45):
        unit = f"synthetic-{i:03}"
        n = len(dates)
        soil = rng.normal(size=(5, n))
        smci = np.convolve(soil[0], [1, .6, .25], mode="full")[:n]
        gleam = np.convolve(soil[1], [1, .5], mode="full")[:n]
        signal = .32 * smci + .16 * gleam + rng.normal(size=n)
        raw = .4 + .04 * np.sin(dates.month * np.pi / 6) + .025 * signal
        parts.append(pd.DataFrame(dict(
            analysis_unit_id=unit, spatial_block_id=f"fake-block-{i:03}", Zone=ZONES[i % 5],
            functional_group=GROUPS[i % 3], spatial_fold=i % 5 + 1, valid_area=1.+i/20,
            Year=dates.year, Month=dates.month, date=dates.to_timestamp(),
            ordinal=dates.year*12+dates.month-1, time_index=np.arange(n)-12,
            row_id=[f"{unit}/{date}" for date in dates], SMCI=smci, GLEAM=gleam,
            D=soil[2], SPEI3=soil[3], SPEI6=soil[4], PET_z=rng.normal(size=n),
            Y=signal, FVC=.5*signal + rng.normal(size=n), y_raw=raw,
            SMCI_month_mean=smci, GLEAM_SMrz_month=gleam, PET=rng.normal(size=n)
        )))
    return pd.concat(parts, ignore_index=True)


def test_monthly(panel):
    cells = pd.DataFrame(dict(GridB_ID=np.arange(6), year=2001, month=1, sm=np.arange(6.)))
    mapping = pd.DataFrame(dict(GridB_ID=np.arange(6), analysis_unit_id="fictional-unit",
                                Area=np.arange(1,7), stable_flag=[True]*5+[False]))
    result = monthly.aggregate_frame(cells, mapping, ["sm"]).set_index("support")
    assert np.isclose(result.loc["stable", "sm"], np.average(np.arange(5), weights=np.arange(1,6)))
    assert result.loc["full", "support_cell_count"] == 6
    q = panel.copy().reset_index(drop=True)
    anomalies, fit = monthly.robust(q, "SMCI")
    residual, trend = monthly.detrend(q, "SMCI", 2014)
    assert np.isfinite(anomalies).all() and np.isfinite(residual).all()
    assert len(fit) == 45*12 and len(trend) == 45
    variables = ["SMCI", "GLEAM", "D", "PET_z", "SPEI3", "SPEI6"]
    try:
        monthly.monthly_lags(q.drop(index=18), variables)
    except ValueError as exc:
        assert "missing calendar months" in str(exc)
    else:
        raise AssertionError("Calendar gap slipped through")
    lagged = monthly.monthly_lags(q, variables)
    assert lagged.iloc[12].SMCI_L12 == q.iloc[0].SMCI
    return lagged


def test_history(panel, lagged, output):
    registry = panel[["analysis_unit_id", "spatial_block_id", "valid_area", "Zone",
                      "functional_group"]].drop_duplicates()
    results, units, sw, bidx, nt = persistence.calendar_persistence(panel, registry, PRODUCTS)
    q, d = results["SMCI"]
    point, draws = persistence.regional_fisher(q, d, units.valid_area.to_numpy(), sw, bidx)
    assert draws.shape == (999,13) and point[0] == 1 and np.isfinite(draws).all()
    keep = units.Zone.eq("Z1").to_numpy()
    est, boot, distinct = persistence.aggregate(q[keep],d[:,keep],units.valid_area.to_numpy()[keep],
                                                 sw,bidx[keep],nt)
    assert est[0] == 1 and (distinct >= 3).sum() >= 950
    assert np.isfinite(boot[:,3]).sum() >= 950
    summary = coupling.run_dlm(lagged, output, "synthetic_U6", "SMCI", "Y", "SMCI", basis="U")
    assert summary["n_obs"] > 0 and summary["convergence_status"] == "SOLVED"
    state = coupling.conditional(lagged, output, "SMCI", "F3", free=True)
    assert state["n_bootstrap"] == 999
    sub = lagged[lagged.Year.between(2001,2014)&lagged.Month.between(5,9)].copy()
    x, terms, projection = coupling.design(sub, "SMCI", "Y", 6, "U")
    x, names, maps = zonal.projection(terms, x, projection, sub, "SMCI", "Y", [0])
    s, t, bi, ti = coupling.weights(sub)
    beta, draws, failures = zonal.fit_boot(x, sub.Y.to_numpy(), s[:,bi], t[:,ti])
    offsets = np.array([maps[z][1][0] @ beta for z in ZONES[1:]])
    proj = np.vstack([maps[z][1][0] for z in ZONES[1:]])
    p, low, high, raw, n = zonal.max_family(offsets, draws @ proj.T)
    assert len(names) == len(beta) and n >= 950 and 0 < p <= 1
    assert (low <= high).all()
    covariates = units.assign(D_mm_season=np.random.default_rng(210).normal(-250,80,len(units)))
    xenv = zonal.environment_design(covariates)
    outcomes = q[:,[3,6]]
    p_env, boot_env, issues = zonal.regression(xenv, outcomes, d[:,:,[3,6]], sw[:,bidx])
    assert np.isfinite(p_env).all() and np.isfinite(boot_env[:,1,0]).sum() >= 950
    print("PASS monthly, persistence, conditional state, DLM, zone effects, environmental refits", flush=True)


def test_prediction(panel):
    panel = panel[panel.Year.ge(2001)].copy()
    climate = panel[["analysis_unit_id","date","SPEI3"]].rename(columns={"SPEI3":"SPEI3_fold"})
    climate["transform_id"] = "SPEI3_1960_2014"
    task = prediction.tasks()[0] | {"end_eval":2018}
    pred, params = prediction.evaluate_fold(panel, climate, task)
    assert len(pred) and params.loc[params.fit_role.eq("training"),"fit_time_end"].max().startswith("2014")
    shifted = panel.assign(y_raw=panel.y_raw + np.where(panel.Year.ge(2015),3,0))
    first, _ = prediction.prepare(panel, climate, task)
    second, _ = prediction.prepare(shifted, climate, task)
    np.testing.assert_allclose(first.y_offset, second.y_offset)
    np.testing.assert_allclose(first.y_scale, second.y_scale)
    summary, draws = prediction.bootstrap(prediction.wide(pred), task["task_id"])
    assert len(summary) == 9 and len(draws) == 18000 and np.isfinite(summary.CI_lower).all()
    assert np.max(abs(summary.identity_residual)) < 1e-10
    print("PASS frozen training transform, five same-sample fits, paired prediction bootstrap", flush=True)


def test_events():
    spell = np.zeros(30)
    spell[[3,5,14,15]] = -1.5
    y = np.zeros(30)
    y[4:7] = -1.2
    y[15:17] = -1.3
    y[17] = np.nan
    registry, end = events.identify_recovery(spell, y)
    assert registry.eligibility_class.tolist() == ["new_suppression","new_suppression"]
    main = end[end.followup_rule.eq("met_end_plus12") & end.recovery_threshold.eq(-.25)]
    assert main.t.tolist() == [4,1] and main.recovered.tolist() == [1,0]
    assert main.censor_reason.tolist() == ["", "internal_missing"]
    km, _, nr, _, _ = events.curve(main, max_t=8)
    assert nr[0,0] == 2 and km[0,4] == 1
    competing = events.before_recurrence(pd.DataFrame(dict(t=[4,4], recovered=[1,1], next_i=[7,12], origin_i=[3,3])), True)
    cif, _, _, _, _ = events.curve(competing, competing=True)
    assert competing.recovered.tolist()==[0,1] and cif[0,4] == .5
    p = pd.DataFrame(dict(unit_id="fictional",time_index=np.arange(30)))
    for prod,col in events.PRODUCTS.items():
        p[col] = .3
        for version in ("pooled","calendar_month"):
            p[prod+"_low_"+version] = False
            p.loc[2:3,prod+"_low_"+version] = True
    eps=pd.DataFrame([dict(unit_id="fictional",L=4,veg_episode_id="fake",block_id="fake",
                           SPEI_L=-.3,elapsed_drought_L=2,initial_y=-1.2,month=5,zone="Z1",stratum="WOODY",entry_year=2001)])
    risk, elig, _ = events.make_risk(p,eps,main.assign(veg_episode_id=["fake","other"]),len(p))
    assert len(risk)==12 and risk.D_pre.eq(2).all() and elig.exclusion_reason.eq("").all()
    rng=np.random.default_rng(113)
    n=700
    data=pd.DataFrame(dict(D_pre=rng.integers(0,5,n),SM_previous=rng.normal(size=n),SPEI_L=rng.normal(size=n),
        elapsed_drought_L=rng.integers(1,5,n),initial_y=rng.normal(-1.3,.2,n),month=rng.integers(5,10,n),
        stratum=rng.choice(GROUPS,n),zone=rng.choice(ZONES,n),time_bin=rng.choice(["2","3","4","5-6","7-9","10+"],n),
        block_id=rng.integers(0,25,n),entry_year=rng.integers(2001,2015,n)))
    data["recovered_indicator"]=rng.binomial(1,1/(1+np.exp(.28*data.D_pre)))
    coef, block, tw = events.fit_landmark(data)
    assert coef.D_pre < 0 and block.shape == tw.shape
    print("PASS event eligibility, confirmation clock, censoring, competing risk and landmark model", flush=True)
    return pd.DataFrame(dict(month_since_origin=np.arange(9),estimate=km[0],
                             ci_low=np.maximum(0,km[0]-.1),ci_high=np.minimum(1,km[0]+.1),n_risk=nr[0]))


def test_climate():
    from scipy.sparse import csr_matrix
    values, area = exposure.aggregate(csr_matrix([[.5,.5]]),np.array([[1.,3.],[2.,np.nan]]))
    assert values[0,0]==2 and np.isnan(values[1,0]) and area[1,0]==.5
    rng=np.random.default_rng(3232)
    parts=[]
    for zone in ZONES:
        for model in exposure.MODELS:
            for scenario in ("historical",*exposure.SSPS):
                dates=pd.period_range("1994-11","2014-12",freq="M") if scenario=="historical" else pd.period_range("2020-05","2060-12",freq="M")
                vals=rng.normal(size=len(dates)) + (-.12 if scenario!="historical" else 0)
                parts.append(pd.DataFrame(dict(support="zone",support_id=zone,GCM=model,scenario=scenario,
                                               Year=dates.year,Month=dates.month,water_balance_old=vals)))
    monthly_data = pd.concat(parts, ignore_index=True)
    corrected=exposure.corrected_cwbi(monthly_data)
    assert corrected.loc[corrected.scenario.eq("historical") & corrected.Year.eq(1995),"CORRECTED_WARMUP"].notna().all()
    per_model, ensemble=exposure.summarize_exposure(corrected)
    assert len(per_model)==150 and len(ensemble)==30 and ensemble.n_gcm.eq(5).all()
    assert (ensemble.min_gcm <= ensemble.five_GCM_median).all()
    print("PASS complete-area weighting, warm-up CWBI3 and five-model climate-only summaries", flush=True)
    return ensemble


def test_figures(output, recovery, ensemble):
    corr=pd.DataFrame([dict(product=p,transform_id=transform,lag_months=l,estimate=.85**l,
        ci_low=.85**l-.06,ci_high=.85**l+.06,period_id=period,stratum_or_region="REGION",
        mask_version="original2020",aggregation_weight_id="fixed_valid_area_FisherZ")
        for p in PRODUCTS for transform in ("seasonal","seasonal_linear_detrend")
        for period in ("2001-2014","2001-2022") for l in range(13)])
    terms=pd.DataFrame([dict(product=p,stratum=g,estimate=.48,ci_low=.31,ci_high=.62) for p in PRODUCTS for g in GROUPS])
    figures.fig3(corr[corr.period_id.eq("2001-2022")],terms,output/"fig3.png")
    figures.fig_s5(corr,output/"fig-s5.png")
    ids=['C_SMCI_SPEI3_U6','V_SMCI_Y_U6','V_SMCI_FVC_U6','C_GLEAM_SPEI3_U6','V_GLEAM_Y_U6','V_GLEAM_FVC_U6']
    coef=np.array([.44,.07,.04,.01,-.02,.025,.008])
    co=pd.DataFrame([dict(spec_id=s,stratum_or_region="REGION",lag_months=k,coefficient=v,
        ci_low=v-.025,ci_high=v+.025) for s in ids for k,v in enumerate(coef)])
    info=pd.DataFrame([dict(spec_id=s,cumulative_net=coef.sum(),centroid_lag=np.average(np.arange(7),weights=abs(coef))) for s in ids])
    figures.fig4(info,co,output/"fig4.png")
    geo=gpd.GeoDataFrame(dict(spatial_block_id=[f"g{i}" for i in range(5)],assigned_zone_id=ZONES),
        geometry=[box(i+1,1,i+1.95,2) for i in range(5)],crs=4326)
    figures.fig5c(geo,pd.DataFrame(dict(block_id=geo.spatial_block_id,delta=np.arange(5)*.001-.002)),"delta",output/"fig5c.png")
    subset = pd.DataFrame([dict(task_id=t,contrast_id=c,product=p,
        relative_RMSE_reduction_pct=.2,CI_lower=-.1,CI_upper=.4)
        for t in ("TEMPORAL_2015_2022","SPATIAL_NEW_POOLED")
        for c in ("CURRENT","PAST_GIVEN_CURRENT","TOTAL") for p in PRODUCTS])
    windows = pd.DataFrame([dict(task_id=t,product=p,relative_RMSE_reduction_pct=.3,CI_lower=-.1,CI_upper=.5)
        for t in [f"ROLL_{i}" for i in range(6)]+["ROLLING_POOLED"] for p in PRODUCTS])
    r = np.random.default_rng(7)
    observed = r.uniform(.1,.9,60)
    p5 = pd.concat([pd.DataFrame(dict(row_id=np.arange(60),model_id=m,y_raw=observed,
        yhat_raw=observed+r.normal(0,.08,60))) for m in prediction.MODELS],ignore_index=True)
    figures.fig5(subset,p5,windows,geo,
                 pd.DataFrame(dict(block_id=geo.spatial_block_id,delta=np.arange(5)*.001-.002)),
                 "delta",output/"fig5.png")
    from matplotlib.colors import Normalize
    geo2 = geo.assign(field=np.arange(5))
    empty = geo2.iloc[:0]
    specs2 = [("field", f"Synthetic layer {i}", "Synthetic units", "viridis", Normalize(0,4)) for i in range(6)]
    figures.fig2(geo2,geo2.dissolve(),empty,empty,empty,specs2,(0,7,0,3),output/"fig2.png")
    figures.fig6e(recovery,output/"fig6e.png")
    zon=pd.DataFrame([dict(product=p,zone=z,stratum=s,lag=l,kind="curve",estimate=.81**l,ci_low=.81**l-.08,
        ci_high=.81**l+.07,unit_n=9 if s=="ALL" else 3) for z in ZONES for p in PRODUCTS
        for s in ("ALL","HERBACEOUS") for l in range(13)])
    zp=pd.DataFrame([dict(product=p,stratum=s,lag=l,contrast_estimate=.13,contrast_simultaneous_low=-.03,
        contrast_simultaneous_high=.21) for p in PRODUCTS for s in ("ALL","HERBACEOUS") for l in (3,6)])
    figures.fig7(zon[zon.lag.isin((3,6))],zp,output/"fig7.png")
    diff=pd.DataFrame([dict(zone=z,lag=l,kind="product_difference",ci_low=-.08,ci_high=.15,estimate=.03)
        for z in ZONES for l in (3,6)])
    figures.fig_s3(pd.concat([zon,diff],ignore_index=True),output/"fig-s3.png")
    specs=["climate_only","climate_coarse_cover","climate_common_herb","climate_cover_zone",
           "climate_cover_zone_PAWC33","climate_cover_fractions"]+[f"LOZO_without_{z}" for z in ZONES]
    env=pd.DataFrame([dict(product=p,lag=l,term="D_per_100_mm",specification=s,n_resamples=999,
         estimate=.09,ci_low=.04,ci_high=.13) for p in PRODUCTS for l in (3,6) for s in specs])
    units=pd.DataFrame({"D_mm_season":np.arange(-700,-200,10),
        **{f"{p}_z{k}":np.linspace(.6,1.1,50) for p in PRODUCTS for k in (3,6)}})
    figures.fig_s4(units,env,output/"fig-s4.png")
    figures.fig8(ensemble,geo,output/"fig8.png",limit=max(.774,float(ensemble.five_GCM_median.abs().max())+.01))
    assert len(list(output.glob("*.png")))==11 and all(x.stat().st_size > 1000 for x in output.glob("*.png"))
    print("PASS eleven figure renderers using only generated values and geometries", flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True,help="A new directory for generated outputs")
    output=parser.parse_args().out
    output.mkdir(parents=True,exist_ok=False)
    panel=sample_panel()
    lagged=test_monthly(panel)
    test_history(panel,lagged,output)
    test_prediction(panel)
    test_figures(output,test_events(),test_climate())
    print("PASS synthetic integration suite", flush=True)


if __name__ == "__main__":
    main()
