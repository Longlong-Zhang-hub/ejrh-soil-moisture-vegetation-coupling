# Copyright (c) 2026 Longqi Zhang

import numpy as np
import pandas as pd

def aggregate(weight_matrix,x):
 ok=np.isfinite(x);den=np.asarray(weight_matrix@ok.astype(float).T).T;num=np.asarray(weight_matrix@np.nan_to_num(x).T).T
 return np.where(den>=1-1e-10,num,np.nan),den

MODELS = ("ACCESS-CM2", "GFDL-ESM4", "IPSL-CM6A-LR", "MIROC6", "MRI-ESM2-0")
SSPS = ("ssp126", "ssp245", "ssp585")
PERIODS = ((2021, 2040), (2041, 2060))


def corrected_cwbi(monthly):
    """Calculate corrected three-month CWBI from authorized monthly water balance."""
    keys = ["support", "support_id", "GCM", "scenario"]
    full = monthly.sort_values(keys + ["Year", "Month"]).reset_index(drop=True).copy()
    ordinal = full.Year * 12 + full.Month - 1
    if not full.Month.between(1, 12).all() or not ordinal.groupby(
            [full[k] for k in keys]).diff().dropna().eq(1).all():
        raise ValueError("Use one row per consecutive calendar month and climate key")
    full["CWBI3_raw"] = full.groupby(keys, sort=False).water_balance_old.transform(
        lambda x: x.rolling(3, min_periods=3).sum())
    refkeys = keys[:-1] + ["Month"]
    baseline = full.loc[full.scenario.eq("historical") & full.Year.between(1995, 2014)]
    reference = baseline.groupby(refkeys).CWBI3_raw.agg(["mean", "std", "count"])
    parameters = reference.reindex(pd.MultiIndex.from_frame(full[refkeys]))
    if not parameters["count"].eq(20).all() or not parameters["std"].gt(0).all():
        raise ValueError("Each support and GCM needs a full nonconstant 1995–2014 reference, with Nov–Dec 1994 antecedents")
    full["CORRECTED_WARMUP"] = (full.CWBI3_raw.to_numpy() - parameters["mean"].to_numpy()) / parameters["std"].to_numpy()
    return full


def summarize_exposure(full):
    """Five-zone primary Fig. 8 statistic after authorized native-grid aggregation."""
    keys = ["support", "support_id", "GCM"]
    full = full[full.support.eq("zone")]
    rows = []
    for (support, uid, gcm), group in full.groupby(keys):
        hist = group.loc[group.scenario.eq("historical") & group.Year.between(1995, 2014), "CORRECTED_WARMUP"]
        for ssp in SSPS:
            for lo, hi in PERIODS:
                future = group.loc[group.scenario.eq(ssp) & group.Year.between(lo, hi), "CORRECTED_WARMUP"]
                if hist.count() != 240 or future.count() != 240:
                    raise ValueError("Incomplete baseline or scenario period; supply the needed antecedent months")
                rows.append(dict(support=support, support_id=uid, gcm=gcm, ssp=ssp,
                                 period_start=lo, period_end=hi,
                                 variant="CORRECTED_WARMUP", period_statistic="median",
                                 exposure_change=float(future.median() - hist.median())))
    per_model = pd.DataFrame(rows)
    ens_keys = ["support", "support_id", "ssp", "period_start", "period_end", "variant", "period_statistic"]
    ensemble = []
    for key, group in per_model.groupby(ens_keys):
        if len(group) != len(MODELS) or set(group.gcm) != set(MODELS):
            raise ValueError("Each zone, scenario and period requires the five declared GCMs")
        x = group.exposure_change.to_numpy()
        med = float(np.median(x))
        ensemble.append(dict(zip(ens_keys, key), five_GCM_median=med, n_gcm=len(group),
                             direction_agreement=int((np.sign(x) == np.sign(med)).sum()),
                             n_negative=int((x < 0).sum()), min_gcm=float(x.min()), max_gcm=float(x.max())))
    return per_model, pd.DataFrame(ensemble)
