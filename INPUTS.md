# Inputs and execution order

All inputs are supplied by the caller. Use pandas DataFrames with the named columns, or GeoDataFrames for cartography. Load authorized Parquet with pandas and authorized GeoJSON or GeoParquet with GeoPandas. Where dates are represented by `Year`, `Month` and `ordinal`, use `ordinal = Year*12 + Month - 1`. One row represents one calendar month per key. Preserve calendar gaps as explicit rows with missing values and keep the original area and mask definitions.

## 1. Historical aggregation and anomalies

- `aggregate_frame(cell_table, mapping, variables)` uses `GridB_ID, year, month` and named numeric values. The mapping has `GridB_ID, analysis_unit_id, Area, stable_flag`. The values must be at the intended stage of transformation. The fixed area average needs five valid contributing cells by default and returns full and stable rows with support counts. Convert the returned year/month field names when building the analysis panel.
- For `robust` and `detrend`, sort the unit panel and reset it to a zero-based RangeIndex. Use `analysis_unit_id, Year, Month, time_index` and a numeric value column. Robust anomalies use 2001–2014 unit/calendar-month references, median and scaled MAD with an SD fallback. Detrending uses the same numeric monthly index within each unit and a user-specified calibration end year. Retain the resulting transformation parameters alongside the caller's panel.
- For the historical lagged panel supply `analysis_unit_id, spatial_block_id, functional_group, Zone, Year, Month, ordinal`, soil anomalies `SMCI, GLEAM`, forcing `D, PET_z, SPEI3, SPEI6`, and response columns `Y, FVC` or the stated sensitivity response. `monthly_lags(panel, variables)` creates the 0–12 actual-calendar lag columns from a contiguous monthly panel. Preserve antecedents from 2000.

## 2. State memory and distributed lag models

- `calendar_persistence(panel, registry, variables, end_year)` takes one registry row per `analysis_unit_id` with `spatial_block_id, valid_area, Zone, functional_group`. Its common panel has finite values for each requested variable and May–September target and antecedent month. Pass seasonal or detrended variables explicitly. The return values are a dictionary of (unit × lag point, draw × unit × lag), the registry in matrix order, spatial block counts, unit-to-block indices, and target temporal counts. Apply `regional_fisher` for the A regional/stratum pooling. Apply `persistence.aggregate` to fixed zones or strata for the T4 valid-area estimates and its bootstrap eligibility rule.
- Call `conditional(lagged, output, product, model, free)` for predeclared state designs. Fig. 3 uses SMCI F3 and GLEAM F2 with free forcing history. Call `run_dlm(lagged, output, sid, driver, response, product, L=6, basis="U")` for the U6 profiles. This creates a new subdirectory under an existing caller-owned output directory. Keep the common complete-case lag vector and May–September 2001–2014 design when comparing pathways.

## 3. Historical prediction and uncertainty

- The raw unit panel takes `row_id, analysis_unit_id, spatial_block_id, Zone, functional_group, spatial_fold, date, Year, Month, time_index, y_raw, PET, SMCI_month_mean, GLEAM_SMrz_month`. Date is the first day of the calendar month as a datetime64 value. The one-row-per-unit-month key and spatial folds must be fixed before fitting. `functional_group` uses WOODY, HERBACEOUS and SPARSE, with units nested within zones.
- Supply `analysis_unit_id, date, transform_id, SPEI3_fold` in the climate table. Reference IDs follow `SPEI3_1960_<training end>` or `SPEI3_1960_2000` for the spatial tasks. Build those climate indices separately from the corresponding frozen climate reference before calling this code.
- Use a task from `prediction.tasks()`. For each task call `evaluate_fold` and retain the returned evaluation predictions and transform records. Training-frozen baselines differ between temporal, new-location and local-baseline estimands. Evaluate raw kNDVI on a common finite M_ALL sample. Group pooled outputs by the declared evaluation estimand, use `wide` for the paired models and use `bootstrap` for paired block/year intervals conditional on the fitted models. Overlapping windows share information. Use the same held-out row IDs for all five models within a comparison. A two-year window uses spatial-only resampling conditional on its dates.

## 4. Events, risks and censoring

- Provide monthly SPEI3 and vegetation anomaly z arrays for the same unit to `identify_recovery`. Time is the zero-based month offset. The formal start is January 2001. Group the endpoints by threshold and administrative rule. For the primary Kaplan–Meier curve, use the `met_end_plus12` rule, threshold −0.25, and positive follow-up. A recovery is recorded at the second consecutive evaluable month above the recovery threshold. Internal missingness censors at the last evaluable month. Use `before_recurrence` and then `curve(..., competing=True)` for the competing-event estimand. `bootweights` supplies paired cluster resampling multiplicities.
- For `make_risk(p, eps, end, n_months)`, supply `unit_id,time_index,SMCI_month_mean_z,GLEAM_SMrz_month_z` and `{SMCI,GLEAM}_low_{pooled,calendar_month}` for the same monthly span. The low flags use the matched unit soil series and 2001–2014 pooled or calendar-month P20 thresholds. Supply eligible episode rows with `unit_id,L,veg_episode_id,block_id,SPEI_L,elapsed_drought_L,initial_y,month,zone,stratum,entry_year`. Join `veg_episode_id` to the endpoint rows from the corresponding registry. Select one product/scenario and `common_sample` from the output risk table before `fit_landmark`. Restrict elapsed times or follow-up eligibility when running those sensitivity scenarios.

## 5. Zonal and environmental analysis

Apply the valid-area Fisher-z method on fixed unit subsets for each of Z1–Z5. Use matched common-herbaceous units for that stratum. For the U6 modification, call `zonal.projection` on the original design with `mode=[0]` for contemporaneous offsets or the relevant declared joint lag design. Use the same spatial and calendar multiplicities for `fit_boot`, and use `max_family` for the selected simultaneous coefficient family. `holm` adjusts the declared omnibus families. Keep conditional offsets and cover-weighted marginal zone coefficients distinct.

For environmental slopes, supply a single fixed row per unit with 2001–2014 mean May–September `D_mm_season`, `functional_group` and `Zone`, plus the matching Fisher-z memory outcomes and temporal refits. `environment_design` returns intercept, centered climate per 100 mm, coarse cover and zone reference coding. `regression` refits using spatial counts. Use a separately qualified static PAWC33 input and predeclared design if fitting the PAWC sensitivity. Treat slopes as conditional associations.

## 6. Fig. 8 climate exposure

`aggregate(W, x)` takes row-normalized area weights by reporting support × native cell and a month × native-cell array. Form monthly precipitation minus ET0 at native-cell support before area aggregation, using the required product versions and daily completeness. Provide `support, support_id, GCM, scenario, Year, Month, water_balance_old` to `corrected_cwbi`, in consistent monthly water-balance units. Use the historical label `historical`, with 1994 Nov–Dec antecedents for 1995, and future scenario series with the archived 2020 antecedent months. Dates must be consecutive within each key. Summarize with `summarize_exposure` using full periods 2021–2040 and 2041–2060, three SSPs and the five declared GCMs. Preserve model-level changes, ensemble medians and direction agreement counts.

## 7. Figure input tables

The callable `figures` module renders directly from supplied tables and an output path with a file extension such as .png or .pdf.

| Function | Required table contents |
| --- | --- |
| fig2 | Mapped values, outer and contextual boundary layers with CRS, six tuples of value-column/title/units/colormap/norm, and longitude/latitude extent |
| fig3, fig_s5 | Product/transform lag curves with `lag_months,estimate,ci_low,ci_high`. Fig. 3 also takes product/stratum final lag-1 terms. Fig. S5 selects regional original-mask valid-area rows for both periods |
| fig4 | Six named U6 `spec_id` summaries with cumulative net and centroid and a regional 0–6 coefficient table with pointwise intervals |
| fig5, fig5c | Panel A primary task/product/contrast RMSE intervals, panel B paired held-out row/model/raw observed/raw prediction, panel D six rolling windows plus pooled intervals by product, panel C block polygons keyed by `spatial_block_id` and supplied signed MSE-reduction contrast keyed by `block_id` |
| fig6e | Primary cumulative curve with `month_since_origin,estimate,ci_low,ci_high,n_risk` |
| fig7 | Z1–Z5 × product × stratum × lags 3 and 6 unit counts and intervals; eight signed Z5-minus-Z3 comparisons with simultaneous intervals |
| fig8 | Full-domain five-zone ensemble from step 6 and one CRS-tagged polygon per `assigned_zone_id`. Final plotting convention uses fixed diverging bounds and direction-agreement hatching |
| fig_s3 | Five-zone full lag curves and paired product contrasts, distinguished by `kind=curve` or `product_difference` |
| fig_s4 | Unit climate and product-specific Fisher-z values at lags 3 and 6, and the complete declared specification/product/lag slope table including valid resample counts |
