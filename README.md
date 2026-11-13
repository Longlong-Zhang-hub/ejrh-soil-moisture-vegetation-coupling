# Root-zone soil-moisture memory and vegetation coupling

**Version 0.1.0 · MIT License · copyright © 2026 Longqi Zhang.** The existing code modules and synthetic test are inherited unchanged from the 10 November 2026 candidate. No raw observations, derived research tables, gridded products, geometries or complete end-to-end processing pipeline are included.

For the precise reproduction boundary, see `REPRODUCIBILITY_SCOPE.md`. For the locally checked test status, see `SYNTHETIC_TEST_STATUS.md`. `RELEASE_GATE.md` summarizes the completed pre-release validation.

**Author and copyright holder** Longqi Zhang (张龙齐), © 2026. See COPYRIGHT.md and CITATION.cff. The copyright holder authorized the MIT License on 12 November 2026; its active text is in `LICENSE`. Source-fragment provenance and the publication/test gate are described in `DEPENDENCY_NOTICES.md` and `RELEASE_GATE.md`.

The historical modules correspond to the Beijing–Tianjin Sand Source Region study on soil-state persistence, vegetation coupling, same-month conditional prediction, drought recovery and zonal differences. The scenario module calculates the full-domain five-zone climate exposure summaries behind Fig. 8. Its outputs are changes in standardized three-month climatic water balance and intermodel direction agreement.

## Install and run the synthetic integration test

Use Python 3.10 in an isolated environment. From this directory, using that environment's interpreter:

```text
python -m pip install -r requirements.txt
python -B -m synthetic_test --out synthetic-output
```

The output directory must be new. The test creates fictional monthly panels and geometric shapes in memory. It exercises calendar pairing, conditional state estimates, distributed lag fits, prediction transforms and evaluation, event and risk estimators, zonal uncertainty, climate exposure and figure rendering. Output is generated locally. The synthetic assertions test execution and several method invariants.

## Workflow with separately authorized input

The code exposes Python functions for each stage. INPUTS.md describes the tables and call order. Use caller-selected output paths in a new directory. The package is organized around analysis-ready inputs and plot tables.

| Stage | Entry points | Paper connection |
| --- | --- | --- |
| Monthly panel | `monthly.aggregate_frame`, `robust`, `detrend`, `monthly_lags` | Fixed-area full and stable support, within-unit calendar-month anomalies and retrospective lag alignment |
| Historical state memory | `persistence.calendar_persistence`, `regional_fisher`, `coupling.conditional` | Fig. 3 and SI persistence profiles. Separate direct state correlation from forcing-adjusted lag terms |
| Distributed lag coupling | `coupling.design`, `run_dlm` | Fig. 4 and U6 pathway terms. Common-support OLS analogue, with paired block-time refits or two-way covariance draws |
| Conditional prediction | `prediction.tasks`, `prepare`, `evaluate_fold`, `wide`, `bootstrap` | Fig. 5. Common-row M0/M1/M2 model ladder, training-frozen transforms, fixed-effect-only evaluation and paired conditional uncertainty |
| Drought events and recovery | `events.identify_recovery`, `before_recurrence`, `curve`, `make_risk`, `fit_landmark` | Fig. 6e and SI event analysis. The event definition, second-month confirmation, positive follow-up, competing events and pre-landmark exposure are distinct steps |
| Five-zone analysis | `persistence.aggregate`, `zonal.projection`, `fit_boot`, `max_family`, `holm`, `environment_design`, `regression` | Fig. 7 and S3–S4. Fixed-support zone curves, matched U6 zone modifications, simultaneous families and climate gradients |
| Climate exposure | `exposure.aggregate`, `corrected_cwbi`, `summarize_exposure` | Fig. 8. Area-weighted native monthly water balance upstream, trailing three months, fixed 1995–2014 normalization, within-model period median change and five-model ensemble median |
| Figures | `figures.fig2`, `fig3`, `fig4`, `fig5`, `fig5c`, `fig6e`, `fig7`, `fig8`, `fig_s3`, `fig_s4`, `fig_s5` | Plotting from caller-supplied estimates and geometries |

## Reproduction boundary

Paper values, rasters, station records, geometries, third-party raw data and supplementary data files are outside this archive. Exact formal results require the authorized native products, matching registered masks and analysis units, historical transformed panels, fold-specific climate tables, accepted estimator inputs and plot source tables. The code starts after some product-specific processing, including the retrospective vegetation STL, fold-specific SPEI construction and native daily climate QC. The T4 functions provide the inference kernels. Complete sensitivity, PAWC qualification, leave-one-zone-out and reporting workflows require additional branch-specific inputs and orchestration. The event functions provide the revised endpoints, risk and association kernels. Historical event reconciliation and full sensitivity-table assembly require the original event and monthly inputs.

The plotting functions redraw selected panels and layouts from supplied inputs. Exact historical cartography, original Fig. 1 and Fig. 6a–d artwork, the final PDF composition, and standalone Fig. S1–S2 rendering are outside the supplied figure workflow. Fig. 2 starts from mapped values, and Fig. 8 starts from authorized zone geometry and climate values. Future ecological projection code is excluded. Scientific results were not recomputed during packaging.

Cite this software using `CITATION.cff` and the DOI for the particular version, when its Zenodo archive becomes available. The DOI is assigned by the archival service and must not be guessed. This repository does not redistribute any underlying third-party research datasets.
