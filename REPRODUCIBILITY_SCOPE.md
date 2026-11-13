# Reproducibility scope and figure coverage

This source archive supplies research **analysis kernels**, a deterministic synthetic integration test and plotting functions. It is not an end-to-end reproduction of the manuscript from public raw data. The reader must acquire licensed upstream products and the matching analysis-ready inputs separately; the corresponding file schemas and function order are documented in `INPUTS.md`.

## Included algorithm families

| Research component | Implemented entry points | What the code does not supply |
| --- | --- | --- |
| Fixed-area monthly aggregation and transformations | `ejrh_code.monthly` | Raw satellite downloads, native product corrections, vegetation STL and construction of registered masks |
| Calendar-lag soil-state persistence and conditional history | `ejrh_code.persistence`, `ejrh_code.coupling.conditional` | Primary observational monthly panels and original fixed spatial units |
| Distributed-lag association and uncertainty | `ejrh_code.coupling.run_dlm` | Historical model registry and all required processed predictor/response columns |
| Same-month conditional prediction | `ejrh_code.prediction` | Fold-specific SPEI generation, complete source panel build and frozen climate reference generation |
| Drought suppression, censoring and recovery associations | `ejrh_code.events` | Original drought-event reconciliations, endpoint registry and full sensitivity report assembly |
| Fixed-zone comparisons and spatially paired inference | `ejrh_code.zonal` | Original zone geometries, PAWC eligibility layers and complete qualified sensitivity input tables |
| Five-zone climate exposure | `ejrh_code.exposure` | Native precipitation and evapotranspiration processing; observationally constrained future ecosystem projections are not included |

## Figure-function coverage

| Figure(s) | Function(s) in `ejrh_code.figures` | Coverage |
| --- | --- | --- |
| Main Fig. 1 | — | Final source map composition **not included** |
| Main Fig. 2 | `fig2` | Plotting from provided mapped values and boundaries |
| Main Fig. 3 | `fig3` | Plotting from precomputed persistence profiles |
| Main Fig. 4 | `fig4` | Plotting from precomputed lag estimates and summary statistics |
| Main Fig. 5 | `fig5`, `fig5c` | Plotting from precomputed prediction and geometry tables |
| Main Fig. 6(a–d) | — | Original artwork **not included** |
| Main Fig. 6(e) | `fig6e` | Recovery curve plotting from provided results |
| Main Fig. 7 | `fig7` | Zone comparisons, including available count-bar form |
| Main Fig. 8 | `fig8` | Five-zone *climate-exposure* visualization from area-aggregated input values |
| Supplement Figs. S1–S2 | — | Standalone final renderers **not included** |
| Supplement Figs. S3–S5 | `fig_s3`, `fig_s4`, `fig_s5` | Plotting from precomputed input tables |

The source has figure rendering functions for 11 synthetic PNG outputs; these are **generated demonstrations**, not the paper's original figures. Exact published values, maps, tables and summary statistics are not derivable without external datasets and additional preprocessing workflows.

## Interpretation boundaries

- Forcing-adjusted lag coefficients are conditional statistical associations, not physical residence times or causal mediation effects.
- Fold-specific prediction improvements depend on the stated evaluation task and training-only preprocessing. A local-history baseline is not a new-location prediction.
- The climate module provides conditional change summaries in meteorological water balance only. It does not generate future soil moisture, vegetation, ecological degradation, or carrying-capacity exceedance estimates.
- Synthetic tests exercise internal logic and invariants. They do not verify published numerical results independently.
