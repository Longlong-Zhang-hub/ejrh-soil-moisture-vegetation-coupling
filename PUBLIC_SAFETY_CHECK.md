# Public package safety check

Initial candidate checked on 2026-11-10; source-rights and native-Parquet checks completed on 2026-11-12. This record describes the bounded checks on the public source-code archive.

| Check | Result |
| --- | --- |
| Archive membership | Text-only allowlist of code, documentation, citation, dependency specification and integrity records. No data, figures, rasters, geometry, binary assets, vendor packages or internal QA artifacts are included. |
| Excluded data | All candidate member names and SHA-256 values were checked against the pending supplementary files and the internal source register. Zero matching files. |
| Text checks | Scanned all allowed text for absolute local paths, network/share paths, email and credential-like patterns, known internal artifacts and network/process launch calls. Zero matches. The named author is credited as requested. |
| Python review | All included Python sources parsed and compiled. External imports resolve to declared, separately installed packages. Source selection excludes future ecological forecast and legacy bridge code. |
| Dependencies and provenance | Direct pinned versions and their active installed dependency constraints checked on Windows with Python 3.10. No vendor files included. The copyright holder approved MIT and public release on 12 November 2026; a file-by-file review found no identifiable embedded third-party snippets requiring separate permission. |
| Synthetic execution | The integration test passed with generated observations and polygons, including historical estimation, prediction evaluation, events, zone inference, climate-only exposure and eleven plot renderers. No formal scientific result was recalculated. |
| Integrity | SHA256SUMS.txt lists SHA-256 for every other archive member. The ZIP hash covers the manifest itself. |

These checks support local scientific coordination review. Author release approval is recorded. Actual GitHub/Zenodo publication and the assigned DOI must be verified on the hosting services; this package preparation does not itself publish anything.
