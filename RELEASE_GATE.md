# Release validation — version 0.1.0 (prepared 12 November 2026)

- Code copyright holder: Longqi Zhang (张龙齐); MIT license authorized on 12 November 2026.
- Public publication: expressly approved by the copyright holder on 12 November 2026.
- Source provenance: ten Python files examined in a bounded rights review; no identifiable incorporated third-party source requiring separate authorization was found. Dependency licensing remains separate.
- Native integration: the unmodified R2 code passed `python -B -m synthetic_test --out ...` in isolated Python 3.10.8 with pinned requirements including PyArrow 24.0.0. Exit code 0, 29 Python asserts, two `numpy.testing.assert_allclose` checks, two real Parquet datasets (3,150 rows × 10 columns each) with round-trip verification, and 11 nonempty decodable synthetic PNGs were recorded. See `SYNTHETIC_TEST_STATUS.md`.
- Scientific code and pinned requirements: identical to the independently verified R2p1 candidate; this v0.1.0 preparation changes public-facing release metadata/documentation only, not the Python implementation or synthetic test.
- Reproducibility: this is analysis code operating on separately authorized and preprocessed inputs; it is not a full raw-data-to-paper reproduction pipeline. Read `REPRODUCIBILITY_SCOPE.md`.
- Excluded: original or derived research data, underlying third-party products, internal QA and execution logs, author correspondence, masks, geometries, and confidential paths.

This file documents checks **completed before publication**. Its presence in a public repository is not itself evidence that GitHub or Zenodo publication has occurred. Cite only host links and DOIs that have actually been created and verified.
