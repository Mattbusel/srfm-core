# Changelog

All notable changes to this project are documented here. Versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-15

First release, carved out of [srfm-lab](https://github.com/Mattbusel/srfm-lab) (commit `8cbfcb0`).

### Added
- Python package `srfm-core`: batch functions `beta`, `bh_mass`, `geodesic`, `hawking_z`, `wilder_atr` (numpy kernels in `srfm_core.kernels`, pandas front end at the top level) and `add_srfm_columns` for freqtrade-style OHLCV frames.
- Streaming classes from srfm-lab's `lib/srfm_core.py` (`MinkowskiClassifier`, `BlackHoleDetector`, `GeodesicAnalyzer`, `GravitationalLens`, `HawkingMonitor`) plus `WilderATR`, with every LARSA constant as a documented parameter (defaults unchanged).
- `srfm_core.vbt`: vectorbt `BHMass`, `SRFMBeta`, `HawkingZ` indicators (optional extra, imported lazily).
- Rust crate `srfm-core`: slice functions and streaming structs for the same transforms; `ta` feature with `ta::Next` / `ta::Reset` impls. No default dependencies, no unsafe code.
- `bindings/python-rs`: thin pyo3 layer over the crate (unpublished), replacing the physics half of srfm-lab's `larsa-core`.
- `golden/golden_vectors.json`: six cases generated from srfm-lab's untouched `lib/srfm_core.py`, checked by both test suites.
- `golden/drift/compare_lab_variants.py` and `docs/DRIFT.md`: the other BH-mass copies inside srfm-lab scored against the vectors.
- `examples/freqtrade/SRFMBlackHole.py`: a dependency-free freqtrade strategy with the recurrence inlined and tested against the vectors.

### Changed (relative to srfm-lab)
- Batch outputs use NaN for warm-up and gaps instead of 0.
- The batch BH path no longer has the one-bar lag of srfm-lab's `BlackHoleDetector.detect()` (which skipped the classifier on the first bar).
- LARSA's geodesic "rapidity" (identically 0 as shipped) is not part of the batch API or the Rust crate.

[0.1.0]: https://github.com/Mattbusel/srfm-core/releases/tag/v0.1.0
