# Provenance

srfm-core is carved out of [srfm-lab](https://github.com/Mattbusel/srfm-lab) (MIT, Matthew Busel), commit `8cbfcb0551d44e3ed1c20bdae9dc805d61521c62`.

| srfm-core | srfm-lab source | What changed |
|---|---|---|
| `python/srfm_core/streaming.py` | `lib/srfm_core.py` (`MinkowskiClassifier`, `BlackHoleDetector`, `GeodesicAnalyzer`, `GravitationalLens`, `HawkingMonitor`, `_ATR`) | LARSA literals became constructor parameters with the same defaults; lab batch wrappers removed; `WilderATR` returns NaN instead of 0 on its first bar. Tested to be identical to the original on random inputs. |
| `python/srfm_core/kernels.py`, `frames.py` | new | numpy/pandas batch functions over the same recurrences; gap handling and NaN warm-up made explicit. |
| `python/srfm_core/vbt.py` | new | vectorbt `IndicatorFactory` wrappers. |
| `src/lib.rs` | `crates/tick-backtest/src/bh_physics.rs` (structure), `lib/srfm_core.py` (math) | Rust port of the Python reference, replacing the gamma-accumulation rule in `bh_physics.rs`, which had drifted (see DRIFT.md). |
| `bindings/python-rs` | `crates/larsa-core` | The pyo3 layer split from the math: no formulas in the binding crate. The backtest, Monte Carlo and sweep parts of `larsa-core` are not carried over. |
| `golden/reference/srfm_core_lab.py` | `lib/srfm_core.py` | Byte-for-byte copy, used only to generate the golden vectors. |
| `python/tests/`, `tests/` | ideas from `tests/test_srfm_core.py`, `tests/test_bh_physics.py` | Rewritten around the golden vectors and reference-equivalence checks. |

Left behind on purpose: LEAN/QuantConnect glue, hard-coded paths, notebooks, the multi-timeframe engine, the backtester and every performance claim. In particular the QuantConnect backtest figure in srfm-lab's RESULTS.md is not replicated (RESULTS.md says so itself), so this package describes these as feature transforms only.
