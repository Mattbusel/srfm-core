# Using srfm-core in other ecosystems

## vectorbt

No upstream change is needed. `srfm_core.vbt` builds indicators with `vbt.IndicatorFactory`, so any vectorbt notebook can run them, including parameter grids:

```python
from srfm_core.vbt import BHMass, SRFMBeta, HawkingZ

bh = BHMass.run(close, cf=[0.0006, 0.0008, 0.001], param_product=True)
entries = bh.active & ~bh.active.shift(1, fill_value=False) & (bh.dir == 1)
pf = vbt.Portfolio.from_signals(close, entries, ~bh.active)
```

vectorbt is an optional extra (`pip install "srfm-core[vbt]"`) and is imported only when an indicator is first used. Tested with vectorbt 1.2.

## freqtrade

Two ways in:

* With the package: `return srfm_core.add_srfm_columns(dataframe)` in `populate_indicators` adds `srfm_beta, srfm_bh_mass, srfm_bh_active, srfm_bh_dir, srfm_ctl, srfm_geo_dev, srfm_geo_slope, srfm_causal_frac, srfm_hawking_z, srfm_hawking_t, srfm_atr`.
* Without a dependency: `examples/freqtrade/SRFMBlackHole.py` is a single self-contained `IStrategy` that inlines the beta and mass recurrence (numpy only). Its inlined math is tested against the golden vectors in this repo, so it cannot silently drift from the package.

## Rust `ta` crate

Enable the `ta` feature: `Beta`, `BhMass`, `HawkingZ`, `Geodesic` and `WilderAtr` implement `ta::Next` and `ta::Reset` (and `ta::Period` where it applies).

## pandas-ta and other indicator libraries

Libraries that only accept indicators with a published reference (for example pandas-ta-classic's reference-port rule) need a citable definition first. `CITATION.cff` and `.zenodo.json` are in place so a Zenodo DOI can be minted from a GitHub release; `docs/PARAMETERS.md` and `docs/GOLDEN.md` are the definition and the test oracle.

## Other languages

Any port (C++, Julia, C, WASM, R) should load `golden/golden_vectors.json` in its test suite. See [GOLDEN.md](GOLDEN.md).
