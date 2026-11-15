# srfm-core

The SRFM bar transforms from [srfm-lab](https://github.com/Mattbusel/srfm-lab), as a small package you can drop into a pandas, vectorbt or freqtrade workflow. NumPy and pandas only.

| Function | What it gives you per bar |
|---|---|
| `beta(close, cf=0.001)` | `abs(dC) / C_prev / cf`. Below 1 is a TIMELIKE (ordered) bar, 1 or more is SPACELIKE. |
| `bh_mass(close, cf, bh_form=1.5, bh_collapse=1.0, timelike_decay=0.95, spacelike_decay=0.7, ctl_min=5)` | DataFrame `bh_mass, bh_active, bh_dir, ctl`: a decaying accumulator of ordered momentum and whether a "well" is active. |
| `geodesic(close, atr, window=20)` | DataFrame `geo_dev, geo_slope, causal_frac`: rolling log-linear fit and the share of lags with a TIMELIKE interval. |
| `hawking_z(close, period=20)` | DataFrame `hawking_z, hawking_t`: Bollinger z and LARSA's `z * (z - z_prev)` proxy. |
| `add_srfm_columns(df)` | All of the above added to an OHLCV frame as `srfm_*` columns (made for freqtrade `populate_indicators`). |

```python
import srfm_core as sc

df = sc.add_srfm_columns(ohlcv)            # srfm_bh_mass, srfm_bh_active, ...
bh = sc.bh_mass(ohlcv["close"], cf=0.002)  # tune cf to your bar size
```

vectorbt (optional, `pip install "srfm-core[vbt]"`):

```python
from srfm_core.vbt import BHMass
bh = BHMass.run(close, cf=[0.001, 0.002])  # parameter grids work
```

Streaming classes (`MinkowskiClassifier`, `BlackHoleDetector`, `GeodesicAnalyzer`, `GravitationalLens`, `HawkingMonitor`) keep the `update()` bar-by-bar API from srfm-lab.

Every LARSA constant is a documented parameter with LARSA's value as the default. The batch functions are checked against `golden/golden_vectors.json`, generated from srfm-lab's untouched `lib/srfm_core.py`; the Rust crate [`srfm-core`](https://crates.io/crates/srfm-core) is checked against the same file.

These are feature transforms, not a strategy. Nothing here claims a trading edge.

Docs: https://github.com/Mattbusel/srfm-core/tree/main/docs. License: MIT.
