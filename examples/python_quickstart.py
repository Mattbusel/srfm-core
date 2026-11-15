"""pandas quickstart: every SRFM column on an OHLCV frame.

    pip install srfm-core
    python examples/python_quickstart.py
"""

import numpy as np
import pandas as pd

import srfm_core as sc

# synthetic hourly bars: calm drift, a volatile stretch, calm again
rng = np.random.default_rng(42)
r = np.concatenate([rng.uniform(0.0003, 0.0009, 150) * np.where(rng.random(150) < 0.3, -1, 1),
                    rng.normal(0, 0.004, 50),
                    rng.uniform(0.0003, 0.0009, 150) * np.where(rng.random(150) < 0.7, -1, 1)])
close = 100 * np.exp(np.cumsum(r))
df = pd.DataFrame({"open": close, "high": close * 1.0006, "low": close * 0.9994,
                   "close": close, "volume": 1.0},
                  index=pd.date_range("2025-01-01", periods=close.size, freq="h"))

sc.add_srfm_columns(df, cf=0.001)
print(df[["close", "srfm_beta", "srfm_bh_mass", "srfm_bh_active", "srfm_bh_dir",
          "srfm_geo_slope", "srfm_hawking_z"]].iloc[::25].round(4).to_string())

formed = df["srfm_bh_active"] & ~df["srfm_bh_active"].shift(1, fill_value=False)
print("\nwells formed:", int(formed.sum()),
      "| long:", int((formed & (df["srfm_bh_dir"] == 1)).sum()),
      "| short:", int((formed & (df["srfm_bh_dir"] == -1)).sum()))

# streaming, one bar at a time (live feeds)
clf, det = sc.MinkowskiClassifier(cf=0.001), sc.BlackHoleDetector()
prev = None
for c in close:
    bit = clf.update(c)
    if prev is not None:
        det.update(bit, c, prev)
    prev = c
print("streaming final mass:", round(det.bh_mass, 6), "== batch:",
      round(float(df["srfm_bh_mass"].iloc[-1]), 6))
