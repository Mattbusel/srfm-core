"""vectorbt: BH mass as an IndicatorFactory indicator, with a cf grid.

    pip install "srfm-core[vbt]"
    python examples/vectorbt_bhmass.py
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

from srfm_core.vbt import BHMass

rng = np.random.default_rng(0)
n = 2000
r = rng.uniform(0.0003, 0.0009, n) * np.where(rng.random(n) < 0.45, -1, 1)
shock = rng.random(n) < 0.03          # occasional SPACELIKE bars
r[shock] = rng.normal(0, 0.006, shock.sum())
close = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.date_range("2024-01-01", periods=n, freq="h"))

bh = BHMass.run(close, cf=[0.0006, 0.0008, 0.001], param_product=True)
entries = bh.active & ~bh.active.shift(1, fill_value=False) & (bh.dir == 1)
exits = ~bh.active
pf = vbt.Portfolio.from_signals(close, entries, exits, fees=0.0005)
print(pf.total_return().rename("total_return"))
print(pf.trades.count().rename("trades"))
