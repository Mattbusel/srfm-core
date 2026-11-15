# pragma pylint: disable=missing-docstring, invalid-name, too-many-locals
"""
SRFMBlackHole: a feature-transform example strategy.

Each bar is classified with a Minkowski-style "speed of light" cf:
    beta = |close - prev_close| / prev_close / cf
beta < 1 is a TIMELIKE (ordered) bar, beta >= 1 is SPACELIKE.

A "black-hole mass" accumulates over runs of TIMELIKE bars and decays on
SPACELIKE ones. A well is active when the mass clears a threshold after at
least ctl_min ordered bars in a row; bh_dir is the sign of the move over that
run. This strategy enters long when a long well forms and exits when it
collapses. It is a demonstration of the transform, not a tuned strategy:
backtest it on your own pairs and timeframe, and tune cf to your bar size
(roughly the typical absolute bar return you consider "orderly").

The math is inlined (about 60 lines, numpy only) so the file needs no extra
dependency. It is the same recurrence as the srfm-core package
(https://github.com/Mattbusel/srfm-core), which checks it against golden
vectors generated from the original srfm-lab implementation. With srfm-core
installed, populate_indicators can instead call
srfm_core.add_srfm_columns(dataframe).
"""

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IStrategy


def srfm_beta(close: np.ndarray, cf: float) -> np.ndarray:
    """|dC| / C_prev / cf per bar (NaN on the first bar)."""
    out = np.full(close.shape, np.nan)
    out[1:] = np.abs(close[1:] - close[:-1]) / (close[:-1] + 1e-9) / cf
    return out


def srfm_bh_mass(close: np.ndarray, cf: float = 0.001, bh_form: float = 1.5,
                 bh_collapse: float = 1.0, timelike_decay: float = 0.95,
                 spacelike_decay: float = 0.7, ctl_min: int = 5):
    """Black-hole mass per bar. Returns (mass, active, direction, ctl)."""
    n = close.shape[0]
    mass_out = np.zeros(n)
    active_out = np.zeros(n, dtype=bool)
    dir_out = np.zeros(n, dtype=np.int8)
    ctl_out = np.zeros(n, dtype=np.int64)
    mass, active, bdir, ctl, cum, prev_mass, reform = 0.0, False, 0, 0, 0.0, 0.0, 0
    for i in range(1, n):
        prev, c = float(close[i - 1]), float(close[i])
        br = (c - prev) / (prev + 1e-9)
        if abs(c - prev) / (prev + 1e-9) / cf < 1.0:          # TIMELIKE
            ctl += 1
            cum += br
            boost = min(2.0, 1.0 + ctl * 0.1)
            mass = mass * timelike_decay + abs(br) * 100.0 * boost
        else:                                                  # SPACELIKE
            ctl = 0
            mass *= spacelike_decay
        if cum > 0:
            bdir = 1
        elif cum < 0:
            bdir = -1
        was = active
        if not was:
            if 0 < reform < 15:                                # re-form memory
                mass += prev_mass * 0.5
                prev_mass = 0.0
            active = mass > bh_form and ctl >= ctl_min
        else:
            active = mass > bh_collapse and ctl >= ctl_min
        if active and not was:
            reform = 0
        elif was and not active:
            prev_mass, reform = mass, 1
        if not active:
            if reform > 0:
                reform += 1
            if ctl == 0:
                cum = 0.0
        mass_out[i], active_out[i], dir_out[i], ctl_out[i] = mass, active, bdir, ctl
    return mass_out, active_out, dir_out, ctl_out


class SRFMBlackHole(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    minimal_roi = {"0": 0.05, "240": 0.02, "720": 0}
    stoploss = -0.06
    trailing_stop = False
    process_only_new_candles = True
    startup_candle_count = 100

    # cf: the per-bar return treated as the "speed of light" (0.001 = 0.1%).
    cf = DecimalParameter(0.0005, 0.005, default=0.001, decimals=4, space="buy")
    bh_form = DecimalParameter(1.0, 2.5, default=1.5, decimals=2, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        close = dataframe["close"].to_numpy(dtype=float)
        cf = float(self.cf.value)
        mass, active, bdir, ctl = srfm_bh_mass(close, cf=cf, bh_form=float(self.bh_form.value))
        dataframe["srfm_beta"] = srfm_beta(close, cf)
        dataframe["bh_mass"] = mass
        dataframe["bh_active"] = active.astype(int)
        dataframe["bh_dir"] = bdir
        dataframe["bh_ctl"] = ctl
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        formed = (dataframe["bh_active"] == 1) & (dataframe["bh_active"].shift(1) == 0)
        dataframe.loc[formed & (dataframe["bh_dir"] == 1) & (dataframe["volume"] > 0),
                      ["enter_long", "enter_tag"]] = (1, "bh_formed_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        collapsed = (dataframe["bh_active"] == 0) & (dataframe["bh_active"].shift(1) == 1)
        dataframe.loc[collapsed, ["exit_long", "exit_tag"]] = (1, "bh_collapsed")
        return dataframe
