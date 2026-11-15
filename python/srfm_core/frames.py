"""pandas front end over :mod:`srfm_core.kernels`.

Each function accepts a list, ndarray or pandas Series. A Series keeps its
index; anything else gets a RangeIndex.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import kernels as _k

__all__ = ["beta", "bh_mass", "geodesic", "hawking_z", "wilder_atr", "add_srfm_columns"]


def _index(x) -> Optional[pd.Index]:
    return x.index if isinstance(x, (pd.Series, pd.DataFrame)) else None


def beta(close, cf: float = 0.001) -> pd.Series:
    """Per-bar beta (``|dC| / C_prev / cf``). ``< 1`` is TIMELIKE."""
    return pd.Series(_k.beta(close, cf), index=_index(close), name="beta")


def bh_mass(
    close,
    cf: float = 0.001,
    bh_form: float = 1.5,
    bh_collapse: float = 1.0,
    timelike_decay: float = 0.95,
    spacelike_decay: float = 0.7,
    ctl_min: int = 5,
    **advanced,
) -> pd.DataFrame:
    """Black-hole mass. Returns a DataFrame with columns
    ``bh_mass, bh_active, bh_dir, ctl``.

    ``advanced`` accepts ``gain``, ``boost_step``, ``boost_cap``,
    ``reform_window`` and ``reform_frac`` (see docs/PARAMETERS.md).
    """
    unknown = set(advanced) - {"gain", "boost_step", "boost_cap", "reform_window", "reform_frac"}
    if unknown:
        raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")
    out = _k.bh_mass(close, cf=cf, bh_form=bh_form, bh_collapse=bh_collapse,
                     timelike_decay=timelike_decay, spacelike_decay=spacelike_decay,
                     ctl_min=ctl_min, **advanced)
    return pd.DataFrame(out, index=_index(close))


def geodesic(close, atr, window: int = 20, cf: float = 0.001) -> pd.DataFrame:
    """Rolling geodesic. Columns ``geo_dev, geo_slope, causal_frac``."""
    a = atr.to_numpy(dtype=float) if isinstance(atr, pd.Series) else atr
    return pd.DataFrame(_k.geodesic(close, a, window=window, cf=cf), index=_index(close))


def hawking_z(close, period: int = 20) -> pd.DataFrame:
    """Bollinger z and Hawking proxy. Columns ``hawking_z, hawking_t``."""
    return pd.DataFrame(_k.hawking_z(close, period=period), index=_index(close))


def wilder_atr(high, low, close, period: int = 14) -> pd.Series:
    """Wilder ATR with srfm-lab's warm-up (NaN on the first bar)."""
    def arr(x):
        return x.to_numpy(dtype=float) if isinstance(x, pd.Series) else x
    return pd.Series(_k.wilder_atr(arr(high), arr(low), arr(close), period),
                     index=_index(close), name="atr")


def add_srfm_columns(
    df: pd.DataFrame,
    cf: float = 0.001,
    window: int = 20,
    period: int = 20,
    atr_period: int = 14,
    prefix: str = "srfm_",
    close_col: str = "close",
    **bh_params,
) -> pd.DataFrame:
    """Add every SRFM column to ``df`` in place and return it.

    Made for a freqtrade ``populate_indicators``::

        from srfm_core import add_srfm_columns

        def populate_indicators(self, dataframe, metadata):
            return add_srfm_columns(dataframe, cf=0.001)

    Adds ``beta, bh_mass, bh_active, bh_dir, ctl, geo_dev, geo_slope,
    causal_frac, hawking_z, hawking_t, atr``, each named with ``prefix``
    (default ``srfm_``, so ``srfm_bh_mass``; pass ``prefix=""`` for bare
    names). ATR is Wilder ATR from ``high``/``low``/``close``; on a close-only
    frame the true range falls back to ``|dC|``.
    ``bh_params`` go to :func:`bh_mass`.
    """
    close = df[close_col].to_numpy(dtype=float)
    if "high" in df.columns and "low" in df.columns:
        atr = _k.wilder_atr(df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float),
                            close, atr_period)
    else:
        atr = _k.wilder_atr(close, close, close, atr_period)
    cols = {"beta": _k.beta(close, cf)}
    cols.update(_k.bh_mass(close, cf=cf, **bh_params))
    cols.update(_k.geodesic(close, atr, window=window, cf=cf))
    cols.update(_k.hawking_z(close, period=period))
    cols["atr"] = atr
    for name, values in cols.items():
        df[prefix + name] = values
    return df
