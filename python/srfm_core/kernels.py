"""NumPy-only batch kernels. No pandas import.

Every function takes 1-D float arrays and returns ndarrays (or a dict of
ndarrays). Shared conventions, identical in the Rust crate and checked by
``golden/golden_vectors.json``:

* Non-finite closes (NaN, inf) are gaps. A gap bar outputs NaN (``bh_active``
  False, ``bh_dir`` 0, ``ctl`` 0), does not touch the state, and the next
  finite bar is compared with the last finite close.
* Finite closes must be > 0, otherwise ``ValueError``.
* Warm-up outputs are NaN, never a fake 0.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

__all__ = ["beta", "bh_mass", "geodesic", "hawking_z", "wilder_atr", "BH_DEFAULTS"]

_EPS = 1e-9

#: LARSA's BH constants, now parameters. See docs/PARAMETERS.md.
BH_DEFAULTS = dict(
    cf=0.001,
    bh_form=1.5,
    bh_collapse=1.0,
    timelike_decay=0.95,
    spacelike_decay=0.7,
    ctl_min=5,
    gain=100.0,
    boost_step=0.1,
    boost_cap=2.0,
    reform_window=15,
    reform_frac=0.5,
)


def _as_close(close) -> Tuple[np.ndarray, np.ndarray]:
    c = np.asarray(close, dtype=float)
    if c.ndim != 1:
        raise ValueError("close must be 1-D")
    finite = np.isfinite(c)
    if np.any(c[finite] <= 0):
        raise ValueError("close prices must be > 0 (non-finite values are treated as gaps)")
    return c, finite


def _check_cf(cf: float) -> None:
    if not (cf > 0 and math.isfinite(cf)):
        raise ValueError("cf must be a finite number > 0")


def beta(close, cf: float = 0.001) -> np.ndarray:
    """``|C_t - C_{t-1}| / C_{t-1} / cf`` per bar. NaN on the first finite bar and on gaps.

    ``beta < 1`` is a TIMELIKE bar, ``beta >= 1`` SPACELIKE.
    """
    _check_cf(cf)
    c, finite = _as_close(close)
    out = np.full(c.shape, np.nan)
    idx = np.flatnonzero(finite)
    if idx.size >= 2:
        v = c[idx]
        out[idx[1:]] = np.abs(v[1:] - v[:-1]) / (v[:-1] + _EPS) / cf
    return out


def bh_mass(
    close,
    cf: float = 0.001,
    bh_form: float = 1.5,
    bh_collapse: float = 1.0,
    timelike_decay: float = 0.95,
    spacelike_decay: float = 0.7,
    ctl_min: int = 5,
    gain: float = 100.0,
    boost_step: float = 0.1,
    boost_cap: float = 2.0,
    reform_window: int = 15,
    reform_frac: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Black-hole mass per bar.

    Returns a dict with ``bh_mass`` (float), ``bh_active`` (bool), ``bh_dir``
    (int8, +1/-1/0) and ``ctl`` (int64, consecutive TIMELIKE bars). See
    :class:`srfm_core.BlackHoleDetector` for the recurrence.
    """
    _check_cf(cf)
    c, finite = _as_close(close)
    n = c.shape[0]
    mass_out = np.full(n, np.nan)
    active_out = np.zeros(n, dtype=bool)
    dir_out = np.zeros(n, dtype=np.int8)
    ctl_out = np.zeros(n, dtype=np.int64)

    mass = 0.0
    active = False
    bdir = 0
    ctl = 0
    cum = 0.0
    prev_mass = 0.0
    reform = 0
    prev = None
    for i in np.flatnonzero(finite).tolist():
        x = float(c[i])
        if prev is not None:
            diff = x - prev
            timelike = abs(diff) / (prev + _EPS) / cf < 1.0
            br = diff / (prev + _EPS)
            if timelike:
                ctl += 1
                cum += br
                sb = min(boost_cap, 1.0 + ctl * boost_step)
                mass = mass * timelike_decay + abs(br) * gain * sb
            else:
                ctl = 0
                mass *= spacelike_decay
            if cum > 0:
                bdir = 1
            elif cum < 0:
                bdir = -1
            was = active
            if not was:
                if 0 < reform < reform_window:
                    mass += prev_mass * reform_frac
                    prev_mass = 0.0
                active = mass > bh_form and ctl >= ctl_min
            else:
                active = mass > bh_collapse and ctl >= ctl_min
            if active and not was:
                reform = 0
            elif was and not active:
                prev_mass = mass
                reform = 1
            if not active:
                if reform > 0:
                    reform += 1
                if ctl == 0:
                    cum = 0.0
        mass_out[i] = mass
        active_out[i] = active
        dir_out[i] = bdir
        ctl_out[i] = ctl
        prev = x
    return {"bh_mass": mass_out, "bh_active": active_out, "bh_dir": dir_out, "ctl": ctl_out}


def geodesic(close, atr, window: int = 20, cf: float = 0.001) -> Dict[str, np.ndarray]:
    """Rolling log-linear "geodesic" over ``window`` finite closes.

    Returns ``geo_dev`` = tanh((C - fitted C) / atr), ``geo_slope`` =
    100 * slope of log price per bar, and ``causal_frac`` = share of lags
    1..window-1 with a TIMELIKE interval. NaN until ``window`` finite closes.
    ``atr`` is aligned with ``close``; NaN ATR gives NaN ``geo_dev`` only.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    _check_cf(cf)
    c, finite = _as_close(close)
    a = np.asarray(atr, dtype=float)
    if a.shape != c.shape:
        raise ValueError("atr must have the same length as close")
    n = c.shape[0]
    dev = np.full(n, np.nan)
    slope_out = np.full(n, np.nan)
    causal = np.full(n, np.nan)
    idx = np.flatnonzero(finite)
    m = idx.size
    if m >= window:
        v = c[idx]
        w = np.lib.stride_tricks.sliding_window_view(v, window)  # oldest -> newest
        lp = np.log(w + _EPS)
        x = np.arange(window, dtype=float)
        nn = float(window)
        sx = x.sum()
        sxx = float(np.dot(x, x))
        slp = lp.sum(axis=1)
        sxlp = lp @ x
        slope = (nn * sxlp - sx * slp) / (nn * sxx - sx * sx + _EPS)
        intercept = (slp - slope * sx) / nn
        geo_p = np.exp(slope * (window - 1) + intercept)
        last = w[:, -1]
        tgt = idx[window - 1:]
        dev[tgt] = np.tanh((last - geo_p) / (a[tgt] + _EPS))
        slope_out[tgt] = slope * 100
        k = np.arange(1, window, dtype=float)
        past = w[:, :-1][:, ::-1]  # column j = lag j+1
        dp = np.abs(last[:, None] - past) / (past + _EPS)
        ds2 = -(cf * k) ** 2 + dp * dp
        causal[tgt] = (ds2 < 0).sum(axis=1) / (window - 1)
    return {"geo_dev": dev, "geo_slope": slope_out, "causal_frac": causal}


def hawking_z(close, period: int = 20) -> Dict[str, np.ndarray]:
    """Bollinger z-score and the LARSA Hawking-temperature proxy.

    ``hawking_z = (C - mean) / std`` over the last ``period`` finite closes
    (population std, current bar included) and
    ``hawking_t = z * (z - z_prev)``, where ``z_prev`` is the previous defined
    z (0 before the first). NaN in warm-up and where std == 0.
    """
    if period < 2:
        raise ValueError("period must be >= 2")
    c, finite = _as_close(close)
    n = c.shape[0]
    z_out = np.full(n, np.nan)
    t_out = np.full(n, np.nan)
    idx = np.flatnonzero(finite)
    if idx.size >= period:
        v = c[idx]
        w = np.lib.stride_tricks.sliding_window_view(v, period)
        mid = w.sum(axis=1) / period
        std = w.std(axis=1)
        last = w[:, -1]
        tgt = idx[period - 1:]
        ok = std > 0
        z = np.full(last.shape, np.nan)
        z[ok] = (last[ok] - mid[ok]) / std[ok]
        zv = z[ok]
        pz = np.concatenate(([0.0], zv[:-1]))
        t = np.full(last.shape, np.nan)
        t[ok] = zv * (zv - pz)
        z_out[tgt] = z
        t_out[tgt] = t
    return {"hawking_z": z_out, "hawking_t": t_out}


def wilder_atr(high, low, close, period: int = 14) -> np.ndarray:
    """Wilder ATR (mean of true ranges for the first ``period`` bars, then
    Wilder smoothing). NaN on the first finite close and on gaps."""
    if period < 1:
        raise ValueError("period must be >= 1")
    c, finite = _as_close(close)
    h = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    if h.shape != c.shape or lo.shape != c.shape:
        raise ValueError("high, low and close must have the same length")
    out = np.full(c.shape, np.nan)
    prev = None
    val = 0.0
    k = 0
    for i in np.flatnonzero(finite).tolist():
        x = float(c[i])
        if prev is not None:
            tr = max(float(h[i]) - float(lo[i]), abs(float(h[i]) - prev), abs(float(lo[i]) - prev))
            if k < period:
                val = (val * k + tr) / (k + 1)
            else:
                val = (val * (period - 1) + tr) / period
            k += 1
            out[i] = val
        prev = x
    return out
