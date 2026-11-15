"""Streaming (bar-by-bar) SRFM classes.

These are the classes from srfm-lab ``lib/srfm_core.py`` (commit 8cbfcb0) with
two changes and no change to the math at default settings:

* every LARSA constant is now a documented constructor parameter whose
  default is the value LARSA shipped with;
* lab-only batch wrappers (``detect``, ``analyze``, ``compute_mu``,
  ``compute_temperature``, ``classify``) are gone; use the functions in
  :mod:`srfm_core` instead, which do not have the one-bar lag of ``detect``.

The golden vectors in ``golden/golden_vectors.json`` are produced from the
untouched srfm-lab file and these classes are tested against them.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

__all__ = [
    "MinkowskiClassifier",
    "BlackHoleDetector",
    "GeodesicAnalyzer",
    "GravitationalLens",
    "HawkingMonitor",
    "WilderATR",
    "TIMELIKE",
    "SPACELIKE",
    "UNKNOWN",
]

TIMELIKE = "TIMELIKE"
SPACELIKE = "SPACELIKE"
UNKNOWN = "UNKNOWN"

_EPS = 1e-9  # denominator guard used throughout LARSA; kept for bit-exactness


class MinkowskiClassifier:
    """Classify each bar as TIMELIKE or SPACELIKE.

    ``beta = |close_t - close_{t-1}| / close_{t-1} / cf``.
    ``beta < 1`` is TIMELIKE (an ordered move smaller than the "speed of
    light" ``cf``), ``beta >= 1`` is SPACELIKE.

    Also tracks ``tl_confirm`` (consecutive TIMELIKE bars, capped at
    ``tl_cap``) and ``proper_time`` (sum of 1/gamma with
    ``v = min(0.99, |dC/C| / max_vol)``).

    Parameters
    ----------
    cf : float, default 0.001
        Speed-of-light constant as a fraction of price per bar. A bar whose
        absolute return is below ``cf`` is TIMELIKE. Tune per instrument and
        bar size; LARSA used 0.001 on hourly index futures.
    max_vol : float, default 0.01
        Return at which proper-time velocity saturates (v capped at 0.99).
    tl_cap : int, default 3
        Cap on the ``tl_confirm`` counter.
    """

    def __init__(self, cf: float = 0.001, max_vol: float = 0.01, tl_cap: int = 3):
        if cf <= 0:
            raise ValueError("cf must be > 0")
        self.cf = cf
        self.max_vol = max_vol
        self.tl_cap = tl_cap
        self.beta: float = 0.0
        self.bit: str = UNKNOWN
        self.tl_confirm: int = 0
        self.proper_time: float = 0.0
        self._prev_close: Optional[float] = None

    def update(self, close: float) -> str:
        """Feed one close. Returns ``"TIMELIKE"``, ``"SPACELIKE"`` or ``"UNKNOWN"`` (first bar)."""
        if self._prev_close is None or self._prev_close <= 0:
            self._prev_close = close
            return self.bit
        price_diff = abs(close - self._prev_close)
        self.beta = price_diff / (self._prev_close + _EPS) / self.cf
        self.bit = TIMELIKE if self.beta < 1.0 else SPACELIKE
        if self.bit == TIMELIKE:
            self.tl_confirm = min(self.tl_confirm + 1, self.tl_cap)
        else:
            self.tl_confirm = 0
        hv = price_diff / (self._prev_close + _EPS)
        v = min(0.99, hv / self.max_vol)
        gamma = 1.0 / math.sqrt(max(1e-9, 1.0 - v * v))
        self.proper_time += 1.0 / gamma
        self._prev_close = close
        return self.bit

    def reset_proper_time(self) -> None:
        self.proper_time = 0.0

    def reset(self) -> None:
        self.__init__(self.cf, self.max_vol, self.tl_cap)

    @property
    def is_timelike(self) -> bool:
        return self.bit == TIMELIKE


class BlackHoleDetector:
    """Black-hole (BH) mass: a decaying accumulator of ordered momentum.

    Per bar, with ``br = (close - prev_close) / prev_close``:

    * TIMELIKE: ``ctl += 1``; ``sb = min(boost_cap, 1 + ctl * boost_step)``;
      ``mass = mass * timelike_decay + |br| * gain * sb``
    * otherwise: ``ctl = 0``; ``mass *= spacelike_decay``

    A well *forms* when ``mass > bh_form`` and ``ctl >= ctl_min``; it stays
    active while ``mass > bh_collapse`` and ``ctl >= ctl_min``. ``bh_dir`` is
    the sign of the return accumulated over the current TIMELIKE run.
    Within ``reform_window`` bars of a collapse, the next formation attempt
    adds ``reform_frac`` of the mass at collapse (once).

    All defaults are the values LARSA shipped with.
    """

    def __init__(
        self,
        bh_form: float = 1.5,
        bh_collapse: float = 1.0,
        bh_decay: float = 0.95,
        *,
        spacelike_decay: float = 0.7,
        ctl_min: int = 5,
        gain: float = 100.0,
        boost_step: float = 0.1,
        boost_cap: float = 2.0,
        reform_window: int = 15,
        reform_frac: float = 0.5,
    ):
        self.bh_form = bh_form
        self.bh_collapse = bh_collapse
        self.bh_decay = bh_decay  # TIMELIKE decay; name kept from srfm-lab
        self.spacelike_decay = spacelike_decay
        self.ctl_min = ctl_min
        self.gain = gain
        self.boost_step = boost_step
        self.boost_cap = boost_cap
        self.reform_window = reform_window
        self.reform_frac = reform_frac
        self.reset()
        self.well_events: List[dict] = []

    @property
    def timelike_decay(self) -> float:
        return self.bh_decay

    def update(self, bit: str, close: float, prev_close: float) -> bool:
        """Feed one bar (its classification and the two closes). Returns ``bh_active``."""
        br = (close - prev_close) / (prev_close + _EPS)
        if bit == TIMELIKE:
            self.ctl += 1
            self.cum_disp += br
            sb = min(self.boost_cap, 1.0 + self.ctl * self.boost_step)
            self.bh_mass = self.bh_mass * self.bh_decay + abs(br) * self.gain * sb
        else:
            self.ctl = 0
            self.bh_mass *= self.spacelike_decay

        if self.cum_disp > 0:
            self.bh_dir = 1
        elif self.cum_disp < 0:
            self.bh_dir = -1

        prev_active = self.bh_active
        if not prev_active:
            if 0 < self.reform_bars < self.reform_window:
                self.bh_mass += self.prev_bh_mass * self.reform_frac
                self.prev_bh_mass = 0.0
            self.bh_active = self.bh_mass > self.bh_form and self.ctl >= self.ctl_min
        else:
            self.bh_active = self.bh_mass > self.bh_collapse and self.ctl >= self.ctl_min

        if self.bh_active and not prev_active:
            self.reform_bars = 0
            self.well_events.append(
                {"event": "formed", "mass": self.bh_mass, "dir": self.bh_dir, "ctl": self.ctl})
        elif not self.bh_active and prev_active:
            self.prev_bh_mass = self.bh_mass
            self.reform_bars = 1
            self.well_events.append({"event": "collapsed", "mass": self.bh_mass})

        if not self.bh_active:
            if self.reform_bars > 0:
                self.reform_bars += 1
            if self.ctl == 0:
                self.cum_disp = 0.0
        return self.bh_active

    def reset(self) -> None:
        self.bh_mass = 0.0
        self.bh_active = False
        self.bh_dir = 0
        self.ctl = 0
        self.cum_disp = 0.0
        self.prev_bh_mass = 0.0
        self.reform_bars = 0
        self.well_events = []


class GeodesicAnalyzer:
    """Rolling log-linear regression ("geodesic") and Minkowski causal fraction.

    Over the last ``window`` closes:

    * ``geo_slope = 100 * slope`` of ``log(close)`` on bar index;
    * ``geo_dev = tanh((close - fitted_close) / atr)``;
    * ``causal_frac`` = share of lags ``k = 1..window-1`` whose interval
      ``ds2 = -(cf*k)^2 + (|C_t - C_{t-k}| / C_{t-k})^2`` is negative.

    ``update`` returns ``(geo_dev, geo_slope, causal_frac, rapidity)`` and
    ``(0, 0, 1, 0)`` during warm-up, exactly as in srfm-lab. ``rapidity`` is
    the LARSA production formula, which is identically 0 (its denominator
    guard never passes); it is kept only for drop-in compatibility.
    """

    def __init__(self, cf: float = 0.001, window: int = 20):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.cf = cf
        self.window = window
        self._closes: Deque[float] = deque(maxlen=window)

    def update(self, close: float, atr: float) -> Tuple[float, float, float, float]:
        self._closes.appendleft(close)
        if len(self._closes) < self.window:
            return 0.0, 0.0, 1.0, 0.0
        prices = list(reversed(list(self._closes)))
        lp = np.log(np.array(prices) + _EPS)
        x = np.arange(self.window, dtype=float)
        n = float(self.window)
        sx = x.sum()
        slp = lp.sum()
        sxx = np.dot(x, x)
        sxlp = np.dot(x, lp)
        slope = (n * sxlp - sx * slp) / (n * sxx - sx * sx + _EPS)
        intercept = (slp - slope * sx) / n
        geo_p = float(np.exp(slope * (self.window - 1) + intercept))
        geo_dev = float(np.tanh((close - geo_p) / (atr + _EPS)))
        geo_slope = float(slope * 100)

        closes_list = list(self._closes)
        cc = 0
        for k in range(1, self.window):
            pp = closes_list[k]
            dp = abs(close - pp) / (pp + _EPS)
            if -(self.cf * k) ** 2 + dp * dp < 0:
                cc += 1
        causal_frac = cc / (self.window - 1)

        e = (close - closes_list[self.window - 1]) / (closes_list[self.window - 1] + _EPS)
        denom = e - e + _EPS
        num = e + e
        if abs(denom) > _EPS and num / denom > 0:
            rapidity = float(np.tanh(0.5 * np.log(num / denom + _EPS)))
        else:
            rapidity = 0.0
        return geo_dev, geo_slope, causal_frac, rapidity

    @property
    def ready(self) -> bool:
        return len(self._closes) >= self.window

    def reset(self) -> None:
        self._closes.clear()


class GravitationalLens:
    """Lensing amplification ``mu`` from the distance to the TIMELIKE-bar VWAP.

    ``M = ctl + (1 if TIMELIKE)``, ``R_E = sqrt(M)``,
    ``r = |close - VWAP_TL| / atr``, ``mu = 1 + R_E / (r + R_E)``;
    ``mu = max(0.3, M / 3)`` when ``M < 2``.
    """

    def __init__(self):
        self._closes_tl: List[float] = []
        self._vols_tl: List[float] = []
        self.mu: float = 1.0

    def update(self, close: float, volume: float, bit: str, ctl: int, atr: float) -> float:
        if bit == TIMELIKE:
            self._closes_tl.append(close)
            self._vols_tl.append(volume)
        else:
            self._closes_tl = []
            self._vols_tl = []
        m = float(ctl + (1 if bit == TIMELIKE else 0))
        r_e = math.sqrt(m + _EPS)
        if m < 2.0:
            self.mu = max(0.3, m / 3.0)
        elif self._closes_tl:
            sv = sum(self._vols_tl)
            if sv > 0:
                vwap = sum(p * v for p, v in zip(self._closes_tl, self._vols_tl)) / sv
            else:
                vwap = sum(self._closes_tl) / len(self._closes_tl)
            r = abs(close - vwap) / (atr + _EPS)
            self.mu = float(1.0 + r_e / (r + r_e))
        else:
            self.mu = 1.0
        return self.mu


class HawkingMonitor:
    """Hawking-temperature proxy: ``ht = z * (z - z_prev)`` on a Bollinger z-score.

    ``z = (close - bb_middle) / std``. LARSA treated ``ht > hot`` (1.8) as a
    hot well (avoid long entry) and ``ht < inverted`` (-1.5) as a boost.
    """

    def __init__(self, hot: float = 1.8, inverted: float = -1.5):
        self.hot = hot
        self.inverted = inverted
        self.ht: float = 0.0
        self.z: float = 0.0
        self._pz: float = 0.0

    def update(self, close: float, bb_middle: float, std: float) -> float:
        if std <= 0:
            return self.ht
        z = (close - bb_middle) / std
        self.ht = z * (z - self._pz)
        self._pz = z
        self.z = z
        return self.ht

    @property
    def is_hot(self) -> bool:
        return self.ht > self.hot

    @property
    def is_inverted(self) -> bool:
        return self.ht < self.inverted

    def reset(self) -> None:
        self.ht = 0.0
        self.z = 0.0
        self._pz = 0.0


class WilderATR:
    """Wilder ATR as used by srfm-lab: simple mean of true ranges for the first
    ``period`` bars, Wilder smoothing after. The first bar has no true range
    and returns ``nan`` (srfm-lab returned 0.0)."""

    def __init__(self, period: int = 14):
        self.period = period
        self.reset()

    def update(self, high: float, low: float, close: float) -> float:
        if self._prev_c is None:
            self._prev_c = close
            return float("nan")
        tr = max(high - low, abs(high - self._prev_c), abs(low - self._prev_c))
        if self._n < self.period:
            self.val = (self.val * self._n + tr) / (self._n + 1)
        else:
            self.val = (self.val * (self.period - 1) + tr) / self.period
        self._n += 1
        self._prev_c = close
        return self.val

    def reset(self) -> None:
        self._prev_c: Optional[float] = None
        self.val = 0.0
        self._n = 0
