"""vectorbt indicators (optional extra: ``pip install "srfm-core[vbt]"``).

vectorbt is imported the first time an indicator is used, not when
``srfm_core`` is imported::

    import vectorbt as vbt
    from srfm_core.vbt import BHMass

    close = vbt.YFData.download("BTC-USD").get("Close")
    bh = BHMass.run(close, cf=[0.001, 0.002])      # parameter grid works
    entries = bh.active & ~bh.active.shift(1, fill_value=False)
    pf = vbt.Portfolio.from_signals(close, entries, ~bh.active)

Indicators: ``BHMass`` (outputs mass, active, dir, ctl), ``SRFMBeta``
(beta) and ``HawkingZ`` (z, t). Each column of a 2-D input is run
independently; NaN padding between columns is treated as gaps.
"""

from __future__ import annotations

import numpy as np

from . import kernels as _k

__all__ = ["BHMass", "SRFMBeta", "HawkingZ"]

_cache: dict = {}


def _per_column(fn, close2d, keys, dtypes):
    close2d = np.asarray(close2d, dtype=float)
    if close2d.ndim == 1:
        close2d = close2d[:, None]
    outs = [np.empty(close2d.shape, dtype=d) for d in dtypes]
    for j in range(close2d.shape[1]):
        res = fn(close2d[:, j])
        for o, k in zip(outs, keys):
            o[:, j] = res[k]
    return tuple(outs) if len(outs) > 1 else outs[0]


def _bh_apply(close, cf, bh_form, bh_collapse, timelike_decay, spacelike_decay, ctl_min):
    return _per_column(
        lambda c: _k.bh_mass(c, cf=cf, bh_form=bh_form, bh_collapse=bh_collapse,
                             timelike_decay=timelike_decay, spacelike_decay=spacelike_decay,
                             ctl_min=int(ctl_min)),
        close, ("bh_mass", "bh_active", "bh_dir", "ctl"),
        (float, bool, np.int8, np.int64))


def _beta_apply(close, cf):
    return _per_column(lambda c: {"beta": _k.beta(c, cf)}, close, ("beta",), (float,))


def _hz_apply(close, period):
    return _per_column(lambda c: _k.hawking_z(c, int(period)), close,
                       ("hawking_z", "hawking_t"), (float, float))


def _build(name: str):
    try:
        import vectorbt as vbt
    except ImportError as exc:  # pragma: no cover - exercised only without vectorbt
        raise ImportError(
            "srfm_core.vbt needs vectorbt: pip install \"srfm-core[vbt]\"") from exc
    if name == "BHMass":
        d = _k.BH_DEFAULTS
        return vbt.IndicatorFactory(
            class_name="BHMass", short_name="bh", input_names=["close"],
            param_names=["cf", "bh_form", "bh_collapse", "timelike_decay",
                         "spacelike_decay", "ctl_min"],
            output_names=["mass", "active", "dir", "ctl"],
        ).from_apply_func(
            _bh_apply, cf=d["cf"], bh_form=d["bh_form"], bh_collapse=d["bh_collapse"],
            timelike_decay=d["timelike_decay"], spacelike_decay=d["spacelike_decay"],
            ctl_min=d["ctl_min"])
    if name == "SRFMBeta":
        return vbt.IndicatorFactory(
            class_name="SRFMBeta", short_name="srfm_beta", input_names=["close"],
            param_names=["cf"], output_names=["beta"],
        ).from_apply_func(_beta_apply, cf=0.001)
    if name == "HawkingZ":
        return vbt.IndicatorFactory(
            class_name="HawkingZ", short_name="hawking", input_names=["close"],
            param_names=["period"], output_names=["z", "t"],
        ).from_apply_func(_hz_apply, period=20)
    raise AttributeError(name)


def __getattr__(name: str):
    if name in __all__:
        if name not in _cache:
            _cache[name] = _build(name)
        return _cache[name]
    raise AttributeError(f"module 'srfm_core.vbt' has no attribute {name!r}")
