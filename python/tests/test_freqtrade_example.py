"""The self-contained freqtrade example inlines the BH math; keep it in step
with the package and the golden vectors. freqtrade itself is stubbed."""

import importlib.util
import sys
import types

import numpy as np
import pandas as pd
import pytest

import srfm_core as sc
from conftest import ROOT, to_array

STRATEGY = ROOT / "examples" / "freqtrade" / "SRFMBlackHole.py"


@pytest.fixture(scope="module")
def strategy_module():
    if not STRATEGY.exists():
        pytest.skip("examples/ not available")
    real = sys.modules.get("freqtrade.strategy")
    if real is None:
        class _Param:
            def __init__(self, low, high, default, **kw):
                self.value = default

        class _IStrategy:
            def __init__(self, config=None):
                self.config = config or {}

        ft = types.ModuleType("freqtrade")
        fts = types.ModuleType("freqtrade.strategy")
        fts.IStrategy = _IStrategy
        fts.DecimalParameter = _Param
        sys.modules.setdefault("freqtrade", ft)
        sys.modules["freqtrade.strategy"] = fts
    spec = importlib.util.spec_from_file_location("SRFMBlackHole", STRATEGY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_inlined_math_matches_golden(strategy_module, golden):
    for case in golden["cases"]:
        if any(v is None for v in case["close"]):
            continue  # the inlined copy assumes gap-free exchange candles
        p = case["params"]
        close = to_array(case["close"])
        mass, active, bdir, ctl = strategy_module.srfm_bh_mass(
            close, cf=p["cf"], bh_form=p["bh_form"], bh_collapse=p["bh_collapse"],
            timelike_decay=p["timelike_decay"], spacelike_decay=p["spacelike_decay"],
            ctl_min=p["ctl_min"])
        exp = case["expected"]
        np.testing.assert_allclose(mass, to_array(exp["bh_mass"]), rtol=1e-12, atol=1e-15)
        assert active.tolist() == exp["bh_active"], case["name"]
        assert bdir.tolist() == exp["bh_dir"], case["name"]
        assert ctl.tolist() == exp["ctl"], case["name"]
        np.testing.assert_allclose(strategy_module.srfm_beta(close, p["cf"]),
                                   sc.kernels.beta(close, p["cf"]), rtol=1e-12, equal_nan=True)


def test_strategy_signals(strategy_module):
    n = 300
    rng = np.random.default_rng(1)
    r = rng.uniform(0.0004, 0.0009, n) * np.where(rng.random(n) < 0.3, -1, 1)
    r[::40] = 0.006  # a SPACELIKE shock every 40 bars collapses any open well
    close = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                       "close": close, "volume": 10.0})
    s = strategy_module.SRFMBlackHole({})
    df = s.populate_indicators(df, {"pair": "TEST/USDT"})
    df = s.populate_entry_trend(df, {"pair": "TEST/USDT"})
    df = s.populate_exit_trend(df, {"pair": "TEST/USDT"})
    assert df["enter_long"].fillna(0).sum() >= 1
    assert df["exit_long"].fillna(0).sum() >= 1
    np.testing.assert_allclose(df["bh_mass"].to_numpy()[1:], sc.bh_mass(close)["bh_mass"].to_numpy()[1:])
