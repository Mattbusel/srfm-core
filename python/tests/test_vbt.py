import numpy as np
import pandas as pd
import pytest

vbt = pytest.importorskip("vectorbt")

import srfm_core as sc
from srfm_core.vbt import BHMass, HawkingZ, SRFMBeta


def _close(n=300, cols=("a", "b")):
    rng = np.random.default_rng(7)
    data = {c: 100 * np.exp(np.cumsum(rng.uniform(0.0004, 0.0009, n) *
                                      np.where(rng.random(n) < 0.35, -1, 1))) for c in cols}
    return pd.DataFrame(data, index=pd.date_range("2025-01-01", periods=n, freq="h"))


def test_bhmass_matches_functional():
    close = _close()
    bh = BHMass.run(close)
    for col in close.columns:
        ref = sc.bh_mass(close[col])
        np.testing.assert_allclose(bh.mass[col].to_numpy(), ref["bh_mass"].to_numpy())
        assert bh.active[col].tolist() == ref["bh_active"].tolist()
        assert bh.ctl[col].tolist() == ref["ctl"].tolist()


def test_bhmass_parameter_grid_and_portfolio():
    close = _close(cols=("a",))["a"]
    bh = BHMass.run(close, cf=[0.001, 0.002], param_product=True)
    assert bh.mass.shape == (len(close), 2)
    single = BHMass.run(close)
    entries = single.active & ~single.active.shift(1, fill_value=False)
    pf = vbt.Portfolio.from_signals(close, entries, ~single.active)
    assert np.isfinite(pf.total_return())


def test_beta_and_hawking_indicators():
    close = _close(cols=("a",))["a"]
    np.testing.assert_allclose(SRFMBeta.run(close).beta.to_numpy(), sc.beta(close).to_numpy())
    hz = HawkingZ.run(close, period=10)
    np.testing.assert_allclose(hz.z.to_numpy(), sc.hawking_z(close, 10)["hawking_z"].to_numpy())
