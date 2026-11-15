import numpy as np
import pandas as pd
import pytest

import srfm_core as sc


def test_beta_hand_computed():
    b = sc.beta([5000.0, 5002.5, 5012.5], cf=0.001)
    assert np.isnan(b.iloc[0])
    assert b.iloc[1] == pytest.approx(0.5, abs=1e-6)
    assert b.iloc[2] == pytest.approx(10.0 / 5002.5 / 0.001, rel=1e-9)


def test_bh_forms_on_steady_ordered_moves():
    # 0.08% per bar < cf: every bar TIMELIKE, mass climbs past 1.5 with ctl >= 5
    close = 100 * 1.0008 ** np.arange(30)
    df = sc.bh_mass(close)
    first = int(np.argmax(df["bh_active"].to_numpy()))
    assert df["bh_active"].any()
    assert df["ctl"].iloc[first] >= 5
    assert df["bh_mass"].iloc[first] > 1.5
    assert (df["bh_dir"].iloc[1:] == 1).all()


def test_spacelike_bar_kills_the_well():
    close = list(100 * 1.0008 ** np.arange(30)) + [110.0]
    df = sc.bh_mass(close)
    assert df["bh_active"].iloc[-2]
    assert not df["bh_active"].iloc[-1]
    assert df["ctl"].iloc[-1] == 0
    assert df["bh_mass"].iloc[-1] == pytest.approx(df["bh_mass"].iloc[-2] * 0.7, rel=1e-12)


def test_gaps_are_skipped_not_zeroed():
    clean = 100 * 1.0005 ** np.arange(40)
    gappy = clean.copy().astype(float)
    gappy = np.insert(gappy, 10, np.nan)
    a = sc.bh_mass(clean)["bh_mass"].to_numpy()
    b = sc.bh_mass(gappy)["bh_mass"].to_numpy()
    assert np.isnan(b[10])
    np.testing.assert_array_equal(np.delete(b, 10), a)


def test_non_positive_close_raises():
    with pytest.raises(ValueError):
        sc.bh_mass([1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        sc.beta([1.0, 2.0], cf=0)


def test_unknown_kwarg_raises():
    with pytest.raises(TypeError):
        sc.bh_mass([1.0, 1.001], bogus=1)


def test_warmup_is_nan():
    close = 100 + np.arange(25, dtype=float) * 0.01
    hz = sc.hawking_z(close, period=20)
    assert hz["hawking_z"].iloc[:19].isna().all()
    assert hz["hawking_z"].iloc[19:].notna().all()
    geo = sc.geodesic(close, np.full(25, 0.05), window=20)
    assert geo["geo_slope"].iloc[:19].isna().all()
    assert geo["geo_slope"].iloc[19] > 0


def test_first_hawking_t_is_z_squared():
    close = 100 + np.sin(np.arange(40))
    hz = sc.hawking_z(close, period=10)
    z0 = hz["hawking_z"].iloc[9]
    assert hz["hawking_t"].iloc[9] == pytest.approx(z0 * z0)
    z1 = hz["hawking_z"].iloc[10]
    assert hz["hawking_t"].iloc[10] == pytest.approx(z1 * (z1 - z0))


def test_add_srfm_columns_on_ohlcv_frame():
    n = 120
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0006, n)))
    df = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                       "close": close, "volume": 1.0},
                      index=pd.date_range("2025-01-01", periods=n, freq="5min"))
    out = sc.add_srfm_columns(df)
    assert out is df
    for col in ("srfm_beta", "srfm_bh_mass", "srfm_bh_active", "srfm_bh_dir", "srfm_ctl",
                "srfm_geo_dev", "srfm_geo_slope", "srfm_causal_frac",
                "srfm_hawking_z", "srfm_hawking_t", "srfm_atr"):
        assert col in df.columns, col
    np.testing.assert_array_equal(df["srfm_bh_mass"].to_numpy(), sc.bh_mass(close)["bh_mass"].to_numpy())
    bare = sc.add_srfm_columns(df[["close"]].copy(), prefix="")
    assert "bh_mass" in bare.columns and "atr" in bare.columns


def test_bh_defaults_documented():
    assert sc.BH_DEFAULTS == dict(cf=0.001, bh_form=1.5, bh_collapse=1.0, timelike_decay=0.95,
                                  spacelike_decay=0.7, ctl_min=5, gain=100.0, boost_step=0.1,
                                  boost_cap=2.0, reform_window=15, reform_frac=0.5)


def test_importing_package_does_not_import_vectorbt():
    import subprocess
    import sys
    code = "import sys, srfm_core, srfm_core.vbt; print('vectorbt' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
