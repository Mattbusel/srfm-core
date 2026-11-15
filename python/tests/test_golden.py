"""The numeric reference check: every batch function against golden_vectors.json,
which was generated from srfm-lab's untouched lib/srfm_core.py."""

import numpy as np
import pytest

import srfm_core as sc
from srfm_core import kernels

from conftest import assert_close, field_tol, to_array


def _cases(golden):
    return {c["name"]: c for c in golden["cases"]}


def test_reference_is_pinned(golden):
    ref = golden["reference"]
    assert ref["path"] == "lib/srfm_core.py"
    assert len(ref["commit"]) == 40
    assert len(golden["cases"]) >= 6


@pytest.mark.parametrize("name", ["quiet_random_walk", "regime_mix", "nan_gaps",
                                  "custom_params", "constant_price", "short_series"])
def test_case(golden, name):
    case = _cases(golden)[name]
    p, e, tol = case["params"], case["expected"], golden["tolerance"]
    close = to_array(case["close"])
    high = to_array(case["high"])
    low = to_array(case["low"])

    assert_close(kernels.beta(close, p["cf"]), e["beta"], tol, f"{name}.beta")

    bh = kernels.bh_mass(close, cf=p["cf"], bh_form=p["bh_form"], bh_collapse=p["bh_collapse"],
                         timelike_decay=p["timelike_decay"], spacelike_decay=p["spacelike_decay"],
                         ctl_min=p["ctl_min"])
    assert_close(bh["bh_mass"], e["bh_mass"], tol, f"{name}.bh_mass")
    assert bh["bh_active"].tolist() == e["bh_active"], f"{name}.bh_active"
    assert bh["bh_dir"].tolist() == e["bh_dir"], f"{name}.bh_dir"
    assert bh["ctl"].tolist() == e["ctl"], f"{name}.ctl"

    atr = kernels.wilder_atr(high, low, close, p["atr_period"])
    assert_close(atr, e["atr"], tol, f"{name}.atr")

    geo = kernels.geodesic(close, atr, window=p["window"], cf=p["cf"])
    for k in ("geo_dev", "geo_slope", "causal_frac"):
        assert_close(geo[k], e[k], field_tol(golden, k), f"{name}.{k}")

    hz = kernels.hawking_z(close, period=p["period"])
    assert_close(hz["hawking_z"], e["hawking_z"], tol, f"{name}.hawking_z")
    assert_close(hz["hawking_t"], e["hawking_t"], tol, f"{name}.hawking_t")


def test_pandas_front_end_matches_kernels(golden):
    import pandas as pd
    case = _cases(golden)["regime_mix"]
    idx = pd.date_range("2024-01-01", periods=len(case["close"]), freq="h")
    close = pd.Series(to_array(case["close"]), index=idx)
    df = sc.bh_mass(close)
    assert list(df.columns) == ["bh_mass", "bh_active", "bh_dir", "ctl"]
    assert df.index.equals(idx)
    assert_close(df["bh_mass"].to_numpy(), case["expected"]["bh_mass"], golden["tolerance"], "frame")
    assert df["bh_active"].tolist() == case["expected"]["bh_active"]
