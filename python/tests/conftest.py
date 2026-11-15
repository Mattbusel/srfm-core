import json
import os
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = pathlib.Path(os.environ.get("SRFM_GOLDEN", ROOT / "golden" / "golden_vectors.json"))
REFERENCE_DIR = ROOT / "golden" / "reference"


@pytest.fixture(scope="session")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"golden vectors not found at {GOLDEN} (run from a git checkout)")
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


def to_array(values, dtype=float):
    if dtype is float:
        return np.array([np.nan if v is None else v for v in values], dtype=float)
    return np.array(values, dtype=dtype)


def field_tol(golden, field):
    return golden.get("field_tolerance", {}).get(field, golden["tolerance"])


def assert_close(actual, expected, tol, label):
    actual = np.asarray(actual, dtype=float)
    expected = to_array(expected)
    assert actual.shape == expected.shape, label
    nan_a = np.isnan(actual)
    nan_e = np.isnan(expected)
    bad = np.flatnonzero(nan_a != nan_e)
    assert bad.size == 0, f"{label}: NaN pattern differs at {bad[:5].tolist()}"
    ok = ~nan_e
    diff = np.abs(actual[ok] - expected[ok])
    lim = tol["abs"] + tol["rel"] * np.abs(expected[ok])
    worst = np.flatnonzero(diff > lim)
    assert worst.size == 0, (
        f"{label}: {worst.size} values off, first at {np.flatnonzero(ok)[worst[0]]}: "
        f"{actual[ok][worst[0]]!r} vs {expected[ok][worst[0]]!r}")
