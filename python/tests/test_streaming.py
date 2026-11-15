"""Streaming classes against the untouched srfm-lab file and against the batch kernels."""

import importlib.util
import math

import numpy as np
import pytest

import srfm_core as sc
from srfm_core import kernels

from conftest import REFERENCE_DIR


@pytest.fixture(scope="module")
def lab():
    path = REFERENCE_DIR / "srfm_core_lab.py"
    if not path.exists():
        pytest.skip("reference copy not available")
    spec = importlib.util.spec_from_file_location("srfm_core_lab", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _walk(seed, n=600, sigma=0.0007, spike=0.05):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, sigma, n) + np.where(rng.random(n) < spike, rng.normal(0, 0.004, n), 0)
    return 100.0 * np.exp(np.cumsum(r))


@pytest.mark.parametrize("seed", range(8))
def test_bh_detector_is_identical_to_lab(lab, seed):
    close = _walk(seed)
    a_clf, a_bh = sc.MinkowskiClassifier(), sc.BlackHoleDetector()
    b_clf, b_bh = lab.MinkowskiClassifier(), lab.BlackHoleDetector()
    prev = None
    for c in close:
        ba, bb = a_clf.update(c), b_clf.update(c)
        assert ba == bb
        assert a_clf.beta == b_clf.beta
        assert a_clf.proper_time == b_clf.proper_time
        if prev is not None:
            assert a_bh.update(ba, c, prev) == b_bh.update(bb, c, prev)
        assert (a_bh.bh_mass, a_bh.bh_dir, a_bh.ctl) == (b_bh.bh_mass, b_bh.bh_dir, b_bh.ctl)
        prev = c
    assert a_bh.well_events == b_bh.well_events


def test_geodesic_lens_hawking_identical_to_lab(lab):
    close = _walk(99, n=300)
    g_a, g_b = sc.GeodesicAnalyzer(), lab.GeodesicAnalyzer()
    l_a, l_b = sc.GravitationalLens(), lab.GravitationalLens()
    h_a, h_b = sc.HawkingMonitor(), lab.HawkingMonitor()
    clf = sc.MinkowskiClassifier()
    for i, c in enumerate(close):
        atr = 0.05 + 0.01 * math.sin(i)
        assert g_a.update(c, atr) == g_b.update(c, atr)
        bit = clf.update(c)
        assert l_a.update(c, 1.0 + i % 3, bit, clf.tl_confirm, atr) == \
            l_b.update(c, 1.0 + i % 3, bit, clf.tl_confirm, atr)
        mid, std = c * 0.999, 0.1 + 0.01 * (i % 5)
        assert h_a.update(c, mid, std) == h_b.update(c, mid, std)


def test_rapidity_is_identically_zero():
    g = sc.GeodesicAnalyzer(window=5)
    vals = [g.update(c, 1.0)[3] for c in _walk(3, n=200, sigma=0.01)]
    assert all(v == 0.0 for v in vals)


def test_parameters_reach_the_math():
    close = _walk(5)
    base = kernels.bh_mass(close)["bh_mass"]
    for k, v in [("gain", 50.0), ("boost_cap", 1.5), ("boost_step", 0.2),
                 ("spacelike_decay", 0.5), ("timelike_decay", 0.9)]:
        other = kernels.bh_mass(close, **{k: v})["bh_mass"]
        assert not np.allclose(base[1:], other[1:]), k


def test_streaming_with_custom_constants_matches_kernel():
    close = _walk(11)
    kw = dict(bh_form=1.2, bh_collapse=0.9, spacelike_decay=0.6, ctl_min=3, gain=80.0,
              boost_step=0.05, boost_cap=1.8, reform_window=10, reform_frac=0.25)
    det = sc.BlackHoleDetector(bh_form=kw["bh_form"], bh_collapse=kw["bh_collapse"], bh_decay=0.93,
                               **{k: v for k, v in kw.items() if k not in ("bh_form", "bh_collapse")})
    clf = sc.MinkowskiClassifier(cf=0.002)
    batch = kernels.bh_mass(close, cf=0.002, timelike_decay=0.93, **kw)
    prev = None
    for i, c in enumerate(close):
        bit = clf.update(c)
        if prev is not None:
            det.update(bit, c, prev)
        assert det.bh_mass == pytest.approx(batch["bh_mass"][i], rel=1e-12, abs=1e-15)
        assert det.bh_active == batch["bh_active"][i]
        assert det.ctl == batch["ctl"][i]
        prev = c


def test_wilder_atr_streaming_matches_kernel():
    close = _walk(2, n=100)
    high, low = close * 1.001, close * 0.998
    atr = sc.WilderATR(14)
    batch = kernels.wilder_atr(high, low, close, 14)
    for i in range(close.size):
        v = atr.update(high[i], low[i], close[i])
        if i == 0:
            assert math.isnan(v) and math.isnan(batch[0])
        else:
            assert v == pytest.approx(batch[i], rel=1e-12)
