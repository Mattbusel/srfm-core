"""Generate golden_vectors.json from the srfm-lab reference implementation.

The expected outputs come from ``reference/srfm_core_lab.py``, which is a
byte-for-byte copy of ``lib/srfm_core.py`` in srfm-lab (commit 8cbfcb0). Only
its streaming classes are used, driven bar by bar the way LARSA drives them:

    bit = classifier.update(close)
    if there is a previous close:
        detector.update(bit, close, prev_close)

Nothing in this file re-implements the math. If a port disagrees with these
vectors, the port is wrong (or the reference changed and the vectors must be
regenerated on purpose).

Conventions written into the vectors, which every port follows:
  * A non-finite close (NaN/inf) is a gap: it is not fed to the reference,
    its outputs are null (bh_active false, bh_dir 0, ctl 0) and the next
    finite bar is compared against the last finite close.
  * beta on the first finite bar is null (there is no previous close).
  * hawking_z / hawking_t are null until `period` finite closes are buffered,
    and on any bar where the rolling population std is 0.
  * geo_dev / geo_slope / causal_frac are null until `window` finite closes
    are buffered.
  * atr (Wilder) is null on the first finite bar.

Run:  python golden/generate.py           (writes golden/golden_vectors.json)
      python golden/generate.py --check   (regenerates in memory and checks the
                                           committed file still matches, within
                                           the tolerances it declares)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from typing import List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "reference"))

import srfm_core_lab as ref  # noqa: E402

LAB_COMMIT = "8cbfcb0551d44e3ed1c20bdae9dc805d61521c62"


def _f(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else x


def _finite(x: float) -> bool:
    return x is not None and math.isfinite(x)


def run_reference(close, high, low, p):
    n = len(close)
    out = {k: [None] * n for k in (
        "beta", "bh_mass", "geo_dev", "geo_slope", "causal_frac",
        "hawking_z", "hawking_t", "atr")}
    out["bh_active"] = [False] * n
    out["bh_dir"] = [0] * n
    out["ctl"] = [0] * n

    clf = ref.MinkowskiClassifier(cf=p["cf"])
    det = ref.BlackHoleDetector(
        bh_form=p["bh_form"], bh_collapse=p["bh_collapse"], bh_decay=p["timelike_decay"])
    atr = ref._ATR(p["atr_period"])
    geo = ref.GeodesicAnalyzer(cf=p["cf"], window=p["window"])
    bb = ref._BB(p["period"])
    hm = ref.HawkingMonitor()

    prev = None
    for i in range(n):
        c = close[i]
        if not _finite(c):
            continue
        # MinkowskiClassifier + BlackHoleDetector (LARSA driving order)
        bit = clf.update(c)
        if prev is not None:
            out["beta"][i] = _f(clf.beta)
            det.update(bit, c, prev)
        out["bh_mass"][i] = _f(det.bh_mass)
        out["bh_active"][i] = bool(det.bh_active)
        out["bh_dir"][i] = int(det.bh_dir)
        out["ctl"][i] = int(det.ctl)

        # Wilder ATR (reference _ATR); first finite bar has no true range
        a = atr.update(high[i], low[i], c)
        a_val = a if prev is not None else None
        out["atr"][i] = _f(a_val)

        # GeodesicAnalyzer, fed the ATR above (NaN while ATR is undefined)
        g = geo.update(c, a_val if a_val is not None else float("nan"))
        if geo.ready:
            out["geo_dev"][i] = _f(g[0])
            out["geo_slope"][i] = _f(g[1])
            out["causal_frac"][i] = _f(g[2])

        # HawkingMonitor on the reference rolling band
        mid, std = bb.update(c)
        if len(bb._buf) >= p["period"] and std > 0:
            hm.update(c, mid, std)
            out["hawking_z"][i] = _f(hm._pz)
            out["hawking_t"][i] = _f(hm.ht)
        prev = c
    return out


DEFAULTS = dict(cf=0.001, bh_form=1.5, bh_collapse=1.0, timelike_decay=0.95,
                spacelike_decay=0.7, ctl_min=5, window=20, period=20, atr_period=14)


def _hl(close, rng, spread=0.0004):
    high, low = [], []
    for c in close:
        if not _finite(c):
            high.append(float("nan")); low.append(float("nan")); continue
        u = abs(rng.normal(0, spread)); d = abs(rng.normal(0, spread))
        high.append(c * (1 + u)); low.append(c * (1 - d))
    return high, low


def make_cases():
    cases = []
    rng = np.random.default_rng(20261115)

    # 1. quiet random walk: mostly TIMELIKE bars, wells form and collapse
    mag = rng.uniform(0.0004, 0.00095, 400)
    sign = np.where(rng.random(400) < 0.6, 1.0, -1.0)
    spike = np.where(rng.random(400) < 0.04, rng.normal(0, 0.004, 400), 0.0)
    r = mag * sign + spike
    close = list(100.0 * np.exp(np.cumsum(r)))
    h, l = _hl(close, rng)
    cases.append(("quiet_random_walk", close, h, l, dict(DEFAULTS)))

    # 2. regime mix: calm trend, then a volatile block of SPACELIKE bars, then calm down-trend
    r = np.concatenate([rng.normal(0.0003, 0.0003, 120),
                        rng.normal(0.0, 0.004, 60),
                        rng.normal(-0.0003, 0.0003, 120)])
    close = list(5000.0 * np.exp(np.cumsum(r)))
    h, l = _hl(close, rng)
    cases.append(("regime_mix", close, h, l, dict(DEFAULTS)))

    # 3. gaps: NaN bars inside the series and at the start
    r = rng.uniform(0.0004, 0.0009, 150) * np.where(rng.random(150) < 0.3, -1.0, 1.0)
    close = list(42.0 * np.exp(np.cumsum(r)))
    for k in (0, 1, 37, 38, 39, 90, 149):
        close[k] = float("nan")
    h, l = _hl(close, rng)
    cases.append(("nan_gaps", close, h, l, dict(DEFAULTS)))

    # 4. non-default parameters for every documented constant the ports share
    r = rng.normal(0.0002, 0.003, 250)
    close = list(1.2345 * np.exp(np.cumsum(r)))
    h, l = _hl(close, rng, 0.002)
    p = dict(DEFAULTS, cf=0.005, bh_form=1.2, bh_collapse=0.8, timelike_decay=0.9,
             window=10, period=10, atr_period=7)
    cases.append(("custom_params", close, h, l, p))

    # 5. constant price: beta 0, rolling std 0 so hawking is null
    close = [250.0] * 40
    cases.append(("constant_price", close, [c * 1.001 for c in close],
                  [c * 0.999 for c in close], dict(DEFAULTS)))

    # 6. tiny series shorter than every window
    close = [10.0, 10.005, 10.004]
    cases.append(("short_series", close, [c * 1.0005 for c in close],
                  [c * 0.9995 for c in close], dict(DEFAULTS)))
    return cases


def _check(doc, path):
    with open(path, encoding="utf-8") as fh:
        committed = json.load(fh)
    bad = []
    if committed["reference"] != doc["reference"]:
        bad.append("reference block differs")
    for new, old in zip(doc["cases"], committed["cases"]):
        for field, vals in new["expected"].items():
            tol = committed.get("field_tolerance", {}).get(field, committed["tolerance"])
            for i, (a, b) in enumerate(zip(vals, old["expected"][field])):
                if isinstance(a, bool) or isinstance(a, int) or a is None or b is None:
                    if a != b:
                        bad.append(f"{new['name']}.{field}[{i}]: {a} != {b}")
                elif abs(a - b) > tol["abs"] + tol["rel"] * abs(b):
                    bad.append(f"{new['name']}.{field}[{i}]: {a} vs {b}")
    if len(doc["cases"]) != len(committed["cases"]):
        bad.append("case count differs")
    for line in bad[:20]:
        print(line)
    print("golden vectors OK" if not bad else f"{len(bad)} mismatches")
    return 1 if bad else 0


def main():
    with open(os.path.join(HERE, "reference", "srfm_core_lab.py"), "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    doc = {
        "format": 1,
        "generator": "golden/generate.py",
        "reference": {
            "repository": "https://github.com/Mattbusel/srfm-lab",
            "path": "lib/srfm_core.py",
            "commit": LAB_COMMIT,
            "sha256": sha,
        },
        "tolerance": {"rel": 1e-9, "abs": 1e-12},
        "field_tolerance": {
            "geo_dev": {"rel": 1e-6, "abs": 1e-9},
            "geo_slope": {"rel": 1e-6, "abs": 1e-9},
        },
        "note": ("null = NaN / undefined. bh_active, bh_dir and ctl are exact. "
                 "geo_* use a looser tolerance because LARSA's closed-form "
                 "regression (n*Sxy - Sx*Sy on log prices) cancels badly and "
                 "summation order differs between numpy and other languages."),
        "cases": [],
    }
    for name, close, high, low, p in make_cases():
        exp = run_reference(close, high, low, p)
        doc["cases"].append({
            "name": name,
            "params": p,
            "close": [_f(x) for x in close],
            "high": [_f(x) for x in high],
            "low": [_f(x) for x in low],
            "expected": exp,
        })
    path = os.path.join(HERE, "golden_vectors.json")
    if "--check" in sys.argv:
        sys.exit(_check(doc, path))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1, allow_nan=False)
        fh.write("\n")
    print(f"wrote {path}: {len(doc['cases'])} cases")


if __name__ == "__main__":
    main()
