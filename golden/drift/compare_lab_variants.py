"""Score the other BH-mass copies inside srfm-lab against the golden vectors.

Each variant below is a line-for-line transcription of the update rule in the
named srfm-lab file (commit 8cbfcb0), run with the same defaults the golden
vectors use (cf=0.001, bh_form=1.5, bh_collapse=1.0, decay=0.95, ctl_min=5).
It reports, per golden case, how many bars each variant flags as an active
well against the reference and how many bars disagree.

Run:  python golden/drift/compare_lab_variants.py
"""

from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CF, FORM, COLLAPSE, DECAY, CTL = 0.001, 1.5, 1.0, 0.95, 5


def larsa_core(close):
    """crates/larsa-core/src/lib.rs run_bh_physics (same rule in
    srfm-tools srfm_turbo.rs / srfm_stream.rs and larsa-wasm minkowski.rs
    with mass_alpha = 0.03)."""
    mass, ctl, act, out = 0.0, 0, False, []
    prev = None
    for c in close:
        if prev is not None:
            beta = abs(c - prev) / (prev * CF + 1e-12)
            if beta < 1.0:
                mass = mass * 0.97 + 0.03
                ctl += 1
            else:
                mass *= DECAY
                ctl = 0
            if not act and mass >= FORM and ctl >= CTL:
                act = True
            if act and mass < COLLAPSE:
                act = False
        out.append(act)
        prev = c
    return out


def tick_backtest(close):
    """crates/tick-backtest/src/bh_physics.rs BHState::update."""
    mass, act, out = 0.0, False, []
    prev = None
    for c in close:
        if prev is not None:
            dx = c / prev - 1.0
            if CF * CF - dx * dx > 0.0:
                b = min(abs(dx / CF), 0.9999)
                mass = min(mass + 1.0 / math.sqrt(1.0 - b * b), 2.0)
            else:
                mass *= DECAY
                if mass < 1e-9:
                    mass = 0.0
            if not act and mass >= FORM:
                act = True
            elif act and mass < COLLAPSE:
                act = False
        out.append(act)
        prev = c
    return out


def fast_indicators_c(close):
    """extensions/fast_indicators/src/bh_fast.c bh_update."""
    mass, ctl, act, out = 0.0, 0, False, []
    prev = None
    for c in close:
        if prev is not None:
            beta = abs(math.log(c / prev))
            tl = beta < CF
            ctl = ctl + 1 if tl else 0
            dm = CF * 0.5 if tl else (beta - CF) * 2.0
            mass = mass * DECAY + dm
            if not act:
                if mass >= FORM and ctl >= CTL:
                    act = True
            elif mass < COLLAPSE:
                act, mass, ctl = False, 0.0, 0
        out.append(act)
        prev = c
    return out


VARIANTS = {
    "larsa-core / srfm-tools / larsa-wasm": larsa_core,
    "tick-backtest bh_physics.rs": tick_backtest,
    "fast_indicators bh_fast.c": fast_indicators_c,
}


def main():
    with open(os.path.join(HERE, "..", "golden_vectors.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    print(f"{'case':<20} {'variant':<38} {'ref active':>10} {'var active':>10} {'disagree':>9}")
    for case in doc["cases"]:
        if case["params"]["cf"] != CF or case["params"]["bh_form"] != FORM:
            continue
        idx = [i for i, v in enumerate(case["close"]) if v is not None]
        close = [case["close"][i] for i in idx]
        ref = [case["expected"]["bh_active"][i] for i in idx]
        for name, fn in VARIANTS.items():
            got = fn(close)
            dis = sum(a != b for a, b in zip(ref, got))
            print(f"{case['name']:<20} {name:<38} {sum(ref):>10} {sum(got):>10} {dis:>9}")


if __name__ == "__main__":
    main()
