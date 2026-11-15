# Drift inside srfm-lab

srfm-lab carries the black-hole mass recurrence in many places (Python, several Rust crates, a C extension, a WASM crate, C++, Julia notebooks). They were rewritten by hand with no shared test vectors. Reading the formulas side by side is not a reliable way to find drift: for example `larsa-core` writes beta as `abs(dC) / (C_prev * cf)` and Python as `(abs(dC) / C_prev) / cf`, which look different and are the same. Running them on the same inputs is.

`golden/drift/compare_lab_variants.py` transcribes three update rules from srfm-lab (commit `8cbfcb0`) and runs them on the golden cases with the defaults the vectors use (`cf=0.001`, `bh_form=1.5`, `bh_collapse=1.0`, decay 0.95, `ctl_min=5`). Bars flagged as an active well:

| Case | Reference `lib/srfm_core.py` | `larsa-core` / `srfm-tools` / `larsa-wasm` | `tick-backtest` `bh_physics.rs` | `fast_indicators` `bh_fast.c` |
|---|---|---|---|---|
| quiet_random_walk (400 bars) | 306 | 0 | 398 | 0 |
| regime_mix (300) | 29 | 0 | 298 | 0 |
| nan_gaps (143 finite) | 113 | 0 | 141 | 0 |
| constant_price (40) | 0 | 0 | 38 | 0 |

What the numbers say:

* **larsa-core, srfm-tools (`srfm_turbo`, `srfm_stream`), larsa-wasm**: TIMELIKE bars do `mass = mass * 0.97 + 0.03`, which converges to 1.0. With `bh_form = 1.5` a well can never form. (`larsa-core`'s own `sensitivity_sweep` defaults to `cf = 0.02`, but the cap at 1.0 holds for any `cf`.)
* **tick-backtest**: TIMELIKE bars add a Lorentz gamma (at least 1.0) and mass is capped at 2.0, with no run-length requirement, so a well forms after two ordered bars and stays on. It is active even on a flat price.
* **fast_indicators C extension**: beta is `abs(log return)` compared with `cf` directly (no division), TIMELIKE bars add `cf / 2`, and SPACELIKE bars *add* `2 * (beta - cf)` (a shock raises mass, while it resets the run counter). In these cases it forms no well at all.

None of these is a parameter difference that a config change would fix; the recurrences differ. srfm-core takes `lib/srfm_core.py` as the definition (it is the copy srfm-lab's own tests and docs describe as "taken directly from the LARSA strategy code") and ships the golden vectors so any copy can be checked in one test run.

Not covered by the script yet: the C++ copy in `cpp/signal-engine/src/bh_physics` and the Julia notebook versions. They can be checked by loading `golden/golden_vectors.json` the same way.
