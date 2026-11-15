# Parameters

In srfm-lab these were literals inside LARSA's code. Here each one is a named parameter, the default is the value LARSA shipped with, and the same name is used in Python and Rust.

## Shared

| Name | Default | Meaning | How to choose it |
|---|---|---|---|
| `cf` | 0.001 | "Speed of light": the absolute per-bar return below which a bar is TIMELIKE (`beta = abs(dC) / C_prev / cf < 1`). | The single most important knob. It is in return units per bar, so it scales with bar size and instrument volatility. 0.001 (0.1%) was LARSA's value on hourly index futures. A sensible start is a low percentile (roughly the 40th to 60th) of `abs(returns)` on your data, so ordered bars and impulsive bars are both common. |

## Black-hole mass (`bh_mass`, `BlackHoleDetector`, `BhParams`)

Per bar, with `br = (C - C_prev) / C_prev`:

* TIMELIKE: `ctl += 1`, `sb = min(boost_cap, 1 + ctl * boost_step)`, `mass = mass * timelike_decay + abs(br) * gain * sb`
* SPACELIKE: `ctl = 0`, `mass *= spacelike_decay`

| Name | Default | Meaning |
|---|---|---|
| `bh_form` | 1.5 | Mass a well must exceed to form (with `ctl >= ctl_min`). |
| `bh_collapse` | 1.0 | Mass an active well must stay above (hysteresis below `bh_form`). |
| `timelike_decay` | 0.95 | Mass retained per TIMELIKE bar before accretion. Python streaming class name: `bh_decay` (kept from srfm-lab). |
| `spacelike_decay` | 0.7 | Mass retained per SPACELIKE bar. |
| `ctl_min` | 5 | Consecutive TIMELIKE bars required for a well to form or stay active. |
| `gain` | 100 | Scale on `abs(br)` when accreting. With `cf = 0.001` a full-speed TIMELIKE bar adds at most `0.1 * sb`. |
| `boost_step` | 0.1 | Growth of the run boost `sb` per consecutive TIMELIKE bar. |
| `boost_cap` | 2.0 | Cap on `sb`. |
| `reform_window` | 15 | Bars after a collapse during which a re-forming well gets the reform boost. |
| `reform_frac` | 0.5 | Share of the collapse mass added back (once) on the first formation attempt inside the reform window. |

Rough steady state: on a long run of TIMELIKE bars with mean `abs(br) = m`, mass tends to `m * gain * boost_cap / (1 - timelike_decay)`, so with defaults it is `4000 * m`. A well can only form if that exceeds `bh_form`, i.e. `m > 0.000375`, while every bar stays under `cf = 0.001`. If you change `cf`, scale `gain` (or `bh_form`) with it, otherwise wells either never form or never stop forming.

`bh_dir` is the sign of the return accumulated over the current TIMELIKE run (reset when no well is active and a SPACELIKE bar arrives).

## Geodesic (`geodesic`, `GeodesicAnalyzer`)

| Name | Default | Meaning |
|---|---|---|
| `window` | 20 | Bars in the rolling log-linear fit and the causal-fraction scan. |
| `cf` | 0.001 | Light cone for `causal_frac`: lag `k` counts as TIMELIKE when `abs(C_t - C_{t-k}) / C_{t-k} < cf * k`. |
| `atr` | input | Scale for `geo_dev = tanh((C - fitted C) / atr)`. Use `wilder_atr(high, low, close, 14)`. |

LARSA also computed a "rapidity" here. As shipped its denominator guard never passes, so it is identically 0; the streaming class still returns it for drop-in compatibility, the batch functions and Rust do not.

## Hawking (`hawking_z`, `HawkingMonitor`)

| Name | Default | Meaning |
|---|---|---|
| `period` | 20 | Bollinger window (population std, current bar included). |
| `hot` | 1.8 | `HawkingMonitor.is_hot` threshold on `hawking_t` (LARSA avoided long entries above it). |
| `inverted` | -1.5 | `HawkingMonitor.is_inverted` threshold. |

## Wilder ATR (`wilder_atr`, `WilderAtr`)

| Name | Default | Meaning |
|---|---|---|
| `period` | 14 | Running mean of true ranges for the first `period` bars, Wilder smoothing after. NaN on the first bar (srfm-lab returned 0). |

## Constants that stay constants

`1e-9` denominator guards are kept exactly as in LARSA so results match the reference bit for bit; they only matter for prices near 0, which the batch functions reject.
