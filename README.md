# srfm-core: if your indicator has two implementations and no shared test vectors, you have two indicators

[![crates.io](https://img.shields.io/crates/v/srfm-core.svg)](https://crates.io/crates/srfm-core)
[![PyPI](https://img.shields.io/pypi/v/srfm-core.svg)](https://pypi.org/project/srfm-core/)
[![docs.rs](https://docs.rs/srfm-core/badge.svg)](https://docs.rs/srfm-core)

Research happens in Python, production runs in Rust or C, and the indicator gets rewritten by hand on the way. Nobody notices until the live system stops agreeing with the backtest. We found it in our own code first: srfm-lab carried the same black-hole mass recurrence in many hand-written places (Python, several Rust crates, a WASM build, a C extension, C++, Julia notebooks), with no shared test vectors. Run four of those update rules on the same 400-bar random walk with the same parameters and they do not agree on anything:

| Bars flagged as an active well (quiet_random_walk, 400 bars) | Count |
|---|---|
| Reference `lib/srfm_core.py` | 306 |
| `larsa-core`, `srfm-tools`, `larsa-wasm` | 0 |
| `tick-backtest` `bh_physics.rs` | 398 |
| `fast_indicators` `bh_fast.c` | 0 |

Reproduce it with `python golden/drift/compare_lab_variants.py`; the full table and why each copy differs is in [docs/DRIFT.md](docs/DRIFT.md). None of it is a config difference; the recurrences themselves differ.

srfm-core takes the other side: **one definition, two languages, one file of golden vectors both must match.** The Python package and this Rust crate are both tested against `golden/golden_vectors.json`, generated from the untouched reference, so they cannot drift apart without a test failing.

**Why it exists.** To give the SRFM bar transforms (Minkowski beta, black-hole mass, rolling geodesic, the Hawking z proxy) one checked definition that research and production can both import.

**Why this instead of porting it yourself.** A port is a sixth copy. This one ships the vectors that prove it matches, every LARSA constant is a named parameter with LARSA's value as the default ([PARAMETERS.md](docs/PARAMETERS.md)), and bad input is an `Error`, not a quiet NaN.

**Why today.** No default dependencies in Rust (MSRV 1.81, no unsafe), NumPy and pandas only in Python, a vectorbt indicator, a freqtrade helper, and `ta::Next` with the `ta` feature. `cargo add srfm-core` or `pip install srfm-core`.

**Who it is for:** anyone moving a Python research feature into a Rust or C++ trading system who needs the production numbers to equal the research numbers.

**What it does not claim:** these are feature transforms, not a strategy. Nothing here claims a trading edge.

The SRFM bar transforms from [srfm-lab](https://github.com/Mattbusel/srfm-lab) as a small, dependency-free Rust crate.

| Transform | Slice function | Streaming struct |
|---|---|---|
| Minkowski beta, `abs(dC) / C_prev / cf` (below 1 = TIMELIKE bar) | `beta(&close, cf)` | `Beta` |
| Black-hole mass, well state, direction, run length | `bh_mass(&close, &BhParams)` | `BhMass` |
| Rolling log-linear geodesic and causal fraction | `geodesic(&close, &atr, window, cf)` | `Geodesic` |
| Bollinger z and the Hawking proxy `z * (z - z_prev)` | `hawking_z(&close, period)` | `HawkingZ` |
| Wilder ATR (srfm-lab warm-up) | `wilder_atr(&high, &low, &close, period)` | `WilderAtr` |

```rust
use srfm_core::{bh_mass, BhParams};

fn main() -> Result<(), srfm_core::Error> {
    let close = vec![100.0, 100.07, 100.13, 100.2, 100.28, 100.35, 100.41, 100.5];
    let bh = bh_mass(&close, &BhParams { cf: 0.001, ..BhParams::default() })?;
    println!("{:?} {:?}", bh.mass, bh.active);
    Ok(())
}
```

Feature `ta`: every streaming struct implements `ta::Next` and `ta::Reset` (bare `f64` closes or any `&T: ta::Close`), so it drops into code built on the [`ta`](https://crates.io/crates/ta) crate.

Every LARSA constant is a field of `BhParams` with LARSA's value as the default (see [PARAMETERS.md](https://github.com/Mattbusel/srfm-core/blob/main/docs/PARAMETERS.md)). The crate is tested against `golden/golden_vectors.json`, generated from srfm-lab's untouched `lib/srfm_core.py` and shared with the Python package [`srfm-core`](https://pypi.org/project/srfm-core/). No unsafe code, no default dependencies, MSRV 1.81.

These are feature transforms, not a strategy. Nothing here claims a trading edge.

License: MIT. Provenance and attribution: [NOTICE](https://github.com/Mattbusel/srfm-core/blob/main/NOTICE).
