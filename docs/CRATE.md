# srfm-core (Rust)

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
