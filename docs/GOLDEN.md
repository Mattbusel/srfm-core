# Golden vectors

`golden/golden_vectors.json` is the contract every implementation of these transforms must meet: the Python package and the Rust crate today, and any later port (C++, Julia, C, WASM, R).

## Where the numbers come from

`golden/generate.py` loads `golden/reference/srfm_core_lab.py`, a byte-for-byte copy of `lib/srfm_core.py` from [srfm-lab](https://github.com/Mattbusel/srfm-lab) at commit `8cbfcb0551d44e3ed1c20bdae9dc805d61521c62` (its SHA-256 is recorded in the JSON). It drives the reference's own streaming classes bar by bar, the way LARSA drives them, and writes what they output. It does not re-implement any math.

```
python golden/generate.py
```

Regenerate only on purpose (a deliberate change of reference). CI fails if a port drifts from the committed file.

## Format

```jsonc
{
  "format": 1,
  "reference": {"repository": "...", "path": "lib/srfm_core.py", "commit": "...", "sha256": "..."},
  "tolerance": {"rel": 1e-9, "abs": 1e-12},
  "field_tolerance": {"geo_dev": {...}, "geo_slope": {...}},
  "cases": [
    {
      "name": "quiet_random_walk",
      "params": {"cf": 0.001, "bh_form": 1.5, "bh_collapse": 1.0, "timelike_decay": 0.95,
                 "spacelike_decay": 0.7, "ctl_min": 5, "window": 20, "period": 20, "atr_period": 14},
      "close": [...], "high": [...], "low": [...],
      "expected": {
        "beta": [...], "bh_mass": [...], "bh_active": [...], "bh_dir": [...], "ctl": [...],
        "atr": [...], "geo_dev": [...], "geo_slope": [...], "causal_frac": [...],
        "hawking_z": [...], "hawking_t": [...]
      }
    }
  ]
}
```

`null` means NaN or undefined. `bh_active`, `bh_dir` and `ctl` must match exactly. Floats pass when `abs(actual - expected) <= abs + rel * abs(expected)`, using `field_tolerance[field]` when present. The geodesic fields get a looser bound because LARSA's closed-form regression (`n*Sxy - Sx*Sy` over log prices) cancels badly, so a different summation order moves the last few digits.

## Cases

| Case | What it exercises |
|---|---|
| `quiet_random_walk` | Mostly TIMELIKE bars with rare shocks: many formations, collapses and reform boosts. |
| `regime_mix` | Calm up-trend, a volatile SPACELIKE block, calm down-trend: direction flips. |
| `nan_gaps` | NaN closes at the start and inside the series. |
| `custom_params` | Non-default `cf`, `bh_form`, `bh_collapse`, `timelike_decay`, `window`, `period`, `atr_period`. |
| `constant_price` | Zero returns, zero rolling std (Hawking undefined). |
| `short_series` | Three bars, shorter than every window. |

## Conventions a port must follow

1. A non-finite close is a gap: output NaN (inactive, `bh_dir` 0, `ctl` 0), do not touch state, compare the next finite bar with the last finite close.
2. beta and ATR are NaN on the first finite bar.
3. Geodesic outputs are NaN until `window` finite closes; Hawking outputs until `period` finite closes and wherever the rolling std is 0.
4. The Hawking proxy uses the previous *defined* z, starting from 0.
5. Keep the `1e-9` denominator guards.

## Using the vectors from another language

Load the JSON, run each case with its `params`, compare field by field. The Rust test (`tests/golden.rs`) is about 90 lines and is a good template.
