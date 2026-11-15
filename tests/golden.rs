//! The numeric reference check: the Rust port against golden/golden_vectors.json,
//! generated from srfm-lab's untouched lib/srfm_core.py.

use serde_json::Value;
use srfm_core::{beta, bh_mass, geodesic, hawking_z, wilder_atr, BhParams};

const GOLDEN: &str = include_str!("../golden/golden_vectors.json");

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

fn tol(doc: &Value, field: &str) -> (f64, f64) {
    let t = doc["field_tolerance"]
        .get(field)
        .unwrap_or(&doc["tolerance"]);
    (t["rel"].as_f64().unwrap(), t["abs"].as_f64().unwrap())
}

fn check(case: &str, field: &str, actual: &[f64], expected: &[f64], (rel, abs): (f64, f64)) {
    assert_eq!(actual.len(), expected.len(), "{case}.{field} length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{case}.{field}[{i}]: expected NaN, got {a}");
        } else {
            assert!(
                (a - e).abs() <= abs + rel * e.abs(),
                "{case}.{field}[{i}]: {a} vs {e} (diff {})",
                (a - e).abs()
            );
        }
    }
}

#[test]
fn golden_vectors() {
    let doc: Value = serde_json::from_str(GOLDEN).expect("golden json");
    let base = tol(&doc, "_");
    let cases = doc["cases"].as_array().unwrap();
    assert!(cases.len() >= 6);
    for c in cases {
        let name = c["name"].as_str().unwrap();
        let p = &c["params"];
        let e = &c["expected"];
        let close = floats(&c["close"]);
        let high = floats(&c["high"]);
        let low = floats(&c["low"]);
        let cf = p["cf"].as_f64().unwrap();

        check(
            name,
            "beta",
            &beta(&close, cf).unwrap(),
            &floats(&e["beta"]),
            base,
        );

        let params = BhParams {
            cf,
            bh_form: p["bh_form"].as_f64().unwrap(),
            bh_collapse: p["bh_collapse"].as_f64().unwrap(),
            timelike_decay: p["timelike_decay"].as_f64().unwrap(),
            spacelike_decay: p["spacelike_decay"].as_f64().unwrap(),
            ctl_min: p["ctl_min"].as_u64().unwrap() as u32,
            ..BhParams::default()
        };
        let bh = bh_mass(&close, &params).unwrap();
        check(name, "bh_mass", &bh.mass, &floats(&e["bh_mass"]), base);
        let active: Vec<bool> = e["bh_active"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        let dir: Vec<i8> = e["bh_dir"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i8)
            .collect();
        let ctl: Vec<u32> = e["ctl"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(bh.active, active, "{name}.bh_active");
        assert_eq!(bh.dir, dir, "{name}.bh_dir");
        assert_eq!(bh.ctl, ctl, "{name}.ctl");

        let atr = wilder_atr(
            &high,
            &low,
            &close,
            p["atr_period"].as_u64().unwrap() as usize,
        )
        .unwrap();
        check(name, "atr", &atr, &floats(&e["atr"]), base);

        let window = p["window"].as_u64().unwrap() as usize;
        let geo = geodesic(&close, &atr, window, cf).unwrap();
        check(
            name,
            "geo_dev",
            &geo.dev,
            &floats(&e["geo_dev"]),
            tol(&doc, "geo_dev"),
        );
        check(
            name,
            "geo_slope",
            &geo.slope,
            &floats(&e["geo_slope"]),
            tol(&doc, "geo_slope"),
        );
        check(
            name,
            "causal_frac",
            &geo.causal_frac,
            &floats(&e["causal_frac"]),
            tol(&doc, "causal_frac"),
        );

        let hz = hawking_z(&close, p["period"].as_u64().unwrap() as usize).unwrap();
        check(name, "hawking_z", &hz.z, &floats(&e["hawking_z"]), base);
        check(name, "hawking_t", &hz.t, &floats(&e["hawking_t"]), base);
    }
}
