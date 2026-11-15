//! Slice API: compute every transform over a close series.
//!
//! cargo run --example bh_mass

use srfm_core::{beta, bh_mass, geodesic, hawking_z, wilder_atr, BhParams};

fn main() -> Result<(), srfm_core::Error> {
    // deterministic toy series: alternating calm runs and volatile bursts
    let mut close = Vec::with_capacity(200);
    let mut p = 100.0_f64;
    for i in 0..200u32 {
        let r = if (i / 25) % 2 == 0 {
            0.0006
        } else if i % 2 == 0 {
            0.004
        } else {
            -0.0035
        };
        p *= 1.0 + r;
        close.push(p);
    }
    let high: Vec<f64> = close.iter().map(|c| c * 1.0005).collect();
    let low: Vec<f64> = close.iter().map(|c| c * 0.9995).collect();

    let b = beta(&close, 0.001)?;
    let bh = bh_mass(&close, &BhParams::default())?;
    let atr = wilder_atr(&high, &low, &close, 14)?;
    let geo = geodesic(&close, &atr, 20, 0.001)?;
    let hz = hawking_z(&close, 20)?;

    for i in (0..close.len()).step_by(10) {
        println!(
            "{i:>3} beta={:>6.2} mass={:>6.3} active={:<5} geo_slope={:>7.4} causal={:>5.2} z={:>6.2}",
            b[i], bh.mass[i], bh.active[i], geo.slope[i], geo.causal_frac[i], hz.z[i]
        );
    }
    let wells = (1..close.len())
        .filter(|&i| bh.active[i] && !bh.active[i - 1])
        .count();
    println!("wells formed: {wells}");
    Ok(())
}
