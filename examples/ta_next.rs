//! Streaming use through the `ta` crate traits.
//!
//! cargo run --example ta_next --features ta

use srfm_core::{BhMass, BhParams, HawkingZ};
use ta::Next;

fn main() {
    let mut bh = BhMass::new(BhParams {
        cf: 0.001,
        ..BhParams::default()
    })
    .unwrap();
    let mut hz = HawkingZ::new(20).unwrap();

    let mut price = 100.0_f64;
    for i in 0..60 {
        // calm ordered drift with a shock at bar 40
        price *= if i == 40 { 1.01 } else { 1.0007 };
        let b = bh.next(price);
        let h = hz.next(price);
        println!(
            "{i:>3} close={price:>9.4} mass={:>6.3} active={:<5} dir={:>2} ctl={:>2} z={:>6.2}",
            b.mass, b.active, b.dir, b.ctl, h.z
        );
    }
}
