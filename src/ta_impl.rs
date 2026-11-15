//! `ta` crate integration (feature `ta`).
//!
//! Every streaming struct implements [`ta::Next`] and [`ta::Reset`], taking a
//! bare `f64` close or any `&T` that implements the `ta` data-item traits.

use crate::{Beta, BhBar, BhMass, GeoBar, Geodesic, HawkingBar, HawkingZ, WilderAtr};
use ta::{Close, High, Low, Next, Period, Reset};

impl Next<f64> for Beta {
    type Output = f64;
    fn next(&mut self, close: f64) -> f64 {
        self.update(close)
    }
}

impl<T: Close> Next<&T> for Beta {
    type Output = f64;
    fn next(&mut self, bar: &T) -> f64 {
        self.update(bar.close())
    }
}

impl Reset for Beta {
    fn reset(&mut self) {
        Beta::reset(self)
    }
}

impl Next<f64> for BhMass {
    type Output = BhBar;
    fn next(&mut self, close: f64) -> BhBar {
        self.update(close)
    }
}

impl<T: Close> Next<&T> for BhMass {
    type Output = BhBar;
    fn next(&mut self, bar: &T) -> BhBar {
        self.update(bar.close())
    }
}

impl Reset for BhMass {
    fn reset(&mut self) {
        BhMass::reset(self)
    }
}

impl Next<f64> for HawkingZ {
    type Output = HawkingBar;
    fn next(&mut self, close: f64) -> HawkingBar {
        self.update(close)
    }
}

impl<T: Close> Next<&T> for HawkingZ {
    type Output = HawkingBar;
    fn next(&mut self, bar: &T) -> HawkingBar {
        self.update(bar.close())
    }
}

impl Reset for HawkingZ {
    fn reset(&mut self) {
        HawkingZ::reset(self)
    }
}

impl Period for HawkingZ {
    fn period(&self) -> usize {
        self.period
    }
}

/// `(close, atr)`.
impl Next<(f64, f64)> for Geodesic {
    type Output = GeoBar;
    fn next(&mut self, (close, atr): (f64, f64)) -> GeoBar {
        self.update(close, atr)
    }
}

impl Reset for Geodesic {
    fn reset(&mut self) {
        Geodesic::reset(self)
    }
}

impl Period for Geodesic {
    fn period(&self) -> usize {
        self.window
    }
}

impl<T: High + Low + Close> Next<&T> for WilderAtr {
    type Output = f64;
    fn next(&mut self, bar: &T) -> f64 {
        self.update(bar.high(), bar.low(), bar.close())
    }
}

impl Reset for WilderAtr {
    fn reset(&mut self) {
        WilderAtr::reset(self)
    }
}

impl Period for WilderAtr {
    fn period(&self) -> usize {
        self.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BhParams;

    struct Bar(f64);
    impl Close for Bar {
        fn close(&self) -> f64 {
            self.0
        }
    }

    #[test]
    fn next_matches_update_and_reset_works() {
        let closes: Vec<f64> = (0..40).map(|i| 100.0 * 1.0007_f64.powi(i)).collect();
        let mut a = BhMass::new(BhParams::default()).unwrap();
        let mut b = a.clone();
        for &c in &closes {
            assert_eq!(Next::next(&mut a, c), b.update(c));
        }
        Reset::reset(&mut a);
        let bars: Vec<Bar> = closes.iter().map(|&c| Bar(c)).collect();
        let mut fresh = BhMass::new(BhParams::default()).unwrap();
        for bar in &bars {
            assert_eq!(a.next(bar), fresh.update(bar.0));
        }
    }
}
