//! SRFM bar transforms, carved out of [srfm-lab](https://github.com/Mattbusel/srfm-lab).
//!
//! Four per-bar transforms over close prices, each available as a slice
//! function and as a streaming struct:
//!
//! | Transform | Slice function | Streaming struct |
//! |---|---|---|
//! | Minkowski beta, `abs(dC) / C_prev / cf` | [`beta`] | [`Beta`] |
//! | Black-hole mass and well state | [`bh_mass`] | [`BhMass`] |
//! | Rolling log-linear geodesic + causal fraction | [`geodesic`] | [`Geodesic`] |
//! | Bollinger z and Hawking proxy `z * (z - z_prev)` | [`hawking_z`] | [`HawkingZ`] |
//!
//! plus [`wilder_atr`] / [`WilderAtr`] for the geodesic's ATR input.
//!
//! The slice functions are thin loops over the streaming structs, so the two
//! can never disagree. Both are checked against `golden/golden_vectors.json`,
//! which is generated from srfm-lab's untouched `lib/srfm_core.py` and shared
//! with the Python package `srfm-core`.
//!
//! # Conventions (shared with the Python package)
//!
//! * A non-finite close (NaN, inf) is a gap: its output is NaN (inactive, dir
//!   0, ctl 0), the state is untouched, and the next finite bar is compared
//!   with the last finite close.
//! * Slice functions reject finite closes `<= 0` with [`Error::NonPositiveClose`];
//!   the streaming structs treat them as gaps.
//! * Warm-up outputs are NaN, never a fake 0.
//!
//! ```
//! use srfm_core::{bh_mass, BhParams};
//!
//! let close: Vec<f64> = (0..40).map(|i| 100.0 * 1.0008_f64.powi(i)).collect();
//! let bh = bh_mass(&close, &BhParams::default()).unwrap();
//! assert!(bh.active.iter().any(|&a| a));      // steady ordered moves form a well
//! ```
//!
//! These are feature transforms. Nothing here claims a trading edge.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::VecDeque;
use std::fmt;

#[cfg(feature = "ta")]
mod ta_impl;

/// Denominator guard used throughout LARSA, kept so results match bit for bit.
pub const EPS: f64 = 1e-9;

/// Errors from the slice functions.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A finite close at `index` was zero or negative.
    NonPositiveClose {
        /// Position in the input slice.
        index: usize,
    },
    /// Input slices that must be aligned have different lengths.
    LengthMismatch,
    /// A parameter is out of range; the string names it.
    InvalidParam(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NonPositiveClose { index } => {
                write!(
                    f,
                    "close at index {index} is <= 0 (non-finite values are gaps)"
                )
            }
            Error::LengthMismatch => f.write_str("input slices have different lengths"),
            Error::InvalidParam(p) => write!(f, "invalid parameter: {p}"),
        }
    }
}

impl std::error::Error for Error {}

fn check_close(close: &[f64]) -> Result<(), Error> {
    match close.iter().position(|&c| c.is_finite() && c <= 0.0) {
        Some(index) => Err(Error::NonPositiveClose { index }),
        None => Ok(()),
    }
}

fn check_cf(cf: f64) -> Result<(), Error> {
    if cf.is_finite() && cf > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidParam("cf must be finite and > 0"))
    }
}

#[inline]
fn usable(c: f64) -> bool {
    c.is_finite() && c > 0.0
}

// ---------------------------------------------------------------------------
// Beta
// ---------------------------------------------------------------------------

/// Streaming Minkowski beta: `abs(C_t - C_{t-1}) / C_{t-1} / cf`.
///
/// `beta < 1` is a TIMELIKE bar, `beta >= 1` SPACELIKE. NaN on the first
/// usable bar and on gaps.
#[derive(Debug, Clone, PartialEq)]
pub struct Beta {
    cf: f64,
    prev: Option<f64>,
}

impl Beta {
    /// New classifier with speed-of-light constant `cf` (LARSA: 0.001).
    pub fn new(cf: f64) -> Result<Self, Error> {
        check_cf(cf)?;
        Ok(Self { cf, prev: None })
    }

    /// Feed one close.
    pub fn update(&mut self, close: f64) -> f64 {
        if !usable(close) {
            return f64::NAN;
        }
        let out = match self.prev {
            Some(p) => (close - p).abs() / (p + EPS) / self.cf,
            None => f64::NAN,
        };
        self.prev = Some(close);
        out
    }

    /// Forget all state.
    pub fn reset(&mut self) {
        self.prev = None;
    }
}

/// Beta for every bar. See [`Beta`].
pub fn beta(close: &[f64], cf: f64) -> Result<Vec<f64>, Error> {
    check_close(close)?;
    let mut b = Beta::new(cf)?;
    Ok(close.iter().map(|&c| b.update(c)).collect())
}

// ---------------------------------------------------------------------------
// Black-hole mass
// ---------------------------------------------------------------------------

/// Black-hole mass parameters. `Default` is what LARSA shipped with.
///
/// Per bar, with `br = (C - C_prev) / C_prev` and the bar TIMELIKE when
/// `abs(br) / cf < 1`:
///
/// * TIMELIKE: `ctl += 1`, `sb = min(boost_cap, 1 + ctl * boost_step)`,
///   `mass = mass * timelike_decay + abs(br) * gain * sb`
/// * SPACELIKE: `ctl = 0`, `mass *= spacelike_decay`
///
/// A well forms when `mass > bh_form && ctl >= ctl_min` and stays active
/// while `mass > bh_collapse && ctl >= ctl_min`. Within `reform_window` bars
/// of a collapse the next formation attempt adds `reform_frac` of the
/// collapse mass (once).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BhParams {
    /// Speed-of-light constant as a fraction of price per bar (0.001).
    pub cf: f64,
    /// Formation threshold (1.5).
    pub bh_form: f64,
    /// Collapse threshold while active (1.0).
    pub bh_collapse: f64,
    /// Mass multiplier on TIMELIKE bars before accretion (0.95).
    pub timelike_decay: f64,
    /// Mass multiplier on SPACELIKE bars (0.7).
    pub spacelike_decay: f64,
    /// Minimum consecutive TIMELIKE bars for a well (5).
    pub ctl_min: u32,
    /// Accretion scale on `abs(br)` (100).
    pub gain: f64,
    /// Per-bar growth of the run boost (0.1).
    pub boost_step: f64,
    /// Cap on the run boost (2.0).
    pub boost_cap: f64,
    /// Bars after a collapse during which the reform boost applies (15).
    pub reform_window: u32,
    /// Share of the collapse mass added back on reform (0.5).
    pub reform_frac: f64,
}

impl Default for BhParams {
    fn default() -> Self {
        Self {
            cf: 0.001,
            bh_form: 1.5,
            bh_collapse: 1.0,
            timelike_decay: 0.95,
            spacelike_decay: 0.7,
            ctl_min: 5,
            gain: 100.0,
            boost_step: 0.1,
            boost_cap: 2.0,
            reform_window: 15,
            reform_frac: 0.5,
        }
    }
}

impl BhParams {
    /// Check the parameters are usable.
    pub fn validate(&self) -> Result<(), Error> {
        check_cf(self.cf)?;
        let finite = [
            self.bh_form,
            self.bh_collapse,
            self.timelike_decay,
            self.spacelike_decay,
            self.gain,
            self.boost_step,
            self.boost_cap,
            self.reform_frac,
        ];
        if finite.iter().all(|x| x.is_finite()) {
            Ok(())
        } else {
            Err(Error::InvalidParam("black-hole parameters must be finite"))
        }
    }
}

/// One bar of black-hole output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BhBar {
    /// Mass after this bar (NaN on a gap).
    pub mass: f64,
    /// Whether a well is active after this bar.
    pub active: bool,
    /// +1 long well, -1 short well, 0 undetermined.
    pub dir: i8,
    /// Consecutive TIMELIKE bars.
    pub ctl: u32,
}

impl BhBar {
    const GAP: BhBar = BhBar {
        mass: f64::NAN,
        active: false,
        dir: 0,
        ctl: 0,
    };
}

/// Streaming black-hole mass. See [`BhParams`] for the recurrence.
#[derive(Debug, Clone, PartialEq)]
pub struct BhMass {
    p: BhParams,
    prev: Option<f64>,
    mass: f64,
    active: bool,
    dir: i8,
    ctl: u32,
    cum: f64,
    prev_mass: f64,
    reform: u32,
}

impl BhMass {
    /// New detector.
    pub fn new(params: BhParams) -> Result<Self, Error> {
        params.validate()?;
        Ok(Self {
            p: params,
            prev: None,
            mass: 0.0,
            active: false,
            dir: 0,
            ctl: 0,
            cum: 0.0,
            prev_mass: 0.0,
            reform: 0,
        })
    }

    /// The parameters in use.
    pub fn params(&self) -> &BhParams {
        &self.p
    }

    /// Feed one close.
    pub fn update(&mut self, close: f64) -> BhBar {
        if !usable(close) {
            return BhBar::GAP;
        }
        if let Some(prev) = self.prev {
            let p = &self.p;
            let diff = close - prev;
            let timelike = diff.abs() / (prev + EPS) / p.cf < 1.0;
            let br = diff / (prev + EPS);
            if timelike {
                self.ctl += 1;
                self.cum += br;
                let sb = p.boost_cap.min(1.0 + self.ctl as f64 * p.boost_step);
                self.mass = self.mass * p.timelike_decay + br.abs() * p.gain * sb;
            } else {
                self.ctl = 0;
                self.mass *= p.spacelike_decay;
            }
            if self.cum > 0.0 {
                self.dir = 1;
            } else if self.cum < 0.0 {
                self.dir = -1;
            }
            let was = self.active;
            if !was {
                if self.reform > 0 && self.reform < p.reform_window {
                    self.mass += self.prev_mass * p.reform_frac;
                    self.prev_mass = 0.0;
                }
                self.active = self.mass > p.bh_form && self.ctl >= p.ctl_min;
            } else {
                self.active = self.mass > p.bh_collapse && self.ctl >= p.ctl_min;
            }
            if self.active && !was {
                self.reform = 0;
            } else if was && !self.active {
                self.prev_mass = self.mass;
                self.reform = 1;
            }
            if !self.active {
                if self.reform > 0 {
                    self.reform += 1;
                }
                if self.ctl == 0 {
                    self.cum = 0.0;
                }
            }
        }
        self.prev = Some(close);
        BhBar {
            mass: self.mass,
            active: self.active,
            dir: self.dir,
            ctl: self.ctl,
        }
    }

    /// Forget all state, keep the parameters.
    pub fn reset(&mut self) {
        *self = Self::new(self.p).expect("parameters were validated");
    }
}

/// Column-oriented black-hole output, one entry per input bar.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BhSeries {
    /// Mass per bar (NaN on gaps).
    pub mass: Vec<f64>,
    /// Active flag per bar.
    pub active: Vec<bool>,
    /// Direction per bar.
    pub dir: Vec<i8>,
    /// Consecutive TIMELIKE count per bar.
    pub ctl: Vec<u32>,
}

/// Black-hole mass for every bar. See [`BhParams`].
pub fn bh_mass(close: &[f64], params: &BhParams) -> Result<BhSeries, Error> {
    check_close(close)?;
    let mut bh = BhMass::new(*params)?;
    let mut s = BhSeries {
        mass: Vec::with_capacity(close.len()),
        active: Vec::with_capacity(close.len()),
        dir: Vec::with_capacity(close.len()),
        ctl: Vec::with_capacity(close.len()),
    };
    for &c in close {
        let b = bh.update(c);
        s.mass.push(b.mass);
        s.active.push(b.active);
        s.dir.push(b.dir);
        s.ctl.push(b.ctl);
    }
    Ok(s)
}

// ---------------------------------------------------------------------------
// Geodesic
// ---------------------------------------------------------------------------

/// One bar of geodesic output (all NaN during warm-up).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoBar {
    /// `tanh((C - fitted C) / atr)`.
    pub dev: f64,
    /// 100 * slope of log price per bar.
    pub slope: f64,
    /// Share of lags `1..window` whose interval is TIMELIKE.
    pub causal_frac: f64,
}

impl GeoBar {
    const NAN: GeoBar = GeoBar {
        dev: f64::NAN,
        slope: f64::NAN,
        causal_frac: f64::NAN,
    };
}

/// Streaming rolling log-linear "geodesic" over the last `window` closes.
///
/// LARSA's `rapidity` output is not ported: as shipped it is identically 0.
#[derive(Debug, Clone, PartialEq)]
pub struct Geodesic {
    cf: f64,
    window: usize,
    buf: VecDeque<f64>,
    lp: Vec<f64>,
}

impl Geodesic {
    /// New analyzer. `window >= 2` (LARSA: 20), `cf` as in [`Beta`].
    pub fn new(window: usize, cf: f64) -> Result<Self, Error> {
        check_cf(cf)?;
        if window < 2 {
            return Err(Error::InvalidParam("window must be >= 2"));
        }
        Ok(Self {
            cf,
            window,
            buf: VecDeque::with_capacity(window),
            lp: vec![0.0; window],
        })
    }

    /// Feed one close and its ATR (NaN ATR gives NaN `dev` only).
    pub fn update(&mut self, close: f64, atr: f64) -> GeoBar {
        if !usable(close) {
            return GeoBar::NAN;
        }
        if self.buf.len() == self.window {
            self.buf.pop_front();
        }
        self.buf.push_back(close);
        if self.buf.len() < self.window {
            return GeoBar::NAN;
        }
        let w = self.window;
        let n = w as f64;
        let (mut sx, mut sxx, mut slp, mut sxlp) = (0.0, 0.0, 0.0, 0.0);
        for (j, &c) in self.buf.iter().enumerate() {
            let x = j as f64;
            let l = (c + EPS).ln();
            self.lp[j] = l;
            sx += x;
            sxx += x * x;
            slp += l;
            sxlp += x * l;
        }
        let slope = (n * sxlp - sx * slp) / (n * sxx - sx * sx + EPS);
        let intercept = (slp - slope * sx) / n;
        let geo_p = (slope * (w as f64 - 1.0) + intercept).exp();
        let dev = ((close - geo_p) / (atr + EPS)).tanh();
        let mut cc = 0usize;
        for k in 1..w {
            let pp = self.buf[w - 1 - k];
            let dp = (close - pp).abs() / (pp + EPS);
            let ck = self.cf * k as f64;
            if -(ck * ck) + dp * dp < 0.0 {
                cc += 1;
            }
        }
        GeoBar {
            dev,
            slope: slope * 100.0,
            causal_frac: cc as f64 / (w as f64 - 1.0),
        }
    }

    /// Forget all state.
    pub fn reset(&mut self) {
        self.buf.clear();
    }
}

/// Column-oriented geodesic output.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GeoSeries {
    /// See [`GeoBar::dev`].
    pub dev: Vec<f64>,
    /// See [`GeoBar::slope`].
    pub slope: Vec<f64>,
    /// See [`GeoBar::causal_frac`].
    pub causal_frac: Vec<f64>,
}

/// Geodesic for every bar. `atr` is aligned with `close`.
pub fn geodesic(close: &[f64], atr: &[f64], window: usize, cf: f64) -> Result<GeoSeries, Error> {
    if close.len() != atr.len() {
        return Err(Error::LengthMismatch);
    }
    check_close(close)?;
    let mut g = Geodesic::new(window, cf)?;
    let mut s = GeoSeries::default();
    for (&c, &a) in close.iter().zip(atr) {
        let b = g.update(c, a);
        s.dev.push(b.dev);
        s.slope.push(b.slope);
        s.causal_frac.push(b.causal_frac);
    }
    Ok(s)
}

// ---------------------------------------------------------------------------
// Hawking z
// ---------------------------------------------------------------------------

/// One bar of Hawking output (NaN in warm-up and where the rolling std is 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HawkingBar {
    /// `(C - mean) / std` over the last `period` closes (population std).
    pub z: f64,
    /// `z * (z - z_prev)`, `z_prev` = previous defined z (0 before the first).
    pub t: f64,
}

impl HawkingBar {
    const NAN: HawkingBar = HawkingBar {
        z: f64::NAN,
        t: f64::NAN,
    };
}

/// Streaming Bollinger z and the LARSA Hawking-temperature proxy.
#[derive(Debug, Clone, PartialEq)]
pub struct HawkingZ {
    period: usize,
    buf: VecDeque<f64>,
    pz: f64,
}

impl HawkingZ {
    /// New monitor, `period >= 2` (LARSA: 20).
    pub fn new(period: usize) -> Result<Self, Error> {
        if period < 2 {
            return Err(Error::InvalidParam("period must be >= 2"));
        }
        Ok(Self {
            period,
            buf: VecDeque::with_capacity(period),
            pz: 0.0,
        })
    }

    /// Feed one close.
    pub fn update(&mut self, close: f64) -> HawkingBar {
        if !usable(close) {
            return HawkingBar::NAN;
        }
        if self.buf.len() == self.period {
            self.buf.pop_front();
        }
        self.buf.push_back(close);
        if self.buf.len() < self.period {
            return HawkingBar::NAN;
        }
        let n = self.period as f64;
        let mean = self.buf.iter().sum::<f64>() / n;
        let var = self
            .buf
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>()
            / n;
        let std = var.sqrt();
        if std.is_nan() || std <= 0.0 {
            return HawkingBar::NAN;
        }
        let z = (close - mean) / std;
        let t = z * (z - self.pz);
        self.pz = z;
        HawkingBar { z, t }
    }

    /// Forget all state.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.pz = 0.0;
    }
}

/// Column-oriented Hawking output.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HawkingSeries {
    /// See [`HawkingBar::z`].
    pub z: Vec<f64>,
    /// See [`HawkingBar::t`].
    pub t: Vec<f64>,
}

/// Hawking z and proxy for every bar.
pub fn hawking_z(close: &[f64], period: usize) -> Result<HawkingSeries, Error> {
    check_close(close)?;
    let mut h = HawkingZ::new(period)?;
    let mut s = HawkingSeries::default();
    for &c in close {
        let b = h.update(c);
        s.z.push(b.z);
        s.t.push(b.t);
    }
    Ok(s)
}

// ---------------------------------------------------------------------------
// Wilder ATR
// ---------------------------------------------------------------------------

/// Streaming Wilder ATR with srfm-lab's warm-up: a running mean of true
/// ranges for the first `period` bars, Wilder smoothing after. NaN on the
/// first usable bar.
#[derive(Debug, Clone, PartialEq)]
pub struct WilderAtr {
    period: usize,
    prev: Option<f64>,
    val: f64,
    n: usize,
}

impl WilderAtr {
    /// New ATR, `period >= 1` (LARSA: 14).
    pub fn new(period: usize) -> Result<Self, Error> {
        if period < 1 {
            return Err(Error::InvalidParam("period must be >= 1"));
        }
        Ok(Self {
            period,
            prev: None,
            val: 0.0,
            n: 0,
        })
    }

    /// Feed one bar.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        if !usable(close) {
            return f64::NAN;
        }
        let out = match self.prev {
            None => f64::NAN,
            Some(pc) => {
                let tr = (high - low).max((high - pc).abs()).max((low - pc).abs());
                let p = self.period as f64;
                self.val = if self.n < self.period {
                    (self.val * self.n as f64 + tr) / (self.n as f64 + 1.0)
                } else {
                    (self.val * (p - 1.0) + tr) / p
                };
                self.n += 1;
                self.val
            }
        };
        self.prev = Some(close);
        out
    }

    /// Forget all state.
    pub fn reset(&mut self) {
        self.prev = None;
        self.val = 0.0;
        self.n = 0;
    }
}

/// Wilder ATR for every bar.
pub fn wilder_atr(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Result<Vec<f64>, Error> {
    if high.len() != close.len() || low.len() != close.len() {
        return Err(Error::LengthMismatch);
    }
    check_close(close)?;
    let mut a = WilderAtr::new(period)?;
    Ok((0..close.len())
        .map(|i| a.update(high[i], low[i], close[i]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_hand_computed() {
        let b = beta(&[5000.0, 5002.5], 0.001).unwrap();
        assert!(b[0].is_nan());
        assert!((b[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn spacelike_bar_kills_the_well() {
        let mut close: Vec<f64> = (0..30).map(|i| 100.0 * 1.0008_f64.powi(i)).collect();
        close.push(110.0);
        let s = bh_mass(&close, &BhParams::default()).unwrap();
        let n = close.len();
        assert!(s.active[n - 2]);
        assert!(!s.active[n - 1]);
        assert_eq!(s.ctl[n - 1], 0);
        assert!((s.mass[n - 1] - s.mass[n - 2] * 0.7).abs() < 1e-12);
        assert!(s.dir[1..].iter().all(|&d| d == 1));
    }

    #[test]
    fn gaps_are_skipped() {
        let clean: Vec<f64> = (0..40).map(|i| 100.0 * 1.0005_f64.powi(i)).collect();
        let mut gappy = clean.clone();
        gappy.insert(10, f64::NAN);
        let a = bh_mass(&clean, &BhParams::default()).unwrap();
        let b = bh_mass(&gappy, &BhParams::default()).unwrap();
        assert!(b.mass[10].is_nan());
        let mut bm = b.mass.clone();
        bm.remove(10);
        assert_eq!(bm, a.mass);
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(
            bh_mass(&[1.0, 0.0], &BhParams::default()),
            Err(Error::NonPositiveClose { index: 1 })
        );
        assert!(beta(&[1.0], 0.0).is_err());
        assert!(geodesic(&[1.0, 2.0], &[1.0], 2, 0.001).is_err());
        assert!(HawkingZ::new(1).is_err());
    }

    #[test]
    fn reset_restores_fresh_state() {
        let close: Vec<f64> = (0..50).map(|i| 100.0 * 1.0006_f64.powi(i)).collect();
        let mut bh = BhMass::new(BhParams::default()).unwrap();
        let first: Vec<BhBar> = close.iter().map(|&c| bh.update(c)).collect();
        bh.reset();
        let second: Vec<BhBar> = close.iter().map(|&c| bh.update(c)).collect();
        assert_eq!(first, second);
    }
}
