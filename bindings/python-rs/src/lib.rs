//! Thin pyo3 layer over the `srfm-core` crate: no math lives here.
//!
//! This replaces the physics half of srfm-lab's `larsa-core` crate (whose
//! `bh_mass_series` had drifted from the Python reference; see
//! docs/DRIFT.md). Build with `maturin develop` in this directory.

#![allow(clippy::useless_conversion)] // pyo3 0.22 macro expansion trips this lint

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn err(e: srfm_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// beta(close, cf=0.001) -> list[float]
#[pyfunction]
#[pyo3(signature = (close, cf = 0.001))]
fn beta(close: Vec<f64>, cf: f64) -> PyResult<Vec<f64>> {
    srfm_core::beta(&close, cf).map_err(err)
}

/// bh_mass(close, cf=0.001, bh_form=1.5, bh_collapse=1.0, timelike_decay=0.95,
///         spacelike_decay=0.7, ctl_min=5) -> (mass, active, dir, ctl)
#[pyfunction]
#[pyo3(signature = (close, cf = 0.001, bh_form = 1.5, bh_collapse = 1.0,
                    timelike_decay = 0.95, spacelike_decay = 0.7, ctl_min = 5))]
#[allow(clippy::type_complexity)]
fn bh_mass(
    close: Vec<f64>,
    cf: f64,
    bh_form: f64,
    bh_collapse: f64,
    timelike_decay: f64,
    spacelike_decay: f64,
    ctl_min: u32,
) -> PyResult<(Vec<f64>, Vec<bool>, Vec<i8>, Vec<u32>)> {
    let p = srfm_core::BhParams {
        cf,
        bh_form,
        bh_collapse,
        timelike_decay,
        spacelike_decay,
        ctl_min,
        ..Default::default()
    };
    let s = srfm_core::bh_mass(&close, &p).map_err(err)?;
    Ok((s.mass, s.active, s.dir, s.ctl))
}

/// geodesic(close, atr, window=20, cf=0.001) -> (geo_dev, geo_slope, causal_frac)
#[pyfunction]
#[pyo3(signature = (close, atr, window = 20, cf = 0.001))]
fn geodesic(
    close: Vec<f64>,
    atr: Vec<f64>,
    window: usize,
    cf: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let s = srfm_core::geodesic(&close, &atr, window, cf).map_err(err)?;
    Ok((s.dev, s.slope, s.causal_frac))
}

/// hawking_z(close, period=20) -> (z, t)
#[pyfunction]
#[pyo3(signature = (close, period = 20))]
fn hawking_z(close: Vec<f64>, period: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let s = srfm_core::hawking_z(&close, period).map_err(err)?;
    Ok((s.z, s.t))
}

#[pymodule]
fn srfm_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(beta, m)?)?;
    m.add_function(wrap_pyfunction!(bh_mass, m)?)?;
    m.add_function(wrap_pyfunction!(geodesic, m)?)?;
    m.add_function(wrap_pyfunction!(hawking_z, m)?)?;
    Ok(())
}
