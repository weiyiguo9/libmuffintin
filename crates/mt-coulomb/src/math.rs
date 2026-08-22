//! Factorials, Gaunt-like `gmat`, and plane-wave phases used by SPEX `coulombmatrix.f`.

use crate::CoulombError;
use muffintin_core::{Bohr, InverseBohr, lm_count, lm_index};
use num_complex::Complex64;
use std::f64::consts::PI;

/// `sqrt(n!)` for `n = 0..=max_n`, matching SPEX `sfac` (`getinput.f:1348-1362`).
pub(crate) fn sfac_table(max_n: usize) -> Result<Vec<f64>, CoulombError> {
    let mut table = vec![1.0; max_n + 1];
    for n in 1..=max_n {
        table[n] = table[n - 1] * (n as f64).sqrt();
        if !table[n].is_finite() {
            return Err(CoulombError::FactorialOverflow(n));
        }
    }
    Ok(table)
}

/// `i^l`.
pub(crate) fn i_pow(l: u32) -> Complex64 {
    match l % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

/// `(-1)^n` for `n >= 0`.
pub(crate) fn parity(n: i32) -> f64 {
    if n.rem_euclid(2) == 0 { 1.0 } else { -1.0 }
}

/// SPEX `gmat` (`coulombmatrix.f:223-240`).
///
/// Symmetric in the two `(l,m)` channels. Couples two solid harmonics of ranks
/// `l1` and `l2` into rank `l1+l2`.
pub fn weinert_gmat(l1: u32, m1: i32, l2: u32, m2: i32, sfac: &[f64]) -> Result<f64, CoulombError> {
    let a = (l1 + l2) as i32 + m2 - m1;
    let b = (l1 + l2) as i32 + m1 - m2;
    let c = l1 as i32 + m1;
    let d = l1 as i32 - m1;
    let e = l2 as i32 + m2;
    let f = l2 as i32 - m2;
    if [a, b, c, d, e, f].iter().any(|&n| n < 0) {
        return Ok(0.0);
    }
    let lookup = |n: i32| -> Result<f64, CoulombError> {
        sfac.get(n as usize)
            .copied()
            .ok_or(CoulombError::FactorialOverflow(n as usize))
    };
    let numerator = lookup(a)? * lookup(b)?;
    let denominator = lookup(c)? * lookup(d)? * lookup(e)? * lookup(f)?;
    if denominator == 0.0 {
        return Ok(0.0);
    }
    let angular = ((2 * l1 + 1) as f64 * (2 * l2 + 1) as f64 * (2 * (l1 + l2) + 1) as f64).sqrt();
    Ok(numerator / denominator / angular * (4.0 * PI).powf(1.5))
}

/// Cartesian norm of an `InverseBohr` vector.
pub(crate) fn inverse_norm(vector: [InverseBohr; 3]) -> f64 {
    vector
        .iter()
        .map(|component| component.get().powi(2))
        .sum::<f64>()
        .sqrt()
}

/// `exp(+i q · r)` with Cartesian `q` and `r`.
pub(crate) fn plane_wave_phase(q: [InverseBohr; 3], r: [Bohr; 3]) -> Complex64 {
    let phase = q
        .iter()
        .zip(r)
        .map(|(component, coordinate)| component.get() * coordinate.get())
        .sum();
    Complex64::from_polar(1.0, phase)
}

/// Structure-constant compound index `l(l+1)+m` through `2 lexp`.
pub(crate) fn structure_lm(l: u32, m: i32, l_max: u32) -> Result<usize, CoulombError> {
    if l > l_max || m.unsigned_abs() > l {
        return Err(CoulombError::StructureIndex {
            index: 0,
            count: lm_count(l_max),
        });
    }
    Ok(lm_index(l, m)?)
}

pub(crate) fn is_gamma(q: [InverseBohr; 3]) -> bool {
    inverse_norm(q) <= 1.0e-12
}

pub(crate) fn is_zero_norm(value: f64) -> bool {
    value <= 1.0e-12
}
