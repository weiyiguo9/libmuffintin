//! Multipole moments and spherical-Bessel radial integrals (`coulombmatrix.f:242-309`, `1331-1391`).

use crate::CoulombError;
use libmuffintin_core::{ExponentialMesh, spherical_bessel_j};
use std::f64::consts::PI;

/// $\int r^{L+1} b(r)\,\mathrm{d}r$ (`coulombmatrix.f:247`).
pub fn multipole_moment(l: u32, mesh: &ExponentialMesh, basm: &[f64]) -> Result<f64, CoulombError> {
    let integrand: Vec<f64> = mesh
        .radii()
        .iter()
        .zip(basm)
        .map(|(radius, sample)| sample * radius.get().powi(l as i32 + 1))
        .collect();
    Ok(mesh.integrate(&integrand)?)
}

/// $\int r^{3} b(r)\,\mathrm{d}r$ for $L=0$ (`coulombmatrix.f:251`).
pub fn second_moment(mesh: &ExponentialMesh, basm: &[f64]) -> Result<f64, CoulombError> {
    let integrand: Vec<f64> = mesh
        .radii()
        .iter()
        .zip(basm)
        .map(|(radius, sample)| sample * radius.get().powi(3))
        .collect();
    Ok(mesh.integrate(&integrand)?)
}

/// $\int_0^R r^{L+2} j_L(qr)\,\mathrm{d}r$ (`coulombmatrix.f:278-288`).
///
/// At $q=0$ only $L=0$ survives: $R^3/3$. Otherwise $R^{L+2} j_{L+1}(qR)/q$.
pub fn spherical_bessel_moment(l: u32, q_norm: f64, radius: f64) -> f64 {
    if q_norm.abs() <= 1.0e-14 {
        if l == 0 { radius.powi(3) / 3.0 } else { 0.0 }
    } else {
        radius.powi(l as i32 + 2) * spherical_bessel_j(l + 1, q_norm * radius) / q_norm
    }
}

/// Overlap $\int r\, b(r)\, j_L(qr)\,\mathrm{d}r$ (`coulombmatrix.f:304`).
pub fn bessel_overlap(
    l: u32,
    q_norm: f64,
    mesh: &ExponentialMesh,
    basm: &[f64],
) -> Result<f64, CoulombError> {
    let integrand: Vec<f64> = mesh
        .radii()
        .iter()
        .zip(basm)
        .map(|(radius, sample)| {
            let r = radius.get();
            sample * r * spherical_bessel_j(l, q_norm * r)
        })
        .collect();
    Ok(mesh.integrate(&integrand)?)
}

/// SPEX `sphbesmoment1` (`coulombmatrix.f:261-298`).
pub fn weinert_bessel_kernel(l: u32, q_norm: f64, r: f64, radius: f64) -> f64 {
    if q_norm.abs() <= 1.0e-14 {
        if l == 0 {
            r * r / 3.0 + (radius * radius - r * r) / 2.0
        } else {
            0.0
        }
    } else if l == 0 {
        let j1_r = spherical_bessel_j(1, q_norm * r);
        let inner = r * j1_r / q_norm;
        let outer_radius = -(q_norm * radius).cos() / q_norm;
        let outer_r = -(q_norm * r).cos() / q_norm;
        (inner + (outer_radius - outer_r)) / q_norm
    } else {
        let jl1_r = spherical_bessel_j(l + 1, q_norm * r);
        let jlm1_r = spherical_bessel_j(l - 1, q_norm * r);
        let jlm1_radius = spherical_bessel_j(l - 1, q_norm * radius);
        let rdum1_r = -r.powi(1 - l as i32) * jlm1_r;
        let rdum1_radius = -radius.powi(1 - l as i32) * jlm1_radius;
        (r * jl1_r + (rdum1_radius - rdum1_r) * r.powi(l as i32)) / q_norm
    }
}

/// $\int r\, b(r)\, \mathrm{sphbesmoment1}(r,L)\,\mathrm{d}r$ (`coulombmatrix.f:305`).
pub fn bessel_weinert_integral(
    l: u32,
    q_norm: f64,
    mesh: &ExponentialMesh,
    basm: &[f64],
) -> Result<f64, CoulombError> {
    let radius = mesh.last().get();
    let integrand: Vec<f64> = mesh
        .radii()
        .iter()
        .zip(basm)
        .map(|(sample_r, sample)| {
            let r = sample_r.get();
            sample * r * weinert_bessel_kernel(l, q_norm, r, radius)
        })
        .collect();
    Ok(mesh.integrate(&integrand)?)
}

/// SPEX `sphbessel_integral` (`coulombmatrix.f:1331-1391`) for the same-MT PW-PW kernel.
pub fn sphbessel_pw_integral(l: u32, q1: f64, q2: f64, radius: f64) -> f64 {
    let s = radius;
    if q1.abs() <= 1.0e-14 && q2.abs() <= 1.0e-14 {
        if l > 0 { 0.0 } else { 2.0 * s.powi(5) / 15.0 }
    } else if q1.abs() <= 1.0e-14 || q2.abs() <= 1.0e-14 {
        if l > 0 {
            0.0
        } else if q1.abs() <= 1.0e-14 {
            s.powi(3) / (3.0 * q2 * q2)
                * (q2 * s * spherical_bessel_j(1, q2 * s) + spherical_bessel_j(2, q2 * s))
        } else {
            s.powi(3) / (3.0 * q1 * q1)
                * (q1 * s * spherical_bessel_j(1, q1 * s) + spherical_bessel_j(2, q1 * s))
        }
    } else if (q1 - q2).abs() < 1.0e-6 {
        let sb = |n: i32| -> f64 {
            if n < 0 {
                (q1 * s).cos() / (q1 * s)
            } else {
                spherical_bessel_j(n as u32, q1 * s)
            }
        };
        let dq = q2 - q1;
        s.powi(3) / (2.0 * q1 * q1)
            * ((2 * l + 3) as f64 * sb(l as i32 + 1).powi(2)
                - (2 * l + 1) as f64 * sb(l as i32) * sb(l as i32 + 2)
                + ((2 * l + 3) as f64 * 5.0 * sb(l as i32 + 1) * sb(l as i32 - 1)
                    - (sb(l as i32 + 1) + 5.0 * sb(l as i32 - 1)) * q1 * s * sb(l as i32))
                    * dq
                    / (2.0 * q1))
    } else {
        let sb = |q: f64, n: i32| -> f64 {
            if n < 0 {
                if (q * s).abs() <= 1.0e-14 {
                    0.0
                } else {
                    (q * s).cos() / (q * s)
                }
            } else {
                spherical_bessel_j(n as u32, q * s)
            }
        };
        let sb01 = sb(q1, l as i32 - 1);
        let sb11 = sb(q1, l as i32);
        let sb21 = sb(q1, l as i32 + 1);
        let sb31 = sb(q1, l as i32 + 2);
        let sb02 = sb(q2, l as i32 - 1);
        let sb12 = sb(q2, l as i32);
        let sb22 = sb(q2, l as i32 + 1);
        let sb32 = sb(q2, l as i32 + 2);
        let dq = q1 * q1 - q2 * q2;
        let a1 = q2 / q1 * sb21 * sb02;
        let a2 = q1 / q2 * sb22 * sb01;
        let da = a1 - a2;
        let b1 = sb31 * sb12;
        let b2 = sb32 * sb11;
        let db = b1 - b2;
        let c1 = sb21 * sb22 / (q1 * q2);
        let c2 = db / dq * (2 * l + 1) as f64 / (2 * l + 3) as f64;
        let dc = c1 + c2;
        let r1 = if a1.abs() == 0.0 {
            f64::INFINITY
        } else {
            (da / a1).abs()
        };
        let r2 = if b1.abs() == 0.0 {
            f64::INFINITY
        } else {
            (db / b1).abs().min(if c1.abs() == 0.0 {
                f64::INFINITY
            } else {
                (dc / c1).abs()
            })
        };
        if r1 > r2 {
            s.powi(3) / dq * da
        } else {
            s.powi(3) * dc
        }
    }
}

/// Sphere form factor $\int_{\mathrm{MT}} \exp(i\mathbf{G}\cdot\mathbf{r})\,\mathrm{d}^3r$.
pub fn sphere_plane_wave_integral(gnorm: f64, radius: f64) -> f64 {
    if gnorm.abs() <= 1.0e-14 {
        4.0 * PI * radius.powi(3) / 3.0
    } else {
        let x = radius * gnorm;
        4.0 * PI * (x.sin() - x * x.cos()) / gnorm.powi(3)
    }
}
