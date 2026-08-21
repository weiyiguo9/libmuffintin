//! APW value/slope matching and site augmentation coefficients.

use crate::BasisError;
use libmuffintin_core::{
    Bohr, InverseBohr, Lm, VolumeBohr3, lm_count, spherical_bessel_j, spherical_bessel_j_derivative,
};
use libmuffintin_envelope::{PlaneWave, rayleigh_coefficient, site_translation_phase};
use libmuffintin_radial::BoundaryData;
use num_complex::Complex64;

/// The two primitive radial boundary columns `(u_l, du_l/dE)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApwBoundaryBasis {
    pub u: BoundaryData,
    pub udot: BoundaryData,
}

/// Coefficients and direct substitution residuals of one APW radial match.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApwMatch {
    pub l: u32,
    /// Coefficients multiplying `(u_l, du_l/dE)`.
    pub coefficients: [f64; 2],
    /// `A u(R) + B udot(R) - j_l(qR)`.
    pub value_residual: f64,
    /// `A u'(R) + B udot'(R) - q j_l'(qR)`.
    pub slope_residual: f64,
}

/// APW augmentation coefficients indexed by the public contiguous `lm` index.
/// Each channel stores coefficients multiplying `(u_l, du_l/dE)`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveAugmentation {
    pub coefficients: Vec<[Complex64; 2]>,
}

/// Solve the SPEX `2 x 2` boundary system for a fixed angular momentum.
///
/// The right-hand side is exactly `(j_l(qR), q j_l'(qR))`; the returned
/// residuals are obtained by substituting the computed coefficients back into
/// the un-inverted boundary matrix.
pub fn match_apw_boundary(
    l: u32,
    q: InverseBohr,
    radius: Bohr,
    basis: ApwBoundaryBasis,
) -> Result<ApwMatch, BasisError> {
    if !radius.get().is_finite() || radius.get() <= 0.0 {
        return Err(BasisError::InvalidRadius(radius.get()));
    }
    if !q.get().is_finite() || q.get() < 0.0 {
        return Err(BasisError::InvalidWaveVector(q.get()));
    }
    let determinant = basis.u.value * basis.udot.derivative - basis.udot.value * basis.u.derivative;
    let matrix_scale = (basis.u.value.abs() + basis.udot.value.abs())
        * (basis.u.derivative.abs() + basis.udot.derivative.abs());
    if !determinant.is_finite()
        || determinant.abs() <= 64.0 * f64::EPSILON * matrix_scale.max(f64::MIN_POSITIVE)
    {
        return Err(BasisError::SingularBoundaryMatrix { l, determinant });
    }

    let x = q.get() * radius.get();
    let target_value = spherical_bessel_j(l, x);
    let target_slope = q.get() * spherical_bessel_j_derivative(l, x);
    let a = (basis.udot.derivative * target_value - basis.udot.value * target_slope) / determinant;
    let b = (-basis.u.derivative * target_value + basis.u.value * target_slope) / determinant;

    Ok(ApwMatch {
        l,
        coefficients: [a, b],
        value_residual: a.mul_add(basis.u.value, b * basis.udot.value) - target_value,
        slope_residual: a.mul_add(basis.u.derivative, b * basis.udot.derivative) - target_slope,
    })
}

/// Build all site-centered APW coefficients through `l_max`.
///
/// `matches[l]` is the real boundary match for this plane wave and site type.
/// The output includes both the Rayleigh factor and `exp(+i q dot R_a)`.
pub fn augmentation_coefficients(
    plane_wave: &PlaneWave,
    site: [Bohr; 3],
    cell_volume: VolumeBohr3,
    matches: &[ApwMatch],
) -> Result<PlaneWaveAugmentation, BasisError> {
    let l_max = matches.len().saturating_sub(1) as u32;
    let phase = site_translation_phase(plane_wave.q, site);
    let mut coefficients = Vec::with_capacity(lm_count(l_max));
    for (l, matched) in matches.iter().enumerate() {
        let l = l as u32;
        if matched.l != l {
            return Err(BasisError::MatchAngularMomentum {
                expected: l,
                actual: matched.l,
            });
        }
        for m in -(l as i32)..=l as i32 {
            let angular = phase
                * rayleigh_coefficient(
                    Lm::new(l, m).expect("loop bounds validate m"),
                    plane_wave.q,
                    cell_volume,
                )?;
            coefficients.push([
                angular * matched.coefficients[0],
                angular * matched.coefficients[1],
            ]);
        }
    }
    Ok(PlaneWaveAugmentation { coefficients })
}
