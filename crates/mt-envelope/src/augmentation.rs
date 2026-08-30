//! APW value/slope matching and site augmentation coefficients.

use crate::BasisError;
use crate::{PlaneWave, rayleigh_coefficient, site_translation_phase};
use muffintin_core::{
    Bohr, InverseBohr, Kappa, Lm, RelativisticChannel, SpinProjection, VolumeBohr3, lm_count,
    spherical_bessel_j, spherical_bessel_j_derivative,
};
use muffintin_sphere::BoundaryData;
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

/// One `kappa`-resolved SRA value/slope match used by spinor augmentation.
///
/// The match must already have been formed from the large-component boundary
/// pair `(U, U_r)`.  This type deliberately contains no Dirac `(P, Q)` trace,
/// preventing a four-component trace from being passed to the scalar
/// value/slope matcher by accident.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinorApwMatch {
    pub kappa: Kappa,
    pub apw: ApwMatch,
}

/// SRA augmentation of one spatial plane wave for both Pauli-spin columns.
///
/// `channels` is in canonical `(kappa, twice_mu)` order.  For each spin
/// (`0`, then `1`), `coefficients[spin][channel]` stores the two coefficients
/// multiplying that `kappa` channel's radial `(u, udot)` columns.  A compiled
/// site projection therefore traverses global columns as `spin`, then `g`,
/// and reads `site_augmentations[site][g].coefficient(spin, channel)`.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorPlaneWaveAugmentation {
    pub channels: Vec<RelativisticChannel>,
    pub coefficients: [Vec<[Complex64; 2]>; 2],
}

impl SpinorPlaneWaveAugmentation {
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    pub fn augmented_site_coordinate_count(&self) -> usize {
        2 * self.channel_count()
    }

    pub fn coefficient(&self, spin: usize, channel: usize) -> &[Complex64; 2] {
        &self.coefficients[spin][channel]
    }

    /// Row index in a site's augmented radial coordinate block.
    ///
    /// Rows use `(channel, radial_column)` order with `radial_column` fastest:
    /// `2 * channel + radial_column`.  Explicit local orbitals follow this
    /// entire `0..2*channel_count` block in a site projection.
    pub fn site_coordinate_index(&self, channel: usize, radial_column: usize) -> Option<usize> {
        if channel >= self.channel_count() || radial_column >= 2 {
            return None;
        }
        Some(2 * channel + radial_column)
    }
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

/// Build the SRA spinor augmentation of one spatial plane wave.
///
/// Each `SpinorApwMatch` is the result of matching the large-component SRA
/// boundary pair `(U, U_r)` for one `kappa`; a Dirac `(P, Q)` trace is not an
/// accepted input.  Output channels are ordered by increasing signed `kappa`
/// and then increasing exact `twice_mu`.  For every channel and incident
/// Pauli spin, the coefficient is
/// `site phase * Rayleigh(l,m_l) * CG * [a_kappa,b_kappa]`.
pub fn spinor_augmentation_coefficients(
    plane_wave: &PlaneWave,
    site: [Bohr; 3],
    cell_volume: VolumeBohr3,
    matches: &[SpinorApwMatch],
) -> Result<SpinorPlaneWaveAugmentation, BasisError> {
    let mut matches = matches.to_vec();
    matches.sort_unstable_by_key(|matched| matched.kappa.get());
    for pair in matches.windows(2) {
        if pair[0].kappa == pair[1].kappa {
            return Err(BasisError::DuplicateSpinorMatch {
                kappa: pair[0].kappa.get(),
            });
        }
    }

    let phase = site_translation_phase(plane_wave.q, site);
    let channel_count = matches
        .iter()
        .map(|matched| matched.kappa.degeneracy() as usize)
        .sum();
    let mut channels = Vec::with_capacity(channel_count);
    let mut coefficients = [
        Vec::with_capacity(channel_count),
        Vec::with_capacity(channel_count),
    ];
    for matched in matches {
        let expected_l = matched.kappa.large_l();
        if matched.apw.l != expected_l {
            return Err(BasisError::SpinorMatchAngularMomentum {
                kappa: matched.kappa.get(),
                expected: expected_l,
                actual: matched.apw.l,
            });
        }
        for channel in matched.kappa.channels() {
            let mut channel_coefficients = [[Complex64::default(); 2]; 2];
            for term in channel.spinor_harmonic_terms().into_iter().flatten() {
                let spin = match term.spin {
                    SpinProjection::Up => 0,
                    SpinProjection::Down => 1,
                };
                let angular = phase
                    * rayleigh_coefficient(term.orbital, plane_wave.q, cell_volume)?
                    * term.coefficient;
                channel_coefficients[spin] = [
                    angular * matched.apw.coefficients[0],
                    angular * matched.apw.coefficients[1],
                ];
            }
            channels.push(channel);
            coefficients[0].push(channel_coefficients[0]);
            coefficients[1].push(channel_coefficients[1]);
        }
    }
    Ok(SpinorPlaneWaveAugmentation {
        channels,
        coefficients,
    })
}
