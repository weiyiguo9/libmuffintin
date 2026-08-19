//! Linearized augmented-plane-wave matching and overlap assembly.
//!
//! Plane waves use `exp(+i (k+G) dot r) / sqrt(Omega)`.  The Rayleigh
//! coefficient and the site-translation phase are exposed separately so that
//! neither convention is hidden inside the real, radial boundary solve.

#![forbid(unsafe_code)]

use mt_core::{
    Bohr, GVector, InterstitialGeometry, InverseBohr, Lm, VolumeBohr3, complex_spherical_harmonics,
    lm_count, lm_from_index, spherical_bessel_j, spherical_bessel_j_derivative,
};
use mt_radial::BoundaryData;
use num_complex::Complex64;
use std::f64::consts::PI;
use std::ops::Index;
use thiserror::Error;

/// A normalized plane wave identified by `G`, with Cartesian `q = k + G`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneWave {
    /// Bloch vector in Cartesian bohr^-1.
    pub k: [InverseBohr; 3],
    /// Reciprocal-lattice vector.
    pub g: GVector,
    /// Cartesian `k + G` in bohr^-1.
    pub q: [InverseBohr; 3],
    /// Norm `|k + G|` in bohr^-1.
    pub q_norm: InverseBohr,
}

impl PlaneWave {
    /// Form `q = k + G` without changing reciprocal-lattice coordinates.
    pub fn new(k: [InverseBohr; 3], g: GVector) -> Self {
        let q = std::array::from_fn(|axis| InverseBohr(k[axis].get() + g.cartesian[axis].get()));
        let q_norm = InverseBohr(
            q.iter()
                .map(|component| component.get().powi(2))
                .sum::<f64>()
                .sqrt(),
        );
        Self { k, g, q, q_norm }
    }
}

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

/// LAPW construction or overlap-assembly error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum LapwError {
    #[error("muffin-tin radius must be finite and positive, got {0}")]
    InvalidRadius(f64),
    #[error("APW boundary matrix for l={l} is singular (determinant {determinant})")]
    SingularBoundaryMatrix { l: u32, determinant: f64 },
    #[error("cell volume must be finite and positive, got {0}")]
    InvalidCellVolume(f64),
    #[error("site has {actual} plane-wave coefficient sets, expected {expected}")]
    PlaneWaveCount { expected: usize, actual: usize },
    #[error("site plane wave {plane_wave} has {actual} lm channels, expected {expected}")]
    ChannelCount {
        plane_wave: usize,
        expected: usize,
        actual: usize,
    },
    #[error("APW matches must be ordered by l: expected {expected}, found {actual}")]
    MatchAngularMomentum { expected: u32, actual: u32 },
    #[error("all plane waves in an overlap matrix must share one k point")]
    MixedKPoints,
    #[error("interstitial coefficient failed: {0}")]
    StepFunction(String),
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
) -> Result<ApwMatch, LapwError> {
    if !radius.get().is_finite() || radius.get() <= 0.0 {
        return Err(LapwError::InvalidRadius(radius.get()));
    }
    let determinant = basis.u.value * basis.udot.derivative - basis.udot.value * basis.u.derivative;
    let matrix_scale = (basis.u.value.abs() + basis.udot.value.abs())
        * (basis.u.derivative.abs() + basis.udot.derivative.abs());
    if !determinant.is_finite()
        || determinant.abs() <= 64.0 * f64::EPSILON * matrix_scale.max(f64::MIN_POSITIVE)
    {
        return Err(LapwError::SingularBoundaryMatrix { l, determinant });
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

/// `4 pi i^l conj(Y_lm(qhat)) / sqrt(Omega)` for the `exp(+i q dot r)`
/// Rayleigh expansion.
///
/// This coefficient contains no site phase and no radial matching
/// coefficient.  At `q=0`, `mt-core`'s deterministic direction convention
/// leaves only the `l=m=0` channel nonzero.
pub fn rayleigh_coefficient(
    lm: Lm,
    q: [InverseBohr; 3],
    cell_volume: VolumeBohr3,
) -> Result<Complex64, LapwError> {
    if !cell_volume.get().is_finite() || cell_volume.get() <= 0.0 {
        return Err(LapwError::InvalidCellVolume(cell_volume.get()));
    }
    let direction = q.map(InverseBohr::get);
    let harmonic = complex_spherical_harmonics(lm.l, direction)[lm.index()].conj();
    Ok(i_pow(lm.l) * harmonic * (4.0 * PI / cell_volume.get().sqrt()))
}

/// Translation phase `exp(+i q dot R_a)` for expansion about site `R_a`.
pub fn site_translation_phase(q: [InverseBohr; 3], site: [Bohr; 3]) -> Complex64 {
    let phase = q
        .iter()
        .zip(site)
        .map(|(component, coordinate)| component.get() * coordinate.get())
        .sum();
    Complex64::from_polar(1.0, phase)
}

/// APW augmentation coefficients indexed by the public contiguous `lm` index.
/// Each channel stores coefficients multiplying `(u_l, du_l/dE)`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveAugmentation {
    pub coefficients: Vec<[Complex64; 2]>,
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
) -> Result<PlaneWaveAugmentation, LapwError> {
    let l_max = matches.len().saturating_sub(1) as u32;
    let phase = site_translation_phase(plane_wave.q, site);
    let mut coefficients = Vec::with_capacity(lm_count(l_max));
    for (l, matched) in matches.iter().enumerate() {
        let l = l as u32;
        if matched.l != l {
            return Err(LapwError::MatchAngularMomentum {
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

/// Per-site APW coefficients and real radial overlap blocks.
///
/// `radial_overlaps[l]` uses the ordered radial basis
/// `(u_l, du_l/dE)`.  Local orbitals are deliberately absent from this M-D
/// representation.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteAugmentation {
    pub plane_waves: Vec<PlaneWaveAugmentation>,
    pub radial_overlaps: Vec<RadialOverlapBlock>,
}

/// A real symmetric `2 x 2` radial overlap block in the `(u, du/dE)` basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialOverlapBlock {
    pub uu: f64,
    pub u_udot: f64,
    pub udot_udot: f64,
}

impl RadialOverlapBlock {
    fn element(self, left: usize, right: usize) -> f64 {
        match (left, right) {
            (0, 0) => self.uu,
            (0, 1) | (1, 0) => self.u_udot,
            (1, 1) => self.udot_udot,
            _ => unreachable!("radial indices are fixed to 0 and 1"),
        }
    }
}

/// Dense row-major complex matrix produced by overlap assembly.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseOverlapMatrix {
    dimension: usize,
    data: Vec<Complex64>,
}

impl DenseOverlapMatrix {
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn as_slice(&self) -> &[Complex64] {
        &self.data
    }
}

impl Index<(usize, usize)> for DenseOverlapMatrix {
    type Output = Complex64;

    fn index(&self, (row, column): (usize, usize)) -> &Self::Output {
        &self.data[row * self.dimension + column]
    }
}

/// Assemble the LAPW overlap
/// `Theta(G_i-G_j) + sum_a,lm c_i^dagger O^a_l c_j`.
///
/// Only the upper triangle is evaluated; the lower triangle is filled by
/// conjugation, making the returned dense matrix Hermitian by construction.
pub fn assemble_overlap(
    plane_waves: &[PlaneWave],
    geometry: &InterstitialGeometry,
    sites: &[SiteAugmentation],
) -> Result<DenseOverlapMatrix, LapwError> {
    if let Some(first) = plane_waves.first() {
        if plane_waves.iter().any(|wave| wave.k != first.k) {
            return Err(LapwError::MixedKPoints);
        }
    }
    let dimension = plane_waves.len();
    for site in sites {
        validate_site(site, dimension)?;
    }

    let mut data = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for i in 0..dimension {
        for j in i..dimension {
            let difference = std::array::from_fn(|axis| {
                InverseBohr(
                    plane_waves[i].g.cartesian[axis].get() - plane_waves[j].g.cartesian[axis].get(),
                )
            });
            let mut value = geometry
                .coefficient(difference)
                .map_err(|error| LapwError::StepFunction(error.to_string()))?;
            for site in sites {
                let left = &site.plane_waves[i].coefficients;
                let right = &site.plane_waves[j].coefficients;
                for channel in 0..left.len() {
                    let l = lm_from_index(channel).l as usize;
                    let overlap = site.radial_overlaps[l];
                    for (alpha, left_coefficient) in left[channel].iter().enumerate() {
                        for (beta, right_coefficient) in right[channel].iter().enumerate() {
                            value += left_coefficient.conj()
                                * overlap.element(alpha, beta)
                                * right_coefficient;
                        }
                    }
                }
            }
            data[i * dimension + j] = value;
            data[j * dimension + i] = value.conj();
        }
    }
    Ok(DenseOverlapMatrix { dimension, data })
}

fn validate_site(site: &SiteAugmentation, plane_wave_count: usize) -> Result<(), LapwError> {
    if site.plane_waves.len() != plane_wave_count {
        return Err(LapwError::PlaneWaveCount {
            expected: plane_wave_count,
            actual: site.plane_waves.len(),
        });
    }
    let expected_channels = site
        .radial_overlaps
        .len()
        .saturating_mul(site.radial_overlaps.len());
    for (plane_wave, augmentation) in site.plane_waves.iter().enumerate() {
        if augmentation.coefficients.len() != expected_channels {
            return Err(LapwError::ChannelCount {
                plane_wave,
                expected: expected_channels,
                actual: augmentation.coefficients.len(),
            });
        }
    }
    Ok(())
}

fn i_pow(l: u32) -> Complex64 {
    match l % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mt_core::{ReciprocalLattice, Sphere};

    fn boundary(value: f64, derivative: f64) -> BoundaryData {
        BoundaryData {
            value,
            derivative,
            log_derivative: (value != 0.0).then(|| InverseBohr(derivative / value)),
            scaled_log_derivative: None,
        }
    }

    fn waves() -> Vec<PlaneWave> {
        let lattice = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        lattice
            .enumerate(InverseBohr(1.0))
            .unwrap()
            .into_iter()
            .map(|g| PlaneWave::new([InverseBohr(0.1), InverseBohr(-0.2), InverseBohr(0.05)], g))
            .collect()
    }

    #[test]
    fn matching_residuals_are_small() {
        let basis = ApwBoundaryBasis {
            u: boundary(0.73, -0.21),
            udot: boundary(-0.18, 1.14),
        };
        for l in 0..=8 {
            let matched = match_apw_boundary(l, InverseBohr(2.3), Bohr(1.7), basis).unwrap();
            assert!(matched.value_residual.abs() <= 1.0e-10);
            assert!(matched.slope_residual.abs() <= 1.0e-10);
        }
    }

    #[test]
    fn overlap_is_hermitian() {
        let waves = waves();
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: [Bohr(0.3), Bohr(-0.4), Bohr(0.2)],
                radius: Bohr(0.8),
            }],
        )
        .unwrap();
        let boundary = ApwBoundaryBasis {
            u: boundary(0.8, -0.1),
            udot: boundary(0.2, 1.1),
        };
        let plane_waves = waves
            .iter()
            .map(|wave| {
                let matches = (0..=2)
                    .map(|l| match_apw_boundary(l, wave.q_norm, Bohr(0.8), boundary).unwrap())
                    .collect::<Vec<_>>();
                augmentation_coefficients(
                    wave,
                    [Bohr(0.3), Bohr(-0.4), Bohr(0.2)],
                    VolumeBohr3(100.0),
                    &matches,
                )
                .unwrap()
            })
            .collect();
        let site = SiteAugmentation {
            plane_waves,
            radial_overlaps: vec![
                RadialOverlapBlock {
                    uu: 1.0,
                    u_udot: 0.04,
                    udot_udot: 0.7,
                };
                3
            ],
        };
        let overlap = assemble_overlap(&waves, &geometry, &[site]).unwrap();
        for i in 0..overlap.dimension() {
            for j in 0..overlap.dimension() {
                assert!((overlap[(i, j)] - overlap[(j, i)].conj()).norm() < 2.0e-14);
            }
        }
    }

    #[test]
    fn empty_sphere_geometry_is_identity() {
        let waves = waves();
        let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), vec![]).unwrap();
        let overlap = assemble_overlap(&waves, &geometry, &[]).unwrap();
        for i in 0..overlap.dimension() {
            for j in 0..overlap.dimension() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((overlap[(i, j)] - expected).norm() < 2.0e-14);
            }
        }
    }
}
