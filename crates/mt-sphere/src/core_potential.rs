//! Boundary-checked continuation of a spherical core potential.
//!
//! The muffin-tin samples, including the exact central `-Z/r` singularity,
//! remain unchanged. Beyond the sphere, the potential is the spherical
//! average of an explicitly supplied periodic Fourier continuation. No
//! atomic tail, constant, slope, or decay law is manufactured here.

use muffintin_core::{ExponentialMesh, spherical_bessel_j};
use num_complex::Complex64;
use thiserror::Error;

/// One already site-centered Fourier contribution `c_G exp(i G dot R_site)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CenteredSphericalFourierMode {
    /// `|G|` in inverse Bohr.
    pub wave_number: f64,
    /// Centered potential coefficient in Hartree.
    pub coefficient: Complex64,
}

/// Validation tolerances for joining MT and periodic representations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorePotentialContinuationSpec {
    /// Absolute-plus-relative tolerance for the physical value at the MT boundary.
    pub boundary_tolerance: f64,
    /// Tolerance for the first-four-point extrapolation of `r V(r) -> -Z`.
    pub coulomb_tolerance: f64,
}

impl Default for CorePotentialContinuationSpec {
    fn default() -> Self {
        Self {
            boundary_tolerance: 1.0e-7,
            coulomb_tolerance: 1.0e-5,
        }
    }
}

/// Complete physical spherical potential and representation-join evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtendedCorePotential {
    pub mesh: ExponentialMesh,
    pub values: Vec<f64>,
    pub muffin_tin_points: usize,
    pub muffin_tin_boundary: f64,
    pub periodic_boundary: f64,
    pub boundary_mismatch: f64,
    /// Residual of the extrapolated `lim_(r->0) r V(r) = -Z` coefficient.
    pub origin_coulomb_residual: f64,
}

/// Copy the physical MT potential and continue it with a periodic spherical average.
pub fn continue_core_spherical_potential(
    muffin_tin_mesh: &ExponentialMesh,
    muffin_tin_potential: &[f64],
    extended_mesh: &ExponentialMesh,
    nuclear_charge: f64,
    periodic_modes: &[CenteredSphericalFourierMode],
    spec: CorePotentialContinuationSpec,
) -> Result<ExtendedCorePotential, CorePotentialContinuationError> {
    validate_inputs(
        muffin_tin_mesh,
        muffin_tin_potential,
        extended_mesh,
        nuclear_charge,
        spec,
    )?;
    validate_modes(periodic_modes)?;
    let outer_potential = extended_mesh.radii()[muffin_tin_mesh.len() - 1..]
        .iter()
        .map(|radius| {
            periodic_spherical_average(radius.get(), periodic_modes, spec.boundary_tolerance)
        })
        .collect::<Result<Vec<_>, _>>()?;
    join_core_spherical_potential(
        muffin_tin_mesh,
        muffin_tin_potential,
        extended_mesh,
        nuclear_charge,
        &outer_potential,
        spec,
    )
}

/// Join MT samples to explicitly evaluated outer spherical-potential samples.
///
/// `outer_potential[0]` is the independently evaluated value at the MT
/// boundary; the remaining entries correspond in order to the extended-mesh
/// samples outside the sphere.  Keeping the duplicate boundary sample is
/// intentional: it makes representation continuity a checked physical
/// contract rather than an assumed or fitted condition.
pub fn join_core_spherical_potential(
    muffin_tin_mesh: &ExponentialMesh,
    muffin_tin_potential: &[f64],
    extended_mesh: &ExponentialMesh,
    nuclear_charge: f64,
    outer_potential: &[f64],
    spec: CorePotentialContinuationSpec,
) -> Result<ExtendedCorePotential, CorePotentialContinuationError> {
    validate_inputs(
        muffin_tin_mesh,
        muffin_tin_potential,
        extended_mesh,
        nuclear_charge,
        spec,
    )?;
    let expected_outer = extended_mesh.len() - muffin_tin_mesh.len() + 1;
    if outer_potential.len() != expected_outer {
        return Err(CorePotentialContinuationError::OuterPotentialLength {
            expected: expected_outer,
            actual: outer_potential.len(),
        });
    }
    for (index, &value) in outer_potential.iter().enumerate() {
        if !value.is_finite() {
            return Err(CorePotentialContinuationError::NonFiniteOuterPotential { index, value });
        }
    }
    let origin_coulomb_residual =
        (extrapolated_coulomb_coefficient(muffin_tin_mesh, muffin_tin_potential) + nuclear_charge)
            .abs();
    let coulomb_limit = spec.coulomb_tolerance * nuclear_charge.abs().max(1.0);
    if origin_coulomb_residual > coulomb_limit {
        return Err(CorePotentialContinuationError::CoulombOrigin {
            charge: nuclear_charge,
            residual: origin_coulomb_residual,
            tolerance: coulomb_limit,
        });
    }

    let periodic_boundary = outer_potential[0];
    let muffin_tin_boundary = *muffin_tin_potential
        .last()
        .expect("validated MT potential is nonempty");
    let boundary_mismatch = periodic_boundary - muffin_tin_boundary;
    let boundary_limit = spec.boundary_tolerance
        * periodic_boundary
            .abs()
            .max(muffin_tin_boundary.abs())
            .max(1.0);
    if boundary_mismatch.abs() > boundary_limit {
        return Err(CorePotentialContinuationError::BoundaryMismatch {
            muffin_tin: muffin_tin_boundary,
            periodic: periodic_boundary,
            mismatch: boundary_mismatch,
            tolerance: boundary_limit,
        });
    }

    let mut values = muffin_tin_potential.to_vec();
    values.extend_from_slice(&outer_potential[1..]);
    Ok(ExtendedCorePotential {
        mesh: extended_mesh.clone(),
        values,
        muffin_tin_points: muffin_tin_mesh.len(),
        muffin_tin_boundary,
        periodic_boundary,
        boundary_mismatch,
        origin_coulomb_residual,
    })
}

fn extrapolated_coulomb_coefficient(mesh: &ExponentialMesh, potential: &[f64]) -> f64 {
    // For a complete effective potential V=-Z/r+V_regular, rV has a finite
    // intercept -Z and regular O(r) terms.  Interpolating the first up-to-four
    // finite samples to r=0 tests the singular coefficient without mistaking
    // a large but finite electrostatic or XC offset for an incorrect nucleus.
    let count = mesh.len().min(4);
    (0..count)
        .map(|point| {
            let radius = mesh.radii()[point].get();
            let basis_at_zero = (0..count)
                .filter(|&other| other != point)
                .map(|other| {
                    let other_radius = mesh.radii()[other].get();
                    -other_radius / (radius - other_radius)
                })
                .product::<f64>();
            radius * potential[point] * basis_at_zero
        })
        .sum()
}

fn periodic_spherical_average(
    radius: f64,
    modes: &[CenteredSphericalFourierMode],
    reality_tolerance: f64,
) -> Result<f64, CorePotentialContinuationError> {
    let value = modes.iter().fold(Complex64::new(0.0, 0.0), |sum, mode| {
        sum + mode.coefficient * spherical_bessel_j(0, mode.wave_number * radius)
    });
    let tolerance = reality_tolerance * value.re.abs().max(1.0);
    if value.im.abs() > tolerance {
        Err(CorePotentialContinuationError::ComplexPeriodicAverage {
            radius,
            imaginary: value.im,
            tolerance,
        })
    } else {
        Ok(value.re)
    }
}

fn validate_inputs(
    muffin_tin_mesh: &ExponentialMesh,
    muffin_tin_potential: &[f64],
    extended_mesh: &ExponentialMesh,
    nuclear_charge: f64,
    spec: CorePotentialContinuationSpec,
) -> Result<(), CorePotentialContinuationError> {
    if muffin_tin_potential.len() != muffin_tin_mesh.len() {
        return Err(CorePotentialContinuationError::PotentialLength {
            expected: muffin_tin_mesh.len(),
            actual: muffin_tin_potential.len(),
        });
    }
    if extended_mesh.len() <= muffin_tin_mesh.len()
        || extended_mesh.first() != muffin_tin_mesh.first()
        || extended_mesh.increment() != muffin_tin_mesh.increment()
        || extended_mesh.radii()[..muffin_tin_mesh.len()] != *muffin_tin_mesh.radii()
    {
        return Err(CorePotentialContinuationError::MeshPrefix);
    }
    if !nuclear_charge.is_finite() || nuclear_charge < 0.0 {
        return Err(CorePotentialContinuationError::InvalidNuclearCharge(
            nuclear_charge,
        ));
    }
    if !spec.boundary_tolerance.is_finite()
        || spec.boundary_tolerance < 0.0
        || !spec.coulomb_tolerance.is_finite()
        || spec.coulomb_tolerance < 0.0
    {
        return Err(CorePotentialContinuationError::InvalidTolerance {
            boundary: spec.boundary_tolerance,
            coulomb: spec.coulomb_tolerance,
        });
    }
    for (index, &value) in muffin_tin_potential.iter().enumerate() {
        if !value.is_finite() {
            return Err(CorePotentialContinuationError::NonFinitePotential { index, value });
        }
    }
    Ok(())
}

fn validate_modes(
    periodic_modes: &[CenteredSphericalFourierMode],
) -> Result<(), CorePotentialContinuationError> {
    for (index, mode) in periodic_modes.iter().enumerate() {
        if !mode.wave_number.is_finite()
            || mode.wave_number < 0.0
            || !mode.coefficient.re.is_finite()
            || !mode.coefficient.im.is_finite()
        {
            return Err(CorePotentialContinuationError::InvalidFourierMode { index });
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum CorePotentialContinuationError {
    #[error("muffin-tin potential has {actual} samples, expected {expected}")]
    PotentialLength { expected: usize, actual: usize },
    #[error("outer potential has {actual} samples, expected {expected} including the MT boundary")]
    OuterPotentialLength { expected: usize, actual: usize },
    #[error("extended core mesh must strictly extend the exact muffin-tin mesh prefix")]
    MeshPrefix,
    #[error("nuclear charge must be finite and nonnegative, got {0}")]
    InvalidNuclearCharge(f64),
    #[error("continuation tolerances are invalid: boundary={boundary}, coulomb={coulomb}")]
    InvalidTolerance { boundary: f64, coulomb: f64 },
    #[error("muffin-tin potential sample {index} is non-finite: {value}")]
    NonFinitePotential { index: usize, value: f64 },
    #[error("outer potential sample {index} is non-finite: {value}")]
    NonFiniteOuterPotential { index: usize, value: f64 },
    #[error("periodic Fourier mode {index} is invalid")]
    InvalidFourierMode { index: usize },
    #[error(
        "core potential does not retain -Z/r at the origin for Z={charge}: residual {residual}, tolerance {tolerance}"
    )]
    CoulombOrigin {
        charge: f64,
        residual: f64,
        tolerance: f64,
    },
    #[error(
        "MT and periodic spherical potentials disagree at the boundary: MT={muffin_tin}, periodic={periodic}, mismatch={mismatch}, tolerance={tolerance}"
    )]
    BoundaryMismatch {
        muffin_tin: f64,
        periodic: f64,
        mismatch: f64,
        tolerance: f64,
    },
    #[error(
        "periodic spherical average at r={radius} has imaginary part {imaginary}, tolerance {tolerance}"
    )]
    ComplexPeriodicAverage {
        radius: f64,
        imaginary: f64,
        tolerance: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::Bohr;

    #[test]
    fn exact_mt_prefix_and_explicit_periodic_tail_are_preserved() {
        let mt = ExponentialMesh::new(Bohr(0.01), 0.1, 7).unwrap();
        let extended = ExponentialMesh::new(Bohr(0.01), 0.1, 13).unwrap();
        let charge = 0.4;
        let boundary_value = -charge / mt.last().get() + 0.2;
        let inner = mt
            .radii()
            .iter()
            .map(|radius| -charge / radius.get() + 0.2)
            .collect::<Vec<_>>();
        let modes = [CenteredSphericalFourierMode {
            wave_number: 0.0,
            coefficient: Complex64::new(boundary_value, 0.0),
        }];
        let continued = continue_core_spherical_potential(
            &mt,
            &inner,
            &extended,
            charge,
            &modes,
            CorePotentialContinuationSpec {
                boundary_tolerance: 1.0e-13,
                coulomb_tolerance: 0.01,
            },
        )
        .unwrap();
        assert_eq!(&continued.values[..mt.len()], inner);
        assert!(
            continued.values[mt.len()..]
                .iter()
                .all(|value| *value == boundary_value)
        );
        assert!(continued.origin_coulomb_residual < 1.0e-13);
    }

    #[test]
    fn boundary_mismatch_is_not_hidden_by_an_invented_tail() {
        let mt = ExponentialMesh::new(Bohr(1.0e-5), 0.1, 7).unwrap();
        let extended = ExponentialMesh::new(Bohr(1.0e-5), 0.1, 13).unwrap();
        let inner = vec![0.0; mt.len()];
        let modes = [CenteredSphericalFourierMode {
            wave_number: 0.0,
            coefficient: Complex64::new(1.0, 0.0),
        }];
        assert!(matches!(
            continue_core_spherical_potential(
                &mt,
                &inner,
                &extended,
                0.0,
                &modes,
                CorePotentialContinuationSpec::default(),
            ),
            Err(CorePotentialContinuationError::BoundaryMismatch { .. })
        ));
    }

    #[test]
    fn coulomb_intercept_accepts_large_regular_terms_but_rejects_wrong_charge() {
        let mt = ExponentialMesh::new(Bohr(1.0e-4), 0.3, 7).unwrap();
        let extended = ExponentialMesh::new(Bohr(1.0e-4), 0.3, 11).unwrap();
        let charge = 2.0;
        let regular = |radius: f64| 700.0 - 31.0 * radius + 4.0 * radius * radius;
        let inner = mt
            .radii()
            .iter()
            .map(|radius| -charge / radius.get() + regular(radius.get()))
            .collect::<Vec<_>>();
        let boundary = *inner.last().unwrap();
        let outer = vec![boundary; extended.len() - mt.len() + 1];
        let result = join_core_spherical_potential(
            &mt,
            &inner,
            &extended,
            charge,
            &outer,
            CorePotentialContinuationSpec {
                boundary_tolerance: 1.0e-13,
                coulomb_tolerance: 1.0e-10,
            },
        )
        .unwrap();
        assert!(result.origin_coulomb_residual < 1.0e-12);

        let wrong = mt
            .radii()
            .iter()
            .map(|radius| -1.9 / radius.get() + regular(radius.get()))
            .collect::<Vec<_>>();
        assert!(matches!(
            join_core_spherical_potential(
                &mt,
                &wrong,
                &extended,
                charge,
                &outer,
                CorePotentialContinuationSpec {
                    boundary_tolerance: 1.0,
                    coulomb_tolerance: 1.0e-6,
                },
            ),
            Err(CorePotentialContinuationError::CoulombOrigin { .. })
        ));
    }
}
