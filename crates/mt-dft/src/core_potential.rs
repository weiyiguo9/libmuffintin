//! Physical effective-potential continuation for bound-core radial solves.
//!
//! The electrostatic input is the raw, unmasked periodic field. Inside a
//! muffin tin its monopole contains the restored central `-Z/r`; outside its
//! Fourier coefficients represent that same periodic Coulomb potential. XC is
//! not taken from `RegionalPotential::interstitial`, whose coefficients are
//! step-function masked. Instead it is evaluated pointwise from the smooth
//! periodic interstitial-density representation and angularly averaged at
//! every outer radial sample. A compact C1 representation bridge reconciles
//! finite MT/outer value and slope mismatches, then vanishes with zero slope
//! at the extended endpoint so the raw physical outer field controls decay.

use crate::xc_field::{NoncollinearXcRoute, evaluate_interstitial_noncollinear_xc_potential};
use crate::{
    InterstitialField, MuffinTinField, RegionalDensity, RegionalElectrostaticResult,
    RegionalXcError, RegionalXcResult, XcFunctional,
};
use muffintin_core::{
    Bohr, ExponentialMesh, InterstitialGeometry, spherical_bessel_j, spherical_bessel_j_derivative,
};
use muffintin_coulomb::{InterstitialHartreePotential, MuffinTinHartreePotential};
use muffintin_core::{AngularGrid, GridError};
use muffintin_sphere::{
    CenteredSphericalFourierMode, CorePotentialContinuationError, CorePotentialContinuationSpec,
    ExtendedCorePotential, join_core_spherical_potential,
};
use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

const REALITY_TOLERANCE: f64 = 4096.0 * f64::EPSILON;
const GEOMETRY_TOLERANCE: f64 = 1.0e-10;

/// Numerical controls for a physical core-potential continuation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorePotentialBuildSpec {
    pub continuation: CorePotentialContinuationSpec,
    /// The same functional used to produce the supplied `RegionalXcResult`.
    pub xc_functional: XcFunctional,
    /// The same noncollinear reduction used to produce the supplied XC result.
    pub xc_noncollinear_route: NoncollinearXcRoute,
    /// Fibonacci angular points used for each outer XC spherical average.
    pub xc_angular_point_count: usize,
}

/// Evidence for aligning a frozen checkpoint's two potential representations.
///
/// The correction is the unique cubic Hermite bridge that matches the MT
/// value and slope at the inner endpoint and vanishes with zero slope at the
/// extended-mesh endpoint. It restores the supplied periodic field exactly in
/// the asymptotic region and is not an atomic decay or fitted radial tail.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorePotentialJoin {
    pub muffin_tin_boundary: f64,
    pub uncorrected_outer_boundary: f64,
    pub boundary_value_correction: f64,
    pub muffin_tin_boundary_derivative: f64,
    pub uncorrected_outer_boundary_derivative: f64,
    pub boundary_derivative_correction: f64,
    pub periodic_recovery_radius: f64,
    pub corrected_boundary_residual: f64,
    pub outer_correction_residual: f64,
}

/// One extended core potential plus explicit representation-join evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltExtendedCorePotential {
    pub potential: ExtendedCorePotential,
    pub join: CorePotentialJoin,
}

/// Build spherical scalar effective potentials on extended core meshes.
///
/// The `RegionalXcResult` supplies physical inner MT monopoles. Its masked
/// interstitial coefficients are used only for layout identity, never as a raw
/// periodic XC potential. Outer XC values are recomputed from `density` and
/// `spec.xc_functional`. Site order is the common order of the density
/// geometry, raw MT arrays, and `extended_meshes`.
pub fn build_extended_core_potentials(
    electrostatics: &RegionalElectrostaticResult,
    exchange_correlation: &RegionalXcResult,
    density: &RegionalDensity,
    extended_meshes: &[ExponentialMesh],
    spec: CorePotentialBuildSpec,
) -> Result<Vec<BuiltExtendedCorePotential>, CorePotentialBuildError> {
    let raw = &electrostatics.raw_electrostatic;
    let nuclear_charges = electrostatics.raw_nuclear.nuclear_charges();
    let geometry = density.geometry();
    let xc_potential = &exchange_correlation.potential;
    let angular = AngularGrid::fibonacci(spec.xc_angular_point_count)?;
    let site_count = geometry.spheres().len();
    if raw.muffin_tins().len() != site_count {
        return Err(CorePotentialBuildError::RawSiteCount {
            expected: site_count,
            actual: raw.muffin_tins().len(),
        });
    }
    if nuclear_charges.len() != site_count {
        return Err(CorePotentialBuildError::NuclearSiteCount {
            expected: site_count,
            actual: nuclear_charges.len(),
        });
    }
    if extended_meshes.len() != site_count {
        return Err(CorePotentialBuildError::ExtendedMeshCount {
            expected: site_count,
            actual: extended_meshes.len(),
        });
    }
    require_xc_site_count(xc_potential.scalar().muffin_tins().len(), site_count)?;

    build_scalar_channel(
        raw.muffin_tins(),
        raw.interstitial(),
        xc_potential.scalar().muffin_tins(),
        xc_potential.scalar().interstitial(),
        density,
        geometry,
        nuclear_charges,
        extended_meshes,
        spec,
        &angular,
    )
}

/// Bootstrap extended core potentials directly from a frozen checkpoint total potential.
///
/// This path does not require a neutral valence-plus-core density or a Poisson
/// solve. The physical MT monopole, including `-Z/r`, is copied unchanged. The
/// checkpoint interstitial Fourier field supplies the outer radial shape. Since
/// independently stored checkpoint representations need not join pointwise, a
/// compact cubic Hermite correction aligns value and slope at the MT boundary
/// and vanishes with zero slope at the extended-mesh endpoint. Thus the raw
/// periodic field, rather than a guessed atomic tail, controls the asymptote.
/// Every join parameter and endpoint residual is returned as
/// [`CorePotentialJoin`].
pub fn build_extended_checkpoint_core_potentials(
    checkpoint_total: &crate::RegionalPotential,
    geometry: &InterstitialGeometry,
    nuclear_charges: &[f64],
    extended_meshes: &[ExponentialMesh],
    continuation: CorePotentialContinuationSpec,
) -> Result<Vec<BuiltExtendedCorePotential>, CorePotentialBuildError> {
    let site_count = geometry.spheres().len();
    if nuclear_charges.len() != site_count {
        return Err(CorePotentialBuildError::NuclearSiteCount {
            expected: site_count,
            actual: nuclear_charges.len(),
        });
    }
    if extended_meshes.len() != site_count {
        return Err(CorePotentialBuildError::ExtendedMeshCount {
            expected: site_count,
            actual: extended_meshes.len(),
        });
    }
    require_checkpoint_site_count(checkpoint_total.scalar().muffin_tins().len(), site_count)?;
    build_checkpoint_scalar(
        checkpoint_total.scalar().muffin_tins(),
        checkpoint_total.scalar().interstitial(),
        geometry,
        nuclear_charges,
        extended_meshes,
        continuation,
    )
}

fn build_checkpoint_scalar(
    muffin_tins: &[MuffinTinField],
    interstitial: &InterstitialField,
    geometry: &InterstitialGeometry,
    nuclear_charges: &[f64],
    extended_meshes: &[ExponentialMesh],
    continuation: CorePotentialContinuationSpec,
) -> Result<Vec<BuiltExtendedCorePotential>, CorePotentialBuildError> {
    muffin_tins
        .iter()
        .zip(geometry.spheres())
        .zip(nuclear_charges)
        .zip(extended_meshes)
        .enumerate()
        .map(|(site, (((muffin_tin, sphere), &charge), extended_mesh))| {
            let mesh_radius = muffin_tin.mesh().last().get();
            let geometry_radius = sphere.radius.get();
            let radius_tolerance = GEOMETRY_TOLERANCE * geometry_radius.max(1.0);
            if (mesh_radius - geometry_radius).abs() > radius_tolerance {
                return Err(CorePotentialBuildError::MuffinTinRadius {
                    site,
                    mesh: mesh_radius,
                    geometry: geometry_radius,
                    tolerance: radius_tolerance,
                });
            }
            let monopole = physical_checkpoint_monopole(site, "scalar", muffin_tin)?;
            let modes = centered_checkpoint_modes(interstitial, sphere.center.map(Bohr::get));
            let mut outer = extended_mesh.radii()[muffin_tin.mesh().len() - 1..]
                .iter()
                .enumerate()
                .map(|(radial, radius)| {
                    periodic_spherical_average(site, "scalar", radial, radius.get(), &modes)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let uncorrected_outer_boundary_derivative =
                periodic_spherical_derivative(muffin_tin.mesh().last().get(), &modes);
            bridge_and_join_core_potential(
                site,
                "scalar",
                muffin_tin.mesh(),
                &monopole,
                extended_mesh,
                charge,
                &mut outer,
                uncorrected_outer_boundary_derivative,
                continuation,
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_scalar_channel(
    electrostatic_muffin_tins: &[MuffinTinHartreePotential],
    electrostatic_interstitial: &InterstitialHartreePotential,
    xc_muffin_tins: &[MuffinTinField],
    xc_operator_interstitial: &InterstitialField,
    density: &RegionalDensity,
    geometry: &InterstitialGeometry,
    nuclear_charges: &[f64],
    extended_meshes: &[ExponentialMesh],
    spec: CorePotentialBuildSpec,
    angular: &AngularGrid,
) -> Result<Vec<BuiltExtendedCorePotential>, CorePotentialBuildError> {
    let density_layout = density.charge().interstitial().layout();
    if electrostatic_interstitial.layout() != xc_operator_interstitial.layout()
        || electrostatic_interstitial.layout() != density_layout
    {
        return Err(CorePotentialBuildError::InterstitialLayout);
    }
    let modes_by_site = geometry
        .spheres()
        .iter()
        .map(|sphere| {
            centered_electrostatic_modes(electrostatic_interstitial, sphere.center.map(Bohr::get))
        })
        .collect::<Vec<_>>();

    electrostatic_muffin_tins
        .iter()
        .zip(xc_muffin_tins)
        .zip(geometry.spheres())
        .zip(nuclear_charges)
        .zip(extended_meshes)
        .zip(modes_by_site)
        .enumerate()
        .map(
            |(site, (((((electrostatic_mt, xc_mt), sphere), &charge), extended_mesh), modes))| {
                if electrostatic_mt.mesh() != xc_mt.mesh() {
                    return Err(CorePotentialBuildError::MuffinTinMesh { site });
                }
                let mesh_radius = electrostatic_mt.mesh().last().get();
                let geometry_radius = sphere.radius.get();
                let radius_tolerance = GEOMETRY_TOLERANCE * geometry_radius.max(1.0);
                if (mesh_radius - geometry_radius).abs() > radius_tolerance {
                    return Err(CorePotentialBuildError::MuffinTinRadius {
                        site,
                        mesh: mesh_radius,
                        geometry: geometry_radius,
                        tolerance: radius_tolerance,
                    });
                }
                let effective_monopole =
                    physical_effective_monopole(site, "scalar", electrostatic_mt, xc_mt)?;
                let mut outer_potential = extended_mesh.radii()
                    [electrostatic_mt.mesh().len() - 1..]
                    .iter()
                    .enumerate()
                    .map(|(outer_index, radius)| {
                        let electrostatic = periodic_spherical_average(
                            site,
                            "scalar",
                            outer_index,
                            radius.get(),
                            &modes,
                        )?;
                        let xc = xc_spherical_average(
                            site,
                            sphere.center,
                            radius.get(),
                            density,
                            spec.xc_functional,
                            spec.xc_noncollinear_route,
                            angular,
                        )?;
                        Ok(electrostatic + xc)
                    })
                    .collect::<Result<Vec<_>, CorePotentialBuildError>>()?;
                let outer_radii = &extended_mesh.radii()[electrostatic_mt.mesh().len() - 1..];
                let uncorrected_outer_boundary_derivative =
                    endpoint_derivative(outer_radii, &outer_potential, 0);
                bridge_and_join_core_potential(
                    site,
                    "scalar",
                    electrostatic_mt.mesh(),
                    &effective_monopole,
                    extended_mesh,
                    charge,
                    &mut outer_potential,
                    uncorrected_outer_boundary_derivative,
                    spec.continuation,
                )
            },
        )
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn bridge_and_join_core_potential(
    site: usize,
    spin: &'static str,
    muffin_tin_mesh: &ExponentialMesh,
    muffin_tin_potential: &[f64],
    extended_mesh: &ExponentialMesh,
    charge: f64,
    outer_potential: &mut [f64],
    uncorrected_outer_boundary_derivative: f64,
    continuation: CorePotentialContinuationSpec,
) -> Result<BuiltExtendedCorePotential, CorePotentialBuildError> {
    let muffin_tin_boundary = *muffin_tin_potential
        .last()
        .expect("MT potential follows a nonempty mesh");
    let uncorrected_outer_boundary = outer_potential[0];
    let uncorrected_outer_endpoint = *outer_potential
        .last()
        .expect("strictly extended mesh has outer samples");
    let boundary_value_correction = muffin_tin_boundary - uncorrected_outer_boundary;
    let muffin_tin_boundary_derivative = endpoint_derivative(
        muffin_tin_mesh.radii(),
        muffin_tin_potential,
        muffin_tin_mesh.len() - 1,
    );
    let boundary_derivative_correction =
        muffin_tin_boundary_derivative - uncorrected_outer_boundary_derivative;
    let periodic_recovery_radius = extended_mesh.last().get();
    let bridge_length = periodic_recovery_radius - muffin_tin_mesh.last().get();
    for (&radius, value) in extended_mesh.radii()[muffin_tin_mesh.len() - 1..]
        .iter()
        .zip(outer_potential.iter_mut())
    {
        let fraction = (radius.get() - muffin_tin_mesh.last().get()) / bridge_length;
        let value_basis = 2.0 * fraction.powi(3) - 3.0 * fraction.powi(2) + 1.0;
        let slope_basis = fraction.powi(3) - 2.0 * fraction.powi(2) + fraction;
        *value += value_basis * boundary_value_correction
            + slope_basis * bridge_length * boundary_derivative_correction;
    }
    let corrected_boundary_residual = outer_potential[0] - muffin_tin_boundary;
    let outer_correction_residual =
        *outer_potential.last().expect("outer samples are nonempty") - uncorrected_outer_endpoint;
    let potential = join_core_spherical_potential(
        muffin_tin_mesh,
        muffin_tin_potential,
        extended_mesh,
        charge,
        outer_potential,
        continuation,
    )
    .map_err(|source| CorePotentialBuildError::Continuation { site, spin, source })?;
    Ok(BuiltExtendedCorePotential {
        potential,
        join: CorePotentialJoin {
            muffin_tin_boundary,
            uncorrected_outer_boundary,
            boundary_value_correction,
            muffin_tin_boundary_derivative,
            uncorrected_outer_boundary_derivative,
            boundary_derivative_correction,
            periodic_recovery_radius,
            corrected_boundary_residual,
            outer_correction_residual,
        },
    })
}

fn physical_effective_monopole(
    site: usize,
    spin: &'static str,
    electrostatic: &MuffinTinHartreePotential,
    exchange_correlation: &MuffinTinField,
) -> Result<Vec<f64>, CorePotentialBuildError> {
    let electrostatic =
        electrostatic
            .channel(0, 0)
            .ok_or(CorePotentialBuildError::MissingMonopole {
                site,
                spin,
                component: "electrostatic",
            })?;
    let exchange_correlation = exchange_correlation.field().channel(0, 0).ok_or(
        CorePotentialBuildError::MissingMonopole {
            site,
            spin,
            component: "exchange-correlation",
        },
    )?;
    let normalization = (4.0 * PI).sqrt();
    electrostatic
        .iter()
        .zip(exchange_correlation)
        .enumerate()
        .map(|(radial, (&electrostatic, &exchange_correlation))| {
            let value = (electrostatic.as_complex() + exchange_correlation) / normalization;
            let tolerance = REALITY_TOLERANCE * value.norm().max(1.0);
            if value.im.abs() > tolerance {
                Err(CorePotentialBuildError::ComplexMonopole {
                    site,
                    spin,
                    radial,
                    imaginary: value.im,
                    tolerance,
                })
            } else {
                Ok(value.re)
            }
        })
        .collect()
}

fn physical_checkpoint_monopole(
    site: usize,
    spin: &'static str,
    muffin_tin: &MuffinTinField,
) -> Result<Vec<f64>, CorePotentialBuildError> {
    let monopole =
        muffin_tin
            .field()
            .channel(0, 0)
            .ok_or(CorePotentialBuildError::MissingMonopole {
                site,
                spin,
                component: "frozen checkpoint",
            })?;
    let normalization = (4.0 * PI).sqrt();
    monopole
        .iter()
        .enumerate()
        .map(|(radial, &value)| {
            let value = value / normalization;
            let tolerance = REALITY_TOLERANCE * value.norm().max(1.0);
            if value.im.abs() > tolerance {
                Err(CorePotentialBuildError::ComplexMonopole {
                    site,
                    spin,
                    radial,
                    imaginary: value.im,
                    tolerance,
                })
            } else {
                Ok(value.re)
            }
        })
        .collect()
}

fn centered_electrostatic_modes(
    electrostatic: &InterstitialHartreePotential,
    center: [f64; 3],
) -> Vec<CenteredSphericalFourierMode> {
    electrostatic
        .layout()
        .vectors()
        .iter()
        .zip(electrostatic.coefficients())
        .map(|(vector, electrostatic)| {
            let angle = vector
                .cartesian
                .iter()
                .zip(center)
                .map(|(wave_vector, position)| wave_vector.get() * position)
                .sum::<f64>();
            let phase = Complex64::new(angle.cos(), angle.sin());
            CenteredSphericalFourierMode {
                wave_number: vector.norm.get(),
                coefficient: electrostatic.as_complex() * phase,
            }
        })
        .collect()
}

fn centered_checkpoint_modes(
    interstitial: &InterstitialField,
    center: [f64; 3],
) -> Vec<CenteredSphericalFourierMode> {
    interstitial
        .field()
        .iter()
        .map(|(vector, coefficient)| {
            let angle = vector
                .cartesian
                .iter()
                .zip(center)
                .map(|(wave_vector, position)| wave_vector.get() * position)
                .sum::<f64>();
            let phase = Complex64::new(angle.cos(), angle.sin());
            CenteredSphericalFourierMode {
                wave_number: vector.norm.get(),
                coefficient: *coefficient * phase,
            }
        })
        .collect()
}

fn periodic_spherical_average(
    site: usize,
    spin: &'static str,
    radial: usize,
    radius: f64,
    modes: &[CenteredSphericalFourierMode],
) -> Result<f64, CorePotentialBuildError> {
    let value = modes.iter().fold(Complex64::new(0.0, 0.0), |sum, mode| {
        sum + mode.coefficient * spherical_bessel_j(0, mode.wave_number * radius)
    });
    let tolerance = REALITY_TOLERANCE * value.norm().max(1.0);
    if value.im.abs() > tolerance {
        Err(CorePotentialBuildError::ComplexOuterElectrostatic {
            site,
            spin,
            radial,
            imaginary: value.im,
            tolerance,
        })
    } else {
        Ok(value.re)
    }
}

fn periodic_spherical_derivative(radius: f64, modes: &[CenteredSphericalFourierMode]) -> f64 {
    modes
        .iter()
        .fold(Complex64::new(0.0, 0.0), |sum, mode| {
            sum + mode.coefficient
                * mode.wave_number
                * spherical_bessel_j_derivative(0, mode.wave_number * radius)
        })
        .re
}

fn endpoint_derivative(radii: &[Bohr], values: &[f64], point: usize) -> f64 {
    let count = radii.len().min(4);
    let start = (point + 1).saturating_sub(count);
    let end = (start + count).min(radii.len());
    let x = radii[point].get();
    (start..end)
        .map(|basis| {
            let x_basis = radii[basis].get();
            let derivative = (start..end)
                .filter(|&omitted| omitted != basis)
                .map(|omitted| {
                    (1.0 / (x_basis - radii[omitted].get()))
                        * (start..end)
                            .filter(|&other| other != basis && other != omitted)
                            .map(|other| (x - radii[other].get()) / (x_basis - radii[other].get()))
                            .product::<f64>()
                })
                .sum::<f64>();
            values[basis] * derivative
        })
        .sum()
}

#[allow(clippy::too_many_arguments)]
fn xc_spherical_average(
    site: usize,
    center: [Bohr; 3],
    radius: f64,
    density: &RegionalDensity,
    functional: XcFunctional,
    route: NoncollinearXcRoute,
    angular: &AngularGrid,
) -> Result<f64, CorePotentialBuildError> {
    let mut average = 0.0;
    for point in angular.points() {
        let position =
            std::array::from_fn(|axis| Bohr(center[axis].get() + radius * point.direction[axis]));
        let xc =
            evaluate_interstitial_noncollinear_xc_potential(functional, route, density, position)
                .map_err(|source| CorePotentialBuildError::InterstitialDensity {
                site,
                radius,
                source,
            })?;
        average += point.weight * xc.0.get();
    }
    Ok(average / (4.0 * PI))
}

fn require_xc_site_count(actual: usize, expected: usize) -> Result<(), CorePotentialBuildError> {
    if actual == expected {
        Ok(())
    } else {
        Err(CorePotentialBuildError::XcSiteCount { expected, actual })
    }
}

fn require_checkpoint_site_count(
    actual: usize,
    expected: usize,
) -> Result<(), CorePotentialBuildError> {
    if actual == expected {
        Ok(())
    } else {
        Err(CorePotentialBuildError::CheckpointSiteCount { expected, actual })
    }
}

/// Invalid or physically discontinuous effective-potential continuation.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum CorePotentialBuildError {
    #[error(transparent)]
    AngularGrid(#[from] GridError),
    #[error("raw electrostatic potential has {actual} sites, expected {expected}")]
    RawSiteCount { expected: usize, actual: usize },
    #[error("raw nuclear potential has {actual} sites, expected {expected}")]
    NuclearSiteCount { expected: usize, actual: usize },
    #[error("extended core mesh list has {actual} sites, expected {expected}")]
    ExtendedMeshCount { expected: usize, actual: usize },
    #[error("scalar XC muffin-tin potential has {actual} sites, expected {expected}")]
    XcSiteCount { expected: usize, actual: usize },
    #[error("scalar frozen checkpoint muffin-tin potential has {actual} sites, expected {expected}")]
    CheckpointSiteCount { expected: usize, actual: usize },
    #[error("charge/scalar-XC/raw-electrostatic interstitial layouts differ")]
    InterstitialLayout,
    #[error("site {site} scalar XC and raw electrostatic muffin-tin meshes differ")]
    MuffinTinMesh { site: usize },
    #[error(
        "site {site} muffin-tin mesh radius {mesh} differs from geometry radius {geometry}, tolerance {tolerance}"
    )]
    MuffinTinRadius {
        site: usize,
        mesh: f64,
        geometry: f64,
        tolerance: f64,
    },
    #[error("site {site} {spin} {component} potential has no (l,m)=(0,0) channel")]
    MissingMonopole {
        site: usize,
        spin: &'static str,
        component: &'static str,
    },
    #[error(
        "site {site} {spin} effective monopole sample {radial} has imaginary part {imaginary}, tolerance {tolerance}"
    )]
    ComplexMonopole {
        site: usize,
        spin: &'static str,
        radial: usize,
        imaginary: f64,
        tolerance: f64,
    },
    #[error(
        "site {site} {spin} outer electrostatic sample {radial} has imaginary part {imaginary}, tolerance {tolerance}"
    )]
    ComplexOuterElectrostatic {
        site: usize,
        spin: &'static str,
        radial: usize,
        imaginary: f64,
        tolerance: f64,
    },
    #[error("site {site} interstitial density/XC evaluation failed at r={radius}: {source}")]
    InterstitialDensity {
        site: usize,
        radius: f64,
        source: RegionalXcError,
    },
    #[error("site {site} {spin} core-potential continuation failed: {source}")]
    Continuation {
        site: usize,
        spin: &'static str,
        source: CorePotentialContinuationError,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ElectrostaticSpec, RegionalDensity, XcFieldSpec, electron_count,
        evaluate_regional_electrostatics, evaluate_regional_xc,
    };
    use muffintin_core::{
        GVector, HermitianFourierField, InverseBohr, ReciprocalLattice, Sphere, VolumeBohr3,
    };
    use muffintin_coulomb::WeinertHartreeSpec;
    use muffintin_sphere::{HarmonicConvention, SphereField};

    const SIDE: f64 = 8.0;
    const RADIUS: f64 = 1.0;

    fn mesh(points: usize) -> ExponentialMesh {
        let first: f64 = 1.0e-5;
        ExponentialMesh::new(Bohr(first), (RADIUS / first).ln() / 100.0, points).unwrap()
    }

    fn layout() -> muffintin_core::FourierLayout {
        let reciprocal = ReciprocalLattice::from_direct([
            [Bohr(SIDE), Bohr(0.0), Bohr(0.0)],
            [Bohr(0.0), Bohr(SIDE), Bohr(0.0)],
            [Bohr(0.0), Bohr(0.0), Bohr(SIDE)],
        ])
        .unwrap();
        muffintin_core::FourierLayout::new(
            reciprocal,
            vec![GVector {
                index: [0; 3],
                cartesian: [InverseBohr(0.0); 3],
                norm: InverseBohr(0.0),
            }],
        )
        .unwrap()
    }

    fn geometry() -> InterstitialGeometry {
        InterstitialGeometry::new(
            VolumeBohr3(SIDE.powi(3)),
            vec![Sphere {
                center: [Bohr(SIDE / 2.0); 3],
                radius: Bohr(RADIUS),
            }],
        )
        .unwrap()
    }

    fn interstitial(layout: &muffintin_core::FourierLayout, value: f64) -> InterstitialField {
        InterstitialField::from_fourier_field(
            HermitianFourierField::new(layout.clone(), vec![Complex64::new(value, 0.0)]).unwrap(),
        )
    }

    fn muffin_tin(mesh: &ExponentialMesh, value: impl Fn(f64) -> f64) -> MuffinTinField {
        MuffinTinField::new(
            mesh.clone(),
            SphereField::new(
                HarmonicConvention::Real,
                [(
                    (0, 0),
                    mesh.radii()
                        .iter()
                        .map(|radius| Complex64::new((4.0 * PI).sqrt() * value(radius.get()), 0.0))
                        .collect(),
                )],
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn uniform_density(up: f64, down: f64) -> RegionalDensity {
        uniform_pauli_density(up + down, [0.0, 0.0, up - down])
    }

    fn uniform_pauli_density(charge_value: f64, magnetization: [f64; 3]) -> RegionalDensity {
        let radial = mesh(101);
        let reciprocal_layout = layout();
        let charge = crate::RegionalScalarField::new(
            geometry(),
            vec![muffin_tin(&radial, |_| charge_value)],
            interstitial(&reciprocal_layout, charge_value),
        )
        .unwrap();
        let components = magnetization.map(|value| {
            crate::RegionalScalarField::new(
                geometry(),
                vec![muffin_tin(&radial, |_| value)],
                interstitial(&reciprocal_layout, value),
            )
            .unwrap()
        });
        RegionalDensity::new(charge, components).unwrap()
    }

    #[test]
    fn nonzero_xc_uses_unmasked_pointwise_outer_field_and_joins_boundary() {
        let density = uniform_density(0.02, 0.01);
        let charge = electron_count(&density).unwrap();
        let electrostatics = evaluate_regional_electrostatics(
            density.charge(),
            &ElectrostaticSpec::new(WeinertHartreeSpec::electronic(4).unwrap(), vec![charge])
                .unwrap(),
        )
        .unwrap();
        let xc = evaluate_regional_xc(
            XcFunctional::LdaPw92,
            &density,
            XcFieldSpec {
                interstitial_divisions: [8; 3],
                angular_point_count: 14,
                output_l_max: 0,
                noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
            },
        )
        .unwrap();
        let extended = mesh(111);
        let built = build_extended_core_potentials(
            &electrostatics,
            &xc,
            &density,
            std::slice::from_ref(&extended),
            CorePotentialBuildSpec {
                continuation: CorePotentialContinuationSpec {
                    boundary_tolerance: 1.0e-10,
                    coulomb_tolerance: 1.0e-7,
                },
                xc_functional: XcFunctional::LdaPw92,
                xc_noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
                xc_angular_point_count: 26,
            },
        )
        .unwrap();
        assert_eq!(built[0].potential.mesh, extended);
        assert!(built[0].potential.boundary_mismatch.abs() < 1.0e-11);
        assert!(built[0].potential.origin_coulomb_residual < 1.0e-7 * charge);

        let transverse_density = uniform_pauli_density(0.03, [0.01, 0.0, 0.0]);
        let transverse_xc = evaluate_regional_xc(
            XcFunctional::LdaPw92,
            &transverse_density,
            XcFieldSpec {
                interstitial_divisions: [8; 3],
                angular_point_count: 14,
                output_l_max: 0,
                noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
            },
        )
        .unwrap();
        let transverse_built = build_extended_core_potentials(
            &electrostatics,
            &transverse_xc,
            &transverse_density,
            std::slice::from_ref(&extended),
            CorePotentialBuildSpec {
                continuation: CorePotentialContinuationSpec {
                    boundary_tolerance: 1.0e-10,
                    coulomb_tolerance: 1.0e-7,
                },
                xc_functional: XcFunctional::LdaPw92,
                xc_noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
                xc_angular_point_count: 26,
            },
        )
        .unwrap();
        assert!(
            built[0]
                .potential
                .values
                .iter()
                .zip(&transverse_built[0].potential.values)
                .all(|(&longitudinal, &transverse)| (longitudinal - transverse).abs() < 2.0e-12)
        );
        let masked_xc_g0 = xc
            .potential
            .scalar()
            .interstitial()
            .coefficient([0; 3])
            .unwrap()
            .re;
        let pointwise_xc = evaluate_interstitial_noncollinear_xc_potential(
            XcFunctional::LdaPw92,
            NoncollinearXcRoute::LocalSpinFrame,
            &density,
            [Bohr(0.0); 3],
        )
        .unwrap()
        .0
        .get();
        assert!((masked_xc_g0 - pointwise_xc).abs() > 1.0e-4);

        let mismatch = 0.03;
        let shift = muffin_tin(&mesh(101), |_| mismatch);
        let mut shifted_muffin_tins = xc.potential.scalar().muffin_tins().to_vec();
        shifted_muffin_tins[0].add_scaled(1.0, &shift).unwrap();
        let shifted_scalar = crate::RegionalScalarField::new(
            density.geometry().clone(),
            shifted_muffin_tins,
            xc.potential.scalar().interstitial().clone(),
        )
        .unwrap();
        let shifted_xc = crate::RegionalXcResult {
            potential: crate::RegionalPotential::new(
                shifted_scalar,
                xc.potential.magnetic().clone(),
            )
            .unwrap(),
            exchange_correlation_energy: xc.exchange_correlation_energy,
            density_potential_integral: xc.density_potential_integral,
        };
        let shifted = build_extended_core_potentials(
            &electrostatics,
            &shifted_xc,
            &density,
            std::slice::from_ref(&extended),
            CorePotentialBuildSpec {
                continuation: CorePotentialContinuationSpec {
                    boundary_tolerance: 1.0e-10,
                    coulomb_tolerance: 1.0e-7,
                },
                xc_functional: XcFunctional::LdaPw92,
                xc_noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
                xc_angular_point_count: 26,
            },
        )
        .unwrap();
        assert!(
            (shifted[0].join.boundary_value_correction
                - built[0].join.boundary_value_correction
                - mismatch)
                .abs()
                < 1.0e-12
        );
        assert!(shifted[0].join.corrected_boundary_residual.abs() < 1.0e-12);
        assert!(shifted[0].join.outer_correction_residual.abs() < 1.0e-12);
        assert_eq!(
            shifted[0].potential.values.last(),
            built[0].potential.values.last()
        );
    }

    #[test]
    fn checkpoint_bridge_preserves_mt_and_reports_compact_join() {
        let radial = mesh(101);
        let extended = mesh(108);
        let charge = 3.0;
        let regular = 0.7;
        let checkpoint_mt = muffin_tin(&radial, |radius| -charge / radius + regular);
        let reciprocal_layout = layout();
        let outer_level = -1.25;
        let scalar = crate::RegionalScalarField::new(
            geometry(),
            vec![checkpoint_mt],
            interstitial(&reciprocal_layout, outer_level),
        )
        .unwrap();
        let zero = scalar.zero_like();
        let checkpoint =
            crate::RegionalPotential::new(scalar, [zero.clone(), zero.clone(), zero]).unwrap();
        let built = build_extended_checkpoint_core_potentials(
            &checkpoint,
            &geometry(),
            &[charge],
            &[extended],
            CorePotentialContinuationSpec {
                boundary_tolerance: 1.0e-13,
                coulomb_tolerance: 1.0e-10,
            },
        )
        .unwrap();
        let site = &built[0];
        let original = radial
            .radii()
            .iter()
            .map(|radius| -charge / radius.get() + regular)
            .collect::<Vec<_>>();
        for (&actual, expected) in site.potential.values[..radial.len()].iter().zip(original) {
            assert!((actual - expected).abs() < 1.0e-12 * expected.abs().max(1.0));
        }
        assert_eq!(site.join.uncorrected_outer_boundary, outer_level);
        assert!(
            (site.join.boundary_value_correction - (site.join.muffin_tin_boundary - outer_level))
                .abs()
                < 1.0e-14
        );
        assert!(site.join.corrected_boundary_residual.abs() < 1.0e-14);
        assert!(site.join.outer_correction_residual.abs() < 1.0e-14);
        assert!((site.potential.values.last().unwrap() - outer_level).abs() < 1.0e-14);
    }
}
