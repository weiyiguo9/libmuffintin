use muffintin_dft::{
    AtomicNumber, AtomicSuperpositionChargeClosure, AtomicSuperpositionError,
    AtomicSuperpositionSite, AtomicSuperpositionSpec, FreeAtomScfSpec,
    LinearizationEnergyGenerator, ScfConfig, ScfRelativity, build_atomic_superposition_density,
};
use muffintin_grid::{AngularGrid, Cell, GridError};
use muffintin_io::{
    BasisHintsV1, DensityV2, FieldRepresentationV2, FieldUnitV2, FourierNormalizationV1,
    FourierPhaseV1, InitialV2, InverseLengthUnitV1, MetaV1, PotentialV2, SnapshotV2,
};
use thiserror::Error;

use super::convert_v2::{regional_density_from_v2, regional_scalar_to_v2};
use super::{
    SnapshotDftError, build_production_potential, convert_snapshot_geometry,
    production_density_layout,
};

/// Complete structure/task request for a neutral atomic-superposition V2 restart.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomicSnapshotRequest {
    pub meta: MetaV1,
    pub geometry: muffintin_io::GeometryV2,
    pub scf: ScfConfig,
    pub free_atom_scf: FreeAtomScfSpec,
    pub atomic_superposition_angular_points: usize,
}

/// Validated restart snapshot and its exact finite-layout charge accounting.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomicSnapshotResult {
    pub snapshot: SnapshotV2,
    pub charge_closure: AtomicSuperpositionChargeClosure,
}

/// Invalid stage-4 request or failure in the production density/potential path.
#[derive(Debug, Error)]
pub enum AtomicSnapshotError {
    #[error(transparent)]
    SnapshotDft(#[from] SnapshotDftError),
    #[error(transparent)]
    AtomicSuperposition(#[from] AtomicSuperpositionError),
    #[error(transparent)]
    AngularGrid(#[from] GridError),
    #[error("neutral atomic snapshot requires electron count {neutral}, got {electron_count}")]
    NonNeutralElectronCount { electron_count: f64, neutral: f64 },
    #[error("site {site:?} atomic number {atomic_number} is outside the free-atom route 1..=103")]
    UnsupportedAtomicNumber { site: String, atomic_number: u16 },
    #[error("regular k mesh division {axis} is zero")]
    ZeroKMeshDivision { axis: usize },
    #[error("plane-wave cutoff must be finite and positive, got {0}")]
    InvalidPlaneWaveCutoff(f64),
    #[error("atomic snapshot basis l_max must be positive")]
    ZeroLMax,
    #[error("atomic snapshot muffin-tin output angular momentum overflows u32")]
    AngularMomentumOverflow,
    #[error("atomic snapshot request must not carry pre-resolved basis channels")]
    PreResolvedBasis,
    #[error("site {site:?} frozen-snapshot basis generation needs an authored snapshot anchor")]
    FrozenSnapshotGenerator { site: String },
    #[error(
        "site {site:?} spectral generator {generator:?} needs provisional bands or chemical potential"
    )]
    SpectralGenerator {
        site: String,
        generator: LinearizationEnergyGenerator,
    },
    #[error(
        "site {site:?} generator {generator:?} needs an explicit seed without a snapshot anchor"
    )]
    MissingGeneratorSeed {
        site: String,
        generator: LinearizationEnergyGenerator,
    },
    #[error("radial basis for {relativity:?} must use {expected:?}; site {site:?} uses {actual:?}")]
    RadialEquationRoute {
        site: String,
        relativity: ScfRelativity,
        expected: muffintin_io::RadialEquationTagV1,
        actual: muffintin_io::RadialEquationTagV1,
    },
}

/// Materialize a nonmagnetic neutral SAD density and its production crystal potential.
pub fn materialize_atomic_snapshot_v2(
    request: AtomicSnapshotRequest,
) -> Result<AtomicSnapshotResult, AtomicSnapshotError> {
    validate_request(&request)?;
    let converted = convert_snapshot_geometry(&request.geometry)?;
    let fourier_layout = production_density_layout(
        converted.reciprocal,
        request.scf.k_mesh,
        request.scf.basis.plane_wave_cutoff,
    )?;
    let muffin_tin_l_max = match request.scf.relativity {
        ScfRelativity::Scalar | ScfRelativity::SocSecondVariation { .. } => request
            .scf
            .basis
            .l_max
            .checked_mul(2)
            .ok_or(AtomicSnapshotError::AngularMomentumOverflow)?,
        ScfRelativity::SpinorFirstVariation => request
            .scf
            .basis
            .l_max
            .checked_add(1)
            .and_then(|value| value.checked_mul(2))
            .ok_or(AtomicSnapshotError::AngularMomentumOverflow)?,
    };
    if i32::try_from(muffin_tin_l_max).is_err() {
        return Err(AtomicSnapshotError::AngularMomentumOverflow);
    }
    let angular_grid = AngularGrid::fibonacci(request.atomic_superposition_angular_points)?;
    let direct_lattice = Cell::new(converted.direct)?;
    let sites = request
        .geometry
        .sites
        .iter()
        .zip(&converted.sites)
        .map(|(site, converted)| {
            let atomic_number = u8::try_from(site.atomic_number)
                .ok()
                .and_then(AtomicNumber::new)
                .ok_or_else(|| AtomicSnapshotError::UnsupportedAtomicNumber {
                    site: site.id.clone(),
                    atomic_number: site.atomic_number,
                })?;
            Ok(AtomicSuperpositionSite {
                atomic_number,
                position: converted.position,
                muffin_tin_mesh: converted.up.mesh.clone(),
            })
        })
        .collect::<Result<Vec<_>, AtomicSnapshotError>>()?;
    let atomic = build_atomic_superposition_density(&AtomicSuperpositionSpec {
        direct_lattice,
        sites,
        fourier_layout,
        muffin_tin_l_max,
        angular_grid,
        target_electron_count: request.scf.electron_count,
        free_atom_scf: request.free_atom_scf,
    })?;
    let angular_basis = request.meta.potential_convention.angular_basis;
    let basis_hints = BasisHintsV1 {
        reciprocal_length_unit: InverseLengthUnitV1::BohrInverse,
        plane_wave_cutoff: Some(request.scf.basis.plane_wave_cutoff),
        coefficient_cutoff: None,
        normalization: FourierNormalizationV1::CellNormalized,
        phase: FourierPhaseV1::NegativeExponent,
    };
    let density = DensityV2 {
        unit: FieldUnitV2::BohrMinus3,
        representation: FieldRepresentationV2::PeriodicExtension,
        angular_basis,
        basis_hints,
        n: regional_scalar_to_v2(
            atomic.density.charge(),
            &request.geometry.sites,
            angular_basis,
        )?,
        mx: regional_scalar_to_v2(
            &atomic.density.magnetization()[0],
            &request.geometry.sites,
            angular_basis,
        )?,
        my: regional_scalar_to_v2(
            &atomic.density.magnetization()[1],
            &request.geometry.sites,
            angular_basis,
        )?,
        mz: regional_scalar_to_v2(
            &atomic.density.magnetization()[2],
            &request.geometry.sites,
            angular_basis,
        )?,
    };
    let production_density = regional_density_from_v2(
        &density,
        &converted.geometry,
        &converted.sites,
        converted.reciprocal,
    )?;
    let built_potential = build_production_potential(
        &production_density,
        &converted.nuclear_charges,
        request.scf.exchange_correlation,
    )?;
    let potential = PotentialV2 {
        unit: FieldUnitV2::Hartree,
        representation: FieldRepresentationV2::MaskedOperator,
        angular_basis,
        basis_hints,
        v0: regional_scalar_to_v2(
            built_potential.potential.scalar(),
            &request.geometry.sites,
            angular_basis,
        )?,
        bx: regional_scalar_to_v2(
            &built_potential.potential.magnetic()[0],
            &request.geometry.sites,
            angular_basis,
        )?,
        by: regional_scalar_to_v2(
            &built_potential.potential.magnetic()[1],
            &request.geometry.sites,
            angular_basis,
        )?,
        bz: regional_scalar_to_v2(
            &built_potential.potential.magnetic()[2],
            &request.geometry.sites,
            angular_basis,
        )?,
    };
    let snapshot = SnapshotV2::new(
        request.meta,
        request.geometry,
        InitialV2::Restart { density, potential },
    );
    snapshot.validate().map_err(SnapshotDftError::from)?;
    Ok(AtomicSnapshotResult {
        snapshot,
        charge_closure: atomic.charge_closure,
    })
}

fn validate_request(request: &AtomicSnapshotRequest) -> Result<(), AtomicSnapshotError> {
    let neutral = request
        .geometry
        .sites
        .iter()
        .map(|site| f64::from(site.atomic_number))
        .sum::<f64>();
    if request.scf.electron_count != neutral {
        return Err(AtomicSnapshotError::NonNeutralElectronCount {
            electron_count: request.scf.electron_count,
            neutral,
        });
    }
    for (axis, &division) in request.scf.k_mesh.divisions.iter().enumerate() {
        if division == 0 {
            return Err(AtomicSnapshotError::ZeroKMeshDivision { axis });
        }
    }
    let cutoff = request.scf.basis.plane_wave_cutoff;
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(AtomicSnapshotError::InvalidPlaneWaveCutoff(cutoff));
    }
    if request.scf.basis.l_max == 0 {
        return Err(AtomicSnapshotError::ZeroLMax);
    }
    if !request.scf.basis.resolved_channels.is_empty() {
        return Err(AtomicSnapshotError::PreResolvedBasis);
    }
    for recipe in &request.scf.basis.channels {
        match recipe.generator {
            LinearizationEnergyGenerator::FrozenSnapshot => {
                return Err(AtomicSnapshotError::FrozenSnapshotGenerator {
                    site: recipe.site.clone(),
                });
            }
            generator @ (LinearizationEnergyGenerator::BandCog
            | LinearizationEnergyGenerator::FermiOffset) => {
                return Err(AtomicSnapshotError::SpectralGenerator {
                    site: recipe.site.clone(),
                    generator,
                });
            }
            generator @ (LinearizationEnergyGenerator::Explicit
            | LinearizationEnergyGenerator::BandCenter
            | LinearizationEnergyGenerator::LogDerivative)
                if recipe.seed.is_none() =>
            {
                return Err(AtomicSnapshotError::MissingGeneratorSeed {
                    site: recipe.site.clone(),
                    generator,
                });
            }
            LinearizationEnergyGenerator::Explicit
            | LinearizationEnergyGenerator::Atomic
            | LinearizationEnergyGenerator::BandCenter
            | LinearizationEnergyGenerator::LogDerivative => {}
        }
    }
    let expected = match request.scf.relativity {
        ScfRelativity::Scalar | ScfRelativity::SocSecondVariation { .. } => {
            muffintin_io::RadialEquationTagV1::ScalarKoellingHarmon
        }
        ScfRelativity::SpinorFirstVariation => {
            muffintin_io::RadialEquationTagV1::FullyRelativisticDirac
        }
    };
    for radial in &request.geometry.radial_basis {
        if radial.radial_equation != expected {
            return Err(AtomicSnapshotError::RadialEquationRoute {
                site: radial.site_id.clone(),
                relativity: request.scf.relativity,
                expected,
                actual: radial.radial_equation,
            });
        }
    }
    Ok(())
}
