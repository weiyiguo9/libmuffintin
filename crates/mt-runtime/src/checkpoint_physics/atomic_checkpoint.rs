use muffintin_core::{
    AngularGrid, Cell, FourierFieldError, FourierLayout, GridError, HermitianFourierField,
    InverseBohr, LatticeError,
};
use muffintin_dft::{
    AtomicNumber, AtomicSuperpositionChargeClosure, AtomicSuperpositionError,
    AtomicSuperpositionSite, AtomicSuperpositionSpec, FreeAtomScfSpec, ScfExchangeCorrelation,
    build_atomic_superposition_density, build_scf_potential, g_vector,
};
use muffintin_io::{
    BasisHints, CheckpointMeta, CheckpointV2, DensityV2, FieldRepresentationV2, FieldUnitV2,
    FourierNormalization, FourierPhase, GeometryV2, InitialV2, InverseLengthUnit, IoError,
    PotentialV2,
};
use num_complex::Complex64;
use thiserror::Error;

use super::convert_v2::{regional_density_from_v2, regional_scalar_to_v2};
use super::{CheckpointPhysicsError, ConvertedCheckpointGeometry, convert_checkpoint_geometry};

/// Validated crystal structure and exact per-site radial meshes for regional fields.
#[derive(Clone, Debug)]
pub struct Structure {
    geometry: GeometryV2,
    converted: ConvertedCheckpointGeometry,
}

impl Structure {
    /// Validate and convert one source-neutral V2 geometry into internal atomic units.
    pub fn new(geometry: GeometryV2) -> Result<Self, CheckpointPhysicsError> {
        geometry.validate().map_err(IoError::from)?;
        let converted = convert_checkpoint_geometry(&geometry)?;
        Ok(Self {
            geometry,
            converted,
        })
    }

    /// Exact checkpoint geometry retained for serialization.
    pub const fn geometry(&self) -> &GeometryV2 {
        &self.geometry
    }

    /// Runtime interstitial partition derived from the validated geometry.
    pub const fn interstitial_geometry(&self) -> &muffintin_core::InterstitialGeometry {
        &self.converted.geometry
    }

    /// Exact reciprocal lattice associated with the direct lattice.
    pub const fn reciprocal(&self) -> &muffintin_core::ReciprocalLattice {
        &self.converted.reciprocal
    }

    /// Exact per-site radial meshes in geometry site order.
    pub fn site_meshes(&self) -> impl ExactSizeIterator<Item = &muffintin_core::ExponentialMesh> {
        self.converted.sites.iter().map(|site| site.up().mesh())
    }

    /// Positive nuclear charges in geometry site order.
    pub fn nuclear_charges(&self) -> &[f64] {
        &self.converted.nuclear_charges
    }
}

/// Exact scalar regional-field layout selected independently of an orbital basis.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionalFieldLayout {
    fourier: FourierLayout,
    muffin_tin_l_max: u32,
    g_cutoff: Option<InverseBohr>,
}

impl RegionalFieldLayout {
    /// Construct from an exact ordered reciprocal-index list.
    pub fn new(
        structure: &Structure,
        reciprocal_indices: Vec<[i32; 3]>,
        muffin_tin_l_max: u32,
    ) -> Result<Self, RegionalFieldLayoutError> {
        let reciprocal = structure.converted.reciprocal;
        let vectors = reciprocal_indices
            .into_iter()
            .map(|index| g_vector(reciprocal, index))
            .collect();
        Self::from_fourier_layout(
            FourierLayout::new(reciprocal, vectors)?,
            muffin_tin_l_max,
            None,
        )
    }

    /// Construct the exact reciprocal sphere selected by a positive `bohr^-1` cutoff.
    pub fn from_g_cutoff(
        structure: &Structure,
        g_cutoff: InverseBohr,
        muffin_tin_l_max: u32,
    ) -> Result<Self, RegionalFieldLayoutError> {
        if !g_cutoff.get().is_finite() || g_cutoff.get() <= 0.0 {
            return Err(RegionalFieldLayoutError::InvalidGCutoff(g_cutoff));
        }
        let reciprocal = structure.converted.reciprocal;
        let mut vectors = reciprocal.enumerate(g_cutoff)?;
        vectors.sort_unstable_by_key(|vector| vector.index);
        Self::from_fourier_layout(
            FourierLayout::new(reciprocal, vectors)?,
            muffin_tin_l_max,
            Some(g_cutoff),
        )
    }

    fn from_fourier_layout(
        fourier: FourierLayout,
        muffin_tin_l_max: u32,
        g_cutoff: Option<InverseBohr>,
    ) -> Result<Self, RegionalFieldLayoutError> {
        if fourier.index([0; 3]).is_none() {
            return Err(RegionalFieldLayoutError::MissingZeroVector);
        }
        HermitianFourierField::new(
            fourier.clone(),
            vec![Complex64::new(0.0, 0.0); fourier.len()],
        )?;
        Ok(Self {
            fourier,
            muffin_tin_l_max,
            g_cutoff,
        })
    }

    pub const fn fourier(&self) -> &FourierLayout {
        &self.fourier
    }

    pub const fn muffin_tin_l_max(&self) -> u32 {
        self.muffin_tin_l_max
    }

    pub const fn g_cutoff(&self) -> Option<InverseBohr> {
        self.g_cutoff
    }
}

/// Invalid explicit reciprocal layout for a physically real regional field.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RegionalFieldLayoutError {
    #[error("regional-field G cutoff must be finite and positive, got {0}")]
    InvalidGCutoff(InverseBohr),
    #[error("regional-field Fourier layout must contain G=0")]
    MissingZeroVector,
    #[error(transparent)]
    Lattice(#[from] LatticeError),
    #[error(transparent)]
    Fourier(#[from] FourierFieldError),
}

/// Exact request for a neutral atomic-superposition regional start.
#[derive(Clone, Debug)]
pub struct AtomicStartRequest {
    pub meta: CheckpointMeta,
    pub structure: Structure,
    pub field_layout: RegionalFieldLayout,
    pub exchange_correlation: ScfExchangeCorrelation,
    pub free_atom_scf: FreeAtomScfSpec,
    pub angular_grid: AngularGrid,
}

/// Validated restart checkpoint and exact finite-layout charge accounting.
#[derive(Clone, Debug)]
pub struct AtomicStart {
    pub checkpoint: CheckpointV2,
    pub charge_closure: AtomicSuperpositionChargeClosure,
}

/// Invalid common atomic-start request or failure in the production density/potential path.
#[derive(Debug, Error)]
pub enum AtomicStartError {
    #[error(transparent)]
    PotentialBuild(#[from] muffintin_dft::ScfPotentialBuildError),
    #[error(transparent)]
    CheckpointPhysics(#[from] CheckpointPhysicsError),
    #[error(transparent)]
    AtomicSuperposition(#[from] AtomicSuperpositionError),
    #[error(transparent)]
    Grid(#[from] GridError),
    #[error("regional-field layout belongs to a different direct lattice")]
    FieldLayoutStructureMismatch,
    #[error("site {site:?} atomic number {atomic_number} is outside the free-atom route 1..=103")]
    UnsupportedAtomicNumber { site: String, atomic_number: u16 },
}

/// Materialize a neutral nonmagnetic density and production crystal potential.
pub fn materialize_atomic_start(
    request: AtomicStartRequest,
) -> Result<AtomicStart, AtomicStartError> {
    use crate::hf_diagnostics::HfPhaseTimer;
    let _start_timer = HfPhaseTimer::new("atomic_start.total");
    let AtomicStartRequest {
        meta,
        structure,
        field_layout,
        exchange_correlation,
        free_atom_scf,
        angular_grid,
    } = request;
    if field_layout.fourier.reciprocal() != &structure.converted.reciprocal {
        return Err(AtomicStartError::FieldLayoutStructureMismatch);
    }
    let target_electron_count = structure
        .geometry
        .sites
        .iter()
        .map(|site| f64::from(site.atomic_number))
        .sum();
    let direct_lattice = Cell::new(structure.converted.direct)?;
    let sites = structure
        .geometry
        .sites
        .iter()
        .zip(&structure.converted.sites)
        .map(|(site, converted)| {
            let atomic_number = u8::try_from(site.atomic_number)
                .ok()
                .and_then(AtomicNumber::new)
                .ok_or_else(|| AtomicStartError::UnsupportedAtomicNumber {
                    site: site.id.clone(),
                    atomic_number: site.atomic_number,
                })?;
            Ok(AtomicSuperpositionSite {
                atomic_number,
                position: converted.position(),
                muffin_tin_mesh: converted.up().mesh().clone(),
            })
        })
        .collect::<Result<Vec<_>, AtomicStartError>>()?;
    let density_timer = HfPhaseTimer::new("atomic_start.superposition_density");
    let atomic = build_atomic_superposition_density(&AtomicSuperpositionSpec {
        direct_lattice,
        sites,
        fourier_layout: field_layout.fourier.clone(),
        muffin_tin_l_max: field_layout.muffin_tin_l_max,
        angular_grid,
        target_electron_count,
        free_atom_scf,
    })?;
    drop(density_timer);
    let conversion_timer = HfPhaseTimer::new("atomic_start.density_conversion");
    let angular_basis = meta.potential_convention.angular_basis;
    let basis_hints = BasisHints {
        reciprocal_length_unit: InverseLengthUnit::BohrInverse,
        plane_wave_cutoff: field_layout.g_cutoff.map(InverseBohr::get),
        coefficient_cutoff: None,
        normalization: FourierNormalization::CellNormalized,
        phase: FourierPhase::NegativeExponent,
    };
    let density = DensityV2 {
        unit: FieldUnitV2::BohrMinus3,
        representation: FieldRepresentationV2::PeriodicExtension,
        angular_basis,
        basis_hints,
        n: regional_scalar_to_v2(
            atomic.density.charge(),
            &structure.geometry.sites,
            angular_basis,
        )?,
        mx: regional_scalar_to_v2(
            &atomic.density.magnetization()[0],
            &structure.geometry.sites,
            angular_basis,
        )?,
        my: regional_scalar_to_v2(
            &atomic.density.magnetization()[1],
            &structure.geometry.sites,
            angular_basis,
        )?,
        mz: regional_scalar_to_v2(
            &atomic.density.magnetization()[2],
            &structure.geometry.sites,
            angular_basis,
        )?,
    };
    let production_density = regional_density_from_v2(
        &density,
        &structure.converted.geometry,
        &structure.converted.sites,
        structure.converted.reciprocal,
    )?;
    drop(conversion_timer);
    let potential_timer = HfPhaseTimer::new("atomic_start.potential");
    let built_potential = build_scf_potential(
        &production_density,
        &structure.converted.nuclear_charges,
        exchange_correlation,
    )?;
    drop(potential_timer);
    let _checkpoint_timer = HfPhaseTimer::new("atomic_start.checkpoint");
    let potential = PotentialV2 {
        unit: FieldUnitV2::Hartree,
        representation: FieldRepresentationV2::MaskedOperator,
        angular_basis,
        basis_hints,
        v0: regional_scalar_to_v2(
            built_potential.potential.scalar(),
            &structure.geometry.sites,
            angular_basis,
        )?,
        bx: regional_scalar_to_v2(
            &built_potential.potential.magnetic()[0],
            &structure.geometry.sites,
            angular_basis,
        )?,
        by: regional_scalar_to_v2(
            &built_potential.potential.magnetic()[1],
            &structure.geometry.sites,
            angular_basis,
        )?,
        bz: regional_scalar_to_v2(
            &built_potential.potential.magnetic()[2],
            &structure.geometry.sites,
            angular_basis,
        )?,
    };
    let checkpoint = CheckpointV2::new(
        meta,
        structure.geometry,
        InitialV2::Restart { density, potential },
    );
    checkpoint
        .validate()
        .map_err(CheckpointPhysicsError::from)?;
    Ok(AtomicStart {
        checkpoint,
        charge_closure: atomic.charge_closure,
    })
}
