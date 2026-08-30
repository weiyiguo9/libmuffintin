//! Checkpoint conversion shell around the runtime-internal material kernel.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;

use muffintin_core::{
    Bohr, ExponentialMesh, FourierFieldError, FourierLayout, Hartree, HermitianFourierField,
    InterstitialGeometry, LatticeError, MeshError, ReciprocalLattice, Sphere, StepFunctionError,
    VolumeBohr3,
};
use muffintin_dft::{
    InterstitialField, LinearizationEnergyGenerator, MuffinTinField, RegionalDensity,
    RegionalPotential, RegionalScalarField, ScfBasis, ScfChannelIdentity, ScfChannelTreatment,
    ScfResolvedChannelEnergy, ScfState, channel_l, generate_frozen_checkpoint_energy,
};
use muffintin_io::{
    AngularBasis, CheckpointV2, Complex64V2, DensityV2, FieldRepresentationV2, FieldUnitV2,
    FourierCoefficientV2, GeometryV2, InitialV2, InterstitialFieldV2, IoError, MuffinTinFieldV2,
    PotentialV2, RadialBasisSpinV2, RadialEquationTag, RegionalFieldV2, SpexMaterialBasisRecipeV1,
    SpexMaterialChannelKind, SphericalChannelV2,
};
use muffintin_sphere::{HarmonicConvention, SphereField, SphereFieldError};
use num_complex::Complex64;
use thiserror::Error;

mod atomic_checkpoint;
mod convert_v2;
mod material_kernel;

pub use atomic_checkpoint::{
    AtomicCheckpointError, AtomicCheckpointRequest, AtomicCheckpointResult,
    materialize_atomic_checkpoint_v2,
};
pub use convert_v2::checkpoint_v2_from_state;
use convert_v2::{convert_v2_site_bases, regional_density_from_v2, regional_potential_from_v2};
pub use material_kernel::MaterialKernelError;
pub(crate) use material_kernel::{
    CheckpointBandSolution, CheckpointKPointSolution, MaterialKernel, g_vector,
    production_density_layout, regular_k_points,
};
use material_kernel::{
    CheckpointSite, CheckpointSpin, RadialRoute, SpexBoundSpinorChannel, SpexSpinorMaterialBinding,
};

const CHECKPOINT_RADIUS_TOLERANCE: f64 = 1.0e-10;

/// Checkpoint conversion and orchestration shell.
#[derive(Debug)]
pub struct CheckpointPhysics {
    checkpoint_template: CheckpointV2,
    pub(crate) kernel: MaterialKernel,
}

pub(super) struct ConvertedCheckpointGeometry {
    pub(super) direct: [[Bohr; 3]; 3],
    pub(super) reciprocal: ReciprocalLattice,
    pub(super) geometry: InterstitialGeometry,
    pub(super) sites: Vec<CheckpointSite>,
    pub(super) nuclear_charges: Vec<f64>,
}

impl CheckpointPhysics {
    /// Convert a validated V2 checkpoint into exact internal units and conventions.
    pub fn new(checkpoint: &CheckpointV2) -> Result<Self, CheckpointPhysicsError> {
        checkpoint.validate()?;
        let converted = convert_checkpoint_geometry(&checkpoint.geometry)?;
        let restart_density = match &checkpoint.initial {
            InitialV2::FrozenPotential { .. } => None,
            InitialV2::Restart { density, .. } => Some(density),
        };
        let potential = match &checkpoint.initial {
            InitialV2::FrozenPotential { potential } | InitialV2::Restart { potential, .. } => {
                potential
            }
        };
        let frozen_potential = regional_potential_from_v2(
            potential,
            &converted.geometry,
            &converted.sites,
            converted.reciprocal,
        )?;
        let restart_density = restart_density
            .map(|density| {
                regional_density_from_v2(
                    density,
                    &converted.geometry,
                    &converted.sites,
                    converted.reciprocal,
                )
            })
            .transpose()?;
        Ok(Self {
            checkpoint_template: checkpoint.clone(),
            kernel: MaterialKernel {
                reciprocal: converted.reciprocal,
                geometry: converted.geometry,
                sites: converted.sites,
                frozen_potential,
                restart_density,
                nuclear_charges: converted.nuclear_charges,
                core_potentials: BTreeMap::new(),
                density_template: None,
                energy_terms: BTreeMap::new(),
                spex_spinor_binding: None,
            },
        })
    }

    /// Bind a caller-owned signed-kappa material recipe to one runtime basis.
    pub fn new_spex_material(
        checkpoint: &CheckpointV2,
        recipe: &SpexMaterialBasisRecipeV1,
        basis: &ScfBasis,
    ) -> Result<Self, CheckpointPhysicsError> {
        let mut physics = Self::new(checkpoint)?;
        let recorded_sha256 = checkpoint
            .meta
            .annotations
            .get("material_basis.recipe_sha256");
        let recorded_producer = checkpoint.meta.annotations.get("material_basis.producer");
        if recorded_sha256 != Some(&recipe.recipe_sha256)
            || recorded_producer != Some(&recipe.producer)
        {
            return Err(CheckpointPhysicsError::SpexMaterialProvenanceMismatch);
        }
        for site in &physics.kernel.sites {
            for (spin, source) in [&site.up, &site.down].into_iter().enumerate() {
                if source.route != RadialRoute::ScalarKoellingHarmon {
                    return Err(CheckpointPhysicsError::SpexMaterialSourceRadialEquation {
                        site: site.id.clone(),
                        spin,
                        equation: radial_route_tag(source.route),
                    });
                }
            }
        }

        let mut keys = BTreeSet::new();
        let mut channels = Vec::with_capacity(recipe.channels.len());
        for channel in &recipe.channels {
            let treatment = match channel.kind {
                SpexMaterialChannelKind::Lo | SpexMaterialChannelKind::Rlo => {
                    ScfChannelTreatment::Lo
                }
                SpexMaterialChannelKind::Hdlo => ScfChannelTreatment::Hdlo,
            };
            let identity = ScfChannelIdentity::Kappa {
                n: channel.n,
                kappa: channel.kappa,
            };
            let key = (
                channel.site_id.clone(),
                channel.n,
                channel.l,
                channel.kappa,
                match treatment {
                    ScfChannelTreatment::Lo => 0_u8,
                    ScfChannelTreatment::Hdlo => 1_u8,
                    ScfChannelTreatment::Core | ScfChannelTreatment::Valence => unreachable!(),
                },
                channel.derivative_order,
            );
            let matches = basis
                .channels
                .iter()
                .filter(|requested| {
                    requested.site == channel.site_id
                        && requested.identity == identity
                        && requested.treatment == treatment
                        && requested.derivative_order == channel.derivative_order
                        && requested.generator == LinearizationEnergyGenerator::FrozenCheckpoint
                })
                .collect::<Vec<_>>();
            if channel_l(identity) != channel.l || !keys.insert(key) || matches.len() != 1 {
                return Err(spex_material_channel_mismatch(channel, treatment));
            }
            let requested = matches[0].clone();
            let generated =
                generate_frozen_checkpoint_energy(Hartree(channel.energy)).map_err(|source| {
                    MaterialKernelError::ChannelGenerator {
                        site: requested.site.clone(),
                        identity: requested.identity,
                        treatment: requested.treatment,
                        generator: requested.generator,
                        source,
                    }
                })?;
            channels.push(SpexBoundSpinorChannel {
                l: channel.l,
                requested: requested.clone(),
                resolved: ScfResolvedChannelEnergy {
                    recipe: requested,
                    energy: generated.energy,
                    components: vec![generated],
                },
            });
        }
        physics
            .kernel
            .bind_spex_spinor(SpexSpinorMaterialBinding { channels }, basis)?;
        Ok(physics)
    }

    pub const fn reciprocal(&self) -> &ReciprocalLattice {
        self.kernel.reciprocal()
    }

    pub const fn geometry(&self) -> &InterstitialGeometry {
        self.kernel.geometry()
    }

    pub const fn frozen_potential(&self) -> &RegionalPotential {
        self.kernel.frozen_potential()
    }

    pub(crate) fn nuclear_charges(&self) -> &[f64] {
        self.kernel.nuclear_charges()
    }

    /// Execute one prepared workflow through the internal material kernel.
    pub fn execute_prepared(
        &mut self,
        workflow: &crate::PreparedWorkflow,
    ) -> Result<crate::WorkflowResult, crate::InputError> {
        crate::runner::execute_prepared_with(workflow, &mut self.kernel)
    }

    /// Serialize a converged state as a V2 restart while preserving immutable input identity.
    pub fn restart_checkpoint(
        &self,
        state: &ScfState,
    ) -> Result<CheckpointV2, CheckpointPhysicsError> {
        checkpoint_v2_from_state(&self.checkpoint_template, state)
    }
}

fn spex_material_channel_mismatch(
    channel: &muffintin_io::SpexMaterialChannelV1,
    treatment: ScfChannelTreatment,
) -> CheckpointPhysicsError {
    CheckpointPhysicsError::SpexMaterialChannelMismatch {
        site: channel.site_id.clone(),
        n: channel.n,
        l: channel.l,
        kappa: channel.kappa,
        treatment,
        derivative_order: channel.derivative_order,
        energy: channel.energy,
    }
}

pub(super) fn convert_checkpoint_geometry(
    checkpoint: &GeometryV2,
) -> Result<ConvertedCheckpointGeometry, CheckpointPhysicsError> {
    let direct = checkpoint.lattice.vectors.map(|vector| vector.map(Bohr));
    let reciprocal = ReciprocalLattice::from_direct(direct)?;
    let mut sites = Vec::with_capacity(checkpoint.sites.len());
    for site in &checkpoint.sites {
        let position = fractional_to_cartesian(site.fractional_position, direct);
        let (up, down, nonmagnetic_scalar) =
            convert_v2_site_bases(&site.id, &checkpoint.radial_basis)?;
        if up.mesh != down.mesh {
            return Err(CheckpointPhysicsError::SpinMeshMismatch {
                site: site.id.clone(),
            });
        }
        let radius = up.mesh.last();
        let scale = site
            .muffin_tin_radius
            .abs()
            .max(radius.get().abs())
            .max(1.0);
        if (site.muffin_tin_radius - radius.get()).abs() > CHECKPOINT_RADIUS_TOLERANCE * scale {
            return Err(CheckpointPhysicsError::MuffinTinMeshRadius {
                site: site.id.clone(),
                declared: site.muffin_tin_radius,
                mesh: radius.get(),
            });
        }
        sites.push(CheckpointSite {
            id: site.id.clone(),
            position,
            radius,
            up,
            down,
            nonmagnetic_scalar,
        });
    }
    let geometry = InterstitialGeometry::new(
        VolumeBohr3(determinant(checkpoint.lattice.vectors)),
        sites
            .iter()
            .map(|site| Sphere {
                center: site.position,
                radius: site.radius,
            })
            .collect::<Vec<_>>(),
    )?;
    Ok(ConvertedCheckpointGeometry {
        direct,
        reciprocal,
        geometry,
        sites,
        nuclear_charges: checkpoint
            .sites
            .iter()
            .map(|site| f64::from(site.atomic_number))
            .collect(),
    })
}

fn determinant(matrix: [[f64; 3]; 3]) -> f64 {
    let [a, b, c] = matrix;
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

fn fractional_to_cartesian(fractional: [f64; 3], direct: [[Bohr; 3]; 3]) -> [Bohr; 3] {
    let fractional = fractional.map(|value| value.rem_euclid(1.0));
    std::array::from_fn(|axis| {
        Bohr(
            fractional
                .iter()
                .zip(direct)
                .map(|(&coefficient, vector)| coefficient * vector[axis].get())
                .sum(),
        )
    })
}

fn radial_route_tag(route: RadialRoute) -> RadialEquationTag {
    match route {
        RadialRoute::Schroedinger => RadialEquationTag::Schroedinger,
        RadialRoute::ScalarKoellingHarmon => RadialEquationTag::ScalarKoellingHarmon,
        RadialRoute::Dirac => RadialEquationTag::FullyRelativisticDirac,
    }
}

/// Checkpoint conversion, SPEX construction, product bridge, or material-kernel failure.
#[derive(Debug, Error)]
pub enum CheckpointPhysicsError {
    #[error(transparent)]
    Checkpoint(#[from] IoError),
    #[error(transparent)]
    Lattice(#[from] LatticeError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    StepFunction(#[from] StepFunctionError),
    #[error(transparent)]
    Fourier(#[from] FourierFieldError),
    #[error(transparent)]
    Sphere(#[from] SphereFieldError),
    #[error(transparent)]
    Regional(#[from] muffintin_dft::RegionalError),
    #[error(transparent)]
    MaterialKernel(#[from] MaterialKernelError),
    #[error("site {site:?} radial basis must be exactly scalar or an up/down pair")]
    InvalidRadialBasisSpins { site: String },
    #[error("V2 regional field is missing site {0:?}")]
    MissingV2FieldSite(String),
    #[error("cannot export {actual} muffin-tin fields against {expected} checkpoint sites")]
    ExportSiteCount { expected: usize, actual: usize },
    #[error("cannot export {from:?} spherical fields as {target:?} fields")]
    UnsupportedAngularConversion {
        from: HarmonicConvention,
        target: HarmonicConvention,
    },
    #[error("real-tesseral channel l={l}, m={m} has no signed-m partner")]
    UnpairedRealTesseralChannel { l: u32, m: i32 },
    #[error("site {site:?} has different up/down radial meshes")]
    SpinMeshMismatch { site: String },
    #[error("site {site:?} muffin-tin radius is {declared}, radial mesh ends at {mesh}")]
    MuffinTinMeshRadius {
        site: String,
        declared: f64,
        mesh: f64,
    },
    #[error("checkpoint interstitial potential must contain G=0")]
    MissingInterstitialZero,
    #[error("SPEX material checkpoint annotations do not match the caller-owned recipe")]
    SpexMaterialProvenanceMismatch,
    #[error(
        "SPEX material source at site {site:?}, spin {spin} must remain scalar Koelling-Harmon; got {equation:?}"
    )]
    SpexMaterialSourceRadialEquation {
        site: String,
        spin: usize,
        equation: RadialEquationTag,
    },
    #[error(
        "SPEX material channel site={site:?}, n={n}, l={l}, kappa={kappa}, treatment={treatment:?}, derivative_order={derivative_order}, energy={energy} is not bound exactly to the runtime basis"
    )]
    SpexMaterialChannelMismatch {
        site: String,
        n: u32,
        l: u32,
        kappa: i32,
        treatment: ScfChannelTreatment,
        derivative_order: u32,
        energy: f64,
    },
    #[error("k point {0:?} contains a non-finite coordinate")]
    NonFiniteKPoint([f64; 3]),
    #[error("regular k-point set is empty")]
    EmptyKPointSet,
    #[error("scalar product input requires scalar Koelling-Harmon relativity")]
    ScalarProductRequiresScalarRelativity,
    #[error(
        "spinor product input requires ScfRelativity::SpinorFirstVariation, not scalar Koelling-Harmon"
    )]
    SpinorProductRejectsScalarRelativity,
    #[error(
        "spinor product input requires ScfRelativity::SpinorFirstVariation, not SOC second variation; signed-kappa is not routed through second variation"
    )]
    SpinorProductRejectsSocSecondVariation,
    #[error(
        "spinor product k-mesh, compiled bases, eigenvectors, energies, available-band counts, and k-q map must share one ordered k slice"
    )]
    SpinorProductKSliceMismatch,
    #[error("spinor product source transfer q does not match the frozen q-slice")]
    SpinorProductTransferQMismatch,
    #[error(transparent)]
    DiracProduct(#[from] muffintin_prodbasis::DiracProductError),
    #[error(
        "folded k-q {folded:?} from k={k:?} q_in={q_in:?} q_canonical={q_canonical:?} is not on the regular mesh"
    )]
    OffMeshTransfer {
        k: [f64; 3],
        q_in: [f64; 3],
        q_canonical: [f64; 3],
        folded: [f64; 3],
    },
    #[error("collinear product input needs equal up/down band counts, got {up} and {down}")]
    CollinearBandCount { up: usize, down: usize },
    #[error(transparent)]
    Product(#[from] muffintin_prodbasis::AuxiliaryIrError),
    #[error("angular momentum does not fit the signed-kappa representation")]
    AngularMomentumOverflow,
    #[error("one band solution mixed scalar and spinor k-point routes")]
    InconsistentRelativityRoute,
    #[error("k points retained inconsistent band counts")]
    InconsistentBandCount,
}

#[cfg(test)]
mod atomic_checkpoint_test;
#[cfg(test)]
mod tests;
