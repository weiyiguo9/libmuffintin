use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use muffintin::{
    AtomicStartRequest, CheckpointPhysics, GammaExchangeTreatment, RegionalFieldLayout,
    RelaxedCoreHfIterationDiagnostic, RelaxedCoreHfResult, RelaxedCoreHfSpec, Structure,
    checkpoint_v2_from_regional_state, materialize_atomic_start, run_gamma_relaxed_core_hf,
};
use muffintin_core::{AngularGrid, Bohr, ExponentialMesh, Hartree, InverseBohr};
use muffintin_coulomb::CoulombRequest;
use muffintin_dft::{
    AtomicChannelTreatment, AtomicNumber, CoreFixedPotentialSpec, CoreShellOccupations,
    LinearizationEnergyGenerator, NoncollinearXcRoute, ScfBasis, ScfChannelIdentity,
    ScfChannelProvenance, ScfChannelRecipe, ScfChannelTreatment, ScfConfig, ScfConvergence,
    ScfCoreSite, ScfCoreState, ScfExchangeCorrelation, ScfKMesh, ScfKReduction, ScfMixing,
    ScfOccupations, ScfRelativity, XcFunctional, fleur_default_atomic_configuration,
};
use muffintin_io::{
    AngularBasis, CheckpointFile, CheckpointMeta, EnergyUnit, ExponentialMeshSpec, GeometryV2,
    LatticeV1, LengthUnit, LinearizationV1, PotentialConventionV1, PotentialRadialQuantityV1,
    RadialBasisSpinV2, RadialEquationTag, SiteRadialBasisV2, SiteV2, SphericalChannelConvention,
    checkpoint_file_to_toml,
};
use muffintin_prodbasis::mpb::DEFAULT_TOLERANCE;
use serde::Serialize;

const KR_Z: u8 = 36;
const SITE_ID: &str = "Kr-1";
const RADIAL_FIRST_BOHR: f64 = 1.0e-6;
const FREE_ATOM_FIRST_BOHR: f64 = 1.0e-8;
const FREE_ATOM_INCREMENT: f64 = 0.01;
const FREE_ATOM_POINTS: usize = 2_143;
const MINIMUM_ANGULAR_GRID_POINTS: usize = 50;
const OUTER_MAX_ITERATIONS: usize = 2;
const CORE_MAX_ITERATIONS: usize = 2;
const OUTER_MIXING_ALPHA: f64 = 0.1;
const MAX_FOCK_ITERATIONS: usize = 32;
const LOOSE_TOLERANCE: f64 = 1.0e100;
const FOCK_DENSITY_TOLERANCE: f64 = 1.0e-7;
const FOCK_MIXING: f64 = 0.5;
const SECTOR_NUMERICAL_TOLERANCE_HARTREE: f64 = 1.0e-8;
const MAXIMUM_CORE_SHELL_SPILL: f64 = 1.0;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
enum HdloSelection {
    None,
    All,
}

impl HdloSelection {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "none" => Ok(Self::None),
            "all" => Ok(Self::All),
            _ => Err(invalid_input(format!(
                "--hdlo must be none or all, got {value:?}"
            ))),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::All => "all",
        }
    }
}

#[derive(Debug)]
struct Cli {
    out: PathBuf,
    box_size: f64,
    orbital_g: f64,
    field_g: f64,
    orbital_l_max: u32,
    product_g: f64,
    product_l_max: u32,
    lexp: u32,
    muffin_tin_radius: f64,
    radial_points: usize,
    hdlo: HdloSelection,
    temperature: f64,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            out: PathBuf::from("kr-relaxed-core-hf-p0"),
            box_size: 8.0,
            orbital_g: 1.0,
            field_g: 4.5,
            orbital_l_max: 1,
            product_g: 1.0,
            product_l_max: 2,
            lexp: 2,
            muffin_tin_radius: 2.0,
            radial_points: 2_401,
            hdlo: HdloSelection::None,
            temperature: 0.02,
        }
    }
}

impl Cli {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut cli = Self::default();
        let mut arguments = env::args().skip(1);
        while let Some(name) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| invalid_input(format!("missing value after {name}")))?;
            match name.as_str() {
                "--out" => cli.out = PathBuf::from(value),
                "--box" => cli.box_size = parse_value(&name, &value)?,
                "--orbital-g" => cli.orbital_g = parse_value(&name, &value)?,
                "--field-g" => cli.field_g = parse_value(&name, &value)?,
                "--orbital-lmax" => cli.orbital_l_max = parse_value(&name, &value)?,
                "--product-g" => cli.product_g = parse_value(&name, &value)?,
                "--product-lmax" => cli.product_l_max = parse_value(&name, &value)?,
                "--lexp" => cli.lexp = parse_value(&name, &value)?,
                "--rmt" => cli.muffin_tin_radius = parse_value(&name, &value)?,
                "--radial-points" => cli.radial_points = parse_value(&name, &value)?,
                "--hdlo" => cli.hdlo = HdloSelection::parse(&value)?,
                "--temperature" => cli.temperature = parse_value(&name, &value)?,
                _ => return Err(invalid_input(format!("unknown option {name:?}"))),
            }
        }
        cli.validate()?;
        Ok(cli)
    }

    fn validate(&self) -> Result<(), Box<dyn Error>> {
        for (name, value) in [
            ("--box", self.box_size),
            ("--rmt", self.muffin_tin_radius),
            ("--orbital-g", self.orbital_g),
            ("--field-g", self.field_g),
            ("--product-g", self.product_g),
            ("--temperature", self.temperature),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid_input(format!(
                    "{name} must be finite and positive, got {value}"
                )));
            }
        }
        if self.orbital_l_max == 0 {
            return Err(invalid_input("--orbital-lmax must be positive"));
        }
        if self.product_l_max == 0 {
            return Err(invalid_input("--product-lmax must be positive"));
        }
        if self.muffin_tin_radius >= self.box_size / 2.0 {
            return Err(invalid_input("--rmt must be smaller than --box / 2"));
        }
        if self.radial_points < 2 {
            return Err(invalid_input("--radial-points must be at least 2"));
        }
        if self.product_l_max > self.lexp || self.lexp > 12 {
            return Err(invalid_input(
                "angular cutoffs must satisfy --product-lmax <= --lexp <= 12",
            ));
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct Manifest {
    schema_version: u32,
    status: &'static str,
    system: SystemManifest,
    units: UnitsManifest,
    parameters: ParameterManifest,
    derived: DerivedManifest,
    provenance: ProvenanceManifest,
}

#[derive(Serialize)]
struct SystemManifest {
    element: &'static str,
    atomic_number: u8,
    nucleus: &'static str,
    site_id: &'static str,
    fractional_position: [f64; 3],
}

#[derive(Serialize)]
struct UnitsManifest {
    length: &'static str,
    inverse_length: &'static str,
    energy: &'static str,
    density: &'static str,
    position: &'static str,
    occupation_and_electron_count: &'static str,
    radial_norm_and_residual: &'static str,
}

#[derive(Serialize)]
struct ParameterManifest {
    output_directory: String,
    box_bohr: f64,
    muffin_tin_radius_bohr: f64,
    radial_first_bohr: f64,
    radial_points: usize,
    radial_log_increment: f64,
    orbital_g_max_bohr_inverse: f64,
    orbital_l_max: u32,
    field_g_max_bohr_inverse: f64,
    field_l_max: u32,
    product_g_max_bohr_inverse: f64,
    product_l_max: u32,
    weinert_lexp: u32,
    hdlo: &'static str,
    temperature_hartree: f64,
    k_mesh_divisions: [usize; 3],
    k_mesh_shift: [f64; 3],
    k_mesh_reduction: &'static str,
    gamma_exchange: &'static str,
    relativity: &'static str,
    exchange_correlation: &'static str,
    outer_mixing_alpha: f64,
    outer_energy_tolerance_hartree: f64,
    outer_density_tolerance: f64,
    outer_max_iterations: usize,
    core_action_mixing: f64,
    core_energy_tolerance_hartree: f64,
    core_radial_tolerance: f64,
    core_vc_imaginary_tolerance: f64,
    core_max_iterations: usize,
    max_fock_iterations: usize,
    fock_density_tolerance: f64,
    fock_mixing: f64,
    overlap_tolerance: f64,
    sector_numerical_tolerance_hartree: f64,
    maximum_core_shell_spill: f64,
    free_atom_radial_first_bohr: f64,
    free_atom_radial_log_increment: f64,
    free_atom_radial_points: usize,
    free_atom_mixing: f64,
    free_atom_potential_tolerance_hartree: f64,
    free_atom_tail_tolerance: f64,
    free_atom_max_iterations: usize,
    angular_grid_points: usize,
}

#[derive(Serialize)]
struct DerivedManifest {
    core_charge_electrons: f64,
    valence_charge_electrons: f64,
    total_charge_electrons: f64,
    core_states: Vec<CoreStateRecord>,
    basis_channels: Vec<BasisChannelRecord>,
}

#[derive(Clone, Serialize)]
struct CoreStateRecord {
    principal_quantum_number: u32,
    kappa: i32,
    occupation_electrons: f64,
}

#[derive(Serialize)]
struct BasisChannelRecord {
    principal_quantum_number: u32,
    angular_momentum: u32,
    kappa: Option<i32>,
    treatment: &'static str,
    derivative_order: u32,
    generator: &'static str,
}

#[derive(Serialize)]
struct ProvenanceManifest {
    crate_name: &'static str,
    crate_version: &'static str,
    git_sha: String,
    git_dirty: bool,
    atomic_configuration: &'static str,
}

#[derive(Serialize)]
struct IterationsFile {
    status: &'static str,
    iterations: Vec<IterationRecord>,
}

#[derive(Serialize)]
struct IterationRecord {
    iteration: usize,
    fock_iterations: usize,
    exchange_rebuilds: usize,
    trace_vv_hartree: f64,
    trace_cv_hartree: f64,
    trace_vc_hartree: f64,
    trace_cc_hartree: f64,
    exchange_vv_hartree: f64,
    exchange_cross_hartree: f64,
    exchange_cc_hartree: f64,
    exchange_total_hartree: f64,
    cv_vc_trace_mismatch_hartree: f64,
    valence_feedback_vv_cv_trace_hartree: f64,
    maximum_antihermitian_residual: f64,
    fock_fixed_point_residual: f64,
    valence_density_rms: f64,
    total_density_rms: f64,
    valence_electron_count: f64,
    core_electron_count: f64,
    total_electron_count: f64,
    core_inner_iterations: Vec<usize>,
    core_maximum_energy_change_hartree: f64,
    core_maximum_radial_residual: f64,
    core_h0_trace_hartree: f64,
    total_energy_hartree: f64,
    energy_change_hartree: Option<f64>,
    valence_eigenvalue_identity_residual: f64,
    lifting_identity_residual: f64,
    first_global_solve_identity_residual: Option<f64>,
    fresh_core_replacement_rms: f64,
    delta_c: Vec<DeltaCRecord>,
    weighted_delta_closure_residual_hartree: f64,
}

#[derive(Serialize)]
struct ResultFile {
    status: &'static str,
    total_energy_hartree: f64,
    sector_energies: SectorEnergies,
    sector_traces: SectorTraces,
    cv_vc_trace_mismatch_hartree: f64,
    core_h0_trace_hartree: f64,
    orbital_energies_and_occupations: Vec<OrbitalRecord>,
    core_shells: Vec<CoreShellRecord>,
    core_h0_shells: Vec<CoreH0ShellRecord>,
    delta_c: Vec<DeltaCRecord>,
    electron_counts: ElectronCounts,
    residuals: ResidualRecord,
    exchange_rebuilds: usize,
    k_fractional: Vec<[f64; 3]>,
    q_fractional: Vec<[f64; 3]>,
    k_weights: Vec<f64>,
}

#[derive(Serialize)]
struct SectorEnergies {
    vv_hartree: f64,
    cross_hartree: f64,
    cc_hartree: f64,
    total_exchange_hartree: f64,
}

#[derive(Serialize)]
struct SectorTraces {
    vv_hartree: f64,
    cv_hartree: f64,
    vc_hartree: f64,
    cc_hartree: f64,
}

#[derive(Serialize)]
struct OrbitalRecord {
    k_index: usize,
    band_index: usize,
    energy_hartree: f64,
    occupation: f64,
}

#[derive(Serialize)]
struct CoreShellRecord {
    site_index: usize,
    site_id: String,
    shell_index: usize,
    principal_quantum_number: u32,
    kappa: i32,
    energy_hartree: f64,
    occupation: f64,
    norm_total: f64,
    norm_muffin_tin: f64,
    spill: f64,
}

#[derive(Serialize)]
struct CoreH0ShellRecord {
    site_index: usize,
    site_id: String,
    shell_index: usize,
    principal_quantum_number: u32,
    kappa: i32,
    occupation: f64,
    expectation_hartree: f64,
    contribution_hartree: f64,
}

#[derive(Clone, Serialize)]
struct DeltaCRecord {
    core_index: usize,
    site_index: usize,
    principal_quantum_number: u32,
    kappa: i32,
    twice_mu: i64,
    occupation: f64,
    exact_vc_hartree: f64,
    spherical_vc_hartree: f64,
    delta_c_hartree: f64,
}

#[derive(Serialize)]
struct ElectronCounts {
    valence: f64,
    core: f64,
    total: f64,
}

#[derive(Serialize)]
struct ResidualRecord {
    maximum_antihermitian: f64,
    fock_fixed_point: f64,
    valence_density_rms: f64,
    total_density_rms: f64,
    valence_eigenvalue_identity: f64,
    lifting_identity: f64,
    first_global_solve_identity: Option<f64>,
    fresh_core_replacement_rms: f64,
    weighted_delta_closure_hartree: f64,
    vc_action_legacy_radial_hartree: f64,
    vc_action_cross_cv_mpb_difference_hartree: f64,
    vc_action_mpb_difference_hartree: f64,
    mpb_cross_trace_hartree: f64,
    maximum_measured_shell_spill: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse()?;
    let field_l_max = cli
        .orbital_l_max
        .checked_add(1)
        .and_then(|l_max| l_max.checked_mul(2))
        .ok_or_else(|| invalid_input("--orbital-lmax is too large to derive field_lmax"))?;
    let angular_grid_points = usize::try_from(field_l_max + 1)?
        .checked_pow(2)
        .and_then(|modes| modes.checked_mul(2))
        .ok_or_else(|| invalid_input("--orbital-lmax is too large to derive angular grid"))?
        .max(MINIMUM_ANGULAR_GRID_POINTS);
    let radial_log_increment =
        (cli.muffin_tin_radius / RADIAL_FIRST_BOHR).ln() / (cli.radial_points - 1) as f64;

    let atomic_number = AtomicNumber::new(KR_Z).expect("Kr is in the FLEUR catalogue");
    let configuration = fleur_default_atomic_configuration(atomic_number);
    let (core_states, core_charge, valence_charge) = derive_core_partition(&configuration);
    if (core_charge + valence_charge - f64::from(KR_Z)).abs() > 1.0e-12 {
        return Err(invalid_input(format!(
            "FLEUR Kr core and valence charges must sum to {KR_Z}, got {}",
            core_charge + valence_charge
        )));
    }
    let channels = derive_basis_channels(&configuration, cli.orbital_l_max, cli.hdlo);

    let geometry = GeometryV2 {
        lattice: LatticeV1 {
            unit: LengthUnit::Bohr,
            vectors: [
                [cli.box_size, 0.0, 0.0],
                [0.0, cli.box_size, 0.0],
                [0.0, 0.0, cli.box_size],
            ],
        },
        sites: vec![SiteV2 {
            id: SITE_ID.to_owned(),
            atomic_number: u16::from(KR_Z),
            fractional_position: [0.5; 3],
            muffin_tin_radius_unit: LengthUnit::Bohr,
            muffin_tin_radius: cli.muffin_tin_radius,
        }],
        radial_basis: vec![SiteRadialBasisV2 {
            site_id: SITE_ID.to_owned(),
            spin: RadialBasisSpinV2::Scalar,
            mesh: ExponentialMeshSpec {
                radius_unit: LengthUnit::Bohr,
                first: RADIAL_FIRST_BOHR,
                log_increment: radial_log_increment,
                point_count: cli.radial_points,
                last: cli.muffin_tin_radius,
                consistency_tolerance: 1.0e-12,
            },
            radial_equation: RadialEquationTag::FullyRelativisticDirac,
            linearization: LinearizationV1 {
                energy_unit: EnergyUnit::Hartree,
                linearization_energies: Vec::new(),
                local_orbital_energies: Vec::new(),
            },
        }],
    };
    let meta = CheckpointMeta {
        title: "Kr point-nucleus Gamma relaxed-core HF smoke".to_owned(),
        producer: "libmuffintin-runtime kr_relaxed_core_hf example".to_owned(),
        producer_version: Some(env!("CARGO_PKG_VERSION").to_owned()),
        energy_zero: "periodic finite-cell electrostatic reference".to_owned(),
        potential_convention: PotentialConventionV1 {
            angular_basis: AngularBasis::ComplexCondonShortley,
            radial_quantity: PotentialRadialQuantityV1::Potential,
            spherical_channel: SphericalChannelConvention::PhysicalValue,
        },
        annotations: BTreeMap::from([
            ("nucleus.model".to_owned(), "point".to_owned()),
            (
                "atomic_configuration.source".to_owned(),
                "fleur_default_atomic_configuration".to_owned(),
            ),
        ]),
    };
    let config = ScfConfig {
        electron_count: core_charge + valence_charge,
        k_mesh: ScfKMesh {
            divisions: [1, 1, 1],
            shift: [0.0; 3],
            reduction: ScfKReduction::Full,
        },
        basis: ScfBasis {
            plane_wave_cutoff: InverseBohr(cli.orbital_g),
            l_max: cli.orbital_l_max,
            channels: channels.clone(),
            resolved_channels: Vec::new(),
        },
        occupations: ScfOccupations::FermiDirac {
            temperature: Hartree(cli.temperature),
        },
        exchange_correlation: ScfExchangeCorrelation {
            functional: XcFunctional::LdaPw92,
            noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
        },
        mixing: ScfMixing::Linear {
            alpha: OUTER_MIXING_ALPHA,
        },
        relativity: ScfRelativity::SpinorFirstVariation,
        convergence: ScfConvergence {
            energy_tolerance: Hartree(LOOSE_TOLERANCE),
            density_tolerance: LOOSE_TOLERANCE,
            max_iterations: OUTER_MAX_ITERATIONS,
        },
        core_sites: vec![ScfCoreSite {
            id: SITE_ID.to_owned(),
            states: core_states.clone(),
        }],
    };

    let structure = Structure::new(geometry)?;
    let field_layout =
        RegionalFieldLayout::from_g_cutoff(&structure, InverseBohr(cli.field_g), field_l_max)?;
    let free_atom_mesh = ExponentialMesh::new(
        Bohr(FREE_ATOM_FIRST_BOHR),
        FREE_ATOM_INCREMENT,
        FREE_ATOM_POINTS,
    )?;
    let start = materialize_atomic_start(AtomicStartRequest {
        meta,
        structure,
        field_layout,
        exchange_correlation: config.exchange_correlation,
        free_atom_scf: muffintin_dft::FreeAtomScfSpec {
            mesh: free_atom_mesh,
            mixing: 0.3,
            potential_tolerance: 2.0e-5,
            tail_tolerance: 1.0e-7,
            max_iterations: 120,
        },
        angular_grid: AngularGrid::fibonacci(angular_grid_points)?,
    })?;

    let (git_sha, git_dirty) = git_provenance()?;
    let manifest = Manifest {
        schema_version: 1,
        status: "prepared_smoke",
        system: SystemManifest {
            element: "Kr",
            atomic_number: KR_Z,
            nucleus: "point",
            site_id: SITE_ID,
            fractional_position: [0.5; 3],
        },
        units: UnitsManifest {
            length: "Bohr",
            inverse_length: "InverseBohr",
            energy: "Hartree",
            density: "Bohr^-3",
            position: "fractional direct-lattice coordinates",
            occupation_and_electron_count: "electrons",
            radial_norm_and_residual: "dimensionless",
        },
        parameters: ParameterManifest {
            output_directory: cli.out.display().to_string(),
            box_bohr: cli.box_size,
            muffin_tin_radius_bohr: cli.muffin_tin_radius,
            radial_first_bohr: RADIAL_FIRST_BOHR,
            radial_points: cli.radial_points,
            radial_log_increment,
            orbital_g_max_bohr_inverse: cli.orbital_g,
            orbital_l_max: cli.orbital_l_max,
            field_g_max_bohr_inverse: cli.field_g,
            field_l_max,
            product_g_max_bohr_inverse: cli.product_g,
            product_l_max: cli.product_l_max,
            weinert_lexp: cli.lexp,
            hdlo: cli.hdlo.as_str(),
            temperature_hartree: cli.temperature,
            k_mesh_divisions: [1, 1, 1],
            k_mesh_shift: [0.0; 3],
            k_mesh_reduction: "full",
            gamma_exchange: "finite-body",
            relativity: "spinor-first-variation with fully-relativistic Dirac radial equation",
            exchange_correlation: "LDA-PW92 local-spin-frame",
            outer_mixing_alpha: OUTER_MIXING_ALPHA,
            outer_energy_tolerance_hartree: LOOSE_TOLERANCE,
            outer_density_tolerance: LOOSE_TOLERANCE,
            outer_max_iterations: OUTER_MAX_ITERATIONS,
            core_action_mixing: 1.0,
            core_energy_tolerance_hartree: LOOSE_TOLERANCE,
            core_radial_tolerance: LOOSE_TOLERANCE,
            core_vc_imaginary_tolerance: 1.0e-8,
            core_max_iterations: CORE_MAX_ITERATIONS,
            max_fock_iterations: MAX_FOCK_ITERATIONS,
            fock_density_tolerance: FOCK_DENSITY_TOLERANCE,
            fock_mixing: FOCK_MIXING,
            overlap_tolerance: DEFAULT_TOLERANCE,
            sector_numerical_tolerance_hartree: SECTOR_NUMERICAL_TOLERANCE_HARTREE,
            maximum_core_shell_spill: MAXIMUM_CORE_SHELL_SPILL,
            free_atom_radial_first_bohr: FREE_ATOM_FIRST_BOHR,
            free_atom_radial_log_increment: FREE_ATOM_INCREMENT,
            free_atom_radial_points: FREE_ATOM_POINTS,
            free_atom_mixing: 0.3,
            free_atom_potential_tolerance_hartree: 2.0e-5,
            free_atom_tail_tolerance: 1.0e-7,
            free_atom_max_iterations: 120,
            angular_grid_points,
        },
        derived: DerivedManifest {
            core_charge_electrons: core_charge,
            valence_charge_electrons: valence_charge,
            total_charge_electrons: core_charge + valence_charge,
            core_states: core_states.iter().map(core_state_record).collect(),
            basis_channels: channels.iter().map(basis_channel_record).collect(),
        },
        provenance: ProvenanceManifest {
            crate_name: env!("CARGO_PKG_NAME"),
            crate_version: env!("CARGO_PKG_VERSION"),
            git_sha,
            git_dirty,
            atomic_configuration: "FLEUR default.econfig catalogue embedded by libmuffintin-dft",
        },
    };

    fs::create_dir_all(&cli.out)?;
    write_toml(&cli.out.join("manifest.toml"), &manifest)?;
    write_checkpoint(&cli.out.join("initial-checkpoint.toml"), &start.checkpoint)?;

    let spec = RelaxedCoreHfSpec {
        config,
        product_l_max: cli.product_l_max,
        product_g_max: InverseBohr(cli.product_g),
        overlap_tolerance: DEFAULT_TOLERANCE,
        coulomb: CoulombRequest::cubic(cli.box_size, cli.lexp)?,
        gamma: GammaExchangeTreatment::FiniteBody,
        max_fock_iterations: MAX_FOCK_ITERATIONS,
        fock_density_tolerance: FOCK_DENSITY_TOLERANCE,
        fock_mixing: FOCK_MIXING,
        core: CoreFixedPotentialSpec {
            action_mixing: 1.0,
            energy_tolerance: Hartree(LOOSE_TOLERANCE),
            radial_tolerance: LOOSE_TOLERANCE,
            vc_imaginary_tolerance: 1.0e-8,
            max_iterations: CORE_MAX_ITERATIONS,
        },
        sector_numerical_tolerance: Hartree(SECTOR_NUMERICAL_TOLERANCE_HARTREE),
        maximum_core_shell_spill: MAXIMUM_CORE_SHELL_SPILL,
    };
    let mut physics = CheckpointPhysics::new(&start.checkpoint)?;
    let result = run_gamma_relaxed_core_hf(&mut physics, &spec)?;

    let iterations = IterationsFile {
        status: "smoke_completed",
        iterations: result.diagnostics.iter().map(iteration_record).collect(),
    };
    write_toml(&cli.out.join("iterations.toml"), &iterations)?;
    let result_file = result_record(&result);
    write_toml(&cli.out.join("result.toml"), &result_file)?;

    let final_checkpoint = checkpoint_v2_from_regional_state(
        &start.checkpoint,
        &result.total_density,
        &result.potential,
        BTreeMap::from([
            (
                "hf.driver".to_owned(),
                "run_gamma_relaxed_core_hf".to_owned(),
            ),
            ("hf.status".to_owned(), "smoke_completed".to_owned()),
        ]),
    )?;
    write_checkpoint(&cli.out.join("final-checkpoint.toml"), &final_checkpoint)?;

    println!(
        "status=smoke_completed out={} total_energy_hartree={} outer_iterations={} exchange_rebuilds={}",
        cli.out.display(),
        result.total_energy.get(),
        result.diagnostics.len(),
        result.exchange_rebuilds
    );
    Ok(())
}

fn derive_core_partition(
    configuration: &muffintin_dft::AtomicElectronicConfiguration,
) -> (Vec<ScfCoreState>, f64, f64) {
    let mut core_states = Vec::new();
    let mut core_charge = 0.0;
    let mut valence_charge = 0.0;
    for occupied in configuration.occupations() {
        match occupied.treatment {
            AtomicChannelTreatment::Core => {
                core_charge += occupied.occupation;
                core_states.push(ScfCoreState {
                    principal_quantum_number: u32::from(
                        occupied.orbital.principal_quantum_number(),
                    ),
                    kappa: i32::from(occupied.orbital.kappa()),
                    occupation: occupied.occupation,
                });
            }
            AtomicChannelTreatment::Valence | AtomicChannelTreatment::RelativisticLocalOrbital => {
                valence_charge += occupied.occupation;
            }
        }
    }
    (core_states, core_charge, valence_charge)
}

fn derive_basis_channels(
    configuration: &muffintin_dft::AtomicElectronicConfiguration,
    l_max: u32,
    hdlo: HdloSelection,
) -> Vec<ScfChannelRecipe> {
    let mut channels = Vec::new();
    let mut collapsed_valence = BTreeSet::new();
    for occupied in configuration.occupations() {
        let n = u32::from(occupied.orbital.principal_quantum_number());
        let kappa = i32::from(occupied.orbital.kappa());
        let l = angular_momentum(kappa);
        match occupied.treatment {
            AtomicChannelTreatment::Core => {}
            AtomicChannelTreatment::Valence => {
                if collapsed_valence.insert((n, l)) {
                    channels.push(channel(
                        ScfChannelIdentity::ScalarL { n, l },
                        ScfChannelTreatment::Valence,
                        0,
                    ));
                }
            }
            AtomicChannelTreatment::RelativisticLocalOrbital => channels.push(channel(
                ScfChannelIdentity::Kappa { n, kappa },
                ScfChannelTreatment::Lo,
                0,
            )),
        }
    }
    for l in 0..=l_max {
        if !channels.iter().any(|recipe| {
            recipe.treatment == ScfChannelTreatment::Valence && identity_l(recipe.identity) == l
        }) {
            let mut n = l + 1;
            while channels
                .iter()
                .any(|recipe| identity_l(recipe.identity) == l && identity_n(recipe.identity) == n)
            {
                n += 1;
            }
            channels.push(channel(
                ScfChannelIdentity::ScalarL { n, l },
                ScfChannelTreatment::Valence,
                0,
            ));
        }
    }
    if matches!(hdlo, HdloSelection::All) {
        for l in 0..=l_max {
            let n = channels
                .iter()
                .find(|recipe| {
                    recipe.treatment == ScfChannelTreatment::Valence
                        && identity_l(recipe.identity) == l
                })
                .map(|recipe| identity_n(recipe.identity))
                .expect("every l has one valence channel");
            channels.push(channel(
                ScfChannelIdentity::ScalarL { n, l },
                ScfChannelTreatment::Hdlo,
                2,
            ));
        }
    }
    channels
}

fn channel(
    identity: ScfChannelIdentity,
    treatment: ScfChannelTreatment,
    derivative_order: u32,
) -> ScfChannelRecipe {
    ScfChannelRecipe {
        site: SITE_ID.to_owned(),
        identity,
        treatment,
        derivative_order,
        generator: LinearizationEnergyGenerator::Atomic,
        seed: None,
        provenance: ScfChannelProvenance::BuiltIn,
    }
}

fn angular_momentum(kappa: i32) -> u32 {
    if kappa > 0 {
        kappa as u32
    } else {
        (-kappa - 1) as u32
    }
}

fn identity_n(identity: ScfChannelIdentity) -> u32 {
    match identity {
        ScfChannelIdentity::ScalarL { n, .. } | ScfChannelIdentity::Kappa { n, .. } => n,
    }
}

fn identity_l(identity: ScfChannelIdentity) -> u32 {
    match identity {
        ScfChannelIdentity::ScalarL { l, .. } => l,
        ScfChannelIdentity::Kappa { kappa, .. } => angular_momentum(kappa),
    }
}

fn core_state_record(state: &ScfCoreState) -> CoreStateRecord {
    CoreStateRecord {
        principal_quantum_number: state.principal_quantum_number,
        kappa: state.kappa,
        occupation_electrons: state.occupation,
    }
}

fn basis_channel_record(recipe: &ScfChannelRecipe) -> BasisChannelRecord {
    let (principal_quantum_number, angular_momentum, kappa) = match recipe.identity {
        ScfChannelIdentity::ScalarL { n, l } => (n, l, None),
        ScfChannelIdentity::Kappa { n, kappa } => (n, angular_momentum(kappa), Some(kappa)),
    };
    BasisChannelRecord {
        principal_quantum_number,
        angular_momentum,
        kappa,
        treatment: match recipe.treatment {
            ScfChannelTreatment::Core => "core",
            ScfChannelTreatment::Valence => "valence",
            ScfChannelTreatment::Lo => "lo",
            ScfChannelTreatment::Hdlo => "hdlo",
        },
        derivative_order: recipe.derivative_order,
        generator: "atomic",
    }
}

fn delta_c_records(diagnostics: &[muffintin::CoreValenceDeltaDiagnostic]) -> Vec<DeltaCRecord> {
    diagnostics
        .iter()
        .map(|item| DeltaCRecord {
            core_index: item.core_index,
            site_index: item.site_index,
            principal_quantum_number: item.n,
            kappa: item.kappa.get(),
            twice_mu: item.twice_mu.get(),
            occupation: item.occupation,
            exact_vc_hartree: item.exact_vc.get(),
            spherical_vc_hartree: item.spherical_vc.get(),
            delta_c_hartree: item.delta_c.get(),
        })
        .collect()
}

fn iteration_record(item: &RelaxedCoreHfIterationDiagnostic) -> IterationRecord {
    IterationRecord {
        iteration: item.iteration,
        fock_iterations: item.fock_iterations,
        exchange_rebuilds: item.exchange_rebuilds,
        trace_vv_hartree: item.trace_vv.get(),
        trace_cv_hartree: item.trace_cv.get(),
        trace_vc_hartree: item.trace_vc.get(),
        trace_cc_hartree: item.trace_cc.get(),
        exchange_vv_hartree: item.exchange_vv.get(),
        exchange_cross_hartree: item.exchange_cv.get(),
        exchange_cc_hartree: item.exchange_cc.get(),
        exchange_total_hartree: item.exchange_total.get(),
        cv_vc_trace_mismatch_hartree: item.cv_vc_trace_mismatch.get(),
        valence_feedback_vv_cv_trace_hartree: item.valence_feedback_vv_cv_trace.get(),
        maximum_antihermitian_residual: item.maximum_antihermitian_residual,
        fock_fixed_point_residual: item.fock_fixed_point_residual,
        valence_density_rms: item.valence_density_rms,
        total_density_rms: item.total_density_rms,
        valence_electron_count: item.valence_electron_count,
        core_electron_count: item.core_electron_count,
        total_electron_count: item.total_electron_count,
        core_inner_iterations: item.core_inner_iterations.clone(),
        core_maximum_energy_change_hartree: item.core_maximum_energy_change.get(),
        core_maximum_radial_residual: item.core_maximum_radial_residual,
        core_h0_trace_hartree: item.core_h0_trace.get(),
        total_energy_hartree: item.total_energy.get(),
        energy_change_hartree: item.energy_change.map(Hartree::get),
        valence_eigenvalue_identity_residual: item.valence_eigenvalue_identity_residual,
        lifting_identity_residual: item.lifting_identity_residual,
        first_global_solve_identity_residual: item.first_global_solve_identity_residual,
        fresh_core_replacement_rms: item.fresh_core_replacement_rms,
        delta_c: delta_c_records(&item.delta_c),
        weighted_delta_closure_residual_hartree: item.weighted_delta_closure_residual.get(),
    }
}

fn result_record(result: &RelaxedCoreHfResult) -> ResultFile {
    let final_iteration = result
        .diagnostics
        .last()
        .expect("a completed HF run has a final diagnostic");
    let mut orbitals = Vec::new();
    for (k_index, energies) in result.orbital_energies.iter().enumerate() {
        for (band_index, energy) in energies.iter().enumerate() {
            orbitals.push(OrbitalRecord {
                k_index,
                band_index,
                energy_hartree: energy.get(),
                occupation: result.occupations[k_index][band_index],
            });
        }
    }
    let mut core_shells = Vec::new();
    for site in &result.core_orbitals {
        for (shell_index, shell) in site.shells.iter().enumerate() {
            let occupation = match &shell.occupations {
                CoreShellOccupations::MuResolved(values) => {
                    values.iter().map(|(_, value)| value).sum()
                }
                CoreShellOccupations::ExplicitCollinear { up, down } => up + down,
            };
            core_shells.push(CoreShellRecord {
                site_index: site.site_index,
                site_id: site.site_id.clone(),
                shell_index,
                principal_quantum_number: shell.state.n,
                kappa: shell.state.kappa.get(),
                energy_hartree: shell.energy.get(),
                occupation,
                norm_total: shell.norm_total,
                norm_muffin_tin: shell.norm_mt,
                spill: shell.spill,
            });
        }
    }
    let mut core_h0_shells = Vec::new();
    for (site, trace) in result
        .core_orbitals
        .iter()
        .zip(&result.core_one_body_traces)
    {
        for shell in &trace.shells {
            core_h0_shells.push(CoreH0ShellRecord {
                site_index: site.site_index,
                site_id: site.site_id.clone(),
                shell_index: shell.shell_index,
                principal_quantum_number: shell.state.n,
                kappa: shell.state.kappa.get(),
                occupation: shell.occupation,
                expectation_hartree: shell.expectation.get(),
                contribution_hartree: shell.contribution.get(),
            });
        }
    }
    let exchange = &result.sector_exchange;
    let comparison = &result.core_valence_comparison;
    ResultFile {
        status: "smoke_completed",
        total_energy_hartree: result.total_energy.get(),
        sector_energies: SectorEnergies {
            vv_hartree: exchange.exchange_vv.get(),
            cross_hartree: exchange.exchange_cv.get(),
            cc_hartree: exchange.exchange_cc.get(),
            total_exchange_hartree: exchange.exchange_total.get(),
        },
        sector_traces: SectorTraces {
            vv_hartree: exchange.vv.trace.get(),
            cv_hartree: exchange.cv.trace.get(),
            vc_hartree: exchange.vc.trace.get(),
            cc_hartree: exchange.cc.trace.get(),
        },
        cv_vc_trace_mismatch_hartree: exchange.cross_trace_mismatch.get(),
        core_h0_trace_hartree: result.core_h0_trace.get(),
        orbital_energies_and_occupations: orbitals,
        core_shells,
        core_h0_shells,
        delta_c: delta_c_records(&comparison.deltas),
        electron_counts: ElectronCounts {
            valence: final_iteration.valence_electron_count,
            core: final_iteration.core_electron_count,
            total: final_iteration.total_electron_count,
        },
        residuals: ResidualRecord {
            maximum_antihermitian: result.maximum_antihermitian_residual,
            fock_fixed_point: result.fock_fixed_point_residual,
            valence_density_rms: result.valence_density_rms,
            total_density_rms: result.total_density_rms,
            valence_eigenvalue_identity: final_iteration.valence_eigenvalue_identity_residual,
            lifting_identity: final_iteration.lifting_identity_residual,
            first_global_solve_identity: final_iteration.first_global_solve_identity_residual,
            fresh_core_replacement_rms: final_iteration.fresh_core_replacement_rms,
            weighted_delta_closure_hartree: comparison.weighted_delta_closure_residual.get(),
            vc_action_legacy_radial_hartree: comparison.vc_action_legacy_radial_residual.get(),
            vc_action_cross_cv_mpb_difference_hartree: comparison
                .vc_action_cross_cv_mpb_difference
                .get(),
            vc_action_mpb_difference_hartree: comparison.vc_action_mpb_difference.get(),
            mpb_cross_trace_hartree: comparison.mpb_cross_trace_residual.get(),
            maximum_measured_shell_spill: comparison.maximum_measured_shell_spill,
        },
        exchange_rebuilds: result.exchange_rebuilds,
        k_fractional: result.k_fractional.clone(),
        q_fractional: result.q_fractional.clone(),
        k_weights: result.k_weights.clone(),
    }
}

fn write_toml(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let mut text = toml::to_string_pretty(value)?;
    text.push('\n');
    fs::write(path, text)?;
    Ok(())
}

fn write_checkpoint(
    path: &Path,
    checkpoint: &muffintin_io::CheckpointV2,
) -> Result<(), Box<dyn Error>> {
    let text = checkpoint_file_to_toml(&CheckpointFile::V2(checkpoint.clone()))?;
    fs::write(path, text)?;
    Ok(())
}

fn git_provenance() -> Result<(String, bool), Box<dyn Error>> {
    let sha = git_output(&["rev-parse", "HEAD"])?;
    let status = git_output(&["status", "--porcelain", "--untracked-files=normal"])?;
    Ok((sha, !status.is_empty()))
}

fn git_output(arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(env!("CARGO_MANIFEST_DIR"))
        .args(arguments)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "git {} failed with status {}: {}",
            arguments.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ))
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn parse_value<T>(name: &str, value: &str) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    value
        .parse()
        .map_err(|source| invalid_input(format!("invalid value for {name}: {source}")))
}

fn invalid_input(message: impl Into<String>) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}
