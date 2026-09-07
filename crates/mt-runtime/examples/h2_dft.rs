use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use muffintin::{
    AtomicStartRequest, Basis, BasisEnvelope, BasisEnvelopeKind, ChannelEnergyGenerator,
    Convergence, ExchangeCorrelation, Input, KMesh, Mixing, NoncollinearXcRoute, Occupations,
    RegionalFieldLayout, Relativity, Structure, Symmetry, Task, Workflow, input_to_toml,
    materialize_atomic_start, run_dft_scf,
};
use muffintin_core::{AngularGrid, Bohr, ExponentialMesh, InverseBohr};
use muffintin_dft::{
    FreeAtomScfSpec, NoncollinearXcRoute as DftNoncollinearXcRoute, ScfExchangeCorrelation,
    XcFunctional,
};
use muffintin_io::{
    AngularBasis, CheckpointFile, CheckpointMeta, EnergyUnit, ExponentialMeshSpec, GeometryV2,
    LatticeV1, LengthUnit, LinearizationV1, PotentialConventionV1, PotentialRadialQuantityV1,
    RadialBasisSpinV2, RadialEquationTag, SiteRadialBasisV2, SiteV2, SphericalChannelConvention,
    checkpoint_file_to_toml,
};

const BOND_LENGTH_BOHR: f64 = 1.4;
const DEFAULT_MUFFIN_TIN_RADIUS_BOHR: f64 = 0.65;
const RADIAL_FIRST_BOHR: f64 = 1.0e-6;
const RADIAL_POINTS: usize = 401;
const FREE_ATOM_FIRST_BOHR: f64 = 1.0e-8;
const FREE_ATOM_LOG_INCREMENT: f64 = 0.01;
const FREE_ATOM_POINTS: usize = 2_143;
const FREE_ATOM_MIXING: f64 = 0.3;
const FREE_ATOM_POTENTIAL_TOLERANCE: f64 = 2.0e-5;
const FREE_ATOM_TAIL_TOLERANCE: f64 = 1.0e-7;
const FREE_ATOM_MAX_ITERATIONS: usize = 120;
const ATOMIC_START_FIELD_L_MAX: u32 = 8;
const SCF_ORBITAL_L_MAX: u32 = 8;
const ELECTRON_COUNT: f64 = 2.0;
const FERMI_TEMPERATURE_HARTREE: f64 = 0.001;
const MIXING_BETA: f64 = 0.4;
const MIXING_HISTORY: usize = 6;
const SCF_ENERGY_TOLERANCE_HARTREE: f64 = 1.0e-8;
const SCF_DENSITY_TOLERANCE: f64 = 1.0e-7;
const SCF_MAX_ITERATIONS: usize = 80;

#[derive(Debug)]
struct Options {
    output_directory: PathBuf,
    box_size_bohr: f64,
    orbital_g_cutoff: f64,
    field_g_cutoff: f64,
    speed_of_light: f64,
    xc_grid: Option<[usize; 3]>,
    muffin_tin_radius: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            output_directory: PathBuf::from("h2-dft-output"),
            box_size_bohr: 10.0,
            orbital_g_cutoff: 6.0,
            field_g_cutoff: 12.0,
            speed_of_light: muffintin_core::SPEX_SPEED_OF_LIGHT,
            xc_grid: None,
            muffin_tin_radius: DEFAULT_MUFFIN_TIN_RADIUS_BOHR,
        }
    }
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut options = Self::default();
        let mut arguments = env::args().skip(1);
        if let Some(value) = arguments.next() {
            options.output_directory = PathBuf::from(value);
        }
        if let Some(value) = arguments.next() {
            options.box_size_bohr = value.parse()?;
        }
        if let Some(value) = arguments.next() {
            options.orbital_g_cutoff = value.parse()?;
        }
        if let Some(value) = arguments.next() {
            options.field_g_cutoff = value.parse()?;
        }
        if let Some(value) = arguments.next() {
            options.speed_of_light = value.parse()?;
        }
        if let Some(value) = arguments.next() {
            options.xc_grid = Some([value.parse()?; 3]);
        }
        if let Some(value) = arguments.next() {
            options.muffin_tin_radius = value.parse()?;
        }
        if arguments.next().is_some() {
            return Err(
                "usage: h2_dft [output-directory] [box-bohr] [orbital-g] [field-g] [speed-of-light-au] [xc-grid-size] [muffin-tin-radius-bohr]"
                    .into(),
            );
        }
        for (name, value) in [
            ("box-bohr", options.box_size_bohr),
            ("orbital-g-bohr-inverse", options.orbital_g_cutoff),
            ("field-g-bohr-inverse", options.field_g_cutoff),
            ("speed-of-light-au", options.speed_of_light),
            ("muffin-tin-radius-bohr", options.muffin_tin_radius),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive, got {value}").into());
            }
        }
        if 2.0 * options.muffin_tin_radius >= BOND_LENGTH_BOHR {
            return Err(format!(
                "muffin-tin spheres overlap: radius {} exceeds half the H2 bond",
                options.muffin_tin_radius
            )
            .into());
        }
        if options.box_size_bohr <= BOND_LENGTH_BOHR + 2.0 * options.muffin_tin_radius {
            return Err(format!(
                "box-bohr must exceed the H2 bond plus both muffin-tin radii, got {}",
                options.box_size_bohr
            )
            .into());
        }
        Ok(options)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    fs::create_dir_all(&options.output_directory)?;
    println!("speed_of_light_au={:.10}", options.speed_of_light);

    let radial_log_increment =
        (options.muffin_tin_radius / RADIAL_FIRST_BOHR).ln() / (RADIAL_POINTS - 1) as f64;
    let geometry = h2_geometry(
        options.box_size_bohr,
        options.muffin_tin_radius,
        radial_log_increment,
    );
    let structure = Structure::new(geometry)?;
    let field_layout = RegionalFieldLayout::from_g_cutoff(
        &structure,
        InverseBohr(options.field_g_cutoff),
        ATOMIC_START_FIELD_L_MAX,
    )?;
    let start = materialize_atomic_start(AtomicStartRequest {
        meta: checkpoint_meta(options.speed_of_light),
        structure,
        field_layout,
        exchange_correlation: ScfExchangeCorrelation {
            functional: XcFunctional::LdaPw92,
            noncollinear_route: DftNoncollinearXcRoute::LocalSpinFrame,
            interstitial_grid: options.xc_grid,
        },
        free_atom_scf: FreeAtomScfSpec {
            speed_of_light: options.speed_of_light,
            mesh: ExponentialMesh::new(
                Bohr(FREE_ATOM_FIRST_BOHR),
                FREE_ATOM_LOG_INCREMENT,
                FREE_ATOM_POINTS,
            )?,
            mixing: FREE_ATOM_MIXING,
            potential_tolerance: FREE_ATOM_POTENTIAL_TOLERANCE,
            tail_tolerance: FREE_ATOM_TAIL_TOLERANCE,
            max_iterations: FREE_ATOM_MAX_ITERATIONS,
        },
        angular_grid: AngularGrid::fibonacci(302)?,
    })?;

    let atomic_start_path = options.output_directory.join("h2.atomic-start.toml");
    fs::write(
        &atomic_start_path,
        checkpoint_file_to_toml(&CheckpointFile::V2(start.checkpoint))?,
    )?;
    let input_path = options.output_directory.join("h2.input.toml");
    fs::write(
        &input_path,
        input_to_toml(&scf_input(options.orbital_g_cutoff, options.xc_grid))?,
    )?;

    println!("system=H2 bond_bohr={BOND_LENGTH_BOHR:.6} electron_count={ELECTRON_COUNT:.1}");
    println!(
        "route=scalar-koelling-harmon exchange_correlation=lda-pw92 cores=none box_bohr={:.6} rmt_bohr={:.6} orbital_g_bohr_inverse={:.6} orbital_l_max={SCF_ORBITAL_L_MAX} field_g_bohr_inverse={:.6} field_l_max={ATOMIC_START_FIELD_L_MAX}",
        options.box_size_bohr,
        options.muffin_tin_radius,
        options.orbital_g_cutoff,
        options.field_g_cutoff,
    );
    println!(
        "atomic_start_checkpoint={} charge_target={:.16e} charge_represented={:.16e} charge_error={:.3e} normalization_scale={:.16e}",
        atomic_start_path.display(),
        start.charge_closure.target_electron_count,
        start.charge_closure.represented_electron_count,
        start.charge_closure.represented_electron_count
            - start.charge_closure.target_electron_count,
        start.charge_closure.normalization_scale,
    );
    println!("input={}", input_path.display());

    let result = run_dft_scf(&input_path)?;
    for diagnostic in &result.state.diagnostics {
        let energy_change = diagnostic.energy_change.map_or_else(
            || "none".to_owned(),
            |value| format!("{:.16e}", value.get()),
        );
        println!(
            "scf_iteration={} total_energy_ha={:.16e} energy_change_ha={} density_rms={:.16e}",
            diagnostic.iteration,
            diagnostic.energy.total.get(),
            energy_change,
            diagnostic.density_rms,
        );
    }

    let restart_path = options.output_directory.join("h2.restart.toml");
    fs::write(
        &restart_path,
        checkpoint_file_to_toml(&CheckpointFile::V2(result.checkpoint))?,
    )?;
    println!(
        "final_electron_count={:.16e}",
        muffintin_dft::electron_count(&result.state.density)?
    );
    let final_diagnostic = result
        .state
        .diagnostics
        .last()
        .expect("a converged SCF state has at least one diagnostic");
    println!(
        "scf_final iterations={} total_energy_ha={:.16e} energy_change_ha={:?} density_rms={:.16e} restart_checkpoint={}",
        result.state.iterations(),
        result.state.energy.total.get(),
        final_diagnostic.energy_change.map(|value| value.get()),
        final_diagnostic.density_rms,
        restart_path.display(),
    );
    let energy = result.state.energy;
    println!(
        "energy_terms_ha band={:.16e} core_eigenvalues={:.16e} madelung={:.16e} coulomb={:.16e} exchange_correlation={:.16e} exchange_correlation_potential={:.16e} occupation_correction={:.16e} total={:.16e}",
        energy.band.get(),
        energy.core_eigenvalues.get(),
        energy.madelung.get(),
        energy.coulomb.get(),
        energy.exchange_correlation.get(),
        energy.exchange_correlation_potential.get(),
        energy.occupation.correction().get(),
        energy.total.get(),
    );
    Ok(())
}

fn h2_geometry(
    box_size_bohr: f64,
    muffin_tin_radius: f64,
    radial_log_increment: f64,
) -> GeometryV2 {
    let half_separation_fraction = BOND_LENGTH_BOHR / (2.0 * box_size_bohr);
    let sites = [
        ("H-1", [0.5 - half_separation_fraction, 0.5, 0.5]),
        ("H-2", [0.5 + half_separation_fraction, 0.5, 0.5]),
    ];
    GeometryV2 {
        lattice: LatticeV1 {
            unit: LengthUnit::Bohr,
            vectors: [
                [box_size_bohr, 0.0, 0.0],
                [0.0, box_size_bohr, 0.0],
                [0.0, 0.0, box_size_bohr],
            ],
        },
        sites: sites
            .iter()
            .map(|(id, fractional_position)| SiteV2 {
                id: (*id).to_owned(),
                atomic_number: 1,
                fractional_position: *fractional_position,
                muffin_tin_radius_unit: LengthUnit::Bohr,
                muffin_tin_radius,
            })
            .collect(),
        radial_basis: sites
            .iter()
            .map(|(id, _)| SiteRadialBasisV2 {
                site_id: (*id).to_owned(),
                spin: RadialBasisSpinV2::Scalar,
                mesh: ExponentialMeshSpec {
                    radius_unit: LengthUnit::Bohr,
                    first: RADIAL_FIRST_BOHR,
                    log_increment: radial_log_increment,
                    point_count: RADIAL_POINTS,
                    last: muffin_tin_radius,
                    consistency_tolerance: 1.0e-12,
                },
                radial_equation: RadialEquationTag::ScalarKoellingHarmon,
                linearization: LinearizationV1 {
                    energy_unit: EnergyUnit::Hartree,
                    linearization_energies: Vec::new(),
                    local_orbital_energies: Vec::new(),
                },
            })
            .collect(),
    }
}

fn checkpoint_meta(speed_of_light: f64) -> CheckpointMeta {
    CheckpointMeta {
        speed_of_light,
        title: "H2 Gamma scalar Koelling-Harmon LDA-PW92".to_owned(),
        producer: "libmuffintin-runtime h2_dft example".to_owned(),
        producer_version: Some(env!("CARGO_PKG_VERSION").to_owned()),
        energy_zero: "periodic molecule-in-box electrostatic reference".to_owned(),
        potential_convention: PotentialConventionV1 {
            angular_basis: AngularBasis::ComplexCondonShortley,
            radial_quantity: PotentialRadialQuantityV1::Potential,
            spherical_channel: SphericalChannelConvention::PhysicalValue,
        },
        annotations: BTreeMap::from([
            ("system".to_owned(), "H2 molecule-in-box".to_owned()),
            ("relativity".to_owned(), "scalar-koelling-harmon".to_owned()),
            ("core-treatment".to_owned(), "none".to_owned()),
        ]),
    }
}

fn scf_input(orbital_g_cutoff: f64, xc_grid: Option<[usize; 3]>) -> Input {
    let mut tasks = BTreeMap::new();
    tasks.insert(
        "scf".to_owned(),
        Task::DftScf {
            source: None,
            electron_count: ELECTRON_COUNT,
            k_mesh: KMesh {
                mesh: [1, 1, 1],
                shift: [0.0; 3],
            },
            symmetry: Symmetry::default(),
            basis: Basis {
                l_max: SCF_ORBITAL_L_MAX,
                energy_generator: Some(ChannelEnergyGenerator::Atomic),
                recipe: None,
                envelope: BasisEnvelope {
                    kind: BasisEnvelopeKind::PlaneWave,
                    g_cutoff: Some(orbital_g_cutoff),
                    energy_cutoff: None,
                },
                channels: BTreeMap::from([(
                    "H".to_owned(),
                    BTreeMap::from([(
                        muffintin::ChannelTreatment::Valence,
                        vec![
                            "1s@-0.4".to_owned(),
                            "2p@-0.4".to_owned(),
                            "3d@-0.4".to_owned(),
                            "4f@-0.4".to_owned(),
                            "5g@-0.4".to_owned(),
                            "6h@-0.4".to_owned(),
                            "7i@-0.4".to_owned(),
                            "8k@-0.4".to_owned(),
                            "9l@-0.4".to_owned(),
                        ],
                    )]),
                )]),
            },
            occupations: Occupations::FermiDirac {
                temperature: FERMI_TEMPERATURE_HARTREE,
            },
            xc: ExchangeCorrelation::LdaPw92 {
                noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
                interstitial_grid: xc_grid,
            },
            mixing: Mixing::PulayAnderson {
                beta: MIXING_BETA,
                history: MIXING_HISTORY,
            },
            relativity: Relativity::Scalar {},
            convergence: Convergence {
                energy_tolerance: SCF_ENERGY_TOLERANCE_HARTREE,
                density_tolerance: SCF_DENSITY_TOLERANCE,
                max_iterations: SCF_MAX_ITERATIONS,
            },
        },
    );
    Input::new(
        PathBuf::from("h2.atomic-start.toml"),
        Workflow {
            tasks: vec!["scf".to_owned()],
        },
        tasks,
    )
}
