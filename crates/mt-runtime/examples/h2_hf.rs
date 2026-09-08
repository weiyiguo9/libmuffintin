use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use muffintin::{
    AtomicStartRequest, CheckpointPhysics, FockMixing, GammaValenceHfSpec, RegionalFieldLayout,
    Structure, materialize_atomic_start, run_gamma_valence_hf,
};
use muffintin_core::{AngularGrid, Bohr, ExponentialMesh, Hartree, InverseBohr};
use muffintin_coulomb::{CoulombRequest, DEFAULT_LEXP, MAX_LEXP};
use muffintin_dft::{
    LinearizationEnergyGenerator, NoncollinearXcRoute, ScfBasis, ScfChannelIdentity,
    ScfChannelProvenance, ScfChannelRecipe, ScfChannelTreatment, ScfConfig, ScfConvergence,
    ScfCoreSite, ScfExchangeCorrelation, ScfKMesh, ScfKReduction, ScfMixing, ScfOccupations,
    ScfRelativity, XcFunctional, electron_count,
};
use muffintin_io::{
    AngularBasis, CheckpointFile, CheckpointMeta, EnergyUnit, ExponentialMeshSpec, GeometryV2,
    LatticeV1, LengthUnit, LinearizationV1, PotentialConventionV1, PotentialRadialQuantityV1,
    RadialBasisSpinV2, RadialEquationTag, SiteRadialBasisV2, SiteV2, SphericalChannelConvention,
    checkpoint_file_to_toml,
};
use muffintin_prodbasis::mpb::DEFAULT_TOLERANCE;

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
const FIELD_L_MAX: u32 = 8;
const ORBITAL_L_MAX: u32 = 8;
const ELECTRON_COUNT: f64 = 2.0;
const FERMI_TEMPERATURE_HARTREE: f64 = 0.001;
const OUTER_MIXING_BETA: f64 = 0.4;
const OUTER_MIXING_HISTORY: usize = 6;
const OUTER_ENERGY_TOLERANCE_HARTREE: f64 = 1.0e-8;
const OUTER_DENSITY_TOLERANCE: f64 = 1.0e-7;
const OUTER_MAX_ITERATIONS: usize = 80;
const FOCK_MAX_ITERATIONS: usize = 128;
// The valence identity residual tracks the Fock feedback tolerance with a
// prefactor of 16 to 34 (evd-0006, evd-0007); 1e-10 clears the driver's 2e-8 gate.
const FOCK_DENSITY_TOLERANCE: f64 = 1.0e-9;
const FOCK_FEEDBACK_TOLERANCE_HARTREE: f64 = 1.0e-10;
const FOCK_DIIS_HISTORY: usize = 8;
const FOCK_DIIS_STARTUP_STEPS: usize = 2;
const FOCK_DIIS_DAMPING: f64 = 0.5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExchangeCoulomb {
    PeriodicFiniteBody,
    SpencerAlaviSphere,
    SmoothedSpencerAlaviSphere,
}

impl ExchangeCoulomb {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "periodic-finite-body" => Ok(Self::PeriodicFiniteBody),
            "spencer-alavi-sphere" => Ok(Self::SpencerAlaviSphere),
            "smoothed-spencer-alavi-sphere" => Ok(Self::SmoothedSpencerAlaviSphere),
            _ => Err(format!(
                "--exchange-coulomb must be periodic-finite-body, spencer-alavi-sphere, or smoothed-spencer-alavi-sphere, got {value:?}"
            )
            .into()),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::PeriodicFiniteBody => "periodic-finite-body",
            Self::SpencerAlaviSphere => "spencer-alavi-sphere",
            Self::SmoothedSpencerAlaviSphere => "smoothed-spencer-alavi-sphere",
        }
    }
}

#[derive(Debug)]
struct Options {
    output_directory: PathBuf,
    box_size_bohr: f64,
    orbital_g: f64,
    field_g: f64,
    product_g: f64,
    product_l_max: u32,
    overlap_tolerance: f64,
    exchange_coulomb: ExchangeCoulomb,
    fock_fourier_g: Option<f64>,
    fock_smoothing_omega: Option<f64>,
    lexp: u32,
    speed_of_light: f64,
    muffin_tin_radius: f64,
    max_fock_iterations: usize,
    verbosity: muffintin::HfVerbosity,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            output_directory: PathBuf::from("h2-hf-output"),
            box_size_bohr: 8.0,
            orbital_g: 4.0,
            field_g: 12.0,
            product_g: 4.0,
            product_l_max: 2,
            overlap_tolerance: DEFAULT_TOLERANCE,
            exchange_coulomb: ExchangeCoulomb::PeriodicFiniteBody,
            fock_fourier_g: None,
            fock_smoothing_omega: None,
            lexp: DEFAULT_LEXP,
            speed_of_light: 137_035.989_5,
            muffin_tin_radius: DEFAULT_MUFFIN_TIN_RADIUS_BOHR,
            max_fock_iterations: FOCK_MAX_ITERATIONS,
            verbosity: muffintin::HfVerbosity::Quiet,
        }
    }
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut options = Self::default();
        let mut arguments = env::args().skip(1).peekable();
        if arguments
            .peek()
            .is_some_and(|value| !value.starts_with("--"))
        {
            options.output_directory = PathBuf::from(arguments.next().expect("peeked argument"));
        }
        while let Some(name) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| format!("missing value after {name}"))?;
            match name.as_str() {
                "--box" => options.box_size_bohr = value.parse()?,
                "--orbital-g" => options.orbital_g = value.parse()?,
                "--field-g" => options.field_g = value.parse()?,
                "--product-g" => options.product_g = value.parse()?,
                "--product-lmax" => options.product_l_max = value.parse()?,
                "--overlap-tolerance" => options.overlap_tolerance = value.parse()?,
                "--exchange-coulomb" => options.exchange_coulomb = ExchangeCoulomb::parse(&value)?,
                "--fock-fourier-g" => options.fock_fourier_g = Some(value.parse()?),
                "--fock-smoothing-omega" => options.fock_smoothing_omega = Some(value.parse()?),
                "--lexp" => options.lexp = value.parse()?,
                "--speed-of-light" => options.speed_of_light = value.parse()?,
                "--rmt" => options.muffin_tin_radius = value.parse()?,
                "--fock-max-iterations" => options.max_fock_iterations = value.parse()?,
                "--verbosity" => {
                    options.verbosity = match value.as_str() {
                        "0" => muffintin::HfVerbosity::Quiet,
                        "1" => muffintin::HfVerbosity::Progress,
                        "2" => muffintin::HfVerbosity::Timings,
                        _ => return Err("--verbosity must be 0, 1, or 2".into()),
                    };
                }
                _ => return Err(format!("unknown option {name:?}").into()),
            }
        }
        options.validate()?;
        Ok(options)
    }

    fn validate(&self) -> Result<(), Box<dyn Error>> {
        for (name, value) in [
            ("--box", self.box_size_bohr),
            ("--orbital-g", self.orbital_g),
            ("--field-g", self.field_g),
            ("--product-g", self.product_g),
            ("--overlap-tolerance", self.overlap_tolerance),
            ("--speed-of-light", self.speed_of_light),
            ("--rmt", self.muffin_tin_radius),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive, got {value}").into());
            }
        }
        if let Some(value) = self.fock_fourier_g
            && (!value.is_finite() || value <= 0.0)
        {
            return Err(
                format!("--fock-fourier-g must be finite and positive, got {value}").into(),
            );
        }
        match (self.exchange_coulomb, self.fock_smoothing_omega) {
            (ExchangeCoulomb::SmoothedSpencerAlaviSphere, Some(value))
                if value.is_finite() && value > 0.0 => {}
            (ExchangeCoulomb::SmoothedSpencerAlaviSphere, _) => {
                return Err("smoothed-spencer-alavi-sphere requires --fock-smoothing-omega".into());
            }
            (_, Some(_)) => {
                return Err(
                    "--fock-smoothing-omega applies only to smoothed-spencer-alavi-sphere".into(),
                );
            }
            (_, None) => {}
        }
        if self.product_l_max > self.lexp || self.lexp > MAX_LEXP {
            return Err(format!(
                "angular cutoffs must satisfy --product-lmax <= --lexp <= {MAX_LEXP}"
            )
            .into());
        }
        if self.max_fock_iterations < 2 {
            return Err("--fock-max-iterations must be at least 2".into());
        }
        if 2.0 * self.muffin_tin_radius >= BOND_LENGTH_BOHR {
            return Err("muffin-tin spheres overlap".into());
        }
        if self.box_size_bohr <= BOND_LENGTH_BOHR + 2.0 * self.muffin_tin_radius {
            return Err("box is too small for the molecule and muffin-tin spheres".into());
        }
        Ok(())
    }

    fn fock_fourier_g(&self) -> f64 {
        self.fock_fourier_g.unwrap_or(self.product_g)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let options = Options::parse()?;
    muffintin::set_hf_verbosity(options.verbosity);
    fs::create_dir_all(&options.output_directory)?;
    let radial_log_increment =
        (options.muffin_tin_radius / RADIAL_FIRST_BOHR).ln() / (RADIAL_POINTS - 1) as f64;
    let geometry = h2_geometry(
        options.box_size_bohr,
        options.muffin_tin_radius,
        radial_log_increment,
    );
    let structure = Structure::new(geometry)?;
    let field_layout =
        RegionalFieldLayout::from_g_cutoff(&structure, InverseBohr(options.field_g), FIELD_L_MAX)?;
    let config = scf_config(&options);
    let start = materialize_atomic_start(AtomicStartRequest {
        meta: checkpoint_meta(options.speed_of_light),
        structure,
        field_layout,
        exchange_correlation: config.exchange_correlation,
        free_atom_scf: muffintin_dft::FreeAtomScfSpec {
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
        checkpoint_file_to_toml(&CheckpointFile::V2(start.checkpoint.clone()))?,
    )?;
    println!(
        "system=H2 bond_bohr={BOND_LENGTH_BOHR:.6} electron_count={ELECTRON_COUNT:.1} route=spinor-first-variation cores=none box_bohr={:.6} rmt_bohr={:.6} orbital_g_bohr_inverse={:.6} field_g_bohr_inverse={:.6} product_g_bohr_inverse={:.6} product_lmax={} overlap_tolerance={:.6e} exchange_coulomb={} fock_fourier_g_bohr_inverse={:.6} fock_smoothing_omega_bohr_inverse={} lexp={} speed_of_light_au={:.10}",
        options.box_size_bohr,
        options.muffin_tin_radius,
        options.orbital_g,
        options.field_g,
        options.product_g,
        options.product_l_max,
        options.overlap_tolerance,
        options.exchange_coulomb.as_str(),
        options.fock_fourier_g(),
        options
            .fock_smoothing_omega
            .map_or_else(|| "none".to_owned(), |value| format!("{value:.6}")),
        options.lexp,
        options.speed_of_light,
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

    let spec = GammaValenceHfSpec {
        config,
        product_l_max: options.product_l_max,
        product_g_max: InverseBohr(options.product_g),
        overlap_tolerance: options.overlap_tolerance,
        coulomb: exchange_coulomb_request(&options)?,
        max_fock_iterations: options.max_fock_iterations,
        fock_density_tolerance: FOCK_DENSITY_TOLERANCE,
        fock_feedback_tolerance: Hartree(FOCK_FEEDBACK_TOLERANCE_HARTREE),
        fock_mixing: FockMixing::CommutatorDiis {
            history: FOCK_DIIS_HISTORY,
            startup_steps: FOCK_DIIS_STARTUP_STEPS,
            damping: FOCK_DIIS_DAMPING,
        },
    };
    let mut physics = CheckpointPhysics::new(&start.checkpoint)?;
    let result = run_gamma_valence_hf(&mut physics, &spec)?;
    for diagnostic in &result.diagnostics {
        println!(
            "hf_iteration={} fock_iterations={} exchange_rebuilds={} density_rms={:.16e} electron_count={:.16e} h0_ha={:.16e} electron_hartree_ha={:.16e} electron_nuclear_ha={:.16e} nuclear_hartree_ha={:.16e} exchange_ha={:.16e} total_ha={:.16e} exchange_identity_ha={:.16e} eigenvalue_identity_ha={:.16e} total_identity_ha={:.16e}",
            diagnostic.iteration,
            diagnostic.fock_iterations,
            diagnostic.exchange_rebuilds,
            diagnostic.regional_density_rms,
            diagnostic.density_electron_count,
            diagnostic.h0_expectation.get(),
            diagnostic.electron_hartree.get(),
            diagnostic.electron_nuclear.get(),
            diagnostic.nuclear_nuclear.get(),
            diagnostic.exchange_energy.get(),
            diagnostic.total_energy.get(),
            diagnostic.exchange_energy_identity_residual,
            diagnostic.eigenvalue_identity_residual,
            diagnostic.total_energy_identity_residual,
        );
    }
    let final_diagnostic = result
        .diagnostics
        .last()
        .expect("a converged HF state has at least one diagnostic");
    let band_energy = result
        .orbital_energies
        .iter()
        .zip(&result.occupations)
        .zip(&result.k_weights)
        .map(|((energies, occupations), weight)| {
            weight
                * energies
                    .iter()
                    .zip(occupations)
                    .map(|(energy, occupation)| energy.get() * occupation)
                    .sum::<f64>()
        })
        .sum::<f64>();
    let homo = result
        .orbital_energies
        .iter()
        .zip(&result.occupations)
        .flat_map(|(energies, occupations)| energies.iter().zip(occupations))
        .filter(|(_, occupation)| **occupation >= 0.5)
        .map(|(energy, _)| energy.get())
        .reduce(f64::max)
        .ok_or("no occupied HF orbital")?;
    let occupation_correction = result.total_energy.get() - result.h0_expectation.get()
        + result.electron_hartree.get()
        - result.nuclear_nuclear.get()
        - result.exchange_energy.get();
    let hartree_exchange =
        (result.exchange_energy.get() + 0.5 * result.electron_hartree.get()).abs();
    let final_electron_count = electron_count(&result.density)?;
    let wall_seconds = started.elapsed().as_secs_f64();
    println!(
        "hf_final outer_iterations={} fock_iterations={} exchange_rebuilds={} electron_count={:.16e} homo_ha={:.16e} wall_s={:.6}",
        result.diagnostics.len(),
        final_diagnostic.fock_iterations,
        result.exchange_rebuilds,
        final_electron_count,
        homo,
        wall_seconds,
    );
    println!(
        "hf_energy_terms_ha h0={:.16e} electron_hartree={:.16e} nuclear_hartree={:.16e} exchange={:.16e} occupation_correction={:.16e} band={:.16e} total={:.16e}",
        result.h0_expectation.get(),
        result.electron_hartree.get(),
        result.nuclear_nuclear.get(),
        result.exchange_energy.get(),
        occupation_correction,
        band_energy,
        result.total_energy.get(),
    );
    println!(
        "hf_identity_ha exchange={:.16e} eigenvalue={:.16e} total={:.16e} hartree_exchange={:.16e}",
        final_diagnostic.exchange_energy_identity_residual,
        final_diagnostic.eigenvalue_identity_residual,
        final_diagnostic.total_energy_identity_residual,
        hartree_exchange,
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
                radial_equation: RadialEquationTag::FullyRelativisticDirac,
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
        title: "H2 Gamma spinor-first valence HF".to_owned(),
        producer: "libmuffintin-runtime h2_hf example".to_owned(),
        producer_version: Some(env!("CARGO_PKG_VERSION").to_owned()),
        energy_zero: "periodic molecule-in-box electrostatic reference".to_owned(),
        potential_convention: PotentialConventionV1 {
            angular_basis: AngularBasis::ComplexCondonShortley,
            radial_quantity: PotentialRadialQuantityV1::Potential,
            spherical_channel: SphericalChannelConvention::PhysicalValue,
        },
        annotations: BTreeMap::from([
            ("system".to_owned(), "H2 molecule-in-box".to_owned()),
            ("relativity".to_owned(), "spinor-first-variation".to_owned()),
            ("core-treatment".to_owned(), "none".to_owned()),
        ]),
    }
}

fn scf_config(options: &Options) -> ScfConfig {
    let sites = ["H-1", "H-2"];
    let channels = sites
        .iter()
        .flat_map(|site| {
            (0..=ORBITAL_L_MAX).map(move |l| ScfChannelRecipe {
                site: (*site).to_owned(),
                identity: ScfChannelIdentity::ScalarL { n: l + 1, l },
                treatment: ScfChannelTreatment::Valence,
                derivative_order: 0,
                generator: LinearizationEnergyGenerator::Explicit,
                seed: Some(Hartree(-0.4)),
                provenance: ScfChannelProvenance::TaskDefault,
            })
        })
        .collect();
    ScfConfig {
        electron_count: ELECTRON_COUNT,
        k_mesh: ScfKMesh {
            divisions: [1, 1, 1],
            shift: [0.0; 3],
            reduction: ScfKReduction::Full,
        },
        basis: ScfBasis {
            plane_wave_cutoff: InverseBohr(options.orbital_g),
            l_max: ORBITAL_L_MAX,
            channels,
            resolved_channels: Vec::new(),
        },
        occupations: ScfOccupations::FermiDirac {
            temperature: Hartree(FERMI_TEMPERATURE_HARTREE),
        },
        exchange_correlation: ScfExchangeCorrelation {
            functional: XcFunctional::LdaPw92,
            noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
            interstitial_grid: None,
        },
        mixing: ScfMixing::PulayAnderson {
            alpha: OUTER_MIXING_BETA,
            history: OUTER_MIXING_HISTORY,
        },
        relativity: ScfRelativity::SpinorFirstVariation,
        convergence: ScfConvergence {
            energy_tolerance: Hartree(OUTER_ENERGY_TOLERANCE_HARTREE),
            density_tolerance: OUTER_DENSITY_TOLERANCE,
            max_iterations: OUTER_MAX_ITERATIONS,
        },
        core_sites: sites
            .iter()
            .map(|site| ScfCoreSite {
                id: (*site).to_owned(),
                states: Vec::new(),
            })
            .collect(),
    }
}

fn exchange_coulomb_request(options: &Options) -> Result<CoulombRequest, Box<dyn Error>> {
    let request = CoulombRequest::cubic(options.box_size_bohr, options.lexp)?;
    match options.exchange_coulomb {
        ExchangeCoulomb::PeriodicFiniteBody => Ok(request),
        ExchangeCoulomb::SpencerAlaviSphere => {
            Ok(request.with_spencer_alavi_sphere(1, InverseBohr(options.fock_fourier_g()))?)
        }
        ExchangeCoulomb::SmoothedSpencerAlaviSphere => Ok(request
            .with_smoothed_spencer_alavi_sphere(
                1,
                InverseBohr(options.fock_fourier_g()),
                InverseBohr(
                    options
                        .fock_smoothing_omega
                        .expect("validated smoothing omega"),
                ),
            )?),
    }
}
