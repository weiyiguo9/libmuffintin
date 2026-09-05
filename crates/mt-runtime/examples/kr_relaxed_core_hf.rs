use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::f64::consts::PI;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use muffintin::{
    AtomicStartRequest, CheckpointPhysics, FockMixing, GammaExchangeTreatment, KhSocCoreTreatment,
    KhSocHartreeUpdate, KhSocValenceHfIterationDiagnostic, KhSocValenceHfResult,
    KhSocValenceHfSpec, RegionalFieldLayout, RelaxedCoreHfIterationDiagnostic, RelaxedCoreHfResult,
    RelaxedCoreHfSpec, Structure, checkpoint_v2_from_regional_state, materialize_atomic_start,
    run_gamma_kh_soc_valence_hf, run_gamma_relaxed_core_hf,
};
use muffintin_core::{
    AngularGrid, Bohr, ExponentialMesh, Hartree, InverseBohr, SpinProjection, lm_count,
};
use muffintin_coulomb::{CoulombRequest, DEFAULT_LEXP, MAX_LEXP};
use muffintin_dft::{
    AtomicChannelTreatment, AtomicNumber, CheckpointKPointSolution, CoreFixedPotentialSpec,
    CoreShellOccupations, FirstVariationWindow, LinearizationEnergyGenerator, NoncollinearXcRoute,
    ScfBasis, ScfChannelIdentity, ScfChannelProvenance, ScfChannelRecipe, ScfChannelTreatment,
    ScfConfig, ScfConvergence, ScfCoreSite, ScfCoreState, ScfExchangeCorrelation, ScfKMesh,
    ScfKReduction, ScfMixing, ScfOccupations, ScfRelativity, XcFunctional, electron_count,
    fleur_default_atomic_configuration,
};
use muffintin_io::{
    AngularBasis, CheckpointFile, CheckpointMeta, EnergyUnit, ExponentialMeshSpec, GeometryV2,
    LatticeV1, LengthUnit, LinearizationV1, PotentialConventionV1, PotentialRadialQuantityV1,
    RadialBasisSpinV2, RadialEquationTag, SiteRadialBasisV2, SiteV2, SphericalChannelConvention,
    checkpoint_file_to_toml,
};
use muffintin_operators::CompiledSiteProjection;
use muffintin_prodbasis::mpb::DEFAULT_TOLERANCE;
use num_complex::Complex64;
use serde::Serialize;

#[path = "kr_relaxed_core_hf/frozen_sra.rs"]
mod frozen_sra;

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
const OUTER_MIXING_HISTORY: usize = 8;
const MAX_FOCK_ITERATIONS: usize = 128;
const LOOSE_TOLERANCE: f64 = 1.0e100;
const FOCK_DENSITY_TOLERANCE: f64 = 1.0e-5;
const FOCK_FEEDBACK_TOLERANCE_HARTREE: f64 = 1.0e-5;
const FOCK_COMMUTATOR_TOLERANCE_HARTREE: f64 = 1.0e-5;
const FOCK_DIIS_HISTORY: usize = 8;
const FOCK_DIIS_LEVEL_SHIFT_HARTREE: f64 = 0.05;
const FOCK_DIIS_STARTUP_STEPS: usize = 2;
const FOCK_DIIS_DAMPING: f64 = 0.5;
const SECTOR_NUMERICAL_TOLERANCE_HARTREE: f64 = 1.0e-8;
const MAXIMUM_CORE_SHELL_SPILL: f64 = 1.0;
const HOMO_OCCUPATION_THRESHOLD: f64 = 0.5;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
enum HdloSelection {
    None,
    All,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum FockMixerSelection {
    Linear,
    Pulay,
    Cdiis,
    QuasiNewtonCdiis,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum RelativitySelection {
    SpinorFirst,
    SpinorFrozen,
    KhSoc,
}

impl RelativitySelection {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "spinor-first" => Ok(Self::SpinorFirst),
            "spinor-frozen" => Ok(Self::SpinorFrozen),
            "kh-soc" => Ok(Self::KhSoc),
            _ => Err(invalid_input(format!(
                "--relativity must be spinor-first, spinor-frozen, or kh-soc, got {value:?}"
            ))),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::SpinorFirst => {
                "spinor-first-variation with fully-relativistic Dirac radial equation"
            }
            Self::SpinorFrozen => {
                "coupled frozen-core SRA first variation (4c MT, 2c interstitial)"
            }
            Self::KhSoc => "self-consistent Pauli-spinor Koelling-Harmon plus SOC HF",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum OuterMixerSelection {
    Linear,
    Pulay,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum ExchangeCoulombSelection {
    PeriodicFiniteBody,
    SpencerAlaviSphere,
    SmoothedSpencerAlaviSphere,
}

impl ExchangeCoulombSelection {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "periodic-finite-body" => Ok(Self::PeriodicFiniteBody),
            "spencer-alavi-sphere" => Ok(Self::SpencerAlaviSphere),
            "smoothed-spencer-alavi-sphere" => Ok(Self::SmoothedSpencerAlaviSphere),
            _ => Err(invalid_input(format!(
                "--exchange-coulomb must be periodic-finite-body, spencer-alavi-sphere, or smoothed-spencer-alavi-sphere, got {value:?}"
            ))),
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

impl OuterMixerSelection {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "linear" => Ok(Self::Linear),
            "pulay" => Ok(Self::Pulay),
            _ => Err(invalid_input(format!(
                "--outer-mixing must be linear or pulay, got {value:?}"
            ))),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear-total-density",
            Self::Pulay => "pulay-anderson-total-density",
        }
    }

    const fn build(self, alpha: f64, history: usize) -> ScfMixing {
        match self {
            Self::Linear => ScfMixing::Linear { alpha },
            Self::Pulay => ScfMixing::PulayAnderson { alpha, history },
        }
    }
}

impl FockMixerSelection {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "linear" => Ok(Self::Linear),
            "pulay" => Ok(Self::Pulay),
            "cdiis" => Ok(Self::Cdiis),
            "quasi-newton-cdiis" => Ok(Self::QuasiNewtonCdiis),
            _ => Err(invalid_input(format!(
                "--fock-mixing must be linear, pulay, cdiis, or quasi-newton-cdiis, got {value:?}"
            ))),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear-global-feedback",
            Self::Pulay => "pulay-anderson-global-feedback",
            Self::Cdiis => "commutator-cdiis-global-feedback",
            Self::QuasiNewtonCdiis => "quasi-newton-commutator-cdiis-global-feedback",
        }
    }

    const fn build(
        self,
        alpha: f64,
        history: usize,
        level_shift: f64,
        startup_steps: usize,
        damping: f64,
    ) -> FockMixing {
        match self {
            Self::Linear => FockMixing::Linear { alpha },
            Self::Pulay => FockMixing::PulayAnderson { alpha, history },
            Self::Cdiis => FockMixing::CommutatorDiis {
                history,
                startup_steps,
                damping,
            },
            Self::QuasiNewtonCdiis => FockMixing::QuasiNewtonDiis {
                history,
                level_shift: Hartree(level_shift),
                startup_steps,
                damping,
            },
        }
    }
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
    verbosity: muffintin::HfVerbosity,
    relativity: RelativitySelection,
    soc_bands: usize,
    kh_hartree_update: KhSocHartreeUpdate,
    box_size: f64,
    orbital_g: f64,
    field_g: f64,
    orbital_l_max: u32,
    product_g: f64,
    product_l_max: u32,
    lexp: u32,
    exchange_coulomb: ExchangeCoulombSelection,
    fock_fourier_g: f64,
    fock_smoothing_omega: Option<f64>,
    muffin_tin_radius: f64,
    radial_points: usize,
    hdlo: HdloSelection,
    temperature: f64,
    outer_max_iterations: usize,
    outer_energy_tolerance: f64,
    outer_density_tolerance: f64,
    outer_mixing: OuterMixerSelection,
    outer_mixing_alpha: f64,
    outer_mixing_history: usize,
    core_max_iterations: usize,
    core_energy_tolerance: f64,
    core_radial_tolerance: f64,
    max_fock_iterations: usize,
    fock_density_tolerance: f64,
    fock_feedback_tolerance: f64,
    fock_commutator_tolerance: f64,
    spinor_virtual_level_shift: f64,
    scalar_fock_mixing: FockMixerSelection,
    fock_mixing: FockMixerSelection,
    fock_mixing_alpha: f64,
    fock_diis_history: usize,
    fock_diis_level_shift: f64,
    fock_diis_startup_steps: usize,
    fock_diis_damping: f64,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            out: PathBuf::from("kr-relaxed-core-hf-p0"),
            verbosity: muffintin::HfVerbosity::Progress,
            relativity: RelativitySelection::SpinorFirst,
            soc_bands: 7,
            kh_hartree_update: KhSocHartreeUpdate::OuterDensity,
            box_size: 8.0,
            orbital_g: 1.0,
            field_g: 4.5,
            orbital_l_max: 1,
            product_g: 1.0,
            product_l_max: 2,
            lexp: DEFAULT_LEXP,
            exchange_coulomb: ExchangeCoulombSelection::PeriodicFiniteBody,
            fock_fourier_g: 4.5,
            fock_smoothing_omega: None,
            muffin_tin_radius: 2.0,
            radial_points: 2_401,
            hdlo: HdloSelection::None,
            temperature: 0.02,
            outer_max_iterations: OUTER_MAX_ITERATIONS,
            outer_energy_tolerance: LOOSE_TOLERANCE,
            outer_density_tolerance: LOOSE_TOLERANCE,
            outer_mixing: OuterMixerSelection::Pulay,
            outer_mixing_alpha: OUTER_MIXING_ALPHA,
            outer_mixing_history: OUTER_MIXING_HISTORY,
            core_max_iterations: CORE_MAX_ITERATIONS,
            core_energy_tolerance: LOOSE_TOLERANCE,
            core_radial_tolerance: LOOSE_TOLERANCE,
            max_fock_iterations: MAX_FOCK_ITERATIONS,
            fock_density_tolerance: FOCK_DENSITY_TOLERANCE,
            fock_feedback_tolerance: FOCK_FEEDBACK_TOLERANCE_HARTREE,
            fock_commutator_tolerance: FOCK_COMMUTATOR_TOLERANCE_HARTREE,
            spinor_virtual_level_shift: 0.0,
            scalar_fock_mixing: FockMixerSelection::Cdiis,
            fock_mixing: FockMixerSelection::Cdiis,
            fock_mixing_alpha: 0.5,
            fock_diis_history: FOCK_DIIS_HISTORY,
            fock_diis_level_shift: FOCK_DIIS_LEVEL_SHIFT_HARTREE,
            fock_diis_startup_steps: FOCK_DIIS_STARTUP_STEPS,
            fock_diis_damping: FOCK_DIIS_DAMPING,
        }
    }
}

impl Cli {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut cli = Self::default();
        let mut smoke_configuration = true;
        let mut arguments = env::args().skip(1);
        while let Some(name) = arguments.next() {
            smoke_configuration &= matches!(name.as_str(), "--out" | "--verbosity");
            let value = arguments
                .next()
                .ok_or_else(|| invalid_input(format!("missing value after {name}")))?;
            match name.as_str() {
                "--out" => cli.out = PathBuf::from(value),
                "--verbosity" => {
                    cli.verbosity = match value.as_str() {
                        "0" => muffintin::HfVerbosity::Quiet,
                        "1" => muffintin::HfVerbosity::Progress,
                        "2" => muffintin::HfVerbosity::Timings,
                        _ => return Err(invalid_input("--verbosity must be 0, 1, or 2")),
                    };
                }
                "--relativity" => cli.relativity = RelativitySelection::parse(&value)?,
                "--soc-bands" => cli.soc_bands = parse_value(&name, &value)?,
                "--kh-hartree-update" => {
                    cli.kh_hartree_update = match value.as_str() {
                        "outer-density" => KhSocHartreeUpdate::OuterDensity,
                        "coupled-fock" => KhSocHartreeUpdate::CoupledFock,
                        _ => {
                            return Err(invalid_input(format!(
                                "unknown KH Hartree update {value}"
                            )));
                        }
                    };
                }
                "--box" => cli.box_size = parse_value(&name, &value)?,
                "--orbital-g" => cli.orbital_g = parse_value(&name, &value)?,
                "--field-g" => cli.field_g = parse_value(&name, &value)?,
                "--orbital-lmax" => cli.orbital_l_max = parse_value(&name, &value)?,
                "--product-g" => cli.product_g = parse_value(&name, &value)?,
                "--product-lmax" => cli.product_l_max = parse_value(&name, &value)?,
                "--lexp" => cli.lexp = parse_value(&name, &value)?,
                "--exchange-coulomb" => {
                    cli.exchange_coulomb = ExchangeCoulombSelection::parse(&value)?
                }
                "--fock-fourier-g" => cli.fock_fourier_g = parse_value(&name, &value)?,
                "--fock-smoothing-omega" => {
                    cli.fock_smoothing_omega = Some(parse_value(&name, &value)?)
                }
                "--rmt" => cli.muffin_tin_radius = parse_value(&name, &value)?,
                "--radial-points" => cli.radial_points = parse_value(&name, &value)?,
                "--hdlo" => cli.hdlo = HdloSelection::parse(&value)?,
                "--temperature" => cli.temperature = parse_value(&name, &value)?,
                "--outer-max-iterations" => cli.outer_max_iterations = parse_value(&name, &value)?,
                "--outer-energy-tolerance" => {
                    cli.outer_energy_tolerance = parse_value(&name, &value)?
                }
                "--outer-density-tolerance" => {
                    cli.outer_density_tolerance = parse_value(&name, &value)?
                }
                "--outer-mixing" => cli.outer_mixing = OuterMixerSelection::parse(&value)?,
                "--outer-mixing-alpha" => cli.outer_mixing_alpha = parse_value(&name, &value)?,
                "--outer-mixing-history" => cli.outer_mixing_history = parse_value(&name, &value)?,
                "--core-max-iterations" => cli.core_max_iterations = parse_value(&name, &value)?,
                "--core-energy-tolerance" => {
                    cli.core_energy_tolerance = parse_value(&name, &value)?
                }
                "--core-radial-tolerance" => {
                    cli.core_radial_tolerance = parse_value(&name, &value)?
                }
                "--fock-max-iterations" => cli.max_fock_iterations = parse_value(&name, &value)?,
                "--fock-density-tolerance" => {
                    cli.fock_density_tolerance = parse_value(&name, &value)?
                }
                "--fock-feedback-tolerance" => {
                    cli.fock_feedback_tolerance = parse_value(&name, &value)?
                }
                "--fock-commutator-tolerance" => {
                    cli.fock_commutator_tolerance = parse_value(&name, &value)?
                }
                "--scalar-fock-mixing" => {
                    cli.scalar_fock_mixing = FockMixerSelection::parse(&value)?
                }
                "--spinor-virtual-level-shift" => {
                    cli.spinor_virtual_level_shift = parse_value(&name, &value)?
                }
                "--fock-mixing" => cli.fock_mixing = FockMixerSelection::parse(&value)?,
                "--fock-mixing-alpha" => cli.fock_mixing_alpha = parse_value(&name, &value)?,
                "--fock-diis-history" => cli.fock_diis_history = parse_value(&name, &value)?,
                "--fock-diis-level-shift" => {
                    cli.fock_diis_level_shift = parse_value(&name, &value)?
                }
                "--fock-diis-startup-steps" => {
                    cli.fock_diis_startup_steps = parse_value(&name, &value)?
                }
                "--fock-diis-damping" => cli.fock_diis_damping = parse_value(&name, &value)?,
                _ => return Err(invalid_input(format!("unknown option {name:?}"))),
            }
        }
        cli.validate()?;
        if smoke_configuration {
            eprintln!(
                "warning: default smoke configuration; not for physical assessment (see doc/23_core_valence_exchange.md section 3.4)"
            );
        }
        Ok(cli)
    }

    fn validate(&self) -> Result<(), Box<dyn Error>> {
        match (self.exchange_coulomb, self.fock_smoothing_omega) {
            (ExchangeCoulombSelection::SmoothedSpencerAlaviSphere, Some(omega))
                if omega.is_finite() && omega > 0.0 => {}
            (ExchangeCoulombSelection::SmoothedSpencerAlaviSphere, _) => {
                return Err(invalid_input(
                    "smoothed-spencer-alavi-sphere requires explicit finite positive --fock-smoothing-omega",
                ));
            }
            (_, Some(_)) => {
                return Err(invalid_input(
                    "--fock-smoothing-omega applies only to smoothed-spencer-alavi-sphere",
                ));
            }
            (_, None) => {}
        }
        for (name, value) in [
            ("--box", self.box_size),
            ("--rmt", self.muffin_tin_radius),
            ("--orbital-g", self.orbital_g),
            ("--field-g", self.field_g),
            ("--product-g", self.product_g),
            ("--fock-fourier-g", self.fock_fourier_g),
            ("--temperature", self.temperature),
            ("--outer-energy-tolerance", self.outer_energy_tolerance),
            ("--outer-density-tolerance", self.outer_density_tolerance),
            ("--outer-mixing-alpha", self.outer_mixing_alpha),
            ("--core-energy-tolerance", self.core_energy_tolerance),
            ("--core-radial-tolerance", self.core_radial_tolerance),
            ("--fock-mixing-alpha", self.fock_mixing_alpha),
            ("--fock-density-tolerance", self.fock_density_tolerance),
            ("--fock-feedback-tolerance", self.fock_feedback_tolerance),
            (
                "--fock-commutator-tolerance",
                self.fock_commutator_tolerance,
            ),
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
        if self.outer_max_iterations < 2 {
            return Err(invalid_input("--outer-max-iterations must be at least 2"));
        }
        if self.outer_mixing_alpha > 1.0 {
            return Err(invalid_input("--outer-mixing-alpha must not exceed 1"));
        }
        if self.outer_mixing_history < 2 {
            return Err(invalid_input("--outer-mixing-history must be at least 2"));
        }
        if self.core_max_iterations == 0 {
            return Err(invalid_input("--core-max-iterations must be positive"));
        }
        if self.max_fock_iterations < 2 {
            return Err(invalid_input("--fock-max-iterations must be at least 2"));
        }
        if self.soc_bands == 0 {
            return Err(invalid_input("--soc-bands must be positive"));
        }
        if self.fock_mixing_alpha > 1.0 {
            return Err(invalid_input("--fock-mixing-alpha must not exceed 1"));
        }
        if self.fock_diis_history < 2 {
            return Err(invalid_input("--fock-diis-history must be at least 2"));
        }
        if !self.fock_diis_level_shift.is_finite() || self.fock_diis_level_shift < 0.0 {
            return Err(invalid_input(
                "--fock-diis-level-shift must be finite and nonnegative",
            ));
        }
        if !self.spinor_virtual_level_shift.is_finite() || self.spinor_virtual_level_shift < 0.0 {
            return Err(invalid_input(
                "--spinor-virtual-level-shift must be finite and nonnegative",
            ));
        }
        if self.relativity == RelativitySelection::SpinorFirst
            && self.spinor_virtual_level_shift != 0.0
        {
            return Err(invalid_input(
                "--spinor-virtual-level-shift requires --relativity kh-soc or spinor-frozen",
            ));
        }
        if !self.fock_diis_damping.is_finite()
            || self.fock_diis_damping < 0.0
            || self.fock_diis_damping >= 1.0
        {
            return Err(invalid_input(
                "--fock-diis-damping must be finite and in [0, 1)",
            ));
        }
        if self.product_l_max > self.lexp || self.lexp > MAX_LEXP {
            return Err(invalid_input(format!(
                "angular cutoffs must satisfy --product-lmax <= --lexp <= {MAX_LEXP}",
            )));
        }
        match (self.relativity, self.fock_mixing) {
            (
                RelativitySelection::SpinorFirst,
                FockMixerSelection::Cdiis | FockMixerSelection::QuasiNewtonCdiis,
            )
            | (
                RelativitySelection::KhSoc | RelativitySelection::SpinorFrozen,
                FockMixerSelection::Linear
                | FockMixerSelection::Pulay
                | FockMixerSelection::Cdiis
                | FockMixerSelection::QuasiNewtonCdiis,
            ) => {}
            (RelativitySelection::SpinorFirst, _) => {
                return Err(invalid_input(
                    "spinor-first requires --fock-mixing cdiis or quasi-newton-cdiis",
                ));
            }
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
    verbosity: u8,
    coupled_scf_max_iterations: Option<usize>,
    coupled_scf_density_tolerance: Option<f64>,
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
    exchange_coulomb: &'static str,
    spencer_alavi_radius_bohr: Option<f64>,
    spencer_alavi_smoothing_omega_bohr_inverse: Option<f64>,
    spencer_alavi_smoothing_eta: Option<f64>,
    fock_fourier_g_max_bohr_inverse: Option<f64>,
    hdlo: &'static str,
    temperature_hartree: f64,
    k_mesh_divisions: [usize; 3],
    k_mesh_shift: [f64; 3],
    k_mesh_reduction: &'static str,
    gamma_exchange: &'static str,
    relativity: &'static str,
    soc_first_variation_bands: Option<usize>,
    kh_hartree_update: Option<&'static str>,
    core_treatment: &'static str,
    exchange_correlation: &'static str,
    outer_mixing_algorithm: &'static str,
    outer_mixing_alpha: f64,
    outer_mixing_history: Option<usize>,
    outer_energy_tolerance_hartree: f64,
    outer_density_tolerance: f64,
    outer_max_iterations: usize,
    core_action_mixing: Option<f64>,
    core_energy_tolerance_hartree: Option<f64>,
    core_radial_tolerance: Option<f64>,
    core_vc_imaginary_tolerance: Option<f64>,
    core_max_iterations: Option<usize>,
    max_fock_iterations: usize,
    fock_density_tolerance: f64,
    fock_feedback_tolerance_hartree: f64,
    fock_commutator_tolerance_hartree: Option<f64>,
    spinor_virtual_level_shift_hartree: Option<f64>,
    scalar_fock_mixing_algorithm: Option<&'static str>,
    scalar_fock_diis_level_shift_hartree: Option<f64>,
    fock_mixing_algorithm: &'static str,
    fock_mixing_alpha: Option<f64>,
    fock_mixing_history: usize,
    fock_diis_level_shift_hartree: Option<f64>,
    fock_diis_startup_steps: Option<usize>,
    fock_diis_damping: Option<f64>,
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
    fock_feedback_residual_hartree: f64,
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
struct KhSocIterationsFile {
    status: &'static str,
    failure: Option<String>,
    iterations: Vec<KhSocIterationRecord>,
}

#[derive(Serialize)]
struct KhSocIterationRecord {
    iteration: usize,
    fock_iterations: usize,
    exchange_rebuilds: usize,
    scalar_fock_iterations: usize,
    spinor_fock_iterations: usize,
    scalar_exchange_rebuilds: usize,
    spinor_exchange_rebuilds: usize,
    scalar_fock_fixed_point_residual: f64,
    scalar_fock_feedback_residual_hartree: f64,
    spinor_fock_commutator_residual_hartree: f64,
    spinor_active_feedback_residual_hartree: f64,
    vv_exchange_energy_hartree: f64,
    fock_fixed_point_residual: f64,
    fock_feedback_residual_hartree: f64,
    fock_commutator_residual_hartree: f64,
    active_feedback_residual_hartree: f64,
    valence_density_rms: f64,
    muffin_tin_density_rms: f64,
    interstitial_density_rms: f64,
    total_density_rms: f64,
    total_energy_hartree: f64,
    energy_change_hartree: Option<f64>,
}

#[derive(Serialize)]
struct KhSocResultFile {
    status: &'static str,
    energy_reference: &'static str,
    homo_energy_hartree: f64,
    fermi_shift_hartree: f64,
    total_energy_hartree: f64,
    vv_exchange_energy_hartree: f64,
    core_valence_exchange_hartree: f64,
    core_core_exchange_hartree: f64,
    core_h0_trace_hartree: f64,
    orbital_energies_and_occupations: Vec<KhSocOrbitalRecord>,
    core_shells: Vec<CoreShellRecord>,
    scalar_core_orthogonalization: Vec<CoreOrthogonalizationRecord>,
    electron_counts: ElectronCounts,
    fock_fixed_point_residual: f64,
    fock_feedback_residual_hartree: f64,
    fock_commutator_residual_hartree: f64,
    active_feedback_residual_hartree: f64,
    muffin_tin_density_rms: f64,
    interstitial_density_rms: f64,
    total_density_rms: f64,
    scalar_to_soc_density_rms: f64,
    exchange_rebuilds: usize,
    k_fractional: Vec<[f64; 3]>,
    q_fractional: Vec<[f64; 3]>,
    k_weights: Vec<f64>,
}

#[derive(Serialize)]
struct CoreOrthogonalizationRecord {
    k_index: usize,
    expanded_basis_dimension: usize,
    active_basis_dimension: usize,
    retained_scalar_bands: usize,
    constraint_count: usize,
    maximum_radial_overlap_residual: f64,
}

#[derive(Serialize)]
struct KhSocOrbitalRecord {
    k_index: usize,
    band_index: usize,
    kramers_pair_index: usize,
    kramers_splitting_hartree: f64,
    energy_hartree: f64,
    homo_shifted_hartree: f64,
    homo_shifted_ev: f64,
    occupation: f64,
    /// MT-only overlap using the same scalar P/Q contraction as static core exchange.
    core_overlap_weights: Vec<CoreOverlapRecord>,
    source_weights: Vec<SourceWeightRecord>,
}

#[derive(Clone, Serialize)]
struct CoreOverlapRecord {
    site_index: usize,
    principal_quantum_number: u32,
    kappa: i32,
    summed_mu_overlap_squared: f64,
}

#[derive(Serialize)]
struct SourceWeightRecord {
    source_band: usize,
    scalar_energy_hartree: f64,
    spin_up: f64,
    spin_down: f64,
    total: f64,
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
    fock_feedback_hartree: f64,
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
    muffintin::set_hf_verbosity(cli.verbosity);
    if cli.verbosity != muffintin::HfVerbosity::Quiet {
        eprintln!(
            "[hf] preparing Kr {} calculation; output={}",
            cli.relativity.as_str(),
            cli.out.display()
        );
    }
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
    let channels =
        derive_basis_channels(&configuration, cli.orbital_l_max, cli.hdlo, cli.relativity);

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
            radial_equation: match cli.relativity {
                RelativitySelection::SpinorFirst | RelativitySelection::SpinorFrozen => {
                    RadialEquationTag::FullyRelativisticDirac
                }
                RelativitySelection::KhSoc => RadialEquationTag::ScalarKoellingHarmon,
            },
            linearization: LinearizationV1 {
                energy_unit: EnergyUnit::Hartree,
                linearization_energies: Vec::new(),
                local_orbital_energies: Vec::new(),
            },
        }],
    };
    let meta = CheckpointMeta {
        title: format!("Kr point-nucleus Gamma {} HF", cli.relativity.as_str()),
        producer: "libmuffintin-runtime kr_relativistic_hf example".to_owned(),
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
        mixing: cli
            .outer_mixing
            .build(cli.outer_mixing_alpha, cli.outer_mixing_history),
        relativity: match cli.relativity {
            RelativitySelection::SpinorFirst | RelativitySelection::SpinorFrozen => {
                ScfRelativity::SpinorFirstVariation
            }
            RelativitySelection::KhSoc => ScfRelativity::SocSecondVariation {
                window: FirstVariationWindow::new(0, cli.soc_bands)?,
            },
        },
        convergence: ScfConvergence {
            energy_tolerance: Hartree(cli.outer_energy_tolerance),
            density_tolerance: cli.outer_density_tolerance,
            max_iterations: cli.outer_max_iterations,
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
        schema_version: 5,
        status: "prepared",
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
            verbosity: cli.verbosity as u8,
            coupled_scf_max_iterations: (cli.relativity == RelativitySelection::SpinorFrozen)
                .then_some(cli.max_fock_iterations),
            coupled_scf_density_tolerance: (cli.relativity == RelativitySelection::SpinorFrozen)
                .then_some(cli.fock_density_tolerance),
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
            exchange_coulomb: cli.exchange_coulomb.as_str(),
            spencer_alavi_radius_bohr: (cli.exchange_coulomb
                != ExchangeCoulombSelection::PeriodicFiniteBody)
                .then_some((3.0 * cli.box_size.powi(3) / (4.0 * PI)).cbrt()),
            spencer_alavi_smoothing_omega_bohr_inverse: cli.fock_smoothing_omega,
            spencer_alavi_smoothing_eta: cli
                .fock_smoothing_omega
                .map(|omega| omega * (3.0 * cli.box_size.powi(3) / (4.0 * PI)).cbrt()),
            fock_fourier_g_max_bohr_inverse: (cli.exchange_coulomb
                != ExchangeCoulombSelection::PeriodicFiniteBody)
                .then_some(cli.fock_fourier_g),
            hdlo: cli.hdlo.as_str(),
            temperature_hartree: cli.temperature,
            k_mesh_divisions: [1, 1, 1],
            k_mesh_shift: [0.0; 3],
            k_mesh_reduction: "full",
            gamma_exchange: match cli.exchange_coulomb {
                ExchangeCoulombSelection::PeriodicFiniteBody => "finite-body",
                ExchangeCoulombSelection::SpencerAlaviSphere
                | ExchangeCoulombSelection::SmoothedSpencerAlaviSphere => "finite-kernel-no-head",
            },
            relativity: cli.relativity.as_str(),
            soc_first_variation_bands: (cli.relativity == RelativitySelection::KhSoc)
                .then_some(cli.soc_bands),
            core_treatment: match cli.relativity {
                RelativitySelection::SpinorFirst => "relaxed",
                RelativitySelection::KhSoc | RelativitySelection::SpinorFrozen => {
                    "frozen-checkpoint"
                }
            },
            exchange_correlation: "LDA-PW92 local-spin-frame",
            kh_hartree_update: (cli.relativity == RelativitySelection::KhSoc).then_some(match cli
                .kh_hartree_update
            {
                KhSocHartreeUpdate::OuterDensity => "outer-density",
                KhSocHartreeUpdate::CoupledFock => "coupled-fock",
            }),
            outer_mixing_algorithm: if cli.relativity == RelativitySelection::SpinorFrozen
                || (cli.relativity == RelativitySelection::KhSoc
                    && cli.kh_hartree_update == KhSocHartreeUpdate::CoupledFock)
            {
                "none"
            } else {
                cli.outer_mixing.as_str()
            },
            outer_mixing_alpha: cli.outer_mixing_alpha,
            outer_mixing_history: match cli.outer_mixing {
                OuterMixerSelection::Linear => None,
                OuterMixerSelection::Pulay => Some(cli.outer_mixing_history),
            },
            outer_energy_tolerance_hartree: cli.outer_energy_tolerance,
            outer_density_tolerance: cli.outer_density_tolerance,
            outer_max_iterations: cli.outer_max_iterations,
            core_action_mixing: (cli.relativity == RelativitySelection::SpinorFirst).then_some(1.0),
            core_energy_tolerance_hartree: (cli.relativity == RelativitySelection::SpinorFirst)
                .then_some(cli.core_energy_tolerance),
            core_radial_tolerance: (cli.relativity == RelativitySelection::SpinorFirst)
                .then_some(cli.core_radial_tolerance),
            core_vc_imaginary_tolerance: (cli.relativity == RelativitySelection::SpinorFirst)
                .then_some(1.0e-8),
            core_max_iterations: (cli.relativity == RelativitySelection::SpinorFirst)
                .then_some(cli.core_max_iterations),
            max_fock_iterations: cli.max_fock_iterations,
            fock_density_tolerance: cli.fock_density_tolerance,
            fock_feedback_tolerance_hartree: cli.fock_feedback_tolerance,
            fock_commutator_tolerance_hartree: (cli.relativity != RelativitySelection::SpinorFirst)
                .then_some(cli.fock_commutator_tolerance),
            spinor_virtual_level_shift_hartree: (cli.relativity
                != RelativitySelection::SpinorFirst)
                .then_some(cli.spinor_virtual_level_shift),
            scalar_fock_mixing_algorithm: (cli.relativity == RelativitySelection::KhSoc)
                .then_some(cli.scalar_fock_mixing.as_str()),
            scalar_fock_diis_level_shift_hartree: (cli.relativity == RelativitySelection::KhSoc
                && matches!(cli.scalar_fock_mixing, FockMixerSelection::QuasiNewtonCdiis))
            .then_some(cli.fock_diis_level_shift),
            fock_mixing_algorithm: cli.fock_mixing.as_str(),
            fock_mixing_alpha: matches!(
                cli.fock_mixing,
                FockMixerSelection::Linear | FockMixerSelection::Pulay
            )
            .then_some(cli.fock_mixing_alpha),
            fock_mixing_history: cli.fock_diis_history,
            fock_diis_level_shift_hartree: match cli.fock_mixing {
                FockMixerSelection::Linear
                | FockMixerSelection::Pulay
                | FockMixerSelection::Cdiis => None,
                FockMixerSelection::QuasiNewtonCdiis => Some(cli.fock_diis_level_shift),
            },
            fock_diis_startup_steps: matches!(
                cli.fock_mixing,
                FockMixerSelection::Cdiis | FockMixerSelection::QuasiNewtonCdiis
            )
            .then_some(cli.fock_diis_startup_steps),
            fock_diis_damping: matches!(
                cli.fock_mixing,
                FockMixerSelection::Cdiis | FockMixerSelection::QuasiNewtonCdiis
            )
            .then_some(cli.fock_diis_damping),
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

    match cli.relativity {
        RelativitySelection::SpinorFirst => run_spinor_first_example(&cli, &start, config)?,
        RelativitySelection::SpinorFrozen => frozen_sra::run(&cli, &start, config)?,
        RelativitySelection::KhSoc => run_kh_soc_example(&cli, &start, config)?,
    }
    Ok(())
}

fn run_spinor_first_example(
    cli: &Cli,
    start: &muffintin::AtomicStart,
    config: ScfConfig,
) -> Result<(), Box<dyn Error>> {
    let spec = RelaxedCoreHfSpec {
        config,
        product_l_max: cli.product_l_max,
        product_g_max: InverseBohr(cli.product_g),
        overlap_tolerance: DEFAULT_TOLERANCE,
        coulomb: exchange_coulomb_request(cli)?,
        gamma: GammaExchangeTreatment::FiniteBody,
        max_fock_iterations: cli.max_fock_iterations,
        fock_density_tolerance: cli.fock_density_tolerance,
        fock_feedback_tolerance: Hartree(cli.fock_feedback_tolerance),
        fock_mixing: cli.fock_mixing.build(
            cli.fock_mixing_alpha,
            cli.fock_diis_history,
            cli.fock_diis_level_shift,
            cli.fock_diis_startup_steps,
            cli.fock_diis_damping,
        ),
        core: CoreFixedPotentialSpec {
            action_mixing: 1.0,
            energy_tolerance: Hartree(cli.core_energy_tolerance),
            radial_tolerance: cli.core_radial_tolerance,
            vc_imaginary_tolerance: 1.0e-8,
            max_iterations: cli.core_max_iterations,
        },
        sector_numerical_tolerance: Hartree(SECTOR_NUMERICAL_TOLERANCE_HARTREE),
        maximum_core_shell_spill: MAXIMUM_CORE_SHELL_SPILL,
    };
    let mut physics = CheckpointPhysics::new(&start.checkpoint)?;
    let result = run_gamma_relaxed_core_hf(&mut physics, &spec)?;

    let iterations = IterationsFile {
        status: "configured_convergence_reached",
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
            (
                "hf.status".to_owned(),
                "configured_convergence_reached".to_owned(),
            ),
        ]),
    )?;
    write_checkpoint(&cli.out.join("final-checkpoint.toml"), &final_checkpoint)?;

    println!(
        "status=configured_convergence_reached out={} total_energy_hartree={} outer_iterations={} exchange_rebuilds={}",
        cli.out.display(),
        result.total_energy.get(),
        result.diagnostics.len(),
        result.exchange_rebuilds
    );
    Ok(())
}

fn run_kh_soc_example(
    cli: &Cli,
    start: &muffintin::AtomicStart,
    config: ScfConfig,
) -> Result<(), Box<dyn Error>> {
    let spec = KhSocValenceHfSpec {
        config,
        hartree_update: cli.kh_hartree_update,
        product_l_max: cli.product_l_max,
        product_g_max: InverseBohr(cli.product_g),
        overlap_tolerance: DEFAULT_TOLERANCE,
        coulomb: exchange_coulomb_request(cli)?,
        gamma: GammaExchangeTreatment::FiniteBody,
        max_fock_iterations: cli.max_fock_iterations,
        fock_density_tolerance: cli.fock_density_tolerance,
        fock_feedback_tolerance: Hartree(cli.fock_feedback_tolerance),
        fock_commutator_tolerance: Hartree(cli.fock_commutator_tolerance),
        spinor_virtual_level_shift: Hartree(cli.spinor_virtual_level_shift),
        scalar_fock_mixing: cli.scalar_fock_mixing.build(
            cli.fock_mixing_alpha,
            cli.fock_diis_history,
            cli.fock_diis_level_shift,
            0,
            0.0,
        ),
        spinor_fock_mixing: cli.fock_mixing.build(
            cli.fock_mixing_alpha,
            cli.fock_diis_history,
            cli.fock_diis_level_shift,
            cli.fock_diis_startup_steps,
            cli.fock_diis_damping,
        ),
        core_treatment: KhSocCoreTreatment::Frozen,
    };
    let mut physics = CheckpointPhysics::new(&start.checkpoint)?;
    let mut iterations = KhSocIterationsFile {
        status: "running",
        failure: None,
        iterations: Vec::new(),
    };
    let iterations_path = cli.out.join("iterations.toml");
    let pending_path = cli.out.join("iterations.toml.next");
    let result = run_gamma_kh_soc_valence_hf(&mut physics, &spec, |diagnostics| {
        iterations.iterations = diagnostics.iter().map(kh_soc_iteration_record).collect();
        write_toml(&pending_path, &iterations)?;
        fs::rename(&pending_path, &iterations_path)
    });
    iterations.status = if result.is_ok() {
        "configured_convergence_reached"
    } else {
        "failed"
    };
    iterations.failure = result.as_ref().err().map(ToString::to_string);
    write_toml(&pending_path, &iterations)?;
    fs::rename(&pending_path, &iterations_path)?;
    let result = result?;
    let result_file = kh_soc_result_record(&result)?;
    write_toml(&cli.out.join("result.toml"), &result_file)?;

    let final_checkpoint = checkpoint_v2_from_regional_state(
        &start.checkpoint,
        &result.total_density,
        &result.potential,
        BTreeMap::from([
            (
                "hf.driver".to_owned(),
                "run_gamma_kh_soc_valence_hf".to_owned(),
            ),
            (
                "hf.status".to_owned(),
                "configured_convergence_reached".to_owned(),
            ),
            (
                "hf.core_treatment".to_owned(),
                "frozen-checkpoint".to_owned(),
            ),
            (
                "hf.energy_reference".to_owned(),
                "closed-shell occupation>=0.5 HOMO shifted to zero".to_owned(),
            ),
        ]),
    )?;
    write_checkpoint(&cli.out.join("final-checkpoint.toml"), &final_checkpoint)?;
    println!(
        "status=configured_convergence_reached route=kh-soc out={} total_energy_hartree={} homo_energy_hartree={} outer_iterations={} exchange_rebuilds={}",
        cli.out.display(),
        result.total_energy.get(),
        result_file.homo_energy_hartree,
        result.diagnostics.len(),
        result.exchange_rebuilds
    );
    Ok(())
}

fn exchange_coulomb_request(cli: &Cli) -> Result<CoulombRequest, Box<dyn Error>> {
    let request = CoulombRequest::cubic(cli.box_size, cli.lexp)?;
    match cli.exchange_coulomb {
        ExchangeCoulombSelection::PeriodicFiniteBody => Ok(request),
        ExchangeCoulombSelection::SpencerAlaviSphere => {
            Ok(request.with_spencer_alavi_sphere(1, InverseBohr(cli.fock_fourier_g))?)
        }
        ExchangeCoulombSelection::SmoothedSpencerAlaviSphere => Ok(request
            .with_smoothed_spencer_alavi_sphere(
                1,
                InverseBohr(cli.fock_fourier_g),
                InverseBohr(
                    cli.fock_smoothing_omega
                        .expect("validated explicit smoothing"),
                ),
            )?),
    }
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
    relativity: RelativitySelection,
) -> Vec<ScfChannelRecipe> {
    let mut channels = Vec::new();
    let mut core_channels = BTreeSet::new();
    let mut collapsed_valence = BTreeSet::new();
    let mut collapsed_local = BTreeSet::new();
    for occupied in configuration.occupations() {
        let n = u32::from(occupied.orbital.principal_quantum_number());
        let kappa = i32::from(occupied.orbital.kappa());
        let l = angular_momentum(kappa);
        match occupied.treatment {
            AtomicChannelTreatment::Core => {
                core_channels.insert((n, l));
            }
            AtomicChannelTreatment::Valence => {
                if collapsed_valence.insert((n, l)) {
                    channels.push(channel(
                        ScfChannelIdentity::ScalarL { n, l },
                        ScfChannelTreatment::Valence,
                        0,
                    ));
                }
            }
            AtomicChannelTreatment::RelativisticLocalOrbital => match relativity {
                RelativitySelection::SpinorFirst | RelativitySelection::SpinorFrozen => channels
                    .push(channel(
                        ScfChannelIdentity::Kappa { n, kappa },
                        ScfChannelTreatment::Lo,
                        0,
                    )),
                RelativitySelection::KhSoc => {
                    if collapsed_local.insert((n, l)) {
                        channels.push(channel(
                            ScfChannelIdentity::ScalarL { n, l },
                            ScfChannelTreatment::Lo,
                            0,
                        ));
                    }
                }
            },
        }
    }
    for l in 0..=l_max {
        if !channels.iter().any(|recipe| {
            recipe.treatment == ScfChannelTreatment::Valence && identity_l(recipe.identity) == l
        }) {
            let mut n = l + 1;
            while core_channels.contains(&(n, l))
                || channels.iter().any(|recipe| {
                    identity_l(recipe.identity) == l && identity_n(recipe.identity) == n
                })
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
        fock_feedback_residual_hartree: item.fock_feedback_residual.get(),
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

fn kh_soc_iteration_record(item: &KhSocValenceHfIterationDiagnostic) -> KhSocIterationRecord {
    KhSocIterationRecord {
        iteration: item.iteration,
        fock_iterations: item.fock_iterations,
        exchange_rebuilds: item.exchange_rebuilds,
        scalar_fock_iterations: item.scalar_fock_iterations,
        spinor_fock_iterations: item.spinor_fock_iterations,
        scalar_exchange_rebuilds: item.scalar_exchange_rebuilds,
        spinor_exchange_rebuilds: item.spinor_exchange_rebuilds,
        scalar_fock_fixed_point_residual: item.scalar_fock_fixed_point_residual,
        scalar_fock_feedback_residual_hartree: item.scalar_fock_feedback_residual.get(),
        spinor_fock_commutator_residual_hartree: item.spinor_fock_commutator_residual.get(),
        spinor_active_feedback_residual_hartree: item.spinor_active_feedback_residual.get(),
        vv_exchange_energy_hartree: item.exchange_energy.get(),
        fock_fixed_point_residual: item.fock_fixed_point_residual,
        fock_feedback_residual_hartree: item.fock_feedback_residual.get(),
        fock_commutator_residual_hartree: item.spinor_fock_commutator_residual.get(),
        active_feedback_residual_hartree: item.spinor_active_feedback_residual.get(),
        valence_density_rms: item.valence_density_rms,
        muffin_tin_density_rms: item.muffin_tin_density_rms,
        interstitial_density_rms: item.interstitial_density_rms,
        total_density_rms: item.regional_density_rms,
        total_energy_hartree: item.total_energy.get(),
        energy_change_hartree: item.energy_change.map(Hartree::get),
    }
}

fn kh_soc_result_record(result: &KhSocValenceHfResult) -> Result<KhSocResultFile, Box<dyn Error>> {
    let mut core_spaces = Vec::new();
    for (k, point) in result.scalar_bands.points().iter().enumerate() {
        let CheckpointKPointSolution::Collinear {
            bases, solutions, ..
        } = &point.solution
        else {
            return Err(invalid_input("KH+SOC source must use scalar eigenvectors"));
        };
        if let Some(core) = &bases.up.core_orthogonalization {
            core_spaces.push(CoreOrthogonalizationRecord {
                k_index: k,
                expanded_basis_dimension: core.embedding.shape()[0],
                active_basis_dimension: core.embedding.shape()[1],
                retained_scalar_bands: solutions.up.eigenvalues.len(),
                constraint_count: core.constraint_count,
                maximum_radial_overlap_residual: core.maximum_radial_overlap_residual,
            });
        }
    }
    let homo = result
        .orbital_energies
        .iter()
        .zip(&result.occupations)
        .flat_map(|(energies, occupations)| energies.iter().zip(occupations))
        .filter(|(_, occupation)| **occupation >= HOMO_OCCUPATION_THRESHOLD)
        .map(|(energy, _)| energy.get())
        .max_by(f64::total_cmp)
        .ok_or_else(|| invalid_input("KH+SOC result has no occupied orbital for the HOMO shift"))?;
    let mut orbitals = Vec::new();
    for (k, ((energies, occupations), diagnostics)) in result
        .orbital_energies
        .iter()
        .zip(&result.occupations)
        .zip(&result.second_variation_diagnostics)
        .enumerate()
    {
        if energies.len() != occupations.len() || energies.len() != diagnostics.len() {
            return Err(invalid_input(format!(
                "KH+SOC output arrays have inconsistent lengths at k={k}"
            )));
        }
        let scalar_energies = scalar_source_energies(result, k)?;
        let core_overlaps = kh_soc_core_overlap_records(result, k)?;
        for (band, ((energy, occupation), diagnostic)) in energies
            .iter()
            .zip(occupations)
            .zip(diagnostics)
            .enumerate()
        {
            let pair_start = 2 * (band / 2);
            let pair_end = pair_start + 1;
            let splitting = energies.get(pair_end).map_or(0.0, |partner| {
                (partner.get() - energies[pair_start].get()).abs()
            });
            let source_weights = diagnostic
                .source_weights
                .iter()
                .map(|weight| {
                    let scalar_energy =
                        scalar_energies.get(weight.source_band).ok_or_else(|| {
                            invalid_input(format!(
                                "SOC source band {} is outside scalar spectrum at k={k}",
                                weight.source_band
                            ))
                        })?;
                    Ok(SourceWeightRecord {
                        source_band: weight.source_band,
                        scalar_energy_hartree: scalar_energy.get(),
                        spin_up: weight.spin_up,
                        spin_down: weight.spin_down,
                        total: weight.spin_up + weight.spin_down,
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            let shifted = energy.get() - homo;
            orbitals.push(KhSocOrbitalRecord {
                k_index: k,
                band_index: band,
                kramers_pair_index: band / 2,
                kramers_splitting_hartree: splitting,
                energy_hartree: energy.get(),
                homo_shifted_hartree: shifted,
                homo_shifted_ev: Hartree(shifted).to_ev(),
                occupation: *occupation,
                core_overlap_weights: core_overlaps[band].clone(),
                source_weights,
            });
        }
    }
    Ok(KhSocResultFile {
        status: "configured_convergence_reached",
        energy_reference: "closed-shell HOMO (occupation >= 0.5) set to zero",
        homo_energy_hartree: homo,
        fermi_shift_hartree: -homo,
        total_energy_hartree: result.total_energy.get(),
        vv_exchange_energy_hartree: result.second_variation_exchange.exchange_energy.get(),
        core_valence_exchange_hartree: result.core_valence_exchange.get(),
        core_core_exchange_hartree: result.core_core_exchange.get(),
        core_h0_trace_hartree: result.core_h0_trace.get(),
        orbital_energies_and_occupations: orbitals,
        core_shells: core_shell_records(&result.core_orbitals),
        scalar_core_orthogonalization: core_spaces,
        electron_counts: ElectronCounts {
            valence: electron_count(&result.valence_density)?,
            core: electron_count(&result.core_density)?,
            total: electron_count(&result.total_density)?,
        },
        fock_fixed_point_residual: result.fock_fixed_point_residual,
        fock_feedback_residual_hartree: result.fock_feedback_residual.get(),
        fock_commutator_residual_hartree: result.fock_commutator_residual.get(),
        active_feedback_residual_hartree: result.active_feedback_residual.get(),
        muffin_tin_density_rms: result.muffin_tin_density_rms,
        interstitial_density_rms: result.interstitial_density_rms,
        total_density_rms: result.regional_density_rms,
        scalar_to_soc_density_rms: result.second_variation_density_rms,
        exchange_rebuilds: result.exchange_rebuilds,
        k_fractional: result.k_fractional.clone(),
        q_fractional: result.q_fractional.clone(),
        k_weights: result.k_weights.clone(),
    })
}

fn kh_soc_core_overlap_records(
    result: &KhSocValenceHfResult,
    k: usize,
) -> Result<Vec<Vec<CoreOverlapRecord>>, Box<dyn Error>> {
    let CheckpointKPointSolution::Collinear {
        bases, solutions, ..
    } = &result.bands.points()[k].solution
    else {
        return Err(invalid_input(
            "KH+SOC core overlaps require split Pauli eigenvectors",
        ));
    };
    let band_count = solutions.up.eigenvectors.columns();
    let mut records = vec![Vec::new(); band_count];
    for sidecar in &result.core_orbitals {
        let site = sidecar.site_index;
        let radial_site = &bases.up.radial_sites[site];
        let mesh = &bases.up.density_sites[site].mesh;
        let l_max = (radial_site.linearized.len() - 1) as u32;
        let augmented_count = lm_count(l_max);
        let local_layout = bases.up.compiled.layout.site_layout(site).unwrap();
        let projection = CompiledSiteProjection::scalar(&bases.up.compiled, site)?;
        let up = projection.project_eigenvectors(&solutions.up.eigenvectors)?;
        let down = projection.project_eigenvectors(&solutions.down.eigenvectors)?;
        for core in &sidecar.shells {
            let l = core.state.kappa.large_l();
            if l > l_max {
                continue;
            }
            let linearized = &radial_site.linearized[l as usize];
            let locals = &radial_site.local_orbitals[l as usize];
            let mut radials = vec![
                (&linearized.solution.p, linearized.solution.q.as_deref()),
                (
                    &linearized.energy_derivative.p,
                    linearized.energy_derivative.q.as_deref(),
                ),
            ];
            radials.extend(
                locals
                    .iter()
                    .map(|local| (&local.orbital.p, local.orbital.q.as_deref())),
            );
            let overlaps = radials
                .iter()
                .map(|(p, q)| {
                    let integrand = (0..mesh.len())
                        .map(|r| {
                            (core.p[r] * p[r] + core.q[r] * q.map_or(0.0, |q| q[r]))
                                / core.norm_total.sqrt()
                        })
                        .collect::<Vec<_>>();
                    mesh.integrate(&integrand)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut weights = vec![0.0; band_count];
            for channel in core.state.kappa.channels() {
                let mut amplitudes = vec![Complex64::default(); band_count];
                for term in channel.spinor_harmonic_terms().into_iter().flatten() {
                    let coefficients = match term.spin {
                        SpinProjection::Up => &up,
                        SpinProjection::Down => &down,
                    };
                    for (radial, overlap) in overlaps.iter().enumerate() {
                        let coordinate = if radial < 2 {
                            2 * term.orbital.index() + radial
                        } else {
                            2 * augmented_count
                                + local_layout.index(l, term.orbital.m, radial - 2).unwrap()
                        };
                        for (band, amplitude) in amplitudes.iter_mut().enumerate() {
                            *amplitude +=
                                term.coefficient * overlap * coefficients.at(coordinate, band);
                        }
                    }
                }
                for (weight, amplitude) in weights.iter_mut().zip(amplitudes) {
                    *weight += amplitude.norm_sqr();
                }
            }
            for (record, weight) in records.iter_mut().zip(weights) {
                record.push(CoreOverlapRecord {
                    site_index: site,
                    principal_quantum_number: core.state.n,
                    kappa: core.state.kappa.get(),
                    summed_mu_overlap_squared: weight,
                });
            }
        }
    }
    Ok(records)
}

fn scalar_source_energies(
    result: &KhSocValenceHfResult,
    k: usize,
) -> Result<&[Hartree], Box<dyn Error>> {
    let point = result
        .scalar_bands
        .points()
        .get(k)
        .ok_or_else(|| invalid_input(format!("missing scalar k point {k}")))?;
    let muffintin_dft::CheckpointKPointSolution::Collinear { solutions, .. } = &point.solution
    else {
        return Err(invalid_input("KH+SOC scalar source is not collinear"));
    };
    Ok(&solutions.up.eigenvalues)
}

fn core_shell_records(sidecars: &[muffintin_dft::CoreShellOrbitals]) -> Vec<CoreShellRecord> {
    let mut records = Vec::new();
    for site in sidecars {
        for (shell_index, shell) in site.shells.iter().enumerate() {
            let occupation = match &shell.occupations {
                CoreShellOccupations::MuResolved(values) => {
                    values.iter().map(|(_, value)| value).sum()
                }
                CoreShellOccupations::ExplicitCollinear { up, down } => up + down,
            };
            records.push(CoreShellRecord {
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
    records
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
    let core_shells = core_shell_records(&result.core_orbitals);
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
        status: "configured_convergence_reached",
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
            fock_feedback_hartree: result.fock_feedback_residual.get(),
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

fn write_toml(path: &Path, value: &impl Serialize) -> io::Result<()> {
    let mut text = toml::to_string_pretty(value).map_err(io::Error::other)?;
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
