//! Spherical four-component Dirac bound-core states.
//!
//! The public radial spinor is
//! `Psi = (P Omega_kappa, i Q Omega_-kappa) / r`.  Energies have the
//! electron rest energy subtracted and are measured in Hartree.  The
//! integration variable used for the small component is `q_hat = c Q`.

use std::borrow::Borrow;

use muffintin_core::{Bohr, DiracAngularContract, ExponentialMesh, Hartree, Kappa};
use rayon::prelude::*;
use thiserror::Error;

use crate::valence::{BoundaryData, LocalOrbitalCoefficients, SPEX_SPEED_OF_LIGHT};

const C_SQUARED: f64 = SPEX_SPEED_OF_LIGHT * SPEX_SPEED_OF_LIGHT;
const CORE_NORM_PARTITION_TOLERANCE: f64 = 1.0e-8;

/// The role of a relativistic radial solution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelativisticRole {
    /// A normalizable bound core state on an extended radial domain.
    Core,
    /// A regular fixed-energy four-component valence radial basis.
    Valence,
}

/// Quantum numbers identifying a spherical bound-core channel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoreState {
    /// Principal quantum number.
    pub n: u32,
    pub kappa: Kappa,
}

impl CoreState {
    /// Construct a state and require `n >= l + 1`.
    pub fn new(n: u32, kappa: Kappa) -> Result<Self, DiracError> {
        let minimum = kappa.large_l() + 1;
        if n < minimum {
            Err(DiracError::InvalidPrincipalQuantumNumber {
                n,
                l: kappa.large_l(),
            })
        } else {
            Ok(Self { n, kappa })
        }
    }

    /// Expected nonrelativistic radial node count `n - l - 1`.
    pub const fn expected_nodes(self) -> u32 {
        self.n - self.kappa.large_l() - 1
    }
}

/// A checked energy interval in the rest-energy-subtracted Hartree scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnergyBracket {
    pub lower: Hartree,
    pub upper: Hartree,
}

impl EnergyBracket {
    /// Construct a finite, increasing bracket.
    pub fn new(lower: Hartree, upper: Hartree) -> Result<Self, DiracError> {
        let bracket = Self { lower, upper };
        bracket.validate()?;
        Ok(bracket)
    }

    /// Convenience constructor from raw Hartree values.
    pub fn from_values(lower: f64, upper: f64) -> Result<Self, DiracError> {
        Self::new(Hartree(lower), Hartree(upper))
    }

    /// Return the two raw Hartree values.
    pub const fn values(self) -> (f64, f64) {
        (self.lower.get(), self.upper.get())
    }

    fn validate(self) -> Result<(), DiracError> {
        let (lower, upper) = self.values();
        if lower.is_finite() && upper.is_finite() && lower < upper {
            Ok(())
        } else {
            Err(DiracError::InvalidEnergyBracket { lower, upper })
        }
    }
}

/// Controls for a two-sided bound-core shooting calculation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreDiracSpec {
    pub state: CoreState,
    /// Positive finite nuclear charge used by the regular Coulomb-origin branch.
    pub nuclear_charge: f64,
    /// Energy interval that must isolate one eigenvalue with the node count
    /// implied by `state`; the solver does not search a multi-root interval.
    pub bracket: EnergyBracket,
    /// Muffin-tin cutoff; the supplied mesh must continue beyond it.
    pub muffin_tin_radius: Bohr,
    /// Absolute energy tolerance in Hartree.
    pub energy_tolerance: f64,
    /// Tolerance for the scale-free two-component matching residual.
    pub matching_tolerance: f64,
    pub max_iterations: usize,
}

/// Signed physical nonlocal exchange action on a core radial spinor.
///
/// `p` and `q` are the large- and physical-small-component radial actions
/// `(K psi)_P` and `(K psi)_Q`, sampled on the solver mesh. Their units are
/// Hartree times the corresponding physical reduced radial component. The
/// sign of `K` is already included; the solver does not negate the input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreDiracExchangeAction<'a> {
    pub p: &'a [f64],
    pub q: &'a [f64],
}

/// Root-selection controls for a source-driven core Dirac update.
///
/// The core bracket is a provenance-preserving search window and may contain
/// both norm roots on opposite sides of a homogeneous pole. The predicted
/// energy identifies the physical Picard branch, normally as the previous
/// core energy plus the old-orbital expectation value of the signed action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreDiracSourcedSpec {
    pub core: CoreDiracSpec,
    pub predicted_energy: Hartree,
    /// Number of equal scan intervals across `core.bracket`.
    pub search_intervals: usize,
}

impl CoreDiracSourcedSpec {
    pub const fn new(core: CoreDiracSpec, predicted_energy: Hartree) -> Self {
        Self {
            core,
            predicted_energy,
            search_intervals: 512,
        }
    }

    pub const fn with_search_intervals(mut self, search_intervals: usize) -> Self {
        self.search_intervals = search_intervals;
        self
    }
}

/// Input contract for the regular fixed-energy four-component valence path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValenceDiracSpec {
    pub kappa: Kappa,
    /// Rest-energy-subtracted trial energy in Hartree.
    pub energy: Hartree,
    /// Speed of light in Hartree atomic units.
    pub speed_of_light: f64,
}

impl ValenceDiracSpec {
    pub fn new(kappa: Kappa, energy: Hartree) -> Result<Self, DiracError> {
        if energy.get().is_finite() {
            Ok(Self {
                kappa,
                energy,
                speed_of_light: SPEX_SPEED_OF_LIGHT,
            })
        } else {
            Err(DiracError::NonFiniteEnergy(energy.get()))
        }
    }

    /// Override the default SPEX speed of light.
    pub fn with_speed_of_light(mut self, speed_of_light: f64) -> Result<Self, DiracError> {
        if !speed_of_light.is_finite() || speed_of_light <= 0.0 {
            return Err(DiracError::InvalidSpeedOfLight(speed_of_light));
        }
        self.speed_of_light = speed_of_light;
        Ok(self)
    }
}

impl CoreDiracSpec {
    /// Construct a specification with conservative shooting tolerances.
    pub const fn new(
        state: CoreState,
        nuclear_charge: f64,
        bracket: EnergyBracket,
        muffin_tin_radius: Bohr,
    ) -> Self {
        Self {
            state,
            nuclear_charge,
            bracket,
            muffin_tin_radius,
            energy_tolerance: 1.0e-11,
            matching_tolerance: 1.0e-10,
            max_iterations: 160,
        }
    }

    pub const fn with_tolerances(
        mut self,
        energy_tolerance: f64,
        matching_tolerance: f64,
        max_iterations: usize,
    ) -> Self {
        self.energy_tolerance = energy_tolerance;
        self.matching_tolerance = matching_tolerance;
        self.max_iterations = max_iterations;
        self
    }
}

/// Deterministic finite continuum-below scan for one bound-core bracket.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreBracketSearch {
    pub state: CoreState,
    /// Positive finite nuclear charge used by every scanned outward branch.
    pub nuclear_charge: f64,
    pub muffin_tin_radius: Bohr,
    pub energy_window: EnergyBracket,
    /// Number of equal adjacent intervals in `energy_window`.
    pub intervals: usize,
}

impl CoreBracketSearch {
    pub const fn new(
        state: CoreState,
        nuclear_charge: f64,
        muffin_tin_radius: Bohr,
        energy_window: EnergyBracket,
    ) -> Self {
        Self {
            state,
            nuclear_charge,
            muffin_tin_radius,
            energy_window,
            intervals: 256,
        }
    }

    pub const fn with_intervals(mut self, intervals: usize) -> Self {
        self.intervals = intervals;
        self
    }
}

/// A normalized physical core spinor and its shooting diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreDiracSolution {
    pub role: RelativisticRole,
    pub state: CoreState,
    pub angular: DiracAngularContract,
    /// Rest-energy-subtracted eigenvalue in Hartree.
    pub energy: Hartree,
    /// Large reduced radial component `P`.
    pub p: Vec<f64>,
    /// Physical small reduced radial component `Q` (not `c Q`).
    pub q: Vec<f64>,
    /// Total normalized radial integral (nominally one).
    pub norm_total: f64,
    /// Part of the total norm assigned at or inside the MT cutoff.
    pub norm_mt: f64,
    /// Part of the total norm outside the MT cutoff.
    pub norm_outside: f64,
    /// Alias of `norm_outside`, useful as a core-spill diagnostic.
    pub spill: f64,
    /// Numerically observed nodes of `P`.
    pub nodes: u32,
    pub match_radius: Bohr,
    /// Scale-free Wronskian residual at the matching radius.
    pub matching_residual: f64,
}

/// Exact first-order trace of a physical reduced Dirac radial spinor.
///
/// The derivatives are evaluated from the Dirac equations at the same radius;
/// they are not finite differences of sampled arrays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiracBoundaryTrace<T = f64> {
    pub radius: Bohr,
    pub p: T,
    pub q: T,
    pub p_derivative: T,
    pub q_derivative: T,
}

impl DiracBoundaryTrace<f64> {
    /// Project the large component onto the scalar/SRA LAPW boundary pair.
    ///
    /// With `U=P/r`, this returns `U(R)` and `dU/dr|_R`.
    pub fn sra_large_component(self) -> BoundaryData {
        let r = self.radius.get();
        BoundaryData::new(self.p / r, self.p_derivative / r - self.p / (r * r), r)
    }
}

/// Analytic fixed-potential energy derivative of a normalized valence spinor.
#[derive(Clone, Debug, PartialEq)]
pub struct DiracEnergyDerivative {
    /// `dP/dE` in Hartree⁻¹.
    pub p: Vec<f64>,
    /// Physical `dQ/dE` in Hartree⁻¹.
    pub q: Vec<f64>,
    pub boundary: DiracBoundaryTrace,
    /// Norm after imposing the parallel-transport gauge.
    pub norm_squared: f64,
}

/// Analytic fixed-potential second energy derivative of a normalized spinor.
#[derive(Clone, Debug, PartialEq)]
pub struct DiracSecondEnergyDerivative {
    /// `d²P/dE²` in Hartree⁻².
    pub p: Vec<f64>,
    /// Physical `d²Q/dE²` in Hartree⁻².
    pub q: Vec<f64>,
    pub boundary: DiracBoundaryTrace,
    pub norm_squared: f64,
}

/// A normalized SRA local orbital with both large-component boundary data zero.
#[derive(Clone, Debug, PartialEq)]
pub struct DiracLocalOrbital {
    pub energy: Hartree,
    pub kappa: Kappa,
    pub p: Vec<f64>,
    /// Physical small reduced component; never the internal `cQ`.
    pub q: Vec<f64>,
    pub coefficients: LocalOrbitalCoefficients,
    pub boundary: BoundaryData,
}

/// Normalized regular four-component valence radial solution at fixed energy.
#[derive(Clone, Debug, PartialEq)]
pub struct ValenceDiracSolution {
    pub role: RelativisticRole,
    pub kappa: Kappa,
    pub angular: DiracAngularContract,
    pub energy: Hartree,
    pub speed_of_light: f64,
    /// Physical large reduced radial component `P`.
    pub p: Vec<f64>,
    /// Physical small reduced radial component `Q` (never the internal `cQ`).
    pub q: Vec<f64>,
    pub boundary: DiracBoundaryTrace,
    pub energy_derivative: DiracEnergyDerivative,
    pub second_energy_derivative: DiracSecondEnergyDerivative,
    pub norm_total: f64,
}

impl ValenceDiracSolution {
    pub fn large(&self) -> &[f64] {
        &self.p
    }

    pub fn small(&self) -> &[f64] {
        &self.q
    }

    /// The established scalar/SRA value-and-slope boundary adapter.
    pub fn sra_boundary(&self) -> BoundaryData {
        self.boundary.sra_large_component()
    }

    /// Build a confined SRA local orbital from a solution at a distinct energy.
    ///
    /// The matched spinor is `raw + a * self + b * d(self)/dE`.  Its large
    /// component and radial derivative vanish at the muffin-tin boundary.  The
    /// same coefficients are applied to the physical small component before
    /// normalizing the complete four-component radial norm.
    pub fn sra_local_orbital(
        &self,
        raw_at_lo_energy: &ValenceDiracSolution,
        mesh: &ExponentialMesh,
    ) -> Result<DiracLocalOrbital, DiracError> {
        if self.kappa != raw_at_lo_energy.kappa {
            return Err(DiracError::LocalOrbitalKappaMismatch {
                base: self.kappa.get(),
                raw: raw_at_lo_energy.kappa.get(),
            });
        }
        if self.speed_of_light != raw_at_lo_energy.speed_of_light {
            return Err(DiracError::LocalOrbitalSpeedOfLightMismatch {
                base: self.speed_of_light,
                raw: raw_at_lo_energy.speed_of_light,
            });
        }
        if self.energy == raw_at_lo_energy.energy {
            return Err(DiracError::LocalOrbitalEnergyNotDistinct {
                energy: self.energy.get(),
            });
        }

        let expected = mesh.len();
        for (field, actual) in [
            ("base.p", self.p.len()),
            ("base.q", self.q.len()),
            ("base.energy_derivative.p", self.energy_derivative.p.len()),
            ("base.energy_derivative.q", self.energy_derivative.q.len()),
            ("raw.p", raw_at_lo_energy.p.len()),
            ("raw.q", raw_at_lo_energy.q.len()),
        ] {
            if actual != expected {
                return Err(DiracError::LocalOrbitalSampleCountMismatch {
                    field,
                    mesh: expected,
                    actual,
                });
            }
        }

        let mesh_boundary = mesh.last().get();
        for (field, actual) in [
            ("base.boundary", self.boundary.radius.get()),
            (
                "base.energy_derivative.boundary",
                self.energy_derivative.boundary.radius.get(),
            ),
            ("raw.boundary", raw_at_lo_energy.boundary.radius.get()),
        ] {
            if actual != mesh_boundary {
                return Err(DiracError::LocalOrbitalBoundaryRadiusMismatch {
                    field,
                    mesh: mesh_boundary,
                    actual,
                });
            }
        }

        self.match_sra_local_orbital(
            mesh,
            raw_at_lo_energy.energy,
            &raw_at_lo_energy.p,
            &raw_at_lo_energy.q,
            raw_at_lo_energy.boundary.sra_large_component(),
        )
    }

    /// Match a homogeneous bound core of this same kappa and central potential.
    /// Reuse its two-sided shooting samples rather than reintegrating a deep
    /// bound eigenenergy outward. Sourced HF core samples are not homogeneous
    /// primitives and must not be passed to this method.
    pub fn sra_bound_core_local_orbital(
        &self,
        mesh: &ExponentialMesh,
        potential: &[f64],
        energy: Hartree,
        p: &[f64],
        q: &[f64],
    ) -> Result<DiracLocalOrbital, DiracError> {
        validate_dirac_potential(mesh, potential)?;
        for actual in [p.len(), q.len(), self.p.len()] {
            if actual != mesh.len() {
                return Err(DiracError::ArrayLength {
                    expected: mesh.len(),
                    actual,
                });
            }
        }
        let last = mesh.len() - 1;
        let radius = mesh.last().get();
        let (p_prime, _) = dirac_rhs(
            radius,
            potential[last],
            p[last],
            self.speed_of_light * q[last],
            f64::from(self.kappa.get()),
            energy.get(),
            self.speed_of_light.powi(2),
        );
        let boundary = BoundaryData::new(
            p[last] / radius,
            (p_prime - p[last] / radius) / radius,
            radius,
        );
        self.match_sra_local_orbital(mesh, energy, p, q, boundary)
    }

    fn match_sra_local_orbital(
        &self,
        mesh: &ExponentialMesh,
        energy: Hartree,
        raw_p: &[f64],
        raw_q: &[f64],
        raw: BoundaryData,
    ) -> Result<DiracLocalOrbital, DiracError> {
        let base = self.boundary.sra_large_component();
        let first = self.energy_derivative.boundary.sra_large_component();
        let determinant = base.value * first.derivative - first.value * base.derivative;
        let determinant_scale = (base.value.abs() * first.derivative.abs())
            .max(first.value.abs() * base.derivative.abs())
            .max(1.0);
        if determinant.abs() <= 256.0 * f64::EPSILON * determinant_scale {
            return Err(DiracError::SingularLocalOrbital { determinant });
        }
        let a = (-raw.value * first.derivative + first.value * raw.derivative) / determinant;
        let b = (-base.value * raw.derivative + raw.value * base.derivative) / determinant;

        let mut p: Vec<f64> = raw_p
            .iter()
            .zip(&self.p)
            .zip(&self.energy_derivative.p)
            .map(|((&raw, &base), &first)| raw + a * base + b * first)
            .collect();
        let mut q: Vec<f64> = raw_q
            .iter()
            .zip(&self.q)
            .zip(&self.energy_derivative.q)
            .map(|((&raw, &base), &first)| raw + a * base + b * first)
            .collect();
        let density: Vec<f64> = p
            .iter()
            .zip(&q)
            .map(|(&large, &small)| large * large + small * small)
            .collect();
        let norm_squared = mesh
            .integrate(&density)
            .map_err(|error| DiracError::Quadrature(error.to_string()))?;
        if !norm_squared.is_finite() || norm_squared <= f64::MIN_POSITIVE {
            return Err(DiracError::SingularNorm { norm_squared });
        }
        let normalization_scale = norm_squared.sqrt().recip();
        p.iter_mut().for_each(|value| *value *= normalization_scale);
        q.iter_mut().for_each(|value| *value *= normalization_scale);
        let value = normalization_scale * (raw.value + a * base.value + b * first.value);
        let derivative =
            normalization_scale * (raw.derivative + a * base.derivative + b * first.derivative);

        Ok(DiracLocalOrbital {
            energy,
            kappa: self.kappa,
            p,
            q,
            coefficients: LocalOrbitalCoefficients {
                a,
                b,
                normalization_scale,
            },
            boundary: BoundaryData::new(value, derivative, mesh.last().get()),
        })
    }

    /// Build the confined SRA-HDLO from the normalized second derivative.
    ///
    /// Only the large component is used for the value-and-slope boundary
    /// match.  The same coefficients are applied to both physical Dirac
    /// components before normalizing the complete spinor.
    pub fn sra_hdlo(&self, mesh: &ExponentialMesh) -> Result<DiracLocalOrbital, DiracError> {
        let expected = self.p.len();
        for actual in [
            self.q.len(),
            self.energy_derivative.p.len(),
            self.energy_derivative.q.len(),
            self.second_energy_derivative.p.len(),
            self.second_energy_derivative.q.len(),
            mesh.len(),
        ] {
            if actual != expected {
                return Err(DiracError::ArrayLength { expected, actual });
            }
        }

        let base = self.boundary.sra_large_component();
        let first = self.energy_derivative.boundary.sra_large_component();
        let raw = self.second_energy_derivative.boundary.sra_large_component();
        let determinant = base.value * first.derivative - first.value * base.derivative;
        let determinant_scale = (base.value.abs() * first.derivative.abs())
            .max(first.value.abs() * base.derivative.abs())
            .max(1.0);
        if determinant.abs() <= 256.0 * f64::EPSILON * determinant_scale {
            return Err(DiracError::SingularLocalOrbital { determinant });
        }
        let a = (-raw.value * first.derivative + first.value * raw.derivative) / determinant;
        let b = (-base.value * raw.derivative + raw.value * base.derivative) / determinant;

        let mut p: Vec<f64> = self
            .second_energy_derivative
            .p
            .iter()
            .zip(&self.p)
            .zip(&self.energy_derivative.p)
            .map(|((&raw, &base), &first)| raw + a * base + b * first)
            .collect();
        let mut q: Vec<f64> = self
            .second_energy_derivative
            .q
            .iter()
            .zip(&self.q)
            .zip(&self.energy_derivative.q)
            .map(|((&raw, &base), &first)| raw + a * base + b * first)
            .collect();
        let density: Vec<f64> = p
            .iter()
            .zip(&q)
            .map(|(&large, &small)| large * large + small * small)
            .collect();
        let norm_squared = mesh
            .integrate(&density)
            .map_err(|error| DiracError::Quadrature(error.to_string()))?;
        if !norm_squared.is_finite() || norm_squared <= f64::MIN_POSITIVE {
            return Err(DiracError::SingularNorm { norm_squared });
        }
        let normalization_scale = norm_squared.sqrt().recip();
        p.iter_mut().for_each(|value| *value *= normalization_scale);
        q.iter_mut().for_each(|value| *value *= normalization_scale);
        let value = normalization_scale * (raw.value + a * base.value + b * first.value);
        let derivative =
            normalization_scale * (raw.derivative + a * base.derivative + b * first.derivative);

        Ok(DiracLocalOrbital {
            energy: self.energy,
            kappa: self.kappa,
            p,
            q,
            coefficients: LocalOrbitalCoefficients {
                a,
                b,
                normalization_scale,
            },
            boundary: BoundaryData::new(value, derivative, mesh.last().get()),
        })
    }
}

impl CoreDiracSolution {
    /// Large radial component `P`.
    pub fn large(&self) -> &[f64] {
        &self.p
    }

    /// Physical small radial component `Q`.
    pub fn small(&self) -> &[f64] {
        &self.q
    }
}

/// Evaluate the rest-energy-subtracted local radial Dirac Hamiltonian.
///
/// `p` and `q` are the physical reduced components in the convention
/// `Psi = (P Omega_kappa, i Q Omega_-kappa) / r`. The first derivatives are
/// evaluated with the fifth-order SPEX logarithmic-mesh stencil, and the
/// returned value is the direct radial quadrature
/// `integral (P H0_P + Q H0_Q) dr`.
pub fn dirac_local_hamiltonian_expectation(
    mesh: &ExponentialMesh,
    potential: &[f64],
    kappa: Kappa,
    p: &[f64],
    q: &[f64],
) -> Result<Hartree, DiracLocalHamiltonianError> {
    for (component, values) in [("potential", potential), ("P", p), ("Q", q)] {
        if values.len() != mesh.len() {
            return Err(DiracLocalHamiltonianError::SampleCount {
                component,
                expected: mesh.len(),
                actual: values.len(),
            });
        }
        for (index, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(DiracLocalHamiltonianError::NonFiniteSample {
                    component,
                    index,
                    value,
                });
            }
        }
    }
    let p_derivative = logarithmic_mesh_derivative(mesh, p);
    let q_derivative = logarithmic_mesh_derivative(mesh, q);
    let kappa = f64::from(kappa.get());
    let integrand = mesh
        .radii()
        .iter()
        .zip(potential)
        .zip(p.iter().zip(q))
        .zip(p_derivative.iter().zip(&q_derivative))
        .map(
            |(((radius, &potential), (&p, &q)), (&p_derivative, &q_derivative))| {
                let inverse_radius = radius.get().recip();
                let h_p = potential * p
                    + SPEX_SPEED_OF_LIGHT * (-q_derivative + kappa * inverse_radius * q);
                let h_q = SPEX_SPEED_OF_LIGHT * (p_derivative + kappa * inverse_radius * p)
                    + (potential - 2.0 * C_SQUARED) * q;
                p * h_p + q * h_q
            },
        )
        .collect::<Vec<_>>();
    mesh.integrate(&integrand)
        .map(Hartree)
        .map_err(|error| DiracLocalHamiltonianError::Quadrature(error.to_string()))
}

fn logarithmic_mesh_derivative(mesh: &ExponentialMesh, values: &[f64]) -> Vec<f64> {
    let n = mesh.len();
    let h12 = 12.0 * mesh.increment();
    let mut derivative = vec![0.0; n];
    derivative[0] = (-25.0 * values[0] + 48.0 * values[1] - 36.0 * values[2] + 16.0 * values[3]
        - 3.0 * values[4])
        / (h12 * mesh.radii()[0].get());
    derivative[1] = (-3.0 * values[0] - 10.0 * values[1] + 18.0 * values[2] - 6.0 * values[3]
        + values[4])
        / (h12 * mesh.radii()[1].get());
    for index in 2..n - 2 {
        derivative[index] = (values[index - 2] - values[index + 2]
            + 8.0 * (values[index + 1] - values[index - 1]))
            / (h12 * mesh.radii()[index].get());
    }
    derivative[n - 2] = (-values[n - 5] + 6.0 * values[n - 4] - 18.0 * values[n - 3]
        + 10.0 * values[n - 2]
        + 3.0 * values[n - 1])
        / (h12 * mesh.radii()[n - 2].get());
    derivative[n - 1] = (3.0 * values[n - 5] - 16.0 * values[n - 4] + 36.0 * values[n - 3]
        - 48.0 * values[n - 2]
        + 25.0 * values[n - 1])
        / (h12 * mesh.radii()[n - 1].get());
    derivative
}

/// Invalid physical radial input or quadrature failure for a local Dirac trace.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum DiracLocalHamiltonianError {
    #[error("local Dirac Hamiltonian {component} has {actual} samples, expected {expected}")]
    SampleCount {
        component: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("local Dirac Hamiltonian {component}[{index}] is not finite: {value}")]
    NonFiniteSample {
        component: &'static str,
        index: usize,
        value: f64,
    },
    #[error("local Dirac Hamiltonian quadrature failed: {0}")]
    Quadrature(String),
}

/// Diagnosable input, shooting, and feature-boundary errors.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum DiracError {
    #[error("principal quantum number n={n} is invalid for l={l}")]
    InvalidPrincipalQuantumNumber { n: u32, l: u32 },
    #[error("energy bracket is invalid: [{lower}, {upper}] Ha")]
    InvalidEnergyBracket { lower: f64, upper: f64 },
    #[error("core bracket search window must be finite and ordered, got [{lower}, {upper}] Ha")]
    InvalidCoreSearchWindow { lower: f64, upper: f64 },
    #[error("core bracket search requires at least one interval, got {0}")]
    InvalidCoreSearchIntervals(usize),
    #[error(
        "no bracket for n={n}, kappa={kappa} was found in [{lower}, {upper}] Ha using {intervals} intervals"
    )]
    CoreBracketNotFound {
        n: u32,
        kappa: i32,
        lower: f64,
        upper: f64,
        intervals: usize,
    },
    #[error(
        "core bracket search for n={n}, kappa={kappa} found {candidates} node-compatible roots"
    )]
    CoreBracketAmbiguous {
        n: u32,
        kappa: i32,
        candidates: usize,
    },
    #[error("potential has {actual} samples, but the mesh has {expected}")]
    PotentialLength { expected: usize, actual: usize },
    #[error("potential[{index}] is not finite: {value}")]
    NonFinitePotential { index: usize, value: f64 },
    #[error(
        "core Dirac exchange action {component} has {actual} samples, but the mesh has {expected}"
    )]
    ExchangeActionLength {
        component: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("core Dirac exchange action {component}[{index}] is not finite: {value}")]
    NonFiniteExchangeAction {
        component: &'static str,
        index: usize,
        value: f64,
    },
    #[error(
        "source-driven core prediction {predicted} Ha must lie in the finite search window [{lower}, {upper}] Ha"
    )]
    InvalidSourcedPrediction {
        predicted: f64,
        lower: f64,
        upper: f64,
    },
    #[error("source-driven core search requires at least one interval, got {0}")]
    InvalidSourcedSearchIntervals(usize),
    #[error(
        "no node-compatible source-driven norm root was found in [{lower}, {upper}] Ha using {intervals} intervals"
    )]
    SourcedNormRootNotFound {
        lower: f64,
        upper: f64,
        intervals: usize,
    },
    #[error(
        "source-driven core prediction {predicted} Ha is equidistant from {candidates} node-compatible norm roots"
    )]
    SourcedNormRootAmbiguous { predicted: f64, candidates: usize },
    #[error("energy is not finite: {0}")]
    NonFiniteEnergy(f64),
    #[error("speed of light must be finite and positive, got {0}")]
    InvalidSpeedOfLight(f64),
    #[error("nuclear charge must be finite and positive, got {0}")]
    InvalidNuclearCharge(f64),
    #[error("radial array has {actual} samples, but expected {expected}")]
    ArrayLength { expected: usize, actual: usize },
    #[error("bound-core shooting requires an outward positive radial mesh")]
    InvalidMeshDirection,
    #[error("muffin-tin radius {radius} bohr is not strictly inside mesh [{first}, {last}]")]
    InvalidMuffinTinRadius { radius: f64, first: f64, last: f64 },
    #[error("muffin-tin radius {radius} bohr is not a radial mesh point")]
    MuffinTinRadiusNotOnMesh { radius: f64 },
    #[error("point-Coulomb origin is supercritical: kappa^2-(Z/c)^2={radicand}")]
    SupercriticalOrigin { radicand: f64 },
    #[error("Dirac mass factor at index {index} is non-positive or non-finite: {mass}")]
    InvalidRelativisticMass { index: usize, mass: f64 },
    #[error("outer boundary does not support exponential decay: V-E={delta} Ha")]
    NonDecayingOuterBoundary { delta: f64 },
    #[error("radial integration overflowed at mesh index {index}")]
    IntegrationOverflow { index: usize },
    #[error("matching branch is singular at mesh index {index}")]
    SingularMatch { index: usize },
    #[error("shooting residual does not change sign: f({lower})={f_lower}, f({upper})={f_upper}")]
    RootNotBracketed {
        lower: f64,
        upper: f64,
        f_lower: f64,
        f_upper: f64,
    },
    #[error("bound-core root did not converge after {iterations} iterations")]
    RootDidNotConverge { iterations: usize },
    #[error(
        "invalid shooting tolerances: energy={energy}, matching={matching}, iterations={iterations}"
    )]
    InvalidTolerance {
        energy: f64,
        matching: f64,
        iterations: usize,
    },
    #[error("converged radial function has {actual} nodes, expected {expected}")]
    NodeCountMismatch { expected: u32, actual: u32 },
    #[error("solution norm is singular or non-finite: {norm_squared}")]
    SingularNorm { norm_squared: f64 },
    #[error(
        "muffin-tin core norm {norm_mt} exceeds total norm {norm_total} beyond quadrature tolerance {tolerance}"
    )]
    InconsistentNormPartition {
        norm_total: f64,
        norm_mt: f64,
        tolerance: f64,
    },
    #[error("Dirac local-orbital boundary system is singular (determinant {determinant})")]
    SingularLocalOrbital { determinant: f64 },
    #[error("Dirac local-orbital kappa mismatch: base {base}, raw {raw}")]
    LocalOrbitalKappaMismatch { base: i32, raw: i32 },
    #[error("Dirac local-orbital speed-of-light mismatch: base {base}, raw {raw}")]
    LocalOrbitalSpeedOfLightMismatch { base: f64, raw: f64 },
    #[error("Dirac local-orbital raw energy must differ from the base energy {energy} Ha")]
    LocalOrbitalEnergyNotDistinct { energy: f64 },
    #[error("Dirac local-orbital {field} has {actual} samples, but the mesh has {mesh}")]
    LocalOrbitalSampleCountMismatch {
        field: &'static str,
        mesh: usize,
        actual: usize,
    },
    #[error(
        "Dirac local-orbital {field} radius {actual} bohr does not match the mesh boundary {mesh} bohr"
    )]
    LocalOrbitalBoundaryRadiusMismatch {
        field: &'static str,
        mesh: f64,
        actual: f64,
    },
    #[error("mesh quadrature failed: {0}")]
    Quadrature(String),
}

/// Solve one normalizable spherical four-component Dirac core state.
///
/// `potential` is the total physical spherical potential `V(r)` in Hartree
/// on every point of the extended positive `mesh`. The energy bracket must
/// isolate the requested root; a converged root with a different radial node
/// count is rejected with [`DiracError::NodeCountMismatch`].
pub fn solve_core_dirac<S: Borrow<CoreDiracSpec>>(
    mesh: &ExponentialMesh,
    potential: &[f64],
    spec: S,
) -> Result<CoreDiracSolution, DiracError> {
    let spec = spec.borrow();
    validate_inputs(mesh, potential, spec)?;

    let (mut lower, mut upper) = spec.bracket.values();
    let mut lower_shot = shoot(
        mesh,
        potential,
        spec.nuclear_charge,
        spec.state.kappa,
        lower,
        false,
    )?;
    let upper_shot = shoot(
        mesh,
        potential,
        spec.nuclear_charge,
        spec.state.kappa,
        upper,
        false,
    )?;
    if lower_shot.residual == 0.0 {
        return assemble_solution(
            mesh,
            potential,
            spec,
            lower,
            lower_shot.match_index,
            lower_shot.outer_index,
        );
    }
    if upper_shot.residual == 0.0 {
        return assemble_solution(
            mesh,
            potential,
            spec,
            upper,
            upper_shot.match_index,
            upper_shot.outer_index,
        );
    }
    if lower_shot.residual.signum() == upper_shot.residual.signum() {
        return Err(DiracError::RootNotBracketed {
            lower,
            upper,
            f_lower: lower_shot.residual,
            f_upper: upper_shot.residual,
        });
    }

    for _ in 0..spec.max_iterations {
        let energy = lower + 0.5 * (upper - lower);
        let shot = shoot(
            mesh,
            potential,
            spec.nuclear_charge,
            spec.state.kappa,
            energy,
            false,
        )?;
        if shot.residual.abs() <= spec.matching_tolerance
            && 0.5 * (upper - lower) <= spec.energy_tolerance
        {
            return assemble_solution(
                mesh,
                potential,
                spec,
                energy,
                shot.match_index,
                shot.outer_index,
            );
        }
        if shot.residual.signum() == lower_shot.residual.signum() {
            lower = energy;
            lower_shot = shot;
        } else {
            upper = energy;
        }
    }

    Err(DiracError::RootDidNotConverge {
        iterations: spec.max_iterations,
    })
}

/// Solve one bound core state with a fixed signed nonlocal exchange action.
///
/// In the internal `q_hat = c Q` representation, the supplied physical
/// action enters the first-order equations as
/// `P' = M q_hat - kappa P/r - action.q/c` and
/// `q_hat' = (V-E)P + kappa q_hat/r + action.p`. Following FlapwMBPT
/// `RADSCH`/`RADSCH_b`, the solver integrates homogeneous and source-driven
/// branches from both boundaries and combines them with the `inhomo` match.
/// The energy root is the `rad_eq` convention: the unnormalized physical
/// radial norm is one. The sourced specification scans the preserved core
/// energy window for node-compatible norm roots and selects the unique root
/// nearest its explicit physical Picard prediction.
///
/// An exactly zero action is the homogeneous problem and is evaluated by
/// [`solve_core_dirac`], preserving its root, normalization, and diagnostics.
pub fn solve_core_dirac_with_action<S: Borrow<CoreDiracSourcedSpec>>(
    mesh: &ExponentialMesh,
    potential: &[f64],
    spec: S,
    action: CoreDiracExchangeAction<'_>,
) -> Result<CoreDiracSolution, DiracError> {
    let sourced = spec.borrow();
    let spec = &sourced.core;
    validate_inputs(mesh, potential, spec)?;
    validate_sourced_spec(sourced)?;
    validate_exchange_action(mesh, action)?;
    if action.p.iter().chain(action.q).all(|&value| value == 0.0) {
        return solve_core_dirac(mesh, potential, spec);
    }

    let (window_lower, window_upper) = spec.bracket.values();
    let predicted = sourced.predicted_energy.get();
    let step = (window_upper - window_lower) / sourced.search_intervals as f64;
    let endpoint_energy = |index: usize| {
        if index == sourced.search_intervals {
            window_upper
        } else {
            window_lower + index as f64 * step
        }
    };
    let interval_distance = |index: usize| {
        let lower = endpoint_energy(index);
        let upper = endpoint_energy(index + 1);
        if predicted < lower {
            lower - predicted
        } else if predicted > upper {
            predicted - upper
        } else {
            0.0
        }
    };
    let mut intervals = (0..sourced.search_intervals).collect::<Vec<_>>();
    intervals.sort_by(|&left, &right| {
        interval_distance(left)
            .total_cmp(&interval_distance(right))
            .then_with(|| left.cmp(&right))
    });
    let mut shots = (0..=sourced.search_intervals)
        .map(|_| None)
        .collect::<Vec<Option<DrivenShot>>>();
    let mut candidates = Vec::new();
    let mut best_distance = None::<f64>;
    for interval in intervals {
        let distance = interval_distance(interval);
        if let Some(best) = best_distance {
            let competition_margin = spec
                .energy_tolerance
                .max(128.0 * f64::EPSILON * best.max(1.0));
            if distance > best + competition_margin {
                break;
            }
        }
        let lower = endpoint_energy(interval);
        let upper = endpoint_energy(interval + 1);
        if shots[interval].is_none() {
            shots[interval] = Some(shoot_with_action(
                mesh,
                potential,
                spec.nuclear_charge,
                spec.state.kappa,
                lower,
                action,
            )?);
        }
        if shots[interval + 1].is_none() {
            shots[interval + 1] = Some(shoot_with_action(
                mesh,
                potential,
                spec.nuclear_charge,
                spec.state.kappa,
                upper,
                action,
            )?);
        }
        let lower_shot = shots[interval]
            .as_ref()
            .expect("prediction-centered lower shot is initialized")
            .clone();
        let upper_shot = shots[interval + 1]
            .as_ref()
            .expect("prediction-centered upper shot is initialized")
            .clone();
        let sign_change = lower_shot.root_residual == 0.0
            || upper_shot.root_residual == 0.0
            || lower_shot.root_residual.signum() != upper_shot.root_residual.signum();
        if sign_change {
            match solve_driven_bracket(
                mesh, potential, spec, action, lower, upper, lower_shot, upper_shot,
            ) {
                Ok(solution) => {
                    let duplicate = candidates.iter().any(|candidate: &CoreDiracSolution| {
                        (candidate.energy.get() - solution.energy.get()).abs()
                            <= spec.energy_tolerance
                    });
                    if !duplicate {
                        let candidate_distance = (solution.energy.get() - predicted).abs();
                        best_distance = Some(
                            best_distance
                                .map_or(candidate_distance, |best| best.min(candidate_distance)),
                        );
                        candidates.push(solution);
                    }
                }
                Err(DiracError::NodeCountMismatch { .. }) => {}
                Err(error) => return Err(error),
            }
        }
    }

    if candidates.is_empty() {
        return Err(DiracError::SourcedNormRootNotFound {
            lower: window_lower,
            upper: window_upper,
            intervals: sourced.search_intervals,
        });
    }
    candidates.sort_by(|left, right| {
        (left.energy.get() - predicted)
            .abs()
            .total_cmp(&(right.energy.get() - predicted).abs())
    });
    if candidates.len() > 1 {
        let first = (candidates[0].energy.get() - predicted).abs();
        let second = (candidates[1].energy.get() - predicted).abs();
        let tie_tolerance = 128.0 * f64::EPSILON * first.max(second).max(1.0);
        if (first - second).abs() <= tie_tolerance {
            return Err(DiracError::SourcedNormRootAmbiguous {
                predicted,
                candidates: candidates.len(),
            });
        }
    }
    Ok(candidates.remove(0))
}

/// Isolate the unique adjacent shooting interval with the requested node count.
///
/// The caller supplies the extended mesh and its complete physical potential;
/// this helper does not construct or extrapolate an outer potential. Every
/// residual sign change is solved and accepted only when the converged `P`
/// has [`CoreState::expected_nodes`].
pub fn isolate_core_dirac_bracket<S: Borrow<CoreBracketSearch>>(
    mesh: &ExponentialMesh,
    potential: &[f64],
    search: S,
) -> Result<EnergyBracket, DiracError> {
    let search = search.borrow();
    let (lower, upper) = search.energy_window.values();
    if !lower.is_finite() || !upper.is_finite() || lower >= upper {
        return Err(DiracError::InvalidCoreSearchWindow { lower, upper });
    }
    if search.intervals == 0 {
        return Err(DiracError::InvalidCoreSearchIntervals(search.intervals));
    }
    // Reuse the production input validation, including the exact MT mesh point.
    validate_inputs(
        mesh,
        potential,
        &CoreDiracSpec::new(
            search.state,
            search.nuclear_charge,
            search.energy_window,
            search.muffin_tin_radius,
        ),
    )?;

    let step = (upper - lower) / search.intervals as f64;
    // The scan points are independent, so shoot them together and keep the
    // sign-change walk serial; the walk still sees them in energy order.
    let energies = (0..=search.intervals)
        .map(|interval| match interval {
            0 => lower,
            _ if interval == search.intervals => upper,
            _ => lower + interval as f64 * step,
        })
        .collect::<Vec<_>>();
    let shots = energies
        .par_iter()
        .map(|&energy| {
            shoot(
                mesh,
                potential,
                search.nuclear_charge,
                search.state.kappa,
                energy,
                false,
            )
        })
        .collect::<Vec<_>>();
    let mut shots = shots.into_iter();
    let mut previous_energy = lower;
    let mut previous = shots.next().expect("the scan holds at least two points")?;
    let mut candidates = Vec::new();
    let mut previous_exact_root = false;
    for (interval, shot) in (1..=search.intervals).zip(shots) {
        let energy = energies[interval];
        let current = shot?;
        let sign_change = !previous_exact_root
            && (previous.residual == 0.0
                || current.residual == 0.0
                || previous.residual.signum() != current.residual.signum());
        if sign_change {
            let bracket = EnergyBracket::from_values(previous_energy, energy)?;
            let spec = CoreDiracSpec::new(
                search.state,
                search.nuclear_charge,
                bracket,
                search.muffin_tin_radius,
            );
            match solve_core_dirac(mesh, potential, spec) {
                Ok(_) => candidates.push(bracket),
                Err(
                    DiracError::NodeCountMismatch { .. }
                    | DiracError::RootNotBracketed { .. }
                    | DiracError::RootDidNotConverge { .. },
                ) => {}
                Err(error) => return Err(error),
            }
        }
        previous_exact_root = current.residual == 0.0;
        previous_energy = energy;
        previous = current;
    }
    match candidates.as_slice() {
        [bracket] => Ok(*bracket),
        [] => Err(DiracError::CoreBracketNotFound {
            n: search.state.n,
            kappa: search.state.kappa.get(),
            lower,
            upper,
            intervals: search.intervals,
        }),
        many => Err(DiracError::CoreBracketAmbiguous {
            n: search.state.n,
            kappa: search.state.kappa.get(),
            candidates: many.len(),
        }),
    }
}

/// Integrate the regular spherical Dirac solution outward at fixed real energy.
///
/// The potential is held fixed when differentiating with respect to energy.
/// Both the solution and its analytic first energy derivative are normalized,
/// with the phase fixed by positive `P` at the first mesh point and the
/// derivative in the parallel-transport gauge `⟨R|Ṙ⟩ = 0`.
pub fn solve_valence_dirac<S: Borrow<ValenceDiracSpec>>(
    mesh: &ExponentialMesh,
    potential: &[f64],
    spec: S,
) -> Result<ValenceDiracSolution, DiracError> {
    let spec = spec.borrow();
    validate_dirac_potential(mesh, potential)?;
    let energy = spec.energy.get();
    if !energy.is_finite() {
        return Err(DiracError::NonFiniteEnergy(energy));
    }
    let speed_of_light = spec.speed_of_light;
    if !speed_of_light.is_finite() || speed_of_light <= 0.0 {
        return Err(DiracError::InvalidSpeedOfLight(speed_of_light));
    }
    let c_squared = speed_of_light * speed_of_light;
    for (index, &value) in potential.iter().enumerate() {
        let mass = 2.0 + (energy - value) / c_squared;
        if !mass.is_finite() || mass <= 0.0 {
            return Err(DiracError::InvalidRelativisticMass { index, mass });
        }
    }

    let n = mesh.len();
    let mut p = vec![0.0; n];
    let mut q_hat = vec![0.0; n];
    let mut p_dot = vec![0.0; n];
    let mut q_hat_dot = vec![0.0; n];
    let mut p_second = vec![0.0; n];
    let mut q_hat_second = vec![0.0; n];
    let initial =
        regular_valence_initial_state(mesh, potential, spec.kappa, energy, speed_of_light)?;
    let [p0, q0, p_dot0, q_dot0, p_second0, q_second0] = initial;
    p[0] = p0;
    q_hat[0] = q0;
    p_dot[0] = p_dot0;
    q_hat_dot[0] = q_dot0;
    p_second[0] = p_second0;
    q_hat_second[0] = q_second0;

    let kappa = f64::from(spec.kappa.get());
    for i in 0..n - 1 {
        let next = rk4_energy_derivatives_interval(
            mesh.radii()[i].get(),
            mesh.radii()[i + 1].get(),
            potential[i],
            potential[i + 1],
            [
                p[i],
                q_hat[i],
                p_dot[i],
                q_hat_dot[i],
                p_second[i],
                q_hat_second[i],
            ],
            kappa,
            energy,
            c_squared,
        );
        ensure_finite_state(next[0], next[1], i + 1)?;
        ensure_finite_state(next[2], next[3], i + 1)?;
        ensure_finite_state(next[4], next[5], i + 1)?;
        p[i + 1] = next[0];
        q_hat[i + 1] = next[1];
        p_dot[i + 1] = next[2];
        q_hat_dot[i + 1] = next[3];
        p_second[i + 1] = next[4];
        q_hat_second[i + 1] = next[5];
    }

    let density: Vec<f64> = p
        .iter()
        .zip(&q_hat)
        .map(|(&large, &small)| large * large + small * small / c_squared)
        .collect();
    let norm_squared = mesh
        .integrate(&density)
        .map_err(|error| DiracError::Quadrature(error.to_string()))?;
    if !norm_squared.is_finite() || norm_squared <= f64::MIN_POSITIVE {
        return Err(DiracError::SingularNorm { norm_squared });
    }
    let cross_density: Vec<f64> = p
        .iter()
        .zip(&q_hat)
        .zip(&p_dot)
        .zip(&q_hat_dot)
        .map(|(((&large, &small), &large_dot), &small_dot)| {
            large * large_dot + small * small_dot / c_squared
        })
        .collect();
    let raw_cross = mesh
        .integrate(&cross_density)
        .map_err(|error| DiracError::Quadrature(error.to_string()))?;
    let first_norm_density: Vec<f64> = p_dot
        .iter()
        .zip(&q_hat_dot)
        .map(|(&large, &small)| large * large + small * small / c_squared)
        .collect();
    let solution_second_density: Vec<f64> = p
        .iter()
        .zip(&q_hat)
        .zip(&p_second)
        .zip(&q_hat_second)
        .map(|(((&large, &small), &large_second), &small_second)| {
            large * large_second + small * small_second / c_squared
        })
        .collect();
    let d = mesh
        .integrate(&first_norm_density)
        .map_err(|error| DiracError::Quadrature(error.to_string()))?
        + mesh
            .integrate(&solution_second_density)
            .map_err(|error| DiracError::Quadrature(error.to_string()))?;
    let a = raw_cross / norm_squared;
    let scale = norm_squared.sqrt().recip();
    let second_solution_coefficient = 3.0 * a * a - d / norm_squared;
    for i in 0..n {
        let raw_p = p[i];
        let raw_q = q_hat[i];
        let raw_p_dot = p_dot[i];
        let raw_q_dot = q_hat_dot[i];
        p[i] = scale * raw_p;
        q_hat[i] = scale * raw_q;
        p_dot[i] = scale * (raw_p_dot - a * raw_p);
        q_hat_dot[i] = scale * (raw_q_dot - a * raw_q);
        p_second[i] =
            scale * (p_second[i] - 2.0 * a * raw_p_dot + second_solution_coefficient * raw_p);
        q_hat_second[i] =
            scale * (q_hat_second[i] - 2.0 * a * raw_q_dot + second_solution_coefficient * raw_q);
    }

    let q: Vec<f64> = q_hat.iter().map(|&value| value / speed_of_light).collect();
    let q_dot: Vec<f64> = q_hat_dot
        .iter()
        .map(|&value| value / speed_of_light)
        .collect();
    let q_second: Vec<f64> = q_hat_second
        .iter()
        .map(|&value| value / speed_of_light)
        .collect();
    let radius = mesh.last().get();
    let potential_boundary = potential[n - 1];
    let (p_prime, q_hat_prime) = dirac_rhs(
        radius,
        potential_boundary,
        p[n - 1],
        q_hat[n - 1],
        kappa,
        energy,
        c_squared,
    );
    let derivatives = dirac_energy_derivatives_rhs(
        radius,
        potential_boundary,
        [
            p[n - 1],
            q_hat[n - 1],
            p_dot[n - 1],
            q_hat_dot[n - 1],
            p_second[n - 1],
            q_hat_second[n - 1],
        ],
        kappa,
        energy,
        c_squared,
    );
    let [
        p_dot_prime,
        q_hat_dot_prime,
        p_second_prime,
        q_hat_second_prime,
    ] = derivatives;
    let boundary = DiracBoundaryTrace {
        radius: mesh.last(),
        p: p[n - 1],
        q: q[n - 1],
        p_derivative: p_prime,
        q_derivative: q_hat_prime / speed_of_light,
    };
    let derivative_boundary = DiracBoundaryTrace {
        radius: mesh.last(),
        p: p_dot[n - 1],
        q: q_dot[n - 1],
        p_derivative: p_dot_prime,
        q_derivative: q_hat_dot_prime / speed_of_light,
    };
    let second_derivative_boundary = DiracBoundaryTrace {
        radius: mesh.last(),
        p: p_second[n - 1],
        q: q_second[n - 1],
        p_derivative: p_second_prime,
        q_derivative: q_hat_second_prime / speed_of_light,
    };
    let derivative_density: Vec<f64> = p_dot
        .iter()
        .zip(&q_dot)
        .map(|(&large, &small)| large * large + small * small)
        .collect();
    let derivative_norm_squared = mesh
        .integrate(&derivative_density)
        .map_err(|error| DiracError::Quadrature(error.to_string()))?;
    let second_derivative_density: Vec<f64> = p_second
        .iter()
        .zip(&q_second)
        .map(|(&large, &small)| large * large + small * small)
        .collect();
    let second_derivative_norm_squared = mesh
        .integrate(&second_derivative_density)
        .map_err(|error| DiracError::Quadrature(error.to_string()))?;
    let physical_density: Vec<f64> = p
        .iter()
        .zip(&q)
        .map(|(&large, &small)| large * large + small * small)
        .collect();
    let norm_total = mesh
        .integrate(&physical_density)
        .map_err(|error| DiracError::Quadrature(error.to_string()))?;

    Ok(ValenceDiracSolution {
        role: RelativisticRole::Valence,
        kappa: spec.kappa,
        angular: spec.kappa.angular_contract(),
        energy: spec.energy,
        speed_of_light,
        p,
        q,
        boundary,
        energy_derivative: DiracEnergyDerivative {
            p: p_dot,
            q: q_dot,
            boundary: derivative_boundary,
            norm_squared: derivative_norm_squared,
        },
        second_energy_derivative: DiracSecondEnergyDerivative {
            p: p_second,
            q: q_second,
            boundary: second_derivative_boundary,
            norm_squared: second_derivative_norm_squared,
        },
        norm_total,
    })
}

fn validate_dirac_potential(mesh: &ExponentialMesh, potential: &[f64]) -> Result<(), DiracError> {
    if potential.len() != mesh.len() {
        return Err(DiracError::PotentialLength {
            expected: mesh.len(),
            actual: potential.len(),
        });
    }
    if let Some((index, &value)) = potential
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(DiracError::NonFinitePotential { index, value });
    }
    if mesh.increment() <= 0.0 {
        return Err(DiracError::InvalidMeshDirection);
    }
    Ok(())
}

fn regular_valence_initial_state(
    mesh: &ExponentialMesh,
    potential: &[f64],
    kappa: Kappa,
    energy: f64,
    speed_of_light: f64,
) -> Result<[f64; 6], DiracError> {
    let sample_count = mesh.len().min(4);
    let z = mesh.radii()[..sample_count]
        .iter()
        .zip(&potential[..sample_count])
        .map(|(r, &v)| -r.get() * v)
        .sum::<f64>()
        / sample_count as f64;
    let r = mesh.first().get();
    let k = f64::from(kappa.get());
    let c_squared = speed_of_light * speed_of_light;
    let mass = 2.0 + (energy - potential[0]) / c_squared;
    debug_assert!(mass.is_finite() && mass > 0.0);
    let p = 1.0;
    let p_dot = 0.0;
    if z > 1.0e-8 {
        let radicand = k * k - (z / speed_of_light).powi(2);
        if !radicand.is_finite() || radicand <= 0.0 {
            return Err(DiracError::SupercriticalOrigin { radicand });
        }
        let gamma = radicand.sqrt();
        let q_hat = p * (gamma + k) / (mass * r);
        let q_hat_dot = -q_hat / (mass * c_squared);
        let q_hat_second = 2.0 * q_hat / (mass * mass * c_squared * c_squared);
        Ok([p, q_hat, p_dot, q_hat_dot, 0.0, q_hat_second])
    } else if k > 0.0 {
        let q_hat = p * (2.0 * k + 1.0) / (mass * r);
        let q_hat_dot = -q_hat / (mass * c_squared);
        let q_hat_second = 2.0 * q_hat / (mass * mass * c_squared * c_squared);
        Ok([p, q_hat, p_dot, q_hat_dot, 0.0, q_hat_second])
    } else {
        let l = f64::from(kappa.large_l());
        let denominator = 2.0 * l + 3.0;
        let q_hat = (potential[0] - energy) * p * r / denominator;
        let q_hat_dot = -p * r / denominator;
        Ok([p, q_hat, p_dot, q_hat_dot, 0.0, 0.0])
    }
}

#[derive(Clone, Copy, Debug)]
struct Shot {
    residual: f64,
    match_index: usize,
    outer_index: usize,
}

#[derive(Clone, Debug)]
struct DrivenShot {
    root_residual: f64,
    match_index: usize,
    p: Vec<f64>,
    q_hat: Vec<f64>,
    matching_residual: f64,
}

fn validate_inputs(
    mesh: &ExponentialMesh,
    potential: &[f64],
    spec: &CoreDiracSpec,
) -> Result<(), DiracError> {
    spec.bracket.validate()?;
    if !spec.nuclear_charge.is_finite() || spec.nuclear_charge <= 0.0 {
        return Err(DiracError::InvalidNuclearCharge(spec.nuclear_charge));
    }
    if potential.len() != mesh.len() {
        return Err(DiracError::PotentialLength {
            expected: mesh.len(),
            actual: potential.len(),
        });
    }
    if let Some((index, &value)) = potential
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(DiracError::NonFinitePotential { index, value });
    }
    if mesh.increment() <= 0.0 {
        return Err(DiracError::InvalidMeshDirection);
    }
    let first = mesh.first().get();
    let last = mesh.last().get();
    let radius = spec.muffin_tin_radius.get();
    if !radius.is_finite() || radius < first || radius >= last {
        return Err(DiracError::InvalidMuffinTinRadius {
            radius,
            first,
            last,
        });
    }
    locate_muffin_tin_index(mesh, spec.muffin_tin_radius)?;
    if !spec.energy_tolerance.is_finite()
        || spec.energy_tolerance <= 0.0
        || !spec.matching_tolerance.is_finite()
        || spec.matching_tolerance <= 0.0
        || spec.max_iterations == 0
    {
        return Err(DiracError::InvalidTolerance {
            energy: spec.energy_tolerance,
            matching: spec.matching_tolerance,
            iterations: spec.max_iterations,
        });
    }
    let minimum = spec.state.kappa.large_l() + 1;
    if spec.state.n < minimum {
        return Err(DiracError::InvalidPrincipalQuantumNumber {
            n: spec.state.n,
            l: spec.state.kappa.large_l(),
        });
    }
    Ok(())
}

fn validate_exchange_action(
    mesh: &ExponentialMesh,
    action: CoreDiracExchangeAction<'_>,
) -> Result<(), DiracError> {
    for (component, values) in [("p", action.p), ("q", action.q)] {
        if values.len() != mesh.len() {
            return Err(DiracError::ExchangeActionLength {
                component,
                expected: mesh.len(),
                actual: values.len(),
            });
        }
        if let Some((index, &value)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(DiracError::NonFiniteExchangeAction {
                component,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn validate_sourced_spec(spec: &CoreDiracSourcedSpec) -> Result<(), DiracError> {
    let predicted = spec.predicted_energy.get();
    let (lower, upper) = spec.core.bracket.values();
    if !predicted.is_finite() || predicted < lower || predicted > upper {
        return Err(DiracError::InvalidSourcedPrediction {
            predicted,
            lower,
            upper,
        });
    }
    if spec.search_intervals == 0 {
        return Err(DiracError::InvalidSourcedSearchIntervals(
            spec.search_intervals,
        ));
    }
    Ok(())
}

fn shoot(
    mesh: &ExponentialMesh,
    potential: &[f64],
    nuclear_charge: f64,
    kappa: Kappa,
    energy: f64,
    keep_arrays: bool,
) -> Result<Shot, DiracError> {
    let match_index = select_match_index(mesh, potential, energy);
    let outer_index = select_outer_index(mesh, match_index);
    let outward = integrate_outward(
        mesh,
        potential,
        nuclear_charge,
        kappa,
        energy,
        match_index,
        keep_arrays,
    )?;
    let inward = integrate_inward(
        mesh,
        potential,
        kappa,
        energy,
        match_index,
        outer_index,
        keep_arrays,
    )?;
    let (po, qo) = outward.at_match(match_index);
    let (pi, qi) = inward.at_match(match_index);
    let out_norm = po.hypot(qo);
    let in_norm = pi.hypot(qi);
    if out_norm <= f64::MIN_POSITIVE || in_norm <= f64::MIN_POSITIVE {
        return Err(DiracError::SingularMatch { index: match_index });
    }
    let residual = (po * qi - qo * pi) / (out_norm * in_norm);
    if !residual.is_finite() {
        return Err(DiracError::SingularMatch { index: match_index });
    }
    Ok(Shot {
        residual,
        match_index,
        outer_index,
    })
}

fn shoot_with_action(
    mesh: &ExponentialMesh,
    potential: &[f64],
    nuclear_charge: f64,
    kappa: Kappa,
    energy: f64,
    action: CoreDiracExchangeAction<'_>,
) -> Result<DrivenShot, DiracError> {
    let match_index = select_match_index(mesh, potential, energy);
    let outer_index = select_outer_index(mesh, match_index);
    let homogeneous_out = integrate_outward(
        mesh,
        potential,
        nuclear_charge,
        kappa,
        energy,
        match_index,
        true,
    )?;
    let homogeneous_in = integrate_inward(
        mesh,
        potential,
        kappa,
        energy,
        match_index,
        outer_index,
        true,
    )?;
    // FlapwMBPT RADSCH/RADSCH_b seed the source-driven branch with an
    // arbitrary homogeneous piece, which cancels in `inhomo`. Use the
    // equivalent zero-seeded particular branch here: the homogeneous core
    // integrator deliberately starts from unit P at r_0, and subtracting that
    // exponentially amplified arbitrary piece would lose source precision.
    let driven_out = integrate_outward_with_action(
        mesh,
        potential,
        kappa,
        energy,
        match_index,
        action,
        (0.0, 0.0),
    )?;
    let driven_in = integrate_inward_with_action(
        mesh,
        potential,
        kappa,
        energy,
        match_index,
        outer_index,
        action,
        (0.0, 0.0),
    )?;

    // This is the two-component affine match in FlapwMBPT `inhomo`.
    let ph_out = homogeneous_out.p[match_index];
    let qh_out = homogeneous_out.q_hat[match_index];
    let ph_in = homogeneous_in.p[match_index];
    let qh_in = homogeneous_in.q_hat[match_index];
    let pi_out = driven_out.p[match_index];
    let qi_out = driven_out.q_hat[match_index];
    let pi_in = driven_in.p[match_index];
    let qi_in = driven_in.q_hat[match_index];
    let determinant = qh_out * ph_in - ph_out * qh_in;
    if !determinant.is_finite() || determinant.abs() <= f64::MIN_POSITIVE {
        return Err(DiracError::SingularMatch { index: match_index });
    }
    let delta_p = pi_in - pi_out;
    let delta_q = qi_in - qi_out;
    let outward_scale = (ph_in * delta_q - delta_p * qh_in) / determinant;
    let inward_scale = (ph_out * delta_q - delta_p * qh_out) / determinant;
    if !outward_scale.is_finite() || !inward_scale.is_finite() {
        return Err(DiracError::SingularMatch { index: match_index });
    }

    let mut p = driven_out.p;
    let mut q_hat = driven_out.q_hat;
    for index in 0..=match_index {
        p[index] += outward_scale * homogeneous_out.p[index];
        q_hat[index] += outward_scale * homogeneous_out.q_hat[index];
        ensure_finite_state(p[index], q_hat[index], index)?;
    }
    for index in match_index + 1..=outer_index {
        p[index] = driven_in.p[index] + inward_scale * homogeneous_in.p[index];
        q_hat[index] = driven_in.q_hat[index] + inward_scale * homogeneous_in.q_hat[index];
        ensure_finite_state(p[index], q_hat[index], index)?;
    }

    let matched_out = (
        pi_out + outward_scale * ph_out,
        qi_out + outward_scale * qh_out,
    );
    let matched_in = (pi_in + inward_scale * ph_in, qi_in + inward_scale * qh_in);
    let match_norm = matched_out.0.hypot(matched_out.1) * matched_in.0.hypot(matched_in.1);
    if !match_norm.is_finite() || match_norm <= f64::MIN_POSITIVE {
        return Err(DiracError::SingularMatch { index: match_index });
    }
    let matching_residual =
        (matched_out.0 * matched_in.1 - matched_out.1 * matched_in.0) / match_norm;
    let density = p
        .iter()
        .zip(&q_hat)
        .map(|(&large, &small_scaled)| large * large + small_scaled * small_scaled / C_SQUARED)
        .collect::<Vec<_>>();
    let norm_squared = mesh
        .integrate(&density)
        .map_err(|error| DiracError::Quadrature(error.to_string()))?;
    if !norm_squared.is_finite() || norm_squared <= f64::MIN_POSITIVE {
        return Err(DiracError::SingularNorm { norm_squared });
    }

    Ok(DrivenShot {
        root_residual: norm_squared - 1.0,
        match_index,
        p,
        q_hat,
        matching_residual,
    })
}

#[allow(clippy::too_many_arguments)]
fn solve_driven_bracket(
    mesh: &ExponentialMesh,
    potential: &[f64],
    spec: &CoreDiracSpec,
    action: CoreDiracExchangeAction<'_>,
    mut lower: f64,
    mut upper: f64,
    mut lower_shot: DrivenShot,
    mut upper_shot: DrivenShot,
) -> Result<CoreDiracSolution, DiracError> {
    if lower_shot.root_residual == 0.0 {
        return finalize_driven_solution(mesh, spec, lower, lower_shot);
    }
    if upper_shot.root_residual == 0.0 {
        return finalize_driven_solution(mesh, spec, upper, upper_shot);
    }
    if lower_shot.root_residual.signum() == upper_shot.root_residual.signum() {
        return Err(DiracError::RootNotBracketed {
            lower,
            upper,
            f_lower: lower_shot.root_residual,
            f_upper: upper_shot.root_residual,
        });
    }

    for _ in 0..spec.max_iterations {
        let energy = lower + 0.5 * (upper - lower);
        let shot = shoot_with_action(
            mesh,
            potential,
            spec.nuclear_charge,
            spec.state.kappa,
            energy,
            action,
        )?;
        if shot.root_residual == 0.0 {
            return finalize_driven_solution(mesh, spec, energy, shot);
        }
        if shot.root_residual.signum() == lower_shot.root_residual.signum() {
            lower = energy;
            lower_shot = shot;
        } else {
            upper = energy;
            upper_shot = shot;
        }
        // FlapwMBPT `rad_eq` converges the source-driven norm root from the
        // energy step, not the homogeneous Wronskian tolerance. Retain the
        // endpoint with the smaller |norm - 1| so the final normalization
        // perturbs the fixed-source equation as little as the bracket allows.
        if 0.5 * (upper - lower) <= spec.energy_tolerance {
            return if lower_shot.root_residual.abs() <= upper_shot.root_residual.abs() {
                finalize_driven_solution(mesh, spec, lower, lower_shot)
            } else {
                finalize_driven_solution(mesh, spec, upper, upper_shot)
            };
        }
    }

    Err(DiracError::RootDidNotConverge {
        iterations: spec.max_iterations,
    })
}

#[derive(Clone, Debug)]
struct Branch {
    p: Vec<f64>,
    q_hat: Vec<f64>,
    endpoint: (f64, f64),
}

impl Branch {
    fn at_match(&self, match_index: usize) -> (f64, f64) {
        if self.p.is_empty() {
            self.endpoint
        } else {
            (self.p[match_index], self.q_hat[match_index])
        }
    }
}

fn integrate_outward(
    mesh: &ExponentialMesh,
    potential: &[f64],
    nuclear_charge: f64,
    kappa: Kappa,
    energy: f64,
    stop: usize,
    keep_arrays: bool,
) -> Result<Branch, DiracError> {
    let n = mesh.len();
    let mut p = if keep_arrays {
        vec![0.0; n]
    } else {
        Vec::new()
    };
    let mut q_hat = if keep_arrays {
        vec![0.0; n]
    } else {
        Vec::new()
    };
    let k = f64::from(kappa.get());
    let radicand = k * k - (nuclear_charge / SPEX_SPEED_OF_LIGHT).powi(2);
    if !radicand.is_finite() || radicand <= 0.0 {
        return Err(DiracError::SupercriticalOrigin { radicand });
    }
    let gamma = radicand.sqrt();
    // The arbitrary common amplitude avoids underflow for large |kappa|.
    // The finite-grid first-equation relation supplies the regular Coulomb
    // eigenvector without dropping the nonsingular terms at r_0.
    let mut current_p = 1.0;
    let mass_factor_origin = 2.0 + (energy - potential[0]) / C_SQUARED;
    let mut current_q = (gamma + k) / (mass_factor_origin * mesh.first().get());
    if keep_arrays {
        p[0] = current_p;
        q_hat[0] = current_q;
    }
    for i in 0..stop {
        (current_p, current_q) = rk4_interval(
            mesh.radii()[i].get(),
            mesh.radii()[i + 1].get(),
            potential[i],
            potential[i + 1],
            current_p,
            current_q,
            k,
            energy,
        );
        ensure_finite_state(current_p, current_q, i + 1)?;
        if !keep_arrays {
            // Shooting uses only the endpoint direction of this homogeneous
            // equation. Remove its arbitrary growing amplitude on long trial
            // branches; physical arrays retain their common normalization.
            let scale = current_p.hypot(current_q).max(1.0);
            current_p /= scale;
            current_q /= scale;
        }
        if keep_arrays {
            p[i + 1] = current_p;
            q_hat[i + 1] = current_q;
        }
    }
    Ok(Branch {
        p,
        q_hat,
        endpoint: (current_p, current_q),
    })
}

fn integrate_inward(
    mesh: &ExponentialMesh,
    potential: &[f64],
    kappa: Kappa,
    energy: f64,
    stop: usize,
    outer_index: usize,
    keep_arrays: bool,
) -> Result<Branch, DiracError> {
    let n = mesh.len();
    let mut p = if keep_arrays {
        vec![0.0; n]
    } else {
        Vec::new()
    };
    let mut q_hat = if keep_arrays {
        vec![0.0; n]
    } else {
        Vec::new()
    };
    let delta = potential[outer_index] - energy;
    let mass_factor = 2.0 - delta / C_SQUARED;
    let decay_squared = mass_factor * delta;
    if !decay_squared.is_finite() || decay_squared <= 0.0 {
        return Err(DiracError::NonDecayingOuterBoundary { delta });
    }
    let decay = decay_squared.sqrt();
    let mut current_p = 1.0;
    let mut current_q = -decay / mass_factor;
    if keep_arrays {
        p[outer_index] = current_p;
        q_hat[outer_index] = current_q;
    }
    let k = f64::from(kappa.get());
    for i in (stop + 1..=outer_index).rev() {
        (current_p, current_q) = rk4_interval(
            mesh.radii()[i].get(),
            mesh.radii()[i - 1].get(),
            potential[i],
            potential[i - 1],
            current_p,
            current_q,
            k,
            energy,
        );
        ensure_finite_state(current_p, current_q, i - 1)?;
        if !keep_arrays {
            let scale = current_p.hypot(current_q).max(1.0);
            current_p /= scale;
            current_q /= scale;
        }
        if keep_arrays {
            p[i - 1] = current_p;
            q_hat[i - 1] = current_q;
        }
    }
    Ok(Branch {
        p,
        q_hat,
        endpoint: (current_p, current_q),
    })
}

#[allow(clippy::too_many_arguments)]
fn integrate_outward_with_action(
    mesh: &ExponentialMesh,
    potential: &[f64],
    kappa: Kappa,
    energy: f64,
    stop: usize,
    action: CoreDiracExchangeAction<'_>,
    initial: (f64, f64),
) -> Result<Branch, DiracError> {
    let mut p = vec![0.0; mesh.len()];
    let mut q_hat = vec![0.0; mesh.len()];
    let (mut current_p, mut current_q) = initial;
    p[0] = current_p;
    q_hat[0] = current_q;
    let kappa = f64::from(kappa.get());
    for index in 0..stop {
        let action_p_mid = source_action_midpoint(action.p, index, false);
        let action_q_mid = source_action_midpoint(action.q, index, false);
        (current_p, current_q) = rk4_interval_with_action(
            mesh.radii()[index].get(),
            mesh.radii()[index + 1].get(),
            potential[index],
            potential[index + 1],
            action.p[index],
            action_p_mid,
            action.p[index + 1],
            action.q[index],
            action_q_mid,
            action.q[index + 1],
            current_p,
            current_q,
            kappa,
            energy,
        );
        ensure_finite_state(current_p, current_q, index + 1)?;
        p[index + 1] = current_p;
        q_hat[index + 1] = current_q;
    }
    Ok(Branch {
        p,
        q_hat,
        endpoint: (current_p, current_q),
    })
}

#[allow(clippy::too_many_arguments)]
fn integrate_inward_with_action(
    mesh: &ExponentialMesh,
    potential: &[f64],
    kappa: Kappa,
    energy: f64,
    stop: usize,
    outer_index: usize,
    action: CoreDiracExchangeAction<'_>,
    initial: (f64, f64),
) -> Result<Branch, DiracError> {
    let mut p = vec![0.0; mesh.len()];
    let mut q_hat = vec![0.0; mesh.len()];
    let (mut current_p, mut current_q) = initial;
    p[outer_index] = current_p;
    q_hat[outer_index] = current_q;
    let kappa = f64::from(kappa.get());
    for index in (stop + 1..=outer_index).rev() {
        let interval = index - 1;
        let terminal = index == outer_index;
        let action_p_mid = source_action_midpoint(action.p, interval, terminal);
        let action_q_mid = source_action_midpoint(action.q, interval, terminal);
        (current_p, current_q) = rk4_interval_with_action(
            mesh.radii()[index].get(),
            mesh.radii()[index - 1].get(),
            potential[index],
            potential[index - 1],
            action.p[index],
            action_p_mid,
            action.p[index - 1],
            action.q[index],
            action_q_mid,
            action.q[index - 1],
            current_p,
            current_q,
            kappa,
            energy,
        );
        ensure_finite_state(current_p, current_q, index - 1)?;
        p[index - 1] = current_p;
        q_hat[index - 1] = current_q;
    }
    Ok(Branch {
        p,
        q_hat,
        endpoint: (current_p, current_q),
    })
}

fn source_action_midpoint(values: &[f64], interval: usize, terminal: bool) -> f64 {
    if interval > 0 && interval + 2 < values.len() && !terminal {
        // FlapwMBPT RADSCH/RADSCH_b midpoint interpolation on the uniform
        // exponential-mesh index. This retains fourth-order source accuracy.
        (9.0 * (values[interval] + values[interval + 1])
            - values[interval - 1]
            - values[interval + 2])
            / 16.0
    } else {
        0.5 * (values[interval] + values[interval + 1])
    }
}

#[allow(clippy::too_many_arguments)]
fn rk4_interval(
    ra: f64,
    rc: f64,
    va: f64,
    vc: f64,
    p: f64,
    q_hat: f64,
    kappa: f64,
    energy: f64,
) -> (f64, f64) {
    let rb = 0.5 * (ra + rc);
    let dr = rc - ra;
    // Interpolating rV is exact for a Coulomb singularity.
    let vb = (ra * va + rc * vc) / (ra + rc);
    let (k1, l1) = dirac_rhs(ra, va, p, q_hat, kappa, energy, C_SQUARED);
    let (k2, l2) = dirac_rhs(
        rb,
        vb,
        p + 0.5 * dr * k1,
        q_hat + 0.5 * dr * l1,
        kappa,
        energy,
        C_SQUARED,
    );
    let (k3, l3) = dirac_rhs(
        rb,
        vb,
        p + 0.5 * dr * k2,
        q_hat + 0.5 * dr * l2,
        kappa,
        energy,
        C_SQUARED,
    );
    let (k4, l4) = dirac_rhs(
        rc,
        vc,
        p + dr * k3,
        q_hat + dr * l3,
        kappa,
        energy,
        C_SQUARED,
    );
    (
        p + dr * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0,
        q_hat + dr * (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0,
    )
}

#[allow(clippy::too_many_arguments)]
fn rk4_interval_with_action(
    ra: f64,
    rc: f64,
    va: f64,
    vc: f64,
    action_pa: f64,
    action_pb: f64,
    action_pc: f64,
    action_qa: f64,
    action_qb: f64,
    action_qc: f64,
    p: f64,
    q_hat: f64,
    kappa: f64,
    energy: f64,
) -> (f64, f64) {
    let rb = 0.5 * (ra + rc);
    let dr = rc - ra;
    let vb = (ra * va + rc * vc) / (ra + rc);
    let rhs = |radius, potential, action_p, action_q, p, q_hat| {
        dirac_rhs_with_action(
            radius, potential, action_p, action_q, p, q_hat, kappa, energy, C_SQUARED,
        )
    };
    let (k1, l1) = rhs(ra, va, action_pa, action_qa, p, q_hat);
    let (k2, l2) = rhs(
        rb,
        vb,
        action_pb,
        action_qb,
        p + 0.5 * dr * k1,
        q_hat + 0.5 * dr * l1,
    );
    let (k3, l3) = rhs(
        rb,
        vb,
        action_pb,
        action_qb,
        p + 0.5 * dr * k2,
        q_hat + 0.5 * dr * l2,
    );
    let (k4, l4) = rhs(rc, vc, action_pc, action_qc, p + dr * k3, q_hat + dr * l3);
    (
        p + dr * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0,
        q_hat + dr * (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0,
    )
}

fn dirac_rhs(
    radius: f64,
    potential: f64,
    p: f64,
    q_hat: f64,
    kappa: f64,
    energy: f64,
    c_squared: f64,
) -> (f64, f64) {
    let mass_factor = 2.0 + (energy - potential) / c_squared;
    (
        mass_factor * q_hat - kappa * p / radius,
        (potential - energy) * p + kappa * q_hat / radius,
    )
}

#[allow(clippy::too_many_arguments)]
fn dirac_rhs_with_action(
    radius: f64,
    potential: f64,
    action_p: f64,
    action_q: f64,
    p: f64,
    q_hat: f64,
    kappa: f64,
    energy: f64,
    c_squared: f64,
) -> (f64, f64) {
    let (p_derivative, q_hat_derivative) =
        dirac_rhs(radius, potential, p, q_hat, kappa, energy, c_squared);
    (
        p_derivative - action_q / c_squared.sqrt(),
        q_hat_derivative + action_p,
    )
}

fn dirac_energy_derivatives_rhs(
    radius: f64,
    potential: f64,
    state: [f64; 6],
    kappa: f64,
    energy: f64,
    c_squared: f64,
) -> [f64; 4] {
    let [p, q_hat, p_dot, q_hat_dot, p_second, q_hat_second] = state;
    let mass_factor = 2.0 + (energy - potential) / c_squared;
    [
        mass_factor * q_hat_dot + q_hat / c_squared - kappa * p_dot / radius,
        (potential - energy) * p_dot - p + kappa * q_hat_dot / radius,
        mass_factor * q_hat_second + 2.0 * q_hat_dot / c_squared - kappa * p_second / radius,
        (potential - energy) * p_second - 2.0 * p_dot + kappa * q_hat_second / radius,
    ]
}

#[allow(clippy::too_many_arguments)]
fn rk4_energy_derivatives_interval(
    ra: f64,
    rc: f64,
    va: f64,
    vc: f64,
    state: [f64; 6],
    kappa: f64,
    energy: f64,
    c_squared: f64,
) -> [f64; 6] {
    let rb = 0.5 * (ra + rc);
    let dr = rc - ra;
    let vb = (ra * va + rc * vc) / (ra + rc);
    let rhs = |radius: f64, potential: f64, x: [f64; 6]| {
        let (p_prime, q_prime) = dirac_rhs(radius, potential, x[0], x[1], kappa, energy, c_squared);
        let [p_dot_prime, q_dot_prime, p_second_prime, q_second_prime] =
            dirac_energy_derivatives_rhs(radius, potential, x, kappa, energy, c_squared);
        [
            p_prime,
            q_prime,
            p_dot_prime,
            q_dot_prime,
            p_second_prime,
            q_second_prime,
        ]
    };
    let add = |x: [f64; 6], a: f64, dx: [f64; 6]| {
        [
            x[0] + a * dx[0],
            x[1] + a * dx[1],
            x[2] + a * dx[2],
            x[3] + a * dx[3],
            x[4] + a * dx[4],
            x[5] + a * dx[5],
        ]
    };
    let k1 = rhs(ra, va, state);
    let k2 = rhs(rb, vb, add(state, 0.5 * dr, k1));
    let k3 = rhs(rb, vb, add(state, 0.5 * dr, k2));
    let k4 = rhs(rc, vc, add(state, dr, k3));
    [
        state[0] + dr * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + dr * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        state[2] + dr * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        state[3] + dr * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        state[4] + dr * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0,
        state[5] + dr * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]) / 6.0,
    ]
}

fn ensure_finite_state(p: f64, q_hat: f64, index: usize) -> Result<(), DiracError> {
    if p.is_finite()
        && q_hat.is_finite()
        && p.abs() <= crate::MAX_RADIAL_AMPLITUDE
        && q_hat.abs() <= crate::MAX_RADIAL_AMPLITUDE
    {
        Ok(())
    } else {
        Err(DiracError::IntegrationOverflow { index })
    }
}

fn select_match_index(mesh: &ExponentialMesh, potential: &[f64], energy: f64) -> usize {
    // Stay clear of both asymptotic initializations.  The point closest to a
    // classical turning point gives two well-conditioned branch amplitudes.
    let first = 2.min(mesh.len() - 2);
    let last = mesh.len() - 2;
    (first..=last)
        .min_by(|&a, &b| {
            (potential[a] - energy)
                .abs()
                .total_cmp(&(potential[b] - energy).abs())
        })
        .expect("match-point search range over a nonempty mesh is nonempty")
}

fn select_outer_index(mesh: &ExponentialMesh, match_index: usize) -> usize {
    // Seed the decaying branch far enough beyond the state-dependent turning
    // point without integrating the growing reverse solution across the
    // unused remainder of an arbitrarily extended user mesh.
    let target = 15.0 * mesh.radii()[match_index].get();
    mesh.radii()
        .iter()
        .position(|radius| radius.get() >= target)
        .unwrap_or(mesh.len() - 1)
}

fn locate_muffin_tin_index(
    mesh: &ExponentialMesh,
    muffin_tin_radius: Bohr,
) -> Result<usize, DiracError> {
    let requested = muffin_tin_radius.get();
    let (index, actual) = mesh
        .radii()
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (a.get() - requested)
                .abs()
                .total_cmp(&(b.get() - requested).abs())
        })
        .expect("a constructed exponential mesh is nonempty");
    let tolerance = 128.0 * f64::EPSILON * requested.abs().max(1.0);
    if index < 6 || (actual.get() - requested).abs() > tolerance {
        Err(DiracError::MuffinTinRadiusNotOnMesh { radius: requested })
    } else {
        Ok(index)
    }
}

fn assemble_solution(
    mesh: &ExponentialMesh,
    potential: &[f64],
    spec: &CoreDiracSpec,
    energy: f64,
    match_index: usize,
    outer_index: usize,
) -> Result<CoreDiracSolution, DiracError> {
    let outward = integrate_outward(
        mesh,
        potential,
        spec.nuclear_charge,
        spec.state.kappa,
        energy,
        match_index,
        true,
    )?;
    let inward = integrate_inward(
        mesh,
        potential,
        spec.state.kappa,
        energy,
        match_index,
        outer_index,
        true,
    )?;
    let po = outward.p[match_index];
    let qo = outward.q_hat[match_index];
    let pi = inward.p[match_index];
    let qi = inward.q_hat[match_index];
    let denominator = po.hypot(qo) * pi.hypot(qi);
    if !denominator.is_finite() || denominator <= f64::MIN_POSITIVE {
        return Err(DiracError::SingularMatch { index: match_index });
    }
    let residual = (po * qi - qo * pi) / denominator;
    // Enforce P continuity using one common scale for both inward components.
    let scale_in = if pi.abs() > 64.0 * f64::EPSILON * pi.hypot(qi) {
        po / pi
    } else if qi.abs() > 64.0 * f64::EPSILON * pi.hypot(qi) {
        qo / qi
    } else {
        return Err(DiracError::SingularMatch { index: match_index });
    };

    let mut p = outward.p;
    let mut q_hat = outward.q_hat;
    for i in match_index + 1..=outer_index {
        p[i] = scale_in * inward.p[i];
        q_hat[i] = scale_in * inward.q_hat[i];
    }
    finalize_core_solution(mesh, spec, energy, match_index, residual, p, q_hat)
}

fn finalize_driven_solution(
    mesh: &ExponentialMesh,
    spec: &CoreDiracSpec,
    energy: f64,
    shot: DrivenShot,
) -> Result<CoreDiracSolution, DiracError> {
    finalize_core_solution(
        mesh,
        spec,
        energy,
        shot.match_index,
        shot.matching_residual,
        shot.p,
        shot.q_hat,
    )
}

#[allow(clippy::too_many_arguments)]
fn finalize_core_solution(
    mesh: &ExponentialMesh,
    spec: &CoreDiracSpec,
    energy: f64,
    match_index: usize,
    matching_residual: f64,
    mut p: Vec<f64>,
    mut q_hat: Vec<f64>,
) -> Result<CoreDiracSolution, DiracError> {
    let density: Vec<f64> = p
        .iter()
        .zip(&q_hat)
        .map(|(&large, &small_scaled)| large * large + small_scaled * small_scaled / C_SQUARED)
        .collect();
    let norm_squared = mesh
        .integrate(&density)
        .map_err(|error| DiracError::Quadrature(error.to_string()))?;
    if !norm_squared.is_finite() || norm_squared <= f64::MIN_POSITIVE {
        return Err(DiracError::SingularNorm { norm_squared });
    }
    let scale = norm_squared.sqrt().recip();
    p.iter_mut().for_each(|value| *value *= scale);
    q_hat.iter_mut().for_each(|value| *value *= scale);
    let q: Vec<f64> = q_hat
        .iter()
        .map(|value| value / SPEX_SPEED_OF_LIGHT)
        .collect();
    let normalized_density: Vec<f64> = p
        .iter()
        .zip(&q)
        .map(|(&large, &small)| large * large + small * small)
        .collect();
    let norm_total = mesh
        .integrate(&normalized_density)
        .map_err(|error| DiracError::Quadrature(error.to_string()))?;
    let muffin_tin_index = locate_muffin_tin_index(mesh, spec.muffin_tin_radius)?;
    let muffin_tin_mesh =
        ExponentialMesh::new(mesh.first(), mesh.increment(), muffin_tin_index + 1)
            .map_err(|error| DiracError::Quadrature(error.to_string()))?;
    let raw_norm_mt = muffin_tin_mesh
        .integrate(&normalized_density[..=muffin_tin_index])
        .map_err(|error| DiracError::Quadrature(error.to_string()))?;
    let partition_tolerance =
        CORE_NORM_PARTITION_TOLERANCE * norm_total.abs().max(raw_norm_mt.abs()).max(1.0);
    if raw_norm_mt > norm_total + partition_tolerance {
        return Err(DiracError::InconsistentNormPartition {
            norm_total,
            norm_mt: raw_norm_mt,
            tolerance: partition_tolerance,
        });
    }
    let norm_mt = raw_norm_mt.min(norm_total);
    // The outside is the complement of the independently integrated prefix;
    // no cutoff sample is double counted. Independent seventh-order stencils
    // resolve a smaller prefix excess only as zero spill.
    let norm_outside = norm_total - norm_mt;
    let nodes = count_nodes(&p);
    let expected_nodes = spec.state.expected_nodes();
    if nodes != expected_nodes {
        return Err(DiracError::NodeCountMismatch {
            expected: expected_nodes,
            actual: nodes,
        });
    }

    Ok(CoreDiracSolution {
        role: RelativisticRole::Core,
        state: spec.state,
        angular: spec.state.kappa.angular_contract(),
        energy: Hartree(energy),
        p,
        q,
        norm_total,
        norm_mt,
        norm_outside,
        spill: norm_outside,
        nodes,
        match_radius: mesh.radii()[match_index],
        matching_residual,
    })
}

fn count_nodes(values: &[f64]) -> u32 {
    let largest = values
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    let threshold = largest * 1.0e-10;
    let mut previous = 0.0_f64;
    let mut nodes = 0;
    for &value in values {
        if value.abs() <= threshold {
            continue;
        }
        if previous != 0.0 && value.signum() != previous.signum() {
            nodes += 1;
        }
        previous = value;
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extended_mesh(first: f64, last: f64, increment: f64) -> ExponentialMesh {
        let count = ((last / first).ln() / increment).ceil() as usize + 1;
        ExponentialMesh::new(Bohr(first), increment, count).unwrap()
    }

    #[test]
    fn kappa_mapping_covers_both_spin_orbit_branches() {
        let cases = [
            (-1, 0, 1, 1, 2),
            (1, 1, 0, 1, 2),
            (-2, 1, 2, 3, 4),
            (2, 2, 1, 3, 4),
            (-3, 2, 3, 5, 6),
        ];
        for (value, large_l, small_l, twice_j, degeneracy) in cases {
            let angular = Kappa::new(value).unwrap().angular_contract();
            assert_eq!(angular.large_l, large_l);
            assert_eq!(angular.small_l, small_l);
            assert_eq!(angular.twice_j, twice_j);
            assert_eq!(angular.degeneracy, degeneracy);
        }
        assert!(Kappa::new(0).is_err());
    }

    #[test]
    fn coulomb_one_s_matches_the_shifted_dirac_energy() {
        let mesh = extended_mesh(1.0e-7, 40.0, 0.002);
        let potential: Vec<f64> = mesh
            .radii()
            .iter()
            .map(|radius| -1.0 / radius.get())
            .collect();
        let mt_radius = *mesh
            .radii()
            .iter()
            .min_by(|a, b| (a.get() - 6.0).abs().total_cmp(&(b.get() - 6.0).abs()))
            .unwrap();
        let state = CoreState::new(1, Kappa::new(-1).unwrap()).unwrap();
        let spec = CoreDiracSpec::new(
            state,
            1.0,
            EnergyBracket::from_values(-0.6, -0.4).unwrap(),
            mt_radius,
        );
        let solution = solve_core_dirac(&mesh, &potential, spec).unwrap();
        let exact = C_SQUARED * ((1.0 - 1.0 / C_SQUARED).sqrt() - 1.0);

        assert!((solution.energy.get() - exact).abs() < 1.0e-8);
        assert!((solution.norm_total - 1.0).abs() < 2.0e-13);
        assert!((solution.norm_mt + solution.norm_outside - 1.0).abs() < 2.0e-13);
        assert!(solution.norm_outside >= 0.0);
        assert!(solution.spill > 0.0 && solution.spill < 1.0e-3);
        assert_eq!(solution.nodes, 0);
        assert!(solution.matching_residual.abs() <= spec.matching_tolerance);
    }

    #[test]
    fn long_mesh_iron_one_s_stops_the_inward_branch_beyond_the_match_radius() {
        let mesh = extended_mesh(1.0e-7, 200.0, 0.002);
        let z = 26.0;
        let potential: Vec<f64> = mesh
            .radii()
            .iter()
            .map(|radius| -z / radius.get())
            .collect();
        let mt_radius = *mesh
            .radii()
            .iter()
            .min_by(|a, b| (a.get() - 2.0).abs().total_cmp(&(b.get() - 2.0).abs()))
            .unwrap();
        let state = CoreState::new(1, Kappa::new(-1).unwrap()).unwrap();
        let exact = C_SQUARED * ((1.0 - z * z / C_SQUARED).sqrt() - 1.0);
        let spec = CoreDiracSpec::new(
            state,
            z,
            EnergyBracket::from_values(exact - 20.0, exact + 20.0).unwrap(),
            mt_radius,
        );

        let solution = solve_core_dirac(&mesh, &potential, spec).unwrap();
        let match_index = mesh
            .radii()
            .iter()
            .position(|radius| *radius == solution.match_radius)
            .unwrap();
        let outer_index = select_outer_index(&mesh, match_index);

        assert!(outer_index < mesh.len() - 1);
        assert!(mesh.radii()[outer_index].get() >= 15.0 * solution.match_radius.get());
        assert!(
            solution.p[outer_index + 1..]
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(
            solution.q[outer_index + 1..]
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!((solution.norm_total - 1.0).abs() < 2.0e-13);
        assert!(solution.matching_residual.is_finite());
        assert!(solution.matching_residual.abs() <= spec.matching_tolerance);
    }

    #[test]
    fn deterministic_core_search_isolates_the_node_compatible_bracket() {
        let mesh = extended_mesh(1.0e-7, 100.0, 0.002);
        let potential: Vec<f64> = mesh
            .radii()
            .iter()
            .map(|radius| -1.0 / radius.get())
            .collect();
        let mt_radius = *mesh
            .radii()
            .iter()
            .min_by(|a, b| (a.get() - 6.0).abs().total_cmp(&(b.get() - 6.0).abs()))
            .unwrap();
        let state = CoreState::new(1, Kappa::new(-1).unwrap()).unwrap();
        let search = CoreBracketSearch::new(
            state,
            1.0,
            mt_radius,
            EnergyBracket::from_values(-0.8, -0.2).unwrap(),
        )
        .with_intervals(48);
        let bracket = isolate_core_dirac_bracket(&mesh, &potential, search).unwrap();
        let exact = C_SQUARED * ((1.0 - 1.0 / C_SQUARED).sqrt() - 1.0);
        assert!(bracket.lower.get() < exact && bracket.upper.get() > exact);
        let solution = solve_core_dirac(
            &mesh,
            &potential,
            CoreDiracSpec::new(state, 1.0, bracket, mt_radius),
        )
        .unwrap();
        assert_eq!(solution.nodes, state.expected_nodes());

        let missing = CoreBracketSearch::new(
            state,
            1.0,
            mt_radius,
            EnergyBracket::from_values(-0.2, -0.05).unwrap(),
        )
        .with_intervals(24);
        assert!(matches!(
            isolate_core_dirac_bracket(&mesh, &potential, missing),
            Err(DiracError::CoreBracketNotFound { .. })
        ));
    }

    #[test]
    fn core_search_and_solution_are_covariant_under_global_energy_shift() {
        let mesh = extended_mesh(1.0e-7, 40.0, 0.002);
        let potential = mesh
            .radii()
            .iter()
            .map(|radius| -1.0 / radius.get())
            .collect::<Vec<_>>();
        let mt_radius = *mesh
            .radii()
            .iter()
            .min_by(|a, b| (a.get() - 6.0).abs().total_cmp(&(b.get() - 6.0).abs()))
            .unwrap();
        let state = CoreState::new(1, Kappa::new(-1).unwrap()).unwrap();
        let window = EnergyBracket::from_values(-0.8, -0.2).unwrap();
        let bracket = isolate_core_dirac_bracket(
            &mesh,
            &potential,
            CoreBracketSearch::new(state, 1.0, mt_radius, window).with_intervals(48),
        )
        .unwrap();
        let solution = solve_core_dirac(
            &mesh,
            &potential,
            CoreDiracSpec::new(state, 1.0, bracket, mt_radius),
        )
        .unwrap();

        let shift = 1.25;
        let shifted_potential = potential
            .iter()
            .map(|value| value + shift)
            .collect::<Vec<_>>();
        let shifted_window =
            EnergyBracket::from_values(window.lower.get() + shift, window.upper.get() + shift)
                .unwrap();
        let shifted_bracket = isolate_core_dirac_bracket(
            &mesh,
            &shifted_potential,
            CoreBracketSearch::new(state, 1.0, mt_radius, shifted_window).with_intervals(48),
        )
        .unwrap();
        assert!((shifted_bracket.lower.get() - bracket.lower.get() - shift).abs() < 1.0e-13);
        assert!((shifted_bracket.upper.get() - bracket.upper.get() - shift).abs() < 1.0e-13);
        let shifted_solution = solve_core_dirac(
            &mesh,
            &shifted_potential,
            CoreDiracSpec::new(state, 1.0, shifted_bracket, mt_radius),
        )
        .unwrap();
        assert!((shifted_solution.energy.get() - solution.energy.get() - shift).abs() < 1.0e-11);
        let phase = solution
            .p
            .iter()
            .zip(&shifted_solution.p)
            .map(|(&left, &right)| left * right)
            .sum::<f64>()
            .signum();
        for ((&p, &q), (&shifted_p, &shifted_q)) in solution
            .p
            .iter()
            .zip(&solution.q)
            .zip(shifted_solution.p.iter().zip(&shifted_solution.q))
        {
            assert!((phase * shifted_p - p).abs() < 2.0e-11);
            assert!((phase * shifted_q - q).abs() < 2.0e-13);
        }
    }

    #[test]
    fn coulomb_two_s_selects_the_one_node_root() {
        let mesh = extended_mesh(1.0e-7, 100.0, 0.002);
        let potential: Vec<f64> = mesh
            .radii()
            .iter()
            .map(|radius| -1.0 / radius.get())
            .collect();
        let mt_radius = *mesh
            .radii()
            .iter()
            .min_by(|a, b| (a.get() - 10.0).abs().total_cmp(&(b.get() - 10.0).abs()))
            .unwrap();
        let kappa = Kappa::new(-1).unwrap();
        let state = CoreState::new(2, kappa).unwrap();
        let spec = CoreDiracSpec::new(
            state,
            1.0,
            EnergyBracket::from_values(-0.14, -0.11).unwrap(),
            mt_radius,
        );
        let solution = solve_core_dirac(&mesh, &potential, spec).unwrap();
        let gamma = (1.0 - 1.0 / C_SQUARED).sqrt();
        let denominator = 1.0 + gamma;
        let exact = C_SQUARED
            * ((1.0 + 1.0 / (C_SQUARED * denominator * denominator))
                .sqrt()
                .recip()
                - 1.0);

        assert!((solution.energy.get() - exact).abs() < 1.0e-8);
        assert_eq!(solution.nodes, 1);
    }

    #[test]
    fn coulomb_two_p_one_half_covers_positive_kappa() {
        let mesh = extended_mesh(1.0e-7, 100.0, 0.002);
        let potential: Vec<f64> = mesh
            .radii()
            .iter()
            .map(|radius| -1.0 / radius.get())
            .collect();
        let mt_radius = *mesh
            .radii()
            .iter()
            .min_by(|a, b| (a.get() - 10.0).abs().total_cmp(&(b.get() - 10.0).abs()))
            .unwrap();
        let kappa = Kappa::new(1).unwrap();
        let state = CoreState::new(2, kappa).unwrap();
        let spec = CoreDiracSpec::new(
            state,
            1.0,
            EnergyBracket::from_values(-0.14, -0.11).unwrap(),
            mt_radius,
        );
        let solution = solve_core_dirac(&mesh, &potential, spec).unwrap();
        let gamma = (1.0 - 1.0 / C_SQUARED).sqrt();
        let denominator = 1.0 + gamma;
        let exact = C_SQUARED
            * ((1.0 + 1.0 / (C_SQUARED * denominator * denominator))
                .sqrt()
                .recip()
                - 1.0);

        assert!((solution.energy.get() - exact).abs() < 1.0e-8);
        assert_eq!(solution.angular.large_l, 1);
        assert_eq!(solution.angular.small_l, 0);
        assert_eq!(solution.nodes, 0);
    }
}
