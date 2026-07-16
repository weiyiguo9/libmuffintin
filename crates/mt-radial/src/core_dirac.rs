//! Spherical four-component Dirac bound-core states.
//!
//! The public radial spinor is
//! `Psi = (P Omega_kappa, i Q Omega_-kappa) / r`.  Energies have the
//! electron rest energy subtracted and are measured in Hartree.  The
//! integration variable used for the small component is `q_hat = c Q`.

use std::borrow::Borrow;

use mt_core::{Bohr, ExponentialMesh, Hartree};
use thiserror::Error;

use crate::valence::SPEX_SPEED_OF_LIGHT;

const C_SQUARED: f64 = SPEX_SPEED_OF_LIGHT * SPEX_SPEED_OF_LIGHT;

/// The role of a relativistic radial solution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelativisticRole {
    /// A normalizable bound core state on an extended radial domain.
    Core,
    /// The reserved (not yet implemented) four-component valence basis.
    Valence,
}

/// A nonzero Dirac spin-angular quantum number.
///
/// `kappa = -(l + 1)` labels `j = l + 1/2`, while `kappa = l` labels
/// `j = l - 1/2`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Kappa(i32);

impl Kappa {
    /// Construct a direct Dirac `kappa` label.
    pub fn new(value: i32) -> Result<Self, KappaError> {
        match value {
            0 => Err(KappaError::Zero),
            i32::MIN => Err(KappaError::OutOfRange(value)),
            _ => Ok(Self(value)),
        }
    }

    /// Return the signed integer label.
    pub const fn get(self) -> i32 {
        self.0
    }

    /// Orbital angular momentum of the large component `P`.
    pub const fn orbital_angular_momentum(self) -> u32 {
        if self.0 < 0 {
            self.0.unsigned_abs() - 1
        } else {
            self.0 as u32
        }
    }

    /// Short alias for [`Self::orbital_angular_momentum`].
    pub const fn l(self) -> u32 {
        self.orbital_angular_momentum()
    }

    /// Orbital angular momentum of the small component `Q`.
    pub const fn small_component_angular_momentum(self) -> u32 {
        if self.0 < 0 {
            self.0.unsigned_abs()
        } else {
            self.0 as u32 - 1
        }
    }

    /// `2j`, represented exactly as an integer.
    pub const fn twice_j(self) -> u32 {
        2 * self.0.unsigned_abs() - 1
    }

    /// Total angular momentum `j`.
    pub fn j(self) -> f64 {
        f64::from(self.0.unsigned_abs()) - 0.5
    }

    /// Magnetic degeneracy `2j + 1 = 2 |kappa|`.
    pub const fn degeneracy(self) -> u32 {
        2 * self.0.unsigned_abs()
    }

    /// Materialize the complete angular contract for this channel.
    pub const fn angular_contract(self) -> DiracAngularContract {
        DiracAngularContract {
            kappa: self,
            large_l: self.orbital_angular_momentum(),
            small_l: self.small_component_angular_momentum(),
            twice_j: self.twice_j(),
            degeneracy: self.degeneracy(),
        }
    }
}

impl TryFrom<i32> for Kappa {
    type Error = KappaError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Kappa> for i32 {
    fn from(value: Kappa) -> Self {
        value.get()
    }
}

/// Explicit mapping between `kappa` and both spinor spherical harmonics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiracAngularContract {
    pub kappa: Kappa,
    /// Orbital angular momentum multiplying `P` (`Omega_kappa`).
    pub large_l: u32,
    /// Orbital angular momentum multiplying `Q` (`Omega_-kappa`).
    pub small_l: u32,
    /// Twice the half-integer total angular momentum.
    pub twice_j: u32,
    /// Number of allowed `m_j` values.
    pub degeneracy: u32,
}

impl From<Kappa> for DiracAngularContract {
    fn from(kappa: Kappa) -> Self {
        kappa.angular_contract()
    }
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
    pub fn new(n: u32, kappa: Kappa) -> Result<Self, KappaError> {
        let minimum = kappa.l() + 1;
        if n < minimum {
            Err(KappaError::InvalidPrincipalQuantumNumber { n, l: kappa.l() })
        } else {
            Ok(Self { n, kappa })
        }
    }

    /// Expected nonrelativistic radial node count `n - l - 1`.
    pub const fn expected_nodes(self) -> u32 {
        self.n - self.kappa.l() - 1
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
    pub fn new(lower: Hartree, upper: Hartree) -> Result<Self, KappaError> {
        let bracket = Self { lower, upper };
        bracket.validate()?;
        Ok(bracket)
    }

    /// Convenience constructor from raw Hartree values.
    pub fn from_values(lower: f64, upper: f64) -> Result<Self, KappaError> {
        Self::new(Hartree(lower), Hartree(upper))
    }

    /// Return the two raw Hartree values.
    pub const fn values(self) -> (f64, f64) {
        (self.lower.get(), self.upper.get())
    }

    fn validate(self) -> Result<(), KappaError> {
        let (lower, upper) = self.values();
        if lower.is_finite() && upper.is_finite() && lower < upper {
            Ok(())
        } else {
            Err(KappaError::InvalidEnergyBracket { lower, upper })
        }
    }
}

/// Controls for a two-sided bound-core shooting calculation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreDiracSpec {
    pub state: CoreState,
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

/// Input contract for the deliberately reserved four-component valence path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValenceDiracSpec {
    pub kappa: Kappa,
    /// Rest-energy-subtracted trial energy in Hartree.
    pub energy: Hartree,
}

impl ValenceDiracSpec {
    pub fn new(kappa: Kappa, energy: Hartree) -> Result<Self, KappaError> {
        if energy.get().is_finite() {
            Ok(Self { kappa, energy })
        } else {
            Err(KappaError::NonFiniteEnergy(energy.get()))
        }
    }
}

impl CoreDiracSpec {
    /// Construct a specification with conservative shooting tolerances.
    pub const fn new(state: CoreState, bracket: EnergyBracket, muffin_tin_radius: Bohr) -> Self {
        Self {
            state,
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

/// Reserved result type for a future four-component valence implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct ValenceDiracSolution {
    pub role: RelativisticRole,
    pub kappa: Kappa,
    pub energy: Hartree,
    pub p: Vec<f64>,
    pub q: Vec<f64>,
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

/// Diagnosable input, shooting, and feature-boundary errors.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum KappaError {
    #[error("Dirac kappa cannot be zero")]
    Zero,
    #[error("Dirac kappa is outside the supported range: {0}")]
    OutOfRange(i32),
    #[error("principal quantum number n={n} is invalid for l={l}")]
    InvalidPrincipalQuantumNumber { n: u32, l: u32 },
    #[error("energy bracket is invalid: [{lower}, {upper}] Ha")]
    InvalidEnergyBracket { lower: f64, upper: f64 },
    #[error("potential has {actual} samples, but the mesh has {expected}")]
    PotentialLength { expected: usize, actual: usize },
    #[error("potential[{index}] is not finite: {value}")]
    NonFinitePotential { index: usize, value: f64 },
    #[error("energy is not finite: {0}")]
    NonFiniteEnergy(f64),
    #[error("bound-core shooting requires an outward positive radial mesh")]
    InvalidMeshDirection,
    #[error("muffin-tin radius {radius} bohr is not strictly inside mesh [{first}, {last}]")]
    InvalidMuffinTinRadius { radius: f64, first: f64, last: f64 },
    #[error("muffin-tin radius {radius} bohr is not a radial mesh point")]
    MuffinTinRadiusNotOnMesh { radius: f64 },
    #[error("origin is not Coulombic enough to initialize a core Dirac state (estimated Z={0})")]
    NonCoulombicOrigin(f64),
    #[error("point-Coulomb origin is supercritical: kappa^2-(Z/c)^2={radicand}")]
    SupercriticalOrigin { radicand: f64 },
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
    #[error("mesh quadrature failed: {0}")]
    Quadrature(String),
    #[error("four-component {role:?} radial solving is not supported")]
    Unsupported { role: RelativisticRole },
}

/// Solve one normalizable spherical four-component Dirac core state.
///
/// `potential` is the total physical spherical potential `V(r)` in Hartree
/// on every point of the extended positive `mesh`. The energy bracket must
/// isolate the requested root; a converged root with a different radial node
/// count is rejected with [`KappaError::NodeCountMismatch`].
pub fn solve_core_dirac<S: Borrow<CoreDiracSpec>>(
    mesh: &ExponentialMesh,
    potential: &[f64],
    spec: S,
) -> Result<CoreDiracSolution, KappaError> {
    let spec = spec.borrow();
    validate_inputs(mesh, potential, spec)?;

    let (mut lower, mut upper) = spec.bracket.values();
    let mut lower_shot = shoot(mesh, potential, spec.state.kappa, lower, false)?;
    let upper_shot = shoot(mesh, potential, spec.state.kappa, upper, false)?;
    if lower_shot.residual == 0.0 {
        return assemble_solution(mesh, potential, spec, lower, lower_shot.match_index);
    }
    if upper_shot.residual == 0.0 {
        return assemble_solution(mesh, potential, spec, upper, upper_shot.match_index);
    }
    if lower_shot.residual.signum() == upper_shot.residual.signum() {
        return Err(KappaError::RootNotBracketed {
            lower,
            upper,
            f_lower: lower_shot.residual,
            f_upper: upper_shot.residual,
        });
    }

    for _ in 0..spec.max_iterations {
        let energy = lower + 0.5 * (upper - lower);
        let shot = shoot(mesh, potential, spec.state.kappa, energy, false)?;
        if shot.residual.abs() <= spec.matching_tolerance
            && 0.5 * (upper - lower) <= spec.energy_tolerance
        {
            return assemble_solution(mesh, potential, spec, energy, shot.match_index);
        }
        if shot.residual.signum() == lower_shot.residual.signum() {
            lower = energy;
            lower_shot = shot;
        } else {
            upper = energy;
        }
    }

    Err(KappaError::RootDidNotConverge {
        iterations: spec.max_iterations,
    })
}

/// Reserved four-component valence entry point.
///
/// This deliberately never falls back to the scalar Koelling--Harmon solver.
pub fn solve_valence_dirac<S: Borrow<ValenceDiracSpec>>(
    _mesh: &ExponentialMesh,
    _potential: &[f64],
    _spec: S,
) -> Result<ValenceDiracSolution, KappaError> {
    Err(KappaError::Unsupported {
        role: RelativisticRole::Valence,
    })
}

#[derive(Clone, Copy, Debug)]
struct Shot {
    residual: f64,
    match_index: usize,
}

fn validate_inputs(
    mesh: &ExponentialMesh,
    potential: &[f64],
    spec: &CoreDiracSpec,
) -> Result<(), KappaError> {
    spec.bracket.validate()?;
    if potential.len() != mesh.len() {
        return Err(KappaError::PotentialLength {
            expected: mesh.len(),
            actual: potential.len(),
        });
    }
    if let Some((index, &value)) = potential
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(KappaError::NonFinitePotential { index, value });
    }
    if mesh.increment() <= 0.0 {
        return Err(KappaError::InvalidMeshDirection);
    }
    let first = mesh.first().get();
    let last = mesh.last().get();
    let radius = spec.muffin_tin_radius.get();
    if !radius.is_finite() || radius < first || radius >= last {
        return Err(KappaError::InvalidMuffinTinRadius {
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
        return Err(KappaError::InvalidTolerance {
            energy: spec.energy_tolerance,
            matching: spec.matching_tolerance,
            iterations: spec.max_iterations,
        });
    }
    let minimum = spec.state.kappa.l() + 1;
    if spec.state.n < minimum {
        return Err(KappaError::InvalidPrincipalQuantumNumber {
            n: spec.state.n,
            l: spec.state.kappa.l(),
        });
    }
    Ok(())
}

fn shoot(
    mesh: &ExponentialMesh,
    potential: &[f64],
    kappa: Kappa,
    energy: f64,
    keep_arrays: bool,
) -> Result<Shot, KappaError> {
    let match_index = select_match_index(mesh, potential, energy);
    let outward = integrate_outward(mesh, potential, kappa, energy, match_index, keep_arrays)?;
    let inward = integrate_inward(mesh, potential, kappa, energy, match_index, keep_arrays)?;
    let (po, qo) = outward.at_match(match_index);
    let (pi, qi) = inward.at_match(match_index);
    let out_norm = po.hypot(qo);
    let in_norm = pi.hypot(qi);
    if out_norm <= f64::MIN_POSITIVE || in_norm <= f64::MIN_POSITIVE {
        return Err(KappaError::SingularMatch { index: match_index });
    }
    let residual = (po * qi - qo * pi) / (out_norm * in_norm);
    if !residual.is_finite() {
        return Err(KappaError::SingularMatch { index: match_index });
    }
    Ok(Shot {
        residual,
        match_index,
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
    kappa: Kappa,
    energy: f64,
    stop: usize,
    keep_arrays: bool,
) -> Result<Branch, KappaError> {
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
    // Averaging the first few -rV samples damps harmless grid noise while
    // retaining the Coulomb coefficient exactly for V=-Z/r.
    let sample_count = n.min(4);
    let z = mesh.radii()[..sample_count]
        .iter()
        .zip(&potential[..sample_count])
        .map(|(r, &v)| -r.get() * v)
        .sum::<f64>()
        / sample_count as f64;
    if !z.is_finite() || z <= 1.0e-12 {
        return Err(KappaError::NonCoulombicOrigin(z));
    }
    let k = f64::from(kappa.get());
    let radicand = k * k - (z / SPEX_SPEED_OF_LIGHT).powi(2);
    if !radicand.is_finite() || radicand <= 0.0 {
        return Err(KappaError::SupercriticalOrigin { radicand });
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
    keep_arrays: bool,
) -> Result<Branch, KappaError> {
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
    let delta = potential[n - 1] - energy;
    let mass_factor = 2.0 - delta / C_SQUARED;
    let decay_squared = mass_factor * delta;
    if !decay_squared.is_finite() || decay_squared <= 0.0 {
        return Err(KappaError::NonDecayingOuterBoundary { delta });
    }
    let decay = decay_squared.sqrt();
    let mut current_p = 1.0;
    let mut current_q = -decay / mass_factor;
    if keep_arrays {
        p[n - 1] = current_p;
        q_hat[n - 1] = current_q;
    }
    let k = f64::from(kappa.get());
    for i in (stop + 1..n).rev() {
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
    let (k1, l1) = dirac_rhs(ra, va, p, q_hat, kappa, energy);
    let (k2, l2) = dirac_rhs(
        rb,
        vb,
        p + 0.5 * dr * k1,
        q_hat + 0.5 * dr * l1,
        kappa,
        energy,
    );
    let (k3, l3) = dirac_rhs(
        rb,
        vb,
        p + 0.5 * dr * k2,
        q_hat + 0.5 * dr * l2,
        kappa,
        energy,
    );
    let (k4, l4) = dirac_rhs(rc, vc, p + dr * k3, q_hat + dr * l3, kappa, energy);
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
) -> (f64, f64) {
    let mass_factor = 2.0 + (energy - potential) / C_SQUARED;
    (
        mass_factor * q_hat - kappa * p / radius,
        (potential - energy) * p + kappa * q_hat / radius,
    )
}

fn ensure_finite_state(p: f64, q_hat: f64, index: usize) -> Result<(), KappaError> {
    if p.is_finite() && q_hat.is_finite() && p.abs() <= 1.0e200 && q_hat.abs() <= 1.0e200 {
        Ok(())
    } else {
        Err(KappaError::IntegrationOverflow { index })
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
        .unwrap_or(mesh.len() / 2)
}

fn locate_muffin_tin_index(
    mesh: &ExponentialMesh,
    muffin_tin_radius: Bohr,
) -> Result<usize, KappaError> {
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
        Err(KappaError::MuffinTinRadiusNotOnMesh { radius: requested })
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
) -> Result<CoreDiracSolution, KappaError> {
    let outward = integrate_outward(mesh, potential, spec.state.kappa, energy, match_index, true)?;
    let inward = integrate_inward(mesh, potential, spec.state.kappa, energy, match_index, true)?;
    let po = outward.p[match_index];
    let qo = outward.q_hat[match_index];
    let pi = inward.p[match_index];
    let qi = inward.q_hat[match_index];
    let denominator = po.hypot(qo) * pi.hypot(qi);
    if !denominator.is_finite() || denominator <= f64::MIN_POSITIVE {
        return Err(KappaError::SingularMatch { index: match_index });
    }
    let residual = (po * qi - qo * pi) / denominator;
    // Enforce P continuity using one common scale for both inward components.
    let scale_in = if pi.abs() > 64.0 * f64::EPSILON * pi.hypot(qi) {
        po / pi
    } else if qi.abs() > 64.0 * f64::EPSILON * pi.hypot(qi) {
        qo / qi
    } else {
        return Err(KappaError::SingularMatch { index: match_index });
    };

    let mut p = outward.p;
    let mut q_hat = outward.q_hat;
    for i in match_index + 1..mesh.len() {
        p[i] = scale_in * inward.p[i];
        q_hat[i] = scale_in * inward.q_hat[i];
    }
    let density: Vec<f64> = p
        .iter()
        .zip(&q_hat)
        .map(|(&large, &small_scaled)| large * large + small_scaled * small_scaled / C_SQUARED)
        .collect();
    let norm_squared = mesh
        .integrate(&density)
        .map_err(|error| KappaError::Quadrature(error.to_string()))?;
    if !norm_squared.is_finite() || norm_squared <= f64::MIN_POSITIVE {
        return Err(KappaError::SingularNorm { norm_squared });
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
        .map_err(|error| KappaError::Quadrature(error.to_string()))?;
    let muffin_tin_index = locate_muffin_tin_index(mesh, spec.muffin_tin_radius)?;
    let muffin_tin_mesh =
        ExponentialMesh::new(mesh.first(), mesh.increment(), muffin_tin_index + 1)
            .map_err(|error| KappaError::Quadrature(error.to_string()))?;
    let norm_mt = muffin_tin_mesh
        .integrate(&normalized_density[..=muffin_tin_index])
        .map_err(|error| KappaError::Quadrature(error.to_string()))?;
    // The outside is the complement of the independently integrated prefix;
    // no cutoff sample is double counted.
    let norm_outside = norm_total - norm_mt;
    let nodes = count_nodes(&p);
    let expected_nodes = spec.state.expected_nodes();
    if nodes != expected_nodes {
        return Err(KappaError::NodeCountMismatch {
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
        matching_residual: residual,
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
        assert_eq!(Kappa::new(0), Err(KappaError::Zero));
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
            EnergyBracket::from_values(-0.6, -0.4).unwrap(),
            mt_radius,
        );
        let solution = solve_core_dirac(&mesh, &potential, spec).unwrap();
        let exact = C_SQUARED * ((1.0 - 1.0 / C_SQUARED).sqrt() - 1.0);

        assert!((solution.energy.get() - exact).abs() < 1.0e-8);
        assert!((solution.norm_total - 1.0).abs() < 2.0e-13);
        assert!((solution.norm_mt + solution.norm_outside - 1.0).abs() < 2.0e-13);
        assert!(solution.spill > 0.0 && solution.spill < 1.0e-3);
        assert_eq!(solution.nodes, 0);
        assert!(solution.matching_residual.abs() <= spec.matching_tolerance);
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
