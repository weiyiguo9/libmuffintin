use mt_core::{ExponentialMesh, Hartree, InverseBohr};
use thiserror::Error;

use crate::core_dirac::EnergyBracket;

/// Speed of light used by SPEX (`src/global.f`) in Hartree atomic units.
pub const SPEX_SPEED_OF_LIGHT: f64 = 137.035_989_5;
const C_INV_SQUARED: f64 = 1.0 / (SPEX_SPEED_OF_LIGHT * SPEX_SPEED_OF_LIGHT);

/// Equation used for the two-component valence radial problem.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RadialEquation {
    /// Nonrelativistic radial Schrödinger equation, `T = -1/2 ∇²`.
    Schroedinger,
    /// Scalar-relativistic Koelling--Harmon equation in the SPEX convention.
    ScalarKoellingHarmon,
}

impl RadialEquation {
    fn c_inverse_squared(self) -> f64 {
        match self {
            Self::Schroedinger => 0.0,
            Self::ScalarKoellingHarmon => C_INV_SQUARED,
        }
    }
}

/// Values and slopes at the muffin-tin boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryData {
    /// `u(R)` with `p = r u`.
    pub value: f64,
    /// Dimensional derivative `du/dr` at `R`.
    pub derivative: f64,
    /// Dimensional logarithmic derivative `u'(R)/u(R)` in bohr⁻¹.
    ///
    /// This is `None` when the boundary value vanishes numerically.
    pub log_derivative: Option<InverseBohr>,
    /// Scaled, dimensionless logarithmic derivative `R u'(R)/u(R)`.
    pub scaled_log_derivative: Option<f64>,
}

impl BoundaryData {
    fn new(value: f64, derivative: f64, radius: f64) -> Self {
        let scale = value.abs().max(1.0);
        let logarithmic = (value.abs() > 64.0 * f64::EPSILON * scale).then(|| derivative / value);
        Self {
            value,
            derivative,
            log_derivative: logarithmic.map(InverseBohr),
            scaled_log_derivative: logarithmic.map(|d| radius * d),
        }
    }
}

/// A normalized homogeneous valence radial solution.
#[derive(Clone, Debug, PartialEq)]
pub struct RadialSolution {
    equation: RadialEquation,
    angular_momentum: u32,
    energy: Hartree,
    /// Large reduced radial component `p(r) = r u(r)`.
    pub p: Vec<f64>,
    /// Physical Koelling--Harmon small component `Q(r)`; absent for Schrödinger.
    pub q: Option<Vec<f64>>,
    // Auxiliary first-order component.  This is `c Q` for KH and `r u'/2`
    // for Schrödinger; retaining it avoids differentiating sampled data.
    auxiliary_q: Vec<f64>,
    /// Boundary data for the large component `u = p/r`.
    pub boundary: BoundaryData,
}

impl RadialSolution {
    pub const fn equation(&self) -> RadialEquation {
        self.equation
    }

    pub const fn angular_momentum(&self) -> u32 {
        self.angular_momentum
    }

    pub const fn energy(&self) -> Hartree {
        self.energy
    }

    /// Materialize `u(r) = p(r)/r` on `mesh`.
    pub fn u(&self, mesh: &ExponentialMesh) -> Result<Vec<f64>, RadialError> {
        ensure_mesh_length(mesh, self.p.len())?;
        Ok(self
            .p
            .iter()
            .zip(mesh.radii())
            .map(|(&p, r)| p / r.get())
            .collect())
    }
}

/// Exact energy derivative of a normalized homogeneous solution.
///
/// It is orthogonalized against `solution`, but is deliberately not normalized.
#[derive(Clone, Debug, PartialEq)]
pub struct EnergyDerivative {
    /// `d p / dE` in Hartree⁻¹.
    pub p: Vec<f64>,
    /// Physical `d Q / dE`, present only for Koelling--Harmon.
    pub q: Option<Vec<f64>>,
    pub boundary: BoundaryData,
    /// Metric norm `⟨du/dE | du/dE⟩` after orthogonalization.
    pub norm_squared: f64,
}

impl EnergyDerivative {
    /// Materialize `du/dE = (dp/dE)/r` on `mesh`.
    pub fn u(&self, mesh: &ExponentialMesh) -> Result<Vec<f64>, RadialError> {
        ensure_mesh_length(mesh, self.p.len())?;
        Ok(self
            .p
            .iter()
            .zip(mesh.radii())
            .map(|(&p, r)| p / r.get())
            .collect())
    }
}

/// The LAPW linearization pair `(u, du/dE)`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearizedRadialSolution {
    pub solution: RadialSolution,
    pub energy_derivative: EnergyDerivative,
}

/// Coefficients of the unnormalized matched combination
/// `raw + a * u + b * du/dE`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalOrbitalCoefficients {
    pub a: f64,
    pub b: f64,
    /// Scale subsequently applied to the whole matched combination.
    pub normalization_scale: f64,
}

/// A normalized local orbital whose value and slope vanish at the boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalOrbital {
    pub energy: Hartree,
    pub p: Vec<f64>,
    pub q: Option<Vec<f64>>,
    pub coefficients: LocalOrbitalCoefficients,
    pub boundary: BoundaryData,
}

/// Diagnosable radial-solver failures.
#[derive(Debug, Error, PartialEq)]
pub enum RadialError {
    #[error("potential has {actual} samples, but the mesh has {expected}")]
    PotentialLength { expected: usize, actual: usize },
    #[error("radial array has {actual} samples, but the mesh has {expected}")]
    ArrayLength { expected: usize, actual: usize },
    #[error("potential[{index}] is not finite: {value}")]
    NonFinitePotential { index: usize, value: f64 },
    #[error("energy is not finite: {0}")]
    NonFiniteEnergy(f64),
    #[error("regular-origin exponent is non-real (radicand {radicand})")]
    SupercriticalOrigin { radicand: f64 },
    #[error("Koelling--Harmon mass factor at index {index} is non-positive or non-finite: {mass}")]
    InvalidRelativisticMass { index: usize, mass: f64 },
    #[error("radial integration overflowed at mesh index {index}")]
    IntegrationOverflow { index: usize },
    #[error("solution norm is singular or non-finite: {norm_squared}")]
    SingularNorm { norm_squared: f64 },
    #[error("local-orbital boundary system is singular (determinant {determinant})")]
    SingularLocalOrbital { determinant: f64 },
    #[error("radial equation mismatch: solver uses {solver:?}, solution uses {solution:?}")]
    EquationMismatch {
        solver: RadialEquation,
        solution: RadialEquation,
    },
    #[error("large/small component presence is inconsistent in a radial basis combination")]
    ComponentMismatch,
    #[error("hard-wall energy bracket is invalid: [{lower}, {upper}] Ha")]
    InvalidEnergyBracket { lower: f64, upper: f64 },
    #[error("hard-wall energy tolerance must be finite and positive, got {0} Ha")]
    InvalidEnergyTolerance(f64),
    #[error("hard-wall residual does not change sign: f({lower})={f_lower}, f({upper})={f_upper}")]
    RootNotBracketed {
        lower: f64,
        upper: f64,
        f_lower: f64,
        f_upper: f64,
    },
    #[error("hard-wall root did not converge after {iterations} iterations")]
    RootDidNotConverge { iterations: usize },
    #[error("mesh quadrature failed: {0}")]
    Quadrature(String),
}

/// Solver bound to one mesh, potential, and valence equation.
#[derive(Clone, Copy, Debug)]
pub struct RadialSolver<'a> {
    mesh: &'a ExponentialMesh,
    potential: &'a [f64],
    equation: RadialEquation,
}

impl<'a> RadialSolver<'a> {
    /// Bind a solver to the physical spherical potential `V(r)` in Hartree.
    ///
    /// If a caller starts from a spherical-harmonic coefficient, it must first
    /// convert it to the physical spherical average (for SPEX/FLEUR-style
    /// normalized harmonics, `V = v_00 / sqrt(4π)`).
    pub fn new(
        mesh: &'a ExponentialMesh,
        potential: &'a [f64],
        equation: RadialEquation,
    ) -> Result<Self, RadialError> {
        validate_potential(mesh, potential)?;
        Ok(Self {
            mesh,
            potential,
            equation,
        })
    }

    pub const fn equation(&self) -> RadialEquation {
        self.equation
    }

    pub fn solve(
        &self,
        angular_momentum: u32,
        energy: Hartree,
    ) -> Result<RadialSolution, RadialError> {
        let raw = self.integrate_homogeneous(angular_momentum, energy)?;
        self.normalize(raw, angular_momentum, energy)
    }

    pub fn solve_with_energy_derivative(
        &self,
        angular_momentum: u32,
        energy: Hartree,
    ) -> Result<LinearizedRadialSolution, RadialError> {
        let solution = self.solve(angular_momentum, energy)?;
        let energy_derivative = self.integrate_energy_derivative(&solution)?;
        Ok(LinearizedRadialSolution {
            solution,
            energy_derivative,
        })
    }

    /// Build and normalize a matched local orbital at `lo_energy`.
    pub fn local_orbital(
        &self,
        linearized: &LinearizedRadialSolution,
        lo_energy: Hartree,
    ) -> Result<LocalOrbital, RadialError> {
        if linearized.solution.equation != self.equation {
            return Err(RadialError::EquationMismatch {
                solver: self.equation,
                solution: linearized.solution.equation,
            });
        }
        let raw = self.solve(linearized.solution.angular_momentum, lo_energy)?;
        let u = &linearized.solution;
        let udot = &linearized.energy_derivative;
        ensure_mesh_length(self.mesh, u.p.len())?;
        ensure_mesh_length(self.mesh, udot.p.len())?;

        let det = u.boundary.value * udot.boundary.derivative
            - udot.boundary.value * u.boundary.derivative;
        let determinant_scale = (u.boundary.value.abs() * udot.boundary.derivative.abs())
            .max(udot.boundary.value.abs() * u.boundary.derivative.abs())
            .max(1.0);
        if det.abs() <= 256.0 * f64::EPSILON * determinant_scale {
            return Err(RadialError::SingularLocalOrbital { determinant: det });
        }
        let rhs_value = -raw.boundary.value;
        let rhs_slope = -raw.boundary.derivative;
        let a = (rhs_value * udot.boundary.derivative - udot.boundary.value * rhs_slope) / det;
        let b = (u.boundary.value * rhs_slope - rhs_value * u.boundary.derivative) / det;

        let mut p: Vec<f64> = raw
            .p
            .iter()
            .zip(&u.p)
            .zip(&udot.p)
            .map(|((&raw, &base), &dot)| raw + a * base + b * dot)
            .collect();
        let mut q = match (&raw.q, &u.q, &udot.q) {
            (Some(raw), Some(base), Some(dot)) => Some(
                raw.iter()
                    .zip(base)
                    .zip(dot)
                    .map(|((&raw, &base), &dot)| raw + a * base + b * dot)
                    .collect::<Vec<_>>(),
            ),
            (None, None, None) => None,
            _ => return Err(RadialError::ComponentMismatch),
        };
        let norm_squared = component_norm_squared(self.mesh, &p, q.as_deref())?;
        if !norm_squared.is_finite() || norm_squared <= f64::MIN_POSITIVE {
            return Err(RadialError::SingularNorm { norm_squared });
        }
        let scale = norm_squared.sqrt().recip();
        p.iter_mut().for_each(|x| *x *= scale);
        if let Some(q) = &mut q {
            q.iter_mut().for_each(|x| *x *= scale);
        }
        let radius = self.mesh.last().get();
        let value = scale * (raw.boundary.value + a * u.boundary.value + b * udot.boundary.value);
        let derivative = scale
            * (raw.boundary.derivative + a * u.boundary.derivative + b * udot.boundary.derivative);
        Ok(LocalOrbital {
            energy: lo_energy,
            p,
            q,
            coefficients: LocalOrbitalCoefficients {
                a,
                b,
                normalization_scale: scale,
            },
            boundary: BoundaryData::new(value, derivative, radius),
        })
    }

    /// Bracketed hard-wall helper.  It is useful for analytic tests and finite
    /// spherical boxes; it is not a core-state boundary condition.
    pub fn hard_wall_eigenenergy(
        &self,
        angular_momentum: u32,
        bracket: EnergyBracket,
        energy_tolerance: Hartree,
        max_iterations: usize,
    ) -> Result<Hartree, RadialError> {
        let (mut lo, mut hi) = bracket.values();
        let tolerance = energy_tolerance.get();
        if !lo.is_finite() || !hi.is_finite() || lo >= hi {
            return Err(RadialError::InvalidEnergyBracket {
                lower: lo,
                upper: hi,
            });
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(RadialError::InvalidEnergyTolerance(tolerance));
        }
        let mut flo = self
            .integrate_homogeneous(angular_momentum, Hartree(lo))?
            .p
            .last()
            .copied()
            .unwrap_or(0.0);
        let fhi = self
            .integrate_homogeneous(angular_momentum, Hartree(hi))?
            .p
            .last()
            .copied()
            .unwrap_or(0.0);
        if flo == 0.0 {
            return Ok(Hartree(lo));
        }
        if fhi == 0.0 {
            return Ok(Hartree(hi));
        }
        if flo.signum() == fhi.signum() {
            return Err(RadialError::RootNotBracketed {
                lower: lo,
                upper: hi,
                f_lower: flo,
                f_upper: fhi,
            });
        }
        for _ in 0..max_iterations {
            let mid = lo + 0.5 * (hi - lo);
            let fmid = self
                .integrate_homogeneous(angular_momentum, Hartree(mid))?
                .p
                .last()
                .copied()
                .unwrap_or(0.0);
            if fmid == 0.0 || 0.5 * (hi - lo) <= tolerance {
                return Ok(Hartree(mid));
            }
            if fmid.signum() == flo.signum() {
                lo = mid;
                flo = fmid;
            } else {
                hi = mid;
            }
        }
        Err(RadialError::RootDidNotConverge {
            iterations: max_iterations,
        })
    }

    fn integrate_homogeneous(
        &self,
        angular_momentum: u32,
        energy: Hartree,
    ) -> Result<InternalSolution, RadialError> {
        let e = energy.get();
        if !e.is_finite() {
            return Err(RadialError::NonFiniteEnergy(e));
        }
        let n = self.mesh.len();
        let mut p = vec![0.0; n];
        let mut q = vec![0.0; n];
        let r0 = self.mesh.first().get();
        let l = f64::from(angular_momentum);
        let ll = l * (l + 1.0);
        let cci = self.equation.c_inverse_squared();
        if cci != 0.0 {
            for (index, &potential) in self.potential.iter().enumerate() {
                let mass = 2.0 + (e - potential) * cci;
                if !mass.is_finite() || mass <= 0.0 {
                    return Err(RadialError::InvalidRelativisticMass { index, mass });
                }
            }
        }
        let radicand = 1.0 + ll - (self.potential[0] * r0).powi(2) * cci;
        if radicand <= 0.0 || !radicand.is_finite() {
            return Err(RadialError::SupercriticalOrigin { radicand });
        }
        let exponent = radicand.sqrt();
        let m0 = 2.0 + (e - self.potential[0]) * cci;
        // SPEX deliberately starts the large component with the
        // nonrelativistic regular power and uses the relativistic exponent in
        // the small/large ratio.
        p[0] = r0.powf(f64::from(angular_momentum) + 1.0);
        q[0] = p[0] * (exponent - 1.0) / (m0 * r0);

        for i in 0..n - 1 {
            let ra = self.mesh.radii()[i].get();
            let rc = self.mesh.radii()[i + 1].get();
            let rb = 0.5 * (ra + rc);
            let dr = rc - ra;
            let vb = rv_midpoint(ra, rc, self.potential[i], self.potential[i + 1]);
            let (k1, l1) = scalar_rhs(ra, self.potential[i], p[i], q[i], ll, e, cci);
            let (k2, l2) = scalar_rhs(
                rb,
                vb,
                p[i] + 0.5 * dr * k1,
                q[i] + 0.5 * dr * l1,
                ll,
                e,
                cci,
            );
            let (k3, l3) = scalar_rhs(
                rb,
                vb,
                p[i] + 0.5 * dr * k2,
                q[i] + 0.5 * dr * l2,
                ll,
                e,
                cci,
            );
            let (k4, l4) = scalar_rhs(
                rc,
                self.potential[i + 1],
                p[i] + dr * k3,
                q[i] + dr * l3,
                ll,
                e,
                cci,
            );
            p[i + 1] = p[i] + dr * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            q[i + 1] = q[i] + dr * (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0;
            if !p[i + 1].is_finite() || !q[i + 1].is_finite() || p[i + 1].abs() > 1.0e150 {
                return Err(RadialError::IntegrationOverflow { index: i + 1 });
            }
        }
        Ok(InternalSolution { p, q_tilde: q })
    }

    fn normalize(
        &self,
        raw: InternalSolution,
        angular_momentum: u32,
        energy: Hartree,
    ) -> Result<RadialSolution, RadialError> {
        let cci = self.equation.c_inverse_squared();
        let density: Vec<f64> = raw
            .p
            .iter()
            .zip(&raw.q_tilde)
            .map(|(&p, &q)| p * p + cci * q * q)
            .collect();
        let norm_squared = integrate(self.mesh, &density)?;
        if !norm_squared.is_finite() || norm_squared <= f64::MIN_POSITIVE {
            return Err(RadialError::SingularNorm { norm_squared });
        }
        let scale = norm_squared.sqrt().recip();
        let p: Vec<f64> = raw.p.into_iter().map(|x| x * scale).collect();
        let q_tilde: Vec<f64> = raw.q_tilde.into_iter().map(|x| x * scale).collect();
        let physical_q = (self.equation == RadialEquation::ScalarKoellingHarmon)
            .then(|| q_tilde.iter().map(|q| q / SPEX_SPEED_OF_LIGHT).collect());
        let boundary = self.boundary_from_internal(
            angular_momentum,
            energy.get(),
            p.last().copied().unwrap_or(0.0),
            q_tilde.last().copied().unwrap_or(0.0),
        );
        Ok(RadialSolution {
            equation: self.equation,
            angular_momentum,
            energy,
            p,
            q: physical_q,
            auxiliary_q: q_tilde,
            boundary,
        })
    }

    fn integrate_energy_derivative(
        &self,
        solution: &RadialSolution,
    ) -> Result<EnergyDerivative, RadialError> {
        let n = self.mesh.len();
        ensure_mesh_length(self.mesh, solution.p.len())?;
        let cci = self.equation.c_inverse_squared();
        let q_base = solution.auxiliary_q.clone();
        let mut pdot = vec![0.0; n];
        let mut qdot = vec![0.0; n];
        if cci != 0.0 {
            let m0 = 2.0 + (solution.energy.get() - self.potential[0]) * cci;
            qdot[0] = -q_base[0] * cci / m0;
        }
        let l = f64::from(solution.angular_momentum);
        let ll = l * (l + 1.0);
        let e = solution.energy.get();
        for i in 0..n - 1 {
            let ra = self.mesh.radii()[i].get();
            let rc = self.mesh.radii()[i + 1].get();
            let rb = 0.5 * (ra + rc);
            let dr = rc - ra;
            let vb = rv_midpoint(ra, rc, self.potential[i], self.potential[i + 1]);

            // Differentiate the same discrete RK4 map used by the homogeneous
            // solve.  Endpoint averages are not its midpoint stages and lose
            // several orders of accuracy in boundary derivatives.
            let (base_k1_p, base_k1_q) =
                scalar_rhs(ra, self.potential[i], solution.p[i], q_base[i], ll, e, cci);
            let base_y2_p = solution.p[i] + 0.5 * dr * base_k1_p;
            let base_y2_q = q_base[i] + 0.5 * dr * base_k1_q;
            let (base_k2_p, base_k2_q) = scalar_rhs(rb, vb, base_y2_p, base_y2_q, ll, e, cci);
            let base_y3_p = solution.p[i] + 0.5 * dr * base_k2_p;
            let base_y3_q = q_base[i] + 0.5 * dr * base_k2_q;
            let (base_k3_p, base_k3_q) = scalar_rhs(rb, vb, base_y3_p, base_y3_q, ll, e, cci);
            let base_y4_p = solution.p[i] + dr * base_k3_p;
            let base_y4_q = q_base[i] + dr * base_k3_q;

            let (s1_p, s1_q) = sensitivity_rhs(
                ra,
                self.potential[i],
                pdot[i],
                qdot[i],
                solution.p[i],
                q_base[i],
                ll,
                e,
                cci,
            );
            let (s2_p, s2_q) = sensitivity_rhs(
                rb,
                vb,
                pdot[i] + 0.5 * dr * s1_p,
                qdot[i] + 0.5 * dr * s1_q,
                base_y2_p,
                base_y2_q,
                ll,
                e,
                cci,
            );
            let (s3_p, s3_q) = sensitivity_rhs(
                rb,
                vb,
                pdot[i] + 0.5 * dr * s2_p,
                qdot[i] + 0.5 * dr * s2_q,
                base_y3_p,
                base_y3_q,
                ll,
                e,
                cci,
            );
            let (s4_p, s4_q) = sensitivity_rhs(
                rc,
                self.potential[i + 1],
                pdot[i] + dr * s3_p,
                qdot[i] + dr * s3_q,
                base_y4_p,
                base_y4_q,
                ll,
                e,
                cci,
            );
            pdot[i + 1] = pdot[i] + dr * (s1_p + 2.0 * s2_p + 2.0 * s3_p + s4_p) / 6.0;
            qdot[i + 1] = qdot[i] + dr * (s1_q + 2.0 * s2_q + 2.0 * s3_q + s4_q) / 6.0;
        }

        let overlap_density: Vec<f64> = solution
            .p
            .iter()
            .zip(&pdot)
            .zip(q_base.iter().zip(&qdot))
            .map(|((&p, &pdot), (&q, &qdot))| p * pdot + cci * q * qdot)
            .collect();
        let projection = integrate(self.mesh, &overlap_density)?;
        for i in 0..n {
            pdot[i] -= projection * solution.p[i];
            qdot[i] -= projection * q_base[i];
        }
        let r = self.mesh.last().get();
        let v = self.potential[n - 1];
        let m = 2.0 + (e - v) * cci;
        let pprime = pdot[n - 1] / r + m * qdot[n - 1] + cci * q_base[n - 1];
        let value = pdot[n - 1] / r;
        let derivative = (pprime - pdot[n - 1] / r) / r;
        let physical_q = (self.equation == RadialEquation::ScalarKoellingHarmon)
            .then(|| qdot.into_iter().map(|q| q / SPEX_SPEED_OF_LIGHT).collect());
        let norm_squared = component_norm_squared(self.mesh, &pdot, physical_q.as_deref())?;
        Ok(EnergyDerivative {
            p: pdot,
            q: physical_q,
            boundary: BoundaryData::new(value, derivative, r),
            norm_squared,
        })
    }

    fn boundary_from_internal(
        &self,
        angular_momentum: u32,
        energy: f64,
        p: f64,
        q_tilde: f64,
    ) -> BoundaryData {
        let index = self.mesh.len() - 1;
        let r = self.mesh.radii()[index].get();
        let l = f64::from(angular_momentum);
        let ll = l * (l + 1.0);
        let (pprime, _) = scalar_rhs(
            r,
            self.potential[index],
            p,
            q_tilde,
            ll,
            energy,
            self.equation.c_inverse_squared(),
        );
        let value = p / r;
        BoundaryData::new(value, (pprime - p / r) / r, r)
    }
}

#[derive(Clone, Debug)]
struct InternalSolution {
    p: Vec<f64>,
    q_tilde: Vec<f64>,
}

fn scalar_rhs(
    r: f64,
    potential: f64,
    p: f64,
    q: f64,
    ll: f64,
    energy: f64,
    cci: f64,
) -> (f64, f64) {
    let m = 2.0 + (energy - potential) * cci;
    let w = ll / (m * r * r) + potential - energy;
    (p / r + m * q, -q / r + w * p)
}

#[allow(clippy::too_many_arguments)]
fn sensitivity_rhs(
    r: f64,
    potential: f64,
    pdot: f64,
    qdot: f64,
    p: f64,
    q: f64,
    ll: f64,
    energy: f64,
    cci: f64,
) -> (f64, f64) {
    let m = 2.0 + (energy - potential) * cci;
    let wh = ll / (m * r * r);
    let w = wh + potential - energy;
    let dm = cci;
    let dw = -1.0 - wh * cci / m;
    (pdot / r + m * qdot + dm * q, -qdot / r + w * pdot + dw * p)
}

/// SPEX's optional `BAS_DIFF_R` branch: use the arithmetic-radius RK midpoint
/// and linearly interpolate `r V(r)`, not `V(r)`, to that radius.
fn rv_midpoint(ra: f64, rc: f64, va: f64, vc: f64) -> f64 {
    (ra * va + rc * vc) / (ra + rc)
}

fn validate_potential(mesh: &ExponentialMesh, potential: &[f64]) -> Result<(), RadialError> {
    ensure_mesh_length(mesh, potential.len()).map_err(|_| RadialError::PotentialLength {
        expected: mesh.len(),
        actual: potential.len(),
    })?;
    for (index, &value) in potential.iter().enumerate() {
        if !value.is_finite() {
            return Err(RadialError::NonFinitePotential { index, value });
        }
    }
    Ok(())
}

fn ensure_mesh_length(mesh: &ExponentialMesh, actual: usize) -> Result<(), RadialError> {
    if actual == mesh.len() {
        Ok(())
    } else {
        Err(RadialError::ArrayLength {
            expected: mesh.len(),
            actual,
        })
    }
}

pub(crate) fn integrate(mesh: &ExponentialMesh, values: &[f64]) -> Result<f64, RadialError> {
    mesh.integrate(values)
        .map_err(|error| RadialError::Quadrature(error.to_string()))
}

pub(crate) fn component_norm_squared(
    mesh: &ExponentialMesh,
    p: &[f64],
    q: Option<&[f64]>,
) -> Result<f64, RadialError> {
    ensure_mesh_length(mesh, p.len())?;
    if let Some(q) = q {
        ensure_mesh_length(mesh, q.len())?;
    }
    let density: Vec<f64> = match q {
        Some(q) => p.iter().zip(q).map(|(&p, &q)| p * p + q * q).collect(),
        None => p.iter().map(|&p| p * p).collect(),
    };
    integrate(mesh, &density)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mt_core::Bohr;

    fn mesh(first: f64, last: f64, increment: f64) -> ExponentialMesh {
        let number = ((last / first).ln() / increment).ceil() as usize + 1;
        ExponentialMesh::new(Bohr(first), increment, number).unwrap()
    }

    #[test]
    fn square_well_s_wave_log_derivative_matches_analytic_result() {
        let mesh = mesh(1.0e-7, 3.0, 0.0015);
        let well_bottom = -0.35;
        let potential = vec![well_bottom; mesh.len()];
        let solver = RadialSolver::new(&mesh, &potential, RadialEquation::Schroedinger).unwrap();
        let energy = Hartree(0.73);
        let solution = solver.solve(0, energy).unwrap();
        let radius = mesh.last().get();
        let kr = (2.0 * (energy.get() - well_bottom)).sqrt() * radius;
        let expected = kr / kr.tan() - 1.0;
        assert!((solution.boundary.scaled_log_derivative.unwrap() - expected).abs() < 1.0e-10);
    }

    #[test]
    fn hard_wall_free_sphere_energy_meets_milestone_tolerance() {
        let mesh = mesh(1.0e-7, 5.0, 0.001);
        let potential = vec![0.0; mesh.len()];
        let solver = RadialSolver::new(&mesh, &potential, RadialEquation::Schroedinger).unwrap();
        let radius = mesh.last().get();
        let exact = std::f64::consts::PI.powi(2) / (2.0 * radius * radius);
        let energy = solver
            .hard_wall_eigenenergy(
                0,
                EnergyBracket::new(Hartree(0.8 * exact), Hartree(1.2 * exact)).unwrap(),
                Hartree(2.0e-13),
                80,
            )
            .unwrap();
        assert!((energy.get() - exact).abs() < 1.0e-10);
    }

    #[test]
    fn hydrogenic_ground_state_energy_meets_milestone_tolerance() {
        // The SPEX regular-origin start retains only the leading power.  A
        // sufficiently small r0 makes the omitted Coulomb series term much
        // smaller than the M-B energy tolerance.
        let mesh = mesh(1.0e-12, 40.0, 0.001);
        let potential: Vec<f64> = mesh.radii().iter().map(|r| -1.0 / r.get()).collect();
        let solver = RadialSolver::new(&mesh, &potential, RadialEquation::Schroedinger).unwrap();
        let energy = solver
            .hard_wall_eigenenergy(
                0,
                EnergyBracket::new(Hartree(-0.6), Hartree(-0.4)).unwrap(),
                Hartree(2.0e-13),
                80,
            )
            .unwrap();
        assert!((energy.get() + 0.5).abs() < 1.0e-10);
    }

    #[test]
    fn energy_derivative_matches_finite_difference_and_is_orthogonal() {
        let mesh = mesh(1.0e-7, 4.0, 0.0015);
        let potential: Vec<f64> = mesh.radii().iter().map(|r| -1.0 / r.get()).collect();
        let solver = RadialSolver::new(&mesh, &potential, RadialEquation::Schroedinger).unwrap();
        let energy = Hartree(-0.37);
        let pair = solver.solve_with_energy_derivative(0, energy).unwrap();
        let step = 2.0e-5;
        let plus = solver.solve(0, Hartree(energy.get() + step)).unwrap();
        let minus = solver.solve(0, Hartree(energy.get() - step)).unwrap();
        let fd: Vec<f64> = plus
            .p
            .iter()
            .zip(&minus.p)
            .map(|(&a, &b)| (a - b) / (2.0 * step))
            .collect();
        let error_density: Vec<f64> = fd
            .iter()
            .zip(&pair.energy_derivative.p)
            .map(|(&a, &b)| (a - b).powi(2))
            .collect();
        assert!(integrate(&mesh, &error_density).unwrap().sqrt() < 2.0e-5);
        let overlap_density: Vec<f64> = pair
            .solution
            .p
            .iter()
            .zip(&pair.energy_derivative.p)
            .map(|(&a, &b)| a * b)
            .collect();
        assert!(integrate(&mesh, &overlap_density).unwrap().abs() < 2.0e-12);
    }

    #[test]
    fn local_orbital_is_normalized_and_matched() {
        let mesh = mesh(1.0e-7, 3.5, 0.0015);
        let potential: Vec<f64> = mesh.radii().iter().map(|r| -1.0 / r.get()).collect();
        let solver = RadialSolver::new(&mesh, &potential, RadialEquation::Schroedinger).unwrap();
        let pair = solver
            .solve_with_energy_derivative(1, Hartree(-0.18))
            .unwrap();
        let local = solver.local_orbital(&pair, Hartree(0.4)).unwrap();
        assert!(local.boundary.value.abs() < 2.0e-12);
        assert!(local.boundary.derivative.abs() < 2.0e-12);
        assert!((component_norm_squared(&mesh, &local.p, None).unwrap() - 1.0).abs() < 2.0e-12);
    }

    #[test]
    fn koelling_harmon_approaches_schroedinger_for_light_smooth_problem() {
        let mesh = mesh(1.0e-6, 2.5, 0.0015);
        let potential = vec![-0.2; mesh.len()];
        let sch = RadialSolver::new(&mesh, &potential, RadialEquation::Schroedinger)
            .unwrap()
            .solve(1, Hartree(0.5))
            .unwrap();
        let kh = RadialSolver::new(&mesh, &potential, RadialEquation::ScalarKoellingHarmon)
            .unwrap()
            .solve(1, Hartree(0.5))
            .unwrap();
        let difference: Vec<f64> = sch
            .p
            .iter()
            .zip(&kh.p)
            .map(|(&a, &b)| (a - b).powi(2))
            .collect();
        assert!(integrate(&mesh, &difference).unwrap().sqrt() < 2.0e-4);
        assert!(kh.q.as_ref().unwrap().iter().all(|q| q.is_finite()));
    }

    #[test]
    fn koelling_harmon_energy_derivative_is_exact_and_orthogonal() {
        let mesh = mesh(1.0e-6, 3.0, 0.0015);
        let potential: Vec<f64> = mesh.radii().iter().map(|r| -0.4 / r.get()).collect();
        let solver =
            RadialSolver::new(&mesh, &potential, RadialEquation::ScalarKoellingHarmon).unwrap();
        let energy = Hartree(-0.11);
        let pair = solver.solve_with_energy_derivative(1, energy).unwrap();
        let step = 2.0e-5;
        let plus = solver.solve(1, Hartree(energy.get() + step)).unwrap();
        let minus = solver.solve(1, Hartree(energy.get() - step)).unwrap();
        let fd_boundary_value = (plus.boundary.value - minus.boundary.value) / (2.0 * step);
        let fd_boundary_derivative =
            (plus.boundary.derivative - minus.boundary.derivative) / (2.0 * step);
        assert!((pair.energy_derivative.boundary.value - fd_boundary_value).abs() < 1.0e-9);
        assert!(
            (pair.energy_derivative.boundary.derivative - fd_boundary_derivative).abs() < 1.0e-9
        );
        let fd_p: Vec<f64> = plus
            .p
            .iter()
            .zip(&minus.p)
            .map(|(&a, &b)| (a - b) / (2.0 * step))
            .collect();
        let fd_q: Vec<f64> = plus
            .q
            .as_ref()
            .unwrap()
            .iter()
            .zip(minus.q.as_ref().unwrap())
            .map(|(&a, &b)| (a - b) / (2.0 * step))
            .collect();
        let error: Vec<f64> = fd_p
            .iter()
            .zip(&pair.energy_derivative.p)
            .zip(fd_q.iter().zip(pair.energy_derivative.q.as_ref().unwrap()))
            .map(|((&a, &b), (&aq, &bq))| (a - b).powi(2) + (aq - bq).powi(2))
            .collect();
        assert!(integrate(&mesh, &error).unwrap().sqrt() < 2.0e-5);
        let overlap = crate::radial_integral(
            &mesh,
            &pair.solution,
            &pair.energy_derivative,
            crate::RadialIntegralKernel::Overlap,
        )
        .unwrap();
        assert!(overlap.abs() < 2.0e-12);
        assert!(pair.energy_derivative.norm_squared > 0.0);
    }
}
