//! SPEX-style scalar-relativistic radial factors for second-variation SOC.
//!
//! This is the Koelling--Harmon approximation used by SPEX, not a
//! four-component sphere treatment.  The sampled factor is
//! `dV/dr / (4 c^2 r)` and the angular matrix used by the operator layer is
//! `sigma dot L`; together these are the usual `xi(r) L dot S` operator.

use muffintin_core::{ExponentialMesh, Hartree, MeshError};
use thiserror::Error;

use crate::{
    EnergyDerivative, LinearizedRadialSolution, LocalOrbital, RadialEquation, RadialSolution,
    SPEX_SPEED_OF_LIGHT,
};

const TWO_C_SQUARED: f64 = 2.0 * SPEX_SPEED_OF_LIGHT * SPEX_SPEED_OF_LIGHT;
const FOUR_C_SQUARED: f64 = 2.0 * TWO_C_SQUARED;

/// Spherical potential and the SPEX SOC factor sampled on one radial mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct SpexSpinOrbitPotential {
    potential: Vec<f64>,
    derivative: Vec<f64>,
    /// `dV/dr / (4 c^2 r)`, before the Koelling--Harmon mass factors.
    xi_half: Vec<f64>,
}

impl SpexSpinOrbitPotential {
    /// Differentiate a physical spherical potential on the SPEX exponential
    /// mesh and form `dV/dr / (4 c^2 r)`.
    ///
    /// [`ExponentialMesh`] begins at a strictly positive radius.  Therefore no
    /// artificial value at `r = 0` is introduced; the regular-origin
    /// contribution remains controlled by the radial functions and SPEX
    /// quadrature.
    pub fn new(mesh: &ExponentialMesh, potential: &[f64]) -> Result<Self, SpinOrbitRadialError> {
        validate_samples(mesh, potential, "potential")?;
        let derivative = spex_derivative(mesh, potential);
        let xi_half = derivative
            .iter()
            .zip(mesh.radii())
            .map(|(&dv, radius)| dv / (FOUR_C_SQUARED * radius.get()))
            .collect();
        Ok(Self {
            potential: potential.to_vec(),
            derivative,
            xi_half,
        })
    }

    pub fn potential(&self) -> &[f64] {
        &self.potential
    }

    pub fn derivative(&self) -> &[f64] {
        &self.derivative
    }

    /// SPEX's radial factor multiplying `sigma dot L`.
    pub fn xi_half(&self) -> &[f64] {
        &self.xi_half
    }

    /// Conventional `xi(r)` multiplying `L dot S`.
    pub fn xi(&self) -> Vec<f64> {
        self.xi_half.iter().map(|value| 2.0 * value).collect()
    }
}

/// Real symmetric SOC radial matrix for one angular-momentum shell.
///
/// Coordinates are `(u_l, du_l/dE, LO_0, LO_1, ...)`, matching one fixed
/// `(l,m)` slice of the scalar site projection.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinOrbitRadialShell {
    angular_momentum: u32,
    dimension: usize,
    values: Vec<f64>,
}

impl SpinOrbitRadialShell {
    pub const fn angular_momentum(&self) -> u32 {
        self.angular_momentum
    }

    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn at(&self, row: usize, column: usize) -> f64 {
        self.values[row * self.dimension + column]
    }

    pub fn as_row_major(&self) -> &[f64] {
        &self.values
    }
}

/// Build one SPEX/Koelling--Harmon radial SOC shell.
///
/// The primitive basis is `(u(E_l), du/dE, u(E_LO), ...)`.  The exact SPEX
/// energy-dependent mass hermitianization and the additional `du/dE` terms
/// are evaluated there.  The result is then transformed to the actual
/// normalized matched local orbitals stored by [`LocalOrbital`].
pub fn spex_spin_orbit_radial_shell(
    mesh: &ExponentialMesh,
    potential: &SpexSpinOrbitPotential,
    linearized: &LinearizedRadialSolution,
    local_orbitals: &[LocalOrbital],
) -> Result<SpinOrbitRadialShell, SpinOrbitRadialError> {
    let solution = &linearized.solution;
    let derivative = &linearized.energy_derivative;
    validate_kh_pair(mesh, solution, derivative)?;
    if potential.potential.len() != mesh.len() {
        return Err(SpinOrbitRadialError::ArrayLength {
            array: "SOC potential",
            expected: mesh.len(),
            actual: potential.potential.len(),
        });
    }

    let dimension = 2 + local_orbitals.len();
    let mut primitive_radials = Vec::with_capacity(dimension);
    let mut energies = Vec::with_capacity(dimension);
    primitive_radials.push(solution.p.clone());
    primitive_radials.push(derivative.p.clone());
    energies.push(solution.energy());
    energies.push(solution.energy());

    for local in local_orbitals {
        validate_samples(mesh, &local.p, "local orbital")?;
        if local.q.is_none() {
            return Err(SpinOrbitRadialError::NonKoellingHarmonLocalOrbital);
        }
        let coefficients = local.coefficients;
        let scale = coefficients.normalization_scale;
        if !scale.is_finite() || scale == 0.0 {
            return Err(SpinOrbitRadialError::InvalidLocalOrbitalScale(scale));
        }
        let raw = local
            .p
            .iter()
            .zip(&solution.p)
            .zip(&derivative.p)
            .map(|((&matched, &u), &udot)| {
                matched / scale - coefficients.a * u - coefficients.b * udot
            })
            .collect();
        primitive_radials.push(raw);
        energies.push(local.energy);
    }

    let mut primitive = vec![0.0; dimension * dimension];
    for left in 0..dimension {
        for right in left..dimension {
            let value = primitive_soc_integral(
                mesh,
                potential,
                &primitive_radials,
                &energies,
                left,
                right,
            )?;
            primitive[left * dimension + right] = value;
            primitive[right * dimension + left] = value;
        }
    }

    // primitive_radial[p] * transform[p, coordinate]
    let mut transform = vec![0.0; dimension * dimension];
    transform[0] = 1.0;
    transform[dimension + 1] = 1.0;
    for (local_index, local) in local_orbitals.iter().enumerate() {
        let coordinate = 2 + local_index;
        let coefficients = local.coefficients;
        let scale = coefficients.normalization_scale;
        transform[coordinate] = scale * coefficients.a;
        transform[dimension + coordinate] = scale * coefficients.b;
        transform[coordinate * dimension + coordinate] = scale;
    }

    let mut values = vec![0.0; dimension * dimension];
    for left in 0..dimension {
        for right in left..dimension {
            let mut value = 0.0;
            for p in 0..dimension {
                for q in 0..dimension {
                    value += transform[p * dimension + left]
                        * primitive[p * dimension + q]
                        * transform[q * dimension + right];
                }
            }
            values[left * dimension + right] = value;
            values[right * dimension + left] = value;
        }
    }

    Ok(SpinOrbitRadialShell {
        angular_momentum: solution.angular_momentum(),
        dimension,
        values,
    })
}

fn primitive_soc_integral(
    mesh: &ExponentialMesh,
    soc: &SpexSpinOrbitPotential,
    radials: &[Vec<f64>],
    energies: &[Hartree],
    left: usize,
    right: usize,
) -> Result<f64, SpinOrbitRadialError> {
    let left_mass = reciprocal_mass_powers(soc, energies[left], left)?;
    let right_mass = reciprocal_mass_powers(soc, energies[right], right)?;
    let mut integrand = Vec::with_capacity(mesh.len());
    for index in 0..mesh.len() {
        let mass_factor = 0.5 * (left_mass[index].0 + right_mass[index].0);
        let mut value =
            soc.xi_half[index] * mass_factor * radials[left][index] * radials[right][index];
        if left == 1 {
            value -=
                soc.xi_half[index] * left_mass[index].1 * radials[0][index] * radials[right][index]
                    / TWO_C_SQUARED;
        }
        if right == 1 {
            value -=
                soc.xi_half[index] * right_mass[index].1 * radials[left][index] * radials[0][index]
                    / TWO_C_SQUARED;
        }
        if left == 1 && right == 1 {
            value -= soc.xi_half[index]
                * (left_mass[index].1 + right_mass[index].1)
                * radials[0][index]
                * radials[0][index]
                / TWO_C_SQUARED;
        }
        integrand.push(value);
    }
    mesh.integrate(&integrand).map_err(Into::into)
}

/// `(M^-2, M^-3)` for `M = 1 + (E - V)/(2c^2)`.
fn reciprocal_mass_powers(
    soc: &SpexSpinOrbitPotential,
    energy: Hartree,
    radial_index: usize,
) -> Result<Vec<(f64, f64)>, SpinOrbitRadialError> {
    soc.potential
        .iter()
        .enumerate()
        .map(|(mesh_index, &value)| {
            let mass = 1.0 + (energy.get() - value) / TWO_C_SQUARED;
            if !mass.is_finite() || mass <= 0.0 {
                return Err(SpinOrbitRadialError::InvalidRelativisticMass {
                    radial_index,
                    mesh_index,
                    mass,
                });
            }
            let inverse = mass.recip();
            Ok((inverse * inverse, inverse * inverse * inverse))
        })
        .collect()
}

fn validate_kh_pair(
    mesh: &ExponentialMesh,
    solution: &RadialSolution,
    derivative: &EnergyDerivative,
) -> Result<(), SpinOrbitRadialError> {
    if solution.equation() != RadialEquation::ScalarKoellingHarmon
        || solution.q.is_none()
        || derivative.q.is_none()
    {
        return Err(SpinOrbitRadialError::RequiresScalarKoellingHarmon);
    }
    validate_samples(mesh, &solution.p, "u")?;
    validate_samples(mesh, &derivative.p, "du/dE")
}

fn validate_samples(
    mesh: &ExponentialMesh,
    values: &[f64],
    array: &'static str,
) -> Result<(), SpinOrbitRadialError> {
    if values.len() != mesh.len() {
        return Err(SpinOrbitRadialError::ArrayLength {
            array,
            expected: mesh.len(),
            actual: values.len(),
        });
    }
    if let Some((index, &value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(SpinOrbitRadialError::NonFiniteSample {
            array,
            index,
            value,
        });
    }
    Ok(())
}

// SPEX `numerics.f:365-389`: fifth-order Lagrange derivative on x = log(r).
fn spex_derivative(mesh: &ExponentialMesh, values: &[f64]) -> Vec<f64> {
    let n = mesh.len();
    if values.iter().all(|value| *value == values[0]) {
        return vec![0.0; n];
    }
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

/// Invalid input to the scalar-relativistic SOC radial construction.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SpinOrbitRadialError {
    #[error("{array} has {actual} samples, expected {expected}")]
    ArrayLength {
        array: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{array}[{index}] is non-finite: {value}")]
    NonFiniteSample {
        array: &'static str,
        index: usize,
        value: f64,
    },
    #[error("second-variation SOC requires a scalar Koelling-Harmon radial pair")]
    RequiresScalarKoellingHarmon,
    #[error("second-variation SOC requires Koelling-Harmon local orbitals")]
    NonKoellingHarmonLocalOrbital,
    #[error("local-orbital normalization scale must be finite and nonzero, got {0}")]
    InvalidLocalOrbitalScale(f64),
    #[error(
        "SOC radial {radial_index} has invalid Koelling-Harmon mass {mass} at mesh point {mesh_index}"
    )]
    InvalidRelativisticMass {
        radial_index: usize,
        mesh_index: usize,
        mass: f64,
    },
    #[error(transparent)]
    Mesh(#[from] MeshError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RadialSolver;
    use muffintin_core::Bohr;

    #[test]
    fn spex_log_mesh_derivative_and_zero_soc_are_exact_enough() {
        let mesh = ExponentialMesh::new(Bohr(1.0e-5), 0.01, 301).unwrap();
        let quadratic = mesh
            .radii()
            .iter()
            .map(|radius| 0.4 * radius.get().powi(2) - 0.7)
            .collect::<Vec<_>>();
        let soc = SpexSpinOrbitPotential::new(&mesh, &quadratic).unwrap();
        for (index, radius) in mesh.radii().iter().enumerate().skip(2).take(297) {
            let expected = 0.8 * radius.get();
            assert!((soc.derivative()[index] - expected).abs() < 2.0e-8 * expected.abs().max(1.0));
        }

        let constant = vec![-0.3; mesh.len()];
        let zero = SpexSpinOrbitPotential::new(&mesh, &constant).unwrap();
        assert!(zero.xi_half().iter().all(|value| *value == 0.0));
    }

    #[test]
    fn kh_shell_is_real_symmetric_and_schroedinger_input_is_rejected() {
        let mesh = ExponentialMesh::new(Bohr(1.0e-5), 0.015, 401).unwrap();
        let potential = mesh
            .radii()
            .iter()
            .map(|radius| -0.8 / (radius.get() + 0.2))
            .collect::<Vec<_>>();
        let soc = SpexSpinOrbitPotential::new(&mesh, &potential).unwrap();
        let kh = RadialSolver::new(&mesh, &potential, RadialEquation::ScalarKoellingHarmon)
            .unwrap()
            .solve_with_energy_derivative(1, Hartree(-0.2))
            .unwrap();
        let shell = spex_spin_orbit_radial_shell(&mesh, &soc, &kh, &[]).unwrap();
        assert_eq!(shell.dimension(), 2);
        assert_eq!(shell.at(0, 1), shell.at(1, 0));
        assert!(shell.as_row_major().iter().all(|value| value.is_finite()));

        let nonrel = RadialSolver::new(&mesh, &potential, RadialEquation::Schroedinger)
            .unwrap()
            .solve_with_energy_derivative(1, Hartree(-0.2))
            .unwrap();
        assert_eq!(
            spex_spin_orbit_radial_shell(&mesh, &soc, &nonrel, &[]).unwrap_err(),
            SpinOrbitRadialError::RequiresScalarKoellingHarmon
        );
    }
}
