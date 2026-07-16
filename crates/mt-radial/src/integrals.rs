use mt_core::ExponentialMesh;
use thiserror::Error;

use crate::core_dirac::CoreDiracSolution;
use crate::valence::{EnergyDerivative, LocalOrbital, RadialSolution};

/// Borrowed large and optional physical-small radial components.
///
/// Components use the reduced convention (`p = r u`), so overlap is directly
/// `∫ (p₁ p₂ + Q₁ Q₂) dr`.
pub trait RadialComponents {
    fn large_component(&self) -> &[f64];
    fn small_component(&self) -> Option<&[f64]>;
}

impl RadialComponents for RadialSolution {
    fn large_component(&self) -> &[f64] {
        &self.p
    }

    fn small_component(&self) -> Option<&[f64]> {
        self.q.as_deref()
    }
}

impl RadialComponents for EnergyDerivative {
    fn large_component(&self) -> &[f64] {
        &self.p
    }

    fn small_component(&self) -> Option<&[f64]> {
        self.q.as_deref()
    }
}

impl RadialComponents for LocalOrbital {
    fn large_component(&self) -> &[f64] {
        &self.p
    }

    fn small_component(&self) -> Option<&[f64]> {
        self.q.as_deref()
    }
}

impl RadialComponents for CoreDiracSolution {
    fn large_component(&self) -> &[f64] {
        &self.p
    }

    fn small_component(&self) -> Option<&[f64]> {
        Some(&self.q)
    }
}

/// Kernel multiplying the reduced-component product in a radial integral.
#[derive(Clone, Copy, Debug)]
pub enum RadialIntegralKernel<'a> {
    /// Ordinary overlap.
    Overlap,
    /// Moment `∫ r^exponent (p₁p₂ + Q₁Q₂) dr`.
    Power(i32),
    /// An arbitrary sampled scalar weight.
    Samples(&'a [f64]),
    /// A sampled `v_LM(r)` potential channel.  The angular label is validated
    /// and retained at the call site, while angular Gaunt factors remain L0/L2.
    PotentialMultipole {
        angular_l: u32,
        angular_m: i32,
        values: &'a [f64],
    },
}

/// Invalid radial integral request.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RadialIntegralError {
    #[error("left radial component has {actual} samples, expected {expected}")]
    LeftLength { expected: usize, actual: usize },
    #[error("right radial component has {actual} samples, expected {expected}")]
    RightLength { expected: usize, actual: usize },
    #[error("small component has {actual} samples, expected {expected}")]
    SmallComponentLength { expected: usize, actual: usize },
    #[error("radial weight has {actual} samples, expected {expected}")]
    WeightLength { expected: usize, actual: usize },
    #[error("invalid multipole (l={angular_l}, m={angular_m})")]
    InvalidMultipole { angular_l: u32, angular_m: i32 },
    #[error("radial weight at index {index} is non-finite: {value}")]
    NonFiniteWeight { index: usize, value: f64 },
    #[error("weighted radial product at index {index} is non-finite")]
    NonFiniteProduct { index: usize },
    #[error("mesh quadrature failed: {0}")]
    Quadrature(String),
}

/// Evaluate overlap, weighted, moment, or `v_LM` radial integrals.
///
/// If exactly one operand lacks a small component, that component is treated
/// as zero.  This makes mixed nonrelativistic/scalar-relativistic diagnostics
/// well-defined without inventing a Schrödinger small component.
pub fn radial_integral<L: RadialComponents + ?Sized, R: RadialComponents + ?Sized>(
    mesh: &ExponentialMesh,
    left: &L,
    right: &R,
    kernel: RadialIntegralKernel<'_>,
) -> Result<f64, RadialIntegralError> {
    let n = mesh.len();
    let large_left = left.large_component();
    let large_right = right.large_component();
    if large_left.len() != n {
        return Err(RadialIntegralError::LeftLength {
            expected: n,
            actual: large_left.len(),
        });
    }
    if large_right.len() != n {
        return Err(RadialIntegralError::RightLength {
            expected: n,
            actual: large_right.len(),
        });
    }
    for small in [left.small_component(), right.small_component()]
        .into_iter()
        .flatten()
    {
        if small.len() != n {
            return Err(RadialIntegralError::SmallComponentLength {
                expected: n,
                actual: small.len(),
            });
        }
    }

    let sampled_weight = match kernel {
        RadialIntegralKernel::Overlap | RadialIntegralKernel::Power(_) => None,
        RadialIntegralKernel::Samples(values)
        | RadialIntegralKernel::PotentialMultipole { values, .. } => {
            if values.len() != n {
                return Err(RadialIntegralError::WeightLength {
                    expected: n,
                    actual: values.len(),
                });
            }
            if let Some((index, &value)) = values
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(RadialIntegralError::NonFiniteWeight { index, value });
            }
            Some(values)
        }
    };
    if let RadialIntegralKernel::PotentialMultipole {
        angular_l,
        angular_m,
        ..
    } = kernel
    {
        if angular_m.unsigned_abs() > angular_l {
            return Err(RadialIntegralError::InvalidMultipole {
                angular_l,
                angular_m,
            });
        }
    }

    let small_left = left.small_component();
    let small_right = right.small_component();
    let mut integrand = Vec::with_capacity(n);
    for index in 0..n {
        let mut component_product = large_left[index] * large_right[index];
        if let (Some(left), Some(right)) = (small_left, small_right) {
            component_product += left[index] * right[index];
        }
        let weight = match kernel {
            RadialIntegralKernel::Overlap => 1.0,
            RadialIntegralKernel::Power(exponent) => mesh.radii()[index].get().powi(exponent),
            RadialIntegralKernel::Samples(_) | RadialIntegralKernel::PotentialMultipole { .. } => {
                sampled_weight.expect("sampled kernel was validated")[index]
            }
        };
        let value = component_product * weight;
        if !value.is_finite() {
            return Err(RadialIntegralError::NonFiniteProduct { index });
        }
        integrand.push(value);
    }
    mesh.integrate(&integrand)
        .map_err(|error| RadialIntegralError::Quadrature(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RadialEquation, RadialSolver};
    use mt_core::{Bohr, Hartree};

    fn problem() -> (ExponentialMesh, Vec<f64>) {
        let first: f64 = 1.0e-6;
        let last: f64 = 4.0;
        let increment: f64 = 0.002;
        let number = ((last / first).ln() / increment).ceil() as usize + 1;
        let mesh = ExponentialMesh::new(Bohr(first), increment, number).unwrap();
        let potential: Vec<f64> = mesh.radii().iter().map(|r| -0.7 / r.get()).collect();
        (mesh, potential)
    }

    #[test]
    fn integral_is_symmetric_for_all_kernel_kinds() {
        let (mesh, potential) = problem();
        let solver =
            RadialSolver::new(&mesh, &potential, RadialEquation::ScalarKoellingHarmon).unwrap();
        let a = solver.solve(0, Hartree(-0.2)).unwrap();
        let b = solver.solve(1, Hartree(0.3)).unwrap();
        let weights: Vec<f64> = mesh.radii().iter().map(|r| (-r.get()).exp()).collect();
        let kernels = [
            RadialIntegralKernel::Overlap,
            RadialIntegralKernel::Power(2),
            RadialIntegralKernel::Samples(&weights),
            RadialIntegralKernel::PotentialMultipole {
                angular_l: 2,
                angular_m: -1,
                values: &weights,
            },
        ];
        for kernel in kernels {
            let ab = radial_integral(&mesh, &a, &b, kernel).unwrap();
            let ba = radial_integral(&mesh, &b, &a, kernel).unwrap();
            assert_eq!(ab, ba);
        }
        let aa = radial_integral(&mesh, &a, &a, RadialIntegralKernel::Overlap).unwrap();
        assert!((aa - 1.0).abs() < 2.0e-12);
    }
}
