//! Explicit strategy-level conventions that must never be hidden defaults.

use crate::{Hartree, InverseBohr};
use std::f64::consts::PI;

/// Interstitial kinetic-energy form for discontinuous augmented basis functions.
///
/// The two forms agree on all space for smooth periodic functions but differ by
/// a muffin-tin boundary term when multiplied by the interstitial indicator.
/// There is intentionally no [`Default`] implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KineticOperatorConvention {
    /// Weak/gradient form `1/2 K dot K' Theta(K-K')`, as written in the plan.
    Gradient,
    /// SPEX production form `1/4 (|K|^2+|K'|^2) Theta(K-K')`.
    SpexSymmetricLaplacian,
}

impl KineticOperatorConvention {
    /// Real prefactor multiplying the interstitial step coefficient, in Hartree.
    pub fn prefactor(self, left: [InverseBohr; 3], right: [InverseBohr; 3]) -> Hartree {
        let left_squared = squared_norm(left);
        let right_squared = squared_norm(right);
        let value = match self {
            Self::Gradient => 0.5 * dot(left, right),
            Self::SpexSymmetricLaplacian => 0.25 * (left_squared + right_squared),
        };
        Hartree(value)
    }

    /// SPEX-minus-gradient boundary-term prefactor
    /// `|K-K'|^2 / 4`, in Hartree.
    pub fn symmetric_minus_gradient(left: [InverseBohr; 3], right: [InverseBohr; 3]) -> Hartree {
        let difference = std::array::from_fn(|axis| InverseBohr(left[axis].0 - right[axis].0));
        Hartree(0.25 * squared_norm(difference))
    }
}

/// Convert an expansion coefficient multiplying `Y_00` to the actual constant
/// spherical potential `V(r)` used by radial equations.
///
/// `mt-core` and `mt-radial` use the actual value. This conversion is only for
/// input formats that store `V(r) = v_00(r) Y_00`.
pub fn spherical_value_from_y00_coefficient(coefficient: Hartree) -> Hartree {
    Hartree(coefficient.0 / (4.0 * PI).sqrt())
}

fn dot(left: [InverseBohr; 3], right: [InverseBohr; 3]) -> f64 {
    left.iter().zip(right).map(|(x, y)| x.0 * y.0).sum()
}

fn squared_norm(vector: [InverseBohr; 3]) -> f64 {
    dot(vector, vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_kinetic_forms_differ_by_documented_boundary_term() {
        let left = [InverseBohr(0.3), InverseBohr(-1.1), InverseBohr(2.0)];
        let right = [InverseBohr(-0.7), InverseBohr(0.4), InverseBohr(1.2)];
        let actual = KineticOperatorConvention::SpexSymmetricLaplacian.prefactor(left, right)
            - KineticOperatorConvention::Gradient.prefactor(left, right);
        let expected = KineticOperatorConvention::symmetric_minus_gradient(left, right);
        assert!((actual.0 - expected.0).abs() < 2e-15);
    }
}
