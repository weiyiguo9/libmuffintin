//! Reciprocal lattices and complete Cartesian-norm G-vector enumeration.

use crate::{Bohr, InverseBohr};
use std::f64::consts::TAU;
use thiserror::Error;

/// A reciprocal lattice vector and its integer coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GVector {
    /// Integer coefficients of the reciprocal primitive vectors.
    pub index: [i32; 3],
    /// Cartesian vector in `bohr^-1`.
    pub cartesian: [InverseBohr; 3],
    /// Cartesian norm in `bohr^-1`.
    pub norm: InverseBohr,
}

/// Invalid direct/reciprocal lattice or cutoff.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum LatticeError {
    /// A lattice component is not finite.
    #[error("lattice component must be finite")]
    NonFiniteComponent,
    /// Primitive vectors are linearly dependent or numerically singular.
    #[error("lattice basis is singular")]
    SingularBasis,
    /// The Cartesian cutoff must be finite and nonnegative.
    #[error("G-vector cutoff must be finite and nonnegative, got {0}")]
    InvalidCutoff(f64),
    /// Integer enumeration bounds cannot be represented safely.
    #[error("G-vector cutoff produces integer bounds too large to enumerate")]
    CutoffTooLarge,
}

/// Reciprocal primitive vectors, including the crystallographic `2*pi`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReciprocalLattice {
    basis: [[InverseBohr; 3]; 3],
}

impl ReciprocalLattice {
    /// Construct from reciprocal primitive vectors `b_i` in Cartesian units.
    pub fn new(basis: [[InverseBohr; 3]; 3]) -> Result<Self, LatticeError> {
        if basis.iter().flatten().any(|x| !x.0.is_finite()) {
            return Err(LatticeError::NonFiniteComponent);
        }
        let raw = basis.map(|v| v.map(InverseBohr::get));
        validate_basis(&raw)?;
        Ok(Self { basis })
    }

    /// Construct `b_i` from direct vectors `a_i`, enforcing
    /// `a_i dot b_j = 2*pi delta_ij`.
    pub fn from_direct(direct: [[Bohr; 3]; 3]) -> Result<Self, LatticeError> {
        if direct.iter().flatten().any(|x| !x.0.is_finite()) {
            return Err(LatticeError::NonFiniteComponent);
        }
        let a = direct.map(|v| v.map(Bohr::get));
        validate_basis(&a)?;
        let volume = dot(a[0], cross(a[1], a[2]));
        let b0 = scale(cross(a[1], a[2]), TAU / volume);
        let b1 = scale(cross(a[2], a[0]), TAU / volume);
        let b2 = scale(cross(a[0], a[1]), TAU / volume);
        Self::new([b0, b1, b2].map(|v| v.map(InverseBohr)))
    }

    /// Reciprocal primitive vectors in Cartesian coordinates.
    pub const fn basis(&self) -> &[[InverseBohr; 3]; 3] {
        &self.basis
    }

    /// Convert integer reciprocal coordinates to a Cartesian vector.
    pub fn cartesian(&self, index: [i32; 3]) -> [InverseBohr; 3] {
        let mut result = [InverseBohr(0.0); 3];
        for (coefficient, basis) in index.into_iter().zip(self.basis) {
            for axis in 0..3 {
                result[axis].0 += f64::from(coefficient) * basis[axis].0;
            }
        }
        result
    }

    /// Enumerate every `G` satisfying the Cartesian norm cutoff.
    ///
    /// Bounds come from the dual basis, so the enumeration remains complete
    /// for skewed cells where large integer coefficients can cancel. Output is
    /// deterministic: norm first, then lexicographic integer coordinates.
    pub fn enumerate(&self, cutoff: InverseBohr) -> Result<Vec<GVector>, LatticeError> {
        if !cutoff.0.is_finite() || cutoff.0 < 0.0 {
            return Err(LatticeError::InvalidCutoff(cutoff.0));
        }
        let b = self.basis.map(|v| v.map(InverseBohr::get));
        let determinant = dot(b[0], cross(b[1], b[2]));
        let dual = [
            scale(cross(b[1], b[2]), 1.0 / determinant),
            scale(cross(b[2], b[0]), 1.0 / determinant),
            scale(cross(b[0], b[1]), 1.0 / determinant),
        ];
        let mut bounds = [0_i32; 3];
        for axis in 0..3 {
            let bound = (cutoff.0 * norm(dual[axis])).ceil();
            if bound > f64::from(i32::MAX) {
                return Err(LatticeError::CutoffTooLarge);
            }
            bounds[axis] = bound as i32;
        }

        let cutoff_squared = cutoff.0 * cutoff.0;
        let tolerance = 64.0 * f64::EPSILON * cutoff_squared.max(1.0);
        let mut vectors = Vec::new();
        for n0 in -bounds[0]..=bounds[0] {
            for n1 in -bounds[1]..=bounds[1] {
                for n2 in -bounds[2]..=bounds[2] {
                    let index = [n0, n1, n2];
                    let cartesian = self.cartesian(index);
                    let norm_squared = cartesian.iter().map(|x| x.0 * x.0).sum::<f64>();
                    if norm_squared <= cutoff_squared + tolerance {
                        vectors.push(GVector {
                            index,
                            cartesian,
                            norm: InverseBohr(norm_squared.sqrt()),
                        });
                    }
                }
            }
        }
        vectors.sort_by(|left, right| {
            left.norm
                .0
                .total_cmp(&right.norm.0)
                .then_with(|| left.index.cmp(&right.index))
        });
        Ok(vectors)
    }
}

fn validate_basis(basis: &[[f64; 3]; 3]) -> Result<(), LatticeError> {
    let determinant = dot(basis[0], cross(basis[1], basis[2]));
    let scale = norm(basis[0]) * norm(basis[1]) * norm(basis[2]);
    if scale == 0.0 || determinant.abs() <= 128.0 * f64::EPSILON * scale {
        Err(LatticeError::SingularBasis)
    } else {
        Ok(())
    }
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    vector.map(|x| x * factor)
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_and_reciprocal_are_dual() {
        let direct = [
            [Bohr(2.0), Bohr(0.0), Bohr(0.0)],
            [Bohr(0.2), Bohr(3.0), Bohr(0.0)],
            [Bohr(0.1), Bohr(-0.3), Bohr(4.0)],
        ];
        let reciprocal = ReciprocalLattice::from_direct(direct).unwrap();
        for (i, a) in direct.iter().enumerate() {
            for (j, b) in reciprocal.basis().iter().enumerate() {
                let product = a.iter().zip(b).map(|(x, y)| x.0 * y.0).sum::<f64>();
                let expected = if i == j { TAU } else { 0.0 };
                assert!((product - expected).abs() < 2e-15);
            }
        }
    }

    #[test]
    fn cubic_shell_is_complete_and_deterministic() {
        let lattice = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        let vectors = lattice.enumerate(InverseBohr(1.0)).unwrap();
        assert_eq!(vectors.len(), 7);
        assert_eq!(vectors[0].index, [0, 0, 0]);
        assert_eq!(vectors, lattice.enumerate(InverseBohr(1.0)).unwrap());
    }
}
