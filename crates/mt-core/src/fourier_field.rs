//! Ordered reciprocal layouts and real-valued periodic scalar fields.

use crate::{GVector, ReciprocalLattice};
use num_complex::Complex64;
use std::collections::BTreeMap;
use thiserror::Error;

/// An ordered, uniquely indexed set of reciprocal vectors on one exact lattice.
///
/// The input order is preserved.  Equality is deliberately exact and includes
/// both the reciprocal lattice and the complete ordered vector list; fields on
/// two independently rounded lattices therefore cannot be combined silently.
#[derive(Clone, Debug, PartialEq)]
pub struct FourierLayout {
    reciprocal: ReciprocalLattice,
    vectors: Vec<GVector>,
    by_index: BTreeMap<[i32; 3], usize>,
}

impl FourierLayout {
    /// Validate and store an ordered reciprocal-vector list.
    pub fn new(
        reciprocal: ReciprocalLattice,
        vectors: Vec<GVector>,
    ) -> Result<Self, FourierFieldError> {
        let mut by_index = BTreeMap::new();
        for (position, vector) in vectors.iter().enumerate() {
            if by_index.insert(vector.index, position).is_some() {
                return Err(FourierFieldError::DuplicateVector {
                    index: vector.index,
                });
            }
            let expected_cartesian = reciprocal.cartesian(vector.index);
            let expected_norm = expected_cartesian
                .iter()
                .map(|component| component.get() * component.get())
                .sum::<f64>()
                .sqrt();
            if vector.cartesian != expected_cartesian || vector.norm.get() != expected_norm {
                return Err(FourierFieldError::VectorLatticeMismatch {
                    index: vector.index,
                });
            }
        }
        Ok(Self {
            reciprocal,
            vectors,
            by_index,
        })
    }

    /// Exact reciprocal-lattice identity carried by the layout.
    pub const fn reciprocal(&self) -> &ReciprocalLattice {
        &self.reciprocal
    }

    /// Reciprocal vectors in caller-specified order.
    pub fn vectors(&self) -> &[GVector] {
        &self.vectors
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Position of an integer reciprocal coordinate in the ordered list.
    pub fn index(&self, reciprocal_index: [i32; 3]) -> Option<usize> {
        self.by_index.get(&reciprocal_index).copied()
    }
}

/// Fourier coefficients of a physically real periodic scalar field.
///
/// Coefficients obey `c(-G) = conj(c(G))` exactly, including a real `G=0`
/// coefficient.  A layout used here must consequently contain the inversion
/// partner of every stored vector.
#[derive(Clone, Debug, PartialEq)]
pub struct HermitianFourierField {
    layout: FourierLayout,
    coefficients: Vec<Complex64>,
}

impl HermitianFourierField {
    /// Construct a field and check finite coefficients and Fourier reality.
    pub fn new(
        layout: FourierLayout,
        coefficients: Vec<Complex64>,
    ) -> Result<Self, FourierFieldError> {
        if coefficients.len() != layout.len() {
            return Err(FourierFieldError::CoefficientCount {
                expected: layout.len(),
                actual: coefficients.len(),
            });
        }
        for (position, (&coefficient, vector)) in
            coefficients.iter().zip(layout.vectors()).enumerate()
        {
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(FourierFieldError::NonFiniteCoefficient {
                    position,
                    coefficient,
                });
            }
            let opposite = negate_index(vector.index)
                .and_then(|index| layout.index(index))
                .ok_or(FourierFieldError::MissingConjugate {
                    index: vector.index,
                })?;
            if coefficients[opposite] != coefficient.conj() {
                return Err(FourierFieldError::NonHermitianPair {
                    index: vector.index,
                    coefficient,
                    conjugate: coefficients[opposite],
                });
            }
        }
        Ok(Self {
            layout,
            coefficients,
        })
    }

    pub const fn layout(&self) -> &FourierLayout {
        &self.layout
    }

    pub fn coefficients(&self) -> &[Complex64] {
        &self.coefficients
    }

    pub fn coefficient(&self, reciprocal_index: [i32; 3]) -> Option<Complex64> {
        self.layout
            .index(reciprocal_index)
            .map(|position| self.coefficients[position])
    }

    pub fn iter(&self) -> impl Iterator<Item = (&GVector, &Complex64)> {
        self.layout.vectors().iter().zip(&self.coefficients)
    }

    /// A zero field with the identical lattice and ordered reciprocal layout.
    pub fn zero_like(&self) -> Self {
        Self {
            layout: self.layout.clone(),
            coefficients: vec![Complex64::new(0.0, 0.0); self.coefficients.len()],
        }
    }

    /// Accumulate `self += scale * other` without changing either layout.
    pub fn add_scaled(&mut self, scale: f64, other: &Self) -> Result<(), FourierFieldError> {
        if !scale.is_finite() {
            return Err(FourierFieldError::InvalidScale(scale));
        }
        self.require_same_layout(other)?;
        let coefficients = self
            .coefficients
            .iter()
            .zip(&other.coefficients)
            .map(|(&left, &right)| left + scale * right)
            .collect();
        let updated = Self::new(self.layout.clone(), coefficients)?;
        self.coefficients = updated.coefficients;
        Ok(())
    }

    /// Form `self - other` on the same exact reciprocal layout.
    pub fn difference(&self, other: &Self) -> Result<Self, FourierFieldError> {
        self.require_same_layout(other)?;
        Self::new(
            self.layout.clone(),
            self.coefficients
                .iter()
                .zip(&other.coefficients)
                .map(|(&left, &right)| left - right)
                .collect(),
        )
    }

    fn require_same_layout(&self, other: &Self) -> Result<(), FourierFieldError> {
        if self.layout != other.layout {
            Err(FourierFieldError::LayoutMismatch)
        } else {
            Ok(())
        }
    }
}

fn negate_index(index: [i32; 3]) -> Option<[i32; 3]> {
    Some([
        index[0].checked_neg()?,
        index[1].checked_neg()?,
        index[2].checked_neg()?,
    ])
}

/// Invalid reciprocal layout, coefficient set, or field operation.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum FourierFieldError {
    #[error("duplicate reciprocal vector {index:?}")]
    DuplicateVector { index: [i32; 3] },
    #[error("reciprocal vector {index:?} does not exactly match its lattice")]
    VectorLatticeMismatch { index: [i32; 3] },
    #[error("Fourier field has {actual} coefficients, expected {expected}")]
    CoefficientCount { expected: usize, actual: usize },
    #[error("Fourier coefficient {position} is non-finite: {coefficient}")]
    NonFiniteCoefficient {
        position: usize,
        coefficient: Complex64,
    },
    #[error("reciprocal vector {index:?} has no -G partner")]
    MissingConjugate { index: [i32; 3] },
    #[error(
        "Fourier coefficients at G={index:?} are not conjugates: c(G)={coefficient}, c(-G)={conjugate}"
    )]
    NonHermitianPair {
        index: [i32; 3],
        coefficient: Complex64,
        conjugate: Complex64,
    },
    #[error("Fourier layouts differ")]
    LayoutMismatch,
    #[error("field accumulation scale must be finite, got {0}")]
    InvalidScale(f64),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InverseBohr;

    fn lattice() -> ReciprocalLattice {
        ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap()
    }

    #[test]
    fn ordering_is_preserved_and_indices_are_unique() {
        let reciprocal = lattice();
        let mut vectors = reciprocal.enumerate(InverseBohr(1.0)).unwrap();
        vectors.swap(1, 6);
        let expected = vectors
            .iter()
            .map(|vector| vector.index)
            .collect::<Vec<_>>();
        let layout = FourierLayout::new(reciprocal, vectors).unwrap();
        assert_eq!(
            layout
                .vectors()
                .iter()
                .map(|vector| vector.index)
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(layout.index(expected[1]), Some(1));

        let duplicate = vec![layout.vectors()[0], layout.vectors()[0]];
        assert!(matches!(
            FourierLayout::new(reciprocal, duplicate),
            Err(FourierFieldError::DuplicateVector { .. })
        ));
    }

    #[test]
    fn hermitian_coefficients_and_exact_layout_identity_are_enforced() {
        let reciprocal = lattice();
        let vectors = reciprocal.enumerate(InverseBohr(1.0)).unwrap();
        let layout = FourierLayout::new(reciprocal, vectors).unwrap();
        let mut coefficients = vec![Complex64::new(0.0, 0.0); layout.len()];
        coefficients[layout.index([0, 0, 0]).unwrap()] = Complex64::new(2.0, 0.0);
        coefficients[layout.index([1, 0, 0]).unwrap()] = Complex64::new(0.4, -0.7);
        coefficients[layout.index([-1, 0, 0]).unwrap()] = Complex64::new(0.4, 0.7);
        let field = HermitianFourierField::new(layout.clone(), coefficients.clone()).unwrap();
        assert_eq!(
            field.coefficient([1, 0, 0]),
            Some(Complex64::new(0.4, -0.7))
        );

        coefficients[layout.index([0, 0, 0]).unwrap()].im = 1.0e-15;
        assert!(matches!(
            HermitianFourierField::new(layout.clone(), coefficients),
            Err(FourierFieldError::NonHermitianPair {
                index: [0, 0, 0],
                ..
            })
        ));

        let different_reciprocal = ReciprocalLattice::new([
            [
                InverseBohr(1.0 + f64::EPSILON),
                InverseBohr(0.0),
                InverseBohr(0.0),
            ],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        let different = FourierLayout::new(
            different_reciprocal,
            different_reciprocal.enumerate(InverseBohr(1.0)).unwrap(),
        )
        .unwrap();
        let other = HermitianFourierField::new(
            different.clone(),
            vec![Complex64::new(0.0, 0.0); different.len()],
        )
        .unwrap();
        assert_eq!(
            field.difference(&other),
            Err(FourierFieldError::LayoutMismatch)
        );
    }

    #[test]
    fn semantic_field_algebra_preserves_reality() {
        let reciprocal = lattice();
        let vectors = reciprocal.enumerate(InverseBohr(1.0)).unwrap();
        let layout = FourierLayout::new(reciprocal, vectors).unwrap();
        let mut coefficients = vec![Complex64::new(0.0, 0.0); layout.len()];
        coefficients[layout.index([1, 0, 0]).unwrap()] = Complex64::new(1.0, 2.0);
        coefficients[layout.index([-1, 0, 0]).unwrap()] = Complex64::new(1.0, -2.0);
        let field = HermitianFourierField::new(layout, coefficients).unwrap();
        let mut accumulated = field.zero_like();
        accumulated.add_scaled(0.25, &field).unwrap();
        assert_eq!(
            accumulated.difference(&accumulated).unwrap(),
            field.zero_like()
        );
    }
}
