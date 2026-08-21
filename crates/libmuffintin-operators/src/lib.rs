//! Generic Hermitian operator containers, site projection, and eigensolution.

#![forbid(unsafe_code)]

mod assemble;
mod eigensolve;

pub use assemble::{OperatorSet, SiteOperatorBlocks, add_site_contributions};
pub use eigensolve::{
    Collinear, EigenpairResidual, GeneralizedEigensolution, solve_generalized_hermitian,
};

use libmuffintin_tensor::TensorError;
use thiserror::Error;

/// Operator assembly or generalized-eigensolver error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum OperatorError {
    #[error("site has {actual} plane-wave coefficient sets, expected {expected}")]
    PlaneWaveCount { expected: usize, actual: usize },
    #[error("expected {expected} site blocks, got {actual}")]
    SiteCount { expected: usize, actual: usize },
    #[error("basis layout has {actual} plane waves, expected {expected}")]
    BasisPlaneWaveCount { expected: usize, actual: usize },
    #[error("basis layout has {actual} sites, expected {expected}")]
    BasisSiteCount { expected: usize, actual: usize },
    #[error("site plane wave {plane_wave} has {actual} lm channels, expected {expected}")]
    ChannelCount {
        plane_wave: usize,
        expected: usize,
        actual: usize,
    },
    #[error("site {site} {matrix} block has dimension {actual}, expected {expected}")]
    SiteBlockDimension {
        site: usize,
        matrix: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("matrix data has length {actual}, expected {expected}")]
    MatrixDataLength { expected: usize, actual: usize },
    #[error("matrix dimensions differ: H is {hamiltonian}, S is {overlap}")]
    MatrixDimensionMismatch { hamiltonian: usize, overlap: usize },
    #[error("overlap eigenvalue threshold must be finite and nonnegative, got {0}")]
    InvalidOverlapThreshold(f64),
    #[error("overlap eigensystem retained no positive directions")]
    EmptyOverlapSubspace,
    #[error("overlap matrix is significantly indefinite (eigenvalue {eigenvalue})")]
    IndefiniteOverlap { eigenvalue: f64 },
    #[error("dense self-adjoint eigendecomposition failed")]
    Eigensolver,
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use libmuffintin_tensor::{Axis, DenseHermitianMatrix, MemoryLayout, TensorError};
    use num_complex::Complex64;

    fn site_h(
        dimension: usize,
        element: impl FnMut(usize, usize) -> Complex64,
    ) -> DenseHermitianMatrix {
        DenseHermitianMatrix::from_upper_triangle(dimension, Axis::SiteCoordinate, element).unwrap()
    }

    fn global_h(
        dimension: usize,
        element: impl FnMut(usize, usize) -> Complex64,
    ) -> DenseHermitianMatrix {
        DenseHermitianMatrix::from_upper_triangle(dimension, Axis::GlobalBasis, element).unwrap()
    }

    #[test]
    fn generalized_solver_filters_near_null_overlap_and_rejects_indefinite_overlap() {
        let h = global_h(2, |row, column| {
            if row == column {
                Complex64::new((row + 1) as f64, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        let nearly_singular = global_h(2, |row, column| {
            if row == column {
                Complex64::new(if row == 0 { 1.0 } else { -1.0e-14 }, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        let solution = solve_generalized_hermitian(&h, &nearly_singular, 1.0e-10).unwrap();
        assert_eq!(solution.retained_dimension, 1);
        assert_eq!(solution.filtered_dimension, 1);
        assert!((solution.eigenvalues[0].get() - 1.0).abs() < 1.0e-14);

        let indefinite = global_h(2, |row, column| {
            if row == column {
                Complex64::new(if row == 0 { 1.0 } else { -0.1 }, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        assert!(matches!(
            solve_generalized_hermitian(&h, &indefinite, 1.0e-10),
            Err(OperatorError::IndefiniteOverlap { .. })
        ));
    }

    #[test]
    fn generalized_solver_reports_the_overlap_axis() {
        let h = global_h(1, |_, _| Complex64::new(1.0, 0.0));
        let overlap = site_h(1, |_, _| Complex64::new(1.0, 0.0));
        let error = solve_generalized_hermitian(&h, &overlap, 0.0).unwrap_err();
        assert_eq!(
            error,
            OperatorError::Tensor(TensorError::Axis {
                index: 0,
                expected: Axis::GlobalBasis,
                actual: Axis::SiteCoordinate,
            })
        );
    }

    #[test]
    fn generalized_eigenvectors_are_s_orthonormal_with_small_residuals() {
        let h = global_h(2, |row, column| match (row, column) {
            (0, 0) => Complex64::new(1.0, 0.0),
            (0, 1) => Complex64::new(0.2, 0.1),
            (1, 1) => Complex64::new(2.0, 0.0),
            _ => unreachable!(),
        });
        let s = global_h(2, |row, column| match (row, column) {
            (0, 0) => Complex64::new(1.3, 0.0),
            (0, 1) => Complex64::new(0.1, -0.05),
            (1, 1) => Complex64::new(0.9, 0.0),
            _ => unreachable!(),
        });
        let solution = solve_generalized_hermitian(&h, &s, 1.0e-12).unwrap();
        assert_eq!(solution.eigenvectors.layout(), MemoryLayout::ColumnMajor);
        assert_eq!(
            solution.eigenvectors.as_tensor().layout(),
            MemoryLayout::ColumnMajor
        );
        for left in 0..2 {
            for right in 0..2 {
                let mut value = Complex64::new(0.0, 0.0);
                for i in 0..2 {
                    for j in 0..2 {
                        value += solution.eigenvectors.at(i, left).conj()
                            * s.at(i, j)
                            * solution.eigenvectors.at(j, right);
                    }
                }
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((value - expected).norm() < 1.0e-12);
            }
        }
        assert!(
            solution
                .residuals
                .iter()
                .all(|residual| residual.absolute < 1.0e-12)
        );
    }
}
