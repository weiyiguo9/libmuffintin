//! Generic Hermitian operator containers, site projection, eigensolution,
//! and method recipes.
//!
//! [`recipes`] holds basis-construction recipes; LAPW is one such recipe,
//! and its facade and eigenproblem assembly live in [`lapw`].

#![forbid(unsafe_code)]

mod assemble;
mod eigensolve;
pub mod lapw;
mod projection;
pub mod recipes;
mod soc;
mod spinor;

pub use assemble::{
    OperatorSet, SiteOperatorBlocks, add_site_contributions, assemble_scalar_site_operator,
};
pub use eigensolve::{
    Collinear, EigenpairResidual, GeneralizedEigensolution, RealSymmetricEigensolution,
    lift_band_hermitian_feedback, solve_generalized_hermitian, solve_real_symmetric,
};
pub use projection::{
    CompiledSiteProjection, SiteOrbitalCoefficients, project_eigenvectors_to_site,
    project_spinor_eigenvectors_to_site,
};
pub use soc::{
    SecondVariationMixing, SecondVariationSubspaceSolution, SiteSpinOrbitBlock,
    SocEigenpairResidual, SocOperatorError, project_site_soc_to_subspace,
    project_site_spinor_operator_to_subspace, solve_second_variation_subspace,
};
pub use spinor::{SpinorSiteOperatorBlocks, add_spinor_site_contributions};

use muffintin_tensor::TensorError;
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
    #[error("spinor basis layout has {actual} spatial plane waves, expected {expected}")]
    SpinorBasisPlaneWaveCount { expected: usize, actual: usize },
    #[error("site plane wave {plane_wave} has {actual} lm channels, expected {expected}")]
    ChannelCount {
        plane_wave: usize,
        expected: usize,
        actual: usize,
    },
    #[error("site {site} plane wave {plane_wave} has a different spinor channel layout")]
    SpinorChannelLayout { site: usize, plane_wave: usize },
    #[error("site index {site} is outside a basis with {site_count} sites")]
    SiteIndex { site: usize, site_count: usize },
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
    #[error("eigenvectors have {actual} global-basis rows, expected {expected}")]
    EigenvectorBasisCount { expected: usize, actual: usize },
    #[error("band feedback dimension {feedback} differs from eigenvector band count {bands}")]
    BandFeedbackDimensionMismatch { feedback: usize, bands: usize },
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
    use muffintin_tensor::{Axis, DenseHermitianMatrix, MemoryLayout, TensorError};
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

    #[test]
    fn real_symmetric_eigensolver_returns_real_columns() {
        let solution = solve_real_symmetric(2, |row, column| match (row, column) {
            (0, 0) => 2.0,
            (0, 1) => 0.5,
            (1, 1) => 1.0,
            _ => unreachable!(),
        })
        .unwrap();
        assert_eq!(solution.dimension, 2);
        assert!(solution.eigenvalues[0] <= solution.eigenvalues[1]);
        let left = solution.eigenvectors[0];
        let right = solution.eigenvectors[1];
        let residual0 = 2.0 * left + 0.5 * right - solution.eigenvalues[0] * left;
        let residual1 = 0.5 * left + 1.0 * right - solution.eigenvalues[0] * right;
        assert!(residual0.abs() + residual1.abs() < 1.0e-12);
    }
}
