//! Overlap-filtered generalized Hermitian eigensolver.

use crate::OperatorError;
use libmuffintin_core::Hartree;
use libmuffintin_tensor::{
    Axis, ComplexTensor, DenseEigenvectors, DenseHermitianMatrix, TensorError, einsum,
};
use num_complex::Complex64;

/// Residual diagnostic for one generalized eigenpair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EigenpairResidual {
    pub band_index: usize,
    /// Euclidean norm of `H c - S c epsilon`.
    pub absolute: f64,
    /// Absolute residual divided by `max(||Hc||, |epsilon| ||Sc||)`.
    pub relative: f64,
}

/// Result of the filtered dense Hermitian generalized eigensolve.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralizedEigensolution {
    /// Eigenvalues in nondecreasing Hartree order.
    pub eigenvalues: Vec<Hartree>,
    /// Column-major eigenvector columns on axes `[GlobalBasis, Band]`.
    pub eigenvectors: DenseEigenvectors,
    pub retained_dimension: usize,
    pub filtered_dimension: usize,
    pub residuals: Vec<EigenpairResidual>,
}

/// Independent spin-up and spin-down values for a collinear, no-SOC problem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Collinear<T> {
    pub up: T,
    pub down: T,
}

impl<T> Collinear<T> {
    pub const fn new(up: T, down: T) -> Self {
        Self { up, down }
    }
}

/// Solve `H C = S C epsilon` after removing near-linearly-dependent overlap
/// directions.  An overlap eigenvalue is retained when it is positive and
/// greater than `relative_overlap_threshold * max(eigenvalue(S))`.
pub fn solve_generalized_hermitian(
    hamiltonian: &DenseHermitianMatrix,
    overlap: &DenseHermitianMatrix,
    relative_overlap_threshold: f64,
) -> Result<GeneralizedEigensolution, OperatorError> {
    use faer::{Mat, Side};

    if hamiltonian.axis() != Axis::GlobalBasis {
        return Err(OperatorError::Tensor(TensorError::Axis {
            index: 0,
            expected: Axis::GlobalBasis,
            actual: hamiltonian.axis(),
        }));
    }
    if overlap.axis() != Axis::GlobalBasis {
        return Err(OperatorError::Tensor(TensorError::Axis {
            index: 0,
            expected: Axis::GlobalBasis,
            actual: overlap.axis(),
        }));
    }
    if hamiltonian.dimension() != overlap.dimension() {
        return Err(OperatorError::MatrixDimensionMismatch {
            hamiltonian: hamiltonian.dimension(),
            overlap: overlap.dimension(),
        });
    }
    if !relative_overlap_threshold.is_finite() || relative_overlap_threshold < 0.0 {
        return Err(OperatorError::InvalidOverlapThreshold(
            relative_overlap_threshold,
        ));
    }
    let n = overlap.dimension();
    if n == 0 {
        return Err(OperatorError::EmptyOverlapSubspace);
    }
    let s_matrix = Mat::from_fn(n, n, |row, column| overlap.at(row, column));
    let s_eigen = s_matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| OperatorError::Eigensolver)?;
    let spectral_scale = (0..n)
        .map(|index| s_eigen.S()[index].re)
        .map(f64::abs)
        .fold(0.0, f64::max);
    let cutoff = relative_overlap_threshold * spectral_scale;
    let negative_noise_tolerance = 1024.0 * f64::EPSILON * spectral_scale;
    if let Some(eigenvalue) = (0..n)
        .map(|index| s_eigen.S()[index].re)
        .find(|&eigenvalue| eigenvalue < -negative_noise_tolerance)
    {
        return Err(OperatorError::IndefiniteOverlap { eigenvalue });
    }
    let retained = (0..n)
        .filter(|&index| {
            let eigenvalue = s_eigen.S()[index].re;
            eigenvalue > 0.0 && eigenvalue > cutoff
        })
        .collect::<Vec<_>>();
    if retained.is_empty() {
        return Err(OperatorError::EmptyOverlapSubspace);
    }
    let r = retained.len();

    // X = U_keep diag(s_keep^{-1/2}), so X^H S X = I. Filtering stays here;
    // the products are einsum.
    let mut u_keep = vec![Complex64::default(); n * r];
    let mut scales = vec![Complex64::default(); r];
    for (column, &source_column) in retained.iter().enumerate() {
        scales[column] = Complex64::new(1.0 / s_eigen.S()[source_column].re.sqrt(), 0.0);
        for row in 0..n {
            u_keep[row * r + column] = s_eigen.U()[(row, source_column)];
        }
    }
    let u_keep =
        ComplexTensor::from_host_row_major(&[n, r], &[Axis::GlobalBasis, Axis::Reduced], u_keep)?;
    let scales = ComplexTensor::from_host_row_major(&[r], &[Axis::Reduced], scales)?;
    let x = einsum("ik,k->ik", &[&u_keep, &scales])?;

    let x_conj = x.conjugate();
    let reduced = DenseHermitianMatrix::from_tensor(einsum(
        "ir,ij,js->rs",
        &[&x_conj, hamiltonian.as_tensor(), &x],
    )?)?;
    let reduced_matrix = Mat::from_fn(r, r, |row, column| {
        reduced
            .get(row, column)
            .expect("reduced Hermitian block is square")
    });
    let reduced_eigen = reduced_matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| OperatorError::Eigensolver)?;

    let eigenvalues = (0..r)
        .map(|band| Hartree(reduced_eigen.S()[band].re))
        .collect::<Vec<_>>();
    let mut z = vec![Complex64::default(); r * r];
    for row in 0..r {
        for band in 0..r {
            z[row * r + band] = reduced_eigen.U()[(row, band)];
        }
    }
    let z = ComplexTensor::from_host_row_major(&[r, r], &[Axis::Reduced, Axis::Band], z)?;
    let vectors = einsum("ir,rb->ib", &[&x, &z])?;

    let hc = einsum("ij,jb->ib", &[hamiltonian.as_tensor(), &vectors])?;
    let sc = einsum("ij,jb->ib", &[overlap.as_tensor(), &vectors])?;
    let epsilon = ComplexTensor::from_host_row_major(
        &[r],
        &[Axis::Band],
        eigenvalues
            .iter()
            .map(|value| Complex64::new(value.get(), 0.0))
            .collect(),
    )?;
    let sc_eps = einsum("ib,b->ib", &[&sc, &epsilon])?;
    let residual = hc.sub(&sc_eps)?;
    let residual_conj = residual.conjugate();
    let residual_sq = einsum("ib,ib->b", &[&residual_conj, &residual])?;
    let hc_conj = hc.conjugate();
    let hc_sq = einsum("ib,ib->b", &[&hc_conj, &hc])?;
    let sc_conj = sc.conjugate();
    let sc_sq = einsum("ib,ib->b", &[&sc_conj, &sc])?;
    let residuals = (0..r)
        .map(|band| {
            let absolute = residual_sq
                .get(&[band])
                .expect("band residual")
                .re
                .max(0.0)
                .sqrt();
            let hc_norm = hc_sq.get(&[band]).expect("Hc norm").re.max(0.0).sqrt();
            let sc_norm = sc_sq.get(&[band]).expect("Sc norm").re.max(0.0).sqrt();
            let denominator = hc_norm.max(eigenvalues[band].get().abs() * sc_norm);
            EigenpairResidual {
                band_index: band,
                absolute,
                relative: if denominator == 0.0 {
                    absolute
                } else {
                    absolute / denominator
                },
            }
        })
        .collect();

    Ok(GeneralizedEigensolution {
        eigenvalues,
        eigenvectors: DenseEigenvectors::from_tensor(vectors)?,
        retained_dimension: r,
        filtered_dimension: n - r,
        residuals,
    })
}
