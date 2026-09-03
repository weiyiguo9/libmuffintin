//! Local tensor substrate for muffin-tin numerical contractions.
//!
//! The public contraction language is Einstein summation. Physics modules
//! write `einsum("ci,cd,dj->ij", ...)`; they do not own nested reduction
//! loops. The default backend is RSTSR 0.7.10 linked with TBLIS. tenferro-rs
//! is the second local backend behind the same subscripts, enabled later
//! when it satisfies the workspace MSRV and dependency gate.
//!
//! Backend tensor handles stay private. Host checkpoints remain ordinary
//! `Vec<Complex64>` buffers with an explicit row- or column-major contract.
//! There is no scalar fallback runtime.

#![forbid(unsafe_code)]

mod backend;
#[cfg(feature = "backend-rstsr")]
mod rstsr_tblis;
#[cfg(feature = "backend-tenferro")]
mod tenferro;

#[cfg(feature = "backend-tenferro")]
pub use backend::einsum_tenferro;
pub use backend::{EinsumBackend, active_backend_name, einsum};

use num_complex::Complex64;
use std::fmt;
use thiserror::Error;

/// One-process local execution world.
///
/// Distributed worlds such as CTF/MPI are deferred.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LocalWorld;

impl LocalWorld {
    pub const fn new() -> Self {
        Self
    }
}

/// Storage placement intent. [`Placement::Auto`] is the normative policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Placement {
    #[default]
    Auto,
    Host,
}

/// Named axis of a dense local tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// Site-local coordinates: every `lm` channel's `(u, udot)` pair, then
    /// that site's local orbitals in `(l, m, n)` order.
    SiteCoordinate,
    /// Columns participating in one site projection: all plane waves, then
    /// that site's local orbitals.
    SiteBasis,
    /// Global `[PW][all site LOs]` operator axes.
    GlobalBasis,
    /// Retained overlap subspace after spectral filtering.
    Reduced,
    /// Eigenvector / band columns.
    Band,
    /// Compiled mixed-product or interpolation-point auxiliary index of $V^q$.
    Auxiliary,
    /// Pair-vertex column index of one exchange block.
    PairColumn,
    /// Explicit core spin-orbital index within one site block.
    CoreOrbital,
}

/// Contiguous host layout of a dense tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    Strided,
}

/// Element type of a dense local tensor. Only complex `f64` is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    Complex64,
}

/// Tensor rank, shape, axis, layout, einsum, or Hermiticity error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum TensorError {
    #[error("tensor rank is {actual}, expected {expected}")]
    Rank { expected: usize, actual: usize },
    #[error("tensor shape {actual:?} does not match {expected:?}")]
    Shape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("declared {declared} axes for a rank-{rank} tensor")]
    AxisCount { declared: usize, rank: usize },
    #[error("axis {index} is {actual:?}, expected {expected:?}")]
    Axis {
        index: usize,
        expected: Axis,
        actual: Axis,
    },
    #[error(
        "cannot contract axis {left:?} of length {left_len} with axis {right:?} of length {right_len}"
    )]
    Contraction {
        left: Axis,
        left_len: usize,
        right: Axis,
        right_len: usize,
    },
    #[error(
        "host buffer has length {actual}, expected {expected} for shape {shape:?} in {layout:?} layout"
    )]
    HostLength {
        expected: usize,
        actual: usize,
        shape: Vec<usize>,
        layout: MemoryLayout,
    },
    #[error("index {indices:?} is out of bounds for shape {shape:?}")]
    Index {
        indices: Vec<usize>,
        shape: Vec<usize>,
    },
    #[error("matrix of shape {rows}x{cols} is not square")]
    NotSquare { rows: usize, cols: usize },
    #[error("Hermitian matrix axes must match; got {row:?} and {column:?}")]
    HermitianAxes { row: Axis, column: Axis },
    #[error("tensor is not Hermitian at ({row}, {column})")]
    NonHermitian { row: usize, column: usize },
    #[error("tensor has a non-finite value at {indices:?}")]
    NonFinite { indices: Vec<usize> },
    #[error("rank-{rank} tensors are not supported; local contractions use rank 1, 2, or 3")]
    UnsupportedRank { rank: usize },
    #[error("einsum subscripts `{subscripts}` are not in the form `ops->out`")]
    EinsumSyntax { subscripts: String },
    #[error("einsum `{subscripts}` expects {expected} operands, got {actual}")]
    EinsumArity {
        subscripts: String,
        expected: usize,
        actual: usize,
    },
    #[error("einsum `{subscripts}` uses unbound label {label}")]
    EinsumLabel { subscripts: String, label: char },
    #[error("local tensor backend failed: {0}")]
    Backend(String),
}

/// Owned dense complex tensor with declared axes.
#[derive(Clone, Debug)]
pub struct ComplexTensor {
    #[cfg(feature = "backend-rstsr")]
    pub(crate) data: rstsr::prelude::Tensor<Complex64, rstsr::prelude::DeviceFaer>,
    pub(crate) axes: Vec<Axis>,
    pub(crate) layout: MemoryLayout,
}

impl ComplexTensor {
    /// Copy a C-contiguous host buffer into a local tensor in [`LocalWorld`].
    pub fn from_host_row_major(
        shape: &[usize],
        axes: &[Axis],
        values: Vec<Complex64>,
    ) -> Result<Self, TensorError> {
        Self::from_host(shape, axes, values, MemoryLayout::RowMajor)
    }

    /// Copy a Fortran-contiguous host buffer into a local tensor in [`LocalWorld`].
    pub fn from_host_column_major(
        shape: &[usize],
        axes: &[Axis],
        values: Vec<Complex64>,
    ) -> Result<Self, TensorError> {
        Self::from_host(shape, axes, values, MemoryLayout::ColumnMajor)
    }

    fn from_host(
        shape: &[usize],
        axes: &[Axis],
        values: Vec<Complex64>,
        layout: MemoryLayout,
    ) -> Result<Self, TensorError> {
        if axes.len() != shape.len() {
            return Err(TensorError::AxisCount {
                declared: axes.len(),
                rank: shape.len(),
            });
        }
        if shape.is_empty() || shape.len() > 3 {
            return Err(TensorError::UnsupportedRank { rank: shape.len() });
        }
        let expected = shape.iter().copied().product::<usize>();
        if values.len() != expected {
            return Err(TensorError::HostLength {
                expected,
                actual: values.len(),
                shape: shape.to_vec(),
                layout,
            });
        }
        for (flat, value) in values.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(TensorError::NonFinite {
                    indices: unravel(flat, shape, layout),
                });
            }
        }
        #[cfg(feature = "backend-rstsr")]
        {
            let data = match layout {
                MemoryLayout::RowMajor => rstsr_tblis::asarray_row_major(shape, values)?,
                MemoryLayout::ColumnMajor => rstsr_tblis::asarray_column_major(shape, values)?,
                MemoryLayout::Strided => unreachable!("host constructors require contiguous data"),
            };
            Ok(Self {
                data,
                axes: axes.to_vec(),
                layout,
            })
        }
        #[cfg(not(feature = "backend-rstsr"))]
        {
            let _ = values;
            Err(TensorError::Backend("no tensor backend enabled".into()))
        }
    }

    pub const fn world(&self) -> LocalWorld {
        LocalWorld
    }

    pub const fn placement(&self) -> Placement {
        Placement::Auto
    }

    pub const fn dtype(&self) -> DType {
        DType::Complex64
    }

    pub fn rank(&self) -> usize {
        self.axes.len()
    }

    pub fn shape(&self) -> Vec<usize> {
        #[cfg(feature = "backend-rstsr")]
        {
            rstsr_tblis::shape_of(&self.data)
        }
        #[cfg(not(feature = "backend-rstsr"))]
        {
            Vec::new()
        }
    }

    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    pub const fn layout(&self) -> MemoryLayout {
        self.layout
    }

    /// In-bounds entry. Panics if the index is invalid.
    pub fn at(&self, indices: &[usize]) -> Complex64 {
        self.get(indices).unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn get(&self, indices: &[usize]) -> Result<Complex64, TensorError> {
        let shape = self.shape();
        if indices.len() != shape.len() {
            return Err(TensorError::Rank {
                expected: shape.len(),
                actual: indices.len(),
            });
        }
        if indices
            .iter()
            .zip(&shape)
            .any(|(&index, &extent)| index >= extent)
        {
            return Err(TensorError::Index {
                indices: indices.to_vec(),
                shape,
            });
        }
        #[cfg(feature = "backend-rstsr")]
        {
            rstsr_tblis::get_at(&self.data, indices)
        }
        #[cfg(not(feature = "backend-rstsr"))]
        {
            Err(TensorError::Backend("no tensor backend enabled".into()))
        }
    }

    /// Element-wise difference. Axes and shape must match.
    pub fn sub(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.axes != rhs.axes {
            return Err(TensorError::Axis {
                index: 0,
                expected: self.axes.first().copied().unwrap_or(Axis::GlobalBasis),
                actual: rhs.axes.first().copied().unwrap_or(Axis::GlobalBasis),
            });
        }
        if self.shape() != rhs.shape() {
            return Err(TensorError::Shape {
                expected: self.shape(),
                actual: rhs.shape(),
            });
        }
        #[cfg(feature = "backend-rstsr")]
        {
            let data = rstsr_tblis::subtract(&self.data, &rhs.data);
            let layout = rstsr_tblis::layout_of(&data);
            Ok(Self {
                data,
                axes: self.axes.clone(),
                layout,
            })
        }
        #[cfg(not(feature = "backend-rstsr"))]
        {
            Err(TensorError::Backend("no tensor backend enabled".into()))
        }
    }

    /// Element-wise conjugate, preserving axis order. Use this before einsum
    /// when a factor appears as $A^*$.
    pub fn conjugate(&self) -> Self {
        #[cfg(feature = "backend-rstsr")]
        {
            let data = rstsr_tblis::conjugate(&self.data);
            let layout = rstsr_tblis::layout_of(&data);
            Self {
                data,
                axes: self.axes.clone(),
                layout,
            }
        }
        #[cfg(not(feature = "backend-rstsr"))]
        {
            self.clone()
        }
    }

    /// Copy into a C-contiguous host buffer.
    pub fn to_host_row_major(&self) -> Vec<Complex64> {
        #[cfg(feature = "backend-rstsr")]
        {
            rstsr_tblis::to_host_row_major(&self.data)
        }
        #[cfg(not(feature = "backend-rstsr"))]
        {
            Vec::new()
        }
    }

    /// Copy into a Fortran-contiguous host buffer.
    pub fn to_host_column_major(&self) -> Vec<Complex64> {
        #[cfg(feature = "backend-rstsr")]
        {
            rstsr_tblis::to_host_column_major(&self.data)
        }
        #[cfg(not(feature = "backend-rstsr"))]
        {
            Vec::new()
        }
    }

    fn into_column_major(self) -> Self {
        #[cfg(feature = "backend-rstsr")]
        {
            Self {
                data: rstsr_tblis::into_column_major(self.data),
                axes: self.axes,
                layout: MemoryLayout::ColumnMajor,
            }
        }
        #[cfg(not(feature = "backend-rstsr"))]
        {
            self
        }
    }
}

impl PartialEq for ComplexTensor {
    fn eq(&self, other: &Self) -> bool {
        self.axes == other.axes
            && self.shape() == other.shape()
            && self.to_host_row_major() == other.to_host_row_major()
    }
}

impl fmt::Display for Axis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, formatter)
    }
}

/// Dense square Hermitian rank-2 tensor on a single repeated axis.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseHermitianMatrix {
    tensor: ComplexTensor,
}

impl DenseHermitianMatrix {
    /// Validate a square rank-2 tensor and store it as Hermitian.
    pub fn from_tensor(tensor: ComplexTensor) -> Result<Self, TensorError> {
        if tensor.rank() != 2 {
            return Err(TensorError::Rank {
                expected: 2,
                actual: tensor.rank(),
            });
        }
        let shape = tensor.shape();
        if shape[0] != shape[1] {
            return Err(TensorError::NotSquare {
                rows: shape[0],
                cols: shape[1],
            });
        }
        if tensor.axes[0] != tensor.axes[1] {
            return Err(TensorError::HermitianAxes {
                row: tensor.axes[0],
                column: tensor.axes[1],
            });
        }
        let mut values = tensor.to_host_row_major();
        validate_hermitian(shape[0], &values)?;
        hermitize(shape[0], &mut values);
        let tensor = ComplexTensor::from_host_row_major(&shape, tensor.axes(), values)?;
        Ok(Self { tensor })
    }

    pub fn from_host_row_major(
        dimension: usize,
        axis: Axis,
        values: Vec<Complex64>,
    ) -> Result<Self, TensorError> {
        Self::from_tensor(ComplexTensor::from_host_row_major(
            &[dimension, dimension],
            &[axis, axis],
            values,
        )?)
    }

    /// Build a Hermitian matrix from the upper triangle, including the diagonal.
    pub fn from_upper_triangle(
        dimension: usize,
        axis: Axis,
        mut element: impl FnMut(usize, usize) -> Complex64,
    ) -> Result<Self, TensorError> {
        let mut values = vec![Complex64::default(); dimension * dimension];
        for row in 0..dimension {
            for column in row..dimension {
                let value = element(row, column);
                values[row * dimension + column] = value;
                values[column * dimension + row] = value.conj();
            }
        }
        Self::from_host_row_major(dimension, axis, values)
    }

    pub fn as_tensor(&self) -> &ComplexTensor {
        &self.tensor
    }

    pub fn dimension(&self) -> usize {
        self.tensor.shape()[0]
    }

    pub fn axis(&self) -> Axis {
        self.tensor.axes[0]
    }

    pub fn get(&self, row: usize, column: usize) -> Result<Complex64, TensorError> {
        self.tensor.get(&[row, column])
    }

    /// In-bounds entry. Panics if the index is invalid.
    pub fn at(&self, row: usize, column: usize) -> Complex64 {
        self.get(row, column)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn to_host_row_major(&self) -> Vec<Complex64> {
        self.tensor.to_host_row_major()
    }
}

/// Dense eigenvector columns of a local generalized eigenproblem.
///
/// Axes are `[GlobalBasis, Band]` and storage is column-major, so every band's
/// complete basis expansion is contiguous. This matches the native
/// basis-by-band convention used by Fortran LAPW and plane-wave codes.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseEigenvectors {
    tensor: ComplexTensor,
}

impl DenseEigenvectors {
    pub fn from_tensor(tensor: ComplexTensor) -> Result<Self, TensorError> {
        if tensor.rank() != 2 {
            return Err(TensorError::Rank {
                expected: 2,
                actual: tensor.rank(),
            });
        }
        for (index, expected) in [Axis::GlobalBasis, Axis::Band].into_iter().enumerate() {
            if tensor.axes()[index] != expected {
                return Err(TensorError::Axis {
                    index,
                    expected,
                    actual: tensor.axes()[index],
                });
            }
        }
        Ok(Self {
            tensor: tensor.into_column_major(),
        })
    }

    pub fn from_host_column_major(
        basis_count: usize,
        band_count: usize,
        values: Vec<Complex64>,
    ) -> Result<Self, TensorError> {
        Self::from_tensor(ComplexTensor::from_host_column_major(
            &[basis_count, band_count],
            &[Axis::GlobalBasis, Axis::Band],
            values,
        )?)
    }

    pub fn as_tensor(&self) -> &ComplexTensor {
        &self.tensor
    }

    pub fn rows(&self) -> usize {
        self.tensor.shape()[0]
    }

    pub fn columns(&self) -> usize {
        self.tensor.shape()[1]
    }

    pub const fn layout(&self) -> MemoryLayout {
        MemoryLayout::ColumnMajor
    }

    pub fn get(&self, row: usize, column: usize) -> Result<Complex64, TensorError> {
        self.tensor.get(&[row, column])
    }

    /// In-bounds entry. Panics if the index is invalid.
    pub fn at(&self, row: usize, column: usize) -> Complex64 {
        self.get(row, column)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn to_host_column_major(&self) -> Vec<Complex64> {
        self.tensor.to_host_column_major()
    }

    /// Copy logical `[basis, band]` values into a C-contiguous host buffer.
    pub fn to_host_row_major(&self) -> Vec<Complex64> {
        self.tensor.to_host_row_major()
    }
}

/// Evaluate the site congruence $P^\dagger B P$ as
/// `einsum("ci,cd,dj->ij", [P^*, B, P])`.
///
/// This wrapper only restores the Hermitian invariant of the result. The
/// contraction itself is the einsum layer.
pub fn hermitian_congruence(
    projection: &ComplexTensor,
    block: &DenseHermitianMatrix,
) -> Result<DenseHermitianMatrix, TensorError> {
    if projection.rank() != 2 {
        return Err(TensorError::Rank {
            expected: 2,
            actual: projection.rank(),
        });
    }
    let coord = projection.axes[0];
    let basis = projection.axes[1];
    if coord != block.axis() {
        return Err(TensorError::Axis {
            index: 0,
            expected: block.axis(),
            actual: coord,
        });
    }
    if projection.shape()[0] != block.dimension() {
        return Err(TensorError::Contraction {
            left: block.axis(),
            left_len: block.dimension(),
            right: coord,
            right_len: projection.shape()[0],
        });
    }
    let conjugated = projection.conjugate();
    let projected = einsum(
        "ci,cd,dj->ij",
        &[&conjugated, block.as_tensor(), projection],
    )?;
    if projected.axes() != [basis, basis] {
        return Err(TensorError::Axis {
            index: 0,
            expected: basis,
            actual: projected.axes()[0],
        });
    }
    let dimension = projected.shape()[0];
    let values = projected.to_host_row_major();
    DenseHermitianMatrix::from_upper_triangle(dimension, basis, |row, column| {
        0.5 * values[row * dimension + column] + 0.5 * values[column * dimension + row].conj()
    })
}

fn unravel(mut flat: usize, shape: &[usize], layout: MemoryLayout) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    match layout {
        MemoryLayout::RowMajor => {
            for axis in (0..shape.len()).rev() {
                let extent = shape[axis].max(1);
                indices[axis] = flat % extent;
                flat /= extent;
            }
        }
        MemoryLayout::ColumnMajor => {
            for axis in 0..shape.len() {
                let extent = shape[axis].max(1);
                indices[axis] = flat % extent;
                flat /= extent;
            }
        }
        MemoryLayout::Strided => unreachable!("host constructors require contiguous data"),
    }
    indices
}

fn validate_hermitian(n: usize, values: &[Complex64]) -> Result<(), TensorError> {
    let mut scale = 1.0_f64;
    for row in 0..n {
        for column in 0..n {
            let value = values[row * n + column];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(TensorError::NonFinite {
                    indices: vec![row, column],
                });
            }
            scale = scale.max(value.norm());
        }
    }
    let tolerance = 128.0 * f64::EPSILON * scale;
    for row in 0..n {
        for column in 0..n {
            let value = values[row * n + column];
            let partner = values[column * n + row];
            if (value - partner.conj()).norm() > tolerance {
                return Err(TensorError::NonHermitian { row, column });
            }
        }
    }
    Ok(())
}

fn hermitize(dimension: usize, values: &mut [Complex64]) {
    for row in 0..dimension {
        for column in row..dimension {
            let average =
                0.5 * (values[row * dimension + column] + values[column * dimension + row].conj());
            values[row * dimension + column] = average;
            values[column * dimension + row] = average.conj();
        }
        values[row * dimension + row].im = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn projection_and_block() -> (ComplexTensor, DenseHermitianMatrix) {
        let p = ComplexTensor::from_host_row_major(
            &[3, 2],
            &[Axis::SiteCoordinate, Axis::SiteBasis],
            vec![
                c(0.3, -0.2),
                c(0.0, 0.0),
                c(-0.1, 0.4),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(1.0, 0.0),
            ],
        )
        .unwrap();
        let b = DenseHermitianMatrix::from_host_row_major(
            3,
            Axis::SiteCoordinate,
            vec![
                c(1.1, 0.0),
                c(0.2, 0.1),
                c(-0.3, 0.25),
                c(0.2, -0.1),
                c(0.9, 0.0),
                c(0.15, -0.35),
                c(-0.3, -0.25),
                c(0.15, 0.35),
                c(1.4, 0.0),
            ],
        )
        .unwrap();
        (p, b)
    }

    fn analytic_congruence(p: &ComplexTensor, b: &DenseHermitianMatrix) -> Vec<Complex64> {
        let n_coord = p.shape()[0];
        let n_basis = p.shape()[1];
        let mut result = vec![Complex64::default(); n_basis * n_basis];
        for left in 0..n_basis {
            for right in 0..n_basis {
                let mut value = Complex64::default();
                for row in 0..n_coord {
                    for column in 0..n_coord {
                        value += p.get(&[row, left]).unwrap().conj()
                            * b.get(row, column).unwrap()
                            * p.get(&[column, right]).unwrap();
                    }
                }
                result[left * n_basis + right] = value;
            }
        }
        result
    }

    #[test]
    fn einsum_site_congruence_matches_direct_complex_oracle() {
        let (p, b) = projection_and_block();
        let conjugated = p.conjugate();
        let projected = einsum("ci,cd,dj->ij", &[&conjugated, b.as_tensor(), &p]).unwrap();
        let expected = analytic_congruence(&p, &b);
        assert_eq!(projected.axes(), &[Axis::SiteBasis, Axis::SiteBasis]);
        for (actual, expected) in projected.to_host_row_major().iter().zip(&expected) {
            assert!((actual - expected).norm() < 1.0e-13);
        }
        assert_eq!(active_backend_name(), "rstsr-0.7.10/tblis");
    }

    #[test]
    fn hermitian_congruence_uses_einsum_and_matches_oracle() {
        let (p, b) = projection_and_block();
        let projected = hermitian_congruence(&p, &b).unwrap();
        let expected = analytic_congruence(&p, &b);
        let a = c(0.3, -0.2);
        let bb = c(-0.1, 0.4);
        let apw_lo = a.conj() * b.get(0, 2).unwrap() + bb.conj() * b.get(1, 2).unwrap();
        assert!((projected.get(0, 1).unwrap() - apw_lo).norm() < 1.0e-13);
        assert_eq!(projected.get(1, 1).unwrap(), c(1.4, 0.0));
        for (actual, expected) in projected.to_host_row_major().iter().zip(&expected) {
            assert!((actual - expected).norm() < 1.0e-13);
        }
    }

    #[test]
    fn conjugate_preserves_axes_and_flips_imaginary_part() {
        let p = ComplexTensor::from_host_row_major(
            &[2, 1],
            &[Axis::SiteCoordinate, Axis::SiteBasis],
            vec![c(0.0, 1.0), c(1.0, -0.5)],
        )
        .unwrap();
        let conjugated = p.conjugate();
        assert_eq!(conjugated.axes(), &[Axis::SiteCoordinate, Axis::SiteBasis]);
        assert_eq!(conjugated.get(&[0, 0]).unwrap(), c(0.0, -1.0));
        assert_eq!(conjugated.get(&[1, 0]).unwrap(), c(1.0, 0.5));
        assert_eq!(p.dtype(), DType::Complex64);
        assert_eq!(p.world(), LocalWorld::new());
    }

    #[test]
    fn axis_and_shape_mismatches_are_traceable() {
        let (p, b) = projection_and_block();
        let wrong_axis =
            DenseHermitianMatrix::from_host_row_major(3, Axis::GlobalBasis, b.to_host_row_major())
                .unwrap();
        let error = hermitian_congruence(&p, &wrong_axis).unwrap_err();
        assert_eq!(
            error,
            TensorError::Axis {
                index: 0,
                expected: Axis::GlobalBasis,
                actual: Axis::SiteCoordinate,
            }
        );

        let short = ComplexTensor::from_host_row_major(
            &[2, 2],
            &[Axis::SiteCoordinate, Axis::SiteBasis],
            vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
        )
        .unwrap();
        let error = hermitian_congruence(&short, &b).unwrap_err();
        assert!(matches!(
            error,
            TensorError::Contraction {
                left: Axis::SiteCoordinate,
                left_len: 3,
                right: Axis::SiteCoordinate,
                right_len: 2
            }
        ));
    }

    #[test]
    fn dense_eigenvectors_report_the_mismatched_axis() {
        let tensor = ComplexTensor::from_host_row_major(
            &[2, 1],
            &[Axis::GlobalBasis, Axis::Reduced],
            vec![c(1.0, 0.0), c(0.0, 0.0)],
        )
        .unwrap();
        let error = DenseEigenvectors::from_tensor(tensor).unwrap_err();
        assert_eq!(
            error,
            TensorError::Axis {
                index: 1,
                expected: Axis::Band,
                actual: Axis::Reduced,
            }
        );
    }

    #[test]
    fn dense_eigenvectors_are_column_major_basis_by_band() {
        let row_major = vec![
            c(1.0, 0.1),
            c(2.0, 0.2),
            c(3.0, 0.3),
            c(4.0, 0.4),
            c(5.0, 0.5),
            c(6.0, 0.6),
        ];
        let tensor = ComplexTensor::from_host_row_major(
            &[2, 3],
            &[Axis::GlobalBasis, Axis::Band],
            row_major.clone(),
        )
        .unwrap();
        let eigenvectors = DenseEigenvectors::from_tensor(tensor).unwrap();
        let column_major = vec![
            c(1.0, 0.1),
            c(4.0, 0.4),
            c(2.0, 0.2),
            c(5.0, 0.5),
            c(3.0, 0.3),
            c(6.0, 0.6),
        ];

        assert_eq!(eigenvectors.layout(), MemoryLayout::ColumnMajor);
        assert_eq!(eigenvectors.as_tensor().layout(), MemoryLayout::ColumnMajor);
        assert_eq!(eigenvectors.to_host_row_major(), row_major);
        assert_eq!(eigenvectors.to_host_column_major(), column_major);
        assert_eq!(eigenvectors.at(1, 2), c(6.0, 0.6));

        let imported = DenseEigenvectors::from_host_column_major(2, 3, column_major).unwrap();
        assert_eq!(imported, eigenvectors);

        for (basis_count, band_count) in [(1, 3), (3, 1), (1, 1)] {
            let degenerate = DenseEigenvectors::from_host_column_major(
                basis_count,
                band_count,
                vec![c(1.0, 0.0); basis_count * band_count],
            )
            .unwrap();
            assert_eq!(degenerate.layout(), MemoryLayout::ColumnMajor);
            assert_eq!(degenerate.as_tensor().layout(), MemoryLayout::ColumnMajor);
        }
    }

    #[test]
    fn non_hermitian_host_data_is_rejected() {
        let error = DenseHermitianMatrix::from_host_row_major(
            2,
            Axis::SiteCoordinate,
            vec![c(1.0, 0.0), c(0.0, 1.0), c(0.0, 1.0), c(1.0, 0.0)],
        )
        .unwrap_err();
        assert_eq!(error, TensorError::NonHermitian { row: 0, column: 1 });
    }

    #[test]
    fn no_lo_projection_reduces_to_apw_block() {
        let p = ComplexTensor::from_host_row_major(
            &[2, 1],
            &[Axis::SiteCoordinate, Axis::SiteBasis],
            vec![c(0.3, -0.2), c(-0.1, 0.4)],
        )
        .unwrap();
        let b = DenseHermitianMatrix::from_host_row_major(
            2,
            Axis::SiteCoordinate,
            vec![c(1.2, 0.0), c(0.1, -0.2), c(0.1, 0.2), c(0.8, 0.0)],
        )
        .unwrap();
        let projected = hermitian_congruence(&p, &b).unwrap();
        let expected = analytic_congruence(&p, &b)[0];
        assert!((projected.get(0, 0).unwrap() - expected).norm() < 1.0e-13);
    }

    #[test]
    fn eigensolver_einsums_match_direct_reductions() {
        // X = U diag(s^{-1/2}), H_red = X^H H X, C = X Z, residual HC - S C ε.
        let u = ComplexTensor::from_host_row_major(
            &[2, 2],
            &[Axis::GlobalBasis, Axis::Reduced],
            vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
        )
        .unwrap();
        let scale = ComplexTensor::from_host_row_major(
            &[2],
            &[Axis::Reduced],
            vec![c(0.5, 0.0), c(2.0, 0.0)],
        )
        .unwrap();
        let x = einsum("ik,k->ik", &[&u, &scale]).unwrap();
        assert!((x.get(&[0, 0]).unwrap() - c(0.5, 0.0)).norm() < 1.0e-14);
        assert!((x.get(&[1, 1]).unwrap() - c(2.0, 0.0)).norm() < 1.0e-14);

        let h = DenseHermitianMatrix::from_host_row_major(
            2,
            Axis::GlobalBasis,
            vec![c(2.0, 0.0), c(0.0, 1.0), c(0.0, -1.0), c(3.0, 0.0)],
        )
        .unwrap();
        let x_conj = x.conjugate();
        let reduced = einsum("ir,ij,js->rs", &[&x_conj, h.as_tensor(), &x]).unwrap();
        // X = diag(0.5, 2), H_red_00 = 0.25 * 2 = 0.5, H_red_11 = 4 * 3 = 12,
        // H_red_01 = 0.5 * (0+i) * 2 = i.
        assert!((reduced.get(&[0, 0]).unwrap() - c(0.5, 0.0)).norm() < 1.0e-13);
        assert!((reduced.get(&[1, 1]).unwrap() - c(12.0, 0.0)).norm() < 1.0e-13);
        assert!((reduced.get(&[0, 1]).unwrap() - c(0.0, 1.0)).norm() < 1.0e-13);

        let z = ComplexTensor::from_host_row_major(
            &[2, 1],
            &[Axis::Reduced, Axis::Band],
            vec![c(1.0, 0.0), c(0.0, 0.0)],
        )
        .unwrap();
        let c_mat = einsum("ir,rb->ib", &[&x, &z]).unwrap();
        assert!((c_mat.get(&[0, 0]).unwrap() - c(0.5, 0.0)).norm() < 1.0e-14);
        assert!((c_mat.get(&[1, 0]).unwrap() - c(0.0, 0.0)).norm() < 1.0e-14);

        let hc = einsum("ij,jb->ib", &[h.as_tensor(), &c_mat]).unwrap();
        let s = DenseHermitianMatrix::from_host_row_major(
            2,
            Axis::GlobalBasis,
            vec![c(4.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
        )
        .unwrap();
        let sc = einsum("ij,jb->ib", &[s.as_tensor(), &c_mat]).unwrap();
        let eps =
            ComplexTensor::from_host_row_major(&[1], &[Axis::Band], vec![c(0.5, 0.0)]).unwrap();
        let sc_eps = einsum("ib,b->ib", &[&sc, &eps]).unwrap();
        let residual = hc.sub(&sc_eps).unwrap();
        let residual_conj = residual.conjugate();
        let norm_sq = einsum("ib,ib->b", &[&residual_conj, &residual]).unwrap();
        assert_eq!(norm_sq.axes(), &[Axis::Band]);
        assert!(norm_sq.get(&[0]).unwrap().re >= 0.0);
        assert!(norm_sq.get(&[0]).unwrap().im.abs() < 1.0e-12);
    }

    #[cfg(feature = "backend-tenferro")]
    #[test]
    fn tenferro_einsum_matches_rstsr_tblis_site_congruence() {
        let (p, b) = projection_and_block();
        let conjugated = p.conjugate();
        let operands = [&conjugated, b.as_tensor(), &p];
        let rstsr = einsum("ci,cd,dj->ij", &operands).unwrap();
        let tenferro = einsum_tenferro("ci,cd,dj->ij", &operands).unwrap();
        for (left, right) in rstsr
            .to_host_row_major()
            .iter()
            .zip(tenferro.to_host_row_major())
        {
            assert!((left - right).norm() < 1.0e-12);
        }
        assert_eq!(tenferro.axes(), rstsr.axes());
    }
}
