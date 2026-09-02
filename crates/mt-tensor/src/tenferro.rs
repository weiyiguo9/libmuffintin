//! tenferro-rs CPU/faer einsum engine.
//!
//! Storage stays on the RSTSR tensor. This module converts host buffers,
//! runs `tenferro-einsum`, and copies the result back. AD, GPU, and XLA
//! features are not enabled.

use crate::{ComplexTensor, TensorError, backend::infer_output_axes};
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::Tensor;

/// Einstein summation through tenferro-einsum on `CpuBackend`.
pub fn einsum(subscripts: &str, operands: &[&ComplexTensor]) -> Result<ComplexTensor, TensorError> {
    let axes = infer_output_axes(subscripts, operands)?;
    let mut backend = CpuBackend::new();
    let owned = operands
        .iter()
        .map(|tensor| to_tenferro(tensor))
        .collect::<Result<Vec<_>, _>>()?;
    let refs = owned.iter().collect::<Vec<_>>();
    let result = refs
        .as_slice()
        .einsum(subscripts, &mut backend)
        .map_err(|error| TensorError::Backend(error.to_string()))?;
    from_tenferro(result, axes)
}

fn to_tenferro(tensor: &ComplexTensor) -> Result<Tensor, TensorError> {
    let shape = tensor.shape();
    let column_major = row_major_to_col_major(&shape, tensor.to_host_row_major())?;
    Tensor::from_vec_col_major(shape, column_major)
        .map_err(|error| TensorError::Backend(error.to_string()))
}

fn from_tenferro(tensor: Tensor, axes: Vec<crate::Axis>) -> Result<ComplexTensor, TensorError> {
    let shape = tensor.shape().to_vec();
    let column_major = tensor
        .as_slice::<Complex64>()
        .map_err(|error| TensorError::Backend(error.to_string()))?
        .to_vec();
    let row_major = col_major_to_row_major(&shape, column_major)?;
    ComplexTensor::from_host_row_major(&shape, &axes, row_major)
}

fn row_major_to_col_major(
    shape: &[usize],
    row_major: Vec<Complex64>,
) -> Result<Vec<Complex64>, TensorError> {
    match *shape {
        [_] => Ok(row_major),
        [rows, cols] => {
            let mut column_major = vec![Complex64::default(); rows * cols];
            for row in 0..rows {
                for col in 0..cols {
                    column_major[col * rows + row] = row_major[row * cols + col];
                }
            }
            Ok(column_major)
        }
        [outer, rows, cols] => {
            let mut column_major = vec![Complex64::default(); outer * rows * cols];
            for i in 0..outer {
                for j in 0..rows {
                    for k in 0..cols {
                        column_major[i + outer * (j + rows * k)] =
                            row_major[(i * rows + j) * cols + k];
                    }
                }
            }
            Ok(column_major)
        }
        _ => Err(TensorError::UnsupportedRank { rank: shape.len() }),
    }
}

fn col_major_to_row_major(
    shape: &[usize],
    column_major: Vec<Complex64>,
) -> Result<Vec<Complex64>, TensorError> {
    match *shape {
        [_] => Ok(column_major),
        [rows, cols] => {
            let mut row_major = vec![Complex64::default(); rows * cols];
            for row in 0..rows {
                for col in 0..cols {
                    row_major[row * cols + col] = column_major[col * rows + row];
                }
            }
            Ok(row_major)
        }
        [outer, rows, cols] => {
            let mut row_major = vec![Complex64::default(); outer * rows * cols];
            for i in 0..outer {
                for j in 0..rows {
                    for k in 0..cols {
                        row_major[(i * rows + j) * cols + k] =
                            column_major[i + outer * (j + rows * k)];
                    }
                }
            }
            Ok(row_major)
        }
        _ => Err(TensorError::UnsupportedRank { rank: shape.len() }),
    }
}
