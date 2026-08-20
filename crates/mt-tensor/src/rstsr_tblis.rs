//! RSTSR 0.7.10 + TBLIS implementation of the einsum layer.

extern crate tblis_src;

use crate::{
    Axis, ComplexTensor, MemoryLayout, TensorError,
    backend::{EinsumBackend, infer_output_axes},
};
use num_complex::Complex64;
use rstsr::prelude::*;

pub struct RstsrTblisBackend;

type RstsrTensor = Tensor<Complex64, DeviceFaer>;

impl EinsumBackend for RstsrTblisBackend {
    fn name() -> &'static str {
        "rstsr-0.7.10/tblis"
    }

    fn einsum(subscripts: &str, operands: &[&ComplexTensor]) -> Result<ComplexTensor, TensorError> {
        let axes = infer_output_axes(subscripts, operands)?;
        let data = tblis_einsum(subscripts, operands)?;
        Ok(ComplexTensor::from_rstsr(data, axes))
    }
}

pub fn device() -> DeviceFaer {
    DeviceFaer::default()
}

pub fn asarray_row_major(
    shape: &[usize],
    values: Vec<Complex64>,
) -> Result<RstsrTensor, TensorError> {
    let device = device();
    match *shape {
        [n] => Ok(rt::asarray((values, [n], &device))),
        [rows, cols] => Ok(rt::asarray((values, [rows, cols], &device))),
        _ => Err(TensorError::UnsupportedRank { rank: shape.len() }),
    }
}

pub fn conjugate(data: &RstsrTensor) -> RstsrTensor {
    rt::conj(data).into_owned()
}

pub fn layout_of(data: &RstsrTensor) -> MemoryLayout {
    let layout = data.layout();
    if layout.c_contig() {
        MemoryLayout::RowMajor
    } else if layout.f_contig() {
        MemoryLayout::ColumnMajor
    } else {
        MemoryLayout::Strided
    }
}

pub fn shape_of(data: &RstsrTensor) -> Vec<usize> {
    data.shape().to_vec()
}

pub fn get_at(data: &RstsrTensor, indices: &[usize]) -> Result<Complex64, TensorError> {
    match *indices {
        [i] => Ok(data[[i]]),
        [i, j] => Ok(data[[i, j]]),
        _ => Err(TensorError::UnsupportedRank {
            rank: indices.len(),
        }),
    }
}

pub fn to_host_row_major(data: &RstsrTensor) -> Vec<Complex64> {
    data.to_contig(RowMajor).into_owned().reshape(-1).to_vec()
}

fn tblis_einsum(subscripts: &str, operands: &[&ComplexTensor]) -> Result<RstsrTensor, TensorError> {
    let result = match operands {
        [a] => rt::tblis::einsum_f(subscripts, [a.rstsr()], true, None),
        [a, b] => rt::tblis::einsum_f(subscripts, [a.rstsr(), b.rstsr()], true, None),
        [a, b, c] => rt::tblis::einsum_f(subscripts, [a.rstsr(), b.rstsr(), c.rstsr()], true, None),
        [a, b, c, d] => rt::tblis::einsum_f(
            subscripts,
            [a.rstsr(), b.rstsr(), c.rstsr(), d.rstsr()],
            true,
            None,
        ),
        _ => {
            return Err(TensorError::EinsumArity {
                subscripts: subscripts.to_string(),
                expected: 4,
                actual: operands.len(),
            });
        }
    };
    result.map_err(|error| TensorError::Backend(error.to_string()))
}

impl ComplexTensor {
    pub(crate) fn from_rstsr(data: RstsrTensor, axes: Vec<Axis>) -> Self {
        Self { data, axes }
    }

    pub(crate) fn rstsr(&self) -> &RstsrTensor {
        &self.data
    }
}
