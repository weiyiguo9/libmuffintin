use std::f64::consts::TAU;

use num_complex::Complex64;

use super::{FftError, FftGrid};

/// Separable direct DFT, retained as the dependency-free implementation.
#[derive(Debug)]
pub(crate) struct FftPlan {
    grid: FftGrid,
}

impl FftPlan {
    pub(crate) fn new(grid: FftGrid) -> Result<Self, FftError> {
        Ok(Self { grid })
    }

    pub(crate) fn forward(&mut self, input: &[Complex64]) -> Result<Vec<Complex64>, FftError> {
        self.transform(input, -1.0)
    }

    pub(crate) fn inverse(&mut self, input: &[Complex64]) -> Result<Vec<Complex64>, FftError> {
        let mut result = self.transform(input, 1.0)?;
        for value in &mut result {
            *value /= self.grid.len() as f64;
        }
        Ok(result)
    }

    fn transform(&self, input: &[Complex64], sign: f64) -> Result<Vec<Complex64>, FftError> {
        if input.len() != self.grid.len() {
            return Err(FftError::InputLength {
                expected: self.grid.len(),
                actual: input.len(),
            });
        }
        let mut values = input.to_vec();
        let mut output = vec![Complex64::default(); input.len()];
        for axis in (0..3).rev() {
            let n = self.grid.dimensions[axis];
            let stride: usize = self.grid.dimensions[axis + 1..].iter().product();
            for block in (0..values.len()).step_by(n * stride) {
                for offset in 0..stride {
                    for k in 0..n {
                        let mut sum = Complex64::default();
                        for j in 0..n {
                            let phase = sign * TAU * (j as f64) * (k as f64) / n as f64;
                            sum += values[block + j * stride + offset]
                                * Complex64::from_polar(1.0, phase);
                        }
                        output[block + k * stride + offset] = sum;
                    }
                }
            }
            std::mem::swap(&mut values, &mut output);
        }
        Ok(values)
    }
}
