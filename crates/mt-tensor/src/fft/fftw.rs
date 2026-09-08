use ::fftw::array::AlignedVec;
use ::fftw::plan::{C2CPlan, C2CPlan64};
use ::fftw::types::{Flag, Sign};
use num_complex::Complex64;

use super::{FftError, FftGrid};

/// Reuses FFTW plans and their aligned workspace across field components.
pub struct FftPlan {
    forward: C2CPlan64,
    inverse: C2CPlan64,
    input: AlignedVec<Complex64>,
    output: AlignedVec<Complex64>,
}

impl std::fmt::Debug for FftPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FftPlan")
            .field("len", &self.input.len())
            .finish_non_exhaustive()
    }
}

impl FftPlan {
    pub fn new(grid: FftGrid) -> Result<Self, FftError> {
        let mut input = AlignedVec::new(grid.len());
        let mut output = AlignedVec::new(grid.len());
        let forward = C2CPlan64::new(
            &grid.dimensions,
            &mut input,
            &mut output,
            Sign::Forward,
            Flag::ESTIMATE,
        )
        .map_err(|error| FftError::Backend(error.to_string()))?;
        let inverse = C2CPlan64::new(
            &grid.dimensions,
            &mut input,
            &mut output,
            Sign::Backward,
            Flag::ESTIMATE,
        )
        .map_err(|error| FftError::Backend(error.to_string()))?;
        Ok(Self {
            forward,
            inverse,
            input,
            output,
        })
    }

    pub fn forward(&mut self, input: &[Complex64]) -> Result<Vec<Complex64>, FftError> {
        let mut output = vec![Complex64::default(); self.output.len()];
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute a forward transform into caller-owned storage.
    pub fn forward_into(
        &mut self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), FftError> {
        self.load(input)?;
        self.forward
            .c2c(&mut self.input, &mut self.output)
            .map_err(|error| FftError::Backend(error.to_string()))?;
        output.copy_from_slice(&self.output);
        Ok(())
    }

    pub fn inverse(&mut self, input: &[Complex64]) -> Result<Vec<Complex64>, FftError> {
        let mut output = vec![Complex64::default(); self.output.len()];
        self.inverse_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute a normalized inverse transform into caller-owned storage.
    pub fn inverse_into(
        &mut self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), FftError> {
        assert_eq!(output.len(), self.output.len());
        self.load(input)?;
        self.inverse
            .c2c(&mut self.input, &mut self.output)
            .map_err(|error| FftError::Backend(error.to_string()))?;
        let normalization = self.output.len() as f64;
        for (target, value) in output.iter_mut().zip(self.output.iter()) {
            *target = value / normalization;
        }
        Ok(())
    }

    fn load(&mut self, input: &[Complex64]) -> Result<(), FftError> {
        if input.len() != self.input.len() {
            return Err(FftError::InputLength {
                expected: self.input.len(),
                actual: input.len(),
            });
        }
        self.input.copy_from_slice(input);
        Ok(())
    }
}
