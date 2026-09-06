use ::fftw::array::AlignedVec;
use ::fftw::plan::{C2CPlan, C2CPlan64};
use ::fftw::types::{Flag, Sign};
use num_complex::Complex64;

use super::{FftError, FftGrid};

/// Reuses FFTW plans and their aligned workspace across field components.
pub(crate) struct FftPlan {
    forward: C2CPlan64,
    inverse: C2CPlan64,
    input: AlignedVec<Complex64>,
    output: AlignedVec<Complex64>,
}

impl FftPlan {
    pub(crate) fn new(grid: FftGrid) -> Result<Self, FftError> {
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

    pub(crate) fn forward(&mut self, input: &[Complex64]) -> Result<Vec<Complex64>, FftError> {
        self.load(input)?;
        self.forward
            .c2c(&mut self.input, &mut self.output)
            .map_err(|error| FftError::Backend(error.to_string()))?;
        Ok(self.output.to_vec())
    }

    pub(crate) fn inverse(&mut self, input: &[Complex64]) -> Result<Vec<Complex64>, FftError> {
        self.load(input)?;
        self.inverse
            .c2c(&mut self.input, &mut self.output)
            .map_err(|error| FftError::Backend(error.to_string()))?;
        let normalization = self.output.len() as f64;
        Ok(self
            .output
            .iter()
            .map(|value| value / normalization)
            .collect())
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
