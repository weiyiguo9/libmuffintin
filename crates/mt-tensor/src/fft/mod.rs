//! Dense transforms shared by DFT and product-space consumers: last axis fastest, forward unnormalized,
//! inverse normalized by the number of grid points. Physical sampling and
//! sparse Fourier-field contracts remain with the consumers.

#[cfg_attr(feature = "fft-fftw", allow(dead_code))]
mod direct;
#[cfg(feature = "fft-fftw")]
mod fftw;
mod layout;

#[cfg(not(feature = "fft-fftw"))]
pub use direct::FftPlan;
#[cfg(feature = "fft-fftw")]
pub use fftw::FftPlan;
pub use layout::FftGrid;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FftError {
    #[error("invalid FFT grid dimensions {0:?}")]
    InvalidDimensions([usize; 3]),
    #[error("FFT input length {actual}, expected {expected}")]
    InputLength { expected: usize, actual: usize },
    #[cfg(feature = "fft-fftw")]
    #[error("FFTW: {0}")]
    Backend(String),
}

#[cfg(all(test, feature = "fft-fftw"))]
mod tests {
    use super::{FftGrid, FftPlan};
    use num_complex::Complex64;
    use std::f64::consts::TAU;

    #[test]
    fn fftw_rectangular_complex_modes_and_plan_reuse() {
        let dimensions = [3, 4, 5];
        let grid = FftGrid::new(dimensions).unwrap();
        let mut plan = FftPlan::new(grid).unwrap();
        // Two different inputs exercise workspace reuse, not just a round trip
        // that could hide a shared forward/inverse sign or ordering error.
        for mode in [[1, -1, 2], [-1, 1, -2]] {
            let amplitude = Complex64::new(0.3, -0.2);
            let mut samples = Vec::with_capacity(grid.len());
            for i in 0..dimensions[0] {
                for j in 0..dimensions[1] {
                    for k in 0..dimensions[2] {
                        let phase = TAU
                            * [i, j, k]
                                .into_iter()
                                .enumerate()
                                .map(|(axis, point)| {
                                    point as f64 * f64::from(mode[axis]) / dimensions[axis] as f64
                                })
                                .sum::<f64>();
                        samples.push(amplitude * Complex64::from_polar(1.0, phase));
                    }
                }
            }
            let spectrum = plan.forward(&samples).unwrap();
            for (index, value) in spectrum.iter().enumerate() {
                let expected = if index == grid.index(mode) {
                    amplitude
                } else {
                    Complex64::default()
                };
                assert!((value / grid.len() as f64 - expected).norm() < 2.0e-14);
            }
            let restored = plan.inverse(&spectrum).unwrap();
            for (actual, expected) in restored.iter().zip(&samples) {
                assert!((actual - expected).norm() < 2.0e-14);
            }
        }
    }
}
