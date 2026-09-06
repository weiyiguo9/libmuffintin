//! FFT synthesis on the explicitly natural uniform interstitial parent subset.

use crate::thc_grid::{ThcParentGrid, ThcRegion};
use muffintin_core::{Grid, GridError, InterstitialGrid};
use muffintin_tensor::fft::{FftError, FftGrid, FftPlan};
use num_complex::Complex64;
use std::f64::consts::PI;

#[derive(Debug)]
pub(crate) enum NaturalInterstitialFftError {
    Grid(GridError),
    Fft(FftError),
    ParentGridOrder,
}

/// One reusable transform plan plus the exact natural-grid gather map.
pub(crate) struct NaturalInterstitialFft {
    grid: FftGrid,
    plan: FftPlan,
    gather: Vec<(usize, usize)>,
}

impl NaturalInterstitialFft {
    pub(crate) fn new(parent: &ThcParentGrid) -> Result<Option<Self>, NaturalInterstitialFftError> {
        let Some(uniform) = parent
            .uniform_interstitial_grid()
            .map_err(NaturalInterstitialFftError::Grid)?
        else {
            return Ok(None);
        };
        let interstitial =
            InterstitialGrid::new(&uniform, parent.partition().interstitial().spheres())
                .map_err(NaturalInterstitialFftError::Grid)?;
        let parent_interstitial = parent
            .points()
            .iter()
            .enumerate()
            .filter(|(_, point)| point.region == ThcRegion::Interstitial)
            .collect::<Vec<_>>();
        if parent_interstitial.len() != interstitial.len() {
            return Err(NaturalInterstitialFftError::ParentGridOrder);
        }

        let mut gather = Vec::with_capacity(interstitial.len());
        let mut uniform_index = 0;
        for ((parent_index, parent_point), interstitial_point) in
            parent_interstitial.into_iter().zip(interstitial.points())
        {
            if parent_point.coordinate != interstitial_point.position {
                return Err(NaturalInterstitialFftError::ParentGridOrder);
            }
            while uniform_index < uniform.len()
                && uniform.points()[uniform_index].position != interstitial_point.position
            {
                uniform_index += 1;
            }
            if uniform_index == uniform.len() {
                return Err(NaturalInterstitialFftError::ParentGridOrder);
            }
            gather.push((parent_index, uniform_index));
            uniform_index += 1;
        }

        let grid = FftGrid::new(uniform.divisions()).map_err(NaturalInterstitialFftError::Fft)?;
        let plan = FftPlan::new(grid).map_err(NaturalInterstitialFftError::Fft)?;
        Ok(Some(Self { grid, plan, gather }))
    }

    /// Synthesize one cell-periodic orbital and scatter only the natural
    /// interstitial subset. Reciprocal modes alias by accumulation, and the
    /// midpoint phase preserves the `UniformGrid` sampling convention.
    pub(crate) fn synthesize(
        &mut self,
        modes: impl IntoIterator<Item = ([i32; 3], Complex64)>,
        scale: f64,
        mut scatter: impl FnMut(usize, Complex64),
    ) -> Result<(), FftError> {
        let dimensions = self.grid.dimensions();
        let mut spectrum = vec![Complex64::default(); self.grid.len()];
        for (mode, coefficient) in modes {
            let midpoint_argument = PI
                * mode
                    .into_iter()
                    .enumerate()
                    .map(|(axis, index)| f64::from(index) / dimensions[axis] as f64)
                    .sum::<f64>();
            spectrum[self.grid.index(mode)] += coefficient
                * Complex64::from_polar(scale * self.grid.len() as f64, midpoint_argument);
        }
        let values = self.plan.inverse(&spectrum)?;
        for &(parent_index, uniform_index) in &self.gather {
            scatter(parent_index, values[uniform_index]);
        }
        Ok(())
    }
}
