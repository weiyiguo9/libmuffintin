//! Alias-free dense FFT workspace for sparse orbital-pair correlations.

use muffintin_tensor::fft::{FftError, FftGrid, FftPlan};
use num_complex::Complex64;

/// One reusable workspace for
/// `sum_G conj(left[G]) * right[G + relative]`.
pub(crate) struct PairFft {
    grid: FftGrid,
    plan: FftPlan,
    sparse: Vec<Complex64>,
    product: Vec<Complex64>,
    correlation: Vec<Complex64>,
}

impl PairFft {
    /// Size each axis for the complete linear difference range and every raw
    /// support label that will be sampled. Including the latter prevents a
    /// raw label absent from this orbital pair from wrapping onto a populated
    /// difference.
    pub(crate) fn new(
        left_indices: &[[i32; 3]],
        right_indices: &[[i32; 3]],
        raw_indices: &[[i32; 3]],
        wrap: [i32; 3],
    ) -> Result<Self, FftError> {
        let dimensions = if left_indices.is_empty() || right_indices.is_empty() {
            [1; 3]
        } else {
            std::array::from_fn(|axis| {
                let left_min = left_indices.iter().map(|index| index[axis]).min().unwrap();
                let left_max = left_indices.iter().map(|index| index[axis]).max().unwrap();
                let right_min = right_indices.iter().map(|index| index[axis]).min().unwrap();
                let right_max = right_indices.iter().map(|index| index[axis]).max().unwrap();
                let difference_min = i64::from(right_min) - i64::from(left_max);
                let difference_max = i64::from(right_max) - i64::from(left_min);
                let sampled_min = raw_indices
                    .iter()
                    .map(|index| i64::from(index[axis]) - i64::from(wrap[axis]))
                    .min()
                    .unwrap_or(difference_min);
                let sampled_max = raw_indices
                    .iter()
                    .map(|index| i64::from(index[axis]) - i64::from(wrap[axis]))
                    .max()
                    .unwrap_or(difference_max);
                (difference_max.max(sampled_max) - difference_min.min(sampled_min) + 1) as usize
            })
        };
        let grid = FftGrid::new(dimensions)?;
        let plan = FftPlan::new(grid)?;
        let sparse = vec![Complex64::default(); grid.len()];
        let product = vec![Complex64::default(); grid.len()];
        let correlation = vec![Complex64::default(); grid.len()];
        Ok(Self {
            grid,
            plan,
            sparse,
            product,
            correlation,
        })
    }

    pub(crate) fn grid_len(&self) -> usize {
        self.grid.len()
    }

    /// Gather one sparse band and cache its forward transform.
    pub(crate) fn spectrum(
        &mut self,
        indices: &[[i32; 3]],
        values: &[Complex64],
    ) -> Result<Vec<Complex64>, FftError> {
        self.load_sparse(indices, values);
        let mut spectrum = vec![Complex64::default(); self.grid.len()];
        self.plan.forward_into(&self.sparse, &mut spectrum)?;
        Ok(spectrum)
    }

    /// Sum the two spin correlations in reciprocal space and append the
    /// requested real-space support. All transform workspaces are reused.
    pub(crate) fn correlate_spin_spectra_into(
        &mut self,
        left: [&[Complex64]; 2],
        right: [&[Complex64]; 2],
        raw_indices: &[[i32; 3]],
        wrap: [i32; 3],
        output: &mut Vec<Complex64>,
    ) -> Result<(), FftError> {
        for (index, product) in self.product.iter_mut().enumerate() {
            *product =
                left[0][index].conj() * right[0][index] + left[1][index].conj() * right[1][index];
        }
        self.plan
            .inverse_into(&self.product, &mut self.correlation)?;
        output.extend(raw_indices.iter().map(|index| {
            let relative = std::array::from_fn(|axis| index[axis] - wrap[axis]);
            self.correlation[self.grid.index(relative)]
        }));
        Ok(())
    }

    fn load_sparse(&mut self, indices: &[[i32; 3]], values: &[Complex64]) {
        self.sparse.fill(Complex64::default());
        for (&index, &value) in indices.iter().zip(values) {
            self.sparse[self.grid.index(index)] = value;
        }
    }

    /// Correlate one orbital pair and return only the requested raw support,
    /// in its original sparse order. The transfer wrap shifts the sampled
    /// correlation label; it is not folded into either orbital grid.
    pub(crate) fn correlate(
        &mut self,
        left_indices: &[[i32; 3]],
        left: &[Complex64],
        right_indices: &[[i32; 3]],
        right: &[Complex64],
        raw_indices: &[[i32; 3]],
        wrap: [i32; 3],
    ) -> Result<Vec<Complex64>, FftError> {
        self.load_sparse(left_indices, left);
        self.plan.forward_into(&self.sparse, &mut self.product)?;
        self.load_sparse(right_indices, right);
        self.plan
            .forward_into(&self.sparse, &mut self.correlation)?;
        for (target, right) in self.product.iter_mut().zip(&self.correlation) {
            *target = target.conj() * right;
        }
        self.plan
            .inverse_into(&self.product, &mut self.correlation)?;
        Ok(raw_indices
            .iter()
            .map(|index| {
                let relative = std::array::from_fn(|axis| index[axis] - wrap[axis]);
                self.correlation[self.grid.index(relative)]
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::PairFft;
    use num_complex::Complex64;

    #[test]
    fn sparse_wrapped_correlation_matches_direct_pair_sum() {
        let left_indices = [[-2, 0, 1], [1, -1, 0], [0, 2, -1]];
        let right_indices = [[-1, 1, 0], [2, 0, -2], [0, -2, 1], [3, 1, 0]];
        let left = [
            Complex64::new(0.2, -0.4),
            Complex64::new(-0.7, 0.1),
            Complex64::new(0.3, 0.8),
        ];
        let right = [
            Complex64::new(0.5, 0.2),
            Complex64::new(-0.1, -0.6),
            Complex64::new(0.9, -0.3),
            Complex64::new(-0.2, 0.7),
        ];
        let wrap = [2, -1, 1];
        let raw_indices = [[3, 0, 0], [6, 0, 0], [-3, -2, 2], [0, 4, -1]];
        let expected = raw_indices
            .iter()
            .map(|raw| {
                left_indices
                    .iter()
                    .zip(left)
                    .flat_map(|(left_index, left)| {
                        right_indices
                            .iter()
                            .zip(right)
                            .filter_map(move |(right_index, right)| {
                                let relative = std::array::from_fn(|axis| {
                                    right_index[axis] - left_index[axis] + wrap[axis]
                                });
                                (relative == *raw).then_some(left.conj() * right)
                            })
                    })
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        let mut fft = PairFft::new(&left_indices, &right_indices, &raw_indices, wrap).unwrap();
        let actual = fft
            .correlate(
                &left_indices,
                &left,
                &right_indices,
                &right,
                &raw_indices,
                wrap,
            )
            .unwrap();
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (*actual - expected).norm() < 1.0e-10)
        );
    }
}
