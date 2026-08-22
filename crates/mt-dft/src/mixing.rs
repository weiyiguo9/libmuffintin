//! Stateful density mixers using the regional physical metric.

use crate::{RegionalDensity, RegionalError};
use thiserror::Error;

/// One SCF input and its residual, with `residual = input - output`.
#[derive(Clone, Debug, PartialEq)]
pub struct MixRecord {
    pub input: RegionalDensity,
    pub residual: RegionalDensity,
}

/// Invalid mixer configuration or a regional-algebra failure.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum MixingError {
    #[error("mixing alpha must be finite and in (0, 1], got {0}")]
    InvalidAlpha(f64),
    #[error("nonlinear mixer history must hold at least two records, got {0}")]
    HistoryTooShort(usize),
    #[error(transparent)]
    Regional(#[from] RegionalError),
}

/// Selectable, stateful density mixing algorithm.
///
/// Broyden type 2 uses the multisecant inverse-Jacobian update
/// `B = alpha I + (S - alpha Y) (Y^T Y)^-1 Y^T`. Pulay--Anderson minimizes
/// the residual norm subject to coefficients summing to one, then applies
/// the same linear preconditioner. Every dot product in both algorithms is
/// [`RegionalDensity::physical_inner_product`].
#[derive(Clone, Debug, PartialEq)]
pub enum DensityMixer {
    Linear {
        alpha: f64,
    },
    Broyden2 {
        alpha: f64,
        max_history: usize,
        history: Vec<MixRecord>,
    },
    PulayAnderson {
        alpha: f64,
        max_history: usize,
        history: Vec<MixRecord>,
        last_coefficients: Vec<f64>,
    },
}

impl DensityMixer {
    pub fn linear(alpha: f64) -> Result<Self, MixingError> {
        validate_alpha(alpha)?;
        Ok(Self::Linear { alpha })
    }

    pub fn broyden2(alpha: f64, max_history: usize) -> Result<Self, MixingError> {
        validate_alpha(alpha)?;
        validate_history(max_history)?;
        Ok(Self::Broyden2 {
            alpha,
            max_history,
            history: Vec::new(),
        })
    }

    pub fn pulay_anderson(alpha: f64, max_history: usize) -> Result<Self, MixingError> {
        validate_alpha(alpha)?;
        validate_history(max_history)?;
        Ok(Self::PulayAnderson {
            alpha,
            max_history,
            history: Vec::new(),
            last_coefficients: Vec::new(),
        })
    }

    /// Stored records, in oldest-to-newest order.
    pub fn history(&self) -> &[MixRecord] {
        match self {
            Self::Linear { .. } => &[],
            Self::Broyden2 { history, .. } | Self::PulayAnderson { history, .. } => history,
        }
    }

    /// Coefficients from the last successful Pulay solve.
    ///
    /// An empty slice means that the selected algorithm is not Pulay or that
    /// the latest Pulay step fell back to a linear update.
    pub fn last_pulay_coefficients(&self) -> &[f64] {
        match self {
            Self::PulayAnderson {
                last_coefficients, ..
            } => last_coefficients,
            _ => &[],
        }
    }

    /// Mix one SCF output into its input.
    ///
    /// The residual convention is `input - output`, hence the linear step is
    /// exactly `input - alpha * residual`.
    pub fn mix(
        &mut self,
        input: &RegionalDensity,
        output: &RegionalDensity,
    ) -> Result<RegionalDensity, MixingError> {
        let residual = input.difference(output)?;
        let record = MixRecord {
            input: input.clone(),
            residual,
        };

        match self {
            Self::Linear { alpha } => linear_step(input, &record.residual, *alpha),
            Self::Broyden2 {
                alpha,
                max_history,
                history,
            } => {
                validate_alpha(*alpha)?;
                validate_history(*max_history)?;
                push_bounded(history, record, *max_history);
                match broyden_step(history, *alpha)? {
                    Some(step) => Ok(step),
                    None => {
                        if history.len() > 1 {
                            history.remove(0);
                        }
                        match broyden_step(history, *alpha)? {
                            Some(step) => Ok(step),
                            None => linear_step(input, &history.last().unwrap().residual, *alpha),
                        }
                    }
                }
            }
            Self::PulayAnderson {
                alpha,
                max_history,
                history,
                last_coefficients,
            } => {
                validate_alpha(*alpha)?;
                validate_history(*max_history)?;
                push_bounded(history, record, *max_history);
                if let Some((mixed, coefficients)) = pulay_step(history, *alpha)? {
                    *last_coefficients = coefficients;
                    return Ok(mixed);
                }
                if history.len() > 1 {
                    history.remove(0);
                }
                if let Some((mixed, coefficients)) = pulay_step(history, *alpha)? {
                    *last_coefficients = coefficients;
                    return Ok(mixed);
                }
                last_coefficients.clear();
                linear_step(input, &history.last().unwrap().residual, *alpha)
            }
        }
    }
}

fn validate_alpha(alpha: f64) -> Result<(), MixingError> {
    if alpha.is_finite() && alpha > 0.0 && alpha <= 1.0 {
        Ok(())
    } else {
        Err(MixingError::InvalidAlpha(alpha))
    }
}

fn validate_history(max_history: usize) -> Result<(), MixingError> {
    if max_history >= 2 {
        Ok(())
    } else {
        Err(MixingError::HistoryTooShort(max_history))
    }
}

fn push_bounded(history: &mut Vec<MixRecord>, record: MixRecord, max_history: usize) {
    history.push(record);
    if history.len() > max_history {
        history.remove(0);
    }
}

fn linear_step(
    input: &RegionalDensity,
    residual: &RegionalDensity,
    alpha: f64,
) -> Result<RegionalDensity, MixingError> {
    let mut mixed = input.clone();
    mixed.add_scaled(-alpha, residual)?;
    Ok(mixed)
}

fn broyden_step(history: &[MixRecord], alpha: f64) -> Result<Option<RegionalDensity>, MixingError> {
    if history.len() < 2 {
        return Ok(None);
    }
    let current = history.last().unwrap();
    let mut steps = Vec::with_capacity(history.len() - 1);
    let mut residual_changes = Vec::with_capacity(history.len() - 1);
    for pair in history.windows(2) {
        steps.push(pair[1].input.difference(&pair[0].input)?);
        residual_changes.push(pair[1].residual.difference(&pair[0].residual)?);
    }

    let number = residual_changes.len();
    let gram = metric_matrix(&residual_changes)?;
    let mut projection = Vec::with_capacity(number);
    for change in &residual_changes {
        projection.push(change.physical_inner_product(&current.residual)?);
    }
    let Some(coefficients) = solve_dense(gram, projection) else {
        return Ok(None);
    };

    let mut action = current.residual.zero_like();
    action.add_scaled(alpha, &current.residual)?;
    for ((step, change), coefficient) in steps.iter().zip(&residual_changes).zip(coefficients) {
        action.add_scaled(coefficient, step)?;
        action.add_scaled(-alpha * coefficient, change)?;
    }
    let mut mixed = current.input.clone();
    mixed.add_scaled(-1.0, &action)?;
    Ok(Some(mixed))
}

fn pulay_step(
    history: &[MixRecord],
    alpha: f64,
) -> Result<Option<(RegionalDensity, Vec<f64>)>, MixingError> {
    if history.is_empty() {
        return Ok(None);
    }
    let number = history.len();
    let residuals: Vec<_> = history
        .iter()
        .map(|record| record.residual.clone())
        .collect();
    let gram = metric_matrix(&residuals)?;
    let dimension = number + 1;
    let mut constrained = vec![vec![0.0; dimension]; dimension];
    for row in 0..number {
        constrained[row][..number].copy_from_slice(&gram[row]);
        constrained[row][number] = 1.0;
        constrained[number][row] = 1.0;
    }
    let mut right = vec![0.0; dimension];
    right[number] = 1.0;
    let Some(solution) = solve_dense(constrained, right) else {
        return Ok(None);
    };
    let coefficients = solution[..number].to_vec();

    let mut mixed = history[0].input.zero_like();
    for (record, &coefficient) in history.iter().zip(&coefficients) {
        mixed.add_scaled(coefficient, &record.input)?;
        mixed.add_scaled(-alpha * coefficient, &record.residual)?;
    }
    Ok(Some((mixed, coefficients)))
}

fn metric_matrix(vectors: &[RegionalDensity]) -> Result<Vec<Vec<f64>>, RegionalError> {
    let mut matrix = vec![vec![0.0; vectors.len()]; vectors.len()];
    for row in 0..vectors.len() {
        for column in 0..=row {
            let value = vectors[row].physical_inner_product(&vectors[column])?;
            matrix[row][column] = value;
            matrix[column][row] = value;
        }
    }
    Ok(matrix)
}

/// Deterministic Gaussian elimination with scaled singular-pivot detection.
///
/// This intentionally has no regularization path: a rank-deficient history
/// is handled by dropping its oldest record and, if necessary, falling back
/// to the linear mixer.
fn solve_dense(mut matrix: Vec<Vec<f64>>, mut right: Vec<f64>) -> Option<Vec<f64>> {
    let dimension = right.len();
    if matrix.len() != dimension || matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }
    let scale = matrix
        .iter()
        .flatten()
        .chain(&right)
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    let tolerance = 256.0 * f64::EPSILON * scale * dimension.max(1) as f64;

    for column in 0..dimension {
        let pivot = (column..dimension).max_by(|&left, &right_row| {
            matrix[left][column]
                .abs()
                .total_cmp(&matrix[right_row][column].abs())
        })?;
        if matrix[pivot][column].abs() <= tolerance {
            return None;
        }
        matrix.swap(column, pivot);
        right.swap(column, pivot);

        for row in column + 1..dimension {
            let factor = matrix[row][column] / matrix[column][column];
            matrix[row][column] = 0.0;
            let (upper, lower) = matrix.split_at_mut(row);
            let pivot_tail = &upper[column][column + 1..];
            let target_tail = &mut lower[0][column + 1..];
            for (target, &pivot_entry) in target_tail.iter_mut().zip(pivot_tail) {
                *target -= factor * pivot_entry;
            }
            right[row] -= factor * right[column];
        }
    }

    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let tail = matrix[row][row + 1..]
            .iter()
            .zip(&solution[row + 1..])
            .map(|(coefficient, value)| coefficient * value)
            .sum::<f64>();
        solution[row] = (right[row] - tail) / matrix[row][row];
        if !solution[row].is_finite() {
            return None;
        }
    }
    Some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InterstitialField;
    use muffintin_core::{
        FourierLayout, GVector, InterstitialGeometry, InverseBohr, ReciprocalLattice, VolumeBohr3,
    };
    use muffintin_lapw::Collinear;
    use num_complex::Complex64;
    use std::collections::BTreeMap;
    use std::f64::consts::TAU;

    fn layout(indices: &[[i32; 3]]) -> FourierLayout {
        let reciprocal = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        let vectors = indices
            .iter()
            .map(|&index| {
                let cartesian = reciprocal.cartesian(index);
                let norm = cartesian
                    .iter()
                    .map(|component| component.get().powi(2))
                    .sum::<f64>()
                    .sqrt();
                GVector {
                    index,
                    cartesian,
                    norm: InverseBohr(norm),
                }
            })
            .collect();
        FourierLayout::new(reciprocal, vectors).unwrap()
    }

    fn density(
        indices: &[[i32; 3]],
        coefficients: impl IntoIterator<Item = ([i32; 3], Complex64)>,
    ) -> RegionalDensity {
        let layout = layout(indices);
        let up =
            InterstitialField::new(layout.clone(), coefficients.into_iter().collect()).unwrap();
        let down = InterstitialField::new(
            layout,
            indices
                .iter()
                .copied()
                .map(|g| (g, Complex64::new(0.0, 0.0)))
                .collect::<BTreeMap<_, _>>(),
        )
        .unwrap();
        RegionalDensity::new(
            InterstitialGeometry::new(VolumeBohr3(TAU.powi(3)), Vec::new()).unwrap(),
            Collinear::new(Vec::new(), Vec::new()),
            Collinear::new(up, down),
        )
        .unwrap()
    }

    fn scalar(value: f64) -> RegionalDensity {
        density(&[[0; 3]], [([0; 3], Complex64::new(value, 0.0))])
    }

    fn scalar_value(density: &RegionalDensity) -> f64 {
        density.interstitial().up.coefficient([0; 3]).unwrap().re
    }

    #[test]
    fn dense_solver_rejects_rank_deficiency_without_regularization() {
        assert_eq!(
            solve_dense(vec![vec![1.0, 1.0], vec![1.0, 1.0]], vec![1.0, 1.0]),
            None
        );
    }

    #[test]
    fn dense_solver_solves_a_regular_system() {
        let solution = solve_dense(vec![vec![2.0, 1.0], vec![1.0, 3.0]], vec![1.0, 2.0]).unwrap();
        assert!((solution[0] - 0.2).abs() < 1.0e-14);
        assert!((solution[1] - 0.6).abs() < 1.0e-14);
    }

    #[test]
    fn linear_step_uses_input_minus_alpha_residual() {
        let mut mixer = DensityMixer::linear(0.25).unwrap();
        let mixed = mixer.mix(&scalar(4.0), &scalar(0.0)).unwrap();
        assert!((scalar_value(&mixed) - 3.0).abs() < 1.0e-15);
    }

    #[test]
    fn broyden_type_two_learns_the_scalar_secant() {
        let mut mixer = DensityMixer::broyden2(0.1, 4).unwrap();
        let first = mixer.mix(&scalar(1.0), &scalar(-1.0)).unwrap();
        assert!((scalar_value(&first) - 0.8).abs() < 1.0e-14);

        // residual = 2 x, so the exact inverse residual Jacobian is 1/2.
        let second = mixer.mix(&scalar(0.8), &scalar(-0.8)).unwrap();
        assert!(scalar_value(&second).abs() < 1.0e-13);
        assert_eq!(mixer.history().len(), 2);
        assert!((scalar_value(&mixer.history()[1].residual) - 1.6).abs() < 1.0e-15);
    }

    #[test]
    fn pulay_coefficients_sum_to_one_and_reduce_combined_residual() {
        let indices = [[-1, 0, 0], [0; 3], [1, 0, 0]];
        let zero = density(&indices, indices.map(|g| (g, Complex64::new(0.0, 0.0))));
        let residual_g0 = density(
            &indices,
            [
                ([-1, 0, 0], Complex64::new(0.0, 0.0)),
                ([0; 3], Complex64::new(1.0, 0.0)),
                ([1, 0, 0], Complex64::new(0.0, 0.0)),
            ],
        );
        let pair_value = 1.0 / 2.0_f64.sqrt();
        let residual_pair = density(
            &indices,
            [
                ([-1, 0, 0], Complex64::new(pair_value, 0.0)),
                ([0; 3], Complex64::new(0.0, 0.0)),
                ([1, 0, 0], Complex64::new(pair_value, 0.0)),
            ],
        );
        let output_first = zero.difference(&residual_g0).unwrap();
        let output_second = zero.difference(&residual_pair).unwrap();
        let mut mixer = DensityMixer::pulay_anderson(0.2, 4).unwrap();
        mixer.mix(&zero, &output_first).unwrap();
        mixer.mix(&zero, &output_second).unwrap();

        let coefficients = mixer.last_pulay_coefficients();
        assert_eq!(coefficients.len(), 2);
        assert!((coefficients.iter().sum::<f64>() - 1.0).abs() < 1.0e-14);
        let mut combined = residual_g0.zero_like();
        combined.add_scaled(coefficients[0], &residual_g0).unwrap();
        combined
            .add_scaled(coefficients[1], &residual_pair)
            .unwrap();
        assert!(combined.residual_rms().unwrap() < residual_g0.residual_rms().unwrap());
        assert!(combined.residual_rms().unwrap() < residual_pair.residual_rms().unwrap());
    }

    #[test]
    fn rank_deficient_histories_fall_back_without_nan() {
        let input_first = scalar(1.0);
        let input_second = scalar(2.0);
        let residual = scalar(0.5);
        let output_first = input_first.difference(&residual).unwrap();
        let output_second = input_second.difference(&residual).unwrap();

        let mut broyden = DensityMixer::broyden2(0.2, 4).unwrap();
        broyden.mix(&input_first, &output_first).unwrap();
        let broyden_result = broyden.mix(&input_second, &output_second).unwrap();
        assert!((scalar_value(&broyden_result) - 1.9).abs() < 1.0e-14);
        assert!(scalar_value(&broyden_result).is_finite());

        let mut pulay = DensityMixer::pulay_anderson(0.2, 4).unwrap();
        pulay.mix(&input_first, &output_first).unwrap();
        let pulay_result = pulay.mix(&input_second, &output_second).unwrap();
        assert!((scalar_value(&pulay_result) - 1.9).abs() < 1.0e-14);
        assert!(scalar_value(&pulay_result).is_finite());
    }
}
