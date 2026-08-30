//! Stateful density mixers using the regional physical metric.

use crate::{RegionalDensity, RegionalError};
use std::fmt;
use thiserror::Error;

/// One SCF input and its residual, with `residual = input - output`.
#[derive(Clone, Debug, PartialEq)]
pub struct MixRecord {
    pub input: RegionalDensity,
    pub residual: RegionalDensity,
}

/// Quantity that became non-finite during nonlinear mixing algebra.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixAlgebraQuantity {
    Gram,
    RightHandSide,
    Solution,
}

impl fmt::Display for MixAlgebraQuantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Gram => "Gram matrix",
            Self::RightHandSide => "right-hand side",
            Self::Solution => "solution",
        })
    }
}

/// Invalid mixer configuration, non-finite algebra, or a regional failure.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum MixingError {
    #[error("mixing alpha must be finite and in (0, 1], got {0}")]
    InvalidAlpha(f64),
    #[error("nonlinear mixer history must hold at least two records, got {0}")]
    HistoryTooShort(usize),
    #[error("nonlinear mixing algebra produced a non-finite {quantity}")]
    NonFiniteAlgebra { quantity: MixAlgebraQuantity },
    #[error(transparent)]
    Regional(#[from] RegionalError),
}

/// Public per-step mixing outcome.
///
/// Nonlinear warmup is Broyden's first record or Pulay's one-record
/// coefficient-`[1]` step. Finite rank-deficient Broyden/Pulay algebra may
/// take one explicit linear fallback; non-finite Gram, right-hand-side,
/// solution, or physical-metric results are hard errors, not fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixStatus {
    /// Ordinary linear mixing step.
    Linear,
    /// Broyden's first record, or Pulay's one-record coefficient-`[1]` step.
    NonlinearWarmup,
    /// Successful nonlinear Broyden or Pulay step with sufficient history.
    Nonlinear,
    /// Finite rank-deficient Broyden/Pulay algebra; one explicit linear fallback.
    RankDeficientLinearFallback,
    /// Iteration accepted without mixing, including SCF convergence.
    NotMixed,
}

/// Mixed density together with the algorithm actually used for this step.
#[derive(Clone, Debug, PartialEq)]
pub struct MixStep {
    pub density: RegionalDensity,
    pub status: MixStatus,
}

/// Selectable, stateful density mixing algorithm.
///
/// Broyden type 2 uses the multisecant inverse-Jacobian update
/// `B = alpha I + (S - alpha Y) (Y^T Y)^-1 Y^T`. Pulay--Anderson minimizes
/// the residual norm subject to coefficients summing to one, then applies
/// the same linear preconditioner. Every dot product in both algorithms is
/// [`RegionalDensity::physical_inner_product`].
///
/// Internal mixer state is private so alpha and history bounds cannot bypass
/// the three public constructors.
#[derive(Clone, Debug, PartialEq)]
pub struct DensityMixer {
    inner: InnerMixer,
}

#[derive(Clone, Debug, PartialEq)]
enum InnerMixer {
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
        Ok(Self {
            inner: InnerMixer::Linear { alpha },
        })
    }

    pub fn broyden2(alpha: f64, max_history: usize) -> Result<Self, MixingError> {
        validate_alpha(alpha)?;
        validate_history(max_history)?;
        Ok(Self {
            inner: InnerMixer::Broyden2 {
                alpha,
                max_history,
                history: Vec::new(),
            },
        })
    }

    pub fn pulay_anderson(alpha: f64, max_history: usize) -> Result<Self, MixingError> {
        validate_alpha(alpha)?;
        validate_history(max_history)?;
        Ok(Self {
            inner: InnerMixer::PulayAnderson {
                alpha,
                max_history,
                history: Vec::new(),
                last_coefficients: Vec::new(),
            },
        })
    }

    /// Stored records, in oldest-to-newest order.
    pub fn history(&self) -> &[MixRecord] {
        match &self.inner {
            InnerMixer::Linear { .. } => &[],
            InnerMixer::Broyden2 { history, .. } | InnerMixer::PulayAnderson { history, .. } => {
                history
            }
        }
    }

    /// Coefficients from the last Pulay warmup or successful Pulay solve.
    ///
    /// An empty slice means that the selected algorithm is not Pulay or that
    /// the latest Pulay step reported [`MixStatus::RankDeficientLinearFallback`].
    pub fn last_pulay_coefficients(&self) -> &[f64] {
        match &self.inner {
            InnerMixer::PulayAnderson {
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
    ) -> Result<MixStep, MixingError> {
        let snapshot = self.clone();
        match self.mix_inner(input, output) {
            Ok(step) => Ok(step),
            Err(error) => {
                *self = snapshot;
                Err(error)
            }
        }
    }

    fn mix_inner(
        &mut self,
        input: &RegionalDensity,
        output: &RegionalDensity,
    ) -> Result<MixStep, MixingError> {
        let residual = input.difference(output)?;
        let record = MixRecord {
            input: input.clone(),
            residual,
        };

        match &mut self.inner {
            InnerMixer::Linear { alpha } => {
                let density = linear_step(input, &record.residual, *alpha)?;
                Ok(MixStep {
                    density,
                    status: MixStatus::Linear,
                })
            }
            InnerMixer::Broyden2 {
                alpha,
                max_history,
                history,
            } => {
                push_bounded(history, record, *max_history);
                if history.len() < 2 {
                    let density = linear_step(input, &history.last().unwrap().residual, *alpha)?;
                    return Ok(MixStep {
                        density,
                        status: MixStatus::NonlinearWarmup,
                    });
                }
                match broyden_step(history, *alpha)? {
                    Some(density) => Ok(MixStep {
                        density,
                        status: MixStatus::Nonlinear,
                    }),
                    None => {
                        let density =
                            linear_step(input, &history.last().unwrap().residual, *alpha)?;
                        Ok(MixStep {
                            density,
                            status: MixStatus::RankDeficientLinearFallback,
                        })
                    }
                }
            }
            InnerMixer::PulayAnderson {
                alpha,
                max_history,
                history,
                last_coefficients,
            } => {
                push_bounded(history, record, *max_history);
                if history.len() == 1 {
                    *last_coefficients = vec![1.0];
                    let density = linear_step(input, &history[0].residual, *alpha)?;
                    return Ok(MixStep {
                        density,
                        status: MixStatus::NonlinearWarmup,
                    });
                }
                match pulay_step(history, *alpha)? {
                    Some((density, coefficients)) => {
                        *last_coefficients = coefficients;
                        Ok(MixStep {
                            density,
                            status: MixStatus::Nonlinear,
                        })
                    }
                    None => {
                        last_coefficients.clear();
                        let density =
                            linear_step(input, &history.last().unwrap().residual, *alpha)?;
                        Ok(MixStep {
                            density,
                            status: MixStatus::RankDeficientLinearFallback,
                        })
                    }
                }
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
    let Some(coefficients) = solve_dense(gram, projection)? else {
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
    let Some(solution) = solve_dense(constrained, right)? else {
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
/// A finite rank-deficient system returns `Ok(None)` so the mixer can take one
/// explicit linear fallback. Non-finite Gram, right-hand-side, or solution
/// entries are hard errors, not rank deficiency.
fn solve_dense(
    mut matrix: Vec<Vec<f64>>,
    mut right: Vec<f64>,
) -> Result<Option<Vec<f64>>, MixingError> {
    let dimension = right.len();
    debug_assert_eq!(
        matrix.len(),
        dimension,
        "metric_matrix produces a square Gram of the residual length"
    );
    debug_assert!(
        matrix.iter().all(|row| row.len() == dimension),
        "metric_matrix produces a square Gram of the residual length"
    );
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(MixingError::NonFiniteAlgebra {
            quantity: MixAlgebraQuantity::Gram,
        });
    }
    if right.iter().any(|value| !value.is_finite()) {
        return Err(MixingError::NonFiniteAlgebra {
            quantity: MixAlgebraQuantity::RightHandSide,
        });
    }
    let scale = matrix
        .iter()
        .flatten()
        .chain(&right)
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return Ok(None);
    }
    let tolerance = 256.0 * f64::EPSILON * scale * dimension.max(1) as f64;

    for column in 0..dimension {
        let Some(pivot) = (column..dimension).max_by(|&left, &right_row| {
            matrix[left][column]
                .abs()
                .total_cmp(&matrix[right_row][column].abs())
        }) else {
            return Ok(None);
        };
        if matrix[pivot][column].abs() <= tolerance {
            return Ok(None);
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
            return Err(MixingError::NonFiniteAlgebra {
                quantity: MixAlgebraQuantity::Solution,
            });
        }
    }
    Ok(Some(solution))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InterstitialField, RegionalScalarField};
    use muffintin_core::{
        FourierLayout, GVector, InterstitialGeometry, InverseBohr, ReciprocalLattice, VolumeBohr3,
    };
    use num_complex::Complex64;
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
        let charge_interstitial =
            InterstitialField::new(layout.clone(), coefficients.into_iter().collect()).unwrap();
        let geometry = InterstitialGeometry::new(VolumeBohr3(TAU.powi(3)), Vec::new()).unwrap();
        let charge = RegionalScalarField::new(geometry, Vec::new(), charge_interstitial).unwrap();
        let zero = charge.zero_like();
        RegionalDensity::new(charge, [zero.clone(), zero.clone(), zero]).unwrap()
    }

    fn pauli_scalar(values: [f64; 4]) -> RegionalDensity {
        let layout = layout(&[[0; 3]]);
        let geometry = InterstitialGeometry::new(VolumeBohr3(TAU.powi(3)), Vec::new()).unwrap();
        let fields = values.map(|value| {
            RegionalScalarField::new(
                geometry.clone(),
                Vec::new(),
                InterstitialField::new(
                    layout.clone(),
                    [([0; 3], Complex64::new(value, 0.0))].into_iter().collect(),
                )
                .unwrap(),
            )
            .unwrap()
        });
        let [charge, mx, my, mz] = fields;
        RegionalDensity::new(charge, [mx, my, mz]).unwrap()
    }

    fn scalar(value: f64) -> RegionalDensity {
        density(&[[0; 3]], [([0; 3], Complex64::new(value, 0.0))])
    }

    fn scalar_value(density: &RegionalDensity) -> f64 {
        density
            .charge()
            .interstitial()
            .coefficient([0; 3])
            .unwrap()
            .re
    }

    #[test]
    fn dense_solver_rejects_rank_deficiency_without_regularization() {
        assert_eq!(
            solve_dense(vec![vec![1.0, 1.0], vec![1.0, 1.0]], vec![1.0, 1.0]).unwrap(),
            None
        );
    }

    #[test]
    fn dense_solver_solves_a_regular_system() {
        let solution = solve_dense(vec![vec![2.0, 1.0], vec![1.0, 3.0]], vec![1.0, 2.0]).unwrap();
        let solution = solution.unwrap();
        assert!((solution[0] - 0.2).abs() < 1.0e-14);
        assert!((solution[1] - 0.6).abs() < 1.0e-14);
    }

    #[test]
    fn dense_solver_treats_non_finite_entries_as_hard_errors() {
        assert_eq!(
            solve_dense(
                vec![vec![f64::INFINITY, 0.0], vec![0.0, 1.0]],
                vec![1.0, 1.0]
            ),
            Err(MixingError::NonFiniteAlgebra {
                quantity: MixAlgebraQuantity::Gram,
            })
        );
        assert_eq!(
            solve_dense(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![f64::NAN, 1.0]),
            Err(MixingError::NonFiniteAlgebra {
                quantity: MixAlgebraQuantity::RightHandSide,
            })
        );
    }

    #[test]
    fn linear_step_uses_input_minus_alpha_residual() {
        let mut mixer = DensityMixer::linear(0.25).unwrap();
        let mixed = mixer.mix(&scalar(4.0), &scalar(0.0)).unwrap();
        assert_eq!(mixed.status, MixStatus::Linear);
        assert!((scalar_value(&mixed.density) - 3.0).abs() < 1.0e-15);
    }

    #[test]
    fn linear_step_mixes_charge_and_all_magnetization_components() {
        let mut mixer = DensityMixer::linear(0.25).unwrap();
        let mixed = mixer
            .mix(
                &pauli_scalar([4.0, 1.0, -2.0, 3.0]),
                &pauli_scalar([0.0; 4]),
            )
            .unwrap();
        assert_eq!(mixed.status, MixStatus::Linear);
        let actual = std::iter::once(mixed.density.charge())
            .chain(mixed.density.magnetization())
            .map(|field| field.interstitial().coefficient([0; 3]).unwrap().re)
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![3.0, 0.75, -1.5, 2.25]);
    }

    #[test]
    fn broyden_type_two_learns_the_scalar_secant() {
        let mut mixer = DensityMixer::broyden2(0.1, 4).unwrap();
        let first = mixer.mix(&scalar(1.0), &scalar(-1.0)).unwrap();
        assert_eq!(first.status, MixStatus::NonlinearWarmup);
        assert!((scalar_value(&first.density) - 0.8).abs() < 1.0e-14);

        // residual = 2 x, so the exact inverse residual Jacobian is 1/2.
        let second = mixer.mix(&scalar(0.8), &scalar(-0.8)).unwrap();
        assert_eq!(second.status, MixStatus::Nonlinear);
        assert!(scalar_value(&second.density).abs() < 1.0e-13);
        assert_eq!(mixer.history().len(), 2);
        assert!((scalar_value(&mixer.history()[1].residual) - 1.6).abs() < 1.0e-15);
    }

    #[test]
    fn pulay_one_record_step_is_warmup_with_unit_coefficient() {
        let mut mixer = DensityMixer::pulay_anderson(0.25, 4).unwrap();
        let mixed = mixer.mix(&scalar(4.0), &scalar(0.0)).unwrap();
        assert_eq!(mixed.status, MixStatus::NonlinearWarmup);
        assert_eq!(mixer.last_pulay_coefficients(), &[1.0]);
        assert!((scalar_value(&mixed.density) - 3.0).abs() < 1.0e-15);
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
        let warmup = mixer.mix(&zero, &output_first).unwrap();
        assert_eq!(warmup.status, MixStatus::NonlinearWarmup);
        assert_eq!(mixer.last_pulay_coefficients(), &[1.0]);
        let second = mixer.mix(&zero, &output_second).unwrap();
        assert_eq!(second.status, MixStatus::Nonlinear);

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
    fn rank_deficient_histories_fall_back_without_dropping_records() {
        let input_first = scalar(1.0);
        let input_second = scalar(2.0);
        let residual = scalar(0.5);
        let output_first = input_first.difference(&residual).unwrap();
        let output_second = input_second.difference(&residual).unwrap();

        let mut broyden = DensityMixer::broyden2(0.2, 4).unwrap();
        let warmup = broyden.mix(&input_first, &output_first).unwrap();
        assert_eq!(warmup.status, MixStatus::NonlinearWarmup);
        let broyden_result = broyden.mix(&input_second, &output_second).unwrap();
        assert_eq!(
            broyden_result.status,
            MixStatus::RankDeficientLinearFallback
        );
        assert_eq!(broyden.history().len(), 2);
        assert!((scalar_value(&broyden_result.density) - 1.9).abs() < 1.0e-14);
        assert!(scalar_value(&broyden_result.density).is_finite());

        let mut pulay = DensityMixer::pulay_anderson(0.2, 4).unwrap();
        pulay.mix(&input_first, &output_first).unwrap();
        let pulay_result = pulay.mix(&input_second, &output_second).unwrap();
        assert_eq!(pulay_result.status, MixStatus::RankDeficientLinearFallback);
        assert_eq!(pulay.history().len(), 2);
        assert!(pulay.last_pulay_coefficients().is_empty());
        assert!((scalar_value(&pulay_result.density) - 1.9).abs() < 1.0e-14);
        assert!(scalar_value(&pulay_result.density).is_finite());
    }

    #[test]
    fn non_finite_physical_metric_is_a_hard_error_not_fallback() {
        let mut mixer = DensityMixer::pulay_anderson(0.2, 4).unwrap();
        mixer.mix(&scalar(1.0e200), &scalar(0.0)).unwrap();
        let state_before_failure = mixer.clone();
        let error = mixer.mix(&scalar(2.0e200), &scalar(0.0)).unwrap_err();
        assert!(
            matches!(
                error,
                MixingError::Regional(RegionalError::NonFiniteMetric)
                    | MixingError::NonFiniteAlgebra { .. }
            ),
            "{error:?}"
        );
        assert_eq!(mixer, state_before_failure);
        assert_eq!(mixer.history().len(), 1);
        assert_eq!(mixer.last_pulay_coefficients(), &[1.0]);
    }
}
