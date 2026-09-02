//! Per-q interpolation-vector (ζ) fits and residual diagnostics.

use crate::thc::ThcError;
use crate::thc::gram::InjectedCoulombGram;
use crate::thc::linalg::{hermitian_sqrt, lstsq};
use crate::thc::pair::{ExchangePairBlock, PairBlock};
use crate::thc::select::{matmul, reconstruct_pairs, weighted_residual};
use crate::{ExchangePairLayout, ExchangeSpace, PairColumnLayout, TransferQ};
use num_complex::Complex64;

/// Weighted residual pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedResidual {
    pub frobenius: f64,
    pub column_max: f64,
}

/// Per-q interpolation vectors and residuals.
#[derive(Clone, Debug, PartialEq)]
pub struct PerQFit {
    pub q_index: usize,
    pub q: TransferQ,
    pub rank: usize,
    /// ζ on the fit grid, row-major `n_points × n_mu`.
    pub zeta: Vec<Complex64>,
    pub n_points: usize,
    pub n_mu: usize,
    pub l2_all: WeightedResidual,
    pub l2_core: Option<WeightedResidual>,
    pub l2_valence: Option<WeightedResidual>,
    pub coulomb: Option<WeightedResidual>,
}

/// Weighted residuals of one shared exchange fit, kept separate by sector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExchangeSectorResiduals {
    pub vv: WeightedResidual,
    pub cv: WeightedResidual,
    pub vc: WeightedResidual,
    pub cc: WeightedResidual,
}

/// One per-q interpolation-vector fit shared by VV, CV, VC, and CC.
#[derive(Clone, Debug, PartialEq)]
pub struct ExchangePerQFit {
    pub q_index: usize,
    pub q: TransferQ,
    pub rank: usize,
    /// Shared ζ on the fit grid, row-major `n_points × n_mu`.
    pub zeta: Vec<Complex64>,
    pub n_points: usize,
    pub n_mu: usize,
    pub residuals: ExchangeSectorResiduals,
}

/// Least-squares ζ from selected pair rows onto a (possibly weighted) target grid.
///
/// `selected_rows` is `n_mu × n_col` in the same point order as the compiled
/// interpolation-point auxiliary. This is the ISDF collocation, not a Coulomb
/// kernel.
#[allow(clippy::too_many_arguments)]
pub fn fit_per_q(
    selected_rows: &[Complex64],
    n_mu: usize,
    target: &PairBlock,
    weights: &[f64],
    q: TransferQ,
    gram: Option<&InjectedCoulombGram>,
    weight_target: bool,
) -> Result<PerQFit, ThcError> {
    if n_mu == 0 {
        return Err(ThcError::EmptyRank);
    }
    let n_col = target.n_columns();
    let n_pts = target.n_points;
    if selected_rows.len() != n_mu * n_col {
        return Err(ThcError::PairBlockLength {
            expected: n_mu * n_col,
            actual: selected_rows.len(),
        });
    }
    if weights.len() != n_pts {
        return Err(ThcError::GridWeightCount {
            points: n_pts,
            weights: weights.len(),
        });
    }
    crate::thc::error::validate_quadrature_weights(weights)?;
    target.layout.require_core_orbital()?;
    if let Some(gram) = gram {
        gram.require_context(target.q_index, q, target.layout)?;
    }
    let mut a = vec![Complex64::default(); n_col * n_mu];
    for mu in 0..n_mu {
        for col in 0..n_col {
            a[col * n_mu + mu] = selected_rows[mu * n_col + col];
        }
    }
    let mut b = vec![Complex64::default(); n_col * n_pts];
    for p in 0..n_pts {
        let scale = if weight_target {
            weights[p].sqrt()
        } else {
            1.0
        };
        for col in 0..n_col {
            b[col * n_pts + p] = target.at(p, col) * scale;
        }
    }
    let x = lstsq(&a, n_col, n_mu, &b, n_pts)?;
    let mut zeta = vec![Complex64::default(); n_pts * n_mu];
    for mu in 0..n_mu {
        for p in 0..n_pts {
            let value = x[mu * n_pts + p];
            zeta[p * n_mu + mu] = if weight_target {
                let denom = weights[p].sqrt();
                if denom > 0.0 {
                    value / denom
                } else {
                    Complex64::default()
                }
            } else {
                value
            };
        }
    }
    let reconstructed = reconstruct_pairs(selected_rows, n_mu, n_col, &zeta, n_pts);
    let layout = target.layout;
    let l2_all = residual_pair(target, &reconstructed, weights, |_| true)?;
    let l2_core = if layout.core_orbital.is_some() {
        Some(residual_pair(target, &reconstructed, weights, |col| {
            layout.is_core(col)
        })?)
    } else {
        None
    };
    let l2_valence = if layout.core_orbital.is_some() {
        Some(residual_pair(target, &reconstructed, weights, |col| {
            layout.is_valence(col)
        })?)
    } else {
        None
    };
    let coulomb = match gram {
        Some(gram) => Some(coulomb_residual(target, &reconstructed, weights, gram)?),
        None => None,
    };
    Ok(PerQFit {
        q_index: target.q_index,
        q,
        rank: n_mu,
        zeta,
        n_points: n_pts,
        n_mu,
        l2_all,
        l2_core,
        l2_valence,
        coulomb,
    })
}

/// Fit one shared per-q ζ from all rectangular exchange sectors.
///
/// Both `selected_rows` and `targets` are ordered VV, CV, VC, CC. All columns
/// of all sectors enter the same least-squares system in that order. The fit
/// applies no sector reweighting and reports each sector against its complete
/// input block.
pub fn fit_exchange_per_q(
    selected_rows: [&[Complex64]; 4],
    n_mu: usize,
    targets: [&ExchangePairBlock; 4],
    weights: &[f64],
    q: TransferQ,
) -> Result<ExchangePerQFit, ThcError> {
    if n_mu == 0 {
        return Err(ThcError::EmptyRank);
    }
    let expected_spaces = [
        (ExchangeSpace::Valence, ExchangeSpace::Valence),
        (ExchangeSpace::Core, ExchangeSpace::Valence),
        (ExchangeSpace::Valence, ExchangeSpace::Core),
        (ExchangeSpace::Core, ExchangeSpace::Core),
    ];
    let q_index = targets[0].q_index;
    let n_points = targets[0].n_points;
    if weights.len() != n_points {
        return Err(ThcError::GridWeightCount {
            points: n_points,
            weights: weights.len(),
        });
    }
    crate::thc::error::validate_quadrature_weights(weights)?;
    let mut pooled_columns = 0_usize;
    for (position, target) in targets.iter().enumerate() {
        let (expected_occupied, expected_target) = expected_spaces[position];
        if target.layout.occupied_space != expected_occupied
            || target.layout.target_space != expected_target
        {
            return Err(ThcError::ExchangePairBlockSector {
                index: position,
                expected_occupied,
                expected_target,
                actual_occupied: target.layout.occupied_space,
                actual_target: target.layout.target_space,
            });
        }
        if target.q_index != q_index {
            return Err(ThcError::ExchangePairBlockQIndex {
                index: position,
                expected: q_index,
                actual: target.q_index,
            });
        }
        if target.n_points != n_points {
            return Err(ThcError::PairBlockPointCount {
                index: position,
                expected: n_points,
                actual: target.n_points,
            });
        }
        let expected = crate::thc::error::checked_storage_len(&[n_mu, target.n_columns()])?;
        if selected_rows[position].len() != expected {
            return Err(ThcError::PairBlockLength {
                expected,
                actual: selected_rows[position].len(),
            });
        }
        pooled_columns = pooled_columns
            .checked_add(target.n_columns())
            .ok_or_else(|| ThcError::DimensionOverflow {
                dimensions: targets.iter().map(|block| block.n_columns()).collect(),
            })?;
    }
    let n_k = targets[0].layout.n_k;
    let n_valence = targets[0].layout.n_occupied;
    let n_core = targets[1].layout.n_occupied;
    if n_valence == 0 {
        return Err(ThcError::EmptyExchangeSpace(ExchangeSpace::Valence));
    }
    if n_core == 0 {
        return Err(ThcError::EmptyExchangeSpace(ExchangeSpace::Core));
    }
    let expected_layouts = [
        ExchangePairLayout::new(
            ExchangeSpace::Valence,
            ExchangeSpace::Valence,
            n_k,
            n_valence,
            n_valence,
        ),
        ExchangePairLayout::new(
            ExchangeSpace::Core,
            ExchangeSpace::Valence,
            n_k,
            n_core,
            n_valence,
        ),
        ExchangePairLayout::new(
            ExchangeSpace::Valence,
            ExchangeSpace::Core,
            n_k,
            n_valence,
            n_core,
        ),
        ExchangePairLayout::new(
            ExchangeSpace::Core,
            ExchangeSpace::Core,
            n_k,
            n_core,
            n_core,
        ),
    ];
    for (index, (&target, &expected)) in targets.iter().zip(&expected_layouts).enumerate() {
        if target.layout != expected {
            return Err(ThcError::ExchangePairBlockLayout {
                index,
                expected,
                actual: target.layout,
            });
        }
    }

    let mut a = vec![
        Complex64::default();
        crate::thc::error::checked_storage_len(&[pooled_columns, n_mu])?
    ];
    let mut b = vec![
        Complex64::default();
        crate::thc::error::checked_storage_len(&[pooled_columns, n_points])?
    ];
    let mut column_offset = 0;
    for position in 0..4 {
        let target = targets[position];
        let rows = selected_rows[position];
        for column in 0..target.n_columns() {
            for mu in 0..n_mu {
                a[(column_offset + column) * n_mu + mu] = rows[mu * target.n_columns() + column];
            }
            for point in 0..n_points {
                b[(column_offset + column) * n_points + point] = target.at(point, column);
            }
        }
        column_offset += target.n_columns();
    }
    let x = lstsq(&a, pooled_columns, n_mu, &b, n_points)?;
    let mut zeta =
        vec![Complex64::default(); crate::thc::error::checked_storage_len(&[n_points, n_mu])?];
    for mu in 0..n_mu {
        for point in 0..n_points {
            zeta[point * n_mu + mu] = x[mu * n_points + point];
        }
    }
    let reconstructed: [Vec<Complex64>; 4] = std::array::from_fn(|position| {
        reconstruct_pairs(
            selected_rows[position],
            n_mu,
            targets[position].n_columns(),
            &zeta,
            n_points,
        )
    });
    let residuals = ExchangeSectorResiduals {
        vv: exchange_residual(targets[0], &reconstructed[0], weights)?,
        cv: exchange_residual(targets[1], &reconstructed[1], weights)?,
        vc: exchange_residual(targets[2], &reconstructed[2], weights)?,
        cc: exchange_residual(targets[3], &reconstructed[3], weights)?,
    };
    Ok(ExchangePerQFit {
        q_index,
        q,
        rank: n_mu,
        zeta,
        n_points,
        n_mu,
        residuals,
    })
}

fn exchange_residual(
    exact: &ExchangePairBlock,
    reconstructed: &[Complex64],
    weights: &[f64],
) -> Result<WeightedResidual, ThcError> {
    if reconstructed.len() != exact.values().len() {
        return Err(ThcError::PairBlockLength {
            expected: exact.values().len(),
            actual: reconstructed.len(),
        });
    }
    let n_columns = exact.n_columns();
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut column_numerator = vec![0.0; n_columns];
    let mut column_denominator = vec![0.0; n_columns];
    for point in 0..exact.n_points {
        for column in 0..n_columns {
            let reference = exact.at(point, column);
            let difference = reference - reconstructed[point * n_columns + column];
            numerator += weights[point] * difference.norm_sqr();
            denominator += weights[point] * reference.norm_sqr();
            column_numerator[column] += weights[point] * difference.norm_sqr();
            column_denominator[column] += weights[point] * reference.norm_sqr();
        }
    }
    let frobenius = if denominator > 0.0 {
        (numerator / denominator).sqrt()
    } else {
        0.0
    };
    let scale = column_denominator.iter().copied().fold(0.0_f64, f64::max);
    let floor = f64::EPSILON * scale.max(1.0);
    let mut column_max = 0.0_f64;
    for column in 0..n_columns {
        if column_denominator[column] > floor {
            column_max =
                column_max.max((column_numerator[column] / column_denominator[column]).sqrt());
        }
    }
    Ok(WeightedResidual {
        frobenius,
        column_max,
    })
}

fn residual_pair(
    exact: &PairBlock,
    reconstructed: &[Complex64],
    weights: &[f64],
    mask: impl Fn(usize) -> bool,
) -> Result<WeightedResidual, ThcError> {
    let (frobenius, column_max) = weighted_residual(exact, reconstructed, weights, mask)?;
    Ok(WeightedResidual {
        frobenius,
        column_max,
    })
}

fn coulomb_residual(
    exact: &PairBlock,
    reconstructed: &[Complex64],
    weights: &[f64],
    gram: &InjectedCoulombGram,
) -> Result<WeightedResidual, ThcError> {
    let n = exact.n_columns();
    let sqrt_g = hermitian_sqrt(gram.data(), n)?;
    let whitened_exact = whiten(exact.values(), exact.n_points, n, &sqrt_g);
    let whitened_rec = whiten(reconstructed, exact.n_points, n, &sqrt_g);
    let exact_block = PairBlock::new(
        exact.q_index,
        exact.n_points,
        PairColumnLayout::new(
            exact.layout.n_k,
            exact.layout.n_orb,
            exact.layout.core_orbital,
        ),
        whitened_exact,
    )?;
    residual_pair(&exact_block, &whitened_rec, weights, |_| true)
}

fn whiten(matrix: &[Complex64], rows: usize, cols: usize, sqrt_g: &[Complex64]) -> Vec<Complex64> {
    matmul(matrix, rows, cols, sqrt_g, cols)
}

/// Worst finite-q L2 residual among `reports`, skipping Gamma.
pub fn worst_finite_q(reports: &[PerQFit], is_gamma: impl Fn(usize) -> bool) -> Option<&PerQFit> {
    reports
        .iter()
        .filter(|report| !is_gamma(report.q_index))
        .max_by(|a, b| a.l2_all.frobenius.total_cmp(&b.l2_all.frobenius))
}

/// Worst finite-q Coulomb residual, independent of the L2-worst $q$.
pub fn worst_finite_q_coulomb(
    reports: &[PerQFit],
    is_gamma: impl Fn(usize) -> bool,
) -> Option<&PerQFit> {
    reports
        .iter()
        .filter(|report| !is_gamma(report.q_index) && report.coulomb.is_some())
        .max_by(|left, right| coulomb_frobenius(left).total_cmp(&coulomb_frobenius(right)))
}

fn coulomb_frobenius(fit: &PerQFit) -> f64 {
    fit.coulomb
        .map_or(f64::NEG_INFINITY, |value| value.frobenius)
}

/// Gamma-point report.
pub fn gamma_report(reports: &[PerQFit], is_gamma: impl Fn(usize) -> bool) -> Option<&PerQFit> {
    reports.iter().find(|report| is_gamma(report.q_index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::InverseBohr;

    fn report(q_index: usize, l2: f64, coulomb: f64) -> PerQFit {
        PerQFit {
            q_index,
            q: TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap(),
            rank: 1,
            zeta: Vec::new(),
            n_points: 1,
            n_mu: 1,
            l2_all: WeightedResidual {
                frobenius: l2,
                column_max: l2,
            },
            l2_core: None,
            l2_valence: None,
            coulomb: Some(WeightedResidual {
                frobenius: coulomb,
                column_max: coulomb,
            }),
        }
    }

    #[test]
    fn worst_coulomb_q_is_independent_of_worst_l2_q() {
        let reports = [
            report(0, 0.01, 0.01),
            report(1, 0.90, 0.10),
            report(2, 0.20, 0.80),
        ];
        let worst_l2 = worst_finite_q(&reports, |index| index == 0).unwrap();
        let worst_c = worst_finite_q_coulomb(&reports, |index| index == 0).unwrap();
        assert_eq!(worst_l2.q_index, 1);
        assert_eq!(worst_c.q_index, 2);
        assert!((worst_c.coulomb.unwrap().frobenius - 0.80).abs() < 1.0e-15);
    }
}
