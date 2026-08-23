//! Per-q interpolation-vector (ζ) fits and residual diagnostics.

use crate::ThcError;
use crate::gram::InjectedCoulombGram;
use crate::linalg::{hermitian_sqrt, lstsq};
use crate::pair::{PairBlock, PairColumnLayout};
use crate::select::{matmul, reconstruct_pairs, weighted_residual};
use muffintin_auxiliary_ir::TransferQ;
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
    crate::error::validate_quadrature_weights(weights)?;
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
