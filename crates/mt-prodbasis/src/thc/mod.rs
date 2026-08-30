//! k-point ISDF/THC kernels over the product-space IR.
//!
//! Callers evaluate orbital-pair blocks on a parent grid, select
//! interpolation points with a full weighted L2 engine, fit per-q
//! interpolation vectors, and emit representation-neutral
//! [`crate::CompiledAuxiliaryBasis`] interpolation-point payloads and
//! [`crate::PairVertex`] Bloch pair vertices.
//!
//! This module does **not** assemble Weinert or SPEX $V^q$. Coulomb-aware
//! residual reporting consumes an injected [`InjectedCoulombGram`]. The toy
//! grids, toy k-mesh, toy Bloch orbitals, and selector-strategy sweep
//! harness live in the shared test fixture, not here.

#![forbid(unsafe_code)]

mod error;
mod fit;
mod gram;
pub mod linalg;
mod pair;
mod run;
mod select;

pub use error::{ThcError, checked_storage_len, validate_quadrature_weights};
pub use fit::{
    PerQFit, WeightedResidual, fit_per_q, gamma_report, worst_finite_q, worst_finite_q_coulomb,
};
pub use gram::{CoulombGramSet, GRAM_HERMITIAN_TOLERANCE, GRAM_PSD_TOLERANCE, InjectedCoulombGram};
pub use pair::PairBlock;
pub use run::{
    StrategyDiagnostics, ThcResult, bloch_pair_vertices, fit_allq_l2_pair_blocks,
    interpolation_auxiliary,
};
pub use select::{
    DEFAULT_SELECTOR, GridPath, L2Engine, RankPolicy, Selection, SelectionProvenance,
    SelectorStrategy, UniformShift, cholesky_pivots_from_pair_blocks, interpolation_points,
    matmul, pivots_from_pair_blocks, reconstruct_pairs, truncate_rank, weighted_residual,
};

#[cfg(test)]
mod linalg_smoke {
    use num_complex::Complex64;

    #[test]
    fn column_pivots_prefer_the_large_column() {
        let matrix = [
            Complex64::new(1.0, 0.0),
            Complex64::new(100.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(50.0, 0.0),
        ];
        let (pivots, _) = crate::thc::linalg::column_pivots(&matrix, 2, 3).unwrap();
        assert_eq!(pivots[0], 1);
    }

    #[test]
    fn pivoted_cholesky_matches_qrcp_without_pivot_ties() {
        let matrix = [
            Complex64::new(4.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.2),
            Complex64::new(0.1, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.4),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.2, 0.0),
            Complex64::new(0.1, 0.0),
            Complex64::new(0.8, 0.0),
        ];
        let (qr_pivots, qr_diag) = crate::thc::linalg::column_pivots(&matrix, 4, 3).unwrap();
        let (chol_pivots, chol_diag) =
            crate::thc::linalg::pivoted_cholesky_pivots(&matrix, 4, 3, 3).unwrap();
        assert_eq!(chol_pivots, qr_pivots);
        for (chol, qr) in chol_diag.iter().zip(&qr_diag) {
            assert!((chol - qr).abs() < 1.0e-12, "Cholesky={chol}, QRCP={qr}");
        }
        let relative_rank = |diagonal: &[f64], threshold: f64| {
            diagonal
                .iter()
                .take_while(|&&value| value >= threshold * diagonal[0])
                .count()
        };
        assert_eq!(relative_rank(&chol_diag, 0.7), 2);
        assert_eq!(relative_rank(&chol_diag, 0.7), relative_rank(&qr_diag, 0.7));
    }

    #[test]
    fn lstsq_recovers_a_tall_full_rank_system() {
        let a = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let b = [
            Complex64::new(3.0, -1.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-4.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let x = crate::thc::linalg::lstsq(&a, 3, 2, &b, 2).unwrap();
        assert!((x[0] - Complex64::new(3.0, -1.0)).norm() < 1.0e-12);
        assert!((x[1] - Complex64::new(0.0, 2.0)).norm() < 1.0e-12);
        assert!((x[2] - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
        assert!((x[3] - Complex64::new(-4.0, 0.0)).norm() < 1.0e-12);
    }

    #[test]
    fn lstsq_truncates_tiny_singular_values() {
        let eps = 1.0e-20;
        let a = [
            Complex64::new(1.0, 0.0),
            Complex64::new(eps, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let b = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let x = crate::thc::linalg::lstsq(&a, 3, 2, &b, 1).unwrap();
        assert!(
            x.iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        assert!((x[0] - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
        assert!(x[1].norm() < 1.0e-12);
    }
}
