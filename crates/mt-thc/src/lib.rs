//! Toy k-point ISDF/THC over the product-space IR.
//!
//! This crate selects a q-independent interpolation-point set on a finite
//! periodic toy basis, fits per-q interpolation vectors, and emits
//! representation-neutral [`muffintin_auxiliary_ir::CompiledAuxiliaryBasis`]
//! interpolation-point payloads and [`muffintin_auxiliary_ir::PairVertex`]
//! Bloch pair vertices.
//!
//! It does **not** assemble Weinert or SPEX $V^q$. Coulomb-aware ranking
//! consumes [`gram::InjectedCoulombGram`]. The recorded Python numbers in
//! [`toy`] are finite-cutoff candidate-oracle evidence, not a real-material
//! accuracy claim.

#![forbid(unsafe_code)]

mod error;
mod fit;
mod gram;
mod kmesh;
mod linalg;
mod pair;
mod run;
mod select;
pub mod toy;

pub use error::{ThcError, checked_storage_len, validate_quadrature_weights};
pub use fit::{
    PerQFit, WeightedResidual, fit_per_q, gamma_report, worst_finite_q, worst_finite_q_coulomb,
};
pub use gram::{CoulombGramSet, GRAM_HERMITIAN_TOLERANCE, GRAM_PSD_TOLERANCE, InjectedCoulombGram};
pub use kmesh::{KMesh, umklapp_phase};
pub use pair::{BlochOrbitals, PairBlock, PairColumnLayout, UmklappGauge, evaluate_pair_block};
pub use run::{
    StrategyDiagnostics, ThcResult, bloch_pair_vertices, compare_strategies,
    interpolation_auxiliary, run_thc,
};
pub use select::{
    DEFAULT_POOL_FACTOR, DEFAULT_SELECTOR, DEFAULT_SKETCH_ROWS, GridPath, HEADLINE_SEED, L2Engine,
    RANDOM_SHIFT_SEED, RankPolicy, STRATEGY_SEEDS, Selection, SelectionProvenance,
    SelectionRequest, SelectorStrategy, UniformShift, pivots_from_pair_blocks, select_points,
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
        let (pivots, _) = crate::linalg::column_pivots(&matrix, 2, 3).unwrap();
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
        let (qr_pivots, qr_diag) = crate::linalg::column_pivots(&matrix, 4, 3).unwrap();
        let (chol_pivots, chol_diag) =
            crate::linalg::pivoted_cholesky_pivots(&matrix, 4, 3, 3).unwrap();
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
        let x = crate::linalg::lstsq(&a, 3, 2, &b, 2).unwrap();
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
        let x = crate::linalg::lstsq(&a, 3, 2, &b, 1).unwrap();
        assert!(
            x.iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        assert!((x[0] - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
        assert!(x[1].norm() < 1.0e-12);
    }
}
