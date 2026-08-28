//! Finite-cutoff toy Coulomb/ERI/action path from `thc_lapw_end_to_end_test.py`.
//!
//! Three layers:
//! 1. ordinary plumbing (`candidate_eri_action_path_is_exact_for_a_rank_one_interpolation`)
//!    is an algebraic identity at $10^{-12}$. It must not use
//!    `RECORDED_ERI_ACTION_GATE`.
//! 2. ordinary bounded `three_selectors_distinguish_finite_q_action_at_identical_nmu`
//!    compares `q0_l2` / `allq_l2` / `allq_coulomb_pool` at one $N_\mu$ on a
//!    tiny algebraic fixture. Python never implemented the pool, so the
//!    recorded $8\times10^{-2}$ gate is not claimed here.
//! 3. ignored `source_equivalent_python_lapw_fixture` is the only test that
//!    may assert `RECORDED_REFERENCE_GATE` / `RECORDED_ERI_ACTION_GATE`.
//!    Python table values are evidence, not bit-identity targets.

use muffintin_auxiliary_ir::{PairColumnLayout, TransferQ};
use muffintin_core::InverseBohr;
use muffintin_thc::toy::{
    ACTION_VECTOR_COUNT, ACTION_VECTOR_SEED, LAPW_LATTICE, LAPW_NORB, RECORDED_ERI_ACTION_GATE,
    RECORDED_REFERENCE_GATE, ToyEriActionMetrics, ToyFiniteCutoffKernel, ToyGrid,
    approximate_pair_fourier, compare_candidate_eri_action, lapw_bloch_orbitals,
    lapw_composite_grid, lapw_kmesh, lapw_uniform_grid, pair_fourier, reciprocal_vectors,
    relative_gram_frobenius, toy_coulomb_factors, toy_coulomb_gram, values_fourier,
};
use muffintin_thc::{
    BlochOrbitals, CoulombGramSet, GridPath, KMesh, L2Engine, RankPolicy, SelectionRequest,
    SelectorStrategy, ThcError, UmklappGauge, evaluate_pair_block, fit_per_q, select_points,
    umklapp_phase,
};
use num_complex::Complex64;

#[allow(clippy::too_many_arguments)]
fn candidate_eri_action_over_q(
    orbitals: &BlochOrbitals,
    grid: &ToyGrid,
    mesh: &KMesh,
    kernel: &ToyFiniteCutoffKernel,
    reference_ft: &[Vec<Complex64>],
    n_mu: usize,
    strategy: SelectorStrategy,
    grams: Option<&CoulombGramSet>,
    engine: L2Engine,
    core_orbital: Option<usize>,
) -> Result<ToyEriActionMetrics, ThcError> {
    let selection = select_points(
        orbitals,
        &grid.points,
        &grid.weights,
        &grid.regions,
        mesh,
        &SelectionRequest {
            strategy,
            rank: RankPolicy::Exact { n_mu },
            seed: 7,
            pool_factor: 2,
            engine,
            grid_path: GridPath::Composite {
                name: grid.name.clone(),
            },
        },
        grams,
        core_orbital,
    )?;
    let ids: Vec<usize> = selection.points.iter().map(|point| point.id).collect();
    let layout = orbitals.layout(core_orbital);
    let mut worst: Option<ToyEriActionMetrics> = None;
    for (iq, ref_ft) in reference_ft.iter().enumerate() {
        let q = mesh.transfer_q(iq)?;
        let block = evaluate_pair_block(
            orbitals,
            &grid.points,
            mesh,
            iq,
            core_orbital,
            UmklappGauge::Canonical,
        )?;
        let selected_rows = block.selected_rows(&ids)?;
        let fit = fit_per_q(&selected_rows, n_mu, &block, &grid.weights, q, None, false)?;
        let report = compare_candidate_eri_action(
            &fit.zeta,
            fit.n_mu,
            &selected_rows,
            block.n_columns(),
            grid,
            mesh.fractional()[iq],
            kernel,
            ref_ft,
            iq,
            q,
            layout,
        )?;
        worst = Some(match worst {
            Some(acc) => acc.max_with(report),
            None => report,
        });
    }
    Ok(worst.expect("mesh is nonempty"))
}

fn exact_pair_fourier(
    orbitals: &BlochOrbitals,
    grid: &ToyGrid,
    mesh: &KMesh,
    kernel: &ToyFiniteCutoffKernel,
) -> Vec<Vec<Complex64>> {
    let g_cart = kernel.g_cartesian();
    (0..mesh.len())
        .map(|iq| {
            let block = evaluate_pair_block(
                orbitals,
                &grid.points,
                mesh,
                iq,
                None,
                UmklappGauge::Canonical,
            )
            .unwrap();
            pair_fourier(&block, grid, &g_cart, kernel.volume)
        })
        .collect()
}

#[test]
fn synthetic_lapw_umklapp_is_minus_i() {
    let mesh = lapw_kmesh();
    let probe = [0.0, LAPW_LATTICE / 4.0, 0.0];
    let (_, shift) = mesh.kminus(0, 1).unwrap();
    assert_eq!(shift, [0, -1, 0]);
    let phase = umklapp_phase(probe, shift, mesh.lattice_constant());
    assert!((phase + Complex64::i()).norm() < 2.0e-14);
}

/// Exact rank-one interpolation identity. Mechanical $10^{-12}$ only;
/// not the recorded $8\times10^{-2}$ ERI/action gate.
#[test]
fn candidate_eri_action_path_is_exact_for_a_rank_one_interpolation() {
    assert_eq!(ACTION_VECTOR_SEED, 19);
    assert_eq!(ACTION_VECTOR_COUNT, 8);
    let q = TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
    let layout = PairColumnLayout::new(1, 1, None);
    let grid = lapw_uniform_grid(1);
    let kernel = ToyFiniteCutoffKernel {
        g_integer: reciprocal_vectors(1),
        lattice: LAPW_LATTICE,
        volume: LAPW_LATTICE.powi(3),
    };
    let n_g = kernel.g_integer.len();
    let pair = vec![Complex64::new(0.3, -0.1)];
    let zeta = vec![Complex64::new(1.0, 0.0)];
    let rows = pair.clone();
    let g_cart = kernel.g_cartesian();
    let exact_ft = values_fourier(&pair, 1, 1, &grid, &g_cart, kernel.volume).unwrap();
    let zeta_ft = values_fourier(&zeta, 1, 1, &grid, &g_cart, kernel.volume).unwrap();
    let approx_ft = approximate_pair_fourier(&zeta_ft, n_g, 1, &rows, 1).unwrap();
    for (got, want) in approx_ft.iter().zip(&exact_ft) {
        assert!((got - want).norm() < 1.0e-12);
    }
    let metrics = compare_candidate_eri_action(
        &zeta,
        1,
        &rows,
        1,
        &grid,
        [0.0, 0.0, 0.0],
        &kernel,
        &exact_ft,
        0,
        q,
        layout,
    )
    .unwrap();
    assert!(
        metrics.eri_frobenius < 1.0e-12
            && metrics.eri_max_element < 1.0e-12
            && metrics.action < 1.0e-12
            && metrics.pair_fourier < 1.0e-12,
        "rank-one plumbing identity failed: {metrics:?}"
    );
}

#[test]
fn gram_q_index_is_validated_before_whitening() {
    let q = TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
    let layout = PairColumnLayout::new(1, 1, None);
    let gram = toy_coulomb_gram(1, q, layout, &[Complex64::new(1.0, 0.0)], &[4.0], 1.0).unwrap();
    let error = gram.require_context(0, q, layout).unwrap_err();
    assert!(matches!(
        error,
        muffintin_thc::ThcError::GramQIndex {
            expected: 0,
            actual: 1
        }
    ));
    let mut shifted = gram.clone();
    shifted.layout.n_orb = 3;
    assert!(matches!(
        shifted.require_context(1, q, layout),
        Err(muffintin_thc::ThcError::GramColumnOrder(1))
    ));
    assert!(matches!(
        shifted.require_context(1, q, shifted.layout),
        Err(muffintin_thc::ThcError::GramShape { index: 1, .. })
    ));
}

fn injected_grams_from_pair_fourier(
    mesh: &KMesh,
    layout: PairColumnLayout,
    reference_ft: &[Vec<Complex64>],
    kernel: &ToyFiniteCutoffKernel,
) -> CoulombGramSet {
    let mut grams = Vec::new();
    for (iq, ref_ft) in reference_ft.iter().enumerate() {
        let q = mesh.transfer_q(iq).unwrap();
        let factors = toy_coulomb_factors(mesh.fractional()[iq], &kernel.g_integer, kernel.lattice);
        grams.push(toy_coulomb_gram(iq, q, layout, ref_ft, &factors, kernel.volume).unwrap());
    }
    CoulombGramSet::new(grams, mesh.len(), layout).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn finite_q_action(
    orbitals: &BlochOrbitals,
    grid: &ToyGrid,
    mesh: &KMesh,
    kernel: &ToyFiniteCutoffKernel,
    reference_ft: &[Vec<Complex64>],
    n_mu: usize,
    strategy: SelectorStrategy,
    grams: Option<&CoulombGramSet>,
    engine: L2Engine,
) -> (f64, Vec<usize>) {
    let iq = 1;
    let selection = select_points(
        orbitals,
        &grid.points,
        &grid.weights,
        &grid.regions,
        mesh,
        &SelectionRequest {
            strategy,
            rank: RankPolicy::Exact { n_mu },
            seed: 7,
            pool_factor: 2,
            engine,
            grid_path: GridPath::Uniform {
                divisions: grid.len(),
                shift: muffintin_thc::UniformShift::Origin,
            },
        },
        grams,
        None,
    )
    .unwrap();
    let ids: Vec<usize> = selection.points.iter().map(|point| point.id).collect();
    let q = mesh.transfer_q(iq).unwrap();
    let block = evaluate_pair_block(
        orbitals,
        &grid.points,
        mesh,
        iq,
        None,
        UmklappGauge::Canonical,
    )
    .unwrap();
    let selected_rows = block.selected_rows(&ids).unwrap();
    let fit = fit_per_q(&selected_rows, n_mu, &block, &grid.weights, q, None, false).unwrap();
    let action = compare_candidate_eri_action(
        &fit.zeta,
        fit.n_mu,
        &selected_rows,
        block.n_columns(),
        grid,
        mesh.fractional()[iq],
        kernel,
        &reference_ft[iq],
        iq,
        q,
        orbitals.layout(None),
    )
    .unwrap()
    .action;
    (action, selection.pivots)
}

/// Bounded algebraic action comparison at identical $N_\mu$.
///
/// Metrics must be nontrivial (not self-reconstruction). `q0_l2` must miss a
/// finite-$q$ channel that `allq_l2` sees. Coulomb-pool is judged
/// action-protecting vs the all-q L2 baseline under the existing $1.05$
/// convention. Python did not implement the pool, so `RECORDED_ERI_ACTION_GATE`
/// is not claimed here.
#[test]
fn three_selectors_distinguish_finite_q_action_at_identical_nmu() {
    let mesh = KMesh::gamma_centred([2, 1, 1], 6.0).unwrap();
    let mut points = Vec::new();
    for i in 0..8 {
        points.push([6.0 * i as f64 / 8.0, 0.0, 0.0]);
    }
    let n = points.len();
    let grid = ToyGrid {
        name: "finite-q-action".to_owned(),
        points,
        weights: vec![1.0; n],
        regions: vec![muffintin_auxiliary_ir::InterpolationRegion::Uniform; n],
    };
    let mut values = Vec::new();
    for (p, point) in grid.points.iter().enumerate() {
        let x = point[0];
        values.push(Complex64::new(1.0 + 0.05 * x, 0.0));
        values.push(Complex64::new(
            (4.0 * std::f64::consts::PI * x / 6.0).sin(),
            0.0,
        ));
        values.push(Complex64::new(
            1.0 + 0.2 * (2.0 * std::f64::consts::PI * x / 6.0).cos(),
            0.0,
        ));
        values.push(Complex64::new(if p >= 4 { 1.4 } else { 0.05 }, 0.15 * x));
    }
    let orbitals = BlochOrbitals::new(n, mesh.len(), 2, values).unwrap();
    let kernel = ToyFiniteCutoffKernel {
        g_integer: reciprocal_vectors(8),
        lattice: 6.0,
        volume: 216.0,
    };
    let reference_ft = exact_pair_fourier(&orbitals, &grid, &mesh, &kernel);
    let layout = orbitals.layout(None);
    let grams = injected_grams_from_pair_fourier(&mesh, layout, &reference_ft, &kernel);
    let n_mu = 2;
    let engine = L2Engine::StructuredSketch { rows: 12 };
    let (q0, q0_pivots) = finite_q_action(
        &orbitals,
        &grid,
        &mesh,
        &kernel,
        &reference_ft,
        n_mu,
        SelectorStrategy::Q0L2,
        None,
        engine,
    );
    let (allq, allq_pivots) = finite_q_action(
        &orbitals,
        &grid,
        &mesh,
        &kernel,
        &reference_ft,
        n_mu,
        SelectorStrategy::AllQL2,
        None,
        engine,
    );
    let (pool, pool_pivots) = finite_q_action(
        &orbitals,
        &grid,
        &mesh,
        &kernel,
        &reference_ft,
        n_mu,
        SelectorStrategy::AllQCoulombPool,
        Some(&grams),
        engine,
    );
    eprintln!(
        "three-selector finite-q action Nmu={n_mu}: q0={q0:.6e} allq={allq:.6e} pool={pool:.6e} (1.05 convention; not the recorded 8e-2 gate)"
    );
    assert!(
        q0 > 1.0e-3 && allq > 1.0e-3 && pool > 1.0e-3,
        "action metrics must be nontrivial: q0={q0} allq={allq} pool={pool}"
    );
    assert!(
        q0 > allq * 1.05,
        "q0_l2 must miss a finite-q/action channel relative to allq_l2: q0={q0} pivots={q0_pivots:?} allq={allq} pivots={allq_pivots:?}"
    );
    let _ = pool_pivots.len();
    assert!(
        pool <= allq * 1.05,
        "allq_coulomb_pool must protect action vs allq_l2 within 1.05: pool={pool} allq={allq}"
    );
}

/// Source-equivalent Python fixture from `thc_lapw_end_to_end_test.py`.
///
/// Ordinary workspace tests skip this. Run:
/// `cargo test --release -p libmuffintin-thc --test end_to_end_smoke source_equivalent_python_lapw_fixture --offline -- --ignored --exact --nocapture`
#[ignore = "source-equivalent Python 26x86+18^3 fixture; run with --ignored"]
#[test]
fn source_equivalent_python_lapw_fixture() {
    let mesh = lapw_kmesh();
    let reference = lapw_composite_grid("reference 38x110 + 20^3", 38, 110, 20);
    let medium = lapw_composite_grid("medium reference 30x86 + 18^3", 30, 86, 18);
    let candidate = lapw_composite_grid("adaptive 26x86 + 18^3", 26, 86, 18);
    let ref_orbs = lapw_bloch_orbitals(&reference, &mesh, None).unwrap();
    let medium_orbs = lapw_bloch_orbitals(&medium, &mesh, Some(&reference)).unwrap();
    let cand_orbs = lapw_bloch_orbitals(&candidate, &mesh, Some(&reference)).unwrap();
    let kernel = ToyFiniteCutoffKernel {
        g_integer: reciprocal_vectors(12),
        lattice: LAPW_LATTICE,
        volume: LAPW_LATTICE.powi(3),
    };
    let g_cart = kernel.g_cartesian();
    let layout = cand_orbs.layout(None);
    let mut reference_convergence = 0.0_f64;
    let mut reference_ft = Vec::with_capacity(mesh.len());
    for iq in 0..mesh.len() {
        let q = mesh.transfer_q(iq).unwrap();
        let ref_block = evaluate_pair_block(
            &ref_orbs,
            &reference.points,
            &mesh,
            iq,
            None,
            UmklappGauge::Canonical,
        )
        .unwrap();
        let med_block = evaluate_pair_block(
            &medium_orbs,
            &medium.points,
            &mesh,
            iq,
            None,
            UmklappGauge::Canonical,
        )
        .unwrap();
        let factors = toy_coulomb_factors(mesh.fractional()[iq], &kernel.g_integer, kernel.lattice);
        let ref_ft = pair_fourier(&ref_block, &reference, &g_cart, kernel.volume);
        let med_ft = pair_fourier(&med_block, &medium, &g_cart, kernel.volume);
        let ref_gram = toy_coulomb_gram(iq, q, layout, &ref_ft, &factors, kernel.volume).unwrap();
        let med_gram = toy_coulomb_gram(iq, q, layout, &med_ft, &factors, kernel.volume).unwrap();
        reference_convergence =
            reference_convergence.max(relative_gram_frobenius(&ref_gram, &med_gram).unwrap());
        reference_ft.push(ref_ft);
    }
    eprintln!(
        "source-equivalent independent-reference ERI: {reference_convergence:.6e} (Python 2.498e-2, gate 5e-2); npts ref={} med={} cand={}",
        reference.len(),
        medium.len(),
        candidate.len()
    );
    assert!(
        reference_convergence <= RECORDED_REFERENCE_GATE,
        "independent-reference ERI {reference_convergence:.6e} exceeded 5e-2"
    );

    let n_mu = 16 * LAPW_NORB;
    for engine in [L2Engine::FullColumnPivotedQr, L2Engine::FullPivotedCholesky] {
        let metrics = candidate_eri_action_over_q(
            &cand_orbs,
            &candidate,
            &mesh,
            &kernel,
            &reference_ft,
            n_mu,
            SelectorStrategy::AllQL2,
            None,
            engine,
            None,
        )
        .unwrap();
        eprintln!(
            "source-equivalent fine candidate engine={engine:?} Nmu={n_mu}: pair-G={:.6e} ERI-F={:.6e} ERI-max={:.6e} action={:.6e} (Python 2.362e-2 / 4.932e-2 / 4.560e-2 / 6.230e-2, gate 8e-2)",
            metrics.pair_fourier, metrics.eri_frobenius, metrics.eri_max_element, metrics.action
        );
        assert!(
            metrics.eri_frobenius <= RECORDED_ERI_ACTION_GATE,
            "fine {engine:?} ERI-F {:.6e} exceeded 8e-2",
            metrics.eri_frobenius
        );
        assert!(
            metrics.eri_max_element <= RECORDED_ERI_ACTION_GATE,
            "fine {engine:?} ERI-max {:.6e} exceeded 8e-2",
            metrics.eri_max_element
        );
        assert!(
            metrics.action <= RECORDED_ERI_ACTION_GATE,
            "fine {engine:?} action {:.6e} exceeded 8e-2",
            metrics.action
        );
    }
}
