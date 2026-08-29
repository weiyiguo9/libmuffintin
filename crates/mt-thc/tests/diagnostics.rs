//! Error-contract regressions for the THC public boundary.

use muffintin_auxiliary_ir::{AuxiliaryIrError, InterpolationRegion, PairColumnLayout, TransferQ};
use muffintin_core::InverseBohr;
use muffintin_thc::{
    BlochOrbitals, CoulombGramSet, GridPath, InjectedCoulombGram, KMesh, L2Engine, PairBlock,
    RankPolicy, SelectionRequest, SelectorStrategy, ThcError, UniformShift, select_points,
};
use num_complex::Complex64;

fn gamma() -> TransferQ {
    TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap()
}

fn layout() -> PairColumnLayout {
    PairColumnLayout::new(1, 2, None)
}

fn request(rank: RankPolicy) -> SelectionRequest {
    request_with_engine(rank, L2Engine::FullColumnPivotedQr)
}

fn request_with_engine(rank: RankPolicy, engine: L2Engine) -> SelectionRequest {
    SelectionRequest {
        strategy: SelectorStrategy::AllQL2,
        rank,
        seed: 7,
        pool_factor: 2,
        engine,
        grid_path: GridPath::Uniform {
            divisions: 2,
            shift: UniformShift::Half,
        },
    }
}

fn zero_orbitals(n_points: usize, n_k: usize, n_orb: usize) -> BlochOrbitals {
    BlochOrbitals::new(
        n_points,
        n_k,
        n_orb,
        vec![Complex64::default(); n_points * n_k * n_orb],
    )
    .unwrap()
}

#[test]
fn threshold_on_all_zero_pairs_is_degenerate_not_full_rank() {
    let mesh = KMesh::gamma_centred([2, 1, 1], 6.0).unwrap();
    let n_points = 4;
    let orbitals = zero_orbitals(n_points, mesh.len(), 1);
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let weights = vec![1.0; n_points];
    let regions = vec![InterpolationRegion::Uniform; n_points];
    for engine in [L2Engine::FullColumnPivotedQr, L2Engine::FullPivotedCholesky] {
        let error = select_points(
            &orbitals,
            &points,
            &weights,
            &regions,
            &mesh,
            &request_with_engine(
                RankPolicy::Threshold {
                    thresh: 1.0e-6,
                    n_max: n_points,
                },
                engine,
            ),
            None,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(error, ThcError::DegenerateRank),
            "zero pair matrix must be degenerate for {engine:?}, got {error:?}"
        );
    }
}

#[test]
fn threshold_keeps_leading_nondegenerate_rank() {
    let mesh = KMesh::gamma_centred([1, 1, 1], 6.0).unwrap();
    let n_points = 4;
    let orbitals =
        BlochOrbitals::new(n_points, 1, 1, vec![Complex64::new(1.0, 0.0); n_points]).unwrap();
    let points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let weights = vec![1.0; n_points];
    let regions = vec![InterpolationRegion::Uniform; n_points];
    let mut selections = Vec::new();
    for engine in [L2Engine::FullColumnPivotedQr, L2Engine::FullPivotedCholesky] {
        let selection = select_points(
            &orbitals,
            &points,
            &weights,
            &regions,
            &mesh,
            &request_with_engine(
                RankPolicy::Threshold {
                    thresh: 1.0e-8,
                    n_max: n_points,
                },
                engine,
            ),
            None,
            None,
        )
        .unwrap();
        assert_eq!(selection.pivots.len(), 1);
        assert_eq!(selection.provenance.n_mu, 1);
        assert_eq!(selection.provenance.engine, engine);
        selections.push(selection);
    }
    assert_eq!(selections[0].pivots, selections[1].pivots);
    assert_eq!(selections[0].points, selections[1].points);
}

#[test]
fn relative_threshold_has_the_same_amplitude_scale_for_full_engines() {
    let mesh = KMesh::gamma_centred([1, 1, 1], 6.0).unwrap();
    let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let orbitals = BlochOrbitals::new(
        points.len(),
        1,
        3,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::default(),
            Complex64::default(),
            Complex64::default(),
            Complex64::new(0.1, 0.0),
            Complex64::default(),
            Complex64::default(),
            Complex64::default(),
            Complex64::new(0.01, 0.0),
        ],
    )
    .unwrap();
    let weights = vec![1.0; points.len()];
    let regions = vec![InterpolationRegion::Uniform; points.len()];
    for engine in [L2Engine::FullColumnPivotedQr, L2Engine::FullPivotedCholesky] {
        let selection = select_points(
            &orbitals,
            &points,
            &weights,
            &regions,
            &mesh,
            &request_with_engine(
                RankPolicy::Threshold {
                    thresh: 5.0e-3,
                    n_max: points.len(),
                },
                engine,
            ),
            None,
            None,
        )
        .unwrap();
        assert_eq!(selection.pivots, vec![0, 1], "engine={engine:?}");
    }
}

#[test]
fn anisotropic_kminus_roundtrip_and_umklapp() {
    let mesh = KMesh::gamma_centred([3, 2, 4], 5.0).unwrap();
    assert_eq!(mesh.len(), 24);
    assert_eq!(mesh.divisions(), [3, 2, 4]);
    assert!(!mesh.is_empty());
    for ik in 0..mesh.len() {
        for iq in 0..mesh.len() {
            let (left, shift) = mesh.kminus(ik, iq).unwrap();
            assert!(left < mesh.len());
            let k_ik = mesh.fractional()[ik];
            let k_iq = mesh.fractional()[iq];
            let k_left = mesh.fractional()[left];
            for axis in 0..3 {
                let reconstructed = k_iq[axis] + k_left[axis] + f64::from(shift[axis]);
                assert!(
                    (k_ik[axis] - reconstructed).abs() < 1.0e-12,
                    "kminus({ik},{iq}) axis {axis}: {k_ik:?} vs {reconstructed}"
                );
            }
        }
    }
    let iq = mesh
        .fractional()
        .iter()
        .position(|frac| {
            (frac[1] - 0.5).abs() < 1.0e-12 && frac[0].abs() < 1.0e-12 && frac[2].abs() < 1.0e-12
        })
        .unwrap();
    let (_, shift) = mesh.kminus(0, iq).unwrap();
    assert_eq!(shift, [0, -1, 0]);
    let phase = muffintin_thc::umklapp_phase([0.0, 5.0 / 4.0, 0.0], shift, mesh.lattice_constant());
    assert!((phase + Complex64::i()).norm() < 2.0e-14);
}

#[test]
fn kminus_reports_caller_index_not_a_missing_key() {
    let mesh = KMesh::gamma_centred([2, 2, 1], 6.0).unwrap();
    assert!(matches!(
        mesh.kminus(99, 0),
        Err(ThcError::KMeshIndex {
            index: 99,
            count: 4
        })
    ));
}

#[test]
fn bloch_and_pair_block_overflow_do_not_fabricate_lengths() {
    let overflow = BlochOrbitals::new(usize::MAX, 3, 3, Vec::new()).unwrap_err();
    assert!(
        matches!(
            overflow,
            ThcError::DimensionOverflow { ref dimensions } if dimensions == &[usize::MAX, 3, 3]
        ),
        "{overflow:?}"
    );
    let layout = PairColumnLayout::new(2, 2, None);
    let pair = PairBlock::new(0, usize::MAX, layout, Vec::new()).unwrap_err();
    assert!(
        matches!(pair, ThcError::DimensionOverflow { .. }),
        "{pair:?}"
    );
    let columns = PairColumnLayout::new(usize::MAX, 4, None)
        .n_columns()
        .unwrap_err();
    assert!(matches!(
        columns,
        AuxiliaryIrError::DimensionOverflow { ref dimensions } if dimensions == &[usize::MAX, 4, 4]
    ));
}

#[test]
fn pair_block_length_mismatch_reports_the_true_expected_len() {
    let layout = PairColumnLayout::new(2, 2, None);
    let error = PairBlock::new(0, 2, layout, vec![Complex64::default(); 3]).unwrap_err();
    assert!(matches!(
        error,
        ThcError::PairBlockLength {
            expected: 16,
            actual: 3
        }
    ));
}

#[test]
fn gram_shape_reports_buffer_lengths_not_a_sqrt_guess() {
    let error =
        InjectedCoulombGram::from_dense(0, gamma(), layout(), vec![Complex64::default(); 7])
            .unwrap_err();
    assert!(
        matches!(
            error,
            ThcError::GramShape {
                index: 0,
                expected_len: 16,
                actual_len: 7
            }
        ),
        "{error:?}"
    );
}

#[test]
fn gram_set_get_out_of_range_is_an_index_error() {
    let layout = PairColumnLayout::new(1, 1, None);
    let gram = InjectedCoulombGram::from_dense(0, gamma(), layout, vec![Complex64::new(1.0, 0.0)])
        .unwrap();
    let set = CoulombGramSet::new(vec![gram], 1, layout).unwrap();
    assert!(matches!(
        set.get(3),
        Err(ThcError::KMeshIndex { index: 3, count: 1 })
    ));
    assert!(matches!(
        CoulombGramSet::new(Vec::new(), 1, layout),
        Err(ThcError::MissingCoulombGrams)
    ));
}

#[test]
fn empty_grid_is_not_empty_rank() {
    let mesh = KMesh::gamma_centred([2, 1, 1], 6.0).unwrap();
    let orbitals = zero_orbitals(0, mesh.len(), 1);
    let error = select_points(
        &orbitals,
        &[],
        &[],
        &[],
        &mesh,
        &request(RankPolicy::Exact { n_mu: 4 }),
        None,
        None,
    )
    .unwrap_err();
    assert!(matches!(error, ThcError::EmptyGrid), "{error:?}");
}

#[test]
fn zero_requested_rank_is_empty_rank() {
    let mesh = KMesh::gamma_centred([1, 1, 1], 6.0).unwrap();
    let orbitals = BlochOrbitals::new(2, 1, 1, vec![Complex64::new(1.0, 0.0); 2]).unwrap();
    let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let weights = vec![1.0; 2];
    let regions = vec![InterpolationRegion::Uniform; 2];
    let error = select_points(
        &orbitals,
        &points,
        &weights,
        &regions,
        &mesh,
        &request(RankPolicy::Exact { n_mu: 0 }),
        None,
        None,
    )
    .unwrap_err();
    assert!(matches!(error, ThcError::EmptyRank), "{error:?}");
    let thresh = select_points(
        &orbitals,
        &points,
        &weights,
        &regions,
        &mesh,
        &request(RankPolicy::Threshold {
            thresh: 1.0e-4,
            n_max: 0,
        }),
        None,
        None,
    )
    .unwrap_err();
    assert!(matches!(thresh, ThcError::EmptyRank), "{thresh:?}");
}

#[test]
fn gamma_centred_rejects_zero_divisions_and_nonpositive_lattice() {
    assert!(matches!(
        KMesh::gamma_centred([2, 0, 2], 6.0),
        Err(ThcError::InvalidKMeshDivisions([2, 0, 2]))
    ));
    assert!(matches!(
        KMesh::gamma_centred([2, 2, 2], 0.0),
        Err(ThcError::InvalidLattice(0.0))
    ));
}

#[test]
fn orbital_count_mismatch_uses_the_checked_length() {
    let error = BlochOrbitals::new(2, 2, 2, vec![Complex64::default(); 3]).unwrap_err();
    assert!(matches!(
        error,
        ThcError::OrbitalCount {
            expected: 8,
            actual: 3
        }
    ));
}

#[test]
fn select_points_reports_orbital_and_grid_length_fields_separately() {
    let mesh = KMesh::gamma_centred([2, 1, 1], 6.0).unwrap();
    let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let weights = vec![1.0; 2];
    let regions = vec![InterpolationRegion::Uniform; 2];
    let short_orbs =
        BlochOrbitals::new(1, mesh.len(), 1, vec![Complex64::new(1.0, 0.0); 2]).unwrap();
    assert!(matches!(
        select_points(
            &short_orbs,
            &points,
            &weights,
            &regions,
            &mesh,
            &request(RankPolicy::Exact { n_mu: 1 }),
            None,
            None,
        ),
        Err(ThcError::OrbitalPointCount {
            orbitals: 1,
            points: 2
        })
    ));
    let wrong_k = BlochOrbitals::new(2, 1, 1, vec![Complex64::new(1.0, 0.0); 2]).unwrap();
    assert!(matches!(
        select_points(
            &wrong_k,
            &points,
            &weights,
            &regions,
            &mesh,
            &request(RankPolicy::Exact { n_mu: 1 }),
            None,
            None,
        ),
        Err(ThcError::OrbitalKCount {
            orbitals: 1,
            mesh: 2
        })
    ));
    let orbitals = BlochOrbitals::new(2, mesh.len(), 1, vec![Complex64::new(1.0, 0.0); 4]).unwrap();
    assert!(matches!(
        select_points(
            &orbitals,
            &points,
            &[1.0],
            &regions,
            &mesh,
            &request(RankPolicy::Exact { n_mu: 1 }),
            None,
            None,
        ),
        Err(ThcError::GridWeightCount {
            points: 2,
            weights: 1
        })
    ));
    assert!(matches!(
        select_points(
            &orbitals,
            &points,
            &weights,
            &[InterpolationRegion::Uniform],
            &mesh,
            &request(RankPolicy::Exact { n_mu: 1 }),
            None,
            None,
        ),
        Err(ThcError::GridRegionCount {
            points: 2,
            regions: 1
        })
    ));
}

#[test]
fn select_and_fit_reject_negative_nonfinite_and_all_zero_weights() {
    let mesh = KMesh::gamma_centred([1, 1, 1], 6.0).unwrap();
    let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let regions = vec![InterpolationRegion::Uniform; 2];
    let orbitals = BlochOrbitals::new(2, 1, 1, vec![Complex64::new(1.0, 0.0); 2]).unwrap();
    assert!(matches!(
        select_points(
            &orbitals,
            &points,
            &[1.0, -0.2],
            &regions,
            &mesh,
            &request(RankPolicy::Exact { n_mu: 1 }),
            None,
            None,
        ),
        Err(ThcError::InvalidWeight { index: 1, .. })
    ));
    assert!(matches!(
        select_points(
            &orbitals,
            &points,
            &[1.0, f64::NAN],
            &regions,
            &mesh,
            &request(RankPolicy::Exact { n_mu: 1 }),
            None,
            None,
        ),
        Err(ThcError::InvalidWeight { index: 1, .. })
    ));
    assert!(matches!(
        select_points(
            &orbitals,
            &points,
            &[0.0, 0.0],
            &regions,
            &mesh,
            &request(RankPolicy::Exact { n_mu: 1 }),
            None,
            None,
        ),
        Err(ThcError::NoPositiveWeight)
    ));
    let zeros_ok = select_points(
        &orbitals,
        &points,
        &[1.0, 0.0],
        &regions,
        &mesh,
        &request(RankPolicy::Exact { n_mu: 1 }),
        None,
        None,
    );
    assert!(zeros_ok.is_ok(), "{zeros_ok:?}");
}

#[test]
fn invalid_core_orbital_is_rejected() {
    let mesh = KMesh::gamma_centred([1, 1, 1], 6.0).unwrap();
    let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let weights = vec![1.0; 2];
    let regions = vec![InterpolationRegion::Uniform; 2];
    let orbitals = BlochOrbitals::new(2, 1, 2, vec![Complex64::new(1.0, 0.0); 4]).unwrap();
    let error = select_points(
        &orbitals,
        &points,
        &weights,
        &regions,
        &mesh,
        &request(RankPolicy::Exact { n_mu: 1 }),
        None,
        Some(2),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ThcError::Product(AuxiliaryIrError::InvalidCoreOrbital { index: 2, n_orb: 2 })
    ));
}

#[test]
fn pivots_from_pair_blocks_rejects_shorter_and_relayouted_later_blocks() {
    let layout = PairColumnLayout::new(1, 1, None);
    let other = PairColumnLayout::new(1, 2, None);
    let first = PairBlock::new(0, 2, layout, vec![Complex64::new(1.0, 0.0); 2]).unwrap();
    let shorter = PairBlock::new(1, 1, layout, vec![Complex64::new(1.0, 0.0)]).unwrap();
    let mixed = PairBlock::new(1, 2, other, vec![Complex64::new(1.0, 0.0); 8]).unwrap();
    let weights = vec![1.0; 2];
    assert!(matches!(
        muffintin_thc::pivots_from_pair_blocks(&[first.clone(), shorter], &weights, 1),
        Err(ThcError::PairBlockPointCount {
            index: 1,
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        muffintin_thc::pivots_from_pair_blocks(&[first, mixed], &weights, 1),
        Err(ThcError::PairBlockLayout { index: 1 })
    ));
}
