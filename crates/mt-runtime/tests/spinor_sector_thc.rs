//! Core-aware spinor THC and full MPB quadratic-oracle tests.

use muffintin::{
    CheckpointPhysics, RankPolicy, SpinorMpbSelection, SpinorMpbSpec, SpinorSectorThcError,
    SpinorThcSpec, ThcCandidates, ThcEngine, build_spinor_exchange_mpb, build_spinor_mpb,
    build_spinor_sector_thc, compare_spinor_sector_thc_mpb,
};
use muffintin_prodbasis::mpb::DEFAULT_TOLERANCE;
use muffintin_prodbasis::{ExchangePairLayout, ExchangeSpace, InterpolationRegion, OrbitalPair};

#[path = "spinor_hydrogen.rs"]
mod spinor_hydrogen;

use spinor_hydrogen::{
    core_sidecar, coulomb_spec, exchange_mpb_spec, hydrogen_spinor_checkpoint, parent_grid,
    spinor_config,
};

fn full_rank_spec(n_mu: usize) -> SpinorThcSpec {
    SpinorThcSpec {
        rank: RankPolicy::Exact { n_mu },
        candidates: ThcCandidates::All,
        engine: ThcEngine::FullColumnPivotedQr,
    }
}

fn full_vv_spec(input: &muffintin::SpinorProductInput) -> SpinorMpbSpec {
    let n_k = input.orbitals.k_fractional.len();
    let n_valence = input.orbitals.band_window.count;
    SpinorMpbSpec {
        product_l_max: 2,
        product_g_max: spinor_hydrogen::coulomb_spec().projection.pw_cutoff,
        overlap_tolerance: DEFAULT_TOLERANCE,
        selections: (0..n_k)
            .flat_map(|k| {
                (0..n_valence).flat_map(move |occupied| {
                    (0..n_valence).map(move |target| SpinorMpbSelection {
                        k,
                        left_band: occupied,
                        right_band: target,
                    })
                })
            })
            .collect(),
    }
}

fn expected_pair(layout: ExchangePairLayout, column: usize) -> OrbitalPair {
    let (k_index, occupied, target) = layout.decode(column).unwrap();
    OrbitalPair::Exchange {
        k_index,
        occupied_space: layout.occupied_space,
        occupied,
        target_space: layout.target_space,
        target,
    }
}

#[test]
fn full_positive_grid_reconstructs_all_sectors_and_keeps_core_vertices_mt_only() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let plain = physics
        .spinor_product_input(&spinor_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let low = plain
        .clone()
        .with_core_sidecars(&[core_sidecar(&plain, 0.25)])
        .unwrap();
    let high = plain
        .clone()
        .with_core_sidecars(&[core_sidecar(&plain, 0.75)])
        .unwrap();
    let grid = parent_grid(&low);
    let n_positive = grid
        .points()
        .iter()
        .filter(|point| point.weight > 0.0)
        .count();
    let low_result = build_spinor_sector_thc(
        std::slice::from_ref(&low),
        &grid,
        &full_rank_spec(n_positive),
    )
    .unwrap();
    let high_result = build_spinor_sector_thc(
        std::slice::from_ref(&high),
        &grid,
        &full_rank_spec(n_positive),
    )
    .unwrap();

    assert_eq!(
        low_result, high_result,
        "occupations must not enter vertices"
    );
    assert!(low_result.records_match_parent_grid());
    assert_eq!(low_result.records.len(), 1);
    assert_eq!(low_result.effective_rank, n_positive);
    let scaling = low_result.rank_scaling;
    let n_valence = low.orbitals.band_window.count;
    let n_core = low.core.orbitals.len();
    assert_eq!(scaling.n_k, 1);
    assert_eq!(scaling.n_valence, n_valence);
    assert_eq!(scaling.n_core, n_core);
    assert_eq!(scaling.n_candidates, n_positive);
    assert_eq!(scaling.effective_rank, n_positive);
    assert_eq!(scaling.vv_columns_per_q, n_valence * n_valence);
    assert_eq!(scaling.cv_columns_per_q, n_core * n_valence);
    assert_eq!(scaling.vc_columns_per_q, n_valence * n_core);
    assert_eq!(scaling.cc_columns_per_q, n_core * n_core);
    assert_eq!(
        scaling.pooled_columns_per_q,
        (n_valence + n_core) * (n_valence + n_core)
    );
    assert_eq!(scaling.selector_rows, scaling.pooled_columns_per_q);

    for residual in [
        low_result.diagnostics.vv,
        low_result.diagnostics.cv,
        low_result.diagnostics.vc,
        low_result.diagnostics.cc,
    ] {
        assert!(residual.frobenius < 1.0e-8, "{residual:?}");
        assert!(residual.column_max < 1.0e-8, "{residual:?}");
    }

    let record = &low_result.records[0];
    let sectors = [&record.vv, &record.cv, &record.vc, &record.cc];
    let expected_spaces = [
        (ExchangeSpace::Valence, ExchangeSpace::Valence),
        (ExchangeSpace::Core, ExchangeSpace::Valence),
        (ExchangeSpace::Valence, ExchangeSpace::Core),
        (ExchangeSpace::Core, ExchangeSpace::Core),
    ];
    for (sector, spaces) in sectors.into_iter().zip(expected_spaces) {
        assert_eq!(
            (sector.layout.occupied_space, sector.layout.target_space),
            spaces
        );
        assert_eq!(sector.vertices.len(), sector.layout.n_columns().unwrap());
        for (column, vertex) in sector.vertices.iter().enumerate() {
            assert_eq!(vertex.pair(), expected_pair(sector.layout, column));
            assert_eq!(vertex.layout(), &record.auxiliary.layout());
        }
    }
    let points = record.auxiliary.require_interpolation_points().unwrap();
    for sector in [&record.cv, &record.vc, &record.cc] {
        for vertex in &sector.vertices {
            for (mu, point) in points.iter().enumerate() {
                if point.region == InterpolationRegion::Interstitial {
                    assert_eq!(vertex.coefficients()[mu].norm(), 0.0);
                }
            }
        }
    }
    assert!(
        [&record.cv, &record.vc, &record.cc]
            .into_iter()
            .flat_map(|sector| &sector.vertices)
            .flat_map(|vertex| vertex.coefficients())
            .any(|coefficient| coefficient.norm() > 1.0e-12)
    );
}

#[test]
fn full_mpb_oracle_aggregates_every_quadratic_by_exact_sector() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let plain = physics
        .spinor_product_input(&spinor_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let input = plain
        .clone()
        .with_core_sidecars(&[core_sidecar(&plain, 0.5)])
        .unwrap();
    let grid = parent_grid(&input);
    let n_positive = grid
        .points()
        .iter()
        .filter(|point| point.weight > 0.0)
        .count();
    let thc = build_spinor_sector_thc(
        std::slice::from_ref(&input),
        &grid,
        &full_rank_spec(n_positive),
    )
    .unwrap();
    let vv = build_spinor_mpb(&input, &full_vv_spec(&input)).unwrap();
    let core = build_spinor_exchange_mpb(&input, &exchange_mpb_spec()).unwrap();
    let comparison = compare_spinor_sector_thc_mpb(
        std::slice::from_ref(&input),
        &thc,
        std::slice::from_ref(&vv),
        std::slice::from_ref(&core),
        &coulomb_spec(),
    )
    .unwrap();

    for (sector, expected) in [
        (&comparison.vv, thc.rank_scaling.vv_columns_per_q),
        (&comparison.cv, thc.rank_scaling.cv_columns_per_q),
        (&comparison.vc, thc.rank_scaling.vc_columns_per_q),
        (&comparison.cc, thc.rank_scaling.cc_columns_per_q),
    ] {
        assert_eq!(sector.pairs.len(), expected);
        assert!(sector.pairs.iter().all(|pair| {
            pair.mpb_quadratic.re.is_finite()
                && pair.mpb_quadratic.im.is_finite()
                && pair.thc_quadratic.re.is_finite()
                && pair.thc_quadratic.im.is_finite()
                && pair.absolute.is_finite()
                && pair.relative.is_finite()
        }));
        let max_absolute = sector
            .pairs
            .iter()
            .map(|pair| pair.absolute)
            .fold(0.0_f64, f64::max);
        let max_relative = sector
            .pairs
            .iter()
            .map(|pair| pair.relative)
            .fold(0.0_f64, f64::max);
        assert_eq!(sector.maximum_absolute, max_absolute);
        assert_eq!(sector.maximum_relative, max_relative);
        assert!(sector.pairs.iter().any(|pair| {
            pair.q_index == sector.worst_absolute_q_index
                && pair.column == sector.worst_absolute_column
                && pair.absolute == sector.maximum_absolute
        }));
        assert!(sector.pairs.iter().any(|pair| {
            pair.q_index == sector.worst_relative_q_index
                && pair.column == sector.worst_relative_column
                && pair.relative == sector.maximum_relative
        }));
    }

    let incomplete = build_spinor_mpb(
        &input,
        &SpinorMpbSpec {
            product_l_max: 2,
            product_g_max: coulomb_spec().projection.pw_cutoff,
            overlap_tolerance: DEFAULT_TOLERANCE,
            selections: vec![SpinorMpbSelection {
                k: 0,
                left_band: 0,
                right_band: 0,
            }],
        },
    )
    .unwrap();
    if input.pair_columns.n_columns().unwrap() > 1 {
        assert!(matches!(
            compare_spinor_sector_thc_mpb(
                std::slice::from_ref(&input),
                &thc,
                std::slice::from_ref(&incomplete),
                std::slice::from_ref(&core),
                &coulomb_spec(),
            ),
            Err(SpinorSectorThcError::MpbCoverage {
                q_index: 0,
                occupied_space: ExchangeSpace::Valence,
                target_space: ExchangeSpace::Valence,
            })
        ));
    }
}
