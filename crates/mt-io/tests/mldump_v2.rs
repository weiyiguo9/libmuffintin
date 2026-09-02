//! Core MLDUMP v2 roundtrip and strict trust-boundary tests.

use std::fs;
use std::path::PathBuf;

use hdf5_metno::File;
use muffintin_io::{
    IoError, MLDUMP_EXCHANGE_BACKEND_V2, MLDUMP_EXCHANGE_SOURCE_FRAME_V2, MLDUMP_SCHEMA_VERSION_V1,
    MLDUMP_SCHEMA_VERSION_V2, MldumpCoreOccupationV2, MldumpExchangeFitResidualV2,
    MldumpExchangeLayoutV2, MldumpExchangeMpbQuadraticV2, MldumpExchangeProvenanceV2,
    MldumpExchangeRankScalingV2, MldumpExchangeSectorV2, MldumpExchangeSpaceV2, MldumpExchangeV2,
    MldumpGammaPolicyV2, MldumpRequestedRankV2, MldumpSelectorEngineV2, MldumpSelectorStrategyV2,
    read_mldump_v1, read_mldump_v2, upgrade_mldump_v1_with_exchange_v2,
};

const V1_FIXTURE: &[u8] = include_bytes!("../../../python/tests/fixtures/mldump_spinor_v1.h5");

fn fixture_path(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    fs::write(&path, V1_FIXTURE).unwrap();
    path
}

fn layout(
    occupied_space: MldumpExchangeSpaceV2,
    target_space: MldumpExchangeSpaceV2,
    n_occupied: usize,
    n_target: usize,
) -> MldumpExchangeLayoutV2 {
    MldumpExchangeLayoutV2 {
        occupied_space,
        target_space,
        n_k: 2,
        n_occupied,
        n_target,
    }
}

fn sector(layout: MldumpExchangeLayoutV2, trace_hartree: f64) -> MldumpExchangeSectorV2 {
    MldumpExchangeSectorV2 {
        layout,
        trace_hartree,
        maximum_antihermitian: 1.0e-13,
        fit_residual: MldumpExchangeFitResidualV2 {
            frobenius: 2.0e-8,
            column_max: 3.0e-8,
        },
        mpb_quadratic: MldumpExchangeMpbQuadraticV2 {
            maximum_absolute: 4.0e-8,
            maximum_relative: 5.0e-8,
            worst_absolute_q_index: 1,
            worst_absolute_column: 1,
            worst_relative_q_index: 0,
            worst_relative_column: 0,
        },
    }
}

fn exchange() -> MldumpExchangeV2 {
    let vv = sector(
        layout(
            MldumpExchangeSpaceV2::Valence,
            MldumpExchangeSpaceV2::Valence,
            2,
            2,
        ),
        -4.0,
    );
    let cv = sector(
        layout(
            MldumpExchangeSpaceV2::Core,
            MldumpExchangeSpaceV2::Valence,
            1,
            2,
        ),
        -1.8,
    );
    let vc = sector(
        layout(
            MldumpExchangeSpaceV2::Valence,
            MldumpExchangeSpaceV2::Core,
            2,
            1,
        ),
        -2.2,
    );
    let cc = sector(
        layout(
            MldumpExchangeSpaceV2::Core,
            MldumpExchangeSpaceV2::Core,
            1,
            1,
        ),
        -1.0,
    );
    MldumpExchangeV2 {
        vv,
        cv,
        vc,
        cc,
        exchange_vv_hartree: -2.0,
        exchange_cv_hartree: -2.0,
        exchange_cc_hartree: -0.5,
        exchange_total_hartree: -4.5,
        cross_trace_average_hartree: -2.0,
        cross_trace_mismatch_hartree: 0.4,
        provenance: MldumpExchangeProvenanceV2 {
            source_frame: MLDUMP_EXCHANGE_SOURCE_FRAME_V2.to_owned(),
            backend: MLDUMP_EXCHANGE_BACKEND_V2.to_owned(),
            gamma_policy: MldumpGammaPolicyV2::FiniteBody,
            product_l_max: 4,
            product_g_max_inv_bohr: 3.0,
            overlap_tolerance: 1.0e-10,
            coulomb_lexp: 4,
            interpolation_l_max: 2,
            interpolation_pw_cutoff_inv_bohr: 2.0,
            selector_strategy: MldumpSelectorStrategyV2::AllQL2,
            selector_engine: MldumpSelectorEngineV2::FullColumnPivotedQr,
            requested_rank: MldumpRequestedRankV2::Exact { n_mu: 2 },
            rank_scaling: MldumpExchangeRankScalingV2 {
                n_k: 2,
                n_valence: 2,
                n_core: 1,
                n_candidates: 3,
                effective_rank: 2,
                vv_columns_per_q: 8,
                cv_columns_per_q: 4,
                vc_columns_per_q: 4,
                cc_columns_per_q: 2,
                pooled_columns_per_q: 18,
                selector_rows: 36,
            },
            k_weights: vec![0.5, 0.5],
            valence_occupations: vec![vec![1.0, 0.5], vec![0.75, 0.25]],
            core_occupations: vec![MldumpCoreOccupationV2 {
                site_index: 0,
                n: 1,
                signed_kappa: -1,
                twice_mu: -1,
                occupation: 1.0,
            }],
        },
    }
}

fn schema_version(path: &PathBuf) -> u32 {
    File::open(path)
        .unwrap()
        .attr("schema_version")
        .unwrap()
        .read_scalar()
        .unwrap()
}

#[test]
fn exchange_v2_roundtrip_and_readers_reject_the_other_version() {
    let path = fixture_path("libmuffintin-mldump-v2-roundtrip.h5");
    assert!(matches!(
        read_mldump_v2(&path),
        Err(IoError::UnsupportedVersion {
            supported: MLDUMP_SCHEMA_VERSION_V2,
            found: MLDUMP_SCHEMA_VERSION_V1,
            ..
        })
    ));

    let expected = exchange();
    upgrade_mldump_v1_with_exchange_v2(&path, &expected).unwrap();
    assert_eq!(read_mldump_v2(&path).unwrap().exchange, expected);
    assert!(matches!(
        read_mldump_v1(&path),
        Err(IoError::UnsupportedVersion {
            supported: MLDUMP_SCHEMA_VERSION_V1,
            found: MLDUMP_SCHEMA_VERSION_V2,
            ..
        })
    ));
}

#[test]
fn upgrade_rejects_bad_energy_layout_and_provenance_before_version_flip() {
    let cases = [
        {
            let mut value = exchange();
            value.exchange_total_hartree += 1.0;
            value
        },
        {
            let mut value = exchange();
            value.cv.layout.occupied_space = MldumpExchangeSpaceV2::Valence;
            value
        },
        {
            let mut value = exchange();
            value.provenance.backend = "ambiguous_backend".to_owned();
            value
        },
    ];
    for (index, value) in cases.iter().enumerate() {
        let path = fixture_path(&format!("libmuffintin-mldump-v2-invalid-{index}.h5"));
        assert!(matches!(
            upgrade_mldump_v1_with_exchange_v2(&path, value),
            Err(IoError::Validation(_))
        ));
        assert_eq!(schema_version(&path), MLDUMP_SCHEMA_VERSION_V1);
        read_mldump_v1(&path).unwrap();
    }
}
