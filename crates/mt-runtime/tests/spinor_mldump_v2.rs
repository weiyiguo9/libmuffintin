//! M5 exchange export from the bounded relaxed-core production pipeline.

use std::path::PathBuf;

use muffintin::{
    CheckpointPhysics, FockMixing, GammaExchangeTreatment, RankPolicy, RelaxedCoreHfResult,
    RelaxedCoreHfSpec, SpinorCoulombResult, SpinorCoulombSpec, SpinorMldumpV2Error,
    SpinorMpbSelection, SpinorMpbSpec, SpinorSectorThcMpbComparison, SpinorSectorThcResult,
    SpinorThcResult, build_spinor_coulomb, build_spinor_exchange_mpb, build_spinor_mpb,
    build_spinor_sector_thc, build_spinor_thc, compare_spinor_sector_thc_mpb,
    run_gamma_relaxed_core_hf, write_spinor_mldump_v2,
};
use muffintin_core::{Hartree, InverseBohr};
use muffintin_coulomb::CoulombRequest;
use muffintin_dft::{
    CoreFixedPotentialSpec, ScfChannelIdentity, ScfConvergence, ScfCoreState, ScfMixing,
};
use muffintin_io::{
    InitialV2, MLDUMP_EXCHANGE_BACKEND_V2, MLDUMP_EXCHANGE_SOURCE_FRAME_V2,
    MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION, MldumpExchangeSpaceV2, MldumpGeometryV1,
    MldumpHeaderV1, MldumpKMinusQV1, MldumpKPointV1, MldumpMeshV1, MldumpMetaV1, MldumpQEntryV1,
    MldumpRadialMeshV1, MldumpSiteV1, read_mldump_v2,
};
use muffintin_prodbasis::mpb::DEFAULT_TOLERANCE;

#[path = "spinor_hydrogen.rs"]
mod spinor_hydrogen;

use spinor_hydrogen::{
    LATTICE, coulomb_spec, hydrogen_spinor_checkpoint, parent_grid, spinor_config, thc_spec,
};

struct Pipeline {
    result: RelaxedCoreHfResult,
    spec: RelaxedCoreHfSpec,
    vv_thc: SpinorThcResult,
    vv_coulomb: SpinorCoulombResult,
    sector_thc: SpinorSectorThcResult,
    comparison: SpinorSectorThcMpbComparison,
    header: MldumpHeaderV1,
}

fn fixture_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

fn relaxed_setup() -> (muffintin_io::CheckpointV2, RelaxedCoreHfSpec) {
    let mut checkpoint = hydrogen_spinor_checkpoint();
    checkpoint.meta.title = "neutral closed-shell core+valence MLDUMP v2 smoke".to_owned();
    checkpoint.geometry.sites[0].atomic_number = 4;
    let InitialV2::FrozenPotential { potential } = &mut checkpoint.initial else {
        unreachable!("shared fixture is a frozen-potential checkpoint")
    };
    for channel in &mut potential.v0.muffin_tins[0].channels {
        for value in &mut channel.real {
            *value *= 4.0;
        }
    }

    let mut config = spinor_config([1, 1, 1], 0.5);
    config.electron_count = 4.0;
    config.basis.channels[0].identity = ScfChannelIdentity::ScalarL { n: 2, l: 0 };
    config.core_sites[0].states = vec![ScfCoreState {
        principal_quantum_number: 1,
        kappa: -1,
        occupation: 2.0,
    }];
    config.mixing = ScfMixing::Linear { alpha: 0.5 };
    config.convergence = ScfConvergence {
        energy_tolerance: Hartree(1.0e100),
        density_tolerance: 1.0e100,
        max_iterations: 2,
    };
    let spec = RelaxedCoreHfSpec {
        config,
        product_l_max: 2,
        product_g_max: InverseBohr(1.5),
        overlap_tolerance: DEFAULT_TOLERANCE,
        coulomb: CoulombRequest::cubic(LATTICE, 2).unwrap(),
        gamma: GammaExchangeTreatment::FiniteBody,
        max_fock_iterations: 32,
        fock_density_tolerance: 1.0e-7,
        fock_mixing: FockMixing::Linear { alpha: 0.5 },
        core: CoreFixedPotentialSpec {
            action_mixing: 1.0,
            energy_tolerance: Hartree(1.0e100),
            radial_tolerance: 1.0e100,
            vc_imaginary_tolerance: 1.0e-8,
            max_iterations: 2,
        },
        sector_numerical_tolerance: Hartree(1.0e-8),
        maximum_core_shell_spill: 1.0,
    };
    (checkpoint, spec)
}

fn full_vv_spec(input: &muffintin::SpinorProductInput, spec: &RelaxedCoreHfSpec) -> SpinorMpbSpec {
    let n_k = input.orbitals.k_fractional.len();
    let n_valence = input.orbitals.band_window.count;
    SpinorMpbSpec {
        product_l_max: spec.product_l_max,
        product_g_max: spec.product_g_max,
        overlap_tolerance: spec.overlap_tolerance,
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

fn build_pipeline() -> Pipeline {
    let (checkpoint, spec) = relaxed_setup();
    let mut physics = CheckpointPhysics::new(&checkpoint).unwrap();
    let result = run_gamma_relaxed_core_hf(&mut physics, &spec).unwrap();
    let inputs = &result.final_exchange_inputs;
    let grid = parent_grid(&inputs[0]);
    let vv_thc = build_spinor_thc(inputs, &grid, &thc_spec()).unwrap();
    let vv_spec = SpinorCoulombSpec {
        request: spec.coulomb.clone(),
        projection: coulomb_spec().projection,
    };
    let vv_coulomb = build_spinor_coulomb(inputs, &vv_thc, &vv_spec, &[]).unwrap();
    let sector_thc = build_spinor_sector_thc(inputs, &grid, &thc_spec()).unwrap();
    assert_eq!(sector_thc.requested_rank, RankPolicy::Exact { n_mu: 1 });
    let vv_mpb = inputs
        .iter()
        .map(|input| build_spinor_mpb(input, &full_vv_spec(input, &spec)))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let core_mpb = inputs
        .iter()
        .map(|input| {
            build_spinor_exchange_mpb(
                input,
                &muffintin::SpinorExchangeMpbSpec {
                    product_l_max: spec.product_l_max,
                    product_g_max: spec.product_g_max,
                    overlap_tolerance: spec.overlap_tolerance,
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let comparison =
        compare_spinor_sector_thc_mpb(inputs, &sector_thc, &vv_mpb, &core_mpb, &vv_spec).unwrap();
    let header = header_from_result(&result, &spec);
    Pipeline {
        result,
        spec,
        vv_thc,
        vv_coulomb,
        sector_thc,
        comparison,
        header,
    }
}

fn header_from_result(result: &RelaxedCoreHfResult, spec: &RelaxedCoreHfSpec) -> MldumpHeaderV1 {
    let inputs = &result.final_exchange_inputs;
    let first = &inputs[0];
    let cell = spec.coulomb.cell();
    let sites = first
        .source
        .partition
        .sites()
        .iter()
        .zip(&first.source.radials)
        .enumerate()
        .map(|(index, (site, radials))| MldumpSiteV1 {
            species: Some("Be".to_owned()),
            label: Some(format!("Be-{}", index + 1)),
            position_bohr: site.position.map(|component| component.get()),
            radius_bohr: site.radius.get(),
            radial_mesh: MldumpRadialMeshV1 {
                first_bohr: radials.mesh.first().get(),
                log_increment: radials.mesh.increment(),
                point_count: radials.mesh.len(),
            },
        })
        .collect();
    MldumpHeaderV1::new(
        MldumpMetaV1 {
            producer_name: "libmuffintin-runtime-spinor-mldump-v2-test".to_owned(),
            producer_version: "0.1.0".to_owned(),
            source_revision: "d429d60250a092c0cd41c3d562965caf43a62878".to_owned(),
            feature_representation: MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION.to_owned(),
        },
        MldumpGeometryV1 {
            direct_basis_bohr: std::array::from_fn(|row| {
                std::array::from_fn(|axis| cell.basis()[row][axis].get())
            }),
            reciprocal_basis_inv_bohr: std::array::from_fn(|row| {
                std::array::from_fn(|axis| first.reciprocal.basis()[row][axis].get())
            }),
            cell_volume_bohr3: cell.volume().get(),
            sites,
        },
        MldumpMeshV1 {
            k_points: result
                .k_fractional
                .iter()
                .zip(&result.k_weights)
                .map(|(fractional, weight)| MldumpKPointV1 {
                    fractional: *fractional,
                    weight: *weight,
                })
                .collect(),
            q_entries: inputs
                .iter()
                .zip(&result.q_fractional)
                .map(|(input, canonical)| MldumpQEntryV1 {
                    input_fractional: *canonical,
                    canonical_fractional: *canonical,
                    global_umklapp: input.source.q.umklapp.index,
                    k_minus_q: input
                        .k_minus_q
                        .iter()
                        .map(|mapped| MldumpKMinusQV1 {
                            k_index: mapped.k_index,
                            mapped_index: mapped.kq_index,
                            g_wrap: mapped.umklapp.index,
                        })
                        .collect(),
                })
                .collect(),
        },
    )
}

#[test]
fn relaxed_final_frame_roundtrips_common_payload_and_exchange_summary() {
    let path = fixture_path("libmuffintin-runtime-spinor-mldump-v2.h5");
    let _ = std::fs::remove_file(&path);
    let pipeline = build_pipeline();
    write_spinor_mldump_v2(
        &path,
        &pipeline.header,
        &pipeline.result,
        &pipeline.spec,
        &pipeline.vv_thc,
        &pipeline.vv_coulomb,
        &pipeline.sector_thc,
        &pipeline.comparison,
    )
    .unwrap();

    let read = read_mldump_v2(&path).unwrap();
    assert_eq!(read.header, pipeline.header);
    assert_eq!(
        read.payload.orbitals.k_points.len(),
        pipeline.result.k_fractional.len()
    );
    assert_eq!(
        read.payload.products.q_records.len(),
        pipeline.result.q_fractional.len()
    );
    assert_eq!(
        read.payload.thc.q_records.len(),
        pipeline.result.q_fractional.len()
    );
    assert_eq!(
        read.payload.coulomb.q_records.len(),
        pipeline.result.q_fractional.len()
    );
    assert_eq!(
        read.exchange.provenance.source_frame,
        MLDUMP_EXCHANGE_SOURCE_FRAME_V2
    );
    assert_eq!(read.exchange.provenance.backend, MLDUMP_EXCHANGE_BACKEND_V2);
    assert_eq!(
        read.exchange.vv.layout.occupied_space,
        MldumpExchangeSpaceV2::Valence
    );
    assert_eq!(
        read.exchange.cv.layout.occupied_space,
        MldumpExchangeSpaceV2::Core
    );
    assert_eq!(
        read.exchange.vc.layout.target_space,
        MldumpExchangeSpaceV2::Core
    );
    assert_eq!(
        read.exchange.cc.layout.target_space,
        MldumpExchangeSpaceV2::Core
    );
    assert_eq!(
        read.exchange.exchange_total_hartree,
        pipeline.result.sector_exchange.exchange_total.get()
    );
    assert_eq!(
        read.exchange.exchange_cv_hartree,
        pipeline.result.sector_exchange.cross_trace_average.get()
    );
    assert_eq!(
        read.exchange.provenance.k_weights,
        pipeline.result.k_weights
    );
    assert_eq!(
        read.exchange.provenance.valence_occupations,
        pipeline.result.occupations
    );
    assert_eq!(
        read.exchange.provenance.core_occupations.len(),
        pipeline.result.final_exchange_inputs[0].core.orbitals.len()
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn writer_rejects_a_stale_final_exchange_frame_before_creating_v1() {
    let path = fixture_path("libmuffintin-runtime-spinor-mldump-v2-stale.h5");
    let _ = std::fs::remove_file(&path);
    let mut pipeline = build_pipeline();
    pipeline.result.final_exchange_inputs[0].orbitals.energies[0][0].0 += 1.0;
    let error = write_spinor_mldump_v2(
        &path,
        &pipeline.header,
        &pipeline.result,
        &pipeline.spec,
        &pipeline.vv_thc,
        &pipeline.vv_coulomb,
        &pipeline.sector_thc,
        &pipeline.comparison,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        SpinorMldumpV2Error::FrozenFrame("sector_exchange")
    ));
    assert!(!path.exists());
}
