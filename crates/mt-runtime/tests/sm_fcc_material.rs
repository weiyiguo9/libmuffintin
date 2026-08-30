//! Sm fcc catalogue and bounded SPEX-checkpoint lane.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use muffintin::{
    ChannelIdentity, ChannelTreatment, CheckpointPhysics, RecipeSite, SpinorCoulombSpec,
    SpinorMpbSelection, SpinorMpbSpec, ThcCandidates, ThcParentGrid, ThcPoint, ThcRegion,
    build_spinor_coulomb, build_spinor_mpb, compile_channel_recipe, write_spinor_mldump,
};
use muffintin_core::Cell;
use muffintin_core::{Bohr, Hartree, InverseBohr};
use muffintin_coulomb::{CoulombRequest, InterpolationProjection};
use muffintin_dft::{
    AtomicChannelTreatment, AtomicNumber, LinearizationEnergyGenerator, NoncollinearXcRoute,
    RelativisticOrbital, ScfBasis, ScfChannelIdentity, ScfChannelProvenance, ScfChannelRecipe,
    ScfChannelTreatment, ScfConfig, ScfConvergence, ScfExchangeCorrelation, ScfKMesh, ScfMixing,
    ScfOccupations, ScfRelativity, XcFunctional, fleur_default_atomic_configuration,
};
use muffintin_io::{
    MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION, MldumpGeometryV1, MldumpHeaderV1,
    MldumpKMinusQV1, MldumpKPointV1, MldumpMeshV1, MldumpMetaV1, MldumpPayloadV1, MldumpQEntryV1,
    MldumpRadialMeshV1, MldumpSiteV1, RadialBasisSpinV2, RadialEquationTag,
    SpexMaterialBasisRecipeV1, SpexMaterialChannelKind, SpexMaterialChannelV1,
    materialize_checkpoint_v2, read_mldump_v1, read_spex_snapshot_hdf,
};
use muffintin_operators::lapw::Provenance;
use muffintin_prodbasis::mpb::DEFAULT_TOLERANCE;
use muffintin_prodbasis::thc::RankPolicy;

#[path = "material_lane_common.rs"]
mod material_lane_common;

#[path = "thc_fixture_common.rs"]
mod thc_fixture_common;

use thc_fixture_common::on_shell;

const SM_Z: u8 = 62;

fn sm_number() -> AtomicNumber {
    let atomic_number = AtomicNumber::new(SM_Z).expect("Sm is in the FLEUR catalogue");
    assert_eq!(atomic_number.symbol(), "Sm");
    assert_eq!(AtomicNumber::from_symbol("Sm"), Some(atomic_number));
    atomic_number
}

fn occupation(
    configuration: &muffintin_dft::AtomicElectronicConfiguration,
    n: u8,
    kappa: i8,
) -> muffintin_dft::AtomicOccupation {
    let orbital = RelativisticOrbital::new(n, kappa).expect("admissible (n, kappa)");
    *configuration
        .occupations()
        .iter()
        .find(|channel| channel.orbital == orbital)
        .expect("requested Sm channel must be occupied")
}

#[test]
fn sm_fleur_catalogue_has_4f_valence_and_5p12_relativistic_lo() {
    let configuration = fleur_default_atomic_configuration(sm_number());
    assert!((configuration.total_occupation() - f64::from(SM_Z)).abs() < 1.0e-12);

    let four_f_minus = occupation(&configuration, 4, 3);
    let four_f_plus = occupation(&configuration, 4, -4);
    assert_eq!(four_f_minus.treatment, AtomicChannelTreatment::Valence);
    assert_eq!(four_f_plus.treatment, AtomicChannelTreatment::Valence);
    assert!((four_f_minus.occupation - 6.0 * 3.0 / 7.0).abs() < 1.0e-12);
    assert!((four_f_plus.occupation - 6.0 * 4.0 / 7.0).abs() < 1.0e-12);

    let five_p12 = occupation(&configuration, 5, 1);
    let five_p32 = occupation(&configuration, 5, -2);
    assert_eq!(
        five_p12.treatment,
        AtomicChannelTreatment::RelativisticLocalOrbital
    );
    assert_eq!(five_p32.treatment, AtomicChannelTreatment::Valence);
    assert!((five_p12.occupation - 2.0).abs() < 1.0e-12);
    assert!((five_p32.occupation - 4.0).abs() < 1.0e-12);

    assert!(
        !configuration
            .occupations()
            .iter()
            .any(|channel| channel.treatment == AtomicChannelTreatment::Core
                && channel.orbital.principal_quantum_number() == 4
                && matches!(channel.orbital.kappa(), 3 | -4)),
        "Sm 4f must remain valence in the FLEUR default split"
    );
}

#[test]
fn sm_built_in_recipe_keeps_5p12_as_lo_and_does_not_invent_hdlo() {
    let compiled = compile_channel_recipe(
        &[RecipeSite {
            id: "Sm-1".to_owned(),
            atomic_number: sm_number(),
        }],
        None,
        None,
        &BTreeMap::new(),
    )
    .expect("built-in Sm recipe must compile");
    let site = compiled.site("Sm-1").expect("Sm-1 site");
    assert_eq!(site.atomic_number.get(), SM_Z);

    let has_4f = site.channels.iter().any(|record| {
        record.identity == ChannelIdentity::Kappa { n: 4, kappa: 3 }
            && record.treatment == ChannelTreatment::Valence
    }) && site.channels.iter().any(|record| {
        record.identity == ChannelIdentity::Kappa { n: 4, kappa: -4 }
            && record.treatment == ChannelTreatment::Valence
    });
    assert!(
        has_4f,
        "compiled Sm recipe must retain both 4f kappa partners"
    );

    let five_p12 = site
        .channels
        .iter()
        .find(|record| record.identity == ChannelIdentity::Kappa { n: 5, kappa: 1 })
        .expect("5p1/2 must be present");
    assert_eq!(five_p12.treatment, ChannelTreatment::Lo);

    assert!(
        !site
            .channels
            .iter()
            .any(|record| record.treatment == ChannelTreatment::Hdlo),
        "FLEUR default.econfig does not encode HDLO; the built-in recipe must not invent one"
    );
}

const ARTIFACT: &str = "/tmp/spex-sm-artifact/checkpoint.h5";
const ARTIFACT_SHA256: &str = "9f060f742e9078ec3dc8ee24d8945d38ec74a729e5dee85acfbffd345e132a59";

fn bounded_parent_grid(input: &muffintin::SpinorProductInput) -> ThcParentGrid {
    let origin = input.source.partition.sites()[0].position;
    let mesh = &input.source.radials[0].mesh;
    let mid = mesh.radii().len() / 2;
    let r_mid = mesh.radii()[mid].get();
    let r0 = mesh.radii()[0].get();
    ThcParentGrid::new(
        input.source.partition.clone(),
        Provenance::default(),
        vec![
            ThcPoint {
                coordinate: on_shell(origin, r0, [0.4, -0.3, 0.2]),
                weight: 0.35,
                region: ThcRegion::MuffinTin {
                    site: 0,
                    radial_index: 0,
                },
            },
            ThcPoint {
                coordinate: on_shell(origin, r_mid, [1.0, 0.0, 0.0]),
                weight: 0.0,
                region: ThcRegion::MuffinTin {
                    site: 0,
                    radial_index: mid,
                },
            },
            ThcPoint {
                coordinate: on_shell(origin, r_mid, [0.0, 1.0, 0.0]),
                weight: 0.45,
                region: ThcRegion::MuffinTin {
                    site: 0,
                    radial_index: mid,
                },
            },
            ThcPoint {
                coordinate: [
                    Bohr(origin[0].get() + 0.2),
                    Bohr(origin[1].get() + 0.2),
                    Bohr(origin[2].get() + 0.2),
                ],
                weight: 0.8,
                region: ThcRegion::Interstitial,
            },
            ThcPoint {
                coordinate: [
                    Bohr(origin[0].get() + 1.1),
                    Bohr(origin[1].get() + 0.4),
                    Bohr(origin[2].get() + 0.3),
                ],
                weight: 0.15,
                region: ThcRegion::Interstitial,
            },
            ThcPoint {
                coordinate: [
                    Bohr(origin[0].get() + 0.3),
                    Bohr(origin[1].get() + 1.2),
                    Bohr(origin[2].get() + 0.5),
                ],
                weight: 0.25,
                region: ThcRegion::Interstitial,
            },
        ],
    )
    .unwrap()
}

fn sm_runtime_channels(recipe: &SpexMaterialBasisRecipeV1) -> Vec<ScfChannelRecipe> {
    let mut channels = [(6, 0), (5, 1), (5, 2), (4, 3)]
        .into_iter()
        .map(|(n, l)| ScfChannelRecipe {
            site: "Sm-1".to_owned(),
            identity: ScfChannelIdentity::ScalarL { n, l },
            treatment: ScfChannelTreatment::Valence,
            derivative_order: 0,
            generator: LinearizationEnergyGenerator::FrozenCheckpoint,
            seed: None,
            provenance: ScfChannelProvenance::BuiltIn,
        })
        .collect::<Vec<_>>();
    channels.extend(recipe.channels.iter().map(|record| ScfChannelRecipe {
        site: record.site_id.clone(),
        identity: ScfChannelIdentity::Kappa {
            n: record.n,
            kappa: record.kappa,
        },
        treatment: match record.kind {
            SpexMaterialChannelKind::Lo | SpexMaterialChannelKind::Rlo => ScfChannelTreatment::Lo,
            SpexMaterialChannelKind::Hdlo => ScfChannelTreatment::Hdlo,
        },
        derivative_order: record.derivative_order,
        generator: LinearizationEnergyGenerator::FrozenCheckpoint,
        seed: None,
        provenance: ScfChannelProvenance::ExternalRecipe {
            source: Some(recipe.producer.clone()),
        },
    }));
    channels
}

/// Bounded Sm fcc SPEX snapshot lane at `/tmp/spex-sm-artifact/checkpoint.h5`.
///
/// Ordinary workspace tests skip this. Run:
/// `cargo test -p libmuffintin-runtime --test sm_fcc_material consume_b45d9b9_spex_snapshot_and_run_bounded_sm_lane -- --ignored --exact --nocapture`
#[ignore = "requires local SPEX artifact /tmp/spex-sm-artifact/checkpoint.h5; run with --ignored"]
#[test]
fn consume_b45d9b9_spex_snapshot_and_run_bounded_sm_lane() {
    let path = Path::new(ARTIFACT);
    if !path.is_file() {
        panic!("authorized artifact missing at {ARTIFACT}");
    }
    let started = Instant::now();
    let fields =
        read_spex_snapshot_hdf(path).expect("frozen reader must load b45d9b9 checkpoint.h5");
    assert_eq!(fields.spin_layout, "collinear-up-down");
    assert_eq!(fields.interstitial_phase, "positive-exponent");
    assert_eq!(
        fields.hashes[0].sha256,
        "bd0734b9cfc6268489d10da7cb9bad159cc312a633650e2346e7460e3c17c179"
    );
    assert_eq!(
        fields.hashes[1].sha256,
        "92fdd4e4362e342cca4cefebe8dc6c181b317f599b5588e4e02d6ea67078019f"
    );

    let mut recipe_channels = Vec::new();
    for table in fields
        .scalar_los
        .iter()
        .filter(|table| table.spin == RadialBasisSpinV2::Up)
    {
        for lo in &table.orbitals {
            let kappa = if lo.l == 0 {
                -1
            } else {
                i32::try_from(lo.l).unwrap()
            };
            recipe_channels.push(SpexMaterialChannelV1 {
                site_id: table.site_id.clone(),
                n: lo.n.unwrap_or(5),
                l: lo.l,
                kappa,
                kind: if lo.l == 1 {
                    SpexMaterialChannelKind::Rlo
                } else {
                    SpexMaterialChannelKind::Lo
                },
                derivative_order: 0,
                energy: lo.energy,
            });
        }
    }
    let recipe = SpexMaterialBasisRecipeV1 {
        producer: "libmuffintin-sm-material-recipe".to_owned(),
        recipe_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            .to_owned(),
        channels: recipe_channels,
    };
    let materialized =
        materialize_checkpoint_v2(&fields, &recipe).expect("recipe must match SPEX scalar LOs");
    assert_eq!(materialized.recipe_sha256, recipe.recipe_sha256);
    assert!(
        materialized
            .checkpoint
            .geometry
            .radial_basis
            .iter()
            .all(|basis| basis.radial_equation == RadialEquationTag::ScalarKoellingHarmon),
        "SPEX must remain frozen scalar Koelling-Harmon source provenance"
    );

    let config = ScfConfig {
        electron_count: 16.0,
        k_mesh: ScfKMesh {
            divisions: [1, 1, 1],
            shift: [0.0; 3],
        },
        basis: ScfBasis {
            plane_wave_cutoff: InverseBohr(fields.plane_wave_cutoff.min(0.5)),
            l_max: 3,
            channels: sm_runtime_channels(&recipe),
            resolved_channels: Vec::new(),
        },
        occupations: ScfOccupations::FermiDirac {
            temperature: Hartree(0.02),
        },
        exchange_correlation: ScfExchangeCorrelation {
            functional: XcFunctional::LdaPw92,
            noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
        },
        mixing: ScfMixing::Linear { alpha: 1.0 },
        relativity: ScfRelativity::SpinorFirstVariation,
        convergence: ScfConvergence {
            energy_tolerance: Hartree(1.0),
            density_tolerance: 1.0,
            max_iterations: 1,
        },
        core_sites: Vec::new(),
    };
    let mut mismatched_basis = config.basis.clone();
    mismatched_basis
        .channels
        .iter_mut()
        .find(|channel| channel.treatment == ScfChannelTreatment::Lo)
        .expect("bound Sm local orbital")
        .derivative_order = 1;
    assert!(matches!(
        CheckpointPhysics::new_spex_material(&materialized.checkpoint, &recipe, &mismatched_basis),
        Err(muffintin::CheckpointPhysicsError::SpexMaterialChannelMismatch { .. })
    ));
    let physics =
        CheckpointPhysics::new_spex_material(&materialized.checkpoint, &recipe, &config.basis)
            .expect("typed SPEX recipe must bind to the target Dirac basis");
    let fixture = material_lane_common::MaterialFixture {
        checkpoint: materialized.checkpoint,
        config,
        physics,
        provenance: material_lane_common::MaterialProvenance {
            checkpoint_path: path.to_path_buf(),
            checkpoint_sha256: ARTIFACT_SHA256.to_owned(),
            producer: "b45d9b9e1505d25236c3e78674418b011a471666".to_owned(),
        },
    };
    let inputs = material_lane_common::ordered_q_slice(&fixture)
        .expect("bound SPEX material must execute the target Dirac product route");
    assert_eq!(inputs.len(), 1);
    assert!(inputs[0].source.radials[0].valence.iter().all(|radial| {
        radial.samples.large.len() == radial.samples.small.len()
            && radial.samples.large.iter().all(|value| value.is_finite())
            && radial.samples.small.iter().all(|value| value.is_finite())
    }));
    assert!(
        inputs[0].source.radials[0]
            .valence
            .iter()
            .any(
                |radial| radial.samples.large.iter().any(|value| value.abs() > 0.0)
                    && radial.samples.small.iter().any(|value| value.abs() > 0.0)
            ),
        "target solve must produce physical Dirac P/Q samples"
    );
    let grid = bounded_parent_grid(&inputs[0]);
    let (qrcp, chol) = material_lane_common::compare_qrcp_cholesky(
        &inputs,
        &grid,
        RankPolicy::Exact { n_mu: 1 },
        ThcCandidates::All,
    )
    .expect("QRCP vs pivoted Cholesky on the same parent grid");
    assert_eq!(qrcp.grid.points().len(), chol.grid.points().len());
    assert_eq!(qrcp.effective_rank, 1);
    assert_eq!(chol.effective_rank, 1);
    assert!(qrcp.records_match_parent_grid());
    assert!(chol.records_match_parent_grid());
    let mpb = build_spinor_mpb(
        &inputs[0],
        &SpinorMpbSpec {
            product_l_max: 2,
            product_g_max: InverseBohr(1.5),
            overlap_tolerance: DEFAULT_TOLERANCE,
            selections: vec![SpinorMpbSelection {
                k: 0,
                left_band: 0,
                right_band: 0,
            }],
        },
    )
    .expect("bounded MPB pair");
    assert_eq!(mpb.vertices.len(), 1);
    let cell = Cell::new(std::array::from_fn(|row| {
        std::array::from_fn(|axis| Bohr(fixture.checkpoint.geometry.lattice.vectors[row][axis]))
    }))
    .unwrap();
    let coulomb = build_spinor_coulomb(
        &inputs,
        &qrcp,
        &SpinorCoulombSpec {
            request: CoulombRequest::new(cell, 2).unwrap(),
            projection: InterpolationProjection::new(InverseBohr(1.5), 1).unwrap(),
        },
        &[],
    )
    .expect("bounded spinor Coulomb");
    assert_eq!(coulomb.records().len(), inputs.len());
    let header = header_from_inputs(&inputs, &fixture);
    let dump = std::env::temp_dir().join("sm-fcc-b45d9b9-spinor.mldump.h5");
    write_spinor_mldump(
        &dump,
        &header,
        &inputs,
        &qrcp,
        &coulomb,
        &SpinorCoulombSpec {
            request: CoulombRequest::new(
                Cell::new(std::array::from_fn(|row| {
                    std::array::from_fn(|axis| {
                        Bohr(fixture.checkpoint.geometry.lattice.vectors[row][axis])
                    })
                }))
                .unwrap(),
                2,
            )
            .unwrap(),
            projection: InterpolationProjection::new(InverseBohr(1.5), 1).unwrap(),
        },
    )
    .expect("MLDUMP write");
    let roundtrip = read_mldump_v1(&dump).expect("MLDUMP roundtrip");
    assert_eq!(
        roundtrip.header.meta.source_revision,
        "b45d9b9e1505d25236c3e78674418b011a471666"
    );
    assert!(matches!(roundtrip.payload, MldumpPayloadV1::Spinor(_)));
    eprintln!(
        "Sm bounded lane wall={:?} parent={} qrcp_rank={} chol_rank={} n_q={}",
        started.elapsed(),
        qrcp.grid.points().len(),
        qrcp.effective_rank,
        chol.effective_rank,
        inputs.len()
    );
}

fn header_from_inputs(
    inputs: &[muffintin::SpinorProductInput],
    fixture: &material_lane_common::MaterialFixture,
) -> MldumpHeaderV1 {
    let first = &inputs[0];
    let n_k = first.orbitals.k_fractional.len();
    let weight = 1.0 / n_k as f64;
    let sites = first
        .source
        .partition
        .sites()
        .iter()
        .zip(&first.source.radials)
        .enumerate()
        .map(|(index, (site, radials))| MldumpSiteV1 {
            species: Some("Sm".to_owned()),
            label: if index == 0 {
                Some("Sm-1".to_owned())
            } else {
                None
            },
            position_bohr: site.position.map(|component| component.get()),
            radius_bohr: site.radius.get(),
            radial_mesh: MldumpRadialMeshV1 {
                first_bohr: radials.mesh.first().get(),
                log_increment: radials.mesh.increment(),
                point_count: radials.mesh.len(),
            },
        })
        .collect();
    let cell = Cell::new(std::array::from_fn(|row| {
        std::array::from_fn(|axis| Bohr(fixture.checkpoint.geometry.lattice.vectors[row][axis]))
    }))
    .unwrap();
    MldumpHeaderV1::new(
        MldumpMetaV1 {
            producer_name: "libmuffintin-sm-material".to_owned(),
            producer_version: "0.1.0".to_owned(),
            source_revision: "b45d9b9e1505d25236c3e78674418b011a471666".to_owned(),
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
            k_points: first
                .orbitals
                .k_fractional
                .iter()
                .map(|fractional| MldumpKPointV1 {
                    fractional: *fractional,
                    weight,
                })
                .collect(),
            q_entries: inputs
                .iter()
                .map(|input| {
                    let umklapp = input.source.q.umklapp.index;
                    MldumpQEntryV1 {
                        input_fractional: input.source.q.cartesian.map(|c| c.get()),
                        canonical_fractional: input.source.q.cartesian.map(|c| c.get()),
                        global_umklapp: umklapp,
                        k_minus_q: input
                            .k_minus_q
                            .iter()
                            .map(|mapped| MldumpKMinusQV1 {
                                k_index: mapped.k_index,
                                mapped_index: mapped.kq_index,
                                g_wrap: mapped.umklapp.index,
                            })
                            .collect(),
                    }
                })
                .collect(),
        },
    )
}
