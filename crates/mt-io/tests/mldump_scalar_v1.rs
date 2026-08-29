//! Neutral scalar MLDUMP v1 fixture: HDF5 inspection plus owned-reader equality.

use std::f64::consts::PI;
use std::path::PathBuf;

use hdf5_metno::File;
use hdf5_metno::types::VarLenUnicode;
use muffintin_io::{
    IoError, MLDUMP_CORE_EMPTY_NOT_FITTED, MLDUMP_INTERSTITIAL_SENTINEL,
    MLDUMP_OCCUPATIONS_NOT_EXPORTED, MLDUMP_PAIR_ORDER_K_LEFT_RIGHT,
    MLDUMP_PARENT_REGION_INTERSTITIAL, MLDUMP_PARENT_REGION_MUFFIN_TIN,
    MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON, MLDUMP_STATUS_ABSENT_NOT_COMPUTED,
    MLDUMP_STATUS_PRESENT, MLDUMP_THC_ENGINE_QRCP, MLDUMP_THC_STRATEGY_ALL_QL2, MldumpGeometryV1,
    MldumpHeaderV1, MldumpKMinusQV1, MldumpKPointV1, MldumpMeshV1, MldumpMetaV1, MldumpQEntryV1,
    MldumpRadialMeshV1, MldumpSiteV1, MldumpWriterV1, ScalarApwSiteMatchRefV1,
    ScalarCoulombGammaRefV1, ScalarCoulombQRecordRefV1, ScalarCoulombRefV1,
    ScalarLocalOrbitalTableRefV1, ScalarOrbitalKRefV1, ScalarOrbitalSpinRefV1, ScalarOrbitalsRefV1,
    ScalarProductQRecordRefV1, ScalarProductSiteRefV1, ScalarProductsRefV1,
    ScalarThcParentGridRefV1, ScalarThcQRecordRefV1, ScalarThcRefV1, ScalarThcSelectionRefV1,
    ScalarThcVertexTableRefV1, ValidationError, read_mldump_v1,
};

fn fixture_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

fn header() -> MldumpHeaderV1 {
    MldumpHeaderV1::new(
        MldumpMetaV1 {
            producer_name: "libmuffintin-io-scalar-test".to_owned(),
            producer_version: "0.1.0".to_owned(),
            source_revision: "48021b567713688d09eacc69e61024bf48d38472".to_owned(),
            feature_representation: MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON.to_owned(),
        },
        MldumpGeometryV1 {
            direct_basis_bohr: [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            reciprocal_basis_inv_bohr: [
                [2.0 * PI / 8.0, 0.0, 0.0],
                [0.0, 2.0 * PI / 8.0, 0.0],
                [0.0, 0.0, 2.0 * PI / 8.0],
            ],
            cell_volume_bohr3: 512.0,
            sites: vec![
                MldumpSiteV1 {
                    species: Some("H".to_owned()),
                    label: Some("H-1".to_owned()),
                    position_bohr: [0.0, 0.0, 0.0],
                    radius_bohr: 1.0,
                    radial_mesh: MldumpRadialMeshV1 {
                        first_bohr: 1.0e-4,
                        log_increment: 0.05,
                        point_count: 4,
                    },
                },
                MldumpSiteV1 {
                    species: Some("He".to_owned()),
                    label: None,
                    position_bohr: [4.0, 0.0, 0.0],
                    radius_bohr: 1.2,
                    radial_mesh: MldumpRadialMeshV1 {
                        first_bohr: 2.0e-4,
                        log_increment: 0.04,
                        point_count: 3,
                    },
                },
            ],
        },
        MldumpMeshV1 {
            k_points: vec![
                MldumpKPointV1 {
                    fractional: [0.0, 0.0, 0.0],
                    weight: 0.5,
                },
                MldumpKPointV1 {
                    fractional: [0.5, 0.0, 0.0],
                    weight: 0.5,
                },
            ],
            q_entries: vec![
                MldumpQEntryV1 {
                    input_fractional: [0.0, 0.0, 0.0],
                    canonical_fractional: [0.0, 0.0, 0.0],
                    global_umklapp: [0, 0, 0],
                    k_minus_q: vec![
                        MldumpKMinusQV1 {
                            k_index: 0,
                            mapped_index: 0,
                            g_wrap: [0, 0, 0],
                        },
                        MldumpKMinusQV1 {
                            k_index: 1,
                            mapped_index: 1,
                            g_wrap: [0, 0, 0],
                        },
                    ],
                },
                MldumpQEntryV1 {
                    input_fractional: [1.5, 0.0, 0.0],
                    canonical_fractional: [0.5, 0.0, 0.0],
                    global_umklapp: [1, 0, 0],
                    k_minus_q: vec![
                        MldumpKMinusQV1 {
                            k_index: 0,
                            mapped_index: 1,
                            g_wrap: [-1, 0, 0],
                        },
                        MldumpKMinusQV1 {
                            k_index: 1,
                            mapped_index: 0,
                            g_wrap: [0, 0, 0],
                        },
                    ],
                },
            ],
        },
    )
}

fn bx() -> f64 {
    2.0 * PI / 8.0
}

fn fill_complex(n: usize, seed: f64) -> Vec<f64> {
    (0..n)
        .flat_map(|index| [seed + index as f64, seed + index as f64 + 0.5])
        .collect()
}

fn group_status(file: &File, group: &str) -> String {
    let value: VarLenUnicode = file
        .group(group)
        .unwrap()
        .attr("status")
        .unwrap()
        .read_scalar()
        .unwrap();
    value.as_str().to_owned()
}

fn axes(file: &File, dataset: &str) -> Vec<String> {
    file.dataset(dataset)
        .unwrap()
        .attr("axes")
        .unwrap()
        .read_raw::<VarLenUnicode>()
        .unwrap()
        .iter()
        .map(|value| value.as_str().to_owned())
        .collect()
}

struct OrbitalStore {
    evals: Vec<Vec<Vec<f64>>>,
    evecs: Vec<Vec<Vec<f64>>>,
    g: Vec<Vec<i32>>,
    k_cart: Vec<Vec<f64>>,
    q_cart: Vec<Vec<f64>>,
    lm_l: Vec<i32>,
    lm_m: Vec<i32>,
    matching: Vec<Vec<Vec<Vec<f64>>>>,
    lo_row: Vec<Vec<i64>>,
    lo_site: Vec<i64>,
    lo_l: Vec<i64>,
    lo_m: Vec<i64>,
    lo_ord: Vec<i64>,
    lo_n: Vec<i64>,
}

fn k_record<'a>(
    store: &'a OrbitalStore,
    spin: usize,
    k: usize,
    matches: &'a [ScalarApwSiteMatchRefV1<'a>],
) -> ScalarOrbitalKRefV1<'a> {
    let n_pw = if k == 0 { 2 } else { 3 };
    ScalarOrbitalKRefV1 {
        k_index: k,
        available_bands: if k == 0 { 3 } else { 2 },
        basis_dimension: n_pw + 1,
        eigenvalues: &store.evals[spin][k],
        eigenvectors: &store.evecs[spin][k],
        n_plane_waves: n_pw,
        plane_wave_g: &store.g[k],
        plane_wave_k_cartesian: &store.k_cart[k],
        plane_wave_q_cartesian: &store.q_cart[k],
        site_matches: matches,
        local_orbitals: ScalarLocalOrbitalTableRefV1 {
            n_local_orbitals: 1,
            row_index: &store.lo_row[k],
            site: &store.lo_site,
            l: &store.lo_l,
            m: &store.lo_m,
            ordinal: &store.lo_ord,
            radial_n: &store.lo_n,
        },
    }
}

fn site_matches<'a>(
    store: &'a OrbitalStore,
    spin: usize,
    k: usize,
) -> [ScalarApwSiteMatchRefV1<'a>; 2] {
    [
        ScalarApwSiteMatchRefV1 {
            site_index: 0,
            n_lm: 1,
            lm_l: &store.lm_l,
            lm_m: &store.lm_m,
            matching_coefficients: &store.matching[spin][k][0],
        },
        ScalarApwSiteMatchRefV1 {
            site_index: 1,
            n_lm: 1,
            lm_l: &store.lm_l,
            lm_m: &store.lm_m,
            matching_coefficients: &store.matching[spin][k][1],
        },
    ]
}

impl OrbitalStore {
    fn new() -> Self {
        let n_pw = [2usize, 3];
        let mut g = vec![Vec::new(); 2];
        let mut k_cart = vec![Vec::new(); 2];
        let mut q_cart = vec![Vec::new(); 2];
        for k in 0..2 {
            let k_x = if k == 0 { 0.0 } else { 0.5 * bx() };
            for pw in 0..n_pw[k] {
                let gx = i32::try_from(pw).unwrap();
                g[k].extend_from_slice(&[gx, 0, 0]);
                k_cart[k].extend_from_slice(&[k_x, 0.0, 0.0]);
                q_cart[k].extend_from_slice(&[k_x + f64::from(gx) * bx(), 0.0, 0.0]);
            }
        }
        let mut evals = vec![vec![Vec::new(); 2]; 2];
        let mut evecs = vec![vec![Vec::new(); 2]; 2];
        let mut matching = vec![vec![Vec::<Vec<f64>>::new(); 2]; 2];
        let mut lo_row = vec![Vec::new(); 2];
        for spin in 0..2 {
            for k in 0..2 {
                let basis = n_pw[k] + 1;
                evals[spin][k] = vec![
                    0.1 + spin as f64 + k as f64 * 0.01,
                    0.2 + spin as f64 + k as f64 * 0.01,
                ];
                evecs[spin][k] =
                    fill_complex(basis * 2, 10.0 + 100.0 * spin as f64 + 10.0 * k as f64);
                for site in 0..2 {
                    matching[spin][k].push(fill_complex(n_pw[k] * 2, 3.0 + site as f64));
                }
                lo_row[k] = vec![i64::try_from(n_pw[k]).unwrap()];
            }
        }
        Self {
            evals,
            evecs,
            g,
            k_cart,
            q_cart,
            lm_l: vec![0],
            lm_m: vec![0],
            matching,
            lo_row,
            lo_site: vec![0],
            lo_l: vec![0],
            lo_m: vec![0],
            lo_ord: vec![0],
            lo_n: vec![2],
        }
    }
}

#[test]
fn mldump_scalar_v1_roundtrip_has_inspectable_hdf5_payload() {
    let path = fixture_path("libmuffintin-mldump-scalar-v1-fixture.h5");
    let header = header();
    let orbitals = OrbitalStore::new();
    let n_pw = [2usize, 3];
    let match_00 = site_matches(&orbitals, 0, 0);
    let match_01 = site_matches(&orbitals, 0, 1);
    let match_10 = site_matches(&orbitals, 1, 0);
    let match_11 = site_matches(&orbitals, 1, 1);
    let k0_s0 = k_record(&orbitals, 0, 0, &match_00);
    let k1_s0 = k_record(&orbitals, 0, 1, &match_01);
    let k0_s1 = k_record(&orbitals, 1, 0, &match_10);
    let k1_s1 = k_record(&orbitals, 1, 1, &match_11);
    let k_points_0 = [k0_s0, k1_s0];
    let k_points_1 = [k0_s1, k1_s1];
    let spin0 = ScalarOrbitalSpinRefV1 {
        spin: 0,
        k_points: &k_points_0,
    };
    let spin1 = ScalarOrbitalSpinRefV1 {
        spin: 1,
        k_points: &k_points_1,
    };
    let spins = [spin0, spin1];

    let site_indices = [0_i64, 1];
    let site_positions = [0.0, 0.0, 0.0, 4.0, 0.0, 0.0];
    let site_radii = [1.0, 1.2];
    let site0_kind = [0_i64, 0, 0];
    let site0_l = [0_i64, 0, 0];
    let site0_n = [0_i64, 1, 2];
    let site0_spin = [0_i64, 0, 0];
    let site0_large: Vec<f64> = (0..12).map(|i| 0.01 * i as f64).collect();
    let site0_small: Vec<f64> = (0..12).map(|i| 0.001 * i as f64).collect();
    let site1_kind = [0_i64, 0];
    let site1_l = [0_i64, 0];
    let site1_n = [0_i64, 1];
    let site1_spin = [0_i64, 0];
    let site1_large: Vec<f64> = (0..6).map(|i| 0.02 * i as f64).collect();
    let raw0 = [0_i32, 0, 0, 1, 0, 0];
    let raw1 = [0_i32, 0, 0, -1, 0, 0];
    let q1_cart = [0.5 * bx(), 0.0, 0.0];
    let interstitial = 512.0 - 4.0 / 3.0 * PI * (1.0 + 1.2_f64.powi(3));

    let parent_xyz = [0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 4.1, 0.0, 0.0, 2.0, 2.0, 2.0];
    let parent_w = [0.1, 0.0, 0.2, 0.3];
    let parent_kind = [
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_INTERSTITIAL,
    ];
    let parent_site = [0, 0, 1, MLDUMP_INTERSTITIAL_SENTINEL];
    let parent_radial = [0, 1, 0, MLDUMP_INTERSTITIAL_SENTINEL];
    let pivots = [2_i64, 0];
    let points = [0_i64, 2];
    let zeta0 = fill_complex(4 * 2, 20.0);
    let zeta1 = fill_complex(4 * 2, 40.0);
    let vertex_col = [0_i64, 3];
    let vertex_klr = [0_i64, 0, 0, 0, 1, 1];
    let vertex_c0 = fill_complex(2 * 2, 0.5);
    let vertex_c1 = fill_complex(2 * 2, 1.5);
    let body0 = [1.0, 0.0, 0.2, -0.3, 0.2, 0.3, 2.0, 0.0];
    let body1 = [3.0, 0.0, 0.1, 0.4, 0.1, -0.4, 4.0, 0.0];
    let gamma_c = [1.0, 0.0, 0.25, 0.5];
    let product_sites = [
        ScalarProductSiteRefV1 {
            site_index: 0,
            n_radial: 3,
            n_radial_samples: 4,
            kind: &site0_kind,
            l: &site0_l,
            n: &site0_n,
            spin: &site0_spin,
            large: &site0_large,
            small: Some(&site0_small),
        },
        ScalarProductSiteRefV1 {
            site_index: 1,
            n_radial: 2,
            n_radial_samples: 3,
            kind: &site1_kind,
            l: &site1_l,
            n: &site1_n,
            spin: &site1_spin,
            large: &site1_large,
            small: None,
        },
    ];
    let product_qs = [
        ScalarProductQRecordRefV1 {
            q_index: 0,
            transfer_cartesian: [0.0, 0.0, 0.0],
            global_transfer: [0, 0, 0],
            n_raw_g: 2,
            raw_relative_g: &raw0,
            provenance: "q0-raw-g",
        },
        ScalarProductQRecordRefV1 {
            q_index: 1,
            transfer_cartesian: q1_cart,
            global_transfer: [1, 0, 0],
            n_raw_g: 2,
            raw_relative_g: &raw1,
            provenance: "q1-raw-g",
        },
    ];
    let thc_qs = [
        ScalarThcQRecordRefV1 {
            q_index: 0,
            aux_dimension: 2,
            layout_provenance: "aux-layout-q0",
            zeta: &zeta0,
            residual_l2_all_frobenius: 1.0e-8,
            residual_l2_all_column_max: 2.0e-8,
            vertices: ScalarThcVertexTableRefV1 {
                n_vertex: 2,
                column: &vertex_col,
                k_left_right: &vertex_klr,
                coefficients: &vertex_c0,
            },
        },
        ScalarThcQRecordRefV1 {
            q_index: 1,
            aux_dimension: 2,
            layout_provenance: "aux-layout-q1",
            zeta: &zeta1,
            residual_l2_all_frobenius: 3.0e-8,
            residual_l2_all_column_max: 4.0e-8,
            vertices: ScalarThcVertexTableRefV1 {
                n_vertex: 2,
                column: &vertex_col,
                k_left_right: &vertex_klr,
                coefficients: &vertex_c1,
            },
        },
    ];
    let coulomb_qs = [
        ScalarCoulombQRecordRefV1 {
            q_index: 0,
            aux_dimension: 2,
            layout_provenance: "aux-layout-q0",
            body: &body0,
            gamma: Some(ScalarCoulombGammaRefV1 {
                spherical_average_subtracted: true,
                head_prefactor: 4.0 * PI,
                constant_coefficients: &gamma_c,
            }),
        },
        ScalarCoulombQRecordRefV1 {
            q_index: 1,
            aux_dimension: 2,
            layout_provenance: "aux-layout-q1",
            body: &body1,
            gamma: None,
        },
    ];

    {
        let mut writer = MldumpWriterV1::create(&path, &header).unwrap();
        writer
            .write_scalar_orbitals(&ScalarOrbitalsRefV1 {
                spin_count: 2,
                band_window_start: 0,
                band_window_count: 2,
                spins: &spins,
            })
            .unwrap();
        writer
            .write_scalar_products(&ScalarProductsRefV1 {
                n_k: 2,
                n_orb: 2,
                provenance_recipe: "scalar-product-test",
                provenance_reference: "m-l6b1-neutral-fixture",
                site_indices: &site_indices,
                site_positions: &site_positions,
                site_radii: &site_radii,
                interstitial_volume_bohr3: interstitial,
                sites: &product_sites,
                q_records: &product_qs,
            })
            .unwrap();
        writer
            .write_scalar_thc(&ScalarThcRefV1 {
                parent_grid: ScalarThcParentGridRefV1 {
                    n_points: 4,
                    coordinates: &parent_xyz,
                    weights: &parent_w,
                    region_kind: &parent_kind,
                    site_index: &parent_site,
                    radial_index: &parent_radial,
                    provenance: "neutral-parent-grid",
                },
                strategy: MLDUMP_THC_STRATEGY_ALL_QL2,
                engine: MLDUMP_THC_ENGINE_QRCP,
                requested_rank: 2,
                effective_rank: 2,
                n_candidates: 3,
                selection: ScalarThcSelectionRefV1 {
                    pivots: &pivots,
                    points: &points,
                },
                q_records: &thc_qs,
            })
            .unwrap();
        writer
            .write_scalar_coulomb(&ScalarCoulombRefV1 {
                lexp: 4,
                interpolation_l_max: 2,
                interpolation_pw_cutoff: 2.0,
                q_records: &coulomb_qs,
            })
            .unwrap();
        writer.finish().unwrap();
    }

    let read = read_mldump_v1(&path).unwrap();
    assert_eq!(read.header, header);
    let scalar = read.scalar.expect("scalar payload");
    assert_eq!(scalar.orbitals.spin_count, 2);
    assert_eq!(scalar.orbitals.band_window_start, 0);
    assert_eq!(scalar.orbitals.band_window_count, 2);
    assert_eq!(scalar.orbitals.spins[0].k_points[0].available_bands, 3);
    assert_eq!(scalar.orbitals.spins[0].k_points[1].available_bands, 2);
    assert_eq!(scalar.orbitals.spins[0].k_points[0].eigenvalues.len(), 2);
    assert_eq!(scalar.orbitals.spins[0].k_points[0].basis_dimension, 3);
    assert_eq!(scalar.orbitals.spins[0].k_points[1].basis_dimension, 4);
    assert_eq!(
        scalar.orbitals.spins[0].k_points[0].eigenvalues,
        orbitals.evals[0][0]
    );
    assert_eq!(
        scalar.orbitals.spins[1].k_points[1].eigenvectors,
        orbitals.evecs[1][1]
    );
    assert_eq!(
        scalar.orbitals.spins[0].k_points[1].plane_wave_g.len(),
        n_pw[1]
    );
    assert_eq!(
        scalar.orbitals.spins[0].k_points[0].local_orbitals[0].radial_n,
        2
    );
    assert_eq!(scalar.products.n_k, 2);
    assert_eq!(scalar.products.n_orb, 2);
    assert_eq!(scalar.products.sites[0].n, site0_n);
    assert_eq!(scalar.products.sites[0].large, site0_large);
    assert_eq!(
        scalar.products.sites[0].small.as_deref(),
        Some(site0_small.as_slice())
    );
    assert!(scalar.products.sites[1].small.is_none());
    assert_eq!(scalar.products.q_records[1].global_transfer, [1, 0, 0]);
    assert_eq!(scalar.thc.engine, MLDUMP_THC_ENGINE_QRCP);
    assert_eq!(scalar.thc.selection.pivots, pivots);
    assert_eq!(scalar.thc.selection.points, points);
    assert_ne!(scalar.thc.selection.pivots, scalar.thc.selection.points);
    assert_eq!(scalar.thc.parent_grid.weights[1], 0.0);
    assert_eq!(scalar.thc.q_records[0].zeta, zeta0);
    assert_eq!(scalar.thc.q_records[1].vertices[1].column, 3);
    assert_eq!(
        (
            scalar.thc.q_records[0].vertices[1].k,
            scalar.thc.q_records[0].vertices[1].left,
            scalar.thc.q_records[0].vertices[1].right
        ),
        (0, 1, 1)
    );
    assert_eq!(
        scalar.thc.q_records[0].layout_provenance,
        scalar.coulomb.q_records[0].layout_provenance
    );
    assert_eq!(
        scalar.thc.q_records[1].layout_provenance,
        scalar.coulomb.q_records[1].layout_provenance
    );
    assert_eq!(scalar.coulomb.lexp, 4);
    assert_eq!(scalar.coulomb.q_records[0].body, body0);
    assert!(scalar.coulomb.q_records[0].gamma.is_some());
    assert!(scalar.coulomb.q_records[1].gamma.is_none());
    assert_eq!(
        scalar.coulomb.q_records[0]
            .gamma
            .as_ref()
            .unwrap()
            .constant_coefficients,
        gamma_c
    );

    let file = File::open(&path).unwrap();
    assert_eq!(group_status(&file, "orbitals"), MLDUMP_STATUS_PRESENT);
    assert_eq!(group_status(&file, "products"), MLDUMP_STATUS_PRESENT);
    assert_eq!(group_status(&file, "thc"), MLDUMP_STATUS_PRESENT);
    assert_eq!(group_status(&file, "coulomb"), MLDUMP_STATUS_PRESENT);
    assert_eq!(
        group_status(&file, "mpb"),
        MLDUMP_STATUS_ABSENT_NOT_COMPUTED
    );
    assert!(
        file.group("mpb")
            .unwrap()
            .member_names()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        group_status(&file, "exchange/valence"),
        MLDUMP_STATUS_ABSENT_NOT_COMPUTED
    );
    assert!(file.group("orbitals").unwrap().link_exists("spin_000000"));
    assert!(
        file.group("orbitals/spin_000000")
            .unwrap()
            .link_exists("k_000001")
    );
    assert!(file.group("products").unwrap().link_exists("site_000000"));
    assert!(file.group("thc").unwrap().link_exists("q_000001"));
    assert!(file.group("coulomb/q_000000").unwrap().link_exists("gamma"));
    assert_eq!(
        group_status(&file, "coulomb/q_000000/gamma"),
        MLDUMP_STATUS_PRESENT
    );
    assert_eq!(
        group_status(&file, "coulomb/q_000001/gamma"),
        MLDUMP_STATUS_ABSENT_NOT_COMPUTED
    );
    assert!(
        file.group("coulomb/q_000001/gamma")
            .unwrap()
            .member_names()
            .unwrap()
            .is_empty()
    );

    let representation: VarLenUnicode = file
        .group("orbitals")
        .unwrap()
        .attr("representation")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(
        representation.as_str(),
        MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON
    );
    let occupations: VarLenUnicode = file
        .group("orbitals")
        .unwrap()
        .attr("occupations_status")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(occupations.as_str(), MLDUMP_OCCUPATIONS_NOT_EXPORTED);
    let pair_order: VarLenUnicode = file
        .group("products")
        .unwrap()
        .attr("pair_order")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(pair_order.as_str(), MLDUMP_PAIR_ORDER_K_LEFT_RIGHT);
    let core_status: VarLenUnicode = file
        .group("products")
        .unwrap()
        .attr("core_status")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(core_status.as_str(), MLDUMP_CORE_EMPTY_NOT_FITTED);
    let engine: VarLenUnicode = file
        .group("thc")
        .unwrap()
        .attr("engine")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(engine.as_str(), MLDUMP_THC_ENGINE_QRCP);
    let strategy: VarLenUnicode = file
        .group("thc")
        .unwrap()
        .attr("strategy")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(strategy.as_str(), MLDUMP_THC_STRATEGY_ALL_QL2);

    assert!(
        file.dataset("orbitals/spin_000000/k_000000/eigenvalues")
            .unwrap()
            .dtype()
            .unwrap()
            .is::<f64>()
    );
    assert!(
        file.dataset("orbitals/spin_000001/k_000001/basis/plane_wave_g")
            .unwrap()
            .dtype()
            .unwrap()
            .is::<i32>()
    );
    let available_k0: i64 = file
        .group("orbitals/spin_000000/k_000000")
        .unwrap()
        .attr("available_bands")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(available_k0, 3);
    assert_eq!(
        file.dataset("orbitals/spin_000000/k_000000/eigenvalues")
            .unwrap()
            .shape(),
        [2]
    );
    assert_eq!(
        file.dataset("orbitals/spin_000000/k_000000/eigenvectors")
            .unwrap()
            .shape(),
        [3, 2, 2]
    );
    assert_eq!(
        file.dataset("orbitals/spin_000000/k_000001/eigenvectors")
            .unwrap()
            .shape(),
        [4, 2, 2]
    );
    assert_eq!(
        axes(&file, "orbitals/spin_000000/k_000000/eigenvectors"),
        ["basis_row", "band", "re_im"]
    );
    assert_eq!(
        axes(
            &file,
            "orbitals/spin_000000/k_000000/basis/site_000000/matching_coefficients"
        ),
        ["plane_wave", "lm", "radial_component", "re_im"]
    );
    assert_eq!(
        file.dataset("thc/parent_grid/weights").unwrap().shape(),
        [4]
    );
    let weights = file
        .dataset("thc/parent_grid/weights")
        .unwrap()
        .read_raw::<f64>()
        .unwrap();
    assert_eq!(weights[1], 0.0);
    let stored_pivots = file
        .dataset("thc/pivots")
        .unwrap()
        .read_raw::<i64>()
        .unwrap();
    let stored_points = file
        .dataset("thc/points")
        .unwrap()
        .read_raw::<i64>()
        .unwrap();
    assert_eq!(stored_pivots, pivots);
    assert_eq!(stored_points, points);
    assert_ne!(stored_pivots, stored_points);
    assert_eq!(
        axes(&file, "thc/q_000000/zeta"),
        ["parent_point", "aux", "re_im"]
    );
    assert_eq!(
        file.dataset("thc/q_000000/zeta").unwrap().shape(),
        [4, 2, 2]
    );
    let zeta_raw = file
        .dataset("thc/q_000000/zeta")
        .unwrap()
        .read_raw::<f64>()
        .unwrap();
    assert_eq!(zeta_raw, zeta0);
    assert_eq!(
        axes(&file, "coulomb/q_000000/body"),
        ["aux_row", "aux_col", "re_im"]
    );
    let body_raw = file
        .dataset("coulomb/q_000000/body")
        .unwrap()
        .read_raw::<f64>()
        .unwrap();
    assert_eq!(body_raw, body0);
    let wrap = file
        .dataset("mesh/k_minus_q_g_wrap")
        .unwrap()
        .read_raw::<i32>()
        .unwrap();
    assert_eq!(&wrap[6..9], [-1, 0, 0]);
}

#[test]
fn mldump_scalar_v1_rejects_aux_vertex_shape_before_thc_write() {
    let path = fixture_path("libmuffintin-mldump-scalar-v1-bad-aux.h5");
    let header = header();
    let parent_xyz = [0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 4.1, 0.0, 0.0, 2.0, 2.0, 2.0];
    let parent_w = [0.1, 0.0, 0.2, 0.3];
    let parent_kind = [
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_INTERSTITIAL,
    ];
    let parent_site = [0, 0, 1, MLDUMP_INTERSTITIAL_SENTINEL];
    let parent_radial = [0, 1, 0, MLDUMP_INTERSTITIAL_SENTINEL];
    let pivots = [2_i64, 0];
    let points = [0_i64, 2];
    let zeta = fill_complex(4 * 2, 1.0);
    let vertex_col = [0_i64];
    let vertex_klr = [0_i64, 0, 0];
    let bad_coefficients = fill_complex(3, 0.0);
    let good_q1_coeff = fill_complex(2, 0.0);
    let bad_qs = [
        ScalarThcQRecordRefV1 {
            q_index: 0,
            aux_dimension: 2,
            layout_provenance: "aux-layout-q0",
            zeta: &zeta,
            residual_l2_all_frobenius: 0.0,
            residual_l2_all_column_max: 0.0,
            vertices: ScalarThcVertexTableRefV1 {
                n_vertex: 1,
                column: &vertex_col,
                k_left_right: &vertex_klr,
                coefficients: &bad_coefficients,
            },
        },
        ScalarThcQRecordRefV1 {
            q_index: 1,
            aux_dimension: 2,
            layout_provenance: "aux-layout-q1",
            zeta: &zeta,
            residual_l2_all_frobenius: 0.0,
            residual_l2_all_column_max: 0.0,
            vertices: ScalarThcVertexTableRefV1 {
                n_vertex: 1,
                column: &vertex_col,
                k_left_right: &vertex_klr,
                coefficients: &good_q1_coeff,
            },
        },
    ];
    let mut writer = MldumpWriterV1::create(&path, &header).unwrap();
    let error = writer
        .write_scalar_thc(&ScalarThcRefV1 {
            parent_grid: ScalarThcParentGridRefV1 {
                n_points: 4,
                coordinates: &parent_xyz,
                weights: &parent_w,
                region_kind: &parent_kind,
                site_index: &parent_site,
                radial_index: &parent_radial,
                provenance: "neutral-parent-grid",
            },
            strategy: MLDUMP_THC_STRATEGY_ALL_QL2,
            engine: MLDUMP_THC_ENGINE_QRCP,
            requested_rank: 2,
            effective_rank: 2,
            n_candidates: 3,
            selection: ScalarThcSelectionRefV1 {
                pivots: &pivots,
                points: &points,
            },
            q_records: &bad_qs,
        })
        .unwrap_err();
    match error {
        IoError::Validation(ValidationError::LengthMismatch {
            path,
            expected,
            actual,
        }) => {
            assert_eq!(path, "thc.q_records[0].vertices.coefficients");
            assert_eq!(expected, 4);
            assert_eq!(actual, 6);
        }
        other => panic!("expected vertex coefficient LengthMismatch, got {other:?}"),
    }
    drop(writer);
    let file = File::open(&path).unwrap();
    assert_eq!(
        group_status(&file, "thc"),
        MLDUMP_STATUS_ABSENT_NOT_COMPUTED
    );
    assert!(
        file.group("thc")
            .unwrap()
            .member_names()
            .unwrap()
            .is_empty()
    );
}

fn write_complete_scalar(
    path: &std::path::Path,
    vertex_col: [&[i64]; 2],
    vertex_klr: [&[i64]; 2],
    parent_w: &[f64],
    pivots: &[i64],
    points: &[i64],
) -> Result<(), IoError> {
    let header = header();
    let orbitals = OrbitalStore::new();
    let match_00 = site_matches(&orbitals, 0, 0);
    let match_01 = site_matches(&orbitals, 0, 1);
    let match_10 = site_matches(&orbitals, 1, 0);
    let match_11 = site_matches(&orbitals, 1, 1);
    let k0_s0 = k_record(&orbitals, 0, 0, &match_00);
    let k1_s0 = k_record(&orbitals, 0, 1, &match_01);
    let k0_s1 = k_record(&orbitals, 1, 0, &match_10);
    let k1_s1 = k_record(&orbitals, 1, 1, &match_11);
    let k_points_0 = [k0_s0, k1_s0];
    let k_points_1 = [k0_s1, k1_s1];
    let spins = [
        ScalarOrbitalSpinRefV1 {
            spin: 0,
            k_points: &k_points_0,
        },
        ScalarOrbitalSpinRefV1 {
            spin: 1,
            k_points: &k_points_1,
        },
    ];
    let site_indices = [0_i64, 1];
    let site_positions = [0.0, 0.0, 0.0, 4.0, 0.0, 0.0];
    let site_radii = [1.0, 1.2];
    let site0_kind = [0_i64, 0, 0];
    let site0_l = [0_i64, 0, 0];
    let site0_n = [0_i64, 1, 2];
    let site0_spin = [0_i64, 0, 0];
    let site0_large: Vec<f64> = (0..12).map(|i| 0.01 * i as f64).collect();
    let site0_small: Vec<f64> = (0..12).map(|i| 0.001 * i as f64).collect();
    let site1_kind = [0_i64, 0];
    let site1_l = [0_i64, 0];
    let site1_n = [0_i64, 1];
    let site1_spin = [0_i64, 0];
    let site1_large: Vec<f64> = (0..6).map(|i| 0.02 * i as f64).collect();
    let raw0 = [0_i32, 0, 0, 1, 0, 0];
    let raw1 = [0_i32, 0, 0, -1, 0, 0];
    let q1_cart = [0.5 * bx(), 0.0, 0.0];
    let interstitial = 512.0 - 4.0 / 3.0 * PI * (1.0 + 1.2_f64.powi(3));
    let parent_xyz = [0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 4.1, 0.0, 0.0, 2.0, 2.0, 2.0];
    let parent_kind = [
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_INTERSTITIAL,
    ];
    let parent_site = [0, 0, 1, MLDUMP_INTERSTITIAL_SENTINEL];
    let parent_radial = [0, 1, 0, MLDUMP_INTERSTITIAL_SENTINEL];
    let zeta0 = fill_complex(4 * 2, 20.0);
    let zeta1 = fill_complex(4 * 2, 40.0);
    let vertex_c0 = fill_complex(2 * 2, 0.5);
    let vertex_c1 = fill_complex(2 * 2, 1.5);
    let body0 = [1.0, 0.0, 0.2, -0.3, 0.2, 0.3, 2.0, 0.0];
    let body1 = [3.0, 0.0, 0.1, 0.4, 0.1, -0.4, 4.0, 0.0];
    let gamma_c = [1.0, 0.0, 0.25, 0.5];
    let product_sites = [
        ScalarProductSiteRefV1 {
            site_index: 0,
            n_radial: 3,
            n_radial_samples: 4,
            kind: &site0_kind,
            l: &site0_l,
            n: &site0_n,
            spin: &site0_spin,
            large: &site0_large,
            small: Some(&site0_small),
        },
        ScalarProductSiteRefV1 {
            site_index: 1,
            n_radial: 2,
            n_radial_samples: 3,
            kind: &site1_kind,
            l: &site1_l,
            n: &site1_n,
            spin: &site1_spin,
            large: &site1_large,
            small: None,
        },
    ];
    let product_qs = [
        ScalarProductQRecordRefV1 {
            q_index: 0,
            transfer_cartesian: [0.0, 0.0, 0.0],
            global_transfer: [0, 0, 0],
            n_raw_g: 2,
            raw_relative_g: &raw0,
            provenance: "q0-raw-g",
        },
        ScalarProductQRecordRefV1 {
            q_index: 1,
            transfer_cartesian: q1_cart,
            global_transfer: [1, 0, 0],
            n_raw_g: 2,
            raw_relative_g: &raw1,
            provenance: "q1-raw-g",
        },
    ];
    let thc_qs = [
        ScalarThcQRecordRefV1 {
            q_index: 0,
            aux_dimension: 2,
            layout_provenance: "aux-layout-q0",
            zeta: &zeta0,
            residual_l2_all_frobenius: 1.0e-8,
            residual_l2_all_column_max: 2.0e-8,
            vertices: ScalarThcVertexTableRefV1 {
                n_vertex: 2,
                column: vertex_col[0],
                k_left_right: vertex_klr[0],
                coefficients: &vertex_c0,
            },
        },
        ScalarThcQRecordRefV1 {
            q_index: 1,
            aux_dimension: 2,
            layout_provenance: "aux-layout-q1",
            zeta: &zeta1,
            residual_l2_all_frobenius: 3.0e-8,
            residual_l2_all_column_max: 4.0e-8,
            vertices: ScalarThcVertexTableRefV1 {
                n_vertex: 2,
                column: vertex_col[1],
                k_left_right: vertex_klr[1],
                coefficients: &vertex_c1,
            },
        },
    ];
    let coulomb_qs = [
        ScalarCoulombQRecordRefV1 {
            q_index: 0,
            aux_dimension: 2,
            layout_provenance: "aux-layout-q0",
            body: &body0,
            gamma: Some(ScalarCoulombGammaRefV1 {
                spherical_average_subtracted: true,
                head_prefactor: 4.0 * PI,
                constant_coefficients: &gamma_c,
            }),
        },
        ScalarCoulombQRecordRefV1 {
            q_index: 1,
            aux_dimension: 2,
            layout_provenance: "aux-layout-q1",
            body: &body1,
            gamma: None,
        },
    ];
    let mut writer = MldumpWriterV1::create(path, &header)?;
    writer.write_scalar_orbitals(&ScalarOrbitalsRefV1 {
        spin_count: 2,
        band_window_start: 0,
        band_window_count: 2,
        spins: &spins,
    })?;
    writer.write_scalar_products(&ScalarProductsRefV1 {
        n_k: 2,
        n_orb: 2,
        provenance_recipe: "scalar-product-test",
        provenance_reference: "m-l6b1-neutral-fixture",
        site_indices: &site_indices,
        site_positions: &site_positions,
        site_radii: &site_radii,
        interstitial_volume_bohr3: interstitial,
        sites: &product_sites,
        q_records: &product_qs,
    })?;
    writer.write_scalar_thc(&ScalarThcRefV1 {
        parent_grid: ScalarThcParentGridRefV1 {
            n_points: 4,
            coordinates: &parent_xyz,
            weights: parent_w,
            region_kind: &parent_kind,
            site_index: &parent_site,
            radial_index: &parent_radial,
            provenance: "neutral-parent-grid",
        },
        strategy: MLDUMP_THC_STRATEGY_ALL_QL2,
        engine: MLDUMP_THC_ENGINE_QRCP,
        requested_rank: 2,
        effective_rank: 2,
        n_candidates: 3,
        selection: ScalarThcSelectionRefV1 { pivots, points },
        q_records: &thc_qs,
    })?;
    writer.write_scalar_coulomb(&ScalarCoulombRefV1 {
        lexp: 4,
        interpolation_l_max: 2,
        interpolation_pw_cutoff: 2.0,
        q_records: &coulomb_qs,
    })?;
    writer.finish()
}

#[test]
fn mldump_scalar_v1_rejects_semantic_vertex_mismatch_at_finish_and_read() {
    let path = fixture_path("libmuffintin-mldump-scalar-v1-vertex-mismatch.h5");
    let good_col = [0_i64, 3];
    let good_klr = [0_i64, 0, 0, 0, 1, 1];
    let bad_col = [0_i64, 2];
    let parent_w = [0.1, 0.0, 0.2, 0.3];
    let pivots = [2_i64, 0];
    let points = [0_i64, 2];
    let error = write_complete_scalar(
        &path,
        [&good_col, &bad_col],
        [&good_klr, &good_klr],
        &parent_w,
        &pivots,
        &points,
    )
    .unwrap_err();
    match error {
        IoError::Validation(ValidationError::InvalidValue {
            path,
            expected,
            actual,
        }) => {
            assert_eq!(path, "thc.q_records[1].vertices[1].k_left_right");
            assert!(expected.contains("decode(column=2)"), "{expected}");
            assert_eq!(actual, "(0,1,1)");
        }
        other => panic!("expected vertex identity InvalidValue, got {other:?}"),
    }
    match read_mldump_v1(&path) {
        Err(IoError::Validation(ValidationError::InvalidValue { path, actual, .. })) => {
            assert_eq!(path, "thc.q_records[1].vertices[1].k_left_right");
            assert_eq!(actual, "(0,1,1)");
        }
        other => panic!("expected reader vertex identity InvalidValue, got {other:?}"),
    }
}

#[test]
fn mldump_scalar_v1_rejects_bad_parent_selection() {
    struct Case {
        label: &'static str,
        weights: [f64; 4],
        pivots: [i64; 2],
        points: [i64; 2],
        path_contains: &'static str,
        expected_contains: &'static str,
    }
    let cases = [
        Case {
            label: "negative_weight",
            weights: [0.1, -0.2, 0.2, 0.3],
            pivots: [2, 0],
            points: [0, 2],
            path_contains: "thc.parent_grid.weights[1]",
            expected_contains: "nonnegative",
        },
        Case {
            label: "nan_weight",
            weights: [0.1, f64::NAN, 0.2, 0.3],
            pivots: [2, 0],
            points: [0, 2],
            path_contains: "thc.parent_grid.weights[1]",
            expected_contains: "",
        },
        Case {
            label: "zero_weight_point",
            weights: [0.1, 0.0, 0.2, 0.3],
            pivots: [2, 0],
            points: [0, 1],
            path_contains: "thc.selection.points[1]",
            expected_contains: "strictly positive parent weight",
        },
    ];
    let parent_xyz = [0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 4.1, 0.0, 0.0, 2.0, 2.0, 2.0];
    let parent_kind = [
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_INTERSTITIAL,
    ];
    let parent_site = [0, 0, 1, MLDUMP_INTERSTITIAL_SENTINEL];
    let parent_radial = [0, 1, 0, MLDUMP_INTERSTITIAL_SENTINEL];
    let zeta = fill_complex(4 * 2, 1.0);
    let vertex_col = [0_i64, 3];
    let vertex_klr = [0_i64, 0, 0, 0, 1, 1];
    let vertex_c = fill_complex(2 * 2, 0.5);
    for case in cases {
        let path = fixture_path(&format!(
            "libmuffintin-mldump-scalar-v1-bad-parent-{}.h5",
            case.label
        ));
        let header = header();
        let thc_qs = [
            ScalarThcQRecordRefV1 {
                q_index: 0,
                aux_dimension: 2,
                layout_provenance: "aux-layout-q0",
                zeta: &zeta,
                residual_l2_all_frobenius: 0.0,
                residual_l2_all_column_max: 0.0,
                vertices: ScalarThcVertexTableRefV1 {
                    n_vertex: 2,
                    column: &vertex_col,
                    k_left_right: &vertex_klr,
                    coefficients: &vertex_c,
                },
            },
            ScalarThcQRecordRefV1 {
                q_index: 1,
                aux_dimension: 2,
                layout_provenance: "aux-layout-q1",
                zeta: &zeta,
                residual_l2_all_frobenius: 0.0,
                residual_l2_all_column_max: 0.0,
                vertices: ScalarThcVertexTableRefV1 {
                    n_vertex: 2,
                    column: &vertex_col,
                    k_left_right: &vertex_klr,
                    coefficients: &vertex_c,
                },
            },
        ];
        let mut writer = MldumpWriterV1::create(&path, &header).unwrap();
        let error = writer
            .write_scalar_thc(&ScalarThcRefV1 {
                parent_grid: ScalarThcParentGridRefV1 {
                    n_points: 4,
                    coordinates: &parent_xyz,
                    weights: &case.weights,
                    region_kind: &parent_kind,
                    site_index: &parent_site,
                    radial_index: &parent_radial,
                    provenance: "neutral-parent-grid",
                },
                strategy: MLDUMP_THC_STRATEGY_ALL_QL2,
                engine: MLDUMP_THC_ENGINE_QRCP,
                requested_rank: 2,
                effective_rank: 2,
                n_candidates: 3,
                selection: ScalarThcSelectionRefV1 {
                    pivots: &case.pivots,
                    points: &case.points,
                },
                q_records: &thc_qs,
            })
            .unwrap_err();
        match error {
            IoError::Validation(ValidationError::InvalidValue { path, expected, .. }) => {
                assert!(
                    path.contains(case.path_contains),
                    "{}: path={path}",
                    case.label
                );
                if !case.expected_contains.is_empty() {
                    assert!(
                        expected.contains(case.expected_contains),
                        "{}: expected={expected}",
                        case.label
                    );
                }
            }
            IoError::Validation(ValidationError::NonFinite { path, .. }) => {
                assert_eq!(case.label, "nan_weight", "{}", case.label);
                assert!(
                    path.contains(case.path_contains),
                    "{}: path={path}",
                    case.label
                );
            }
            other => panic!(
                "{}: expected parent-selection validation, got {other:?}",
                case.label
            ),
        }
        drop(writer);

        let good_col = [0_i64, 3];
        let good_klr = [0_i64, 0, 0, 0, 1, 1];
        let good_w = [0.1, 0.0, 0.2, 0.3];
        let good_pivots = [2_i64, 0];
        let good_points = [0_i64, 2];
        write_complete_scalar(
            &path,
            [&good_col, &good_col],
            [&good_klr, &good_klr],
            &good_w,
            &good_pivots,
            &good_points,
        )
        .unwrap();
        {
            let file = File::open_rw(&path).unwrap();
            file.dataset("thc/parent_grid/weights")
                .unwrap()
                .write_raw(&case.weights)
                .unwrap();
            file.dataset("thc/pivots")
                .unwrap()
                .write_raw(&case.pivots)
                .unwrap();
            file.dataset("thc/points")
                .unwrap()
                .write_raw(&case.points)
                .unwrap();
        }
        match read_mldump_v1(&path) {
            Err(IoError::Validation(ValidationError::InvalidValue { path, expected, .. })) => {
                assert!(
                    path.contains(case.path_contains),
                    "{} reader: path={path}",
                    case.label
                );
                if !case.expected_contains.is_empty() {
                    assert!(
                        expected.contains(case.expected_contains),
                        "{} reader: expected={expected}",
                        case.label
                    );
                }
            }
            Err(IoError::Validation(ValidationError::NonFinite { path, .. })) => {
                assert_eq!(case.label, "nan_weight", "{}", case.label);
                assert!(
                    path.contains(case.path_contains),
                    "{} reader: path={path}",
                    case.label
                );
            }
            other => panic!(
                "{} reader: expected parent-selection validation, got {other:?}",
                case.label
            ),
        }
    }
}

#[test]
fn mldump_scalar_v1_rejects_pivot_point_set_mismatch() {
    let path = fixture_path("libmuffintin-mldump-scalar-v1-selection-set.h5");
    let header = header();
    let parent_xyz = [0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 4.1, 0.0, 0.0, 2.0, 2.0, 2.0];
    let parent_w = [0.1, 0.2, 0.3, 0.4];
    let parent_kind = [
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_MUFFIN_TIN,
        MLDUMP_PARENT_REGION_INTERSTITIAL,
    ];
    let parent_site = [0, 0, 1, MLDUMP_INTERSTITIAL_SENTINEL];
    let parent_radial = [0, 1, 0, MLDUMP_INTERSTITIAL_SENTINEL];
    let pivots = [2_i64, 0];
    let points = [0_i64, 3];
    let zeta = fill_complex(4 * 2, 1.0);
    let vertex_col = [0_i64, 3];
    let vertex_klr = [0_i64, 0, 0, 0, 1, 1];
    let vertex_c = fill_complex(2 * 2, 0.5);
    let thc_qs = [
        ScalarThcQRecordRefV1 {
            q_index: 0,
            aux_dimension: 2,
            layout_provenance: "aux-layout-q0",
            zeta: &zeta,
            residual_l2_all_frobenius: 0.0,
            residual_l2_all_column_max: 0.0,
            vertices: ScalarThcVertexTableRefV1 {
                n_vertex: 2,
                column: &vertex_col,
                k_left_right: &vertex_klr,
                coefficients: &vertex_c,
            },
        },
        ScalarThcQRecordRefV1 {
            q_index: 1,
            aux_dimension: 2,
            layout_provenance: "aux-layout-q1",
            zeta: &zeta,
            residual_l2_all_frobenius: 0.0,
            residual_l2_all_column_max: 0.0,
            vertices: ScalarThcVertexTableRefV1 {
                n_vertex: 2,
                column: &vertex_col,
                k_left_right: &vertex_klr,
                coefficients: &vertex_c,
            },
        },
    ];
    let mut writer = MldumpWriterV1::create(&path, &header).unwrap();
    let error = writer
        .write_scalar_thc(&ScalarThcRefV1 {
            parent_grid: ScalarThcParentGridRefV1 {
                n_points: 4,
                coordinates: &parent_xyz,
                weights: &parent_w,
                region_kind: &parent_kind,
                site_index: &parent_site,
                radial_index: &parent_radial,
                provenance: "neutral-parent-grid",
            },
            strategy: MLDUMP_THC_STRATEGY_ALL_QL2,
            engine: MLDUMP_THC_ENGINE_QRCP,
            requested_rank: 2,
            effective_rank: 2,
            n_candidates: 3,
            selection: ScalarThcSelectionRefV1 {
                pivots: &pivots,
                points: &points,
            },
            q_records: &thc_qs,
        })
        .unwrap_err();
    match error {
        IoError::Validation(ValidationError::InvalidValue {
            path,
            expected,
            actual,
        }) => {
            assert_eq!(path, "thc.selection");
            assert!(expected.contains("same parent indices"), "{expected}");
            assert!(actual.contains('3'), "{actual}");
        }
        other => panic!("expected selection-set InvalidValue, got {other:?}"),
    }
    drop(writer);

    let good_col = [0_i64, 3];
    let good_klr = [0_i64, 0, 0, 0, 1, 1];
    let good_w = [0.1, 0.0, 0.2, 0.3];
    let good_pivots = [2_i64, 0];
    let good_points = [0_i64, 2];
    write_complete_scalar(
        &path,
        [&good_col, &good_col],
        [&good_klr, &good_klr],
        &good_w,
        &good_pivots,
        &good_points,
    )
    .unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        file.dataset("thc/points")
            .unwrap()
            .write_raw(&[0_i64, 3])
            .unwrap();
    }
    match read_mldump_v1(&path) {
        Err(IoError::Validation(ValidationError::InvalidValue { path, actual, .. })) => {
            assert_eq!(path, "thc.selection");
            assert!(actual.contains('3'), "{actual}");
        }
        other => panic!("expected reader selection-set InvalidValue, got {other:?}"),
    }
}
