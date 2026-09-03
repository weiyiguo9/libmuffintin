//! Neutral spinor MLDUMP v1 fixture: HDF5 inspection plus owned-reader equality.

use std::f64::consts::PI;
use std::path::PathBuf;

use hdf5_metno::File;
use hdf5_metno::types::VarLenUnicode;
use muffintin_io::{
    IoError, MLDUMP_INTERSTITIAL_SENTINEL, MLDUMP_OCCUPATIONS_NOT_EXPORTED,
    MLDUMP_PAIR_ORDER_K_LEFT_RIGHT, MLDUMP_PARENT_REGION_INTERSTITIAL,
    MLDUMP_PARENT_REGION_MUFFIN_TIN, MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION,
    MLDUMP_STATUS_ABSENT_NOT_COMPUTED, MLDUMP_STATUS_PRESENT, MLDUMP_THC_ENGINE_QRCP,
    MLDUMP_THC_STRATEGY_ALL_QL2, MldumpCoulombBeginV1, MldumpCoulombGammaRefV1,
    MldumpCoulombQRecordRefV1, MldumpGeometryV1, MldumpHeaderV1, MldumpKMinusQV1, MldumpKPointV1,
    MldumpMeshV1, MldumpMetaV1, MldumpPayloadV1, MldumpQEntryV1, MldumpRadialMeshV1, MldumpSiteV1,
    MldumpThcBeginV1, MldumpThcParentGridRefV1, MldumpThcQRecordRefV1, MldumpThcSelectionRefV1,
    MldumpThcVertexTableRefV1, MldumpWriterV1, SpinorLocalOrbitalTableRefV1, SpinorOrbitalKRefV1,
    SpinorOrbitalsBeginV1, SpinorPauliRowMapRefV1, SpinorProductQRecordRefV1,
    SpinorProductSiteRefV1, SpinorProductsBeginV1, SpinorSiteMatchRefV1, ValidationError,
    read_mldump_v1,
};

fn fixture_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

fn header() -> MldumpHeaderV1 {
    MldumpHeaderV1::new(
        MldumpMetaV1 {
            producer_name: "libmuffintin-io-spinor-test".to_owned(),
            producer_version: "0.1.0".to_owned(),
            source_revision: "76ac7cfac81480655b6b5a124e9b81ab680c51e0".to_owned(),
            feature_representation: MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION.to_owned(),
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

fn attr_str(file: &File, group: &str, name: &str) -> String {
    let value: VarLenUnicode = file
        .group(group)
        .unwrap()
        .attr(name)
        .unwrap()
        .read_scalar()
        .unwrap();
    value.as_str().to_owned()
}

fn pauli_map(n_pw: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let n_row = 2 * n_pw;
    let mut row_index = Vec::with_capacity(n_row);
    let mut component = Vec::with_capacity(n_row);
    let mut pw_index = Vec::with_capacity(n_row);
    for pauli in 0..2 {
        for pw in 0..n_pw {
            row_index.push(i64::try_from(pauli * n_pw + pw).unwrap());
            component.push(i64::try_from(pauli).unwrap());
            pw_index.push(i64::try_from(pw).unwrap());
        }
    }
    (row_index, component, pw_index)
}

fn write_spinor_fixture(path: &std::path::Path) -> Result<(), IoError> {
    let header = header();
    let n_pw = [2usize, 3];
    let available = [2usize, 3];
    let band_window = 2usize;
    let k_frac_x = [0.0, 0.5];
    let mut g = [Vec::new(), Vec::new()];
    let mut k_cart = [Vec::new(), Vec::new()];
    let mut q_cart = [Vec::new(), Vec::new()];
    let mut evals = [Vec::new(), Vec::new()];
    let mut evecs = [Vec::new(), Vec::new()];
    let mut pauli_row = [Vec::new(), Vec::new()];
    let mut pauli_comp = [Vec::new(), Vec::new()];
    let mut pauli_pw = [Vec::new(), Vec::new()];
    let mut match0 = [Vec::new(), Vec::new()];
    let mut match1 = [Vec::new(), Vec::new()];
    for (k, &n) in n_pw.iter().enumerate() {
        let kx = k_frac_x[k] * bx();
        for pw in 0..n {
            let gx = i32::try_from(pw).unwrap();
            g[k].extend_from_slice(&[gx, 0, 0]);
            k_cart[k].extend_from_slice(&[kx, 0.0, 0.0]);
            q_cart[k].extend_from_slice(&[kx + f64::from(gx) * bx(), 0.0, 0.0]);
        }
        let (rows, comps, pws) = pauli_map(n);
        pauli_row[k] = rows;
        pauli_comp[k] = comps;
        pauli_pw[k] = pws;
        let basis = 2 * n + 1;
        evals[k] = vec![0.1 + k as f64, 0.2 + k as f64];
        evecs[k] = fill_complex(basis * band_window, 10.0 + 10.0 * k as f64);
        match0[k] = fill_complex(n * 2 * 4, 3.0 + k as f64);
        match1[k] = fill_complex(n * 2 * 2, 5.0 + k as f64);
    }
    let lo_row = [
        [i64::try_from(2 * n_pw[0]).unwrap()],
        [i64::try_from(2 * n_pw[1]).unwrap()],
    ];
    let lo_site = [0_i64];
    let lo_kappa = [-1_i64];
    let lo_mu = [-1_i64];
    let lo_ord = [0_i64];
    let lo_n = [2_i64];
    let proj0_coord = [0_i64, 1, 2, 3, 4];
    let proj0_kappa = [-1_i64, -1, -1, -1, -1];
    let proj0_mu = [-1_i64, -1, 1, 1, -1];
    let proj0_n = [0_i64, 1, 0, 1, 2];
    let proj1_coord = [0_i64, 1];
    let proj1_kappa = [-1_i64, -1];
    let proj1_mu = [-1_i64, -1];
    let proj1_n = [0_i64, 1];

    let site_indices = [0_i64, 1];
    let site_positions = [0.0, 0.0, 0.0, 4.0, 0.0, 0.0];
    let site_radii = [1.0, 1.2];
    let site0_kind = [0_i64, 0, 0];
    let site0_kappa = [-1_i64, -1, -1];
    let site0_n = [0_i64, 1, 2];
    let site0_p: Vec<f64> = (0..12).map(|i| 0.01 * i as f64).collect();
    let site0_q: Vec<f64> = (0..12).map(|i| 0.001 * i as f64).collect();
    let site1_kind = [0_i64, 0];
    let site1_kappa = [-1_i64, -1];
    let site1_n = [0_i64, 1];
    let site1_p: Vec<f64> = (0..6).map(|i| 0.02 * i as f64).collect();
    let site1_q: Vec<f64> = (0..6).map(|i| 0.002 * i as f64).collect();
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

    let mut stream = MldumpWriterV1::create(path, &header)?.begin_spinor()?;
    stream.begin_orbitals(&SpinorOrbitalsBeginV1 {
        band_window_start: 0,
        band_window_count: band_window,
    })?;
    for k in 0..2 {
        let matches = [
            SpinorSiteMatchRefV1 {
                site_index: 0,
                n_projection: 5,
                n_apw_projection: 4,
                coordinate: &proj0_coord,
                signed_kappa: &proj0_kappa,
                twice_mu: &proj0_mu,
                radial_n: &proj0_n,
                matching_coefficients: &match0[k],
            },
            SpinorSiteMatchRefV1 {
                site_index: 1,
                n_projection: 2,
                n_apw_projection: 2,
                coordinate: &proj1_coord,
                signed_kappa: &proj1_kappa,
                twice_mu: &proj1_mu,
                radial_n: &proj1_n,
                matching_coefficients: &match1[k],
            },
        ];
        stream.write_orbital_k(&SpinorOrbitalKRefV1 {
            k_index: k,
            available_bands: available[k],
            basis_dimension: 2 * n_pw[k] + 1,
            eigenvalues: &evals[k],
            eigenvectors: &evecs[k],
            n_plane_waves: n_pw[k],
            plane_wave_g: &g[k],
            plane_wave_k_cartesian: &k_cart[k],
            plane_wave_q_cartesian: &q_cart[k],
            pauli_rows: SpinorPauliRowMapRefV1 {
                n_row: pauli_row[k].len(),
                row_index: &pauli_row[k],
                pauli_component: &pauli_comp[k],
                plane_wave_index: &pauli_pw[k],
            },
            local_orbitals: SpinorLocalOrbitalTableRefV1 {
                n_local_orbitals: 1,
                row_index: &lo_row[k],
                site: &lo_site,
                signed_kappa: &lo_kappa,
                twice_mu: &lo_mu,
                ordinal: &lo_ord,
                radial_n: &lo_n,
            },
            site_matches: &matches,
        })?;
    }
    stream.finish_orbitals()?;
    stream.begin_products(&SpinorProductsBeginV1 {
        n_k: 2,
        n_orb: 2,
        provenance_recipe: "spinor-product-test",
        provenance_reference: "spinor-mldump-neutral-fixture",
        site_indices: &site_indices,
        site_positions: &site_positions,
        site_radii: &site_radii,
        interstitial_volume_bohr3: interstitial,
    })?;
    stream.write_product_site(&SpinorProductSiteRefV1 {
        site_index: 0,
        n_radial: 3,
        n_radial_samples: 4,
        kind: &site0_kind,
        signed_kappa: &site0_kappa,
        n: &site0_n,
        p: &site0_p,
        q: &site0_q,
    })?;
    stream.write_product_site(&SpinorProductSiteRefV1 {
        site_index: 1,
        n_radial: 2,
        n_radial_samples: 3,
        kind: &site1_kind,
        signed_kappa: &site1_kappa,
        n: &site1_n,
        p: &site1_p,
        q: &site1_q,
    })?;
    stream.write_product_q(&SpinorProductQRecordRefV1 {
        q_index: 0,
        transfer_cartesian: [0.0, 0.0, 0.0],
        global_transfer: [0, 0, 0],
        n_raw_g: 2,
        raw_relative_g: &raw0,
        provenance: "q0-raw-g",
    })?;
    stream.write_product_q(&SpinorProductQRecordRefV1 {
        q_index: 1,
        transfer_cartesian: q1_cart,
        global_transfer: [1, 0, 0],
        n_raw_g: 2,
        raw_relative_g: &raw1,
        provenance: "q1-raw-g",
    })?;
    stream.finish_products()?;
    stream.begin_thc(&MldumpThcBeginV1 {
        parent_grid: MldumpThcParentGridRefV1 {
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
        selection: MldumpThcSelectionRefV1 {
            pivots: &pivots,
            points: &points,
        },
    })?;
    stream.write_thc_q(&MldumpThcQRecordRefV1 {
        q_index: 0,
        aux_dimension: 2,
        layout_provenance: "aux-layout-q0",
        zeta: &zeta0,
        residual_l2_all_frobenius: 1.0e-8,
        residual_l2_all_column_max: 2.0e-8,
        vertices: MldumpThcVertexTableRefV1 {
            n_vertex: 2,
            column: &vertex_col,
            k_left_right: &vertex_klr,
            coefficients: &vertex_c0,
        },
    })?;
    stream.write_thc_q(&MldumpThcQRecordRefV1 {
        q_index: 1,
        aux_dimension: 2,
        layout_provenance: "aux-layout-q1",
        zeta: &zeta1,
        residual_l2_all_frobenius: 3.0e-8,
        residual_l2_all_column_max: 4.0e-8,
        vertices: MldumpThcVertexTableRefV1 {
            n_vertex: 2,
            column: &vertex_col,
            k_left_right: &vertex_klr,
            coefficients: &vertex_c1,
        },
    })?;
    stream.finish_thc()?;
    stream.begin_coulomb(&MldumpCoulombBeginV1 {
        lexp: 14,
        interpolation_l_max: 2,
        interpolation_pw_cutoff: 2.0,
    })?;
    stream.write_coulomb_q(&MldumpCoulombQRecordRefV1 {
        q_index: 0,
        aux_dimension: 2,
        layout_provenance: "aux-layout-q0",
        body: &body0,
        gamma: Some(MldumpCoulombGammaRefV1 {
            spherical_average_subtracted: true,
            head_prefactor: 4.0 * PI,
            constant_coefficients: &gamma_c,
        }),
    })?;
    stream.write_coulomb_q(&MldumpCoulombQRecordRefV1 {
        q_index: 1,
        aux_dimension: 2,
        layout_provenance: "aux-layout-q1",
        body: &body1,
        gamma: None,
    })?;
    stream.finish_coulomb()?;
    stream.finish()
}

#[test]
fn mldump_spinor_v1_roundtrip_has_inspectable_hdf5_payload() {
    let path = fixture_path("libmuffintin-mldump-spinor-v1-fixture.h5");
    write_spinor_fixture(&path).unwrap();
    let read = read_mldump_v1(&path).unwrap();
    let MldumpPayloadV1::Spinor(spinor) = read.payload else {
        panic!("expected spinor payload, got {:?}", read.payload);
    };
    assert_eq!(spinor.orbitals.band_window_count, 2);
    assert_eq!(spinor.orbitals.k_points[0].available_bands, 2);
    assert_eq!(spinor.orbitals.k_points[1].available_bands, 3);
    assert_eq!(spinor.orbitals.k_points[0].plane_wave_g.len(), 2);
    assert_eq!(spinor.orbitals.k_points[1].plane_wave_g.len(), 3);
    assert_eq!(spinor.orbitals.k_points[0].pauli_rows.row_index.len(), 4);
    assert_eq!(spinor.orbitals.k_points[1].pauli_rows.row_index.len(), 6);
    assert_eq!(spinor.orbitals.k_points[0].local_orbitals[0].radial_n, 2);
    assert_eq!(
        spinor.orbitals.k_points[0].local_orbitals[0].signed_kappa,
        -1
    );
    assert_eq!(
        spinor.orbitals.k_points[0].site_matches[0].coordinates[4].radial_n,
        2
    );
    assert_eq!(spinor.products.sites[0].n, vec![0, 1, 2]);
    assert_eq!(spinor.products.q_records[1].global_transfer, [1, 0, 0]);
    assert!(spinor.thc.parent_grid.weights.contains(&0.0));
    assert_eq!(spinor.thc.engine, MLDUMP_THC_ENGINE_QRCP);
    assert!(spinor.coulomb.q_records[0].gamma.is_some());
    assert!(spinor.coulomb.q_records[1].gamma.is_none());

    let file = File::open(&path).unwrap();
    assert_eq!(group_status(&file, "orbitals"), MLDUMP_STATUS_PRESENT);
    assert_eq!(group_status(&file, "products"), MLDUMP_STATUS_PRESENT);
    assert_eq!(group_status(&file, "thc"), MLDUMP_STATUS_PRESENT);
    assert_eq!(group_status(&file, "coulomb"), MLDUMP_STATUS_PRESENT);
    assert_eq!(
        group_status(&file, "mpb"),
        MLDUMP_STATUS_ABSENT_NOT_COMPUTED
    );
    assert!(!file.group("orbitals").unwrap().link_exists("spin_000000"));
    assert!(file.group("orbitals").unwrap().link_exists("k_000000"));
    assert!(file.group("orbitals").unwrap().link_exists("k_000001"));
    for group in ["orbitals", "products", "thc", "coulomb"] {
        assert_eq!(
            attr_str(&file, group, "representation"),
            MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION,
            "{group}"
        );
    }
    assert_eq!(
        attr_str(&file, "orbitals", "occupations_status"),
        MLDUMP_OCCUPATIONS_NOT_EXPORTED
    );
    assert_eq!(
        attr_str(&file, "products", "pair_order"),
        MLDUMP_PAIR_ORDER_K_LEFT_RIGHT
    );
    assert_eq!(
        axes(&file, "orbitals/k_000000/eigenvectors"),
        ["basis_row", "band", "re_im"]
    );
    assert_eq!(
        axes(
            &file,
            "orbitals/k_000000/basis/site_000000/matching_coefficients"
        ),
        [
            "plane_wave",
            "pauli_component",
            "projection_coordinate",
            "re_im"
        ]
    );
    assert!(
        file.dataset("orbitals/k_000000/basis/site_000000/matching_coefficients")
            .unwrap()
            .dtype()
            .unwrap()
            .is::<f64>()
    );
    assert!(
        file.dataset("orbitals/k_000000/basis/pauli_row_index")
            .unwrap()
            .dtype()
            .unwrap()
            .is::<i64>()
    );
    assert_eq!(
        axes(&file, "products/site_000000/p"),
        ["radial", "radial_sample"]
    );
    assert_eq!(
        axes(&file, "products/site_000000/q"),
        ["radial", "radial_sample"]
    );
    assert_eq!(
        axes(&file, "thc/q_000000/zeta"),
        ["parent_point", "aux", "re_im"]
    );
    assert_eq!(
        axes(&file, "coulomb/q_000000/body"),
        ["aux_row", "aux_col", "re_im"]
    );
}

#[test]
fn mldump_spinor_v1_rejects_tampered_product_mesh_binding() {
    let path = fixture_path("libmuffintin-mldump-spinor-v1-mesh-tamper.h5");
    write_spinor_fixture(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        file.group("products/site_000000")
            .unwrap()
            .attr("mesh_first_bohr")
            .unwrap()
            .write_scalar(&9.0e-4)
            .unwrap();
    }
    match read_mldump_v1(&path) {
        Err(IoError::Validation(ValidationError::LayoutMismatch { path, reference })) => {
            assert!(path.ends_with("/products/site_000000/mesh"), "path={path}");
            assert!(
                reference.contains("/geometry site 0 radial mesh"),
                "reference={reference}"
            );
        }
        other => panic!("expected product mesh LayoutMismatch, got {other:?}"),
    }
}

#[test]
fn mldump_spinor_v1_rejects_tampered_projection_apw_lo_order() {
    let path = fixture_path("libmuffintin-mldump-spinor-v1-projection-tamper.h5");
    write_spinor_fixture(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        file.dataset("orbitals/k_000000/basis/site_000000/projection_radial_n")
            .unwrap()
            .write_raw(&[0_i64, 2, 0, 1, 1])
            .unwrap();
    }
    match read_mldump_v1(&path) {
        Err(IoError::Validation(ValidationError::InvalidValue { path, expected, .. })) => {
            assert!(
                path.contains("projection"),
                "path={path} expected={expected}"
            );
        }
        other => panic!("expected projection semantic InvalidValue, got {other:?}"),
    }
}

#[test]
fn mldump_spinor_v1_rejects_swapped_apw_channel_pair_order() {
    let path = fixture_path("libmuffintin-mldump-spinor-v1-apw-channel-swap.h5");
    write_spinor_fixture(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let site = "orbitals/k_000000/basis/site_000000";
        file.dataset(&format!("{site}/projection_signed_kappa"))
            .unwrap()
            .write_raw(&[-1_i64, -1, -1, -1, -1])
            .unwrap();
        file.dataset(&format!("{site}/projection_twice_mu"))
            .unwrap()
            .write_raw(&[1_i64, 1, -1, -1, -1])
            .unwrap();
    }
    match read_mldump_v1(&path) {
        Err(IoError::Validation(ValidationError::InvalidValue {
            path,
            expected,
            actual,
        })) => {
            assert!(
                path.contains("apw_projection"),
                "path={path} expected={expected} actual={actual}"
            );
            assert!(
                expected.contains("strictly increasing") && expected.contains("(-1, 1)"),
                "expected={expected}"
            );
            assert_eq!(actual, "(-1, -1)");
        }
        other => panic!("expected non-increasing APW channel InvalidValue, got {other:?}"),
    }
}
