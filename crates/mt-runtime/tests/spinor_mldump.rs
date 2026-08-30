//! Spinor MLDUMP materialization: owned reader plus dense $c^\dagger V c$ oracle.

use std::f64::consts::PI;
use std::path::PathBuf;

use muffintin::{
    SPINOR_COULOMB_EXACTNESS_FLOOR, SPINOR_RADIAL_LO0, CheckpointPhysics, SpinorCoulombError,
    SpinorCoulombResult, SpinorCoulombSpec, SpinorMldumpError, SpinorProductInput,
    build_spinor_coulomb, build_spinor_thc, write_spinor_mldump,
};
use muffintin_core::InverseBohr;
use muffintin_coulomb::InterpolationProjection;
use muffintin_io::{
    MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION, MLDUMP_STATUS_ABSENT_NOT_COMPUTED,
    MLDUMP_THC_ENGINE_QRCP, MldumpGeometryV1, MldumpHeaderV1, MldumpKMinusQV1, MldumpKPointV1,
    MldumpMeshV1, MldumpMetaV1, MldumpPayloadV1, MldumpQEntryV1, MldumpRadialMeshV1, MldumpSiteV1,
    read_mldump_v1,
};
use num_complex::Complex64;

#[path = "spinor_hydrogen.rs"]
mod spinor_hydrogen;

use spinor_hydrogen::{
    LATTICE, coulomb_spec, hydrogen_spinor_checkpoint, parent_grid, spinor_config, thc_spec,
};

fn fixture_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

fn header_from_inputs(inputs: &[SpinorProductInput]) -> MldumpHeaderV1 {
    let first = &inputs[0];
    let spec = coulomb_spec();
    let cell = spec.request.cell();
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
            species: Some("H".to_owned()),
            label: if index == 0 {
                Some("H-1".to_owned())
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
    MldumpHeaderV1::new(
        MldumpMetaV1 {
            producer_name: "libmuffintin-runtime-spinor-mldump-test".to_owned(),
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
                    let canonical = [
                        input.source.q.cartesian[0].get() * LATTICE / (2.0 * PI),
                        input.source.q.cartesian[1].get() * LATTICE / (2.0 * PI),
                        input.source.q.cartesian[2].get() * LATTICE / (2.0 * PI),
                    ];
                    let input_fractional = [
                        canonical[0] + f64::from(umklapp[0]),
                        canonical[1] + f64::from(umklapp[1]),
                        canonical[2] + f64::from(umklapp[2]),
                    ];
                    MldumpQEntryV1 {
                        input_fractional,
                        canonical_fractional: canonical,
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

fn quadratic_from_flat(body: &[f64], coefficients: &[f64]) -> Complex64 {
    let n = coefficients.len() / 2;
    let mut acc = Complex64::default();
    for row in 0..n {
        let mut applied = Complex64::default();
        for column in 0..n {
            let re = body[(row * n + column) * 2];
            let im = body[(row * n + column) * 2 + 1];
            let cr = coefficients[column * 2];
            let ci = coefficients[column * 2 + 1];
            applied += Complex64::new(re, im) * Complex64::new(cr, ci);
        }
        acc += Complex64::new(coefficients[row * 2], coefficients[row * 2 + 1]).conj() * applied;
    }
    acc
}

fn build_path() -> (
    Vec<SpinorProductInput>,
    muffintin::SpinorThcResult,
    SpinorCoulombResult,
    muffintin::SpinorCoulombSpec,
) {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let q0 = physics
        .spinor_product_input(&spinor_config([2, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let q15 = physics
        .spinor_product_input(&spinor_config([2, 1, 1], 0.5), [1.5, 0.0, 0.0])
        .unwrap();
    let grid = parent_grid(&q15);
    let inputs = vec![q0, q15];
    let thc = build_spinor_thc(&inputs, &grid, &thc_spec()).unwrap();
    let spec = coulomb_spec();
    let coulomb = build_spinor_coulomb(&inputs, &thc, &spec, &[]).unwrap();
    (inputs, thc, coulomb, spec)
}

#[test]
fn write_spinor_mldump_roundtrip_matches_runtime_quadratic() {
    let path = fixture_path("libmuffintin-runtime-spinor-mldump.h5");
    let (inputs, thc, coulomb, spec) = build_path();
    assert_eq!(inputs[1].source.q.umklapp.index, [1, 0, 0]);
    assert_ne!(
        inputs[1].k_minus_q[0].umklapp.index,
        inputs[1].k_minus_q[1].umklapp.index
    );
    assert!(thc.grid.points().iter().any(|point| point.weight == 0.0));
    let header = header_from_inputs(&inputs);
    write_spinor_mldump(&path, &header, &inputs, &thc, &coulomb, &spec).unwrap();
    let read = read_mldump_v1(&path).unwrap();
    assert_eq!(read.header.mesh.q_entries[1].global_umklapp, [1, 0, 0]);
    let MldumpPayloadV1::Spinor(spinor) = read.payload else {
        panic!("expected spinor payload, got {:?}", read.payload);
    };
    let window = inputs[0].orbitals.band_window.count;
    assert_eq!(spinor.orbitals.band_window_count, window);
    assert!(
        spinor
            .orbitals
            .k_points
            .iter()
            .any(|k| k.available_bands >= window)
    );
    assert!(
        spinor.orbitals.k_points.iter().any(|k| k.available_bands
            != spinor.orbitals.k_points[0].available_bands
            || k.plane_wave_g.len() != spinor.orbitals.k_points[0].plane_wave_g.len())
            || spinor.orbitals.k_points[0].available_bands >= window
    );
    let k0 = &spinor.orbitals.k_points[0];
    assert_eq!(k0.pauli_rows.row_index.len(), 2 * k0.plane_wave_g.len());
    assert!(
        k0.local_orbitals
            .iter()
            .any(|row| row.radial_n >= SPINOR_RADIAL_LO0)
    );
    let match0 = &k0.site_matches[0];
    assert!(match0.coordinates.iter().any(|coord| coord.radial_n == 0));
    assert!(match0.coordinates.iter().any(|coord| coord.radial_n == 1));
    assert!(
        match0
            .coordinates
            .iter()
            .any(|coord| coord.radial_n >= SPINOR_RADIAL_LO0)
    );
    let n_apw = match0
        .coordinates
        .iter()
        .take_while(|coord| coord.radial_n <= 1)
        .count();
    assert_eq!(
        match0.matching_coefficients.len(),
        k0.plane_wave_g.len() * 2 * n_apw * 2
    );
    let site0 = &spinor.products.sites[0];
    assert!(site0.n.contains(&0));
    assert!(site0.n.contains(&1));
    assert!(site0.n.iter().any(|n| *n >= SPINOR_RADIAL_LO0 as i64));
    assert_eq!(site0.p.len(), site0.q.len());
    assert_eq!(
        site0.p.len(),
        site0.n.len() * header.geometry.sites[0].radial_mesh.point_count
    );
    assert!(site0.p.iter().any(|value| *value != 0.0));
    assert!(site0.q.iter().any(|value| *value != 0.0));
    assert_eq!(spinor.products.q_records[1].global_transfer, [1, 0, 0]);
    assert_ne!(
        read.header.mesh.q_entries[1].k_minus_q[0].g_wrap,
        read.header.mesh.q_entries[1].k_minus_q[1].g_wrap
    );
    assert!(spinor.thc.parent_grid.weights.contains(&0.0));
    assert_eq!(spinor.thc.engine, MLDUMP_THC_ENGINE_QRCP);
    assert!(!spinor.thc.q_records[0].vertices.is_empty());
    assert!(spinor.coulomb.q_records[0].gamma.is_some());
    assert!(spinor.coulomb.q_records[1].gamma.is_none());
    assert_eq!(
        read.exchange.valence.as_str(),
        MLDUMP_STATUS_ABSENT_NOT_COMPUTED
    );

    for (q, record) in coulomb.records().iter().enumerate() {
        let stored_body = &spinor.coulomb.q_records[q].body;
        for (vertex_index, vertex) in record.vertices.iter().enumerate() {
            let stored = &spinor.thc.q_records[q].vertices[vertex_index];
            let reconstructed = quadratic_from_flat(stored_body, &stored.coefficients);
            let original = record.operator.quadratic_form(vertex, vertex).unwrap();
            let abs = (reconstructed - original).norm();
            assert!(
                abs <= SPINOR_COULOMB_EXACTNESS_FLOOR
                    || abs / original.norm().max(SPINOR_COULOMB_EXACTNESS_FLOOR)
                        <= SPINOR_COULOMB_EXACTNESS_FLOOR,
                "q={q} vertex={vertex_index} reconstructed {reconstructed} original {original} abs {abs}"
            );
        }
    }
}

#[test]
fn write_spinor_mldump_rejects_mismatched_coulomb_spec_before_create() {
    let path = fixture_path("libmuffintin-runtime-spinor-mldump-spec.h5");
    let _ = std::fs::remove_file(&path);
    let (inputs, thc, coulomb, spec) = build_path();
    let header = header_from_inputs(&inputs);
    let mismatched = SpinorCoulombSpec {
        request: spec.request.clone(),
        projection: InterpolationProjection::new(InverseBohr(1.5), 0).unwrap(),
    };
    assert_eq!(mismatched.request.reciprocal(), spec.request.reciprocal());
    assert_ne!(mismatched.projection, spec.projection);
    let error =
        write_spinor_mldump(&path, &header, &inputs, &thc, &coulomb, &mismatched).unwrap_err();
    match error {
        SpinorMldumpError::Coulomb(SpinorCoulombError::SpecMismatch) => {}
        other => panic!("expected spec mismatch, got {other}"),
    }
    assert!(
        !path.exists(),
        "mismatched Coulomb spec must not create {}",
        path.display()
    );
}

#[test]
fn write_spinor_mldump_rejects_undersized_site_augmentation_before_create() {
    let path = fixture_path("libmuffintin-runtime-spinor-mldump-aug.h5");
    let _ = std::fs::remove_file(&path);
    let (mut inputs, thc, coulomb, spec) = build_path();
    let header = header_from_inputs(&inputs);
    for input in &mut inputs {
        for basis in &mut input.orbitals.bases {
            basis.site_augmentations[0][0].coefficients[0].clear();
        }
    }
    let error = write_spinor_mldump(&path, &header, &inputs, &thc, &coulomb, &spec).unwrap_err();
    match error {
        SpinorMldumpError::ExportableBasis {
            path: ref field, ..
        } => {
            assert!(
                field.contains("coefficients"),
                "expected coefficient exportability path, got {field}"
            );
        }
        other => panic!("expected exportable basis error, got {other}"),
    }
    assert!(
        !path.exists(),
        "undersized site augmentation must not create {}",
        path.display()
    );
}
