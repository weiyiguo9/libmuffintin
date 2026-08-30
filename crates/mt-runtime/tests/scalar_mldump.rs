//! Scalar MLDUMP materialization: owned reader plus dense $c^\dagger V c$ oracle.

use std::f64::consts::PI;
use std::path::PathBuf;

use muffintin::{
    SCALAR_COULOMB_EXACTNESS_FLOOR, ScalarCoulombError, ScalarCoulombResult, ScalarCoulombSpec,
    ScalarMldumpError, ScalarProductInput, CheckpointPhysics, build_scalar_coulomb,
    build_scalar_thc, write_scalar_mldump,
};
use muffintin_core::InverseBohr;
use muffintin_coulomb::{AuxiliaryKind, InterpolationProjection, assemble_point_charge_oracle};
use muffintin_io::{
    MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON, MLDUMP_STATUS_ABSENT_NOT_COMPUTED,
    MLDUMP_THC_ENGINE_QRCP, MldumpGeometryV1, MldumpHeaderV1, MldumpKMinusQV1, MldumpKPointV1,
    MldumpMeshV1, MldumpMetaV1, MldumpPayloadV1, MldumpQEntryV1, MldumpRadialMeshV1, MldumpSiteV1,
    read_mldump_v1,
};
use num_complex::Complex64;

#[path = "scalar_hydrogen.rs"]
mod scalar_hydrogen;

use scalar_hydrogen::{
    LATTICE, coulomb_spec, hydrogen_checkpoint, parent_grid, scalar_config, thc_spec,
};

fn fixture_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

fn header_from_inputs(inputs: &[ScalarProductInput]) -> MldumpHeaderV1 {
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
            producer_name: "libmuffintin-runtime-scalar-mldump-test".to_owned(),
            producer_version: "0.1.0".to_owned(),
            source_revision: "a55d4cf4301b9d874f0109d754fd8f8deff93c55".to_owned(),
            feature_representation: MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON.to_owned(),
        },
        MldumpGeometryV1 {
            direct_basis_bohr: cell
                .basis()
                .map(|vector| vector.map(|component| component.get())),
            reciprocal_basis_inv_bohr: first
                .reciprocal
                .basis()
                .map(|vector| vector.map(|component| component.get())),
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
    Vec<ScalarProductInput>,
    muffintin::ScalarThcResult,
    ScalarCoulombResult,
    muffintin::ScalarCoulombSpec,
) {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let q0 = physics
        .scalar_product_input(&scalar_config([2, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let q15 = physics
        .scalar_product_input(&scalar_config([2, 1, 1], 0.5), [1.5, 0.0, 0.0])
        .unwrap();
    let grid = parent_grid(&q15);
    let inputs = vec![q0, q15];
    let thc = build_scalar_thc(&inputs, &grid, &thc_spec()).unwrap();
    let spec = coulomb_spec();
    let coulomb = build_scalar_coulomb(&inputs, &thc, &spec, &[]).unwrap();
    (inputs, thc, coulomb, spec)
}

#[test]
fn write_scalar_mldump_roundtrip_matches_runtime_quadratic() {
    let path = fixture_path("libmuffintin-runtime-scalar-mldump.h5");
    let (inputs, thc, coulomb, spec) = build_path();
    assert_eq!(inputs[1].source.q.umklapp.index, [1, 0, 0]);
    assert_ne!(
        inputs[1].k_minus_q[0].umklapp.index,
        inputs[1].k_minus_q[1].umklapp.index
    );
    assert!(thc.grid.points().iter().any(|point| point.weight == 0.0));
    let header = header_from_inputs(&inputs);
    write_scalar_mldump(&path, &header, &inputs, &thc, &coulomb, &spec).unwrap();
    let read = read_mldump_v1(&path).unwrap();
    assert_eq!(read.header.mesh.q_entries[1].global_umklapp, [1, 0, 0]);
    let MldumpPayloadV1::Scalar(scalar) = read.payload else {
        panic!("expected scalar payload, got {:?}", read.payload);
    };
    assert!(scalar.thc.parent_grid.weights.contains(&0.0));
    assert_eq!(scalar.thc.engine, MLDUMP_THC_ENGINE_QRCP);
    let pivot_set = scalar
        .thc
        .selection
        .pivots
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let point_set = scalar
        .thc
        .selection
        .points
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(pivot_set, point_set);
    assert!(scalar.coulomb.q_records[0].gamma.is_some());
    assert!(scalar.coulomb.q_records[1].gamma.is_none());
    assert_eq!(
        read.exchange.valence.as_str(),
        MLDUMP_STATUS_ABSENT_NOT_COMPUTED
    );

    for (q, record) in coulomb.records.iter().enumerate() {
        let stored_body = &scalar.coulomb.q_records[q].body;
        for (vertex_index, vertex) in record.vertices.iter().enumerate() {
            let stored = &scalar.thc.q_records[q].vertices[vertex_index];
            let reconstructed = quadratic_from_flat(stored_body, &stored.coefficients);
            let original = record.operator.quadratic_form(vertex, vertex).unwrap();
            let abs = (reconstructed - original).norm();
            assert!(
                abs <= SCALAR_COULOMB_EXACTNESS_FLOOR
                    || abs / original.norm().max(SCALAR_COULOMB_EXACTNESS_FLOOR)
                        <= SCALAR_COULOMB_EXACTNESS_FLOOR,
                "q={q} vertex={vertex_index} reconstructed {reconstructed} original {original} abs {abs}"
            );
        }
    }
}

#[test]
fn write_scalar_mldump_rejects_tampered_header_before_create() {
    let path = fixture_path("libmuffintin-runtime-scalar-mldump-tamper.h5");
    let _ = std::fs::remove_file(&path);
    let (inputs, thc, coulomb, spec) = build_path();
    let mut header = header_from_inputs(&inputs);
    header.mesh.k_points[0].weight = 0.3;
    let error = write_scalar_mldump(&path, &header, &inputs, &thc, &coulomb, &spec).unwrap_err();
    match error {
        ScalarMldumpError::HeaderMismatch { path, .. } => {
            assert!(path.contains("k_points[0].weight"), "{path}");
        }
        other => panic!("expected header mismatch, got {other}"),
    }
    assert!(
        !path.exists(),
        "tampered header must not create {}",
        path.display()
    );
}

#[test]
fn write_scalar_mldump_rejects_mismatched_coulomb_spec_before_create() {
    let path = fixture_path("libmuffintin-runtime-scalar-mldump-spec.h5");
    let _ = std::fs::remove_file(&path);
    let (inputs, thc, mut coulomb, spec) = build_path();
    let header = header_from_inputs(&inputs);
    let mismatched = ScalarCoulombSpec {
        request: spec.request.clone(),
        projection: InterpolationProjection::new(InverseBohr(1.5), 0).unwrap(),
    };
    assert_eq!(mismatched.request.reciprocal(), spec.request.reciprocal());
    assert_ne!(mismatched.projection, spec.projection);
    let error =
        write_scalar_mldump(&path, &header, &inputs, &thc, &coulomb, &mismatched).unwrap_err();
    match error {
        ScalarMldumpError::Coulomb(ScalarCoulombError::SpecMismatch) => {}
        other => panic!("expected spec mismatch, got {other}"),
    }
    assert!(
        !path.exists(),
        "mismatched Coulomb spec must not create {}",
        path.display()
    );

    coulomb.records.swap(0, 1);
    let error = write_scalar_mldump(&path, &header, &inputs, &thc, &coulomb, &spec).unwrap_err();
    match error {
        ScalarMldumpError::Coulomb(ScalarCoulombError::CoulombRecord { index }) => {
            assert_eq!(index, 0);
        }
        other => panic!("expected Coulomb record mismatch, got {other}"),
    }
    assert!(
        !path.exists(),
        "reordered Coulomb records must not create {}",
        path.display()
    );
}

#[test]
fn write_scalar_mldump_rejects_point_charge_oracle_operator_before_create() {
    let path = fixture_path("libmuffintin-runtime-scalar-mldump-oracle.h5");
    let _ = std::fs::remove_file(&path);
    let (inputs, thc, mut coulomb, spec) = build_path();
    let header = header_from_inputs(&inputs);
    let request = spec
        .request
        .clone()
        .with_interpolation(spec.projection)
        .unwrap();
    let original = &coulomb.records[0];
    let oracle = assemble_point_charge_oracle(&original.auxiliary, &request).unwrap();
    assert_eq!(oracle.dimension(), original.operator.dimension());
    assert_eq!(oracle.q(), original.q);
    assert_eq!(oracle.cell(), spec.request.cell());
    assert_eq!(oracle.reciprocal(), spec.request.reciprocal());
    assert_eq!(oracle.layout(), &original.auxiliary.layout());
    assert_eq!(oracle.kind(), AuxiliaryKind::PointChargeOracle);
    coulomb.records[0].operator = oracle;
    let error = write_scalar_mldump(&path, &header, &inputs, &thc, &coulomb, &spec).unwrap_err();
    match error {
        ScalarMldumpError::Coulomb(ScalarCoulombError::CoulombRecord { index }) => {
            assert_eq!(index, 0);
        }
        other => panic!("expected Coulomb record mismatch, got {other}"),
    }
    assert!(
        !path.exists(),
        "point-charge oracle operator must not create {}",
        path.display()
    );
}
