//! Scalar CoQui-native Cholesky adapter: q0 + finite-q hydrogen oracles.
//!
//! Live CoQui contract: CoQui `chol_reader_t.hpp` / GF2 `build_int`
//! (branch `wg-dev` @ `a19774d03fb979bd852fae4f7f95c045a4cbca78`). This is not MLDUMP.

use std::path::PathBuf;

use muffintin::{
    CheckpointPhysics, SCALAR_COULOMB_EXACTNESS_FLOOR, ScalarCoquiCholeskyError,
    ScalarCoquiCholeskySpec, ScalarCoulombError, ScalarCoulombResult, ScalarProductInput,
    build_scalar_coulomb, build_scalar_thc, write_scalar_coqui_cholesky,
};
use muffintin_core::InverseBohr;
use muffintin_coulomb::InterpolationProjection;
use muffintin_io::read_coqui_cholesky;
use muffintin_prodbasis::thc::{L2Engine, SelectorStrategy};
use num_complex::Complex64;

#[path = "scalar_hydrogen.rs"]
mod scalar_hydrogen;

use scalar_hydrogen::{coulomb_spec, hydrogen_checkpoint, parent_grid, scalar_config, thc_spec};

fn fixture_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
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

fn dense_quadratic(matrix: &[Complex64], coefficients: &[Complex64]) -> Complex64 {
    let n = coefficients.len();
    let mut acc = Complex64::default();
    for row in 0..n {
        let mut applied = Complex64::default();
        for (column, coefficient) in coefficients.iter().enumerate() {
            applied += matrix[row * n + column] * coefficient;
        }
        acc += coefficients[row].conj() * applied;
    }
    acc
}

fn l_inner(
    values: &[f64],
    np: usize,
    n_k: usize,
    n_band: usize,
    k: usize,
    i: usize,
    j: usize,
) -> Complex64 {
    let mut acc = Complex64::default();
    for q in 0..np {
        let base = ((q * n_k + k) * n_band + i) * n_band + j;
        let slot = base * 2;
        let z = Complex64::new(values[slot], values[slot + 1]);
        acc += z.conj() * z;
    }
    acc
}

fn l_plain_product(
    values: &[f64],
    np: usize,
    n_k: usize,
    n_band: usize,
    k: usize,
    i: usize,
    j: usize,
) -> Complex64 {
    let mut acc = Complex64::default();
    for q in 0..np {
        let base = ((q * n_k + k) * n_band + i) * n_band + j;
        let slot = base * 2;
        let z = Complex64::new(values[slot], values[slot + 1]);
        acc += z * z;
    }
    acc
}

fn l_transposed_pair(
    values: &[f64],
    np: usize,
    n_k: usize,
    n_band: usize,
    k: usize,
    i: usize,
    j: usize,
) -> Complex64 {
    l_inner(values, np, n_k, n_band, k, j, i)
}

#[test]
fn write_scalar_coqui_cholesky_matches_independent_quadratic() {
    let path = fixture_path("libmuffintin-runtime-scalar-coqui-cholesky.h5");
    let (inputs, thc, coulomb, spec) = build_path();
    assert!(thc.grid.points().iter().any(|point| point.weight == 0.0));
    assert_eq!(thc.selection.provenance.strategy, SelectorStrategy::AllQL2);
    assert_eq!(
        thc.selection.provenance.engine,
        L2Engine::FullColumnPivotedQr
    );
    assert_eq!(inputs[1].source.q.umklapp.index, [1, 0, 0]);
    assert!(coulomb.records[0].operator.gamma().is_some());
    assert!(coulomb.records[1].operator.gamma().is_none());
    let factor = ScalarCoquiCholeskySpec { tolerance: 1.0e-10 };
    write_scalar_coqui_cholesky(&path, &inputs, &thc, &coulomb, &spec, factor).unwrap();
    let read = read_coqui_cholesky(&path).unwrap();
    assert_eq!(read.header.nspin, 1);
    assert_eq!(read.header.nspin_in_basis, 1);
    assert_eq!(read.header.nbnd_aux, 0);
    assert_eq!(read.header.tol, factor.tolerance);
    assert_eq!(read.header.nkpts, 2);
    assert_eq!(read.records.len(), 2);
    let n_k = inputs[0].orbitals.k_fractional.len();
    let n_band = inputs[0].orbitals.band_window.count;
    assert_eq!(read.header.nbnd, n_band as i32);
    assert_eq!(
        read.header.qpts[0..3],
        [
            inputs[0].source.q.cartesian[0].get(),
            inputs[0].source.q.cartesian[1].get(),
            inputs[0].source.q.cartesian[2].get()
        ]
    );
    assert_eq!(
        read.header.qpts[3..6],
        [
            inputs[1].source.q.cartesian[0].get(),
            inputs[1].source.q.cartesian[1].get(),
            inputs[1].source.q.cartesian[2].get()
        ]
    );
    assert_ne!(
        read.header.qpts[3],
        inputs[1].source.q.cartesian[0].get() + inputs[1].source.q.umklapp.cartesian[0].get(),
        "qpts must be canonical Cartesian without global Umklapp"
    );
    assert_eq!(
        read.header.qk_to_kmq,
        inputs
            .iter()
            .flat_map(|input| input.k_minus_q.iter().map(|mapped| mapped.kq_index as i32))
            .collect::<Vec<_>>()
    );
    assert_ne!(read.header.qk_to_kmq[2], read.header.qk_to_kmq[3]);

    let np = read.header.np as usize;
    let mut saw_negative = false;
    for (q, record) in coulomb.records.iter().enumerate() {
        let stored = &read.records[q].values;
        for (column, vertex) in record.vertices.iter().enumerate() {
            let (k, i, j) = record.layout.decode(column);
            let reconstructed = l_inner(stored, np, n_k, n_band, k, i, j);
            let original = dense_quadratic(record.operator.matrix(), vertex.coefficients());
            let abs = (reconstructed - original).norm();
            assert!(
                abs <= SCALAR_COULOMB_EXACTNESS_FLOOR
                    || abs / original.norm().max(SCALAR_COULOMB_EXACTNESS_FLOOR)
                        <= SCALAR_COULOMB_EXACTNESS_FLOOR,
                "q={q} column={column} L^H L {reconstructed} vs independent c^H V c {original} abs {abs}"
            );
            let wrong_conj = l_plain_product(stored, np, n_k, n_band, k, i, j);
            if (wrong_conj - original).norm()
                > SCALAR_COULOMB_EXACTNESS_FLOOR * original.norm().max(1.0)
            {
                saw_negative = true;
            }
            if i != j {
                let swapped = l_transposed_pair(stored, np, n_k, n_band, k, i, j);
                if (swapped - original).norm()
                    > SCALAR_COULOMB_EXACTNESS_FLOOR * original.norm().max(1.0)
                {
                    saw_negative = true;
                }
            }
        }
    }
    assert!(
        saw_negative,
        "wrong conjugation or pair transpose must fail to reconstruct c^H V c"
    );
}

#[test]
fn write_scalar_coqui_cholesky_rejects_forged_spec_before_create() {
    let path = fixture_path("libmuffintin-runtime-scalar-coqui-cholesky-spec.h5");
    let _ = std::fs::remove_file(&path);
    let (inputs, thc, coulomb, spec) = build_path();
    let mismatched = muffintin::ScalarCoulombSpec {
        request: spec.request.clone(),
        projection: InterpolationProjection::new(InverseBohr(1.5), 0).unwrap(),
    };
    let error = write_scalar_coqui_cholesky(
        &path,
        &inputs,
        &thc,
        &coulomb,
        &mismatched,
        ScalarCoquiCholeskySpec { tolerance: 1.0e-10 },
    )
    .unwrap_err();
    match error {
        ScalarCoquiCholeskyError::Coulomb(ScalarCoulombError::SpecMismatch) => {}
        other => panic!("expected spec mismatch, got {other}"),
    }
    assert!(
        !path.exists(),
        "forged Coulomb spec must not create {}",
        path.display()
    );
}
