//! Facade versus explicit [`BasisSpec`] routes must be tolerance-identical.
//!
//! The facade path calls the canonical recipe facade. The explicit path
//! constructs [`BasisSpec`] by hand and must not call
//! `muffintin_recipes::lapw`. Both paths assemble through
//! `assemble_compiled` / the shared operator layer.

use muffintin_basis::{
    ApwBoundaryBasis, ApwSiteAugmentation, BasisBlock, BasisSpec, LocalOrbitalLayout, Provenance,
    compile,
};
use muffintin_core::{
    Bohr, Hartree, InterstitialGeometry, InverseBohr, KineticOperatorConvention, ReciprocalLattice,
    Sphere, VolumeBohr3,
};
use muffintin_envelope::{PlaneWave, PlaneWaveEnvelope};
use muffintin_lapw::{
    DenseEigenvectors, InterstitialPotential, LapwSiteInput, RadialOverlapBlock,
    SiteOperatorBlocks, assemble_compiled, assemble_eigenproblem, solve_collinear_eigenproblems,
};
use muffintin_operators::{
    Collinear, EigenpairResidual, GeneralizedEigensolution, solve_generalized_hermitian,
};
use muffintin_radial::BoundaryData;
use muffintin_tensor::{Axis, DenseHermitianMatrix};
use num_complex::Complex64;

const MATRIX_TOL: f64 = 1.0e-14;
const ENERGY_TOL: f64 = 1.0e-12;
const VEC_TOL: f64 = 1.0e-12;
/// Bound on each stored `||Hc - Sc ε||` residual and its relative form.
const MF_RESIDUAL_TOL: f64 = 1.0e-12;

fn boundary(value: f64, derivative: f64) -> BoundaryData {
    BoundaryData {
        value,
        derivative,
        log_derivative: (value != 0.0).then(|| InverseBohr(derivative / value)),
        scaled_log_derivative: None,
    }
}

fn waves() -> Vec<PlaneWave> {
    let lattice = ReciprocalLattice::new([
        [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
        [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
        [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
    ])
    .unwrap();
    lattice
        .enumerate(InverseBohr(1.0))
        .unwrap()
        .into_iter()
        .map(|g| PlaneWave::new([InverseBohr(0.1), InverseBohr(-0.2), InverseBohr(0.05)], g))
        .collect()
}

fn envelope_from(waves: &[PlaneWave]) -> PlaneWaveEnvelope {
    PlaneWaveEnvelope::new(waves.to_vec())
}

fn site_h(
    dimension: usize,
    element: impl FnMut(usize, usize) -> Complex64,
) -> DenseHermitianMatrix {
    DenseHermitianMatrix::from_upper_triangle(dimension, Axis::SiteCoordinate, element).unwrap()
}

fn matrices_match(left: &DenseHermitianMatrix, right: &DenseHermitianMatrix, tol: f64) {
    assert_eq!(left.dimension(), right.dimension());
    assert_eq!(left.axis(), right.axis());
    for i in 0..left.dimension() {
        for j in 0..left.dimension() {
            assert!(
                (left.at(i, j) - right.at(i, j)).norm() < tol,
                "matrix mismatch at ({i},{j}): {} vs {}",
                left.at(i, j),
                right.at(i, j)
            );
        }
    }
}

fn is_nondegenerate(values: &[Hartree], index: usize, tol: f64) -> bool {
    let energy = values[index].get();
    let lower_ok = index == 0 || (energy - values[index - 1].get()).abs() > tol;
    let upper_ok = index + 1 == values.len() || (values[index + 1].get() - energy).abs() > tol;
    lower_ok && upper_ok
}

/// Unit-modulus factor such that `left ≈ factor * right` when
/// `overlap = left^H right`.
fn phase_aligning_right_to_left(
    left: &DenseEigenvectors,
    right: &DenseEigenvectors,
    band: usize,
) -> Complex64 {
    let mut overlap = Complex64::new(0.0, 0.0);
    for i in 0..left.rows() {
        overlap += left.at(i, band).conj() * right.at(i, band);
    }
    if overlap.norm() == 0.0 {
        Complex64::new(1.0, 0.0)
    } else {
        overlap.conj() / overlap.norm()
    }
}

fn assert_solution_ranks_and_residuals(solution: &GeneralizedEigensolution) {
    assert_eq!(solution.eigenvalues.len(), solution.retained_dimension);
    assert_eq!(solution.residuals.len(), solution.retained_dimension);
    assert_eq!(solution.eigenvectors.columns(), solution.retained_dimension);
    assert_eq!(
        solution.eigenvectors.rows(),
        solution.retained_dimension + solution.filtered_dimension
    );
    for residual in &solution.residuals {
        assert!(
            residual.absolute.is_finite(),
            "absolute residual for band {} is not finite",
            residual.band_index
        );
        assert!(
            residual.relative.is_finite(),
            "relative residual for band {} is not finite",
            residual.band_index
        );
        assert!(
            residual.absolute < MF_RESIDUAL_TOL,
            "absolute residual {} for band {} exceeds residual bound {MF_RESIDUAL_TOL}",
            residual.absolute,
            residual.band_index
        );
        assert!(
            residual.relative < MF_RESIDUAL_TOL,
            "relative residual {} for band {} exceeds residual bound {MF_RESIDUAL_TOL}",
            residual.relative,
            residual.band_index
        );
    }
}

fn solutions_match(
    left: &GeneralizedEigensolution,
    right: &GeneralizedEigensolution,
    energy_tol: f64,
    vec_tol: f64,
) {
    assert_solution_ranks_and_residuals(left);
    assert_solution_ranks_and_residuals(right);
    assert_eq!(left.retained_dimension, right.retained_dimension);
    assert_eq!(left.filtered_dimension, right.filtered_dimension);
    for (left_energy, right_energy) in left.eigenvalues.iter().zip(&right.eigenvalues) {
        assert!((left_energy.get() - right_energy.get()).abs() <= energy_tol);
    }
    for (left_residual, right_residual) in left.residuals.iter().zip(&right.residuals) {
        assert_eq!(left_residual.band_index, right_residual.band_index);
        assert!((left_residual.absolute - right_residual.absolute).abs() < energy_tol);
        assert!((left_residual.relative - right_residual.relative).abs() < energy_tol);
    }
    for band in 0..left.eigenvectors.columns() {
        if !is_nondegenerate(&left.eigenvalues, band, energy_tol)
            || !is_nondegenerate(&right.eigenvalues, band, energy_tol)
        {
            continue;
        }
        let phase = phase_aligning_right_to_left(&left.eigenvectors, &right.eigenvectors, band);
        for i in 0..left.eigenvectors.rows() {
            assert!(
                (left.eigenvectors.at(i, band) - phase * right.eigenvectors.at(i, band)).norm()
                    < vec_tol
            );
        }
    }
}

fn eigenproblems_match(
    facade: &muffintin_lapw::LapwEigenproblem,
    explicit: &muffintin_lapw::LapwEigenproblem,
    overlap_threshold: f64,
) {
    matrices_match(&facade.overlap, &explicit.overlap, MATRIX_TOL);
    matrices_match(&facade.hamiltonian, &explicit.hamiltonian, MATRIX_TOL);
    let facade_sol =
        solve_generalized_hermitian(&facade.hamiltonian, &facade.overlap, overlap_threshold)
            .unwrap();
    let explicit_sol =
        solve_generalized_hermitian(&explicit.hamiltonian, &explicit.overlap, overlap_threshold)
            .unwrap();
    solutions_match(&facade_sol, &explicit_sol, ENERGY_TOL, VEC_TOL);
}

fn empty_lattice_spec(envelope: PlaneWaveEnvelope, volume: VolumeBohr3) -> BasisSpec {
    BasisSpec {
        blocks: vec![BasisBlock::PlaneWaveEnvelope {
            envelope,
            sites: Vec::new(),
        }],
        cell_volume: volume,
        provenance: Provenance::default(),
    }
}

#[test]
fn empty_lattice_facade_and_explicit_spec_match() {
    let waves = waves();
    let envelope = envelope_from(&waves);
    let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), vec![]).unwrap();
    let potential = InterstitialPotential::default();
    let facade = assemble_eigenproblem(&envelope, &geometry, &potential, &[], &[]).unwrap();

    let compiled = compile(&empty_lattice_spec(
        envelope_from(&waves),
        VolumeBohr3(100.0),
    ))
    .unwrap();
    assert_eq!(compiled.plane_waves, waves);
    assert_eq!(compiled.site_count(), 0);
    let explicit = assemble_compiled(&compiled, &geometry, &potential, &[]).unwrap();

    eigenproblems_match(&facade, &explicit, 0.0);

    let kinetic = KineticOperatorConvention::SpexSymmetricLaplacian;
    for (i, wave) in waves.iter().enumerate() {
        let expected = kinetic.prefactor(wave.q, wave.q).get();
        assert!((facade.hamiltonian.at(i, i) - expected).norm() <= ENERGY_TOL);
        assert!((explicit.hamiltonian.at(i, i) - expected).norm() <= ENERGY_TOL);
    }
}

#[test]
fn translated_sphere_overlap_facade_and_explicit_spec_match() {
    let waves = waves();
    let envelope = envelope_from(&waves);
    let position = [Bohr(0.3), Bohr(-0.4), Bohr(0.2)];
    let radius = Bohr(0.8);
    let volume = VolumeBohr3(100.0);
    let geometry = InterstitialGeometry::new(
        volume,
        vec![Sphere {
            center: position,
            radius,
        }],
    )
    .unwrap();
    let apw_boundary = ApwBoundaryBasis {
        u: boundary(0.8, -0.1),
        udot: boundary(0.2, 1.1),
    };
    let boundaries = vec![apw_boundary, apw_boundary, apw_boundary];
    let local_dimension = 2 * muffintin_core::lm_count(2);
    let radial = RadialOverlapBlock {
        uu: 1.0,
        u_udot: 0.04,
        udot_udot: 0.7,
    };
    let overlap_block = site_h(local_dimension, |row, column| {
        if row / 2 != column / 2 {
            return Complex64::default();
        }
        Complex64::new(
            match (row % 2, column % 2) {
                (0, 0) => radial.uu,
                (0, 1) => radial.u_udot,
                (1, 1) => radial.udot_udot,
                _ => unreachable!(),
            },
            0.0,
        )
    });
    let hamiltonian_block = site_h(local_dimension, |_, _| Complex64::default());
    let local = SiteOperatorBlocks {
        overlap: overlap_block,
        hamiltonian: hamiltonian_block,
    };
    let recipe_sites = [LapwSiteInput {
        position,
        radius,
        boundaries: boundaries.clone(),
        local_orbitals: LocalOrbitalLayout::default(),
    }];
    let facade = assemble_eigenproblem(
        &envelope,
        &geometry,
        &InterstitialPotential::default(),
        &recipe_sites,
        std::slice::from_ref(&local),
    )
    .unwrap();

    let spec = BasisSpec {
        blocks: vec![BasisBlock::PlaneWaveEnvelope {
            envelope: envelope_from(&waves),
            sites: vec![ApwSiteAugmentation {
                position,
                radius,
                boundaries,
            }],
        }],
        cell_volume: volume,
        provenance: Provenance::default(),
    };
    let compiled = compile(&spec).unwrap();
    assert_eq!(compiled.layout.plane_wave_count(), waves.len());
    assert_eq!(compiled.layout.site_count(), 1);
    assert_eq!(compiled.site_geometry[0].position, position);
    assert_eq!(compiled.site_geometry[0].radius, radius);
    let explicit = assemble_compiled(
        &compiled,
        &geometry,
        &InterstitialPotential::default(),
        std::slice::from_ref(&local),
    )
    .unwrap();
    eigenproblems_match(&facade, &explicit, 1.0e-12);
}

#[test]
fn apw_lo_matching_facade_and_explicit_spec_match() {
    let waves = waves();
    let wave = waves[0];
    let envelope = PlaneWaveEnvelope::new(vec![wave]);
    let position = [Bohr(0.2), Bohr(-0.1), Bohr(0.3)];
    let radius = Bohr(0.7);
    let volume = VolumeBohr3(100.0);
    let geometry = InterstitialGeometry::new(
        volume,
        vec![Sphere {
            center: position,
            radius,
        }],
    )
    .unwrap();
    let apw_boundary = ApwBoundaryBasis {
        u: boundary(0.8, -0.1),
        udot: boundary(0.2, 1.1),
    };
    let overlap = site_h(3, |row, column| match (row, column) {
        (0, 0) => Complex64::new(1.1, 0.0),
        (0, 1) => Complex64::new(0.2, 0.1),
        (0, 2) => Complex64::new(-0.3, 0.25),
        (1, 1) => Complex64::new(0.9, 0.0),
        (1, 2) => Complex64::new(0.15, -0.35),
        (2, 2) => Complex64::new(1.4, 0.0),
        _ => unreachable!(),
    });
    let hamiltonian = site_h(3, |row, column| match (row, column) {
        (0, 0) => Complex64::new(0.8, 0.0),
        (0, 1) => Complex64::new(-0.2, 0.05),
        (0, 2) => Complex64::new(0.4, -0.3),
        (1, 1) => Complex64::new(1.2, 0.0),
        (1, 2) => Complex64::new(-0.1, 0.2),
        (2, 2) => Complex64::new(2.3, 0.0),
        _ => unreachable!(),
    });
    let local = SiteOperatorBlocks {
        overlap,
        hamiltonian,
    };
    let recipe_sites = [LapwSiteInput {
        position,
        radius,
        boundaries: vec![apw_boundary],
        local_orbitals: LocalOrbitalLayout::new(vec![1]),
    }];
    let facade = assemble_eigenproblem(
        &envelope,
        &geometry,
        &InterstitialPotential::default(),
        &recipe_sites,
        std::slice::from_ref(&local),
    )
    .unwrap();

    let spec = BasisSpec {
        blocks: vec![
            BasisBlock::PlaneWaveEnvelope {
                envelope: PlaneWaveEnvelope::new(vec![wave]),
                sites: vec![ApwSiteAugmentation {
                    position,
                    radius,
                    boundaries: vec![apw_boundary],
                }],
            },
            BasisBlock::ConfinedSite {
                site: 0,
                local_orbitals: LocalOrbitalLayout::new(vec![1]),
            },
        ],
        cell_volume: volume,
        provenance: Provenance::default(),
    };
    let compiled = compile(&spec).unwrap();
    assert_eq!(compiled.layout.dimension(), 2);
    let explicit = assemble_compiled(
        &compiled,
        &geometry,
        &InterstitialPotential::default(),
        std::slice::from_ref(&local),
    )
    .unwrap();
    eigenproblems_match(&facade, &explicit, 1.0e-12);
}

#[test]
fn collinear_empty_lattice_facade_and_explicit_spec_match() {
    let waves = waves();
    let envelope = envelope_from(&waves);
    let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), vec![]).unwrap();
    let up = InterstitialPotential::new([([0; 3], Complex64::new(0.17, 0.0))]).unwrap();
    let down = InterstitialPotential::new([([0; 3], Complex64::new(-0.09, 0.0))]).unwrap();
    let facade = solve_collinear_eigenproblems(
        &envelope,
        &geometry,
        &[],
        Collinear::new(&up, &down),
        Collinear::new(&[][..], &[][..]),
        0.0,
    )
    .unwrap();

    let compiled = compile(&empty_lattice_spec(
        envelope_from(&waves),
        VolumeBohr3(100.0),
    ))
    .unwrap();
    assert_eq!(compiled.plane_waves, waves);
    let up_explicit = assemble_compiled(&compiled, &geometry, &up, &[]).unwrap();
    let down_explicit = assemble_compiled(&compiled, &geometry, &down, &[]).unwrap();
    eigenproblems_match(&facade.up.eigenproblem, &up_explicit, 0.0);
    eigenproblems_match(&facade.down.eigenproblem, &down_explicit, 0.0);
    solutions_match(
        &facade.up.solution,
        &solve_generalized_hermitian(&up_explicit.hamiltonian, &up_explicit.overlap, 0.0).unwrap(),
        ENERGY_TOL,
        VEC_TOL,
    );
    solutions_match(
        &facade.down.solution,
        &solve_generalized_hermitian(&down_explicit.hamiltonian, &down_explicit.overlap, 0.0)
            .unwrap(),
        ENERGY_TOL,
        VEC_TOL,
    );
}

fn synthetic_solution(vectors: DenseEigenvectors) -> GeneralizedEigensolution {
    let retained = vectors.columns();
    let filtered = vectors.rows() - retained;
    GeneralizedEigensolution {
        eigenvalues: vec![Hartree(1.0); retained],
        eigenvectors: vectors,
        retained_dimension: retained,
        filtered_dimension: filtered,
        residuals: (0..retained)
            .map(|band_index| EigenpairResidual {
                band_index,
                absolute: 0.0,
                relative: 0.0,
            })
            .collect(),
    }
}

#[test]
fn eigenvector_phase_helper_accepts_nontrivial_right_hand_phase() {
    let left_vectors = DenseEigenvectors::from_host_column_major(
        2,
        1,
        vec![Complex64::new(0.6, 0.0), Complex64::new(0.0, 0.8)],
    )
    .unwrap();
    let phase = Complex64::from_polar(1.0, 0.7);
    assert!(phase.im.abs() > 0.5);
    let right_vectors = DenseEigenvectors::from_host_column_major(
        2,
        1,
        vec![left_vectors.at(0, 0) * phase, left_vectors.at(1, 0) * phase],
    )
    .unwrap();
    assert!((left_vectors.at(0, 0) - right_vectors.at(0, 0)).norm() > 0.1);

    let overlap = left_vectors.at(0, 0).conj() * right_vectors.at(0, 0)
        + left_vectors.at(1, 0).conj() * right_vectors.at(1, 0);
    let aligned = phase_aligning_right_to_left(&left_vectors, &right_vectors, 0);
    assert!((aligned - overlap.conj() / overlap.norm()).norm() < 1.0e-14);
    for row in 0..2 {
        assert!((left_vectors.at(row, 0) - aligned * right_vectors.at(row, 0)).norm() < 1.0e-14);
    }

    let left = synthetic_solution(left_vectors);
    let right = synthetic_solution(right_vectors);
    solutions_match(&left, &right, ENERGY_TOL, VEC_TOL);
}
