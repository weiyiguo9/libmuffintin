//! Repository-local large-`c` SRA regression against a frozen synthetic scalar fixture.

use muffintin_envelope::{
    ApwBoundaryBasis, ApwSiteAugmentation, ApwSiteGeometry, BasisBlock, BasisSpec, Provenance,
    SpinorApwMatch, SpinorBasisLayout, SpinorCompiledBasis, SpinorSiteLayout, compile,
    match_apw_boundary, spinor_augmentation_coefficients,
};
use muffintin_core::{
    Bohr, ExponentialMesh, Hartree, InterstitialGeometry, InverseBohr, Kappa, ReciprocalLattice,
    RelativisticChannel, Sphere, VolumeBohr3,
};
use muffintin_envelope::{PlaneWave, PlaneWaveEnvelope};
use muffintin_lapw::{
    InterstitialPauliPotential, InterstitialPotential, SiteOperatorBlocks,
    SpinorSiteOperatorBlocks, assemble_compiled, assemble_sra_spinor_compiled,
    solve_generalized_hermitian,
};
use muffintin_sphere::{BoundaryData, ValenceDiracSpec, solve_valence_dirac};
use muffintin_tensor::{Axis, DenseHermitianMatrix};
use num_complex::Complex64;

const LARGE_C: f64 = 1.0e6;
const S_TOL: f64 = 2.0e-9;
const H_TOL: f64 = 5.0e-9;
const EIG_TOL: f64 = 1.0e-8;

fn mesh_and_potential() -> (ExponentialMesh, Vec<f64>) {
    let mesh = ExponentialMesh::new(Bohr(1.0e-6), 0.006, 2_335).unwrap();
    let potential = mesh
        .radii()
        .iter()
        .map(|radius| -0.7 + 0.15 * radius.get().powi(2))
        .collect();
    (mesh, potential)
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
        .map(|g| {
            PlaneWave::new(
                [InverseBohr(0.13), InverseBohr(-0.17), InverseBohr(0.09)],
                g,
            )
        })
        .collect()
}

fn boundary(value: f64, derivative: f64, radius: Bohr) -> BoundaryData {
    BoundaryData {
        value,
        derivative,
        log_derivative: (value != 0.0).then(|| InverseBohr(derivative / value)),
        scaled_log_derivative: (value != 0.0).then(|| radius.get() * derivative / value),
    }
}

/// Frozen Schrödinger `(u, du/dE)` boundary data for the local synthetic
/// mesh, potential, and `E = -0.23 Ha`; this is not an external oracle.
fn frozen_scalar_boundaries(radius: Bohr) -> [ApwBoundaryBasis; 2] {
    [
        ApwBoundaryBasis {
            u: boundary(1.204_267_469_824_055, -0.358_723_554_435_703_6, radius),
            udot: boundary(-0.282_027_897_962_640_1, -1.055_156_846_317_608_6, radius),
        },
        ApwBoundaryBasis {
            u: boundary(1.634_947_280_386_641_4, 1.097_825_735_048_552, radius),
            udot: boundary(-0.148_250_453_995_328_77, -0.938_632_271_236_412_7, radius),
        },
    ]
}

fn site_matrix(
    dimension: usize,
    element: impl FnMut(usize, usize) -> Complex64,
) -> DenseHermitianMatrix {
    DenseHermitianMatrix::from_upper_triangle(dimension, Axis::SiteCoordinate, element).unwrap()
}

fn frozen_scalar_site() -> SiteOperatorBlocks {
    let overlap = site_matrix(8, |row, column| {
        if row / 2 != column / 2 {
            return Complex64::default();
        }
        let values = if row / 2 == 0 {
            [[1.0, 0.04], [0.04, 0.70]]
        } else {
            [[0.95, -0.03], [-0.03, 0.62]]
        };
        Complex64::new(values[row % 2][column % 2], 0.0)
    });
    let hamiltonian = site_matrix(8, |row, column| {
        let diagonal_block = if row / 2 == 0 {
            [[-0.25, 0.11], [0.11, 0.52]]
        } else {
            [[0.15, -0.08], [-0.08, 0.83]]
        };
        if row / 2 == column / 2 {
            return Complex64::new(diagonal_block[row % 2][column % 2], 0.0);
        }
        match (row, column) {
            (0, 2) => Complex64::new(0.07, 0.02),
            (1, 5) => Complex64::new(-0.04, 0.01),
            (2, 6) => Complex64::new(0.03, -0.025),
            _ => Complex64::default(),
        }
    });
    SiteOperatorBlocks {
        overlap,
        hamiltonian,
    }
}

fn cg_congruence(
    scalar: &DenseHermitianMatrix,
    channels: &[RelativisticChannel],
) -> DenseHermitianMatrix {
    site_matrix(2 * channels.len(), |row, column| {
        let left_channel = channels[row / 2];
        let right_channel = channels[column / 2];
        let left_radial = row % 2;
        let right_radial = column % 2;
        let mut value = Complex64::default();
        for left in left_channel.spinor_harmonic_terms().into_iter().flatten() {
            for right in right_channel.spinor_harmonic_terms().into_iter().flatten() {
                if left.spin == right.spin {
                    let scalar_row = 2 * left.orbital.index() + left_radial;
                    let scalar_column = 2 * right.orbital.index() + right_radial;
                    value +=
                        left.coefficient * scalar.at(scalar_row, scalar_column) * right.coefficient;
                }
            }
        }
        value
    })
}

fn maximum_difference(left: &DenseHermitianMatrix, right: &DenseHermitianMatrix) -> f64 {
    assert_eq!(left.dimension(), right.dimension());
    (0..left.dimension())
        .flat_map(|row| {
            (0..left.dimension())
                .map(move |column| (left.at(row, column) - right.at(row, column)).norm())
        })
        .fold(0.0, f64::max)
}

#[test]
fn large_c_sra_reduces_to_two_frozen_scalar_blocks() {
    let (mesh, radial_potential) = mesh_and_potential();
    let radius = mesh.last();
    let energy = Hartree(-0.23);
    let scalar_boundaries = frozen_scalar_boundaries(radius);

    let position = [Bohr(0.31), Bohr(-0.27), Bohr(0.19)];
    let volume = VolumeBohr3(125.0);
    let waves = waves();
    let geometry = InterstitialGeometry::new(
        volume,
        vec![Sphere {
            center: position,
            radius,
        }],
    )
    .unwrap();
    let scalar_compiled = compile(&BasisSpec {
        blocks: vec![BasisBlock::PlaneWaveEnvelope {
            envelope: PlaneWaveEnvelope::new(waves.clone()),
            sites: vec![ApwSiteAugmentation {
                position,
                radius,
                boundaries: scalar_boundaries.to_vec(),
            }],
        }],
        cell_volume: volume,
        provenance: Provenance {
            recipe: Some("local synthetic large-c regression".into()),
            reference: Some("repository-local frozen scalar fixture".into()),
        },
    })
    .unwrap();

    let kappas = [-2, -1, 1].map(|value| Kappa::new(value).unwrap());
    let spinor_boundaries = kappas.map(|kappa| {
        let solution = solve_valence_dirac(
            &mesh,
            &radial_potential,
            ValenceDiracSpec::new(kappa, energy)
                .unwrap()
                .with_speed_of_light(LARGE_C)
                .unwrap(),
        )
        .unwrap();
        (
            kappa,
            ApwBoundaryBasis {
                u: solution.sra_boundary(),
                udot: solution.energy_derivative.boundary.sra_large_component(),
            },
        )
    });
    let spinor_augmentations = waves
        .iter()
        .map(|wave| {
            let matches = spinor_boundaries.map(|(kappa, boundaries)| SpinorApwMatch {
                kappa,
                apw: match_apw_boundary(kappa.large_l(), wave.q_norm, radius, boundaries).unwrap(),
            });
            spinor_augmentation_coefficients(wave, position, volume, &matches).unwrap()
        })
        .collect::<Vec<_>>();
    let channels = spinor_augmentations[0].channels.clone();
    let spinor_compiled = SpinorCompiledBasis {
        layout: SpinorBasisLayout::new(waves.len(), vec![SpinorSiteLayout::default()]),
        plane_waves: waves.clone(),
        site_augmentations: vec![spinor_augmentations],
        site_geometry: vec![ApwSiteGeometry { position, radius }],
        provenance: Provenance {
            recipe: Some("local synthetic large-c regression".into()),
            reference: Some("repository-local frozen scalar fixture".into()),
        },
    };

    assert!(!scalar_compiled.site_augmentations[0].is_empty());
    assert!(
        scalar_compiled.site_augmentations[0]
            .iter()
            .all(|augmentation| !augmentation.coefficients.is_empty())
    );
    assert!(!spinor_compiled.site_augmentations[0].is_empty());
    assert!(
        spinor_compiled.site_augmentations[0]
            .iter()
            .all(|augmentation| {
                augmentation.channel_count() == 8
                    && augmentation
                        .coefficients
                        .iter()
                        .flatten()
                        .any(|pair| pair.iter().any(|coefficient| coefficient.norm() > 0.0))
            })
    );
    assert_eq!(channels.len(), 8);
    assert_eq!(
        channels
            .iter()
            .map(|channel| channel.kappa().get())
            .collect::<Vec<_>>(),
        vec![-2, -2, -2, -2, -1, -1, 1, 1]
    );

    let scalar_site = frozen_scalar_site();
    assert!(scalar_site.hamiltonian.at(0, 2).norm() > 0.0);
    assert!(scalar_site.hamiltonian.at(2, 6).norm() > 0.0);
    let spinor_site = SpinorSiteOperatorBlocks {
        channels: channels.clone(),
        overlap: cg_congruence(&scalar_site.overlap, &channels),
        hamiltonian: cg_congruence(&scalar_site.hamiltonian, &channels),
    };
    let mut different_kappa_coupling: f64 = 0.0;
    for (left, left_channel) in channels.iter().enumerate() {
        for (right, right_channel) in channels.iter().enumerate() {
            if left_channel.kappa() != right_channel.kappa() {
                different_kappa_coupling = different_kappa_coupling
                    .max(spinor_site.hamiltonian.at(2 * left, 2 * right).norm());
            }
        }
    }
    assert!(different_kappa_coupling > 1.0e-3);

    let potential = InterstitialPotential::new([
        ([0, 0, 0], Complex64::new(-0.08, 0.0)),
        ([1, 0, 0], Complex64::new(0.013, -0.007)),
    ])
    .unwrap();
    let scalar_problem = assemble_compiled(
        &scalar_compiled,
        &geometry,
        &potential,
        std::slice::from_ref(&scalar_site),
    )
    .unwrap();
    let spinor_problem = assemble_sra_spinor_compiled(
        &spinor_compiled,
        &geometry,
        &InterstitialPauliPotential::new(
            potential.clone(),
            InterstitialPotential::default(),
            InterstitialPotential::default(),
            InterstitialPotential::default(),
        ),
        std::slice::from_ref(&spinor_site),
    )
    .unwrap();

    let n_g = waves.len();
    for spin in 0..2 {
        for row in 0..n_g {
            for column in 0..n_g {
                let spinor_row = spinor_compiled.layout.plane_wave_index(spin, row).unwrap();
                let spinor_column = spinor_compiled
                    .layout
                    .plane_wave_index(spin, column)
                    .unwrap();
                assert!(
                    (spinor_problem.overlap.at(spinor_row, spinor_column)
                        - scalar_problem.overlap.at(row, column))
                    .norm()
                        < S_TOL
                );
                assert!(
                    (spinor_problem.hamiltonian.at(spinor_row, spinor_column)
                        - scalar_problem.hamiltonian.at(row, column))
                    .norm()
                        < H_TOL
                );
            }
        }
    }
    for row in 0..n_g {
        for column in 0..n_g {
            let up = spinor_compiled.layout.plane_wave_index(0, row).unwrap();
            let down = spinor_compiled.layout.plane_wave_index(1, column).unwrap();
            assert!(spinor_problem.overlap.at(up, down).norm() < S_TOL);
            assert!(spinor_problem.hamiltonian.at(up, down).norm() < H_TOL);
        }
    }

    let zero_scalar_site = SiteOperatorBlocks {
        overlap: site_matrix(8, |_, _| Complex64::default()),
        hamiltonian: site_matrix(8, |_, _| Complex64::default()),
    };
    let interstitial_only =
        assemble_compiled(&scalar_compiled, &geometry, &potential, &[zero_scalar_site]).unwrap();
    assert!(maximum_difference(&scalar_problem.overlap, &interstitial_only.overlap) > 1.0e-5);
    assert!(
        maximum_difference(&scalar_problem.hamiltonian, &interstitial_only.hamiltonian) > 1.0e-5
    );

    let scalar_solution = solve_generalized_hermitian(
        &scalar_problem.hamiltonian,
        &scalar_problem.overlap,
        1.0e-12,
    )
    .unwrap();
    let spinor_solution = solve_generalized_hermitian(
        &spinor_problem.hamiltonian,
        &spinor_problem.overlap,
        1.0e-12,
    )
    .unwrap();
    assert_eq!(
        spinor_solution.eigenvalues.len(),
        2 * scalar_solution.eigenvalues.len()
    );
    for (pair, scalar) in spinor_solution
        .eigenvalues
        .chunks_exact(2)
        .zip(&scalar_solution.eigenvalues)
    {
        assert!((pair[0].get() - pair[1].get()).abs() < EIG_TOL);
        assert!((pair[0].get() - scalar.get()).abs() < EIG_TOL);
        assert!((pair[1].get() - scalar.get()).abs() < EIG_TOL);
    }
}
