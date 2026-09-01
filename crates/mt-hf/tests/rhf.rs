use muffintin_core::Hartree;
use muffintin_hf::{RestrictedHfError, RestrictedHfProblem, RestrictedHfSpec, solve_restricted_hf};
use muffintin_tensor::{Axis, DenseHermitianMatrix};
use num_complex::Complex64;

fn diagonal(values: [f64; 2]) -> DenseHermitianMatrix {
    DenseHermitianMatrix::from_upper_triangle(2, Axis::GlobalBasis, |row, column| {
        if row == column {
            Complex64::new(values[row], 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })
    .unwrap()
}

fn two_level_problem(electron_count: usize) -> RestrictedHfProblem {
    let mut chemist_eri = vec![0.0; 16];
    // (((0 * 2 + 0) * 2 + 0) * 2 + 0): (00|00) = 0.7 Ha.
    chemist_eri[0] = 0.7;
    RestrictedHfProblem {
        overlap: diagonal([1.0, 1.0]),
        one_electron: diagonal([-1.0, -0.2]),
        chemist_eri,
        electron_count,
        nuclear_repulsion: Hartree(0.2),
    }
}

#[test]
fn two_electron_feedback_energy_and_metric_trace_match_manual_oracle() {
    let spec = RestrictedHfSpec {
        max_iterations: 8,
        energy_tolerance: Hartree(1.0e-13),
        density_tolerance: 1.0e-13,
        density_mixing: 1.0,
        overlap_threshold: 1.0e-12,
    };
    let problem = two_level_problem(2);
    let result = solve_restricted_hf(&problem, &spec).unwrap();

    // P_00 = 2 gives J_00 = 1.4 and K_00 = 1.4, so F_00 = -0.3 Ha.
    // This differs from the -1.0 Ha core-H eigenvalue and proves Fock feedback.
    assert!((result.orbital_energies[0].get() + 0.3).abs() < 1.0e-13);
    assert!((result.orbital_energies[1].get() + 0.2).abs() < 1.0e-13);
    assert_eq!(result.iterations, 1);
    assert_eq!(result.diagnostics.len(), 1);
    assert!(result.diagnostics[0].fixed_point_density_rms < 1.0e-13);

    // 1/2 Tr[P(h+F)] = 1/2 * 2 * (-1.0 - 0.3) = -1.3 Ha.
    assert!((result.electronic_energy.get() + 1.3).abs() < 1.0e-13);
    assert!((result.total_energy.get() + 1.1).abs() < 1.0e-13);

    let mut metric_trace = Complex64::new(0.0, 0.0);
    for mu in 0..2 {
        for nu in 0..2 {
            metric_trace += result.density.at(nu, mu) * problem.overlap.at(mu, nu);
        }
    }
    assert!((metric_trace.re - 2.0).abs() < 1.0e-13);
    assert!(metric_trace.im.abs() < 1.0e-13);

    let odd = solve_restricted_hf(&two_level_problem(1), &spec).unwrap_err();
    assert_eq!(odd, RestrictedHfError::OddElectronCount(1));
}

#[test]
fn tiny_mixing_cannot_hide_an_occupied_subspace_switch() {
    let mut problem = two_level_problem(2);
    // P_00 = 2 raises F_00 to +0.2 Ha, above F_11 = -0.2 Ha, so the first
    // Fock solve switches the occupied subspace from AO 0 to AO 1.
    problem.chemist_eri[0] = 1.2;
    let error = solve_restricted_hf(
        &problem,
        &RestrictedHfSpec {
            max_iterations: 1,
            energy_tolerance: Hartree(1.0e100),
            density_tolerance: 1.0e-6,
            density_mixing: 1.0e-12,
            overlap_threshold: 1.0e-12,
        },
    )
    .unwrap_err();

    let RestrictedHfError::NotConverged {
        fixed_point_density_rms,
        ..
    } = error
    else {
        panic!("expected a fixed-point convergence failure, got {error:?}");
    };
    assert!((fixed_point_density_rms - 2.0_f64.sqrt()).abs() < 1.0e-13);
}
