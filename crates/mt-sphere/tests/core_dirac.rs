use muffintin_core::{Bohr, ExponentialMesh, Hartree, Kappa};
use muffintin_sphere::{
    CoreDiracExchangeAction, CoreDiracSourcedSpec, CoreDiracSpec, CoreState, DiracError,
    EnergyBracket, SPEX_SPEED_OF_LIGHT, ValenceDiracSpec, solve_core_dirac,
    solve_core_dirac_with_action, solve_valence_dirac,
};

fn extended_mesh(first: f64, last: f64, increment: f64) -> ExponentialMesh {
    let count = ((last / first).ln() / increment).ceil() as usize + 1;
    ExponentialMesh::new(Bohr(first), increment, count).unwrap()
}

#[test]
fn hydrogenic_coulomb_1s_has_the_shifted_dirac_energy_and_physical_norm() {
    let mesh = extended_mesh(1.0e-7, 40.0, 0.002);
    let potential: Vec<f64> = mesh
        .radii()
        .iter()
        .map(|radius| -1.0 / radius.get())
        .collect();
    let state = CoreState::new(1, Kappa::new(-1).unwrap()).unwrap();
    let muffin_tin_radius = *mesh
        .radii()
        .iter()
        .min_by(|a, b| (a.get() - 6.0).abs().total_cmp(&(b.get() - 6.0).abs()))
        .unwrap();
    let spec = CoreDiracSpec::new(
        state,
        1.0,
        EnergyBracket::from_values(-0.6, -0.4).unwrap(),
        muffin_tin_radius,
    );
    let solution = solve_core_dirac(&mesh, &potential, spec).unwrap();

    let c = SPEX_SPEED_OF_LIGHT;
    let exact = c * c * ((1.0 - 1.0 / (c * c)).sqrt() - 1.0);
    assert!((solution.energy.get() - exact).abs() < 2.0e-7);
    assert!((solution.norm_total - 1.0).abs() < 2.0e-13);
    assert!((solution.norm_mt + solution.norm_outside - 1.0).abs() < 2.0e-13);
    assert_eq!(solution.spill, solution.norm_outside);
    assert!(solution.norm_outside > 0.0);
    assert!(solution.norm_outside < 1.0e-3);
    assert_eq!(solution.nodes, 0);
    assert!(solution.matching_residual.abs() < 2.0e-9);
    assert_eq!(solution.p.len(), mesh.len());
    assert_eq!(solution.q.len(), mesh.len());
}

#[test]
fn zero_exchange_action_is_identical_to_homogeneous_core_solve() {
    let mesh = extended_mesh(1.0e-7, 40.0, 0.002);
    let potential = mesh
        .radii()
        .iter()
        .map(|radius| -1.0 / radius.get())
        .collect::<Vec<_>>();
    let state = CoreState::new(1, Kappa::new(-1).unwrap()).unwrap();
    let muffin_tin_radius = *mesh
        .radii()
        .iter()
        .min_by(|a, b| (a.get() - 6.0).abs().total_cmp(&(b.get() - 6.0).abs()))
        .unwrap();
    let spec = CoreDiracSpec::new(
        state,
        1.0,
        EnergyBracket::from_values(-0.6, -0.4).unwrap(),
        muffin_tin_radius,
    );
    let homogeneous = solve_core_dirac(&mesh, &potential, spec).unwrap();
    let zeros = vec![0.0; mesh.len()];
    let with_action = solve_core_dirac_with_action(
        &mesh,
        &potential,
        CoreDiracSourcedSpec::new(spec, homogeneous.energy),
        CoreDiracExchangeAction {
            p: &zeros,
            q: &zeros,
        },
    )
    .unwrap();

    assert_eq!(with_action, homogeneous);
}

#[test]
fn manufactured_exchange_action_closes_source_equations_and_norm_root() {
    let mesh = extended_mesh(1.0e-7, 40.0, 0.002);
    let potential = mesh
        .radii()
        .iter()
        .map(|radius| -1.0 / radius.get())
        .collect::<Vec<_>>();
    let kappa = Kappa::new(-1).unwrap();
    let state = CoreState::new(1, kappa).unwrap();
    let muffin_tin_radius = *mesh
        .radii()
        .iter()
        .min_by(|a, b| (a.get() - 6.0).abs().total_cmp(&(b.get() - 6.0).abs()))
        .unwrap();
    let homogeneous_spec = CoreDiracSpec::new(
        state,
        1.0,
        EnergyBracket::from_values(-0.6, -0.4).unwrap(),
        muffin_tin_radius,
    );
    let homogeneous = solve_core_dirac(&mesh, &potential, homogeneous_spec).unwrap();

    // If H0 psi0 = E0 psi0, the fixed action K psi = delta psi0 makes
    // psi0 the unit-norm source-driven solution at E0 + delta. The other
    // norm root E0 - delta shares this provenance window; the explicit
    // Picard prediction selects the physical positive-action branch.
    let delta = 0.04;
    let action_p = homogeneous
        .p
        .iter()
        .map(|value| delta * value)
        .collect::<Vec<_>>();
    let action_q = homogeneous
        .q
        .iter()
        .map(|value| delta * value)
        .collect::<Vec<_>>();
    let expected_energy = homogeneous.energy.get() + delta;
    let driven_spec = CoreDiracSpec::new(
        state,
        1.0,
        EnergyBracket::from_values(-0.56, -0.44).unwrap(),
        muffin_tin_radius,
    )
    .with_tolerances(1.0e-10, 1.0e-8, 160);
    let driven = solve_core_dirac_with_action(
        &mesh,
        &potential,
        CoreDiracSourcedSpec::new(driven_spec, Hartree(expected_energy)),
        CoreDiracExchangeAction {
            p: &action_p,
            q: &action_q,
        },
    )
    .unwrap();

    assert!(
        (driven.energy.get() - expected_energy).abs() < 3.0e-6,
        "source-driven energy {} Ha, expected {expected_energy} Ha",
        driven.energy.get()
    );
    assert!((driven.norm_total - 1.0).abs() < 3.0e-13);
    assert_eq!(driven.nodes, state.expected_nodes());
    assert!(driven.matching_residual.abs() < 2.0e-12);
    let overlap_samples = homogeneous
        .p
        .iter()
        .zip(&homogeneous.q)
        .zip(&driven.p)
        .zip(&driven.q)
        .map(|(((&p0, &q0), &p), &q)| p0 * p + q0 * q)
        .collect::<Vec<_>>();
    assert!(mesh.integrate(&overlap_samples).unwrap() > 1.0 - 2.0e-5);

    let energy = driven.energy.get();
    let c = SPEX_SPEED_OF_LIGHT;
    let kappa = f64::from(kappa.get());
    for index in (24..mesh.len() - 1).step_by(211) {
        if driven.p[index - 1] == 0.0 || driven.p[index + 1] == 0.0 {
            continue;
        }
        let radius = mesh.radii()[index].get();
        let mass = 2.0 + (energy - potential[index]) / (c * c);
        let p_rhs =
            mass * c * driven.q[index] - kappa * driven.p[index] / radius - action_q[index] / c;
        let q_rhs = ((potential[index] - energy) * driven.p[index]
            + kappa * c * driven.q[index] / radius
            + action_p[index])
            / c;
        let p_numeric = sampled_derivative(&mesh, &driven.p, index);
        let q_numeric = sampled_derivative(&mesh, &driven.q, index);
        assert!((p_numeric - p_rhs).abs() <= 6.0e-5 * p_rhs.abs().max(1.0));
        assert!((q_numeric - q_rhs).abs() <= 6.0e-5 * q_rhs.abs().max(1.0));
    }
}

#[test]
fn explicit_charge_initializes_core_on_a_shallow_fleur_like_mesh() {
    let mesh = extended_mesh(1.0e-4, 40.0, 0.02);
    let regular = 20_000.0;
    let potential = mesh
        .radii()
        .iter()
        .map(|radius| -1.0 / radius.get() + regular)
        .collect::<Vec<_>>();
    let inferred_from_first_four = mesh.radii()[..4]
        .iter()
        .zip(&potential[..4])
        .map(|(radius, &value)| -radius.get() * value)
        .sum::<f64>()
        / 4.0;
    assert!(inferred_from_first_four < 0.0);

    let state = CoreState::new(1, Kappa::new(-1).unwrap()).unwrap();
    let muffin_tin_radius = *mesh
        .radii()
        .iter()
        .min_by(|left, right| {
            (left.get() - 6.0)
                .abs()
                .total_cmp(&(right.get() - 6.0).abs())
        })
        .unwrap();
    let spec = CoreDiracSpec::new(
        state,
        1.0,
        EnergyBracket::from_values(regular - 0.6, regular - 0.4).unwrap(),
        muffin_tin_radius,
    );
    let solution = solve_core_dirac(&mesh, &potential, spec).unwrap();
    let exact = regular
        + SPEX_SPEED_OF_LIGHT.powi(2) * ((1.0 - 1.0 / SPEX_SPEED_OF_LIGHT.powi(2)).sqrt() - 1.0);
    assert!((solution.energy.get() - exact).abs() < 2.0e-5);
    assert_eq!(solution.nodes, 0);
}

#[test]
fn core_dirac_rejects_invalid_explicit_nuclear_charge() {
    let mesh = extended_mesh(1.0e-4, 40.0, 0.02);
    let potential = mesh
        .radii()
        .iter()
        .map(|radius| -1.0 / radius.get())
        .collect::<Vec<_>>();
    let state = CoreState::new(1, Kappa::new(-1).unwrap()).unwrap();
    let muffin_tin_radius = mesh.radii()[mesh.len() / 2];
    for nuclear_charge in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        let spec = CoreDiracSpec::new(
            state,
            nuclear_charge,
            EnergyBracket::from_values(-0.6, -0.4).unwrap(),
            muffin_tin_radius,
        );
        assert!(matches!(
            solve_core_dirac(&mesh, &potential, spec),
            Err(DiracError::InvalidNuclearCharge(value))
                if value.to_bits() == nuclear_charge.to_bits()
        ));
    }
}

fn valence_fixture(
    energy: f64,
) -> (
    ExponentialMesh,
    Vec<f64>,
    muffintin_sphere::ValenceDiracSolution,
) {
    valence_fixture_for_kappa(energy, Kappa::new(-1).unwrap())
}

fn valence_fixture_for_kappa(
    energy: f64,
    kappa: Kappa,
) -> (
    ExponentialMesh,
    Vec<f64>,
    muffintin_sphere::ValenceDiracSolution,
) {
    let mesh = extended_mesh(1.0e-6, 8.0, 0.003);
    let potential: Vec<f64> = mesh
        .radii()
        .iter()
        .map(|radius| -1.0 / radius.get())
        .collect();
    let spec = ValenceDiracSpec::new(kappa, Hartree(energy)).unwrap();
    let solution = solve_valence_dirac(&mesh, &potential, spec).unwrap();
    (mesh, potential, solution)
}

fn sampled_derivative(mesh: &ExponentialMesh, values: &[f64], i: usize) -> f64 {
    let xa = mesh.radii()[i - 1].get();
    let xb = mesh.radii()[i].get();
    let xc = mesh.radii()[i + 1].get();
    let ha = xb - xa;
    let hc = xc - xb;
    -hc * values[i - 1] / (ha * (ha + hc))
        + (hc - ha) * values[i] / (ha * hc)
        + ha * values[i + 1] / (hc * (ha + hc))
}

#[test]
fn regular_valence_dirac_satisfies_both_first_order_systems() {
    let energy = -0.3;
    let (mesh, potential, solution) = valence_fixture(energy);
    let c = solution.speed_of_light;
    let kappa = f64::from(solution.kappa.get());
    for i in (16..mesh.len() - 1).step_by(97) {
        let r = mesh.radii()[i].get();
        let mass = 2.0 + (energy - potential[i]) / (c * c);
        let p_rhs = mass * c * solution.q[i] - kappa * solution.p[i] / r;
        let q_rhs = ((potential[i] - energy) * solution.p[i] + kappa * c * solution.q[i] / r) / c;
        let p_numeric = sampled_derivative(&mesh, &solution.p, i);
        let q_numeric = sampled_derivative(&mesh, &solution.q, i);
        assert!((p_numeric - p_rhs).abs() <= 4.0e-5 * p_rhs.abs().max(1.0));
        assert!((q_numeric - q_rhs).abs() <= 4.0e-5 * q_rhs.abs().max(1.0));

        let dot = &solution.energy_derivative;
        let p_dot_rhs = mass * c * dot.q[i] + solution.q[i] / c - kappa * dot.p[i] / r;
        let q_dot_rhs =
            ((potential[i] - energy) * dot.p[i] - solution.p[i] + kappa * c * dot.q[i] / r) / c;
        let p_dot_numeric = sampled_derivative(&mesh, &dot.p, i);
        let q_dot_numeric = sampled_derivative(&mesh, &dot.q, i);
        assert!((p_dot_numeric - p_dot_rhs).abs() <= 8.0e-5 * p_dot_rhs.abs().max(1.0));
        assert!((q_dot_numeric - q_dot_rhs).abs() <= 8.0e-5 * q_dot_rhs.abs().max(1.0));

        let second = &solution.second_energy_derivative;
        let p_second_rhs = mass * c * second.q[i] + 2.0 * dot.q[i] / c - kappa * second.p[i] / r;
        let q_second_rhs = ((potential[i] - energy) * second.p[i] - 2.0 * dot.p[i]
            + kappa * c * second.q[i] / r)
            / c;
        let p_second_numeric = sampled_derivative(&mesh, &second.p, i);
        let q_second_numeric = sampled_derivative(&mesh, &second.q, i);
        assert!((p_second_numeric - p_second_rhs).abs() <= 1.5e-4 * p_second_rhs.abs().max(1.0));
        assert!((q_second_numeric - q_second_rhs).abs() <= 1.5e-4 * q_second_rhs.abs().max(1.0));
    }
}

#[test]
fn valence_normalization_derivative_gauge_and_sra_boundary_are_exact() {
    let (mesh, _potential, solution) = valence_fixture(-0.3);
    let density: Vec<f64> = solution
        .p
        .iter()
        .zip(&solution.q)
        .map(|(&p, &q)| p * p + q * q)
        .collect();
    let cross: Vec<f64> = solution
        .p
        .iter()
        .zip(&solution.q)
        .zip(&solution.energy_derivative.p)
        .zip(&solution.energy_derivative.q)
        .map(|(((&p, &q), &p_dot), &q_dot)| p * p_dot + q * q_dot)
        .collect();
    let second_identity: Vec<f64> = solution
        .p
        .iter()
        .zip(&solution.q)
        .zip(&solution.energy_derivative.p)
        .zip(&solution.energy_derivative.q)
        .zip(&solution.second_energy_derivative.p)
        .zip(&solution.second_energy_derivative.q)
        .map(|(((((&p, &q), &p_dot), &q_dot), &p_second), &q_second)| {
            p * p_second + q * q_second + p_dot * p_dot + q_dot * q_dot
        })
        .collect();
    assert!((mesh.integrate(&density).unwrap() - 1.0).abs() < 3.0e-13);
    assert!(mesh.integrate(&cross).unwrap().abs() < 3.0e-13);
    assert!(mesh.integrate(&second_identity).unwrap().abs() < 2.0e-12);

    let radius = mesh.last().get();
    let sra = solution.sra_boundary();
    assert_eq!(sra.value, solution.boundary.p / radius);
    assert_eq!(
        sra.derivative,
        solution.boundary.p_derivative / radius - solution.boundary.p / (radius * radius)
    );
    assert_ne!(sra.value, solution.boundary.p);

    let derivative_sra = solution.energy_derivative.boundary.sra_large_component();
    assert_eq!(
        derivative_sra.value,
        solution.energy_derivative.boundary.p / radius
    );
    assert_eq!(
        derivative_sra.derivative,
        solution.energy_derivative.boundary.p_derivative / radius
            - solution.energy_derivative.boundary.p / (radius * radius)
    );
}

#[test]
fn analytic_second_derivative_matches_phase_aligned_centered_difference() {
    let energy = -0.3;
    let step = 2.0e-4;
    let (mesh, _, center) = valence_fixture(energy);
    let (_, _, minus) = valence_fixture(energy - step);
    let (_, _, plus) = valence_fixture(energy + step);
    let overlap = |candidate: &muffintin_sphere::ValenceDiracSolution| {
        let values: Vec<f64> = center
            .p
            .iter()
            .zip(&center.q)
            .zip(&candidate.p)
            .zip(&candidate.q)
            .map(|(((&p, &q), &other_p), &other_q)| p * other_p + q * other_q)
            .collect();
        mesh.integrate(&values).unwrap()
    };
    let minus_phase = overlap(&minus).signum();
    let plus_phase = overlap(&plus).signum();
    let second_difference = |hi: f64, mid: f64, lo: f64| {
        (plus_phase * hi - 2.0 * mid + minus_phase * lo) / (step * step)
    };

    let large_errors: Vec<f64> = plus
        .p
        .iter()
        .zip(&center.p)
        .zip(&minus.p)
        .zip(&center.second_energy_derivative.p)
        .map(|(((&hi, &mid), &lo), &analytic)| {
            let error = second_difference(hi, mid, lo) - analytic;
            error * error
        })
        .collect();
    let small_errors: Vec<f64> = plus
        .q
        .iter()
        .zip(&center.q)
        .zip(&minus.q)
        .zip(&center.second_energy_derivative.q)
        .map(|(((&hi, &mid), &lo), &analytic)| {
            let error = second_difference(hi, mid, lo) - analytic;
            error * error
        })
        .collect();
    let error =
        (mesh.integrate(&large_errors).unwrap() + mesh.integrate(&small_errors).unwrap()).sqrt();
    assert!(error < 3.0e-5, "second-energy-derivative L2 error {error}");

    let trace = center.second_energy_derivative.boundary;
    assert!(
        (second_difference(plus.boundary.p, center.boundary.p, minus.boundary.p) - trace.p).abs()
            < 3.0e-5
    );
    assert!(
        (second_difference(plus.boundary.q, center.boundary.q, minus.boundary.q) - trace.q).abs()
            < 3.0e-7
    );
    assert!(
        (second_difference(
            plus.boundary.p_derivative,
            center.boundary.p_derivative,
            minus.boundary.p_derivative,
        ) - trace.p_derivative)
            .abs()
            < 8.0e-5
    );
    assert!(
        (second_difference(
            plus.boundary.q_derivative,
            center.boundary.q_derivative,
            minus.boundary.q_derivative,
        ) - trace.q_derivative)
            .abs()
            < 3.0e-7
    );
}

#[test]
fn sra_hdlo_is_confined_normalized_and_retains_its_small_component() {
    let (mesh, _, solution) = valence_fixture(-0.3);
    let hdlo = solution.sra_hdlo(&mesh).unwrap();
    assert_eq!(hdlo.kappa, solution.kappa);
    assert!(hdlo.boundary.value.abs() <= 1.0e-10);
    assert!(hdlo.boundary.derivative.abs() <= 1.0e-10);
    let density: Vec<f64> = hdlo
        .p
        .iter()
        .zip(&hdlo.q)
        .map(|(&p, &q)| p * p + q * q)
        .collect();
    assert!((mesh.integrate(&density).unwrap() - 1.0).abs() < 2.0e-12);
    assert!(hdlo.q.iter().any(|value| value.abs() > 1.0e-10));
}

#[test]
fn distinct_energy_p_half_sra_local_orbital_is_confined_and_four_component_normalized() {
    let kappa = Kappa::new(1).unwrap();
    let (mesh, _, base) = valence_fixture_for_kappa(-0.3, kappa);
    let (_, _, raw) = valence_fixture_for_kappa(0.4, kappa);
    let local = base.sra_local_orbital(&raw, &mesh).unwrap();

    assert_eq!(local.energy, raw.energy);
    assert_eq!(local.kappa, kappa);
    assert!(local.boundary.value.abs() <= 1.0e-10);
    assert!(local.boundary.derivative.abs() <= 1.0e-10);
    let coefficients = local.coefficients;
    let p_boundary = coefficients.normalization_scale
        * (raw.boundary.p
            + coefficients.a * base.boundary.p
            + coefficients.b * base.energy_derivative.boundary.p);
    let p_derivative_boundary = coefficients.normalization_scale
        * (raw.boundary.p_derivative
            + coefficients.a * base.boundary.p_derivative
            + coefficients.b * base.energy_derivative.boundary.p_derivative);
    assert!(p_boundary.abs() <= 1.0e-10);
    assert!(p_derivative_boundary.abs() <= 1.0e-10);
    let density: Vec<f64> = local
        .p
        .iter()
        .zip(&local.q)
        .map(|(&p, &q)| p * p + q * q)
        .collect();
    assert!((mesh.integrate(&density).unwrap() - 1.0).abs() < 2.0e-12);
    assert!(local.q.iter().any(|value| value.abs() > 1.0e-10));
}

#[test]
fn distinct_lo_energy_changes_the_p_half_local_orbital() {
    let kappa = Kappa::new(1).unwrap();
    let (mesh, _, base) = valence_fixture_for_kappa(-0.3, kappa);
    let (_, _, raw_low) = valence_fixture_for_kappa(0.1, kappa);
    let (_, _, raw_high) = valence_fixture_for_kappa(1.2, kappa);
    let low = base.sra_local_orbital(&raw_low, &mesh).unwrap();
    let high = base.sra_local_orbital(&raw_high, &mesh).unwrap();
    let overlap_samples: Vec<f64> = low
        .p
        .iter()
        .zip(&low.q)
        .zip(&high.p)
        .zip(&high.q)
        .map(|(((&low_p, &low_q), &high_p), &high_q)| low_p * high_p + low_q * high_q)
        .collect();
    let overlap = mesh.integrate(&overlap_samples).unwrap().abs();
    assert!(overlap < 1.0 - 1.0e-8, "local-orbital overlap {overlap}");
}

#[test]
fn distinct_energy_sra_local_orbital_rejects_incompatible_inputs() {
    let kappa = Kappa::new(1).unwrap();
    let (mesh, _, base) = valence_fixture_for_kappa(-0.3, kappa);
    let (_, _, raw) = valence_fixture_for_kappa(0.4, kappa);

    let (_, _, wrong_kappa) = valence_fixture_for_kappa(0.4, Kappa::new(-1).unwrap());
    assert!(matches!(
        base.sra_local_orbital(&wrong_kappa, &mesh),
        Err(DiracError::LocalOrbitalKappaMismatch { base: 1, raw: -1 })
    ));

    let mut wrong_speed = raw.clone();
    wrong_speed.speed_of_light = 40.0;
    assert!(matches!(
        base.sra_local_orbital(&wrong_speed, &mesh),
        Err(DiracError::LocalOrbitalSpeedOfLightMismatch { base: _, raw: 40.0 })
    ));

    let shorter_mesh =
        ExponentialMesh::new(mesh.first(), mesh.increment(), mesh.len() - 1).unwrap();
    assert!(matches!(
        base.sra_local_orbital(&raw, &shorter_mesh),
        Err(DiracError::LocalOrbitalSampleCountMismatch {
            field: "base.p",
            mesh: expected,
            actual,
        }) if expected == mesh.len() - 1 && actual == mesh.len()
    ));

    let shifted_mesh = ExponentialMesh::new(
        Bohr(mesh.first().get() * 1.01),
        mesh.increment(),
        mesh.len(),
    )
    .unwrap();
    assert!(matches!(
        base.sra_local_orbital(&raw, &shifted_mesh),
        Err(DiracError::LocalOrbitalBoundaryRadiusMismatch {
            field: "base.boundary",
            mesh: expected,
            actual,
        }) if expected == shifted_mesh.last().get() && actual == mesh.last().get()
    ));

    assert!(matches!(
        base.sra_local_orbital(&base, &mesh),
        Err(DiracError::LocalOrbitalEnergyNotDistinct { energy: -0.3 })
    ));
}

#[test]
fn analytic_energy_derivative_matches_phase_aligned_centered_difference() {
    let energy = -0.3;
    let step = 2.0e-5;
    let (mesh, _, center) = valence_fixture(energy);
    let (_, _, minus) = valence_fixture(energy - step);
    let (_, _, plus) = valence_fixture(energy + step);
    let overlap = |candidate: &muffintin_sphere::ValenceDiracSolution| {
        let values: Vec<f64> = center
            .p
            .iter()
            .zip(&center.q)
            .zip(&candidate.p)
            .zip(&candidate.q)
            .map(|(((&p, &q), &other_p), &other_q)| p * other_p + q * other_q)
            .collect();
        mesh.integrate(&values).unwrap()
    };
    let minus_phase = overlap(&minus).signum();
    let plus_phase = overlap(&plus).signum();

    let differences: Vec<f64> = plus
        .p
        .iter()
        .zip(&minus.p)
        .zip(&center.energy_derivative.p)
        .map(|((&hi, &lo), &analytic)| {
            let error = (plus_phase * hi - minus_phase * lo) / (2.0 * step) - analytic;
            error * error
        })
        .collect();
    let small_differences: Vec<f64> = plus
        .q
        .iter()
        .zip(&minus.q)
        .zip(&center.energy_derivative.q)
        .map(|((&hi, &lo), &analytic)| {
            let error = (plus_phase * hi - minus_phase * lo) / (2.0 * step) - analytic;
            error * error
        })
        .collect();
    let error = (mesh.integrate(&differences).unwrap()
        + mesh.integrate(&small_differences).unwrap())
    .sqrt();
    assert!(error < 2.0e-7, "energy-derivative L2 error {error}");

    let finite_difference = |hi: f64, lo: f64| (plus_phase * hi - minus_phase * lo) / (2.0 * step);
    let trace = center.energy_derivative.boundary;
    assert!((finite_difference(plus.boundary.p, minus.boundary.p) - trace.p).abs() < 2.0e-7);
    assert!((finite_difference(plus.boundary.q, minus.boundary.q) - trace.q).abs() < 2.0e-9);
    assert!(
        (finite_difference(plus.boundary.p_derivative, minus.boundary.p_derivative,)
            - trace.p_derivative)
            .abs()
            < 5.0e-7
    );
    assert!(
        (finite_difference(plus.boundary.q_derivative, minus.boundary.q_derivative,)
            - trace.q_derivative)
            .abs()
            < 2.0e-9
    );
}

#[test]
fn valence_dirac_rejects_a_nonpositive_mass_away_from_the_origin() {
    let mesh = extended_mesh(1.0e-6, 0.1, 0.1);
    let energy = Hartree(-0.3);
    let mut potential = vec![-1.0; mesh.len()];
    let bad_index = mesh.len() / 2;
    potential[bad_index] = energy.get() + 2.0 * SPEX_SPEED_OF_LIGHT.powi(2);
    let spec = ValenceDiracSpec::new(Kappa::new(-1).unwrap(), energy).unwrap();
    match solve_valence_dirac(&mesh, &potential, spec) {
        Err(DiracError::InvalidRelativisticMass { index, mass }) => {
            assert_eq!(index, bad_index);
            assert!(mass <= 0.0);
        }
        result => panic!("expected an indexed mass-factor error, got {result:?}"),
    }
}

#[test]
fn valence_speed_of_light_override_controls_physical_small_components() {
    let mesh = extended_mesh(1.0e-6, 8.0, 0.003);
    let potential: Vec<f64> = mesh
        .radii()
        .iter()
        .map(|radius| -1.0 / radius.get())
        .collect();
    let c = 40.0;
    let spec = ValenceDiracSpec::new(Kappa::new(-1).unwrap(), Hartree(-0.3))
        .unwrap()
        .with_speed_of_light(c)
        .unwrap();
    let solution = solve_valence_dirac(&mesh, &potential, spec).unwrap();
    assert_eq!(solution.speed_of_light, c);
    let i = mesh.len() / 2;
    let r = mesh.radii()[i].get();
    let mass = 2.0 + (-0.3 - potential[i]) / (c * c);
    let kappa = f64::from(solution.kappa.get());
    let p_rhs = mass * c * solution.q[i] - kappa * solution.p[i] / r;
    let q_rhs = ((potential[i] + 0.3) * solution.p[i] + kappa * c * solution.q[i] / r) / c;
    assert!((sampled_derivative(&mesh, &solution.p, i) - p_rhs).abs() < 5.0e-5);
    assert!((sampled_derivative(&mesh, &solution.q, i) - q_rhs).abs() < 5.0e-5);

    let default = valence_fixture(-0.3).2;
    let custom_small_norm = solution.q.iter().map(|q| q * q).sum::<f64>();
    let default_small_norm = default.q.iter().map(|q| q * q).sum::<f64>();
    assert!(custom_small_norm > default_small_norm);
}

#[test]
fn valence_dirac_rejects_invalid_speed_of_light_and_uses_it_for_mass_validation() {
    let kappa = Kappa::new(-1).unwrap();
    let energy = Hartree(-0.3);
    for invalid in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(matches!(
            ValenceDiracSpec::new(kappa, energy)
                .unwrap()
                .with_speed_of_light(invalid),
            Err(DiracError::InvalidSpeedOfLight(value))
                if value.to_bits() == invalid.to_bits()
        ));
    }

    let mesh = extended_mesh(1.0e-6, 0.1, 0.1);
    let c = 20.0;
    let mut potential = vec![-1.0; mesh.len()];
    let bad_index = mesh.len() / 2;
    potential[bad_index] = energy.get() + 2.0 * c * c;
    let spec = ValenceDiracSpec::new(kappa, energy)
        .unwrap()
        .with_speed_of_light(c)
        .unwrap();
    assert!(matches!(
        solve_valence_dirac(&mesh, &potential, spec),
        Err(DiracError::InvalidRelativisticMass { index, mass })
            if index == bad_index && mass <= 0.0
    ));
}
