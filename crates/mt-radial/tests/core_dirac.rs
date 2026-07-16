use mt_core::{Bohr, ExponentialMesh, Hartree};
use mt_radial::{
    CoreDiracSpec, CoreState, EnergyBracket, Kappa, KappaError, RelativisticRole,
    SPEX_SPEED_OF_LIGHT, ValenceDiracSpec, solve_core_dirac, solve_valence_dirac,
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
fn reserved_valence_dirac_is_a_typed_unsupported_error() {
    let mesh = extended_mesh(1.0e-6, 10.0, 0.01);
    let potential: Vec<f64> = mesh
        .radii()
        .iter()
        .map(|radius| -1.0 / radius.get())
        .collect();
    let spec = ValenceDiracSpec::new(Kappa::new(-1).unwrap(), Hartree(-0.5)).unwrap();
    assert_eq!(
        solve_valence_dirac(&mesh, &potential, spec),
        Err(KappaError::Unsupported {
            role: RelativisticRole::Valence,
        })
    );
}
