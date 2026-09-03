use super::*;
use muffintin::{FrozenCoreHfSpec, run_frozen_core_hf};

#[derive(Serialize)]
struct Iteration {
    iteration: usize,
    total_energy_hartree: f64,
    energy_change_hartree: Option<f64>,
    density_residual: f64,
    commutator_residual_hartree: f64,
    active_feedback_residual_hartree: Option<f64>,
    eigenvalue_identity_residual_hartree: f64,
    core_h0_trace_hartree: f64,
    exchange_vv_hartree: f64,
    exchange_cv_hartree: f64,
    exchange_cc_hartree: f64,
    virtual_level_shift_hartree: f64,
    converged: bool,
}

#[derive(Serialize)]
struct Iterations {
    status: &'static str,
    failure: Option<String>,
    iterations: Vec<Iteration>,
}

#[derive(Serialize)]
struct Orbital {
    k_index: usize,
    band_index: usize,
    energy_hartree: f64,
    homo_shifted_ev: f64,
    occupation: f64,
}

#[derive(Serialize)]
struct CoreSpace {
    k_index: usize,
    expanded_basis_dimension: usize,
    retained_spinor_bands: usize,
    constraint_count: usize,
    maximum_radial_overlap_residual: f64,
}

#[derive(Serialize)]
struct ResultRecord {
    status: &'static str,
    relativity: &'static str,
    core_treatment: &'static str,
    energy_reference: &'static str,
    total_energy_hartree: f64,
    homo_energy_hartree: f64,
    chemical_potential_hartree: f64,
    core_h0_trace_hartree: f64,
    sector_energies: SectorEnergies,
    cv_vc_trace_mismatch_hartree: f64,
    electron_counts: ElectronCounts,
    orbital_energies_and_occupations: Vec<Orbital>,
    core_shells: Vec<CoreShellRecord>,
    core_orthogonalization: Vec<CoreSpace>,
}

pub(super) fn run(
    cli: &Cli,
    start: &muffintin::AtomicStart,
    mut config: ScfConfig,
) -> Result<(), Box<dyn Error>> {
    // This is a single coupled Fock loop, not outer density mixing plus an inner loop.
    config.convergence.max_iterations = cli.max_fock_iterations;
    config.convergence.density_tolerance = cli.fock_density_tolerance;
    let spec = FrozenCoreHfSpec {
        config,
        product_l_max: cli.product_l_max,
        product_g_max: InverseBohr(cli.product_g),
        overlap_tolerance: DEFAULT_TOLERANCE,
        coulomb: exchange_coulomb_request(cli)?,
        gamma: GammaExchangeTreatment::FiniteBody,
        fock_mixing: cli.fock_mixing.build(
            cli.fock_mixing_alpha,
            cli.fock_diis_history,
            cli.fock_diis_level_shift,
            cli.fock_diis_startup_steps,
            cli.fock_diis_damping,
        ),
        feedback_tolerance: Hartree(cli.fock_feedback_tolerance),
        commutator_tolerance: Hartree(cli.fock_commutator_tolerance),
        virtual_level_shift: Hartree(cli.spinor_virtual_level_shift),
        sector_numerical_tolerance: Hartree(SECTOR_NUMERICAL_TOLERANCE_HARTREE),
    };
    let mut physics = CheckpointPhysics::new(&start.checkpoint)?;
    let mut iterations = Iterations {
        status: "running",
        failure: None,
        iterations: Vec::new(),
    };
    let path = cli.out.join("iterations.toml");
    let pending = cli.out.join("iterations.toml.next");
    let result = run_frozen_core_hf(&mut physics, &spec, |d| {
        iterations.iterations.push(Iteration {
            iteration: d.iteration,
            total_energy_hartree: d.total_energy.get(),
            energy_change_hartree: d.energy_change.map(Hartree::get),
            density_residual: d.density_residual,
            commutator_residual_hartree: d.commutator_residual.get(),
            active_feedback_residual_hartree: d.active_feedback_residual.map(Hartree::get),
            eigenvalue_identity_residual_hartree: d.eigenvalue_identity_residual.get(),
            core_h0_trace_hartree: d.core_h0_trace.get(),
            exchange_vv_hartree: d.exchange_vv.get(),
            exchange_cv_hartree: d.exchange_cv.get(),
            exchange_cc_hartree: d.exchange_cc.get(),
            virtual_level_shift_hartree: d.virtual_level_shift.get(),
            converged: d.converged,
        });
        write_toml(&pending, &iterations)?;
        fs::rename(&pending, &path)
    });
    iterations.status = if result.is_ok() {
        "configured_convergence_reached"
    } else {
        "failed"
    };
    iterations.failure = result.as_ref().err().map(ToString::to_string);
    write_toml(&pending, &iterations)?;
    fs::rename(&pending, &path)?;
    let result = result?;
    let homo = result
        .orbital_energies
        .iter()
        .zip(&result.occupations)
        .flat_map(|(energies, occupations)| energies.iter().zip(occupations))
        .filter(|(_, f)| **f >= HOMO_OCCUPATION_THRESHOLD)
        .map(|(e, _)| e.get())
        .reduce(f64::max)
        .ok_or_else(|| invalid_input("frozen SRA result has no occupied HOMO"))?;
    let orbitals = result
        .orbital_energies
        .iter()
        .zip(&result.occupations)
        .enumerate()
        .flat_map(|(k, (energies, occupations))| {
            energies
                .iter()
                .zip(occupations)
                .enumerate()
                .map(move |(band, (energy, f))| Orbital {
                    k_index: k,
                    band_index: band,
                    energy_hartree: energy.get(),
                    homo_shifted_ev: (energy.get() - homo) * 27.211386245988,
                    occupation: *f,
                })
        })
        .collect();
    let spaces = result
        .bands
        .points()
        .iter()
        .enumerate()
        .map(|(k, point)| {
            let CheckpointKPointSolution::Spinor {
                basis, solution, ..
            } = &point.solution
            else {
                unreachable!()
            };
            let core = basis
                .core_orthogonalization
                .as_ref()
                .expect("frozen SRA keeps its core embedding");
            CoreSpace {
                k_index: k,
                expanded_basis_dimension: solution.eigenvectors.rows(),
                retained_spinor_bands: solution.eigenvectors.columns(),
                constraint_count: core.constraint_count,
                maximum_radial_overlap_residual: core.maximum_radial_overlap_residual,
            }
        })
        .collect();
    let record = ResultRecord {
        status: "configured_convergence_reached",
        relativity: "SRA: 4c muffin tins, 2c interstitial",
        core_treatment: "frozen-checkpoint",
        energy_reference: "occupied HOMO shifted to zero; unshifted HF orbital energies",
        total_energy_hartree: result.total_energy.get(),
        homo_energy_hartree: homo,
        chemical_potential_hartree: result.chemical_potential.get(),
        core_h0_trace_hartree: result
            .core_one_body_traces
            .iter()
            .map(|t| t.total.get())
            .sum(),
        sector_energies: SectorEnergies {
            vv_hartree: result.sector_exchange.exchange_vv.get(),
            cross_hartree: result.sector_exchange.exchange_cv.get(),
            cc_hartree: result.sector_exchange.exchange_cc.get(),
            total_exchange_hartree: result.sector_exchange.exchange_total.get(),
        },
        cv_vc_trace_mismatch_hartree: result.sector_exchange.cross_trace_mismatch.get(),
        electron_counts: ElectronCounts {
            valence: electron_count(&result.valence_density)?,
            core: electron_count(&result.core_density)?,
            total: electron_count(&result.total_density)?,
        },
        orbital_energies_and_occupations: orbitals,
        core_shells: core_shell_records(&result.core_orbitals),
        core_orthogonalization: spaces,
    };
    write_toml(&cli.out.join("result.toml"), &record)?;
    let final_checkpoint = checkpoint_v2_from_regional_state(
        &start.checkpoint,
        &result.total_density,
        &result.potential,
        BTreeMap::from([
            ("hf.driver".to_owned(), "run_frozen_core_hf".to_owned()),
            (
                "hf.core_treatment".to_owned(),
                "frozen-checkpoint".to_owned(),
            ),
            (
                "hf.status".to_owned(),
                "configured_convergence_reached".to_owned(),
            ),
        ]),
    )?;
    write_checkpoint(&cli.out.join("final-checkpoint.toml"), &final_checkpoint)?;
    println!(
        "status=configured_convergence_reached route=spinor-frozen out={} total_energy_hartree={} homo_energy_hartree={} iterations={}",
        cli.out.display(),
        result.total_energy.get(),
        homo,
        result.diagnostics.len()
    );
    Ok(())
}
