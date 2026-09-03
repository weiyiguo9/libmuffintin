//! Coupled Hartree and exchange iteration in one frozen-core SRA space.

use super::*;

/// One-loop SRA HF controls. `config.convergence` supplies the iteration
/// limit, energy tolerance, and fresh unmixed density-map tolerance.
/// Only `fock_mixing` is applied: there is no outer regional-density mixer.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenCoreHfSpec {
    pub config: ScfConfig,
    pub product_l_max: u32,
    pub product_g_max: InverseBohr,
    pub overlap_tolerance: f64,
    pub coulomb: CoulombRequest,
    pub gamma: GammaExchangeTreatment,
    pub fock_mixing: FockMixing,
    pub feedback_tolerance: Hartree,
    pub commutator_tolerance: Hartree,
    pub virtual_level_shift: Hartree,
    pub sector_numerical_tolerance: Hartree,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrozenCoreHfIterationDiagnostic {
    pub iteration: usize,
    pub total_energy: Hartree,
    pub energy_change: Option<Hartree>,
    pub density_residual: f64,
    pub commutator_residual: Hartree,
    pub active_feedback_residual: Option<Hartree>,
    pub eigenvalue_identity_residual: Hartree,
    pub valence_electrons: f64,
    pub core_h0_trace: Hartree,
    pub exchange_vv: Hartree,
    pub exchange_cv: Hartree,
    pub exchange_cc: Hartree,
    pub virtual_level_shift: Hartree,
    pub converged: bool,
}

#[derive(Clone, Debug)]
pub struct FrozenCoreHfResult {
    pub valence_density: RegionalDensity,
    pub core_density: RegionalDensity,
    pub total_density: RegionalDensity,
    pub potential: RegionalPotential,
    pub bands: CheckpointBandSolution,
    pub core_orbitals: Vec<CoreShellOrbitals>,
    pub core_one_body_traces: Vec<CoreLocalOneBodyTrace>,
    pub occupations: Vec<Vec<f64>>,
    pub chemical_potential: Hartree,
    pub orbital_energies: Vec<Vec<Hartree>>,
    pub total_energy: Hartree,
    pub sector_exchange: FrozenSpinorSectorExchange,
    pub final_exchange_inputs: Vec<SpinorProductInput>,
    pub k_fractional: Vec<[f64; 3]>,
    pub q_fractional: Vec<[f64; 3]>,
    pub k_weights: Vec<f64>,
    pub diagnostics: Vec<FrozenCoreHfIterationDiagnostic>,
}

/// Solve frozen-core SRA HF with J[D] and K[D] from the same spinor density.
/// Radials, the core-null embedding, auxiliary bases, and Coulomb operators
/// remain fixed. Only the allowed valence orbitals rotate; core is never an
/// active VV orbital. Final acceptance uses the unshifted, unmixed Fock map.
pub fn run_frozen_core_hf(
    physics: &mut CheckpointPhysics,
    spec: &FrozenCoreHfSpec,
    mut on_iteration: impl FnMut(&FrozenCoreHfIterationDiagnostic) -> std::io::Result<()>,
) -> Result<FrozenCoreHfResult, CoreValenceHfError> {
    let _ = ScfLoop::new(spec.config.clone(), None)?;
    let valence_electrons = validate(spec)?;
    let k_fractional = muffintin_dft::regular_k_points(spec.config.k_mesh)?;
    let q_fractional =
        canonical_q_points(&k_fractional).map_err(|_| CoreValenceHfError::QTopology)?;
    let initial = physics.kernel.initial_density_components(&spec.config)?;
    let core = physics
        .kernel
        .frozen_checkpoint_core(&initial.total, &spec.config)?;
    let reference = physics.frozen_potential().clone();
    let (mut bands, _) = solve_h0_bands(
        physics,
        &spec.config,
        &reference,
        &k_fractional,
        valence_electrons,
        initial.total.charge().interstitial().layout(),
        ScfRelativity::SpinorFirstVariation,
        &core.orbitals,
    )?;
    let exchange_spec = CoreExchangeSpec {
        product_l_max: spec.product_l_max,
        product_g_max: spec.product_g_max,
        overlap_tolerance: spec.overlap_tolerance,
        coulomb: &spec.coulomb,
        gamma: spec.gamma,
    };
    let mut cache = None;
    let mut cc_trace = None;
    let mut previous_energy: Option<Hartree> = None;
    let mut previous_feedback: Option<Vec<DenseHermitianMatrix>> = None;
    let mut mixer = FeedbackMixer::new(spec.fock_mixing);
    let mut shift = spec.virtual_level_shift.get();
    let mut diagnostics = Vec::new();
    for iteration in 1..=spec.config.convergence.max_iterations {
        let occupation =
            solve_occupations(bands.states(), valence_electrons, spec.config.occupations)?;
        let rows = occupation_rows(&occupation.values, &bands)?;
        let valence_density = physics
            .kernel
            .synthesize_bands(&bands, &occupation.values)?;
        let total_density = sum_density(&valence_density, &core.density)?;
        let electrostatic = evaluate_regional_electrostatics(
            total_density.charge(),
            &ElectrostaticSpec::new(
                WeinertHartreeSpec::electronic(4)?,
                physics.nuclear_charges().to_vec(),
            )?,
        )?;
        let zero = electrostatic.potential.zero_like();
        let potential = RegionalPotential::new(
            electrostatic.potential.clone(),
            [zero.clone(), zero.clone(), zero],
        )?;
        let local = physics
            .kernel
            .spinor_local_potential_feedback(&bands, &potential)?;
        let rebuilt = rebuild_core_feedback_frame(
            physics,
            &exchange_spec,
            &bands,
            &occupation,
            &k_fractional,
            &q_fractional,
            &core.orbitals,
            &mut cache,
        )?;
        if cc_trace.is_none() {
            let cache = cache
                .as_ref()
                .expect("fixed-basis exchange was initialized");
            let sectors = cache
                .core_bases
                .iter()
                .map(|basis| &basis.cc)
                .collect::<Vec<_>>();
            cc_trace = Some(
                crate::spinor_sector_exchange::cached_core_core_exchange_trace(
                    &rebuilt.inputs,
                    &sectors,
                    &cache.core_operators,
                    &spec.coulomb,
                    &rebuilt.occupations,
                )?,
            );
        }
        let cc = cc_trace.expect("initial frozen CC contraction is retained");
        let band_feedback = relaxed_valence_feedback(&rebuilt.exchange)?;
        let exchange = lift_global_feedback(&bands, &band_feedback)?;
        let fresh = subtract_global_feedback(&exchange, &scale_feedback(&local, -1.0)?)?;
        let commutator = maximum_complex_element(&commutator_diis_error(
            &bands,
            &rows,
            &fresh,
            None,
            FeedbackChannel::Spinor,
        )?);
        let feedback_residual = previous_feedback
            .as_ref()
            .map(|old| active_feedback_difference(&bands, &rows, old, &fresh))
            .transpose()?;
        let solved = bands.solve_spinor_global_feedback(&fresh)?;
        let solved_occupation =
            solve_occupations(solved.states(), valence_electrons, spec.config.occupations)?;
        let solved_density = physics
            .kernel
            .synthesize_bands(&solved, &solved_occupation.values)?;
        let density_residual = valence_density.difference_rms(&solved_density)?;
        let core_one_body_traces = current_core_one_body(physics, &core.orbitals, &electrostatic)?;
        let core_h0 = Hartree(
            core_one_body_traces
                .iter()
                .map(|trace| trace.total.get())
                .sum(),
        );
        let valence_h0 = local_one_body_trace(&bands, &rows, &local)?;
        let vv = rebuilt.exchange.vv.trace.get();
        let cv = rebuilt.exchange.cv.trace.get();
        let energy = Hartree(
            valence_h0 + core_h0.get() - electrostatic.electron_hartree.get()
                + electrostatic.nuclear_nuclear.get()
                + 0.5 * vv
                + cv
                + 0.5 * cc.get()
                + occupation.correction.get(),
        );
        let energy_change = previous_energy.map(|e| Hartree((energy.get() - e.get()).abs()));
        let electron_count = electron_count(&valence_density)?;
        require_relaxed_gate(
            "frozen-core valence electron count",
            (electron_count - valence_electrons).abs(),
            ELECTRON_COUNT_TOLERANCE,
        )?;
        let gates = energy_change.is_some_and(|e| e <= spec.config.convergence.energy_tolerance)
            && density_residual <= spec.config.convergence.density_tolerance
            && commutator <= spec.commutator_tolerance.get()
            && feedback_residual.is_some_and(|r| r <= spec.feedback_tolerance.get());
        let diagnostic = FrozenCoreHfIterationDiagnostic {
            iteration,
            total_energy: energy,
            energy_change,
            density_residual,
            commutator_residual: Hartree(commutator),
            active_feedback_residual: feedback_residual.map(Hartree),
            eigenvalue_identity_residual: Hartree(
                (occupation.band_energy.get() - valence_h0 - vv - cv).abs(),
            ),
            valence_electrons: electron_count,
            core_h0_trace: core_h0,
            exchange_vv: Hartree(0.5 * vv),
            exchange_cv: Hartree(cv),
            exchange_cc: Hartree(0.5 * cc.get()),
            virtual_level_shift: Hartree(shift),
            converged: gates && shift == 0.0,
        };
        on_iteration(&diagnostic)?;
        diagnostics.push(diagnostic);
        if gates && shift == 0.0 {
            let final_frame = complete_core_sector_frame(&exchange_spec, rebuilt)?;
            require_relaxed_gate(
                "frozen-core CV/VC trace identity",
                final_frame.exchange.cross_trace_mismatch.get().abs(),
                spec.sector_numerical_tolerance.get(),
            )?;
            let final_energy = Hartree(
                valence_h0 + core_h0.get() - electrostatic.electron_hartree.get()
                    + electrostatic.nuclear_nuclear.get()
                    + final_frame.exchange.exchange_total.get()
                    + occupation.correction.get(),
            );
            return Ok(FrozenCoreHfResult {
                valence_density,
                core_density: core.density,
                total_density,
                potential,
                orbital_energies: spinor_energies(&bands)?,
                k_weights: k_weights(&bands)?,
                bands,
                core_orbitals: core.orbitals,
                core_one_body_traces,
                occupations: rows,
                chemical_potential: occupation.chemical_potential,
                total_energy: final_energy,
                sector_exchange: final_frame.exchange,
                final_exchange_inputs: final_frame.inputs,
                k_fractional,
                q_fractional,
                diagnostics,
            });
        }
        previous_energy = Some(energy);
        if gates {
            shift = 0.0;
            bands = solved;
            previous_feedback = Some(fresh);
            continue;
        }
        let mixed = match &previous_feedback {
            Some(previous) => mixer.mix(&bands, &rows, previous, &fresh)?,
            None => fresh.clone(),
        };
        let mut update = mixed.clone();
        if shift > 0.0 {
            let virtual_band = rows
                .iter()
                .map(|f| {
                    DenseHermitianMatrix::from_upper_triangle(f.len(), Axis::Band, |i, j| {
                        if i == j {
                            Complex64::new(shift * (1.0 - f[i]), 0.0)
                        } else {
                            Complex64::default()
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let virtual_global = lift_global_feedback(&bands, &virtual_band)?;
            update = subtract_global_feedback(&update, &scale_feedback(&virtual_global, -1.0)?)?;
        }
        bands = bands.solve_spinor_global_feedback(&update)?;
        previous_feedback = Some(mixed);
    }
    let last = diagnostics
        .last()
        .expect("validated positive iteration count");
    Err(CoreValenceHfError::FrozenNotConverged {
        iterations: diagnostics.len(),
        energy_change: last.energy_change.map_or(f64::INFINITY, Hartree::get),
        density_residual: last.density_residual,
        commutator_residual: last.commutator_residual.get(),
        feedback_residual: last
            .active_feedback_residual
            .map_or(f64::INFINITY, Hartree::get),
    })
}

fn active_feedback_difference(
    bands: &CheckpointBandSolution,
    occupations: &[Vec<f64>],
    old: &[DenseHermitianMatrix],
    fresh: &[DenseHermitianMatrix],
) -> Result<f64, GammaValenceHfError> {
    require_feedback_layout(old, fresh)?;
    let mut maximum = 0.0_f64;
    for (k, point) in bands.points().iter().enumerate() {
        let CheckpointKPointSolution::Spinor { solution, .. } = &point.solution else {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        };
        let delta = DenseHermitianMatrix::from_upper_triangle(
            fresh[k].dimension(),
            Axis::GlobalBasis,
            |i, j| fresh[k].at(i, j) - old[k].at(i, j),
        )?;
        let projected = hermitian_congruence(solution.eigenvectors.as_tensor(), &delta)?;
        for i in 0..occupations[k].len() {
            for j in 0..occupations[k].len() {
                maximum = maximum
                    .max(occupations[k][i].max(occupations[k][j]) * projected.at(i, j).norm());
            }
        }
    }
    Ok(maximum)
}

fn local_one_body_trace(
    bands: &CheckpointBandSolution,
    occupations: &[Vec<f64>],
    local: &[DenseHermitianMatrix],
) -> Result<f64, GammaValenceHfError> {
    let mut value = 0.0;
    for (k, point) in bands.points().iter().enumerate() {
        let CheckpointKPointSolution::Spinor {
            solution,
            eigenproblem,
            ..
        } = &point.solution
        else {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        };
        let h0 = DenseHermitianMatrix::from_upper_triangle(
            local[k].dimension(),
            Axis::GlobalBasis,
            |i, j| eigenproblem.hamiltonian.at(i, j) + local[k].at(i, j),
        )?;
        let projected = hermitian_congruence(solution.eigenvectors.as_tensor(), &h0)?;
        value += occupations[k]
            .iter()
            .enumerate()
            .map(|(i, f)| point.weight() * f * projected.at(i, i).re)
            .sum::<f64>();
    }
    Ok(value)
}

fn current_core_one_body(
    physics: &CheckpointPhysics,
    cores: &[CoreShellOrbitals],
    electrostatic: &muffintin_dft::RegionalElectrostaticResult,
) -> Result<Vec<CoreLocalOneBodyTrace>, CoreValenceHfError> {
    let mut meshes = physics
        .kernel
        .sites()
        .iter()
        .map(|site| site.up().mesh().clone())
        .collect::<Vec<_>>();
    for core in cores {
        meshes[core.site_index] = core.extended_mesh.clone();
    }
    let current = muffintin_dft::build_extended_electrostatic_core_potentials(
        electrostatic,
        physics.geometry(),
        &meshes,
        muffintin_sphere::CorePotentialContinuationSpec::default(),
    )
    .map_err(MaterialKernelError::from)?;
    cores
        .iter()
        .map(|core| {
            core_local_one_body_trace(core, &current[core.site_index].potential.values)
                .map_err(Into::into)
        })
        .collect()
}

fn validate(spec: &FrozenCoreHfSpec) -> Result<f64, CoreValenceHfError> {
    if spec.config.k_mesh.reduction != ScfKReduction::Full {
        return Err(CoreValenceHfError::SymmetryReduction);
    }
    if spec.config.relativity != ScfRelativity::SpinorFirstVariation {
        return Err(CoreValenceHfError::SpinorFirstVariation);
    }
    let core: f64 = spec
        .config
        .core_sites
        .iter()
        .flat_map(|s| &s.states)
        .map(|s| s.occupation)
        .sum();
    if !core.is_finite() || core <= 0.0 {
        return Err(CoreValenceHfError::CoreStates);
    }
    let valence = spec.config.electron_count - core;
    if !valence.is_finite() || valence <= 0.0 {
        return Err(CoreValenceHfError::ValenceElectronCount);
    }
    if spec.config.convergence.max_iterations < 2 {
        return Err(CoreValenceHfError::FockIterations);
    }
    if !valid_fock_mixing(spec.fock_mixing) {
        return Err(CoreValenceHfError::FockMixing);
    }
    if !spec.feedback_tolerance.get().is_finite() || spec.feedback_tolerance.get() <= 0.0 {
        return Err(CoreValenceHfError::FockFeedbackTolerance);
    }
    if !spec.commutator_tolerance.get().is_finite() || spec.commutator_tolerance.get() <= 0.0 {
        return Err(CoreValenceHfError::CommutatorTolerance);
    }
    if !spec.virtual_level_shift.get().is_finite() || spec.virtual_level_shift.get() < 0.0 {
        return Err(CoreValenceHfError::VirtualLevelShift);
    }
    if !spec.sector_numerical_tolerance.get().is_finite()
        || spec.sector_numerical_tolerance.get() < 0.0
    {
        return Err(CoreValenceHfError::SectorTolerance);
    }
    Ok(valence)
}
