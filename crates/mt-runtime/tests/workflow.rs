mod common;

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::f64::consts::TAU;
use std::path::PathBuf;

use muffintin::{
    ChannelEnergyGenerator, ChannelIdentity, ChannelProvenance, ChannelRecipeArtifact,
    ChannelRecipeRecord, ChannelScope, ChannelTreatment, InputError, Task, TaskResult,
    execute_prepared_with, prepare_input, prepare_input_with_recipes,
};
use muffintin_core::{
    FourierLayout, GVector, Hartree, InterstitialGeometry, InverseBohr, ReciprocalLattice,
    VolumeBohr3,
};
use muffintin_dft::{
    BandPathRequest, BandState, CoreContribution, DosRequest, FirstVariationWindow,
    InterstitialField, NoncollinearXcRoute, RegionalDensity, RegionalPotential,
    RegionalScalarField, RegularSpectrum, ScfBasis, ScfConfig, ScfCoreSite, ScfEnergyContext,
    ScfEnergyTerms, ScfExchangeCorrelation, ScfKMesh, ScfPhysics, ScfRelativity, ScfState,
    XcFunctional,
};
use muffintin_io::CheckpointFile;
use num_complex::Complex64;

use common::{sample_checkpoint, sample_input, supported_checkpoint, supported_input};

struct WorkflowKernel {
    template: RegionalDensity,
    events: Vec<String>,
    cutoffs: Vec<InverseBohr>,
    band_source: Option<ScfState>,
    dos_source: Option<ScfState>,
    saw_sv_window: bool,
    saw_valence_electron_count: bool,
}

impl WorkflowKernel {
    fn new() -> Self {
        Self {
            template: scalar_density(1.0),
            events: Vec::new(),
            cutoffs: Vec::new(),
            band_source: None,
            dos_source: None,
            saw_sv_window: false,
            saw_valence_electron_count: false,
        }
    }

    fn scaled_density(&self, scale: f64) -> RegionalDensity {
        let mut density = self.template.zero_like();
        density.add_scaled(scale, &self.template).unwrap();
        density
    }
}

impl ScfPhysics for WorkflowKernel {
    type Error = Infallible;
    type OneParticle = ();
    type BandSolution = Vec<BandState>;

    fn initial_density(&mut self, config: &ScfConfig) -> Result<RegionalDensity, Self::Error> {
        self.events.push("initial".to_owned());
        self.cutoffs.push(config.basis.plane_wave_cutoff);
        Ok(self.template.clone())
    }

    fn build_potential(
        &mut self,
        iteration: usize,
        density: &RegionalDensity,
        exchange_correlation: ScfExchangeCorrelation,
    ) -> Result<RegionalPotential, Self::Error> {
        assert_eq!(
            exchange_correlation,
            ScfExchangeCorrelation {
                functional: XcFunctional::LdaPw92,
                noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
            }
        );
        self.events.push(format!("potential:{iteration}"));
        Ok(
            RegionalPotential::new(density.charge().clone(), density.magnetization().clone())
                .unwrap(),
        )
    }

    fn solve_core(
        &mut self,
        iteration: usize,
        site: &ScfCoreSite,
        _potential: &RegionalPotential,
        _basis: &ScfBasis,
        _relativity: ScfRelativity,
    ) -> Result<CoreContribution, Self::Error> {
        assert_eq!(site.id, "Si-1");
        assert!(!site.states.is_empty());
        assert!(
            site.states
                .iter()
                .map(|state| state.occupation)
                .sum::<f64>()
                > 0.0
        );
        self.events.push(format!("core:{iteration}"));
        Ok(CoreContribution {
            site_id: site.id.clone(),
            density: self.scaled_density(0.25),
            eigenvalue_sum: Hartree(-0.5),
        })
    }

    fn assemble_one_particle(
        &mut self,
        iteration: usize,
        _potential: &RegionalPotential,
        basis: &ScfBasis,
        _relativity: ScfRelativity,
    ) -> Result<Self::OneParticle, Self::Error> {
        assert!(!basis.channels.is_empty());
        assert!(basis.resolved_channels.is_empty());
        self.events.push(format!("assemble:{iteration}"));
        Ok(())
    }

    fn solve_regular_bands(
        &mut self,
        iteration: usize,
        _one_particle: &Self::OneParticle,
        k_mesh: ScfKMesh,
        relativity: ScfRelativity,
    ) -> Result<Self::BandSolution, Self::Error> {
        assert_eq!(k_mesh.divisions, [4, 4, 4]);
        self.saw_sv_window = relativity
            == (ScfRelativity::SocSecondVariation {
                window: FirstVariationWindow::new(0, 12).unwrap(),
            });
        self.events.push(format!("bands:{iteration}"));
        Ok(vec![
            BandState::new(Hartree(-1.0), 1.0, 8),
            BandState::new(Hartree(1.0), 1.0, 8),
        ])
    }

    fn band_states<'a>(&self, bands: &'a Self::BandSolution) -> &'a [BandState] {
        bands
    }

    fn synthesize_valence_density(
        &mut self,
        iteration: usize,
        _bands: &Self::BandSolution,
        occupations: &[f64],
    ) -> Result<RegionalDensity, Self::Error> {
        assert_eq!(occupations.len(), 2);
        let represented_electrons = 8.0 * occupations.iter().sum::<f64>();
        self.saw_valence_electron_count = (represented_electrons - 4.0).abs() < 1.0e-11;
        self.events.push(format!("density:{iteration}"));
        Ok(self.template.zero_like())
    }

    fn energy_terms(
        &mut self,
        context: ScfEnergyContext<'_, Self::OneParticle, Self::BandSolution>,
    ) -> Result<ScfEnergyTerms, Self::Error> {
        self.events.push(format!("energy:{}", context.iteration));
        Ok(ScfEnergyTerms {
            madelung: Hartree(0.0),
            coulomb: Hartree(0.0),
            exchange_correlation: Hartree(0.0),
            exchange_correlation_potential: Hartree(0.0),
        })
    }

    fn solve_band_path(
        &mut self,
        state: &ScfState,
        request: &BandPathRequest,
    ) -> Result<Vec<Vec<Hartree>>, Self::Error> {
        self.band_source = Some(state.clone());
        self.events.push("path".to_owned());
        Ok(request
            .points
            .iter()
            .map(|_| {
                (0..request.bands)
                    .map(|band| Hartree(band as f64 * 0.1))
                    .collect()
            })
            .collect())
    }

    fn solve_dos_spectrum(
        &mut self,
        state: &ScfState,
        request: &DosRequest,
    ) -> Result<RegularSpectrum, Self::Error> {
        self.dos_source = Some(state.clone());
        self.events.push("dos".to_owned());
        let k_count = request.k_mesh.divisions.into_iter().product();
        Ok(RegularSpectrum::new(
            request.k_mesh.divisions,
            [
                [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
                [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
                [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
            ],
            vec![Hartree(0.0); k_count],
            vec![2],
        )
        .unwrap())
    }
}

fn scalar_density(value: f64) -> RegionalDensity {
    let reciprocal = ReciprocalLattice::new([
        [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
        [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
        [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
    ])
    .unwrap();
    let layout = FourierLayout::new(
        reciprocal,
        vec![GVector {
            index: [0; 3],
            cartesian: [InverseBohr(0.0); 3],
            norm: InverseBohr(0.0),
        }],
    )
    .unwrap();
    let field = |value| {
        InterstitialField::new(
            layout.clone(),
            BTreeMap::from([([0; 3], Complex64::new(value, 0.0))]),
        )
        .unwrap()
    };
    let geometry = InterstitialGeometry::new(VolumeBohr3(TAU.powi(3)), Vec::new()).unwrap();
    let charge = RegionalScalarField::new(geometry, Vec::new(), field(value)).unwrap();
    let zero = charge.zero_like();
    RegionalDensity::new(charge, [zero.clone(), zero.clone(), zero]).unwrap()
}

#[test]
fn workflow_executes_scf_bands_dos_in_order_with_exact_state_reuse() {
    let mut input = sample_input();
    let Task::DftScf { basis, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    basis.energy_generator = Some(ChannelEnergyGenerator::FrozenCheckpoint);
    basis.recipe = None;
    basis.channels.clear();
    let prepared = prepare_input(&input, CheckpointFile::V1(sample_checkpoint())).unwrap();
    let mut kernel = WorkflowKernel::new();
    let result = execute_prepared_with(&prepared, &mut kernel).unwrap();
    assert_eq!(result.tasks.len(), 3);

    let TaskResult::Scf(state) = &result.tasks[0] else {
        panic!("first task must be SCF");
    };
    let TaskResult::Bands(bands) = &result.tasks[1] else {
        panic!("second task must be bands");
    };
    let TaskResult::Dos(dos) = &result.tasks[2] else {
        panic!("third task must be DOS");
    };
    assert_eq!(bands.points.len(), 2);
    assert_eq!(bands.points[0].energies.len(), 12);
    assert_eq!(dos.tetrahedron.edges.len(), 401);
    assert_eq!(kernel.band_source.as_ref(), Some(state.as_ref()));
    assert_eq!(kernel.dos_source.as_ref(), Some(state.as_ref()));
    assert!(kernel.saw_sv_window);
    assert!(kernel.saw_valence_electron_count);

    let path_index = kernel
        .events
        .iter()
        .position(|event| event == "path")
        .unwrap();
    let dos_index = kernel
        .events
        .iter()
        .position(|event| event == "dos")
        .unwrap();
    assert!(
        kernel.events[..path_index]
            .iter()
            .any(|event| event.starts_with("energy:"))
    );
    assert!(
        !kernel.events[..path_index]
            .iter()
            .any(|event| event == "dos")
    );
    assert!(path_index < dos_index);
    assert_eq!(
        kernel
            .events
            .iter()
            .filter(|event| event.starts_with("core:"))
            .count(),
        state.iterations()
    );
}

#[test]
fn reciprocal_and_energy_cutoffs_reach_the_same_scf_config() {
    let mut reciprocal_input = sample_input();
    let Task::DftScf { basis, .. } = reciprocal_input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    basis.energy_generator = Some(ChannelEnergyGenerator::FrozenCheckpoint);
    basis.recipe = None;
    basis.channels.clear();

    let mut energy_input = reciprocal_input.clone();
    let Task::DftScf { basis, .. } = energy_input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    basis.envelope.g_cutoff = None;
    basis.envelope.energy_cutoff = Some(8.0);

    for input in [reciprocal_input, energy_input] {
        let prepared = prepare_input(&input, CheckpointFile::V1(sample_checkpoint())).unwrap();
        let mut kernel = WorkflowKernel::new();
        execute_prepared_with(&prepared, &mut kernel).unwrap();
        assert_eq!(kernel.cutoffs, vec![InverseBohr(4.0)]);
    }
}

#[test]
fn schema_rich_orbital_config_reaches_execution() {
    let recipes = BTreeMap::from([(
        PathBuf::from("recipes/si.toml"),
        ChannelRecipeArtifact::default(),
    )]);
    let prepared = prepare_input_with_recipes(
        &sample_input(),
        CheckpointFile::V1(sample_checkpoint()),
        &recipes,
    )
    .unwrap();
    assert!(prepared.tasks[0].channel_recipe.is_some());
    let mut kernel = WorkflowKernel::new();
    let result = execute_prepared_with(&prepared, &mut kernel).unwrap();
    assert_eq!(result.tasks.len(), 3);
    assert!(!kernel.events.is_empty());
}

#[test]
fn derivative_order_three_prepares_but_fails_at_execution_boundary() {
    let mut input = supported_input();
    let Task::DftScf { basis, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    let recipe_path = PathBuf::from("recipes/h.toml");
    basis.recipe = Some(recipe_path.clone());
    let identity = ChannelIdentity::ScalarL { n: 2, l: 1 };
    let recipes = BTreeMap::from([(
        recipe_path,
        ChannelRecipeArtifact {
            channels: vec![ChannelRecipeRecord {
                scope: ChannelScope::Site {
                    name: "H-1".to_owned(),
                },
                identity,
                treatment: ChannelTreatment::Hdlo,
                derivative_order: 3,
                generator: ChannelEnergyGenerator::Explicit,
                seed: Some(Hartree(-0.1)),
                provenance: ChannelProvenance::BuiltIn,
            }],
        },
    )]);
    let prepared =
        prepare_input_with_recipes(&input, CheckpointFile::V1(supported_checkpoint()), &recipes)
            .unwrap();
    assert_eq!(
        prepared.tasks[0].channel_recipe.as_ref().unwrap().sites[0].channels[0].derivative_order,
        3
    );

    let mut kernel = WorkflowKernel::new();
    assert!(matches!(
        execute_prepared_with(&prepared, &mut kernel),
        Err(InputError::DerivativeOrderNotImplemented {
            task_id,
            site,
            identity: found_identity,
            derivative_order: 3,
        }) if task_id == "scf" && site == "H-1" && found_identity == identity
    ));
    assert!(kernel.events.is_empty());
}
