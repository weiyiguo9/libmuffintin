use super::*;
use muffintin_dft::NoncollinearXcRoute;
use muffintin_dft::{
    BandPathPoint, FirstVariationWindow, ScfConvergence, ScfCoreSite, ScfCoreState, ScfMixing,
    XcFunctional, run_scf,
};
use muffintin_io::{
    BasisHints, Complex64V1, EnergyParameterV1, EnergyUnit, ExponentialMeshSpec,
    FourierCoefficientV1, FourierNormalization, FourierPhase, GeometryV1, InterstitialV1,
    InverseLengthUnit, LatticeV1, LengthUnit, LinearizationV1, CheckpointMeta, PotentialChannelV1,
    PotentialConventionV1, PotentialRadialQuantityV1, SiteSpinV1, SiteV1, CheckpointFile, CheckpointV1,
    SphericalChannelConvention, SpinTag, checkpoint_file_from_toml, checkpoint_file_to_toml,
};

fn checkpoint_v1() -> muffintin_io::CheckpointV1 {
    let point_count = 61;
    let first: f64 = 1.0e-4;
    let radius: f64 = 1.0;
    let increment = (radius / first).ln() / (point_count - 1) as f64;
    let radii = (0..point_count)
        .map(|index| first * (index as f64 * increment).exp())
        .collect::<Vec<_>>();
    CheckpointV1::new(
        CheckpointMeta {
            title: "checkpoint kernel hydrogen smoke".to_owned(),
            producer: "mt-runtime test".to_owned(),
            producer_version: None,
            energy_zero: "zero interstitial Fourier mean".to_owned(),
            potential_convention: PotentialConventionV1 {
                angular_basis: AngularBasis::ComplexCondonShortley,
                radial_quantity: PotentialRadialQuantityV1::Potential,
                spherical_channel: SphericalChannelConvention::PhysicalValue,
            },
            annotations: BTreeMap::new(),
        },
        GeometryV1 {
            lattice: LatticeV1 {
                unit: LengthUnit::Bohr,
                vectors: [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            },
            sites: vec![SiteV1 {
                id: "H-1".to_owned(),
                atomic_number: 1,
                fractional_position: [1.25, -0.5, 0.5],
                muffin_tin_radius_unit: LengthUnit::Bohr,
                muffin_tin_radius: radius,
                spins: vec![SiteSpinV1 {
                    spin: SpinTag::Scalar,
                    mesh: ExponentialMeshSpec {
                        radius_unit: LengthUnit::Bohr,
                        first,
                        log_increment: increment,
                        point_count,
                        last: first * ((point_count - 1) as f64 * increment).exp(),
                        consistency_tolerance: 1.0e-12,
                    },
                    radial_equation: RadialEquationTag::ScalarKoellingHarmon,
                    potential_unit: EnergyUnit::Hartree,
                    potential_channels: vec![PotentialChannelV1 {
                        l: 0,
                        m: 0,
                        real: radii.iter().map(|radius| -1.0 / radius).collect(),
                        imaginary: Vec::new(),
                    }],
                    linearization: LinearizationV1 {
                        energy_unit: EnergyUnit::Hartree,
                        linearization_energies: vec![
                            EnergyParameterV1 { l: 0, energy: -0.3 },
                            EnergyParameterV1 {
                                l: 1,
                                energy: -0.15,
                            },
                        ],
                        local_orbital_energies: Vec::new(),
                    },
                }],
            }],
        },
        InterstitialV1 {
            coefficient_unit: EnergyUnit::Hartree,
            coefficients: vec![FourierCoefficientV1 {
                g: [0; 3],
                value: Complex64V1 {
                    real: 0.0,
                    imaginary: 0.0,
                },
            }],
            basis_hints: BasisHints {
                reciprocal_length_unit: InverseLengthUnit::BohrInverse,
                plane_wave_cutoff: Some(0.5),
                coefficient_cutoff: Some(1.0),
                normalization: FourierNormalization::CellNormalized,
                phase: FourierPhase::NegativeExponent,
            },
        },
    )
}

fn checkpoint() -> CheckpointV2 {
    checkpoint_v1().normalize_v2().unwrap()
}

fn config(relativity: ScfRelativity) -> ScfConfig {
    ScfConfig {
        electron_count: 1.0,
        k_mesh: ScfKMesh {
            divisions: [1, 1, 1],
            shift: [0.0; 3],
        },
        basis: ScfBasis {
            plane_wave_cutoff: 0.5,
            l_max: 1,
            channels: vec![
                ScfChannelRecipe {
                    site: "H-1".to_owned(),
                    identity: ScfChannelIdentity::ScalarL { n: 1, l: 0 },
                    treatment: ScfChannelTreatment::Valence,
                    derivative_order: 0,
                    generator: LinearizationEnergyGenerator::FrozenCheckpoint,
                    seed: None,
                    provenance: muffintin_dft::ScfChannelProvenance::BuiltIn,
                },
                ScfChannelRecipe {
                    site: "H-1".to_owned(),
                    identity: ScfChannelIdentity::ScalarL { n: 2, l: 1 },
                    treatment: ScfChannelTreatment::Valence,
                    derivative_order: 0,
                    generator: LinearizationEnergyGenerator::FrozenCheckpoint,
                    seed: None,
                    provenance: muffintin_dft::ScfChannelProvenance::BuiltIn,
                },
            ],
            resolved_channels: Vec::new(),
        },
        occupations: ScfOccupations::FermiDirac {
            temperature: Hartree(0.02),
        },
        exchange_correlation: ScfExchangeCorrelation {
            functional: XcFunctional::LdaPw92,
            noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
        },
        mixing: ScfMixing::Linear { alpha: 1.0 },
        relativity,
        convergence: ScfConvergence {
            energy_tolerance: Hartree(1.0e100),
            density_tolerance: 1.0e100,
            max_iterations: 2,
        },
        core_sites: vec![ScfCoreSite {
            id: "H-1".to_owned(),
            states: Vec::new(),
        }],
    }
}

fn core_checkpoint_and_config() -> (CheckpointV2, ScfConfig) {
    let mut checkpoint = checkpoint_v1();
    let first: f64 = 1.0e-5;
    let radius: f64 = 3.0;
    let point_count = 121;
    let increment = (radius / first).ln() / (point_count - 1) as f64;
    checkpoint.geometry.lattice.vectors = [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]];
    checkpoint.geometry.sites[0].atomic_number = 2;
    checkpoint.geometry.sites[0].muffin_tin_radius = radius;
    let spin = &mut checkpoint.geometry.sites[0].spins[0];
    spin.mesh = ExponentialMeshSpec {
        radius_unit: LengthUnit::Bohr,
        first,
        log_increment: increment,
        point_count,
        last: radius,
        consistency_tolerance: 1.0e-12,
    };
    let mesh = ExponentialMesh::new(Bohr(first), increment, point_count).unwrap();
    spin.potential_channels[0].real = mesh
        .radii()
        .iter()
        .map(|radius| -2.0 / radius.get())
        .collect();
    spin.linearization.linearization_energies[0].energy = -0.8;
    spin.linearization.linearization_energies[1].energy = -0.3;
    let mut config = config(ScfRelativity::Scalar);
    config.electron_count = 2.0;
    config.core_sites[0].states.push(ScfCoreState {
        principal_quantum_number: 1,
        kappa: -1,
        occupation: 1.0,
    });
    (checkpoint.normalize_v2().unwrap(), config)
}

#[test]
fn checkpoint_conversion_normalizes_monopole_and_wraps_cartesian_site() {
    let physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    assert_eq!(
        physics.geometry.spheres()[0].center,
        [Bohr(2.0), Bohr(4.0), Bohr(4.0)]
    );
    let physical = checkpoint_v1().geometry.sites[0].spins[0].potential_channels[0].real[17];
    let normalized = physics.frozen_potential.scalar().muffin_tins()[0]
        .field()
        .channel(0, 0)
        .unwrap()[17]
        .re;
    assert!((normalized - (4.0 * PI).sqrt() * physical).abs() < 1.0e-12);
}

#[test]
fn v2_interstitial_components_are_keyed_independently_of_input_order() {
    fn coefficient(g: [i32; 3], real: f64, imaginary: f64) -> FourierCoefficientV2 {
        FourierCoefficientV2 {
            g,
            value: Complex64V2 { real, imaginary },
        }
    }

    let mut checkpoint = checkpoint();
    let InitialV2::FrozenPotential { potential } = &mut checkpoint.initial else {
        unreachable!()
    };
    potential.v0.interstitial.coefficients = vec![
        coefficient([0, 0, 0], 0.0, 0.0),
        coefficient([1, 0, 0], 1.0, 2.0),
        coefficient([-1, 0, 0], 1.0, -2.0),
    ];
    potential.bx.interstitial.coefficients = vec![
        coefficient([1, 0, 0], 3.0, 4.0),
        coefficient([0, 0, 0], 0.5, 0.0),
        coefficient([-1, 0, 0], 3.0, -4.0),
    ];
    potential.by.interstitial.coefficients = vec![
        coefficient([-1, 0, 0], 5.0, -6.0),
        coefficient([1, 0, 0], 5.0, 6.0),
        coefficient([0, 0, 0], 0.25, 0.0),
    ];
    potential.bz.interstitial.coefficients = vec![
        coefficient([0, 0, 0], -0.5, 0.0),
        coefficient([-1, 0, 0], 7.0, -8.0),
        coefficient([1, 0, 0], 7.0, 8.0),
    ];

    let physics = CheckpointPhysics::new(&checkpoint).unwrap();
    let potential = physics.frozen_potential();
    for field in potential.magnetic() {
        assert_eq!(
            field.interstitial().layout(),
            potential.scalar().interstitial().layout()
        );
    }
    assert_eq!(
        potential.magnetic()[0]
            .interstitial()
            .field()
            .coefficient([1, 0, 0]),
        Some(Complex64::new(3.0, 4.0))
    );
    assert_eq!(
        potential.magnetic()[1]
            .interstitial()
            .field()
            .coefficient([-1, 0, 0]),
        Some(Complex64::new(5.0, -6.0))
    );
}

#[test]
fn frozen_checkpoint_produces_initial_density_without_fake_atomic_g_zero() {
    let mut physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    let config = config(ScfRelativity::Scalar);
    let meshes = physics.channel_meshes(&config.basis).unwrap();
    let extended = build_extended_checkpoint_core_potentials(
        &physics.frozen_potential,
        &physics.geometry,
        &physics.nuclear_charges,
        &meshes,
        CorePotentialContinuationSpec::default(),
    )
    .unwrap();
    let materialized = physics
        .materialize_nonspectral_basis(&physics.frozen_potential, &config.basis, &extended)
        .unwrap();
    assert_eq!(
        materialized.resolved_channels[0].energy.get().to_bits(),
        (-0.3_f64).to_bits()
    );
    assert_eq!(
        materialized.resolved_channels[1].energy.get().to_bits(),
        (-0.15_f64).to_bits()
    );
    let density = physics.initial_density(&config).unwrap();
    assert!((muffintin_dft::electron_count(&density).unwrap() - 1.0).abs() < 1.0e-10);
    assert!(
        density
            .charge()
            .interstitial()
            .layout()
            .index([0; 3])
            .is_some()
    );
}

#[test]
fn magnetic_frozen_checkpoint_does_not_turn_spin_splitting_into_kappa_splitting() {
    let mut checkpoint = checkpoint_v1();
    let mut up = checkpoint.geometry.sites[0].spins[0].clone();
    up.spin = SpinTag::Up;
    up.linearization.linearization_energies[1].energy = -0.14;
    let mut down = up.clone();
    down.spin = SpinTag::Down;
    down.linearization.linearization_energies[1].energy = -0.16;
    checkpoint.geometry.sites[0].spins = vec![up, down];
    let physics = CheckpointPhysics::new(&checkpoint.normalize_v2().unwrap()).unwrap();
    let basis = config(ScfRelativity::SpinorFirstVariation).basis;
    let meshes = physics.channel_meshes(&basis).unwrap();
    let extended = build_extended_checkpoint_core_potentials(
        &physics.frozen_potential,
        &physics.geometry,
        &physics.nuclear_charges,
        &meshes,
        CorePotentialContinuationSpec::default(),
    )
    .unwrap();
    let materialized = physics
        .materialize_nonspectral_basis(&physics.frozen_potential, &basis, &extended)
        .unwrap();
    assert_eq!(
        physics
            .scalar_linearization_energies(&materialized, "H-1", 0)
            .unwrap()[1]
            .get()
            .to_bits(),
        (-0.14_f64).to_bits()
    );
    assert_eq!(
        physics
            .scalar_linearization_energies(&materialized, "H-1", 1)
            .unwrap()[1]
            .get()
            .to_bits(),
        (-0.16_f64).to_bits()
    );
    let spinor = physics
        .spinor_linearization_energies(&materialized, "H-1")
        .unwrap();
    let p = spinor
        .iter()
        .filter(|parameter| parameter.kappa.large_l() == 1)
        .map(|parameter| parameter.energy)
        .collect::<Vec<_>>();
    assert_eq!(p.len(), 2);
    assert_eq!(p[0].get().to_bits(), p[1].get().to_bits());
    assert_eq!(p[0], Hartree(0.5 * (-0.14 - 0.16)));
}

#[test]
fn atomic_recipe_materializes_from_the_current_extended_potential() {
    let physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    let mut config = config(ScfRelativity::Scalar);
    for channel in &mut config.basis.channels {
        if channel_l(channel.identity) == 0 {
            channel.generator = LinearizationEnergyGenerator::Atomic;
        }
    }
    let meshes = physics.channel_meshes(&config.basis).unwrap();
    let extended = build_extended_checkpoint_core_potentials(
        &physics.frozen_potential,
        &physics.geometry,
        &physics.nuclear_charges,
        &meshes,
        CorePotentialContinuationSpec::default(),
    )
    .unwrap();
    let first = physics
        .materialize_nonspectral_basis(&physics.frozen_potential, &config.basis, &extended)
        .unwrap();
    let second = physics
        .materialize_nonspectral_basis(&physics.frozen_potential, &config.basis, &extended)
        .unwrap();
    let atomic = first
        .resolved_channels
        .iter()
        .find(|resolved| resolved.recipe.generator == LinearizationEnergyGenerator::Atomic)
        .unwrap();
    assert_eq!(atomic.components.len(), 1);
    assert_eq!(atomic.energy, second.resolved_channels[0].energy);
}

#[test]
fn seeded_radial_search_does_not_require_a_checkpoint_lo_anchor() {
    let physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    for generator in [
        LinearizationEnergyGenerator::BandCenter,
        LinearizationEnergyGenerator::LogDerivative,
    ] {
        let mut basis = config(ScfRelativity::Scalar).basis;
        basis.channels.push(ScfChannelRecipe {
            site: "H-1".to_owned(),
            identity: ScfChannelIdentity::ScalarL { n: 2, l: 0 },
            treatment: ScfChannelTreatment::Lo,
            derivative_order: 0,
            generator,
            seed: Some(Hartree(-0.2)),
            provenance: muffintin_dft::ScfChannelProvenance::Site,
        });
        let meshes = physics.channel_meshes(&basis).unwrap();
        let extended = build_extended_checkpoint_core_potentials(
            &physics.frozen_potential,
            &physics.geometry,
            &physics.nuclear_charges,
            &meshes,
            CorePotentialContinuationSpec::default(),
        )
        .unwrap();
        match physics.materialize_nonspectral_basis(&physics.frozen_potential, &basis, &extended) {
            Ok(materialized) => assert!(
                materialized
                    .resolved_channels
                    .iter()
                    .any(|resolved| resolved.recipe.generator == generator)
            ),
            Err(CheckpointPhysicsError::ChannelGenerator {
                generator: actual, ..
            }) => assert_eq!(actual, generator),
            Err(error) => panic!("seeded {generator:?} failed before its generator: {error}"),
        }
    }
}

#[test]
fn scalar_single_site_checkpoint_runs_two_iteration_scf_smoke() {
    let mut physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    let state = run_scf(&mut physics, &config(ScfRelativity::Scalar), None).unwrap();
    assert_eq!(state.iterations(), 2);
    assert_eq!(state.relativity, ScfRelativity::Scalar);
}

#[test]
fn fermi_offset_refines_inside_each_scf_iteration() {
    let mut physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    let mut config = config(ScfRelativity::Scalar);
    let channel = &mut config.basis.channels[0];
    channel.generator = LinearizationEnergyGenerator::FermiOffset;
    channel.seed = Some(Hartree(-0.1));
    let state = run_scf(&mut physics, &config, None).unwrap();
    assert!(state.diagnostics.iter().all(|diagnostic| {
        diagnostic.resolved_channels.iter().any(|resolved| {
            resolved.recipe.generator == LinearizationEnergyGenerator::FermiOffset
                && matches!(
                    resolved.components[0].diagnostic,
                    LinearizationEnergyDiagnostic::FermiOffset { .. }
                )
        })
    }));
}

#[test]
fn band_cog_uses_physical_projection_inside_the_scf_iteration() {
    let mut physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    let mut config = config(ScfRelativity::Scalar);
    config.basis.channels[0].generator = LinearizationEnergyGenerator::BandCog;
    let state = run_scf(&mut physics, &config, None).unwrap();
    assert!(state.diagnostics.iter().all(|diagnostic| {
        diagnostic.resolved_channels.iter().any(|resolved| {
            resolved.recipe.generator == LinearizationEnergyGenerator::BandCog
                && matches!(
                    resolved.components[0].diagnostic,
                    LinearizationEnergyDiagnostic::BandCog { .. }
                )
        })
    }));
}

#[test]
fn spinor_band_cog_rejects_distinct_channels_with_the_same_kappa_projection() {
    let physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    let mut first = config(ScfRelativity::SpinorFirstVariation).basis.channels[0].clone();
    first.identity = ScfChannelIdentity::Kappa { n: 2, kappa: -1 };
    first.treatment = ScfChannelTreatment::Lo;
    first.generator = LinearizationEnergyGenerator::BandCog;
    first.seed = Some(Hartree(-0.2));
    let mut second = first.clone();
    second.identity = ScfChannelIdentity::Kappa { n: 3, kappa: -1 };

    assert!(matches!(
        physics.validate_band_cog_projection_keys(
            &[&first, &second],
            ScfRelativity::SpinorFirstVariation,
        ),
        Err(CheckpointPhysicsError::AmbiguousBandCogProjection { .. })
    ));
}

#[test]
fn second_variation_is_routed_and_full_spinor_never_falls_back_to_scalar() {
    let mut physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    let sv = config(ScfRelativity::SocSecondVariation {
        window: FirstVariationWindow::new(0, 1).unwrap(),
    });
    assert!(physics.initial_density(&sv).is_ok());

    let mut physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    assert!(matches!(
        physics.initial_density(&config(ScfRelativity::SpinorFirstVariation)),
        Err(CheckpointPhysicsError::SpinorRadialEquation { .. })
    ));
}

#[test]
fn fully_relativistic_checkpoint_uses_full_spinor_solve_and_density() {
    let mut checkpoint = checkpoint_v1();
    checkpoint.geometry.sites[0].spins[0].radial_equation =
        RadialEquationTag::FullyRelativisticDirac;
    let mut physics = CheckpointPhysics::new(&checkpoint.normalize_v2().unwrap()).unwrap();
    let density = physics
        .initial_density(&config(ScfRelativity::SpinorFirstVariation))
        .unwrap();
    assert!((muffintin_dft::electron_count(&density).unwrap() - 1.0).abs() < 1.0e-9);
}

#[test]
fn full_spinor_scf_retains_transverse_magnetization_for_two_iterations() {
    let mut checkpoint = checkpoint_v1();
    checkpoint.geometry.sites[0].spins[0].radial_equation =
        RadialEquationTag::FullyRelativisticDirac;
    let config = config(ScfRelativity::SpinorFirstVariation);
    let mut physics = CheckpointPhysics::new(&checkpoint.normalize_v2().unwrap()).unwrap();
    let mut source = run_scf(&mut physics, &config, None).unwrap();
    let charge = source.density.charge().clone();
    let mut transverse = charge.zero_like();
    transverse.add_scaled(0.1, &charge).unwrap();
    let zero = charge.zero_like();
    source.density = RegionalDensity::new(charge, [transverse, zero.clone(), zero]).unwrap();

    let state = run_scf(&mut physics, &config, Some(&source)).unwrap();
    assert_eq!(state.iterations(), 2);
    assert!(state.density.magnetization()[0].residual_rms().unwrap() > 1.0e-8);

    let restart = physics.restart_checkpoint(&state).unwrap();
    let encoded = checkpoint_file_to_toml(&CheckpointFile::V2(restart)).unwrap();
    let CheckpointFile::V2(reloaded) = checkpoint_file_from_toml(&encoded).unwrap() else {
        unreachable!()
    };
    let mut restarted_physics = CheckpointPhysics::new(&reloaded).unwrap();
    assert!(
        restarted_physics
            .frozen_potential()
            .scalar()
            .difference_rms(state.potential.scalar())
            .unwrap()
            < 1.0e-10
    );
    for (restarted, expected) in restarted_physics
        .frozen_potential()
        .magnetic()
        .iter()
        .zip(state.potential.magnetic())
    {
        assert!(restarted.difference_rms(expected).unwrap() < 1.0e-10);
    }
    let restarted_density = restarted_physics.initial_density(&config).unwrap();
    assert!(state.density.difference_rms(&restarted_density).unwrap() < 1.0e-12);
}

#[test]
fn scalar_route_rejects_a_transverse_potential() {
    let physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    let scalar = physics.frozen_potential.scalar().clone();
    let mut transverse = scalar.zero_like();
    transverse.add_scaled(0.01, &scalar).unwrap();
    let zero = scalar.zero_like();
    let potential = RegionalPotential::new(scalar, [transverse, zero.clone(), zero]).unwrap();
    assert!(matches!(
        physics.solve_points(
            &potential,
            &config(ScfRelativity::Scalar).basis,
            &[[0.0; 3]],
            ScfRelativity::Scalar,
        ),
        Err(CheckpointPhysicsError::TransversePotentialUnsupported { .. })
    ));
}

#[test]
fn signed_kappa_recipe_keeps_multiple_spinor_local_orbitals() {
    let mut checkpoint = checkpoint_v1();
    let spin = &mut checkpoint.geometry.sites[0].spins[0];
    spin.radial_equation = RadialEquationTag::FullyRelativisticDirac;
    let physics = CheckpointPhysics::new(&checkpoint.normalize_v2().unwrap()).unwrap();
    let mut basis = config(ScfRelativity::SpinorFirstVariation).basis;
    for (n, energy) in [(2, -0.1), (3, -0.05)] {
        basis.channels.push(ScfChannelRecipe {
            site: "H-1".to_owned(),
            identity: ScfChannelIdentity::Kappa { n, kappa: 1 },
            treatment: ScfChannelTreatment::Lo,
            derivative_order: 0,
            generator: LinearizationEnergyGenerator::Explicit,
            seed: Some(Hartree(energy)),
            provenance: muffintin_dft::ScfChannelProvenance::Site,
        });
    }
    let meshes = physics.channel_meshes(&basis).unwrap();
    let extended = build_extended_checkpoint_core_potentials(
        &physics.frozen_potential,
        &physics.geometry,
        &physics.nuclear_charges,
        &meshes,
        CorePotentialContinuationSpec::default(),
    )
    .unwrap();
    let basis = physics
        .materialize_nonspectral_basis(&physics.frozen_potential, &basis, &extended)
        .unwrap();
    let inputs = physics
        .spinor_site_inputs(&physics.frozen_potential, &basis)
        .unwrap();
    assert_eq!(
        inputs[0].local_orbitals,
        vec![
            SpinorLocalOrbitalRequest::Lo {
                kappa: Kappa::new(1).unwrap(),
                energy: Hartree(-0.1),
            },
            SpinorLocalOrbitalRequest::Lo {
                kappa: Kappa::new(1).unwrap(),
                energy: Hartree(-0.05),
            },
        ]
    );
}

#[test]
fn scalar_route_omits_signed_kappa_local_orbitals() {
    let physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    let mut basis = config(ScfRelativity::Scalar).basis;
    basis.channels.push(ScfChannelRecipe {
        site: "H-1".to_owned(),
        identity: ScfChannelIdentity::Kappa { n: 2, kappa: 1 },
        treatment: ScfChannelTreatment::Lo,
        derivative_order: 0,
        generator: LinearizationEnergyGenerator::Explicit,
        seed: Some(Hartree(-0.1)),
        provenance: muffintin_dft::ScfChannelProvenance::Site,
    });
    let meshes = physics.channel_meshes(&basis).unwrap();
    let extended = build_extended_checkpoint_core_potentials(
        &physics.frozen_potential,
        &physics.geometry,
        &physics.nuclear_charges,
        &meshes,
        CorePotentialContinuationSpec::default(),
    )
    .unwrap();
    let basis = physics
        .materialize_nonspectral_basis(&physics.frozen_potential, &basis, &extended)
        .unwrap();
    let inputs = physics
        .scalar_site_inputs(&physics.frozen_potential, &basis)
        .unwrap();
    assert!(inputs.up[0].local_orbitals.is_empty());
    assert!(inputs.down[0].local_orbitals.is_empty());
}

#[test]
fn nonempty_core_is_present_initially_and_in_the_scf_iteration() {
    let (checkpoint, config) = core_checkpoint_and_config();
    let mut physics = CheckpointPhysics::new(&checkpoint).unwrap();
    let initial = physics.initial_density(&config).unwrap();
    let initial_count = muffintin_dft::electron_count(&initial).unwrap();
    assert!(
        (initial_count - 2.0).abs() < 1.0e-8,
        "initial core+valence electron count was {initial_count}"
    );

    let mut physics = CheckpointPhysics::new(&checkpoint).unwrap();
    let state = run_scf(&mut physics, &config, None).unwrap();
    assert_eq!(state.iterations(), 2);
}

#[test]
fn frozen_consumers_use_their_source_states_basis_after_a_later_scf() {
    let mut physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    let first_config = config(ScfRelativity::Scalar);
    let first = run_scf(&mut physics, &first_config, None).unwrap();
    let mut later_config = first_config.clone();
    later_config.basis.plane_wave_cutoff = 0.55;
    let later = run_scf(&mut physics, &later_config, Some(&first)).unwrap();
    assert_eq!(first.basis.plane_wave_cutoff, 0.5);
    assert_eq!(later.basis.plane_wave_cutoff, 0.55);

    let request = BandPathRequest {
        bands: 1,
        points: vec![
            BandPathPoint {
                label: "G".to_owned(),
                k: [0.0; 3],
            },
            BandPathPoint {
                label: "X".to_owned(),
                k: [0.5, 0.0, 0.0],
            },
        ],
    };
    assert_eq!(physics.solve_band_path(&first, &request).unwrap().len(), 2);
}

#[test]
fn second_variation_rejects_a_window_that_would_drop_lower_scalar_bands() {
    let mut physics = CheckpointPhysics::new(&checkpoint()).unwrap();
    let config = config(ScfRelativity::SocSecondVariation {
        window: FirstVariationWindow::new(1, 2).unwrap(),
    });
    assert!(matches!(
        physics.initial_density(&config),
        Err(CheckpointPhysicsError::SecondVariationDropsLowerBands { start: 1 })
    ));
}
