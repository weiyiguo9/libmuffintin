use std::collections::BTreeMap;

use muffintin_core::{Bohr, ExponentialMesh, Hartree};
use muffintin_dft::{
    FreeAtomScfSpec, LinearizationEnergyGenerator, NoncollinearXcRoute, ScfBasis,
    ScfChannelIdentity, ScfChannelProvenance, ScfChannelRecipe, ScfChannelTreatment, ScfConfig,
    ScfConvergence, ScfCoreSite, ScfExchangeCorrelation, ScfKMesh, ScfMixing, ScfOccupations,
    ScfPhysics, ScfRelativity, XcFunctional, production_density_layout,
};
use muffintin_io::{
    AngularBasis, CheckpointFile, CheckpointMeta, EnergyParameterV1, EnergyUnit,
    ExponentialMeshSpec, GeometryV2, InitialV2, LatticeV1, LengthUnit, LinearizationV1,
    PotentialConventionV1, PotentialRadialQuantityV1, RadialBasisSpinV2, RadialEquationTag,
    SiteRadialBasisV2, SiteV2, SphericalChannelConvention, checkpoint_file_from_toml,
    checkpoint_file_to_toml,
};

use super::{
    AtomicCheckpointRequest, CheckpointPhysics, materialize_atomic_checkpoint_v2,
};

#[test]
fn neutral_atomic_checkpoint_enters_the_native_restart_and_potential_path() {
    let first = 1.0e-4_f64;
    let radius = 1.5_f64;
    let point_count = 61;
    let log_increment = (radius / first).ln() / (point_count - 1) as f64;
    let geometry = GeometryV2 {
        lattice: LatticeV1 {
            unit: LengthUnit::Bohr,
            vectors: [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        },
        sites: vec![SiteV2 {
            id: "H-1".to_owned(),
            atomic_number: 1,
            fractional_position: [0.5, 0.5, 0.5],
            muffin_tin_radius_unit: LengthUnit::Bohr,
            muffin_tin_radius: radius,
        }],
        radial_basis: vec![SiteRadialBasisV2 {
            site_id: "H-1".to_owned(),
            spin: RadialBasisSpinV2::Scalar,
            mesh: ExponentialMeshSpec {
                radius_unit: LengthUnit::Bohr,
                first,
                log_increment,
                point_count,
                last: radius,
                consistency_tolerance: 1.0e-12,
            },
            radial_equation: RadialEquationTag::ScalarKoellingHarmon,
            linearization: LinearizationV1 {
                energy_unit: EnergyUnit::Hartree,
                linearization_energies: vec![EnergyParameterV1 { l: 0, energy: -0.3 }],
                local_orbital_energies: Vec::new(),
            },
        }],
    };
    let meta = CheckpointMeta {
        title: "neutral atomic checkpoint production test".to_owned(),
        producer: "libmuffintin-runtime".to_owned(),
        producer_version: None,
        energy_zero: "periodic crystal electrostatic reference".to_owned(),
        potential_convention: PotentialConventionV1 {
            angular_basis: AngularBasis::ComplexCondonShortley,
            radial_quantity: PotentialRadialQuantityV1::Potential,
            spherical_channel: SphericalChannelConvention::PhysicalValue,
        },
        annotations: BTreeMap::new(),
    };
    let config = ScfConfig {
        electron_count: 1.0,
        k_mesh: ScfKMesh {
            divisions: [1, 1, 1],
            shift: [0.0; 3],
        },
        basis: ScfBasis {
            plane_wave_cutoff: 4.0,
            l_max: 1,
            channels: vec![ScfChannelRecipe {
                site: "H-1".to_owned(),
                identity: ScfChannelIdentity::ScalarL { n: 1, l: 0 },
                treatment: ScfChannelTreatment::Valence,
                derivative_order: 0,
                generator: LinearizationEnergyGenerator::Explicit,
                seed: Some(Hartree(-0.3)),
                provenance: ScfChannelProvenance::BuiltIn,
            }],
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
        relativity: ScfRelativity::Scalar,
        convergence: ScfConvergence {
            energy_tolerance: Hartree(1.0e-6),
            density_tolerance: 1.0e-6,
            max_iterations: 2,
        },
        core_sites: vec![ScfCoreSite {
            id: "H-1".to_owned(),
            states: Vec::new(),
        }],
    };
    let free_atom_mesh = ExponentialMesh::new(Bohr(1.0e-6), 0.01, 1683).unwrap();
    let generated = materialize_atomic_checkpoint_v2(AtomicCheckpointRequest {
        meta,
        geometry,
        scf: config.clone(),
        free_atom_scf: FreeAtomScfSpec {
            mesh: free_atom_mesh,
            mixing: 0.3,
            potential_tolerance: 2.0e-5,
            tail_tolerance: 1.0e-7,
            max_iterations: 120,
        },
        atomic_superposition_angular_points: 50,
    })
    .unwrap();

    generated.checkpoint.validate().unwrap();
    let text = checkpoint_file_to_toml(&CheckpointFile::V2(generated.checkpoint.clone())).unwrap();
    assert_eq!(
        checkpoint_file_from_toml(&text).unwrap(),
        CheckpointFile::V2(generated.checkpoint.clone())
    );
    let InitialV2::Restart { density, potential } = &generated.checkpoint.initial else {
        panic!("atomic materialization must produce a restart checkpoint");
    };
    let mut physics = CheckpointPhysics::new(&generated.checkpoint).unwrap();
    let expected_layout = production_density_layout(
        *physics.kernel.reciprocal(),
        config.k_mesh,
        config.basis.plane_wave_cutoff,
    )
    .unwrap();
    let expected_g = expected_layout
        .vectors()
        .iter()
        .map(|vector| vector.index)
        .collect::<Vec<_>>();
    assert_eq!(
        density
            .n
            .interstitial
            .coefficients
            .iter()
            .map(|coefficient| coefficient.g)
            .collect::<Vec<_>>(),
        expected_g
    );
    assert_eq!(
        potential
            .v0
            .interstitial
            .coefficients
            .iter()
            .map(|coefficient| coefficient.g)
            .collect::<Vec<_>>(),
        expected_g
    );
    let restart_density = physics.kernel.restart_density().cloned().unwrap();
    let initial_density = ScfPhysics::initial_density(&mut physics.kernel, &config).unwrap();
    assert_eq!(initial_density, restart_density);
    let rebuilt = ScfPhysics::build_potential(
        &mut physics.kernel,
        0,
        &initial_density,
        config.exchange_correlation,
    )
    .unwrap();
    assert_eq!(rebuilt.scalar().interstitial().layout(), &expected_layout);
}
