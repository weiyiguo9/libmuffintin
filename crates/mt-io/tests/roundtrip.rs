use std::collections::BTreeMap;

use libmuffintin_core::Bohr;
use libmuffintin_grid::{Cell, Grid, UniformGrid};
use libmuffintin_io::{
    AngularBasisV1, BasisHintsV1, Complex64V1, EnergyParameterV1, EnergyUnitV1,
    ExponentialMeshSpecV1, FourierCoefficientV1, FourierNormalizationV1, FourierPhaseV1,
    GeometryV1, GridArtifactV1, InterstitialV1, InverseLengthUnitV1, IoError, LatticeV1,
    LengthUnitV1, LinearizationV1, MetaV1, PotentialChannelV1, PotentialConventionV1,
    PotentialRadialQuantityV1, RadialEquationTagV1, SiteSpinV1, SiteV1, SnapshotV1,
    SphericalChannelConventionV1, SpinTagV1, VolumeUnitV1, grid_artifact_from_toml,
    grid_artifact_to_toml, snapshot_from_toml, snapshot_to_toml,
};

fn snapshot() -> SnapshotV1 {
    let point_count = 7;
    let first = 0.1;
    let increment = 0.2;
    SnapshotV1::new(
        MetaV1 {
            title: "minimal silicon snapshot".to_owned(),
            producer: "mt-io test".to_owned(),
            producer_version: Some("1.2.3".to_owned()),
            energy_zero: "cell-average interstitial potential".to_owned(),
            potential_convention: PotentialConventionV1 {
                angular_basis: AngularBasisV1::RealTesseralCondonShortley,
                radial_quantity: PotentialRadialQuantityV1::Potential,
                spherical_channel: SphericalChannelConventionV1::PhysicalValue,
            },
            annotations: BTreeMap::from([("source".to_owned(), "fixture".to_owned())]),
        },
        GeometryV1 {
            lattice: LatticeV1 {
                unit: LengthUnitV1::Bohr,
                vectors: [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            },
            sites: vec![SiteV1 {
                id: "Si-1".to_owned(),
                atomic_number: 14,
                fractional_position: [0.0, 0.0, 0.0],
                muffin_tin_radius_unit: LengthUnitV1::Bohr,
                muffin_tin_radius: 2.0,
                spins: vec![SiteSpinV1 {
                    spin: SpinTagV1::Scalar,
                    mesh: ExponentialMeshSpecV1 {
                        radius_unit: LengthUnitV1::Bohr,
                        first,
                        log_increment: increment,
                        point_count,
                        last: first * ((point_count - 1) as f64 * increment).exp(),
                        consistency_tolerance: 1.0e-12,
                    },
                    radial_equation: RadialEquationTagV1::ScalarKoellingHarmon,
                    potential_unit: EnergyUnitV1::Hartree,
                    potential_channels: vec![PotentialChannelV1 {
                        l: 0,
                        m: 0,
                        real: vec![-1.0, -0.8, -0.6, -0.4, -0.2, -0.1, 0.0],
                        imaginary: Vec::new(),
                    }],
                    linearization: LinearizationV1 {
                        energy_unit: EnergyUnitV1::Hartree,
                        linearization_energies: vec![EnergyParameterV1 { l: 0, energy: -0.2 }],
                        local_orbital_energies: vec![EnergyParameterV1 { l: 1, energy: -1.1 }],
                    },
                }],
            }],
        },
        InterstitialV1 {
            coefficient_unit: EnergyUnitV1::Hartree,
            coefficients: vec![FourierCoefficientV1 {
                g: [0, 0, 0],
                value: Complex64V1 {
                    real: 0.25,
                    imaginary: 0.0,
                },
            }],
            basis_hints: BasisHintsV1 {
                reciprocal_length_unit: InverseLengthUnitV1::BohrInverse,
                plane_wave_cutoff: Some(4.0),
                coefficient_cutoff: Some(8.0),
                normalization: FourierNormalizationV1::CellNormalized,
                phase: FourierPhaseV1::NegativeExponent,
            },
        },
    )
}

#[test]
fn snapshot_toml_round_trips_with_header_first() {
    let snapshot = snapshot();
    let encoded = snapshot_to_toml(&snapshot).unwrap();
    assert!(encoded.starts_with("format = \"libmuffintin-snapshot\"\nversion = 1\n"));
    assert_eq!(snapshot_from_toml(&encoded).unwrap(), snapshot);
}

#[test]
fn grid_artifact_round_trips_independently() {
    let grid = GridArtifactV1::new(
        LengthUnitV1::Bohr,
        VolumeUnitV1::Bohr3,
        vec![[0.0, 0.0, 0.0], [1.0, 0.5, -0.5]],
        vec![0.4, 0.6],
        vec!["interstitial".to_owned(), "muffin-tin:Si-1".to_owned()],
    );
    let encoded = grid_artifact_to_toml(&grid).unwrap();
    assert!(encoded.starts_with("format = \"libmuffintin-grid-artifact\"\nversion = 1\n"));
    assert_eq!(grid_artifact_from_toml(&encoded).unwrap(), grid);
    assert!(!encoded.contains("geometry"));
}

#[test]
fn grid_artifact_preserves_libmuffintin_grid_order_and_units() {
    let cell = Cell::new([
        [Bohr(2.0), Bohr(0.0), Bohr(0.0)],
        [Bohr(0.0), Bohr(3.0), Bohr(0.0)],
        [Bohr(0.0), Bohr(0.0), Bohr(4.0)],
    ])
    .unwrap();
    let grid = UniformGrid::new(cell, [2, 1, 1]).unwrap();
    let artifact = GridArtifactV1::from_grid(&grid);
    assert_eq!(artifact.points.len(), grid.len());
    assert_eq!(artifact.points[0], [0.5, 1.5, 2.0]);
    assert_eq!(artifact.points[1], [1.5, 1.5, 2.0]);
    assert_eq!(artifact.weights, vec![12.0, 12.0]);
    assert_eq!(artifact.region_tags, vec!["uniform", "uniform"]);
}

#[test]
fn snapshot_rejects_unknown_version() {
    let encoded = snapshot_to_toml(&snapshot()).unwrap();
    let wrong = encoded.replacen("version = 1", "version = 2", 1);
    assert!(matches!(
        snapshot_from_toml(&wrong),
        Err(IoError::UnsupportedVersion { found: 2, .. })
    ));
}

#[test]
fn grid_rejects_unknown_version() {
    let grid = GridArtifactV1::new(
        LengthUnitV1::Bohr,
        VolumeUnitV1::Bohr3,
        vec![[0.0, 0.0, 0.0]],
        vec![1.0],
        vec!["interstitial".to_owned()],
    );
    let encoded = grid_artifact_to_toml(&grid).unwrap();
    let wrong = encoded.replacen("version = 1", "version = 9", 1);
    assert!(matches!(
        grid_artifact_from_toml(&wrong),
        Err(IoError::UnsupportedVersion { found: 9, .. })
    ));
}

#[test]
fn snapshot_rejects_bad_lm_length_and_mesh_endpoint() {
    let mut bad_lm = snapshot();
    let channel = &mut bad_lm.geometry.sites[0].spins[0].potential_channels[0];
    channel.l = 1;
    channel.m = 2;
    assert!(snapshot_to_toml(&bad_lm).is_err());

    let mut bad_length = snapshot();
    bad_length.geometry.sites[0].spins[0].potential_channels[0]
        .real
        .pop();
    assert!(snapshot_to_toml(&bad_length).is_err());

    let mut bad_mesh = snapshot();
    bad_mesh.geometry.sites[0].spins[0].mesh.last *= 1.1;
    assert!(snapshot_to_toml(&bad_mesh).is_err());
}

#[test]
fn readers_reject_non_finite_values_and_grid_length_mismatch() {
    let encoded = snapshot_to_toml(&snapshot()).unwrap();
    let non_finite = encoded.replacen("real = 0.25", "real = nan", 1);
    assert!(snapshot_from_toml(&non_finite).is_err());

    let grid = GridArtifactV1::new(
        LengthUnitV1::Bohr,
        VolumeUnitV1::Bohr3,
        vec![[0.0, 0.0, 0.0]],
        Vec::new(),
        vec!["interstitial".to_owned()],
    );
    assert!(grid_artifact_to_toml(&grid).is_err());
}
