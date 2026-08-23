#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use muffintin::{
    BandPathPointV1, BasisV1, ConvergenceV1, ElectronicStateOverrideV1, ElectronicStateTreatmentV1,
    EnergyWindowV1, ExchangeCorrelationV1, InputV1, KMeshV1, LocalOrbitalKindV1, LocalOrbitalV1,
    MixingV1, OccupationsV1, RelativityV1, TaskV1, WorkflowV1, input_to_toml,
};
use muffintin_io::{
    AngularBasisV1, BasisHintsV1, Complex64V1, EnergyParameterV1, EnergyUnitV1,
    ExponentialMeshSpecV1, FourierCoefficientV1, FourierNormalizationV1, FourierPhaseV1,
    GeometryV1, InterstitialV1, InverseLengthUnitV1, LatticeV1, LengthUnitV1, LinearizationV1,
    MetaV1, PotentialChannelV1, PotentialConventionV1, PotentialRadialQuantityV1,
    RadialEquationTagV1, SiteSpinV1, SiteV1, SnapshotV1, SphericalChannelConventionV1, SpinTagV1,
    snapshot_to_toml,
};

pub fn sample_input() -> InputV1 {
    InputV1::new(
        PathBuf::from("data/snapshot.toml"),
        WorkflowV1 {
            tasks: vec!["scf".to_owned(), "bands".to_owned(), "dos".to_owned()],
        },
        BTreeMap::from([
            (
                "scf".to_owned(),
                TaskV1::DftScf {
                    source: None,
                    electron_count: 14.0,
                    k_mesh: KMeshV1 {
                        mesh: [4, 4, 4],
                        shift: [0.5, 0.5, 0.5],
                    },
                    basis: BasisV1 {
                        plane_wave_cutoff: 4.0,
                        l_max: 8,
                        local_orbitals: vec![LocalOrbitalV1 {
                            site: "Si-1".to_owned(),
                            kappa: 1,
                            energy: -0.15,
                            kind: LocalOrbitalKindV1::Lo,
                        }],
                    },
                    occupations: OccupationsV1::FermiDirac { temperature: 0.01 },
                    xc: ExchangeCorrelationV1::LdaPw92 {
                        noncollinear_route: Default::default(),
                    },
                    mixing: MixingV1::PulayAnderson {
                        beta: 0.4,
                        history: 6,
                    },
                    relativity: RelativityV1::SocSecondVariation {
                        band_window: [0, 12],
                    },
                    convergence: ConvergenceV1 {
                        energy_tolerance: 1.0e-8,
                        density_tolerance: 1.0e-7,
                        max_iterations: 80,
                    },
                    state_overrides: vec![ElectronicStateOverrideV1 {
                        site: "Si-1".to_owned(),
                        principal_quantum_number: 2,
                        kappa: 1,
                        treatment: ElectronicStateTreatmentV1::Valence,
                    }],
                },
            ),
            (
                "bands".to_owned(),
                TaskV1::DftBands {
                    source: "scf.state".to_owned(),
                    bands: 12,
                    path: vec![
                        BandPathPointV1 {
                            label: "G".to_owned(),
                            k: [0.0, 0.0, 0.0],
                        },
                        BandPathPointV1 {
                            label: "X".to_owned(),
                            k: [0.5, 0.0, 0.0],
                        },
                    ],
                },
            ),
            (
                "dos".to_owned(),
                TaskV1::DftDos {
                    source: "scf.state".to_owned(),
                    k_mesh: KMeshV1 {
                        mesh: [8, 8, 8],
                        shift: [0.0, 0.0, 0.0],
                    },
                    energy_window: EnergyWindowV1 {
                        minimum: -1.0,
                        maximum: 1.0,
                    },
                    points: 401,
                    broadening: 0.005,
                },
            ),
        ]),
    )
}

pub fn sample_snapshot() -> SnapshotV1 {
    let point_count = 7;
    let first = 0.1;
    let increment = 0.2;
    SnapshotV1::new(
        MetaV1 {
            title: "runtime fixture".to_owned(),
            producer: "mt-runtime test".to_owned(),
            producer_version: None,
            energy_zero: "cell-average interstitial potential".to_owned(),
            potential_convention: PotentialConventionV1 {
                angular_basis: AngularBasisV1::RealTesseralCondonShortley,
                radial_quantity: PotentialRadialQuantityV1::Potential,
                spherical_channel: SphericalChannelConventionV1::PhysicalValue,
            },
            annotations: BTreeMap::new(),
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
                        local_orbital_energies: Vec::new(),
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

pub fn supported_input() -> InputV1 {
    InputV1::new(
        PathBuf::from("data/snapshot.toml"),
        WorkflowV1 {
            tasks: vec!["scf".to_owned()],
        },
        BTreeMap::from([(
            "scf".to_owned(),
            TaskV1::DftScf {
                source: None,
                electron_count: 1.0,
                k_mesh: KMeshV1 {
                    mesh: [1, 1, 1],
                    shift: [0.0; 3],
                },
                basis: BasisV1 {
                    plane_wave_cutoff: 0.5,
                    l_max: 1,
                    local_orbitals: Vec::new(),
                },
                occupations: OccupationsV1::FermiDirac { temperature: 0.02 },
                xc: ExchangeCorrelationV1::LdaPw92 {
                    noncollinear_route: Default::default(),
                },
                mixing: MixingV1::Linear { beta: 1.0 },
                relativity: RelativityV1::Scalar {},
                convergence: ConvergenceV1 {
                    energy_tolerance: 1.0e100,
                    density_tolerance: 1.0e100,
                    max_iterations: 2,
                },
                state_overrides: Vec::new(),
            },
        )]),
    )
}

pub fn supported_snapshot() -> SnapshotV1 {
    let point_count = 61;
    let first: f64 = 1.0e-4;
    let radius: f64 = 1.0;
    let increment = (radius / first).ln() / (point_count - 1) as f64;
    let radii = (0..point_count)
        .map(|index| first * (index as f64 * increment).exp())
        .collect::<Vec<_>>();
    SnapshotV1::new(
        MetaV1 {
            title: "supported runtime hydrogen smoke".to_owned(),
            producer: "mt-runtime test".to_owned(),
            producer_version: None,
            energy_zero: "zero interstitial Fourier mean".to_owned(),
            potential_convention: PotentialConventionV1 {
                angular_basis: AngularBasisV1::ComplexCondonShortley,
                radial_quantity: PotentialRadialQuantityV1::Potential,
                spherical_channel: SphericalChannelConventionV1::PhysicalValue,
            },
            annotations: BTreeMap::new(),
        },
        GeometryV1 {
            lattice: LatticeV1 {
                unit: LengthUnitV1::Bohr,
                vectors: [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            },
            sites: vec![SiteV1 {
                id: "H-1".to_owned(),
                atomic_number: 1,
                fractional_position: [0.5; 3],
                muffin_tin_radius_unit: LengthUnitV1::Bohr,
                muffin_tin_radius: radius,
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
                        real: radii.iter().map(|radius| -1.0 / radius).collect(),
                        imaginary: Vec::new(),
                    }],
                    linearization: LinearizationV1 {
                        energy_unit: EnergyUnitV1::Hartree,
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
            coefficient_unit: EnergyUnitV1::Hartree,
            coefficients: vec![FourierCoefficientV1 {
                g: [0; 3],
                value: Complex64V1 {
                    real: 0.0,
                    imaginary: 0.0,
                },
            }],
            basis_hints: BasisHintsV1 {
                reciprocal_length_unit: InverseLengthUnitV1::BohrInverse,
                plane_wave_cutoff: Some(0.5),
                coefficient_cutoff: Some(1.0),
                normalization: FourierNormalizationV1::CellNormalized,
                phase: FourierPhaseV1::NegativeExponent,
            },
        },
    )
}

pub struct FixtureDirectory {
    root: PathBuf,
}

impl FixtureDirectory {
    pub fn new() -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "libmuffintin-runtime-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("data")).unwrap();
        Self { root }
    }

    pub fn write_workflow(&self) -> PathBuf {
        let input_path = self.root.join("input.toml");
        fs::write(&input_path, input_to_toml(&sample_input()).unwrap()).unwrap();
        fs::write(
            self.root.join("data/snapshot.toml"),
            snapshot_to_toml(&sample_snapshot()).unwrap(),
        )
        .unwrap();
        input_path
    }

    pub fn write_supported_workflow(&self) -> PathBuf {
        let input_path = self.root.join("supported.toml");
        fs::write(&input_path, input_to_toml(&supported_input()).unwrap()).unwrap();
        fs::write(
            self.root.join("data/snapshot.toml"),
            snapshot_to_toml(&supported_snapshot()).unwrap(),
        )
        .unwrap();
        input_path
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl Drop for FixtureDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
