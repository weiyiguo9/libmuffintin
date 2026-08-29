//! Focused `spex.snapshot_hdf` v1 two-source reader.

use std::collections::BTreeMap;
use std::path::PathBuf;

use muffintin_io::{
    AngularBasisV1, BasisHintsV1, Complex64V2, EnergyParameterV1, ExponentialMeshSpecV1,
    FieldRepresentationV2, FieldUnitV2, FourierCoefficientV2, FourierNormalizationV1,
    FourierPhaseV1, GeometryV2, InitialV2, InterstitialFieldV2, IoError, LatticeV1,
    LinearizationV1, MetaV1, MuffinTinFieldV2, PotentialConventionV1, PotentialRadialQuantityV1,
    PotentialV2, RadialBasisSpinV2, RadialEquationTagV1, RegionalFieldV2, SNAPSHOT_FORMAT,
    SNAPSHOT_VERSION_V2, SPEX_FOURIER_HERMITIAN_TOLERANCE, SPEX_SNAPSHOT_HDF_SCHEMA_NAME,
    SPEX_SNAPSHOT_HDF_SCHEMA_VERSION, SPEX_SNAPSHOT_HDF_SOURCE_KIND, SiteRadialBasisV2, SiteV2,
    SnapshotV2, SpexFrozenFieldsV1, SpexMaterialBasisRecipeV1, SpexMaterialChannelKind,
    SpexMaterialChannelV1, SpexScalarLoKind, SpexScalarLoTableV1, SpexScalarLoV1,
    SpexSnapshotHashV1, SphericalChannelConventionV1, SphericalChannelV2, ValidationError,
    materialize_snapshot_v2, read_spex_snapshot_hdf, write_spex_snapshot_hdf,
};
use muffintin_io::{EnergyUnitV1, InverseLengthUnitV1, LengthUnitV1};

fn fixture_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

fn mesh() -> ExponentialMeshSpecV1 {
    let first = 1.0e-4;
    let log_increment = 0.5;
    let point_count = 7;
    let last = first * (((point_count - 1) as f64) * log_increment).exp();
    ExponentialMeshSpecV1 {
        radius_unit: LengthUnitV1::Bohr,
        first,
        log_increment,
        point_count,
        last,
        consistency_tolerance: 1.0e-12,
    }
}

fn radial_samples(mesh: &ExponentialMeshSpecV1, scale: f64) -> (Vec<f64>, Vec<f64>) {
    let real = (0..mesh.point_count)
        .map(|index| scale / (1.0 + index as f64))
        .collect();
    let imaginary = vec![0.0; mesh.point_count];
    (real, imaginary)
}

fn regional(site_id: &str, mesh: &ExponentialMeshSpecV1, scale: f64) -> RegionalFieldV2 {
    let (real, imaginary) = radial_samples(mesh, scale);
    RegionalFieldV2 {
        muffin_tins: vec![MuffinTinFieldV2 {
            site_id: site_id.to_owned(),
            channels: vec![SphericalChannelV2 {
                l: 0,
                m: 0,
                real,
                imaginary,
            }],
        }],
        interstitial: InterstitialFieldV2 {
            coefficients: vec![FourierCoefficientV2 {
                g: [0, 0, 0],
                value: Complex64V2 {
                    real: scale * 0.01,
                    imaginary: 0.0,
                },
            }],
        },
    }
}

fn sample_fields() -> SpexFrozenFieldsV1 {
    let mesh = mesh();
    let site_id = "Sm-1";
    let hints = BasisHintsV1 {
        reciprocal_length_unit: InverseLengthUnitV1::BohrInverse,
        plane_wave_cutoff: Some(3.5),
        coefficient_cutoff: Some(1.0e-8),
        normalization: FourierNormalizationV1::CellNormalized,
        phase: FourierPhaseV1::NegativeExponent,
    };
    let v0 = regional(site_id, &mesh, -1.0);
    let zero = regional(site_id, &mesh, 0.0);
    let snapshot = SnapshotV2 {
        format: SNAPSHOT_FORMAT.to_owned(),
        version: SNAPSHOT_VERSION_V2,
        meta: MetaV1 {
            title: "spex snapshot hdf fixture".to_owned(),
            producer: "spex-test".to_owned(),
            producer_version: Some("06.00pre38".to_owned()),
            energy_zero: "SPEX absolute Hartree".to_owned(),
            potential_convention: PotentialConventionV1 {
                angular_basis: AngularBasisV1::ComplexCondonShortley,
                radial_quantity: PotentialRadialQuantityV1::Potential,
                spherical_channel: SphericalChannelConventionV1::PhysicalValue,
            },
            annotations: BTreeMap::new(),
        },
        geometry: GeometryV2 {
            lattice: LatticeV1 {
                unit: LengthUnitV1::Bohr,
                vectors: [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            },
            sites: vec![SiteV2 {
                id: site_id.to_owned(),
                atomic_number: 62,
                fractional_position: [0.0, 0.0, 0.0],
                muffin_tin_radius_unit: LengthUnitV1::Bohr,
                muffin_tin_radius: mesh.last,
            }],
            radial_basis: vec![SiteRadialBasisV2 {
                site_id: site_id.to_owned(),
                spin: RadialBasisSpinV2::Scalar,
                mesh,
                radial_equation: RadialEquationTagV1::ScalarKoellingHarmon,
                linearization: LinearizationV1 {
                    energy_unit: EnergyUnitV1::Hartree,
                    linearization_energies: vec![
                        EnergyParameterV1 { l: 0, energy: -0.2 },
                        EnergyParameterV1 {
                            l: 3,
                            energy: -0.15,
                        },
                    ],
                    local_orbital_energies: vec![EnergyParameterV1 { l: 1, energy: -0.8 }],
                },
            }],
        },
        initial: InitialV2::FrozenPotential {
            potential: PotentialV2 {
                unit: FieldUnitV2::Hartree,
                representation: FieldRepresentationV2::MaskedOperator,
                angular_basis: AngularBasisV1::ComplexCondonShortley,
                basis_hints: hints,
                v0,
                bx: zero.clone(),
                by: zero.clone(),
                bz: zero,
            },
        },
    };
    SpexFrozenFieldsV1 {
        snapshot,
        source_revision: "89ff8f8c80711eb6ded36efba688c8a7fd640bf9".to_owned(),
        source_kind: SPEX_SNAPSHOT_HDF_SOURCE_KIND.to_owned(),
        plane_wave_cutoff: 3.5,
        coefficient_cutoff: 1.0e-8,
        spin_layout: "collinear-up-down".to_owned(),
        interstitial_phase: "positive-exponent".to_owned(),
        hashes: vec![SpexSnapshotHashV1 {
            role: "spex.inp".to_owned(),
            name: "runs/sm_fcc_3x3_ref/spex.inp".to_owned(),
            sha256: "bd0734b9cfc6268489d10da7cb9bad159cc312a633650e2346e7460e3c17c179".to_owned(),
        }],
        scalar_los: vec![SpexScalarLoTableV1 {
            site_id: site_id.to_owned(),
            spin: RadialBasisSpinV2::Scalar,
            orbitals: vec![SpexScalarLoV1 {
                kind: SpexScalarLoKind::Lo,
                l: 1,
                energy: -0.8,
                n: None,
            }],
        }],
    }
}

fn matching_recipe() -> SpexMaterialBasisRecipeV1 {
    SpexMaterialBasisRecipeV1 {
        producer: "libmuffintin-ml7-recipe".to_owned(),
        recipe_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .to_owned(),
        channels: vec![SpexMaterialChannelV1 {
            site_id: "Sm-1".to_owned(),
            n: 5,
            l: 1,
            kappa: 1,
            kind: SpexMaterialChannelKind::Rlo,
            derivative_order: 0,
            energy: -0.8,
        }],
    }
}

#[test]
fn spex_frozen_fields_roundtrip_without_kappa() {
    let path = fixture_path("libmuffintin-spex-snapshot-hdf-v1-revised.h5");
    let source = sample_fields();
    write_spex_snapshot_hdf(&path, &source).unwrap();
    let hdf = hdf5_metno::File::open(&path).unwrap();
    let schema_attr = hdf.attr("schema_name").unwrap();
    assert_eq!(schema_attr.shape(), [1]);
    match schema_attr.dtype().unwrap().to_descriptor().unwrap() {
        hdf5_metno::types::TypeDescriptor::FixedAscii(len) => {
            assert_eq!(len, SPEX_SNAPSHOT_HDF_SCHEMA_NAME.len());
        }
        other => panic!("schema_name must be H5T_NATIVE_CHARACTER, got {other:?}"),
    }
    let version: i64 = hdf.attr("schema_version").unwrap().read_scalar().unwrap();
    assert_eq!(version, SPEX_SNAPSHOT_HDF_SCHEMA_VERSION);
    let orbitals = hdf
        .group("radial_basis")
        .unwrap()
        .group("basis_000")
        .unwrap()
        .group("orbitals")
        .unwrap();
    assert!(
        !orbitals
            .member_names()
            .unwrap()
            .iter()
            .any(|name| name == "kappa")
    );
    drop(hdf);
    let read = read_spex_snapshot_hdf(&path).unwrap();
    assert_eq!(read.snapshot.geometry, source.snapshot.geometry);
    assert_eq!(read.scalar_los, source.scalar_los);
    assert_eq!(read.spin_layout, "collinear-up-down");
    match read.snapshot.geometry.radial_basis[0].radial_equation {
        RadialEquationTagV1::ScalarKoellingHarmon => {}
        other => panic!("expected KH, got {other:?}"),
    }
    let roles = hdf5_metno::File::open(&path)
        .unwrap()
        .group("hashes")
        .unwrap()
        .dataset("roles")
        .unwrap();
    match roles.dtype().unwrap().to_descriptor().unwrap() {
        hdf5_metno::types::TypeDescriptor::FixedAscii(_) => {}
        other => panic!("hashes/roles must be 1-d native character, got {other:?}"),
    }
}

fn artifact_recipe(fields: &SpexFrozenFieldsV1) -> SpexMaterialBasisRecipeV1 {
    let mut channels = Vec::new();
    for table in &fields.scalar_los {
        for lo in &table.orbitals {
            let kappa = if lo.l == 0 {
                -1
            } else {
                i32::try_from(lo.l).unwrap()
            };
            channels.push(SpexMaterialChannelV1 {
                site_id: table.site_id.clone(),
                n: lo.n.unwrap_or(5),
                l: lo.l,
                kappa,
                kind: if lo.l == 1 {
                    SpexMaterialChannelKind::Rlo
                } else {
                    SpexMaterialChannelKind::Lo
                },
                derivative_order: 0,
                energy: lo.energy,
            });
        }
    }
    SpexMaterialBasisRecipeV1 {
        producer: "libmuffintin-ml7-sm-recipe".to_owned(),
        recipe_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            .to_owned(),
        channels,
    }
}

fn v0_coeff(fields: &SpexFrozenFieldsV1, g: [i32; 3]) -> muffintin_io::Complex64V2 {
    match &fields.snapshot.initial {
        InitialV2::FrozenPotential { potential } => {
            potential
                .v0
                .interstitial
                .coefficients
                .iter()
                .find(|coefficient| coefficient.g == g)
                .expect("G present")
                .value
        }
        InitialV2::Restart { .. } => panic!("expected frozen-potential"),
    }
}

/// Live SPEX snapshot at `/tmp/ml7-spex-artifact/snapshot.h5`.
///
/// Ordinary workspace tests skip this. Run:
/// `cargo test -p libmuffintin-io --test spex_snapshot_v1 consume_wsl_b45d9b9_snapshot_h5 -- --ignored --exact --nocapture`
#[ignore = "requires local SPEX artifact /tmp/ml7-spex-artifact/snapshot.h5; run with --ignored"]
#[test]
fn consume_wsl_b45d9b9_snapshot_h5() {
    let path = std::path::Path::new("/tmp/ml7-spex-artifact/snapshot.h5");
    assert!(
        path.is_file(),
        "authorized artifact missing at /tmp/ml7-spex-artifact/snapshot.h5"
    );
    let fields =
        read_spex_snapshot_hdf(path).expect("frozen reader must accept b45d9b9 snapshot.h5");
    assert_eq!(fields.source_kind, SPEX_SNAPSHOT_HDF_SOURCE_KIND);
    assert_eq!(fields.spin_layout, "collinear-up-down");
    assert_eq!(fields.interstitial_phase, "positive-exponent");
    assert_eq!(fields.hashes.len(), 2);
    assert_eq!(
        fields.hashes[0].sha256,
        "bd0734b9cfc6268489d10da7cb9bad159cc312a633650e2346e7460e3c17c179"
    );
    assert_eq!(
        fields.hashes[1].sha256,
        "92fdd4e4362e342cca4cefebe8dc6c181b317f599b5588e4e02d6ea67078019f"
    );
    assert_eq!(fields.scalar_los.len(), 2);
    assert_eq!(fields.scalar_los[0].orbitals.len(), 2);
    match fields.snapshot.geometry.radial_basis[0].radial_equation {
        RadialEquationTagV1::ScalarKoellingHarmon => {}
        other => panic!("expected KH, got {other:?}"),
    }
}

/// Live SPEX snapshot at `/tmp/ml7-spex-artifact/snapshot.h5`.
///
/// Ordinary workspace tests skip this. Run:
/// `cargo test -p libmuffintin-io --test spex_snapshot_v1 materialize_symmetrizes_live_ulp_and_rejects_large_hermitian_error -- --ignored --exact --nocapture`
#[ignore = "requires local SPEX artifact /tmp/ml7-spex-artifact/snapshot.h5; run with --ignored"]
#[test]
fn materialize_symmetrizes_live_ulp_and_rejects_large_hermitian_error() {
    let path = std::path::Path::new("/tmp/ml7-spex-artifact/snapshot.h5");
    assert!(path.is_file(), "authorized artifact missing");
    let fields = read_spex_snapshot_hdf(path).unwrap();
    let g = [-13, -7, -7];
    let opp = [13, 7, 7];
    let left = v0_coeff(&fields, g);
    let right = v0_coeff(&fields, opp);
    let live = (left.real - right.real).hypot(left.imaginary + right.imaginary);
    let scale = left
        .real
        .abs()
        .max(left.imaginary.abs())
        .max(right.real.abs())
        .max(right.imaginary.abs())
        .max(1.0);
    assert!(
        live > 0.0 && live <= SPEX_FOURIER_HERMITIAN_TOLERANCE * scale,
        "live discrepancy {live} must be positive ULP-scale vs tol {}",
        SPEX_FOURIER_HERMITIAN_TOLERANCE * scale
    );
    let recipe = artifact_recipe(&fields);
    let done = materialize_snapshot_v2(&fields, &recipe).expect("ULP-scale pair must materialize");
    let InitialV2::FrozenPotential { potential } = &done.snapshot.initial else {
        panic!("frozen-potential");
    };
    let left = potential
        .v0
        .interstitial
        .coefficients
        .iter()
        .find(|coefficient| coefficient.g == g)
        .unwrap()
        .value;
    let right = potential
        .v0
        .interstitial
        .coefficients
        .iter()
        .find(|coefficient| coefficient.g == opp)
        .unwrap()
        .value;
    assert_eq!(left.real, right.real);
    assert_eq!(left.imaginary, -right.imaginary);
    let zero = potential
        .v0
        .interstitial
        .coefficients
        .iter()
        .find(|coefficient| coefficient.g == [0, 0, 0])
        .unwrap()
        .value;
    assert_eq!(zero.imaginary, 0.0);

    let mut perturbed = fields.clone();
    match &mut perturbed.snapshot.initial {
        InitialV2::FrozenPotential { potential } => {
            let coefficient = potential
                .v0
                .interstitial
                .coefficients
                .iter_mut()
                .find(|coefficient| coefficient.g == g)
                .unwrap();
            coefficient.value.real += 1.0e-6;
        }
        InitialV2::Restart { .. } => panic!("frozen-potential"),
    }
    match materialize_snapshot_v2(&perturbed, &recipe) {
        Err(IoError::Validation(ValidationError::SpexFourierHermitian {
            g: failed,
            discrepancy,
            ..
        })) => {
            assert!(failed == g || failed == opp, "failed G {failed:?}");
            assert!(
                discrepancy > 1.0e-7,
                "perturbation must remain large, got {discrepancy}"
            );
        }
        Err(other) => panic!("expected SpexFourierHermitian, got {other:?}"),
        Ok(_) => panic!("1e-6 Hermitian break must not materialize"),
    }
}

#[test]
fn empty_lo_kind_table_is_allowed() {
    let path = fixture_path("libmuffintin-spex-snapshot-hdf-v1-empty-lo.h5");
    let mut source = sample_fields();
    source.scalar_los[0].orbitals.clear();
    source.snapshot.geometry.radial_basis[0]
        .linearization
        .local_orbital_energies
        .clear();
    write_spex_snapshot_hdf(&path, &source).unwrap();
    let read = read_spex_snapshot_hdf(&path).unwrap();
    assert!(read.scalar_los[0].orbitals.is_empty());
}

#[test]
fn missing_bx_is_a_typed_blocker() {
    let path = fixture_path("libmuffintin-spex-snapshot-hdf-v1-missing-bx.h5");
    write_spex_snapshot_hdf(&path, &sample_fields()).unwrap();
    {
        let hdf = hdf5_metno::File::open_rw(&path).unwrap();
        hdf.group("initial")
            .unwrap()
            .group("potential")
            .unwrap()
            .unlink("bx")
            .unwrap();
    }
    match read_spex_snapshot_hdf(&path) {
        Err(IoError::Validation(ValidationError::Missing { key, .. })) => {
            assert_eq!(key, "bx");
        }
        Err(other) => panic!("expected missing bx, got {other:?}"),
        Ok(_) => panic!("omitted Bx must not be defaulted"),
    }
}

#[test]
fn kappa_dataset_is_a_typed_blocker() {
    let path = fixture_path("libmuffintin-spex-snapshot-hdf-v1-kappa.h5");
    write_spex_snapshot_hdf(&path, &sample_fields()).unwrap();
    {
        let hdf = hdf5_metno::File::open_rw(&path).unwrap();
        let orbitals = hdf
            .group("radial_basis")
            .unwrap()
            .group("basis_000")
            .unwrap()
            .group("orbitals")
            .unwrap();
        orbitals
            .new_dataset::<i32>()
            .shape([1])
            .create("kappa")
            .unwrap()
            .write_raw(&[1i32])
            .unwrap();
    }
    match read_spex_snapshot_hdf(&path) {
        Err(IoError::Validation(ValidationError::InvalidValue { path, .. })) => {
            assert!(path.contains("kappa"), "{path}");
        }
        Err(other) => panic!("expected kappa forbidden, got {other:?}"),
        Ok(_) => panic!("SPEX kappa must not be accepted"),
    }
}

#[test]
fn materialize_requires_matching_recipe_not_spex_kappa() {
    let path = fixture_path("libmuffintin-spex-snapshot-hdf-v1-materialize.h5");
    let fields = sample_fields();
    write_spex_snapshot_hdf(&path, &fields).unwrap();
    let fields = read_spex_snapshot_hdf(&path).unwrap();
    let done = materialize_snapshot_v2(&fields, &matching_recipe()).unwrap();
    assert_eq!(
        done.snapshot.meta.annotations["material_basis.recipe_sha256"],
        matching_recipe().recipe_sha256
    );
    let mut bad = matching_recipe();
    bad.channels[0].l = 4;
    match materialize_snapshot_v2(&fields, &bad) {
        Err(IoError::Validation(ValidationError::InvalidValue { path, .. })) => {
            assert!(path.contains("lo"), "{path}");
        }
        Err(other) => panic!("expected l mismatch, got {other:?}"),
        Ok(_) => panic!("recipe must not invent an unmatched LO"),
    }
    let empty = SpexMaterialBasisRecipeV1 {
        producer: "x".to_owned(),
        recipe_sha256: matching_recipe().recipe_sha256,
        channels: Vec::new(),
    };
    match materialize_snapshot_v2(&fields, &empty) {
        Err(IoError::Validation(ValidationError::Empty { .. })) => {}
        Err(other) => panic!("expected empty recipe, got {other:?}"),
        Ok(_) => panic!("empty recipe is not a material basis"),
    }
}
