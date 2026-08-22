use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{IoError, ValidationError, finite, nonempty, positive};
use crate::snapshot::{
    AngularBasisV1, BasisHintsV1, Complex64V1, ExponentialMeshSpecV1, FourierCoefficientV1,
    GeometryV1, LatticeV1, LinearizationV1, MetaV1, PotentialChannelV1, RadialEquationTagV1,
    SNAPSHOT_FORMAT, SiteSpinV1, SnapshotV1, SpinTagV1,
};
use crate::units::LengthUnitV1;

/// Schema version for noncollinear Pauli-field snapshots.
pub const SNAPSHOT_VERSION_V2: u32 = 2;

/// A complete V2 snapshot with a frozen potential or restart state.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotV2 {
    pub format: String,
    pub version: u32,
    pub meta: MetaV1,
    pub geometry: GeometryV2,
    pub initial: InitialV2,
}

impl SnapshotV2 {
    pub fn new(meta: MetaV1, geometry: GeometryV2, initial: InitialV2) -> Self {
        Self {
            format: SNAPSHOT_FORMAT.to_owned(),
            version: SNAPSHOT_VERSION_V2,
            meta,
            geometry,
            initial,
        }
    }

    /// Check the V2 header, geometry/basis identity, and all Pauli-field layouts.
    pub fn validate(&self) -> Result<(), IoError> {
        if self.format != SNAPSHOT_FORMAT {
            return Err(IoError::InvalidFormat {
                expected: SNAPSHOT_FORMAT,
                found: self.format.clone(),
            });
        }
        if self.version != SNAPSHOT_VERSION_V2 {
            return Err(IoError::UnsupportedVersion {
                format: SNAPSHOT_FORMAT,
                supported: SNAPSHOT_VERSION_V2,
                found: self.version,
            });
        }
        self.meta.validate()?;
        self.geometry.validate()?;
        self.initial
            .validate(&self.geometry, self.meta.potential_convention.angular_basis)?;
        Ok(())
    }
}

/// Version-dispatched snapshot file.
#[derive(Clone, Debug, PartialEq)]
// Snapshot DTOs intentionally expose owned variants without Box in the public schema API.
#[allow(clippy::large_enum_variant)]
pub enum SnapshotFile {
    V1(SnapshotV1),
    V2(SnapshotV2),
}

impl SnapshotFile {
    pub fn validate(&self) -> Result<(), IoError> {
        match self {
            Self::V1(snapshot) => snapshot.validate(),
            Self::V2(snapshot) => snapshot.validate(),
        }
    }
}

/// V2 geometry and the radial basis metadata shared by all Pauli components.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryV2 {
    pub lattice: LatticeV1,
    pub sites: Vec<SiteV2>,
    pub radial_basis: Vec<SiteRadialBasisV2>,
}

impl GeometryV2 {
    fn validate(&self) -> Result<(), ValidationError> {
        self.lattice.validate()?;
        if self.sites.is_empty() {
            return Err(ValidationError::Empty {
                path: "geometry.sites".to_owned(),
            });
        }

        let mut site_ids = BTreeSet::new();
        let mut site_units = BTreeMap::new();
        for (index, site) in self.sites.iter().enumerate() {
            site.validate(index)?;
            if !site_ids.insert(site.id.clone()) {
                return Err(ValidationError::Duplicate {
                    path: "geometry.sites.id".to_owned(),
                    key: site.id.clone(),
                });
            }
            site_units.insert(site.id.as_str(), site.muffin_tin_radius_unit);
        }

        let mut basis_by_site: BTreeMap<&str, BTreeMap<RadialBasisSpinV2, &SiteRadialBasisV2>> =
            BTreeMap::new();
        for (index, basis) in self.radial_basis.iter().enumerate() {
            let path = format!("geometry.radial_basis[{index}]");
            basis.validate(&path)?;
            let Some(&radius_unit) = site_units.get(basis.site_id.as_str()) else {
                return Err(ValidationError::InvalidValue {
                    path: format!("{path}.site_id"),
                    expected: "an exact geometry site id".to_owned(),
                    actual: basis.site_id.clone(),
                });
            };
            if basis.mesh.radius_unit != radius_unit {
                return Err(ValidationError::InvalidValue {
                    path: format!("{path}.mesh.radius_unit"),
                    expected: format!("{:?}", radius_unit),
                    actual: format!("{:?}", basis.mesh.radius_unit),
                });
            }
            let by_spin = basis_by_site.entry(&basis.site_id).or_default();
            if by_spin.insert(basis.spin, basis).is_some() {
                return Err(ValidationError::Duplicate {
                    path: "geometry.radial_basis".to_owned(),
                    key: format!("{}:{:?}", basis.site_id, basis.spin),
                });
            }
        }

        for site in &self.sites {
            let Some(by_spin) = basis_by_site.get(site.id.as_str()) else {
                return Err(ValidationError::Missing {
                    path: "geometry.radial_basis".to_owned(),
                    key: site.id.clone(),
                });
            };
            match (
                by_spin.get(&RadialBasisSpinV2::Scalar),
                by_spin.get(&RadialBasisSpinV2::Up),
                by_spin.get(&RadialBasisSpinV2::Down),
            ) {
                (Some(_), None, None) => {}
                (None, Some(up), Some(down)) => {
                    if up.mesh != down.mesh {
                        return Err(ValidationError::LayoutMismatch {
                            path: format!("geometry.radial_basis[{}:down].mesh", site.id),
                            reference: format!("geometry.radial_basis[{}:up].mesh", site.id),
                        });
                    }
                }
                _ => {
                    return Err(ValidationError::InvalidValue {
                        path: format!("geometry.radial_basis[{}].spin", site.id),
                        expected: "scalar or the exact up/down pair".to_owned(),
                        actual: by_spin
                            .keys()
                            .map(|spin| format!("{spin:?}"))
                            .collect::<Vec<_>>()
                            .join(","),
                    });
                }
            }
        }
        Ok(())
    }

    fn point_count(&self, site_id: &str) -> Option<usize> {
        self.radial_basis
            .iter()
            .find(|basis| basis.site_id == site_id)
            .map(|basis| basis.mesh.point_count)
    }
}

/// Geometry-only site data; radial metadata is stored once in `GeometryV2::radial_basis`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SiteV2 {
    pub id: String,
    pub atomic_number: u16,
    pub fractional_position: [f64; 3],
    pub muffin_tin_radius_unit: LengthUnitV1,
    pub muffin_tin_radius: f64,
}

impl SiteV2 {
    fn validate(&self, site_index: usize) -> Result<(), ValidationError> {
        let path = format!("geometry.sites[{site_index}]");
        nonempty(format!("{path}.id"), &self.id)?;
        if self.atomic_number == 0 {
            return Err(ValidationError::Zero {
                path: format!("{path}.atomic_number"),
                value: 0.0,
            });
        }
        for (axis, &value) in self.fractional_position.iter().enumerate() {
            finite(format!("{path}.fractional_position[{axis}]"), value)?;
        }
        positive(format!("{path}.muffin_tin_radius"), self.muffin_tin_radius)
    }
}

/// Spin label retained only for the radial basis metadata.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RadialBasisSpinV2 {
    Scalar,
    Up,
    Down,
}

/// One site's radial equation and linearization metadata on an exact mesh.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SiteRadialBasisV2 {
    pub site_id: String,
    pub spin: RadialBasisSpinV2,
    pub mesh: ExponentialMeshSpecV1,
    pub radial_equation: RadialEquationTagV1,
    pub linearization: LinearizationV1,
}

impl SiteRadialBasisV2 {
    fn validate(&self, path: &str) -> Result<(), ValidationError> {
        nonempty(format!("{path}.site_id"), &self.site_id)?;
        self.mesh.validate(&format!("{path}.mesh"))?;
        self.linearization
            .validate(&format!("{path}.linearization"))
    }
}

/// Physical role of a regional field in the muffin-tin partition.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FieldRepresentationV2 {
    /// Density is a real periodic field, including its continuation through spheres.
    PeriodicExtension,
    /// Potential coefficients define the interstitial masked operator.
    MaskedOperator,
}

/// Units accepted by V2 Pauli fields; each field kind validates its required member.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum FieldUnitV2 {
    #[serde(rename = "bohr^-3")]
    BohrMinus3,
    #[serde(rename = "hartree")]
    Hartree,
}

/// One source-neutral spherical-harmonic radial channel.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SphericalChannelV2 {
    pub l: u32,
    pub m: i32,
    pub real: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub imaginary: Vec<f64>,
}

impl SphericalChannelV2 {
    fn validate(&self, path: &str, point_count: usize) -> Result<(), ValidationError> {
        if self.m.unsigned_abs() > self.l {
            return Err(ValidationError::InvalidLm {
                path: path.to_owned(),
                l: self.l,
                m: self.m,
            });
        }
        if self.real.len() != point_count {
            return Err(ValidationError::LengthMismatch {
                path: format!("{path}.real"),
                expected: point_count,
                actual: self.real.len(),
            });
        }
        if !self.imaginary.is_empty() && self.imaginary.len() != point_count {
            return Err(ValidationError::LengthMismatch {
                path: format!("{path}.imaginary"),
                expected: point_count,
                actual: self.imaginary.len(),
            });
        }
        for (index, &value) in self.real.iter().enumerate() {
            finite(format!("{path}.real[{index}]"), value)?;
        }
        for (index, &value) in self.imaginary.iter().enumerate() {
            finite(format!("{path}.imaginary[{index}]"), value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Complex64V2 {
    pub real: f64,
    pub imaginary: f64,
}

/// One source-neutral reciprocal coefficient keyed by integer dual coordinates.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FourierCoefficientV2 {
    pub g: [i32; 3],
    pub value: Complex64V2,
}

impl FourierCoefficientV2 {
    fn validate(&self, path: &str) -> Result<(), ValidationError> {
        finite(format!("{path}.value.real"), self.value.real)?;
        finite(format!("{path}.value.imaginary"), self.value.imaginary)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MuffinTinFieldV2 {
    pub site_id: String,
    pub channels: Vec<SphericalChannelV2>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InterstitialFieldV2 {
    pub coefficients: Vec<FourierCoefficientV2>,
}

/// One scalar component over the exact muffin-tin sites and reciprocal layout.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegionalFieldV2 {
    pub muffin_tins: Vec<MuffinTinFieldV2>,
    pub interstitial: InterstitialFieldV2,
}

impl RegionalFieldV2 {
    fn validate(&self, path: &str, geometry: &GeometryV2) -> Result<(), ValidationError> {
        let expected_sites: BTreeSet<_> = geometry.sites.iter().map(|site| &site.id).collect();
        let mut actual_sites = BTreeSet::new();
        for (site_index, site) in self.muffin_tins.iter().enumerate() {
            let site_path = format!("{path}.muffin_tins[{site_index}]");
            nonempty(format!("{site_path}.site_id"), &site.site_id)?;
            if !actual_sites.insert(&site.site_id) {
                return Err(ValidationError::Duplicate {
                    path: format!("{path}.muffin_tins.site_id"),
                    key: site.site_id.clone(),
                });
            }
            let Some(point_count) = geometry.point_count(&site.site_id) else {
                return Err(ValidationError::InvalidValue {
                    path: format!("{site_path}.site_id"),
                    expected: "an exact geometry site id".to_owned(),
                    actual: site.site_id.clone(),
                });
            };
            if site.channels.is_empty() {
                return Err(ValidationError::Empty {
                    path: format!("{site_path}.channels"),
                });
            }
            let mut lm = BTreeSet::new();
            for (channel_index, channel) in site.channels.iter().enumerate() {
                let channel_path = format!("{site_path}.channels[{channel_index}]");
                channel.validate(&channel_path, point_count)?;
                if !lm.insert((channel.l, channel.m)) {
                    return Err(ValidationError::Duplicate {
                        path: format!("{site_path}.channels"),
                        key: format!("({}, {})", channel.l, channel.m),
                    });
                }
            }
        }
        if actual_sites != expected_sites {
            return Err(ValidationError::LayoutMismatch {
                path: format!("{path}.muffin_tins.site_id"),
                reference: "geometry.sites.id".to_owned(),
            });
        }

        let mut wave_vectors = BTreeSet::new();
        for (index, coefficient) in self.interstitial.coefficients.iter().enumerate() {
            coefficient.validate(&format!("{path}.interstitial.coefficients[{index}]"))?;
            if !wave_vectors.insert(coefficient.g) {
                return Err(ValidationError::Duplicate {
                    path: format!("{path}.interstitial.coefficients.g"),
                    key: format!("{:?}", coefficient.g),
                });
            }
        }
        Ok(())
    }

    fn layout(&self) -> RegionalLayoutV2 {
        RegionalLayoutV2 {
            muffin_tins: self
                .muffin_tins
                .iter()
                .map(|site| {
                    (
                        site.site_id.clone(),
                        site.channels
                            .iter()
                            .map(|channel| (channel.l, channel.m))
                            .collect(),
                    )
                })
                .collect(),
            reciprocal: self
                .interstitial
                .coefficients
                .iter()
                .map(|coefficient| coefficient.g)
                .collect(),
        }
    }
}

#[derive(Eq, PartialEq)]
struct RegionalLayoutV2 {
    muffin_tins: BTreeMap<String, BTreeSet<(u32, i32)>>,
    reciprocal: BTreeSet<[i32; 3]>,
}

/// Charge and Cartesian Pauli magnetization in density-matrix convention.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DensityV2 {
    pub unit: FieldUnitV2,
    pub representation: FieldRepresentationV2,
    pub angular_basis: AngularBasisV1,
    pub basis_hints: BasisHintsV1,
    pub n: RegionalFieldV2,
    pub mx: RegionalFieldV2,
    pub my: RegionalFieldV2,
    pub mz: RegionalFieldV2,
}

impl DensityV2 {
    fn validate(
        &self,
        path: &str,
        geometry: &GeometryV2,
        angular_basis: AngularBasisV1,
    ) -> Result<(), ValidationError> {
        validate_field_header(
            path,
            self.unit,
            FieldUnitV2::BohrMinus3,
            self.representation,
            FieldRepresentationV2::PeriodicExtension,
            self.angular_basis,
            angular_basis,
        )?;
        self.basis_hints.validate(&format!("{path}.basis_hints"))?;
        validate_components(
            path,
            geometry,
            [
                ("n", &self.n),
                ("mx", &self.mx),
                ("my", &self.my),
                ("mz", &self.mz),
            ],
        )
    }
}

/// Scalar and Cartesian magnetic components of `V0 I + B . sigma`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PotentialV2 {
    pub unit: FieldUnitV2,
    pub representation: FieldRepresentationV2,
    pub angular_basis: AngularBasisV1,
    pub basis_hints: BasisHintsV1,
    pub v0: RegionalFieldV2,
    pub bx: RegionalFieldV2,
    pub by: RegionalFieldV2,
    pub bz: RegionalFieldV2,
}

impl PotentialV2 {
    fn validate(
        &self,
        path: &str,
        geometry: &GeometryV2,
        angular_basis: AngularBasisV1,
    ) -> Result<(), ValidationError> {
        validate_field_header(
            path,
            self.unit,
            FieldUnitV2::Hartree,
            self.representation,
            FieldRepresentationV2::MaskedOperator,
            self.angular_basis,
            angular_basis,
        )?;
        self.basis_hints.validate(&format!("{path}.basis_hints"))?;
        validate_components(
            path,
            geometry,
            [
                ("v0", &self.v0),
                ("bx", &self.bx),
                ("by", &self.by),
                ("bz", &self.bz),
            ],
        )
    }
}

fn validate_field_header(
    path: &str,
    unit: FieldUnitV2,
    expected_unit: FieldUnitV2,
    representation: FieldRepresentationV2,
    expected_representation: FieldRepresentationV2,
    angular_basis: AngularBasisV1,
    expected_angular_basis: AngularBasisV1,
) -> Result<(), ValidationError> {
    if unit != expected_unit {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.unit"),
            expected: format!("{expected_unit:?}"),
            actual: format!("{unit:?}"),
        });
    }
    if representation != expected_representation {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.representation"),
            expected: format!("{expected_representation:?}"),
            actual: format!("{representation:?}"),
        });
    }
    if angular_basis != expected_angular_basis {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.angular_basis"),
            expected: format!("{expected_angular_basis:?}"),
            actual: format!("{angular_basis:?}"),
        });
    }
    Ok(())
}

fn validate_components<'a, const N: usize>(
    path: &str,
    geometry: &GeometryV2,
    components: [(&'a str, &'a RegionalFieldV2); N],
) -> Result<(), ValidationError> {
    for &(name, component) in &components {
        component.validate(&format!("{path}.{name}"), geometry)?;
    }
    let (reference_name, reference) = components[0];
    let reference_layout = reference.layout();
    for (name, component) in &components[1..] {
        if component.layout() != reference_layout {
            return Err(ValidationError::LayoutMismatch {
                path: format!("{path}.{name}"),
                reference: format!("{path}.{reference_name}"),
            });
        }
    }
    Ok(())
}

/// Initial data carried by a V2 file.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
// Keeping density and potential as direct serde DTO fields makes the schema explicit.
#[allow(clippy::large_enum_variant)]
pub enum InitialV2 {
    FrozenPotential {
        potential: PotentialV2,
    },
    Restart {
        density: DensityV2,
        potential: PotentialV2,
    },
}

impl InitialV2 {
    fn validate(
        &self,
        geometry: &GeometryV2,
        angular_basis: AngularBasisV1,
    ) -> Result<(), ValidationError> {
        match self {
            Self::FrozenPotential { potential } => {
                potential.validate("initial.potential", geometry, angular_basis)
            }
            Self::Restart { density, potential } => {
                density.validate("initial.density", geometry, angular_basis)?;
                potential.validate("initial.potential", geometry, angular_basis)
            }
        }
    }
}

#[derive(Deserialize)]
struct SnapshotHeader {
    format: String,
    version: u32,
}

/// Parse the header first, dispatch to the exact schema, and validate the result.
pub fn snapshot_file_from_toml(text: &str) -> Result<SnapshotFile, IoError> {
    let header: SnapshotHeader = toml::from_str(text)?;
    if header.format != SNAPSHOT_FORMAT {
        return Err(IoError::InvalidFormat {
            expected: SNAPSHOT_FORMAT,
            found: header.format,
        });
    }
    let file = match header.version {
        1 => SnapshotFile::V1(toml::from_str(text)?),
        SNAPSHOT_VERSION_V2 => SnapshotFile::V2(toml::from_str(text)?),
        found => {
            return Err(IoError::UnsupportedVersion {
                format: SNAPSHOT_FORMAT,
                supported: SNAPSHOT_VERSION_V2,
                found,
            });
        }
    };
    file.validate()?;
    Ok(file)
}

/// Serialize either supported snapshot schema as deterministic pretty TOML.
pub fn snapshot_file_to_toml(file: &SnapshotFile) -> Result<String, IoError> {
    file.validate()?;
    let mut text = match file {
        SnapshotFile::V1(snapshot) => toml::to_string_pretty(snapshot)?,
        SnapshotFile::V2(snapshot) => toml::to_string_pretty(snapshot)?,
    };
    if !text.ends_with('\n') {
        text.push('\n');
    }
    Ok(text)
}

impl SnapshotV1 {
    /// Normalize legacy scalar/up-down potentials into the V2 Pauli convention.
    ///
    /// Scalar data maps to `V0` with zero `B`. For an up/down pair,
    /// `V0 = (Vup + Vdown)/2` and `Bz = (Vup - Vdown)/2` exactly, while
    /// `Bx = By = 0`. V1 has no density payload, so the result is frozen-potential.
    pub fn normalize_v2(&self) -> Result<SnapshotV2, IoError> {
        self.validate()?;

        let geometry = GeometryV2 {
            lattice: self.geometry.lattice,
            sites: self
                .geometry
                .sites
                .iter()
                .map(|site| SiteV2 {
                    id: site.id.clone(),
                    atomic_number: site.atomic_number,
                    fractional_position: site.fractional_position,
                    muffin_tin_radius_unit: site.muffin_tin_radius_unit,
                    muffin_tin_radius: site.muffin_tin_radius,
                })
                .collect(),
            radial_basis: self
                .geometry
                .sites
                .iter()
                .flat_map(|site| {
                    site.spins.iter().map(|spin| SiteRadialBasisV2 {
                        site_id: site.id.clone(),
                        spin: match spin.spin {
                            SpinTagV1::Scalar => RadialBasisSpinV2::Scalar,
                            SpinTagV1::Up => RadialBasisSpinV2::Up,
                            SpinTagV1::Down => RadialBasisSpinV2::Down,
                        },
                        mesh: spin.mesh,
                        radial_equation: spin.radial_equation,
                        linearization: spin.linearization.clone(),
                    })
                })
                .collect(),
        };

        let potential = potential_from_v1(&self.geometry, &self.interstitial, &self.meta)?;
        let snapshot = SnapshotV2::new(
            self.meta.clone(),
            geometry,
            InitialV2::FrozenPotential { potential },
        );
        snapshot.validate()?;
        Ok(snapshot)
    }
}

fn potential_from_v1(
    geometry: &GeometryV1,
    interstitial: &crate::snapshot::InterstitialV1,
    meta: &MetaV1,
) -> Result<PotentialV2, ValidationError> {
    let mut v0_sites = Vec::with_capacity(geometry.sites.len());
    let mut bx_sites = Vec::with_capacity(geometry.sites.len());
    let mut by_sites = Vec::with_capacity(geometry.sites.len());
    let mut bz_sites = Vec::with_capacity(geometry.sites.len());
    for site in &geometry.sites {
        let (v0, bz) = normalize_site_potential(&site.id, &site.spins)?;
        let zero = v0.iter().map(zero_channel).collect::<Vec<_>>();
        v0_sites.push(MuffinTinFieldV2 {
            site_id: site.id.clone(),
            channels: v0,
        });
        bx_sites.push(MuffinTinFieldV2 {
            site_id: site.id.clone(),
            channels: zero.clone(),
        });
        by_sites.push(MuffinTinFieldV2 {
            site_id: site.id.clone(),
            channels: zero,
        });
        bz_sites.push(MuffinTinFieldV2 {
            site_id: site.id.clone(),
            channels: bz,
        });
    }

    let v0_interstitial = InterstitialFieldV2 {
        coefficients: interstitial
            .coefficients
            .iter()
            .map(convert_fourier)
            .collect(),
    };
    let zero_interstitial = InterstitialFieldV2 {
        coefficients: interstitial
            .coefficients
            .iter()
            .map(|coefficient| FourierCoefficientV2 {
                g: coefficient.g,
                value: Complex64V2 {
                    real: 0.0,
                    imaginary: 0.0,
                },
            })
            .collect(),
    };
    Ok(PotentialV2 {
        unit: FieldUnitV2::Hartree,
        representation: FieldRepresentationV2::MaskedOperator,
        angular_basis: meta.potential_convention.angular_basis,
        basis_hints: interstitial.basis_hints,
        v0: RegionalFieldV2 {
            muffin_tins: v0_sites,
            interstitial: v0_interstitial,
        },
        bx: RegionalFieldV2 {
            muffin_tins: bx_sites,
            interstitial: zero_interstitial.clone(),
        },
        by: RegionalFieldV2 {
            muffin_tins: by_sites,
            interstitial: zero_interstitial.clone(),
        },
        bz: RegionalFieldV2 {
            muffin_tins: bz_sites,
            interstitial: zero_interstitial,
        },
    })
}

fn normalize_site_potential(
    site_id: &str,
    spins: &[SiteSpinV1],
) -> Result<(Vec<SphericalChannelV2>, Vec<SphericalChannelV2>), ValidationError> {
    let scalar = spins.iter().find(|spin| spin.spin == SpinTagV1::Scalar);
    let up = spins.iter().find(|spin| spin.spin == SpinTagV1::Up);
    let down = spins.iter().find(|spin| spin.spin == SpinTagV1::Down);
    match (scalar, up, down) {
        (Some(scalar), None, None) => {
            let v0: Vec<_> = scalar
                .potential_channels
                .iter()
                .map(convert_channel)
                .collect();
            let bz = v0.iter().map(zero_channel).collect();
            Ok((v0, bz))
        }
        (None, Some(up), Some(down)) => {
            if up.mesh != down.mesh {
                return Err(ValidationError::LayoutMismatch {
                    path: format!("geometry.sites[{site_id}].spins[down].mesh"),
                    reference: format!("geometry.sites[{site_id}].spins[up].mesh"),
                });
            }
            combine_collinear_channels(site_id, up, down)
        }
        _ => Err(ValidationError::InvalidValue {
            path: format!("geometry.sites[{site_id}].spins"),
            expected: "scalar or the exact up/down pair".to_owned(),
            actual: spins
                .iter()
                .map(|spin| format!("{:?}", spin.spin))
                .collect::<Vec<_>>()
                .join(","),
        }),
    }
}

fn combine_collinear_channels(
    site_id: &str,
    up: &SiteSpinV1,
    down: &SiteSpinV1,
) -> Result<(Vec<SphericalChannelV2>, Vec<SphericalChannelV2>), ValidationError> {
    let down_by_lm: BTreeMap<_, _> = down
        .potential_channels
        .iter()
        .map(|channel| ((channel.l, channel.m), channel))
        .collect();
    let up_layout: BTreeSet<_> = up
        .potential_channels
        .iter()
        .map(|channel| (channel.l, channel.m))
        .collect();
    if up_layout != down_by_lm.keys().copied().collect() {
        return Err(ValidationError::LayoutMismatch {
            path: format!("geometry.sites[{site_id}].spins[down].potential_channels"),
            reference: format!("geometry.sites[{site_id}].spins[up].potential_channels"),
        });
    }

    let mut v0 = Vec::with_capacity(up.potential_channels.len());
    let mut bz = Vec::with_capacity(up.potential_channels.len());
    for up_channel in &up.potential_channels {
        let down_channel = down_by_lm[&(up_channel.l, up_channel.m)];
        v0.push(combine_channel(up_channel, down_channel, 0.5, 0.5));
        bz.push(combine_channel(up_channel, down_channel, 0.5, -0.5));
    }
    Ok((v0, bz))
}

fn combine_channel(
    up: &PotentialChannelV1,
    down: &PotentialChannelV1,
    up_scale: f64,
    down_scale: f64,
) -> SphericalChannelV2 {
    let imaginary = if up.imaginary.is_empty() && down.imaginary.is_empty() {
        Vec::new()
    } else {
        (0..up.real.len())
            .map(|index| {
                up_scale * up.imaginary.get(index).copied().unwrap_or(0.0)
                    + down_scale * down.imaginary.get(index).copied().unwrap_or(0.0)
            })
            .collect()
    };
    SphericalChannelV2 {
        l: up.l,
        m: up.m,
        real: up
            .real
            .iter()
            .zip(&down.real)
            .map(|(&up, &down)| up_scale * up + down_scale * down)
            .collect(),
        imaginary,
    }
}

fn convert_channel(channel: &PotentialChannelV1) -> SphericalChannelV2 {
    SphericalChannelV2 {
        l: channel.l,
        m: channel.m,
        real: channel.real.clone(),
        imaginary: channel.imaginary.clone(),
    }
}

fn zero_channel(channel: &SphericalChannelV2) -> SphericalChannelV2 {
    SphericalChannelV2 {
        l: channel.l,
        m: channel.m,
        real: vec![0.0; channel.real.len()],
        imaginary: if channel.imaginary.is_empty() {
            Vec::new()
        } else {
            vec![0.0; channel.imaginary.len()]
        },
    }
}

fn convert_fourier(coefficient: &FourierCoefficientV1) -> FourierCoefficientV2 {
    FourierCoefficientV2 {
        g: coefficient.g,
        value: convert_complex(coefficient.value),
    }
}

const fn convert_complex(value: Complex64V1) -> Complex64V2 {
    Complex64V2 {
        real: value.real,
        imaginary: value.imaginary,
    }
}
