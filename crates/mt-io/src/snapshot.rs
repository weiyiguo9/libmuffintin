use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{IoError, ValidationError, finite, nonempty, positive};
use crate::units::{EnergyUnitV1, InverseLengthUnitV1, LengthUnitV1};

/// Stable discriminator written at the start of every snapshot.
pub const SNAPSHOT_FORMAT: &str = "libmuffintin-snapshot";
/// Legacy V1 snapshot schema version.
pub const SNAPSHOT_VERSION: u32 = 1;

/// A complete, canonical V1 muffin-tin input snapshot.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotV1 {
    pub format: String,
    pub version: u32,
    pub meta: MetaV1,
    pub geometry: GeometryV1,
    pub interstitial: InterstitialV1,
}

impl SnapshotV1 {
    /// Construct a snapshot with the required V1 header.
    pub fn new(meta: MetaV1, geometry: GeometryV1, interstitial: InterstitialV1) -> Self {
        Self {
            format: SNAPSHOT_FORMAT.to_owned(),
            version: SNAPSHOT_VERSION,
            meta,
            geometry,
            interstitial,
        }
    }

    /// Check the header and all cross-field invariants.
    pub fn validate(&self) -> Result<(), IoError> {
        if self.format != SNAPSHOT_FORMAT {
            return Err(IoError::InvalidFormat {
                expected: SNAPSHOT_FORMAT,
                found: self.format.clone(),
            });
        }
        if self.version != SNAPSHOT_VERSION {
            return Err(IoError::UnsupportedVersion {
                format: SNAPSHOT_FORMAT,
                supported: SNAPSHOT_VERSION,
                found: self.version,
            });
        }
        self.meta.validate()?;
        self.geometry.validate()?;
        self.interstitial.validate()?;
        Ok(())
    }
}

/// Provenance and convention metadata that changes interpretation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetaV1 {
    pub title: String,
    pub producer: String,
    pub producer_version: Option<String>,
    /// Human-readable definition of the common energy reference.
    pub energy_zero: String,
    pub potential_convention: PotentialConventionV1,
    /// Deterministically ordered producer annotations.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub annotations: BTreeMap<String, String>,
}

impl MetaV1 {
    pub(crate) fn validate(&self) -> Result<(), ValidationError> {
        nonempty("meta.title", &self.title)?;
        nonempty("meta.producer", &self.producer)?;
        nonempty("meta.energy_zero", &self.energy_zero)?;
        if let Some(version) = &self.producer_version {
            nonempty("meta.producer_version", version)?;
        }
        for key in self.annotations.keys() {
            nonempty("meta.annotations key", key)?;
        }
        Ok(())
    }
}

/// Angular and radial conventions for all muffin-tin potential channels.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PotentialConventionV1 {
    pub angular_basis: AngularBasisV1,
    pub radial_quantity: PotentialRadialQuantityV1,
    pub spherical_channel: SphericalChannelConventionV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AngularBasisV1 {
    ComplexCondonShortley,
    RealTesseralCondonShortley,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PotentialRadialQuantityV1 {
    /// Samples are `V_LM(r)` in energy units, with no extra radial factor.
    Potential,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SphericalChannelConventionV1 {
    /// The `(0,0)` samples are the physical scalar entering the radial equation.
    PhysicalValue,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryV1 {
    pub lattice: LatticeV1,
    pub sites: Vec<SiteV1>,
}

impl GeometryV1 {
    fn validate(&self) -> Result<(), ValidationError> {
        self.lattice.validate()?;
        if self.sites.is_empty() {
            return Err(ValidationError::Empty {
                path: "geometry.sites".to_owned(),
            });
        }
        let mut ids = BTreeSet::new();
        for (index, site) in self.sites.iter().enumerate() {
            site.validate(index)?;
            if !ids.insert(&site.id) {
                return Err(ValidationError::Duplicate {
                    path: "geometry.sites.id".to_owned(),
                    key: site.id.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Direct primitive vectors stored by row in Cartesian coordinates.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LatticeV1 {
    pub unit: LengthUnitV1,
    pub vectors: [[f64; 3]; 3],
}

impl LatticeV1 {
    pub(crate) fn validate(&self) -> Result<(), ValidationError> {
        for (row, vector) in self.vectors.iter().enumerate() {
            for (column, &value) in vector.iter().enumerate() {
                finite(format!("geometry.lattice.vectors[{row}][{column}]"), value)?;
            }
        }
        let [a, b, c] = self.vectors;
        let determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]);
        if determinant.is_finite() && determinant > 0.0 {
            Ok(())
        } else {
            Err(ValidationError::InvalidLattice { determinant })
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SiteV1 {
    pub id: String,
    pub atomic_number: u16,
    /// Fractional direct-lattice coordinates; equivalent translated values are allowed.
    pub fractional_position: [f64; 3],
    pub muffin_tin_radius_unit: LengthUnitV1,
    pub muffin_tin_radius: f64,
    pub spins: Vec<SiteSpinV1>,
}

impl SiteV1 {
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
        positive(format!("{path}.muffin_tin_radius"), self.muffin_tin_radius)?;
        if self.spins.is_empty() {
            return Err(ValidationError::Empty {
                path: format!("{path}.spins"),
            });
        }
        let mut tags = BTreeSet::new();
        for (spin_index, spin) in self.spins.iter().enumerate() {
            spin.validate(&path, spin_index)?;
            if !tags.insert(spin.spin) {
                return Err(ValidationError::Duplicate {
                    path: format!("{path}.spins.spin"),
                    key: format!("{:?}", spin.spin),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SiteSpinV1 {
    pub spin: SpinTagV1,
    pub mesh: ExponentialMeshSpecV1,
    pub radial_equation: RadialEquationTagV1,
    pub potential_unit: EnergyUnitV1,
    pub potential_channels: Vec<PotentialChannelV1>,
    pub linearization: LinearizationV1,
}

impl SiteSpinV1 {
    fn validate(&self, site_path: &str, spin_index: usize) -> Result<(), ValidationError> {
        let path = format!("{site_path}.spins[{spin_index}]");
        self.mesh.validate(&format!("{path}.mesh"))?;
        if self.potential_channels.is_empty() {
            return Err(ValidationError::Empty {
                path: format!("{path}.potential_channels"),
            });
        }
        let mut lm = BTreeSet::new();
        for (channel_index, channel) in self.potential_channels.iter().enumerate() {
            let channel_path = format!("{path}.potential_channels[{channel_index}]");
            channel.validate(&channel_path, self.mesh.point_count)?;
            if !lm.insert((channel.l, channel.m)) {
                return Err(ValidationError::Duplicate {
                    path: format!("{path}.potential_channels"),
                    key: format!("({}, {})", channel.l, channel.m),
                });
            }
        }
        self.linearization
            .validate(&format!("{path}.linearization"))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SpinTagV1 {
    Scalar,
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RadialEquationTagV1 {
    Schroedinger,
    ScalarKoellingHarmon,
    FullyRelativisticDirac,
}

/// Serialized radial mesh identity `r_i = first * exp(i * log_increment)`.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExponentialMeshSpecV1 {
    pub radius_unit: LengthUnitV1,
    pub first: f64,
    pub log_increment: f64,
    pub point_count: usize,
    pub last: f64,
    /// Relative endpoint tolerance, scaled by `max(1, |last|, |computed|)`.
    pub consistency_tolerance: f64,
}

impl ExponentialMeshSpecV1 {
    pub(crate) fn validate(&self, path: &str) -> Result<(), ValidationError> {
        positive(format!("{path}.first"), self.first)?;
        finite(format!("{path}.log_increment"), self.log_increment)?;
        if self.log_increment == 0.0 {
            return Err(ValidationError::Zero {
                path: format!("{path}.log_increment"),
                value: self.log_increment,
            });
        }
        if self.point_count < 7 {
            return Err(ValidationError::MeshTooShort {
                path: path.to_owned(),
                points: self.point_count,
            });
        }
        positive(format!("{path}.last"), self.last)?;
        positive(
            format!("{path}.consistency_tolerance"),
            self.consistency_tolerance,
        )?;
        let intervals = (self.point_count - 1) as f64;
        let expected = self.first * (intervals * self.log_increment).exp();
        finite(format!("{path}.computed_last"), expected)?;
        let scale = expected.abs().max(self.last.abs()).max(1.0);
        if (self.last - expected).abs() <= self.consistency_tolerance * scale {
            Ok(())
        } else {
            Err(ValidationError::MeshEndpoint {
                path: path.to_owned(),
                expected,
                actual: self.last,
                tolerance: self.consistency_tolerance,
            })
        }
    }
}

/// One normalized-harmonic radial potential channel.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PotentialChannelV1 {
    pub l: u32,
    pub m: i32,
    pub real: Vec<f64>,
    /// Imaginary samples; must be empty for an explicitly real channel.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub imaginary: Vec<f64>,
}

impl PotentialChannelV1 {
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

/// Linearization and local-orbital energy parameters for one spin channel.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LinearizationV1 {
    pub energy_unit: EnergyUnitV1,
    pub linearization_energies: Vec<EnergyParameterV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub local_orbital_energies: Vec<EnergyParameterV1>,
}

impl LinearizationV1 {
    pub(crate) fn validate(&self, path: &str) -> Result<(), ValidationError> {
        let mut angular_momenta = BTreeSet::new();
        for (index, parameter) in self.linearization_energies.iter().enumerate() {
            finite(
                format!("{path}.linearization_energies[{index}].energy"),
                parameter.energy,
            )?;
            if !angular_momenta.insert(parameter.l) {
                return Err(ValidationError::Duplicate {
                    path: format!("{path}.linearization_energies.l"),
                    key: parameter.l.to_string(),
                });
            }
        }
        for (index, parameter) in self.local_orbital_energies.iter().enumerate() {
            finite(
                format!("{path}.local_orbital_energies[{index}].energy"),
                parameter.energy,
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyParameterV1 {
    pub l: u32,
    pub energy: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InterstitialV1 {
    pub coefficient_unit: EnergyUnitV1,
    pub coefficients: Vec<FourierCoefficientV1>,
    pub basis_hints: BasisHintsV1,
}

impl InterstitialV1 {
    fn validate(&self) -> Result<(), ValidationError> {
        let mut wave_vectors = BTreeSet::new();
        for (index, coefficient) in self.coefficients.iter().enumerate() {
            coefficient.validate(index)?;
            if !wave_vectors.insert(coefficient.g) {
                return Err(ValidationError::Duplicate {
                    path: "interstitial.coefficients.g".to_owned(),
                    key: format!("{:?}", coefficient.g),
                });
            }
        }
        self.basis_hints.validate("interstitial.basis_hints")
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FourierCoefficientV1 {
    /// Integer reciprocal coordinates in the lattice-dual basis.
    pub g: [i32; 3],
    pub value: Complex64V1,
}

impl FourierCoefficientV1 {
    fn validate(&self, index: usize) -> Result<(), ValidationError> {
        finite(
            format!("interstitial.coefficients[{index}].value.real"),
            self.value.real,
        )?;
        finite(
            format!("interstitial.coefficients[{index}].value.imaginary"),
            self.value.imaginary,
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Complex64V1 {
    pub real: f64,
    pub imaginary: f64,
}

/// Non-authoritative information useful when reconstructing a plane-wave basis.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BasisHintsV1 {
    pub reciprocal_length_unit: InverseLengthUnitV1,
    pub plane_wave_cutoff: Option<f64>,
    pub coefficient_cutoff: Option<f64>,
    pub normalization: FourierNormalizationV1,
    pub phase: FourierPhaseV1,
}

impl BasisHintsV1 {
    pub(crate) fn validate(&self, path: &str) -> Result<(), ValidationError> {
        if let Some(value) = self.plane_wave_cutoff {
            positive(format!("{path}.plane_wave_cutoff"), value)?;
        }
        if let Some(value) = self.coefficient_cutoff {
            positive(format!("{path}.coefficient_cutoff"), value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FourierNormalizationV1 {
    CellNormalized,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FourierPhaseV1 {
    /// `f_G = Omega^-1 integral f(r) exp(-i G.r) dr`.
    NegativeExponent,
}

/// Serialize a validated V1 snapshot as deterministic pretty TOML.
pub fn snapshot_to_toml(snapshot: &SnapshotV1) -> Result<String, IoError> {
    snapshot.validate()?;
    let mut text = toml::to_string_pretty(snapshot)?;
    if !text.ends_with('\n') {
        text.push('\n');
    }
    Ok(text)
}

/// Parse and validate a V1 snapshot.
pub fn snapshot_from_toml(text: &str) -> Result<SnapshotV1, IoError> {
    let snapshot: SnapshotV1 = toml::from_str(text)?;
    snapshot.validate()?;
    Ok(snapshot)
}
