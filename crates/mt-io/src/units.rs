use serde::{Deserialize, Serialize};

/// Canonical serialized length unit.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LengthUnit {
    Bohr,
}

/// Canonical serialized reciprocal-length unit.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum InverseLengthUnit {
    #[serde(rename = "bohr^-1")]
    BohrInverse,
}

/// Canonical serialized energy unit.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EnergyUnit {
    Hartree,
}

/// Canonical serialized volume unit.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum VolumeUnit {
    #[serde(rename = "bohr^3")]
    Bohr3,
}
