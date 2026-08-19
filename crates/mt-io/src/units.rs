use serde::{Deserialize, Serialize};

/// Canonical serialized length unit.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LengthUnitV1 {
    Bohr,
}

/// Canonical serialized reciprocal-length unit.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum InverseLengthUnitV1 {
    #[serde(rename = "bohr^-1")]
    BohrInverse,
}

/// Canonical serialized energy unit.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EnergyUnitV1 {
    Hartree,
}

/// Canonical serialized volume unit.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum VolumeUnitV1 {
    #[serde(rename = "bohr^3")]
    Bohr3,
}
