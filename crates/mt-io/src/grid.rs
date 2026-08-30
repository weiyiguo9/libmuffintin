use muffintin_core::{Grid, RegionTag};
use serde::{Deserialize, Serialize};

use crate::error::{IoError, ValidationError, finite, nonempty};
use crate::units::{LengthUnitV1, VolumeUnitV1};

/// Stable discriminator for materialized integration grids.
pub const GRID_ARTIFACT_FORMAT: &str = "libmuffintin-grid-artifact";
/// Only grid-artifact schema version currently supported.
pub const GRID_ARTIFACT_VERSION: u32 = 1;

/// A materialized Cartesian quadrature grid, independently versioned from snapshots.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GridArtifactV1 {
    pub format: String,
    pub version: u32,
    pub point_unit: LengthUnitV1,
    pub weight_unit: VolumeUnitV1,
    pub points: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    /// One nonempty producer-defined region label per point.
    pub region_tags: Vec<String>,
}

impl GridArtifactV1 {
    /// Construct a grid with the required independent V1 header.
    pub fn new(
        point_unit: LengthUnitV1,
        weight_unit: VolumeUnitV1,
        points: Vec<[f64; 3]>,
        weights: Vec<f64>,
        region_tags: Vec<String>,
    ) -> Self {
        Self {
            format: GRID_ARTIFACT_FORMAT.to_owned(),
            version: GRID_ARTIFACT_VERSION,
            point_unit,
            weight_unit,
            points,
            weights,
            region_tags,
        }
    }

    /// Materialize a `libmuffintin-grid` point sequence without changing its order.
    pub fn from_grid(grid: &impl Grid) -> Self {
        let points = grid
            .points()
            .iter()
            .map(|point| point.position.map(|coordinate| coordinate.0))
            .collect();
        let weights = grid.points().iter().map(|point| point.weight.0).collect();
        let region_tags = grid
            .points()
            .iter()
            .map(|point| match point.region {
                RegionTag::Atom(index) => format!("atom:{index}"),
                RegionTag::Interstitial => "interstitial".to_owned(),
                RegionTag::Uniform => "uniform".to_owned(),
            })
            .collect();
        Self::new(
            LengthUnitV1::Bohr,
            VolumeUnitV1::Bohr3,
            points,
            weights,
            region_tags,
        )
    }

    /// Check the header, array lengths, and numerical samples.
    pub fn validate(&self) -> Result<(), IoError> {
        if self.format != GRID_ARTIFACT_FORMAT {
            return Err(IoError::InvalidFormat {
                expected: GRID_ARTIFACT_FORMAT,
                found: self.format.clone(),
            });
        }
        if self.version != GRID_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion {
                format: GRID_ARTIFACT_FORMAT,
                supported: GRID_ARTIFACT_VERSION,
                found: self.version,
            });
        }
        let count = self.points.len();
        if self.weights.len() != count {
            return Err(ValidationError::LengthMismatch {
                path: "weights".to_owned(),
                expected: count,
                actual: self.weights.len(),
            }
            .into());
        }
        if self.region_tags.len() != count {
            return Err(ValidationError::LengthMismatch {
                path: "region_tags".to_owned(),
                expected: count,
                actual: self.region_tags.len(),
            }
            .into());
        }
        for (point_index, point) in self.points.iter().enumerate() {
            for (axis, &value) in point.iter().enumerate() {
                finite(format!("points[{point_index}][{axis}]"), value)?;
            }
        }
        for (index, &weight) in self.weights.iter().enumerate() {
            finite(format!("weights[{index}]"), weight)?;
        }
        for (index, tag) in self.region_tags.iter().enumerate() {
            nonempty(format!("region_tags[{index}]"), tag)?;
        }
        Ok(())
    }
}

/// Serialize a validated grid artifact as deterministic pretty TOML.
pub fn grid_artifact_to_toml(grid: &GridArtifactV1) -> Result<String, IoError> {
    grid.validate()?;
    let mut text = toml::to_string_pretty(grid)?;
    if !text.ends_with('\n') {
        text.push('\n');
    }
    Ok(text)
}

/// Parse and validate an independently versioned grid artifact.
pub fn grid_artifact_from_toml(text: &str) -> Result<GridArtifactV1, IoError> {
    let grid: GridArtifactV1 = toml::from_str(text)?;
    grid.validate()?;
    Ok(grid)
}
