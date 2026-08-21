//! Canonical real-space quadrature grids for muffin-tin methods.
//!
//! Coordinates and volume weights use the strong Bohr units from `libmuffintin-core`.
//! Point order is deterministic: atom grids use radial-shell then angular-rule
//! order, uniform grids use lexicographic fractional-cell order, and composite
//! grids use atom index followed by the interstitial region.

#![forbid(unsafe_code)]

mod angular;
mod cell;
mod grids;
#[cfg(feature = "rstsr")]
mod rstsr;

pub use angular::{AngularGrid, AngularPoint};
pub use cell::Cell;
pub use grids::{
    AtomGrid, CompositeGrid, Grid, GridError, GridPoint, InterstitialGrid, RegionTag, UniformGrid,
};
#[cfg(feature = "rstsr")]
pub use rstsr::RstsrGridExt;
