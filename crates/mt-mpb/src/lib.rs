//! SPEX-compatible mixed product basis over the product-space IR.
//!
//! Enumeration, triangle/parity coupling, overlap spectra, the L=0 constant
//! function, and $|q+G|$ interstitial auxiliary support follow `mixedbasis.f`.
//! `TOL` is constructor policy recorded on the retained auxiliary basis, not
//! on [`muffintin_auxiliary_ir::RawProductSpace`]. Raw interstitial orbital-pair
//! reciprocal support is supplied by [`muffintin_auxiliary_ir::ProductSource`]
//! and is not the MPB auxiliary plane-wave set.

#![forbid(unsafe_code)]

mod construct;
mod interstitial;
mod vertices;

pub use construct::{apply_overlap_cutoff, spex_mixed_product_basis};
pub use interstitial::auxiliary_interstitial_support;
pub use vertices::pair_vertex;

use muffintin_auxiliary_ir::{ProductError, ProductRadialId};
use muffintin_core::{LatticeError, MeshError, StepFunctionError};
use muffintin_operators::OperatorError;
use thiserror::Error;

/// Default SPEX `MBASIS` `TOL` (`mixedbasis.f:106`).
pub const DEFAULT_TOLERANCE: f64 = 1.0e-4;

/// Mixed-product construction error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum MpbError {
    #[error(transparent)]
    Product(#[from] ProductError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    Lattice(#[from] LatticeError),
    #[error(transparent)]
    StepFunction(#[from] StepFunctionError),
    #[error("overlap tolerance must be finite and nonnegative, got {0}")]
    InvalidTolerance(f64),
    #[error("nspin factor must be finite and positive, got {0}")]
    InvalidNspinFactor(f64),
    #[error("product G-cutoff must be finite and nonnegative, got {0}")]
    InvalidGCutoff(f64),
    #[error("no muffin-tin products for site {site} and L={l}")]
    EmptyChannel { site: usize, l: u32 },
    #[error("overlap cutoff left no modes for site {site} and L={l}")]
    EmptyRetainedChannel { site: usize, l: u32 },
    #[error("orbital ({kind:?}, l={l}, n={n}, spin={spin}) is not on site {site}")]
    UnknownOrbital {
        site: usize,
        kind: muffintin_auxiliary_ir::ProductOrbitalKind,
        l: u32,
        n: usize,
        spin: u8,
    },
    #[error("magnetic quantum number m={m} is outside [-{l}, {l}]")]
    MagneticQuantumNumber { l: u32, m: i32 },
    #[error("pair vertex left and right must occupy the same muffin-tin site")]
    CrossSitePair,
    #[error("source, raw product space, and auxiliary basis must share the same transfer q")]
    TransferQMismatch,
    #[error("source, raw product space, and auxiliary basis must share the same partition")]
    PartitionMismatch,
    #[error("source and raw interstitial pair support must be identical, including order")]
    InterstitialPairSupportMismatch,
    #[error("requested muffin-tin pair is absent from the raw product space")]
    UnknownMtPair {
        left: ProductRadialId,
        right: ProductRadialId,
    },
    #[error("requested interstitial pair G={g:?} is absent from the raw pair support")]
    UnknownInterstitialPair { g: [i32; 3] },
    #[error("pair-vertex spec must request a muffin-tin and/or interstitial expansion")]
    EmptyPairSpec,
}

#[cfg(test)]
mod tests {
    use super::construct::retain_overlap_eigenvalue;
    use super::*;

    #[test]
    fn overlap_eigenvalues_equal_to_the_threshold_are_kept() {
        assert!(retain_overlap_eigenvalue(
            DEFAULT_TOLERANCE,
            DEFAULT_TOLERANCE
        ));
        assert!(!retain_overlap_eigenvalue(0.0, 0.0));
        assert!(!retain_overlap_eigenvalue(
            DEFAULT_TOLERANCE * 0.5,
            DEFAULT_TOLERANCE
        ));
    }
}
