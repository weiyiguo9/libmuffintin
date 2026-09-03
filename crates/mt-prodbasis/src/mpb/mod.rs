//! SPEX-compatible mixed product basis over the product-space IR.
//!
//! Enumeration, triangle/parity coupling, overlap spectra, the L=0 constant
//! function, and $|q+G|$ interstitial auxiliary support follow `mixedbasis.f`.
//! `TOL` is constructor policy recorded on the retained auxiliary basis, not
//! on [`crate::RawProductSpace`]. Raw interstitial orbital-pair
//! reciprocal support is supplied by [`crate::AuxiliarySource`]
//! and is not the MPB auxiliary plane-wave set. Dirac PP/QQ muffin-tin products,
//! checked vertices, overlap cutoff of the ordered PP/QQ union, and Bloch
//! pair vertices are a parallel path over
//! [`crate::DiracProductSource`]; they do not reuse scalar
//! [`crate::ProductRadialId`].

#![forbid(unsafe_code)]

mod construct;
mod dirac_construct;
mod dirac_vertices;
mod interstitial;
mod overlap;
mod vertices;

pub use construct::{apply_overlap_cutoff, spex_mixed_product_basis};
pub use dirac_construct::{
    DiracProductMode, apply_dirac_overlap_cutoff, require_matching_dirac_source_and_raw,
    untruncated_dirac_product_space,
};
pub use dirac_vertices::{
    DiracBlochVertexAccumulator, DiracMtCompiledPair, DiracMtSectorTable,
    DiracPairVertexAccumulator, DiracVertexContext, dirac_mt_pair_vertex,
    require_matching_dirac_context,
};
pub use interstitial::{InterstitialThetaTable, auxiliary_interstitial_support};
pub use vertices::{
    PairVertexAccumulator, ScalarMtCompiledPair, ScalarMtPairTable, ScalarVertexContext,
    pair_vertex,
};

use crate::{
    AuxiliaryIrError, DiracChargeSector, DiracProductError, DiracRadialId, ProductRadialId,
};
use muffintin_core::{LatticeError, MeshError, StepFunctionError};
use muffintin_operators::OperatorError;
use thiserror::Error;

/// Default SPEX `MBASIS` `TOL` (`mixedbasis.f:106`).
pub const DEFAULT_TOLERANCE: f64 = 1.0e-4;

/// Mixed-product construction error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum MpbError {
    #[error(transparent)]
    Product(#[from] AuxiliaryIrError),
    #[error(transparent)]
    Dirac(#[from] DiracProductError),
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
    #[error("core-valence product mode requires at least one selected core radial")]
    CoreValenceModeWithoutSelectedCore,
    #[error("no muffin-tin products for site {site} and L={l}")]
    EmptyChannel { site: usize, l: u32 },
    #[error("overlap cutoff left no modes for site {site} and L={l}")]
    EmptyRetainedChannel { site: usize, l: u32 },
    #[error("orbital ({kind:?}, l={l}, n={n}, spin={spin}) is not on site {site}")]
    UnknownOrbital {
        site: usize,
        kind: crate::ProductOrbitalKind,
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
    #[error("precomputed interstitial step-function table does not match the auxiliary basis")]
    InterstitialThetaContext,
    #[error("precompiled Dirac muffin-tin pair does not match the auxiliary basis")]
    CompiledDiracMtContext,
    #[error("precompiled scalar muffin-tin pair does not match the auxiliary basis")]
    CompiledScalarMtContext,
    #[error("requested muffin-tin pair is absent from the raw product space")]
    UnknownMtPair {
        left: ProductRadialId,
        right: ProductRadialId,
    },
    #[error("requested interstitial pair G={g:?} is absent from the raw pair support")]
    UnknownInterstitialPair { g: [i32; 3] },
    #[error("pair-vertex spec must request a muffin-tin and/or interstitial expansion")]
    EmptyPairSpec,
    #[error("orbital ({kind:?}, kappa={kappa}, n={n}) is not on Dirac site {site}")]
    UnknownDiracOrbital {
        site: usize,
        kind: crate::ProductOrbitalKind,
        kappa: i32,
        n: usize,
    },
    #[error("requested Dirac muffin-tin pair is absent from the {sector:?} raw product space")]
    UnknownDiracMtPair {
        left: DiracRadialId,
        right: DiracRadialId,
        sector: DiracChargeSector,
    },
    #[error("2mu={twice_mu} is outside [-{twice_j}, {twice_j}] for Dirac kappa={kappa}")]
    DiracMagneticQuantumNumber {
        kappa: i32,
        twice_mu: i64,
        twice_j: u32,
    },
    #[error("Dirac pair vertex left and right must occupy the same muffin-tin site")]
    DiracCrossSitePair,
    #[error("Dirac overlap spectrum is missing for populated site {site} L={l}")]
    MissingDiracOverlapSpectrum { site: usize, l: u32 },
    #[error("Dirac overlap spectrum for site {site} L={l} has no matching raw muffin-tin block")]
    UnmatchedDiracOverlapSpectrum { site: usize, l: u32 },
    #[error(
        "Dirac overlap spectrum for site {site} L={l} has {n_eigenvalues} eigenvalues and {n_eigenvectors} eigenvector entries, expected {n_products} union products (column-major {n_products}×{n_products})"
    )]
    DiracOverlapSpectrumDimension {
        site: usize,
        l: u32,
        n_products: usize,
        n_eigenvalues: usize,
        n_eigenvectors: usize,
    },
    #[error("Dirac Bloch vertex accumulator requires OrbitalPair::Bloch")]
    ExpectedDiracBlochPair,
}

#[cfg(test)]
mod tests {
    use super::overlap::retain_overlap_eigenvalue;
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
