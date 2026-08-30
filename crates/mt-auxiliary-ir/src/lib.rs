//! Historical-method-name-free product-space IR.
//!
//! v0.2 stores a non-overlapping muffin-tin plus interstitial partition, raw
//! muffin-tin products, capability-supplied raw interstitial orbital-pair
//! reciprocal support, a retained auxiliary basis, and pair vertices. The
//! auxiliary payload is a typed mixed-product or interpolation-point variant.
//! There is no MPB `TOL`, ISDF threshold, Coulomb assembler, or trait family.
//! Raw pair support is not the MPB auxiliary $|q+G|$ set. Dirac muffin-tin
//! products live in a parallel IR ([`DiracProductSource`]) and do not extend
//! scalar [`ProductRadialId`] with $\kappa$.

#![forbid(unsafe_code)]

mod auxiliary;
mod dirac;
mod pair_layout;
mod partition;
mod raw;
mod source;
mod vertex;

pub use auxiliary::{
    AuxiliaryInterstitialSupport, AuxiliaryInterstitialWave, AuxiliaryLayout, AuxiliaryRegion,
    AuxiliaryRepresentation, CompiledAuxiliaryBasis, CutoffKind, CutoffRecord,
    InterpolationAuxiliaryPoint, InterpolationPointAuxiliary, InterpolationRegion,
    MixedProductAuxiliary, MtAuxiliaryMode, SiteAuxiliaryBlock, sort_interpolation_points,
};
pub use dirac::{
    DiracChargeSector, DiracMtPairSpec, DiracPairChannel, DiracPairVertex, DiracProductError,
    DiracProductSource, DiracRadial, DiracRadialId, DiracRadialSamples, DiracRawProductSpace,
    DiracRawRadialProduct, DiracSiteRadialSet,
};
pub use pair_layout::PairColumnLayout;
pub use partition::{AuxiliaryPartition, PartitionSite};
pub use raw::{
    ChannelSpectrum, CoupledChannel, PairChannel, RawInterstitialPairComponent,
    RawInterstitialPairSupport, RawProductSpace, RawRadialProduct, sort_raw_pair_components,
};
pub use source::{
    AuxiliarySource, ProductOrbitalKind, ProductRadial, ProductRadialId, RadialSamples,
    SiteRadialSet, TransferQ,
};
pub use vertex::{
    InterstitialPairSpec, MtPairSpec, OrbitalPair, PairOrbital, PairVertex, PairVertexSpec,
};

use muffintin_core::StepFunctionError;
use muffintin_sphere::RadialIntegralError;
use thiserror::Error;

/// Product-space construction or validation error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum AuxiliaryIrError {
    #[error("expected {expected} product sites, got {actual}")]
    SiteCount { expected: usize, actual: usize },
    #[error("site {site} radial mesh has {actual} samples, expected {expected}")]
    MeshLength {
        site: usize,
        expected: usize,
        actual: usize,
    },
    #[error("site {site} radial function ({kind:?}, l={l}, n={n}) is not finite")]
    NonFiniteRadial {
        site: usize,
        kind: ProductOrbitalKind,
        l: u32,
        n: usize,
    },
    #[error("transfer-q component must be finite")]
    NonFiniteTransferQ,
    #[error("raw interstitial pair support transfer q does not match the product q")]
    PairSupportTransferQ,
    #[error("duplicate raw interstitial pair label {index:?}")]
    DuplicatePairComponent { index: [i32; 3] },
    #[error("raw interstitial pair G label contains a non-finite Cartesian component")]
    NonFinitePairComponent,
    #[error("storage length overflow for dimensions {dimensions:?}")]
    DimensionOverflow { dimensions: Vec<usize> },
    #[error("core orbital index {index} is outside n_orb={n_orb}")]
    InvalidCoreOrbital { index: usize, n_orb: usize },
    #[error("duplicate overlap spectrum for site {site} and L={l}")]
    DuplicateChannelSpectrum { site: usize, l: u32 },
    #[error("duplicate coupled channel (site {site}, L={l}, M={m}, n={radial_index})")]
    DuplicateCoupledChannel {
        site: usize,
        l: u32,
        m: i32,
        radial_index: usize,
    },
    #[error("source and raw interstitial pair support must be identical, including order")]
    InterstitialPairSupportMismatch,
    #[error("expected {expected} auxiliary site blocks, got {actual}")]
    AuxiliarySiteCount { expected: usize, actual: usize },
    #[error("auxiliary site block {found} does not match partition site {expected}")]
    AuxiliarySiteIdentity { expected: usize, found: usize },
    #[error("site {site} auxiliary mesh does not match the product source")]
    AuxiliaryMeshMismatch { site: usize },
    #[error("site {site} L={l} n={n} radial length {actual} does not match mesh {expected}")]
    AuxiliaryModeLength {
        site: usize,
        l: u32,
        n: usize,
        expected: usize,
        actual: usize,
    },
    #[error("duplicate auxiliary muffin-tin mode (site {site}, L={l}, n={n})")]
    DuplicateAuxiliaryMode { site: usize, l: u32, n: usize },
    #[error("auxiliary interstitial wave {index} is kinematically inconsistent")]
    AuxiliaryWaveKinematics { index: usize },
    #[error("duplicate auxiliary interstitial G label {index:?}")]
    DuplicateAuxiliaryWave { index: [i32; 3] },
    #[error("auxiliary interstitial wave {index} exceeds the recorded g_cut")]
    AuxiliaryWaveCutoff { index: usize },
    #[error("auxiliary interstitial waves are not in |G| then G-index order")]
    AuxiliaryWaveOrder,
    #[error("auxiliary interstitial support transfer q does not match the compiled q")]
    AuxiliarySupportTransferQ,
    #[error(
        "pair vertex has {actual} coefficients, expected mt {mt} plus interstitial {interstitial}"
    )]
    PairVertexDimension {
        actual: usize,
        mt: usize,
        interstitial: usize,
    },
    #[error("compiled auxiliary basis is interpolation points, not mixed-product")]
    ExpectedMixedProduct,
    #[error("compiled auxiliary basis is mixed-product, not interpolation points")]
    ExpectedInterpolationPoints,
    #[error("duplicate interpolation point id {0}")]
    DuplicateInterpolationPoint(usize),
    #[error("interpolation point {0} has a non-finite coordinate or weight")]
    NonFiniteInterpolationPoint(usize),
    #[error("interpolation point {0} has a negative quadrature weight")]
    NegativeInterpolationWeight(usize),
    #[error("interpolation auxiliary has no strictly positive quadrature weight")]
    NoPositiveInterpolationWeight,
    #[error("interpolation point region site {site} is outside the partition")]
    InterpolationPointSite { site: usize },
    #[error("interpolation auxiliary has no points")]
    EmptyInterpolationPoints,
    #[error("interpolation points are not in muffin-tin/interstitial/uniform id order")]
    InterpolationPointOrder,
    #[error(transparent)]
    Geometry(#[from] StepFunctionError),
    #[error(transparent)]
    RadialIntegral(#[from] RadialIntegralError),
}
