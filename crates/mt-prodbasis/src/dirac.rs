//! Parallel Dirac product IR.
//!
//! Scalar [`crate::ProductRadialId`] is unchanged. Dirac one-particle factors
//! are labelled by signed [`Kappa`] and occupy one site mesh with required
//! physical reduced $P$ and $Q$. There is no hidden $cQ$ / speed-of-light
//! scaling in this IR.
//!
//! Scalar charge density uses two structurally distinct muffin-tin sectors:
//! large-large PP with $\Omega_\kappa$, and small-small QQ with
//! $\Omega_{-\kappa}$. There is no PQ/QP variant. Raw radial samples are
//! $P_i P_j/r$ or $Q_i Q_j/r$ after a shared one-particle factor from
//! $\int(P^2+Q^2)\,dr$. Sectors are not summed before angular contraction.
//!
//! Angular matrix elements reuse `muffintin_core`: Condon--Shortley
//! Clebsch--Gordan $\Omega_{\kappa\mu}$ and the SPEX Gaunt
//! $\int Y_{l_1 m_1}^* Y_{LM} Y_{l_3 m_3}^*$, with the ket conversion
//! $Y_{lm}=(-1)^m Y_{l,-m}^*$. The stored complex-harmonic density
//! coefficient at auxiliary $(L,M)$ is
//! $(-1)^M\langle\Omega|Y_{L,-M}|\Omega\rangle$, not the matrix element
//! of $Y_{LM}$ placed in slot $M$. PP evaluates that reduction with
//! $\Omega_\kappa$ on both bra and ket; QQ uses $\Omega_{-\kappa}$ on
//! both. That is the repository complex-harmonic magnetic-phase
//! convention.

use crate::{
    AuxiliaryIrError, AuxiliaryLayout, AuxiliaryPartition, ChannelSpectrum, CompiledAuxiliaryBasis,
    CoupledChannel, ProductOrbitalKind, RawInterstitialPairSupport, TransferQ,
};
use muffintin_core::{ExponentialMesh, Kappa, RelativisticChannel, TwiceMu};
use muffintin_envelope::Provenance;
use num_complex::Complex64;
use std::collections::{BTreeSet, HashSet};
use thiserror::Error;

/// Scalar-charge muffin-tin radial-product sector.
///
/// The type cannot represent a PQ or QP cross term.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DiracChargeSector {
    /// Large-large PP, contracted with $\Omega_\kappa$.
    LargeLarge,
    /// Small-small QQ, contracted with $\Omega_{-\kappa}$.
    SmallSmall,
}

impl DiracChargeSector {
    /// Spinor spherical-harmonic channel of this sector.
    ///
    /// PP keeps $\Omega_\kappa$. QQ uses the same $\mu$ with $-\kappa$.
    pub const fn omega_channel(self, channel: RelativisticChannel) -> RelativisticChannel {
        match self {
            Self::LargeLarge => channel,
            Self::SmallSmall => channel.opposite_kappa(),
        }
    }

    /// Orbital $l$ of $\Omega$ in this sector for one-particle $\kappa$.
    pub const fn orbital_l(self, kappa: Kappa) -> u32 {
        match self {
            Self::LargeLarge => kappa.large_l(),
            Self::SmallSmall => kappa.small_l(),
        }
    }
}

/// Site-local Dirac radial factor. This is not a scalar [`crate::ProductRadialId`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DiracRadialId {
    pub site: usize,
    pub kind: ProductOrbitalKind,
    pub kappa: Kappa,
    pub n: usize,
}

/// Physical reduced Dirac samples $P$ and $Q$ on one mesh. Both are required.
#[derive(Clone, Debug, PartialEq)]
pub struct DiracRadialSamples {
    pub large: Vec<f64>,
    pub small: Vec<f64>,
}

/// Normalization used by every raw product containing one Dirac radial.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DiracRadialNormalization {
    /// Compute $\int_{0}^{R_{\mathrm{MT}}}(P^2+Q^2)\,dr$ on the stored mesh.
    OnMesh,
    /// Use a caller-retained all-space norm without renormalizing the MT prefix.
    Explicit(f64),
}

/// One valence or core Dirac radial on a site mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct DiracRadial {
    pub kappa: Kappa,
    pub n: usize,
    pub samples: DiracRadialSamples,
    pub normalization: DiracRadialNormalization,
}

/// Dirac radials belonging to one muffin-tin site.
#[derive(Clone, Debug, PartialEq)]
pub struct DiracSiteRadialSet {
    pub mesh: ExponentialMesh,
    pub valence: Vec<DiracRadial>,
    pub cores: Vec<DiracRadial>,
}

/// Method-neutral Dirac product input. It does not own a compiled LAPW basis.
#[derive(Clone, Debug, PartialEq)]
pub struct DiracProductSource {
    pub partition: AuxiliaryPartition,
    pub radials: Vec<DiracSiteRadialSet>,
    pub q: TransferQ,
    pub interstitial_pair_support: RawInterstitialPairSupport,
    pub provenance: Provenance,
}

impl DiracProductSource {
    /// Construct after checking sites, meshes, required $P$/$Q$, and pair support.
    pub fn new(
        partition: AuxiliaryPartition,
        radials: Vec<DiracSiteRadialSet>,
        q: TransferQ,
        interstitial_pair_support: RawInterstitialPairSupport,
        provenance: Provenance,
    ) -> Result<Self, DiracProductError> {
        let source = Self {
            partition,
            radials,
            q,
            interstitial_pair_support,
            provenance,
        };
        source.validate()?;
        Ok(source)
    }

    /// Check site counts, unique $(\kappa,n)$, finite equal-length $P$ and $Q$,
    /// and raw pair-support identity.
    pub fn validate(&self) -> Result<(), DiracProductError> {
        if self.radials.len() != self.partition.site_count() {
            return Err(DiracProductError::SiteCount {
                expected: self.partition.site_count(),
                actual: self.radials.len(),
            });
        }
        self.interstitial_pair_support.validate()?;
        if self.interstitial_pair_support.q != self.q {
            return Err(DiracProductError::PairSupportTransferQ);
        }
        for (site, radials) in self.radials.iter().enumerate() {
            let expected = radials.mesh.len();
            let mut seen = HashSet::new();
            for (kind, functions) in [
                (ProductOrbitalKind::Valence, radials.valence.as_slice()),
                (ProductOrbitalKind::Core, radials.cores.as_slice()),
            ] {
                for function in functions {
                    if !seen.insert((kind, function.kappa, function.n)) {
                        return Err(DiracProductError::DuplicateDiracRadial {
                            site,
                            kind,
                            kappa: function.kappa.get(),
                            n: function.n,
                        });
                    }
                    if function.samples.large.len() != expected {
                        return Err(DiracProductError::MeshLength {
                            site,
                            expected,
                            actual: function.samples.large.len(),
                        });
                    }
                    if function.samples.small.len() != function.samples.large.len() {
                        return Err(DiracProductError::UnequalPqLength {
                            site,
                            kappa: function.kappa.get(),
                            n: function.n,
                            large: function.samples.large.len(),
                            small: function.samples.small.len(),
                        });
                    }
                    let nonfinite = function
                        .samples
                        .large
                        .iter()
                        .chain(&function.samples.small)
                        .any(|value| !value.is_finite());
                    if nonfinite {
                        return Err(DiracProductError::NonFiniteRadial {
                            site,
                            kind,
                            kappa: function.kappa.get(),
                            n: function.n,
                        });
                    }
                    if let DiracRadialNormalization::Explicit(value) = function.normalization
                        && (!value.is_finite() || value <= 0.0)
                    {
                        return Err(DiracProductError::InvalidExplicitNormalization {
                            site,
                            kind,
                            kappa: function.kappa.get(),
                            n: function.n,
                            value,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Radial identified by [`DiracRadialId`], if present on this source.
    pub fn find_radial(&self, id: DiracRadialId) -> Option<&DiracRadial> {
        let site = self.radials.get(id.site)?;
        let pool = match id.kind {
            ProductOrbitalKind::Valence => site.valence.as_slice(),
            ProductOrbitalKind::Core => site.cores.as_slice(),
        };
        pool.iter()
            .find(|radial| radial.kappa == id.kappa && radial.n == id.n)
    }
}

/// q-aware Dirac muffin-tin product of two radial factors in one charge sector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiracPairChannel {
    pub q: TransferQ,
    pub left: DiracRadialId,
    pub right: DiracRadialId,
    pub coupled_l: u32,
    pub sector: DiracChargeSector,
}

/// Untruncated Dirac muffin-tin radial product in one PP or QQ sector.
///
/// `samples` follow $P_i P_j/r$ or $Q_i Q_j/r$ after the shared
/// $\sqrt{\|P_i\|^2+\|Q_i\|^2}$ one-particle factor. Sectors stay separate.
#[derive(Clone, Debug, PartialEq)]
pub struct DiracRawRadialProduct {
    pub channel: DiracPairChannel,
    pub samples: Vec<f64>,
}

/// Untruncated Dirac product space: separate PP/QQ radials, union overlap spectra,
/// and raw pair-G support.
///
/// Overlap spectra are formed from the ordered union of PP and QQ radial
/// products at each $(site, L)$. Retained Löwdin transforms live on the
/// compiled scalar-charge auxiliary, not on this raw space. Sectors stay
/// unmerged in `radial_products`.
#[derive(Clone, Debug, PartialEq)]
pub struct DiracRawProductSpace {
    pub partition: AuxiliaryPartition,
    pub q: TransferQ,
    pub radial_products: Vec<DiracRawRadialProduct>,
    pub channels: Vec<CoupledChannel>,
    pub overlap_spectra: Vec<ChannelSpectrum>,
    pub interstitial_pair_support: RawInterstitialPairSupport,
    pub provenance: Provenance,
}

impl DiracRawProductSpace {
    /// Construct after rejecting internally inconsistent pair support and channels.
    pub fn new(
        partition: AuxiliaryPartition,
        q: TransferQ,
        radial_products: Vec<DiracRawRadialProduct>,
        channels: Vec<CoupledChannel>,
        interstitial_pair_support: RawInterstitialPairSupport,
        provenance: Provenance,
    ) -> Result<Self, DiracProductError> {
        let raw = Self {
            partition,
            q,
            radial_products,
            channels,
            overlap_spectra: Vec::new(),
            interstitial_pair_support,
            provenance,
        };
        raw.validate_internal()?;
        Ok(raw)
    }

    /// Overlap spectrum for one site and coupled $L$, if present.
    pub fn spectrum(&self, site: usize, l: u32) -> Option<&ChannelSpectrum> {
        self.overlap_spectra
            .iter()
            .find(|spectrum| spectrum.site == site && spectrum.l == l)
    }

    /// Reject cross-site pairs, duplicate unordered products, duplicate
    /// spectra, duplicate $(site,L,M,n)$ channels, and an inconsistent pair support.
    ///
    /// Unordered radial identity is `(canonical(left,right), sector, L)`.
    /// Transfer $q$ is space context and is not part of that identity.
    pub fn validate_internal(&self) -> Result<(), DiracProductError> {
        self.interstitial_pair_support.validate()?;
        if self.interstitial_pair_support.q != self.q {
            return Err(DiracProductError::PairSupportTransferQ);
        }
        let mut products = HashSet::new();
        for product in &self.radial_products {
            if product.channel.q != self.q {
                return Err(DiracProductError::PairSupportTransferQ);
            }
            if product.channel.left.site != product.channel.right.site {
                return Err(DiracProductError::CrossSiteRawProduct {
                    left_site: product.channel.left.site,
                    right_site: product.channel.right.site,
                });
            }
            if product.samples.iter().any(|value| !value.is_finite()) {
                return Err(DiracProductError::NonFiniteProduct {
                    site: product.channel.left.site,
                    coupled_l: product.channel.coupled_l,
                });
            }
            if !products.insert(canonical_raw_identity(product.channel)) {
                return Err(DiracProductError::DuplicateRawRadialProduct {
                    left: product.channel.left,
                    right: product.channel.right,
                    sector: product.channel.sector,
                    coupled_l: product.channel.coupled_l,
                });
            }
        }
        let mut spectra = BTreeSet::new();
        for spectrum in &self.overlap_spectra {
            if !spectra.insert((spectrum.site, spectrum.l)) {
                return Err(DiracProductError::DuplicateChannelSpectrum {
                    site: spectrum.site,
                    l: spectrum.l,
                });
            }
        }
        let mut channels = BTreeSet::new();
        for channel in &self.channels {
            if !channels.insert((channel.site, channel.l, channel.m, channel.radial_index)) {
                return Err(DiracProductError::DuplicateCoupledChannel {
                    site: channel.site,
                    l: channel.l,
                    m: channel.m,
                    radial_index: channel.radial_index,
                });
            }
        }
        Ok(())
    }

    /// Exact partition, $q$, pair support, and signed-$\kappa$ orbital identity.
    pub fn validate_against_source(
        &self,
        source: &DiracProductSource,
    ) -> Result<(), DiracProductError> {
        self.validate_internal()?;
        source.validate()?;
        if self.q != source.q
            || self.interstitial_pair_support.q != source.q
            || source.interstitial_pair_support.q != source.q
        {
            return Err(DiracProductError::PairSupportTransferQ);
        }
        if self.partition != source.partition {
            return Err(DiracProductError::PartitionMismatch);
        }
        if self.interstitial_pair_support != source.interstitial_pair_support {
            return Err(DiracProductError::InterstitialPairSupportMismatch);
        }
        for product in &self.radial_products {
            require_source_id(source, product.channel.left)?;
            require_source_id(source, product.channel.right)?;
            let site = product.channel.left.site;
            let expected = source.radials[site].mesh.len();
            if product.samples.len() != expected {
                return Err(DiracProductError::MeshLength {
                    site,
                    expected,
                    actual: product.samples.len(),
                });
            }
        }
        Ok(())
    }
}

fn require_source_id(
    source: &DiracProductSource,
    id: DiracRadialId,
) -> Result<(), DiracProductError> {
    source
        .find_radial(id)
        .map(|_| ())
        .ok_or(DiracProductError::UnknownDiracOrbital {
            site: id.site,
            kind: id.kind,
            kappa: id.kappa.get(),
            n: id.n,
        })
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct CanonicalRawIdentity {
    first: (usize, u8, i32, usize),
    second: (usize, u8, i32, usize),
    sector: DiracChargeSector,
    coupled_l: u32,
}

fn radial_identity(id: DiracRadialId) -> (usize, u8, i32, usize) {
    let kind = match id.kind {
        ProductOrbitalKind::Valence => 0,
        ProductOrbitalKind::Core => 1,
    };
    (id.site, kind, id.kappa.get(), id.n)
}

fn canonical_raw_identity(channel: DiracPairChannel) -> CanonicalRawIdentity {
    let left = radial_identity(channel.left);
    let right = radial_identity(channel.right);
    let (first, second) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    CanonicalRawIdentity {
        first,
        second,
        sector: channel.sector,
        coupled_l: channel.coupled_l,
    }
}

/// Dirac muffin-tin pair labelled by signed $\kappa$ and $2\mu$.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiracMtPairSpec {
    pub left: DiracRadialId,
    pub left_twice_mu: TwiceMu,
    pub right: DiracRadialId,
    pub right_twice_mu: TwiceMu,
}

/// Dirac muffin-tin vertex on an existing scalar-charge auxiliary layout.
#[derive(Clone, Debug, PartialEq)]
pub struct DiracPairVertex {
    layout: AuxiliaryLayout,
    pair: DiracMtPairSpec,
    coefficients: Vec<Complex64>,
    provenance: Provenance,
}

impl DiracPairVertex {
    /// Construct after checking coefficient length against the auxiliary layout.
    pub fn new(
        layout: AuxiliaryLayout,
        pair: DiracMtPairSpec,
        coefficients: Vec<Complex64>,
        provenance: Provenance,
    ) -> Result<Self, DiracProductError> {
        if coefficients.len() != layout.dimension() {
            return Err(DiracProductError::VertexDimension {
                actual: coefficients.len(),
                expected: layout.dimension(),
            });
        }
        Ok(Self {
            layout,
            pair,
            coefficients,
            provenance,
        })
    }

    /// Construct from a compiled auxiliary's layout and provenance.
    pub fn from_auxiliary(
        auxiliary: &CompiledAuxiliaryBasis,
        pair: DiracMtPairSpec,
        coefficients: Vec<Complex64>,
    ) -> Result<Self, DiracProductError> {
        Self::new(
            auxiliary.layout(),
            pair,
            coefficients,
            auxiliary.provenance.clone(),
        )
    }

    /// Exact auxiliary layout ($q$, regions, split).
    pub const fn layout(&self) -> &AuxiliaryLayout {
        &self.layout
    }

    /// Dirac muffin-tin pair identity.
    pub const fn pair(&self) -> DiracMtPairSpec {
        self.pair
    }

    /// Combined coefficients in the compiled auxiliary order.
    pub fn coefficients(&self) -> &[Complex64] {
        &self.coefficients
    }

    /// Provenance copied from the compiled auxiliary basis.
    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

/// Dirac product-space construction or validation error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum DiracProductError {
    #[error("expected {expected} Dirac product sites, got {actual}")]
    SiteCount { expected: usize, actual: usize },
    #[error("site {site} Dirac radial mesh has {actual} samples, expected {expected}")]
    MeshLength {
        site: usize,
        expected: usize,
        actual: usize,
    },
    #[error("site {site} kappa={kappa} n={n} has P length {large} and Q length {small}")]
    UnequalPqLength {
        site: usize,
        kappa: i32,
        n: usize,
        large: usize,
        small: usize,
    },
    #[error("site {site} Dirac radial ({kind:?}, kappa={kappa}, n={n}) is not finite")]
    NonFiniteRadial {
        site: usize,
        kind: ProductOrbitalKind,
        kappa: i32,
        n: usize,
    },
    #[error(
        "site {site} Dirac radial ({kind:?}, kappa={kappa}, n={n}) has invalid explicit normalization {value}"
    )]
    InvalidExplicitNormalization {
        site: usize,
        kind: ProductOrbitalKind,
        kappa: i32,
        n: usize,
        value: f64,
    },
    #[error("duplicate Dirac radial (site {site}, {kind:?}, kappa={kappa}, n={n})")]
    DuplicateDiracRadial {
        site: usize,
        kind: ProductOrbitalKind,
        kappa: i32,
        n: usize,
    },
    #[error("Dirac raw interstitial pair support transfer q does not match the product q")]
    PairSupportTransferQ,
    #[error("Dirac source and raw interstitial pair support must be identical, including order")]
    InterstitialPairSupportMismatch,
    #[error("Dirac source and raw product space must share the same partition")]
    PartitionMismatch,
    #[error("orbital ({kind:?}, kappa={kappa}, n={n}) is not on Dirac site {site}")]
    UnknownDiracOrbital {
        site: usize,
        kind: ProductOrbitalKind,
        kappa: i32,
        n: usize,
    },
    #[error("duplicate Dirac coupled channel (site {site}, L={l}, M={m}, n={radial_index})")]
    DuplicateCoupledChannel {
        site: usize,
        l: u32,
        m: i32,
        radial_index: usize,
    },
    #[error("duplicate Dirac overlap spectrum for site {site} and L={l}")]
    DuplicateChannelSpectrum { site: usize, l: u32 },
    #[error("Dirac radial product at site {site} L={coupled_l} is not finite")]
    NonFiniteProduct { site: usize, coupled_l: u32 },
    #[error(
        "Dirac raw muffin-tin product left site {left_site} differs from right site {right_site}"
    )]
    CrossSiteRawProduct { left_site: usize, right_site: usize },
    #[error("duplicate Dirac raw radial product {left:?} / {right:?} L={coupled_l} {sector:?}")]
    DuplicateRawRadialProduct {
        left: DiracRadialId,
        right: DiracRadialId,
        sector: DiracChargeSector,
        coupled_l: u32,
    },
    #[error("Dirac pair vertex has {actual} coefficients, expected {expected}")]
    VertexDimension { actual: usize, expected: usize },
    #[error(transparent)]
    Product(#[from] AuxiliaryIrError),
}
