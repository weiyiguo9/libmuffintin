//! Untruncated muffin-tin products, coupled channels, and raw pair support.

use crate::{ProductError, ProductPartition, ProductRadialId, ProductSource, TransferQ};
use libmuffintin_basis::Provenance;
use libmuffintin_core::GVector;
use std::collections::BTreeSet;

/// q-aware muffin-tin product of two radial factors, before spectral cutoff.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairChannel {
    pub q: TransferQ,
    pub left: ProductRadialId,
    pub right: ProductRadialId,
    pub coupled_l: u32,
}

/// One untruncated muffin-tin radial product.
///
/// `samples` follow the SPEX mixed-basis convention
/// $(p_i p_j + Q_i Q_j)/r$ after one-particle-norm scaling.
#[derive(Clone, Debug, PartialEq)]
pub struct RawRadialProduct {
    pub channel: PairChannel,
    pub samples: Vec<f64>,
}

/// One $(L,M,n)$ copy of a site radial-product index, in SPEX flatten order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoupledChannel {
    pub site: usize,
    pub l: u32,
    pub m: i32,
    pub radial_index: usize,
}

/// Untruncated overlap eigensystem of one $(site, L)$ channel.
///
/// Eigenvectors are real and column-major.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelSpectrum {
    pub site: usize,
    pub l: u32,
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<f64>,
}

/// One relative reciprocal label of a raw interstitial orbital-pair product.
///
/// This is not an MPB auxiliary plane wave and is not filtered by
/// `product_g_max`. It includes any per-column $k-q$ wrap supplied by the
/// orbital-pair capability, but excludes the global transfer wrap stored on
/// [`TransferQ`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawInterstitialPairComponent {
    pub g_relative: GVector,
}

/// Finite raw interstitial orbital-pair reciprocal support.
///
/// The one-particle/pair capability supplies this list before any MPB
/// auxiliary $|q+G|$ cutoff. It is not an untruncated copy of the mixed-basis
/// plane-wave set.
#[derive(Clone, Debug, PartialEq)]
pub struct RawInterstitialPairSupport {
    pub q: TransferQ,
    pub components: Vec<RawInterstitialPairComponent>,
}

impl RawInterstitialPairSupport {
    /// Empty support at `q`.
    pub fn empty(q: TransferQ) -> Self {
        Self {
            q,
            components: Vec::new(),
        }
    }

    /// Construct after rejecting duplicate labels and non-finite Cartesian G.
    pub fn from_components(
        q: TransferQ,
        components: Vec<RawInterstitialPairComponent>,
    ) -> Result<Self, ProductError> {
        let support = Self { q, components };
        support.validate()?;
        Ok(support)
    }

    /// Same labels at a different transfer q, preserving order.
    pub fn with_q(&self, q: TransferQ) -> Result<Self, ProductError> {
        Self::from_components(q, self.components.clone())
    }

    /// Position of an exact G-label match, including Cartesian values.
    pub fn find(&self, g: &GVector) -> Option<usize> {
        self.components
            .iter()
            .position(|component| &component.g_relative == g)
    }

    /// Check uniqueness, finite Cartesian labels, and a matching q.
    pub fn validate(&self) -> Result<(), ProductError> {
        let mut seen = BTreeSet::new();
        for component in &self.components {
            if component
                .g_relative
                .cartesian
                .iter()
                .any(|value| !value.get().is_finite())
                || !component.g_relative.norm.get().is_finite()
            {
                return Err(ProductError::NonFinitePairComponent);
            }
            if !seen.insert(component.g_relative.index) {
                return Err(ProductError::DuplicatePairComponent {
                    index: component.g_relative.index,
                });
            }
        }
        Ok(())
    }
}

/// Untruncated product space: MT radials/spectra and raw pair-G support.
///
/// Interstitial content here is capability-supplied orbital-pair reciprocal
/// support. It is not the MPB auxiliary $|q+G|$ plane-wave set.
#[derive(Clone, Debug, PartialEq)]
pub struct RawProductSpace {
    pub partition: ProductPartition,
    pub q: TransferQ,
    pub radial_products: Vec<RawRadialProduct>,
    pub channels: Vec<CoupledChannel>,
    pub overlap_spectra: Vec<ChannelSpectrum>,
    pub interstitial_pair_support: RawInterstitialPairSupport,
    pub provenance: Provenance,
}

impl RawProductSpace {
    /// Number of untruncated muffin-tin radial products.
    pub fn radial_product_count(&self) -> usize {
        self.radial_products.len()
    }

    /// Overlap spectrum for one site and coupled $L$, if present.
    pub fn spectrum(&self, site: usize, l: u32) -> Option<&ChannelSpectrum> {
        self.overlap_spectra
            .iter()
            .find(|spectrum| spectrum.site == site && spectrum.l == l)
    }

    /// Reject duplicate spectra/channels and an internally inconsistent pair support.
    pub fn validate_internal(&self) -> Result<(), ProductError> {
        self.interstitial_pair_support.validate()?;
        if self.interstitial_pair_support.q != self.q {
            return Err(ProductError::PairSupportTransferQ);
        }
        let mut spectra = BTreeSet::new();
        for spectrum in &self.overlap_spectra {
            if !spectra.insert((spectrum.site, spectrum.l)) {
                return Err(ProductError::DuplicateChannelSpectrum {
                    site: spectrum.site,
                    l: spectrum.l,
                });
            }
        }
        let mut channels = BTreeSet::new();
        for channel in &self.channels {
            if !channels.insert((channel.site, channel.l, channel.m, channel.radial_index)) {
                return Err(ProductError::DuplicateCoupledChannel {
                    site: channel.site,
                    l: channel.l,
                    m: channel.m,
                    radial_index: channel.radial_index,
                });
            }
        }
        Ok(())
    }

    /// Exact partition, q, and raw pair-support identity with a product source.
    pub fn validate_against_source(&self, source: &ProductSource) -> Result<(), ProductError> {
        self.validate_internal()?;
        if self.q != source.q
            || self.interstitial_pair_support.q != source.q
            || source.interstitial_pair_support.q != source.q
        {
            return Err(ProductError::PairSupportTransferQ);
        }
        if self.interstitial_pair_support != source.interstitial_pair_support {
            return Err(ProductError::InterstitialPairSupportMismatch);
        }
        Ok(())
    }
}
