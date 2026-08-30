//! One-particle product input: partition, radials, transfer q, and pair support.

use crate::{AuxiliaryIrError, AuxiliaryPartition, RawInterstitialPairSupport};
use muffintin_basis::Provenance;
use muffintin_core::{ExponentialMesh, GVector, InverseBohr};
use muffintin_sphere::RadialComponents;

/// Valence or selected core origin of one muffin-tin radial factor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProductOrbitalKind {
    /// Valence linearization function or local orbital.
    Valence,
    /// Selected bound core radial function.
    Core,
}

/// Site-local radial factor used to enumerate muffin-tin products.
///
/// This is not the representation-neutral orbital-pair identity on
/// [`crate::PairVertex`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProductRadialId {
    pub site: usize,
    pub kind: ProductOrbitalKind,
    pub l: u32,
    pub n: usize,
    pub spin: u8,
}

/// Canonical transfer momentum and reciprocal Umklapp shift.
///
/// `cartesian` is the q used in $|q+G|$ tests and site phases.
/// `umklapp` is the reciprocal vector subtracted to reach that q.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransferQ {
    pub cartesian: [InverseBohr; 3],
    pub umklapp: GVector,
}

impl TransferQ {
    /// Unwrapped Cartesian q with a zero Umklapp vector.
    pub fn from_cartesian(cartesian: [InverseBohr; 3]) -> Result<Self, AuxiliaryIrError> {
        if cartesian
            .iter()
            .any(|component| !component.get().is_finite())
        {
            return Err(AuxiliaryIrError::NonFiniteTransferQ);
        }
        Ok(Self {
            cartesian,
            umklapp: GVector {
                index: [0; 3],
                cartesian: [InverseBohr(0.0); 3],
                norm: InverseBohr(0.0),
            },
        })
    }

    /// Canonical q `input - G` with an explicit Umklapp record.
    pub fn fold_by_reciprocal_vector(
        input: [InverseBohr; 3],
        umklapp: GVector,
    ) -> Result<Self, AuxiliaryIrError> {
        if input
            .iter()
            .chain(umklapp.cartesian.iter())
            .any(|component| !component.get().is_finite())
        {
            return Err(AuxiliaryIrError::NonFiniteTransferQ);
        }
        let cartesian = std::array::from_fn(|axis| {
            InverseBohr(input[axis].get() - umklapp.cartesian[axis].get())
        });
        Ok(Self { cartesian, umklapp })
    }

    /// Cartesian norm $|q|$.
    pub fn norm(self) -> InverseBohr {
        InverseBohr(
            self.cartesian
                .iter()
                .map(|component| component.get().powi(2))
                .sum::<f64>()
                .sqrt(),
        )
    }
}

/// Host radial samples in the reduced convention $p = r u$.
#[derive(Clone, Debug, PartialEq)]
pub struct RadialSamples {
    pub large: Vec<f64>,
    pub small: Option<Vec<f64>>,
}

impl RadialComponents for RadialSamples {
    fn large_component(&self) -> &[f64] {
        &self.large
    }

    fn small_component(&self) -> Option<&[f64]> {
        self.small.as_deref()
    }
}

/// One valence or core radial function on a site mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct ProductRadial {
    pub l: u32,
    pub n: usize,
    pub spin: u8,
    pub samples: RadialSamples,
}

/// Radial functions belonging to one muffin-tin site.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteRadialSet {
    pub mesh: ExponentialMesh,
    pub valence: Vec<ProductRadial>,
    pub cores: Vec<ProductRadial>,
}

/// Minimal product-construction input.
///
/// This does not own a one-particle [`muffintin_basis::CompiledBasis`].
/// Cell volume comes from [`AuxiliaryPartition::interstitial`].
/// `interstitial_pair_support` is the finite raw orbital-pair reciprocal
/// support supplied by the one-particle/pair capability, before MPB
/// auxiliary $g_{\mathrm{cut}}$.
#[derive(Clone, Debug, PartialEq)]
pub struct AuxiliarySource {
    pub partition: AuxiliaryPartition,
    pub radials: Vec<SiteRadialSet>,
    pub q: TransferQ,
    pub interstitial_pair_support: RawInterstitialPairSupport,
    pub provenance: Provenance,
}

impl AuxiliarySource {
    /// Construct after checking site counts, mesh lengths, finite samples,
    /// and raw pair-support identity.
    pub fn new(
        partition: AuxiliaryPartition,
        radials: Vec<SiteRadialSet>,
        q: TransferQ,
        interstitial_pair_support: RawInterstitialPairSupport,
        provenance: Provenance,
    ) -> Result<Self, AuxiliaryIrError> {
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

    /// Check site counts, mesh lengths, finite samples, and pair support.
    pub fn validate(&self) -> Result<(), AuxiliaryIrError> {
        if self.radials.len() != self.partition.site_count() {
            return Err(AuxiliaryIrError::SiteCount {
                expected: self.partition.site_count(),
                actual: self.radials.len(),
            });
        }
        self.interstitial_pair_support.validate()?;
        if self.interstitial_pair_support.q != self.q {
            return Err(AuxiliaryIrError::PairSupportTransferQ);
        }
        for (site, radials) in self.radials.iter().enumerate() {
            let expected = radials.mesh.len();
            for (kind, functions) in [
                (ProductOrbitalKind::Valence, radials.valence.as_slice()),
                (ProductOrbitalKind::Core, radials.cores.as_slice()),
            ] {
                for function in functions {
                    if function.samples.large.len() != expected {
                        return Err(AuxiliaryIrError::MeshLength {
                            site,
                            expected,
                            actual: function.samples.large.len(),
                        });
                    }
                    if let Some(small) = &function.samples.small {
                        if small.len() != expected {
                            return Err(AuxiliaryIrError::MeshLength {
                                site,
                                expected,
                                actual: small.len(),
                            });
                        }
                    }
                    let nonfinite = function
                        .samples
                        .large
                        .iter()
                        .any(|value| !value.is_finite())
                        || function
                            .samples
                            .small
                            .as_ref()
                            .is_some_and(|small| small.iter().any(|value| !value.is_finite()));
                    if nonfinite {
                        return Err(AuxiliaryIrError::NonFiniteRadial {
                            site,
                            kind,
                            l: function.l,
                            n: function.n,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}
