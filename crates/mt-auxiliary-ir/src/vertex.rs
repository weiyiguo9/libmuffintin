//! Pair vertices onto a compiled auxiliary basis.

use crate::{
    AuxiliaryIrError, AuxiliaryLayout, CompiledAuxiliaryBasis, ProductRadialId, TransferQ,
};
use muffintin_envelope::Provenance;
use muffintin_core::GVector;
use num_complex::Complex64;

/// Representation-neutral factor of an orbital pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PairOrbital {
    /// Muffin-tin radial factor plus magnetic index.
    Radial { id: ProductRadialId, m: i32 },
    /// Interstitial plane-wave factor labelled by $G$ at the pair $q$.
    Interstitial { g: GVector },
}

/// Representation-neutral orbital-pair identity carried by [`PairVertex`].
///
/// Expansion arms on [`PairVertexSpec`] do not invent a left/right
/// decomposition from a single interstitial label. Dual-arm requests keep
/// both the muffin-tin factor pair and the raw pair-G identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OrbitalPair {
    /// Muffin-tin radial-factor pair only.
    MuffinTin {
        left: PairOrbital,
        right: PairOrbital,
    },
    /// Raw interstitial reciprocal product label only.
    Interstitial { g_relative: GVector },
    /// Both representations requested for the same pair vertex.
    Composite {
        muffin_tin: (PairOrbital, PairOrbital),
        interstitial: GVector,
    },
    /// Periodic Bloch orbital pair at one $k$, used by k-point ISDF/THC.
    Bloch {
        k_index: usize,
        left: usize,
        right: usize,
    },
}

/// Muffin-tin radial-factor pair used by the mixed-product constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtPairSpec {
    pub left: ProductRadialId,
    pub left_m: i32,
    pub right: ProductRadialId,
    pub right_m: i32,
}

/// Analytic interstitial pair expansion at canonical $q$.
///
/// The unfolded pair density is
/// $A\Theta_I(r)\exp(+i(q+G_{\mathrm{wrap}}+G_{\mathrm{rel}})\cdot r)$,
/// where $q$ is the canonical transfer and $G_{\mathrm{wrap}}$ is
/// [`TransferQ::umklapp`]. Per-column $k-q$ wrapping is already included in
/// `g_relative`; the global transfer wrap is not.
/// `g_relative` must name a component of the raw interstitial pair support.
/// Umklapp stored on [`TransferQ`] shifts the Fourier argument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterstitialPairSpec {
    pub g_relative: GVector,
    pub amplitude: Complex64,
}

/// Explicit MT and/or interstitial expansion request.
///
/// At least one arm must be present. Missing arms stay zero rather than being
/// silently invented by a LAPW adapter. The pair identity is the composite of
/// the requested arms, not an inference from one arm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairVertexSpec {
    pub muffin_tin: Option<MtPairSpec>,
    pub interstitial: Option<InterstitialPairSpec>,
}

impl PairVertexSpec {
    /// Representation-neutral identity implied by the requested arms.
    pub fn pair_identity(self) -> Option<OrbitalPair> {
        match (self.muffin_tin, self.interstitial) {
            (None, None) => None,
            (Some(mt), None) => Some(OrbitalPair::MuffinTin {
                left: PairOrbital::Radial {
                    id: mt.left,
                    m: mt.left_m,
                },
                right: PairOrbital::Radial {
                    id: mt.right,
                    m: mt.right_m,
                },
            }),
            (None, Some(interstitial)) => Some(OrbitalPair::Interstitial {
                g_relative: interstitial.g_relative,
            }),
            (Some(mt), Some(interstitial)) => Some(OrbitalPair::Composite {
                muffin_tin: (
                    PairOrbital::Radial {
                        id: mt.left,
                        m: mt.left_m,
                    },
                    PairOrbital::Radial {
                        id: mt.right,
                        m: mt.right_m,
                    },
                ),
                interstitial: interstitial.g_relative,
            }),
        }
    }
}

/// Expansion of one orbital pair onto the combined auxiliary basis.
///
/// Coefficients are muffin-tin then interstitial, matching
/// [`AuxiliaryLayout::regions`]. The layout is the exact $q$ plus region
/// sequence, not a count split that could be forged. Fields are private so a
/// caller cannot panic on [`Self::mt`] / [`Self::interstitial`].
#[derive(Clone, Debug, PartialEq)]
pub struct PairVertex {
    layout: AuxiliaryLayout,
    pair: OrbitalPair,
    coefficients: Vec<Complex64>,
    provenance: Provenance,
}

impl PairVertex {
    /// Construct from a layout after checking coefficient length.
    pub fn new(
        layout: AuxiliaryLayout,
        pair: OrbitalPair,
        coefficients: Vec<Complex64>,
        provenance: Provenance,
    ) -> Result<Self, AuxiliaryIrError> {
        if coefficients.len() != layout.dimension() {
            return Err(AuxiliaryIrError::PairVertexDimension {
                actual: coefficients.len(),
                mt: layout.mt_dimension(),
                interstitial: layout.interstitial_dimension(),
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
        pair: OrbitalPair,
        coefficients: Vec<Complex64>,
    ) -> Result<Self, AuxiliaryIrError> {
        Self::new(
            auxiliary.layout(),
            pair,
            coefficients,
            auxiliary.provenance.clone(),
        )
    }

    /// Exact auxiliary layout stored on the vertex.
    pub const fn layout(&self) -> &AuxiliaryLayout {
        &self.layout
    }

    /// Canonical transfer q stored on the vertex.
    pub const fn q(&self) -> TransferQ {
        self.layout.q()
    }

    /// Representation-neutral pair identity, including both arms when present.
    pub const fn pair(&self) -> OrbitalPair {
        self.pair
    }

    /// Muffin-tin coefficient count.
    pub const fn mt_dimension(&self) -> usize {
        self.layout.mt_dimension()
    }

    /// Interstitial coefficient count.
    pub const fn interstitial_dimension(&self) -> usize {
        self.layout.interstitial_dimension()
    }

    /// Combined coefficients in muffin-tin then interstitial order.
    pub fn coefficients(&self) -> &[Complex64] {
        &self.coefficients
    }

    /// Provenance copied from the compiled auxiliary basis.
    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// Muffin-tin coefficient block.
    pub fn mt(&self) -> &[Complex64] {
        &self.coefficients[..self.layout.mt_dimension()]
    }

    /// Interstitial coefficient block.
    pub fn interstitial(&self) -> &[Complex64] {
        &self.coefficients[self.layout.mt_dimension()..]
    }
}
