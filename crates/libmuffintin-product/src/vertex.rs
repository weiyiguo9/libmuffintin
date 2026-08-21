//! Pair vertices onto a compiled auxiliary basis.

use crate::{ProductError, ProductRadialId, TransferQ};
use libmuffintin_basis::Provenance;
use libmuffintin_core::GVector;
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
/// The pair density is `amplitude * Θ_I(r) exp(+i (q + G_rel)·r)`.
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
/// [`crate::CompiledAuxiliaryBasis::regions`]. Interpolation-point
/// auxiliaries use muffin-tin-tagged points then interstitial/uniform
/// points. This is not a Coulomb matrix element. Fields are private so a
/// caller cannot forge dimensions that panic on [`Self::mt`] /
/// [`Self::interstitial`].
#[derive(Clone, Debug, PartialEq)]
pub struct PairVertex {
    q: TransferQ,
    pair: OrbitalPair,
    mt_dimension: usize,
    interstitial_dimension: usize,
    coefficients: Vec<Complex64>,
    provenance: Provenance,
}

impl PairVertex {
    /// Construct after checking that the coefficient vector matches the
    /// validated muffin-tin then interstitial split.
    pub fn new(
        q: TransferQ,
        pair: OrbitalPair,
        mt_dimension: usize,
        interstitial_dimension: usize,
        coefficients: Vec<Complex64>,
        provenance: Provenance,
    ) -> Result<Self, ProductError> {
        let expected = mt_dimension.checked_add(interstitial_dimension).ok_or(
            ProductError::PairVertexDimension {
                actual: coefficients.len(),
                mt: mt_dimension,
                interstitial: interstitial_dimension,
            },
        )?;
        if coefficients.len() != expected {
            return Err(ProductError::PairVertexDimension {
                actual: coefficients.len(),
                mt: mt_dimension,
                interstitial: interstitial_dimension,
            });
        }
        Ok(Self {
            q,
            pair,
            mt_dimension,
            interstitial_dimension,
            coefficients,
            provenance,
        })
    }

    /// Canonical transfer q stored on the vertex.
    pub const fn q(&self) -> TransferQ {
        self.q
    }

    /// Representation-neutral pair identity, including both arms when present.
    pub const fn pair(&self) -> OrbitalPair {
        self.pair
    }

    /// Muffin-tin coefficient count.
    pub const fn mt_dimension(&self) -> usize {
        self.mt_dimension
    }

    /// Interstitial coefficient count.
    pub const fn interstitial_dimension(&self) -> usize {
        self.interstitial_dimension
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
        &self.coefficients[..self.mt_dimension]
    }

    /// Interstitial coefficient block.
    pub fn interstitial(&self) -> &[Complex64] {
        &self.coefficients[self.mt_dimension..]
    }
}
