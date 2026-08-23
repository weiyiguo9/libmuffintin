//! MPB auxiliary $|q+G|$ interstitial plane-wave support.

use crate::MpbError;
use muffintin_auxiliary_ir::{AuxiliaryInterstitialSupport, AuxiliaryInterstitialWave, TransferQ};
use muffintin_core::{InverseBohr, ReciprocalLattice};

/// SPEX mixed-basis interstitial PW set: $|q+G|\le g_{\mathrm{cut}}$, ordered
/// by $|G|$ then $G$ index.
///
/// Membership is the SPEX `mixedbasis.f` test `rdum<=gcutm**2` on `kvec+g`.
/// This is the MPB auxiliary plane-wave basis, not raw orbital-pair
/// reciprocal support.
pub fn auxiliary_interstitial_support(
    lattice: &ReciprocalLattice,
    q: TransferQ,
    g_cut: InverseBohr,
) -> Result<AuxiliaryInterstitialSupport, MpbError> {
    if !g_cut.get().is_finite() || g_cut.get() < 0.0 {
        return Err(MpbError::InvalidGCutoff(g_cut.get()));
    }
    let bound = InverseBohr(g_cut.get() + q.norm().get());
    let cutoff_squared = g_cut.get() * g_cut.get();
    let tolerance = 64.0 * f64::EPSILON * cutoff_squared.max(1.0);
    let mut waves = Vec::new();
    for g in lattice.enumerate(bound)? {
        let q_plus_g = std::array::from_fn(|axis| {
            InverseBohr(q.cartesian[axis].get() + g.cartesian[axis].get())
        });
        let qg_squared = q_plus_g
            .iter()
            .map(|component| component.get().powi(2))
            .sum::<f64>();
        if qg_squared <= cutoff_squared + tolerance {
            waves.push(AuxiliaryInterstitialWave {
                g,
                q_plus_g,
                q_plus_g_norm: InverseBohr(qg_squared.sqrt()),
            });
        }
    }
    Ok(AuxiliaryInterstitialSupport { q, g_cut, waves })
}
