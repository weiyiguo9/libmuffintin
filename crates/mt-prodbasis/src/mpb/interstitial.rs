//! MPB auxiliary $|q+G|$ interstitial plane-wave support and raw-pair $\Theta_I$.

use crate::mpb::MpbError;
use crate::{
    AuxiliaryInterstitialSupport, AuxiliaryInterstitialWave, AuxiliaryLayout,
    CompiledAuxiliaryBasis, InterstitialPairSpec, RawInterstitialPairSupport, TransferQ,
};
use muffintin_core::{InverseBohr, ReciprocalLattice};
use num_complex::Complex64;
use std::collections::HashMap;

/// Precomputed interstitial step-function rows for every raw pair momentum.
#[derive(Clone, Debug, PartialEq)]
pub struct InterstitialThetaTable {
    layout: AuxiliaryLayout,
    rows: HashMap<[i32; 3], Vec<Complex64>>,
}

impl InterstitialThetaTable {
    pub fn new(
        pair_support: &RawInterstitialPairSupport,
        auxiliary: &CompiledAuxiliaryBasis,
    ) -> Result<Self, MpbError> {
        let wrap = auxiliary.q.umklapp;
        let waves = &auxiliary.require_mixed_product()?.interstitial.waves;
        let mut rows = HashMap::with_capacity(pair_support.components.len());
        for component in &pair_support.components {
            let mut row = Vec::with_capacity(waves.len());
            for wave in waves {
                let argument = std::array::from_fn(|axis| {
                    InverseBohr(
                        wave.g.cartesian[axis].get()
                            - wrap.cartesian[axis].get()
                            - component.g_relative.cartesian[axis].get(),
                    )
                });
                row.push(auxiliary.partition.interstitial().coefficient(argument)?);
            }
            rows.insert(component.g_relative.index, row);
        }
        Ok(Self {
            layout: auxiliary.layout(),
            rows,
        })
    }

    pub(crate) fn add(
        &self,
        auxiliary: &CompiledAuxiliaryBasis,
        spec: InterstitialPairSpec,
        coefficients: &mut [Complex64],
    ) -> Result<(), MpbError> {
        if self.layout != auxiliary.layout() {
            return Err(MpbError::InterstitialThetaContext);
        }
        let row =
            self.rows
                .get(&spec.g_relative.index)
                .ok_or(MpbError::UnknownInterstitialPair {
                    g: spec.g_relative.index,
                })?;
        let offset = auxiliary.mt_dimension();
        for (coefficient, &theta) in coefficients[offset..].iter_mut().zip(row) {
            *coefficient += spec.amplitude * theta;
        }
        Ok(())
    }
}

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

/// Add $A\Theta_I(G_{\mathrm{aux}}-G_{\mathrm{wrap}}-G_{\mathrm{rel}})$ onto a vertex.
///
/// `g_relative` must exist on the raw pair support. Global
/// [`TransferQ::umklapp`] enters the $\Theta_I$ argument only.
pub(crate) fn add_raw_support_theta_i(
    pair_support: &RawInterstitialPairSupport,
    auxiliary: &CompiledAuxiliaryBasis,
    spec: InterstitialPairSpec,
    coefficients: &mut [Complex64],
) -> Result<(), MpbError> {
    if pair_support.find(&spec.g_relative).is_none() {
        return Err(MpbError::UnknownInterstitialPair {
            g: spec.g_relative.index,
        });
    }
    let wrap = auxiliary.q.umklapp;
    let offset = auxiliary.mt_dimension();
    let payload = auxiliary.require_mixed_product()?;
    for (local, wave) in payload.interstitial.waves.iter().enumerate() {
        let argument = std::array::from_fn(|axis| {
            InverseBohr(
                wave.g.cartesian[axis].get()
                    - wrap.cartesian[axis].get()
                    - spec.g_relative.cartesian[axis].get(),
            )
        });
        let theta = auxiliary.partition.interstitial().coefficient(argument)?;
        coefficients[offset + local] += spec.amplitude * theta;
    }
    Ok(())
}
