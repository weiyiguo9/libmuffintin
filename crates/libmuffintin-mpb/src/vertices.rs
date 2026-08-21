//! Pair vertices onto a SPEX mixed product basis.

use crate::MpbError;
use crate::construct::require_matching_context;
use libmuffintin_core::{InverseBohr, gaunt};
use libmuffintin_envelope::site_translation_phase;
use libmuffintin_product::{
    CompiledAuxiliaryBasis, InterstitialPairSpec, MtPairSpec, PairVertex, PairVertexSpec,
    ProductOrbitalKind, ProductRadial, ProductRadialId, ProductSource, RawProductSpace,
};
use num_complex::Complex64;

/// Expand an explicit MT and/or interstitial pair onto the auxiliary basis.
///
/// Muffin-tin coefficients are Gaunt-weighted radial overlaps times
/// $\exp(+i q\cdot R_a)$. Interstitial coefficients are
/// $A\Theta_I(G_{\mathrm{aux}}-G_{\mathrm{wrap}}-G_{\mathrm{rel}})$ using the partition step
/// function. The interstitial G label must exist on the raw pair support.
/// Missing spec arms stay zero. This is not a Coulomb kernel.
pub fn pair_vertex(
    source: &ProductSource,
    raw: &RawProductSpace,
    auxiliary: &CompiledAuxiliaryBasis,
    spec: PairVertexSpec,
) -> Result<PairVertex, MpbError> {
    require_matching_context(source, raw, auxiliary)?;
    let pair = spec.pair_identity().ok_or(MpbError::EmptyPairSpec)?;
    let mut coefficients = vec![Complex64::default(); auxiliary.dimension()];
    if let Some(mt) = spec.muffin_tin {
        fill_muffin_tin(source, raw, auxiliary, mt, &mut coefficients)?;
    }
    if let Some(interstitial) = spec.interstitial {
        fill_interstitial(raw, auxiliary, interstitial, &mut coefficients)?;
    }
    Ok(PairVertex::from_auxiliary(auxiliary, pair, coefficients)?)
}

fn fill_muffin_tin(
    source: &ProductSource,
    raw: &RawProductSpace,
    auxiliary: &CompiledAuxiliaryBasis,
    spec: MtPairSpec,
    coefficients: &mut [Complex64],
) -> Result<(), MpbError> {
    if spec.left.site != spec.right.site {
        return Err(MpbError::CrossSitePair);
    }
    if spec.left_m.unsigned_abs() > spec.left.l {
        return Err(MpbError::MagneticQuantumNumber {
            l: spec.left.l,
            m: spec.left_m,
        });
    }
    if spec.right_m.unsigned_abs() > spec.right.l {
        return Err(MpbError::MagneticQuantumNumber {
            l: spec.right.l,
            m: spec.right_m,
        });
    }
    let site = spec.left.site;
    let radials = source.radials.get(site).ok_or(MpbError::UnknownOrbital {
        site,
        kind: spec.left.kind,
        l: spec.left.l,
        n: spec.left.n,
        spin: spec.left.spin,
    })?;
    find_radial(radials, spec.left)?;
    find_radial(radials, spec.right)?;
    if !raw.radial_products.iter().any(|product| {
        product.channel.left.site == site
            && pair_matches(
                product.channel.left,
                product.channel.right,
                spec.left,
                spec.right,
            )
    }) {
        return Err(MpbError::UnknownMtPair {
            left: spec.left,
            right: spec.right,
        });
    }
    let mesh = auxiliary.site_mesh(site).ok_or(MpbError::UnknownOrbital {
        site,
        kind: spec.left.kind,
        l: spec.left.l,
        n: spec.left.n,
        spin: spec.left.spin,
    })?;
    let position = source.partition.sites()[site].position;
    let phase = site_translation_phase(source.q.cartesian, position);
    let m = spec.right_m - spec.left_m;
    let block = auxiliary
        .require_mixed_product()?
        .sites
        .iter()
        .find(|block| block.site == site)
        .ok_or(MpbError::UnknownOrbital {
            site,
            kind: spec.left.kind,
            l: spec.left.l,
            n: spec.left.n,
            spin: spec.left.spin,
        })?;
    for mode in &block.modes {
        if m.unsigned_abs() > mode.l {
            continue;
        }
        let Some(product) = raw.radial_products.iter().find(|product| {
            product.channel.coupled_l == mode.l
                && product.channel.left.site == site
                && pair_matches(
                    product.channel.left,
                    product.channel.right,
                    spec.left,
                    spec.right,
                )
        }) else {
            continue;
        };
        let integrand = product
            .samples
            .iter()
            .zip(&mode.radial)
            .map(|(sample, mode)| sample * mode)
            .collect::<Vec<_>>();
        let radial_overlap = mesh.integrate(&integrand)?;
        let angular = gaunt(
            spec.left.l,
            spec.right.l,
            mode.l,
            spec.left_m,
            spec.right_m,
            m,
        );
        if let Some(index) = auxiliary.mt_index(site, mode.l, m, mode.n) {
            coefficients[index] = phase * Complex64::new(angular * radial_overlap, 0.0);
        }
    }
    Ok(())
}

fn fill_interstitial(
    raw: &RawProductSpace,
    auxiliary: &CompiledAuxiliaryBasis,
    spec: InterstitialPairSpec,
    coefficients: &mut [Complex64],
) -> Result<(), MpbError> {
    if raw
        .interstitial_pair_support
        .find(&spec.g_relative)
        .is_none()
    {
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
        coefficients[offset + local] = spec.amplitude * theta;
    }
    Ok(())
}

fn pair_matches(
    stored_left: ProductRadialId,
    stored_right: ProductRadialId,
    left: ProductRadialId,
    right: ProductRadialId,
) -> bool {
    (stored_left == left && stored_right == right) || (stored_left == right && stored_right == left)
}

fn find_radial(
    radials: &libmuffintin_product::SiteRadialSet,
    id: ProductRadialId,
) -> Result<&ProductRadial, MpbError> {
    let pool = match id.kind {
        ProductOrbitalKind::Valence => radials.valence.as_slice(),
        ProductOrbitalKind::Core => radials.cores.as_slice(),
    };
    pool.iter()
        .find(|radial| radial.l == id.l && radial.n == id.n && radial.spin == id.spin)
        .ok_or(MpbError::UnknownOrbital {
            site: id.site,
            kind: id.kind,
            l: id.l,
            n: id.n,
            spin: id.spin,
        })
}
