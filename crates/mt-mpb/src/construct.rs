//! SPEX mixed-basis enumeration, spectra, and retained transforms.

use crate::overlap::{lowdin_modes, overlap_spectrum, product_channel_functions};
use crate::{MpbError, auxiliary_interstitial_support};
use muffintin_auxiliary_ir::{
    AuxiliaryRepresentation, AuxiliarySource, CompiledAuxiliaryBasis, CoupledChannel, CutoffKind,
    CutoffRecord, MixedProductAuxiliary, PairChannel, ProductOrbitalKind, ProductRadial,
    ProductRadialId, RawProductSpace, RawRadialProduct, SiteAuxiliaryBlock, SiteRadialSet,
};
use muffintin_envelope::Provenance;
use muffintin_core::{InverseBohr, ReciprocalLattice};
use muffintin_sphere::RadialComponents;
use std::collections::BTreeSet;

fn spex_provenance(cutoff: Option<&CutoffRecord>) -> Provenance {
    Provenance {
        recipe: Some(if cutoff.is_some() {
            "spex_mpb+tol".to_owned()
        } else {
            "spex_mpb".to_owned()
        }),
        reference: Some("SPEX mixedbasis.f".to_owned()),
    }
}

/// Untruncated SPEX mixed product basis: full spectra and MPB auxiliary PW.
///
/// Raw interstitial orbital-pair support is copied from [`AuxiliarySource`].
/// The auxiliary $|q+G|$ set is built separately from `lattice`, canonical
/// `q`, and `product_g_max`. `TOL` is not applied.
pub fn spex_mixed_product_basis(
    source: &AuxiliarySource,
    product_l_max: u32,
    product_g_max: InverseBohr,
    lattice: &ReciprocalLattice,
) -> Result<(RawProductSpace, CompiledAuxiliaryBasis), MpbError> {
    let raw = untruncated_product_space(source, product_l_max)?;
    let auxiliary = retained_auxiliary(&raw, source, None, lattice, product_g_max)?;
    Ok((raw, auxiliary))
}

/// Apply SPEX `TOL` after untruncated spectra already exist.
///
/// Retained eigenvalues satisfy $\lambda \ge \mathrm{tol}\times n_{\mathrm{spin}}$.
/// Interstitial auxiliary PW are reconstructed from `lattice` and
/// `product_g_max`, independently of raw pair support.
pub fn apply_overlap_cutoff(
    raw: &RawProductSpace,
    source: &AuxiliarySource,
    tolerance: f64,
    nspin_factor: f64,
    lattice: &ReciprocalLattice,
    product_g_max: InverseBohr,
) -> Result<CompiledAuxiliaryBasis, MpbError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(MpbError::InvalidTolerance(tolerance));
    }
    if !nspin_factor.is_finite() || nspin_factor <= 0.0 {
        return Err(MpbError::InvalidNspinFactor(nspin_factor));
    }
    require_matching_source_and_raw(source, raw)?;
    retained_auxiliary(
        raw,
        source,
        Some(CutoffRecord {
            kind: CutoffKind::SpectralOverlap,
            value: tolerance,
            nspin_factor,
        }),
        lattice,
        product_g_max,
    )
}

pub(crate) fn require_matching_source_and_raw(
    source: &AuxiliarySource,
    raw: &RawProductSpace,
) -> Result<(), MpbError> {
    source.validate()?;
    raw.validate_internal()?;
    if source.q != raw.q
        || source.q != source.interstitial_pair_support.q
        || raw.q != raw.interstitial_pair_support.q
    {
        return Err(MpbError::TransferQMismatch);
    }
    if source.partition != raw.partition {
        return Err(MpbError::PartitionMismatch);
    }
    if source.interstitial_pair_support != raw.interstitial_pair_support {
        return Err(MpbError::InterstitialPairSupportMismatch);
    }
    Ok(())
}

pub(crate) fn require_matching_context(
    source: &AuxiliarySource,
    raw: &RawProductSpace,
    auxiliary: &CompiledAuxiliaryBasis,
) -> Result<(), MpbError> {
    require_matching_source_and_raw(source, raw)?;
    auxiliary.validate_against_source(source)?;
    let payload = auxiliary.require_mixed_product()?;
    if auxiliary.q != source.q || payload.interstitial.q != source.q {
        return Err(MpbError::TransferQMismatch);
    }
    if auxiliary.partition != source.partition || auxiliary.partition != raw.partition {
        return Err(MpbError::PartitionMismatch);
    }
    Ok(())
}

fn untruncated_product_space(
    source: &AuxiliarySource,
    product_l_max: u32,
) -> Result<RawProductSpace, MpbError> {
    source.validate()?;
    let mut radial_products = Vec::new();
    let mut channels = Vec::new();
    let mut overlap_spectra = Vec::new();
    for (site, radials) in source.radials.iter().enumerate() {
        for l in 0..=product_l_max {
            let products = enumerate_site_channel(source, site, radials, l)?;
            if products.is_empty() {
                continue;
            }
            let functions = product_channel_functions(
                &radials.mesh,
                l,
                products.iter().map(|product| product.samples.as_slice()),
            )?;
            let spectrum = overlap_spectrum(site, l, &radials.mesh, &functions)?;
            let coupled_l = l as i32;
            for m in -coupled_l..=coupled_l {
                for radial_index in 0..products.len() {
                    channels.push(CoupledChannel {
                        site,
                        l,
                        m,
                        radial_index,
                    });
                }
            }
            radial_products.extend(products);
            overlap_spectra.push(spectrum);
        }
    }
    let raw = RawProductSpace {
        partition: source.partition.clone(),
        q: source.q,
        radial_products,
        channels,
        overlap_spectra,
        interstitial_pair_support: source.interstitial_pair_support.clone(),
        provenance: spex_provenance(None),
    };
    raw.validate_internal()?;
    Ok(raw)
}

fn enumerate_site_channel(
    source: &AuxiliarySource,
    site: usize,
    radials: &SiteRadialSet,
    l: u32,
) -> Result<Vec<RawRadialProduct>, MpbError> {
    let mut products = Vec::new();
    for spin in present_spins(radials) {
        let valence = radials
            .valence
            .iter()
            .filter(|radial| radial.spin == spin)
            .collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        for left in &valence {
            for right in &valence {
                let key = (left.l, left.n, right.l, right.n);
                let swapped = (right.l, right.n, left.l, left.n);
                if !seen.insert(key) {
                    continue;
                }
                seen.insert(swapped);
                if !allowed_coupling(l, left.l, right.l) {
                    continue;
                }
                products.push(pair_product(
                    source,
                    site,
                    radials,
                    l,
                    (ProductOrbitalKind::Valence, left),
                    (ProductOrbitalKind::Valence, right),
                )?);
            }
        }
        for core in radials.cores.iter().filter(|radial| radial.spin == spin) {
            for valence in &valence {
                if !allowed_coupling(l, core.l, valence.l) {
                    continue;
                }
                products.push(pair_product(
                    source,
                    site,
                    radials,
                    l,
                    (ProductOrbitalKind::Core, core),
                    (ProductOrbitalKind::Valence, valence),
                )?);
            }
        }
    }
    Ok(products)
}

fn present_spins(radials: &SiteRadialSet) -> Vec<u8> {
    let mut spins = BTreeSet::new();
    for radial in radials.valence.iter().chain(&radials.cores) {
        spins.insert(radial.spin);
    }
    spins.into_iter().collect()
}

fn allowed_coupling(l: u32, l1: u32, l2: u32) -> bool {
    (l + l1 + l2) % 2 == 0 && l >= l1.abs_diff(l2) && l <= l1 + l2
}

fn pair_product(
    source: &AuxiliarySource,
    site: usize,
    radials: &SiteRadialSet,
    coupled_l: u32,
    left: (ProductOrbitalKind, &ProductRadial),
    right: (ProductOrbitalKind, &ProductRadial),
) -> Result<RawRadialProduct, MpbError> {
    let (left_kind, left) = left;
    let (right_kind, right) = right;
    let radii = radials.mesh.radii();
    let mut samples = Vec::with_capacity(radii.len());
    let left_small = left.samples.small_component();
    let right_small = right.samples.small_component();
    for (index, radius) in radii.iter().enumerate() {
        let mut value = left.samples.large[index] * right.samples.large[index];
        if let (Some(left_q), Some(right_q)) = (left_small, right_small) {
            value += left_q[index] * right_q[index];
        }
        samples.push(value / radius.get());
    }
    let scale = (one_particle_norm(radials, left)? * one_particle_norm(radials, right)?).sqrt();
    if scale > 0.0 {
        for sample in &mut samples {
            *sample /= scale;
        }
    }
    Ok(RawRadialProduct {
        channel: PairChannel {
            q: source.q,
            left: ProductRadialId {
                site,
                kind: left_kind,
                l: left.l,
                n: left.n,
                spin: left.spin,
            },
            right: ProductRadialId {
                site,
                kind: right_kind,
                l: right.l,
                n: right.n,
                spin: right.spin,
            },
            coupled_l,
        },
        samples,
    })
}

fn one_particle_norm(radials: &SiteRadialSet, radial: &ProductRadial) -> Result<f64, MpbError> {
    let small = radial.samples.small_component();
    let integrand = radial
        .samples
        .large
        .iter()
        .enumerate()
        .map(|(index, large)| {
            let mut value = large * large;
            if let Some(small) = small {
                value += small[index] * small[index];
            }
            value
        })
        .collect::<Vec<_>>();
    Ok(radials.mesh.integrate(&integrand)?)
}

fn retained_auxiliary(
    raw: &RawProductSpace,
    source: &AuxiliarySource,
    cutoff: Option<CutoffRecord>,
    lattice: &ReciprocalLattice,
    product_g_max: InverseBohr,
) -> Result<CompiledAuxiliaryBasis, MpbError> {
    require_matching_source_and_raw(source, raw)?;
    let mut sites = Vec::with_capacity(source.radials.len());
    for (site, radials) in source.radials.iter().enumerate() {
        let mut channel_l = raw
            .overlap_spectra
            .iter()
            .filter(|spectrum| spectrum.site == site)
            .map(|spectrum| spectrum.l)
            .collect::<Vec<_>>();
        channel_l.sort_unstable();
        channel_l.dedup();
        let mut modes = Vec::new();
        for l in channel_l {
            let products = raw
                .radial_products
                .iter()
                .filter(|product| {
                    product.channel.left.site == site && product.channel.coupled_l == l
                })
                .cloned()
                .collect::<Vec<_>>();
            let functions = product_channel_functions(
                &radials.mesh,
                l,
                products.iter().map(|product| product.samples.as_slice()),
            )?;
            let spectrum = raw
                .spectrum(site, l)
                .ok_or(MpbError::EmptyChannel { site, l })?;
            modes.extend(lowdin_modes(
                l,
                &radials.mesh,
                &functions,
                spectrum,
                cutoff.as_ref(),
            )?);
        }
        modes.sort_by_key(|mode| (mode.l, mode.n));
        sites.push(SiteAuxiliaryBlock {
            site,
            mesh: radials.mesh.clone(),
            modes,
        });
    }
    let auxiliary = CompiledAuxiliaryBasis {
        partition: raw.partition.clone(),
        q: raw.q,
        representation: AuxiliaryRepresentation::MixedProduct(MixedProductAuxiliary {
            sites,
            interstitial: auxiliary_interstitial_support(lattice, raw.q, product_g_max)?,
            cutoff,
        }),
        provenance: spex_provenance(cutoff.as_ref()),
    };
    auxiliary.validate_against_source(source)?;
    Ok(auxiliary)
}
