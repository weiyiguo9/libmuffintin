//! Untruncated Dirac PP/QQ muffin-tin products.

use crate::MpbError;
use muffintin_auxiliary_ir::{
    CoupledChannel, DiracChargeSector, DiracPairChannel, DiracProductSource, DiracRadial,
    DiracRadialId, DiracRawProductSpace, DiracRawRadialProduct, DiracSiteRadialSet,
    ProductOrbitalKind,
};
use muffintin_basis::Provenance;
use muffintin_core::ExponentialMesh;
use std::collections::HashSet;

/// Untruncated Dirac mixed-product space: separate PP and QQ radial products.
///
/// Each allowed scalar $L$ emits PP using $P_i P_j/r$ and QQ using $Q_i Q_j/r$
/// without merging sectors. `CoupledChannel::radial_index` is local to each
/// $(site,L)$ block (`0..products_for_this_site_L`, SPEX flatten
/// $site\to L\to M\to n$). Raw product storage stays a global list. Overlap
/// spectra and retained Löwdin transforms are not computed; those remain a
/// later seam on the existing scalar-charge auxiliary layout.
pub fn untruncated_dirac_product_space(
    source: &DiracProductSource,
    product_l_max: u32,
) -> Result<DiracRawProductSpace, MpbError> {
    source.validate()?;
    let mut radial_products = Vec::new();
    let mut channels = Vec::new();
    for (site, radials) in source.radials.iter().enumerate() {
        for l in 0..=product_l_max {
            let products = enumerate_site_channel(source, site, radials, l)?;
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
        }
    }
    Ok(DiracRawProductSpace::new(
        source.partition.clone(),
        source.q,
        radial_products,
        channels,
        source.interstitial_pair_support.clone(),
        Provenance {
            recipe: Some("dirac-mt-pp-qq".to_owned()),
            reference: None,
        },
    )?)
}

/// Exact partition, $q$, pair support, and signed-$\kappa$ identity.
pub fn require_matching_dirac_source_and_raw(
    source: &DiracProductSource,
    raw: &DiracRawProductSpace,
) -> Result<(), MpbError> {
    Ok(raw.validate_against_source(source)?)
}

fn enumerate_site_channel(
    source: &DiracProductSource,
    site: usize,
    radials: &DiracSiteRadialSet,
    l: u32,
) -> Result<Vec<DiracRawRadialProduct>, MpbError> {
    let mut products = Vec::new();
    let valence = radials.valence.iter().collect::<Vec<_>>();
    let mut seen = HashSet::new();
    for left in &valence {
        for right in &valence {
            push_pair(
                source,
                site,
                radials,
                l,
                (ProductOrbitalKind::Valence, left),
                (ProductOrbitalKind::Valence, right),
                &mut seen,
                &mut products,
            )?;
        }
    }
    for core in &radials.cores {
        for valence in &valence {
            push_pair(
                source,
                site,
                radials,
                l,
                (ProductOrbitalKind::Core, core),
                (ProductOrbitalKind::Valence, valence),
                &mut seen,
                &mut products,
            )?;
        }
    }
    Ok(products)
}

#[allow(clippy::too_many_arguments)]
fn push_pair(
    source: &DiracProductSource,
    site: usize,
    radials: &DiracSiteRadialSet,
    l: u32,
    left: (ProductOrbitalKind, &DiracRadial),
    right: (ProductOrbitalKind, &DiracRadial),
    seen: &mut HashSet<(
        i32,
        usize,
        ProductOrbitalKind,
        i32,
        usize,
        ProductOrbitalKind,
    )>,
    products: &mut Vec<DiracRawRadialProduct>,
) -> Result<(), MpbError> {
    let (left_kind, left) = left;
    let (right_kind, right) = right;
    let key = (
        left.kappa.get(),
        left.n,
        left_kind,
        right.kappa.get(),
        right.n,
        right_kind,
    );
    let swapped = (
        right.kappa.get(),
        right.n,
        right_kind,
        left.kappa.get(),
        left.n,
        left_kind,
    );
    if !seen.insert(key) {
        return Ok(());
    }
    seen.insert(swapped);
    for sector in [DiracChargeSector::LargeLarge, DiracChargeSector::SmallSmall] {
        if !allowed_coupling(
            l,
            sector.orbital_l(left.kappa),
            sector.orbital_l(right.kappa),
        ) {
            continue;
        }
        products.push(sector_product(
            source,
            site,
            radials,
            l,
            sector,
            (left_kind, left),
            (right_kind, right),
        )?);
    }
    Ok(())
}

fn allowed_coupling(l: u32, l1: u32, l2: u32) -> bool {
    (l + l1 + l2) % 2 == 0 && l >= l1.abs_diff(l2) && l <= l1 + l2
}

fn sector_product(
    source: &DiracProductSource,
    site: usize,
    radials: &DiracSiteRadialSet,
    coupled_l: u32,
    sector: DiracChargeSector,
    left: (ProductOrbitalKind, &DiracRadial),
    right: (ProductOrbitalKind, &DiracRadial),
) -> Result<DiracRawRadialProduct, MpbError> {
    let (left_kind, left) = left;
    let (right_kind, right) = right;
    let radii = radials.mesh.radii();
    let mut samples = Vec::with_capacity(radii.len());
    for (index, radius) in radii.iter().enumerate() {
        let value = match sector {
            DiracChargeSector::LargeLarge => left.samples.large[index] * right.samples.large[index],
            DiracChargeSector::SmallSmall => left.samples.small[index] * right.samples.small[index],
        };
        samples.push(value / radius.get());
    }
    let scale =
        (one_particle_norm(&radials.mesh, left)? * one_particle_norm(&radials.mesh, right)?).sqrt();
    if scale > 0.0 {
        for sample in &mut samples {
            *sample /= scale;
        }
    }
    Ok(DiracRawRadialProduct {
        channel: DiracPairChannel {
            q: source.q,
            left: DiracRadialId {
                site,
                kind: left_kind,
                kappa: left.kappa,
                n: left.n,
            },
            right: DiracRadialId {
                site,
                kind: right_kind,
                kappa: right.kappa,
                n: right.n,
            },
            coupled_l,
            sector,
        },
        samples,
    })
}

fn one_particle_norm(mesh: &ExponentialMesh, radial: &DiracRadial) -> Result<f64, MpbError> {
    let integrand = radial
        .samples
        .large
        .iter()
        .zip(&radial.samples.small)
        .map(|(large, small)| large * large + small * small)
        .collect::<Vec<_>>();
    Ok(mesh.integrate(&integrand)?)
}
