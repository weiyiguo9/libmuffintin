//! Untruncated Dirac PP/QQ muffin-tin products and retained overlap cutoff.

use crate::mpb::overlap::{lowdin_modes, overlap_spectrum, product_channel_functions};
use crate::mpb::{MpbError, auxiliary_interstitial_support};
use crate::{
    AuxiliaryRepresentation, CompiledAuxiliaryBasis, CoupledChannel, CutoffKind, CutoffRecord,
    DiracChargeSector, DiracPairChannel, DiracProductSource, DiracRadial, DiracRadialId,
    DiracRawProductSpace, DiracRawRadialProduct, DiracSiteRadialSet, MixedProductAuxiliary,
    ProductOrbitalKind, SiteAuxiliaryBlock,
};
use muffintin_core::{ExponentialMesh, InverseBohr, ReciprocalLattice};
use muffintin_envelope::Provenance;
use std::collections::{BTreeMap, HashSet};

/// Untruncated Dirac mixed-product space: separate PP and QQ radial products.
///
/// Each allowed scalar $L$ emits PP using $P_i P_j/r$ and QQ using $Q_i Q_j/r$
/// without merging sectors. Each populated $(site,L)$ stores a complete PP
/// prefix in canonical orbital-pair order, then a complete QQ suffix in that
/// same pair order. `CoupledChannel::radial_index` is local to each
/// $(site,L)$ block (`0..products_for_this_site_L`, SPEX flatten
/// $site\to L\to M\to n`). Raw product storage stays a global list. Overlap
/// spectra are the real-symmetric eigensystems of that ordered PP then QQ
/// union.
pub fn untruncated_dirac_product_space(
    source: &DiracProductSource,
    product_l_max: u32,
) -> Result<DiracRawProductSpace, MpbError> {
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
            overlap_spectra.push(overlap_spectrum(site, l, &radials.mesh, &functions)?);
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
    let raw = DiracRawProductSpace {
        partition: source.partition.clone(),
        q: source.q,
        radial_products,
        channels,
        overlap_spectra,
        interstitial_pair_support: source.interstitial_pair_support.clone(),
        provenance: dirac_provenance(None),
    };
    raw.validate_against_source(source)?;
    Ok(raw)
}

/// Apply SPEX `TOL` to Dirac PP/QQ union spectra without merging sectors.
///
/// Retained eigenvalues satisfy $\lambda \ge \mathrm{tol}\times n_{\mathrm{spin}}$.
/// Each $(site,L)$ overlap is the ordered PP-prefix then QQ-suffix union;
/// Löwdin modes are the scalar-charge retained functions those union vectors
/// are projected onto. Every populated muffin-tin block must already carry
/// exactly one dimensionally valid spectrum; dummy empty `overlap_spectra`
/// are rejected here rather than omitted. Interstitial auxiliary PW are
/// reconstructed from `lattice` and `product_g_max`, independently of raw
/// pair support.
pub fn apply_dirac_overlap_cutoff(
    raw: &DiracRawProductSpace,
    source: &DiracProductSource,
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
    require_matching_dirac_source_and_raw(source, raw)?;
    retained_dirac_auxiliary(
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

/// Exact partition, $q$, pair support, and signed-$\kappa$ identity.
pub fn require_matching_dirac_source_and_raw(
    source: &DiracProductSource,
    raw: &DiracRawProductSpace,
) -> Result<(), MpbError> {
    Ok(raw.validate_against_source(source)?)
}

fn dirac_provenance(cutoff: Option<&CutoffRecord>) -> Provenance {
    Provenance {
        recipe: Some(if cutoff.is_some() {
            "dirac-mt-pp-qq+tol".to_owned()
        } else {
            "dirac-mt-pp-qq".to_owned()
        }),
        reference: None,
    }
}

fn retained_dirac_auxiliary(
    raw: &DiracRawProductSpace,
    source: &DiracProductSource,
    cutoff: Option<CutoffRecord>,
    lattice: &ReciprocalLattice,
    product_g_max: InverseBohr,
) -> Result<CompiledAuxiliaryBasis, MpbError> {
    require_matching_dirac_source_and_raw(source, raw)?;
    require_dirac_overlap_spectra(raw)?;
    let mut sites = Vec::with_capacity(source.radials.len());
    for (site, radials) in source.radials.iter().enumerate() {
        let mut channel_l = raw
            .radial_products
            .iter()
            .filter(|product| product.channel.left.site == site)
            .map(|product| product.channel.coupled_l)
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
                .ok_or(MpbError::MissingDiracOverlapSpectrum { site, l })?;
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
        provenance: dirac_provenance(cutoff.as_ref()),
    };
    require_dirac_auxiliary(source, &auxiliary)?;
    Ok(auxiliary)
}

fn require_dirac_auxiliary(
    source: &DiracProductSource,
    auxiliary: &CompiledAuxiliaryBasis,
) -> Result<(), MpbError> {
    auxiliary.validate()?;
    auxiliary.require_mixed_product()?;
    if auxiliary.q != source.q {
        return Err(MpbError::TransferQMismatch);
    }
    if auxiliary.partition != source.partition {
        return Err(MpbError::PartitionMismatch);
    }
    let payload = auxiliary.require_mixed_product()?;
    if payload.interstitial.q != source.q {
        return Err(MpbError::TransferQMismatch);
    }
    if payload.sites.len() != source.radials.len() {
        return Err(MpbError::Product(
            crate::AuxiliaryIrError::AuxiliarySiteCount {
                expected: source.radials.len(),
                actual: payload.sites.len(),
            },
        ));
    }
    for (site, (block, radials)) in payload.sites.iter().zip(&source.radials).enumerate() {
        if block.mesh != radials.mesh {
            return Err(MpbError::Product(
                crate::AuxiliaryIrError::AuxiliaryMeshMismatch { site },
            ));
        }
    }
    Ok(())
}

fn require_dirac_overlap_spectra(raw: &DiracRawProductSpace) -> Result<(), MpbError> {
    let mut n_products = BTreeMap::new();
    for product in &raw.radial_products {
        *n_products
            .entry((product.channel.left.site, product.channel.coupled_l))
            .or_insert(0) += 1;
    }
    for spectrum in &raw.overlap_spectra {
        let Some(&expected) = n_products.get(&(spectrum.site, spectrum.l)) else {
            return Err(MpbError::UnmatchedDiracOverlapSpectrum {
                site: spectrum.site,
                l: spectrum.l,
            });
        };
        let n_eigenvalues = spectrum.eigenvalues.len();
        let n_eigenvectors = spectrum.eigenvectors.len();
        if n_eigenvalues != expected || n_eigenvectors != expected * expected {
            return Err(MpbError::DiracOverlapSpectrumDimension {
                site: spectrum.site,
                l: spectrum.l,
                n_products: expected,
                n_eigenvalues,
                n_eigenvectors,
            });
        }
    }
    for &(site, l) in n_products.keys() {
        if raw.spectrum(site, l).is_none() {
            return Err(MpbError::MissingDiracOverlapSpectrum { site, l });
        }
    }
    Ok(())
}

fn enumerate_site_channel(
    source: &DiracProductSource,
    site: usize,
    radials: &DiracSiteRadialSet,
    l: u32,
) -> Result<Vec<DiracRawRadialProduct>, MpbError> {
    let valence = radials.valence.iter().collect::<Vec<_>>();
    let mut seen = HashSet::new();
    let mut pairs = Vec::new();
    for &left in &valence {
        for &right in &valence {
            record_unique_pair(
                (ProductOrbitalKind::Valence, left),
                (ProductOrbitalKind::Valence, right),
                &mut seen,
                &mut pairs,
            );
        }
    }
    for core in &radials.cores {
        for &valence in &valence {
            record_unique_pair(
                (ProductOrbitalKind::Core, core),
                (ProductOrbitalKind::Valence, valence),
                &mut seen,
                &mut pairs,
            );
        }
    }
    let mut products = Vec::new();
    for sector in [DiracChargeSector::LargeLarge, DiracChargeSector::SmallSmall] {
        for &(left, right) in &pairs {
            if !allowed_coupling(
                l,
                sector.orbital_l(left.1.kappa),
                sector.orbital_l(right.1.kappa),
            ) {
                continue;
            }
            products.push(sector_product(
                source, site, radials, l, sector, left, right,
            )?);
        }
    }
    Ok(products)
}

type DiracOrbitalRef<'a> = (ProductOrbitalKind, &'a DiracRadial);
type UniquePairKey = (
    i32,
    usize,
    ProductOrbitalKind,
    i32,
    usize,
    ProductOrbitalKind,
);

fn record_unique_pair<'a>(
    left: DiracOrbitalRef<'a>,
    right: DiracOrbitalRef<'a>,
    seen: &mut HashSet<UniquePairKey>,
    pairs: &mut Vec<(DiracOrbitalRef<'a>, DiracOrbitalRef<'a>)>,
) {
    let (left_kind, left_radial) = left;
    let (right_kind, right_radial) = right;
    let key = (
        left_radial.kappa.get(),
        left_radial.n,
        left_kind,
        right_radial.kappa.get(),
        right_radial.n,
        right_kind,
    );
    let swapped = (
        right_radial.kappa.get(),
        right_radial.n,
        right_kind,
        left_radial.kappa.get(),
        left_radial.n,
        left_kind,
    );
    if !seen.insert(key) {
        return;
    }
    seen.insert(swapped);
    pairs.push((left, right));
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
