//! SPEX mixed-basis enumeration, spectra, and retained transforms.

use crate::{MpbError, auxiliary_interstitial_support};
use libmuffintin_basis::Provenance;
use libmuffintin_core::{ExponentialMesh, InverseBohr, ReciprocalLattice};
use libmuffintin_operators::solve_real_symmetric;
use libmuffintin_product::{
    ChannelSpectrum, CompiledAuxiliaryBasis, CoupledChannel, CutoffKind, CutoffRecord,
    MtAuxiliaryMode, PairChannel, ProductOrbitalKind, ProductRadial, ProductRadialId,
    ProductSource, RawProductSpace, RawRadialProduct, SiteAuxiliaryBlock, SiteRadialSet,
};
use libmuffintin_radial::RadialComponents;
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

/// Keep strictly positive eigenvalues that are not below the SPEX threshold.
///
/// SPEX drops `eig < tolerance*nspin` (`mixedbasis.f:463`), so equality is kept.
pub(crate) fn retain_overlap_eigenvalue(eigenvalue: f64, threshold: f64) -> bool {
    eigenvalue > 0.0 && eigenvalue >= threshold
}

/// Untruncated SPEX mixed product basis: full spectra and MPB auxiliary PW.
///
/// Raw interstitial orbital-pair support is copied from [`ProductSource`].
/// The auxiliary $|q+G|$ set is built separately from `lattice`, canonical
/// `q`, and `product_g_max`. `TOL` is not applied.
pub fn spex_mixed_product_basis(
    source: &ProductSource,
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
    source: &ProductSource,
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
    source: &ProductSource,
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
    source: &ProductSource,
    raw: &RawProductSpace,
    auxiliary: &CompiledAuxiliaryBasis,
) -> Result<(), MpbError> {
    require_matching_source_and_raw(source, raw)?;
    auxiliary.validate_against_source(source)?;
    if auxiliary.q != source.q || auxiliary.interstitial.q != source.q {
        return Err(MpbError::TransferQMismatch);
    }
    if auxiliary.partition != source.partition || auxiliary.partition != raw.partition {
        return Err(MpbError::PartitionMismatch);
    }
    Ok(())
}

fn untruncated_product_space(
    source: &ProductSource,
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
            let functions = channel_functions(&radials.mesh, l, &products)?;
            let spectrum = spectrum_with_mesh(site, l, &radials.mesh, &functions)?;
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
    source: &ProductSource,
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
    source: &ProductSource,
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

pub(crate) fn channel_functions(
    mesh: &ExponentialMesh,
    l: u32,
    products: &[RawRadialProduct],
) -> Result<Vec<Vec<f64>>, MpbError> {
    let radius = mesh.last().get();
    let constant_norm = (radius.powi(3) / 3.0).sqrt();
    let mut functions = Vec::with_capacity(products.len());
    for product in products {
        let mut samples = product.samples.clone();
        if l == 0 {
            let projection_integrand = mesh
                .radii()
                .iter()
                .zip(&samples)
                .map(|(radius, sample)| sample * radius.get())
                .collect::<Vec<_>>();
            let projection = mesh.integrate(&projection_integrand)? / constant_norm;
            for (sample, radius) in samples.iter_mut().zip(mesh.radii()) {
                *sample -= projection * radius.get() / constant_norm;
            }
        }
        let norm_sq = mesh.integrate(
            &samples
                .iter()
                .map(|value| value * value)
                .collect::<Vec<_>>(),
        )?;
        let scale = norm_sq.max(0.0).sqrt();
        if scale > 0.0 {
            for sample in &mut samples {
                *sample /= scale;
            }
        }
        functions.push(samples);
    }
    Ok(functions)
}

fn spectrum_with_mesh(
    site: usize,
    l: u32,
    mesh: &ExponentialMesh,
    functions: &[Vec<f64>],
) -> Result<ChannelSpectrum, MpbError> {
    let n = functions.len();
    if n == 0 {
        return Err(MpbError::EmptyChannel { site, l });
    }
    let mut overlaps = vec![0.0; n * n];
    for row in 0..n {
        for column in row..n {
            let integrand = functions[row]
                .iter()
                .zip(&functions[column])
                .map(|(left, right)| left * right)
                .collect::<Vec<_>>();
            let value = mesh.integrate(&integrand)?;
            overlaps[row * n + column] = value;
            overlaps[column * n + row] = value;
        }
    }
    let solution = solve_real_symmetric(n, |row, column| overlaps[row * n + column])?;
    Ok(ChannelSpectrum {
        site,
        l,
        eigenvalues: solution.eigenvalues,
        eigenvectors: solution.eigenvectors,
    })
}

fn retained_auxiliary(
    raw: &RawProductSpace,
    source: &ProductSource,
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
            let functions = channel_functions(&radials.mesh, l, &products)?;
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
        sites,
        interstitial: auxiliary_interstitial_support(lattice, raw.q, product_g_max)?,
        cutoff,
        provenance: spex_provenance(cutoff.as_ref()),
    };
    auxiliary.validate_against_source(source)?;
    Ok(auxiliary)
}

fn lowdin_modes(
    l: u32,
    mesh: &ExponentialMesh,
    functions: &[Vec<f64>],
    spectrum: &ChannelSpectrum,
    cutoff: Option<&CutoffRecord>,
) -> Result<Vec<MtAuxiliaryMode>, MpbError> {
    let n = functions.len();
    let threshold = cutoff
        .map(|record| record.value * record.nspin_factor)
        .unwrap_or(0.0);
    let mut kept = Vec::new();
    for (index, &eigenvalue) in spectrum.eigenvalues.iter().enumerate() {
        if retain_overlap_eigenvalue(eigenvalue, threshold) {
            kept.push(index);
        }
    }
    if kept.is_empty() {
        return Err(MpbError::EmptyRetainedChannel {
            site: spectrum.site,
            l,
        });
    }
    let n_mesh = functions[0].len();
    let mut transformed = vec![vec![0.0; n_mesh]; kept.len()];
    for (kept_index, &column) in kept.iter().enumerate() {
        let scale = 1.0 / spectrum.eigenvalues[column].sqrt();
        for (basis, function) in functions.iter().enumerate() {
            let coefficient = spectrum.eigenvectors[basis + column * n] * scale;
            for (sample, value) in transformed[kept_index].iter_mut().zip(function) {
                *sample += coefficient * value;
            }
        }
    }
    let mut modes = Vec::new();
    let mut n_aux = 0;
    if l == 0 {
        let radius = mesh.last().get();
        let constant_norm = (radius.powi(3) / 3.0).sqrt();
        modes.push(MtAuxiliaryMode {
            l: 0,
            n: 0,
            radial: mesh
                .radii()
                .iter()
                .map(|sample| sample.get() / constant_norm)
                .collect(),
        });
        n_aux = 1;
    }
    for radial in transformed {
        modes.push(MtAuxiliaryMode {
            l,
            n: n_aux,
            radial,
        });
        n_aux += 1;
    }
    Ok(modes)
}
