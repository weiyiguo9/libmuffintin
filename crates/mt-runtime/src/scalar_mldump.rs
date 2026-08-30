//! Runtime materialization of scalar MLDUMP v1 from frozen scalar product-input, mixed-product, THC, and Coulomb objects.

use std::path::Path;

use muffintin_core::lm_from_index;
use muffintin_io::{
    IoError, MLDUMP_RADIAL_KIND_VALENCE, MldumpHeaderV1, MldumpWriterV1, ScalarApwSiteMatchRefV1,
    ScalarLocalOrbitalTableRefV1, ScalarOrbitalKRefV1, ScalarOrbitalsBeginV1,
    ScalarProductQRecordRefV1, ScalarProductSiteRefV1, ScalarProductsBeginV1, ValidationError,
};
use muffintin_operators::lapw::CompiledBasis;
use muffintin_prodbasis::ProductRadial;
use muffintin_prodbasis::thc::L2Engine;
use thiserror::Error;

use crate::mldump_header::HeaderBindError;
use crate::mldump_write::{
    flatten_eigenvectors, index_i64, interstitial_volume, preflight_header, provenance_key,
    write_coulomb_result, write_thc,
};
use crate::scalar_coulomb::{
    ScalarCoulombError, ScalarCoulombResult, ScalarCoulombSpec,
    require_scalar_coulomb_export_context,
};
use crate::scalar_product::{
    SCALAR_RADIAL_LO0, ScalarProductInput, ScalarQSliceError, require_scalar_q_slice,
};
use crate::scalar_thc::{ScalarThcError, ScalarThcResult};

/// Failure while preflighting or streaming a scalar MLDUMP file.
#[derive(Debug, Error)]
pub enum ScalarMldumpError {
    #[error(transparent)]
    Io(#[from] IoError),
    #[error(transparent)]
    Thc(#[from] ScalarThcError),
    #[error(transparent)]
    Coulomb(#[from] ScalarCoulombError),
    #[error("scalar MLDUMP header mismatch at {path}: expected {expected}, found {actual}")]
    HeaderMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("scalar MLDUMP cannot serialize THC engine {0:?}")]
    UnsupportedEngine(L2Engine),
    #[error("scalar MLDUMP THC selection strategy must be AllQL2")]
    UnsupportedStrategy,
}

impl From<ValidationError> for ScalarMldumpError {
    fn from(error: ValidationError) -> Self {
        Self::Io(error.into())
    }
}

impl From<ScalarQSliceError> for ScalarMldumpError {
    fn from(error: ScalarQSliceError) -> Self {
        Self::Thc(error.into())
    }
}

impl From<HeaderBindError> for ScalarMldumpError {
    fn from(error: HeaderBindError) -> Self {
        Self::HeaderMismatch {
            path: error.path,
            expected: error.expected,
            actual: error.actual,
        }
    }
}

/// Write a populated scalar MLDUMP v1 file from frozen runtime objects.
///
/// `header` is caller-owned because species/labels cannot be reconstructed
/// from [`ScalarProductInput`]. Recoverable geometry, mesh, q-slice, THC, and
/// Coulomb bindings are preflighted before the HDF5 file is created.
pub fn write_scalar_mldump(
    path: impl AsRef<Path>,
    header: &MldumpHeaderV1,
    inputs: &[ScalarProductInput],
    thc: &ScalarThcResult,
    coulomb: &ScalarCoulombResult,
    spec: &ScalarCoulombSpec,
) -> Result<(), ScalarMldumpError> {
    let first = preflight_scalar_mldump(header, inputs, thc, coulomb, spec)?;
    let path = path.as_ref();
    let mut stream = MldumpWriterV1::create(path, header)?.begin_scalar()?;
    write_orbitals(&mut stream, first)?;
    write_products(&mut stream, inputs)?;
    write_thc::<_, ScalarMldumpError, _>(&mut stream, thc)?;
    write_coulomb_result::<_, ScalarMldumpError>(&mut stream, coulomb)?;
    stream.finish()?;
    Ok(())
}

fn preflight_scalar_mldump<'a>(
    header: &MldumpHeaderV1,
    inputs: &'a [ScalarProductInput],
    thc: &ScalarThcResult,
    coulomb: &ScalarCoulombResult,
    spec: &ScalarCoulombSpec,
) -> Result<&'a ScalarProductInput, ScalarMldumpError> {
    header.validate()?;
    let first = require_scalar_q_slice(inputs)?;
    require_scalar_coulomb_export_context(inputs, thc, coulomb, spec)?;
    if !first
        .orbitals
        .channels
        .iter()
        .any(|channel| channel.spin == thc.spin)
    {
        return Err(ScalarThcError::InvalidSpin(thc.spin).into());
    }
    preflight_header::<ScalarMldumpError, _>(header, first, inputs, spec.request.cell())?;
    Ok(first)
}

fn write_orbitals(
    stream: &mut muffintin_io::ScalarMldumpStreamV1,
    first: &ScalarProductInput,
) -> Result<(), ScalarMldumpError> {
    let n_orb = first.orbitals.band_window.count;
    stream.begin_orbitals(&ScalarOrbitalsBeginV1 {
        spin_count: 2,
        band_window_start: index_i64(first.orbitals.band_window.start),
        band_window_count: n_orb,
    })?;
    let n_sites = first.source.partition.site_count();
    for spin in 0..2 {
        let channel = first
            .orbitals
            .channels
            .iter()
            .find(|channel| channel.spin == spin as u8)
            .ok_or(ScalarThcError::InvalidSpin(spin as u8))?;
        for (k, (evecs, energies, basis)) in channel
            .eigenvectors
            .iter()
            .zip(&channel.energies)
            .zip(&channel.bases)
            .map(|((evecs, energies), basis)| (evecs, energies, basis))
            .enumerate()
        {
            let evals = energies.iter().map(|value| value.get()).collect::<Vec<_>>();
            let evec_scratch = flatten_eigenvectors(evecs, n_orb)?;
            let n_pw = basis.plane_waves.len();
            let mut g = Vec::with_capacity(n_pw * 3);
            let mut k_cart = Vec::with_capacity(n_pw * 3);
            let mut q_cart = Vec::with_capacity(n_pw * 3);
            for wave in &basis.plane_waves {
                g.extend_from_slice(&wave.g.index);
                k_cart.extend(wave.k.iter().map(|component| component.get()));
                q_cart.extend(wave.q.iter().map(|component| component.get()));
            }
            let mut site_matches = Vec::with_capacity(n_sites);
            let mut match_store = Vec::with_capacity(n_sites);
            for site in 0..n_sites {
                match_store.push(apw_match_scratch(basis, site, n_pw)?);
            }
            for (site, matching) in match_store.iter().enumerate() {
                site_matches.push(ScalarApwSiteMatchRefV1 {
                    site_index: site,
                    n_lm: matching.n_lm,
                    lm_l: &matching.lm_l,
                    lm_m: &matching.lm_m,
                    matching_coefficients: &matching.coefficients,
                });
            }
            let lo = local_orbital_tables(basis)?;
            let available = channel.available_bands.get(k).copied().unwrap_or(n_orb);
            stream.write_orbital_k(
                spin,
                &ScalarOrbitalKRefV1 {
                    k_index: k,
                    available_bands: available,
                    basis_dimension: evecs.rows(),
                    eigenvalues: &evals,
                    eigenvectors: &evec_scratch,
                    n_plane_waves: n_pw,
                    plane_wave_g: &g,
                    plane_wave_k_cartesian: &k_cart,
                    plane_wave_q_cartesian: &q_cart,
                    site_matches: &site_matches,
                    local_orbitals: ScalarLocalOrbitalTableRefV1 {
                        n_local_orbitals: lo.n_lo,
                        row_index: &lo.row_index,
                        site: &lo.site,
                        l: &lo.l,
                        m: &lo.m,
                        ordinal: &lo.ordinal,
                        radial_n: &lo.radial_n,
                    },
                },
            )?;
        }
    }
    stream.finish_orbitals()?;
    Ok(())
}

fn write_products(
    stream: &mut muffintin_io::ScalarMldumpStreamV1,
    inputs: &[ScalarProductInput],
) -> Result<(), ScalarMldumpError> {
    let first = &inputs[0];
    let n_site = first.source.partition.site_count();
    let site_indices = (0..n_site).map(|site| site as i64).collect::<Vec<_>>();
    let mut positions = Vec::with_capacity(n_site * 3);
    let mut radii = Vec::with_capacity(n_site);
    for site in first.source.partition.sites() {
        positions.extend(site.position.iter().map(|component| component.get()));
        radii.push(site.radius.get());
    }
    let interstitial = interstitial_volume(&first.source.partition);
    let recipe = first
        .source
        .provenance
        .recipe
        .as_deref()
        .unwrap_or("scalar-product");
    let reference = first
        .source
        .provenance
        .reference
        .as_deref()
        .unwrap_or("checkpoint-dft-frozen-scalar-product-input");
    stream.begin_products(&ScalarProductsBeginV1 {
        n_k: first.orbitals.k_fractional.len(),
        n_orb: first.orbitals.band_window.count,
        provenance_recipe: recipe,
        provenance_reference: reference,
        site_indices: &site_indices,
        site_positions: &positions,
        site_radii: &radii,
        interstitial_volume_bohr3: interstitial,
    })?;
    for (site, radials) in first.source.radials.iter().enumerate() {
        let packed = pack_site_radials(radials.valence.as_slice());
        stream.write_product_site(&ScalarProductSiteRefV1 {
            site_index: site,
            n_radial: packed.n_radial,
            n_radial_samples: radials.mesh.len(),
            kind: &packed.kind,
            l: &packed.l,
            n: &packed.n,
            spin: &packed.spin,
            large: &packed.large,
            small: packed.small.as_deref(),
        })?;
    }
    for (q, input) in inputs.iter().enumerate() {
        let mut raw =
            Vec::with_capacity(input.source.interstitial_pair_support.components.len() * 3);
        for component in &input.source.interstitial_pair_support.components {
            raw.extend_from_slice(&component.g_relative.index);
        }
        let cart = input.source.q.cartesian.map(|component| component.get());
        let provenance = provenance_key(&input.source.provenance);
        stream.write_product_q(&ScalarProductQRecordRefV1 {
            q_index: q,
            transfer_cartesian: cart,
            global_transfer: input.source.q.umklapp.index,
            n_raw_g: input.source.interstitial_pair_support.components.len(),
            raw_relative_g: &raw,
            provenance: &provenance,
        })?;
    }
    stream.finish_products()?;
    Ok(())
}

struct ApwMatchScratch {
    n_lm: usize,
    lm_l: Vec<i32>,
    lm_m: Vec<i32>,
    coefficients: Vec<f64>,
}

fn apw_match_scratch(
    basis: &CompiledBasis,
    site: usize,
    n_pw: usize,
) -> Result<ApwMatchScratch, ScalarMldumpError> {
    let n_lm = basis
        .site_augmentations
        .get(site)
        .and_then(|waves| waves.first())
        .map(|wave| wave.coefficients.len())
        .unwrap_or(0);
    let mut lm_l = Vec::with_capacity(n_lm);
    let mut lm_m = Vec::with_capacity(n_lm);
    for index in 0..n_lm {
        let lm = lm_from_index(index);
        lm_l.push(i32::try_from(lm.l).expect("l fits i32"));
        lm_m.push(lm.m);
    }
    let mut coefficients = Vec::with_capacity(n_pw * n_lm * 4);
    for wave in basis
        .site_augmentations
        .get(site)
        .map(Vec::as_slice)
        .unwrap_or(&[])
    {
        for lm in 0..n_lm {
            let [u, udot] = wave.coefficients.get(lm).copied().unwrap_or_default();
            coefficients.extend_from_slice(&[u.re, u.im, udot.re, udot.im]);
        }
    }
    Ok(ApwMatchScratch {
        n_lm,
        lm_l,
        lm_m,
        coefficients,
    })
}

struct LoTables {
    n_lo: usize,
    row_index: Vec<i64>,
    site: Vec<i64>,
    l: Vec<i64>,
    m: Vec<i64>,
    ordinal: Vec<i64>,
    radial_n: Vec<i64>,
}

fn local_orbital_tables(basis: &CompiledBasis) -> Result<LoTables, ScalarMldumpError> {
    let n_pw = basis.layout.plane_wave_count();
    let mut tables = LoTables {
        n_lo: 0,
        row_index: Vec::new(),
        site: Vec::new(),
        l: Vec::new(),
        m: Vec::new(),
        ordinal: Vec::new(),
        radial_n: Vec::new(),
    };
    let mut row = n_pw;
    for site in 0..basis.layout.site_count() {
        let Some(layout) = basis.layout.site_layout(site) else {
            continue;
        };
        for (l, count) in layout.counts_by_l().iter().enumerate() {
            let l_i64 = index_i64(l);
            let l_i32 = i32::try_from(l).expect("l fits i32");
            for m in -l_i32..=l_i32 {
                for ordinal in 0..*count {
                    tables.row_index.push(index_i64(row));
                    tables.site.push(index_i64(site));
                    tables.l.push(l_i64);
                    tables.m.push(i64::from(m));
                    tables.ordinal.push(index_i64(ordinal));
                    tables.radial_n.push(index_i64(SCALAR_RADIAL_LO0 + ordinal));
                    row += 1;
                    tables.n_lo += 1;
                }
            }
        }
    }
    Ok(tables)
}

struct PackedRadials {
    n_radial: usize,
    kind: Vec<i64>,
    l: Vec<i64>,
    n: Vec<i64>,
    spin: Vec<i64>,
    large: Vec<f64>,
    small: Option<Vec<f64>>,
}

fn pack_site_radials(radials: &[ProductRadial]) -> PackedRadials {
    let n_radial = radials.len();
    let mut packed = PackedRadials {
        n_radial,
        kind: Vec::with_capacity(n_radial),
        l: Vec::with_capacity(n_radial),
        n: Vec::with_capacity(n_radial),
        spin: Vec::with_capacity(n_radial),
        large: Vec::new(),
        small: None,
    };
    let all_small = radials.iter().all(|radial| radial.samples.small.is_some());
    if all_small {
        packed.small = Some(Vec::new());
    }
    for radial in radials {
        packed.kind.push(MLDUMP_RADIAL_KIND_VALENCE);
        packed.l.push(i64::from(radial.l));
        packed.n.push(index_i64(radial.n));
        packed.spin.push(i64::from(radial.spin));
        packed.large.extend_from_slice(&radial.samples.large);
        if let Some(small) = packed.small.as_mut() {
            small.extend_from_slice(radial.samples.small.as_deref().unwrap_or(&[]));
        }
    }
    packed
}
