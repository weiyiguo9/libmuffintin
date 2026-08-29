//! Runtime materialization of scalar MLDUMP v1 from frozen M-L1–L4 objects.

use std::f64::consts::PI;
use std::path::Path;

use muffintin_auxiliary_ir::{OrbitalPair, ProductRadial};
use muffintin_core::lm_from_index;
use muffintin_io::{
    IoError, MLDUMP_INTERSTITIAL_SENTINEL, MLDUMP_PARENT_REGION_INTERSTITIAL,
    MLDUMP_PARENT_REGION_MUFFIN_TIN, MLDUMP_RADIAL_KIND_VALENCE,
    MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY, MLDUMP_THC_ENGINE_QRCP, MLDUMP_THC_STRATEGY_ALL_QL2,
    MldumpCoulombBeginV1, MldumpCoulombGammaRefV1, MldumpCoulombQRecordRefV1, MldumpHeaderV1,
    MldumpThcBeginV1, MldumpThcParentGridRefV1, MldumpThcQRecordRefV1, MldumpThcSelectionRefV1,
    MldumpThcVertexTableRefV1, MldumpWriterV1, ScalarApwSiteMatchRefV1,
    ScalarLocalOrbitalTableRefV1, ScalarOrbitalKRefV1, ScalarOrbitalsBeginV1,
    ScalarProductQRecordRefV1, ScalarProductSiteRefV1, ScalarProductsBeginV1, ValidationError,
};
use muffintin_lapw::{CompiledBasis, Provenance};
use muffintin_tensor::DenseEigenvectors;
use muffintin_thc::{L2Engine, RankPolicy, SelectorStrategy};
use num_complex::Complex64;
use thiserror::Error;

use crate::mldump_header::{
    HeaderBind, HeaderBindError, HeaderBindKMinusQ, HeaderBindQ, HeaderBindSite,
    preflight_mldump_header,
};
use crate::scalar_coulomb::{
    ScalarCoulombError, ScalarCoulombResult, ScalarCoulombSpec,
    require_scalar_coulomb_export_context,
};
use crate::scalar_product::{
    SCALAR_RADIAL_LO0, ScalarProductInput, ScalarQSliceError, require_scalar_q_slice,
};
use crate::scalar_thc::{ScalarThcError, ScalarThcResult};
use crate::thc_grid::ThcRegion;

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
    write_thc(&mut stream, thc)?;
    write_coulomb(&mut stream, coulomb)?;
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
    preflight_header(header, first, inputs, spec)?;
    Ok(first)
}

fn preflight_header(
    header: &MldumpHeaderV1,
    first: &ScalarProductInput,
    inputs: &[ScalarProductInput],
    spec: &ScalarCoulombSpec,
) -> Result<(), ScalarMldumpError> {
    let cell = spec.request.cell();
    let sites = first
        .source
        .partition
        .sites()
        .iter()
        .zip(&first.source.radials)
        .map(|(site, radials)| HeaderBindSite {
            position: site.position.map(|component| component.get()),
            radius: site.radius.get(),
            mesh_first: radials.mesh.first().get(),
            mesh_increment: radials.mesh.increment(),
            mesh_count: radials.mesh.len(),
        })
        .collect::<Vec<_>>();
    let k_maps = inputs
        .iter()
        .map(|input| {
            input
                .k_minus_q
                .iter()
                .map(|mapped| HeaderBindKMinusQ {
                    k_index: mapped.k_index,
                    mapped_index: mapped.kq_index,
                    g_wrap: mapped.umklapp.index,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let q_records = inputs
        .iter()
        .zip(&k_maps)
        .map(|(input, k_minus_q)| HeaderBindQ {
            cartesian: input.source.q.cartesian.map(|component| component.get()),
            umklapp: input.source.q.umklapp.index,
            k_minus_q,
        })
        .collect::<Vec<_>>();
    preflight_mldump_header(
        header,
        &HeaderBind {
            direct_basis: std::array::from_fn(|row| {
                std::array::from_fn(|axis| cell.basis()[row][axis].get())
            }),
            reciprocal_basis: std::array::from_fn(|row| {
                std::array::from_fn(|axis| first.reciprocal.basis()[row][axis].get())
            }),
            cell_volume: cell.volume().get(),
            partition_volume: first.source.partition.interstitial().cell_volume().get(),
            sites: &sites,
            k_fractional: &first.orbitals.k_fractional,
            q_records: &q_records,
        },
    )?;
    Ok(())
}

fn write_orbitals(
    stream: &mut muffintin_io::ScalarMldumpStreamV1,
    first: &ScalarProductInput,
) -> Result<(), ScalarMldumpError> {
    let n_orb = first.orbitals.band_window.count;
    stream.begin_orbitals(&ScalarOrbitalsBeginV1 {
        spin_count: 2,
        band_window_start: i64::try_from(first.orbitals.band_window.start).unwrap_or(0),
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
    let interstitial = interstitial_volume(first);
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
        .unwrap_or("snapshot-dft-frozen-scalar-ml1");
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

fn write_thc(
    stream: &mut muffintin_io::ScalarMldumpStreamV1,
    thc: &ScalarThcResult,
) -> Result<(), ScalarMldumpError> {
    if thc.selection.provenance.strategy != SelectorStrategy::AllQL2 {
        return Err(ScalarMldumpError::UnsupportedStrategy);
    }
    let engine = match thc.selection.provenance.engine {
        L2Engine::FullColumnPivotedQr => MLDUMP_THC_ENGINE_QRCP,
        L2Engine::FullPivotedCholesky => MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY,
        other => return Err(ScalarMldumpError::UnsupportedEngine(other)),
    };
    let n_points = thc.grid.points().len();
    let mut coordinates = Vec::with_capacity(n_points * 3);
    let mut weights = Vec::with_capacity(n_points);
    let mut region_kind = Vec::with_capacity(n_points);
    let mut site_index = Vec::with_capacity(n_points);
    let mut radial_index = Vec::with_capacity(n_points);
    for point in thc.grid.points() {
        coordinates.extend(point.coordinate.iter().map(|component| component.get()));
        weights.push(point.weight);
        match point.region {
            ThcRegion::MuffinTin {
                site,
                radial_index: radial,
            } => {
                region_kind.push(MLDUMP_PARENT_REGION_MUFFIN_TIN);
                site_index.push(i64::try_from(site).unwrap_or(i64::MIN));
                radial_index.push(i64::try_from(radial).unwrap_or(i64::MIN));
            }
            ThcRegion::Interstitial => {
                region_kind.push(MLDUMP_PARENT_REGION_INTERSTITIAL);
                site_index.push(MLDUMP_INTERSTITIAL_SENTINEL);
                radial_index.push(MLDUMP_INTERSTITIAL_SENTINEL);
            }
        }
    }
    let pivots = thc
        .selection
        .pivots
        .iter()
        .map(|index| i64::try_from(*index).unwrap_or(i64::MIN))
        .collect::<Vec<_>>();
    let points = thc
        .selection
        .points
        .iter()
        .map(|point| i64::try_from(point.id).unwrap_or(i64::MIN))
        .collect::<Vec<_>>();
    let n_candidates = weights.iter().filter(|weight| **weight > 0.0).count();
    let requested_rank = match thc.requested_rank {
        RankPolicy::Exact { n_mu } => n_mu,
        RankPolicy::Threshold { n_max, .. } => n_max,
    };
    let grid_provenance = provenance_key(thc.grid.provenance());
    stream.begin_thc(&MldumpThcBeginV1 {
        parent_grid: MldumpThcParentGridRefV1 {
            n_points,
            coordinates: &coordinates,
            weights: &weights,
            region_kind: &region_kind,
            site_index: &site_index,
            radial_index: &radial_index,
            provenance: &grid_provenance,
        },
        strategy: MLDUMP_THC_STRATEGY_ALL_QL2,
        engine,
        requested_rank,
        effective_rank: thc.effective_rank,
        n_candidates,
        selection: MldumpThcSelectionRefV1 {
            pivots: &pivots,
            points: &points,
        },
    })?;
    for record in &thc.records {
        let zeta = flatten_complex(&record.fit.zeta);
        let n_vertex = record.vertices.len();
        let mut column = Vec::with_capacity(n_vertex);
        let mut k_left_right = Vec::with_capacity(n_vertex * 3);
        let mut coefficients = Vec::new();
        for (index, vertex) in record.vertices.iter().enumerate() {
            column.push(i64::try_from(index).unwrap_or(i64::MIN));
            match vertex.pair() {
                OrbitalPair::Bloch {
                    k_index,
                    left,
                    right,
                } => {
                    k_left_right.push(i64::try_from(k_index).unwrap_or(i64::MIN));
                    k_left_right.push(i64::try_from(left).unwrap_or(i64::MIN));
                    k_left_right.push(i64::try_from(right).unwrap_or(i64::MIN));
                }
                _ => {
                    return Err(ScalarCoulombError::VertexIdentity {
                        index: record.q_index,
                        column: index,
                    }
                    .into());
                }
            }
            coefficients.extend(flatten_complex(vertex.coefficients()));
        }
        let layout_provenance = provenance_key(&record.auxiliary.provenance);
        stream.write_thc_q(&MldumpThcQRecordRefV1 {
            q_index: record.q_index,
            aux_dimension: record.fit.n_mu,
            layout_provenance: &layout_provenance,
            zeta: &zeta,
            residual_l2_all_frobenius: record.fit.l2_all.frobenius,
            residual_l2_all_column_max: record.fit.l2_all.column_max,
            vertices: MldumpThcVertexTableRefV1 {
                n_vertex,
                column: &column,
                k_left_right: &k_left_right,
                coefficients: &coefficients,
            },
        })?;
    }
    stream.finish_thc()?;
    Ok(())
}

fn write_coulomb(
    stream: &mut muffintin_io::ScalarMldumpStreamV1,
    coulomb: &ScalarCoulombResult,
) -> Result<(), ScalarMldumpError> {
    stream.begin_coulomb(&MldumpCoulombBeginV1 {
        lexp: coulomb.context.request.lexp(),
        interpolation_l_max: coulomb.context.projection.l_max,
        interpolation_pw_cutoff: coulomb.context.projection.pw_cutoff.get(),
    })?;
    for record in &coulomb.records {
        let body = flatten_complex(record.operator.matrix());
        let gamma_scratch = record.operator.gamma().map(|gamma| {
            (
                gamma.spherical_average_subtracted,
                gamma.head_prefactor,
                flatten_complex(&gamma.constant_coefficients),
            )
        });
        let layout_provenance = provenance_key(&record.auxiliary.provenance);
        stream.write_coulomb_q(&MldumpCoulombQRecordRefV1 {
            q_index: record.q_index,
            aux_dimension: record.operator.dimension(),
            layout_provenance: &layout_provenance,
            body: &body,
            gamma: gamma_scratch
                .as_ref()
                .map(|(subtracted, prefactor, coeffs)| MldumpCoulombGammaRefV1 {
                    spherical_average_subtracted: *subtracted,
                    head_prefactor: *prefactor,
                    constant_coefficients: coeffs,
                }),
        })?;
    }
    stream.finish_coulomb()?;
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
        lm_l.push(i32::try_from(lm.l).unwrap_or(i32::MIN));
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
            let l_i64 = i64::try_from(l).unwrap_or(i64::MIN);
            let l_i32 = i32::try_from(l).unwrap_or(0);
            for m in -l_i32..=l_i32 {
                for ordinal in 0..*count {
                    tables
                        .row_index
                        .push(i64::try_from(row).unwrap_or(i64::MIN));
                    tables.site.push(i64::try_from(site).unwrap_or(i64::MIN));
                    tables.l.push(l_i64);
                    tables.m.push(i64::from(m));
                    tables
                        .ordinal
                        .push(i64::try_from(ordinal).unwrap_or(i64::MIN));
                    tables
                        .radial_n
                        .push(i64::try_from(SCALAR_RADIAL_LO0 + ordinal).unwrap_or(i64::MIN));
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
        packed.n.push(i64::try_from(radial.n).unwrap_or(i64::MIN));
        packed.spin.push(i64::from(radial.spin));
        packed.large.extend_from_slice(&radial.samples.large);
        if let Some(small) = packed.small.as_mut() {
            small.extend_from_slice(radial.samples.small.as_deref().unwrap_or(&[]));
        }
    }
    packed
}

fn flatten_eigenvectors(
    evecs: &DenseEigenvectors,
    n_orb: usize,
) -> Result<Vec<f64>, ScalarMldumpError> {
    let n_basis = evecs.rows();
    let mut out = Vec::with_capacity(n_basis * n_orb * 2);
    for row in 0..n_basis {
        for band in 0..n_orb {
            let value = evecs
                .get(row, band)
                .map_err(|error| ValidationError::InvalidValue {
                    path: "orbitals.eigenvectors".to_owned(),
                    expected: "in-bounds leading window".to_owned(),
                    actual: error.to_string(),
                })?;
            out.push(value.re);
            out.push(value.im);
        }
    }
    Ok(out)
}

fn flatten_complex(values: &[Complex64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.push(value.re);
        out.push(value.im);
    }
    out
}

fn provenance_key(provenance: &Provenance) -> String {
    format!(
        "{}|{}",
        provenance.recipe.as_deref().unwrap_or(""),
        provenance.reference.as_deref().unwrap_or("")
    )
}

fn interstitial_volume(input: &ScalarProductInput) -> f64 {
    let cell = input.source.partition.interstitial().cell_volume().get();
    let muffin = input
        .source
        .partition
        .sites()
        .iter()
        .map(|site| 4.0 / 3.0 * PI * site.radius.get().powi(3))
        .sum::<f64>();
    cell - muffin
}
