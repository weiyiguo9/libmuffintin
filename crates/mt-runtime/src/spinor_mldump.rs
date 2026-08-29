//! Runtime materialization of spinor MLDUMP v1 from frozen M-L5b–L5d objects.

use std::f64::consts::PI;
use std::path::Path;

use muffintin_auxiliary_ir::{DiracRadial, OrbitalPair, PartitionSite};
use muffintin_core::{Bohr, RelativisticChannel};
use muffintin_io::{
    IoError, MLDUMP_INTERSTITIAL_SENTINEL, MLDUMP_PARENT_REGION_INTERSTITIAL,
    MLDUMP_PARENT_REGION_MUFFIN_TIN, MLDUMP_RADIAL_KIND_VALENCE,
    MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY, MLDUMP_THC_ENGINE_QRCP, MLDUMP_THC_STRATEGY_ALL_QL2,
    MldumpCoulombBeginV1, MldumpCoulombGammaRefV1, MldumpCoulombQRecordRefV1, MldumpHeaderV1,
    MldumpSiteV1, MldumpThcBeginV1, MldumpThcParentGridRefV1, MldumpThcQRecordRefV1,
    MldumpThcSelectionRefV1, MldumpThcVertexTableRefV1, MldumpWriterV1,
    SpinorLocalOrbitalTableRefV1, SpinorOrbitalKRefV1, SpinorOrbitalsBeginV1,
    SpinorPauliRowMapRefV1, SpinorProductQRecordRefV1, SpinorProductSiteRefV1,
    SpinorProductsBeginV1, SpinorSiteMatchRefV1, ValidationError,
};
use muffintin_lapw::{Provenance, SpinorCompiledBasis};
use muffintin_tensor::DenseEigenvectors;
use muffintin_thc::{L2Engine, RankPolicy, SelectorStrategy};
use num_complex::Complex64;
use thiserror::Error;

use crate::mldump_header::{
    HeaderBind, HeaderBindError, HeaderBindKMinusQ, HeaderBindQ, HeaderBindSite,
    PREFLIGHT_TOLERANCE, preflight_mldump_header,
};
use crate::spinor_coulomb::{
    SpinorCoulombError, SpinorCoulombResult, SpinorCoulombSpec,
    require_spinor_coulomb_export_context,
};
use crate::spinor_product::{
    SPINOR_RADIAL_LO0, SpinorProductInput, SpinorQSliceError, require_spinor_q_slice,
};
use crate::spinor_thc::{SpinorThcError, SpinorThcResult};
use crate::thc_grid::ThcRegion;

const N_PAULI: usize = 2;

/// Failure while preflighting or streaming a spinor MLDUMP file.
#[derive(Debug, Error)]
pub enum SpinorMldumpError {
    #[error(transparent)]
    Io(#[from] IoError),
    #[error(transparent)]
    Thc(#[from] SpinorThcError),
    #[error(transparent)]
    Coulomb(#[from] SpinorCoulombError),
    #[error("spinor MLDUMP header mismatch at {path}: expected {expected}, found {actual}")]
    HeaderMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("spinor MLDUMP cannot serialize THC engine {0:?}")]
    UnsupportedEngine(L2Engine),
    #[error("spinor MLDUMP THC selection strategy must be AllQL2")]
    UnsupportedStrategy,
    #[error(
        "spinor MLDUMP compiled basis is not exportable at {path}: expected {expected}, found {actual}"
    )]
    ExportableBasis {
        path: String,
        expected: String,
        actual: String,
    },
}

impl From<ValidationError> for SpinorMldumpError {
    fn from(error: ValidationError) -> Self {
        Self::Io(error.into())
    }
}

impl From<SpinorQSliceError> for SpinorMldumpError {
    fn from(error: SpinorQSliceError) -> Self {
        Self::Thc(error.into())
    }
}

impl From<HeaderBindError> for SpinorMldumpError {
    fn from(error: HeaderBindError) -> Self {
        Self::HeaderMismatch {
            path: error.path,
            expected: error.expected,
            actual: error.actual,
        }
    }
}

/// Write a populated spinor MLDUMP v1 file from frozen runtime objects.
///
/// `header` is caller-owned because species/labels cannot be reconstructed
/// from [`SpinorProductInput`]. Recoverable geometry, mesh, q-slice, THC, and
/// Coulomb bindings are preflighted before the HDF5 file is created.
pub fn write_spinor_mldump(
    path: impl AsRef<Path>,
    header: &MldumpHeaderV1,
    inputs: &[SpinorProductInput],
    thc: &SpinorThcResult,
    coulomb: &SpinorCoulombResult,
    spec: &SpinorCoulombSpec,
) -> Result<(), SpinorMldumpError> {
    let first = preflight_spinor_mldump(header, inputs, thc, coulomb, spec)?;
    let path = path.as_ref();
    let mut stream = MldumpWriterV1::create(path, header)?.begin_spinor()?;
    write_orbitals(&mut stream, first)?;
    write_products(&mut stream, inputs)?;
    write_thc(&mut stream, thc)?;
    write_coulomb(&mut stream, coulomb)?;
    stream.finish()?;
    Ok(())
}

fn preflight_spinor_mldump<'a>(
    header: &MldumpHeaderV1,
    inputs: &'a [SpinorProductInput],
    thc: &SpinorThcResult,
    coulomb: &SpinorCoulombResult,
    spec: &SpinorCoulombSpec,
) -> Result<&'a SpinorProductInput, SpinorMldumpError> {
    header.validate()?;
    let first = require_spinor_q_slice(inputs)?;
    require_spinor_coulomb_export_context(inputs, thc, coulomb, spec)?;
    preflight_header(header, first, inputs, spec)?;
    require_spinor_bases_exportable(first, header)?;
    Ok(first)
}

fn preflight_header(
    header: &MldumpHeaderV1,
    first: &SpinorProductInput,
    inputs: &[SpinorProductInput],
    spec: &SpinorCoulombSpec,
) -> Result<(), SpinorMldumpError> {
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

fn require_spinor_bases_exportable(
    first: &SpinorProductInput,
    header: &MldumpHeaderV1,
) -> Result<(), SpinorMldumpError> {
    let partition_sites = first.source.partition.sites();
    let header_sites = header.geometry.sites.as_slice();
    let n_sites = partition_sites.len();
    if header_sites.len() != n_sites {
        return Err(exportable_basis(
            "header.geometry.sites",
            n_sites.to_string(),
            header_sites.len().to_string(),
        ));
    }
    for (k, basis) in first.orbitals.bases.iter().enumerate() {
        if basis.layout.site_count() != n_sites
            || basis.site_augmentations.len() != n_sites
            || basis.site_geometry.len() != n_sites
            || basis.site_count() != n_sites
        {
            return Err(exportable_basis(
                &format!("orbitals.bases[{k}].sites"),
                n_sites.to_string(),
                format!(
                    "layout={} augmentations={} geometry={}",
                    basis.layout.site_count(),
                    basis.site_augmentations.len(),
                    basis.site_geometry.len()
                ),
            ));
        }
        for (site, (partition_site, header_site)) in
            partition_sites.iter().zip(header_sites).enumerate()
        {
            require_exportable_site_augmentation(basis, site)?;
            require_site_geometry_binding(k, site, basis, partition_site, header_site)?;
        }
    }
    Ok(())
}

fn require_exportable_site_augmentation(
    basis: &SpinorCompiledBasis,
    site: usize,
) -> Result<&[RelativisticChannel], SpinorMldumpError> {
    let n_sites = basis.layout.site_count();
    if basis.site_augmentations.len() != n_sites || basis.site_geometry.len() != n_sites {
        return Err(exportable_basis(
            "orbitals.site_augmentations",
            n_sites.to_string(),
            format!(
                "augmentations={} geometry={}",
                basis.site_augmentations.len(),
                basis.site_geometry.len()
            ),
        ));
    }
    let n_pw = basis.plane_waves.len();
    if n_pw != basis.layout.spatial_plane_wave_count() {
        return Err(exportable_basis(
            "orbitals.plane_waves",
            basis.layout.spatial_plane_wave_count().to_string(),
            n_pw.to_string(),
        ));
    }
    let waves = basis.site_augmentations.get(site).ok_or_else(|| {
        exportable_basis(
            &format!("orbitals.site_matches[{site}]"),
            "compiled site augmentation".to_owned(),
            "missing".to_owned(),
        )
    })?;
    if waves.len() != n_pw {
        return Err(exportable_basis(
            &format!("orbitals.site_matches[{site}].plane_waves"),
            n_pw.to_string(),
            waves.len().to_string(),
        ));
    }
    let channels = waves
        .first()
        .map(|wave| wave.channels.as_slice())
        .unwrap_or(&[]);
    require_canonical_channel_sequence(site, channels)?;
    for (pw, wave) in waves.iter().enumerate() {
        if wave.channels.as_slice() != channels {
            return Err(exportable_basis(
                &format!("orbitals.site_matches[{site}].channels[{pw}]"),
                "native signed-kappa then twice_mu channel order shared by every plane wave"
                    .to_owned(),
                format!("{} channels", wave.channel_count()),
            ));
        }
        for pauli in 0..N_PAULI {
            let coefficients = wave.coefficients.get(pauli).ok_or_else(|| {
                exportable_basis(
                    &format!("orbitals.site_matches[{site}].coefficients[{pw}][{pauli}]"),
                    channels.len().to_string(),
                    "missing Pauli component".to_owned(),
                )
            })?;
            if coefficients.len() != channels.len() {
                return Err(exportable_basis(
                    &format!("orbitals.site_matches[{site}].coefficients[{pw}][{pauli}]"),
                    channels.len().to_string(),
                    coefficients.len().to_string(),
                ));
            }
        }
    }
    Ok(channels)
}

fn require_canonical_channel_sequence(
    site: usize,
    channels: &[RelativisticChannel],
) -> Result<(), SpinorMldumpError> {
    let mut kappas = Vec::new();
    for channel in channels {
        let kappa = channel.kappa();
        if let Some(&last) = kappas.last() {
            if kappa == last {
                continue;
            }
            if kappas.contains(&kappa) || kappa.get() <= last.get() {
                return Err(exportable_basis(
                    &format!("orbitals.site_matches[{site}].apw_projection"),
                    "strictly increasing signed-kappa then twice_mu".to_owned(),
                    format!("kappa={}", kappa.get()),
                ));
            }
        }
        kappas.push(kappa);
    }
    let expected = kappas
        .into_iter()
        .flat_map(|kappa| kappa.channels())
        .collect::<Vec<_>>();
    if expected.as_slice() != channels {
        return Err(exportable_basis(
            &format!("orbitals.site_matches[{site}].apw_projection"),
            "live canonical signed-kappa ascending then twice_mu ascending".to_owned(),
            format!("{} channels", channels.len()),
        ));
    }
    Ok(())
}

fn require_site_geometry_binding(
    k: usize,
    site: usize,
    basis: &SpinorCompiledBasis,
    partition_site: &PartitionSite,
    header_site: &MldumpSiteV1,
) -> Result<(), SpinorMldumpError> {
    let geometry = basis.site_geometry.get(site).ok_or_else(|| {
        exportable_basis(
            &format!("orbitals.bases[{k}].site_geometry[{site}]"),
            "compiled site geometry".to_owned(),
            "missing".to_owned(),
        )
    })?;
    require_geometry_match(
        &format!("orbitals.bases[{k}].site_geometry[{site}].position"),
        geometry.position,
        partition_site.position,
    )?;
    require_approx(
        &format!("orbitals.bases[{k}].site_geometry[{site}].radius"),
        geometry.radius.get(),
        partition_site.radius.get(),
    )?;
    require_geometry_match(
        &format!("header.geometry.sites[{site}].position_bohr"),
        [
            Bohr(header_site.position_bohr[0]),
            Bohr(header_site.position_bohr[1]),
            Bohr(header_site.position_bohr[2]),
        ],
        partition_site.position,
    )?;
    require_approx(
        &format!("header.geometry.sites[{site}].radius_bohr"),
        header_site.radius_bohr,
        partition_site.radius.get(),
    )?;
    Ok(())
}

fn require_geometry_match(
    path: &str,
    stored: [Bohr; 3],
    expected: [Bohr; 3],
) -> Result<(), SpinorMldumpError> {
    for (axis, (stored_comp, expected_comp)) in stored.iter().zip(&expected).enumerate() {
        require_approx(
            &format!("{path}[{axis}]"),
            stored_comp.get(),
            expected_comp.get(),
        )?;
    }
    Ok(())
}

fn require_approx(path: &str, stored: f64, expected: f64) -> Result<(), SpinorMldumpError> {
    if !stored.is_finite() || !expected.is_finite() {
        return Err(exportable_basis(
            path,
            expected.to_string(),
            stored.to_string(),
        ));
    }
    let scale = stored.abs().max(expected.abs()).max(1.0);
    if (stored - expected).abs() <= PREFLIGHT_TOLERANCE * scale {
        Ok(())
    } else {
        Err(exportable_basis(
            path,
            expected.to_string(),
            stored.to_string(),
        ))
    }
}

fn exportable_basis(path: &str, expected: String, actual: String) -> SpinorMldumpError {
    SpinorMldumpError::ExportableBasis {
        path: path.to_owned(),
        expected,
        actual,
    }
}

fn write_orbitals(
    stream: &mut muffintin_io::SpinorMldumpStreamV1,
    first: &SpinorProductInput,
) -> Result<(), SpinorMldumpError> {
    let n_orb = first.orbitals.band_window.count;
    stream.begin_orbitals(&SpinorOrbitalsBeginV1 {
        band_window_start: i64::try_from(first.orbitals.band_window.start).unwrap_or(0),
        band_window_count: n_orb,
    })?;
    let n_sites = first.source.partition.site_count();
    for (k, (evecs, energies, basis)) in first
        .orbitals
        .eigenvectors
        .iter()
        .zip(&first.orbitals.energies)
        .zip(&first.orbitals.bases)
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
        let pauli = pauli_row_map(n_pw);
        let lo = local_orbital_tables(basis)?;
        let mut match_store = Vec::with_capacity(n_sites);
        for site in 0..n_sites {
            match_store.push(site_match_scratch(basis, site, n_pw)?);
        }
        let mut site_matches = Vec::with_capacity(n_sites);
        for (site, matching) in match_store.iter().enumerate() {
            site_matches.push(SpinorSiteMatchRefV1 {
                site_index: site,
                n_projection: matching.n_projection,
                n_apw_projection: matching.n_apw_projection,
                coordinate: &matching.coordinate,
                signed_kappa: &matching.signed_kappa,
                twice_mu: &matching.twice_mu,
                radial_n: &matching.radial_n,
                matching_coefficients: &matching.coefficients,
            });
        }
        let available = first
            .orbitals
            .available_bands
            .get(k)
            .copied()
            .unwrap_or(n_orb);
        stream.write_orbital_k(&SpinorOrbitalKRefV1 {
            k_index: k,
            available_bands: available,
            basis_dimension: evecs.rows(),
            eigenvalues: &evals,
            eigenvectors: &evec_scratch,
            n_plane_waves: n_pw,
            plane_wave_g: &g,
            plane_wave_k_cartesian: &k_cart,
            plane_wave_q_cartesian: &q_cart,
            pauli_rows: SpinorPauliRowMapRefV1 {
                n_row: pauli.n_row,
                row_index: &pauli.row_index,
                pauli_component: &pauli.pauli_component,
                plane_wave_index: &pauli.plane_wave_index,
            },
            local_orbitals: SpinorLocalOrbitalTableRefV1 {
                n_local_orbitals: lo.n_lo,
                row_index: &lo.row_index,
                site: &lo.site,
                signed_kappa: &lo.signed_kappa,
                twice_mu: &lo.twice_mu,
                ordinal: &lo.ordinal,
                radial_n: &lo.radial_n,
            },
            site_matches: &site_matches,
        })?;
    }
    stream.finish_orbitals()?;
    Ok(())
}

fn write_products(
    stream: &mut muffintin_io::SpinorMldumpStreamV1,
    inputs: &[SpinorProductInput],
) -> Result<(), SpinorMldumpError> {
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
        .unwrap_or("spinor-product");
    let reference = first
        .source
        .provenance
        .reference
        .as_deref()
        .unwrap_or("snapshot-dft-frozen-spinor-ml5b");
    stream.begin_products(&SpinorProductsBeginV1 {
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
        stream.write_product_site(&SpinorProductSiteRefV1 {
            site_index: site,
            n_radial: packed.n_radial,
            n_radial_samples: radials.mesh.len(),
            kind: &packed.kind,
            signed_kappa: &packed.signed_kappa,
            n: &packed.n,
            p: &packed.p,
            q: &packed.q,
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
        stream.write_product_q(&SpinorProductQRecordRefV1 {
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
    stream: &mut muffintin_io::SpinorMldumpStreamV1,
    thc: &SpinorThcResult,
) -> Result<(), SpinorMldumpError> {
    if thc.selection.provenance.strategy != SelectorStrategy::AllQL2 {
        return Err(SpinorMldumpError::UnsupportedStrategy);
    }
    let engine = match thc.selection.provenance.engine {
        L2Engine::FullColumnPivotedQr => MLDUMP_THC_ENGINE_QRCP,
        L2Engine::FullPivotedCholesky => MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY,
        other => return Err(SpinorMldumpError::UnsupportedEngine(other)),
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
                    return Err(SpinorCoulombError::VertexIdentity {
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
    stream: &mut muffintin_io::SpinorMldumpStreamV1,
    coulomb: &SpinorCoulombResult,
) -> Result<(), SpinorMldumpError> {
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

struct PauliTables {
    n_row: usize,
    row_index: Vec<i64>,
    pauli_component: Vec<i64>,
    plane_wave_index: Vec<i64>,
}

fn pauli_row_map(n_pw: usize) -> PauliTables {
    let n_row = N_PAULI * n_pw;
    let mut tables = PauliTables {
        n_row,
        row_index: Vec::with_capacity(n_row),
        pauli_component: Vec::with_capacity(n_row),
        plane_wave_index: Vec::with_capacity(n_row),
    };
    for pauli in 0..N_PAULI {
        for pw in 0..n_pw {
            tables
                .row_index
                .push(i64::try_from(pauli * n_pw + pw).unwrap_or(i64::MIN));
            tables
                .pauli_component
                .push(i64::try_from(pauli).unwrap_or(i64::MIN));
            tables
                .plane_wave_index
                .push(i64::try_from(pw).unwrap_or(i64::MIN));
        }
    }
    tables
}

struct LoTables {
    n_lo: usize,
    row_index: Vec<i64>,
    site: Vec<i64>,
    signed_kappa: Vec<i64>,
    twice_mu: Vec<i64>,
    ordinal: Vec<i64>,
    radial_n: Vec<i64>,
}

fn local_orbital_tables(basis: &SpinorCompiledBasis) -> Result<LoTables, SpinorMldumpError> {
    let mut tables = LoTables {
        n_lo: 0,
        row_index: Vec::new(),
        site: Vec::new(),
        signed_kappa: Vec::new(),
        twice_mu: Vec::new(),
        ordinal: Vec::new(),
        radial_n: Vec::new(),
    };
    for site in 0..basis.layout.site_count() {
        let Some(layout) = basis.layout.site_layout(site) else {
            continue;
        };
        for &(kappa, count) in layout.counts_by_kappa() {
            for twice_mu in kappa.twice_mu_values() {
                for ordinal in 0..count {
                    let row = basis
                        .layout
                        .site_spinor_index(site, kappa, twice_mu, ordinal)
                        .ok_or_else(|| ValidationError::InvalidValue {
                            path: format!("orbitals.local_orbitals site {site}"),
                            expected: "compiled LO/RLO row".to_owned(),
                            actual: format!(
                                "kappa={} twice_mu={} n={}",
                                kappa.get(),
                                twice_mu.get(),
                                ordinal
                            ),
                        })?;
                    tables
                        .row_index
                        .push(i64::try_from(row).unwrap_or(i64::MIN));
                    tables.site.push(i64::try_from(site).unwrap_or(i64::MIN));
                    tables.signed_kappa.push(i64::from(kappa.get()));
                    tables.twice_mu.push(twice_mu.get());
                    tables
                        .ordinal
                        .push(i64::try_from(ordinal).unwrap_or(i64::MIN));
                    tables
                        .radial_n
                        .push(i64::try_from(SPINOR_RADIAL_LO0 + ordinal).unwrap_or(i64::MIN));
                    tables.n_lo += 1;
                }
            }
        }
    }
    Ok(tables)
}

struct SiteMatchScratch {
    n_projection: usize,
    n_apw_projection: usize,
    coordinate: Vec<i64>,
    signed_kappa: Vec<i64>,
    twice_mu: Vec<i64>,
    radial_n: Vec<i64>,
    coefficients: Vec<f64>,
}

fn site_match_scratch(
    basis: &SpinorCompiledBasis,
    site: usize,
    n_pw: usize,
) -> Result<SiteMatchScratch, SpinorMldumpError> {
    let channels = require_exportable_site_augmentation(basis, site)?;
    let waves = basis.site_augmentations.get(site).ok_or_else(|| {
        exportable_basis(
            &format!("orbitals.site_matches[{site}]"),
            "compiled site augmentation".to_owned(),
            "missing".to_owned(),
        )
    })?;
    if waves.len() != n_pw {
        return Err(exportable_basis(
            &format!("orbitals.site_matches[{site}].plane_waves"),
            n_pw.to_string(),
            waves.len().to_string(),
        ));
    }
    let n_channel = channels.len();
    let n_apw = n_channel.saturating_mul(2);
    let mut coordinate = Vec::new();
    let mut signed_kappa = Vec::new();
    let mut twice_mu = Vec::new();
    let mut radial_n = Vec::new();
    let mut coord = 0i64;
    for channel in channels {
        for n in 0..2 {
            coordinate.push(coord);
            signed_kappa.push(i64::from(channel.kappa().get()));
            twice_mu.push(channel.twice_mu().get());
            radial_n.push(n);
            coord += 1;
        }
    }
    if let Some(layout) = basis.layout.site_layout(site) {
        for &(kappa, count) in layout.counts_by_kappa() {
            for mu in kappa.twice_mu_values() {
                for ordinal in 0..count {
                    coordinate.push(coord);
                    signed_kappa.push(i64::from(kappa.get()));
                    twice_mu.push(mu.get());
                    radial_n.push(i64::try_from(SPINOR_RADIAL_LO0 + ordinal).unwrap_or(i64::MIN));
                    coord += 1;
                }
            }
        }
    }
    let mut coefficients = Vec::with_capacity(n_pw * N_PAULI * n_apw * 2);
    for (pw, wave) in waves.iter().enumerate() {
        for pauli in 0..N_PAULI {
            let pauli_coefficients = wave.coefficients.get(pauli).ok_or_else(|| {
                exportable_basis(
                    &format!("orbitals.site_matches[{site}].coefficients[{pw}][{pauli}]"),
                    n_channel.to_string(),
                    "missing Pauli component".to_owned(),
                )
            })?;
            for channel in 0..n_channel {
                let [u, udot] = pauli_coefficients.get(channel).copied().ok_or_else(|| {
                    exportable_basis(
                        &format!(
                            "orbitals.site_matches[{site}].coefficients[{pw}][{pauli}][{channel}]"
                        ),
                        n_channel.to_string(),
                        pauli_coefficients.len().to_string(),
                    )
                })?;
                coefficients.extend_from_slice(&[u.re, u.im, udot.re, udot.im]);
            }
        }
    }
    Ok(SiteMatchScratch {
        n_projection: coordinate.len(),
        n_apw_projection: n_apw,
        coordinate,
        signed_kappa,
        twice_mu,
        radial_n,
        coefficients,
    })
}

struct PackedRadials {
    n_radial: usize,
    kind: Vec<i64>,
    signed_kappa: Vec<i64>,
    n: Vec<i64>,
    p: Vec<f64>,
    q: Vec<f64>,
}

fn pack_site_radials(radials: &[DiracRadial]) -> PackedRadials {
    let n_radial = radials.len();
    let mut packed = PackedRadials {
        n_radial,
        kind: Vec::with_capacity(n_radial),
        signed_kappa: Vec::with_capacity(n_radial),
        n: Vec::with_capacity(n_radial),
        p: Vec::new(),
        q: Vec::new(),
    };
    for radial in radials {
        packed.kind.push(MLDUMP_RADIAL_KIND_VALENCE);
        packed.signed_kappa.push(i64::from(radial.kappa.get()));
        packed.n.push(i64::try_from(radial.n).unwrap_or(i64::MIN));
        packed.p.extend_from_slice(&radial.samples.large);
        packed.q.extend_from_slice(&radial.samples.small);
    }
    packed
}

fn flatten_eigenvectors(
    evecs: &DenseEigenvectors,
    n_orb: usize,
) -> Result<Vec<f64>, SpinorMldumpError> {
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

fn interstitial_volume(input: &SpinorProductInput) -> f64 {
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

#[cfg(test)]
#[path = "../tests/spinor_hydrogen.rs"]
mod spinor_hydrogen;

#[cfg(test)]
mod export_oracles {
    use super::*;
    use crate::{SnapshotDftPhysics, SpinorCoulombSpec, build_spinor_coulomb, build_spinor_thc};
    use muffintin_coulomb::{AuxiliaryKind, assemble_point_charge_oracle};
    use muffintin_io::{
        MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION, MldumpGeometryV1, MldumpKMinusQV1,
        MldumpKPointV1, MldumpMeshV1, MldumpMetaV1, MldumpQEntryV1, MldumpRadialMeshV1,
        MldumpSiteV1,
    };

    use super::spinor_hydrogen::{
        LATTICE, coulomb_spec, hydrogen_spinor_snapshot, parent_grid, spinor_config, thc_spec,
    };

    fn header_from_inputs(inputs: &[SpinorProductInput]) -> MldumpHeaderV1 {
        let first = &inputs[0];
        let spec = coulomb_spec();
        let cell = spec.request.cell();
        let n_k = first.orbitals.k_fractional.len();
        let weight = 1.0 / n_k as f64;
        let sites = first
            .source
            .partition
            .sites()
            .iter()
            .zip(&first.source.radials)
            .enumerate()
            .map(|(index, (site, radials))| MldumpSiteV1 {
                species: Some("H".to_owned()),
                label: if index == 0 {
                    Some("H-1".to_owned())
                } else {
                    None
                },
                position_bohr: site.position.map(|component| component.get()),
                radius_bohr: site.radius.get(),
                radial_mesh: MldumpRadialMeshV1 {
                    first_bohr: radials.mesh.first().get(),
                    log_increment: radials.mesh.increment(),
                    point_count: radials.mesh.len(),
                },
            })
            .collect();
        MldumpHeaderV1::new(
            MldumpMetaV1 {
                producer_name: "libmuffintin-runtime-spinor-mldump-unit".to_owned(),
                producer_version: "0.1.0".to_owned(),
                source_revision: "d429d60250a092c0cd41c3d562965caf43a62878".to_owned(),
                feature_representation: MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION
                    .to_owned(),
            },
            MldumpGeometryV1 {
                direct_basis_bohr: std::array::from_fn(|row| {
                    std::array::from_fn(|axis| cell.basis()[row][axis].get())
                }),
                reciprocal_basis_inv_bohr: std::array::from_fn(|row| {
                    std::array::from_fn(|axis| first.reciprocal.basis()[row][axis].get())
                }),
                cell_volume_bohr3: cell.volume().get(),
                sites,
            },
            MldumpMeshV1 {
                k_points: first
                    .orbitals
                    .k_fractional
                    .iter()
                    .map(|fractional| MldumpKPointV1 {
                        fractional: *fractional,
                        weight,
                    })
                    .collect(),
                q_entries: inputs
                    .iter()
                    .map(|input| {
                        let umklapp = input.source.q.umklapp.index;
                        let canonical = [
                            input.source.q.cartesian[0].get() * LATTICE / (2.0 * PI),
                            input.source.q.cartesian[1].get() * LATTICE / (2.0 * PI),
                            input.source.q.cartesian[2].get() * LATTICE / (2.0 * PI),
                        ];
                        let input_fractional = [
                            canonical[0] + f64::from(umklapp[0]),
                            canonical[1] + f64::from(umklapp[1]),
                            canonical[2] + f64::from(umklapp[2]),
                        ];
                        MldumpQEntryV1 {
                            input_fractional,
                            canonical_fractional: canonical,
                            global_umklapp: umklapp,
                            k_minus_q: input
                                .k_minus_q
                                .iter()
                                .map(|mapped| MldumpKMinusQV1 {
                                    k_index: mapped.k_index,
                                    mapped_index: mapped.kq_index,
                                    g_wrap: mapped.umklapp.index,
                                })
                                .collect(),
                        }
                    })
                    .collect(),
            },
        )
    }

    fn two_k_path() -> (
        Vec<SpinorProductInput>,
        crate::SpinorThcResult,
        crate::SpinorCoulombResult,
        SpinorCoulombSpec,
    ) {
        let physics = SnapshotDftPhysics::new(&hydrogen_spinor_snapshot()).unwrap();
        let q0 = physics
            .spinor_product_input(&spinor_config([2, 1, 1], 0.5), [0.0; 3])
            .unwrap();
        let q15 = physics
            .spinor_product_input(&spinor_config([2, 1, 1], 0.5), [1.5, 0.0, 0.0])
            .unwrap();
        let grid = parent_grid(&q15);
        let inputs = vec![q0, q15];
        let thc = build_spinor_thc(&inputs, &grid, &thc_spec()).unwrap();
        let spec = coulomb_spec();
        let coulomb = build_spinor_coulomb(&inputs, &thc, &spec, &[]).unwrap();
        (inputs, thc, coulomb, spec)
    }

    #[test]
    fn write_spinor_mldump_rejects_replaced_operator_before_create() {
        let path = std::env::temp_dir().join("libmuffintin-runtime-spinor-mldump-operator.h5");
        let _ = std::fs::remove_file(&path);
        let (inputs, thc, mut coulomb, spec) = two_k_path();
        let header = header_from_inputs(&inputs);
        let request = spec
            .request
            .clone()
            .with_interpolation(spec.projection)
            .unwrap();
        let original = &coulomb.records[0];
        let oracle = assemble_point_charge_oracle(&original.auxiliary, &request).unwrap();
        assert_eq!(oracle.dimension(), original.operator.dimension());
        assert_eq!(oracle.q(), original.q);
        assert_eq!(oracle.cell(), spec.request.cell());
        assert_eq!(oracle.reciprocal(), spec.request.reciprocal());
        assert_eq!(oracle.layout(), &original.auxiliary.layout());
        assert_eq!(oracle.kind(), AuxiliaryKind::PointChargeOracle);
        coulomb.records[0].operator = oracle;
        let error =
            write_spinor_mldump(&path, &header, &inputs, &thc, &coulomb, &spec).unwrap_err();
        match error {
            SpinorMldumpError::Coulomb(SpinorCoulombError::CoulombRecord { index }) => {
                assert_eq!(index, 0);
            }
            other => panic!("expected Coulomb record mismatch, got {other}"),
        }
        assert!(
            !path.exists(),
            "replaced Coulomb operator must not create {}",
            path.display()
        );
    }
}
