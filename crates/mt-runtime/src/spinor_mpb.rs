//! Selected-band spinor mixed-product bridge from frozen [`SpinorProductInput`].

use crate::hf_diagnostics::HfPhaseTimer;
use crate::spinor_product::{
    SpinorCoreTable, SpinorKMinusQ, SpinorProductInput, spinor_pair_site_phases,
};
use muffintin_core::{InverseBohr, ReciprocalLattice, RelativisticChannel};
use muffintin_operators::lapw::SpinorCompiledBasis;
use muffintin_operators::{CompiledSiteProjection, OperatorError, SiteOrbitalCoefficients};
use muffintin_prodbasis::mpb::{
    DiracMtSectorTable, DiracProductMode, DiracVertexContext, MpbError, apply_dirac_overlap_cutoff,
    untruncated_dirac_product_space,
};
use muffintin_prodbasis::{
    AuxiliaryPartition, CompiledAuxiliaryBasis, DiracChargeSector, DiracMtPairSpec, DiracRadialId,
    DiracRawProductSpace, OrbitalPair, PairColumnLayout, PairVertex, TransferQ,
};
use muffintin_tensor::{Axis, ComplexTensor, TensorError, einsum};
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use thiserror::Error;

/// SPEX overlap-cutoff spin factor for one spinor band manifold (`nspin = 1`).
pub const SPINOR_MPB_NSPIN: f64 = 1.0;

/// Explicit mixed-product construction and spinor band-pair selection.
///
/// The reciprocal lattice is the frozen [`SpinorProductInput::reciprocal`];
/// callers do not supply a second lattice.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorMpbSpec {
    /// Maximum coupled muffin-tin $L$ of the raw Dirac mixed-product space.
    pub product_l_max: u32,
    /// Auxiliary interstitial cutoff $|q+G|\le g_{\mathrm{cut}}$.
    pub product_g_max: InverseBohr,
    /// SPEX `TOL` applied with [`SPINOR_MPB_NSPIN`].
    pub overlap_tolerance: f64,
    /// Nonempty selections `(k, left_band, right_band)` in the spinor product-input window.
    ///
    /// `left_band` is the orbital at the mapped $k-q$ side; `right_band` is
    /// the orbital at $k$. There is no collinear spin tag.
    pub selections: Vec<SpinorMpbSelection>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpinorMpbBasis {
    source: muffintin_prodbasis::DiracProductSource,
    raw: DiracRawProductSpace,
    pub(crate) auxiliary: CompiledAuxiliaryBasis,
    reciprocal: ReciprocalLattice,
    pair_columns: PairColumnLayout,
    product_l_max: u32,
    product_g_max: InverseBohr,
    overlap_tolerance: f64,
    interstitial_theta: ComplexTensor,
    mt_coordinate_tensors: Vec<ComplexTensor>,
}

/// One spinor band pair at one k-point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpinorMpbSelection {
    pub k: usize,
    pub left_band: usize,
    pub right_band: usize,
}

/// Mixed-product output for one requested transfer.
///
/// Construction seals a runtime-private frozen-input identity so a later
/// Coulomb match cannot pair this result with an unrelated
/// [`SpinorProductInput`] that merely shares cell, $q$, and layout. The
/// stamp is not present in a public constructor or external struct
/// literal.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorMpbResult {
    /// Untruncated Dirac mixed-product space, before `TOL`.
    pub raw: DiracRawProductSpace,
    /// Overlap-cutoff retained mixed-product auxiliary basis.
    pub auxiliary: CompiledAuxiliaryBasis,
    /// Authoritative reciprocal lattice copied from the frozen input.
    pub reciprocal: ReciprocalLattice,
    /// Pair-column layout copied from the frozen input.
    pub pair_columns: PairColumnLayout,
    /// Selected band-pair vertices, in spec order.
    pub vertices: Vec<SpinorMpbPairVertex>,
    frozen_input: SpinorFrozenInputIdentity,
}

impl SpinorMpbResult {
    pub(crate) fn frozen_input_identity(&self) -> &SpinorFrozenInputIdentity {
        &self.frozen_input
    }
}

/// One selected spinor band-pair expansion onto the retained MPB.
///
/// The checked vertex identity is [`OrbitalPair::Bloch`]. There is no
/// collinear spin field.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorMpbPairVertex {
    pub k: usize,
    /// Pair-column index $k\cdot N_{\mathrm{orb}}^2+i\cdot N_{\mathrm{orb}}+j$
    /// with $i$ the $k-q$ band and $j$ the $k$ band.
    pub column: usize,
    pub left_band: usize,
    pub right_band: usize,
    pub vertex: PairVertex,
}

/// Spinor mixed-product stage-boundary error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SpinorMpbError {
    #[error(transparent)]
    Mpb(#[from] MpbError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("spinor MPB selections must be nonempty")]
    EmptySelection,
    #[error(
        "spinor MPB selection k={k} left={left_band} right={right_band} is outside the frozen product input"
    )]
    InvalidSelection {
        k: usize,
        left_band: usize,
        right_band: usize,
    },
    #[error("spinor MPB pair-column layout is incompatible with the frozen orbitals")]
    IncompatiblePairLayout,
    #[error("spinor MPB static basis does not match the current fixed-potential input")]
    IncompatibleBasisContext,
}

/// Construct the Dirac mixed-product basis and selected spinor-band vertices.
///
/// Raw PP/QQ products are the valence–valence pairs of
/// [`SpinorProductInput::source`] (SPEX `mixedbasis` mode 1); core radials in
/// the source do not enter this basis because only valence band pairs are
/// expanded on it. Overlap cutoff uses [`SPINOR_MPB_NSPIN`]. Muffin-tin contraction projects every site
/// coordinate of the exact per-$k$ [`SpinorCompiledBasis`]: large coordinates
/// onto PP/$\Omega_\kappa$ and small coordinates onto QQ/$\Omega_{-\kappa}$.
/// Interstitial contraction is the same-component Pauli sum
/// $\sum_s\mathrm{conj}(C_{\mathrm{left}}[s,G_{\mathrm{left}}])
/// C_{\mathrm{right}}[s,G_{\mathrm{right}}]/\Omega$ at
/// $G_{\mathrm{rel}}=G_{\mathrm{right}}-G_{\mathrm{left}}+G_{\mathrm{wrap}}$.
pub fn build_spinor_mpb(
    input: &SpinorProductInput,
    spec: &SpinorMpbSpec,
) -> Result<SpinorMpbResult, SpinorMpbError> {
    let basis = compile_spinor_mpb_basis(input, spec)?;
    build_spinor_mpb_from_basis(input, spec, &basis)
}

pub(crate) fn compile_spinor_mpb_basis(
    input: &SpinorProductInput,
    spec: &SpinorMpbSpec,
) -> Result<SpinorMpbBasis, SpinorMpbError> {
    let validation_timer = HfPhaseTimer::new("exchange.cache.vv_basis.validate");
    input
        .validate()
        .map_err(|_| SpinorMpbError::IncompatiblePairLayout)?;
    require_compatible_layout(input)?;
    drop(validation_timer);
    let raw_timer = HfPhaseTimer::new("exchange.cache.vv_basis.raw_dirac_space");
    let raw = untruncated_dirac_product_space(
        &input.source,
        spec.product_l_max,
        DiracProductMode::ValenceValence,
    )?;
    drop(raw_timer);
    let overlap_timer = HfPhaseTimer::new("exchange.cache.vv_basis.overlap_cutoff");
    let auxiliary = apply_dirac_overlap_cutoff(
        &raw,
        &input.source,
        spec.overlap_tolerance,
        SPINOR_MPB_NSPIN,
        &input.reciprocal,
        spec.product_g_max,
    )?;
    drop(overlap_timer);
    let context_timer = HfPhaseTimer::new("exchange.cache.vv_basis.vertex_context");
    let context = DiracVertexContext::new(&input.source, &raw, &auxiliary)?;
    drop(context_timer);
    let table_timer = HfPhaseTimer::new("exchange.cache.vv_basis.interstitial_table");
    let interstitial_table = context.interstitial_table()?;
    drop(table_timer);
    let theta_timer = HfPhaseTimer::new("exchange.cache.vv_basis.interstitial_theta");
    let pair_support = &input.source.interstitial_pair_support.components;
    let mut theta = Vec::new();
    for component in pair_support {
        theta.extend_from_slice(interstitial_table.row(&auxiliary, component.g_relative.index)?);
    }
    let interstitial_theta = ComplexTensor::from_host_row_major(
        &[
            pair_support.len(),
            auxiliary.dimension() - auxiliary.mt_dimension(),
        ],
        &[Axis::Auxiliary, Axis::Auxiliary],
        theta,
    )?;
    drop(theta_timer);
    let mt_timer = HfPhaseTimer::new("exchange.cache.vv_basis.mt_coordinate_tensors");
    let known_pp = raw_mt_pairs(&raw, DiracChargeSector::LargeLarge);
    let known_qq = raw_mt_pairs(&raw, DiracChargeSector::SmallSmall);
    let mut table = context.sector_table();
    let mt_coordinate_tensors = compile_mt_coordinate_tensors(
        input,
        &mut table,
        auxiliary.mt_dimension(),
        &known_pp,
        &known_qq,
    )?;
    drop(mt_timer);
    Ok(SpinorMpbBasis {
        source: input.source.clone(),
        raw,
        auxiliary,
        reciprocal: input.reciprocal,
        pair_columns: input.pair_columns,
        product_l_max: spec.product_l_max,
        product_g_max: spec.product_g_max,
        overlap_tolerance: spec.overlap_tolerance,
        interstitial_theta,
        mt_coordinate_tensors,
    })
}

pub(crate) fn build_spinor_mpb_from_basis(
    input: &SpinorProductInput,
    spec: &SpinorMpbSpec,
    basis: &SpinorMpbBasis,
) -> Result<SpinorMpbResult, SpinorMpbError> {
    let _total_timer = HfPhaseTimer::new("vv.mpb_rebuild");
    let validation_timer = HfPhaseTimer::new("vv.validate_frozen_inputs");
    if spec.selections.is_empty() {
        return Err(SpinorMpbError::EmptySelection);
    }
    input
        .validate()
        .map_err(|_| SpinorMpbError::IncompatiblePairLayout)?;
    require_compatible_layout(input)?;
    for selection in &spec.selections {
        require_selection(input, *selection)?;
    }
    if basis.source != input.source
        || basis.reciprocal != input.reciprocal
        || basis.pair_columns != input.pair_columns
        || basis.product_l_max != spec.product_l_max
        || basis.product_g_max != spec.product_g_max
        || basis.overlap_tolerance != spec.overlap_tolerance
    {
        return Err(SpinorMpbError::IncompatibleBasisContext);
    }
    drop(validation_timer);
    let raw = &basis.raw;
    let auxiliary = &basis.auxiliary;
    let context = DiracVertexContext::new(&input.source, &raw, &auxiliary)?;
    let projection_timer = HfPhaseTimer::new("vv.site_projection");
    let projected_by_k = input
        .k_minus_q
        .iter()
        .copied()
        .map(|mapped| Ok((mapped.k_index, project_k_sites(input, mapped)?)))
        .collect::<Result<Vec<_>, SpinorMpbError>>()?;
    drop(projection_timer);
    let mt_timer = HfPhaseTimer::new("vv.mt_contraction");
    let muffin_tin_vertices = contract_muffin_tin_selections(
        input,
        spec,
        auxiliary.dimension(),
        &projected_by_k,
        &basis.mt_coordinate_tensors,
    )?;
    drop(mt_timer);
    let interstitial_vertices =
        contract_interstitial_selections(input, spec, &basis.interstitial_theta)?;
    let _assembly_timer = HfPhaseTimer::new("vv.vertex_assembly_and_frozen_identity");
    let vertices = spec
        .selections
        .par_iter()
        .zip(muffin_tin_vertices.par_iter())
        .zip(interstitial_vertices.par_iter())
        .map(|((selection, muffin_tin), interstitial)| {
            contract_selection(
                input,
                &raw,
                &auxiliary,
                context,
                muffin_tin,
                interstitial,
                *selection,
            )
        })
        .collect::<Result<Vec<_>, SpinorMpbError>>()?;
    Ok(SpinorMpbResult {
        raw: raw.clone(),
        auxiliary: auxiliary.clone(),
        reciprocal: input.reciprocal,
        pair_columns: input.pair_columns,
        vertices,
        frozen_input: spinor_frozen_input_identity(input),
    })
}

fn require_compatible_layout(input: &SpinorProductInput) -> Result<(), SpinorMpbError> {
    let n_k = input.orbitals.k_fractional.len();
    let n_orb = input.orbitals.band_window.count;
    let compatible = input.orbitals.band_window.start == 0
        && input.pair_columns.n_k == n_k
        && input.pair_columns.n_orb == n_orb
        && input.k_minus_q.len() == n_k
        && input.orbitals.eigenvectors.len() == n_k
        && input.orbitals.bases.len() == n_k
        && input
            .orbitals
            .eigenvectors
            .iter()
            .zip(&input.orbitals.bases)
            .all(|(evecs, basis)| {
                evecs.columns() == n_orb && evecs.rows() == basis.layout.dimension()
            });
    if compatible {
        Ok(())
    } else {
        Err(SpinorMpbError::IncompatiblePairLayout)
    }
}

fn require_selection(
    input: &SpinorProductInput,
    selection: SpinorMpbSelection,
) -> Result<(), SpinorMpbError> {
    let n_k = input.orbitals.k_fractional.len();
    let n_orb = input.orbitals.band_window.count;
    let valid = selection.k < n_k && selection.left_band < n_orb && selection.right_band < n_orb;
    if valid {
        Ok(())
    } else {
        Err(SpinorMpbError::InvalidSelection {
            k: selection.k,
            left_band: selection.left_band,
            right_band: selection.right_band,
        })
    }
}

fn contract_selection(
    input: &SpinorProductInput,
    raw: &DiracRawProductSpace,
    auxiliary: &CompiledAuxiliaryBasis,
    context: DiracVertexContext<'_>,
    muffin_tin: &[Complex64],
    interstitial: &[Complex64],
    selection: SpinorMpbSelection,
) -> Result<SpinorMpbPairVertex, SpinorMpbError> {
    if auxiliary.q != input.source.q || raw.q != input.source.q {
        return Err(MpbError::TransferQMismatch.into());
    }
    if auxiliary.partition != input.source.partition || raw.partition != input.source.partition {
        return Err(MpbError::PartitionMismatch.into());
    }
    let bloch = OrbitalPair::Bloch {
        k_index: selection.k,
        left: selection.left_band,
        right: selection.right_band,
    };
    let mut acc = context.bloch_accumulator(bloch)?;
    let mut coefficients = muffin_tin.to_vec();
    for (target, value) in coefficients[auxiliary.mt_dimension()..]
        .iter_mut()
        .zip(interstitial)
    {
        *target += value;
    }
    acc.add_auxiliary_coefficients(&coefficients)?;
    let vertex = acc.finish()?;
    if vertex.pair() != bloch {
        return Err(SpinorMpbError::IncompatiblePairLayout);
    }
    Ok(SpinorMpbPairVertex {
        k: selection.k,
        column: input
            .pair_columns
            .encode(selection.k, selection.left_band, selection.right_band),
        left_band: selection.left_band,
        right_band: selection.right_band,
        vertex,
    })
}

struct ProjectedSitePair {
    left: SiteOrbitalCoefficients,
    right: SiteOrbitalCoefficients,
}

fn project_k_sites(
    input: &SpinorProductInput,
    mapped: SpinorKMinusQ,
) -> Result<Vec<ProjectedSitePair>, SpinorMpbError> {
    let left_basis = &input.orbitals.bases[mapped.kq_index];
    let right_basis = &input.orbitals.bases[mapped.k_index];
    let left_ev = &input.orbitals.eigenvectors[mapped.kq_index];
    let right_ev = &input.orbitals.eigenvectors[mapped.k_index];
    let mut sites = Vec::with_capacity(input.source.partition.site_count());
    for site in 0..input.source.partition.site_count() {
        let left_channels = site_channels(left_basis, site)?;
        let right_channels = site_channels(right_basis, site)?;
        let left = CompiledSiteProjection::spinor(left_basis, site, left_channels)?
            .project_eigenvectors(left_ev)?;
        let right = CompiledSiteProjection::spinor(right_basis, site, right_channels)?
            .project_eigenvectors(right_ev)?;
        sites.push(ProjectedSitePair {
            left: normalized_mpb_site_projection(input, site, left)?,
            right: normalized_mpb_site_projection(input, site, right)?,
        });
    }
    Ok(sites)
}

/// MPB primitives use P/sqrt(N), Q/sqrt(N), while LAPW projection rows
/// multiply the original P,Q. Change coefficients to d_hat=sqrt(N)*d.
/// This affects only MT coordinates, never interstitial plane-wave amplitudes.
pub(crate) fn normalized_mpb_site_projection(
    input: &SpinorProductInput,
    site: usize,
    projected: SiteOrbitalCoefficients,
) -> Result<SiteOrbitalCoefficients, SpinorMpbError> {
    let coordinates = projected.coordinate_count();
    let bands = projected.band_count();
    let mut values = projected.to_host_row_major();
    for (coordinate, row) in values.chunks_exact_mut(bands).enumerate() {
        let (id, _) = input
            .site_projection_identity(site, coordinate)
            .ok_or(SpinorMpbError::IncompatiblePairLayout)?;
        let radial = input
            .source
            .find_radial(id)
            .ok_or(SpinorMpbError::IncompatiblePairLayout)?;
        let norm = match radial.normalization {
            muffintin_prodbasis::DiracRadialNormalization::Explicit(value) => value,
            muffintin_prodbasis::DiracRadialNormalization::OnMesh => {
                let integrand = radial
                    .samples
                    .large
                    .iter()
                    .zip(&radial.samples.small)
                    .map(|(p, q)| p * p + q * q)
                    .collect::<Vec<_>>();
                input.source.radials[site]
                    .mesh
                    .integrate(&integrand)
                    .map_err(MpbError::from)?
            }
        };
        let scale = norm.sqrt();
        for value in row {
            *value *= scale;
        }
    }
    Ok(SiteOrbitalCoefficients::from_tensor(
        ComplexTensor::from_host_row_major(
            &[coordinates, bands],
            &[Axis::SiteCoordinate, Axis::Band],
            values,
        )?,
    )?)
}

fn compile_mt_coordinate_tensors(
    input: &SpinorProductInput,
    table: &mut DiracMtSectorTable<'_>,
    mt_dimension: usize,
    known_pp: &HashSet<(DiracRadialId, DiracRadialId)>,
    known_qq: &HashSet<(DiracRadialId, DiracRadialId)>,
) -> Result<Vec<ComplexTensor>, SpinorMpbError> {
    let basis = input
        .orbitals
        .bases
        .first()
        .ok_or(SpinorMpbError::IncompatiblePairLayout)?;
    let mut sites = Vec::with_capacity(input.source.partition.site_count());
    for site in 0..input.source.partition.site_count() {
        let channels = site_channels(basis, site)?;
        let coordinate_count =
            CompiledSiteProjection::spinor(basis, site, channels)?.coordinate_count();
        // MT pair kernels have no interstitial auxiliary support. Do not
        // allocate those zero rows in the static site-coordinate cache.
        let auxiliary_count = mt_dimension;
        let mut coefficients =
            vec![Complex64::default(); auxiliary_count * coordinate_count * coordinate_count];
        for left in 0..coordinate_count {
            let (left_id, left_twice_mu) = input
                .site_projection_identity(site, left)
                .ok_or(SpinorMpbError::IncompatiblePairLayout)?;
            for right in 0..coordinate_count {
                let (right_id, right_twice_mu) = input
                    .site_projection_identity(site, right)
                    .ok_or(SpinorMpbError::IncompatiblePairLayout)?;
                let mut sectors = Vec::with_capacity(2);
                if known_pp.contains(&(left_id, right_id)) {
                    sectors.push(DiracChargeSector::LargeLarge);
                }
                if known_qq.contains(&(left_id, right_id)) {
                    sectors.push(DiracChargeSector::SmallSmall);
                }
                if sectors.is_empty() {
                    continue;
                }
                let compiled = table.compile_pair(
                    &input.source,
                    DiracMtPairSpec {
                        left: left_id,
                        left_twice_mu,
                        right: right_id,
                        right_twice_mu,
                    },
                    &sectors,
                )?;
                for &(auxiliary, factor) in compiled.coefficients() {
                    coefficients
                        [(auxiliary * coordinate_count + left) * coordinate_count + right] +=
                        factor;
                }
            }
        }
        sites.push(ComplexTensor::from_host_row_major(
            &[auxiliary_count, coordinate_count, coordinate_count],
            &[Axis::Auxiliary, Axis::SiteCoordinate, Axis::SiteCoordinate],
            coefficients,
        )?);
    }
    Ok(sites)
}

fn contract_muffin_tin_selections(
    input: &SpinorProductInput,
    spec: &SpinorMpbSpec,
    auxiliary_count: usize,
    projected_by_k: &[(usize, Vec<ProjectedSitePair>)],
    mt_coordinate_tensors: &[ComplexTensor],
) -> Result<Vec<Vec<Complex64>>, SpinorMpbError> {
    let mut by_selection = HashMap::new();
    let selected_k = spec
        .selections
        .iter()
        .map(|selection| selection.k)
        .collect::<BTreeSet<_>>();
    for k in selected_k {
        let projected = projected_by_k
            .iter()
            .find(|(projected_k, _)| *projected_k == k)
            .map(|(_, projected)| projected.as_slice())
            .ok_or(SpinorMpbError::IncompatiblePairLayout)?;
        if projected.len() != mt_coordinate_tensors.len() {
            return Err(SpinorMpbError::IncompatiblePairLayout);
        }
        let left_bands = spec
            .selections
            .iter()
            .filter(|selection| selection.k == k)
            .map(|selection| selection.left_band)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let right_bands = spec
            .selections
            .iter()
            .filter(|selection| selection.k == k)
            .map(|selection| selection.right_band)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mapped = input
            .k_minus_q
            .iter()
            .copied()
            .find(|mapped| mapped.k_index == k)
            .ok_or(SpinorMpbError::IncompatiblePairLayout)?;
        let mut contracted =
            vec![Complex64::default(); auxiliary_count * left_bands.len() * right_bands.len()];
        for (site, (projected, coordinate_tensor)) in
            projected.iter().zip(mt_coordinate_tensors).enumerate()
        {
            let left = select_site_bands(&projected.left, &left_bands)?;
            let right = select_site_bands(&projected.right, &right_bands)?;
            let site_contracted = einsum(
                "il,aij,jr->alr",
                &[&left.conjugate(), coordinate_tensor, &right],
            )?
            .to_host_row_major();
            let phase = spinor_pair_site_phases(input, mapped, site)
                .ok_or(SpinorMpbError::IncompatiblePairLayout)?
                .auxiliary_compensation;
            for (target, value) in contracted.iter_mut().zip(site_contracted) {
                *target += phase * value;
            }
        }
        for (left_position, &left_band) in left_bands.iter().enumerate() {
            for (right_position, &right_band) in right_bands.iter().enumerate() {
                let coefficients = (0..auxiliary_count)
                    .map(|auxiliary| {
                        contracted[(auxiliary * left_bands.len() + left_position)
                            * right_bands.len()
                            + right_position]
                    })
                    .collect();
                by_selection.insert((k, left_band, right_band), coefficients);
            }
        }
    }
    spec.selections
        .iter()
        .map(|selection| {
            by_selection
                .get(&(selection.k, selection.left_band, selection.right_band))
                .cloned()
                .ok_or(SpinorMpbError::IncompatiblePairLayout)
        })
        .collect()
}

fn select_site_bands(
    projected: &SiteOrbitalCoefficients,
    bands: &[usize],
) -> Result<ComplexTensor, SpinorMpbError> {
    let mut values = Vec::with_capacity(projected.coordinate_count() * bands.len());
    let coefficients = projected.to_host_row_major();
    for row in coefficients.chunks_exact(projected.band_count()) {
        for &band in bands {
            values.push(row[band]);
        }
    }
    Ok(ComplexTensor::from_host_row_major(
        &[projected.coordinate_count(), bands.len()],
        &[Axis::SiteCoordinate, Axis::Band],
        values,
    )?)
}

fn contract_interstitial_selections(
    input: &SpinorProductInput,
    spec: &SpinorMpbSpec,
    theta: &ComplexTensor,
) -> Result<Vec<Vec<Complex64>>, SpinorMpbError> {
    let _total_timer = HfPhaseTimer::new("vv.interstitial");
    let components = &input.source.interstitial_pair_support.components;
    let n_raw = components.len();
    let n_pw = theta.shape()[1];
    if n_raw == 0 || n_pw == 0 {
        return Ok(vec![
            vec![Complex64::default(); n_pw];
            spec.selections.len()
        ]);
    }
    let lookup_timer = HfPhaseTimer::new("vv.interstitial.dense_lookup");
    // Dense reciprocal-index lookup is shared by all k points and bands.
    let mut lower = components[0].g_relative.index;
    let mut upper = lower;
    for component in components {
        for axis in 0..3 {
            lower[axis] = lower[axis].min(component.g_relative.index[axis]);
            upper[axis] = upper[axis].max(component.g_relative.index[axis]);
        }
    }
    let shape: [usize; 3] = std::array::from_fn(|axis| (upper[axis] - lower[axis] + 1) as usize);
    let offset = |g: [i32; 3]| {
        ((g[0] - lower[0]) as usize * shape[1] + (g[1] - lower[1]) as usize) * shape[2]
            + (g[2] - lower[2]) as usize
    };
    let mut relative_positions = vec![usize::MAX; shape.iter().product()];
    for (position, component) in components.iter().enumerate() {
        relative_positions[offset(component.g_relative.index)] = position;
    }
    drop(lookup_timer);
    let mut selections_by_k = BTreeMap::<usize, BTreeMap<usize, Vec<usize>>>::new();
    for (position, selection) in spec.selections.iter().enumerate() {
        selections_by_k
            .entry(selection.k)
            .or_default()
            .entry(selection.left_band)
            .or_default()
            .push(position);
    }
    let volume = input.source.partition.interstitial().cell_volume().get();
    let mut vertices = vec![Vec::new(); spec.selections.len()];
    for (k, selections_by_left) in selections_by_k {
        let preparation_timer =
            HfPhaseTimer::new("vv.interstitial.k_gather_indices_and_coefficients");
        let mapped = input
            .k_minus_q
            .iter()
            .find(|mapped| mapped.k_index == k)
            .ok_or(SpinorMpbError::IncompatiblePairLayout)?;
        let left_basis = &input.orbitals.bases[mapped.kq_index];
        let right_basis = &input.orbitals.bases[mapped.k_index];
        let n_right = right_basis.plane_waves.len();
        let band_count = input.orbitals.band_window.count;
        let right_bands = selections_by_left
            .values()
            .flatten()
            .map(|&position| spec.selections[position].right_band)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mut right_positions = vec![usize::MAX; band_count];
        for (position, &band) in right_bands.iter().enumerate() {
            right_positions[band] = position;
        }

        // Build G_left(G_rel, G_right) once per k, not once per band pair.
        // Missing left sphere entries remain zero in the gathered matrix.
        let mut gather = vec![usize::MAX; n_raw * n_right];
        let wrap = mapped.umklapp.index;
        for (left_g, left_wave) in left_basis.plane_waves.iter().enumerate() {
            for (right_g, right_wave) in right_basis.plane_waves.iter().enumerate() {
                let g = std::array::from_fn(|axis| {
                    right_wave.g.index[axis] - left_wave.g.index[axis] + wrap[axis]
                });
                if (0..3).any(|axis| g[axis] < lower[axis] || g[axis] > upper[axis]) {
                    return Err(MpbError::UnknownInterstitialPair { g }.into());
                }
                let position = relative_positions[offset(g)];
                if position == usize::MAX {
                    return Err(MpbError::UnknownInterstitialPair { g }.into());
                }
                gather[position * n_right + right_g] = left_g;
            }
        }
        let left_values = input.orbitals.eigenvectors[mapped.kq_index].to_host_column_major();
        let right_values = input.orbitals.eigenvectors[mapped.k_index].to_host_row_major();
        let mut left_rows = Vec::with_capacity(2);
        let mut right_blocks = Vec::with_capacity(2);
        for spin in 0..2 {
            left_rows.push(
                (0..left_basis.plane_waves.len())
                    .map(|g| {
                        left_basis
                            .layout
                            .plane_wave_index(spin, g)
                            .ok_or(SpinorMpbError::IncompatiblePairLayout)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let mut coefficients = Vec::with_capacity(n_right * right_bands.len());
            for g in 0..n_right {
                let row = right_basis
                    .layout
                    .plane_wave_index(spin, g)
                    .ok_or(SpinorMpbError::IncompatiblePairLayout)?;
                for &band in &right_bands {
                    coefficients.push(right_values[row * band_count + band]);
                }
            }
            right_blocks.push(ComplexTensor::from_host_row_major(
                &[n_right, right_bands.len()],
                &[Axis::GlobalBasis, Axis::Band],
                coefficients,
            )?);
        }
        drop(preparation_timer);

        // Keep only one left band's raw amplitudes live; parallelizing these
        // blocks would multiply the large G_rel x band temporary by thread count.
        for (left_band, positions) in selections_by_left {
            let mut amplitudes = vec![Complex64::default(); n_raw * right_bands.len()];
            let left_column = &left_values[left_band * left_basis.layout.dimension()
                ..(left_band + 1) * left_basis.layout.dimension()];
            for (rows, right) in left_rows.iter().zip(&right_blocks) {
                let left = rows
                    .iter()
                    .map(|&row| left_column[row].conj())
                    .collect::<Vec<_>>();
                let spin_amplitudes =
                    interstitial_amplitudes(&gather, &left, right, n_raw, n_right)?;
                for (amplitude, spin_amplitude) in amplitudes.iter_mut().zip(spin_amplitudes) {
                    *amplitude += spin_amplitude / volume;
                }
            }
            // Preserve the existing bounded pair-to-Theta contraction and
            // scatter back to the original order, including duplicate selections.
            let theta_timer = HfPhaseTimer::new("vv.interstitial.theta");
            let blocks = positions
                .par_chunks(64)
                .map(|positions| {
                    let mut selected = Vec::with_capacity(positions.len() * n_raw);
                    for &position in positions {
                        let right = right_positions[spec.selections[position].right_band];
                        selected.extend(
                            (0..n_raw).map(|raw| amplitudes[raw * right_bands.len() + right]),
                        );
                    }
                    let selected = ComplexTensor::from_host_row_major(
                        &[positions.len(), n_raw],
                        &[Axis::PairColumn, Axis::Auxiliary],
                        selected,
                    )?;
                    let contracted = einsum("pr,ra->pa", &[&selected, theta])?.to_host_row_major();
                    Ok(contracted
                        .chunks_exact(n_pw)
                        .map(<[Complex64]>::to_vec)
                        .collect::<Vec<_>>())
                })
                .collect::<Result<Vec<_>, SpinorMpbError>>()?;
            for (position, vertex) in positions.into_iter().zip(blocks.into_iter().flatten()) {
                vertices[position] = vertex;
            }
            drop(theta_timer);
        }
    }
    Ok(vertices)
}

fn interstitial_amplitudes(
    gather: &[usize],
    conjugate_left: &[Complex64],
    right: &ComplexTensor,
    n_raw: usize,
    n_right: usize,
) -> Result<Vec<Complex64>, SpinorMpbError> {
    let gather_timer = HfPhaseTimer::new("vv.interstitial.gather");
    let mut values = vec![Complex64::default(); gather.len()];
    for (value, &left_g) in values.iter_mut().zip(gather) {
        if left_g != usize::MAX {
            *value = conjugate_left[left_g];
        }
    }
    let gathered = ComplexTensor::from_host_row_major(
        &[n_raw, n_right],
        &[Axis::Auxiliary, Axis::GlobalBasis],
        values,
    )?;
    drop(gather_timer);
    let _gemm_timer = HfPhaseTimer::new("vv.interstitial.gemm");
    Ok(einsum("rg,gb->rb", &[&gathered, right])?.to_host_row_major())
}

fn site_channels(
    compiled: &SpinorCompiledBasis,
    site: usize,
) -> Result<&[RelativisticChannel], SpinorMpbError> {
    compiled
        .site_augmentations
        .get(site)
        .and_then(|waves| waves.first())
        .map(|wave| wave.channels.as_slice())
        .ok_or(SpinorMpbError::IncompatiblePairLayout)
}

fn raw_mt_pairs(
    raw: &DiracRawProductSpace,
    sector: DiracChargeSector,
) -> HashSet<(DiracRadialId, DiracRadialId)> {
    let mut pairs = HashSet::new();
    for product in &raw.radial_products {
        if product.channel.sector != sector {
            continue;
        }
        pairs.insert((product.channel.left, product.channel.right));
        pairs.insert((product.channel.right, product.channel.left));
    }
    pairs
}

/// Runtime-private identifying fields of the frozen [`SpinorProductInput`]
/// used to construct a [`SpinorMpbResult`].
///
/// Compared by derived [`PartialEq`] on transfer q, product partition,
/// pair-column layout, reciprocal lattice, and the complete frozen orbital
/// payload. Rotating orbitals therefore invalidates every MPB/Coulomb context
/// built from the old coefficients.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpinorFrozenInputIdentity {
    q: TransferQ,
    partition: AuxiliaryPartition,
    pair_columns: PairColumnLayout,
    reciprocal: ReciprocalLattice,
    orbitals: crate::spinor_product::SpinorFrozenOrbitals,
    core: SpinorCoreTable,
}

impl SpinorFrozenInputIdentity {
    pub(crate) fn matches(&self, input: &SpinorProductInput) -> bool {
        self.q == input.source.q
            && self.partition == input.source.partition
            && self.pair_columns == input.pair_columns
            && self.reciprocal == input.reciprocal
            && self.orbitals == input.orbitals
            && self.core == input.core
    }
}

pub(crate) fn spinor_frozen_input_identity(
    input: &SpinorProductInput,
) -> SpinorFrozenInputIdentity {
    SpinorFrozenInputIdentity {
        q: input.source.q,
        partition: input.source.partition.clone(),
        pair_columns: input.pair_columns,
        reciprocal: input.reciprocal,
        orbitals: input.orbitals.clone(),
        core: input.core.clone(),
    }
}
