//! Selected-band spinor mixed-product bridge from frozen [`SpinorProductInput`].

use crate::spinor_product::{SpinorBandWindow, SpinorProductInput};
use muffintin_auxiliary_ir::{
    AuxiliaryPartition, CompiledAuxiliaryBasis, DiracChargeSector, DiracMtPairSpec, DiracRadialId,
    DiracRawProductSpace, InterstitialPairSpec, OrbitalPair, PairColumnLayout, PairVertex,
    TransferQ,
};
use muffintin_core::{InverseBohr, ReciprocalLattice, RelativisticChannel};
use muffintin_envelope::site_translation_phase;
use muffintin_lapw::SpinorCompiledBasis;
use muffintin_mpb::{
    DiracBlochVertexAccumulator, MpbError, apply_dirac_overlap_cutoff,
    untruncated_dirac_product_space,
};
use muffintin_operators::{CompiledSiteProjection, OperatorError};
use muffintin_tensor::DenseEigenvectors;
use num_complex::Complex64;
use std::collections::{HashMap, HashSet};
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
}

/// Construct the Dirac mixed-product basis and selected spinor-band vertices.
///
/// Raw PP/QQ products come from [`SpinorProductInput::source`]. Overlap cutoff
/// uses [`SPINOR_MPB_NSPIN`]. Muffin-tin contraction projects every site
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
    let raw = untruncated_dirac_product_space(&input.source, spec.product_l_max)?;
    let auxiliary = apply_dirac_overlap_cutoff(
        &raw,
        &input.source,
        spec.overlap_tolerance,
        SPINOR_MPB_NSPIN,
        &input.reciprocal,
        spec.product_g_max,
    )?;
    let relative_g_by_index = input
        .source
        .interstitial_pair_support
        .components
        .iter()
        .map(|component| (component.g_relative.index, component.g_relative))
        .collect::<HashMap<_, _>>();
    let mut vertices = Vec::with_capacity(spec.selections.len());
    for selection in &spec.selections {
        vertices.push(contract_selection(
            input,
            &raw,
            &auxiliary,
            &relative_g_by_index,
            *selection,
        )?);
    }
    Ok(SpinorMpbResult {
        raw,
        auxiliary,
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
    relative_g_by_index: &HashMap<[i32; 3], muffintin_core::GVector>,
    selection: SpinorMpbSelection,
) -> Result<SpinorMpbPairVertex, SpinorMpbError> {
    if auxiliary.q != input.source.q || raw.q != input.source.q {
        return Err(MpbError::TransferQMismatch.into());
    }
    if auxiliary.partition != input.source.partition || raw.partition != input.source.partition {
        return Err(MpbError::PartitionMismatch.into());
    }
    let mapped = input
        .k_minus_q
        .iter()
        .copied()
        .find(|mapped| mapped.k_index == selection.k)
        .ok_or(SpinorMpbError::IncompatiblePairLayout)?;
    let pair = BandPair {
        left_band: selection.left_band,
        right_band: selection.right_band,
        left_basis: &input.orbitals.bases[mapped.kq_index],
        right_basis: &input.orbitals.bases[mapped.k_index],
        left_ev: &input.orbitals.eigenvectors[mapped.kq_index],
        right_ev: &input.orbitals.eigenvectors[mapped.k_index],
        wrap: mapped.umklapp,
    };
    if pair.left_ev.rows() != pair.left_basis.layout.dimension()
        || pair.right_ev.rows() != pair.right_basis.layout.dimension()
        || pair.left_ev.columns() != input.orbitals.band_window.count
        || pair.right_ev.columns() != input.orbitals.band_window.count
    {
        return Err(SpinorMpbError::IncompatiblePairLayout);
    }
    let bloch = OrbitalPair::Bloch {
        k_index: selection.k,
        left: selection.left_band,
        right: selection.right_band,
    };
    let mut acc = DiracBlochVertexAccumulator::new(&input.source, raw, auxiliary, bloch)?;
    add_muffin_tin_terms(&mut acc, input, raw, &pair)?;
    add_interstitial_terms(&mut acc, input, relative_g_by_index, &pair)?;
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

struct BandPair<'a> {
    left_band: usize,
    right_band: usize,
    left_basis: &'a SpinorCompiledBasis,
    right_basis: &'a SpinorCompiledBasis,
    left_ev: &'a DenseEigenvectors,
    right_ev: &'a DenseEigenvectors,
    wrap: muffintin_core::GVector,
}

fn add_muffin_tin_terms(
    acc: &mut DiracBlochVertexAccumulator<'_>,
    input: &SpinorProductInput,
    raw: &DiracRawProductSpace,
    pair: &BandPair<'_>,
) -> Result<(), SpinorMpbError> {
    let known_pp = raw_mt_pairs(raw, DiracChargeSector::LargeLarge);
    let known_qq = raw_mt_pairs(raw, DiracChargeSector::SmallSmall);
    for (site, region) in input.source.partition.sites().iter().enumerate() {
        let left_channels = site_channels(pair.left_basis, site)?;
        let right_channels = site_channels(pair.right_basis, site)?;
        let left_proj = CompiledSiteProjection::spinor(pair.left_basis, site, left_channels)?;
        let right_proj = CompiledSiteProjection::spinor(pair.right_basis, site, right_channels)?;
        let left_site = left_proj.project_eigenvectors(pair.left_ev)?;
        let right_site = right_proj.project_eigenvectors(pair.right_ev)?;
        if left_site.coordinate_count() != right_site.coordinate_count() {
            return Err(SpinorMpbError::IncompatiblePairLayout);
        }
        let phase = site_translation_phase(input.source.q.cartesian, region.position).conj();
        for left_coord in 0..left_site.coordinate_count() {
            let Some((left_id, left_mu)) = input.site_projection_identity(site, left_coord) else {
                return Err(SpinorMpbError::IncompatiblePairLayout);
            };
            for right_coord in 0..right_site.coordinate_count() {
                let Some((right_id, right_mu)) = input.site_projection_identity(site, right_coord)
                else {
                    return Err(SpinorMpbError::IncompatiblePairLayout);
                };
                let spec = DiracMtPairSpec {
                    left: left_id,
                    left_twice_mu: left_mu,
                    right: right_id,
                    right_twice_mu: right_mu,
                };
                let amplitude = left_site.at(left_coord, pair.left_band).conj()
                    * right_site.at(right_coord, pair.right_band)
                    * phase;
                if known_pp.contains(&(left_id, right_id)) {
                    acc.add_pp(spec, amplitude)?;
                }
                if known_qq.contains(&(left_id, right_id)) {
                    acc.add_qq(spec, amplitude)?;
                }
            }
        }
    }
    Ok(())
}

fn add_interstitial_terms(
    acc: &mut DiracBlochVertexAccumulator<'_>,
    input: &SpinorProductInput,
    relative_g_by_index: &HashMap<[i32; 3], muffintin_core::GVector>,
    pair: &BandPair<'_>,
) -> Result<(), SpinorMpbError> {
    let volume = input.source.partition.interstitial().cell_volume().get();
    let wrap = pair.wrap.index;
    for (left_g, left_wave) in pair.left_basis.plane_waves.iter().enumerate() {
        for (right_g, right_wave) in pair.right_basis.plane_waves.iter().enumerate() {
            let index = [
                right_wave.g.index[0] - left_wave.g.index[0] + wrap[0],
                right_wave.g.index[1] - left_wave.g.index[1] + wrap[1],
                right_wave.g.index[2] - left_wave.g.index[2] + wrap[2],
            ];
            let g_relative = relative_g_by_index
                .get(&index)
                .copied()
                .ok_or(MpbError::UnknownInterstitialPair { g: index })?;
            let mut amplitude = Complex64::default();
            for spin in 0..2 {
                let left_row = pair
                    .left_basis
                    .layout
                    .plane_wave_index(spin, left_g)
                    .ok_or(SpinorMpbError::IncompatiblePairLayout)?;
                let right_row = pair
                    .right_basis
                    .layout
                    .plane_wave_index(spin, right_g)
                    .ok_or(SpinorMpbError::IncompatiblePairLayout)?;
                amplitude += pair.left_ev.at(left_row, pair.left_band).conj()
                    * pair.right_ev.at(right_row, pair.right_band);
            }
            amplitude /= volume;
            acc.add_interstitial(InterstitialPairSpec {
                g_relative,
                amplitude,
            })?;
        }
    }
    Ok(())
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
/// pair-column layout, reciprocal lattice, band window, and per-k orbital
/// counts. This is not a hash of eigenvector coefficients.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpinorFrozenInputIdentity {
    q: TransferQ,
    partition: AuxiliaryPartition,
    pair_columns: PairColumnLayout,
    reciprocal: ReciprocalLattice,
    band_window: SpinorBandWindow,
    available_bands: Vec<usize>,
}

impl SpinorFrozenInputIdentity {
    pub(crate) fn matches(&self, input: &SpinorProductInput) -> bool {
        self.q == input.source.q
            && self.partition == input.source.partition
            && self.pair_columns == input.pair_columns
            && self.reciprocal == input.reciprocal
            && self.band_window == input.orbitals.band_window
            && self.available_bands == input.orbitals.available_bands
    }
}

fn spinor_frozen_input_identity(input: &SpinorProductInput) -> SpinorFrozenInputIdentity {
    SpinorFrozenInputIdentity {
        q: input.source.q,
        partition: input.source.partition.clone(),
        pair_columns: input.pair_columns,
        reciprocal: input.reciprocal,
        band_window: input.orbitals.band_window,
        available_bands: input.orbitals.available_bands.clone(),
    }
}
