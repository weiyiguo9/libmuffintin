//! Scalar mixed-product bridge from frozen [`ScalarProductInput`] to MPB vertices.

use crate::scalar_product::{ScalarProductInput, ScalarSpinChannel};
use crate::site_coords::site_coordinate;
use muffintin_auxiliary_ir::{
    CompiledAuxiliaryBasis, InterstitialPairSpec, MtPairSpec, OrbitalPair, PairVertex,
    ProductRadialId, RawProductSpace,
};
use muffintin_core::{InverseBohr, ReciprocalLattice};
use muffintin_envelope::site_translation_phase;
use muffintin_lapw::CompiledBasis;
use muffintin_mpb::{
    MpbError, PairVertexAccumulator, apply_overlap_cutoff, spex_mixed_product_basis,
};
use muffintin_operators::{CompiledSiteProjection, OperatorError};
use muffintin_tensor::DenseEigenvectors;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// SPEX overlap-cutoff spin factor for collinear scalar mixed-product construction (`nspin = 2`).
pub const SCALAR_MPB_NSPIN: f64 = 2.0;

/// Explicit mixed-product construction and same-spin band-pair selection.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarMpbSpec {
    /// Reciprocal lattice used by [`muffintin_mpb::spex_mixed_product_basis`].
    pub lattice: ReciprocalLattice,
    /// Maximum coupled muffin-tin $L$ of the raw mixed-product space.
    pub product_l_max: u32,
    /// Auxiliary interstitial cutoff $|q+G|\le g_{\mathrm{cut}}$.
    pub product_g_max: InverseBohr,
    /// SPEX `TOL` applied with [`SCALAR_MPB_NSPIN`].
    pub overlap_tolerance: f64,
    /// Nonempty same-spin selections `(spin, k, left_band, right_band)`.
    ///
    /// `left_band` is the orbital at the mapped $k-q$ side; `right_band` is
    /// the orbital at $k$. Band indices are in the published scalar product-input window.
    pub selections: Vec<ScalarMpbSelection>,
}

/// One same-spin band pair at one k-point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScalarMpbSelection {
    pub spin: u8,
    pub k: usize,
    pub left_band: usize,
    pub right_band: usize,
}

/// Mixed-product output for one requested transfer.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarMpbResult {
    /// Untruncated SPEX mixed-product space, before `TOL`.
    pub raw: RawProductSpace,
    /// Overlap-cutoff retained mixed-product auxiliary basis.
    pub auxiliary: CompiledAuxiliaryBasis,
    /// Selected band-pair vertices, in spec order.
    pub vertices: Vec<ScalarMpbPairVertex>,
}

/// One selected same-spin band-pair expansion onto the retained MPB.
///
/// Spin is stored here rather than on [`OrbitalPair`]. The checked vertex
/// identity is [`OrbitalPair::Bloch`] with the selected $k$ and band indices.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarMpbPairVertex {
    pub spin: u8,
    pub k: usize,
    /// Pair-column index $k\cdot N_{\mathrm{orb}}^2+i\cdot N_{\mathrm{orb}}+j$
    /// with $i$ the $k-q$ band and $j$ the $k$ band.
    pub column: usize,
    pub left_band: usize,
    pub right_band: usize,
    pub vertex: PairVertex,
}

/// Scalar mixed-product stage-boundary error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ScalarMpbError {
    #[error(transparent)]
    Mpb(#[from] MpbError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error("scalar MPB selections must be nonempty")]
    EmptySelection,
    #[error(
        "scalar MPB selection spin={spin} k={k} left={left_band} right={right_band} is outside the frozen product input"
    )]
    InvalidSelection {
        spin: u8,
        k: usize,
        left_band: usize,
        right_band: usize,
    },
    #[error("scalar MPB pair-column layout is incompatible with the frozen orbitals")]
    IncompatiblePairLayout,
}

/// Construct the SPEX mixed-product basis and selected real-orbital vertices.
///
/// The raw space comes from [`ScalarProductInput::source`]. Overlap cutoff
/// uses [`SCALAR_MPB_NSPIN`]. Vertices contract every APW $u$, APW $\dot u$,
/// and LO site coordinate present in the exact per-$k$ [`CompiledBasis`],
/// plus every PW-only pair with relative label
/// $G_{\mathrm{right}}-G_{\mathrm{left}}+G_{\mathrm{wrap}}$.
pub fn build_scalar_mpb(
    input: &ScalarProductInput,
    spec: &ScalarMpbSpec,
) -> Result<ScalarMpbResult, ScalarMpbError> {
    if spec.selections.is_empty() {
        return Err(ScalarMpbError::EmptySelection);
    }
    require_compatible_layout(input)?;
    for selection in &spec.selections {
        require_selection(input, *selection)?;
    }
    let (raw, _) = spex_mixed_product_basis(
        &input.source,
        spec.product_l_max,
        spec.product_g_max,
        &spec.lattice,
    )?;
    let auxiliary = apply_overlap_cutoff(
        &raw,
        &input.source,
        spec.overlap_tolerance,
        SCALAR_MPB_NSPIN,
        &spec.lattice,
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
    Ok(ScalarMpbResult {
        raw,
        auxiliary,
        vertices,
    })
}

fn require_compatible_layout(input: &ScalarProductInput) -> Result<(), ScalarMpbError> {
    let n_k = input.orbitals.k_fractional.len();
    let n_orb = input.orbitals.band_window.count;
    let compatible = input.orbitals.band_window.start == 0
        && input.pair_columns.n_k == n_k
        && input.pair_columns.n_orb == n_orb
        && input.k_minus_q.len() == n_k
        && input.orbitals.channels.iter().all(|channel| {
            channel.eigenvectors.len() == n_k
                && channel.bases.len() == n_k
                && channel
                    .eigenvectors
                    .iter()
                    .all(|evecs| evecs.columns() == n_orb)
        });
    if compatible {
        Ok(())
    } else {
        Err(ScalarMpbError::IncompatiblePairLayout)
    }
}

fn require_selection(
    input: &ScalarProductInput,
    selection: ScalarMpbSelection,
) -> Result<(), ScalarMpbError> {
    let n_k = input.orbitals.k_fractional.len();
    let n_orb = input.orbitals.band_window.count;
    let valid = input
        .orbitals
        .channels
        .iter()
        .any(|channel| channel.spin == selection.spin)
        && selection.k < n_k
        && selection.left_band < n_orb
        && selection.right_band < n_orb;
    if valid {
        Ok(())
    } else {
        Err(ScalarMpbError::InvalidSelection {
            spin: selection.spin,
            k: selection.k,
            left_band: selection.left_band,
            right_band: selection.right_band,
        })
    }
}

fn contract_selection(
    input: &ScalarProductInput,
    raw: &RawProductSpace,
    auxiliary: &CompiledAuxiliaryBasis,
    relative_g_by_index: &HashMap<[i32; 3], muffintin_core::GVector>,
    selection: ScalarMpbSelection,
) -> Result<ScalarMpbPairVertex, ScalarMpbError> {
    let channel = spin_channel(input, selection.spin, selection)?;
    let mapped = input
        .k_minus_q
        .iter()
        .copied()
        .find(|mapped| mapped.k_index == selection.k)
        .ok_or(ScalarMpbError::IncompatiblePairLayout)?;
    let pair = BandPair {
        spin: selection.spin,
        left_band: selection.left_band,
        right_band: selection.right_band,
        left_basis: &channel.bases[mapped.kq_index],
        right_basis: &channel.bases[mapped.k_index],
        left_ev: &channel.eigenvectors[mapped.kq_index],
        right_ev: &channel.eigenvectors[mapped.k_index],
        wrap: mapped.umklapp,
    };
    let mut acc = PairVertexAccumulator::new(
        &input.source,
        raw,
        auxiliary,
        OrbitalPair::Bloch {
            k_index: selection.k,
            left: selection.left_band,
            right: selection.right_band,
        },
    )?;
    add_muffin_tin_terms(&mut acc, input, raw, &pair)?;
    add_interstitial_terms(&mut acc, input, relative_g_by_index, &pair)?;
    Ok(ScalarMpbPairVertex {
        spin: selection.spin,
        k: selection.k,
        column: input
            .pair_columns
            .encode(selection.k, selection.left_band, selection.right_band),
        left_band: selection.left_band,
        right_band: selection.right_band,
        vertex: acc.finish()?,
    })
}

fn spin_channel(
    input: &ScalarProductInput,
    spin: u8,
    selection: ScalarMpbSelection,
) -> Result<&ScalarSpinChannel, ScalarMpbError> {
    input
        .orbitals
        .channels
        .iter()
        .find(|channel| channel.spin == spin)
        .ok_or(ScalarMpbError::InvalidSelection {
            spin: selection.spin,
            k: selection.k,
            left_band: selection.left_band,
            right_band: selection.right_band,
        })
}

struct BandPair<'a> {
    spin: u8,
    left_band: usize,
    right_band: usize,
    left_basis: &'a CompiledBasis,
    right_basis: &'a CompiledBasis,
    left_ev: &'a DenseEigenvectors,
    right_ev: &'a DenseEigenvectors,
    wrap: muffintin_core::GVector,
}

fn add_muffin_tin_terms(
    acc: &mut PairVertexAccumulator<'_>,
    input: &ScalarProductInput,
    raw: &RawProductSpace,
    pair: &BandPair<'_>,
) -> Result<(), ScalarMpbError> {
    let known = raw_mt_pairs(raw);
    for (site, region) in input.source.partition.sites().iter().enumerate() {
        let left_proj = CompiledSiteProjection::scalar(pair.left_basis, site)?;
        let right_proj = CompiledSiteProjection::scalar(pair.right_basis, site)?;
        let left_site = left_proj.project_eigenvectors(pair.left_ev)?;
        let right_site = right_proj.project_eigenvectors(pair.right_ev)?;
        let phase = site_translation_phase(input.source.q.cartesian, region.position).conj();
        for left_coord in 0..left_site.coordinate_count() {
            let Some((left_id, left_m)) =
                site_coordinate(pair.left_basis, site, pair.spin, left_coord)
            else {
                return Err(ScalarMpbError::IncompatiblePairLayout);
            };
            for right_coord in 0..right_site.coordinate_count() {
                let Some((right_id, right_m)) =
                    site_coordinate(pair.right_basis, site, pair.spin, right_coord)
                else {
                    return Err(ScalarMpbError::IncompatiblePairLayout);
                };
                if !known.contains(&(left_id, right_id)) {
                    continue;
                }
                let amplitude = left_site.at(left_coord, pair.left_band).conj()
                    * right_site.at(right_coord, pair.right_band)
                    * phase;
                acc.add_muffin_tin(
                    MtPairSpec {
                        left: left_id,
                        left_m,
                        right: right_id,
                        right_m,
                    },
                    amplitude,
                )?;
            }
        }
    }
    Ok(())
}

fn add_interstitial_terms(
    acc: &mut PairVertexAccumulator<'_>,
    input: &ScalarProductInput,
    relative_g_by_index: &HashMap<[i32; 3], muffintin_core::GVector>,
    pair: &BandPair<'_>,
) -> Result<(), ScalarMpbError> {
    let volume = input.source.partition.interstitial().cell_volume().get();
    let wrap = pair.wrap.index;
    for (left_row, left_wave) in pair.left_basis.plane_waves.iter().enumerate() {
        for (right_row, right_wave) in pair.right_basis.plane_waves.iter().enumerate() {
            let index = [
                right_wave.g.index[0] - left_wave.g.index[0] + wrap[0],
                right_wave.g.index[1] - left_wave.g.index[1] + wrap[1],
                right_wave.g.index[2] - left_wave.g.index[2] + wrap[2],
            ];
            let g_relative = relative_g_by_index
                .get(&index)
                .copied()
                .ok_or(MpbError::UnknownInterstitialPair { g: index })?;
            let amplitude = pair.left_ev.at(left_row, pair.left_band).conj()
                * pair.right_ev.at(right_row, pair.right_band)
                / volume;
            acc.add_interstitial(InterstitialPairSpec {
                g_relative,
                amplitude,
            })?;
        }
    }
    Ok(())
}

fn raw_mt_pairs(raw: &RawProductSpace) -> HashSet<(ProductRadialId, ProductRadialId)> {
    let mut pairs = HashSet::new();
    for product in &raw.radial_products {
        pairs.insert((product.channel.left, product.channel.right));
        pairs.insert((product.channel.right, product.channel.left));
    }
    pairs
}
