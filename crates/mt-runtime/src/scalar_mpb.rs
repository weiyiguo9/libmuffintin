//! Scalar mixed-product bridge from frozen [`ScalarProductInput`] to MPB vertices.

use crate::scalar_product::ScalarProductInput;
use crate::site_coords::site_coordinate;
use muffintin_core::{InverseBohr, ReciprocalLattice};
use muffintin_dft::ScfRelativity;
use muffintin_envelope::site_translation_phase;
use muffintin_operators::lapw::CompiledBasis;
use muffintin_operators::{CompiledSiteProjection, OperatorError, SiteOrbitalCoefficients};
use muffintin_prodbasis::mpb::{
    InterstitialThetaTable, MpbError, ScalarMtPairTable, ScalarVertexContext, apply_overlap_cutoff,
    spex_mixed_product_basis,
};
use muffintin_prodbasis::{
    CompiledAuxiliaryBasis, MtPairSpec, OrbitalPair, PairVertex, ProductRadialId, RawProductSpace,
};
use muffintin_tensor::{Axis, ComplexTensor, TensorError, einsum};
use num_complex::Complex64;
use std::collections::{BTreeSet, HashMap, HashSet};
use thiserror::Error;

/// SPEX overlap-cutoff spin factor for collinear scalar mixed-product construction (`nspin = 2`).
pub const SCALAR_MPB_NSPIN: f64 = 2.0;
/// SPEX overlap-cutoff spin factor for one Pauli-spinor second-variation manifold.
pub const SECOND_VARIATION_MPB_NSPIN: f64 = 1.0;

/// Explicit mixed-product construction and same-spin band-pair selection.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarMpbSpec {
    /// Reciprocal lattice used by [`muffintin_prodbasis::mpb::spex_mixed_product_basis`].
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
    /// Authoritative reciprocal lattice copied from the frozen input.
    pub reciprocal: ReciprocalLattice,
    /// Pair-column layout copied from the frozen input.
    pub pair_columns: muffintin_prodbasis::PairColumnLayout,
    frozen_input: ScalarProductInput,
}

impl ScalarMpbResult {
    pub(crate) fn frozen_input_matches(&self, input: &ScalarProductInput) -> bool {
        &self.frozen_input == input
    }
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

/// Pauli-spinor band-pair selection for SOC second variation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SecondVariationMpbSelection {
    pub k: usize,
    pub left_band: usize,
    pub right_band: usize,
}

/// Mixed-product construction for Pauli spinors represented on a scalar KH basis.
#[derive(Clone, Debug, PartialEq)]
pub struct SecondVariationMpbSpec {
    pub lattice: ReciprocalLattice,
    pub product_l_max: u32,
    pub product_g_max: InverseBohr,
    pub overlap_tolerance: f64,
    pub selections: Vec<SecondVariationMpbSelection>,
}

/// One SOC second-variation pair vertex after summing its two Pauli components.
#[derive(Clone, Debug, PartialEq)]
pub struct SecondVariationMpbPairVertex {
    pub k: usize,
    pub column: usize,
    pub left_band: usize,
    pub right_band: usize,
    pub vertex: PairVertex,
}

/// Exact scalar-radial MPB and Pauli-summed pair vertices.
#[derive(Clone, Debug, PartialEq)]
pub struct SecondVariationMpbResult {
    pub raw: RawProductSpace,
    pub auxiliary: CompiledAuxiliaryBasis,
    pub reciprocal: ReciprocalLattice,
    pub pair_columns: muffintin_prodbasis::PairColumnLayout,
    pub vertices: Vec<SecondVariationMpbPairVertex>,
    frozen_input: ScalarProductInput,
}

impl SecondVariationMpbResult {
    pub(crate) fn frozen_input_matches(&self, input: &ScalarProductInput) -> bool {
        &self.frozen_input == input
    }
}

/// Scalar mixed-product stage-boundary error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ScalarMpbError {
    #[error(transparent)]
    Mpb(#[from] MpbError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
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
    #[error("Pauli-summed MPB vertices require SOC second-variation orbitals")]
    RequiresSecondVariation,
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
    let context = ScalarVertexContext::new(&input.source, &raw, &auxiliary)?;
    let interstitial_table = context.interstitial_table()?;
    let muffin_tin_vertices = contract_muffin_tin_selections(
        input,
        spec,
        &raw,
        auxiliary.dimension(),
        context.muffin_tin_table(),
    )?;
    let interstitial_vertices = contract_interstitial_selections(
        input,
        spec,
        &auxiliary,
        &relative_g_by_index,
        &interstitial_table,
    )?;
    let vertices = assemble_scalar_vertices(
        input,
        &auxiliary,
        spec,
        &muffin_tin_vertices,
        &interstitial_vertices,
    )?;
    Ok(ScalarMpbResult {
        raw,
        auxiliary,
        vertices,
        reciprocal: input.reciprocal,
        pair_columns: input.pair_columns,
        frozen_input: input.clone(),
    })
}

/// Construct Pauli-spinor pair vertices by summing the up/down components of
/// every SOC second-variation orbital on one common scalar mixed-product basis.
pub fn build_second_variation_mpb(
    input: &ScalarProductInput,
    spec: &SecondVariationMpbSpec,
) -> Result<SecondVariationMpbResult, ScalarMpbError> {
    if !matches!(
        input.orbitals.relativity,
        ScfRelativity::SocSecondVariation { .. }
    ) {
        return Err(ScalarMpbError::RequiresSecondVariation);
    }
    if spec.selections.is_empty() {
        return Err(ScalarMpbError::EmptySelection);
    }
    require_compatible_layout(input)?;
    for selection in &spec.selections {
        for spin in [0, 1] {
            require_selection(
                input,
                ScalarMpbSelection {
                    spin,
                    k: selection.k,
                    left_band: selection.left_band,
                    right_band: selection.right_band,
                },
            )?;
        }
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
        SECOND_VARIATION_MPB_NSPIN,
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
    let scalar_spec = ScalarMpbSpec {
        lattice: spec.lattice,
        product_l_max: spec.product_l_max,
        product_g_max: spec.product_g_max,
        overlap_tolerance: spec.overlap_tolerance,
        selections: spec
            .selections
            .iter()
            .flat_map(|selection| {
                [0, 1].map(move |spin| ScalarMpbSelection {
                    spin,
                    k: selection.k,
                    left_band: selection.left_band,
                    right_band: selection.right_band,
                })
            })
            .collect(),
    };
    let context = ScalarVertexContext::new(&input.source, &raw, &auxiliary)?;
    let interstitial_table = context.interstitial_table()?;
    let muffin_tin_vertices = contract_muffin_tin_selections(
        input,
        &scalar_spec,
        &raw,
        auxiliary.dimension(),
        context.muffin_tin_table(),
    )?;
    let interstitial_vertices = contract_interstitial_selections(
        input,
        &scalar_spec,
        &auxiliary,
        &relative_g_by_index,
        &interstitial_table,
    )?;
    let components = assemble_scalar_vertices(
        input,
        &auxiliary,
        &scalar_spec,
        &muffin_tin_vertices,
        &interstitial_vertices,
    )?;
    let vertices = spec
        .selections
        .iter()
        .zip(components.chunks_exact(2))
        .map(|(selection, components)| {
            let pair = OrbitalPair::Bloch {
                k_index: selection.k,
                left: selection.left_band,
                right: selection.right_band,
            };
            let coefficients = components[0]
                .vertex
                .coefficients()
                .iter()
                .zip(components[1].vertex.coefficients())
                .map(|(up, down)| up + down)
                .collect();
            Ok(SecondVariationMpbPairVertex {
                k: selection.k,
                column: input.pair_columns.encode(
                    selection.k,
                    selection.left_band,
                    selection.right_band,
                ),
                left_band: selection.left_band,
                right_band: selection.right_band,
                vertex: PairVertex::from_auxiliary(&auxiliary, pair, coefficients)
                    .map_err(MpbError::from)?,
            })
        })
        .collect::<Result<Vec<_>, ScalarMpbError>>()?;
    Ok(SecondVariationMpbResult {
        raw,
        auxiliary,
        reciprocal: input.reciprocal,
        pair_columns: input.pair_columns,
        vertices,
        frozen_input: input.clone(),
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

fn assemble_scalar_vertices(
    input: &ScalarProductInput,
    auxiliary: &CompiledAuxiliaryBasis,
    spec: &ScalarMpbSpec,
    muffin_tin: &[Vec<Complex64>],
    interstitial: &[Vec<Complex64>],
) -> Result<Vec<ScalarMpbPairVertex>, ScalarMpbError> {
    spec.selections
        .iter()
        .zip(muffin_tin)
        .zip(interstitial)
        .map(|((selection, muffin_tin), interstitial)| {
            let pair = OrbitalPair::Bloch {
                k_index: selection.k,
                left: selection.left_band,
                right: selection.right_band,
            };
            let coefficients = muffin_tin
                .iter()
                .zip(interstitial)
                .map(|(muffin_tin, interstitial)| muffin_tin + interstitial)
                .collect();
            Ok(ScalarMpbPairVertex {
                spin: selection.spin,
                k: selection.k,
                column: input.pair_columns.encode(
                    selection.k,
                    selection.left_band,
                    selection.right_band,
                ),
                left_band: selection.left_band,
                right_band: selection.right_band,
                vertex: PairVertex::from_auxiliary(auxiliary, pair, coefficients)
                    .map_err(MpbError::from)?,
            })
        })
        .collect()
}

struct ProjectedSitePair {
    left: SiteOrbitalCoefficients,
    right: SiteOrbitalCoefficients,
}

fn project_k_sites(
    input: &ScalarProductInput,
    spin: u8,
    mapped: crate::scalar_product::ScalarKMinusQ,
) -> Result<Vec<ProjectedSitePair>, ScalarMpbError> {
    let channel = input
        .orbitals
        .channels
        .iter()
        .find(|channel| channel.spin == spin)
        .ok_or(ScalarMpbError::IncompatiblePairLayout)?;
    let left_basis = &channel.bases[mapped.kq_index];
    let right_basis = &channel.bases[mapped.k_index];
    let left_ev = &channel.eigenvectors[mapped.kq_index];
    let right_ev = &channel.eigenvectors[mapped.k_index];
    let mut sites = Vec::with_capacity(input.source.partition.site_count());
    for site in 0..input.source.partition.site_count() {
        let left =
            CompiledSiteProjection::scalar(left_basis, site)?.project_eigenvectors(left_ev)?;
        let right =
            CompiledSiteProjection::scalar(right_basis, site)?.project_eigenvectors(right_ev)?;
        sites.push(ProjectedSitePair { left, right });
    }
    Ok(sites)
}

fn compile_mt_coordinate_tensor(
    input: &ScalarProductInput,
    spin: u8,
    site: usize,
    left_basis: &CompiledBasis,
    right_basis: &CompiledBasis,
    left_coordinate_count: usize,
    right_coordinate_count: usize,
    known: &HashSet<(ProductRadialId, ProductRadialId)>,
    table: &mut ScalarMtPairTable<'_>,
) -> Result<ComplexTensor, ScalarMpbError> {
    let auxiliary_count = table.auxiliary_dimension();
    let mut coefficients = vec![
        Complex64::default();
        auxiliary_count * left_coordinate_count * right_coordinate_count
    ];
    for left_coordinate in 0..left_coordinate_count {
        let (left, left_m) = site_coordinate(left_basis, site, spin, left_coordinate)
            .ok_or(ScalarMpbError::IncompatiblePairLayout)?;
        for right_coordinate in 0..right_coordinate_count {
            let (right, right_m) = site_coordinate(right_basis, site, spin, right_coordinate)
                .ok_or(ScalarMpbError::IncompatiblePairLayout)?;
            if !known.contains(&(left, right)) {
                continue;
            }
            let compiled = table.compile_pair(
                &input.source,
                MtPairSpec {
                    left,
                    left_m,
                    right,
                    right_m,
                },
            )?;
            for &(auxiliary, factor) in compiled.coefficients() {
                coefficients[(auxiliary * left_coordinate_count + left_coordinate)
                    * right_coordinate_count
                    + right_coordinate] += factor;
            }
        }
    }
    Ok(ComplexTensor::from_host_row_major(
        &[
            auxiliary_count,
            left_coordinate_count,
            right_coordinate_count,
        ],
        &[Axis::Auxiliary, Axis::SiteCoordinate, Axis::SiteCoordinate],
        coefficients,
    )?)
}

fn contract_muffin_tin_selections(
    input: &ScalarProductInput,
    spec: &ScalarMpbSpec,
    raw: &RawProductSpace,
    auxiliary_count: usize,
    mut table: ScalarMtPairTable<'_>,
) -> Result<Vec<Vec<Complex64>>, ScalarMpbError> {
    let known = raw_mt_pairs(raw);
    let mut by_selection = HashMap::new();
    let groups = spec
        .selections
        .iter()
        .map(|selection| (selection.spin, selection.k))
        .collect::<BTreeSet<_>>();
    for (spin, k) in groups {
        let mapped = input
            .k_minus_q
            .iter()
            .copied()
            .find(|mapped| mapped.k_index == k)
            .ok_or(ScalarMpbError::IncompatiblePairLayout)?;
        let channel = input
            .orbitals
            .channels
            .iter()
            .find(|channel| channel.spin == spin)
            .ok_or(ScalarMpbError::IncompatiblePairLayout)?;
        let left_basis = &channel.bases[mapped.kq_index];
        let right_basis = &channel.bases[mapped.k_index];
        let projected = project_k_sites(input, spin, mapped)?;
        let left_bands = spec
            .selections
            .iter()
            .filter(|selection| selection.spin == spin && selection.k == k)
            .map(|selection| selection.left_band)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let right_bands = spec
            .selections
            .iter()
            .filter(|selection| selection.spin == spin && selection.k == k)
            .map(|selection| selection.right_band)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mut contracted =
            vec![Complex64::default(); auxiliary_count * left_bands.len() * right_bands.len()];
        for (site, projected) in projected.iter().enumerate() {
            let coordinate_tensor = compile_mt_coordinate_tensor(
                input,
                spin,
                site,
                left_basis,
                right_basis,
                projected.left.coordinate_count(),
                projected.right.coordinate_count(),
                &known,
                &mut table,
            )?;
            let left = select_site_bands(&projected.left, &left_bands)?;
            let right = select_site_bands(&projected.right, &right_bands)?;
            let site_contracted = einsum(
                "il,aij,jr->alr",
                &[&left.conjugate(), &coordinate_tensor, &right],
            )?
            .to_host_row_major();
            let phase = site_translation_phase(
                input.source.q.cartesian,
                input.source.partition.sites()[site].position,
            )
            .conj();
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
                by_selection.insert((spin, k, left_band, right_band), coefficients);
            }
        }
    }
    spec.selections
        .iter()
        .map(|selection| {
            by_selection
                .get(&(
                    selection.spin,
                    selection.k,
                    selection.left_band,
                    selection.right_band,
                ))
                .cloned()
                .ok_or(ScalarMpbError::IncompatiblePairLayout)
        })
        .collect()
}

fn select_site_bands(
    projected: &SiteOrbitalCoefficients,
    bands: &[usize],
) -> Result<ComplexTensor, ScalarMpbError> {
    let mut values = Vec::with_capacity(projected.coordinate_count() * bands.len());
    for coordinate in 0..projected.coordinate_count() {
        for &band in bands {
            values.push(projected.at(coordinate, band));
        }
    }
    Ok(ComplexTensor::from_host_row_major(
        &[projected.coordinate_count(), bands.len()],
        &[Axis::SiteCoordinate, Axis::Band],
        values,
    )?)
}

fn contract_interstitial_selections(
    input: &ScalarProductInput,
    spec: &ScalarMpbSpec,
    auxiliary: &CompiledAuxiliaryBasis,
    relative_g_by_index: &HashMap<[i32; 3], muffintin_core::GVector>,
    table: &InterstitialThetaTable,
) -> Result<Vec<Vec<Complex64>>, ScalarMpbError> {
    let auxiliary_count = auxiliary.dimension();
    let mt_count = auxiliary.mt_dimension();
    let interstitial_count = auxiliary.interstitial_dimension();
    let volume = input.source.partition.interstitial().cell_volume().get();
    let mut by_selection = HashMap::new();
    let groups = spec
        .selections
        .iter()
        .map(|selection| (selection.spin, selection.k))
        .collect::<BTreeSet<_>>();
    for (spin, k) in groups {
        let mapped = input
            .k_minus_q
            .iter()
            .copied()
            .find(|mapped| mapped.k_index == k)
            .ok_or(ScalarMpbError::IncompatiblePairLayout)?;
        let channel = input
            .orbitals
            .channels
            .iter()
            .find(|channel| channel.spin == spin)
            .ok_or(ScalarMpbError::IncompatiblePairLayout)?;
        let left_basis = &channel.bases[mapped.kq_index];
        let right_basis = &channel.bases[mapped.k_index];
        let left_bands = selected_bands(spec, spin, k, true);
        let right_bands = selected_bands(spec, spin, k, false);
        let left = select_plane_wave_bands(
            &channel.eigenvectors[mapped.kq_index],
            left_basis.plane_waves.len(),
            &left_bands,
        )?;
        let right = select_plane_wave_bands(
            &channel.eigenvectors[mapped.k_index],
            right_basis.plane_waves.len(),
            &right_bands,
        )?;
        let mut kernel =
            vec![
                Complex64::default();
                interstitial_count * left_basis.plane_waves.len() * right_basis.plane_waves.len()
            ];
        for (left_row, left_wave) in left_basis.plane_waves.iter().enumerate() {
            for (right_row, right_wave) in right_basis.plane_waves.iter().enumerate() {
                let index = [
                    right_wave.g.index[0] - left_wave.g.index[0] + mapped.umklapp.index[0],
                    right_wave.g.index[1] - left_wave.g.index[1] + mapped.umklapp.index[1],
                    right_wave.g.index[2] - left_wave.g.index[2] + mapped.umklapp.index[2],
                ];
                let g_relative = relative_g_by_index
                    .get(&index)
                    .ok_or(MpbError::UnknownInterstitialPair { g: index })?;
                for (auxiliary_position, &theta) in
                    table.row(auxiliary, g_relative.index)?.iter().enumerate()
                {
                    kernel[(auxiliary_position * left_basis.plane_waves.len() + left_row)
                        * right_basis.plane_waves.len()
                        + right_row] = theta / volume;
                }
            }
        }
        let kernel = ComplexTensor::from_host_row_major(
            &[
                interstitial_count,
                left_basis.plane_waves.len(),
                right_basis.plane_waves.len(),
            ],
            &[Axis::Auxiliary, Axis::GlobalBasis, Axis::GlobalBasis],
            kernel,
        )?;
        let contracted =
            einsum("gi,agh,hj->aij", &[&left.conjugate(), &kernel, &right])?.to_host_row_major();
        for (left_position, &left_band) in left_bands.iter().enumerate() {
            for (right_position, &right_band) in right_bands.iter().enumerate() {
                let mut coefficients = vec![Complex64::default(); auxiliary_count];
                for auxiliary_position in 0..interstitial_count {
                    coefficients[mt_count + auxiliary_position] =
                        contracted[(auxiliary_position * left_bands.len() + left_position)
                            * right_bands.len()
                            + right_position];
                }
                by_selection.insert((spin, k, left_band, right_band), coefficients);
            }
        }
    }
    spec.selections
        .iter()
        .map(|selection| {
            by_selection
                .get(&(
                    selection.spin,
                    selection.k,
                    selection.left_band,
                    selection.right_band,
                ))
                .cloned()
                .ok_or(ScalarMpbError::IncompatiblePairLayout)
        })
        .collect()
}

fn selected_bands(spec: &ScalarMpbSpec, spin: u8, k: usize, left: bool) -> Vec<usize> {
    spec.selections
        .iter()
        .filter(|selection| selection.spin == spin && selection.k == k)
        .map(|selection| {
            if left {
                selection.left_band
            } else {
                selection.right_band
            }
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn select_plane_wave_bands(
    eigenvectors: &muffintin_tensor::DenseEigenvectors,
    plane_wave_count: usize,
    bands: &[usize],
) -> Result<ComplexTensor, ScalarMpbError> {
    let mut values = Vec::with_capacity(plane_wave_count * bands.len());
    for plane_wave in 0..plane_wave_count {
        for &band in bands {
            values.push(eigenvectors.at(plane_wave, band));
        }
    }
    Ok(ComplexTensor::from_host_row_major(
        &[plane_wave_count, bands.len()],
        &[Axis::GlobalBasis, Axis::Band],
        values,
    )?)
}

fn raw_mt_pairs(raw: &RawProductSpace) -> HashSet<(ProductRadialId, ProductRadialId)> {
    let mut pairs = HashSet::new();
    for product in &raw.radial_products {
        pairs.insert((product.channel.left, product.channel.right));
        pairs.insert((product.channel.right, product.channel.left));
    }
    pairs
}
