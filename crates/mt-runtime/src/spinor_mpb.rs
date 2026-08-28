//! Selected-band spinor mixed-product bridge from frozen [`SpinorProductInput`].

use crate::spinor_product::{SpinorFrozenOrbitals, SpinorKMinusQ, SpinorProductInput};
use muffintin_auxiliary_ir::{
    CompiledAuxiliaryBasis, DiracChargeSector, DiracMtPairSpec, DiracProductSource, DiracRadial,
    DiracRadialId, DiracRawProductSpace, DiracSiteRadialSet, InterstitialPairSpec, OrbitalPair,
    PairColumnLayout, PairVertex, ProductOrbitalKind, ProductPartition, RawInterstitialPairSupport,
    TransferQ,
};
use muffintin_core::{Bohr, GVector, InverseBohr, ReciprocalLattice, RelativisticChannel};
use muffintin_envelope::site_translation_phase;
use muffintin_lapw::{Provenance, SpinorCompiledBasis};
use muffintin_mpb::{
    DiracBlochVertexAccumulator, MpbError, apply_dirac_overlap_cutoff,
    untruncated_dirac_product_space,
};
use muffintin_operators::{CompiledSiteProjection, OperatorError};
use muffintin_tensor::{DenseEigenvectors, MemoryLayout};
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
    /// Nonempty selections `(k, left_band, right_band)` in the M-L5b window.
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
    pub(crate) const fn frozen_input_identity(&self) -> SpinorFrozenInputIdentity {
        self.frozen_input
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

/// Runtime-private binding stamp of the frozen [`SpinorProductInput`] used to
/// construct a [`SpinorMpbResult`].
///
/// The mixer is the same splitmix-style 64-bit fold as the parent-grid
/// construction fingerprint: ordered lengths, type tags, and `f64`/complex
/// bit patterns. It is an internal binding stamp, not scientific provenance
/// or a cryptographic digest. Distinct inputs can collide at a residual of
/// one part in $2^{64}$ per comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SpinorFrozenInputIdentity(u64);

const FROZEN_INPUT_IDENTITY_SEED: u64 = 0x5F10_1D00_0001_0001;

pub(crate) fn spinor_frozen_input_identity(
    input: &SpinorProductInput,
) -> SpinorFrozenInputIdentity {
    let mut fingerprint = Fingerprint::new();
    fingerprint.mix_dirac_source(&input.source);
    fingerprint.mix_orbitals(&input.orbitals);
    fingerprint.mix_k_minus_q(&input.k_minus_q);
    fingerprint.mix_pair_layout(input.pair_columns);
    fingerprint.mix_reciprocal(&input.reciprocal);
    fingerprint.finish()
}

struct Fingerprint {
    hash: u64,
}

impl Fingerprint {
    fn new() -> Self {
        Self {
            hash: mix(FROZEN_INPUT_IDENTITY_SEED, 1),
        }
    }

    fn mix(&mut self, lane: u64) {
        self.hash = mix(self.hash, lane);
    }

    fn mix_usize(&mut self, value: usize) {
        self.mix(value as u64);
    }

    fn mix_i32(&mut self, value: i32) {
        self.mix(u64::from(value as u32));
    }

    fn mix_i64(&mut self, value: i64) {
        self.mix(value as u64);
    }

    fn mix_f64(&mut self, value: f64) {
        self.mix(value.to_bits());
    }

    fn mix_complex(&mut self, value: Complex64) {
        self.mix_f64(value.re);
        self.mix_f64(value.im);
    }

    fn mix_opt_str(&mut self, value: Option<&str>) {
        self.hash = mix_opt_str(self.hash, value);
    }

    fn mix_provenance(&mut self, provenance: &Provenance) {
        self.mix_opt_str(provenance.recipe.as_deref());
        self.mix_opt_str(provenance.reference.as_deref());
    }

    fn mix_bohr3(&mut self, value: [Bohr; 3]) {
        for component in value {
            self.mix_f64(component.get());
        }
    }

    fn mix_inverse_bohr3(&mut self, value: [InverseBohr; 3]) {
        for component in value {
            self.mix_f64(component.get());
        }
    }

    fn mix_gvector(&mut self, g: GVector) {
        for index in g.index {
            self.mix_i32(index);
        }
        self.mix_inverse_bohr3(g.cartesian);
        self.mix_f64(g.norm.get());
    }

    fn mix_transfer_q(&mut self, q: TransferQ) {
        self.mix(1);
        self.mix_inverse_bohr3(q.cartesian);
        self.mix_gvector(q.umklapp);
    }

    fn mix_kind(&mut self, kind: ProductOrbitalKind) {
        match kind {
            ProductOrbitalKind::Valence => self.mix(1),
            ProductOrbitalKind::Core => self.mix(2),
        }
    }

    fn mix_dirac_source(&mut self, source: &DiracProductSource) {
        self.mix(1);
        self.mix_partition(&source.partition);
        self.mix_usize(source.radials.len());
        for (site, radials) in source.radials.iter().enumerate() {
            self.mix_site_radials(site, radials);
        }
        self.mix_transfer_q(source.q);
        self.mix_pair_support(&source.interstitial_pair_support);
        self.mix_provenance(&source.provenance);
    }

    fn mix_partition(&mut self, partition: &ProductPartition) {
        self.mix_usize(partition.site_count());
        for site in partition.sites() {
            self.mix_usize(site.index);
            self.mix_bohr3(site.position);
            self.mix_f64(site.radius.get());
        }
        let interstitial = partition.interstitial();
        self.mix_f64(interstitial.cell_volume().get());
        self.mix_usize(interstitial.spheres().len());
        for sphere in interstitial.spheres() {
            self.mix_bohr3(sphere.center);
            self.mix_f64(sphere.radius.get());
        }
        self.mix_provenance(partition.provenance());
    }

    fn mix_site_radials(&mut self, site: usize, radials: &DiracSiteRadialSet) {
        self.mix_usize(site);
        self.mix_f64(radials.mesh.first().get());
        self.mix_f64(radials.mesh.increment());
        self.mix_usize(radials.mesh.radii().len());
        for radius in radials.mesh.radii() {
            self.mix_f64(radius.get());
        }
        self.mix_usize(radials.mesh.weights().len());
        for weight in radials.mesh.weights() {
            self.mix_f64(*weight);
        }
        self.mix_usize(radials.valence.len());
        for radial in &radials.valence {
            self.mix_dirac_radial(site, ProductOrbitalKind::Valence, radial);
        }
        self.mix_usize(radials.cores.len());
        for radial in &radials.cores {
            self.mix_dirac_radial(site, ProductOrbitalKind::Core, radial);
        }
    }

    fn mix_dirac_radial(&mut self, site: usize, kind: ProductOrbitalKind, radial: &DiracRadial) {
        self.mix_usize(site);
        self.mix_kind(kind);
        self.mix_i32(radial.kappa.get());
        self.mix_usize(radial.n);
        self.mix_usize(radial.samples.large.len());
        for sample in &radial.samples.large {
            self.mix_f64(*sample);
        }
        self.mix_usize(radial.samples.small.len());
        for sample in &radial.samples.small {
            self.mix_f64(*sample);
        }
    }

    fn mix_pair_support(&mut self, support: &RawInterstitialPairSupport) {
        self.mix_transfer_q(support.q);
        self.mix_usize(support.components.len());
        for component in &support.components {
            self.mix_gvector(component.g_relative);
        }
    }

    fn mix_orbitals(&mut self, orbitals: &SpinorFrozenOrbitals) {
        self.mix(2);
        self.mix_usize(orbitals.k_fractional.len());
        for k in &orbitals.k_fractional {
            for component in k {
                self.mix_f64(*component);
            }
        }
        self.mix_usize(orbitals.available_bands.len());
        for count in &orbitals.available_bands {
            self.mix_usize(*count);
        }
        self.mix_usize(orbitals.band_window.start);
        self.mix_usize(orbitals.band_window.count);
        self.mix_usize(orbitals.energies.len());
        for (k, energies) in orbitals.energies.iter().enumerate() {
            self.mix_usize(k);
            self.mix_usize(energies.len());
            for energy in energies {
                self.mix_f64(energy.get());
            }
        }
        self.mix_usize(orbitals.bases.len());
        for (k, basis) in orbitals.bases.iter().enumerate() {
            self.mix_usize(k);
            self.mix_compiled_basis(basis);
        }
        self.mix_usize(orbitals.eigenvectors.len());
        for (k, eigenvectors) in orbitals.eigenvectors.iter().enumerate() {
            self.mix_usize(k);
            self.mix_eigenvectors(eigenvectors);
        }
    }

    fn mix_compiled_basis(&mut self, basis: &SpinorCompiledBasis) {
        self.mix_usize(basis.layout.spatial_plane_wave_count());
        self.mix_usize(basis.layout.site_count());
        for site in 0..basis.layout.site_count() {
            let Some(layout) = basis.layout.site_layout(site) else {
                self.mix(0);
                continue;
            };
            self.mix(1);
            self.mix_usize(site);
            self.mix_usize(layout.counts_by_kappa().len());
            for &(kappa, count) in layout.counts_by_kappa() {
                self.mix_i32(kappa.get());
                self.mix_usize(count);
            }
        }
        self.mix_usize(basis.plane_waves.len());
        for wave in &basis.plane_waves {
            self.mix_inverse_bohr3(wave.k);
            self.mix_gvector(wave.g);
            self.mix_inverse_bohr3(wave.q);
            self.mix_f64(wave.q_norm.get());
        }
        self.mix_usize(basis.site_augmentations.len());
        for (site, waves) in basis.site_augmentations.iter().enumerate() {
            self.mix_usize(site);
            self.mix_usize(waves.len());
            for (g, augmentation) in waves.iter().enumerate() {
                self.mix_usize(g);
                self.mix_usize(augmentation.channels.len());
                for channel in &augmentation.channels {
                    self.mix_channel(*channel);
                }
                for (spin, coefficients) in augmentation.coefficients.iter().enumerate() {
                    self.mix_usize(spin);
                    self.mix_usize(coefficients.len());
                    for pair in coefficients {
                        self.mix_complex(pair[0]);
                        self.mix_complex(pair[1]);
                    }
                }
            }
        }
        self.mix_usize(basis.site_geometry.len());
        for geometry in &basis.site_geometry {
            self.mix_bohr3(geometry.position);
            self.mix_f64(geometry.radius.get());
        }
        self.mix_provenance(&basis.provenance);
    }

    fn mix_channel(&mut self, channel: RelativisticChannel) {
        self.mix_i32(channel.kappa().get());
        self.mix_i64(channel.twice_mu().get());
    }

    fn mix_eigenvectors(&mut self, eigenvectors: &DenseEigenvectors) {
        self.mix_usize(eigenvectors.rows());
        self.mix_usize(eigenvectors.columns());
        match eigenvectors.layout() {
            MemoryLayout::RowMajor => self.mix(1),
            MemoryLayout::ColumnMajor => self.mix(2),
            MemoryLayout::Strided => self.mix(3),
        }
        let values = eigenvectors.to_host_column_major();
        self.mix_usize(values.len());
        for value in values {
            self.mix_complex(value);
        }
    }

    fn mix_k_minus_q(&mut self, mapped: &[SpinorKMinusQ]) {
        self.mix(3);
        self.mix_usize(mapped.len());
        for record in mapped {
            self.mix_usize(record.k_index);
            self.mix_usize(record.kq_index);
            self.mix_gvector(record.umklapp);
        }
    }

    fn mix_pair_layout(&mut self, layout: PairColumnLayout) {
        self.mix(4);
        self.mix_usize(layout.n_k);
        self.mix_usize(layout.n_orb);
        match layout.core_orbital {
            None => self.mix(0),
            Some(core) => {
                self.mix(1);
                self.mix_usize(core);
            }
        }
    }

    fn mix_reciprocal(&mut self, reciprocal: &ReciprocalLattice) {
        self.mix(5);
        for vector in reciprocal.basis() {
            self.mix_inverse_bohr3(*vector);
        }
    }

    fn finish(self) -> SpinorFrozenInputIdentity {
        SpinorFrozenInputIdentity(self.hash)
    }
}

fn mix(hash: u64, lane: u64) -> u64 {
    hash.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(lane)
}

fn mix_opt_str(hash: u64, value: Option<&str>) -> u64 {
    match value {
        None => mix(hash, 0),
        Some(text) => {
            let mut hash = mix(hash, 1);
            hash = mix(hash, text.len() as u64);
            for &byte in text.as_bytes() {
                hash = mix(hash, u64::from(byte));
            }
            hash
        }
    }
}
