//! Retained auxiliary basis shared by mixed-product and interpolation-point paths.

use crate::{AuxiliaryIrError, AuxiliaryPartition, AuxiliarySource, TransferQ};
use muffintin_core::{Bohr, ExponentialMesh, GVector, InverseBohr, VolumeBohr3};
use muffintin_envelope::Provenance;
use std::cmp::Ordering;
use std::collections::BTreeSet;

const WAVE_TOLERANCE: f64 = 1.0e-12;

/// Spectral cutoff recorded by a mixed-product constructor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CutoffKind {
    /// Drop overlap eigenvalues strictly below `value * nspin_factor`.
    SpectralOverlap,
}

/// Explicit mixed-product cutoff; never stored on raw products.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutoffRecord {
    pub kind: CutoffKind,
    pub value: f64,
    pub nspin_factor: f64,
}

/// One retained muffin-tin radial mode, $M$-degenerate.
#[derive(Clone, Debug, PartialEq)]
pub struct MtAuxiliaryMode {
    pub l: u32,
    pub n: usize,
    pub radial: Vec<f64>,
}

/// Per-site retained muffin-tin block, including the radial mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteAuxiliaryBlock {
    pub site: usize,
    pub mesh: ExponentialMesh,
    /// Modes sorted by $L$ then $n$.
    pub modes: Vec<MtAuxiliaryMode>,
}

/// One interstitial $|q+G|$ auxiliary plane wave of the mixed product basis.
///
/// This is not a raw orbital-pair reciprocal label.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuxiliaryInterstitialWave {
    pub g: GVector,
    pub q_plus_g: [InverseBohr; 3],
    pub q_plus_g_norm: InverseBohr,
}

/// MPB auxiliary interstitial PW support, filtered by $|q+G|\le g_{\mathrm{cut}}$.
#[derive(Clone, Debug, PartialEq)]
pub struct AuxiliaryInterstitialSupport {
    pub q: TransferQ,
    pub g_cut: InverseBohr,
    pub waves: Vec<AuxiliaryInterstitialWave>,
}

/// Region of an interpolation-point auxiliary function.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum InterpolationRegion {
    /// Point inside muffin-tin site `site`.
    MuffinTin { site: usize },
    /// Point in the partitioned interstitial.
    Interstitial,
    /// Point on an unpartitioned uniform grid.
    Uniform,
}

/// One real-space interpolation point of a THC/ISDF auxiliary basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpolationAuxiliaryPoint {
    /// Stable parent-grid index.
    pub id: usize,
    pub coordinate: [Bohr; 3],
    pub weight: VolumeBohr3,
    pub region: InterpolationRegion,
}

/// Mixed-product payload: retained muffin-tin modes plus $|q+G|$ plane waves.
#[derive(Clone, Debug, PartialEq)]
pub struct MixedProductAuxiliary {
    pub sites: Vec<SiteAuxiliaryBlock>,
    pub interstitial: AuxiliaryInterstitialSupport,
    pub cutoff: Option<CutoffRecord>,
}

/// Interpolation-point payload shared by k-point ISDF/THC.
///
/// Points are stored in muffin-tin (by site, then id), then interstitial id,
/// then uniform id. This is not an empty mixed-product block.
#[derive(Clone, Debug, PartialEq)]
pub struct InterpolationPointAuxiliary {
    pub points: Vec<InterpolationAuxiliaryPoint>,
}

/// Typed auxiliary representation. Not a trait family and not a compatibility
/// shim around a fake mixed-product payload.
#[derive(Clone, Debug, PartialEq)]
pub enum AuxiliaryRepresentation {
    /// SPEX-style mixed product basis.
    MixedProduct(MixedProductAuxiliary),
    /// Real-space interpolation points (ISDF/THC).
    InterpolationPoints(InterpolationPointAuxiliary),
}

/// Region of a compiled auxiliary function in global flatten order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AuxiliaryRegion {
    /// Muffin-tin channel in SPEX order $(site, L, M, n)$.
    MuffinTin {
        site: usize,
        l: u32,
        m: i32,
        n: usize,
    },
    /// Interstitial auxiliary plane wave labelled by $G$ at the stored $q$.
    Interstitial { g: GVector },
    /// Interpolation point in muffin-tin / interstitial / uniform order.
    InterpolationPoint {
        id: usize,
        region: InterpolationRegion,
    },
}

impl AuxiliaryRegion {
    /// Whether this channel belongs to the muffin-tin block of the flatten.
    pub const fn is_mt_block(self) -> bool {
        match self {
            Self::MuffinTin { .. } => true,
            Self::InterpolationPoint {
                region: InterpolationRegion::MuffinTin { .. },
                ..
            } => true,
            Self::Interstitial { .. }
            | Self::InterpolationPoint {
                region: InterpolationRegion::Interstitial | InterpolationRegion::Uniform,
                ..
            } => false,
        }
    }
}

/// Stable auxiliary layout: transfer $q$, exact region sequence, and MT/I split.
///
/// This is the identity compared by pair vertices and the Coulomb operator.
/// Recipe strings are not part of the identity.
#[derive(Clone, Debug, PartialEq)]
pub struct AuxiliaryLayout {
    q: TransferQ,
    regions: Vec<AuxiliaryRegion>,
    mt_dimension: usize,
    interstitial_dimension: usize,
}

impl AuxiliaryLayout {
    /// Build from an explicit region list. The muffin-tin count follows
    /// [`AuxiliaryRegion::is_mt_block`].
    pub fn from_regions(q: TransferQ, regions: Vec<AuxiliaryRegion>) -> Self {
        let mt_dimension = regions.iter().filter(|region| region.is_mt_block()).count();
        let interstitial_dimension = regions.len() - mt_dimension;
        Self {
            q,
            regions,
            mt_dimension,
            interstitial_dimension,
        }
    }

    /// Canonical transfer $q$.
    pub const fn q(&self) -> TransferQ {
        self.q
    }

    /// Combined regions in flatten order.
    pub fn regions(&self) -> &[AuxiliaryRegion] {
        &self.regions
    }

    /// Muffin-tin block length.
    pub const fn mt_dimension(&self) -> usize {
        self.mt_dimension
    }

    /// Interstitial (or uniform interpolation-point) block length.
    pub const fn interstitial_dimension(&self) -> usize {
        self.interstitial_dimension
    }

    /// Total dimension: muffin-tin then interstitial.
    pub fn dimension(&self) -> usize {
        self.regions.len()
    }
}

/// Retained auxiliary basis at one transfer $q$.
///
/// Mixed-product consumers use muffin-tin meshes and $|q+G|$ waves.
/// Interpolation-point consumers use the selected real-space points. Pair
/// vertices stay muffin-tin then interstitial: mixed-product $M$-expanded
/// modes then plane waves, or muffin-tin-tagged points then
/// interstitial/uniform points.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledAuxiliaryBasis {
    pub partition: AuxiliaryPartition,
    pub q: TransferQ,
    pub representation: AuxiliaryRepresentation,
    pub provenance: Provenance,
}

impl CompiledAuxiliaryBasis {
    /// Mixed-product payload, if this basis is not interpolation points.
    pub fn mixed_product(&self) -> Option<&MixedProductAuxiliary> {
        match &self.representation {
            AuxiliaryRepresentation::MixedProduct(payload) => Some(payload),
            AuxiliaryRepresentation::InterpolationPoints(_) => None,
        }
    }

    /// Mutable mixed-product payload.
    pub fn mixed_product_mut(&mut self) -> Option<&mut MixedProductAuxiliary> {
        match &mut self.representation {
            AuxiliaryRepresentation::MixedProduct(payload) => Some(payload),
            AuxiliaryRepresentation::InterpolationPoints(_) => None,
        }
    }

    /// Interpolation-point payload, if this basis is not mixed-product.
    pub fn interpolation_points(&self) -> Option<&[InterpolationAuxiliaryPoint]> {
        match &self.representation {
            AuxiliaryRepresentation::InterpolationPoints(payload) => Some(&payload.points),
            AuxiliaryRepresentation::MixedProduct(_) => None,
        }
    }

    /// Require a mixed-product payload.
    pub fn require_mixed_product(&self) -> Result<&MixedProductAuxiliary, AuxiliaryIrError> {
        self.mixed_product()
            .ok_or(AuxiliaryIrError::ExpectedMixedProduct)
    }

    /// Require interpolation points.
    pub fn require_interpolation_points(
        &self,
    ) -> Result<&[InterpolationAuxiliaryPoint], AuxiliaryIrError> {
        self.interpolation_points()
            .ok_or(AuxiliaryIrError::ExpectedInterpolationPoints)
    }

    /// Global muffin-tin auxiliary dimension.
    ///
    /// Mixed product: expand $M=-L,\ldots,L$. Interpolation points: muffin-tin
    /// tagged points only.
    pub fn mt_dimension(&self) -> usize {
        match &self.representation {
            AuxiliaryRepresentation::MixedProduct(payload) => payload
                .sites
                .iter()
                .flat_map(|block| block.modes.iter())
                .map(|mode| 2 * mode.l as usize + 1)
                .sum(),
            AuxiliaryRepresentation::InterpolationPoints(payload) => payload
                .points
                .iter()
                .filter(|point| matches!(point.region, InterpolationRegion::MuffinTin { .. }))
                .count(),
        }
    }

    /// Interstitial (or uniform) auxiliary dimension.
    pub fn interstitial_dimension(&self) -> usize {
        match &self.representation {
            AuxiliaryRepresentation::MixedProduct(payload) => payload.interstitial.waves.len(),
            AuxiliaryRepresentation::InterpolationPoints(payload) => payload
                .points
                .iter()
                .filter(|point| !matches!(point.region, InterpolationRegion::MuffinTin { .. }))
                .count(),
        }
    }

    /// Total auxiliary dimension: muffin-tin block then interstitial block.
    pub fn dimension(&self) -> usize {
        self.mt_dimension() + self.interstitial_dimension()
    }

    /// Layout identity: $q$, exact regions, and the muffin-tin/interstitial split.
    pub fn layout(&self) -> AuxiliaryLayout {
        AuxiliaryLayout::from_regions(self.q, self.regions())
    }

    /// Combined regions in muffin-tin then interstitial order.
    ///
    /// Mixed product flatten is $site \to L \to M=-L..L \to n$, then $G$.
    /// Interpolation points are muffin-tin (site, then id), interstitial id,
    /// then uniform id.
    pub fn regions(&self) -> Vec<AuxiliaryRegion> {
        match &self.representation {
            AuxiliaryRepresentation::MixedProduct(payload) => mixed_product_regions(payload),
            AuxiliaryRepresentation::InterpolationPoints(payload) => payload
                .points
                .iter()
                .map(|point| AuxiliaryRegion::InterpolationPoint {
                    id: point.id,
                    region: point.region,
                })
                .collect(),
        }
    }

    /// Deterministic muffin-tin index: $site \to L \to M \to n$.
    ///
    /// Computed from the mixed-product block arithmetic rather than by scanning
    /// [`Self::regions`]; the flatten order is the one that function emits.
    /// Interpolation-point bases carry no muffin-tin region and yield `None`.
    pub fn mt_index(&self, site: usize, l: u32, m: i32, n: usize) -> Option<usize> {
        let payload = self.mixed_product()?;
        let mut offset = 0;
        for block in &payload.sites {
            let mut coupled = block.modes.iter().map(|mode| mode.l).min();
            while let Some(current) = coupled {
                let modes = block.modes.iter().filter(|mode| mode.l == current);
                let count = modes.clone().count();
                if block.site == site && current == l && m.unsigned_abs() <= l {
                    let slot = modes.clone().filter(|mode| mode.n < n).count();
                    if modes.clone().any(|mode| mode.n == n) {
                        return Some(offset + (m + l as i32) as usize * count + slot);
                    }
                }
                offset += (2 * current as usize + 1) * count;
                coupled = block
                    .modes
                    .iter()
                    .map(|mode| mode.l)
                    .filter(|&value| value > current)
                    .min();
            }
        }
        None
    }

    /// Mesh of one muffin-tin site (mixed product only).
    pub fn site_mesh(&self, site: usize) -> Option<&ExponentialMesh> {
        self.mixed_product().and_then(|payload| {
            payload
                .sites
                .iter()
                .find(|block| block.site == site)
                .map(|block| &block.mesh)
        })
    }

    /// Retained radial mode $(site, L, n)$ (mixed product only).
    pub fn mt_mode(&self, site: usize, l: u32, n: usize) -> Option<&MtAuxiliaryMode> {
        self.mixed_product().and_then(|payload| {
            payload
                .sites
                .iter()
                .find(|block| block.site == site)
                .and_then(|block| block.modes.iter().find(|mode| mode.l == l && mode.n == n))
        })
    }

    /// Mixed-product cutoff, if recorded.
    pub fn cutoff(&self) -> Option<CutoffRecord> {
        self.mixed_product().and_then(|payload| payload.cutoff)
    }

    /// Reject inconsistent payloads.
    pub fn validate(&self) -> Result<(), AuxiliaryIrError> {
        match &self.representation {
            AuxiliaryRepresentation::MixedProduct(payload) => self.validate_mixed_product(payload),
            AuxiliaryRepresentation::InterpolationPoints(payload) => {
                self.validate_interpolation_points(payload)
            }
        }
    }

    /// Site meshes must equal the source meshes used to build mixed-product modes.
    ///
    /// Interpolation-point auxiliaries check partition identity and transfer $q$
    /// only; they do not invent empty muffin-tin radial blocks.
    pub fn validate_against_source(
        &self,
        source: &AuxiliarySource,
    ) -> Result<(), AuxiliaryIrError> {
        self.validate()?;
        if self.q != source.q {
            return Err(AuxiliaryIrError::AuxiliarySupportTransferQ);
        }
        if self.partition != source.partition {
            return Err(AuxiliaryIrError::AuxiliarySiteCount {
                expected: source.partition.site_count(),
                actual: self.partition.site_count(),
            });
        }
        match &self.representation {
            AuxiliaryRepresentation::MixedProduct(payload) => {
                if payload.interstitial.q != source.q {
                    return Err(AuxiliaryIrError::AuxiliarySupportTransferQ);
                }
                if payload.sites.len() != source.radials.len() {
                    return Err(AuxiliaryIrError::AuxiliarySiteCount {
                        expected: source.radials.len(),
                        actual: payload.sites.len(),
                    });
                }
                for (site, (block, radials)) in
                    payload.sites.iter().zip(&source.radials).enumerate()
                {
                    if block.mesh != radials.mesh {
                        return Err(AuxiliaryIrError::AuxiliaryMeshMismatch { site });
                    }
                }
            }
            AuxiliaryRepresentation::InterpolationPoints(_) => {}
        }
        Ok(())
    }

    fn validate_mixed_product(
        &self,
        payload: &MixedProductAuxiliary,
    ) -> Result<(), AuxiliaryIrError> {
        if payload.sites.len() != self.partition.site_count() {
            return Err(AuxiliaryIrError::AuxiliarySiteCount {
                expected: self.partition.site_count(),
                actual: payload.sites.len(),
            });
        }
        let mut seen_sites = BTreeSet::new();
        for (expected, block) in payload.sites.iter().enumerate() {
            let partition_index = self.partition.site(expected).map(|site| site.index);
            if block.site != expected || partition_index != Some(block.site) {
                return Err(AuxiliaryIrError::AuxiliarySiteIdentity {
                    expected,
                    found: block.site,
                });
            }
            if !seen_sites.insert(block.site) {
                return Err(AuxiliaryIrError::AuxiliarySiteIdentity {
                    expected,
                    found: block.site,
                });
            }
            let mesh_len = block.mesh.len();
            let mut seen_modes = BTreeSet::new();
            for mode in &block.modes {
                if mode.radial.len() != mesh_len {
                    return Err(AuxiliaryIrError::AuxiliaryModeLength {
                        site: block.site,
                        l: mode.l,
                        n: mode.n,
                        expected: mesh_len,
                        actual: mode.radial.len(),
                    });
                }
                if !seen_modes.insert((mode.l, mode.n)) {
                    return Err(AuxiliaryIrError::DuplicateAuxiliaryMode {
                        site: block.site,
                        l: mode.l,
                        n: mode.n,
                    });
                }
            }
        }
        validate_interstitial(self.q, &payload.interstitial)
    }

    fn validate_interpolation_points(
        &self,
        payload: &InterpolationPointAuxiliary,
    ) -> Result<(), AuxiliaryIrError> {
        if payload.points.is_empty() {
            return Err(AuxiliaryIrError::EmptyInterpolationPoints);
        }
        let mut seen = BTreeSet::new();
        let mut any_positive = false;
        for (index, point) in payload.points.iter().enumerate() {
            if !seen.insert(point.id) {
                return Err(AuxiliaryIrError::DuplicateInterpolationPoint(point.id));
            }
            if point
                .coordinate
                .iter()
                .any(|component| !component.get().is_finite())
                || !point.weight.get().is_finite()
            {
                return Err(AuxiliaryIrError::NonFiniteInterpolationPoint(index));
            }
            if point.weight.get() < 0.0 {
                return Err(AuxiliaryIrError::NegativeInterpolationWeight(index));
            }
            if point.weight.get() > 0.0 {
                any_positive = true;
            }
            if let InterpolationRegion::MuffinTin { site } = point.region
                && site >= self.partition.site_count()
            {
                return Err(AuxiliaryIrError::InterpolationPointSite { site });
            }
        }
        if !any_positive {
            return Err(AuxiliaryIrError::NoPositiveInterpolationWeight);
        }
        if !interpolation_point_order(&payload.points) {
            return Err(AuxiliaryIrError::InterpolationPointOrder);
        }
        Ok(())
    }
}

fn mixed_product_regions(payload: &MixedProductAuxiliary) -> Vec<AuxiliaryRegion> {
    let mut regions = Vec::new();
    for block in &payload.sites {
        let mut angular = block.modes.iter().map(|mode| mode.l).collect::<Vec<_>>();
        angular.sort_unstable();
        angular.dedup();
        for l in angular {
            let mut radial = block
                .modes
                .iter()
                .filter(|mode| mode.l == l)
                .collect::<Vec<_>>();
            radial.sort_by_key(|mode| mode.n);
            let l_i = l as i32;
            for m in -l_i..=l_i {
                for mode in &radial {
                    regions.push(AuxiliaryRegion::MuffinTin {
                        site: block.site,
                        l,
                        m,
                        n: mode.n,
                    });
                }
            }
        }
    }
    for wave in &payload.interstitial.waves {
        regions.push(AuxiliaryRegion::Interstitial { g: wave.g });
    }
    regions
}

/// Sort interpolation points into the IR flatten order.
pub fn sort_interpolation_points(points: &mut [InterpolationAuxiliaryPoint]) {
    points.sort_by_key(interpolation_order_key);
}

fn interpolation_order_key(point: &InterpolationAuxiliaryPoint) -> (u8, usize, usize) {
    match point.region {
        InterpolationRegion::MuffinTin { site } => (0, site, point.id),
        InterpolationRegion::Interstitial => (1, 0, point.id),
        InterpolationRegion::Uniform => (2, 0, point.id),
    }
}

fn interpolation_point_order(points: &[InterpolationAuxiliaryPoint]) -> bool {
    points
        .windows(2)
        .all(|pair| interpolation_order_key(&pair[0]) <= interpolation_order_key(&pair[1]))
}

fn validate_interstitial(
    q: TransferQ,
    interstitial: &AuxiliaryInterstitialSupport,
) -> Result<(), AuxiliaryIrError> {
    if interstitial.q != q {
        return Err(AuxiliaryIrError::AuxiliarySupportTransferQ);
    }
    let g_cut = interstitial.g_cut.get();
    if !g_cut.is_finite() || g_cut < 0.0 {
        return Err(AuxiliaryIrError::AuxiliaryWaveCutoff { index: 0 });
    }
    let cutoff_squared = g_cut * g_cut;
    let cutoff_tolerance = 64.0 * f64::EPSILON * cutoff_squared.max(1.0);
    let mut seen = BTreeSet::new();
    for (index, wave) in interstitial.waves.iter().enumerate() {
        if !seen.insert(wave.g.index) {
            return Err(AuxiliaryIrError::DuplicateAuxiliaryWave {
                index: wave.g.index,
            });
        }
        if !wave_kinematics_match(q, wave) {
            return Err(AuxiliaryIrError::AuxiliaryWaveKinematics { index });
        }
        let qg_squared = wave
            .q_plus_g
            .iter()
            .map(|component| component.get().powi(2))
            .sum::<f64>();
        if qg_squared > cutoff_squared + cutoff_tolerance {
            return Err(AuxiliaryIrError::AuxiliaryWaveCutoff { index });
        }
    }
    if !spex_g_order(&interstitial.waves) {
        return Err(AuxiliaryIrError::AuxiliaryWaveOrder);
    }
    Ok(())
}

fn wave_kinematics_match(q: TransferQ, wave: &AuxiliaryInterstitialWave) -> bool {
    let expected: [f64; 3] =
        std::array::from_fn(|axis| q.cartesian[axis].get() + wave.g.cartesian[axis].get());
    let qg_ok = wave
        .q_plus_g
        .iter()
        .zip(expected)
        .all(|(actual, want)| (actual.get() - want).abs() <= WAVE_TOLERANCE);
    let expected_norm = expected
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let g_norm = wave
        .g
        .cartesian
        .iter()
        .map(|component| component.get().powi(2))
        .sum::<f64>()
        .sqrt();
    qg_ok
        && (wave.q_plus_g_norm.get() - expected_norm).abs() <= WAVE_TOLERANCE
        && (wave.g.norm.get() - g_norm).abs() <= WAVE_TOLERANCE
        && wave
            .g
            .cartesian
            .iter()
            .chain(wave.q_plus_g.iter())
            .all(|component| component.get().is_finite())
}

fn spex_g_order(waves: &[AuxiliaryInterstitialWave]) -> bool {
    waves.windows(2).all(|pair| {
        pair[0]
            .g
            .norm
            .get()
            .total_cmp(&pair[1].g.norm.get())
            .then_with(|| pair[0].g.index.cmp(&pair[1].g.index))
            != Ordering::Greater
    })
}
