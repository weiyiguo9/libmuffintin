//! Retained auxiliary basis shared by mixed-product and later THC paths.

use crate::{ProductError, ProductPartition, ProductSource, TransferQ};
use libmuffintin_basis::Provenance;
use libmuffintin_core::{ExponentialMesh, GVector, InverseBohr};
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
}

/// Retained auxiliary basis at one transfer $q$.
///
/// A consumer can integrate muffin-tin modes using each site's mesh without
/// a [`crate::ProductSource`]. Interstitial support is the MPB $|q+G|$ view,
/// not raw orbital-pair reciprocal support.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledAuxiliaryBasis {
    pub partition: ProductPartition,
    pub q: TransferQ,
    pub sites: Vec<SiteAuxiliaryBlock>,
    pub interstitial: AuxiliaryInterstitialSupport,
    pub cutoff: Option<CutoffRecord>,
    pub provenance: Provenance,
}

impl CompiledAuxiliaryBasis {
    /// Global muffin-tin auxiliary dimension, expanding $M = -L,\ldots,L$.
    pub fn mt_dimension(&self) -> usize {
        self.sites
            .iter()
            .flat_map(|block| block.modes.iter())
            .map(|mode| 2 * mode.l as usize + 1)
            .sum()
    }

    /// Number of interstitial $|q+G|$ plane waves.
    pub fn interstitial_dimension(&self) -> usize {
        self.interstitial.waves.len()
    }

    /// Total auxiliary dimension: muffin-tin block then interstitial block.
    pub fn dimension(&self) -> usize {
        self.mt_dimension() + self.interstitial_dimension()
    }

    /// Combined regions in SPEX muffin-tin order, then interstitial $G$.
    ///
    /// Muffin-tin flatten is $site \to L \to M=-L..L \to n$.
    pub fn regions(&self) -> Vec<AuxiliaryRegion> {
        let mut regions = Vec::with_capacity(self.dimension());
        for block in &self.sites {
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
        for wave in &self.interstitial.waves {
            regions.push(AuxiliaryRegion::Interstitial { g: wave.g });
        }
        regions
    }

    /// Deterministic muffin-tin index: $site \to L \to M \to n$.
    pub fn mt_index(&self, site: usize, l: u32, m: i32, n: usize) -> Option<usize> {
        self.regions().into_iter().position(|region| {
            matches!(
                region,
                AuxiliaryRegion::MuffinTin {
                    site: s,
                    l: ll,
                    m: mm,
                    n: nn,
                } if s == site && ll == l && mm == m && nn == n
            )
        })
    }

    /// Mesh of one muffin-tin site.
    pub fn site_mesh(&self, site: usize) -> Option<&ExponentialMesh> {
        self.sites
            .iter()
            .find(|block| block.site == site)
            .map(|block| &block.mesh)
    }

    /// Retained radial mode $(site, L, n)$.
    pub fn mt_mode(&self, site: usize, l: u32, n: usize) -> Option<&MtAuxiliaryMode> {
        self.sites
            .iter()
            .find(|block| block.site == site)
            .and_then(|block| block.modes.iter().find(|mode| mode.l == l && mode.n == n))
    }

    /// Reject inconsistent site identity, mode lengths, duplicates, and waves.
    pub fn validate(&self) -> Result<(), ProductError> {
        if self.sites.len() != self.partition.site_count() {
            return Err(ProductError::AuxiliarySiteCount {
                expected: self.partition.site_count(),
                actual: self.sites.len(),
            });
        }
        let mut seen_sites = BTreeSet::new();
        for (expected, block) in self.sites.iter().enumerate() {
            let partition_index = self.partition.sites.get(expected).map(|site| site.index);
            if block.site != expected || partition_index != Some(block.site) {
                return Err(ProductError::AuxiliarySiteIdentity {
                    expected,
                    found: block.site,
                });
            }
            if !seen_sites.insert(block.site) {
                return Err(ProductError::AuxiliarySiteIdentity {
                    expected,
                    found: block.site,
                });
            }
            let mesh_len = block.mesh.len();
            let mut seen_modes = BTreeSet::new();
            for mode in &block.modes {
                if mode.radial.len() != mesh_len {
                    return Err(ProductError::AuxiliaryModeLength {
                        site: block.site,
                        l: mode.l,
                        n: mode.n,
                        expected: mesh_len,
                        actual: mode.radial.len(),
                    });
                }
                if !seen_modes.insert((mode.l, mode.n)) {
                    return Err(ProductError::DuplicateAuxiliaryMode {
                        site: block.site,
                        l: mode.l,
                        n: mode.n,
                    });
                }
            }
        }
        self.validate_interstitial()?;
        Ok(())
    }

    /// Site meshes must equal the source meshes used to build the modes.
    pub fn validate_against_source(&self, source: &ProductSource) -> Result<(), ProductError> {
        self.validate()?;
        if self.q != source.q || self.interstitial.q != source.q {
            return Err(ProductError::AuxiliarySupportTransferQ);
        }
        if self.sites.len() != source.radials.len() {
            return Err(ProductError::AuxiliarySiteCount {
                expected: source.radials.len(),
                actual: self.sites.len(),
            });
        }
        for (site, (block, radials)) in self.sites.iter().zip(&source.radials).enumerate() {
            if block.mesh != radials.mesh {
                return Err(ProductError::AuxiliaryMeshMismatch { site });
            }
        }
        Ok(())
    }

    fn validate_interstitial(&self) -> Result<(), ProductError> {
        if self.interstitial.q != self.q {
            return Err(ProductError::AuxiliarySupportTransferQ);
        }
        let g_cut = self.interstitial.g_cut.get();
        if !g_cut.is_finite() || g_cut < 0.0 {
            return Err(ProductError::AuxiliaryWaveCutoff { index: 0 });
        }
        let cutoff_squared = g_cut * g_cut;
        let cutoff_tolerance = 64.0 * f64::EPSILON * cutoff_squared.max(1.0);
        let mut seen = BTreeSet::new();
        for (index, wave) in self.interstitial.waves.iter().enumerate() {
            if !seen.insert(wave.g.index) {
                return Err(ProductError::DuplicateAuxiliaryWave {
                    index: wave.g.index,
                });
            }
            if !wave_kinematics_match(self.q, wave) {
                return Err(ProductError::AuxiliaryWaveKinematics { index });
            }
            let qg_squared = wave
                .q_plus_g
                .iter()
                .map(|component| component.get().powi(2))
                .sum::<f64>();
            if qg_squared > cutoff_squared + cutoff_tolerance {
                return Err(ProductError::AuxiliaryWaveCutoff { index });
            }
        }
        if !spex_g_order(&self.interstitial.waves) {
            return Err(ProductError::AuxiliaryWaveOrder);
        }
        Ok(())
    }
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
