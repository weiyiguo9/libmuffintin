//! Non-overlapping muffin-tin plus interstitial product partition.

use muffintin_basis::Provenance;
use muffintin_core::{Bohr, InterstitialGeometry};

/// One muffin-tin region of a [`AuxiliaryPartition`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartitionSite {
    pub index: usize,
    pub position: [Bohr; 3],
    pub radius: Bohr,
}

/// Independent auxiliary partition: muffin-tin spheres plus interstitial.
///
/// v0.2 implements only the non-overlapping LAPW/full-potential geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct AuxiliaryPartition {
    sites: Vec<PartitionSite>,
    interstitial: InterstitialGeometry,
    provenance: Provenance,
}

impl AuxiliaryPartition {
    /// Build a partition from validated interstitial geometry.
    pub fn from_interstitial(interstitial: InterstitialGeometry) -> Self {
        let sites = interstitial
            .spheres()
            .iter()
            .enumerate()
            .map(|(index, sphere)| PartitionSite {
                index,
                position: sphere.center,
                radius: sphere.radius,
            })
            .collect();
        Self {
            sites,
            interstitial,
            provenance: Provenance::default(),
        }
    }

    /// Number of muffin-tin sites.
    pub fn site_count(&self) -> usize {
        self.sites.len()
    }

    /// Muffin-tin regions derived from the interstitial spheres.
    pub fn sites(&self) -> &[PartitionSite] {
        &self.sites
    }

    /// One muffin-tin region by stable site index.
    pub fn site(&self, index: usize) -> Option<&PartitionSite> {
        self.sites.get(index)
    }

    /// Interstitial geometry from which the site list was derived.
    pub const fn interstitial(&self) -> &InterstitialGeometry {
        &self.interstitial
    }

    /// Construction provenance.
    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}
