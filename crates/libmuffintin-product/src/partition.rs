//! Non-overlapping muffin-tin plus interstitial product partition.

use libmuffintin_basis::Provenance;
use libmuffintin_core::{Bohr, InterstitialGeometry};

/// One muffin-tin region of a [`ProductPartition`].
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
pub struct ProductPartition {
    pub sites: Vec<PartitionSite>,
    pub interstitial: InterstitialGeometry,
    pub provenance: Provenance,
}

impl ProductPartition {
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
}
