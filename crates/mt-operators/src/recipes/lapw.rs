//! SPEX-compatible LAPW (APW+lo) recipe.

use super::lapw_provenance;
use muffintin_envelope::{
    ApwBoundaryBasis, ApwSiteAugmentation, BasisBlock, BasisSpec, LocalOrbitalLayout,
};
use muffintin_core::{Bohr, VolumeBohr3};
use muffintin_envelope::PlaneWaveEnvelope;

/// One muffin-tin site in the LAPW recipe.
#[derive(Clone, Debug, PartialEq)]
pub struct LapwSiteInput {
    pub position: [Bohr; 3],
    pub radius: Bohr,
    pub boundaries: Vec<ApwBoundaryBasis>,
    pub local_orbitals: LocalOrbitalLayout,
}

/// Construct a [`BasisSpec`] for the SPEX APW+lo preset.
///
/// This fills the v0.2 plane-wave envelope block, APW site augmentations,
/// confined local-orbital overlays, and provenance. It does not assemble `H`
/// or `S` and does not introduce a trait hierarchy.
pub fn lapw(
    envelope: PlaneWaveEnvelope,
    cell_volume: VolumeBohr3,
    sites: &[LapwSiteInput],
) -> BasisSpec {
    let provenance = lapw_provenance();
    let apw_sites = sites
        .iter()
        .map(|site| ApwSiteAugmentation {
            position: site.position,
            radius: site.radius,
            boundaries: site.boundaries.clone(),
        })
        .collect();
    let mut blocks = vec![BasisBlock::PlaneWaveEnvelope {
        envelope,
        sites: apw_sites,
    }];
    for (index, site) in sites.iter().enumerate() {
        if !site.local_orbitals.is_empty() {
            blocks.push(BasisBlock::ConfinedSite {
                site: index,
                local_orbitals: site.local_orbitals.clone(),
            });
        }
    }
    BasisSpec {
        blocks,
        cell_volume,
        provenance,
    }
}
