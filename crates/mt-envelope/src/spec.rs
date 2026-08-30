//! Anonymous [`BasisSpec`] and compilation to host augmentation coefficients.

use crate::{
    ApwBoundaryBasis, BasisError, BasisLayout, LocalOrbitalLayout, PlaneWaveAugmentation,
    SpinorBasisLayout, SpinorPlaneWaveAugmentation, augmentation_coefficients, match_apw_boundary,
};
use muffintin_core::{Bohr, VolumeBohr3};
use crate::{PlaneWave, PlaneWaveEnvelope};

/// Provenance for a basis specification or compiled result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Provenance {
    pub recipe: Option<String>,
    pub reference: Option<String>,
}

/// APW matching data for one muffin-tin site belonging to a plane-wave envelope.
#[derive(Clone, Debug, PartialEq)]
pub struct ApwSiteAugmentation {
    pub position: [Bohr; 3],
    pub radius: Bohr,
    pub boundaries: Vec<ApwBoundaryBasis>,
}

/// One historical-method-name-free block of a [`BasisSpec`].
///
/// v0.2 typed variants implement one [`BasisBlock::PlaneWaveEnvelope`] with
/// APW site augmentation plus zero or more [`BasisBlock::ConfinedSite`]
/// overlays. This is not an arbitrary method-neutral payload and does not
/// introduce a trait hierarchy.
#[derive(Clone, Debug, PartialEq)]
pub enum BasisBlock {
    /// Owned plane-wave envelope and the APW site augmentations of that envelope.
    PlaneWaveEnvelope {
        envelope: PlaneWaveEnvelope,
        sites: Vec<ApwSiteAugmentation>,
    },
    /// Site-local confined functions (local orbitals) on an existing APW site.
    ConfinedSite {
        site: usize,
        local_orbitals: LocalOrbitalLayout,
    },
}

/// Historical-method-name-free basis specification.
///
/// v0.2 stores only the typed variants above. Later recipes may add blocks
/// without renaming these ones after a historical method.
#[derive(Clone, Debug, PartialEq)]
pub struct BasisSpec {
    pub blocks: Vec<BasisBlock>,
    pub cell_volume: VolumeBohr3,
    pub provenance: Provenance,
}

/// Position and muffin-tin radius of one compiled APW site.
///
/// Assembly compares this metadata with interstitial spheres. It is not a
/// generic geometry payload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApwSiteGeometry {
    pub position: [Bohr; 3],
    pub radius: Bohr,
}

/// Host-side compiled layout, APW augmentation coefficients, and site geometry.
///
/// Projection tensors are built later in `libmuffintin-operators`; this type
/// does not store backend tensor handles.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledBasis {
    pub layout: BasisLayout,
    pub plane_waves: Vec<PlaneWave>,
    pub site_augmentations: Vec<Vec<PlaneWaveAugmentation>>,
    pub site_geometry: Vec<ApwSiteGeometry>,
    pub provenance: Provenance,
}

impl CompiledBasis {
    pub fn site_count(&self) -> usize {
        self.layout.site_count()
    }
}

/// Host-side compiled spinor layout and SRA plane-wave augmentations.
///
/// `plane_waves` contains the spatial `G` list once.  The two global
/// plane-wave columns for each entry are addressed by [`SpinorBasisLayout`]
/// as `spin * n_g + g`.  `site_augmentations[site][g]` stores both Pauli-spin
/// columns for direct construction of a site projection.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorCompiledBasis {
    pub layout: SpinorBasisLayout,
    pub plane_waves: Vec<PlaneWave>,
    pub site_augmentations: Vec<Vec<SpinorPlaneWaveAugmentation>>,
    pub site_geometry: Vec<ApwSiteGeometry>,
    pub provenance: Provenance,
}

impl SpinorCompiledBasis {
    pub fn site_count(&self) -> usize {
        self.layout.site_count()
    }
}

/// Compile a [`BasisSpec`] into a host [`CompiledBasis`].
///
/// Plane waves come only from the envelope block. An empty spec therefore
/// yields an empty basis. Duplicate confined-site blocks are rejected rather
/// than silently overwritten. v0.2 accepts at most one plane-wave envelope.
pub fn compile(spec: &BasisSpec) -> Result<CompiledBasis, BasisError> {
    let mut envelope_block = None;
    let mut confined = Vec::new();
    for block in &spec.blocks {
        match block {
            BasisBlock::PlaneWaveEnvelope { envelope, sites } => {
                if envelope_block.is_some() {
                    return Err(BasisError::MultiplePlaneWaveEnvelopes);
                }
                envelope_block = Some((envelope, sites.as_slice()));
            }
            BasisBlock::ConfinedSite {
                site,
                local_orbitals,
            } => confined.push((*site, local_orbitals)),
        }
    }

    let Some((envelope, apw_sites)) = envelope_block else {
        if let Some((site, _)) = confined.first() {
            return Err(BasisError::UnknownSite { site: *site });
        }
        return Ok(CompiledBasis {
            layout: BasisLayout::new(0, Vec::new()),
            plane_waves: Vec::new(),
            site_augmentations: Vec::new(),
            site_geometry: Vec::new(),
            provenance: spec.provenance.clone(),
        });
    };

    let waves = envelope.waves();
    let mut local_orbitals = vec![LocalOrbitalLayout::default(); apw_sites.len()];
    let mut occupied = vec![false; apw_sites.len()];
    for (site, layout) in confined {
        if site >= apw_sites.len() {
            return Err(BasisError::UnknownSite { site });
        }
        if occupied[site] {
            return Err(BasisError::DuplicateConfinedSite { site });
        }
        occupied[site] = true;
        local_orbitals[site] = layout.clone();
    }

    let layout = BasisLayout::new(waves.len(), local_orbitals);
    let mut site_augmentations = Vec::with_capacity(apw_sites.len());
    for site in apw_sites {
        let mut augmentations = Vec::with_capacity(waves.len());
        for wave in waves {
            let matches = (0..site.boundaries.len())
                .map(|l| match_apw_boundary(l as u32, wave.q_norm, site.radius, site.boundaries[l]))
                .collect::<Result<Vec<_>, _>>()?;
            augmentations.push(augmentation_coefficients(
                wave,
                site.position,
                spec.cell_volume,
                &matches,
            )?);
        }
        site_augmentations.push(augmentations);
    }

    Ok(CompiledBasis {
        layout,
        plane_waves: waves.to_vec(),
        site_augmentations,
        site_geometry: apw_sites
            .iter()
            .map(|site| ApwSiteGeometry {
                position: site.position,
                radius: site.radius,
            })
            .collect(),
        provenance: spec.provenance.clone(),
    })
}
