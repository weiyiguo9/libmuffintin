//! Validated basis presets, defaults, and provenance.

#![forbid(unsafe_code)]

mod lapw;

pub use lapw::{LapwSiteInput, lapw};

use muffintin_basis::Provenance;

pub(crate) fn lapw_provenance() -> Provenance {
    Provenance {
        recipe: Some("lapw".to_owned()),
        reference: Some("SPEX APW+lo".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_basis::{
        ApwBoundaryBasis, ApwSiteAugmentation, BasisBlock, BasisSpec, LocalOrbitalLayout, compile,
    };
    use muffintin_core::{Bohr, InverseBohr, ReciprocalLattice, VolumeBohr3};
    use muffintin_envelope::{PlaneWave, PlaneWaveEnvelope};
    use muffintin_radial::BoundaryData;

    fn boundary(value: f64, derivative: f64) -> BoundaryData {
        BoundaryData {
            value,
            derivative,
            log_derivative: (value != 0.0).then(|| InverseBohr(derivative / value)),
            scaled_log_derivative: None,
        }
    }

    #[test]
    fn lapw_recipe_matches_handwritten_compiled_spec() {
        let lattice = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        let waves = lattice
            .enumerate(InverseBohr(1.0))
            .unwrap()
            .into_iter()
            .map(|g| PlaneWave::new([InverseBohr(0.0); 3], g))
            .collect::<Vec<_>>();
        let envelope = PlaneWaveEnvelope::new(waves);
        let volume = VolumeBohr3(100.0);
        let position = [Bohr(0.1); 3];
        let radius = Bohr(0.8);
        let apw_boundary = ApwBoundaryBasis {
            u: boundary(0.8, -0.1),
            udot: boundary(0.2, 1.1),
        };
        let boundaries = vec![apw_boundary];
        let local_orbitals = LocalOrbitalLayout::new(vec![1]);
        let sites = [LapwSiteInput {
            position,
            radius,
            boundaries: boundaries.clone(),
            local_orbitals: local_orbitals.clone(),
        }];

        let recipe = lapw(envelope.clone(), volume, &sites);
        let handwritten = BasisSpec {
            blocks: vec![
                BasisBlock::PlaneWaveEnvelope {
                    envelope,
                    sites: vec![ApwSiteAugmentation {
                        position,
                        radius,
                        boundaries,
                    }],
                },
                BasisBlock::ConfinedSite {
                    site: 0,
                    local_orbitals,
                },
            ],
            cell_volume: volume,
            provenance: Provenance::default(),
        };

        assert_eq!(recipe.provenance.recipe.as_deref(), Some("lapw"));
        assert_eq!(recipe.provenance.reference.as_deref(), Some("SPEX APW+lo"));
        assert_eq!(recipe.blocks, handwritten.blocks);
        assert_eq!(recipe.cell_volume, handwritten.cell_volume);

        let recipe_compiled = compile(&recipe).unwrap();
        let handwritten_compiled = compile(&handwritten).unwrap();
        assert_eq!(recipe_compiled.layout, handwritten_compiled.layout);
        assert_eq!(
            recipe_compiled.plane_waves,
            handwritten_compiled.plane_waves
        );
        assert_eq!(
            recipe_compiled.site_augmentations,
            handwritten_compiled.site_augmentations
        );
        assert_eq!(
            recipe_compiled.site_geometry,
            handwritten_compiled.site_geometry
        );
        assert_eq!(recipe_compiled.site_geometry[0].position, position);
        assert_eq!(recipe_compiled.site_geometry[0].radius, radius);
        assert_eq!(recipe_compiled.provenance, recipe.provenance);
        assert_eq!(recipe_compiled.layout.site_count(), 1);
        assert_eq!(
            recipe_compiled.layout.dimension(),
            recipe_compiled.plane_waves.len() + 1
        );
        assert!(!recipe_compiled.site_augmentations[0].is_empty());
        assert!(
            !recipe_compiled.site_augmentations[0][0]
                .coefficients
                .is_empty()
        );
    }
}
