//! Historical-method-name-free basis specification, layouts, and APW maps.
//!
//! v0.2 implements one plane-wave envelope with APW site augmentation and
//! confined site-local overlays. There is no trait hierarchy and no generic
//! method-neutral payload.

#![forbid(unsafe_code)]

mod augmentation;
mod layout;
mod spec;

pub use augmentation::{
    ApwBoundaryBasis, ApwMatch, PlaneWaveAugmentation, augmentation_coefficients,
    match_apw_boundary,
};
pub use layout::{BasisLayout, LocalOrbitalLayout};
pub use spec::{
    ApwSiteAugmentation, ApwSiteGeometry, BasisBlock, BasisSpec, CompiledBasis, Provenance, compile,
};

use libmuffintin_envelope::EnvelopeError;
use thiserror::Error;

/// Basis construction, matching, or compilation error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum BasisError {
    #[error("muffin-tin radius must be finite and positive, got {0}")]
    InvalidRadius(f64),
    #[error("APW boundary matrix for l={l} is singular (determinant {determinant})")]
    SingularBoundaryMatrix { l: u32, determinant: f64 },
    #[error("APW matches must be ordered by l: expected {expected}, found {actual}")]
    MatchAngularMomentum { expected: u32, actual: u32 },
    #[error("wave-vector norm must be finite and nonnegative, got {0}")]
    InvalidWaveVector(f64),
    #[error("site {site} is not present in the envelope APW site list")]
    UnknownSite { site: usize },
    #[error("confined-site block for site {site} appears more than once")]
    DuplicateConfinedSite { site: usize },
    #[error("v0.2 compile accepts at most one plane-wave envelope block")]
    MultiplePlaneWaveEnvelopes,
    #[error(transparent)]
    Envelope(#[from] EnvelopeError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use libmuffintin_core::{Bohr, InverseBohr, ReciprocalLattice, VolumeBohr3};
    use libmuffintin_envelope::{PlaneWave, PlaneWaveEnvelope, site_translation_phase};
    use libmuffintin_radial::BoundaryData;

    fn boundary(value: f64, derivative: f64) -> BoundaryData {
        BoundaryData {
            value,
            derivative,
            log_derivative: (value != 0.0).then(|| InverseBohr(derivative / value)),
            scaled_log_derivative: None,
        }
    }

    fn waves() -> Vec<PlaneWave> {
        let lattice = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        lattice
            .enumerate(InverseBohr(1.0))
            .unwrap()
            .into_iter()
            .map(|g| PlaneWave::new([InverseBohr(0.1), InverseBohr(-0.2), InverseBohr(0.05)], g))
            .collect()
    }

    #[test]
    fn matching_residuals_are_small() {
        let basis = ApwBoundaryBasis {
            u: boundary(0.73, -0.21),
            udot: boundary(-0.18, 1.14),
        };
        for l in 0..=8 {
            let matched = match_apw_boundary(l, InverseBohr(2.3), Bohr(1.7), basis).unwrap();
            assert!(matched.value_residual.abs() <= 1.0e-10);
            assert!(matched.slope_residual.abs() <= 1.0e-10);
        }
    }

    #[test]
    fn local_orbital_layout_uses_site_l_m_n_order_and_global_offsets() {
        let first = LocalOrbitalLayout::new(vec![2, 1]);
        let second = LocalOrbitalLayout::new(vec![0, 2, 1]);
        assert_eq!(first.len(), 5);
        assert_eq!(first.index(0, 0, 0), Some(0));
        assert_eq!(first.index(0, 0, 1), Some(1));
        assert_eq!(first.index(1, -1, 0), Some(2));
        assert_eq!(first.index(1, 1, 0), Some(4));
        assert_eq!(first.index(1, 1, 1), None);

        let layout = BasisLayout::new(7, vec![first, second]);
        assert_eq!(layout.site_local_orbital_range(0), Some(7..12));
        assert_eq!(layout.site_local_orbital_range(1), Some(12..23));
        assert_eq!(layout.local_orbital_index(1, 1, -1, 1), Some(13));
        assert_eq!(layout.local_orbital_index(1, 2, 2, 0), Some(22));
        assert_eq!(layout.dimension(), 23);
    }

    #[test]
    fn nonzero_k_site_phase_is_carried_by_augmentation_coefficients() {
        let wave = waves()[0];
        assert!(wave.k.iter().any(|component| component.get() != 0.0));
        let matched = [ApwMatch {
            l: 0,
            coefficients: [0.7, -0.2],
            value_residual: 0.0,
            slope_residual: 0.0,
        }];
        let origin =
            augmentation_coefficients(&wave, [Bohr(0.0); 3], VolumeBohr3(80.0), &matched).unwrap();
        let site = [Bohr(0.31), Bohr(-0.27), Bohr(0.19)];
        let translated =
            augmentation_coefficients(&wave, site, VolumeBohr3(80.0), &matched).unwrap();
        let phase = site_translation_phase(wave.q, site);
        for radial in 0..2 {
            assert!(
                (translated.coefficients[0][radial] - phase * origin.coefficients[0][radial])
                    .norm()
                    < 1.0e-14
            );
        }
    }

    fn empty_volume() -> VolumeBohr3 {
        VolumeBohr3(100.0)
    }

    #[test]
    fn empty_spec_compiles_to_empty_basis() {
        let compiled = compile(&BasisSpec {
            blocks: Vec::new(),
            cell_volume: empty_volume(),
            provenance: Provenance::default(),
        })
        .unwrap();
        assert!(compiled.plane_waves.is_empty());
        assert_eq!(compiled.layout.dimension(), 0);
        assert_eq!(compiled.site_count(), 0);
    }

    #[test]
    fn compile_reads_waves_from_the_envelope_block() {
        let waves = waves();
        let compiled = compile(&BasisSpec {
            blocks: vec![BasisBlock::PlaneWaveEnvelope {
                envelope: PlaneWaveEnvelope::new(waves.clone()),
                sites: Vec::new(),
            }],
            cell_volume: empty_volume(),
            provenance: Provenance::default(),
        })
        .unwrap();
        assert_eq!(compiled.plane_waves, waves);
        assert_eq!(compiled.site_count(), 0);
        assert!(compiled.site_geometry.is_empty());
    }

    #[test]
    fn compile_retains_apw_site_geometry() {
        let position = [Bohr(0.31), Bohr(-0.27), Bohr(0.19)];
        let radius = Bohr(0.8);
        let compiled = compile(&BasisSpec {
            blocks: vec![BasisBlock::PlaneWaveEnvelope {
                envelope: PlaneWaveEnvelope::new(waves()),
                sites: vec![ApwSiteAugmentation {
                    position,
                    radius,
                    boundaries: Vec::new(),
                }],
            }],
            cell_volume: empty_volume(),
            provenance: Provenance::default(),
        })
        .unwrap();
        assert_eq!(
            compiled.site_geometry,
            vec![ApwSiteGeometry { position, radius }]
        );
    }

    #[test]
    fn duplicate_confined_site_is_an_error() {
        let spec = BasisSpec {
            blocks: vec![
                BasisBlock::PlaneWaveEnvelope {
                    envelope: PlaneWaveEnvelope::new(waves()),
                    sites: vec![ApwSiteAugmentation {
                        position: [Bohr(0.0); 3],
                        radius: Bohr(0.8),
                        boundaries: Vec::new(),
                    }],
                },
                BasisBlock::ConfinedSite {
                    site: 0,
                    local_orbitals: LocalOrbitalLayout::new(vec![1]),
                },
                BasisBlock::ConfinedSite {
                    site: 0,
                    local_orbitals: LocalOrbitalLayout::new(vec![2]),
                },
            ],
            cell_volume: empty_volume(),
            provenance: Provenance::default(),
        };
        assert_eq!(
            compile(&spec),
            Err(BasisError::DuplicateConfinedSite { site: 0 })
        );
    }

    #[test]
    fn two_envelope_blocks_are_rejected() {
        let envelope = PlaneWaveEnvelope::new(waves());
        let spec = BasisSpec {
            blocks: vec![
                BasisBlock::PlaneWaveEnvelope {
                    envelope: envelope.clone(),
                    sites: Vec::new(),
                },
                BasisBlock::PlaneWaveEnvelope {
                    envelope,
                    sites: Vec::new(),
                },
            ],
            cell_volume: empty_volume(),
            provenance: Provenance::default(),
        };
        assert_eq!(compile(&spec), Err(BasisError::MultiplePlaneWaveEnvelopes));
    }

    #[test]
    fn confined_site_without_apw_site_is_unknown() {
        let spec = BasisSpec {
            blocks: vec![
                BasisBlock::PlaneWaveEnvelope {
                    envelope: PlaneWaveEnvelope::new(waves()),
                    sites: Vec::new(),
                },
                BasisBlock::ConfinedSite {
                    site: 0,
                    local_orbitals: LocalOrbitalLayout::new(vec![1]),
                },
            ],
            cell_volume: empty_volume(),
            provenance: Provenance::default(),
        };
        assert_eq!(compile(&spec), Err(BasisError::UnknownSite { site: 0 }));
    }
}
