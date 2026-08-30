//! Plane-wave envelope evaluation plus historical-method-name-free basis
//! specification, layouts, and APW maps.
//!
//! v0.2 implements one plane-wave envelope with APW site augmentation and
//! confined site-local overlays. There is no trait hierarchy and no generic
//! method-neutral payload.

#![forbid(unsafe_code)]

mod augmentation;
mod envelope;
mod layout;
mod spec;
mod spinor_spec;

pub use envelope::{
    EnvelopeError, PlaneWave, PlaneWaveEnvelope, rayleigh_coefficient, site_translation_phase,
};

pub use augmentation::{
    ApwBoundaryBasis, ApwMatch, PlaneWaveAugmentation, SpinorApwMatch, SpinorPlaneWaveAugmentation,
    augmentation_coefficients, match_apw_boundary, spinor_augmentation_coefficients,
};
pub use layout::{BasisLayout, LocalOrbitalLayout, SpinorBasisLayout, SpinorSiteLayout};
pub use spec::{
    ApwSiteAugmentation, ApwSiteGeometry, BasisBlock, BasisSpec, CompiledBasis, Provenance,
    SpinorCompiledBasis, compile,
};
pub use spinor_spec::{SpinorBasisSite, SpinorBasisSpec, compile_spinor};

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
    #[error("spinor site layout contains kappa={kappa} more than once")]
    DuplicateKappa { kappa: i32 },
    #[error("spinor APW matches contain kappa={kappa} more than once")]
    DuplicateSpinorMatch { kappa: i32 },
    #[error("spinor basis site {site} has no valence radial solutions")]
    EmptySpinorRadialSet { site: usize },
    #[error("spinor basis site {site} contains kappa={kappa} more than once")]
    DuplicateSpinorSolution { site: usize, kappa: i32 },
    #[error("spinor basis site {site} kappa={kappa} is not a valence radial solution")]
    NonValenceSpinorSolution { site: usize, kappa: i32 },
    #[error(
        "spinor site {site} kappa={kappa} {boundary} boundary radius {actual} does not match {expected}"
    )]
    SpinorBoundaryRadius {
        site: usize,
        kappa: i32,
        boundary: &'static str,
        expected: f64,
        actual: f64,
    },
    #[error("spinor site {site} local-orbital kappa={kappa} has no APW radial solution")]
    SpinorLocalOrbitalWithoutRadialSolution { site: usize, kappa: i32 },
    #[error("spinor APW match for kappa={kappa} must use l={expected}, found l={actual}")]
    SpinorMatchAngularMomentum {
        kappa: i32,
        expected: u32,
        actual: u32,
    },
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
    use muffintin_core::{Bohr, InverseBohr, Kappa, Lm, ReciprocalLattice, TwiceMu, VolumeBohr3};
    use crate::{
        PlaneWave, PlaneWaveEnvelope, rayleigh_coefficient, site_translation_phase,
    };
    use muffintin_sphere::BoundaryData;

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
    fn spinor_layout_uses_spin_g_then_site_kappa_mu_n_order() {
        let minus_two = Kappa::new(-2).unwrap();
        let minus_one = Kappa::new(-1).unwrap();
        let plus_one = Kappa::new(1).unwrap();
        let first =
            SpinorSiteLayout::new(vec![(plus_one, 1), (minus_one, 2), (minus_two, 1)]).unwrap();
        assert_eq!(
            first.counts_by_kappa(),
            &[(minus_two, 1), (minus_one, 2), (plus_one, 1)]
        );
        assert_eq!(first.len(), 10);
        assert_eq!(
            first.index(minus_two, TwiceMu::new(-3).unwrap(), 0),
            Some(0)
        );
        assert_eq!(first.index(minus_two, TwiceMu::new(3).unwrap(), 0), Some(3));
        assert_eq!(
            first.index(minus_one, TwiceMu::new(-1).unwrap(), 0),
            Some(4)
        );
        assert_eq!(
            first.index(minus_one, TwiceMu::new(-1).unwrap(), 1),
            Some(5)
        );
        assert_eq!(first.index(minus_one, TwiceMu::new(1).unwrap(), 0), Some(6));
        assert_eq!(first.index(plus_one, TwiceMu::new(1).unwrap(), 0), Some(9));
        assert_eq!(first.index(plus_one, TwiceMu::new(3).unwrap(), 0), None);

        let second = SpinorSiteLayout::new(vec![(minus_one, 1)]).unwrap();
        let layout = SpinorBasisLayout::new(3, vec![first, second]);
        assert_eq!(layout.plane_wave_range(), 0..6);
        assert_eq!(layout.plane_wave_spin_range(0), Some(0..3));
        assert_eq!(layout.plane_wave_spin_range(1), Some(3..6));
        assert_eq!(layout.plane_wave_index(0, 2), Some(2));
        assert_eq!(layout.plane_wave_index(1, 0), Some(3));
        assert_eq!(layout.plane_wave_index(2, 0), None);
        assert_eq!(layout.site_spinor_range(0), Some(6..16));
        assert_eq!(layout.site_spinor_range(1), Some(16..18));
        assert_eq!(
            layout.site_spinor_index(0, minus_one, TwiceMu::new(1).unwrap(), 1),
            Some(13)
        );
        assert_eq!(layout.dimension(), 18);
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

    #[test]
    fn spinor_augmentation_selects_rayleigh_orbitals_and_cg_coefficients() {
        let wave = waves()[0];
        let kappa = Kappa::new(1).unwrap();
        let radial = [0.7, -0.2];
        let augmentation = spinor_augmentation_coefficients(
            &wave,
            [Bohr(0.0); 3],
            VolumeBohr3(80.0),
            &[SpinorApwMatch {
                kappa,
                apw: ApwMatch {
                    l: 1,
                    coefficients: radial,
                    value_residual: 0.0,
                    slope_residual: 0.0,
                },
            }],
        )
        .unwrap();
        assert_eq!(augmentation.channels.len(), 2);
        assert_eq!(augmentation.augmented_site_coordinate_count(), 4);
        assert_eq!(augmentation.site_coordinate_index(1, 0), Some(2));
        assert_eq!(augmentation.site_coordinate_index(1, 1), Some(3));
        assert_eq!(augmentation.channels[1].kappa(), kappa);
        assert_eq!(
            augmentation.channels[1].twice_mu(),
            TwiceMu::new(1).unwrap()
        );

        let up_angular = rayleigh_coefficient(Lm::new(1, 0).unwrap(), wave.q, VolumeBohr3(80.0))
            .unwrap()
            * -(1.0_f64 / 3.0).sqrt();
        let down_angular = rayleigh_coefficient(Lm::new(1, 1).unwrap(), wave.q, VolumeBohr3(80.0))
            .unwrap()
            * (2.0_f64 / 3.0).sqrt();
        for (radial_column, &radial_coefficient) in radial.iter().enumerate() {
            assert!(
                (augmentation.coefficient(0, 1)[radial_column] - up_angular * radial_coefficient)
                    .norm()
                    < 1.0e-14
            );
            assert!(
                (augmentation.coefficient(1, 1)[radial_column] - down_angular * radial_coefficient)
                    .norm()
                    < 1.0e-14
            );
        }
    }

    #[test]
    fn spinor_augmentation_carries_translated_site_phase() {
        let wave = waves()[0];
        let matched = [SpinorApwMatch {
            kappa: Kappa::new(1).unwrap(),
            apw: ApwMatch {
                l: 1,
                coefficients: [0.7, -0.2],
                value_residual: 0.0,
                slope_residual: 0.0,
            },
        }];
        let origin =
            spinor_augmentation_coefficients(&wave, [Bohr(0.0); 3], VolumeBohr3(80.0), &matched)
                .unwrap();
        let site = [Bohr(0.31), Bohr(-0.27), Bohr(0.19)];
        let translated =
            spinor_augmentation_coefficients(&wave, site, VolumeBohr3(80.0), &matched).unwrap();
        let phase = site_translation_phase(wave.q, site);
        for spin in 0..2 {
            for channel in 0..origin.channels.len() {
                for radial in 0..2 {
                    assert!(
                        (translated.coefficient(spin, channel)[radial]
                            - phase * origin.coefficient(spin, channel)[radial])
                            .norm()
                            < 1.0e-14
                    );
                }
            }
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
