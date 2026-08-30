//! Typed SRA spinor basis specification and compilation.

use crate::PlaneWaveEnvelope;
use crate::{
    ApwBoundaryBasis, ApwSiteGeometry, BasisError, Provenance, SpinorApwMatch, SpinorBasisLayout,
    SpinorCompiledBasis, SpinorSiteLayout, match_apw_boundary, spinor_augmentation_coefficients,
};
use muffintin_core::{Bohr, VolumeBohr3};
use muffintin_sphere::{RelativisticRole, ValenceDiracSolution};
use std::collections::BTreeSet;

/// One muffin-tin site in a first-variation SRA spinor basis.
///
/// `radial_solutions` contains one four-component linearization solution for
/// every APW `kappa`. Its base and analytic energy derivative supply the
/// `(u, udot)` boundary pair. Local-orbital counts remain explicitly typed by
/// signed `kappa`; no scalar `lm` layout is reinterpreted.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorBasisSite {
    pub position: [Bohr; 3],
    pub radius: Bohr,
    pub radial_solutions: Vec<ValenceDiracSolution>,
    pub local_orbitals: SpinorSiteLayout,
}

/// Owned input for compiling a two-component-interstitial/four-component-site basis.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorBasisSpec {
    pub envelope: PlaneWaveEnvelope,
    pub sites: Vec<SpinorBasisSite>,
    pub cell_volume: VolumeBohr3,
    pub provenance: Provenance,
}

/// Compile canonical `(kappa, twice_mu)` augmentation channels and typed LOs.
pub fn compile_spinor(spec: &SpinorBasisSpec) -> Result<SpinorCompiledBasis, BasisError> {
    let plane_waves = spec.envelope.waves().to_vec();
    let mut site_layouts = Vec::with_capacity(spec.sites.len());
    let mut site_augmentations = Vec::with_capacity(spec.sites.len());
    let mut site_geometry = Vec::with_capacity(spec.sites.len());

    for (site_index, site) in spec.sites.iter().enumerate() {
        if !site.radius.get().is_finite() || site.radius.get() <= 0.0 {
            return Err(BasisError::InvalidRadius(site.radius.get()));
        }
        if site.radial_solutions.is_empty() {
            return Err(BasisError::EmptySpinorRadialSet { site: site_index });
        }
        let mut solutions = site.radial_solutions.iter().collect::<Vec<_>>();
        solutions.sort_unstable_by_key(|solution| solution.kappa.get());
        let mut kappas = BTreeSet::new();
        for solution in &solutions {
            if !kappas.insert(solution.kappa) {
                return Err(BasisError::DuplicateSpinorSolution {
                    site: site_index,
                    kappa: solution.kappa.get(),
                });
            }
            if solution.role != RelativisticRole::Valence {
                return Err(BasisError::NonValenceSpinorSolution {
                    site: site_index,
                    kappa: solution.kappa.get(),
                });
            }
            for (boundary, actual) in [
                ("u", solution.boundary.radius),
                ("udot", solution.energy_derivative.boundary.radius),
            ] {
                if actual != site.radius {
                    return Err(BasisError::SpinorBoundaryRadius {
                        site: site_index,
                        kappa: solution.kappa.get(),
                        boundary,
                        expected: site.radius.get(),
                        actual: actual.get(),
                    });
                }
            }
        }
        for &(kappa, count) in site.local_orbitals.counts_by_kappa() {
            if count != 0 && !kappas.contains(&kappa) {
                return Err(BasisError::SpinorLocalOrbitalWithoutRadialSolution {
                    site: site_index,
                    kappa: kappa.get(),
                });
            }
        }

        let mut augmentations = Vec::with_capacity(plane_waves.len());
        for plane_wave in &plane_waves {
            let mut matches = Vec::with_capacity(solutions.len());
            for solution in &solutions {
                let kappa = solution.kappa;
                let apw = match_apw_boundary(
                    kappa.large_l(),
                    plane_wave.q_norm,
                    site.radius,
                    ApwBoundaryBasis {
                        u: solution.sra_boundary(),
                        udot: solution.energy_derivative.boundary.sra_large_component(),
                    },
                )?;
                matches.push(SpinorApwMatch { kappa, apw });
            }
            augmentations.push(spinor_augmentation_coefficients(
                plane_wave,
                site.position,
                spec.cell_volume,
                &matches,
            )?);
        }
        site_layouts.push(site.local_orbitals.clone());
        site_augmentations.push(augmentations);
        site_geometry.push(ApwSiteGeometry {
            position: site.position,
            radius: site.radius,
        });
    }

    Ok(SpinorCompiledBasis {
        layout: SpinorBasisLayout::new(plane_waves.len(), site_layouts),
        plane_waves,
        site_augmentations,
        site_geometry,
        provenance: spec.provenance.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlaneWave;
    use muffintin_core::Kappa;
    use muffintin_core::{DiracAngularContract, Hartree, InverseBohr, ReciprocalLattice, TwiceMu};
    use muffintin_sphere::{
        DiracBoundaryTrace, DiracEnergyDerivative, DiracSecondEnergyDerivative,
    };

    fn solution(kappa: Kappa, radius: Bohr) -> ValenceDiracSolution {
        let boundary = DiracBoundaryTrace {
            radius,
            p: radius.get(),
            q: 0.1,
            p_derivative: 0.0,
            q_derivative: 0.0,
        };
        let derivative_boundary = DiracBoundaryTrace {
            radius,
            p: 0.2 * radius.get(),
            q: 0.02,
            p_derivative: 1.0,
            q_derivative: 0.0,
        };
        ValenceDiracSolution {
            role: RelativisticRole::Valence,
            kappa,
            angular: DiracAngularContract::from(kappa),
            energy: Hartree(-0.3),
            speed_of_light: 137.0,
            p: vec![0.0; 7],
            q: vec![0.0; 7],
            boundary,
            energy_derivative: DiracEnergyDerivative {
                p: vec![0.0; 7],
                q: vec![0.0; 7],
                boundary: derivative_boundary,
                norm_squared: 0.0,
            },
            second_energy_derivative: DiracSecondEnergyDerivative {
                p: vec![0.0; 7],
                q: vec![0.0; 7],
                boundary: derivative_boundary,
                norm_squared: 0.0,
            },
            norm_total: 1.0,
        }
    }

    fn envelope() -> PlaneWaveEnvelope {
        let reciprocal = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        PlaneWaveEnvelope::new(
            reciprocal
                .enumerate(InverseBohr(0.0))
                .unwrap()
                .into_iter()
                .map(|g| PlaneWave::new([InverseBohr(0.0); 3], g))
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn compiler_canonicalizes_channels_and_keeps_positive_kappa_lo() {
        let radius = Bohr(2.0);
        let minus_one = Kappa::new(-1).unwrap();
        let plus_one = Kappa::new(1).unwrap();
        let spec = SpinorBasisSpec {
            envelope: envelope(),
            sites: vec![SpinorBasisSite {
                position: [Bohr(0.0); 3],
                radius,
                radial_solutions: vec![solution(plus_one, radius), solution(minus_one, radius)],
                local_orbitals: SpinorSiteLayout::new(vec![(plus_one, 1)]).unwrap(),
            }],
            cell_volume: VolumeBohr3(100.0),
            provenance: Provenance::default(),
        };
        let compiled = compile_spinor(&spec).unwrap();
        let channels = &compiled.site_augmentations[0][0].channels;
        assert_eq!(channels.len(), 4);
        assert_eq!(channels[0].kappa(), minus_one);
        assert_eq!(channels[0].twice_mu(), TwiceMu::new(-1).unwrap());
        assert_eq!(channels[2].kappa(), plus_one);
        assert_eq!(
            compiled.layout.site_layout(0).unwrap().counts_by_kappa(),
            &[(plus_one, 1)]
        );
        assert_eq!(
            compiled
                .layout
                .site_spinor_index(0, plus_one, TwiceMu::new(-1).unwrap(), 0),
            Some(2)
        );
    }
}
