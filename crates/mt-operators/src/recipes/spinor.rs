//! First-variation SRA spinor LAPW recipe.

use super::spinor_provenance;
use muffintin_envelope::{SpinorBasisSite, SpinorBasisSpec};
use muffintin_core::VolumeBohr3;
use muffintin_envelope::PlaneWaveEnvelope;

/// Construct the typed two-component-interstitial/four-component-site recipe.
///
/// Signed `kappa` radial solutions and local-orbital layouts pass through
/// unchanged. [`muffintin_envelope::compile_spinor`] performs canonical channel
/// ordering and boundary matching.
pub fn spinor_lapw(
    envelope: PlaneWaveEnvelope,
    cell_volume: VolumeBohr3,
    sites: &[SpinorBasisSite],
) -> SpinorBasisSpec {
    SpinorBasisSpec {
        envelope,
        sites: sites.to_vec(),
        cell_volume,
        provenance: spinor_provenance(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_envelope::{SpinorSiteLayout, compile_spinor};
    use muffintin_core::{Bohr, DiracAngularContract, GVector, Hartree, InverseBohr, Kappa};
    use muffintin_envelope::PlaneWave;
    use muffintin_sphere::{
        DiracBoundaryTrace, DiracEnergyDerivative, DiracSecondEnergyDerivative, RelativisticRole,
        ValenceDiracSolution,
    };

    fn solution(kappa: Kappa, radius: Bohr) -> ValenceDiracSolution {
        let boundary = DiracBoundaryTrace {
            radius,
            p: radius.get(),
            q: 0.1,
            p_derivative: 0.0,
            q_derivative: 0.0,
        };
        let derivative = DiracBoundaryTrace {
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
                boundary: derivative,
                norm_squared: 0.0,
            },
            second_energy_derivative: DiracSecondEnergyDerivative {
                p: vec![0.0; 7],
                q: vec![0.0; 7],
                boundary: derivative,
                norm_squared: 0.0,
            },
            norm_total: 1.0,
        }
    }

    #[test]
    fn recipe_compiles_without_manual_spinor_basis_assembly() {
        let radius = Bohr(2.0);
        let kappa = Kappa::new(1).unwrap();
        let g = GVector {
            index: [0; 3],
            cartesian: [InverseBohr(0.0); 3],
            norm: InverseBohr(0.0),
        };
        let envelope = PlaneWaveEnvelope::new(vec![PlaneWave::new([InverseBohr(0.0); 3], g)]);
        let sites = [SpinorBasisSite {
            position: [Bohr(0.0); 3],
            radius,
            radial_solutions: vec![solution(kappa, radius)],
            local_orbitals: SpinorSiteLayout::new(vec![(kappa, 1)]).unwrap(),
        }];
        let spec = spinor_lapw(envelope, VolumeBohr3(100.0), &sites);
        let compiled = compile_spinor(&spec).unwrap();
        assert_eq!(spec.provenance.recipe.as_deref(), Some("spinor-lapw"));
        assert_eq!(compiled.site_count(), 1);
        assert_eq!(compiled.site_augmentations[0][0].channels.len(), 2);
        assert_eq!(
            compiled.layout.site_layout(0).unwrap().counts_by_kappa(),
            &[(kappa, 1)]
        );
    }
}
