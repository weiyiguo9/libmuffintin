//! Exact static core exchange in scalar/KH site coordinates.

use std::collections::BTreeSet;

use muffintin_core::{Lm, SpinProjection, lm_count};
use muffintin_coulomb::{
    ScalarCoreExchangeRadial, StaticCoreExchangeError as RadialCoreExchangeError,
    StaticCoreExchangeMode, StaticCoreExchangeShell, static_core_exchange_block,
};
use muffintin_operators::CompiledSiteProjection;
use muffintin_tensor::{Axis, DenseHermitianMatrix, TensorError};
use num_complex::Complex64;
use thiserror::Error;

use crate::{CoreShellOccupations, CoreShellOrbitals, ScalarIterationBasis};

/// One site's exact core-exchange operator on doubled scalar coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct StaticCoreExchangeSiteBlock {
    mode: StaticCoreExchangeMode,
    scalar_coordinate_count: usize,
    matrix: DenseHermitianMatrix,
}

impl StaticCoreExchangeSiteBlock {
    pub const fn mode(&self) -> StaticCoreExchangeMode {
        self.mode
    }

    pub const fn scalar_coordinate_count(&self) -> usize {
        self.scalar_coordinate_count
    }

    pub const fn matrix(&self) -> &DenseHermitianMatrix {
        &self.matrix
    }

    /// Extract either identical spin block of a scalar-averaged operator.
    pub fn scalar_block(&self) -> Result<DenseHermitianMatrix, StaticCoreSiteExchangeError> {
        if self.mode != StaticCoreExchangeMode::ScalarAverage {
            return Err(StaticCoreSiteExchangeError::ResolvedScalarBlock);
        }
        let dimension = self.scalar_coordinate_count;
        DenseHermitianMatrix::from_upper_triangle(dimension, Axis::SiteCoordinate, |row, column| {
            self.matrix.at(row, column)
        })
        .map_err(Into::into)
    }
}

/// Invalid sidecar or scalar-basis input for static core exchange.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum StaticCoreSiteExchangeError {
    #[error("static core exchange sidecar site {site} is outside 0..{site_count}")]
    SiteIndex { site: usize, site_count: usize },
    #[error("static core exchange has duplicate sidecars for site {0}")]
    DuplicateSite(usize),
    #[error(
        "static core exchange sidecar site {site} mesh is not an exact extension of the MT mesh"
    )]
    MeshPrefix { site: usize },
    #[error("static core exchange site {site} shell {shell} uses explicit collinear occupations")]
    ExplicitCollinear { site: usize, shell: usize },
    #[error(
        "static core exchange site {site} shell {shell} has an incomplete magnetic channel set"
    )]
    MagneticChannels { site: usize, shell: usize },
    #[error("static core exchange site {site} shell {shell} is not closed")]
    OpenShell { site: usize, shell: usize },
    #[error("static core exchange site {site} has no scalar radial shells")]
    EmptyValenceRadials { site: usize },
    #[error("resolved static core exchange cannot be reduced to one scalar spin block")]
    ResolvedScalarBlock,
    #[error(transparent)]
    Radial(#[from] RadialCoreExchangeError),
    #[error(transparent)]
    Operator(#[from] muffintin_operators::OperatorError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

/// Build exact static core exchange on every scalar site-coordinate frame.
pub fn build_static_core_exchange_site_blocks(
    basis: &ScalarIterationBasis,
    sidecars: &[CoreShellOrbitals],
    mode: StaticCoreExchangeMode,
) -> Result<Vec<StaticCoreExchangeSiteBlock>, StaticCoreSiteExchangeError> {
    let site_count = basis.radial_sites.len();
    let mut by_site = vec![None; site_count];
    for sidecar in sidecars {
        if sidecar.site_index >= site_count {
            return Err(StaticCoreSiteExchangeError::SiteIndex {
                site: sidecar.site_index,
                site_count,
            });
        }
        if by_site[sidecar.site_index].replace(sidecar).is_some() {
            return Err(StaticCoreSiteExchangeError::DuplicateSite(
                sidecar.site_index,
            ));
        }
    }

    basis
        .radial_sites
        .iter()
        .zip(&basis.density_sites)
        .enumerate()
        .map(|(site, (radial_site, density_site))| {
            let sidecar = by_site[site];
            if let Some(sidecar) = sidecar
                && (sidecar.extended_mesh.len() < density_site.mesh.len()
                    || sidecar.extended_mesh.radii()[..density_site.mesh.len()]
                        != density_site.mesh.radii()[..])
            {
                return Err(StaticCoreSiteExchangeError::MeshPrefix { site });
            }
            let cores = sidecar
                .map(|sidecar| {
                    sidecar
                        .shells
                        .iter()
                        .enumerate()
                        .map(|(shell, orbital)| {
                            Ok(StaticCoreExchangeShell {
                                kappa: orbital.state.kappa,
                                p: &orbital.p,
                                q: &orbital.q,
                                normalization: orbital.norm_total,
                                occupation_per_mu: closed_occupation(site, shell, orbital)?,
                            })
                        })
                        .collect::<Result<Vec<_>, StaticCoreSiteExchangeError>>()
                })
                .transpose()?
                .unwrap_or_default();
            build_site_block(basis, site, radial_site, density_site, &cores, mode)
        })
        .collect()
}

fn closed_occupation(
    site: usize,
    shell: usize,
    orbital: &crate::CoreShellOrbital,
) -> Result<f64, StaticCoreSiteExchangeError> {
    let CoreShellOccupations::MuResolved(occupations) = &orbital.occupations else {
        return Err(StaticCoreSiteExchangeError::ExplicitCollinear { site, shell });
    };
    let expected = orbital
        .state
        .kappa
        .twice_mu_values()
        .collect::<BTreeSet<_>>();
    let found = occupations
        .iter()
        .map(|(twice_mu, _)| *twice_mu)
        .collect::<BTreeSet<_>>();
    if occupations.len() != expected.len() || found != expected {
        return Err(StaticCoreSiteExchangeError::MagneticChannels { site, shell });
    }
    let occupation = occupations[0].1;
    if occupations
        .iter()
        .any(|(_, value)| value.to_bits() != occupation.to_bits())
    {
        return Err(StaticCoreSiteExchangeError::OpenShell { site, shell });
    }
    Ok(occupation)
}

fn build_site_block(
    basis: &ScalarIterationBasis,
    site: usize,
    radial_site: &crate::ScalarRadialSite,
    density_site: &crate::ScalarSiteBasis,
    cores: &[StaticCoreExchangeShell<'_>],
    mode: StaticCoreExchangeMode,
) -> Result<StaticCoreExchangeSiteBlock, StaticCoreSiteExchangeError> {
    let Some(l_max) = radial_site.linearized.len().checked_sub(1) else {
        return Err(StaticCoreSiteExchangeError::EmptyValenceRadials { site });
    };
    let l_max = u32::try_from(l_max)
        .map_err(|_| StaticCoreSiteExchangeError::EmptyValenceRadials { site })?;
    let augmented_count = lm_count(l_max);
    let layout = basis
        .compiled
        .layout
        .site_layout(site)
        .expect("scalar radial site has a compiled local-orbital layout");
    let scalar_coordinate_count =
        CompiledSiteProjection::scalar(&basis.compiled, site)?.coordinate_count();
    let dimension = 2 * scalar_coordinate_count;
    let mut values = vec![Complex64::default(); dimension * dimension];

    for (l, (linearized, locals)) in radial_site
        .linearized
        .iter()
        .zip(&radial_site.local_orbitals)
        .enumerate()
    {
        let l = u32::try_from(l)
            .map_err(|_| StaticCoreSiteExchangeError::EmptyValenceRadials { site })?;
        let mut radials = vec![
            ScalarCoreExchangeRadial {
                p: &linearized.solution.p,
                q: linearized.solution.q.as_deref(),
            },
            ScalarCoreExchangeRadial {
                p: &linearized.energy_derivative.p,
                q: linearized.energy_derivative.q.as_deref(),
            },
        ];
        radials.extend(locals.iter().map(|local| ScalarCoreExchangeRadial {
            p: &local.orbital.p,
            q: local.orbital.q.as_deref(),
        }));
        let shell = static_core_exchange_block(&density_site.mesh, l, &radials, cores, mode)?;
        for left_spin in [SpinProjection::Up, SpinProjection::Down] {
            for left_m in -(l as i32)..=(l as i32) {
                for left_radial in 0..radials.len() {
                    let left_shell = shell
                        .coordinate(left_spin, left_m, left_radial)
                        .expect("enumerated core-exchange shell coordinate is valid");
                    let left = doubled_site_coordinate(
                        layout,
                        augmented_count,
                        scalar_coordinate_count,
                        left_spin,
                        l,
                        left_m,
                        left_radial,
                    );
                    for right_spin in [SpinProjection::Up, SpinProjection::Down] {
                        for right_m in -(l as i32)..=(l as i32) {
                            for right_radial in 0..radials.len() {
                                let right_shell = shell
                                    .coordinate(right_spin, right_m, right_radial)
                                    .expect("enumerated core-exchange shell coordinate is valid");
                                let right = doubled_site_coordinate(
                                    layout,
                                    augmented_count,
                                    scalar_coordinate_count,
                                    right_spin,
                                    l,
                                    right_m,
                                    right_radial,
                                );
                                values[left * dimension + right] +=
                                    shell.matrix.at(left_shell, right_shell);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(StaticCoreExchangeSiteBlock {
        mode,
        scalar_coordinate_count,
        matrix: DenseHermitianMatrix::from_host_row_major(dimension, Axis::SiteCoordinate, values)?,
    })
}

fn doubled_site_coordinate(
    local_orbitals: &muffintin_operators::lapw::LocalOrbitalLayout,
    augmented_count: usize,
    scalar_coordinate_count: usize,
    spin: SpinProjection,
    l: u32,
    m: i32,
    radial: usize,
) -> usize {
    let scalar = if radial < 2 {
        2 * Lm::new(l, m)
            .expect("enumerated scalar harmonic is valid")
            .index()
            + radial
    } else {
        2 * augmented_count
            + local_orbitals
                .index(l, m, radial - 2)
                .expect("scalar radial shell matches the local-orbital layout")
    };
    match spin {
        SpinProjection::Up => scalar,
        SpinProjection::Down => scalar_coordinate_count + scalar,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CoreShellOrbital, CoreShellOrbitalsProvenance, ScalarSiteInput,
        build_scalar_iteration_basis,
    };
    use muffintin_core::{
        Bohr, ExponentialMesh, GVector, Hartree, InterstitialGeometry, InverseBohr, Kappa, Sphere,
        VolumeBohr3,
    };
    use muffintin_envelope::{PlaneWave, PlaneWaveEnvelope};
    use muffintin_sphere::{CoreState, HarmonicConvention, SphereField};

    fn fixture() -> (ScalarIterationBasis, CoreShellOrbitals) {
        let mesh = ExponentialMesh::new(Bohr(1.0e-5), 0.02, 501).unwrap();
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: [Bohr(0.0); 3],
                radius: mesh.last(),
            }],
        )
        .unwrap();
        let envelope = PlaneWaveEnvelope::new([PlaneWave::new(
            [InverseBohr(0.17), InverseBohr(0.0), InverseBohr(0.0)],
            GVector {
                index: [0, 0, 0],
                cartesian: [InverseBohr(0.0); 3],
                norm: InverseBohr(0.0),
            },
        )]);
        let potential = vec![-0.2; mesh.len()];
        let basis = build_scalar_iteration_basis(
            &envelope,
            &geometry,
            &[ScalarSiteInput {
                position: [Bohr(0.0); 3],
                radius: mesh.last(),
                mesh: mesh.clone(),
                spherical_potential: potential.clone(),
                potential: SphereField::new(
                    HarmonicConvention::Complex,
                    [(
                        (0, 0),
                        vec![
                            Complex64::new(-(4.0 * std::f64::consts::PI).sqrt() * 0.2, 0.0);
                            mesh.len()
                        ],
                    )],
                )
                .unwrap(),
                linearization_energies: vec![Hartree(0.2), Hartree(0.28)],
                local_orbitals: Vec::new(),
            }],
        )
        .unwrap();
        let p = mesh
            .radii()
            .iter()
            .map(|radius| radius.get().powi(2) * (-radius.get()).exp())
            .collect::<Vec<_>>();
        let q = mesh
            .radii()
            .iter()
            .map(|radius| 0.05 * radius.get().powi(3) * (-radius.get()).exp())
            .collect::<Vec<_>>();
        let norm = mesh
            .integrate(
                &p.iter()
                    .zip(&q)
                    .map(|(p, q)| p * p + q * q)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let kappa = Kappa::new(-2).unwrap();
        let occupations = kappa
            .twice_mu_values()
            .map(|twice_mu| (twice_mu, 1.0))
            .collect();
        let core = CoreShellOrbitals {
            site_index: 0,
            site_id: "X-1".to_owned(),
            extended_mesh: mesh,
            shells: vec![CoreShellOrbital {
                state: CoreState::new(2, kappa).unwrap(),
                energy: Hartree(-1.0),
                p,
                q,
                norm_total: norm,
                norm_mt: norm,
                spill: 0.0,
                occupations: CoreShellOccupations::MuResolved(occupations),
            }],
            provenance: CoreShellOrbitalsProvenance {
                extended_potential: Vec::new(),
                solve_specs: Vec::new(),
                sourced_searches: Vec::new(),
            },
        };
        (basis, core)
    }

    #[test]
    fn scalar_and_soc_core_blocks_share_the_exact_compiled_site_frame() {
        let (basis, core) = fixture();
        let averaged = build_static_core_exchange_site_blocks(
            &basis,
            std::slice::from_ref(&core),
            StaticCoreExchangeMode::ScalarAverage,
        )
        .unwrap();
        let resolved = build_static_core_exchange_site_blocks(
            &basis,
            &[core],
            StaticCoreExchangeMode::SpinOrbitResolved,
        )
        .unwrap();
        let coordinates = CompiledSiteProjection::scalar(&basis.compiled, 0)
            .unwrap()
            .coordinate_count();
        assert_eq!(averaged[0].scalar_coordinate_count(), coordinates);
        assert_eq!(averaged[0].matrix().dimension(), 2 * coordinates);
        let scalar = averaged[0].scalar_block().unwrap();
        for row in 0..coordinates {
            for column in 0..coordinates {
                assert_eq!(
                    averaged[0].matrix().at(row, column),
                    averaged[0]
                        .matrix()
                        .at(coordinates + row, coordinates + column)
                );
                assert_eq!(
                    averaged[0].matrix().at(row, coordinates + column),
                    Complex64::new(0.0, 0.0)
                );
                assert_eq!(scalar.at(row, column), averaged[0].matrix().at(row, column));
            }
        }
        let resolved_matrix = resolved[0].matrix();
        let cross_spin = (0..coordinates)
            .flat_map(|row| {
                (0..coordinates)
                    .map(move |column| resolved_matrix.at(row, coordinates + column).norm())
            })
            .fold(0.0_f64, f64::max);
        assert!(cross_spin > 1.0e-8);
    }

    #[test]
    fn open_core_shell_is_rejected_before_operator_assembly() {
        let (basis, mut core) = fixture();
        let CoreShellOccupations::MuResolved(occupations) = &mut core.shells[0].occupations else {
            unreachable!()
        };
        occupations[0].1 = 0.5;
        assert_eq!(
            build_static_core_exchange_site_blocks(
                &basis,
                &[core],
                StaticCoreExchangeMode::SpinOrbitResolved,
            )
            .unwrap_err(),
            StaticCoreSiteExchangeError::OpenShell { site: 0, shell: 0 }
        );
    }
}
