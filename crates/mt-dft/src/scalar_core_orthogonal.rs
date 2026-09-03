//! Boundary-preserving scalar core orthogonality with eliminated local orbitals.

use crate::{
    CoreShellOrbitals, ScalarBuilderError, ScalarIterationBasis, ScalarLocalOrbitalRequest,
    ScalarSiteInput, build_scalar_iteration_basis,
};
use muffintin_core::{InterstitialGeometry, Lm, MeshError};
use muffintin_envelope::PlaneWaveEnvelope;
use muffintin_sphere::{MatrixElementError, SphereFieldError, matrix_element};
use muffintin_tensor::{Axis, ComplexTensor, DenseHermitianMatrix, TensorError};
use num_complex::Complex64;
use thiserror::Error;

/// Fixed map from active coefficients to the expanded physical LAPW basis.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarCoreOrthogonalization {
    pub embedding: ComplexTensor,
    pub constraint_count: usize,
    pub maximum_radial_overlap_residual: f64,
}

#[derive(Debug, Error)]
pub enum ScalarCoreOrthogonalizationError {
    #[error("core sidecar site {site} does not have a matching scalar MT mesh")]
    SiteMesh { site: usize },
    #[error("duplicate core sidecar for site {site}")]
    DuplicateSite { site: usize },
    #[error("core cancellation system is singular at site {site}, l={l}, pivot={pivot}")]
    SingularCancellation { site: usize, l: u32, pivot: usize },
    #[error(transparent)]
    Build(#[from] ScalarBuilderError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    Field(#[from] SphereFieldError),
    #[error(transparent)]
    MatrixElement(#[from] MatrixElementError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

/// Add one matched KH LO per core radial shell, then eliminate exactly those
/// added coordinates by imposing all MT `P_c P_v + Q_c Q_v` overlaps to zero.
/// The s primitives reuse homogeneous Dirac core samples, avoiding unstable
/// outward shooting of deeply bound states. Non-s primitives remain KH solves.
/// The original plane-wave and valence-LO coordinates remain the active space.
/// Radials use `reference_sites`; the complete physical potential difference
/// from `physical_sites` is added to the local Hamiltonian, including its monopole.
pub fn build_core_orthogonal_scalar_iteration_basis(
    envelope: &PlaneWaveEnvelope,
    geometry: &InterstitialGeometry,
    reference_sites: &[ScalarSiteInput],
    physical_sites: &[ScalarSiteInput],
    cores: &[CoreShellOrbitals],
) -> Result<ScalarIterationBasis, ScalarCoreOrthogonalizationError> {
    let mut augmented = reference_sites.to_vec();
    let mut by_site = vec![None; reference_sites.len()];
    let old_counts = reference_sites
        .iter()
        .map(|site| {
            (0..site.linearization_energies.len())
                .map(|l| {
                    site.local_orbitals
                        .iter()
                        .filter(|lo| lo.angular_momentum() as usize == l)
                        .count()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for core in cores {
        let site = core.site_index;
        let Some(input) = augmented.get_mut(site) else {
            return Err(ScalarCoreOrthogonalizationError::SiteMesh { site });
        };
        if core.extended_mesh.len() < input.mesh.len()
            || core.extended_mesh.radii()[..input.mesh.len()] != input.mesh.radii()[..]
        {
            return Err(ScalarCoreOrthogonalizationError::SiteMesh { site });
        }
        if by_site[site].replace(core).is_some() {
            return Err(ScalarCoreOrthogonalizationError::DuplicateSite { site });
        }
        for shell in &core.shells {
            let l = shell.state.kappa.large_l();
            if (l as usize) < input.linearization_energies.len() {
                input.local_orbitals.push(if l == 0 {
                    ScalarLocalOrbitalRequest::BoundSCore {
                        energy: shell.energy,
                        p: shell.p[..input.mesh.len()].to_vec(),
                        q: shell.q[..input.mesh.len()].to_vec(),
                    }
                } else {
                    ScalarLocalOrbitalRequest::Lo {
                        l,
                        energy: shell.energy,
                    }
                });
            }
        }
    }
    let mut basis = build_scalar_iteration_basis(envelope, geometry, &augmented)?;
    for site in 0..reference_sites.len() {
        let delta = physical_sites[site]
            .potential
            .difference(&reference_sites[site].potential)?;
        let density = &basis.density_sites[site];
        let dimension = density.orbitals.len();
        let mut correction = vec![Complex64::default(); dimension * dimension];
        for row in 0..dimension {
            for column in row..dimension {
                correction[row * dimension + column] = matrix_element(
                    &density.mesh,
                    &density.orbitals[row],
                    &delta,
                    &density.orbitals[column],
                )?;
            }
        }
        let reference = &basis.site_blocks[site].hamiltonian;
        basis.site_blocks[site].hamiltonian = DenseHermitianMatrix::from_upper_triangle(
            dimension,
            Axis::SiteCoordinate,
            |row, column| reference.at(row, column) + correction[row * dimension + column],
        )?;
    }
    let expanded = basis.compiled.layout.dimension();
    let plane_waves = basis.compiled.layout.plane_wave_count();
    let mut original_indices = (0..plane_waves).collect::<Vec<_>>();
    for (site, counts) in old_counts.iter().enumerate() {
        for (l, &count) in counts.iter().enumerate() {
            for m in -(l as i32)..=(l as i32) {
                for local in 0..count {
                    original_indices.push(
                        basis
                            .compiled
                            .layout
                            .local_orbital_index(site, l as u32, m, local)
                            .unwrap(),
                    );
                }
            }
        }
    }
    let active = original_indices.len();
    let mut active_index = vec![None; expanded];
    let mut embedding = vec![Complex64::default(); expanded * active];
    for (column, &row) in original_indices.iter().enumerate() {
        active_index[row] = Some(column);
        embedding[row * active + column] = Complex64::new(1.0, 0.0);
    }
    let mut maximum_radial_overlap_residual = 0.0_f64;
    for (site, core) in by_site
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.map(|c| (i, c)))
    {
        let radial_site = &basis.radial_sites[site];
        let mesh = &basis.density_sites[site].mesh;
        for (l, linearized) in radial_site.linearized.iter().enumerate() {
            let shells = core
                .shells
                .iter()
                .filter(|shell| shell.state.kappa.large_l() as usize == l)
                .collect::<Vec<_>>();
            let count = shells.len();
            if count == 0 {
                continue;
            }
            let old = old_counts[site][l];
            let locals = &radial_site.local_orbitals[l];
            let mut radials = vec![
                (&linearized.solution.p, linearized.solution.q.as_deref()),
                (
                    &linearized.energy_derivative.p,
                    linearized.energy_derivative.q.as_deref(),
                ),
            ];
            radials.extend(
                locals
                    .iter()
                    .map(|local| (&local.orbital.p, local.orbital.q.as_deref())),
            );
            let radial_count = radials.len();
            let mut overlaps = Vec::with_capacity(count * radial_count);
            for shell in &shells {
                for (p, q) in &radials {
                    let integrand = (0..mesh.len())
                        .map(|r| {
                            (shell.p[r] * p[r] + shell.q[r] * q.map_or(0.0, |q| q[r]))
                                / shell.norm_total.sqrt()
                        })
                        .collect::<Vec<_>>();
                    overlaps.push(mesh.integrate(&integrand)?);
                }
            }
            let width = 2 + old;
            let mut matrix = vec![0.0; count * count];
            let mut coefficients = vec![0.0; count * width];
            for row in 0..count {
                for column in 0..count {
                    matrix[row * count + column] = overlaps[row * radial_count + width + column];
                }
                coefficients[row * width..(row + 1) * width]
                    .copy_from_slice(&overlaps[row * radial_count..row * radial_count + width]);
            }
            // Small real radial system with pivoting; no truncation of core constraints.
            for pivot in 0..count {
                let selected = (pivot..count)
                    .max_by(|&a, &b| {
                        matrix[a * count + pivot]
                            .abs()
                            .total_cmp(&matrix[b * count + pivot].abs())
                    })
                    .unwrap();
                let value = matrix[selected * count + pivot];
                if value == 0.0 || !value.is_finite() {
                    return Err(ScalarCoreOrthogonalizationError::SingularCancellation {
                        site,
                        l: l as u32,
                        pivot,
                    });
                }
                for column in 0..count {
                    matrix.swap(pivot * count + column, selected * count + column);
                }
                for column in 0..width {
                    coefficients.swap(pivot * width + column, selected * width + column);
                }
                let diagonal = matrix[pivot * count + pivot];
                for column in pivot..count {
                    matrix[pivot * count + column] /= diagonal;
                }
                for column in 0..width {
                    coefficients[pivot * width + column] /= diagonal;
                }
                for row in 0..count {
                    if row == pivot {
                        continue;
                    }
                    let factor = matrix[row * count + pivot];
                    for column in pivot..count {
                        matrix[row * count + column] -= factor * matrix[pivot * count + column];
                    }
                    for column in 0..width {
                        coefficients[row * width + column] -=
                            factor * coefficients[pivot * width + column];
                    }
                }
            }
            for row in 0..count {
                for column in 0..width {
                    let residual = overlaps[row * radial_count + column]
                        - (0..count)
                            .map(|i| {
                                overlaps[row * radial_count + width + i]
                                    * coefficients[i * width + column]
                            })
                            .sum::<f64>();
                    maximum_radial_overlap_residual =
                        maximum_radial_overlap_residual.max(residual.abs());
                }
            }
            for m in -(l as i32)..=(l as i32) {
                let lm = Lm::new(l as u32, m).unwrap().index();
                for eliminated in 0..count {
                    let row = basis
                        .compiled
                        .layout
                        .local_orbital_index(site, l as u32, m, old + eliminated)
                        .unwrap();
                    let coefficients = &coefficients[eliminated * width..(eliminated + 1) * width];
                    for pw in 0..plane_waves {
                        let augmentation =
                            basis.compiled.site_augmentations[site][pw].coefficients[lm];
                        embedding[row * active + pw] =
                            -coefficients[0] * augmentation[0] - coefficients[1] * augmentation[1];
                    }
                    for local in 0..old {
                        let global = basis
                            .compiled
                            .layout
                            .local_orbital_index(site, l as u32, m, local)
                            .unwrap();
                        embedding[row * active + active_index[global].unwrap()] =
                            Complex64::new(-coefficients[2 + local], 0.0);
                    }
                }
            }
        }
    }
    basis.core_orthogonalization = Some(ScalarCoreOrthogonalization {
        embedding: ComplexTensor::from_host_row_major(
            &[expanded, active],
            &[Axis::GlobalBasis, Axis::Reduced],
            embedding,
        )?,
        constraint_count: expanded - active,
        maximum_radial_overlap_residual,
    });
    Ok(basis)
}
