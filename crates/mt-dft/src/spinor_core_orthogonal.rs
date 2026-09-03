//! Frozen-core orthogonality in the kappa-resolved SRA first-variation basis.

use crate::{
    CoreShellOrbitals, LocalPauliPotential, SpinorBuilderError, SpinorFirstVariationError,
    SpinorIterationBasis, SpinorLocalOrbitalRequest, SpinorSiteInput, build_spinor_iteration_basis,
};
use muffintin_core::{InterstitialGeometry, MeshError};
use muffintin_envelope::PlaneWaveEnvelope;
use muffintin_operators::{CompiledSiteProjection, OperatorError};
use muffintin_sphere::{SphereField, SphereFieldError};
use muffintin_tensor::{Axis, ComplexTensor, TensorError};
use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

/// Map into the expanded physical basis, with one constraint per core mu.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorCoreOrthogonalization {
    pub embedding: ComplexTensor,
    pub constraint_count: usize,
    pub maximum_radial_overlap_residual: f64,
}

#[derive(Debug, Error)]
pub enum SpinorCoreOrthogonalizationError {
    #[error("core sidecar site {site} does not match the spinor MT mesh or reference potential")]
    SiteMesh { site: usize },
    #[error("duplicate core sidecar for site {site}")]
    DuplicateSite { site: usize },
    #[error("core cancellation system is singular at site {site}, kappa={kappa}, pivot={pivot}")]
    SingularCancellation {
        site: usize,
        kappa: i32,
        pivot: usize,
    },
    #[error(transparent)]
    Build(#[from] SpinorBuilderError),
    #[error(transparent)]
    FirstVariation(#[from] SpinorFirstVariationError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    Field(#[from] SphereFieldError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

/// Build SRA valence orbitals orthogonal to homogeneous frozen Dirac cores.
///
/// Stable core samples supply the cancellation primitives in the identical
/// kappa channel. The reference radial potential is taken from each sidecar;
/// the physical Hamiltonian still uses `sites`' full current potential.
/// Constraints use MT PP+QQ with the proper Omega_kappa/Omega_-kappa angular
/// orthogonality. Core spill is not projected outside the MT spheres.
/// Sourced, relaxed HF cores are not homogeneous primitives for this builder.
pub fn build_frozen_core_orthogonal_spinor_iteration_basis(
    envelope: &PlaneWaveEnvelope,
    geometry: &InterstitialGeometry,
    sites: &[SpinorSiteInput],
    cores: &[CoreShellOrbitals],
) -> Result<SpinorIterationBasis, SpinorCoreOrthogonalizationError> {
    let mut reference = sites.to_vec();
    let mut by_site = vec![None; sites.len()];
    for core in cores {
        let site = core.site_index;
        let Some(input) = reference.get_mut(site) else {
            return Err(SpinorCoreOrthogonalizationError::SiteMesh { site });
        };
        let n = input.mesh.len();
        if core.extended_mesh.len() < n
            || core.extended_mesh.radii()[..n] != input.mesh.radii()[..]
            || core.provenance.extended_potential.len() < n
        {
            return Err(SpinorCoreOrthogonalizationError::SiteMesh { site });
        }
        if by_site[site].replace(core).is_some() {
            return Err(SpinorCoreOrthogonalizationError::DuplicateSite { site });
        }
        input.spherical_potential = core.provenance.extended_potential[..n]
            .iter()
            .map(|v| v.get())
            .collect();
        let scalar = SphereField::new(
            input.potential.scalar().convention(),
            [(
                (0, 0),
                input
                    .spherical_potential
                    .iter()
                    .map(|v| Complex64::new((4.0 * PI).sqrt() * v, 0.0))
                    .collect(),
            )],
        )?;
        let zero = SphereField::new(
            input.potential.scalar().convention(),
            [((0, 0), vec![Complex64::default(); n])],
        )?;
        input.potential = LocalPauliPotential::new(scalar, [zero.clone(), zero.clone(), zero])?;
        for shell in &core.shells {
            if shell.state.kappa.large_l() <= input.l_max {
                input
                    .local_orbitals
                    .push(SpinorLocalOrbitalRequest::BoundCore {
                        kappa: shell.state.kappa,
                        energy: shell.energy,
                        p: shell.p[..n].to_vec(),
                        q: shell.q[..n].to_vec(),
                    });
            }
        }
    }
    let mut basis = build_spinor_iteration_basis(envelope, geometry, &reference)?;
    for (full, physical) in basis.full_spinor_sites.iter_mut().zip(sites) {
        full.potential = physical.potential.clone();
    }
    let layout = &basis.compiled.layout;
    let expanded = layout.dimension();
    let mut originals = layout.plane_wave_range().collect::<Vec<_>>();
    for (site, input) in sites.iter().enumerate() {
        for radial in &basis.radial_sites[site].solutions {
            let old = input
                .local_orbitals
                .iter()
                .filter(|r| r.kappa() == radial.kappa)
                .count();
            for mu in radial.kappa.twice_mu_values() {
                for local in 0..old {
                    originals.push(
                        layout
                            .site_spinor_index(site, radial.kappa, mu, local)
                            .unwrap(),
                    );
                }
            }
        }
    }
    let active = originals.len();
    let mut active_index = vec![None; expanded];
    let mut embedding = vec![Complex64::default(); expanded * active];
    for (column, &row) in originals.iter().enumerate() {
        active_index[row] = Some(column);
        embedding[row * active + column] = Complex64::new(1.0, 0.0);
    }
    let mut maximum_radial_overlap_residual = 0.0_f64;
    for (site, core) in by_site
        .iter()
        .enumerate()
        .filter_map(|(site, c)| c.map(|c| (site, c)))
    {
        let radial_site = &basis.radial_sites[site];
        let full = &basis.full_spinor_sites[site];
        let projection = CompiledSiteProjection::spinor(&basis.compiled, site, &full.channels)?;
        let projected = projection.matrix().to_host_row_major();
        let mesh = &full.mesh;
        for (radial, locals) in radial_site
            .solutions
            .iter()
            .zip(&radial_site.local_orbitals)
        {
            let kappa = radial.kappa;
            let shells = core
                .shells
                .iter()
                .filter(|s| s.state.kappa == kappa)
                .collect::<Vec<_>>();
            let count = shells.len();
            if count == 0 {
                continue;
            }
            let old = sites[site]
                .local_orbitals
                .iter()
                .filter(|r| r.kappa() == kappa)
                .count();
            let mut functions = vec![
                (&radial.p, &radial.q),
                (&radial.energy_derivative.p, &radial.energy_derivative.q),
            ];
            functions.extend(locals.iter().map(|lo| (&lo.orbital.p, &lo.orbital.q)));
            let radial_count = functions.len();
            let mut overlaps = Vec::with_capacity(count * radial_count);
            for shell in &shells {
                for (p, q) in &functions {
                    let values = (0..mesh.len())
                        .map(|r| (shell.p[r] * p[r] + shell.q[r] * q[r]) / shell.norm_total.sqrt())
                        .collect::<Vec<_>>();
                    overlaps.push(mesh.integrate(&values)?);
                }
            }
            let width = 2 + old;
            let mut matrix = vec![0.0; count * count];
            let mut coefficients = vec![0.0; count * width];
            for row in 0..count {
                matrix[row * count..(row + 1) * count].copy_from_slice(
                    &overlaps[row * radial_count + width..(row + 1) * radial_count],
                );
                coefficients[row * width..(row + 1) * width]
                    .copy_from_slice(&overlaps[row * radial_count..row * radial_count + width]);
            }
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
                    return Err(SpinorCoreOrthogonalizationError::SingularCancellation {
                        site,
                        kappa: kappa.get(),
                        pivot,
                    });
                }
                for column in 0..count {
                    matrix.swap(pivot * count + column, selected * count + column);
                }
                for column in 0..width {
                    coefficients.swap(pivot * width + column, selected * width + column);
                }
                for column in pivot..count {
                    matrix[pivot * count + column] /= value;
                }
                for column in 0..width {
                    coefficients[pivot * width + column] /= value;
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
            for channel in kappa.channels() {
                let coordinates = full
                    .orbitals
                    .iter()
                    .enumerate()
                    .filter_map(|(i, o)| (o.channel() == channel).then_some(i))
                    .collect::<Vec<_>>();
                for eliminated in 0..count {
                    let row = layout
                        .site_spinor_index(site, kappa, channel.twice_mu(), old + eliminated)
                        .unwrap();
                    for (column, &global) in projection.global_indices().iter().enumerate() {
                        let Some(active_column) = active_index[global] else {
                            continue;
                        };
                        embedding[row * active + active_column] = -(0..width)
                            .map(|r| {
                                coefficients[eliminated * width + r]
                                    * projected[coordinates[r] * projection.basis_count() + column]
                            })
                            .sum::<Complex64>();
                    }
                }
            }
        }
    }
    basis.core_orthogonalization = Some(SpinorCoreOrthogonalization {
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
