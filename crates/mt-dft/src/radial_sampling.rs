//! Scalar radial solutions sampled directly from a regional potential.

use std::f64::consts::PI;

use muffintin_core::{Bohr, Hartree, InverseBohr};
use muffintin_sphere::{RadialEquation, RadialError, RadialSolver};
use thiserror::Error;

use crate::RegionalPotential;

/// Scalar radial solutions sampled on one regional-potential muffin-tin mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarRadialSamples {
    pub site_index: usize,
    pub angular_momentum: u32,
    pub energies: Vec<Hartree>,
    pub mesh_first: Bohr,
    pub mesh_increment: f64,
    pub mesh_count: usize,
    pub mesh_radii: Vec<Bohr>,
    /// Energy-major `u(r)` samples.
    pub radial_samples: Vec<f64>,
    /// Energy-major physical small-component `Q(r) / r` samples.
    ///
    /// Schrödinger solutions use zero rows so consumers have one fixed shape.
    pub small_radial_samples: Vec<f64>,
    pub boundary_radius: Bohr,
    /// Rows `[u(R), du/dr(R)]`.
    pub boundary_radial: Vec<[f64; 2]>,
    pub log_derivative: Vec<Option<InverseBohr>>,
    /// Rows `[du/dE(R), d(du/dr)/dE(R)]`.
    pub energy_derivative_boundary_radial: Vec<[f64; 2]>,
}

/// Failure while sampling scalar radial solutions from a regional potential.
#[derive(Debug, Error)]
pub enum ScalarRadialSamplingError {
    #[error("regional potential has {site_count} muffin-tin sites, not site index {site_index}")]
    SiteIndexOutOfBounds {
        site_index: usize,
        site_count: usize,
    },
    #[error("regional potential site {site_index} has no scalar (l=0, m=0) channel")]
    MissingScalarMonopole { site_index: usize },
    #[error(transparent)]
    Radial(#[from] RadialError),
}

/// Solve scalar radial equations in the current regional potential.
///
/// The scalar muffin-tin coefficient is converted from the normalized
/// spherical-harmonic channel `V00` to the physical spherical average
/// `V(r) = V00(r) / sqrt(4 pi)` before solving.
pub fn sample_scalar_radials(
    potential: &RegionalPotential,
    site_index: usize,
    equation: RadialEquation,
    angular_momentum: u32,
    energies: &[Hartree],
) -> Result<ScalarRadialSamples, ScalarRadialSamplingError> {
    let muffin_tins = potential.scalar().muffin_tins();
    let muffin_tin = muffin_tins.get(site_index).ok_or(
        ScalarRadialSamplingError::SiteIndexOutOfBounds {
            site_index,
            site_count: muffin_tins.len(),
        },
    )?;
    let v00 = muffin_tin
        .field()
        .channel(0, 0)
        .ok_or(ScalarRadialSamplingError::MissingScalarMonopole { site_index })?;
    let spherical_potential = v00
        .iter()
        .map(|coefficient| coefficient.re / (4.0 * PI).sqrt())
        .collect::<Vec<_>>();
    let mesh = muffin_tin.mesh();
    let solver = RadialSolver::new(mesh, &spherical_potential, equation)?;

    let mut radial_samples = Vec::with_capacity(energies.len() * mesh.len());
    let mut small_radial_samples = Vec::with_capacity(energies.len() * mesh.len());
    let mut boundary_radial = Vec::with_capacity(energies.len());
    let mut log_derivative = Vec::with_capacity(energies.len());
    let mut energy_derivative_boundary_radial = Vec::with_capacity(energies.len());
    for &energy in energies {
        let linearized = solver.solve_with_energy_derivative(angular_momentum, energy)?;
        radial_samples.extend(linearized.solution.u(mesh)?);
        if let Some(q) = &linearized.solution.q {
            small_radial_samples.extend(
                q.iter()
                    .zip(mesh.radii())
                    .map(|(&sample, radius)| sample / radius.get()),
            );
        } else {
            small_radial_samples.extend(std::iter::repeat_n(0.0, mesh.len()));
        }
        boundary_radial.push([
            linearized.solution.boundary.value,
            linearized.solution.boundary.derivative,
        ]);
        log_derivative.push(linearized.solution.boundary.log_derivative);
        energy_derivative_boundary_radial.push([
            linearized.energy_derivative.boundary.value,
            linearized.energy_derivative.boundary.derivative,
        ]);
    }

    Ok(ScalarRadialSamples {
        site_index,
        angular_momentum,
        energies: energies.to_vec(),
        mesh_first: mesh.first(),
        mesh_increment: mesh.increment(),
        mesh_count: mesh.len(),
        mesh_radii: mesh.radii().to_vec(),
        radial_samples,
        small_radial_samples,
        boundary_radius: mesh.last(),
        boundary_radial,
        log_derivative,
        energy_derivative_boundary_radial,
    })
}
