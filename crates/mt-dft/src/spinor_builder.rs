//! Production radial and basis builder for the full first-variation spinor route.

use crate::{
    FullSpinorDensitySiteBasis, FullSpinorSiteInput, LocalPauliPotential, RelativisticSpinorRoute,
    SolvedFullSpinorFirstVariation, SpinorFirstVariationError, solve_full_spinor_first_variation,
};
use muffintin_core::{Bohr, ExponentialMesh, Hartree, InterstitialGeometry, Kappa, KappaError};
use muffintin_envelope::{
    BasisError, Provenance, SpinorBasisSite, SpinorBasisSpec, SpinorCompiledBasis,
    SpinorSiteLayout, compile_spinor,
};
use muffintin_operators::lapw::{InterstitialPauliPotential, PlaneWaveEnvelope};
use muffintin_sphere::{
    DiracError, DiracLocalOrbital, DiracSecondEnergyDerivative, RadialComponents,
    RadialIntegralError, RadialIntegralKernel, ValenceDiracSolution, ValenceDiracSpec,
    radial_integral, solve_valence_dirac,
};
use muffintin_sphere::{SphereFieldError, SphereOrbitalError, SpinorSphereOrbital};
use muffintin_tensor::{Axis, DenseHermitianMatrix, TensorError};
use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

const POTENTIAL_TOLERANCE: f64 = 4096.0 * f64::EPSILON;

/// One explicitly signed spinor local-orbital request.
///
/// In particular, `kappa=+1` is the relativistic `p1/2` channel and remains
/// distinct from `kappa=-2` (`p3/2`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpinorLocalOrbitalRequest {
    Lo { kappa: Kappa, energy: Hartree },
    Hdlo { kappa: Kappa },
}

impl SpinorLocalOrbitalRequest {
    pub const fn kappa(self) -> Kappa {
        match self {
            Self::Lo { kappa, .. } | Self::Hdlo { kappa } => kappa,
        }
    }
}

/// Base linearization energy for one explicitly signed Dirac channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinorLinearizationEnergy {
    pub kappa: Kappa,
    pub energy: Hartree,
}

/// Complete physical input for one full-spinor muffin-tin site.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorSiteInput {
    pub position: [Bohr; 3],
    pub radius: Bohr,
    pub mesh: ExponentialMesh,
    /// Central scalar potential used only to generate the Dirac radial set.
    pub spherical_potential: Vec<f64>,
    /// Full local `V0 I + B . sigma` potential used by first variation.
    pub potential: LocalPauliPotential,
    /// Largest orbital angular momentum represented by the base radial set.
    pub l_max: u32,
    /// One base energy for every signed `kappa` through `l_max`.
    pub linearization_energies: Vec<SpinorLinearizationEnergy>,
    pub local_orbitals: Vec<SpinorLocalOrbitalRequest>,
}

/// Exact primitive from which a confined spinor local orbital was built.
#[derive(Clone, Debug, PartialEq)]
pub enum SpinorLocalOrbitalOrigin {
    DistinctEnergy(Box<ValenceDiracSolution>),
    Hdlo(DiracSecondEnergyDerivative),
}

/// Matched full-`P/Q` local orbital and its untransformed radial origin.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltSpinorLocalOrbital {
    pub request: SpinorLocalOrbitalRequest,
    pub orbital: DiracLocalOrbital,
    pub origin: SpinorLocalOrbitalOrigin,
}

/// Signed-`kappa` radial solutions retained for one site and iteration.
///
/// `solutions` is in increasing signed-`kappa` order. The corresponding
/// entry of `local_orbitals` contains requests for exactly that `kappa`, in
/// request order.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorRadialSite {
    pub solutions: Vec<ValenceDiracSolution>,
    pub local_orbitals: Vec<Vec<BuiltSpinorLocalOrbital>>,
}

/// Compiled SRA envelope plus mandatory four-component muffin-tin data.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorIterationBasis {
    pub basis_spec: SpinorBasisSpec,
    pub compiled: SpinorCompiledBasis,
    pub radial_sites: Vec<SpinorRadialSite>,
    pub density_sites: Vec<FullSpinorDensitySiteBasis>,
    pub full_spinor_sites: Vec<FullSpinorSiteInput>,
}

/// Build signed-`kappa` Dirac radials, typed LOs, the compiled SRA spinor
/// basis, and the reference muffin-tin Hamiltonians for one iteration.
pub fn build_spinor_iteration_basis(
    envelope: &PlaneWaveEnvelope,
    geometry: &InterstitialGeometry,
    sites: &[SpinorSiteInput],
) -> Result<SpinorIterationBasis, SpinorBuilderError> {
    if sites.len() != geometry.spheres().len() {
        return Err(SpinorBuilderError::SiteCount {
            expected: geometry.spheres().len(),
            actual: sites.len(),
        });
    }
    let built = sites
        .iter()
        .enumerate()
        .map(|(site, input)| build_site(site, input))
        .collect::<Result<Vec<_>, _>>()?;
    let basis_spec = SpinorBasisSpec {
        envelope: envelope.clone(),
        sites: built.iter().map(|site| site.basis.clone()).collect(),
        cell_volume: geometry.cell_volume(),
        provenance: Provenance {
            recipe: Some("spinor-lapw".to_owned()),
            reference: Some("SRA first variation: 2c interstitial and 4c muffin tins".to_owned()),
        },
    };
    let compiled = compile_spinor(&basis_spec)?;
    for (site, (compiled_site, sphere)) in compiled
        .site_geometry
        .iter()
        .zip(geometry.spheres())
        .enumerate()
    {
        if compiled_site.position != sphere.center || compiled_site.radius != sphere.radius {
            return Err(SpinorBuilderError::GeometryMismatch { site });
        }
    }
    let density_sites = built
        .iter()
        .map(|site| FullSpinorDensitySiteBasis {
            mesh: site.full.mesh.clone(),
            channels: site.full.channels.clone(),
            orbitals: site.full.orbitals.clone(),
        })
        .collect();
    Ok(SpinorIterationBasis {
        basis_spec,
        compiled,
        radial_sites: built.iter().map(|site| site.radials.clone()).collect(),
        density_sites,
        full_spinor_sites: built.into_iter().map(|site| site.full).collect(),
    })
}

/// Assemble and solve the production full-spinor route at one k point.
pub fn solve_spinor_k_point(
    basis: &SpinorIterationBasis,
    geometry: &InterstitialGeometry,
    interstitial: &InterstitialPauliPotential,
    relative_overlap_threshold: f64,
) -> Result<SolvedFullSpinorFirstVariation, SpinorBuilderError> {
    solve_full_spinor_first_variation(
        RelativisticSpinorRoute::FullFourComponentFirstVariation,
        &basis.compiled,
        geometry,
        interstitial,
        &basis.full_spinor_sites,
        relative_overlap_threshold,
    )
    .map_err(Into::into)
}

#[derive(Clone)]
struct BuiltSite {
    basis: SpinorBasisSite,
    radials: SpinorRadialSite,
    full: FullSpinorSiteInput,
}

fn build_site(site: usize, input: &SpinorSiteInput) -> Result<BuiltSite, SpinorBuilderError> {
    if input.mesh.last() != input.radius {
        return Err(SpinorBuilderError::MeshRadius {
            site,
            mesh: input.mesh.last(),
            radius: input.radius,
        });
    }
    if input.spherical_potential.len() != input.mesh.len() {
        return Err(SpinorBuilderError::SphericalPotentialLength {
            site,
            expected: input.mesh.len(),
            actual: input.spherical_potential.len(),
        });
    }
    validate_full_potential(site, input)?;

    let mut linearization_energies = input.linearization_energies.clone();
    linearization_energies.sort_by_key(|parameter| parameter.kappa.get());
    for parameter in &linearization_energies {
        if !parameter.energy.get().is_finite() {
            return Err(SpinorBuilderError::NonFiniteBaseLinearizationEnergy {
                site,
                kappa: parameter.kappa.get(),
                energy: parameter.energy,
            });
        }
        if parameter.kappa.large_l() > input.l_max {
            return Err(SpinorBuilderError::BaseLinearizationKappaOutOfRange {
                site,
                kappa: parameter.kappa.get(),
                l_max: input.l_max,
            });
        }
    }
    if let Some(duplicate) = linearization_energies
        .windows(2)
        .find(|pair| pair[0].kappa == pair[1].kappa)
    {
        return Err(SpinorBuilderError::DuplicateBaseLinearizationEnergy {
            site,
            kappa: duplicate[0].kappa.get(),
        });
    }
    for l in 0..=input.l_max {
        for kappa in kappas_for_l(l)? {
            if linearization_energies
                .binary_search_by_key(&kappa.get(), |parameter| parameter.kappa.get())
                .is_err()
            {
                return Err(SpinorBuilderError::MissingBaseLinearizationEnergy {
                    site,
                    kappa: kappa.get(),
                    l,
                });
            }
        }
    }

    let solutions = linearization_energies
        .iter()
        .map(|parameter| {
            solve_valence_dirac(
                &input.mesh,
                &input.spherical_potential,
                ValenceDiracSpec::new(parameter.kappa, parameter.energy)?,
            )
        })
        .collect::<Result<Vec<_>, DiracError>>()?;

    let mut local_orbitals = vec![Vec::new(); solutions.len()];
    for &request in &input.local_orbitals {
        let kappa = request.kappa();
        let shell = solutions
            .binary_search_by_key(&kappa.get(), |solution| solution.kappa.get())
            .map_err(|_| SpinorBuilderError::LocalOrbitalKappa {
                site,
                kappa: kappa.get(),
                l_max: input.l_max,
            })?;
        let base = &solutions[shell];
        let built = match request {
            SpinorLocalOrbitalRequest::Lo { energy, .. } => {
                if energy == base.energy {
                    return Err(SpinorBuilderError::LocalOrbitalEnergyNotDistinct {
                        site,
                        kappa: kappa.get(),
                        energy,
                    });
                }
                let raw = solve_valence_dirac(
                    &input.mesh,
                    &input.spherical_potential,
                    ValenceDiracSpec::new(kappa, energy)?,
                )?;
                BuiltSpinorLocalOrbital {
                    request,
                    orbital: base.sra_local_orbital(&raw, &input.mesh)?,
                    origin: SpinorLocalOrbitalOrigin::DistinctEnergy(Box::new(raw)),
                }
            }
            SpinorLocalOrbitalRequest::Hdlo { .. } => BuiltSpinorLocalOrbital {
                request,
                orbital: base.sra_hdlo(&input.mesh)?,
                origin: SpinorLocalOrbitalOrigin::Hdlo(base.second_energy_derivative.clone()),
            },
        };
        local_orbitals[shell].push(built);
    }
    let radials = SpinorRadialSite {
        solutions,
        local_orbitals,
    };
    let layout = SpinorSiteLayout::new(
        radials
            .solutions
            .iter()
            .zip(&radials.local_orbitals)
            .map(|(solution, locals)| (solution.kappa, locals.len()))
            .collect(),
    )?;
    let basis = SpinorBasisSite {
        position: input.position,
        radius: input.radius,
        radial_solutions: radials.solutions.clone(),
        local_orbitals: layout,
    };
    let channels = radials
        .solutions
        .iter()
        .flat_map(|solution| solution.kappa.channels())
        .collect::<Vec<_>>();
    let orbitals = coordinate_orbitals(&radials)?;
    let reference_hamiltonian = reference_hamiltonian(input, &radials, orbitals.len())?;
    let full = FullSpinorSiteInput {
        mesh: input.mesh.clone(),
        channels,
        orbitals,
        reference_hamiltonian,
        potential: input.potential.clone(),
    };
    Ok(BuiltSite {
        basis,
        radials,
        full,
    })
}

fn kappas_for_l(l: u32) -> Result<Vec<Kappa>, SpinorBuilderError> {
    let l = i32::try_from(l).map_err(|_| SpinorBuilderError::AngularMomentumOverflow)?;
    let upper = l
        .checked_add(1)
        .and_then(i32::checked_neg)
        .ok_or(SpinorBuilderError::AngularMomentumOverflow)?;
    let mut kappas = vec![Kappa::new(upper)?];
    if l != 0 {
        kappas.push(Kappa::new(l)?);
    }
    Ok(kappas)
}

fn validate_full_potential(site: usize, input: &SpinorSiteInput) -> Result<(), SpinorBuilderError> {
    let scalar = input.potential.scalar();
    if scalar.sample_count() != Some(input.mesh.len()) {
        return Err(SpinorBuilderError::FullPotentialLength {
            site,
            expected: input.mesh.len(),
            actual: scalar.sample_count(),
        });
    }
    let Some(monopole) = scalar.channel(0, 0) else {
        return Err(SpinorBuilderError::MissingMonopole { site });
    };
    let normalization = (4.0 * PI).sqrt();
    for (index, (&coefficient, &physical)) in
        monopole.iter().zip(&input.spherical_potential).enumerate()
    {
        let expected = Complex64::new(normalization * physical, 0.0);
        if (coefficient - expected).norm() > POTENTIAL_TOLERANCE * (1.0 + expected.norm()) {
            return Err(SpinorBuilderError::SphericalPotentialMismatch {
                site,
                index,
                expected,
                actual: coefficient,
            });
        }
    }
    Ok(())
}

fn coordinate_orbitals(
    site: &SpinorRadialSite,
) -> Result<Vec<SpinorSphereOrbital>, SpinorBuilderError> {
    let mut orbitals = Vec::new();
    for solution in &site.solutions {
        for channel in solution.kappa.channels() {
            orbitals.push(SpinorSphereOrbital::new(
                channel,
                solution.p.clone(),
                solution.q.clone(),
            )?);
            orbitals.push(SpinorSphereOrbital::new(
                channel,
                solution.energy_derivative.p.clone(),
                solution.energy_derivative.q.clone(),
            )?);
        }
    }
    for (solution, locals) in site.solutions.iter().zip(&site.local_orbitals) {
        for channel in solution.kappa.channels() {
            for local in locals {
                orbitals.push(SpinorSphereOrbital::new(
                    channel,
                    local.orbital.p.clone(),
                    local.orbital.q.clone(),
                )?);
            }
        }
    }
    Ok(orbitals)
}

fn reference_hamiltonian(
    input: &SpinorSiteInput,
    site: &SpinorRadialSite,
    dimension: usize,
) -> Result<DenseHermitianMatrix, SpinorBuilderError> {
    let mut reference = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    let augmented_count = site
        .solutions
        .iter()
        .map(|solution| 2 * solution.kappa.degeneracy() as usize)
        .sum::<usize>();
    for (shell_index, (solution, locals)) in
        site.solutions.iter().zip(&site.local_orbitals).enumerate()
    {
        let shell =
            radial_shell_matrices(&input.mesh, &input.spherical_potential, solution, locals)?;
        for mu in 0..solution.kappa.degeneracy() as usize {
            for left in 0..shell.dimension {
                let left_coordinate =
                    spinor_coordinate(site, augmented_count, shell_index, mu, left);
                for right in 0..shell.dimension {
                    let right_coordinate =
                        spinor_coordinate(site, augmented_count, shell_index, mu, right);
                    reference[left_coordinate * dimension + right_coordinate] = Complex64::new(
                        shell.central_hamiltonian[left * shell.dimension + right]
                            - shell.spherical_potential[left * shell.dimension + right],
                        0.0,
                    );
                }
            }
        }
    }
    DenseHermitianMatrix::from_host_row_major(dimension, Axis::SiteCoordinate, reference)
        .map_err(Into::into)
}

struct RadialShellMatrices {
    dimension: usize,
    central_hamiltonian: Vec<f64>,
    spherical_potential: Vec<f64>,
}

struct PrimitiveDiracRadial {
    energy: Hartree,
    derivative_order: u8,
    lower_derivative: Option<usize>,
    p: Vec<f64>,
    q: Vec<f64>,
}

impl RadialComponents for PrimitiveDiracRadial {
    fn large_component(&self) -> &[f64] {
        &self.p
    }

    fn small_component(&self) -> Option<&[f64]> {
        Some(&self.q)
    }
}

fn radial_shell_matrices(
    mesh: &ExponentialMesh,
    spherical_potential: &[f64],
    solution: &ValenceDiracSolution,
    locals: &[BuiltSpinorLocalOrbital],
) -> Result<RadialShellMatrices, SpinorBuilderError> {
    let dimension = 2 + locals.len();
    let energy = solution.energy;
    let mut primitives = vec![
        PrimitiveDiracRadial {
            energy,
            derivative_order: 0,
            lower_derivative: None,
            p: solution.p.clone(),
            q: solution.q.clone(),
        },
        PrimitiveDiracRadial {
            energy,
            derivative_order: 1,
            lower_derivative: Some(0),
            p: solution.energy_derivative.p.clone(),
            q: solution.energy_derivative.q.clone(),
        },
    ];
    for local in locals {
        primitives.push(match &local.origin {
            SpinorLocalOrbitalOrigin::DistinctEnergy(raw) => PrimitiveDiracRadial {
                energy: raw.energy,
                derivative_order: 0,
                lower_derivative: None,
                p: raw.p.clone(),
                q: raw.q.clone(),
            },
            SpinorLocalOrbitalOrigin::Hdlo(second) => PrimitiveDiracRadial {
                energy,
                derivative_order: 2,
                lower_derivative: Some(1),
                p: second.p.clone(),
                q: second.q.clone(),
            },
        });
    }
    let mut overlap = vec![0.0; dimension * dimension];
    let mut potential = vec![0.0; dimension * dimension];
    for left in 0..dimension {
        for right in left..dimension {
            let s = radial_integral(
                mesh,
                &primitives[left],
                &primitives[right],
                RadialIntegralKernel::Overlap,
            )?;
            let v = radial_integral(
                mesh,
                &primitives[left],
                &primitives[right],
                RadialIntegralKernel::Samples(spherical_potential),
            )?;
            overlap[left * dimension + right] = s;
            overlap[right * dimension + left] = s;
            potential[left * dimension + right] = v;
            potential[right * dimension + left] = v;
        }
    }
    let mut central_hamiltonian = vec![0.0; dimension * dimension];
    for left in 0..dimension {
        for right in left..dimension {
            let mut numerator = (primitives[left].energy.get() + primitives[right].energy.get())
                * overlap[left * dimension + right];
            if let Some(lower) = primitives[right].lower_derivative {
                numerator += f64::from(primitives[right].derivative_order)
                    * overlap[left * dimension + lower];
            }
            if let Some(lower) = primitives[left].lower_derivative {
                numerator += f64::from(primitives[left].derivative_order)
                    * overlap[lower * dimension + right];
            }
            let value = 0.5 * numerator;
            central_hamiltonian[left * dimension + right] = value;
            central_hamiltonian[right * dimension + left] = value;
        }
    }
    let mut transform = vec![0.0; dimension * dimension];
    transform[0] = 1.0;
    transform[dimension + 1] = 1.0;
    for (index, local) in locals.iter().enumerate() {
        let coordinate = 2 + index;
        let coefficients = local.orbital.coefficients;
        let scale = coefficients.normalization_scale;
        transform[coordinate] = scale * coefficients.a;
        transform[dimension + coordinate] = scale * coefficients.b;
        transform[coordinate * dimension + coordinate] = scale;
    }
    Ok(RadialShellMatrices {
        dimension,
        central_hamiltonian: transform_symmetric(dimension, &central_hamiltonian, &transform),
        spherical_potential: transform_symmetric(dimension, &potential, &transform),
    })
}

fn transform_symmetric(dimension: usize, matrix: &[f64], transform: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; dimension * dimension];
    for left in 0..dimension {
        for right in left..dimension {
            let mut value = 0.0;
            for p in 0..dimension {
                for q in 0..dimension {
                    value += transform[p * dimension + left]
                        * matrix[p * dimension + q]
                        * transform[q * dimension + right];
                }
            }
            result[left * dimension + right] = value;
            result[right * dimension + left] = value;
        }
    }
    result
}

fn spinor_coordinate(
    site: &SpinorRadialSite,
    augmented_count: usize,
    shell: usize,
    mu: usize,
    radial: usize,
) -> usize {
    if radial < 2 {
        let preceding_channels = site.solutions[..shell]
            .iter()
            .map(|solution| solution.kappa.degeneracy() as usize)
            .sum::<usize>();
        2 * (preceding_channels + mu) + radial
    } else {
        let preceding_locals = site.solutions[..shell]
            .iter()
            .zip(&site.local_orbitals[..shell])
            .map(|(solution, locals)| solution.kappa.degeneracy() as usize * locals.len())
            .sum::<usize>();
        let count = site.local_orbitals[shell].len();
        augmented_count + preceding_locals + mu * count + radial - 2
    }
}

/// Invalid full-spinor radial, basis, site-operator, or eigensolve input.
#[derive(Debug, Error)]
pub enum SpinorBuilderError {
    #[error("received {actual} spinor sites, expected {expected}")]
    SiteCount { expected: usize, actual: usize },
    #[error("site {site} radial mesh ends at {mesh}, but its radius is {radius}")]
    MeshRadius {
        site: usize,
        mesh: Bohr,
        radius: Bohr,
    },
    #[error("site {site} spherical potential has {actual} samples, expected {expected}")]
    SphericalPotentialLength {
        site: usize,
        expected: usize,
        actual: usize,
    },
    #[error("site {site} full scalar potential has sample count {actual:?}, expected {expected}")]
    FullPotentialLength {
        site: usize,
        expected: usize,
        actual: Option<usize>,
    },
    #[error("site {site} has no scalar-potential spherical-harmonic monopole")]
    MissingMonopole { site: usize },
    #[error(
        "site {site} spherical potential mismatch at {index}: expected {expected}, found {actual}"
    )]
    SphericalPotentialMismatch {
        site: usize,
        index: usize,
        expected: Complex64,
        actual: Complex64,
    },
    #[error("site {site} has duplicate base linearization energy for kappa={kappa}")]
    DuplicateBaseLinearizationEnergy { site: usize, kappa: i32 },
    #[error("site {site} base energy for kappa={kappa} is not finite: {energy}")]
    NonFiniteBaseLinearizationEnergy {
        site: usize,
        kappa: i32,
        energy: Hartree,
    },
    #[error("site {site} is missing the l={l}, kappa={kappa} base linearization energy")]
    MissingBaseLinearizationEnergy { site: usize, l: u32, kappa: i32 },
    #[error("site {site} base kappa={kappa} exceeds l_max={l_max}")]
    BaseLinearizationKappaOutOfRange { site: usize, kappa: i32, l_max: u32 },
    #[error("angular momentum does not fit the signed-kappa representation")]
    AngularMomentumOverflow,
    #[error("site {site} local orbital kappa={kappa} is absent through l_max={l_max}")]
    LocalOrbitalKappa { site: usize, kappa: i32, l_max: u32 },
    #[error("site {site} kappa={kappa} local-orbital energy {energy} is not distinct")]
    LocalOrbitalEnergyNotDistinct {
        site: usize,
        kappa: i32,
        energy: Hartree,
    },
    #[error("compiled spinor site {site} does not match the interstitial geometry")]
    GeometryMismatch { site: usize },
    #[error(transparent)]
    Kappa(#[from] KappaError),
    #[error(transparent)]
    Dirac(#[from] DiracError),
    #[error(transparent)]
    RadialIntegral(#[from] RadialIntegralError),
    #[error(transparent)]
    Basis(#[from] BasisError),
    #[error(transparent)]
    SphereField(#[from] SphereFieldError),
    #[error(transparent)]
    SphereOrbital(#[from] SphereOrbitalError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    FirstVariation(#[from] SpinorFirstVariationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_full_spinor_site_blocks;
    use muffintin_core::{GVector, InverseBohr, Sphere, VolumeBohr3};
    use muffintin_operators::lapw::PlaneWave;
    use muffintin_sphere::{HarmonicConvention, SphereField};

    fn mesh() -> ExponentialMesh {
        ExponentialMesh::new(Bohr(1.0e-5), 0.015, 801).unwrap()
    }

    fn envelope() -> PlaneWaveEnvelope {
        PlaneWaveEnvelope::new([PlaneWave::new(
            [InverseBohr(0.17), InverseBohr(0.0), InverseBohr(0.0)],
            GVector {
                index: [0, 0, 0],
                cartesian: [InverseBohr(0.0); 3],
                norm: InverseBohr(0.0),
            },
        )])
    }

    fn geometry(mesh: &ExponentialMesh) -> InterstitialGeometry {
        InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: [Bohr(0.0); 3],
                radius: mesh.last(),
            }],
        )
        .unwrap()
    }

    fn field(mesh: &ExponentialMesh, value: f64) -> SphereField {
        SphereField::new(
            HarmonicConvention::Complex,
            [(
                (0, 0),
                vec![Complex64::new((4.0 * PI).sqrt() * value, 0.0); mesh.len()],
            )],
        )
        .unwrap()
    }

    fn input(mesh: &ExponentialMesh) -> SpinorSiteInput {
        let kappa = Kappa::new(1).unwrap();
        SpinorSiteInput {
            position: [Bohr(0.0); 3],
            radius: mesh.last(),
            mesh: mesh.clone(),
            spherical_potential: vec![-0.2; mesh.len()],
            potential: LocalPauliPotential::new(
                field(mesh, -0.2),
                [field(mesh, 0.0), field(mesh, 0.0), field(mesh, 0.0)],
            )
            .unwrap(),
            l_max: 1,
            linearization_energies: vec![
                SpinorLinearizationEnergy {
                    kappa: Kappa::new(1).unwrap(),
                    energy: Hartree(0.34),
                },
                SpinorLinearizationEnergy {
                    kappa: Kappa::new(-1).unwrap(),
                    energy: Hartree(0.2),
                },
                SpinorLinearizationEnergy {
                    kappa: Kappa::new(-2).unwrap(),
                    energy: Hartree(0.28),
                },
            ],
            local_orbitals: vec![
                SpinorLocalOrbitalRequest::Lo {
                    kappa,
                    energy: Hartree(0.7),
                },
                SpinorLocalOrbitalRequest::Hdlo { kappa },
            ],
        }
    }

    #[test]
    fn signed_kappa_energies_remain_distinct_and_keep_p_half_los() {
        let mesh = mesh();
        let built =
            build_spinor_iteration_basis(&envelope(), &geometry(&mesh), &[input(&mesh)]).unwrap();
        let radial = &built.radial_sites[0];
        assert_eq!(
            radial
                .solutions
                .iter()
                .map(|solution| (solution.kappa.get(), solution.energy))
                .collect::<Vec<_>>(),
            vec![(-2, Hartree(0.28)), (-1, Hartree(0.2)), (1, Hartree(0.34))]
        );
        assert_eq!(radial.local_orbitals[2].len(), 2);
        assert!(radial.local_orbitals[2].iter().all(|local| {
            local.orbital.kappa.get() == 1
                && local.orbital.q.iter().any(|value| value.abs() > 0.0)
                && local.orbital.boundary.value.abs() < 1.0e-10
                && local.orbital.boundary.derivative.abs() < 1.0e-10
        }));
        let expected_layout = SpinorSiteLayout::new(vec![
            (Kappa::new(-2).unwrap(), 0),
            (Kappa::new(-1).unwrap(), 0),
            (Kappa::new(1).unwrap(), 2),
        ])
        .unwrap();
        assert_eq!(built.compiled.layout.site_layout(0), Some(&expected_layout));
        let site = &built.full_spinor_sites[0];
        let density_site = &built.density_sites[0];
        assert_eq!(site.channels.len(), 8);
        assert_eq!(site.orbitals.len(), 20);
        assert_eq!(site.reference_hamiltonian.dimension(), 20);
        assert_eq!(density_site.mesh, site.mesh);
        assert_eq!(density_site.channels, site.channels);
        assert_eq!(density_site.orbitals, site.orbitals);
        assert!(site.orbitals.iter().all(|orbital| {
            orbital.p().len() == mesh.len()
                && orbital.q().len() == mesh.len()
                && orbital.q().iter().any(|value| value.abs() > 0.0)
        }));
    }

    #[test]
    fn duplicate_base_kappa_is_a_typed_error() {
        let mesh = mesh();
        let mut input = input(&mesh);
        input
            .linearization_energies
            .push(SpinorLinearizationEnergy {
                kappa: Kappa::new(-2).unwrap(),
                energy: Hartree(0.31),
            });
        let error =
            build_spinor_iteration_basis(&envelope(), &geometry(&mesh), &[input]).unwrap_err();
        assert!(matches!(
            error,
            SpinorBuilderError::DuplicateBaseLinearizationEnergy { site: 0, kappa: -2 }
        ));
    }

    #[test]
    fn missing_base_partner_is_a_typed_error() {
        let mesh = mesh();
        let mut input = input(&mesh);
        input
            .linearization_energies
            .retain(|parameter| parameter.kappa.get() != 1);
        let error =
            build_spinor_iteration_basis(&envelope(), &geometry(&mesh), &[input]).unwrap_err();
        assert!(matches!(
            error,
            SpinorBuilderError::MissingBaseLinearizationEnergy {
                site: 0,
                l: 1,
                kappa: 1
            }
        ));
    }

    #[test]
    fn non_finite_base_energy_is_a_typed_error() {
        let mesh = mesh();
        let mut input = input(&mesh);
        input.linearization_energies[0].energy = Hartree(f64::NAN);
        let error =
            build_spinor_iteration_basis(&envelope(), &geometry(&mesh), &[input]).unwrap_err();
        assert!(matches!(
            error,
            SpinorBuilderError::NonFiniteBaseLinearizationEnergy {
                site: 0,
                kappa: 1,
                energy
            } if energy.get().is_nan()
        ));
    }

    #[test]
    fn spherical_reference_is_added_once_and_full_route_solves() {
        let mesh = mesh();
        let geometry = geometry(&mesh);
        let built = build_spinor_iteration_basis(&envelope(), &geometry, &[input(&mesh)]).unwrap();
        let blocks = build_full_spinor_site_blocks(
            RelativisticSpinorRoute::FullFourComponentFirstVariation,
            &built.compiled,
            &built.full_spinor_sites,
        )
        .unwrap();
        let overlap = blocks[0].overlap.at(0, 0).re;
        let energy = built.radial_sites[0].solutions[0].energy.get();
        assert!((blocks[0].hamiltonian.at(0, 0).re - energy * overlap).abs() < 2.0e-11);

        let interstitial = InterstitialPauliPotential::default();
        let solved = solve_spinor_k_point(&built, &geometry, &interstitial, 1.0e-10).unwrap();
        assert!(solved.solution.retained_dimension > 0);
        assert!(
            solved
                .solution
                .residuals
                .iter()
                .all(|residual| residual.relative < 2.0e-10)
        );
    }
}
