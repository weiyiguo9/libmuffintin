//! Concrete scalar Koelling--Harmon LAPW first-variation builder.

use muffintin_core::{Bohr, ExponentialMesh, Hartree, InterstitialGeometry, Lm};
use muffintin_lapw::{
    ApwBoundaryBasis, CompiledBasis, GeneralizedEigensolution, InterstitialPotential,
    LapwEigenproblem, LapwError, LapwSiteInput, LocalOrbitalLayout, PlaneWaveEnvelope,
    SiteOperatorBlocks, assemble_compiled, compile, lapw, solve_generalized_hermitian,
};
use muffintin_operators::{Collinear, OperatorError, SiteSpinOrbitBlock};
use muffintin_radial::{
    LinearizedRadialSolution, LocalOrbital, RadialComponents, RadialEquation, RadialError,
    RadialIntegralError, RadialIntegralKernel, RadialSolution, RadialSolver,
    SecondEnergyDerivative, radial_integral,
};
use muffintin_sphere::{
    HarmonicConvention, MatrixElementError, SphereField, SphereFieldError, SphereOrbital,
    SphereOrbitalError, matrix_element,
};
use muffintin_tensor::{Axis, DenseHermitianMatrix, TensorError};
use num_complex::Complex64;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use thiserror::Error;

const POTENTIAL_TOLERANCE: f64 = 4096.0 * f64::EPSILON;

/// One scalar local-orbital request, resolved by angular momentum rather than
/// signed `kappa`. `p1/2` requests belong to the full-spinor route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarLocalOrbitalRequest {
    Lo { l: u32, energy: Hartree },
    Hdlo { l: u32 },
}

impl ScalarLocalOrbitalRequest {
    pub const fn angular_momentum(self) -> u32 {
        match self {
            Self::Lo { l, .. } | Self::Hdlo { l } => l,
        }
    }
}

/// Complete physical input for one scalar/KH muffin-tin site.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarSiteInput {
    pub position: [Bohr; 3],
    pub radius: Bohr,
    pub mesh: ExponentialMesh,
    /// Physical spherical average `V(r)` in Hartree.
    pub spherical_potential: Vec<f64>,
    /// Full physical potential in normalized spherical harmonics.
    pub potential: SphereField,
    /// `linearization_energies[l]`.
    pub linearization_energies: Vec<Hartree>,
    pub local_orbitals: Vec<ScalarLocalOrbitalRequest>,
}

/// Exact primitive from which one matched scalar local orbital was built.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarLocalOrbitalOrigin {
    DistinctEnergy(RadialSolution),
    Hdlo(SecondEnergyDerivative),
}

/// Matched local orbital plus its untransformed radial origin.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltScalarLocalOrbital {
    pub request: ScalarLocalOrbitalRequest,
    pub orbital: LocalOrbital,
    pub origin: ScalarLocalOrbitalOrigin,
}

/// Radial solutions retained for one iteration and site.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarRadialSite {
    pub linearized: Vec<LinearizedRadialSolution>,
    /// Same `l`, then request order used by [`LocalOrbitalLayout`].
    pub local_orbitals: Vec<Vec<BuiltScalarLocalOrbital>>,
}

/// Concrete scalar basis and production site operators for one iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarIterationBasis {
    pub recipe_sites: Vec<LapwSiteInput>,
    pub compiled: CompiledBasis,
    pub radial_sites: Vec<ScalarRadialSite>,
    pub density_sites: Vec<crate::ScalarSiteBasis>,
    pub site_blocks: Vec<SiteOperatorBlocks>,
}

/// Assembled and solved scalar first-variation k point.
#[derive(Clone, Debug, PartialEq)]
pub struct SolvedScalarKPoint {
    pub eigenproblem: LapwEigenproblem,
    pub solution: GeneralizedEigensolution,
}

/// Build all scalar/KH radials, matched LOs, site-coordinate density orbitals,
/// compiled APW maps, and production muffin-tin `S/H` blocks.
pub fn build_scalar_iteration_basis(
    envelope: &PlaneWaveEnvelope,
    geometry: &InterstitialGeometry,
    sites: &[ScalarSiteInput],
) -> Result<ScalarIterationBasis, ScalarBuilderError> {
    if sites.len() != geometry.spheres().len() {
        return Err(ScalarBuilderError::SiteCount {
            expected: geometry.spheres().len(),
            actual: sites.len(),
        });
    }
    let built = sites
        .iter()
        .enumerate()
        .map(|(site, input)| build_site(site, input))
        .collect::<Result<Vec<_>, _>>()?;
    let recipe_sites = built
        .iter()
        .map(|site| site.recipe.clone())
        .collect::<Vec<_>>();
    let spec = lapw(envelope.clone(), geometry.cell_volume(), &recipe_sites);
    let compiled = compile(&spec).map_err(LapwError::from)?;
    // Geometry identity is checked here rather than deferred until every k solve.
    for (site, (compiled_site, sphere)) in compiled
        .site_geometry
        .iter()
        .zip(geometry.spheres())
        .enumerate()
    {
        if compiled_site.position != sphere.center || compiled_site.radius != sphere.radius {
            return Err(ScalarBuilderError::GeometryMismatch { site });
        }
    }
    Ok(ScalarIterationBasis {
        recipe_sites,
        compiled,
        radial_sites: built.iter().map(|site| site.radials.clone()).collect(),
        density_sites: built.iter().map(|site| site.density.clone()).collect(),
        site_blocks: built.into_iter().map(|site| site.block).collect(),
    })
}

/// Build two genuinely independent scalar/KH bases for collinear potentials.
pub fn build_collinear_scalar_iteration_bases(
    envelope: &PlaneWaveEnvelope,
    geometry: &InterstitialGeometry,
    sites: Collinear<&[ScalarSiteInput]>,
) -> Result<Collinear<ScalarIterationBasis>, ScalarBuilderError> {
    Ok(Collinear::new(
        build_scalar_iteration_basis(envelope, geometry, sites.up)?,
        build_scalar_iteration_basis(envelope, geometry, sites.down)?,
    ))
}

/// Assemble interstitial terms with the built site blocks and solve `H C = S C e`.
pub fn solve_scalar_k_point(
    basis: &ScalarIterationBasis,
    geometry: &InterstitialGeometry,
    interstitial_potential: &InterstitialPotential,
    relative_overlap_threshold: f64,
) -> Result<SolvedScalarKPoint, ScalarBuilderError> {
    let eigenproblem = assemble_compiled(
        &basis.compiled,
        geometry,
        interstitial_potential,
        &basis.site_blocks,
    )?;
    let solution = solve_generalized_hermitian(
        &eigenproblem.hamiltonian,
        &eigenproblem.overlap,
        relative_overlap_threshold,
    )?;
    Ok(SolvedScalarKPoint {
        eigenproblem,
        solution,
    })
}

/// Solve both independently generated collinear scalar channels.
pub fn solve_collinear_scalar_k_point(
    bases: Collinear<&ScalarIterationBasis>,
    geometry: &InterstitialGeometry,
    potentials: Collinear<&InterstitialPotential>,
    relative_overlap_threshold: f64,
) -> Result<Collinear<SolvedScalarKPoint>, ScalarBuilderError> {
    Ok(Collinear::new(
        solve_scalar_k_point(
            bases.up,
            geometry,
            potentials.up,
            relative_overlap_threshold,
        )?,
        solve_scalar_k_point(
            bases.down,
            geometry,
            potentials.down,
            relative_overlap_threshold,
        )?,
    ))
}

/// Apply the optional nonmagnetic SPEX second variation to a solved scalar/KH
/// k point. Magnetic routing is deliberately absent from this adapter.
pub fn solve_scalar_second_variation(
    basis: &ScalarIterationBasis,
    solved: &SolvedScalarKPoint,
    window: crate::FirstVariationWindow,
    site_soc_blocks: &[SiteSpinOrbitBlock],
) -> Result<crate::SecondVariationResult, ScalarBuilderError> {
    let first = crate::FirstVariationSubspace::select(
        window,
        &solved.solution.eigenvalues,
        &solved.solution.eigenvectors,
    )?;
    crate::solve_spex_second_variation(
        crate::FirstVariationRoute::NonmagneticScalarKoellingHarmon,
        &basis.compiled,
        &first,
        site_soc_blocks,
    )
    .map_err(Into::into)
}

#[derive(Clone)]
struct BuiltSite {
    recipe: LapwSiteInput,
    radials: ScalarRadialSite,
    density: crate::ScalarSiteBasis,
    block: SiteOperatorBlocks,
}

fn build_site(site: usize, input: &ScalarSiteInput) -> Result<BuiltSite, ScalarBuilderError> {
    if input.mesh.last() != input.radius {
        return Err(ScalarBuilderError::MeshRadius {
            site,
            mesh: input.mesh.last(),
            radius: input.radius,
        });
    }
    if input.spherical_potential.len() != input.mesh.len() {
        return Err(ScalarBuilderError::SphericalPotentialLength {
            site,
            expected: input.mesh.len(),
            actual: input.spherical_potential.len(),
        });
    }
    if input.linearization_energies.is_empty() {
        return Err(ScalarBuilderError::MissingLinearizationEnergies { site });
    }
    validate_full_potential(site, input)?;

    let solver = RadialSolver::new(
        &input.mesh,
        &input.spherical_potential,
        RadialEquation::ScalarKoellingHarmon,
    )?;
    let linearized = input
        .linearization_energies
        .iter()
        .enumerate()
        .map(|(l, &energy)| -> Result<_, ScalarBuilderError> {
            let l = u32::try_from(l).map_err(|_| ScalarBuilderError::AngularMomentumOverflow)?;
            Ok(solver.solve_with_energy_derivative(l, energy)?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut local_orbitals = vec![Vec::new(); linearized.len()];
    for &request in &input.local_orbitals {
        let l = usize::try_from(request.angular_momentum())
            .map_err(|_| ScalarBuilderError::AngularMomentumOverflow)?;
        let Some(base) = linearized.get(l) else {
            return Err(ScalarBuilderError::LocalOrbitalAngularMomentum {
                site,
                l: request.angular_momentum(),
                l_max: linearized.len() - 1,
            });
        };
        let built = match request {
            ScalarLocalOrbitalRequest::Lo { energy, .. } => {
                if energy == base.solution.energy() {
                    return Err(ScalarBuilderError::LocalOrbitalEnergyNotDistinct {
                        site,
                        l: request.angular_momentum(),
                        energy,
                    });
                }
                let raw = solver.solve(request.angular_momentum(), energy)?;
                let orbital = solver.local_orbital(base, energy)?;
                BuiltScalarLocalOrbital {
                    request,
                    orbital,
                    origin: ScalarLocalOrbitalOrigin::DistinctEnergy(raw),
                }
            }
            ScalarLocalOrbitalRequest::Hdlo { .. } => BuiltScalarLocalOrbital {
                request,
                orbital: base.hdlo(&input.mesh)?,
                origin: ScalarLocalOrbitalOrigin::Hdlo(base.second_energy_derivative.clone()),
            },
        };
        local_orbitals[l].push(built);
    }
    let radials = ScalarRadialSite {
        linearized,
        local_orbitals,
    };
    let local_layout =
        LocalOrbitalLayout::new(radials.local_orbitals.iter().map(Vec::len).collect());
    let recipe = LapwSiteInput {
        position: input.position,
        radius: input.radius,
        boundaries: radials
            .linearized
            .iter()
            .map(|radial| ApwBoundaryBasis {
                u: radial.solution.boundary,
                udot: radial.energy_derivative.boundary,
            })
            .collect(),
        local_orbitals: local_layout,
    };
    let orbitals = coordinate_orbitals(&radials)?;
    let density = crate::ScalarSiteBasis {
        mesh: input.mesh.clone(),
        orbitals: orbitals.clone(),
    };
    let block = build_site_operator_block(input, &radials, &orbitals)?;
    Ok(BuiltSite {
        recipe,
        radials,
        density,
        block,
    })
}

fn validate_full_potential(site: usize, input: &ScalarSiteInput) -> Result<(), ScalarBuilderError> {
    if input.potential.sample_count() != Some(input.mesh.len()) {
        return Err(ScalarBuilderError::FullPotentialLength {
            site,
            expected: input.mesh.len(),
            actual: input.potential.sample_count(),
        });
    }
    input
        .potential
        .validate_physical_reality(POTENTIAL_TOLERANCE)?;
    let Some(monopole) = input.potential.channel(0, 0) else {
        return Err(ScalarBuilderError::MissingMonopole { site });
    };
    let normalization = (4.0 * PI).sqrt();
    for (index, (&coefficient, &physical)) in
        monopole.iter().zip(&input.spherical_potential).enumerate()
    {
        let expected = Complex64::new(normalization * physical, 0.0);
        if (coefficient - expected).norm() > POTENTIAL_TOLERANCE * (1.0 + expected.norm()) {
            return Err(ScalarBuilderError::SphericalPotentialMismatch {
                site,
                index,
                expected,
                actual: coefficient,
            });
        }
    }
    Ok(())
}

fn nonspherical_potential(input: &ScalarSiteInput) -> Result<SphereField, ScalarBuilderError> {
    let normalization = (4.0 * PI).sqrt();
    let monopole = Lm::new(0, 0).expect("monopole is valid");
    let mut complex_channels = BTreeMap::<Lm, Vec<Complex64>>::new();
    for (channel, source) in input.potential.channels() {
        let values = if channel == monopole {
            source
                .iter()
                .zip(&input.spherical_potential)
                .map(|(&value, &spherical)| value - Complex64::new(normalization * spherical, 0.0))
                .collect::<Vec<_>>()
        } else {
            source.to_vec()
        };
        let expansion = match input.potential.convention() {
            HarmonicConvention::Complex => vec![(channel.m, Complex64::new(1.0, 0.0))],
            HarmonicConvention::Real => real_to_complex(channel.m),
        };
        for (m, coefficient) in expansion {
            let target = complex_channels
                .entry(Lm::new(channel.l, m).expect("real-harmonic transform preserves m"))
                .or_insert_with(|| vec![Complex64::new(0.0, 0.0); input.mesh.len()]);
            for (target, &value) in target.iter_mut().zip(&values) {
                *target += coefficient * value;
            }
        }
    }
    SphereField::new(
        HarmonicConvention::Complex,
        complex_channels
            .into_iter()
            .map(|(channel, values)| ((channel.l, channel.m), values)),
    )
    .map_err(Into::into)
}

fn real_to_complex(m: i32) -> Vec<(i32, Complex64)> {
    if m == 0 {
        return vec![(0, Complex64::new(1.0, 0.0))];
    }
    let q = i32::try_from(m.unsigned_abs()).expect("absolute magnetic index fits i32");
    let phase = if q & 1 == 0 { 1.0 } else { -1.0 };
    let inverse_sqrt_two = 1.0 / 2.0_f64.sqrt();
    if m > 0 {
        vec![
            (q, Complex64::new(phase * inverse_sqrt_two, 0.0)),
            (-q, Complex64::new(inverse_sqrt_two, 0.0)),
        ]
    } else {
        vec![
            (q, Complex64::new(0.0, inverse_sqrt_two)),
            (-q, Complex64::new(0.0, -phase * inverse_sqrt_two)),
        ]
    }
}

fn coordinate_orbitals(site: &ScalarRadialSite) -> Result<Vec<SphereOrbital>, ScalarBuilderError> {
    let mut orbitals = Vec::new();
    for (l, radial) in site.linearized.iter().enumerate() {
        let l = u32::try_from(l).map_err(|_| ScalarBuilderError::AngularMomentumOverflow)?;
        for m in -(l as i32)..=l as i32 {
            orbitals.push(SphereOrbital::new(
                l,
                m,
                radial.solution.p.clone(),
                radial.solution.q.clone(),
            )?);
            orbitals.push(SphereOrbital::new(
                l,
                m,
                radial.energy_derivative.p.clone(),
                radial.energy_derivative.q.clone(),
            )?);
        }
    }
    for (l, locals) in site.local_orbitals.iter().enumerate() {
        let l = u32::try_from(l).map_err(|_| ScalarBuilderError::AngularMomentumOverflow)?;
        for m in -(l as i32)..=l as i32 {
            for local in locals {
                orbitals.push(SphereOrbital::new(
                    l,
                    m,
                    local.orbital.p.clone(),
                    local.orbital.q.clone(),
                )?);
            }
        }
    }
    Ok(orbitals)
}

fn build_site_operator_block(
    input: &ScalarSiteInput,
    site: &ScalarRadialSite,
    orbitals: &[SphereOrbital],
) -> Result<SiteOperatorBlocks, ScalarBuilderError> {
    let dimension = orbitals.len();
    let mut overlap = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    let mut hamiltonian = overlap.clone();
    let shells = site
        .linearized
        .iter()
        .zip(&site.local_orbitals)
        .map(|(linearized, locals)| radial_shell_matrices(&input.mesh, linearized, locals))
        .collect::<Result<Vec<_>, _>>()?;
    let augmented_count = site
        .linearized
        .iter()
        .enumerate()
        .map(|(l, _)| 2 * (2 * l + 1))
        .sum::<usize>();
    for (l, shell) in shells.iter().enumerate() {
        let l_u32 = u32::try_from(l).map_err(|_| ScalarBuilderError::AngularMomentumOverflow)?;
        for m in -(l as i32)..=l as i32 {
            for left in 0..shell.dimension {
                let left_coordinate = scalar_coordinate(site, augmented_count, l_u32, m, left);
                for right in 0..shell.dimension {
                    let right_coordinate =
                        scalar_coordinate(site, augmented_count, l_u32, m, right);
                    overlap[left_coordinate * dimension + right_coordinate] =
                        Complex64::new(shell.overlap[left * shell.dimension + right], 0.0);
                    hamiltonian[left_coordinate * dimension + right_coordinate] =
                        Complex64::new(shell.hamiltonian[left * shell.dimension + right], 0.0);
                }
            }
        }
    }
    let nonspherical = nonspherical_potential(input)?;
    for left in 0..dimension {
        for right in left..dimension {
            let value = matrix_element(
                &input.mesh,
                &orbitals[left],
                &nonspherical,
                &orbitals[right],
            )?;
            hamiltonian[left * dimension + right] += value;
            if left != right {
                hamiltonian[right * dimension + left] += value.conj();
            }
        }
    }
    Ok(SiteOperatorBlocks {
        overlap: DenseHermitianMatrix::from_host_row_major(
            dimension,
            Axis::SiteCoordinate,
            overlap,
        )?,
        hamiltonian: DenseHermitianMatrix::from_host_row_major(
            dimension,
            Axis::SiteCoordinate,
            hamiltonian,
        )?,
    })
}

struct RadialShellMatrices {
    dimension: usize,
    overlap: Vec<f64>,
    hamiltonian: Vec<f64>,
}

#[derive(Clone)]
struct PrimitiveRadial {
    energy: Hartree,
    derivative_order: u8,
    lower_derivative: Option<usize>,
    p: Vec<f64>,
    q: Option<Vec<f64>>,
}

impl RadialComponents for PrimitiveRadial {
    fn large_component(&self) -> &[f64] {
        &self.p
    }

    fn small_component(&self) -> Option<&[f64]> {
        self.q.as_deref()
    }
}

fn radial_shell_matrices(
    mesh: &ExponentialMesh,
    linearized: &LinearizedRadialSolution,
    locals: &[BuiltScalarLocalOrbital],
) -> Result<RadialShellMatrices, ScalarBuilderError> {
    let dimension = 2 + locals.len();
    let energy = linearized.solution.energy();
    let mut primitives = vec![
        PrimitiveRadial {
            energy,
            derivative_order: 0,
            lower_derivative: None,
            p: linearized.solution.p.clone(),
            q: linearized.solution.q.clone(),
        },
        PrimitiveRadial {
            energy,
            derivative_order: 1,
            lower_derivative: Some(0),
            p: linearized.energy_derivative.p.clone(),
            q: linearized.energy_derivative.q.clone(),
        },
    ];
    for local in locals {
        primitives.push(match &local.origin {
            ScalarLocalOrbitalOrigin::DistinctEnergy(raw) => PrimitiveRadial {
                energy: raw.energy(),
                derivative_order: 0,
                lower_derivative: None,
                p: raw.p.clone(),
                q: raw.q.clone(),
            },
            ScalarLocalOrbitalOrigin::Hdlo(second) => PrimitiveRadial {
                energy,
                derivative_order: 2,
                lower_derivative: Some(1),
                p: second.p.clone(),
                q: second.q.clone(),
            },
        });
    }
    let mut primitive_overlap = vec![0.0; dimension * dimension];
    for left in 0..dimension {
        for right in left..dimension {
            let value = radial_integral(
                mesh,
                &primitives[left],
                &primitives[right],
                RadialIntegralKernel::Overlap,
            )?;
            primitive_overlap[left * dimension + right] = value;
            primitive_overlap[right * dimension + left] = value;
        }
    }
    let mut primitive_hamiltonian = vec![0.0; dimension * dimension];
    for left in 0..dimension {
        for right in left..dimension {
            let mut numerator = (primitives[left].energy.get() + primitives[right].energy.get())
                * primitive_overlap[left * dimension + right];
            if let Some(lower) = primitives[right].lower_derivative {
                numerator += f64::from(primitives[right].derivative_order)
                    * primitive_overlap[left * dimension + lower];
            }
            if let Some(lower) = primitives[left].lower_derivative {
                numerator += f64::from(primitives[left].derivative_order)
                    * primitive_overlap[lower * dimension + right];
            }
            let value = 0.5 * numerator;
            primitive_hamiltonian[left * dimension + right] = value;
            primitive_hamiltonian[right * dimension + left] = value;
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
        overlap: transform_symmetric(dimension, &primitive_overlap, &transform),
        hamiltonian: transform_symmetric(dimension, &primitive_hamiltonian, &transform),
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

fn scalar_coordinate(
    site: &ScalarRadialSite,
    augmented_count: usize,
    l: u32,
    m: i32,
    radial: usize,
) -> usize {
    if radial < 2 {
        2 * Lm::new(l, m).expect("assembly validates m").index() + radial
    } else {
        let preceding = site
            .local_orbitals
            .iter()
            .enumerate()
            .take(l as usize)
            .map(|(previous_l, locals)| (2 * previous_l + 1) * locals.len())
            .sum::<usize>();
        let count = site.local_orbitals[l as usize].len();
        augmented_count + preceding + (m + l as i32) as usize * count + radial - 2
    }
}

/// Invalid scalar/KH radial, site-operator, compilation, or eigensolve input.
#[derive(Debug, Error)]
pub enum ScalarBuilderError {
    #[error("received {actual} scalar sites, expected {expected}")]
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
    #[error("site {site} full potential has sample count {actual:?}, expected {expected}")]
    FullPotentialLength {
        site: usize,
        expected: usize,
        actual: Option<usize>,
    },
    #[error("site {site} has no spherical-harmonic monopole")]
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
    #[error("site {site} has no linearization energies")]
    MissingLinearizationEnergies { site: usize },
    #[error("angular momentum does not fit the host index type")]
    AngularMomentumOverflow,
    #[error("site {site} local orbital l={l} exceeds l_max={l_max}")]
    LocalOrbitalAngularMomentum { site: usize, l: u32, l_max: usize },
    #[error("site {site} l={l} local-orbital energy {energy} is not distinct")]
    LocalOrbitalEnergyNotDistinct {
        site: usize,
        l: u32,
        energy: Hartree,
    },
    #[error("compiled site {site} does not match the interstitial geometry")]
    GeometryMismatch { site: usize },
    #[error(transparent)]
    Radial(#[from] RadialError),
    #[error(transparent)]
    RadialIntegral(#[from] RadialIntegralError),
    #[error(transparent)]
    SphereField(#[from] SphereFieldError),
    #[error(transparent)]
    SphereOrbital(#[from] SphereOrbitalError),
    #[error(transparent)]
    MatrixElement(#[from] MatrixElementError),
    #[error(transparent)]
    Lapw(#[from] LapwError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    SecondVariation(#[from] crate::SecondVariationError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::{GVector, InverseBohr, Sphere, VolumeBohr3};
    use muffintin_lapw::{PlaneWave, RadialOverlapBlock, spex_spherical_radial_hamiltonian};

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

    fn site_input(
        mesh: &ExponentialMesh,
        l_max: u32,
        nonspherical: bool,
        local_orbitals: Vec<ScalarLocalOrbitalRequest>,
    ) -> ScalarSiteInput {
        let spherical_potential = vec![-0.2; mesh.len()];
        let mut channels = vec![(
            (0, 0),
            vec![Complex64::new(-(4.0 * PI).sqrt() * 0.2, 0.0); mesh.len()],
        )];
        if nonspherical {
            channels.push((
                (1, 0),
                mesh.radii()
                    .iter()
                    .map(|radius| Complex64::new(0.03 * radius.get(), 0.0))
                    .collect(),
            ));
        }
        ScalarSiteInput {
            position: [Bohr(0.0); 3],
            radius: mesh.last(),
            mesh: mesh.clone(),
            spherical_potential,
            potential: SphereField::new(HarmonicConvention::Complex, channels).unwrap(),
            linearization_energies: (0..=l_max)
                .map(|l| Hartree(0.2 + 0.08 * f64::from(l)))
                .collect(),
            local_orbitals,
        }
    }

    #[test]
    fn constant_spherical_site_matches_overlap_and_spex_hamiltonian_identity() {
        let mesh = mesh();
        let input = site_input(&mesh, 0, false, Vec::new());
        let built = build_scalar_iteration_basis(&envelope(), &geometry(&mesh), &[input]).unwrap();
        let radial = &built.radial_sites[0].linearized[0];
        let uu = radial_integral(
            &mesh,
            &radial.solution,
            &radial.solution,
            RadialIntegralKernel::Overlap,
        )
        .unwrap();
        let u_udot = radial_integral(
            &mesh,
            &radial.solution,
            &radial.energy_derivative,
            RadialIntegralKernel::Overlap,
        )
        .unwrap();
        let udot_udot = radial_integral(
            &mesh,
            &radial.energy_derivative,
            &radial.energy_derivative,
            RadialIntegralKernel::Overlap,
        )
        .unwrap();
        let block = &built.site_blocks[0];
        assert!((block.overlap.at(0, 0).re - uu).abs() < 2.0e-13);
        assert!((block.overlap.at(0, 1).re - u_udot).abs() < 2.0e-13);
        assert!((block.overlap.at(1, 1).re - udot_udot).abs() < 2.0e-13);
        let reference = spex_spherical_radial_hamiltonian(
            radial.solution.energy(),
            RadialOverlapBlock {
                uu,
                u_udot,
                udot_udot,
            },
        );
        for row in 0..2 {
            for column in 0..2 {
                assert!(
                    (block.hamiltonian.at(row, column) - reference.at(row, column)).norm()
                        < 2.0e-13
                );
            }
        }

        let pp = mesh
            .integrate(&radial.solution.p.iter().map(|p| p * p).collect::<Vec<_>>())
            .unwrap();
        assert!(uu > pp, "KH physical Q must contribute to overlap");
    }

    #[test]
    fn lo_hdlo_order_boundary_and_positive_overlap_follow_the_compiled_layout() {
        let mesh = mesh();
        let input = site_input(
            &mesh,
            1,
            false,
            vec![
                ScalarLocalOrbitalRequest::Lo {
                    l: 1,
                    energy: Hartree(0.7),
                },
                ScalarLocalOrbitalRequest::Hdlo { l: 0 },
            ],
        );
        let built = build_scalar_iteration_basis(&envelope(), &geometry(&mesh), &[input]).unwrap();
        assert_eq!(built.recipe_sites[0].local_orbitals.counts_by_l(), &[1, 1]);
        let site = &built.radial_sites[0];
        for locals in &site.local_orbitals {
            for local in locals {
                assert!(local.orbital.boundary.value.abs() < 5.0e-11);
                assert!(local.orbital.boundary.derivative.abs() < 5.0e-11);
            }
        }
        let orbitals = &built.density_sites[0].orbitals;
        assert_eq!(orbitals[8].angular(), Lm::new(0, 0).unwrap());
        assert_eq!(orbitals[9].angular(), Lm::new(1, -1).unwrap());
        assert_eq!(orbitals[10].angular(), Lm::new(1, 0).unwrap());
        assert_eq!(orbitals[11].angular(), Lm::new(1, 1).unwrap());

        let overlap = &built.site_blocks[0].overlap;
        for index in 0..overlap.dimension() {
            assert!(overlap.at(index, index).re > 0.0);
            for other in 0..overlap.dimension() {
                assert_eq!(overlap.at(index, other), overlap.at(other, index).conj());
            }
        }
        let trial = (0..overlap.dimension())
            .map(|index| Complex64::new(0.2 + 0.03 * index as f64, -0.01 * index as f64))
            .collect::<Vec<_>>();
        let mut norm = Complex64::new(0.0, 0.0);
        for left in 0..overlap.dimension() {
            for right in 0..overlap.dimension() {
                norm += trial[left].conj() * overlap.at(left, right) * trial[right];
            }
        }
        assert!(norm.re > 0.0);
        assert!(norm.im.abs() < 2.0e-12);
    }

    #[test]
    fn nonspherical_monopole_subtraction_leaves_gaunt_offdiagonal_and_solve_residuals() {
        let mesh = mesh();
        let input = site_input(&mesh, 1, true, Vec::new());
        let built = build_scalar_iteration_basis(&envelope(), &geometry(&mesh), &[input]).unwrap();
        let s_u = 0;
        let p0_u = 2 * Lm::new(1, 0).unwrap().index();
        let h = &built.site_blocks[0].hamiltonian;
        assert!(h.at(s_u, p0_u).norm() > 1.0e-10);
        assert_eq!(h.at(s_u, p0_u), h.at(p0_u, s_u).conj());

        let solved = solve_scalar_k_point(
            &built,
            &geometry(&mesh),
            &InterstitialPotential::default(),
            1.0e-10,
        )
        .unwrap();
        assert!(
            solved
                .solution
                .residuals
                .iter()
                .all(|residual| residual.relative < 2.0e-11)
        );
    }

    #[test]
    fn collinear_channels_build_independent_radials() {
        let mesh = mesh();
        let up = site_input(&mesh, 0, false, Vec::new());
        let mut down = up.clone();
        down.spherical_potential.fill(-0.35);
        down.potential = SphereField::new(
            HarmonicConvention::Complex,
            [(
                (0, 0),
                vec![Complex64::new(-(4.0 * PI).sqrt() * 0.35, 0.0); mesh.len()],
            )],
        )
        .unwrap();
        let bases = build_collinear_scalar_iteration_bases(
            &envelope(),
            &geometry(&mesh),
            Collinear::new(std::slice::from_ref(&up), std::slice::from_ref(&down)),
        )
        .unwrap();
        assert_ne!(
            bases.up.radial_sites[0].linearized[0].solution.p,
            bases.down.radial_sites[0].linearized[0].solution.p
        );
    }
}
