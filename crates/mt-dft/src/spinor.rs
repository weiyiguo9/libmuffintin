//! Full four-component first-variation magnetic and noncollinear route.

use muffintin_envelope::{SpinorCompiledBasis, SpinorSiteLayout};
use muffintin_core::{
    ExponentialMesh, InterstitialGeometry, Lm, RelativisticChannel, SpinProjection, gaunt,
};
use muffintin_operators::lapw::{
    GeneralizedEigensolution, InterstitialPauliPotential, LapwEigenproblem, LapwError,
    assemble_sra_spinor_compiled, solve_generalized_hermitian,
};
use muffintin_operators::{CompiledSiteProjection, OperatorError, SpinorSiteOperatorBlocks};
use muffintin_sphere::{
    HarmonicConvention, MatrixElementError, SphereField, SphereFieldError, SpinorSphereOrbital,
    spinor_matrix_element,
};
use muffintin_tensor::{Axis, DenseHermitianMatrix, TensorError};
use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

const REALITY_TOLERANCE: f64 = 4096.0 * f64::EPSILON;

/// Explicit relativistic route gate.
///
/// Magnetic, SOC, and noncollinear calculations in this module accept only
/// the full four-component first-variation route. `SecondVariation` exists so
/// callers cannot accidentally route such a request through a similarly named
/// scalar-basis adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelativisticSpinorRoute {
    FullFourComponentFirstVariation,
    SecondVariation,
}

/// Local potential `V0 I + Bx sigma_x + By sigma_y + Bz sigma_z`.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalPauliPotential {
    scalar: SphereField,
    magnetic: [SphereField; 3],
}

impl LocalPauliPotential {
    /// Validate one harmonic convention, one radial mesh size, and physical reality.
    pub fn new(
        scalar: SphereField,
        magnetic: [SphereField; 3],
    ) -> Result<Self, SpinorFirstVariationError> {
        scalar.validate_physical_reality(REALITY_TOLERANCE)?;
        for field in &magnetic {
            field.validate_physical_reality(REALITY_TOLERANCE)?;
            if field.convention() != scalar.convention() {
                return Err(SpinorFirstVariationError::PotentialConvention);
            }
            if field.sample_count() != scalar.sample_count() {
                return Err(SpinorFirstVariationError::PotentialSampleCount {
                    expected: scalar.sample_count(),
                    actual: field.sample_count(),
                });
            }
        }
        Ok(Self { scalar, magnetic })
    }

    pub const fn scalar(&self) -> &SphereField {
        &self.scalar
    }

    /// Magnetic fields in Cartesian order `[Bx, By, Bz]`.
    pub const fn magnetic(&self) -> &[SphereField; 3] {
        &self.magnetic
    }
}

/// Four-component site-coordinate data in the compiled projection order.
///
/// The first coordinates are `(channel, u/udot)` and the remainder are typed
/// `(kappa, twice_mu, n)` local orbitals. Every entry is a
/// [`SpinorSphereOrbital`] with mandatory physical `P` and `Q`; there is no
/// optional-small-component fallback. `reference_hamiltonian` carries the
/// field-independent radial Dirac part on the same coordinate axis.
#[derive(Clone, Debug, PartialEq)]
pub struct FullSpinorSiteInput {
    pub mesh: ExponentialMesh,
    pub channels: Vec<RelativisticChannel>,
    pub orbitals: Vec<SpinorSphereOrbital>,
    pub reference_hamiltonian: DenseHermitianMatrix,
    pub potential: LocalPauliPotential,
}

/// Assembled and solved full first-variation generalized eigenproblem.
#[derive(Clone, Debug, PartialEq)]
pub struct SolvedFullSpinorFirstVariation {
    pub site_blocks: Vec<SpinorSiteOperatorBlocks>,
    pub eigenproblem: LapwEigenproblem,
    pub solution: GeneralizedEigensolution,
}

/// Form all four-component site overlap and Hamiltonian blocks.
pub fn build_full_spinor_site_blocks(
    route: RelativisticSpinorRoute,
    compiled: &SpinorCompiledBasis,
    sites: &[FullSpinorSiteInput],
) -> Result<Vec<SpinorSiteOperatorBlocks>, SpinorFirstVariationError> {
    require_first_variation(route)?;
    if sites.len() != compiled.site_count() {
        return Err(SpinorFirstVariationError::SiteCount {
            expected: compiled.site_count(),
            actual: sites.len(),
        });
    }
    sites
        .iter()
        .enumerate()
        .map(|(site_index, site)| build_site_block(compiled, site_index, site))
        .collect()
}

/// Build, assemble, and solve the unique magnetic/SOC/noncollinear route.
///
/// The interstitial argument carries the complete `V0 I + B . sigma` field
/// consumed by SRA assembly. No interstitial Dirac small component is
/// introduced.
pub fn solve_full_spinor_first_variation(
    route: RelativisticSpinorRoute,
    compiled: &SpinorCompiledBasis,
    geometry: &InterstitialGeometry,
    interstitial: &InterstitialPauliPotential,
    sites: &[FullSpinorSiteInput],
    relative_overlap_threshold: f64,
) -> Result<SolvedFullSpinorFirstVariation, SpinorFirstVariationError> {
    let site_blocks = build_full_spinor_site_blocks(route, compiled, sites)?;
    let eigenproblem =
        assemble_sra_spinor_compiled(compiled, geometry, interstitial, &site_blocks)?;
    let solution = solve_generalized_hermitian(
        &eigenproblem.hamiltonian,
        &eigenproblem.overlap,
        relative_overlap_threshold,
    )?;
    Ok(SolvedFullSpinorFirstVariation {
        site_blocks,
        eigenproblem,
        solution,
    })
}

fn require_first_variation(
    route: RelativisticSpinorRoute,
) -> Result<(), SpinorFirstVariationError> {
    if route == RelativisticSpinorRoute::FullFourComponentFirstVariation {
        Ok(())
    } else {
        Err(SpinorFirstVariationError::SecondVariationRejected)
    }
}

fn build_site_block(
    compiled: &SpinorCompiledBasis,
    site_index: usize,
    site: &FullSpinorSiteInput,
) -> Result<SpinorSiteOperatorBlocks, SpinorFirstVariationError> {
    CompiledSiteProjection::spinor(compiled, site_index, &site.channels)?;
    let site_layout = compiled
        .layout
        .site_layout(site_index)
        .ok_or(SpinorFirstVariationError::SiteIndex(site_index))?;
    let geometry = compiled
        .site_geometry
        .get(site_index)
        .ok_or(SpinorFirstVariationError::SiteGeometry(site_index))?;
    if site.mesh.last() != geometry.radius {
        return Err(SpinorFirstVariationError::SiteMeshRadius {
            site: site_index,
            expected: geometry.radius.get(),
            actual: site.mesh.last().get(),
        });
    }
    let expected_channels = coordinate_channels(&site.channels, site_layout);
    if site.orbitals.len() != expected_channels.len() {
        return Err(SpinorFirstVariationError::OrbitalCount {
            site: site_index,
            expected: expected_channels.len(),
            actual: site.orbitals.len(),
        });
    }
    for (coordinate, (orbital, &expected)) in
        site.orbitals.iter().zip(&expected_channels).enumerate()
    {
        if orbital.channel() != expected {
            return Err(SpinorFirstVariationError::CoordinateChannel {
                site: site_index,
                coordinate,
                expected,
                actual: orbital.channel(),
            });
        }
        for (component, actual) in [("P", orbital.p().len()), ("Q", orbital.q().len())] {
            if actual != site.mesh.len() {
                return Err(SpinorFirstVariationError::OrbitalMesh {
                    site: site_index,
                    coordinate,
                    component,
                    expected: site.mesh.len(),
                    actual,
                });
            }
        }
    }
    let dimension = expected_channels.len();
    if site.reference_hamiltonian.dimension() != dimension {
        return Err(SpinorFirstVariationError::ReferenceDimension {
            site: site_index,
            expected: dimension,
            actual: site.reference_hamiltonian.dimension(),
        });
    }
    if site.reference_hamiltonian.axis() != Axis::SiteCoordinate {
        return Err(SpinorFirstVariationError::Tensor(TensorError::Axis {
            index: 0,
            expected: Axis::SiteCoordinate,
            actual: site.reference_hamiltonian.axis(),
        }));
    }
    for field in std::iter::once(site.potential.scalar()).chain(site.potential.magnetic().iter()) {
        if field.sample_count() != Some(site.mesh.len()) {
            return Err(SpinorFirstVariationError::PotentialMesh {
                site: site_index,
                expected: site.mesh.len(),
                actual: field.sample_count(),
            });
        }
    }

    let constant = SphereField::new(
        HarmonicConvention::Complex,
        [(
            (0, 0),
            vec![Complex64::new((4.0 * PI).sqrt(), 0.0); site.mesh.len()],
        )],
    )?;
    let mut overlap = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    let mut hamiltonian = overlap.clone();
    for left in 0..dimension {
        for right in left..dimension {
            let overlap_value = spinor_matrix_element(
                &site.mesh,
                &site.orbitals[left],
                &constant,
                &site.orbitals[right],
            )?;
            let scalar = spinor_matrix_element(
                &site.mesh,
                &site.orbitals[left],
                site.potential.scalar(),
                &site.orbitals[right],
            )?;
            let mut magnetic = Complex64::new(0.0, 0.0);
            for (axis, field) in site.potential.magnetic().iter().enumerate() {
                magnetic += pauli_field_matrix_element(
                    &site.mesh,
                    &site.orbitals[left],
                    field,
                    &site.orbitals[right],
                    axis,
                )?;
            }
            set_hermitian(&mut overlap, dimension, left, right, overlap_value);
            set_hermitian(
                &mut hamiltonian,
                dimension,
                left,
                right,
                site.reference_hamiltonian.at(left, right) + scalar + magnetic,
            );
        }
    }
    Ok(SpinorSiteOperatorBlocks {
        channels: site.channels.clone(),
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

fn coordinate_channels(
    apw_channels: &[RelativisticChannel],
    local_orbitals: &SpinorSiteLayout,
) -> Vec<RelativisticChannel> {
    let mut result = Vec::with_capacity(2 * apw_channels.len() + local_orbitals.len());
    for &channel in apw_channels {
        result.extend([channel, channel]);
    }
    for &(kappa, count) in local_orbitals.counts_by_kappa() {
        for channel in kappa.channels() {
            result.extend(std::iter::repeat_n(channel, count));
        }
    }
    result
}

fn pauli_field_matrix_element(
    mesh: &ExponentialMesh,
    left: &SpinorSphereOrbital,
    field: &SphereField,
    right: &SpinorSphereOrbital,
    axis: usize,
) -> Result<Complex64, SpinorFirstVariationError> {
    let mut result = Complex64::new(0.0, 0.0);
    for (channel, values) in field.channels() {
        let pp_angular = pauli_angular(
            field.convention(),
            left.channel(),
            channel,
            right.channel(),
            axis,
        );
        let qq_angular = pauli_angular(
            field.convention(),
            left.channel().opposite_kappa(),
            channel,
            right.channel().opposite_kappa(),
            axis,
        );
        result += pp_angular * integrate_component(mesh, left.p(), values, right.p())?
            + qq_angular * integrate_component(mesh, left.q(), values, right.q())?;
    }
    Ok(result)
}

fn pauli_angular(
    convention: HarmonicConvention,
    left: RelativisticChannel,
    field: Lm,
    right: RelativisticChannel,
    axis: usize,
) -> Complex64 {
    let mut value = Complex64::new(0.0, 0.0);
    for left_term in left.spinor_harmonic_terms().into_iter().flatten() {
        for right_term in right.spinor_harmonic_terms().into_iter().flatten() {
            value += left_term.coefficient
                * right_term.coefficient
                * pauli(axis, left_term.spin, right_term.spin)
                * orbital_field_gaunt(convention, left_term.orbital, field, right_term.orbital);
        }
    }
    value
}

fn pauli(axis: usize, left: SpinProjection, right: SpinProjection) -> Complex64 {
    use SpinProjection::{Down, Up};
    match (axis, left, right) {
        (0, Up, Down) | (0, Down, Up) => Complex64::new(1.0, 0.0),
        (1, Up, Down) => Complex64::new(0.0, -1.0),
        (1, Down, Up) => Complex64::new(0.0, 1.0),
        (2, Up, Up) => Complex64::new(1.0, 0.0),
        (2, Down, Down) => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, 0.0),
    }
}

fn orbital_field_gaunt(
    convention: HarmonicConvention,
    left: Lm,
    field: Lm,
    right: Lm,
) -> Complex64 {
    if convention == HarmonicConvention::Complex || field.m == 0 {
        return Complex64::new(complex_matrix_gaunt(left, field, right), 0.0);
    }
    let q = i32::try_from(field.m.unsigned_abs()).expect("validated M fits i32");
    let positive = complex_matrix_gaunt(
        left,
        Lm::new(field.l, q).expect("absolute M remains valid"),
        right,
    );
    let negative = complex_matrix_gaunt(
        left,
        Lm::new(field.l, -q).expect("negated M remains valid"),
        right,
    );
    let inverse_sqrt_two = 1.0 / 2.0_f64.sqrt();
    if field.m > 0 {
        Complex64::new(
            (magnetic_phase(q) * positive + negative) * inverse_sqrt_two,
            0.0,
        )
    } else {
        Complex64::new(
            0.0,
            (positive - magnetic_phase(q) * negative) * inverse_sqrt_two,
        )
    }
}

fn complex_matrix_gaunt(left: Lm, field: Lm, right: Lm) -> f64 {
    magnetic_phase(right.m) * gaunt(left.l, field.l, right.l, left.m, field.m, -right.m)
}

fn magnetic_phase(m: i32) -> f64 {
    if m.unsigned_abs() % 2 == 0 { 1.0 } else { -1.0 }
}

fn integrate_component(
    mesh: &ExponentialMesh,
    left: &[f64],
    field: &[Complex64],
    right: &[f64],
) -> Result<Complex64, SpinorFirstVariationError> {
    let real = left
        .iter()
        .zip(field)
        .zip(right)
        .map(|((&left, &field), &right)| left * field.re * right)
        .collect::<Vec<_>>();
    let imaginary = left
        .iter()
        .zip(field)
        .zip(right)
        .map(|((&left, &field), &right)| left * field.im * right)
        .collect::<Vec<_>>();
    Ok(Complex64::new(
        mesh.integrate(&real)?,
        mesh.integrate(&imaginary)?,
    ))
}

fn set_hermitian(
    data: &mut [Complex64],
    dimension: usize,
    row: usize,
    column: usize,
    value: Complex64,
) {
    data[row * dimension + column] = value;
    data[column * dimension + row] = value.conj();
}

/// Invalid full first-variation route, coordinate layout, or local matrix.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SpinorFirstVariationError {
    #[error("second variation is not the magnetic/noncollinear full-4c route")]
    SecondVariationRejected,
    #[error("spinor first variation has {actual} sites, expected {expected}")]
    SiteCount { expected: usize, actual: usize },
    #[error("spinor site index {0} is absent from the compiled layout")]
    SiteIndex(usize),
    #[error("spinor site index {0} is absent from compiled geometry")]
    SiteGeometry(usize),
    #[error("spinor site {site} mesh radius {actual} does not match compiled radius {expected}")]
    SiteMeshRadius {
        site: usize,
        expected: f64,
        actual: f64,
    },
    #[error("spinor site {site} has {actual} orbitals, expected {expected}")]
    OrbitalCount {
        site: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "spinor site {site} coordinate {coordinate} has channel {actual:?}, expected {expected:?}"
    )]
    CoordinateChannel {
        site: usize,
        coordinate: usize,
        expected: RelativisticChannel,
        actual: RelativisticChannel,
    },
    #[error(
        "spinor site {site} coordinate {coordinate} {component} has {actual} samples, expected {expected}"
    )]
    OrbitalMesh {
        site: usize,
        coordinate: usize,
        component: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("spinor site {site} reference Hamiltonian has dimension {actual}, expected {expected}")]
    ReferenceDimension {
        site: usize,
        expected: usize,
        actual: usize,
    },
    #[error("local Pauli-potential fields use different harmonic conventions")]
    PotentialConvention,
    #[error("local Pauli-potential sample counts differ: expected {expected:?}, got {actual:?}")]
    PotentialSampleCount {
        expected: Option<usize>,
        actual: Option<usize>,
    },
    #[error("spinor site {site} potential has {actual:?} samples, expected {expected}")]
    PotentialMesh {
        site: usize,
        expected: usize,
        actual: Option<usize>,
    },
    #[error(transparent)]
    SphereField(#[from] SphereFieldError),
    #[error(transparent)]
    MatrixElement(#[from] MatrixElementError),
    #[error(transparent)]
    Mesh(#[from] muffintin_core::MeshError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Lapw(#[from] LapwError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::{Bohr, GVector, InverseBohr, Kappa, Sphere, TwiceMu, VolumeBohr3};
    use muffintin_operators::lapw::{
        ApwSiteGeometry, PlaneWave, Provenance, SpinorBasisLayout, SpinorCompiledBasis,
        SpinorPlaneWaveAugmentation, SpinorSiteLayout,
    };

    fn mesh() -> ExponentialMesh {
        ExponentialMesh::new(Bohr(1.0e-4), 0.02, 301).unwrap()
    }

    fn radial(mesh: &ExponentialMesh, scale: f64) -> Vec<f64> {
        mesh.radii()
            .iter()
            .map(|radius| scale * radius.get() * (-radius.get()).exp())
            .collect()
    }

    fn field(mesh: &ExponentialMesh, physical_value: f64) -> SphereField {
        SphereField::new(
            HarmonicConvention::Complex,
            [(
                (0, 0),
                vec![Complex64::new((4.0 * PI).sqrt() * physical_value, 0.0); mesh.len()],
            )],
        )
        .unwrap()
    }

    fn potential(mesh: &ExponentialMesh, scalar: f64, magnetic: [f64; 3]) -> LocalPauliPotential {
        LocalPauliPotential::new(
            field(mesh, scalar),
            magnetic.map(|value| field(mesh, value)),
        )
        .unwrap()
    }

    fn channels() -> Vec<RelativisticChannel> {
        let kappa = Kappa::new(-1).unwrap();
        vec![
            RelativisticChannel::new(kappa, TwiceMu::new(-1).unwrap()).unwrap(),
            RelativisticChannel::new(kappa, TwiceMu::new(1).unwrap()).unwrap(),
        ]
    }

    fn compiled(mesh: &ExponentialMesh) -> SpinorCompiledBasis {
        let channels = channels();
        let g = GVector {
            index: [0; 3],
            cartesian: [InverseBohr(0.0); 3],
            norm: InverseBohr(0.0),
        };
        SpinorCompiledBasis {
            layout: SpinorBasisLayout::new(1, vec![SpinorSiteLayout::default()]),
            plane_waves: vec![PlaneWave::new([InverseBohr(0.0); 3], g)],
            site_augmentations: vec![vec![SpinorPlaneWaveAugmentation {
                channels: channels.clone(),
                coefficients: [
                    vec![
                        [Complex64::new(0.0, 0.0); 2],
                        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    ],
                    vec![
                        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                        [Complex64::new(0.0, 0.0); 2],
                    ],
                ],
            }]],
            site_geometry: vec![ApwSiteGeometry {
                position: [Bohr(0.0); 3],
                radius: mesh.last(),
            }],
            provenance: Provenance::default(),
        }
    }

    fn site(
        mesh: &ExponentialMesh,
        q_scale: f64,
        potential: LocalPauliPotential,
    ) -> FullSpinorSiteInput {
        let channels = channels();
        let p = radial(mesh, 1.0);
        let p_dot = radial(mesh, 0.35);
        let q = radial(mesh, q_scale);
        let q_dot = radial(mesh, 0.35 * q_scale);
        let mut orbitals = Vec::new();
        for &channel in &channels {
            orbitals.push(SpinorSphereOrbital::new(channel, p.clone(), q.clone()).unwrap());
            orbitals.push(SpinorSphereOrbital::new(channel, p_dot.clone(), q_dot.clone()).unwrap());
        }
        FullSpinorSiteInput {
            mesh: mesh.clone(),
            channels,
            orbitals,
            reference_hamiltonian: DenseHermitianMatrix::from_upper_triangle(
                4,
                Axis::SiteCoordinate,
                |_, _| Complex64::new(0.0, 0.0),
            )
            .unwrap(),
            potential,
        }
    }

    #[test]
    fn scalar_matrix_contains_physical_q_and_has_the_large_c_limit() {
        let mesh = mesh();
        let compiled = compiled(&mesh);
        let without_q = site(&mesh, 0.0, potential(&mesh, 1.0, [0.0; 3]));
        let with_q = site(&mesh, 0.4, potential(&mesh, 1.0, [0.0; 3]));
        let almost_large_c = site(&mesh, 1.0e-6, potential(&mesh, 1.0, [0.0; 3]));
        let scalar = build_full_spinor_site_blocks(
            RelativisticSpinorRoute::FullFourComponentFirstVariation,
            &compiled,
            &[without_q],
        )
        .unwrap();
        let four_component = build_full_spinor_site_blocks(
            RelativisticSpinorRoute::FullFourComponentFirstVariation,
            &compiled,
            &[with_q],
        )
        .unwrap();
        let reduced = build_full_spinor_site_blocks(
            RelativisticSpinorRoute::FullFourComponentFirstVariation,
            &compiled,
            &[almost_large_c],
        )
        .unwrap();
        assert!(four_component[0].hamiltonian.at(0, 0).re > scalar[0].hamiltonian.at(0, 0).re);
        assert!(
            (reduced[0].hamiltonian.at(0, 0) - scalar[0].hamiltonian.at(0, 0)).norm() < 1.0e-11
        );
    }

    #[test]
    fn collinear_bz_and_transverse_fields_form_pauli_blocks() {
        let mesh = mesh();
        let compiled = compiled(&mesh);
        let collinear = site(&mesh, 0.0, potential(&mesh, 0.0, [0.0, 0.0, 0.3]));
        let block = build_full_spinor_site_blocks(
            RelativisticSpinorRoute::FullFourComponentFirstVariation,
            &compiled,
            &[collinear],
        )
        .unwrap();
        let down = block[0].hamiltonian.at(0, 0).re;
        let up = block[0].hamiltonian.at(2, 2).re;
        assert!(down < 0.0 && up > 0.0);
        assert!((down + up).abs() < 2.0e-14);
        assert_eq!(block[0].hamiltonian.at(0, 2), Complex64::new(0.0, 0.0));

        let transverse = site(&mesh, 0.0, potential(&mesh, 0.0, [0.2, -0.1, 0.0]));
        let mixed = build_full_spinor_site_blocks(
            RelativisticSpinorRoute::FullFourComponentFirstVariation,
            &compiled,
            &[transverse],
        )
        .unwrap();
        let off_diagonal = mixed[0].hamiltonian.at(0, 2);
        assert!(off_diagonal.re.abs() > 1.0e-6);
        assert!(off_diagonal.im.abs() > 1.0e-6);
        assert_eq!(mixed[0].hamiltonian.at(2, 0), off_diagonal.conj());
    }

    #[test]
    fn full_route_assembles_and_solves_while_second_variation_is_rejected() {
        let mesh = mesh();
        let compiled = compiled(&mesh);
        let input = site(&mesh, 0.02, potential(&mesh, 0.1, [0.01, 0.0, 0.03]));
        assert_eq!(
            build_full_spinor_site_blocks(
                RelativisticSpinorRoute::SecondVariation,
                &compiled,
                std::slice::from_ref(&input),
            ),
            Err(SpinorFirstVariationError::SecondVariationRejected)
        );
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: [Bohr(0.0); 3],
                radius: mesh.last(),
            }],
        )
        .unwrap();
        let interstitial = InterstitialPauliPotential::default();
        let solved = solve_full_spinor_first_variation(
            RelativisticSpinorRoute::FullFourComponentFirstVariation,
            &compiled,
            &geometry,
            &interstitial,
            &[input],
            1.0e-12,
        )
        .unwrap();
        assert_eq!(solved.solution.retained_dimension, 2);
        assert!(
            solved
                .solution
                .residuals
                .iter()
                .all(|residual| residual.absolute < 1.0e-11)
        );
    }
}
