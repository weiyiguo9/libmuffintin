//! LAPW facade over historical-method-name-free basis, operator, and recipe crates.
//!
//! Plane waves use `exp(+i (k+G) dot r) / sqrt(Omega)`. Site muffin-tin
//! contributions are `einsum("ci,cd,dj->ij", [P^*, B, P])` in
//! `libmuffintin-tensor`. Interstitial kinetic assembly stays here and
//! explicitly selects [`KineticOperatorConvention::SpexSymmetricLaplacian`].

#![forbid(unsafe_code)]

use muffintin_basis::BasisError;
use muffintin_core::{Bohr, Hartree, InterstitialGeometry, InverseBohr, KineticOperatorConvention};
use muffintin_envelope::EnvelopeError;
use muffintin_operators::{OperatorError, add_site_contributions, add_spinor_site_contributions};
use muffintin_tensor::{Axis, TensorError};
use num_complex::Complex64;
use std::collections::BTreeMap;
use thiserror::Error;

pub use muffintin_basis::BasisLayout as LapwBasisLayout;
pub use muffintin_basis::{
    ApwBoundaryBasis, ApwMatch, ApwSiteAugmentation, ApwSiteGeometry, BasisBlock, BasisLayout,
    BasisSpec, CompiledBasis, LocalOrbitalLayout, PlaneWaveAugmentation, Provenance,
    SpinorApwMatch, SpinorBasisLayout, SpinorCompiledBasis, SpinorPlaneWaveAugmentation,
    SpinorSiteLayout, augmentation_coefficients, compile, match_apw_boundary,
    spinor_augmentation_coefficients,
};
pub use muffintin_envelope::{
    PlaneWave, PlaneWaveEnvelope, rayleigh_coefficient, site_translation_phase,
};
pub use muffintin_operators::{
    Collinear, EigenpairResidual, GeneralizedEigensolution, OperatorSet as LapwEigenproblem,
    SiteOperatorBlocks, SpinorSiteOperatorBlocks, solve_generalized_hermitian,
};
pub use muffintin_recipes::{LapwSiteInput, lapw};
pub use muffintin_tensor::{ComplexTensor, DenseEigenvectors, DenseHermitianMatrix};

const INTERSTITIAL_KINETIC: KineticOperatorConvention =
    KineticOperatorConvention::SpexSymmetricLaplacian;

/// LAPW construction, assembly, or reference-comparison error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum LapwError {
    #[error("all plane waves in an overlap matrix must share one k point")]
    MixedKPoints,
    #[error("interstitial coefficient failed: {0}")]
    StepFunction(String),
    #[error("potential coefficient for G={g:?} conflicts with its Hermitian partner")]
    NonHermitianPotential { g: [i32; 3] },
    #[error("wave-vector norm must be finite and nonnegative, got {0}")]
    InvalidWaveVector(f64),
    #[error("reference comparison points to missing k={k_index}, band={band_index}")]
    MissingBand { k_index: usize, band_index: usize },
    #[error("reference tolerance must be finite and nonnegative, got {0}")]
    InvalidReferenceTolerance(f64),
    #[error("at least one band reference is required")]
    MissingReferenceData,
    #[error("compiled basis stores {actual} APW site geometries, layout has {expected} sites")]
    CompiledSiteGeometryCount { expected: usize, actual: usize },
    #[error(
        "compiled basis has {compiled} APW sites, interstitial geometry has {geometry} spheres"
    )]
    SiteGeometryCount { compiled: usize, geometry: usize },
    #[error(
        "site {site} compiled position {compiled:?} does not match interstitial sphere {geometry:?}"
    )]
    SitePositionMismatch {
        site: usize,
        compiled: [Bohr; 3],
        geometry: [Bohr; 3],
    },
    #[error("site {site} compiled radius {compiled} does not match interstitial sphere {geometry}")]
    SiteRadiusMismatch {
        site: usize,
        compiled: Bohr,
        geometry: Bohr,
    },
    #[error(transparent)]
    Envelope(#[from] EnvelopeError),
    #[error(transparent)]
    Basis(#[from] BasisError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

/// Cell-normalized interstitial potential coefficients keyed by integer
/// reciprocal-lattice coordinates.  Supplying one member also installs its
/// conjugate partner, so lookup is explicitly Hermitian.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InterstitialPotential {
    coefficients: BTreeMap<[i32; 3], Complex64>,
}

impl InterstitialPotential {
    pub fn new(
        coefficients: impl IntoIterator<Item = ([i32; 3], Complex64)>,
    ) -> Result<Self, LapwError> {
        let mut result = Self::default();
        for (g, value) in coefficients {
            let minus_g = g.map(|component| -component);
            if let Some(previous) = result.coefficients.get(&g) {
                if (*previous - value).norm() > 64.0 * f64::EPSILON * value.norm().max(1.0) {
                    return Err(LapwError::NonHermitianPotential { g });
                }
            }
            if let Some(previous) = result.coefficients.get(&minus_g) {
                if (*previous - value.conj()).norm() > 64.0 * f64::EPSILON * value.norm().max(1.0) {
                    return Err(LapwError::NonHermitianPotential { g });
                }
            }
            if g == [0; 3] && value.im.abs() > 64.0 * f64::EPSILON * value.re.abs().max(1.0) {
                return Err(LapwError::NonHermitianPotential { g });
            }
            result.coefficients.insert(g, value);
            result.coefficients.insert(minus_g, value.conj());
        }
        Ok(result)
    }

    pub fn coefficient(&self, g: [i32; 3]) -> Complex64 {
        self.coefficients.get(&g).copied().unwrap_or_default()
    }
}

/// Cell-normalized Fourier fields for an interstitial Pauli potential
/// `V0 I + B dot sigma`.
///
/// In the Pauli order `(up, down)`, the spin blocks are `V0 + Bz`,
/// `Bx - i By`, `Bx + i By`, and `V0 - Bz`. Each Cartesian field is a
/// Hermitian real-space field represented by [`InterstitialPotential`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InterstitialPauliPotential {
    pub v0: InterstitialPotential,
    pub bx: InterstitialPotential,
    pub by: InterstitialPotential,
    pub bz: InterstitialPotential,
}

impl InterstitialPauliPotential {
    pub fn new(
        v0: InterstitialPotential,
        bx: InterstitialPotential,
        by: InterstitialPotential,
        bz: InterstitialPotential,
    ) -> Self {
        Self { v0, bx, by, bz }
    }
}

/// A real symmetric `2 x 2` radial overlap block in the `(u, du/dE)` basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialOverlapBlock {
    pub uu: f64,
    pub u_udot: f64,
    pub udot_udot: f64,
}

/// One angular-channel boundary trace used by the SRA variational surface term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SraSurfaceTrace {
    pub value: Complex64,
    /// Derivative along the outward normal of the muffin-tin sphere.
    pub outward_derivative: Complex64,
}

/// Schlosser--Marcus surface correction for one normalized angular channel.
///
/// `interstitial` and `muffin_tin` traces are kept separate on each side of
/// the matrix element. The factor is `-1/4` in this crate's Hartree convention
/// (`T = -1/2 laplacian`); Kutepov's Appendix A writes `-1/2` in Rydberg
/// units. A continuous value and derivative make the correction vanish.
pub fn sra_schlosser_marcus_surface_correction(
    radius: Bohr,
    left_interstitial: SraSurfaceTrace,
    left_muffin_tin: SraSurfaceTrace,
    right_interstitial: SraSurfaceTrace,
    right_muffin_tin: SraSurfaceTrace,
) -> Result<Complex64, LapwError> {
    if !radius.get().is_finite() || radius.get() <= 0.0 {
        return Err(BasisError::InvalidRadius(radius.get()).into());
    }
    let bracket = left_muffin_tin.value.conj() * right_interstitial.outward_derivative
        - left_muffin_tin.outward_derivative.conj() * right_interstitial.value
        - left_interstitial.value.conj() * right_muffin_tin.outward_derivative
        + left_interstitial.outward_derivative.conj() * right_muffin_tin.value;
    Ok(-0.25 * radius.get().powi(2) * bracket)
}

/// Matrices and eigensolution for one spin channel.
#[derive(Clone, Debug, PartialEq)]
pub struct SolvedLapwEigenproblem {
    pub eigenproblem: LapwEigenproblem,
    pub solution: GeneralizedEigensolution,
}

/// Assemble `S` and `H` through the canonical LAPW recipe path.
///
/// The facade constructs a [`BasisSpec`] with [`lapw`], compiles it, then
/// uses [`assemble_compiled`]. Interstitial terms occupy only the plane-wave
/// block and use [`KineticOperatorConvention::SpexSymmetricLaplacian`].
/// Every site is added as `P^dagger B P` through the shared operator layer.
pub fn assemble_eigenproblem(
    envelope: &PlaneWaveEnvelope,
    geometry: &InterstitialGeometry,
    potential: &InterstitialPotential,
    recipe_sites: &[LapwSiteInput],
    local_operators: &[SiteOperatorBlocks],
) -> Result<LapwEigenproblem, LapwError> {
    let spec = lapw(envelope.clone(), geometry.cell_volume(), recipe_sites);
    let compiled = compile(&spec)?;
    assemble_compiled(&compiled, geometry, potential, local_operators)
}

/// Assemble `S` and `H` from a compiled basis and local site operator blocks.
///
/// This is the shared LAPW interstitial + site-projection path used by both
/// the facade and an explicit [`muffintin_basis::BasisSpec`] route. Every
/// compiled APW site position and radius is checked against
/// [`InterstitialGeometry::spheres`].
pub fn assemble_compiled(
    compiled: &CompiledBasis,
    geometry: &InterstitialGeometry,
    potential: &InterstitialPotential,
    sites: &[SiteOperatorBlocks],
) -> Result<LapwEigenproblem, LapwError> {
    let plane_waves = &compiled.plane_waves;
    validate_plane_wave_norms(plane_waves)?;
    if let Some(first) = plane_waves.first() {
        if plane_waves.iter().any(|wave| wave.k != first.k) {
            return Err(LapwError::MixedKPoints);
        }
    }
    validate_compiled_geometry(compiled, geometry)?;
    if sites.len() != geometry.spheres().len() {
        return Err(OperatorError::SiteCount {
            expected: geometry.spheres().len(),
            actual: sites.len(),
        }
        .into());
    }

    let dimension = compiled.layout.dimension();
    let mut overlap = vec![Complex64::default(); dimension * dimension];
    let mut hamiltonian = vec![Complex64::default(); dimension * dimension];
    for i in 0..plane_waves.len() {
        for j in i..plane_waves.len() {
            let cartesian_difference = std::array::from_fn(|axis| {
                InverseBohr(
                    plane_waves[i].g.cartesian[axis].get() - plane_waves[j].g.cartesian[axis].get(),
                )
            });
            let integer_difference = std::array::from_fn(|axis| {
                plane_waves[i].g.index[axis] - plane_waves[j].g.index[axis]
            });
            let theta = geometry
                .coefficient(cartesian_difference)
                .map_err(|error| LapwError::StepFunction(error.to_string()))?;
            let kinetic = INTERSTITIAL_KINETIC.prefactor(plane_waves[i].q, plane_waves[j].q);
            set_hermitian(&mut overlap, dimension, i, j, theta);
            set_hermitian(
                &mut hamiltonian,
                dimension,
                i,
                j,
                theta * kinetic.get() + potential.coefficient(integer_difference),
            );
        }
    }
    Ok(add_site_contributions(
        &mut overlap,
        &mut hamiltonian,
        dimension,
        compiled,
        sites,
    )?)
}

/// Assemble the first-variation SRA spinor problem from a typed spinor basis.
///
/// The interstitial has two Pauli components ordered as `spin * n_g + g`.
/// Its overlap and kinetic energy are lifted into equal-spin blocks. The
/// interstitial potential is `V0 I + B dot sigma`, so transverse magnetic
/// fields mix the two Pauli blocks as `Bx - i By` and `Bx + i By`. Additional
/// spin mixing and all large/small component physics inside muffin tins enters
/// through the typed site projection; there is no interstitial small-component
/// fallback.
pub fn assemble_sra_spinor_compiled(
    compiled: &SpinorCompiledBasis,
    geometry: &InterstitialGeometry,
    potential: &InterstitialPauliPotential,
    sites: &[SpinorSiteOperatorBlocks],
) -> Result<LapwEigenproblem, LapwError> {
    let plane_waves = &compiled.plane_waves;
    validate_plane_wave_norms(plane_waves)?;
    if let Some(first) = plane_waves.first() {
        if plane_waves.iter().any(|wave| wave.k != first.k) {
            return Err(LapwError::MixedKPoints);
        }
    }
    validate_spinor_compiled_geometry(compiled, geometry)?;
    if sites.len() != geometry.spheres().len() {
        return Err(OperatorError::SiteCount {
            expected: geometry.spheres().len(),
            actual: sites.len(),
        }
        .into());
    }

    let layout = &compiled.layout;
    let dimension = layout.dimension();
    let mut overlap = vec![Complex64::default(); dimension * dimension];
    let mut hamiltonian = vec![Complex64::default(); dimension * dimension];
    for i in 0..plane_waves.len() {
        for j in i..plane_waves.len() {
            let cartesian_difference = std::array::from_fn(|axis| {
                InverseBohr(
                    plane_waves[i].g.cartesian[axis].get() - plane_waves[j].g.cartesian[axis].get(),
                )
            });
            let integer_difference = std::array::from_fn(|axis| {
                plane_waves[i].g.index[axis] - plane_waves[j].g.index[axis]
            });
            let theta = geometry
                .coefficient(cartesian_difference)
                .map_err(|error| LapwError::StepFunction(error.to_string()))?;
            let kinetic = INTERSTITIAL_KINETIC.prefactor(plane_waves[i].q, plane_waves[j].q);
            for spin in 0..2 {
                let left = layout
                    .plane_wave_index(spin, i)
                    .expect("loop bounds and Pauli spin are valid");
                let right = layout
                    .plane_wave_index(spin, j)
                    .expect("loop bounds and Pauli spin are valid");
                set_hermitian(&mut overlap, dimension, left, right, theta);
                set_hermitian(
                    &mut hamiltonian,
                    dimension,
                    left,
                    right,
                    theta * kinetic.get()
                        + potential.v0.coefficient(integer_difference)
                        + if spin == 0 {
                            potential.bz.coefficient(integer_difference)
                        } else {
                            -potential.bz.coefficient(integer_difference)
                        },
                );
            }
        }
    }
    for i in 0..plane_waves.len() {
        for j in 0..plane_waves.len() {
            let integer_difference = std::array::from_fn(|axis| {
                plane_waves[i].g.index[axis] - plane_waves[j].g.index[axis]
            });
            let up = layout
                .plane_wave_index(0, i)
                .expect("loop bounds and Pauli spin are valid");
            let down = layout
                .plane_wave_index(1, j)
                .expect("loop bounds and Pauli spin are valid");
            let transverse = potential.bx.coefficient(integer_difference)
                - Complex64::i() * potential.by.coefficient(integer_difference);
            set_hermitian(&mut hamiltonian, dimension, up, down, transverse);
        }
    }
    Ok(add_spinor_site_contributions(
        &mut overlap,
        &mut hamiltonian,
        dimension,
        compiled,
        sites,
    )?)
}

/// SPEX spherical radial Hamiltonian in the `(u, udot)` basis.
///
/// This is the symmetrized identity from `hamilton.f:396-425`.  In the
/// `(u, udot)` basis it gives `H_uu = E O_uu`,
/// `H_u,udot = E O_u,udot + O_uu / 2`, and `H_udot,udot = E O_udot,udot`.
pub fn spex_spherical_radial_hamiltonian(
    linearization_energy: Hartree,
    overlap: RadialOverlapBlock,
) -> DenseHermitianMatrix {
    let energy = linearization_energy.get();
    DenseHermitianMatrix::from_upper_triangle(2, Axis::SiteCoordinate, |row, column| {
        let value = match (row, column) {
            (0, 0) => energy * overlap.uu,
            (0, 1) => energy * overlap.u_udot + 0.5 * overlap.uu,
            (1, 1) => energy * overlap.udot_udot,
            _ => unreachable!("only the upper triangle is requested"),
        };
        Complex64::new(value, 0.0)
    })
    .expect("SPEX radial block is Hermitian")
}

/// Assemble and solve two independent collinear spin channels.
///
/// The compiled basis is shared. Potentials and local operator blocks are
/// channel-specific, and no doubled spin matrix is formed.
pub fn solve_collinear_eigenproblems(
    envelope: &PlaneWaveEnvelope,
    geometry: &InterstitialGeometry,
    recipe_sites: &[LapwSiteInput],
    potentials: Collinear<&InterstitialPotential>,
    sites: Collinear<&[SiteOperatorBlocks]>,
    relative_overlap_threshold: f64,
) -> Result<Collinear<SolvedLapwEigenproblem>, LapwError> {
    let spec = lapw(envelope.clone(), geometry.cell_volume(), recipe_sites);
    let compiled = compile(&spec)?;
    let solve_channel = |potential: &InterstitialPotential,
                         site_blocks: &[SiteOperatorBlocks]|
     -> Result<SolvedLapwEigenproblem, LapwError> {
        let eigenproblem = assemble_compiled(&compiled, geometry, potential, site_blocks)?;
        let solution = solve_generalized_hermitian(
            &eigenproblem.hamiltonian,
            &eigenproblem.overlap,
            relative_overlap_threshold,
        )?;
        Ok(SolvedLapwEigenproblem {
            eigenproblem,
            solution,
        })
    };
    Ok(Collinear {
        up: solve_channel(potentials.up, sites.up)?,
        down: solve_channel(potentials.down, sites.down)?,
    })
}

/// Default SPEX reference-comparison tolerance: one meV in Hartree.
pub const DEFAULT_BAND_TOLERANCE_HARTREE: Hartree = Hartree(1.0e-3 / 27.211_386_245_988);

/// One reference energy, indexed explicitly by k point and band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandReference {
    pub k_index: usize,
    pub band_index: usize,
    pub energy: Hartree,
}

/// Numerical comparison for one `(k, band)` pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandComparison {
    pub k_index: usize,
    pub band_index: usize,
    pub calculated: Hartree,
    pub reference: Hartree,
    pub difference: Hartree,
    pub within_tolerance: bool,
}

/// Aggregate of explicitly supplied reference points; no material fixture is
/// embedded in this crate.
#[derive(Clone, Debug, PartialEq)]
pub struct BandReferenceReport {
    pub tolerance: Hartree,
    pub comparisons: Vec<BandComparison>,
    pub maximum_absolute_difference: Hartree,
    pub all_within_tolerance: bool,
}

/// Compare calculated Hartree energies with selected `(k, band)` references.
/// Passing `None` selects [`DEFAULT_BAND_TOLERANCE_HARTREE`].
pub fn compare_band_references(
    calculated: &[Vec<Hartree>],
    references: &[BandReference],
    tolerance: Option<Hartree>,
) -> Result<BandReferenceReport, LapwError> {
    if references.is_empty() {
        return Err(LapwError::MissingReferenceData);
    }
    let tolerance = tolerance.unwrap_or(DEFAULT_BAND_TOLERANCE_HARTREE);
    if !tolerance.get().is_finite() || tolerance.get() < 0.0 {
        return Err(LapwError::InvalidReferenceTolerance(tolerance.get()));
    }
    let mut comparisons = Vec::with_capacity(references.len());
    let mut maximum_absolute_difference: f64 = 0.0;
    for reference in references {
        let calculated_energy = calculated
            .get(reference.k_index)
            .and_then(|bands| bands.get(reference.band_index))
            .copied()
            .ok_or(LapwError::MissingBand {
                k_index: reference.k_index,
                band_index: reference.band_index,
            })?;
        let difference = calculated_energy - reference.energy;
        maximum_absolute_difference = maximum_absolute_difference.max(difference.get().abs());
        comparisons.push(BandComparison {
            k_index: reference.k_index,
            band_index: reference.band_index,
            calculated: calculated_energy,
            reference: reference.energy,
            difference,
            within_tolerance: difference.get().abs() <= tolerance.get(),
        });
    }
    let all_within_tolerance = comparisons
        .iter()
        .all(|comparison| comparison.within_tolerance);
    Ok(BandReferenceReport {
        tolerance,
        comparisons,
        maximum_absolute_difference: Hartree(maximum_absolute_difference),
        all_within_tolerance,
    })
}

fn validate_compiled_geometry(
    compiled: &CompiledBasis,
    geometry: &InterstitialGeometry,
) -> Result<(), LapwError> {
    if compiled.site_geometry.len() != compiled.site_count() {
        return Err(LapwError::CompiledSiteGeometryCount {
            expected: compiled.site_count(),
            actual: compiled.site_geometry.len(),
        });
    }
    if compiled.site_geometry.len() != geometry.spheres().len() {
        return Err(LapwError::SiteGeometryCount {
            compiled: compiled.site_geometry.len(),
            geometry: geometry.spheres().len(),
        });
    }
    for (index, (site, sphere)) in compiled
        .site_geometry
        .iter()
        .zip(geometry.spheres())
        .enumerate()
    {
        if site.position != sphere.center {
            return Err(LapwError::SitePositionMismatch {
                site: index,
                compiled: site.position,
                geometry: sphere.center,
            });
        }
        if site.radius != sphere.radius {
            return Err(LapwError::SiteRadiusMismatch {
                site: index,
                compiled: site.radius,
                geometry: sphere.radius,
            });
        }
    }
    Ok(())
}

fn validate_spinor_compiled_geometry(
    compiled: &SpinorCompiledBasis,
    geometry: &InterstitialGeometry,
) -> Result<(), LapwError> {
    if compiled.site_geometry.len() != compiled.site_count() {
        return Err(LapwError::CompiledSiteGeometryCount {
            expected: compiled.site_count(),
            actual: compiled.site_geometry.len(),
        });
    }
    if compiled.site_geometry.len() != geometry.spheres().len() {
        return Err(LapwError::SiteGeometryCount {
            compiled: compiled.site_geometry.len(),
            geometry: geometry.spheres().len(),
        });
    }
    for (index, (site, sphere)) in compiled
        .site_geometry
        .iter()
        .zip(geometry.spheres())
        .enumerate()
    {
        if site.position != sphere.center {
            return Err(LapwError::SitePositionMismatch {
                site: index,
                compiled: site.position,
                geometry: sphere.center,
            });
        }
        if site.radius != sphere.radius {
            return Err(LapwError::SiteRadiusMismatch {
                site: index,
                compiled: site.radius,
                geometry: sphere.radius,
            });
        }
    }
    Ok(())
}

fn validate_plane_wave_norms(plane_waves: &[PlaneWave]) -> Result<(), LapwError> {
    if let Some(q_norm) = plane_waves
        .iter()
        .map(|wave| wave.q_norm.get())
        .find(|q_norm| !q_norm.is_finite() || *q_norm < 0.0)
    {
        return Err(LapwError::InvalidWaveVector(q_norm));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::{Bohr, Kappa, ReciprocalLattice, Sphere, VolumeBohr3};
    use muffintin_radial::BoundaryData;
    use num_complex::Complex64;

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

    fn envelope() -> PlaneWaveEnvelope {
        PlaneWaveEnvelope::new(waves())
    }

    fn site_h(
        dimension: usize,
        element: impl FnMut(usize, usize) -> Complex64,
    ) -> DenseHermitianMatrix {
        DenseHermitianMatrix::from_upper_triangle(dimension, Axis::SiteCoordinate, element).unwrap()
    }

    fn compiled_with(
        plane_waves: Vec<PlaneWave>,
        local_orbitals: Vec<LocalOrbitalLayout>,
        site_augmentations: Vec<Vec<PlaneWaveAugmentation>>,
        site_geometry: Vec<ApwSiteGeometry>,
    ) -> CompiledBasis {
        CompiledBasis {
            layout: LapwBasisLayout::new(plane_waves.len(), local_orbitals),
            plane_waves,
            site_augmentations,
            site_geometry,
            provenance: Provenance::default(),
        }
    }

    #[test]
    fn overlap_is_hermitian() {
        let waves = waves();
        let position = [Bohr(0.3), Bohr(-0.4), Bohr(0.2)];
        let radius = Bohr(0.8);
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: position,
                radius,
            }],
        )
        .unwrap();
        let apw_boundary = ApwBoundaryBasis {
            u: boundary(0.8, -0.1),
            udot: boundary(0.2, 1.1),
        };
        let boundaries = vec![apw_boundary, apw_boundary, apw_boundary];
        let first_augmentation = {
            let matches = (0..=2)
                .map(|l| match_apw_boundary(l, waves[0].q_norm, radius, apw_boundary).unwrap())
                .collect::<Vec<_>>();
            augmentation_coefficients(&waves[0], position, VolumeBohr3(100.0), &matches).unwrap()
        };
        let radial = RadialOverlapBlock {
            uu: 1.0,
            u_udot: 0.04,
            udot_udot: 0.7,
        };
        let local_dimension = 2 * first_augmentation.coefficients.len();
        let overlap_block = site_h(local_dimension, |row, column| {
            if row / 2 != column / 2 {
                return Complex64::default();
            }
            Complex64::new(
                match (row % 2, column % 2) {
                    (0, 0) => radial.uu,
                    (0, 1) => radial.u_udot,
                    (1, 1) => radial.udot_udot,
                    _ => unreachable!(),
                },
                0.0,
            )
        });
        let overlap = assemble_eigenproblem(
            &PlaneWaveEnvelope::new(waves),
            &geometry,
            &InterstitialPotential::default(),
            &[LapwSiteInput {
                position,
                radius,
                boundaries,
                local_orbitals: LocalOrbitalLayout::default(),
            }],
            &[SiteOperatorBlocks {
                overlap: overlap_block,
                hamiltonian: site_h(local_dimension, |_, _| Complex64::default()),
            }],
        )
        .unwrap()
        .overlap;
        for i in 0..overlap.dimension() {
            for j in 0..overlap.dimension() {
                assert!((overlap.at(i, j) - overlap.at(j, i).conj()).norm() < 2.0e-14);
            }
        }
    }

    #[test]
    fn empty_sphere_geometry_is_identity() {
        let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), vec![]).unwrap();
        let overlap = assemble_eigenproblem(
            &envelope(),
            &geometry,
            &InterstitialPotential::default(),
            &[],
            &[],
        )
        .unwrap()
        .overlap;
        for i in 0..overlap.dimension() {
            for j in 0..overlap.dimension() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((overlap.at(i, j) - expected).norm() < 2.0e-14);
            }
        }
    }

    #[test]
    fn empty_lattice_hamiltonian_is_free_electron() {
        let waves = waves();
        let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), vec![]).unwrap();
        let potential = InterstitialPotential::default();
        let problem = assemble_eigenproblem(
            &PlaneWaveEnvelope::new(waves.clone()),
            &geometry,
            &potential,
            &[],
            &[],
        )
        .unwrap();
        let overlap = problem.overlap;
        let hamiltonian = problem.hamiltonian;
        for (i, wave) in waves.iter().enumerate() {
            for j in 0..hamiltonian.dimension() {
                let expected = if i == j {
                    0.5 * wave.q_norm.squared()
                } else {
                    0.0
                };
                assert!((hamiltonian.at(i, j) - expected).norm() <= 1.0e-12);
            }
        }
        let solution = solve_generalized_hermitian(&hamiltonian, &overlap, 0.0).unwrap();
        let mut expected = waves
            .iter()
            .map(|wave| 0.5 * wave.q_norm.squared())
            .collect::<Vec<_>>();
        expected.sort_by(f64::total_cmp);
        for (actual, expected) in solution.eigenvalues.iter().zip(expected) {
            assert!((actual.get() - expected).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn assembled_hamiltonian_is_hermitian() {
        let waves = waves();
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: [Bohr(0.0); 3],
                radius: Bohr(0.8),
            }],
        )
        .unwrap();
        let augmentations = (0..waves.len())
            .map(|index| PlaneWaveAugmentation {
                coefficients: vec![[
                    Complex64::new(0.01 * index as f64, -0.02),
                    Complex64::new(-0.03, 0.015 * index as f64),
                ]],
            })
            .collect();
        let block = site_h(2, |row, column| match (row, column) {
            (0, 0) => Complex64::new(1.2, 0.0),
            (0, 1) => Complex64::new(0.1, -0.2),
            (1, 1) => Complex64::new(0.8, 0.0),
            _ => unreachable!(),
        });
        let compiled = compiled_with(
            waves,
            vec![LocalOrbitalLayout::default()],
            vec![augmentations],
            vec![ApwSiteGeometry {
                position: [Bohr(0.0); 3],
                radius: Bohr(0.8),
            }],
        );
        let hamiltonian = assemble_compiled(
            &compiled,
            &geometry,
            &InterstitialPotential::default(),
            &[SiteOperatorBlocks {
                overlap: site_h(2, |row, column| {
                    Complex64::new(if row == column { 1.0 } else { 0.0 }, 0.0)
                }),
                hamiltonian: block,
            }],
        )
        .unwrap()
        .hamiltonian;
        for i in 0..hamiltonian.dimension() {
            for j in 0..hamiltonian.dimension() {
                assert!((hamiltonian.at(i, j) - hamiltonian.at(j, i).conj()).norm() < 1.0e-14);
            }
        }
    }

    #[test]
    fn full_local_blocks_generate_complex_apw_lo_and_lo_lo_elements() {
        let wave = waves()[0];
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: [Bohr(0.2), Bohr(-0.1), Bohr(0.3)],
                radius: Bohr(0.7),
            }],
        )
        .unwrap();
        let a = Complex64::new(0.3, -0.2);
        let b = Complex64::new(-0.1, 0.4);
        let overlap = site_h(3, |row, column| match (row, column) {
            (0, 0) => Complex64::new(1.1, 0.0),
            (0, 1) => Complex64::new(0.2, 0.1),
            (0, 2) => Complex64::new(-0.3, 0.25),
            (1, 1) => Complex64::new(0.9, 0.0),
            (1, 2) => Complex64::new(0.15, -0.35),
            (2, 2) => Complex64::new(1.4, 0.0),
            _ => unreachable!(),
        });
        let hamiltonian = site_h(3, |row, column| match (row, column) {
            (0, 0) => Complex64::new(0.8, 0.0),
            (0, 1) => Complex64::new(-0.2, 0.05),
            (0, 2) => Complex64::new(0.4, -0.3),
            (1, 1) => Complex64::new(1.2, 0.0),
            (1, 2) => Complex64::new(-0.1, 0.2),
            (2, 2) => Complex64::new(2.3, 0.0),
            _ => unreachable!(),
        });
        let local = SiteOperatorBlocks {
            overlap,
            hamiltonian,
        };
        let compiled = compiled_with(
            vec![wave],
            vec![LocalOrbitalLayout::new(vec![1])],
            vec![vec![PlaneWaveAugmentation {
                coefficients: vec![[a, b]],
            }]],
            vec![ApwSiteGeometry {
                position: [Bohr(0.2), Bohr(-0.1), Bohr(0.3)],
                radius: Bohr(0.7),
            }],
        );
        let problem = assemble_compiled(
            &compiled,
            &geometry,
            &InterstitialPotential::default(),
            std::slice::from_ref(&local),
        )
        .unwrap();
        let expected_s_apw_lo =
            a.conj() * local.overlap.at(0, 2) + b.conj() * local.overlap.at(1, 2);
        let expected_h_apw_lo =
            a.conj() * local.hamiltonian.at(0, 2) + b.conj() * local.hamiltonian.at(1, 2);
        let expected_s_apw_apw = a.conj()
            * (local.overlap.at(0, 0) * a + local.overlap.at(0, 1) * b)
            + b.conj() * (local.overlap.at(1, 0) * a + local.overlap.at(1, 1) * b);
        let expected_h_apw_apw = a.conj()
            * (local.hamiltonian.at(0, 0) * a + local.hamiltonian.at(0, 1) * b)
            + b.conj() * (local.hamiltonian.at(1, 0) * a + local.hamiltonian.at(1, 1) * b);
        let theta = geometry
            .coefficient(std::array::from_fn(|_| InverseBohr(0.0)))
            .unwrap();
        assert!((problem.overlap.at(0, 0) - (theta + expected_s_apw_apw)).norm() < 1.0e-14);
        assert!(
            (problem.hamiltonian.at(0, 0)
                - (0.5 * wave.q_norm.squared() * theta + expected_h_apw_apw))
                .norm()
                < 1.0e-14
        );
        assert!((problem.overlap.at(0, 1) - expected_s_apw_lo).norm() < 1.0e-14);
        assert!((problem.hamiltonian.at(0, 1) - expected_h_apw_lo).norm() < 1.0e-14);
        assert_eq!(problem.overlap.at(1, 1), local.overlap.at(2, 2));
        assert_eq!(problem.hamiltonian.at(1, 1), local.hamiltonian.at(2, 2));
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (problem.overlap.at(i, j) - problem.overlap.at(j, i).conj()).norm() < 1.0e-14
                );
                assert!(
                    (problem.hamiltonian.at(i, j) - problem.hamiltonian.at(j, i).conj()).norm()
                        < 1.0e-14
                );
            }
        }
    }

    #[test]
    fn no_lo_site_projection_reduces_to_apw_congruence() {
        let wave = waves()[0];
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: [Bohr(0.2), Bohr(-0.1), Bohr(0.3)],
                radius: Bohr(0.7),
            }],
        )
        .unwrap();
        let a = Complex64::new(0.3, -0.2);
        let b = Complex64::new(-0.1, 0.4);
        let overlap = site_h(2, |row, column| match (row, column) {
            (0, 0) => Complex64::new(1.1, 0.0),
            (0, 1) => Complex64::new(0.2, 0.1),
            (1, 1) => Complex64::new(0.9, 0.0),
            _ => unreachable!(),
        });
        let hamiltonian = site_h(2, |row, column| match (row, column) {
            (0, 0) => Complex64::new(0.8, 0.0),
            (0, 1) => Complex64::new(-0.2, 0.05),
            (1, 1) => Complex64::new(1.2, 0.0),
            _ => unreachable!(),
        });
        let local = SiteOperatorBlocks {
            overlap,
            hamiltonian,
        };
        let compiled = compiled_with(
            vec![wave],
            vec![LocalOrbitalLayout::default()],
            vec![vec![PlaneWaveAugmentation {
                coefficients: vec![[a, b]],
            }]],
            vec![ApwSiteGeometry {
                position: [Bohr(0.2), Bohr(-0.1), Bohr(0.3)],
                radius: Bohr(0.7),
            }],
        );
        let problem = assemble_compiled(
            &compiled,
            &geometry,
            &InterstitialPotential::default(),
            std::slice::from_ref(&local),
        )
        .unwrap();
        let expected_s = a.conj() * (local.overlap.at(0, 0) * a + local.overlap.at(0, 1) * b)
            + b.conj() * (local.overlap.at(1, 0) * a + local.overlap.at(1, 1) * b);
        let expected_h = a.conj()
            * (local.hamiltonian.at(0, 0) * a + local.hamiltonian.at(0, 1) * b)
            + b.conj() * (local.hamiltonian.at(1, 0) * a + local.hamiltonian.at(1, 1) * b);
        let theta = geometry
            .coefficient(std::array::from_fn(|_| InverseBohr(0.0)))
            .unwrap();
        assert!((problem.overlap.at(0, 0) - (theta + expected_s)).norm() < 1.0e-14);
        assert!(
            (problem.hamiltonian.at(0, 0) - (0.5 * wave.q_norm.squared() * theta + expected_h))
                .norm()
                < 1.0e-14
        );
        assert_eq!(problem.overlap.dimension(), 1);
    }

    #[test]
    fn collinear_driver_matches_two_independent_channel_solves() {
        let waves = waves();
        let envelope = PlaneWaveEnvelope::new(waves.clone());
        let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), vec![]).unwrap();
        let up = InterstitialPotential::new([([0; 3], Complex64::new(0.17, 0.0))]).unwrap();
        let down = InterstitialPotential::new([([0; 3], Complex64::new(-0.09, 0.0))]).unwrap();
        let spins = solve_collinear_eigenproblems(
            &envelope,
            &geometry,
            &[],
            Collinear::new(&up, &down),
            Collinear::new(&[][..], &[][..]),
            0.0,
        )
        .unwrap();
        let up_problem = assemble_eigenproblem(&envelope, &geometry, &up, &[], &[]).unwrap();
        let down_problem = assemble_eigenproblem(&envelope, &geometry, &down, &[], &[]).unwrap();
        let up_solution =
            solve_generalized_hermitian(&up_problem.hamiltonian, &up_problem.overlap, 0.0).unwrap();
        let down_solution =
            solve_generalized_hermitian(&down_problem.hamiltonian, &down_problem.overlap, 0.0)
                .unwrap();
        assert_eq!(spins.up.solution.eigenvalues, up_solution.eigenvalues);
        assert_eq!(spins.down.solution.eigenvalues, down_solution.eigenvalues);
        assert_ne!(
            spins.up.solution.eigenvalues,
            spins.down.solution.eigenvalues
        );
        assert_eq!(spins.up.eigenproblem.overlap.dimension(), waves.len());
        assert_eq!(spins.down.eigenproblem.overlap.dimension(), waves.len());
    }

    fn one_site_compiled(position: [Bohr; 3], radius: Bohr) -> CompiledBasis {
        compile(&BasisSpec {
            blocks: vec![BasisBlock::PlaneWaveEnvelope {
                envelope: PlaneWaveEnvelope::new(waves()),
                sites: vec![ApwSiteAugmentation {
                    position,
                    radius,
                    boundaries: vec![ApwBoundaryBasis {
                        u: boundary(0.8, -0.1),
                        udot: boundary(0.2, 1.1),
                    }],
                }],
            }],
            cell_volume: VolumeBohr3(100.0),
            provenance: Provenance::default(),
        })
        .unwrap()
    }

    #[test]
    fn assemble_compiled_rejects_site_position_mismatch() {
        let position = [Bohr(0.3), Bohr(-0.4), Bohr(0.2)];
        let radius = Bohr(0.8);
        let compiled = one_site_compiled(position, radius);
        let geometry_position = [Bohr(0.1), Bohr(-0.4), Bohr(0.2)];
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: geometry_position,
                radius,
            }],
        )
        .unwrap();
        assert_eq!(
            assemble_compiled(&compiled, &geometry, &InterstitialPotential::default(), &[]),
            Err(LapwError::SitePositionMismatch {
                site: 0,
                compiled: position,
                geometry: geometry_position,
            })
        );
    }

    #[test]
    fn assemble_compiled_rejects_site_radius_mismatch() {
        let position = [Bohr(0.3), Bohr(-0.4), Bohr(0.2)];
        let compiled_radius = Bohr(0.8);
        let geometry_radius = Bohr(0.5);
        let compiled = one_site_compiled(position, compiled_radius);
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: position,
                radius: geometry_radius,
            }],
        )
        .unwrap();
        assert_eq!(
            assemble_compiled(&compiled, &geometry, &InterstitialPotential::default(), &[]),
            Err(LapwError::SiteRadiusMismatch {
                site: 0,
                compiled: compiled_radius,
                geometry: geometry_radius,
            })
        );
    }

    #[test]
    fn spherical_radial_helper_matches_spex_identity() {
        let overlap = RadialOverlapBlock {
            uu: 1.2,
            u_udot: 0.3,
            udot_udot: 0.8,
        };
        let h = spex_spherical_radial_hamiltonian(Hartree(0.5), overlap);
        assert_eq!(h.at(0, 0), Complex64::new(0.6, 0.0));
        assert_eq!(h.at(0, 1), Complex64::new(0.75, 0.0));
        assert_eq!(h.at(1, 1), Complex64::new(0.4, 0.0));
    }

    #[test]
    fn band_reference_report_uses_one_mev_default_and_checks_indices() {
        let tolerance = DEFAULT_BAND_TOLERANCE_HARTREE.get();
        let calculated = vec![vec![Hartree(0.2), Hartree(0.3)]];
        let references = [
            BandReference {
                k_index: 0,
                band_index: 0,
                energy: Hartree(0.2 + 0.5 * tolerance),
            },
            BandReference {
                k_index: 0,
                band_index: 1,
                energy: Hartree(0.3 + 2.0 * tolerance),
            },
        ];
        let report = compare_band_references(&calculated, &references, None).unwrap();
        assert!(report.comparisons[0].within_tolerance);
        assert!(!report.comparisons[1].within_tolerance);
        assert!(!report.all_within_tolerance);
        assert!((report.maximum_absolute_difference.get() - 2.0 * tolerance).abs() < 1.0e-15);

        let missing = [BandReference {
            k_index: 1,
            band_index: 0,
            energy: Hartree(0.0),
        }];
        assert!(matches!(
            compare_band_references(&calculated, &missing, None),
            Err(LapwError::MissingBand {
                k_index: 1,
                band_index: 0
            })
        ));
        assert_eq!(
            compare_band_references(&calculated, &[], None),
            Err(LapwError::MissingReferenceData)
        );
    }

    #[test]
    fn sra_surface_term_is_hermitian_and_vanishes_for_continuous_traces() {
        let trace = SraSurfaceTrace {
            value: Complex64::new(0.8, -0.2),
            outward_derivative: Complex64::new(-0.3, 0.4),
        };
        assert_eq!(
            sra_schlosser_marcus_surface_correction(Bohr(1.7), trace, trace, trace, trace,)
                .unwrap(),
            Complex64::default()
        );

        let li = SraSurfaceTrace {
            value: Complex64::new(0.3, 0.1),
            outward_derivative: Complex64::new(-0.2, 0.4),
        };
        let lt = SraSurfaceTrace {
            value: Complex64::new(0.7, -0.3),
            outward_derivative: Complex64::new(0.1, 0.2),
        };
        let ri = SraSurfaceTrace {
            value: Complex64::new(-0.4, 0.2),
            outward_derivative: Complex64::new(0.6, -0.1),
        };
        let rt = SraSurfaceTrace {
            value: Complex64::new(0.2, 0.5),
            outward_derivative: Complex64::new(-0.3, -0.2),
        };
        let left_right =
            sra_schlosser_marcus_surface_correction(Bohr(2.0), li, lt, ri, rt).unwrap();
        let right_left =
            sra_schlosser_marcus_surface_correction(Bohr(2.0), ri, rt, li, lt).unwrap();
        assert!((left_right - right_left.conj()).norm() < 2.0e-15);
    }

    fn empty_spinor_compiled(plane_waves: Vec<PlaneWave>) -> SpinorCompiledBasis {
        SpinorCompiledBasis {
            layout: SpinorBasisLayout::new(plane_waves.len(), Vec::new()),
            plane_waves,
            site_augmentations: Vec::new(),
            site_geometry: Vec::new(),
            provenance: Provenance::default(),
        }
    }

    #[test]
    fn sra_zero_pauli_field_reproduces_scalar_interstitial_in_both_spin_blocks() {
        let waves = waves();
        let n_g = waves.len();
        let scalar_compiled = compiled_with(waves.clone(), Vec::new(), Vec::new(), Vec::new());
        let compiled = empty_spinor_compiled(waves);
        let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), Vec::new()).unwrap();
        let scalar = assemble_compiled(
            &scalar_compiled,
            &geometry,
            &InterstitialPotential::default(),
            &[],
        )
        .unwrap();
        let problem = assemble_sra_spinor_compiled(
            &compiled,
            &geometry,
            &InterstitialPauliPotential::default(),
            &[],
        )
        .unwrap();
        for i in 0..n_g {
            for j in 0..n_g {
                let up_i = compiled.layout.plane_wave_index(0, i).unwrap();
                let up_j = compiled.layout.plane_wave_index(0, j).unwrap();
                let down_i = compiled.layout.plane_wave_index(1, i).unwrap();
                let down_j = compiled.layout.plane_wave_index(1, j).unwrap();
                assert_eq!(problem.overlap.at(up_i, up_j), scalar.overlap.at(i, j));
                assert_eq!(problem.overlap.at(down_i, down_j), scalar.overlap.at(i, j));
                assert_eq!(problem.overlap.at(up_i, down_j), Complex64::default());
                assert_eq!(problem.hamiltonian.at(up_i, down_j), Complex64::default());
                assert_eq!(
                    problem.hamiltonian.at(up_i, up_j),
                    scalar.hamiltonian.at(i, j)
                );
                assert_eq!(
                    problem.hamiltonian.at(down_i, down_j),
                    scalar.hamiltonian.at(i, j)
                );
            }
        }
    }

    #[test]
    fn sra_bz_reduces_to_independent_collinear_channels() {
        let waves = waves();
        let n_g = waves.len();
        let scalar_compiled = compiled_with(waves.clone(), Vec::new(), Vec::new(), Vec::new());
        let compiled = empty_spinor_compiled(waves);
        let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), Vec::new()).unwrap();
        let v0 = InterstitialPotential::new([
            ([0; 3], Complex64::new(0.12, 0.0)),
            ([1, 0, 0], Complex64::new(0.03, 0.02)),
        ])
        .unwrap();
        let bz = InterstitialPotential::new([
            ([0; 3], Complex64::new(-0.07, 0.0)),
            ([1, 0, 0], Complex64::new(0.01, -0.015)),
        ])
        .unwrap();
        let up = InterstitialPotential::new([
            ([0; 3], Complex64::new(0.05, 0.0)),
            ([1, 0, 0], Complex64::new(0.04, 0.005)),
        ])
        .unwrap();
        let down = InterstitialPotential::new([
            ([0; 3], Complex64::new(0.19, 0.0)),
            ([1, 0, 0], Complex64::new(0.02, 0.035)),
        ])
        .unwrap();
        let pauli = InterstitialPauliPotential::new(
            v0,
            InterstitialPotential::default(),
            InterstitialPotential::default(),
            bz,
        );
        let problem = assemble_sra_spinor_compiled(&compiled, &geometry, &pauli, &[]).unwrap();
        let up_problem = assemble_compiled(&scalar_compiled, &geometry, &up, &[]).unwrap();
        let down_problem = assemble_compiled(&scalar_compiled, &geometry, &down, &[]).unwrap();

        for i in 0..n_g {
            for j in 0..n_g {
                let up_i = compiled.layout.plane_wave_index(0, i).unwrap();
                let up_j = compiled.layout.plane_wave_index(0, j).unwrap();
                let down_i = compiled.layout.plane_wave_index(1, i).unwrap();
                let down_j = compiled.layout.plane_wave_index(1, j).unwrap();
                assert!(
                    (problem.hamiltonian.at(up_i, up_j) - up_problem.hamiltonian.at(i, j)).norm()
                        < 2.0e-14
                );
                assert!(
                    (problem.hamiltonian.at(down_i, down_j) - down_problem.hamiltonian.at(i, j))
                        .norm()
                        < 2.0e-14
                );
                assert_eq!(problem.hamiltonian.at(up_i, down_j), Complex64::default());
            }
        }
    }

    #[test]
    fn sra_bx_by_mix_pauli_blocks_with_standard_signs() {
        let compiled = empty_spinor_compiled(vec![waves()[0]]);
        let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), Vec::new()).unwrap();
        let bx = InterstitialPotential::new([([0; 3], Complex64::new(0.2, 0.0))]).unwrap();
        let by = InterstitialPotential::new([([0; 3], Complex64::new(-0.3, 0.0))]).unwrap();
        let pauli = InterstitialPauliPotential::new(
            InterstitialPotential::default(),
            bx,
            by,
            InterstitialPotential::default(),
        );
        let problem = assemble_sra_spinor_compiled(&compiled, &geometry, &pauli, &[]).unwrap();
        let up = compiled.layout.plane_wave_index(0, 0).unwrap();
        let down = compiled.layout.plane_wave_index(1, 0).unwrap();
        assert_eq!(problem.hamiltonian.at(up, down), Complex64::new(0.2, 0.3));
        assert_eq!(problem.hamiltonian.at(down, up), Complex64::new(0.2, -0.3));
        assert_eq!(problem.overlap.at(up, down), Complex64::default());
    }

    #[test]
    fn sra_general_interstitial_pauli_potential_is_hermitian() {
        let compiled = empty_spinor_compiled(waves());
        let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), Vec::new()).unwrap();
        let field = |zero, one| {
            InterstitialPotential::new([([0; 3], Complex64::new(zero, 0.0)), ([1, 0, 0], one)])
                .unwrap()
        };
        let pauli = InterstitialPauliPotential::new(
            field(0.11, Complex64::new(0.02, -0.01)),
            field(-0.07, Complex64::new(0.03, 0.04)),
            field(0.05, Complex64::new(-0.02, 0.015)),
            field(-0.13, Complex64::new(0.01, -0.025)),
        );
        let problem = assemble_sra_spinor_compiled(&compiled, &geometry, &pauli, &[]).unwrap();

        for i in 0..problem.hamiltonian.dimension() {
            for j in 0..problem.hamiltonian.dimension() {
                assert!(
                    (problem.hamiltonian.at(i, j) - problem.hamiltonian.at(j, i).conj()).norm()
                        < 2.0e-14
                );
            }
        }
    }

    #[test]
    fn typed_spinor_site_projection_can_mix_pauli_blocks_and_remains_hermitian() {
        let wave = waves()[0];
        let position = [Bohr(0.0); 3];
        let radius = Bohr(0.8);
        let kappa = Kappa::new(-1).unwrap();
        let channels = kappa.channels().collect::<Vec<_>>();
        let zero = [Complex64::default(); 2];
        let up = [Complex64::new(0.4, -0.1), Complex64::new(0.1, 0.0)];
        let down = [Complex64::new(-0.2, 0.3), Complex64::new(0.0, 0.05)];
        let compiled = SpinorCompiledBasis {
            layout: SpinorBasisLayout::new(1, vec![SpinorSiteLayout::default()]),
            plane_waves: vec![wave],
            site_augmentations: vec![vec![SpinorPlaneWaveAugmentation {
                channels,
                coefficients: [vec![zero, up], vec![down, zero]],
            }]],
            site_geometry: vec![ApwSiteGeometry { position, radius }],
            provenance: Provenance::default(),
        };
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: position,
                radius,
            }],
        )
        .unwrap();
        let overlap = site_h(4, |row, column| {
            Complex64::new(if row == column { 0.2 } else { 0.0 }, 0.0)
        });
        let hamiltonian = site_h(4, |row, column| match (row, column) {
            (0, 0) | (2, 2) => Complex64::new(0.7, 0.0),
            (1, 1) | (3, 3) => Complex64::new(0.4, 0.0),
            (0, 2) => Complex64::new(0.1, 0.2),
            _ => Complex64::default(),
        });
        let site = SpinorSiteOperatorBlocks {
            channels: kappa.channels().collect(),
            overlap,
            hamiltonian,
        };
        let potential = InterstitialPauliPotential::default();
        let problem = assemble_sra_spinor_compiled(
            &compiled,
            &geometry,
            &potential,
            std::slice::from_ref(&site),
        )
        .unwrap();
        assert!(problem.hamiltonian.at(0, 1).norm() > 1.0e-6);
        for i in 0..problem.hamiltonian.dimension() {
            for j in 0..problem.hamiltonian.dimension() {
                assert!(
                    (problem.hamiltonian.at(i, j) - problem.hamiltonian.at(j, i).conj()).norm()
                        < 2.0e-14
                );
            }
        }
        let solution =
            solve_generalized_hermitian(&problem.hamiltonian, &problem.overlap, 1.0e-12).unwrap();
        assert_eq!(solution.retained_dimension, 2);
        assert!(
            solution
                .residuals
                .iter()
                .all(|residual| residual.absolute < 1.0e-11)
        );

        let wrong_channels = SpinorSiteOperatorBlocks {
            channels: Kappa::new(1).unwrap().channels().collect(),
            overlap: site.overlap.clone(),
            hamiltonian: site.hamiltonian.clone(),
        };
        assert_eq!(
            assemble_sra_spinor_compiled(&compiled, &geometry, &potential, &[wrong_channels],),
            Err(LapwError::Operator(OperatorError::SpinorChannelLayout {
                site: 0,
                plane_wave: 0,
            }))
        );
    }
}
