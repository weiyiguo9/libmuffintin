//! Linearized augmented-plane-wave matching and overlap assembly.
//!
//! Plane waves use `exp(+i (k+G) dot r) / sqrt(Omega)`.  The Rayleigh
//! coefficient and the site-translation phase are exposed separately so that
//! neither convention is hidden inside the real, radial boundary solve.
//! Site muffin-tin contributions are `einsum("ci,cd,dj->ij", [P^*, B, P])`
//! in `mt-tensor`. The filtered generalized solver keeps faer for Hermitian
//! EVD and uses einsum for $X$, $X^\dagger H X$, $C=XZ$, and residuals.

#![forbid(unsafe_code)]

use mt_core::{
    Bohr, GVector, Hartree, InterstitialGeometry, InverseBohr, Lm, VolumeBohr3,
    complex_spherical_harmonics, lm_count, spherical_bessel_j, spherical_bessel_j_derivative,
};
use mt_radial::BoundaryData;
use mt_tensor::{Axis, TensorError, einsum};

pub use mt_tensor::{ComplexTensor, DenseEigenvectors, DenseHermitianMatrix};
use num_complex::Complex64;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::ops::Range;
use thiserror::Error;

/// A normalized plane wave identified by `G`, with Cartesian `q = k + G`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneWave {
    /// Bloch vector in Cartesian bohr^-1.
    pub k: [InverseBohr; 3],
    /// Reciprocal-lattice vector.
    pub g: GVector,
    /// Cartesian `k + G` in bohr^-1.
    pub q: [InverseBohr; 3],
    /// Norm `|k + G|` in bohr^-1.
    pub q_norm: InverseBohr,
}

impl PlaneWave {
    /// Form `q = k + G` without changing reciprocal-lattice coordinates.
    pub fn new(k: [InverseBohr; 3], g: GVector) -> Self {
        let q = std::array::from_fn(|axis| InverseBohr(k[axis].get() + g.cartesian[axis].get()));
        let q_norm = InverseBohr(
            q.iter()
                .map(|component| component.get().powi(2))
                .sum::<f64>()
                .sqrt(),
        );
        Self { k, g, q, q_norm }
    }
}

/// The two primitive radial boundary columns `(u_l, du_l/dE)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApwBoundaryBasis {
    pub u: BoundaryData,
    pub udot: BoundaryData,
}

/// Coefficients and direct substitution residuals of one APW radial match.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApwMatch {
    pub l: u32,
    /// Coefficients multiplying `(u_l, du_l/dE)`.
    pub coefficients: [f64; 2],
    /// `A u(R) + B udot(R) - j_l(qR)`.
    pub value_residual: f64,
    /// `A u'(R) + B udot'(R) - q j_l'(qR)`.
    pub slope_residual: f64,
}

/// LAPW construction or overlap-assembly error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum LapwError {
    #[error("muffin-tin radius must be finite and positive, got {0}")]
    InvalidRadius(f64),
    #[error("APW boundary matrix for l={l} is singular (determinant {determinant})")]
    SingularBoundaryMatrix { l: u32, determinant: f64 },
    #[error("cell volume must be finite and positive, got {0}")]
    InvalidCellVolume(f64),
    #[error("site has {actual} plane-wave coefficient sets, expected {expected}")]
    PlaneWaveCount { expected: usize, actual: usize },
    #[error("geometry has {expected} spheres, but {actual} site blocks were supplied")]
    SiteCount { expected: usize, actual: usize },
    #[error("basis layout has {actual} plane waves, expected {expected}")]
    BasisPlaneWaveCount { expected: usize, actual: usize },
    #[error("basis layout has {actual} sites, expected {expected}")]
    BasisSiteCount { expected: usize, actual: usize },
    #[error("site plane wave {plane_wave} has {actual} lm channels, expected {expected}")]
    ChannelCount {
        plane_wave: usize,
        expected: usize,
        actual: usize,
    },
    #[error("APW matches must be ordered by l: expected {expected}, found {actual}")]
    MatchAngularMomentum { expected: u32, actual: u32 },
    #[error("all plane waves in an overlap matrix must share one k point")]
    MixedKPoints,
    #[error("interstitial coefficient failed: {0}")]
    StepFunction(String),
    #[error("matrix data has length {actual}, expected {expected}")]
    MatrixDataLength { expected: usize, actual: usize },
    #[error("site {site} {matrix} block has dimension {actual}, expected {expected}")]
    SiteBlockDimension {
        site: usize,
        matrix: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("matrix dimensions differ: H is {hamiltonian}, S is {overlap}")]
    MatrixDimensionMismatch { hamiltonian: usize, overlap: usize },
    #[error("matrix is not Hermitian at ({row}, {column})")]
    NonHermitianMatrix { row: usize, column: usize },
    #[error("{matrix} matrix has a non-finite value at ({row}, {column})")]
    NonFiniteMatrix {
        matrix: &'static str,
        row: usize,
        column: usize,
    },
    #[error("potential coefficient for G={g:?} conflicts with its Hermitian partner")]
    NonHermitianPotential { g: [i32; 3] },
    #[error("overlap eigenvalue threshold must be finite and nonnegative, got {0}")]
    InvalidOverlapThreshold(f64),
    #[error("overlap eigensystem retained no positive directions")]
    EmptyOverlapSubspace,
    #[error("overlap matrix is significantly indefinite (eigenvalue {eigenvalue})")]
    IndefiniteOverlap { eigenvalue: f64 },
    #[error("wave-vector norm must be finite and nonnegative, got {0}")]
    InvalidWaveVector(f64),
    #[error("dense self-adjoint eigendecomposition failed")]
    Eigensolver,
    #[error("reference comparison points to missing k={k_index}, band={band_index}")]
    MissingBand { k_index: usize, band_index: usize },
    #[error("reference tolerance must be finite and nonnegative, got {0}")]
    InvalidReferenceTolerance(f64),
    #[error("at least one band reference is required")]
    MissingReferenceData,
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

/// Solve the SPEX `2 x 2` boundary system for a fixed angular momentum.
///
/// The right-hand side is exactly `(j_l(qR), q j_l'(qR))`; the returned
/// residuals are obtained by substituting the computed coefficients back into
/// the un-inverted boundary matrix.
pub fn match_apw_boundary(
    l: u32,
    q: InverseBohr,
    radius: Bohr,
    basis: ApwBoundaryBasis,
) -> Result<ApwMatch, LapwError> {
    if !radius.get().is_finite() || radius.get() <= 0.0 {
        return Err(LapwError::InvalidRadius(radius.get()));
    }
    if !q.get().is_finite() || q.get() < 0.0 {
        return Err(LapwError::InvalidWaveVector(q.get()));
    }
    let determinant = basis.u.value * basis.udot.derivative - basis.udot.value * basis.u.derivative;
    let matrix_scale = (basis.u.value.abs() + basis.udot.value.abs())
        * (basis.u.derivative.abs() + basis.udot.derivative.abs());
    if !determinant.is_finite()
        || determinant.abs() <= 64.0 * f64::EPSILON * matrix_scale.max(f64::MIN_POSITIVE)
    {
        return Err(LapwError::SingularBoundaryMatrix { l, determinant });
    }

    let x = q.get() * radius.get();
    let target_value = spherical_bessel_j(l, x);
    let target_slope = q.get() * spherical_bessel_j_derivative(l, x);
    let a = (basis.udot.derivative * target_value - basis.udot.value * target_slope) / determinant;
    let b = (-basis.u.derivative * target_value + basis.u.value * target_slope) / determinant;

    Ok(ApwMatch {
        l,
        coefficients: [a, b],
        value_residual: a.mul_add(basis.u.value, b * basis.udot.value) - target_value,
        slope_residual: a.mul_add(basis.u.derivative, b * basis.udot.derivative) - target_slope,
    })
}

/// `4 pi i^l conj(Y_lm(qhat)) / sqrt(Omega)` for the `exp(+i q dot r)`
/// Rayleigh expansion.
///
/// This coefficient contains no site phase and no radial matching
/// coefficient.  At `q=0`, `mt-core`'s deterministic direction convention
/// leaves only the `l=m=0` channel nonzero.
pub fn rayleigh_coefficient(
    lm: Lm,
    q: [InverseBohr; 3],
    cell_volume: VolumeBohr3,
) -> Result<Complex64, LapwError> {
    if !cell_volume.get().is_finite() || cell_volume.get() <= 0.0 {
        return Err(LapwError::InvalidCellVolume(cell_volume.get()));
    }
    if let Some(component) = q
        .iter()
        .map(|component| component.get())
        .find(|x| !x.is_finite())
    {
        return Err(LapwError::InvalidWaveVector(component));
    }
    let direction = q.map(InverseBohr::get);
    let harmonic = complex_spherical_harmonics(lm.l, direction)[lm.index()].conj();
    Ok(i_pow(lm.l) * harmonic * (4.0 * PI / cell_volume.get().sqrt()))
}

/// Translation phase `exp(+i q dot R_a)` for expansion about site `R_a`.
pub fn site_translation_phase(q: [InverseBohr; 3], site: [Bohr; 3]) -> Complex64 {
    let phase = q
        .iter()
        .zip(site)
        .map(|(component, coordinate)| component.get() * coordinate.get())
        .sum();
    Complex64::from_polar(1.0, phase)
}

/// APW augmentation coefficients indexed by the public contiguous `lm` index.
/// Each channel stores coefficients multiplying `(u_l, du_l/dE)`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveAugmentation {
    pub coefficients: Vec<[Complex64; 2]>,
}

/// Build all site-centered APW coefficients through `l_max`.
///
/// `matches[l]` is the real boundary match for this plane wave and site type.
/// The output includes both the Rayleigh factor and `exp(+i q dot R_a)`.
pub fn augmentation_coefficients(
    plane_wave: &PlaneWave,
    site: [Bohr; 3],
    cell_volume: VolumeBohr3,
    matches: &[ApwMatch],
) -> Result<PlaneWaveAugmentation, LapwError> {
    let l_max = matches.len().saturating_sub(1) as u32;
    let phase = site_translation_phase(plane_wave.q, site);
    let mut coefficients = Vec::with_capacity(lm_count(l_max));
    for (l, matched) in matches.iter().enumerate() {
        let l = l as u32;
        if matched.l != l {
            return Err(LapwError::MatchAngularMomentum {
                expected: l,
                actual: matched.l,
            });
        }
        for m in -(l as i32)..=l as i32 {
            let angular = phase
                * rayleigh_coefficient(
                    Lm::new(l, m).expect("loop bounds validate m"),
                    plane_wave.q,
                    cell_volume,
                )?;
            coefficients.push([
                angular * matched.coefficients[0],
                angular * matched.coefficients[1],
            ]);
        }
    }
    Ok(PlaneWaveAugmentation { coefficients })
}

/// Counts and ordering of one site's local orbitals.
///
/// Orbitals are contiguous in `(l, m, n)` order: increasing `l`, then
/// `m = -l..l`, then the local-orbital number `n` for that `l`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LocalOrbitalLayout {
    counts_by_l: Vec<usize>,
}

impl LocalOrbitalLayout {
    pub fn new(counts_by_l: Vec<usize>) -> Self {
        Self { counts_by_l }
    }

    pub fn counts_by_l(&self) -> &[usize] {
        &self.counts_by_l
    }

    pub fn len(&self) -> usize {
        self.counts_by_l
            .iter()
            .enumerate()
            .map(|(l, count)| (2 * l + 1) * count)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Site-local LO index in the documented `(l, m, n)` order.
    pub fn index(&self, l: u32, m: i32, n: usize) -> Option<usize> {
        let count = *self.counts_by_l.get(l as usize)?;
        if m < -(l as i32) || m > l as i32 || n >= count {
            return None;
        }
        let preceding = self
            .counts_by_l
            .iter()
            .enumerate()
            .take(l as usize)
            .map(|(previous_l, count)| (2 * previous_l + 1) * count)
            .sum::<usize>();
        Some(preceding + (m + l as i32) as usize * count + n)
    }
}

/// Global LAPW basis order: all plane waves, followed by each site's LOs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LapwBasisLayout {
    plane_wave_count: usize,
    local_orbitals: Vec<LocalOrbitalLayout>,
}

impl LapwBasisLayout {
    pub fn new(plane_wave_count: usize, local_orbitals: Vec<LocalOrbitalLayout>) -> Self {
        Self {
            plane_wave_count,
            local_orbitals,
        }
    }

    pub const fn plane_wave_count(&self) -> usize {
        self.plane_wave_count
    }

    pub fn plane_wave_range(&self) -> Range<usize> {
        0..self.plane_wave_count
    }

    pub fn site_count(&self) -> usize {
        self.local_orbitals.len()
    }

    pub fn site_layout(&self, site: usize) -> Option<&LocalOrbitalLayout> {
        self.local_orbitals.get(site)
    }

    pub fn site_local_orbital_range(&self, site: usize) -> Option<Range<usize>> {
        let site_layout = self.local_orbitals.get(site)?;
        let start = self.plane_wave_count
            + self.local_orbitals[..site]
                .iter()
                .map(LocalOrbitalLayout::len)
                .sum::<usize>();
        Some(start..start + site_layout.len())
    }

    pub fn local_orbital_index(&self, site: usize, l: u32, m: i32, n: usize) -> Option<usize> {
        let range = self.site_local_orbital_range(site)?;
        Some(range.start + self.local_orbitals[site].index(l, m, n)?)
    }

    pub fn dimension(&self) -> usize {
        self.plane_wave_count
            + self
                .local_orbitals
                .iter()
                .map(LocalOrbitalLayout::len)
                .sum::<usize>()
    }
}

/// A real symmetric `2 x 2` radial overlap block in the `(u, du/dE)` basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialOverlapBlock {
    pub uu: f64,
    pub u_udot: f64,
    pub udot_udot: f64,
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
            if let Some(previous) = result.coefficients.get(&g)
                && (*previous - value).norm() > 64.0 * f64::EPSILON * value.norm().max(1.0)
            {
                return Err(LapwError::NonHermitianPotential { g });
            }
            if let Some(previous) = result.coefficients.get(&minus_g)
                && (*previous - value.conj()).norm() > 64.0 * f64::EPSILON * value.norm().max(1.0)
            {
                return Err(LapwError::NonHermitianPotential { g });
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

/// Both muffin-tin operators in the same local basis.
///
/// The block order is all APW radial functions as contiguous `lm` channels
/// `(u, udot)`, followed by all LOs in [`LocalOrbitalLayout`] order.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteOperatorBlocks {
    pub plane_waves: Vec<PlaneWaveAugmentation>,
    pub overlap: DenseHermitianMatrix,
    pub hamiltonian: DenseHermitianMatrix,
}

/// The two matrices of one generalized LAPW eigenproblem.
#[derive(Clone, Debug, PartialEq)]
pub struct LapwEigenproblem {
    pub overlap: DenseHermitianMatrix,
    pub hamiltonian: DenseHermitianMatrix,
}

/// Assemble `S` and `H` together in the global [`LapwBasisLayout`].
///
/// Interstitial terms occupy only the plane-wave block.  Every site is added
/// as `P^dagger B P`; APW--LO and LO--LO entries therefore come from the same
/// full local block as the APW--APW entries.
pub fn assemble_eigenproblem(
    plane_waves: &[PlaneWave],
    geometry: &InterstitialGeometry,
    layout: &LapwBasisLayout,
    potential: &InterstitialPotential,
    sites: &[SiteOperatorBlocks],
) -> Result<LapwEigenproblem, LapwError> {
    validate_plane_wave_norms(plane_waves)?;
    if let Some(first) = plane_waves.first()
        && plane_waves.iter().any(|wave| wave.k != first.k)
    {
        return Err(LapwError::MixedKPoints);
    }
    if layout.plane_wave_count != plane_waves.len() {
        return Err(LapwError::BasisPlaneWaveCount {
            expected: plane_waves.len(),
            actual: layout.plane_wave_count,
        });
    }
    if sites.len() != geometry.spheres().len() {
        return Err(LapwError::SiteCount {
            expected: geometry.spheres().len(),
            actual: sites.len(),
        });
    }
    if layout.site_count() != sites.len() {
        return Err(LapwError::BasisSiteCount {
            expected: sites.len(),
            actual: layout.site_count(),
        });
    }
    for (site_index, site) in sites.iter().enumerate() {
        validate_operator_site(site_index, site, layout)?;
    }

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
            let kinetic =
                0.25 * (plane_waves[i].q_norm.squared() + plane_waves[j].q_norm.squared());
            set_hermitian(&mut overlap, dimension, i, j, theta);
            set_hermitian(
                &mut hamiltonian,
                dimension,
                i,
                j,
                kinetic * theta + potential.coefficient(integer_difference),
            );
        }
    }

    for (site_index, site) in sites.iter().enumerate() {
        add_site_projection(
            &mut overlap,
            dimension,
            site_index,
            site,
            &site.overlap,
            layout,
        )?;
        add_site_projection(
            &mut hamiltonian,
            dimension,
            site_index,
            site,
            &site.hamiltonian,
            layout,
        )?;
    }
    Ok(LapwEigenproblem {
        overlap: DenseHermitianMatrix::from_host_row_major(dimension, Axis::GlobalBasis, overlap)?,
        hamiltonian: DenseHermitianMatrix::from_host_row_major(
            dimension,
            Axis::GlobalBasis,
            hamiltonian,
        )?,
    })
}

fn validate_operator_site(
    site_index: usize,
    site: &SiteOperatorBlocks,
    layout: &LapwBasisLayout,
) -> Result<(), LapwError> {
    if site.plane_waves.len() != layout.plane_wave_count {
        return Err(LapwError::PlaneWaveCount {
            expected: layout.plane_wave_count,
            actual: site.plane_waves.len(),
        });
    }
    let channels = site
        .plane_waves
        .first()
        .map_or(0, |augmentation| augmentation.coefficients.len());
    for (plane_wave, augmentation) in site.plane_waves.iter().enumerate() {
        if augmentation.coefficients.len() != channels {
            return Err(LapwError::ChannelCount {
                plane_wave,
                expected: channels,
                actual: augmentation.coefficients.len(),
            });
        }
    }
    let expected = 2 * channels + layout.local_orbitals[site_index].len();
    for (name, block) in [
        ("overlap", &site.overlap),
        ("Hamiltonian", &site.hamiltonian),
    ] {
        if block.dimension() != expected {
            return Err(LapwError::SiteBlockDimension {
                site: site_index,
                matrix: name,
                expected,
                actual: block.dimension(),
            });
        }
        if block.axis() != Axis::SiteCoordinate {
            return Err(LapwError::Tensor(TensorError::Axis {
                index: 0,
                expected: Axis::SiteCoordinate,
                actual: block.axis(),
            }));
        }
    }
    Ok(())
}

fn add_site_projection(
    global: &mut [Complex64],
    dimension: usize,
    site_index: usize,
    site: &SiteOperatorBlocks,
    block: &DenseHermitianMatrix,
    layout: &LapwBasisLayout,
) -> Result<(), LapwError> {
    let channels = site
        .plane_waves
        .first()
        .map_or(0, |augmentation| augmentation.coefficients.len());
    let apw_dimension = 2 * channels;
    let lo_range = layout
        .site_local_orbital_range(site_index)
        .expect("validated site index");
    let global_indices = layout
        .plane_wave_range()
        .chain(lo_range.clone())
        .collect::<Vec<_>>();
    let n_coord = block.dimension();
    let n_basis = global_indices.len();
    if n_coord == 0 || n_basis == 0 {
        return Ok(());
    }

    let mut projection = vec![Complex64::default(); n_coord * n_basis];
    for (column, &global_index) in global_indices.iter().enumerate() {
        if global_index < layout.plane_wave_count {
            for (channel, coefficients) in site.plane_waves[global_index]
                .coefficients
                .iter()
                .enumerate()
            {
                projection[(2 * channel) * n_basis + column] = coefficients[0];
                projection[(2 * channel + 1) * n_basis + column] = coefficients[1];
            }
        } else {
            let row = apw_dimension + global_index - lo_range.start;
            projection[row * n_basis + column] = Complex64::new(1.0, 0.0);
        }
    }

    let projection = ComplexTensor::from_host_row_major(
        &[n_coord, n_basis],
        &[Axis::SiteCoordinate, Axis::SiteBasis],
        projection,
    )?;
    let conjugated = projection.conjugate();
    let site_matrix = DenseHermitianMatrix::from_tensor(einsum(
        "ci,cd,dj->ij",
        &[&conjugated, block.as_tensor(), &projection],
    )?)?;
    let values = site_matrix.to_host_row_major();
    for left in 0..n_basis {
        for right in left..n_basis {
            add_hermitian(
                global,
                dimension,
                global_indices[left],
                global_indices[right],
                values[left * n_basis + right],
            );
        }
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

fn add_hermitian(
    data: &mut [Complex64],
    dimension: usize,
    row: usize,
    column: usize,
    value: Complex64,
) {
    data[row * dimension + column] += value;
    if row == column {
        data[row * dimension + row].im = 0.0;
    } else {
        data[column * dimension + row] += value.conj();
    }
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

/// Residual diagnostic for one generalized eigenpair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EigenpairResidual {
    pub band_index: usize,
    /// Euclidean norm of `H c - S c epsilon`.
    pub absolute: f64,
    /// Absolute residual divided by `max(||Hc||, |epsilon| ||Sc||)`.
    pub relative: f64,
}

/// Result of the filtered dense Hermitian generalized eigensolve.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralizedEigensolution {
    /// Eigenvalues in nondecreasing Hartree order.
    pub eigenvalues: Vec<Hartree>,
    /// Dense eigenvector columns on axes `[GlobalBasis, Band]`.
    pub eigenvectors: DenseEigenvectors,
    pub retained_dimension: usize,
    pub filtered_dimension: usize,
    pub residuals: Vec<EigenpairResidual>,
}

/// Solve `H C = S C epsilon` after removing near-linearly-dependent overlap
/// directions.  An overlap eigenvalue is retained when it is positive and
/// greater than `relative_overlap_threshold * max(eigenvalue(S))`.
pub fn solve_generalized_hermitian(
    hamiltonian: &DenseHermitianMatrix,
    overlap: &DenseHermitianMatrix,
    relative_overlap_threshold: f64,
) -> Result<GeneralizedEigensolution, LapwError> {
    use faer::{Mat, Side};

    if hamiltonian.axis() != Axis::GlobalBasis || overlap.axis() != Axis::GlobalBasis {
        return Err(LapwError::Tensor(TensorError::Axis {
            index: 0,
            expected: Axis::GlobalBasis,
            actual: hamiltonian.axis(),
        }));
    }
    if hamiltonian.dimension() != overlap.dimension() {
        return Err(LapwError::MatrixDimensionMismatch {
            hamiltonian: hamiltonian.dimension(),
            overlap: overlap.dimension(),
        });
    }
    if !relative_overlap_threshold.is_finite() || relative_overlap_threshold < 0.0 {
        return Err(LapwError::InvalidOverlapThreshold(
            relative_overlap_threshold,
        ));
    }
    let n = overlap.dimension();
    if n == 0 {
        return Err(LapwError::EmptyOverlapSubspace);
    }
    let s_matrix = Mat::from_fn(n, n, |row, column| overlap.at(row, column));
    let s_eigen = s_matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| LapwError::Eigensolver)?;
    let spectral_scale = (0..n)
        .map(|index| s_eigen.S()[index].re)
        .map(f64::abs)
        .fold(0.0, f64::max);
    let cutoff = relative_overlap_threshold * spectral_scale;
    let negative_noise_tolerance = 1024.0 * f64::EPSILON * spectral_scale;
    if let Some(eigenvalue) = (0..n)
        .map(|index| s_eigen.S()[index].re)
        .find(|&eigenvalue| eigenvalue < -negative_noise_tolerance)
    {
        return Err(LapwError::IndefiniteOverlap { eigenvalue });
    }
    let retained = (0..n)
        .filter(|&index| {
            let eigenvalue = s_eigen.S()[index].re;
            eigenvalue > 0.0 && eigenvalue > cutoff
        })
        .collect::<Vec<_>>();
    if retained.is_empty() {
        return Err(LapwError::EmptyOverlapSubspace);
    }
    let r = retained.len();

    // X = U_keep diag(s_keep^{-1/2}), so X^H S X = I. Filtering stays here;
    // the products are einsum.
    let mut u_keep = vec![Complex64::default(); n * r];
    let mut scales = vec![Complex64::default(); r];
    for (column, &source_column) in retained.iter().enumerate() {
        scales[column] = Complex64::new(1.0 / s_eigen.S()[source_column].re.sqrt(), 0.0);
        for row in 0..n {
            u_keep[row * r + column] = s_eigen.U()[(row, source_column)];
        }
    }
    let u_keep =
        ComplexTensor::from_host_row_major(&[n, r], &[Axis::GlobalBasis, Axis::Reduced], u_keep)?;
    let scales = ComplexTensor::from_host_row_major(&[r], &[Axis::Reduced], scales)?;
    let x = einsum("ik,k->ik", &[&u_keep, &scales])?;

    let x_conj = x.conjugate();
    let reduced = DenseHermitianMatrix::from_tensor(einsum(
        "ir,ij,js->rs",
        &[&x_conj, hamiltonian.as_tensor(), &x],
    )?)?;
    let reduced_matrix = Mat::from_fn(r, r, |row, column| {
        reduced
            .get(row, column)
            .expect("reduced Hermitian block is square")
    });
    let reduced_eigen = reduced_matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| LapwError::Eigensolver)?;

    let eigenvalues = (0..r)
        .map(|band| Hartree(reduced_eigen.S()[band].re))
        .collect::<Vec<_>>();
    let mut z = vec![Complex64::default(); r * r];
    for row in 0..r {
        for band in 0..r {
            z[row * r + band] = reduced_eigen.U()[(row, band)];
        }
    }
    let z = ComplexTensor::from_host_row_major(&[r, r], &[Axis::Reduced, Axis::Band], z)?;
    let vectors = einsum("ir,rb->ib", &[&x, &z])?;

    let hc = einsum("ij,jb->ib", &[hamiltonian.as_tensor(), &vectors])?;
    let sc = einsum("ij,jb->ib", &[overlap.as_tensor(), &vectors])?;
    let epsilon = ComplexTensor::from_host_row_major(
        &[r],
        &[Axis::Band],
        eigenvalues
            .iter()
            .map(|value| Complex64::new(value.get(), 0.0))
            .collect(),
    )?;
    let sc_eps = einsum("ib,b->ib", &[&sc, &epsilon])?;
    let residual = hc.sub(&sc_eps)?;
    let residual_conj = residual.conjugate();
    let residual_sq = einsum("ib,ib->b", &[&residual_conj, &residual])?;
    let hc_conj = hc.conjugate();
    let hc_sq = einsum("ib,ib->b", &[&hc_conj, &hc])?;
    let sc_conj = sc.conjugate();
    let sc_sq = einsum("ib,ib->b", &[&sc_conj, &sc])?;
    let residuals = (0..r)
        .map(|band| {
            let absolute = residual_sq
                .get(&[band])
                .expect("band residual")
                .re
                .max(0.0)
                .sqrt();
            let hc_norm = hc_sq.get(&[band]).expect("Hc norm").re.max(0.0).sqrt();
            let sc_norm = sc_sq.get(&[band]).expect("Sc norm").re.max(0.0).sqrt();
            let denominator = hc_norm.max(eigenvalues[band].get().abs() * sc_norm);
            EigenpairResidual {
                band_index: band,
                absolute,
                relative: if denominator == 0.0 {
                    absolute
                } else {
                    absolute / denominator
                },
            }
        })
        .collect();

    Ok(GeneralizedEigensolution {
        eigenvalues,
        eigenvectors: DenseEigenvectors::from_tensor(vectors)?,
        retained_dimension: r,
        filtered_dimension: n - r,
        residuals,
    })
}

/// Independent spin-up and spin-down values for a collinear, no-SOC problem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Collinear<T> {
    pub up: T,
    pub down: T,
}

impl<T> Collinear<T> {
    pub const fn new(up: T, down: T) -> Self {
        Self { up, down }
    }
}

/// Matrices and eigensolution for one spin channel.
#[derive(Clone, Debug, PartialEq)]
pub struct SolvedLapwEigenproblem {
    pub eigenproblem: LapwEigenproblem,
    pub solution: GeneralizedEigensolution,
}

/// Assemble and solve two independent collinear spin channels.
///
/// Plane waves, geometry, and global layout are shared.  Potentials and local
/// operator blocks are channel-specific, and no doubled spin matrix is formed.
pub fn solve_collinear_eigenproblems(
    plane_waves: &[PlaneWave],
    geometry: &InterstitialGeometry,
    layout: &LapwBasisLayout,
    potentials: Collinear<&InterstitialPotential>,
    sites: Collinear<&[SiteOperatorBlocks]>,
    relative_overlap_threshold: f64,
) -> Result<Collinear<SolvedLapwEigenproblem>, LapwError> {
    let solve_channel = |potential: &InterstitialPotential,
                         site_blocks: &[SiteOperatorBlocks]|
     -> Result<SolvedLapwEigenproblem, LapwError> {
        let eigenproblem =
            assemble_eigenproblem(plane_waves, geometry, layout, potential, site_blocks)?;
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

fn i_pow(l: u32) -> Complex64 {
    match l % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mt_core::{ReciprocalLattice, Sphere};

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

    fn site_h(
        dimension: usize,
        element: impl FnMut(usize, usize) -> Complex64,
    ) -> DenseHermitianMatrix {
        DenseHermitianMatrix::from_upper_triangle(dimension, Axis::SiteCoordinate, element).unwrap()
    }

    fn global_h(
        dimension: usize,
        element: impl FnMut(usize, usize) -> Complex64,
    ) -> DenseHermitianMatrix {
        DenseHermitianMatrix::from_upper_triangle(dimension, Axis::GlobalBasis, element).unwrap()
    }

    #[test]
    fn matching_residuals_are_small() {
        let basis = ApwBoundaryBasis {
            u: boundary(0.73, -0.21),
            udot: boundary(-0.18, 1.14),
        };
        for l in 0..=8 {
            let matched = match_apw_boundary(l, InverseBohr(2.3), Bohr(1.7), basis).unwrap();
            assert!(matched.value_residual.abs() <= 1.0e-10);
            assert!(matched.slope_residual.abs() <= 1.0e-10);
        }
    }

    #[test]
    fn overlap_is_hermitian() {
        let waves = waves();
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: [Bohr(0.3), Bohr(-0.4), Bohr(0.2)],
                radius: Bohr(0.8),
            }],
        )
        .unwrap();
        let boundary = ApwBoundaryBasis {
            u: boundary(0.8, -0.1),
            udot: boundary(0.2, 1.1),
        };
        let plane_waves = waves
            .iter()
            .map(|wave| {
                let matches = (0..=2)
                    .map(|l| match_apw_boundary(l, wave.q_norm, Bohr(0.8), boundary).unwrap())
                    .collect::<Vec<_>>();
                augmentation_coefficients(
                    wave,
                    [Bohr(0.3), Bohr(-0.4), Bohr(0.2)],
                    VolumeBohr3(100.0),
                    &matches,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let radial = RadialOverlapBlock {
            uu: 1.0,
            u_udot: 0.04,
            udot_udot: 0.7,
        };
        let local_dimension = 2 * plane_waves[0].coefficients.len();
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
        let site = SiteOperatorBlocks {
            plane_waves,
            overlap: overlap_block,
            hamiltonian: site_h(local_dimension, |_, _| Complex64::default()),
        };
        let layout = LapwBasisLayout::new(waves.len(), vec![LocalOrbitalLayout::default()]);
        let overlap = assemble_eigenproblem(
            &waves,
            &geometry,
            &layout,
            &InterstitialPotential::default(),
            &[site],
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
        let waves = waves();
        let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), vec![]).unwrap();
        let layout = LapwBasisLayout::new(waves.len(), vec![]);
        let overlap = assemble_eigenproblem(
            &waves,
            &geometry,
            &layout,
            &InterstitialPotential::default(),
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
        let layout = LapwBasisLayout::new(waves.len(), vec![]);
        let problem = assemble_eigenproblem(&waves, &geometry, &layout, &potential, &[]).unwrap();
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
        let layout = LapwBasisLayout::new(waves.len(), vec![LocalOrbitalLayout::default()]);
        let hamiltonian = assemble_eigenproblem(
            &waves,
            &geometry,
            &layout,
            &InterstitialPotential::default(),
            &[SiteOperatorBlocks {
                plane_waves: augmentations,
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
    fn local_orbital_layout_uses_site_l_m_n_order_and_global_offsets() {
        let first = LocalOrbitalLayout::new(vec![2, 1]);
        let second = LocalOrbitalLayout::new(vec![0, 2, 1]);
        assert_eq!(first.len(), 5);
        assert_eq!(first.index(0, 0, 0), Some(0));
        assert_eq!(first.index(0, 0, 1), Some(1));
        assert_eq!(first.index(1, -1, 0), Some(2));
        assert_eq!(first.index(1, 1, 0), Some(4));
        assert_eq!(first.index(1, 1, 1), None);

        let layout = LapwBasisLayout::new(7, vec![first, second]);
        assert_eq!(layout.site_local_orbital_range(0), Some(7..12));
        assert_eq!(layout.site_local_orbital_range(1), Some(12..23));
        assert_eq!(layout.local_orbital_index(1, 1, -1, 1), Some(13));
        assert_eq!(layout.local_orbital_index(1, 2, 2, 0), Some(22));
        assert_eq!(layout.dimension(), 23);
    }

    #[test]
    fn full_local_blocks_generate_complex_apw_lo_and_lo_lo_elements() {
        let wave = waves()[0];
        let plane_waves = [wave];
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
        let site = SiteOperatorBlocks {
            plane_waves: vec![PlaneWaveAugmentation {
                coefficients: vec![[a, b]],
            }],
            overlap,
            hamiltonian,
        };
        let layout = LapwBasisLayout::new(1, vec![LocalOrbitalLayout::new(vec![1])]);
        let problem = assemble_eigenproblem(
            &plane_waves,
            &geometry,
            &layout,
            &InterstitialPotential::default(),
            std::slice::from_ref(&site),
        )
        .unwrap();
        let expected_s_apw_lo = a.conj() * site.overlap.at(0, 2) + b.conj() * site.overlap.at(1, 2);
        let expected_h_apw_lo =
            a.conj() * site.hamiltonian.at(0, 2) + b.conj() * site.hamiltonian.at(1, 2);
        let expected_s_apw_apw = a.conj() * (site.overlap.at(0, 0) * a + site.overlap.at(0, 1) * b)
            + b.conj() * (site.overlap.at(1, 0) * a + site.overlap.at(1, 1) * b);
        let expected_h_apw_apw = a.conj()
            * (site.hamiltonian.at(0, 0) * a + site.hamiltonian.at(0, 1) * b)
            + b.conj() * (site.hamiltonian.at(1, 0) * a + site.hamiltonian.at(1, 1) * b);
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
        assert_eq!(problem.overlap.at(1, 1), site.overlap.at(2, 2));
        assert_eq!(problem.hamiltonian.at(1, 1), site.hamiltonian.at(2, 2));
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
        let plane_waves = [wave];
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
        let site = SiteOperatorBlocks {
            plane_waves: vec![PlaneWaveAugmentation {
                coefficients: vec![[a, b]],
            }],
            overlap,
            hamiltonian,
        };
        let layout = LapwBasisLayout::new(1, vec![LocalOrbitalLayout::default()]);
        let problem = assemble_eigenproblem(
            &plane_waves,
            &geometry,
            &layout,
            &InterstitialPotential::default(),
            std::slice::from_ref(&site),
        )
        .unwrap();
        let expected_s = a.conj() * (site.overlap.at(0, 0) * a + site.overlap.at(0, 1) * b)
            + b.conj() * (site.overlap.at(1, 0) * a + site.overlap.at(1, 1) * b);
        let expected_h = a.conj() * (site.hamiltonian.at(0, 0) * a + site.hamiltonian.at(0, 1) * b)
            + b.conj() * (site.hamiltonian.at(1, 0) * a + site.hamiltonian.at(1, 1) * b);
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
    fn nonzero_k_site_phase_is_carried_by_augmentation_coefficients() {
        let wave = waves()[0];
        assert!(wave.k.iter().any(|component| component.get() != 0.0));
        let matched = [ApwMatch {
            l: 0,
            coefficients: [0.7, -0.2],
            value_residual: 0.0,
            slope_residual: 0.0,
        }];
        let origin =
            augmentation_coefficients(&wave, [Bohr(0.0); 3], VolumeBohr3(80.0), &matched).unwrap();
        let site = [Bohr(0.31), Bohr(-0.27), Bohr(0.19)];
        let translated =
            augmentation_coefficients(&wave, site, VolumeBohr3(80.0), &matched).unwrap();
        let phase = site_translation_phase(wave.q, site);
        for radial in 0..2 {
            assert!(
                (translated.coefficients[0][radial] - phase * origin.coefficients[0][radial])
                    .norm()
                    < 1.0e-14
            );
        }
    }

    #[test]
    fn collinear_driver_matches_two_independent_channel_solves() {
        let waves = waves();
        let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), vec![]).unwrap();
        let layout = LapwBasisLayout::new(waves.len(), vec![]);
        let up = InterstitialPotential::new([([0; 3], Complex64::new(0.17, 0.0))]).unwrap();
        let down = InterstitialPotential::new([([0; 3], Complex64::new(-0.09, 0.0))]).unwrap();
        let spins = solve_collinear_eigenproblems(
            &waves,
            &geometry,
            &layout,
            Collinear::new(&up, &down),
            Collinear::new(&[][..], &[][..]),
            0.0,
        )
        .unwrap();
        let up_problem = assemble_eigenproblem(&waves, &geometry, &layout, &up, &[]).unwrap();
        let down_problem = assemble_eigenproblem(&waves, &geometry, &layout, &down, &[]).unwrap();
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
        assert_eq!(
            spins.up.eigenproblem.overlap.dimension(),
            layout.dimension()
        );
        assert_eq!(
            spins.down.eigenproblem.overlap.dimension(),
            layout.dimension()
        );
    }

    #[test]
    fn generalized_solver_filters_near_null_overlap_and_rejects_indefinite_overlap() {
        let h = global_h(2, |row, column| {
            if row == column {
                Complex64::new((row + 1) as f64, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        let nearly_singular = global_h(2, |row, column| {
            if row == column {
                Complex64::new(if row == 0 { 1.0 } else { -1.0e-14 }, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        let solution = solve_generalized_hermitian(&h, &nearly_singular, 1.0e-10).unwrap();
        assert_eq!(solution.retained_dimension, 1);
        assert_eq!(solution.filtered_dimension, 1);
        assert!((solution.eigenvalues[0].get() - 1.0).abs() < 1.0e-14);

        let indefinite = global_h(2, |row, column| {
            if row == column {
                Complex64::new(if row == 0 { 1.0 } else { -0.1 }, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        assert!(matches!(
            solve_generalized_hermitian(&h, &indefinite, 1.0e-10),
            Err(LapwError::IndefiniteOverlap { .. })
        ));
    }

    #[test]
    fn generalized_eigenvectors_are_s_orthonormal_with_small_residuals() {
        let h = global_h(2, |row, column| match (row, column) {
            (0, 0) => Complex64::new(1.0, 0.0),
            (0, 1) => Complex64::new(0.2, 0.1),
            (1, 1) => Complex64::new(2.0, 0.0),
            _ => unreachable!(),
        });
        let s = global_h(2, |row, column| match (row, column) {
            (0, 0) => Complex64::new(1.3, 0.0),
            (0, 1) => Complex64::new(0.1, -0.05),
            (1, 1) => Complex64::new(0.9, 0.0),
            _ => unreachable!(),
        });
        let solution = solve_generalized_hermitian(&h, &s, 1.0e-12).unwrap();
        for left in 0..2 {
            for right in 0..2 {
                let mut value = Complex64::new(0.0, 0.0);
                for i in 0..2 {
                    for j in 0..2 {
                        value += solution.eigenvectors.at(i, left).conj()
                            * s.at(i, j)
                            * solution.eigenvectors.at(j, right);
                    }
                }
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((value - expected).norm() < 1.0e-12);
            }
        }
        assert!(
            solution
                .residuals
                .iter()
                .all(|residual| residual.absolute < 1.0e-12)
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
}
