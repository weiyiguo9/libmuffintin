//! Linearized augmented-plane-wave matching and overlap assembly.
//!
//! Plane waves use `exp(+i (k+G) dot r) / sqrt(Omega)`.  The Rayleigh
//! coefficient and the site-translation phase are exposed separately so that
//! neither convention is hidden inside the real, radial boundary solve.

#![forbid(unsafe_code)]

use mt_core::{
    Bohr, GVector, Hartree, InterstitialGeometry, InverseBohr, Lm, VolumeBohr3,
    complex_spherical_harmonics, lm_count, lm_from_index, spherical_bessel_j,
    spherical_bessel_j_derivative,
};
use mt_radial::BoundaryData;
use num_complex::Complex64;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::ops::Index;
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

/// Per-site APW coefficients and real radial overlap blocks.
///
/// `radial_overlaps[l]` uses the ordered radial basis
/// `(u_l, du_l/dE)`.  Local orbitals are deliberately absent from this M-D
/// representation.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteAugmentation {
    pub plane_waves: Vec<PlaneWaveAugmentation>,
    pub radial_overlaps: Vec<RadialOverlapBlock>,
}

/// A real symmetric `2 x 2` radial overlap block in the `(u, du/dE)` basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialOverlapBlock {
    pub uu: f64,
    pub u_udot: f64,
    pub udot_udot: f64,
}

impl RadialOverlapBlock {
    fn element(self, left: usize, right: usize) -> f64 {
        match (left, right) {
            (0, 0) => self.uu,
            (0, 1) | (1, 0) => self.u_udot,
            (1, 1) => self.udot_udot,
            _ => unreachable!("radial indices are fixed to 0 and 1"),
        }
    }
}

/// Dense row-major complex Hermitian matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseHermitianMatrix {
    dimension: usize,
    data: Vec<Complex64>,
}

impl DenseHermitianMatrix {
    /// Construct a Hermitian matrix from its upper triangle.
    pub fn from_upper_triangle(
        dimension: usize,
        mut element: impl FnMut(usize, usize) -> Complex64,
    ) -> Self {
        let mut data = vec![Complex64::new(0.0, 0.0); dimension * dimension];
        for row in 0..dimension {
            for column in row..dimension {
                let value = element(row, column);
                data[row * dimension + column] = value;
                data[column * dimension + row] = value.conj();
            }
        }
        Self { dimension, data }
    }

    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn as_slice(&self) -> &[Complex64] {
        &self.data
    }
}

impl Index<(usize, usize)> for DenseHermitianMatrix {
    type Output = Complex64;

    fn index(&self, (row, column): (usize, usize)) -> &Self::Output {
        &self.data[row * self.dimension + column]
    }
}

/// Dense row-major matrix whose columns are vectors in the original LAPW basis.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseEigenvectors {
    rows: usize,
    columns: usize,
    data: Vec<Complex64>,
}

impl DenseEigenvectors {
    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub fn as_slice(&self) -> &[Complex64] {
        &self.data
    }
}

impl Index<(usize, usize)> for DenseEigenvectors {
    type Output = Complex64;

    fn index(&self, (row, column): (usize, usize)) -> &Self::Output {
        &self.data[row * self.columns + column]
    }
}

/// Assemble the LAPW overlap
/// `Theta(G_i-G_j) + sum_a,lm c_i^dagger O^a_l c_j`.
///
/// Only the upper triangle is evaluated; the lower triangle is filled by
/// conjugation, making the returned dense matrix Hermitian by construction.
pub fn assemble_overlap(
    plane_waves: &[PlaneWave],
    geometry: &InterstitialGeometry,
    sites: &[SiteAugmentation],
) -> Result<DenseHermitianMatrix, LapwError> {
    validate_plane_wave_norms(plane_waves)?;
    if let Some(first) = plane_waves.first() {
        if plane_waves.iter().any(|wave| wave.k != first.k) {
            return Err(LapwError::MixedKPoints);
        }
    }
    let dimension = plane_waves.len();
    if sites.len() != geometry.spheres().len() {
        return Err(LapwError::SiteCount {
            expected: geometry.spheres().len(),
            actual: sites.len(),
        });
    }
    for site in sites {
        validate_site(site, dimension)?;
    }

    let mut data = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for i in 0..dimension {
        for j in i..dimension {
            let difference = std::array::from_fn(|axis| {
                InverseBohr(
                    plane_waves[i].g.cartesian[axis].get() - plane_waves[j].g.cartesian[axis].get(),
                )
            });
            let mut value = geometry
                .coefficient(difference)
                .map_err(|error| LapwError::StepFunction(error.to_string()))?;
            for site in sites {
                let left = &site.plane_waves[i].coefficients;
                let right = &site.plane_waves[j].coefficients;
                for channel in 0..left.len() {
                    let l = lm_from_index(channel).l as usize;
                    let overlap = site.radial_overlaps[l];
                    for (alpha, left_coefficient) in left[channel].iter().enumerate() {
                        for (beta, right_coefficient) in right[channel].iter().enumerate() {
                            value += left_coefficient.conj()
                                * overlap.element(alpha, beta)
                                * right_coefficient;
                        }
                    }
                }
            }
            data[i * dimension + j] = value;
            data[j * dimension + i] = value.conj();
        }
    }
    Ok(DenseHermitianMatrix { dimension, data })
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

/// One site's dense muffin-tin Hamiltonian block.  Its basis is flattened as
/// contiguous `lm`, then `(u, udot)` within each `lm` channel.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteHamiltonian {
    pub plane_waves: Vec<PlaneWaveAugmentation>,
    pub block: DenseHermitianMatrix,
}

/// Assemble the dense LAPW Hamiltonian in Hartree.
///
/// The interstitial kinetic term is
/// `0.25 (|q_i|^2 + |q_j|^2) Theta(G_i-G_j)`, and the muffin-tin term is
/// evaluated as `c_i^dagger h c_j` in the documented flattened basis.
pub fn assemble_hamiltonian(
    plane_waves: &[PlaneWave],
    geometry: &InterstitialGeometry,
    potential: &InterstitialPotential,
    sites: &[SiteHamiltonian],
) -> Result<DenseHermitianMatrix, LapwError> {
    validate_plane_wave_norms(plane_waves)?;
    if let Some(first) = plane_waves.first()
        && plane_waves.iter().any(|wave| wave.k != first.k)
    {
        return Err(LapwError::MixedKPoints);
    }
    let dimension = plane_waves.len();
    if sites.len() != geometry.spheres().len() {
        return Err(LapwError::SiteCount {
            expected: geometry.spheres().len(),
            actual: sites.len(),
        });
    }
    for site in sites {
        validate_hamiltonian_site(site, dimension)?;
    }

    let mut data = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for i in 0..dimension {
        for j in i..dimension {
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
            let mut value = kinetic * theta + potential.coefficient(integer_difference);

            for site in sites {
                let left = &site.plane_waves[i].coefficients;
                let right = &site.plane_waves[j].coefficients;
                for (left_channel, left_coefficients) in left.iter().enumerate() {
                    for (left_radial, left_coefficient) in left_coefficients.iter().enumerate() {
                        let left_index = 2 * left_channel + left_radial;
                        for (right_channel, right_coefficients) in right.iter().enumerate() {
                            for (right_radial, right_coefficient) in
                                right_coefficients.iter().enumerate()
                            {
                                let right_index = 2 * right_channel + right_radial;
                                value += left_coefficient.conj()
                                    * site.block[(left_index, right_index)]
                                    * right_coefficient;
                            }
                        }
                    }
                }
            }
            data[i * dimension + j] = value;
            data[j * dimension + i] = value.conj();
        }
    }
    Ok(DenseHermitianMatrix { dimension, data })
}

fn validate_hamiltonian_site(
    site: &SiteHamiltonian,
    plane_wave_count: usize,
) -> Result<(), LapwError> {
    if site.plane_waves.len() != plane_wave_count {
        return Err(LapwError::PlaneWaveCount {
            expected: plane_wave_count,
            actual: site.plane_waves.len(),
        });
    }
    let channels = site
        .plane_waves
        .first()
        .map_or(0, |augmentation| augmentation.coefficients.len());
    let expected = 2 * channels;
    if site.block.dimension() != expected {
        return Err(LapwError::MatrixDimensionMismatch {
            hamiltonian: site.block.dimension(),
            overlap: expected,
        });
    }
    for (plane_wave, augmentation) in site.plane_waves.iter().enumerate() {
        if augmentation.coefficients.len() != channels {
            return Err(LapwError::ChannelCount {
                plane_wave,
                expected: channels,
                actual: augmentation.coefficients.len(),
            });
        }
    }
    Ok(())
}

/// SPEX spherical radial Hamiltonian in the `(u, udot)` basis.
///
/// This is the symmetrized identity from `hamilton.f:396-425`:
/// `(E_1 + E_2) O / 2`, plus `O(u, *) / 2` for every side occupied by
/// `udot`.
pub fn spex_spherical_radial_hamiltonian(
    linearization_energy: Hartree,
    overlap: RadialOverlapBlock,
) -> DenseHermitianMatrix {
    let energy = linearization_energy.get();
    DenseHermitianMatrix::from_upper_triangle(2, |row, column| {
        let value = match (row, column) {
            (0, 0) => energy * overlap.uu,
            (0, 1) => energy * overlap.u_udot + 0.5 * overlap.uu,
            (1, 1) => energy * overlap.udot_udot,
            _ => unreachable!("only the upper triangle is requested"),
        };
        Complex64::new(value, 0.0)
    })
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
    /// Eigenvector columns expressed in the original plane-wave basis.
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

    if hamiltonian.dimension != overlap.dimension {
        return Err(LapwError::MatrixDimensionMismatch {
            hamiltonian: hamiltonian.dimension,
            overlap: overlap.dimension,
        });
    }
    if !relative_overlap_threshold.is_finite() || relative_overlap_threshold < 0.0 {
        return Err(LapwError::InvalidOverlapThreshold(
            relative_overlap_threshold,
        ));
    }
    validate_dense_hermitian(hamiltonian, "Hamiltonian")?;
    validate_dense_hermitian(overlap, "overlap")?;
    let n = overlap.dimension;
    if n == 0 {
        return Err(LapwError::EmptyOverlapSubspace);
    }
    let s_matrix = Mat::from_fn(n, n, |row, column| overlap[(row, column)]);
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

    // X = U_keep diag(s_keep^-1/2), so X^H S X = I.
    let mut x = vec![Complex64::new(0.0, 0.0); n * r];
    for (column, &source_column) in retained.iter().enumerate() {
        let scale = 1.0 / s_eigen.S()[source_column].re.sqrt();
        for row in 0..n {
            x[row * r + column] = s_eigen.U()[(row, source_column)] * scale;
        }
    }

    let reduced = Mat::from_fn(r, r, |left, right| {
        let mut value = Complex64::new(0.0, 0.0);
        for i in 0..n {
            for j in 0..n {
                value += x[i * r + left].conj() * hamiltonian[(i, j)] * x[j * r + right];
            }
        }
        value
    });
    let reduced_eigen = reduced
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| LapwError::Eigensolver)?;

    let eigenvalues = (0..r)
        .map(|band| Hartree(reduced_eigen.S()[band].re))
        .collect::<Vec<_>>();
    let mut vectors = vec![Complex64::new(0.0, 0.0); n * r];
    for row in 0..n {
        for band in 0..r {
            for reduced_row in 0..r {
                vectors[row * r + band] +=
                    x[row * r + reduced_row] * reduced_eigen.U()[(reduced_row, band)];
            }
        }
    }

    let residuals = eigenvalues
        .iter()
        .enumerate()
        .map(|(band, eigenvalue)| {
            let mut residual_squared = 0.0;
            let mut hc_squared = 0.0;
            let mut sc_squared = 0.0;
            for row in 0..n {
                let mut hc = Complex64::new(0.0, 0.0);
                let mut sc = Complex64::new(0.0, 0.0);
                for column in 0..n {
                    let coefficient = vectors[column * r + band];
                    hc += hamiltonian[(row, column)] * coefficient;
                    sc += overlap[(row, column)] * coefficient;
                }
                residual_squared += (hc - sc * eigenvalue.get()).norm_sqr();
                hc_squared += hc.norm_sqr();
                sc_squared += sc.norm_sqr();
            }
            let absolute = residual_squared.sqrt();
            let denominator = hc_squared
                .sqrt()
                .max(eigenvalue.get().abs() * sc_squared.sqrt());
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
        eigenvectors: DenseEigenvectors {
            rows: n,
            columns: r,
            data: vectors,
        },
        retained_dimension: r,
        filtered_dimension: n - r,
        residuals,
    })
}

fn validate_dense_hermitian(
    matrix: &DenseHermitianMatrix,
    matrix_name: &'static str,
) -> Result<(), LapwError> {
    let scale = matrix
        .data
        .iter()
        .filter(|value| value.re.is_finite() && value.im.is_finite())
        .map(|value| value.norm())
        .fold(0.0, f64::max)
        .max(1.0);
    let tolerance = 128.0 * f64::EPSILON * scale;
    for row in 0..matrix.dimension {
        for column in 0..matrix.dimension {
            let value = matrix[(row, column)];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(LapwError::NonFiniteMatrix {
                    matrix: matrix_name,
                    row,
                    column,
                });
            }
            if (value - matrix[(column, row)].conj()).norm() > tolerance {
                return Err(LapwError::NonHermitianMatrix { row, column });
            }
        }
    }
    Ok(())
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

fn validate_site(site: &SiteAugmentation, plane_wave_count: usize) -> Result<(), LapwError> {
    if site.plane_waves.len() != plane_wave_count {
        return Err(LapwError::PlaneWaveCount {
            expected: plane_wave_count,
            actual: site.plane_waves.len(),
        });
    }
    let expected_channels = site
        .radial_overlaps
        .len()
        .saturating_mul(site.radial_overlaps.len());
    for (plane_wave, augmentation) in site.plane_waves.iter().enumerate() {
        if augmentation.coefficients.len() != expected_channels {
            return Err(LapwError::ChannelCount {
                plane_wave,
                expected: expected_channels,
                actual: augmentation.coefficients.len(),
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
            .collect();
        let site = SiteAugmentation {
            plane_waves,
            radial_overlaps: vec![
                RadialOverlapBlock {
                    uu: 1.0,
                    u_udot: 0.04,
                    udot_udot: 0.7,
                };
                3
            ],
        };
        let overlap = assemble_overlap(&waves, &geometry, &[site]).unwrap();
        for i in 0..overlap.dimension() {
            for j in 0..overlap.dimension() {
                assert!((overlap[(i, j)] - overlap[(j, i)].conj()).norm() < 2.0e-14);
            }
        }
    }

    #[test]
    fn empty_sphere_geometry_is_identity() {
        let waves = waves();
        let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), vec![]).unwrap();
        let overlap = assemble_overlap(&waves, &geometry, &[]).unwrap();
        for i in 0..overlap.dimension() {
            for j in 0..overlap.dimension() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((overlap[(i, j)] - expected).norm() < 2.0e-14);
            }
        }
    }

    #[test]
    fn empty_lattice_hamiltonian_is_free_electron() {
        let waves = waves();
        let geometry = InterstitialGeometry::new(VolumeBohr3(100.0), vec![]).unwrap();
        let potential = InterstitialPotential::default();
        let overlap = assemble_overlap(&waves, &geometry, &[]).unwrap();
        let hamiltonian = assemble_hamiltonian(&waves, &geometry, &potential, &[]).unwrap();
        for i in 0..hamiltonian.dimension() {
            for j in 0..hamiltonian.dimension() {
                let expected = if i == j {
                    0.5 * waves[i].q_norm.squared()
                } else {
                    0.0
                };
                assert!((hamiltonian[(i, j)] - expected).norm() <= 1.0e-12);
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
        let block =
            DenseHermitianMatrix::from_upper_triangle(2, |row, column| match (row, column) {
                (0, 0) => Complex64::new(1.2, 0.0),
                (0, 1) => Complex64::new(0.1, -0.2),
                (1, 1) => Complex64::new(0.8, 0.0),
                _ => unreachable!(),
            });
        let hamiltonian = assemble_hamiltonian(
            &waves,
            &geometry,
            &InterstitialPotential::default(),
            &[SiteHamiltonian {
                plane_waves: augmentations,
                block,
            }],
        )
        .unwrap();
        for i in 0..hamiltonian.dimension() {
            for j in 0..hamiltonian.dimension() {
                assert!((hamiltonian[(i, j)] - hamiltonian[(j, i)].conj()).norm() < 1.0e-14);
            }
        }
    }

    #[test]
    fn generalized_solver_filters_near_null_overlap_and_rejects_indefinite_overlap() {
        let h = DenseHermitianMatrix::from_upper_triangle(2, |row, column| {
            if row == column {
                Complex64::new((row + 1) as f64, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        let nearly_singular = DenseHermitianMatrix::from_upper_triangle(2, |row, column| {
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

        let indefinite = DenseHermitianMatrix::from_upper_triangle(2, |row, column| {
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
        let h = DenseHermitianMatrix::from_upper_triangle(2, |row, column| match (row, column) {
            (0, 0) => Complex64::new(1.0, 0.0),
            (0, 1) => Complex64::new(0.2, 0.1),
            (1, 1) => Complex64::new(2.0, 0.0),
            _ => unreachable!(),
        });
        let s = DenseHermitianMatrix::from_upper_triangle(2, |row, column| match (row, column) {
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
                        value += solution.eigenvectors[(i, left)].conj()
                            * s[(i, j)]
                            * solution.eigenvectors[(j, right)];
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
        assert_eq!(h[(0, 0)], Complex64::new(0.6, 0.0));
        assert_eq!(h[(0, 1)], Complex64::new(0.75, 0.0));
        assert_eq!(h[(1, 1)], Complex64::new(0.4, 0.0));
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
