//! Nonmagnetic SPEX-style second-variation spin--orbit coupling.
//!
//! The first variation must be scalar Koelling--Harmon. This module does not
//! route magnetic, noncollinear, or spinor-first-variation states through the
//! scalar approximation, and it does not claim four-component accuracy in the
//! muffin-tin spheres.

use muffintin_basis::CompiledBasis;
use muffintin_core::Hartree;
use muffintin_operators::{
    CompiledSiteProjection, OperatorError, SecondVariationMixing, SiteSpinOrbitBlock,
    SocEigenpairResidual, SocOperatorError, project_site_soc_to_subspace,
    solve_second_variation_subspace,
};
use muffintin_tensor::{DenseEigenvectors, TensorError};
use num_complex::Complex64;
use thiserror::Error;

/// Half-open source-band window `[start, end)` for the first variation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FirstVariationWindow {
    start: usize,
    end: usize,
}

impl FirstVariationWindow {
    pub fn new(start: usize, end: usize) -> Result<Self, SecondVariationError> {
        if start >= end {
            return Err(SecondVariationError::EmptyWindow { start, end });
        }
        Ok(Self { start, end })
    }

    pub const fn start(self) -> usize {
        self.start
    }

    pub const fn end(self) -> usize {
        self.end
    }

    pub const fn len(self) -> usize {
        self.end - self.start
    }

    pub const fn is_empty(self) -> bool {
        false
    }
}

/// Selected scalar eigenpairs entering second variation.
#[derive(Clone, Debug, PartialEq)]
pub struct FirstVariationSubspace {
    pub window: FirstVariationWindow,
    pub source_bands: Vec<usize>,
    pub eigenvalues: Vec<Hartree>,
    pub eigenvectors: DenseEigenvectors,
}

impl FirstVariationSubspace {
    /// Copy a contiguous window from one scalar first-variation eigensolution.
    pub fn select(
        window: FirstVariationWindow,
        eigenvalues: &[Hartree],
        eigenvectors: &DenseEigenvectors,
    ) -> Result<Self, SecondVariationError> {
        if eigenvalues.len() != eigenvectors.columns() {
            return Err(SecondVariationError::FirstVariationCount {
                eigenvalues: eigenvalues.len(),
                eigenvectors: eigenvectors.columns(),
            });
        }
        if window.end > eigenvalues.len() {
            return Err(SecondVariationError::WindowOutOfRange {
                end: window.end,
                band_count: eigenvalues.len(),
            });
        }
        let source_bands = (window.start..window.end).collect::<Vec<_>>();
        let selected_values = eigenvalues[window.start..window.end].to_vec();
        let mut selected_vectors = Vec::with_capacity(eigenvectors.rows() * window.len());
        for band in window.start..window.end {
            for basis in 0..eigenvectors.rows() {
                selected_vectors.push(eigenvectors.at(basis, band));
            }
        }
        Ok(Self {
            window,
            source_bands,
            eigenvalues: selected_values,
            eigenvectors: DenseEigenvectors::from_host_column_major(
                eigenvectors.rows(),
                window.len(),
                selected_vectors,
            )?,
        })
    }
}

/// First-variation route presented to the optional second-variation path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FirstVariationRoute {
    /// The only route accepted by [`solve_spex_second_variation`].
    NonmagneticScalarKoellingHarmon,
    CollinearMagnetic,
    Noncollinear,
    SpinorFirstVariation,
}

/// Spin-resolved contribution of one source scalar band to one SOC band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceBandWeight {
    pub source_band: usize,
    pub spin_up: f64,
    pub spin_down: f64,
}

/// Diagnostics for one second-variation eigenstate.
#[derive(Clone, Debug, PartialEq)]
pub struct SecondVariationBandDiagnostic {
    pub band_index: usize,
    pub residual: SocEigenpairResidual,
    pub source_weights: Vec<SourceBandWeight>,
}

/// Eigenpairs reconstructed on the doubled global scalar basis.
#[derive(Clone, Debug, PartialEq)]
pub struct SecondVariationResult {
    pub eigenvalues: Vec<Hartree>,
    /// Rows are spin slow: `(up global basis, down global basis)`.
    pub eigenvectors: DenseEigenvectors,
    /// Exact first-variation band indices corresponding to mixing-matrix rows.
    pub source_bands: Vec<usize>,
    pub mixing: SecondVariationMixing,
    pub diagnostics: Vec<SecondVariationBandDiagnostic>,
}

/// Project every site's scalar eigenvectors, solve the ordinary Hermitian SOC
/// problem, and reconstruct doubled global Pauli-spinor eigenvectors.
pub fn solve_spex_second_variation(
    route: FirstVariationRoute,
    compiled: &CompiledBasis,
    first_variation: &FirstVariationSubspace,
    site_blocks: &[SiteSpinOrbitBlock],
) -> Result<SecondVariationResult, SecondVariationError> {
    if route != FirstVariationRoute::NonmagneticScalarKoellingHarmon {
        return Err(SecondVariationError::UnsupportedRoute(route));
    }
    if first_variation.eigenvalues.len() != first_variation.eigenvectors.columns()
        || first_variation.source_bands.len() != first_variation.eigenvalues.len()
    {
        return Err(SecondVariationError::MalformedSubspace);
    }
    if compiled.layout.dimension() != first_variation.eigenvectors.rows() {
        return Err(SecondVariationError::GlobalBasisDimension {
            expected: compiled.layout.dimension(),
            actual: first_variation.eigenvectors.rows(),
        });
    }
    if site_blocks.len() != compiled.site_count() {
        return Err(SecondVariationError::SiteCount {
            expected: compiled.site_count(),
            actual: site_blocks.len(),
        });
    }

    let mut site_contributions = Vec::with_capacity(site_blocks.len());
    for (site, block) in site_blocks.iter().enumerate() {
        let projection = CompiledSiteProjection::scalar(compiled, site)?;
        let coefficients = projection.project_eigenvectors(&first_variation.eigenvectors)?;
        site_contributions.push(project_site_soc_to_subspace(block, &coefficients)?);
    }
    let subspace =
        solve_second_variation_subspace(&first_variation.eigenvalues, &site_contributions)?;
    let eigenvectors = reconstruct_global_spinors(&first_variation.eigenvectors, &subspace.mixing)?;
    let bands = first_variation.eigenvalues.len();
    let diagnostics = (0..2 * bands)
        .map(|output_band| SecondVariationBandDiagnostic {
            band_index: output_band,
            residual: subspace.residuals[output_band],
            source_weights: first_variation
                .source_bands
                .iter()
                .enumerate()
                .map(|(source, &source_band)| SourceBandWeight {
                    source_band,
                    spin_up: subspace.mixing.at(source, output_band).norm_sqr(),
                    spin_down: subspace.mixing.at(bands + source, output_band).norm_sqr(),
                })
                .collect(),
        })
        .collect();

    Ok(SecondVariationResult {
        eigenvalues: subspace.eigenvalues,
        eigenvectors,
        source_bands: first_variation.source_bands.clone(),
        mixing: subspace.mixing,
        diagnostics,
    })
}

fn reconstruct_global_spinors(
    scalar: &DenseEigenvectors,
    mixing: &SecondVariationMixing,
) -> Result<DenseEigenvectors, TensorError> {
    let global_dimension = scalar.rows();
    let bands = scalar.columns();
    if mixing.dimension() != 2 * bands {
        return Err(TensorError::Shape {
            expected: vec![2 * bands, 2 * bands],
            actual: vec![mixing.dimension(), mixing.dimension()],
        });
    }
    let output_bands = 2 * bands;
    let mut values = Vec::with_capacity(4 * global_dimension * bands);
    for output in 0..output_bands {
        for spin in 0..2 {
            for global in 0..global_dimension {
                let mut value = Complex64::default();
                for source in 0..bands {
                    value += scalar.at(global, source) * mixing.at(spin * bands + source, output);
                }
                values.push(value);
            }
        }
    }
    DenseEigenvectors::from_host_column_major(2 * global_dimension, output_bands, values)
}

/// Invalid second-variation routing, selection, projection, or reconstruction.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SecondVariationError {
    #[error("first-variation window [{start}, {end}) is empty or reversed")]
    EmptyWindow { start: usize, end: usize },
    #[error("first-variation window ends at {end}, but only {band_count} bands exist")]
    WindowOutOfRange { end: usize, band_count: usize },
    #[error("first variation has {eigenvalues} eigenvalues but {eigenvectors} eigenvector columns")]
    FirstVariationCount {
        eigenvalues: usize,
        eigenvectors: usize,
    },
    #[error("second-variation SOC does not accept first-variation route {0:?}")]
    UnsupportedRoute(FirstVariationRoute),
    #[error("first-variation subspace arrays have inconsistent lengths")]
    MalformedSubspace,
    #[error("first-variation eigenvectors have {actual} global rows, expected {expected}")]
    GlobalBasisDimension { expected: usize, actual: usize },
    #[error("received {actual} site SOC blocks, expected {expected}")]
    SiteCount { expected: usize, actual: usize },
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    SocOperator(#[from] SocOperatorError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_basis::{BasisLayout, Provenance};

    fn scalar_vectors() -> DenseEigenvectors {
        DenseEigenvectors::from_host_column_major(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .unwrap()
    }

    #[test]
    fn window_preserves_exact_source_band_mapping() {
        let vectors = DenseEigenvectors::from_host_column_major(
            2,
            3,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.6, 0.0),
                Complex64::new(0.8, 0.0),
            ],
        )
        .unwrap();
        let selected = FirstVariationSubspace::select(
            FirstVariationWindow::new(1, 3).unwrap(),
            &[Hartree(-0.5), Hartree(0.1), Hartree(0.7)],
            &vectors,
        )
        .unwrap();
        assert_eq!(selected.source_bands, vec![1, 2]);
        assert_eq!(selected.eigenvalues, vec![Hartree(0.1), Hartree(0.7)]);
        assert_eq!(selected.eigenvectors.at(1, 0), Complex64::new(1.0, 0.0));
        assert_eq!(selected.eigenvectors.at(0, 1), Complex64::new(0.6, 0.0));
    }

    #[test]
    fn zero_soc_reconstructs_original_scalar_columns_in_both_spins() {
        let scalar = scalar_vectors();
        let solution =
            solve_second_variation_subspace(&[Hartree(-0.2), Hartree(0.4)], &[]).unwrap();
        let doubled = reconstruct_global_spinors(&scalar, &solution.mixing).unwrap();
        let expected_source = [(0, 0), (0, 1), (1, 0), (1, 1)];
        for (output, (source, spin)) in expected_source.into_iter().enumerate() {
            for candidate_spin in 0..2 {
                for global in 0..2 {
                    let expected = if candidate_spin == spin {
                        scalar.at(global, source)
                    } else {
                        Complex64::default()
                    };
                    assert_eq!(doubled.at(candidate_spin * 2 + global, output), expected);
                }
            }
        }
    }

    #[test]
    fn magnetic_and_noncollinear_routes_are_rejected_before_projection() {
        let compiled = CompiledBasis {
            layout: BasisLayout::new(0, Vec::new()),
            plane_waves: Vec::new(),
            site_augmentations: Vec::new(),
            site_geometry: Vec::new(),
            provenance: Provenance::default(),
        };
        let first = FirstVariationSubspace {
            window: FirstVariationWindow::new(0, 2).unwrap(),
            source_bands: vec![0, 1],
            eigenvalues: vec![Hartree(-0.2), Hartree(0.4)],
            eigenvectors: scalar_vectors(),
        };
        for route in [
            FirstVariationRoute::CollinearMagnetic,
            FirstVariationRoute::Noncollinear,
            FirstVariationRoute::SpinorFirstVariation,
        ] {
            assert_eq!(
                solve_spex_second_variation(route, &compiled, &first, &[]).unwrap_err(),
                SecondVariationError::UnsupportedRoute(route)
            );
        }
    }
}
