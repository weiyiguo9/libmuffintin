//! Site projection $P^\dagger B P$ scattered into a global operator pair.

use crate::{CompiledSiteProjection, OperatorError};
use muffintin_basis::CompiledBasis;
use muffintin_tensor::{Axis, DenseHermitianMatrix, TensorError};
use num_complex::Complex64;

/// Both muffin-tin operators in the same local coordinate basis.
///
/// The block order is all APW radial functions as contiguous `lm` channels
/// `(u, udot)`, followed by all LOs in the compiled site layout. Plane-wave
/// augmentation coefficients live on [`CompiledBasis`], not here.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteOperatorBlocks {
    pub overlap: DenseHermitianMatrix,
    pub hamiltonian: DenseHermitianMatrix,
}

/// The two matrices of one generalized eigenproblem.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatorSet {
    pub overlap: DenseHermitianMatrix,
    pub hamiltonian: DenseHermitianMatrix,
}

/// Add every site congruence onto interstitial host buffers and wrap `S` and `H`.
pub fn add_site_contributions(
    overlap: &mut [Complex64],
    hamiltonian: &mut [Complex64],
    dimension: usize,
    compiled: &CompiledBasis,
    sites: &[SiteOperatorBlocks],
) -> Result<OperatorSet, OperatorError> {
    if compiled.layout.plane_wave_count() != compiled.plane_waves.len() {
        return Err(OperatorError::BasisPlaneWaveCount {
            expected: compiled.plane_waves.len(),
            actual: compiled.layout.plane_wave_count(),
        });
    }
    if sites.len() != compiled.site_count() {
        return Err(OperatorError::BasisSiteCount {
            expected: compiled.site_count(),
            actual: sites.len(),
        });
    }
    if compiled.site_augmentations.len() != sites.len() {
        return Err(OperatorError::SiteCount {
            expected: sites.len(),
            actual: compiled.site_augmentations.len(),
        });
    }
    if overlap.len() != dimension * dimension || hamiltonian.len() != dimension * dimension {
        return Err(OperatorError::MatrixDataLength {
            expected: dimension * dimension,
            actual: overlap.len().min(hamiltonian.len()),
        });
    }
    if dimension != compiled.layout.dimension() {
        return Err(OperatorError::MatrixDimensionMismatch {
            hamiltonian: dimension,
            overlap: compiled.layout.dimension(),
        });
    }

    for (site_index, site) in sites.iter().enumerate() {
        let projection = CompiledSiteProjection::scalar(compiled, site_index)?;
        validate_operator_site(site_index, site, projection.coordinate_count())?;
        projection.add_congruence_to(overlap, dimension, &site.overlap)?;
        projection.add_congruence_to(hamiltonian, dimension, &site.hamiltonian)?;
    }

    Ok(OperatorSet {
        overlap: DenseHermitianMatrix::from_host_row_major(
            dimension,
            Axis::GlobalBasis,
            overlap.to_vec(),
        )?,
        hamiltonian: DenseHermitianMatrix::from_host_row_major(
            dimension,
            Axis::GlobalBasis,
            hamiltonian.to_vec(),
        )?,
    })
}

fn validate_operator_site(
    site_index: usize,
    site: &SiteOperatorBlocks,
    expected: usize,
) -> Result<(), OperatorError> {
    for (name, block) in [
        ("overlap", &site.overlap),
        ("Hamiltonian", &site.hamiltonian),
    ] {
        if block.dimension() != expected {
            return Err(OperatorError::SiteBlockDimension {
                site: site_index,
                matrix: name,
                expected,
                actual: block.dimension(),
            });
        }
        if block.axis() != Axis::SiteCoordinate {
            return Err(OperatorError::Tensor(TensorError::Axis {
                index: 0,
                expected: Axis::SiteCoordinate,
                actual: block.axis(),
            }));
        }
    }
    Ok(())
}
