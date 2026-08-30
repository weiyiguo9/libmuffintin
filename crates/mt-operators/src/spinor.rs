//! Site projection for the parallel typed spinor basis.

use crate::{CompiledSiteProjection, OperatorError, OperatorSet};
use muffintin_core::RelativisticChannel;
use muffintin_envelope::SpinorCompiledBasis;
use muffintin_tensor::{Axis, DenseHermitianMatrix, TensorError};
use num_complex::Complex64;

/// Both muffin-tin operators in one typed spinor site-coordinate basis.
///
/// Coordinates are all `channels` in the stored order, with the two radial
/// columns `(u, du/dE)` contiguous for each channel, followed by the site's
/// explicit spinor local orbitals. The channel list is checked exactly against
/// every plane-wave augmentation before a congruence is formed.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorSiteOperatorBlocks {
    pub channels: Vec<RelativisticChannel>,
    pub overlap: DenseHermitianMatrix,
    pub hamiltonian: DenseHermitianMatrix,
}

/// Add all spinor-site congruences to interstitial host buffers.
///
/// Site coordinates are ordered as the two radial columns `(u, du/dE)` for
/// every typed `(kappa, mu)` augmentation channel, followed by the site's
/// explicit `(kappa, mu, n)` local orbitals. Global plane-wave columns use
/// `spin * n_g + g`; no scalar `lm` layout is reinterpreted here.
pub fn add_spinor_site_contributions(
    overlap: &mut [Complex64],
    hamiltonian: &mut [Complex64],
    dimension: usize,
    compiled: &SpinorCompiledBasis,
    sites: &[SpinorSiteOperatorBlocks],
) -> Result<OperatorSet, OperatorError> {
    let layout = &compiled.layout;
    if layout.spatial_plane_wave_count() != compiled.plane_waves.len() {
        return Err(OperatorError::SpinorBasisPlaneWaveCount {
            expected: compiled.plane_waves.len(),
            actual: layout.spatial_plane_wave_count(),
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
    if dimension != layout.dimension() {
        return Err(OperatorError::MatrixDimensionMismatch {
            hamiltonian: dimension,
            overlap: layout.dimension(),
        });
    }

    for (site_index, site) in sites.iter().enumerate() {
        let projection = CompiledSiteProjection::spinor(compiled, site_index, &site.channels)?;
        validate_site(site_index, site, projection.coordinate_count())?;
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

fn validate_site(
    site_index: usize,
    site: &SpinorSiteOperatorBlocks,
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
