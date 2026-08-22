//! Site projection for the parallel typed spinor basis.

use crate::{OperatorError, OperatorSet};
use libmuffintin_basis::{SpinorCompiledBasis, SpinorPlaneWaveAugmentation};
use libmuffintin_core::RelativisticChannel;
use libmuffintin_tensor::{
    Axis, ComplexTensor, DenseHermitianMatrix, TensorError, hermitian_congruence,
};
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
        let augmentations = &compiled.site_augmentations[site_index];
        let channel_count = validate_site(site_index, site, augmentations, compiled)?;
        add_site_projection(
            overlap,
            dimension,
            site_index,
            augmentations,
            channel_count,
            &site.overlap,
            compiled,
        )?;
        add_site_projection(
            hamiltonian,
            dimension,
            site_index,
            augmentations,
            channel_count,
            &site.hamiltonian,
            compiled,
        )?;
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
    augmentations: &[SpinorPlaneWaveAugmentation],
    compiled: &SpinorCompiledBasis,
) -> Result<usize, OperatorError> {
    let layout = &compiled.layout;
    if augmentations.len() != layout.spatial_plane_wave_count() {
        return Err(OperatorError::PlaneWaveCount {
            expected: layout.spatial_plane_wave_count(),
            actual: augmentations.len(),
        });
    }
    for (plane_wave, augmentation) in augmentations.iter().enumerate() {
        if augmentation.channels != site.channels {
            return Err(OperatorError::SpinorChannelLayout {
                site: site_index,
                plane_wave,
            });
        }
        for spin in 0..2 {
            if augmentation.coefficients[spin].len() != site.channels.len() {
                return Err(OperatorError::ChannelCount {
                    plane_wave,
                    expected: site.channels.len(),
                    actual: augmentation.coefficients[spin].len(),
                });
            }
        }
    }
    let expected = 2 * site.channels.len()
        + layout
            .site_layout(site_index)
            .map_or(0, libmuffintin_basis::SpinorSiteLayout::len);
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
    Ok(site.channels.len())
}

fn add_site_projection(
    global: &mut [Complex64],
    dimension: usize,
    site_index: usize,
    augmentations: &[SpinorPlaneWaveAugmentation],
    channel_count: usize,
    block: &DenseHermitianMatrix,
    compiled: &SpinorCompiledBasis,
) -> Result<(), OperatorError> {
    let layout = &compiled.layout;
    let apw_dimension = 2 * channel_count;
    let lo_range = layout
        .site_spinor_range(site_index)
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
    let n_g = layout.spatial_plane_wave_count();
    for (column, &global_index) in global_indices.iter().enumerate() {
        if global_index < layout.plane_wave_count() {
            let spin = global_index / n_g;
            let g = global_index % n_g;
            for channel in 0..channel_count {
                let coefficients = augmentations[g].coefficient(spin, channel);
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
    let site_matrix = hermitian_congruence(&projection, block)?;
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

fn add_hermitian(
    global: &mut [Complex64],
    dimension: usize,
    row: usize,
    column: usize,
    value: Complex64,
) {
    global[row * dimension + column] += value;
    if row != column {
        global[column * dimension + row] += value.conj();
    }
}
