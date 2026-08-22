//! Site projection $P^\dagger B P$ scattered into a global operator pair.

use crate::OperatorError;
use libmuffintin_basis::{CompiledBasis, PlaneWaveAugmentation};
use libmuffintin_tensor::{
    Axis, ComplexTensor, DenseHermitianMatrix, TensorError, hermitian_congruence,
};
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
        validate_operator_site(
            site_index,
            site,
            &compiled.site_augmentations[site_index],
            compiled,
        )?;
        add_site_projection(
            overlap,
            dimension,
            site_index,
            &compiled.site_augmentations[site_index],
            &site.overlap,
            compiled,
        )?;
        add_site_projection(
            hamiltonian,
            dimension,
            site_index,
            &compiled.site_augmentations[site_index],
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

fn validate_operator_site(
    site_index: usize,
    site: &SiteOperatorBlocks,
    augmentations: &[PlaneWaveAugmentation],
    compiled: &CompiledBasis,
) -> Result<(), OperatorError> {
    let layout = &compiled.layout;
    if augmentations.len() != layout.plane_wave_count() {
        return Err(OperatorError::PlaneWaveCount {
            expected: layout.plane_wave_count(),
            actual: augmentations.len(),
        });
    }
    let channels = augmentations
        .first()
        .map_or(0, |augmentation| augmentation.coefficients.len());
    for (plane_wave, augmentation) in augmentations.iter().enumerate() {
        if augmentation.coefficients.len() != channels {
            return Err(OperatorError::ChannelCount {
                plane_wave,
                expected: channels,
                actual: augmentation.coefficients.len(),
            });
        }
    }
    let expected = 2 * channels
        + layout
            .site_layout(site_index)
            .map_or(0, libmuffintin_basis::LocalOrbitalLayout::len);
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

fn add_site_projection(
    global: &mut [Complex64],
    dimension: usize,
    site_index: usize,
    augmentations: &[PlaneWaveAugmentation],
    block: &DenseHermitianMatrix,
    compiled: &CompiledBasis,
) -> Result<(), OperatorError> {
    let layout = &compiled.layout;
    let channels = augmentations
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
        if global_index < layout.plane_wave_count() {
            for (channel, coefficients) in
                augmentations[global_index].coefficients.iter().enumerate()
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
