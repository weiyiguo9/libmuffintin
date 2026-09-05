//! Shared scalar and spinor maps from global basis to site coordinates.

use crate::OperatorError;
use muffintin_core::RelativisticChannel;
use muffintin_envelope::{CompiledBasis, SpinorCompiledBasis};
use muffintin_tensor::{
    Axis, ComplexTensor, DenseEigenvectors, DenseHermitianMatrix, einsum, hermitian_congruence,
};
use num_complex::Complex64;

/// One compiled `GlobalBasis -> SiteCoordinate` map.
///
/// Rows are the site coordinates used by sphere operators and densities.
/// Columns are all global plane waves followed by the selected site's local
/// orbitals.  APW augmentation coefficients and LO identity rows are formed
/// once and reused for both operator congruences and eigenvector projection.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledSiteProjection {
    matrix: ComplexTensor,
    global_indices: Vec<usize>,
    global_dimension: usize,
}

impl CompiledSiteProjection {
    /// Compile the scalar `(lm, radial column)` site-coordinate map.
    pub fn scalar(compiled: &CompiledBasis, site: usize) -> Result<Self, OperatorError> {
        if compiled.layout.plane_wave_count() != compiled.plane_waves.len() {
            return Err(OperatorError::BasisPlaneWaveCount {
                expected: compiled.plane_waves.len(),
                actual: compiled.layout.plane_wave_count(),
            });
        }
        if compiled.site_augmentations.len() != compiled.site_count() {
            return Err(OperatorError::SiteCount {
                expected: compiled.site_count(),
                actual: compiled.site_augmentations.len(),
            });
        }
        let augmentations =
            compiled
                .site_augmentations
                .get(site)
                .ok_or(OperatorError::SiteIndex {
                    site,
                    site_count: compiled.site_count(),
                })?;
        if augmentations.len() != compiled.layout.plane_wave_count() {
            return Err(OperatorError::PlaneWaveCount {
                expected: compiled.layout.plane_wave_count(),
                actual: augmentations.len(),
            });
        }
        let channel_count = augmentations
            .first()
            .map_or(0, |augmentation| augmentation.coefficients.len());
        for (plane_wave, augmentation) in augmentations.iter().enumerate() {
            if augmentation.coefficients.len() != channel_count {
                return Err(OperatorError::ChannelCount {
                    plane_wave,
                    expected: channel_count,
                    actual: augmentation.coefficients.len(),
                });
            }
        }
        let lo_range = compiled
            .layout
            .site_local_orbital_range(site)
            .expect("augmentation site was checked against the layout");
        let global_indices = compiled
            .layout
            .plane_wave_range()
            .chain(lo_range.clone())
            .collect::<Vec<_>>();
        let coordinate_count = 2 * channel_count + lo_range.len();
        let basis_count = global_indices.len();
        let mut values = vec![Complex64::new(0.0, 0.0); coordinate_count * basis_count];
        for (column, &global_index) in global_indices.iter().enumerate() {
            if global_index < compiled.layout.plane_wave_count() {
                for (channel, coefficients) in
                    augmentations[global_index].coefficients.iter().enumerate()
                {
                    values[(2 * channel) * basis_count + column] = coefficients[0];
                    values[(2 * channel + 1) * basis_count + column] = coefficients[1];
                }
            } else {
                let row = 2 * channel_count + global_index - lo_range.start;
                values[row * basis_count + column] = Complex64::new(1.0, 0.0);
            }
        }
        Self::from_parts(
            coordinate_count,
            global_indices,
            compiled.layout.dimension(),
            values,
        )
    }

    /// Compile the spinor `(kappa,mu, radial column)` site-coordinate map.
    pub fn spinor(
        compiled: &SpinorCompiledBasis,
        site: usize,
        channels: &[RelativisticChannel],
    ) -> Result<Self, OperatorError> {
        let layout = &compiled.layout;
        if layout.spatial_plane_wave_count() != compiled.plane_waves.len() {
            return Err(OperatorError::SpinorBasisPlaneWaveCount {
                expected: compiled.plane_waves.len(),
                actual: layout.spatial_plane_wave_count(),
            });
        }
        if compiled.site_augmentations.len() != compiled.site_count() {
            return Err(OperatorError::SiteCount {
                expected: compiled.site_count(),
                actual: compiled.site_augmentations.len(),
            });
        }
        let augmentations =
            compiled
                .site_augmentations
                .get(site)
                .ok_or(OperatorError::SiteIndex {
                    site,
                    site_count: compiled.site_count(),
                })?;
        if augmentations.len() != layout.spatial_plane_wave_count() {
            return Err(OperatorError::PlaneWaveCount {
                expected: layout.spatial_plane_wave_count(),
                actual: augmentations.len(),
            });
        }
        for (plane_wave, augmentation) in augmentations.iter().enumerate() {
            if augmentation.channels != channels {
                return Err(OperatorError::SpinorChannelLayout { site, plane_wave });
            }
            for spin in 0..2 {
                if augmentation.coefficients[spin].len() != channels.len() {
                    return Err(OperatorError::ChannelCount {
                        plane_wave,
                        expected: channels.len(),
                        actual: augmentation.coefficients[spin].len(),
                    });
                }
            }
        }
        let lo_range = layout
            .site_spinor_range(site)
            .expect("augmentation site was checked against the layout");
        let global_indices = layout
            .plane_wave_range()
            .chain(lo_range.clone())
            .collect::<Vec<_>>();
        let coordinate_count = 2 * channels.len() + lo_range.len();
        let basis_count = global_indices.len();
        let mut values = vec![Complex64::new(0.0, 0.0); coordinate_count * basis_count];
        let n_g = layout.spatial_plane_wave_count();
        for (column, &global_index) in global_indices.iter().enumerate() {
            if global_index < layout.plane_wave_count() {
                let spin = global_index / n_g;
                let g = global_index % n_g;
                for channel in 0..channels.len() {
                    let coefficients = augmentations[g].coefficient(spin, channel);
                    values[(2 * channel) * basis_count + column] = coefficients[0];
                    values[(2 * channel + 1) * basis_count + column] = coefficients[1];
                }
            } else {
                let row = 2 * channels.len() + global_index - lo_range.start;
                values[row * basis_count + column] = Complex64::new(1.0, 0.0);
            }
        }
        Self::from_parts(coordinate_count, global_indices, layout.dimension(), values)
    }

    fn from_parts(
        coordinate_count: usize,
        global_indices: Vec<usize>,
        global_dimension: usize,
        values: Vec<Complex64>,
    ) -> Result<Self, OperatorError> {
        let matrix = ComplexTensor::from_host_row_major(
            &[coordinate_count, global_indices.len()],
            &[Axis::SiteCoordinate, Axis::SiteBasis],
            values,
        )?;
        Ok(Self {
            matrix,
            global_indices,
            global_dimension,
        })
    }

    pub const fn matrix(&self) -> &ComplexTensor {
        &self.matrix
    }

    pub fn global_indices(&self) -> &[usize] {
        &self.global_indices
    }

    pub fn coordinate_count(&self) -> usize {
        self.matrix.shape()[0]
    }

    pub fn basis_count(&self) -> usize {
        self.global_indices.len()
    }

    /// Form the site-coordinate coefficients of every eigenvector column.
    pub fn project_eigenvectors(
        &self,
        eigenvectors: &DenseEigenvectors,
    ) -> Result<SiteOrbitalCoefficients, OperatorError> {
        if eigenvectors.rows() != self.global_dimension {
            return Err(OperatorError::EigenvectorBasisCount {
                expected: self.global_dimension,
                actual: eigenvectors.rows(),
            });
        }
        let band_count = eigenvectors.columns();
        let mut selected = Vec::with_capacity(self.basis_count() * band_count);
        for &global_index in &self.global_indices {
            for band in 0..band_count {
                selected.push(eigenvectors.at(global_index, band));
            }
        }
        let selected = ComplexTensor::from_host_row_major(
            &[self.basis_count(), band_count],
            &[Axis::SiteBasis, Axis::Band],
            selected,
        )?;
        let coefficients = einsum("ci,ib->cb", &[&self.matrix, &selected])?;
        SiteOrbitalCoefficients::from_tensor(coefficients).map_err(Into::into)
    }

    pub(crate) fn add_congruence_to(
        &self,
        global: &mut [Complex64],
        dimension: usize,
        block: &DenseHermitianMatrix,
    ) -> Result<(), OperatorError> {
        let site_matrix = hermitian_congruence(&self.matrix, block)?;
        let values = site_matrix.to_host_row_major();
        let n_basis = self.basis_count();
        for left in 0..n_basis {
            for right in left..n_basis {
                add_hermitian(
                    global,
                    dimension,
                    self.global_indices[left],
                    self.global_indices[right],
                    values[left * n_basis + right],
                );
            }
        }
        Ok(())
    }
}

/// Site-coordinate orbital coefficients, stored as `[SiteCoordinate, Band]`.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteOrbitalCoefficients {
    tensor: ComplexTensor,
}

impl SiteOrbitalCoefficients {
    /// Checked site-coordinate coefficients, including an explicit change of
    /// radial-coordinate normalization performed by the caller.
    pub fn from_tensor(tensor: ComplexTensor) -> Result<Self, muffintin_tensor::TensorError> {
        if tensor.rank() != 2 {
            return Err(muffintin_tensor::TensorError::Rank {
                expected: 2,
                actual: tensor.rank(),
            });
        }
        for (index, expected) in [Axis::SiteCoordinate, Axis::Band].into_iter().enumerate() {
            if tensor.axes()[index] != expected {
                return Err(muffintin_tensor::TensorError::Axis {
                    index,
                    expected,
                    actual: tensor.axes()[index],
                });
            }
        }
        Ok(Self { tensor })
    }

    pub const fn as_tensor(&self) -> &ComplexTensor {
        &self.tensor
    }

    pub fn coordinate_count(&self) -> usize {
        self.tensor.shape()[0]
    }

    pub fn band_count(&self) -> usize {
        self.tensor.shape()[1]
    }

    pub fn at(&self, coordinate: usize, band: usize) -> Complex64 {
        self.tensor.at(&[coordinate, band])
    }

    pub fn to_host_row_major(&self) -> Vec<Complex64> {
        self.tensor.to_host_row_major()
    }
}

/// Compile and apply the scalar site projection to all eigenvector columns.
pub fn project_eigenvectors_to_site(
    compiled: &CompiledBasis,
    site: usize,
    eigenvectors: &DenseEigenvectors,
) -> Result<SiteOrbitalCoefficients, OperatorError> {
    CompiledSiteProjection::scalar(compiled, site)?.project_eigenvectors(eigenvectors)
}

/// Compile and apply the spinor site projection to all eigenvector columns.
pub fn project_spinor_eigenvectors_to_site(
    compiled: &SpinorCompiledBasis,
    site: usize,
    channels: &[RelativisticChannel],
    eigenvectors: &DenseEigenvectors,
) -> Result<SiteOrbitalCoefficients, OperatorError> {
    CompiledSiteProjection::spinor(compiled, site, channels)?.project_eigenvectors(eigenvectors)
}

fn add_hermitian(
    global: &mut [Complex64],
    dimension: usize,
    row: usize,
    column: usize,
    value: Complex64,
) {
    global[row * dimension + column] += value;
    if row == column {
        global[row * dimension + row].im = 0.0;
    } else {
        global[column * dimension + row] += value.conj();
    }
}
