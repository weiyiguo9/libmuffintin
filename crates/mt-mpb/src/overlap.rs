//! Shared sample-vector overlap, $L=0$ constant, and Löwdin retained modes.
//!
//! Scalar mixed products and Dirac PP/QQ products both reduce to real sample
//! vectors on one site mesh. This module owns that algebra; it does not know
//! about `ProductRadialId` or `DiracRadialId`.

use crate::MpbError;
use muffintin_auxiliary_ir::{ChannelSpectrum, CutoffRecord, MtAuxiliaryMode};
use muffintin_core::ExponentialMesh;
use muffintin_operators::solve_real_symmetric;

/// Keep strictly positive eigenvalues that are not below the SPEX threshold.
///
/// SPEX drops `eig < tolerance*nspin` (`mixedbasis.f:463`), so equality is kept.
pub(crate) fn retain_overlap_eigenvalue(eigenvalue: f64, threshold: f64) -> bool {
    eigenvalue > 0.0 && eigenvalue >= threshold
}

/// $L=0$ constant projection and per-product $L^2$ normalization.
pub(crate) fn product_channel_functions<'a, I>(
    mesh: &ExponentialMesh,
    l: u32,
    samples: I,
) -> Result<Vec<Vec<f64>>, MpbError>
where
    I: IntoIterator<Item = &'a [f64]>,
{
    let radius = mesh.last().get();
    let constant_norm = (radius.powi(3) / 3.0).sqrt();
    let mut functions = Vec::new();
    for product in samples {
        let mut values = product.to_vec();
        if l == 0 {
            let projection_integrand = mesh
                .radii()
                .iter()
                .zip(&values)
                .map(|(radius, sample)| sample * radius.get())
                .collect::<Vec<_>>();
            let projection = mesh.integrate(&projection_integrand)? / constant_norm;
            for (sample, radius) in values.iter_mut().zip(mesh.radii()) {
                *sample -= projection * radius.get() / constant_norm;
            }
        }
        let norm_sq =
            mesh.integrate(&values.iter().map(|value| value * value).collect::<Vec<_>>())?;
        let scale = norm_sq.max(0.0).sqrt();
        if scale > 0.0 {
            for sample in &mut values {
                *sample /= scale;
            }
        }
        functions.push(values);
    }
    Ok(functions)
}

/// Real-symmetric overlap eigensolve of one $(site, L)$ product-function list.
pub(crate) fn overlap_spectrum(
    site: usize,
    l: u32,
    mesh: &ExponentialMesh,
    functions: &[Vec<f64>],
) -> Result<ChannelSpectrum, MpbError> {
    let n = functions.len();
    if n == 0 {
        return Err(MpbError::EmptyChannel { site, l });
    }
    let mut overlaps = vec![0.0; n * n];
    for row in 0..n {
        for column in row..n {
            let integrand = functions[row]
                .iter()
                .zip(&functions[column])
                .map(|(left, right)| left * right)
                .collect::<Vec<_>>();
            let value = mesh.integrate(&integrand)?;
            overlaps[row * n + column] = value;
            overlaps[column * n + row] = value;
        }
    }
    let solution = solve_real_symmetric(n, |row, column| overlaps[row * n + column])?;
    Ok(ChannelSpectrum {
        site,
        l,
        eigenvalues: solution.eigenvalues,
        eigenvectors: solution.eigenvectors,
    })
}

/// Löwdin-orthonormal retained muffin-tin modes, with the $L=0$ constant first.
pub(crate) fn lowdin_modes(
    l: u32,
    mesh: &ExponentialMesh,
    functions: &[Vec<f64>],
    spectrum: &ChannelSpectrum,
    cutoff: Option<&CutoffRecord>,
) -> Result<Vec<MtAuxiliaryMode>, MpbError> {
    let n = functions.len();
    let threshold = cutoff
        .map(|record| record.value * record.nspin_factor)
        .unwrap_or(0.0);
    let mut kept = Vec::new();
    for (index, &eigenvalue) in spectrum.eigenvalues.iter().enumerate() {
        if retain_overlap_eigenvalue(eigenvalue, threshold) {
            kept.push(index);
        }
    }
    if kept.is_empty() {
        return Err(MpbError::EmptyRetainedChannel {
            site: spectrum.site,
            l,
        });
    }
    let n_mesh = functions[0].len();
    let mut transformed = vec![vec![0.0; n_mesh]; kept.len()];
    for (kept_index, &column) in kept.iter().enumerate() {
        let scale = 1.0 / spectrum.eigenvalues[column].sqrt();
        for (basis, function) in functions.iter().enumerate() {
            let coefficient = spectrum.eigenvectors[basis + column * n] * scale;
            for (sample, value) in transformed[kept_index].iter_mut().zip(function) {
                *sample += coefficient * value;
            }
        }
    }
    let mut modes = Vec::new();
    let mut n_aux = 0;
    if l == 0 {
        let radius = mesh.last().get();
        let constant_norm = (radius.powi(3) / 3.0).sqrt();
        modes.push(MtAuxiliaryMode {
            l: 0,
            n: 0,
            radial: mesh
                .radii()
                .iter()
                .map(|sample| sample.get() / constant_norm)
                .collect(),
        });
        n_aux = 1;
    }
    for radial in transformed {
        modes.push(MtAuxiliaryMode {
            l,
            n: n_aux,
            radial,
        });
        n_aux += 1;
    }
    Ok(modes)
}
