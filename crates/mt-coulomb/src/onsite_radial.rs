//! Onsite Gamma restriction of the assembled MT Coulomb operator.
//!
//! Inputs are channel-major `b_LM(r) = r rho_LM(r)` and outputs are
//! `u_LM(r) = r V_LM(r)`. This is neither an extended-core kernel nor the
//! complete MT/interstitial operator. All magnetic-channel couplings are kept.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use muffintin_core::{
    ExponentialMesh, InverseBohr, complex_spherical_harmonics, lm_count, lm_from_index,
    spherical_bessel_j,
};
use muffintin_prodbasis::{AuxiliaryPartition, TransferQ};
use num_complex::Complex64;
use thiserror::Error;

use crate::assemble::{smoothed_truncation_kernel, spencer_alavi_kernel};
use crate::expansion::auxiliary_waves;
use crate::math::{i_pow, parity, sfac_table, weinert_gmat};
use crate::{CoulombError, CoulombKernel, CoulombRequest, radial_primitive, structure_constants};

/// Invalid onsite domain or radial payload.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum OnsiteRadialCoulombError {
    #[error(transparent)]
    Coulomb(#[from] CoulombError),
    #[error("onsite Coulomb site {0} is not in the auxiliary partition")]
    Site(usize),
    #[error("onsite Coulomb mesh ends at {mesh}, not the exact MT radius {radius}")]
    MeshRadius { mesh: f64, radius: f64 },
    #[error("onsite Coulomb payload has {actual} samples, expected {expected}")]
    Shape { actual: usize, expected: usize },
    #[error("onsite Coulomb radial sample {index} is not finite")]
    NonFinite { index: usize },
}

#[derive(Clone, Debug)]
struct FourierWave {
    shell: usize,
    weight: f64,
    angular: Vec<Complex64>,
}

/// Compiled onsite, MT-supported Gamma Coulomb action with full angular coupling.
#[derive(Clone, Debug)]
pub struct OnsiteRadialCoulomb {
    request: CoulombRequest,
    mesh: ExponentialMesh,
    site_index: usize,
    l_max: u32,
    volume: f64,
    /// Present for the periodic body (also used by the smoothed kernel).
    regular: Option<Vec<Complex64>>,
    /// Per exact-norm shell, then L, then radial point: r j_L(G r).
    bessel: Vec<Vec<f64>>,
    waves: Vec<FourierWave>,
}

impl OnsiteRadialCoulomb {
    /// Compile the same onsite Gamma kernel selected by `request`.
    ///
    /// The mesh must end exactly at the selected partition sphere radius.
    /// `l_max <= request.lexp()` matches the auxiliary assembly contract.
    pub fn new(
        request: &CoulombRequest,
        partition: &AuxiliaryPartition,
        site_index: usize,
        mesh: &ExponentialMesh,
        l_max: u32,
    ) -> Result<Self, OnsiteRadialCoulombError> {
        let site = partition
            .site(site_index)
            .ok_or(OnsiteRadialCoulombError::Site(site_index))?;
        if mesh.last() != site.radius {
            return Err(OnsiteRadialCoulombError::MeshRadius {
                mesh: mesh.last().get(),
                radius: site.radius.get(),
            });
        }
        if l_max > request.lexp() {
            return Err(CoulombError::AuxiliaryLExceedsLexp {
                l: l_max,
                lexp: request.lexp(),
            }
            .into());
        }
        let volume = request.cell().volume().get();
        let partition_volume = partition.interstitial().cell_volume().get();
        if (volume - partition_volume).abs() > 1.0e-8 * volume.max(partition_volume) {
            return Err(CoulombError::CellVolumeMismatch {
                cell: volume,
                partition: partition_volume,
            }
            .into());
        }
        let gamma =
            TransferQ::from_cartesian([InverseBohr(0.0); 3]).expect("zero is a valid transfer");
        let n_lm = lm_count(l_max);
        let regular = if matches!(request.kernel(), CoulombKernel::SpencerAlaviSphere { .. }) {
            None
        } else {
            let structure = structure_constants(
                request.cell(),
                request.reciprocal(),
                partition,
                gamma,
                request.lexp(),
            )?;
            let sfac = sfac_table((4 * request.lexp() + 4) as usize)?;
            let mut matrix = vec![Complex64::default(); n_lm * n_lm];
            for a in 0..n_lm {
                let left = lm_from_index(a);
                for b in a..n_lm {
                    let right = lm_from_index(b);
                    let value = parity(right.l as i32 + right.m)
                        * weinert_gmat(left.l, left.m, right.l, right.m, &sfac)?
                        * structure.get(
                            site_index,
                            site_index,
                            left.l + right.l,
                            left.m - right.m,
                        )?;
                    matrix[a * n_lm + b] = value;
                    if a != b {
                        matrix[b * n_lm + a] = value.conj();
                    }
                }
            }
            Some(matrix)
        };
        let mut bessel = Vec::new();
        let mut waves = Vec::new();
        let reciprocal = match request.kernel() {
            CoulombKernel::PeriodicWeinert => None,
            CoulombKernel::SpencerAlaviSphere {
                reciprocal_cutoff, ..
            }
            | CoulombKernel::SmoothedSpencerAlaviSphere {
                reciprocal_cutoff, ..
            } => Some(reciprocal_cutoff),
        };
        if let Some(cutoff) = reciprocal {
            let radius = request
                .spencer_alavi_radius()
                .expect("selected sphere")
                .get();
            let mut shells = BTreeMap::new();
            for wave in auxiliary_waves(request, gamma, cutoff)? {
                let norm = wave.q_plus_g_norm.get();
                let shell = *shells.entry(norm.to_bits()).or_insert_with(|| {
                    let index = bessel.len();
                    let values = (0..=l_max)
                        .flat_map(|l| {
                            mesh.radii()
                                .iter()
                                .map(move |r| r.get() * spherical_bessel_j(l, norm * r.get()))
                        })
                        .collect();
                    bessel.push(values);
                    index
                });
                let weight = match request.kernel() {
                    CoulombKernel::SpencerAlaviSphere { .. } => spencer_alavi_kernel(norm, radius),
                    CoulombKernel::SmoothedSpencerAlaviSphere { smoothing, .. } => {
                        smoothed_truncation_kernel(norm, radius, smoothing.get())
                    }
                    CoulombKernel::PeriodicWeinert => unreachable!("no reciprocal correction"),
                };
                let harmonics =
                    complex_spherical_harmonics(l_max, wave.q_plus_g.map(InverseBohr::get));
                let angular = harmonics
                    .into_iter()
                    .enumerate()
                    .map(|(lm, y)| 4.0 * PI / volume.sqrt() * i_pow(lm_from_index(lm).l).conj() * y)
                    .collect();
                waves.push(FourierWave {
                    shell,
                    weight,
                    angular,
                });
            }
        }
        Ok(Self {
            request: request.clone(),
            mesh: mesh.clone(),
            site_index,
            l_max,
            volume,
            regular,
            bessel,
            waves,
        })
    }

    /// Exact muffin-tin radial mesh on which this action is defined.
    pub const fn mesh(&self) -> &ExponentialMesh {
        &self.mesh
    }

    /// Exact source request, retained for pairing with the valence operator.
    pub const fn request(&self) -> &CoulombRequest {
        &self.request
    }

    /// Selected auxiliary-partition site.
    pub const fn site_index(&self) -> usize {
        self.site_index
    }

    /// Largest retained angular channel.
    pub const fn l_max(&self) -> u32 {
        self.l_max
    }

    /// Apply to dense `lm_index`-ordered channels, each containing `mesh.len()` samples.
    /// The returned array has the same layout and contains `r V_LM`.
    pub fn apply(&self, source: &[Complex64]) -> Result<Vec<Complex64>, OnsiteRadialCoulombError> {
        let n = self.mesh.len();
        let n_lm = lm_count(self.l_max);
        if source.len() != n_lm * n {
            return Err(OnsiteRadialCoulombError::Shape {
                actual: source.len(),
                expected: n_lm * n,
            });
        }
        if let Some(index) = source
            .iter()
            .position(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(OnsiteRadialCoulombError::NonFinite { index });
        }
        let mut result = vec![Complex64::default(); source.len()];
        if let Some(regular) = &self.regular {
            let mut moments = Vec::with_capacity(n_lm);
            for (lm, radial) in source.chunks_exact(n).enumerate() {
                let l = lm_from_index(lm).l;
                moments.push(integrate(
                    &self.mesh,
                    radial
                        .iter()
                        .zip(self.mesh.radii())
                        .map(|(value, r)| *value * r.get().powi(l as i32 + 1)),
                )?);
                let real = radial.iter().map(|z| z.re).collect::<Vec<_>>();
                let imag = radial.iter().map(|z| z.im).collect::<Vec<_>>();
                let real = poisson(&self.mesh, l, &real)?;
                let imag = poisson(&self.mesh, l, &imag)?;
                for i in 0..n {
                    result[lm * n + i] = Complex64::new(real[i], imag[i]);
                }
            }
            for a in 0..n_lm {
                let coefficient: Complex64 = regular[a * n_lm..(a + 1) * n_lm]
                    .iter()
                    .zip(&moments)
                    .map(|(w, moment)| w * moment)
                    .sum();
                let l = lm_from_index(a).l;
                for (i, radius) in self.mesh.radii().iter().enumerate() {
                    result[a * n + i] += radius.get().powi(l as i32 + 1) * coefficient;
                }
            }
            self.subtract_gamma_average(source, &mut result)?;
        }
        let mut moments = Vec::with_capacity(self.bessel.len());
        for shell in &self.bessel {
            let mut channels = Vec::with_capacity(n_lm);
            for (lm, radial) in source.chunks_exact(n).enumerate() {
                let l = lm_from_index(lm).l as usize;
                channels.push(integrate(
                    &self.mesh,
                    radial
                        .iter()
                        .zip(&shell[l * n..(l + 1) * n])
                        .map(|(b, j)| b * j),
                )?);
            }
            moments.push(channels);
        }
        // Accumulate the full direction-dependent angular adjoint on each shell.
        let mut coefficients = vec![vec![Complex64::default(); n_lm]; self.bessel.len()];
        for wave in &self.waves {
            let fourier: Complex64 = wave
                .angular
                .iter()
                .zip(&moments[wave.shell])
                .map(|(a, m)| a * m)
                .sum();
            for (coefficient, angular) in coefficients[wave.shell].iter_mut().zip(&wave.angular) {
                *coefficient += angular.conj() * wave.weight * fourier;
            }
        }
        for (shell, coefficients) in self.bessel.iter().zip(coefficients) {
            for (lm, coefficient) in coefficients.into_iter().enumerate() {
                let l = lm_from_index(lm).l as usize;
                for i in 0..n {
                    result[lm * n + i] += shell[l * n + i] * coefficient;
                }
            }
        }
        if let Some(index) = result
            .iter()
            .position(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(OnsiteRadialCoulombError::NonFinite { index });
        }
        Ok(result)
    }

    fn subtract_gamma_average(
        &self,
        source: &[Complex64],
        result: &mut [Complex64],
    ) -> Result<(), CoulombError> {
        let n = self.mesh.len();
        let c_scale = (4.0 * PI / self.volume).sqrt();
        let c = integrate(
            &self.mesh,
            source[..n]
                .iter()
                .zip(self.mesh.radii())
                .map(|(b, r)| b * r.get() * c_scale),
        )?;
        let laplace = integrate(
            &self.mesh,
            source[..n]
                .iter()
                .zip(self.mesh.radii())
                .map(|(b, r)| -b * r.get().powi(3) * c_scale),
        )?;
        let prefactor = 4.0 * PI / 3.0;
        for (i, radius) in self.mesh.radii().iter().enumerate() {
            let r = radius.get();
            result[i] -= prefactor * c_scale * (r * laplace - r.powi(3) * c) / 2.0;
        }
        if self.l_max > 0 {
            let derivative_scale = Complex64::new(0.0, -(4.0 * PI / (3.0 * self.volume)).sqrt());
            for lm in 1..=3 {
                let derivative = integrate(
                    &self.mesh,
                    source[lm * n..(lm + 1) * n]
                        .iter()
                        .zip(self.mesh.radii())
                        .map(|(b, r)| b * r.get().powi(2) * derivative_scale),
                )?;
                for (i, radius) in self.mesh.radii().iter().enumerate() {
                    result[lm * n + i] -=
                        prefactor * derivative_scale.conj() * radius.get().powi(2) * derivative;
                }
            }
        }
        Ok(())
    }
}

fn integrate(
    mesh: &ExponentialMesh,
    values: impl Iterator<Item = Complex64>,
) -> Result<Complex64, CoulombError> {
    let (real, imag): (Vec<_>, Vec<_>) = values.map(|z| (z.re, z.im)).unzip();
    Ok(Complex64::new(
        mesh.integrate(&real)?,
        mesh.integrate(&imag)?,
    ))
}

fn poisson(mesh: &ExponentialMesh, l: u32, source: &[f64]) -> Result<Vec<f64>, CoulombError> {
    let outward = source
        .iter()
        .zip(mesh.radii())
        .map(|(b, r)| b * r.get().powi(l as i32 + 1))
        .collect::<Vec<_>>();
    let inward = source
        .iter()
        .zip(mesh.radii())
        .map(|(b, r)| b / r.get().powi(l as i32))
        .collect::<Vec<_>>();
    let outward = radial_primitive(mesh, &outward, false)?;
    let inward = radial_primitive(mesh, &inward, true)?;
    let scale = 4.0 * PI / (2 * l + 1) as f64;
    Ok(mesh
        .radii()
        .iter()
        .enumerate()
        .map(|(i, r)| {
            scale * (outward[i] / r.get().powi(l as i32) + inward[i] * r.get().powi(l as i32 + 1))
        })
        .collect())
}
