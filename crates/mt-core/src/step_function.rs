//! Analytic Fourier coefficients of the periodic interstitial indicator.

use crate::{Bohr, GVector, InverseBohr, VolumeBohr3, spherical_bessel_j};
use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

/// A muffin-tin sphere in Cartesian Bohr coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    /// Sphere center in the chosen unit cell.
    pub center: [Bohr; 3],
    /// Muffin-tin radius.
    pub radius: Bohr,
}

/// Invalid geometry for a cell-normalized interstitial coefficient.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum StepFunctionError {
    /// Cell volume must be finite and positive.
    #[error("cell volume must be finite and positive, got {0}")]
    InvalidCellVolume(f64),
    /// Sphere radius must be finite and positive.
    #[error("sphere {index} radius must be finite and positive, got {radius}")]
    InvalidRadius { index: usize, radius: f64 },
    /// Sphere center must contain finite coordinates.
    #[error("sphere {index} center contains a non-finite coordinate")]
    InvalidCenter { index: usize },
    /// Nonoverlapping muffin-tin sphere volume cannot exceed the cell volume.
    #[error("total muffin-tin sphere volume {sphere_volume} exceeds cell volume {cell_volume}")]
    SphereVolumeExceedsCell {
        sphere_volume: f64,
        cell_volume: f64,
    },
    /// Two explicitly supplied spheres overlap in Cartesian space.
    #[error(
        "muffin-tin spheres {first} and {second} overlap: center distance {distance} < radius sum {radius_sum}"
    )]
    OverlappingSpheres {
        first: usize,
        second: usize,
        distance: f64,
        radius_sum: f64,
    },
    /// The reciprocal vector must be finite.
    #[error("reciprocal vector contains a non-finite coordinate")]
    InvalidReciprocalVector,
}

/// Validated cell and muffin-tin spheres defining `Theta_I(r)`.
#[derive(Clone, Debug, PartialEq)]
pub struct InterstitialGeometry {
    cell_volume: VolumeBohr3,
    spheres: Vec<Sphere>,
}

impl InterstitialGeometry {
    /// Validate the geometry used by the closed-form transform.
    pub fn new(
        cell_volume: VolumeBohr3,
        spheres: impl Into<Vec<Sphere>>,
    ) -> Result<Self, StepFunctionError> {
        if !cell_volume.0.is_finite() || cell_volume.0 <= 0.0 {
            return Err(StepFunctionError::InvalidCellVolume(cell_volume.0));
        }
        let spheres = spheres.into();
        let mut sphere_volume = 0.0;
        for (index, sphere) in spheres.iter().enumerate() {
            if !sphere.radius.0.is_finite() || sphere.radius.0 <= 0.0 {
                return Err(StepFunctionError::InvalidRadius {
                    index,
                    radius: sphere.radius.0,
                });
            }
            if sphere.center.iter().any(|x| !x.0.is_finite()) {
                return Err(StepFunctionError::InvalidCenter { index });
            }
            sphere_volume += 4.0 * PI * sphere.radius.0.powi(3) / 3.0;
        }
        for first in 0..spheres.len() {
            for second in first + 1..spheres.len() {
                let distance_squared: f64 = spheres[first]
                    .center
                    .iter()
                    .zip(spheres[second].center)
                    .map(|(left, right)| (left.0 - right.0).powi(2))
                    .sum();
                let distance = distance_squared.sqrt();
                let radius_sum = spheres[first].radius.0 + spheres[second].radius.0;
                if distance < radius_sum * (1.0 - 64.0 * f64::EPSILON) {
                    return Err(StepFunctionError::OverlappingSpheres {
                        first,
                        second,
                        distance,
                        radius_sum,
                    });
                }
            }
        }
        if sphere_volume > cell_volume.0 * (1.0 + 64.0 * f64::EPSILON) {
            return Err(StepFunctionError::SphereVolumeExceedsCell {
                sphere_volume,
                cell_volume: cell_volume.0,
            });
        }
        Ok(Self {
            cell_volume,
            spheres,
        })
    }

    /// Unit-cell volume.
    pub const fn cell_volume(&self) -> VolumeBohr3 {
        self.cell_volume
    }

    /// Muffin-tin spheres in the coefficient sum.
    pub fn spheres(&self) -> &[Sphere] {
        &self.spheres
    }

    /// SPEX-style, cell-normalized Fourier coefficient of the interstitial
    /// indicator at a Cartesian reciprocal vector.
    ///
    /// The Fourier phase is `exp(-i G dot R_a)`. At zero the result is the
    /// interstitial volume fraction; otherwise it is the negative sum of the
    /// excluded-sphere transforms.
    pub fn coefficient(
        &self,
        reciprocal: [InverseBohr; 3],
    ) -> Result<Complex64, StepFunctionError> {
        if reciprocal.iter().any(|x| !x.0.is_finite()) {
            return Err(StepFunctionError::InvalidReciprocalVector);
        }
        let norm_squared = reciprocal.iter().map(|x| x.0 * x.0).sum::<f64>();
        let is_zero = norm_squared == 0.0;
        let norm = norm_squared.sqrt();
        let mut coefficient = if is_zero {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
        for sphere in &self.spheres {
            let volume = 4.0 * PI * sphere.radius.0.powi(3) / 3.0;
            let radial = if is_zero {
                1.0
            } else {
                sphere_form_factor(InverseBohr(norm), sphere.radius)
            };
            let phase = -reciprocal
                .iter()
                .zip(sphere.center)
                .map(|(g, r)| g.0 * r.0)
                .sum::<f64>();
            coefficient -= Complex64::from_polar(volume * radial / self.cell_volume.0, phase);
        }
        Ok(coefficient)
    }

    /// Convenience overload for an enumerated reciprocal vector.
    pub fn coefficient_for_g(&self, g: &GVector) -> Result<Complex64, StepFunctionError> {
        self.coefficient(g.cartesian)
    }
}

/// Normalized transform of a solid sphere, `3 j_1(qR)/(qR)`.
///
/// A series avoids cancellation at the origin and gives the exact limit one.
pub fn sphere_form_factor(q: InverseBohr, radius: Bohr) -> f64 {
    let x = q.0.abs() * radius.0;
    if x < 1e-3 {
        let x2 = x * x;
        1.0 - x2 / 10.0 + x2 * x2 / 280.0 - x2 * x2 * x2 / 15_120.0
    } else {
        3.0 * spherical_bessel_j(1, x) / x
    }
}

/// Required coefficient cutoff for all differences of plane waves with
/// Cartesian norm at most `pw_cutoff`.
pub fn step_function_coefficient_cutoff(pw_cutoff: InverseBohr) -> InverseBohr {
    InverseBohr(2.0 * pw_cutoff.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_coefficient_is_interstitial_fraction() {
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: [Bohr(1.0), Bohr(2.0), Bohr(3.0)],
                radius: Bohr(1.0),
            }],
        )
        .unwrap();
        let actual = geometry.coefficient([InverseBohr(0.0); 3]).unwrap();
        let expected = 1.0 - 4.0 * PI / 300.0;
        assert!((actual.re - expected).abs() < 2e-15);
        assert_eq!(actual.im, 0.0);
    }

    #[test]
    fn opposite_g_vectors_are_complex_conjugates() {
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![Sphere {
                center: [Bohr(0.2), Bohr(-0.7), Bohr(1.1)],
                radius: Bohr(0.8),
            }],
        )
        .unwrap();
        let g = [InverseBohr(0.4), InverseBohr(-1.2), InverseBohr(0.8)];
        let minus_g = g.map(|x| InverseBohr(-x.0));
        assert_eq!(
            geometry.coefficient(minus_g).unwrap(),
            geometry.coefficient(g).unwrap().conj()
        );
    }

    #[test]
    fn explicitly_overlapping_spheres_are_rejected() {
        let result = InterstitialGeometry::new(
            VolumeBohr3(100.0),
            vec![
                Sphere {
                    center: [Bohr(0.0); 3],
                    radius: Bohr(1.0),
                },
                Sphere {
                    center: [Bohr(1.5), Bohr(0.0), Bohr(0.0)],
                    radius: Bohr(1.0),
                },
            ],
        );
        assert!(matches!(
            result,
            Err(StepFunctionError::OverlappingSpheres {
                first: 0,
                second: 1,
                ..
            })
        ));
    }
}
