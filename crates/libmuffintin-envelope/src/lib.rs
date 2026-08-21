//! Plane-wave envelope evaluation.
//!
//! Plane waves use `exp(+i (k+G) dot r) / sqrt(Omega)`. The Rayleigh
//! coefficient and the site-translation phase are exposed separately so that
//! neither convention is hidden inside a radial boundary solve.

#![forbid(unsafe_code)]

use libmuffintin_core::{Bohr, GVector, InverseBohr, Lm, VolumeBohr3, complex_spherical_harmonics};
use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

/// Envelope evaluation error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum EnvelopeError {
    #[error("cell volume must be finite and positive, got {0}")]
    InvalidCellVolume(f64),
    #[error("wave-vector norm must be finite and nonnegative, got {0}")]
    InvalidWaveVector(f64),
}

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

/// Owned plane-wave set for one envelope.
///
/// v0.2 has this concrete envelope only. There is no envelope trait family.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveEnvelope {
    waves: Vec<PlaneWave>,
}

impl PlaneWaveEnvelope {
    /// Take ownership of a plane-wave list.
    pub fn new(waves: impl Into<Vec<PlaneWave>>) -> Self {
        Self {
            waves: waves.into(),
        }
    }

    pub fn waves(&self) -> &[PlaneWave] {
        &self.waves
    }

    pub fn len(&self) -> usize {
        self.waves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.waves.is_empty()
    }
}

/// `4 pi i^l conj(Y_lm(qhat)) / sqrt(Omega)` for the `exp(+i q dot r)`
/// Rayleigh expansion.
///
/// This coefficient contains no site phase and no radial matching
/// coefficient.  At `q=0`, `libmuffintin-core`'s deterministic direction
/// convention leaves only the `l=m=0` channel nonzero.
pub fn rayleigh_coefficient(
    lm: Lm,
    q: [InverseBohr; 3],
    cell_volume: VolumeBohr3,
) -> Result<Complex64, EnvelopeError> {
    if !cell_volume.get().is_finite() || cell_volume.get() <= 0.0 {
        return Err(EnvelopeError::InvalidCellVolume(cell_volume.get()));
    }
    if let Some(component) = q
        .iter()
        .map(|component| component.get())
        .find(|x| !x.is_finite())
    {
        return Err(EnvelopeError::InvalidWaveVector(component));
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

pub(crate) fn i_pow(l: u32) -> Complex64 {
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
    use libmuffintin_core::Lm;

    #[test]
    fn origin_site_phase_is_unity() {
        let q = [InverseBohr(0.3), InverseBohr(-0.2), InverseBohr(0.1)];
        let phase = site_translation_phase(q, [Bohr(0.0); 3]);
        assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1.0e-15);
    }

    #[test]
    fn q_zero_rayleigh_is_only_s_channel() {
        let volume = VolumeBohr3(8.0);
        let q = [InverseBohr(0.0); 3];
        let s = rayleigh_coefficient(Lm::new(0, 0).unwrap(), q, volume).unwrap();
        assert!(s.norm() > 0.0);
        let p = rayleigh_coefficient(Lm::new(1, 0).unwrap(), q, volume).unwrap();
        assert!(p.norm() < 1.0e-14);
    }

    #[test]
    fn envelope_owns_its_plane_waves() {
        let lattice = libmuffintin_core::ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        let wave = PlaneWave::new(
            [InverseBohr(0.1); 3],
            lattice.enumerate(InverseBohr(0.0)).unwrap()[0],
        );
        let envelope = PlaneWaveEnvelope::new(vec![wave]);
        assert_eq!(envelope.len(), 1);
        assert_eq!(envelope.waves()[0], wave);
        assert!(!envelope.is_empty());
    }
}
