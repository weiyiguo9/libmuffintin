//! Condon--Shortley complex and real spherical harmonics.

use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

/// A validated angular-momentum pair.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Lm {
    /// Angular momentum `l`.
    pub l: u32,
    /// Magnetic quantum number `m`, with `-l <= m <= l`.
    pub m: i32,
}

/// Invalid angular-momentum input.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum LmError {
    /// The magnetic quantum number lies outside its channel.
    #[error("magnetic quantum number m={m} is outside [-{l}, {l}]")]
    MagneticQuantumNumber { l: u32, m: i32 },
}

impl Lm {
    /// Validate an `(l,m)` pair.
    pub fn new(l: u32, m: i32) -> Result<Self, LmError> {
        if m.unsigned_abs() <= l {
            Ok(Self { l, m })
        } else {
            Err(LmError::MagneticQuantumNumber { l, m })
        }
    }

    /// Return the zero-based, channel-contiguous index `l(l+1)+m`.
    pub fn index(self) -> usize {
        let l = self.l as usize;
        (l * (l + 1)).wrapping_add_signed(self.m as isize)
    }
}

/// Number of `(l,m)` entries through `l_max` inclusive.
pub const fn lm_count(l_max: u32) -> usize {
    let n = l_max as usize + 1;
    n * n
}

/// Zero-based, channel-contiguous index `l(l+1)+m`.
pub fn lm_index(l: u32, m: i32) -> Result<usize, LmError> {
    Lm::new(l, m).map(Lm::index)
}

/// Inverse of [`lm_index`]. Every nonnegative index is valid.
pub fn lm_from_index(index: usize) -> Lm {
    // Floating square root is only an initial estimate; integer corrections
    // make the mapping exact even above f64's contiguous-integer range.
    let mut l = (index as f64).sqrt() as usize;
    while (l + 1).checked_mul(l + 1).is_some_and(|x| x <= index) {
        l += 1;
    }
    while l * l > index {
        l -= 1;
    }
    let m = index as i128 - (l as i128 * (l as i128 + 1));
    Lm {
        l: u32::try_from(l).expect("lm index exceeds supported u32 angular momentum"),
        m: i32::try_from(m).expect("lm index exceeds supported i32 magnetic quantum number"),
    }
}

/// Evaluate the normalized complex `Y_lm(theta,phi)`.
///
/// The associated Legendre polynomial contains the Condon--Shortley `(-1)^m`
/// phase. Angles are in radians.
pub fn complex_spherical_harmonic(
    l: u32,
    m: i32,
    theta: f64,
    phi: f64,
) -> Result<Complex64, LmError> {
    Lm::new(l, m)?;
    let x = theta.cos().clamp(-1.0, 1.0);
    let abs_m = m.unsigned_abs();
    let positive = positive_m_harmonic(l, abs_m, x, phi);
    if m >= 0 {
        Ok(positive)
    } else {
        Ok(positive.conj() * parity(abs_m))
    }
}

/// Evaluate all complex harmonics through `l_max` for a Cartesian direction.
///
/// Results use [`lm_index`] order. At the zero vector, only `Y_00` is set;
/// this matches SPEX's deterministic convention for an undefined direction.
pub fn complex_spherical_harmonics(l_max: u32, direction: [f64; 3]) -> Vec<Complex64> {
    let mut result = vec![Complex64::new(0.0, 0.0); lm_count(l_max)];
    result[0] = Complex64::new(0.5 / PI.sqrt(), 0.0);
    let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
    if l_max == 0 || norm <= 16.0 * f64::EPSILON {
        return result;
    }
    let cos_theta = (direction[2] / norm).clamp(-1.0, 1.0);
    let phi = direction[1].atan2(direction[0]);
    for l in 1..=l_max {
        for m in -(l as i32)..=l as i32 {
            let abs_m = m.unsigned_abs();
            let positive = positive_m_harmonic(l, abs_m, cos_theta, phi);
            result[lm_index(l, m).expect("loop bounds validate m")] = if m >= 0 {
                positive
            } else {
                positive.conj() * parity(abs_m)
            };
        }
    }
    result
}

/// Evaluate the real tesseral harmonic identified by the same signed `m`.
///
/// `m>0` is cosine-like and `m<0` is sine-like. Thus the `l=1` channels in
/// order `m=-1,0,1` are proportional to `y,z,x`.
pub fn real_spherical_harmonic(l: u32, m: i32, theta: f64, phi: f64) -> Result<f64, LmError> {
    Lm::new(l, m)?;
    if m == 0 {
        return Ok(complex_spherical_harmonic(l, 0, theta, phi)?.re);
    }
    let q = m.unsigned_abs();
    let y = complex_spherical_harmonic(l, i32::try_from(q).expect("q fits i32"), theta, phi)?;
    Ok(if m > 0 {
        2.0_f64.sqrt() * parity(q) * y.re
    } else {
        -2.0_f64.sqrt() * y.im
    })
}

/// Evaluate all real harmonics through `l_max` in [`lm_index`] order.
pub fn real_spherical_harmonics(l_max: u32, direction: [f64; 3]) -> Vec<f64> {
    let complex = complex_spherical_harmonics(l_max, direction);
    let mut real = vec![0.0; complex.len()];
    for l in 0..=l_max {
        real[lm_index(l, 0).expect("m=0 is valid")] = complex[lm_index(l, 0).unwrap()].re;
        for q in 1..=l {
            let y = complex[lm_index(l, i32::try_from(q).expect("q fits i32")).unwrap()];
            real[lm_index(l, i32::try_from(q).unwrap()).unwrap()] =
                2.0_f64.sqrt() * parity(q) * y.re;
            real[lm_index(l, -i32::try_from(q).unwrap()).unwrap()] = -2.0_f64.sqrt() * y.im;
        }
    }
    real
}

fn positive_m_harmonic(l: u32, m: u32, cos_theta: f64, phi: f64) -> Complex64 {
    let p = associated_legendre(l, m, cos_theta);
    let factorial_ratio = ((l - m + 1)..=(l + m)).fold(1.0, |acc, n| acc / f64::from(n));
    let normalization = ((f64::from(2 * l + 1) / (4.0 * PI)) * factorial_ratio).sqrt();
    Complex64::from_polar(normalization * p, f64::from(m) * phi)
}

fn associated_legendre(l: u32, m: u32, x: f64) -> f64 {
    debug_assert!(m <= l);
    let mut p_mm = 1.0;
    if m > 0 {
        let root = (1.0 - x * x).max(0.0).sqrt();
        let mut factor = 1.0;
        for _ in 1..=m {
            p_mm *= -factor * root;
            factor += 2.0;
        }
    }
    if l == m {
        return p_mm;
    }
    let p_m1_m = f64::from(2 * m + 1) * x * p_mm;
    if l == m + 1 {
        return p_m1_m;
    }
    let mut previous = p_mm;
    let mut current = p_m1_m;
    for ll in (m + 2)..=l {
        let next = (f64::from(2 * ll - 1) * x * current - f64::from(ll + m - 1) * previous)
            / f64::from(ll - m);
        previous = current;
        current = next;
    }
    current
}

pub(crate) const fn parity(n: u32) -> f64 {
    if n % 2 == 0 { 1.0 } else { -1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lm_order_is_channel_contiguous() {
        let expected = [
            (0, 0),
            (1, -1),
            (1, 0),
            (1, 1),
            (2, -2),
            (2, -1),
            (2, 0),
            (2, 1),
            (2, 2),
        ];
        for (index, &(l, m)) in expected.iter().enumerate() {
            assert_eq!(lm_index(l, m), Ok(index));
            assert_eq!(lm_from_index(index), Lm { l, m });
        }
    }

    #[test]
    fn p_orbitals_have_documented_real_orientation() {
        let c = (3.0 / (4.0 * PI)).sqrt();
        let along_x = real_spherical_harmonics(1, [1.0, 0.0, 0.0]);
        let along_y = real_spherical_harmonics(1, [0.0, 1.0, 0.0]);
        assert!((along_x[lm_index(1, 1).unwrap()] - c).abs() < 2e-15);
        assert!((along_y[lm_index(1, -1).unwrap()] - c).abs() < 2e-15);
    }
}
