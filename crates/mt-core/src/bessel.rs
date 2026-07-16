//! Spherical Bessel and Neumann functions and their first derivatives.

use thiserror::Error;

/// Domain errors for the singular spherical Neumann functions.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum BesselError {
    /// `y_l(0)` and its derivative are singular.
    #[error("spherical Neumann functions are singular at zero")]
    ZeroArgument,
    /// The argument must be finite.
    #[error("spherical Bessel argument must be finite, got {0}")]
    NonFiniteArgument(f64),
}

/// Spherical Bessel function of the first kind, `j_l(x)`.
///
/// Small arguments use the convergent power series. The difficult `l > x`
/// region uses a scaled Miller recurrence, while the oscillatory region uses
/// stable upward recurrence.
pub fn spherical_bessel_j(l: u32, x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if l == 0 { 1.0 } else { 0.0 };
    }
    let ax = x.abs();
    let positive = if ax < 0.25 {
        spherical_j_series(l, ax)
    } else if l <= 1 {
        let [j0, j1] = j_bases(ax);
        if l == 0 { j0 } else { j1 }
    } else if ax > f64::from(l) + 12.0 {
        upward_j(l, ax)
    } else {
        miller_j(l, ax)
    };
    if x.is_sign_negative() && l % 2 == 1 {
        -positive
    } else {
        positive
    }
}

/// Derivative `d j_l(x) / dx`.
pub fn spherical_bessel_j_derivative(l: u32, x: f64) -> f64 {
    if x == 0.0 {
        return if l == 1 { 1.0 / 3.0 } else { 0.0 };
    }
    if l == 0 {
        -spherical_bessel_j(1, x)
    } else {
        spherical_bessel_j(l - 1, x) - (f64::from(l) + 1.0) * spherical_bessel_j(l, x) / x
    }
}

/// Spherical Bessel function of the second kind (Neumann function), `y_l(x)`.
pub fn spherical_bessel_y(l: u32, x: f64) -> Result<f64, BesselError> {
    validate_neumann_argument(x)?;
    let ax = x.abs();
    let mut previous = -ax.cos() / ax;
    if l == 0 {
        return Ok(apply_y_parity(l, x, previous));
    }
    let mut current = -ax.cos() / ax.powi(2) - ax.sin() / ax;
    if l == 1 {
        return Ok(apply_y_parity(l, x, current));
    }
    for ll in 1..l {
        let next = f64::from(2 * ll + 1) * current / ax - previous;
        previous = current;
        current = next;
    }
    Ok(apply_y_parity(l, x, current))
}

/// Derivative `d y_l(x) / dx`.
pub fn spherical_bessel_y_derivative(l: u32, x: f64) -> Result<f64, BesselError> {
    validate_neumann_argument(x)?;
    if l == 0 {
        Ok(-spherical_bessel_y(1, x)?)
    } else {
        Ok(spherical_bessel_y(l - 1, x)? - (f64::from(l) + 1.0) * spherical_bessel_y(l, x)? / x)
    }
}

fn validate_neumann_argument(x: f64) -> Result<(), BesselError> {
    if !x.is_finite() {
        Err(BesselError::NonFiniteArgument(x))
    } else if x == 0.0 {
        Err(BesselError::ZeroArgument)
    } else {
        Ok(())
    }
}

fn apply_y_parity(l: u32, x: f64, positive: f64) -> f64 {
    // y_l(-x) = (-1)^(l+1) y_l(x).
    if x.is_sign_negative() && l % 2 == 0 {
        -positive
    } else {
        positive
    }
}

fn j_bases(x: f64) -> [f64; 2] {
    if x.abs() < 0.01 {
        [spherical_j_series(0, x), spherical_j_series(1, x)]
    } else {
        let j0 = x.sin() / x;
        [j0, (j0 - x.cos()) / x]
    }
}

fn upward_j(l: u32, x: f64) -> f64 {
    let [mut previous, mut current] = j_bases(x);
    for ll in 1..l {
        let next = f64::from(2 * ll + 1) * current / x - previous;
        previous = current;
        current = next;
    }
    current
}

fn miller_j(l: u32, x: f64) -> f64 {
    let l = l as usize;
    let start = (l + 50).max(x.ceil() as usize + 50 + (2.0 * x.sqrt()) as usize);
    let mut values = vec![0.0; start + 2];
    values[start] = 1.0;
    for n in (1..=start).rev() {
        values[n - 1] = (2 * n + 1) as f64 * values[n] / x - values[n + 1];
        if values[n - 1].abs() > 1e200 {
            for value in &mut values[(n - 1)..] {
                *value *= 1e-200;
            }
        }
    }
    let [j0, j1] = j_bases(x);
    let scale = if j0.abs() >= j1.abs() {
        j0 / values[0]
    } else {
        j1 / values[1]
    };
    values[l] * scale
}

fn spherical_j_series(l: u32, x: f64) -> f64 {
    let mut leading = 1.0;
    for k in 1..=l {
        leading *= x / f64::from(2 * k + 1);
    }
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..=512_u32 {
        term *= -x * x / (2.0 * f64::from(k) * f64::from(2 * l + 2 * k + 1));
        sum += term;
        if term.abs() <= 2.0 * f64::EPSILON * sum.abs().max(1.0) {
            break;
        }
    }
    leading * sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values_and_zero_limits() {
        assert_eq!(spherical_bessel_j(0, 0.0), 1.0);
        assert_eq!(spherical_bessel_j(4, 0.0), 0.0);
        let x = 1.7;
        assert!((spherical_bessel_j(0, x) - x.sin() / x).abs() < 2e-15);
        assert!((spherical_bessel_y(0, x).unwrap() + x.cos() / x).abs() < 2e-15);
    }

    #[test]
    fn wronskian_is_inverse_x_squared() {
        for l in 0..=20 {
            for x in [0.15, 0.7, 3.0, 15.0, 45.0] {
                let j = spherical_bessel_j(l, x);
                let jp = spherical_bessel_j_derivative(l, x);
                let y = spherical_bessel_y(l, x).unwrap();
                let yp = spherical_bessel_y_derivative(l, x).unwrap();
                let relative = ((j * yp - jp * y) * x * x - 1.0).abs();
                assert!(relative < 2e-12, "l={l}, x={x}, residual={relative:e}");
            }
        }
    }

    #[test]
    fn parity_holds_for_negative_arguments() {
        for l in 0..8 {
            let x = 2.3;
            assert_eq!(
                spherical_bessel_j(l, -x),
                if l % 2 == 0 {
                    spherical_bessel_j(l, x)
                } else {
                    -spherical_bessel_j(l, x)
                }
            );
        }
    }
}
