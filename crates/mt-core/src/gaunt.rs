//! Integer-angular-momentum Wigner 3j and Gaunt coefficients.

use num_complex::Complex64;
use std::f64::consts::PI;

/// Wigner 3j symbol for integer angular momenta, evaluated with Racah's sum.
///
/// Invalid triangles or magnetic quantum numbers return zero. Log-factorials
/// keep the calculation finite well beyond the angular momenta used in LAPW.
pub fn wigner_3j(l1: u32, l2: u32, l3: u32, m1: i32, m2: i32, m3: i32) -> f64 {
    let j1 = i64::from(l1);
    let j2 = i64::from(l2);
    let j3 = i64::from(l3);
    let m1 = i64::from(m1);
    let m2 = i64::from(m2);
    let m3 = i64::from(m3);
    if m1 + m2 + m3 != 0
        || m1.abs() > j1
        || m2.abs() > j2
        || m3.abs() > j3
        || j3 < (j1 - j2).abs()
        || j3 > j1 + j2
    {
        return 0.0;
    }

    // This is the same parametrization of Racah's formula as SPEX
    // src/numerics.f:wigner3j.
    let f1 = j3 - j2 + m1;
    let f2 = j3 - j1 - m2;
    let f3 = j1 + j2 - j3;
    let f4 = j1 - m1;
    let f5 = j2 + m2;
    let t_min = 0_i64.max(-f1).max(-f2);
    let t_max = f3.min(f4).min(f5);
    if t_min > t_max {
        return 0.0;
    }

    let log_prefactor = 0.5
        * (ln_factorial(j1 + j2 - j3) + ln_factorial(j1 - j2 + j3) + ln_factorial(-j1 + j2 + j3)
            - ln_factorial(j1 + j2 + j3 + 1)
            + ln_factorial(j1 + m1)
            + ln_factorial(j1 - m1)
            + ln_factorial(j2 + m2)
            + ln_factorial(j2 - m2)
            + ln_factorial(j3 + m3)
            + ln_factorial(j3 - m3));

    let mut terms = Vec::with_capacity(usize::try_from(t_max - t_min + 1).unwrap_or(0));
    let mut max_log = f64::NEG_INFINITY;
    for t in t_min..=t_max {
        let log_abs = log_prefactor
            - (ln_factorial(t)
                + ln_factorial(f1 + t)
                + ln_factorial(f2 + t)
                + ln_factorial(f3 - t)
                + ln_factorial(f4 - t)
                + ln_factorial(f5 - t));
        max_log = max_log.max(log_abs);
        terms.push((phase(t), log_abs));
    }

    // Kahan sum after common scaling limits cancellation in the Racah sum.
    let mut scaled_sum = 0.0;
    let mut correction = 0.0;
    for (sign, log_abs) in terms {
        let value = sign * (log_abs - max_log).exp();
        let adjusted = value - correction;
        let next = scaled_sum + adjusted;
        correction = (next - scaled_sum) - adjusted;
        scaled_sum = next;
    }
    phase(j1 - j2 - m3) * max_log.exp() * scaled_sum
}

/// SPEX-convention complex Gaunt coefficient.
///
/// This is
/// `integral conj(Y_l1m1) * Y_l2m2 * conj(Y_l3m3) dOmega`, not the
/// occasionally used unconjugated triple product.
pub fn gaunt(l1: u32, l2: u32, l3: u32, m1: i32, m2: i32, m3: i32) -> f64 {
    if m3 != m2 - m1
        || m1.unsigned_abs() > l1
        || m2.unsigned_abs() > l2
        || m3.unsigned_abs() > l3
        || (u64::from(l1) + u64::from(l2) + u64::from(l3)) % 2 != 0
        || l3 < l1.abs_diff(l2)
        || u64::from(l3) > u64::from(l1) + u64::from(l2)
    {
        return 0.0;
    }
    phase(i64::from(m1) + i64::from(m3))
        * (((2.0 * f64::from(l1) + 1.0)
            * (2.0 * f64::from(l2) + 1.0)
            * (2.0 * f64::from(l3) + 1.0)
            / (4.0 * PI))
            .sqrt())
        * wigner_3j(l1, l2, l3, -m1, m2, -m3)
        * wigner_3j(l1, l2, l3, 0, 0, 0)
}

/// Triple-product Gaunt coefficient for the real harmonics in this crate.
///
/// Since real harmonics are their own conjugates, the value is simply
/// `integral R_l1m1 R_l2m2 R_l3m3 dOmega`.
pub fn real_gaunt(l1: u32, l2: u32, l3: u32, m1: i32, m2: i32, m3: i32) -> f64 {
    if m1.unsigned_abs() > l1 || m2.unsigned_abs() > l2 || m3.unsigned_abs() > l3 {
        return 0.0;
    }
    let mut value = Complex64::new(0.0, 0.0);
    for (a, ca) in real_to_complex(m1) {
        for (b, cb) in real_to_complex(m2) {
            for (c, cc) in real_to_complex(m3) {
                value += ca * cb * cc * complex_triple_gaunt(l1, l2, l3, a, b, c);
            }
        }
    }
    debug_assert!(value.im.abs() <= 2e-13 * (1.0 + value.re.abs()));
    value.re
}

fn complex_triple_gaunt(l1: u32, l2: u32, l3: u32, m1: i32, m2: i32, m3: i32) -> f64 {
    if m1 + m2 + m3 != 0 || (u64::from(l1) + u64::from(l2) + u64::from(l3)) % 2 != 0 {
        return 0.0;
    }
    (((2.0 * f64::from(l1) + 1.0) * (2.0 * f64::from(l2) + 1.0) * (2.0 * f64::from(l3) + 1.0)
        / (4.0 * PI))
        .sqrt())
        * wigner_3j(l1, l2, l3, 0, 0, 0)
        * wigner_3j(l1, l2, l3, m1, m2, m3)
}

fn real_to_complex(m: i32) -> Vec<(i32, Complex64)> {
    if m == 0 {
        return vec![(0, Complex64::new(1.0, 0.0))];
    }
    let q_u = m.unsigned_abs();
    let q = i32::try_from(q_u).expect("absolute i32 magnetic quantum number fits i32");
    let inv_sqrt_two = 1.0 / 2.0_f64.sqrt();
    if m > 0 {
        vec![
            (q, Complex64::new(phase(i64::from(q)), 0.0) * inv_sqrt_two),
            (-q, Complex64::new(inv_sqrt_two, 0.0)),
        ]
    } else {
        vec![
            (q, Complex64::new(0.0, inv_sqrt_two)),
            (-q, Complex64::new(0.0, -phase(i64::from(q)) * inv_sqrt_two)),
        ]
    }
}

fn ln_factorial(n: i64) -> f64 {
    debug_assert!(n >= 0);
    (2..=n).map(|x| (x as f64).ln()).sum()
}

const fn phase(exponent: i64) -> f64 {
    if exponent & 1 == 0 { 1.0 } else { -1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_low_order_wigner_values() {
        assert!((wigner_3j(1, 1, 0, 0, 0, 0) + 1.0 / 3.0_f64.sqrt()).abs() < 2e-15);
        assert!((wigner_3j(1, 1, 2, 0, 0, 0) - (2.0 / 15.0_f64).sqrt()).abs() < 2e-15);
    }

    #[test]
    fn symbolic_gaunt_reference_values() {
        let y00 = 1.0 / (4.0 * PI).sqrt();
        assert!((gaunt(0, 0, 0, 0, 0, 0) - y00).abs() < 1e-15);
        assert!((gaunt(1, 1, 0, 1, 1, 0) - y00).abs() < 1e-15);
        assert!((gaunt(1, 1, 2, 0, 0, 0) - 5.0_f64.sqrt() / (5.0 * PI.sqrt())).abs() < 2e-15);
        assert!((gaunt(1, 1, 2, 1, 1, 0) + 5.0_f64.sqrt() / (10.0 * PI.sqrt())).abs() < 2e-15);
    }
}
