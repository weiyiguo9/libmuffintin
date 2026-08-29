//! SPEX-compatible LDA/PBE exchange-correlation point evaluation.

use muffintin_core::Hartree;
use std::f64::consts::PI;
use thiserror::Error;

const CLDA: f64 = -0.738_558_766_382_022_3;
const CS: f64 = 0.161_620_459_673_995_5;
const GRADIENT_THRESHOLD: f64 = 1.0e-10;

/// Minimal exchange-correlation choices frozen for M-Kb.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XcFunctional {
    /// SPEX `xlda + cpw92`.
    LdaPw92,
    /// SPEX `xpbe + cpbe`.
    Pbe,
}

/// Spin-resolved density, gradient, and Hessian at one real-space point.
///
/// Hessian components are ordered as `[xx, yy, zz, xy, xz, yz]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityJet2 {
    pub rho: [f64; 2],
    pub gradient: [[f64; 3]; 2],
    pub hessian: [[f64; 6]; 2],
}

/// Exchange-correlation energy per volume and spin potentials at one point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XcPoint {
    /// `rho * epsilon_xc` in Hartree per cubic bohr.
    pub energy_density: f64,
    pub potential: [Hartree; 2],
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum XcError {
    #[error("{field} contains a non-finite value")]
    NonFiniteInput { field: &'static str },
    #[error("spin density {spin} is negative: {value}")]
    NegativeDensity { spin: usize, value: f64 },
    #[error("exchange-correlation evaluation produced a non-finite result")]
    NonFiniteResult,
}

/// Evaluate SPEX LDA/PW92 or PBE exchange-correlation at one point.
pub fn evaluate_xc_point(functional: XcFunctional, jet: DensityJet2) -> Result<XcPoint, XcError> {
    validate_jet(jet)?;
    if jet.rho[0] + jet.rho[1] == 0.0 {
        return Ok(XcPoint {
            energy_density: 0.0,
            potential: [Hartree(0.0); 2],
        });
    }

    let mut energy_density = 0.0;
    let mut potential = [0.0; 2];
    for (spin, &spin_density) in jet.rho.iter().enumerate() {
        if spin_density == 0.0 {
            continue;
        }
        let rho = 2.0 * spin_density;
        let gradient = jet.gradient[spin].map(|value| 2.0 * value);
        let hessian = jet.hessian[spin].map(|value| 2.0 * value);
        let invariants = DifferentialInvariants::new(gradient, hessian);
        let (epsilon, value) = exchange(functional, rho, invariants);
        energy_density += spin_density * epsilon;
        potential[spin] += value;
    }

    if jet.rho.iter().all(|&rho| rho > 0.0) {
        let gradient = add3(jet.gradient[0], jet.gradient[1]);
        let hessian = add6(jet.hessian[0], jet.hessian[1]);
        let invariants = DifferentialInvariants::new(gradient, hessian);
        let (epsilon, values) = correlation(functional, jet.rho, jet.gradient, invariants);
        energy_density += (jet.rho[0] + jet.rho[1]) * epsilon;
        potential[0] += values[0];
        potential[1] += values[1];
    }

    if !energy_density.is_finite() || potential.iter().any(|value| !value.is_finite()) {
        return Err(XcError::NonFiniteResult);
    }
    Ok(XcPoint {
        energy_density,
        potential: potential.map(Hartree),
    })
}

fn validate_jet(jet: DensityJet2) -> Result<(), XcError> {
    for (spin, value) in jet.rho.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(XcError::NonFiniteInput { field: "rho" });
        }
        if value < 0.0 {
            return Err(XcError::NegativeDensity { spin, value });
        }
    }
    if jet
        .gradient
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(XcError::NonFiniteInput { field: "gradient" });
    }
    if jet.hessian.iter().flatten().any(|value| !value.is_finite()) {
        return Err(XcError::NonFiniteInput { field: "hessian" });
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct DifferentialInvariants {
    gradient_norm: f64,
    laplacian: f64,
    gradient_hessian_gradient: f64,
}

impl DifferentialInvariants {
    fn new(gradient: [f64; 3], hessian: [f64; 6]) -> Self {
        let gradient_norm_squared = dot3(gradient, gradient);
        let gradient_norm = gradient_norm_squared.sqrt();
        let laplacian = hessian[0] + hessian[1] + hessian[2];
        let hg = [
            hessian[0] * gradient[0] + hessian[3] * gradient[1] + hessian[4] * gradient[2],
            hessian[3] * gradient[0] + hessian[1] * gradient[1] + hessian[5] * gradient[2],
            hessian[4] * gradient[0] + hessian[5] * gradient[1] + hessian[2] * gradient[2],
        ];
        let gradient_hessian_gradient = if gradient_norm_squared == 0.0 {
            0.0
        } else {
            dot3(gradient, hg) / gradient_norm_squared
        };
        Self {
            gradient_norm,
            laplacian,
            gradient_hessian_gradient,
        }
    }
}

fn exchange(functional: XcFunctional, rho: f64, derivatives: DifferentialInvariants) -> (f64, f64) {
    let mut epsilon = CLDA * rho.cbrt();
    if functional == XcFunctional::LdaPw92 {
        return (epsilon, 4.0 * epsilon / 3.0);
    }
    let sg = CS / rho.powf(4.0 / 3.0);
    let s = sg * derivatives.gradient_norm;
    let mu = 0.219_514_972_764_517_1;
    let kappa = 0.804;
    let denominator = 1.0 + mu * s * s / kappa;
    let enhancement = 1.0 + kappa - kappa / denominator;
    let first = 2.0 * mu * s / denominator.powi(2);
    let second = 2.0 * mu * (1.0 - 3.0 * mu * s * s / kappa) / denominator.powi(3);
    let epsilon_rho = epsilon * (enhancement - first * s) * 4.0 / 3.0;
    let epsilon_gradient = epsilon * first * sg;
    let epsilon_gradient_gradient = epsilon * second * sg * sg;
    let epsilon_gradient_rho = epsilon * second * sg * s * (-4.0 / 3.0) - epsilon_gradient;
    epsilon *= enhancement;
    let potential = variational_potential(
        epsilon_rho,
        epsilon_gradient,
        epsilon_gradient_gradient,
        epsilon_gradient_rho,
        rho,
        derivatives,
    );
    (epsilon, potential)
}

fn correlation(
    functional: XcFunctional,
    rho: [f64; 2],
    spin_gradients: [[f64; 3]; 2],
    derivatives: DifferentialInvariants,
) -> (f64, [f64; 2]) {
    let parameters = CorrelationParameters::new(rho, derivatives.gradient_norm);
    let mut pw92 = pw92_base(parameters);
    if functional == XcFunctional::LdaPw92 {
        let values = correlation_potential(
            pw92,
            CorrelationGradient::default(),
            parameters,
            derivatives,
            spin_gradient_zeta(rho, spin_gradients, derivatives.gradient_norm),
        );
        return (pw92.epsilon, values);
    }
    let gradient = pbe_correlation(&mut pw92, parameters);
    let values = correlation_potential(
        pw92,
        gradient,
        parameters,
        derivatives,
        spin_gradient_zeta(rho, spin_gradients, derivatives.gradient_norm),
    );
    (pw92.epsilon, values)
}

#[derive(Clone, Copy)]
struct CorrelationParameters {
    total: f64,
    rs: f64,
    zeta: f64,
    zeta_plus_root: f64,
    zeta_minus_root: f64,
    t: f64,
    t_gradient_scale: f64,
    phi: f64,
    phi_zeta: f64,
}

impl CorrelationParameters {
    fn new(rho: [f64; 2], gradient_norm: f64) -> Self {
        let total = rho[0] + rho[1];
        let rs = (3.0 / (4.0 * PI * total)).cbrt();
        let zeta = (rho[0] - rho[1]) / total;
        let zeta_plus_root = (1.0 + zeta).cbrt();
        let zeta_minus_root = (1.0 - zeta).cbrt();
        let phi = (zeta_plus_root.powi(2) + zeta_minus_root.powi(2)) / 2.0;
        // Fully-spin-polarized PBE phi(zeta) singularity guard; 1e-2 floor matches reference PBE implementations.
        let phi_zeta =
            (zeta_plus_root.max(1.0e-2).recip() - zeta_minus_root.max(1.0e-2).recip()) / 3.0;
        let t_gradient_scale = (PI / 3.0).powf(1.0 / 6.0) / (4.0 * phi * total.powf(7.0 / 6.0));
        let t = gradient_norm * t_gradient_scale;
        Self {
            total,
            rs,
            zeta,
            zeta_plus_root,
            zeta_minus_root,
            t,
            t_gradient_scale,
            phi,
            phi_zeta,
        }
    }
}

#[derive(Clone, Copy)]
struct CorrelationValue {
    epsilon: f64,
    rs_derivative: f64,
    zeta_derivative: f64,
}

fn pw92_base(parameters: CorrelationParameters) -> CorrelationValue {
    let p = pw92_channel(
        parameters.rs,
        -1.0,
        0.031_090_7,
        0.213_70,
        [7.5957, 3.5876, 1.6382, 0.49294],
    );
    let f = pw92_channel(
        parameters.rs,
        -1.0,
        0.015_545_35,
        0.205_48,
        [14.1189, 6.1977, 3.3662, 0.62517],
    );
    let a = pw92_channel(
        parameters.rs,
        1.0,
        0.016_886_9,
        0.111_25,
        [10.357, 3.6231, 0.88026, 0.49671],
    );
    let denominator = 2.0 * (2.0_f64.cbrt() - 1.0);
    let interpolation = (parameters.zeta_plus_root.powi(4) + parameters.zeta_minus_root.powi(4)
        - 2.0)
        / denominator;
    let interpolation_derivative =
        (parameters.zeta_plus_root - parameters.zeta_minus_root) * (4.0 / 3.0) / denominator;
    let ddf = 4.0 / (9.0 * (2.0_f64.cbrt() - 1.0));
    let zeta4 = parameters.zeta.powi(4);
    let epsilon = p.value
        + a.value * interpolation / ddf * (1.0 - zeta4)
        + (f.value - p.value) * interpolation * zeta4;
    let zeta_derivative = a.value * interpolation_derivative / ddf
        + (f.value - p.value - a.value / ddf)
            * (interpolation_derivative * parameters.zeta + 4.0 * interpolation)
            * parameters.zeta.powi(3);
    let rs_derivative = p.derivative
        + a.derivative * interpolation / ddf * (1.0 - zeta4)
        + (f.derivative - p.derivative) * interpolation * zeta4;
    CorrelationValue {
        epsilon,
        rs_derivative,
        zeta_derivative,
    }
}

#[derive(Clone, Copy)]
struct ChannelValue {
    value: f64,
    derivative: f64,
}

fn pw92_channel(rs: f64, sign: f64, a: f64, a1: f64, b: [f64; 4]) -> ChannelValue {
    let sqrt_rs = rs.sqrt();
    let g = b[0] * sqrt_rs + b[1] * rs + b[2] * sqrt_rs * rs + b[3] * rs * rs;
    let dg = b[0] / (2.0 * sqrt_rs) + b[1] + 1.5 * b[2] * sqrt_rs + 2.0 * b[3] * rs;
    let value = sign * 2.0 * a * (1.0 + a1 * rs) * (1.0 + (2.0 * a * g).recip()).ln();
    let derivative = value * a1 / (1.0 + a1 * rs)
        - sign * 2.0 * a * (1.0 + a1 * rs) * dg / ((1.0 + 2.0 * a * g) * g);
    ChannelValue { value, derivative }
}

#[derive(Clone, Copy, Default)]
struct CorrelationGradient {
    t_derivative: f64,
    t_rs_derivative: f64,
    t_zeta_derivative: f64,
    t_second_derivative: f64,
}

fn pbe_correlation(
    value: &mut CorrelationValue,
    parameters: CorrelationParameters,
) -> CorrelationGradient {
    let beta = 0.066_724_550_603_149_22;
    let gamma = 0.031_090_690_869_654_895;
    let beta_gamma = beta / gamma;
    let phi3 = parameters.phi.powi(3);
    let dphi_phi = parameters.phi_zeta / parameters.phi;
    let exponential = (-value.epsilon / (gamma * phi3)).exp();
    let a = beta_gamma / (exponential - 1.0);
    let af = a * a * exponential / (beta * phi3);
    let a_rs = af * value.rs_derivative;
    let a_zeta = af * (value.zeta_derivative - 3.0 * dphi_phi * value.epsilon);
    let at2 = a * parameters.t * parameters.t;
    let mf = parameters.t * parameters.t * (1.0 + at2);
    let mf_t = 2.0 * parameters.t * (1.0 + 2.0 * at2);
    let mf_rs = parameters.t.powi(4) * a_rs;
    let mf_zeta = parameters.t.powi(4) * a_zeta;
    let m = (1.0 + a * mf) * (1.0 + (a + beta_gamma) * mf);
    let m_rs = (a * mf_rs + a_rs * mf) * (1.0 + (a + beta_gamma) * mf)
        + (1.0 + a * mf) * ((a + beta_gamma) * mf_rs + a_rs * mf);
    let m_zeta = (a * mf_zeta + a_zeta * mf) * (1.0 + (a + beta_gamma) * mf)
        + (1.0 + a * mf) * ((a + beta_gamma) * mf_zeta + a_zeta * mf);
    let m_t = a * mf_t * (1.0 + (a + beta_gamma) * mf) + (1.0 + a * mf) * (a + beta_gamma) * mf_t;
    let correction = gamma * phi3 * (1.0 + beta_gamma * mf / (1.0 + at2 + at2 * at2)).ln();
    value.epsilon += correction;
    let ecf = beta * phi3 / m;
    value.rs_derivative -= ecf * a * parameters.t.powi(6) * (2.0 + at2) * a_rs;
    value.zeta_derivative +=
        -ecf * a * parameters.t.powi(6) * (2.0 + at2) * a_zeta + 3.0 * dphi_phi * correction;
    CorrelationGradient {
        t_derivative: ecf * parameters.t * (2.0 + 4.0 * at2),
        t_rs_derivative: ecf
            * parameters.t
            * ((2.0 + 4.0 * at2) * (-m_rs / m) + 4.0 * parameters.t.powi(2) * a_rs),
        t_zeta_derivative: ecf
            * parameters.t
            * ((2.0 + 4.0 * at2) * (-m_zeta / m + 3.0 * dphi_phi)
                + 4.0 * parameters.t.powi(2) * a_zeta),
        t_second_derivative: ecf
            * ((2.0 + 4.0 * at2) * (-m_t / m * parameters.t + 1.0)
                + 8.0 * a * parameters.t.powi(2)),
    }
}

fn correlation_potential(
    value: CorrelationValue,
    gradient: CorrelationGradient,
    parameters: CorrelationParameters,
    derivatives: DifferentialInvariants,
    gradient_zeta: f64,
) -> [f64; 2] {
    let rs_rho = -parameters.rs / 3.0;
    let t_rho = -7.0 * parameters.t / 6.0;
    let epsilon_gradient = gradient.t_derivative * parameters.t_gradient_scale;
    let epsilon_gradient_gradient =
        gradient.t_second_derivative * parameters.t_gradient_scale.powi(2);
    let epsilon_rho = value.epsilon + value.rs_derivative * rs_rho + gradient.t_derivative * t_rho;
    let epsilon_gradient_rho = (-7.0 * gradient.t_derivative / 6.0
        + gradient.t_rs_derivative * rs_rho
        + gradient.t_second_derivative * t_rho)
        * parameters.t_gradient_scale;
    let common = variational_potential(
        epsilon_rho,
        epsilon_gradient,
        epsilon_gradient_gradient,
        epsilon_gradient_rho,
        parameters.total,
        derivatives,
    );
    let dphi_phi = parameters.phi_zeta / parameters.phi;
    let spin = value.zeta_derivative - dphi_phi * parameters.t * gradient.t_derivative;
    let mut result = [
        common + (1.0 - parameters.zeta) * spin,
        common + (-1.0 - parameters.zeta) * spin,
    ];
    if derivatives.gradient_norm >= GRADIENT_THRESHOLD {
        let correction = (gradient.t_zeta_derivative
            - dphi_phi * (gradient.t_derivative + parameters.t * gradient.t_second_derivative))
            * parameters.t_gradient_scale
            * gradient_zeta;
        result[0] -= correction;
        result[1] -= correction;
    }
    result
}

fn variational_potential(
    local: f64,
    gradient: f64,
    gradient_gradient: f64,
    gradient_rho: f64,
    rho: f64,
    derivatives: DifferentialInvariants,
) -> f64 {
    if derivatives.gradient_norm < GRADIENT_THRESHOLD {
        local - gradient_gradient * rho * derivatives.laplacian
    } else {
        local
            - gradient
                * (derivatives.gradient_norm
                    + rho / derivatives.gradient_norm
                        * (derivatives.laplacian - derivatives.gradient_hessian_gradient))
            - rho * gradient_gradient * derivatives.gradient_hessian_gradient
            - gradient_rho * derivatives.gradient_norm
    }
}

fn spin_gradient_zeta(rho: [f64; 2], gradients: [[f64; 3]; 2], total_gradient_norm: f64) -> f64 {
    if total_gradient_norm == 0.0 {
        return 0.0;
    }
    let total = rho[0] + rho[1];
    let zeta = (rho[0] - rho[1]) / total;
    let total_gradient = add3(gradients[0], gradients[1]);
    let difference_gradient = sub3(gradients[0], gradients[1]);
    (dot3(difference_gradient, total_gradient) - zeta * total_gradient_norm.powi(2))
        / total_gradient_norm
}

fn add3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|index| left[index] + right[index])
}

fn sub3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|index| left[index] - right[index])
}

fn add6(left: [f64; 6], right: [f64; 6]) -> [f64; 6] {
    std::array::from_fn(|index| left[index] + right[index])
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(rho: [f64; 2]) -> DensityJet2 {
        DensityJet2 {
            rho,
            gradient: [[0.0; 3]; 2],
            hessian: [[0.0; 6]; 2],
        }
    }

    #[test]
    fn unpolarized_lda_exchange_has_the_analytic_value() {
        let density = 0.08;
        let result = evaluate_xc_point(XcFunctional::LdaPw92, uniform([density / 2.0; 2])).unwrap();
        let epsilon_x = -0.75 * (3.0 / PI).cbrt() * density.cbrt();
        let correlation = pw92_base(CorrelationParameters::new([density / 2.0; 2], 0.0));
        let correlation_potential = correlation_potential(
            correlation,
            CorrelationGradient::default(),
            CorrelationParameters::new([density / 2.0; 2], 0.0),
            DifferentialInvariants::new([0.0; 3], [0.0; 6]),
            0.0,
        );
        assert!(
            (result.energy_density - density * (epsilon_x + correlation.epsilon)).abs() < 2.0e-15
        );
        assert!(
            (result.potential[0].get() - (4.0 * epsilon_x / 3.0 + correlation_potential[0])).abs()
                < 2.0e-15
        );
        assert_eq!(result.potential[0], result.potential[1]);
    }

    #[test]
    fn pbe_reduces_exactly_to_lda_for_a_uniform_density() {
        let jet = uniform([0.037, 0.019]);
        let lda = evaluate_xc_point(XcFunctional::LdaPw92, jet).unwrap();
        let pbe = evaluate_xc_point(XcFunctional::Pbe, jet).unwrap();
        assert!((lda.energy_density - pbe.energy_density).abs() < 2.0e-15);
        for (left, right) in lda.potential.into_iter().zip(pbe.potential) {
            assert!((left.get() - right.get()).abs() < 2.0e-14);
        }
    }

    #[test]
    fn exchanging_spin_channels_exchanges_only_the_potentials() {
        let jet = DensityJet2 {
            rho: [0.041, 0.023],
            gradient: [[0.01, -0.02, 0.03], [-0.004, 0.008, 0.002]],
            hessian: [
                [0.004, -0.003, 0.002, 0.001, -0.002, 0.0005],
                [-0.002, 0.001, 0.003, -0.0007, 0.0003, 0.0009],
            ],
        };
        let swapped = DensityJet2 {
            rho: [jet.rho[1], jet.rho[0]],
            gradient: [jet.gradient[1], jet.gradient[0]],
            hessian: [jet.hessian[1], jet.hessian[0]],
        };
        let left = evaluate_xc_point(XcFunctional::Pbe, jet).unwrap();
        let right = evaluate_xc_point(XcFunctional::Pbe, swapped).unwrap();
        assert!((left.energy_density - right.energy_density).abs() < 1.0e-14);
        assert!((left.potential[0].get() - right.potential[1].get()).abs() < 1.0e-13);
        assert!((left.potential[1].get() - right.potential[0].get()).abs() < 1.0e-13);
    }

    #[test]
    fn vacuum_and_invalid_input_are_handled_at_the_boundary() {
        assert_eq!(
            evaluate_xc_point(XcFunctional::Pbe, uniform([0.0, 0.0])).unwrap(),
            XcPoint {
                energy_density: 0.0,
                potential: [Hartree(0.0); 2]
            }
        );
        let mut invalid = uniform([0.1, 0.1]);
        invalid.gradient[0][0] = f64::NAN;
        assert_eq!(
            evaluate_xc_point(XcFunctional::Pbe, invalid),
            Err(XcError::NonFiniteInput { field: "gradient" })
        );
        assert_eq!(
            evaluate_xc_point(XcFunctional::LdaPw92, uniform([-0.1, 0.1])),
            Err(XcError::NegativeDensity {
                spin: 0,
                value: -0.1
            })
        );
    }
}
