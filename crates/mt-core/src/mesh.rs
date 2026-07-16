//! SPEX exponential radial mesh and seventh-order block quadrature.

use crate::Bohr;
use std::ops::Deref;
use thiserror::Error;

const SIMPSON: [f64; 7] = [41.0, 216.0, 27.0, 272.0, 27.0, 216.0, 41.0];

// Columns of SPEX numerics.f:intgr/intgr_init's Fortran `lagrange(7,6)`.
const LAGRANGE_COLUMNS: [[f64; 7]; 6] = [
    [
        19_087.0, 65_112.0, -46_461.0, 37_504.0, -20_211.0, 6_312.0, -863.0,
    ],
    [
        -863.0, 25_128.0, 46_989.0, -16_256.0, 7_299.0, -2_088.0, 271.0,
    ],
    [
        271.0, -2_760.0, 30_819.0, 37_504.0, -6_771.0, 1_608.0, -191.0,
    ],
    [
        -191.0, 1_608.0, -6_771.0, 37_504.0, 30_819.0, -2_760.0, 271.0,
    ],
    [
        271.0, -2_088.0, 7_299.0, -16_256.0, 46_989.0, 25_128.0, -863.0,
    ],
    [
        -863.0, 6_312.0, -20_211.0, 37_504.0, -46_461.0, 65_112.0, 19_087.0,
    ],
];

/// Precomputed `intgr_init` weights, including the radial Jacobian `r`.
#[derive(Clone, Debug, PartialEq)]
pub struct QuadratureWeights(Vec<f64>);

impl QuadratureWeights {
    /// Borrow the raw coefficient array.
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    /// Number of weights.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are no weights (valid meshes are never empty).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn dot(&self, values: &[f64]) -> f64 {
        self.0.iter().zip(values).map(|(w, f)| w * f).sum()
    }
}

impl Deref for QuadratureWeights {
    type Target = [f64];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Diagnostic for SPEX's extrapolated integral from the origin to `r_0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OriginContribution {
    /// No origin correction is used for inward meshes or sign-changing/tiny data.
    None,
    /// The first two samples define `f(r) = c r^exponent`.
    PowerLaw { exponent: f64, value: f64 },
    /// SPEX's finite fallback when the inferred power is non-integrable.
    NonIntegrableFallback { exponent: f64, value: f64 },
}

impl OriginContribution {
    /// Numeric contribution to add to the weighted grid integral.
    pub const fn value(self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::PowerLaw { value, .. } | Self::NonIntegrableFallback { value, .. } => value,
        }
    }
}

/// Invalid exponential mesh or integrand.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum MeshError {
    /// Seventh-order SPEX quadrature requires a complete seven-point stencil.
    #[error("SPEX quadrature requires at least 7 points, got {0}")]
    TooFewPoints(usize),
    /// The first radius must be finite and positive.
    #[error("first mesh radius must be finite and positive, got {0}")]
    InvalidFirstRadius(f64),
    /// The logarithmic increment must be finite and nonzero.
    #[error("mesh increment must be finite and nonzero, got {0}")]
    InvalidIncrement(f64),
    /// Exponentiation produced a non-finite radius.
    #[error("mesh radius at index {index} is not finite")]
    NonFiniteRadius { index: usize },
    /// Integrand and mesh lengths differ.
    #[error("integrand length {values} does not match mesh length {mesh}")]
    LengthMismatch { mesh: usize, values: usize },
    /// A quadrature sample is not finite.
    #[error("integrand sample at index {index} is not finite: {value}")]
    NonFiniteSample { index: usize, value: f64 },
}

/// Exponential mesh `r_i = r_0 exp(i h)` with SPEX quadrature weights.
#[derive(Clone, Debug, PartialEq)]
pub struct ExponentialMesh {
    first: Bohr,
    increment: f64,
    radii: Vec<Bohr>,
    weights: QuadratureWeights,
}

impl ExponentialMesh {
    /// Construct a mesh and exactly reproduce SPEX `intgr_init` weights.
    pub fn new(first: Bohr, increment: f64, number: usize) -> Result<Self, MeshError> {
        if number < 7 {
            return Err(MeshError::TooFewPoints(number));
        }
        if !first.0.is_finite() || first.0 <= 0.0 {
            return Err(MeshError::InvalidFirstRadius(first.0));
        }
        if !increment.is_finite() || increment == 0.0 {
            return Err(MeshError::InvalidIncrement(increment));
        }
        let mut radii = Vec::with_capacity(number);
        let ratio = increment.exp();
        let mut radius = first.0;
        for index in 0..number {
            if !radius.is_finite() {
                return Err(MeshError::NonFiniteRadius { index });
            }
            radii.push(Bohr(radius));
            radius *= ratio;
        }
        let weights = QuadratureWeights(integration_weights(&radii, increment));
        Ok(Self {
            first,
            increment,
            radii,
            weights,
        })
    }

    /// Number of radial points.
    pub fn len(&self) -> usize {
        self.radii.len()
    }

    /// Whether the mesh has no points (always false for a constructed mesh).
    pub fn is_empty(&self) -> bool {
        self.radii.is_empty()
    }

    /// First positive radius `r_0`.
    pub const fn first(&self) -> Bohr {
        self.first
    }

    /// Constant logarithmic increment `h`.
    pub const fn increment(&self) -> f64 {
        self.increment
    }

    /// Radius at an index, if present.
    pub fn radius(&self, index: usize) -> Option<Bohr> {
        self.radii.get(index).copied()
    }

    /// Last mesh radius.
    pub fn last(&self) -> Bohr {
        *self.radii.last().expect("constructed mesh is nonempty")
    }

    /// All mesh radii.
    pub fn radii(&self) -> &[Bohr] {
        &self.radii
    }

    /// Raw `intgr_init` weights.
    pub fn weights(&self) -> &[f64] {
        self.weights.as_slice()
    }

    /// Strongly typed `intgr_init` weight object.
    pub const fn quadrature_weights(&self) -> &QuadratureWeights {
        &self.weights
    }

    /// Determine the SPEX small-radius extrapolation contribution.
    pub fn origin_contribution(&self, values: &[f64]) -> Result<OriginContribution, MeshError> {
        self.validate_values(values)?;
        if self.increment <= 0.0 || values[0] * values[1] <= 1e-28 {
            return Ok(OriginContribution::None);
        }
        let exponent = (values[1] / values[0]).ln() / self.increment;
        if exponent <= -0.99 {
            Ok(OriginContribution::NonIntegrableFallback {
                exponent,
                value: self.first.0 * values[0] / 2.0,
            })
        } else {
            Ok(OriginContribution::PowerLaw {
                exponent,
                value: self.first.0 * values[0] / (exponent + 1.0),
            })
        }
    }

    /// Integrate over the sampled interval only, exactly as a dot product with
    /// SPEX `intgr_init` weights.
    pub fn integrate_without_origin(&self, values: &[f64]) -> Result<f64, MeshError> {
        self.validate_values(values)?;
        Ok(self.weights.dot(values))
    }

    /// Integrate with SPEX `intgr`: origin power-law contribution plus weights.
    pub fn integrate(&self, values: &[f64]) -> Result<f64, MeshError> {
        let origin = self.origin_contribution(values)?.value();
        Ok(origin + self.weights.dot(values))
    }

    fn validate_values(&self, values: &[f64]) -> Result<(), MeshError> {
        if values.len() != self.len() {
            return Err(MeshError::LengthMismatch {
                mesh: self.len(),
                values: values.len(),
            });
        }
        if let Some((index, &value)) = values.iter().enumerate().find(|(_, x)| !x.is_finite()) {
            return Err(MeshError::NonFiniteSample { index, value });
        }
        Ok(())
    }
}

fn integration_weights(radii: &[Bohr], h: f64) -> Vec<f64> {
    let n = radii.len();
    let n_steps = (n - 1) / 6;
    let mut n0 = n - 6 * n_steps; // Fortran's one-based first Simpson index.
    let mut weights = vec![0.0; n];

    if n0 > 1 {
        for (point, weight) in weights.iter_mut().take(7).enumerate() {
            let coefficient_sum: f64 = LAGRANGE_COLUMNS[..(n0 - 1)]
                .iter()
                .map(|column| column[point])
                .sum();
            *weight = h * radii[point].0 * coefficient_sum / 60_480.0;
        }
    }

    for _ in 0..n_steps {
        let start = n0 - 1;
        for local in 0..7 {
            weights[start + local] += h * radii[start + local].0 * SIMPSON[local] / 140.0;
        }
        n0 += 6;
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_block_is_exact_spex_seven_point_formula() {
        let mesh = ExponentialMesh::new(Bohr(0.2), 0.1, 7).unwrap();
        for (i, &weight) in mesh.weights().iter().enumerate() {
            let expected = 0.1 * mesh.radius(i).unwrap().0 * SIMPSON[i] / 140.0;
            assert_eq!(weight, expected);
        }
    }

    #[test]
    fn power_law_origin_is_analytic() {
        let mesh = ExponentialMesh::new(Bohr(1e-4), 0.01, 601).unwrap();
        let values: Vec<_> = mesh.radii().iter().map(|r| r.0.powi(2)).collect();
        let expected = mesh.last().0.powi(3) / 3.0;
        assert!((mesh.integrate(&values).unwrap() - expected).abs() < 2e-13 * expected);
    }
}
