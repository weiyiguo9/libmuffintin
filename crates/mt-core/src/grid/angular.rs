//! User-supplied angular quadratures and a deterministic Fibonacci fallback.

use crate::GridError;
use std::f64::consts::{PI, TAU};

/// A direction on the unit sphere and its solid-angle quadrature weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularPoint {
    /// Cartesian unit direction.
    pub direction: [f64; 3],
    /// Solid-angle weight, normalized so all weights sum to `4*pi`.
    pub weight: f64,
}

/// A validated angular quadrature rule on the unit sphere.
#[derive(Clone, Debug, PartialEq)]
pub struct AngularGrid {
    points: Vec<AngularPoint>,
}

impl AngularGrid {
    /// Validate a user-supplied angular rule.
    pub fn new(points: impl Into<Vec<AngularPoint>>) -> Result<Self, GridError> {
        let points = points.into();
        if points.is_empty() {
            return Err(GridError::EmptyAngularGrid);
        }

        let mut weight_sum = 0.0;
        for (index, point) in points.iter().enumerate() {
            if point
                .direction
                .iter()
                .any(|component| !component.is_finite())
            {
                return Err(GridError::InvalidAngularDirection { index });
            }
            let norm = point
                .direction
                .iter()
                .map(|component| component * component)
                .sum::<f64>()
                .sqrt();
            if (norm - 1.0).abs() > 256.0 * f64::EPSILON {
                return Err(GridError::NonUnitAngularDirection { index, norm });
            }
            if !point.weight.is_finite() || point.weight <= 0.0 {
                return Err(GridError::InvalidAngularWeight {
                    index,
                    weight: point.weight,
                });
            }
            weight_sum += point.weight;
        }

        let expected = 2.0 * TAU;
        if (weight_sum - expected).abs() > 1e-12 * expected {
            return Err(GridError::InvalidAngularWeightSum { weight_sum });
        }
        Ok(Self { points })
    }

    /// Construct a deterministic equal-weight Fibonacci rule.
    pub fn fibonacci(number: usize) -> Result<Self, GridError> {
        if number == 0 {
            return Err(GridError::EmptyAngularGrid);
        }
        let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
        let weight = 2.0 * TAU / number as f64;
        let points: Vec<_> = (0..number)
            .map(|index| {
                let z = 1.0 - 2.0 * (index as f64 + 0.5) / number as f64;
                let radius = (1.0 - z * z).sqrt();
                let phi = golden_angle * index as f64;
                AngularPoint {
                    direction: [radius * phi.cos(), radius * phi.sin(), z],
                    weight,
                }
            })
            .collect();
        Self::new(points)
    }

    /// Angular points in rule order.
    pub fn points(&self) -> &[AngularPoint] {
        &self.points
    }

    /// Number of angular points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the rule is empty (validated rules never are).
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fibonacci_rule_is_normalized_and_deterministic() {
        let first = AngularGrid::fibonacci(101).unwrap();
        let second = AngularGrid::fibonacci(101).unwrap();
        assert_eq!(first, second);
        let sum: f64 = first.points().iter().map(|point| point.weight).sum();
        assert!((sum - 4.0 * PI).abs() < 2e-13);
        for point in first.points() {
            let norm_squared: f64 = point.direction.iter().map(|x| x * x).sum();
            assert!((norm_squared - 1.0).abs() < 4e-15);
        }
    }

    #[test]
    fn supplied_rule_is_validated() {
        let octahedron = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
        .map(|direction| AngularPoint {
            direction,
            weight: 2.0 * PI / 3.0,
        });
        assert_eq!(AngularGrid::new(octahedron.to_vec()).unwrap().len(), 6);
    }
}
