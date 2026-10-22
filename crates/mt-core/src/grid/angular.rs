//! Weighted Lebedev and user-supplied angular quadratures, plus a Fibonacci fallback.

use crate::GridError;
use std::f64::consts::{PI, TAU};
use std::sync::Mutex;

// lebedev_laikov's C routines use per-rule mutable static scratch variables.
static LEBEDEV_GENERATOR: Mutex<()> = Mutex::new(());

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
    /// Supported positive-weight Lebedev algebraic exactness orders.
    pub const LEBEDEV_ORDERS: &[u32] = &[
        3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95,
        101, 107, 113, 119, 125, 131,
    ];

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

    /// Construct the positive-weight Lebedev rule with exactly this algebraic order.
    ///
    /// Unsupported orders are rejected rather than rounded. Calls to the bundled
    /// C generator are serialized because its routines use mutable static scratch.
    pub fn lebedev(order: u32) -> Result<Self, GridError> {
        let number = match order {
            3 => 6,
            5 => 14,
            7 => 26,
            9 => 38,
            11 => 50,
            15 => 86,
            17 => 110,
            19 => 146,
            21 => 170,
            23 => 194,
            29 => 302,
            31 => 350,
            35 => 434,
            41 => 590,
            47 => 770,
            53 => 974,
            59 => 1202,
            65 => 1454,
            71 => 1730,
            77 => 2030,
            83 => 2354,
            89 => 2702,
            95 => 3074,
            101 => 3470,
            107 => 3890,
            113 => 4334,
            119 => 4802,
            125 => 5294,
            131 => 5810,
            _ => return Err(GridError::UnsupportedLebedevOrder { order }),
        };

        let (x, y, z, weights) = {
            let _guard = LEBEDEV_GENERATOR
                .lock()
                .expect("Lebedev C generator lock poisoned");
            lebedev_laikov::ld_vecs(number)
        };
        let points = x
            .into_iter()
            .zip(y)
            .zip(z)
            .zip(weights)
            .map(|(((x, y), z), weight)| AngularPoint {
                direction: [x, y, z],
                weight: 4.0 * PI * weight,
            })
            .collect::<Vec<_>>();
        Self::new(points)
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
    use crate::real_spherical_harmonics;

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

    #[test]
    fn lebedev_rules_are_positive_normalized_and_exact() {
        for &order in AngularGrid::LEBEDEV_ORDERS {
            let grid = AngularGrid::lebedev(order).unwrap();
            assert!(!grid.is_empty());
            assert!(
                grid.points()
                    .iter()
                    .all(|point| point.weight.is_finite() && point.weight > 0.0)
            );
            let weight_sum: f64 = grid.points().iter().map(|point| point.weight).sum();
            assert!(
                (weight_sum - 4.0 * PI).abs() < 2e-12,
                "order {order}: {weight_sum}"
            );

            let exponent = i32::try_from(order - 1).unwrap();
            let x_moment: f64 = grid
                .points()
                .iter()
                .map(|point| point.weight * point.direction[0].powi(exponent))
                .sum();
            let exact_moment = 4.0 * PI / f64::from(order);
            assert!(
                (x_moment - exact_moment).abs() < 2e-12,
                "order {order}: got {x_moment}, expected {exact_moment}"
            );
        }
    }

    #[test]
    fn production_29_rule_has_orthonormal_harmonics_through_l4() {
        let grid = AngularGrid::lebedev(29).unwrap();
        let harmonics: Vec<_> = grid
            .points()
            .iter()
            .map(|point| real_spherical_harmonics(4, point.direction))
            .collect();

        for left in 0..25 {
            for right in 0..25 {
                let overlap: f64 = grid
                    .points()
                    .iter()
                    .zip(&harmonics)
                    .map(|(point, values)| point.weight * values[left] * values[right])
                    .sum();
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!(
                    (overlap - expected).abs() < 2e-12,
                    "harmonic overlap ({left}, {right}) was {overlap}"
                );
            }
        }
    }

    #[test]
    fn unsupported_lebedev_order_is_rejected_without_rounding() {
        for order in [0, 4, 13, 14, 25, 27, 30, 133] {
            assert_eq!(
                AngularGrid::lebedev(order),
                Err(GridError::UnsupportedLebedevOrder { order })
            );
        }
    }
}
