//! Direct-lattice cell geometry and periodic nearest-image distances.

use crate::GridError;
use muffintin_core::{Bohr, VolumeBohr3};

/// A validated direct-lattice unit cell with its origin at Cartesian zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    basis: [[Bohr; 3]; 3],
    inverse: [[f64; 3]; 3],
    volume: VolumeBohr3,
}

impl Cell {
    /// Construct a cell from direct primitive vectors in Cartesian Bohr.
    pub fn new(basis: [[Bohr; 3]; 3]) -> Result<Self, GridError> {
        if basis
            .iter()
            .flatten()
            .any(|component| !component.0.is_finite())
        {
            return Err(GridError::NonFiniteCell);
        }
        let raw = basis.map(|vector| vector.map(Bohr::get));
        let determinant = dot(raw[0], cross(raw[1], raw[2]));
        let scale = norm(raw[0]) * norm(raw[1]) * norm(raw[2]);
        if scale == 0.0 || determinant.abs() <= 128.0 * f64::EPSILON * scale {
            return Err(GridError::SingularCell);
        }

        // Rows of this inverse turn a Cartesian vector into fractional
        // coefficients of the direct primitive vectors.
        let inverse = [
            scale_vector(cross(raw[1], raw[2]), 1.0 / determinant),
            scale_vector(cross(raw[2], raw[0]), 1.0 / determinant),
            scale_vector(cross(raw[0], raw[1]), 1.0 / determinant),
        ];
        Ok(Self {
            basis,
            inverse,
            volume: VolumeBohr3(determinant.abs()),
        })
    }

    /// Direct primitive vectors.
    pub const fn basis(&self) -> &[[Bohr; 3]; 3] {
        &self.basis
    }

    /// Unit-cell volume.
    pub const fn volume(&self) -> VolumeBohr3 {
        self.volume
    }

    /// Map fractional cell coordinates to Cartesian Bohr coordinates.
    pub fn cartesian(&self, fractional: [f64; 3]) -> [Bohr; 3] {
        let mut result = [Bohr(0.0); 3];
        for (coefficient, vector) in fractional.into_iter().zip(self.basis) {
            for axis in 0..3 {
                result[axis].0 += coefficient * vector[axis].0;
            }
        }
        result
    }

    pub(crate) fn nearest_image_distance_squared(&self, displacement: [f64; 3]) -> f64 {
        let fractional = self.inverse.map(|row| dot(row, displacement));
        let rounded = fractional.map(f64::round);
        let initial = self.image_displacement(displacement, rounded);
        let mut best = dot(initial, initial);
        let radius = best.sqrt();

        // If an image is closer than the current best, each one of its
        // fractional components differs from `fractional` by at most
        // |inverse_row| * radius. These bounds make the search complete even
        // for strongly skewed cells where component-wise rounding is not the
        // Euclidean nearest image.
        let mut lower = [0_i64; 3];
        let mut upper = [0_i64; 3];
        for axis in 0..3 {
            let extent = norm(self.inverse[axis]) * radius + 16.0 * f64::EPSILON;
            lower[axis] = (fractional[axis] - extent).ceil() as i64;
            upper[axis] = (fractional[axis] + extent).floor() as i64;
        }
        for n0 in lower[0]..=upper[0] {
            for n1 in lower[1]..=upper[1] {
                for n2 in lower[2]..=upper[2] {
                    let image =
                        self.image_displacement(displacement, [n0 as f64, n1 as f64, n2 as f64]);
                    best = best.min(dot(image, image));
                }
            }
        }
        best
    }

    fn image_displacement(&self, displacement: [f64; 3], image: [f64; 3]) -> [f64; 3] {
        let translation = self.cartesian(image).map(Bohr::get);
        [
            displacement[0] - translation[0],
            displacement[1] - translation[1],
            displacement[2] - translation[2],
        ]
    }
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn scale_vector(vector: [f64; 3], factor: f64) -> [f64; 3] {
    vector.map(|component| component * factor)
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_image_handles_skew_cell() {
        let cell = Cell::new([
            [Bohr(1.0), Bohr(0.0), Bohr(0.0)],
            [Bohr(0.9), Bohr(0.2), Bohr(0.0)],
            [Bohr(0.0), Bohr(0.0), Bohr(1.0)],
        ])
        .unwrap();
        let displacement = [0.95, 0.1, 0.0];
        let expected = (0.95_f64 - 0.9).powi(2) + (0.1_f64 - 0.2).powi(2);
        assert!((cell.nearest_image_distance_squared(displacement) - expected).abs() < 1e-15);
    }
}
