//! Optional conversion of grids to RSTSR tensors.

use crate::Grid;
use ::rstsr::prelude::Tensor;

/// Conversion of a grid's scalar coordinate and weight values to RSTSR tensors.
///
/// Coordinates are expressed in Bohr and weights in Bohr cubed. Both tensors
/// own their data, so they are independent of the source grid after creation.
pub trait RstsrGridExt: Grid {
    /// Cartesian positions as a row-major tensor with shape `(N, 3)`.
    fn positions_tensor(&self) -> Tensor<f64> {
        let values = self
            .points()
            .iter()
            .flat_map(|point| point.position.map(|coordinate| coordinate.0))
            .collect::<Vec<_>>();
        ::rstsr::prelude::rt::asarray((values, [self.len(), 3]))
    }

    /// Volume weights as a tensor with shape `(N,)`.
    fn weights_tensor(&self) -> Tensor<f64> {
        let values = self
            .points()
            .iter()
            .map(|point| point.weight.0)
            .collect::<Vec<_>>();
        ::rstsr::prelude::rt::asarray(values)
    }
}

impl<T: Grid + ?Sized> RstsrGridExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cell, UniformGrid};
    use muffintin_core::Bohr;

    #[test]
    fn grid_tensors_preserve_shapes_and_point_order() {
        let cell = Cell::new([
            [Bohr(2.0), Bohr(0.0), Bohr(0.0)],
            [Bohr(0.0), Bohr(4.0), Bohr(0.0)],
            [Bohr(0.0), Bohr(0.0), Bohr(6.0)],
        ])
        .unwrap();
        let grid = UniformGrid::new(cell, [2, 1, 1]).unwrap();

        let positions = grid.positions_tensor();
        let weights = grid.weights_tensor();

        assert_eq!(positions.shape(), &[2, 3]);
        assert_eq!(positions[[0, 0]], 0.5);
        assert_eq!(positions[[0, 1]], 2.0);
        assert_eq!(positions[[0, 2]], 3.0);
        assert_eq!(positions[[1, 0]], 1.5);
        assert_eq!(positions[[1, 1]], 2.0);
        assert_eq!(positions[[1, 2]], 3.0);
        assert_eq!(weights.shape(), &[2]);
        assert_eq!(weights.to_vec(), vec![24.0, 24.0]);
    }
}
