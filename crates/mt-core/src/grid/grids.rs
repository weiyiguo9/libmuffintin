//! Atom-centred, uniform, interstitial, and composite point grids.

use crate::{AngularGrid, Cell};
use crate::{Bohr, ExponentialMesh, Sphere, VolumeBohr3};
use thiserror::Error;

/// Physical region represented by a quadrature point.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RegionTag {
    /// Point inside the atom-centred grid for the named atom.
    Atom(usize),
    /// Point in the unit-cell interstitial region.
    Interstitial,
    /// Point on an unpartitioned debug uniform grid.
    Uniform,
}

/// One real-space quadrature point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridPoint {
    /// Cartesian coordinates in Bohr.
    pub position: [Bohr; 3],
    /// Three-dimensional quadrature weight in Bohr cubed.
    pub weight: VolumeBohr3,
    /// Region that owns this point.
    pub region: RegionTag,
}

/// Shared read-only surface for all real-space point grids.
pub trait Grid {
    /// Points in the grid's documented deterministic order.
    fn points(&self) -> &[GridPoint];

    /// Number of points.
    fn len(&self) -> usize {
        self.points().len()
    }

    /// Whether the grid has no points.
    fn is_empty(&self) -> bool {
        self.points().is_empty()
    }

    /// Sum a scalar function using the grid's volume weights.
    fn integrate(&self, function: impl Fn([Bohr; 3]) -> f64) -> f64 {
        self.points()
            .iter()
            .map(|point| point.weight.0 * function(point.position))
            .sum()
    }

    /// Sum all volume weights.
    fn volume(&self) -> VolumeBohr3 {
        VolumeBohr3(self.points().iter().map(|point| point.weight.0).sum())
    }
}

/// Invalid grid rule, geometry, or allocation size.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum GridError {
    /// Angular rules must contain at least one point.
    #[error("angular grid must contain at least one point")]
    EmptyAngularGrid,
    /// An angular direction contains a non-finite component.
    #[error("angular direction {index} contains a non-finite component")]
    InvalidAngularDirection { index: usize },
    /// An angular direction is not normalized.
    #[error("angular direction {index} must have unit norm, got {norm}")]
    NonUnitAngularDirection { index: usize, norm: f64 },
    /// Solid-angle weights must be finite and positive.
    #[error("angular weight {index} must be finite and positive, got {weight}")]
    InvalidAngularWeight { index: usize, weight: f64 },
    /// Solid-angle weights must sum to the full sphere.
    #[error("angular weights must sum to 4*pi, got {weight_sum}")]
    InvalidAngularWeightSum { weight_sum: f64 },
    /// Atom centres must have finite Cartesian components.
    #[error("atom centre contains a non-finite coordinate")]
    InvalidAtomCenter,
    /// Atom-grid radial shells must run outwards.
    #[error("atom-grid exponential radial increment must be positive, got {0}")]
    NonOutwardRadialMesh(f64),
    /// A cell basis contains a non-finite component.
    #[error("cell basis contains a non-finite component")]
    NonFiniteCell,
    /// Cell primitive vectors are linearly dependent or numerically singular.
    #[error("cell basis is singular")]
    SingularCell,
    /// Each uniform-grid division count must be nonzero.
    #[error("uniform-grid divisions must be nonzero, got {0:?}")]
    ZeroDivisions([usize; 3]),
    /// Requested point count does not fit in `usize`.
    #[error("grid point count overflows usize")]
    PointCountOverflow,
    /// A muffin-tin centre contains a non-finite component.
    #[error("muffin-tin sphere {index} centre contains a non-finite coordinate")]
    InvalidSphereCenter { index: usize },
    /// A muffin-tin radius must be finite and positive.
    #[error("muffin-tin sphere {index} radius must be finite and positive, got {radius}")]
    InvalidSphereRadius { index: usize, radius: f64 },
}

/// Exponential radial shells crossed with an angular quadrature rule.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomGrid {
    atom_index: usize,
    points: Vec<GridPoint>,
}

impl AtomGrid {
    /// Build an atom-centred grid in shell-major, angular-minor order.
    pub fn new(
        atom_index: usize,
        center: [Bohr; 3],
        radial: &ExponentialMesh,
        angular: &AngularGrid,
    ) -> Result<Self, GridError> {
        if center.iter().any(|component| !component.0.is_finite()) {
            return Err(GridError::InvalidAtomCenter);
        }
        if radial.increment() <= 0.0 {
            return Err(GridError::NonOutwardRadialMesh(radial.increment()));
        }
        let number = radial
            .len()
            .checked_mul(angular.len())
            .ok_or(GridError::PointCountOverflow)?;
        let mut points = Vec::with_capacity(number);
        for (&radius, &radial_weight) in radial.radii().iter().zip(radial.weights()) {
            for angular_point in angular.points() {
                let position = std::array::from_fn(|axis| {
                    Bohr(center[axis].0 + radius.0 * angular_point.direction[axis])
                });
                points.push(GridPoint {
                    position,
                    weight: VolumeBohr3(radial_weight * radius.0.powi(2) * angular_point.weight),
                    region: RegionTag::Atom(atom_index),
                });
            }
        }
        Ok(Self { atom_index, points })
    }

    /// Atom index used by the region tags.
    pub const fn atom_index(&self) -> usize {
        self.atom_index
    }
}

impl Grid for AtomGrid {
    fn points(&self) -> &[GridPoint] {
        &self.points
    }
}

/// Midpoint-rule uniform grid over one direct-lattice cell.
#[derive(Clone, Debug, PartialEq)]
pub struct UniformGrid {
    cell: Cell,
    divisions: [usize; 3],
    points: Vec<GridPoint>,
}

impl UniformGrid {
    /// Build a grid in lexicographic `(i, j, k)` order, with `k` fastest.
    pub fn new(cell: Cell, divisions: [usize; 3]) -> Result<Self, GridError> {
        if divisions.contains(&0) {
            return Err(GridError::ZeroDivisions(divisions));
        }
        let number = divisions.into_iter().try_fold(1_usize, |product, count| {
            product
                .checked_mul(count)
                .ok_or(GridError::PointCountOverflow)
        })?;
        let point_weight = VolumeBohr3(cell.volume().0 / number as f64);
        let mut points = Vec::with_capacity(number);
        for i in 0..divisions[0] {
            for j in 0..divisions[1] {
                for k in 0..divisions[2] {
                    let fractional = [
                        (i as f64 + 0.5) / divisions[0] as f64,
                        (j as f64 + 0.5) / divisions[1] as f64,
                        (k as f64 + 0.5) / divisions[2] as f64,
                    ];
                    points.push(GridPoint {
                        position: cell.cartesian(fractional),
                        weight: point_weight,
                        region: RegionTag::Uniform,
                    });
                }
            }
        }
        Ok(Self {
            cell,
            divisions,
            points,
        })
    }

    /// Direct-lattice cell covered by the grid.
    pub const fn cell(&self) -> Cell {
        self.cell
    }

    /// Division counts along the three primitive directions.
    pub const fn divisions(&self) -> [usize; 3] {
        self.divisions
    }
}

impl Grid for UniformGrid {
    fn points(&self) -> &[GridPoint] {
        &self.points
    }
}

/// Uniform-cell points outside every periodically repeated muffin-tin sphere.
#[derive(Clone, Debug, PartialEq)]
pub struct InterstitialGrid {
    cell: Cell,
    divisions: [usize; 3],
    points: Vec<GridPoint>,
}

impl InterstitialGrid {
    /// Filter a uniform grid using exact periodic nearest-image distances.
    pub fn new(uniform: &UniformGrid, spheres: &[Sphere]) -> Result<Self, GridError> {
        for (index, sphere) in spheres.iter().enumerate() {
            if sphere
                .center
                .iter()
                .any(|component| !component.0.is_finite())
            {
                return Err(GridError::InvalidSphereCenter { index });
            }
            if !sphere.radius.0.is_finite() || sphere.radius.0 <= 0.0 {
                return Err(GridError::InvalidSphereRadius {
                    index,
                    radius: sphere.radius.0,
                });
            }
        }

        let points = uniform
            .points()
            .iter()
            .filter(|point| {
                spheres.iter().all(|sphere| {
                    let displacement =
                        std::array::from_fn(|axis| point.position[axis].0 - sphere.center[axis].0);
                    let distance_squared =
                        uniform.cell.nearest_image_distance_squared(displacement);
                    distance_squared >= sphere.radius.0.powi(2)
                })
            })
            .map(|point| GridPoint {
                region: RegionTag::Interstitial,
                ..*point
            })
            .collect();
        Ok(Self {
            cell: uniform.cell,
            divisions: uniform.divisions,
            points,
        })
    }

    /// Direct-lattice cell covered by the parent uniform grid.
    pub const fn cell(&self) -> Cell {
        self.cell
    }

    /// Division counts of the parent uniform grid.
    pub const fn divisions(&self) -> [usize; 3] {
        self.divisions
    }
}

impl Grid for InterstitialGrid {
    fn points(&self) -> &[GridPoint] {
        &self.points
    }
}

/// Atom grids followed by an interstitial grid in deterministic region order.
#[derive(Clone, Debug, PartialEq)]
pub struct CompositeGrid {
    points: Vec<GridPoint>,
}

impl CompositeGrid {
    /// Concatenate atoms sorted stably by atom index, then the interstitial.
    pub fn new(mut atoms: Vec<AtomGrid>, interstitial: InterstitialGrid) -> Self {
        atoms.sort_by_key(AtomGrid::atom_index);
        let capacity = atoms
            .iter()
            .map(Grid::len)
            .sum::<usize>()
            .saturating_add(interstitial.len());
        let mut points = Vec::with_capacity(capacity);
        for atom in atoms {
            points.extend(atom.points);
        }
        points.extend(interstitial.points);
        Self { points }
    }
}

impl Grid for CompositeGrid {
    fn points(&self) -> &[GridPoint] {
        &self.points
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn cubic_cell(side: f64) -> Cell {
        Cell::new([
            [Bohr(side), Bohr(0.0), Bohr(0.0)],
            [Bohr(0.0), Bohr(side), Bohr(0.0)],
            [Bohr(0.0), Bohr(0.0), Bohr(side)],
        ])
        .unwrap()
    }

    #[test]
    fn atom_grid_integrates_spherical_gaussian_and_slater() {
        let radial = ExponentialMesh::new(Bohr(1e-7), 0.01, 2001).unwrap();
        let angular = AngularGrid::fibonacci(14).unwrap();
        let grid = AtomGrid::new(0, [Bohr(1.2), Bohr(-0.3), Bohr(2.1)], &radial, &angular).unwrap();
        let center = [1.2, -0.3, 2.1];
        let gaussian = grid.integrate(|position| {
            let radius_squared: f64 = position
                .iter()
                .zip(center)
                .map(|(coordinate, center)| (coordinate.0 - center).powi(2))
                .sum();
            (-0.7 * radius_squared).exp()
        });
        let expected_gaussian = (PI / 0.7).powf(1.5);
        assert!((gaussian - expected_gaussian).abs() < 2e-11 * expected_gaussian);

        let slater = grid.integrate(|position| {
            let radius = position
                .iter()
                .zip(center)
                .map(|(coordinate, center)| (coordinate.0 - center).powi(2))
                .sum::<f64>()
                .sqrt();
            (-1.3 * radius).exp()
        });
        let expected_slater = 8.0 * PI / 1.3_f64.powi(3);
        assert!((slater - expected_slater).abs() < 2e-11 * expected_slater);
    }

    #[test]
    fn interstitial_drops_periodic_sphere_images_and_volume_converges() {
        let cell = cubic_cell(4.0);
        let sphere = Sphere {
            center: [Bohr(0.1), Bohr(2.0), Bohr(2.0)],
            radius: Bohr(0.8),
        };
        let coarse =
            InterstitialGrid::new(&UniformGrid::new(cell, [10; 3]).unwrap(), &[sphere]).unwrap();
        let fine =
            InterstitialGrid::new(&UniformGrid::new(cell, [40; 3]).unwrap(), &[sphere]).unwrap();
        for point in fine.points() {
            let displacement =
                std::array::from_fn(|axis| point.position[axis].0 - sphere.center[axis].0);
            assert!(cell.nearest_image_distance_squared(displacement) >= sphere.radius.0.powi(2));
        }
        assert!(fine.len() < 40_usize.pow(3));
        let exact = cell.volume().0 - 4.0 * PI * sphere.radius.0.powi(3) / 3.0;
        assert!((fine.volume().0 - exact).abs() < (coarse.volume().0 - exact).abs());
    }

    #[test]
    fn composite_order_is_stable_and_region_sorted() {
        let radial = ExponentialMesh::new(Bohr(0.1), 0.1, 7).unwrap();
        let angular = AngularGrid::fibonacci(2).unwrap();
        let atom_two = AtomGrid::new(2, [Bohr(2.0); 3], &radial, &angular).unwrap();
        let atom_one_a = AtomGrid::new(1, [Bohr(1.0); 3], &radial, &angular).unwrap();
        let atom_one_b = AtomGrid::new(1, [Bohr(1.5); 3], &radial, &angular).unwrap();
        let interstitial =
            InterstitialGrid::new(&UniformGrid::new(cubic_cell(5.0), [2; 3]).unwrap(), &[])
                .unwrap();

        let expected_first_atom_one = atom_one_a.points()[0];
        let expected_second_atom_one = atom_one_b.points()[0];
        let atom_len = atom_one_a.len();
        let composite = CompositeGrid::new(vec![atom_two, atom_one_a, atom_one_b], interstitial);
        assert_eq!(composite.points()[0], expected_first_atom_one);
        assert_eq!(composite.points()[atom_len], expected_second_atom_one);
        assert!(
            composite
                .points()
                .windows(2)
                .all(|pair| pair[0].region <= pair[1].region)
        );
    }
}
