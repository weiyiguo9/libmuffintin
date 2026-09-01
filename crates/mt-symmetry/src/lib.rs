//! Method-neutral crystal symmetry over the input cell.
//!
//! The root types are the symmetry IR consumed by the rest of the workspace.
//! `moyo_backend` detects them from a crystal structure; external codes such
//! as SPEX populate the same IR through importers instead of re-detection, so
//! downstream consumers never see a backend-native type.

pub mod kmesh;
#[cfg(feature = "backend-moyo")]
pub mod moyo_backend;
pub mod spex;

use muffintin_core::Bohr;
use thiserror::Error;

/// One space-group operation acting on fractional coordinates as `x' = W x + w`,
/// composed with complex conjugation when `time_reversal` is set.
#[derive(Clone, Debug, PartialEq)]
pub struct SymmetryOperation {
    /// Integer rotation part `W` in the input-cell crystallographic basis;
    /// `rotation[i][j]` multiplies fractional component `j` into component `i`.
    pub rotation: [[i32; 3]; 3],
    /// Translation part `w` in fractional coordinates of the input cell.
    pub translation: [f64; 3],
    /// Antiunitary flag: the operation includes time reversal. Detection
    /// backends emit unitary operations only; importers may carry the
    /// time-reversal-doubled set of the producing code.
    pub time_reversal: bool,
}

/// Origin of a dataset, so consumers can tell detected from imported symmetry.
#[derive(Clone, Debug, PartialEq)]
pub enum SymmetryProvenance {
    /// Detected by the moyo backend; `symprec` is the tolerance actually used.
    Moyo { symprec: Bohr },
    /// Imported from an external electronic-structure code without re-detection.
    Imported { code: String },
}

/// Crystal structure input shared by all detection backends.
#[derive(Clone, Debug, PartialEq)]
pub struct CrystalCell {
    /// Direct primitive vectors stored by row in Cartesian coordinates.
    pub lattice: [[Bohr; 3]; 3],
    /// Fractional site positions.
    pub positions: Vec<[f64; 3]>,
    /// Atomic number per site; equal numbers mark interchangeable sites.
    pub atomic_numbers: Vec<u16>,
}

/// Method-neutral symmetry dataset expressed in the input cell.
#[derive(Clone, Debug, PartialEq)]
pub struct SymmetryDataset {
    pub operations: Vec<SymmetryOperation>,
    /// `equivalent_atoms[i]` is the representative input-cell site of site `i`'s orbit.
    pub equivalent_atoms: Vec<usize>,
    /// International space-group number when the source classifies one.
    pub spacegroup_number: Option<i32>,
    /// Hermann-Mauguin short symbol when the source classifies one.
    pub hermann_mauguin: Option<String>,
    pub provenance: SymmetryProvenance,
}

/// One crystal operation compiled into the Cartesian and site-index actions
/// needed by representation-specific consumers.
///
/// The operation remains expressed in the input cell, while
/// `cartesian_rotation` acts on Cartesian polar vectors and `site_map[a]` is
/// the site to which input-cell site `a` is sent.  This type deliberately owns
/// no density, basis, k-mesh, or backend-native state.
#[derive(Clone, Debug, PartialEq)]
pub struct CrystalSymmetryTransform {
    operation: SymmetryOperation,
    direct_lattice: [[Bohr; 3]; 3],
    cartesian_rotation: [[f64; 3]; 3],
    site_map: Vec<usize>,
    tolerance: Bohr,
}

impl CrystalSymmetryTransform {
    /// Compile an operation and derive its site permutation from `cell`.
    pub fn from_cell(
        operation: SymmetryOperation,
        cell: &CrystalCell,
        tolerance: Bohr,
    ) -> Result<Self, SymmetryTransformError> {
        validate_cell(cell, tolerance)?;
        let cartesian_rotation = cartesian_rotation(&operation, cell.lattice)?;
        let mut site_map = Vec::with_capacity(cell.positions.len());
        for (source, (&position, &atomic_number)) in
            cell.positions.iter().zip(&cell.atomic_numbers).enumerate()
        {
            let transformed = affine_fractional(&operation, position);
            let matches = cell
                .positions
                .iter()
                .zip(&cell.atomic_numbers)
                .enumerate()
                .filter(|&(_, (_, &candidate_number))| candidate_number == atomic_number)
                .filter(|&(_, (&candidate, _))| {
                    periodic_cartesian_distance(transformed, candidate, cell.lattice)
                        <= tolerance.get()
                })
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            match matches.as_slice() {
                &[target] => site_map.push(target),
                [] => {
                    return Err(SymmetryTransformError::UnmappedSite {
                        source_site: source,
                    });
                }
                _ => {
                    return Err(SymmetryTransformError::AmbiguousSite {
                        source_site: source,
                    });
                }
            }
        }
        validate_permutation(&site_map)?;
        Ok(Self {
            operation,
            direct_lattice: cell.lattice,
            cartesian_rotation,
            site_map,
            tolerance,
        })
    }

    pub const fn operation(&self) -> &SymmetryOperation {
        &self.operation
    }

    pub const fn direct_lattice(&self) -> &[[Bohr; 3]; 3] {
        &self.direct_lattice
    }

    pub const fn cartesian_rotation(&self) -> &[[f64; 3]; 3] {
        &self.cartesian_rotation
    }

    pub fn site_map(&self) -> &[usize] {
        &self.site_map
    }

    pub const fn tolerance(&self) -> Bohr {
        self.tolerance
    }
}

/// Invalid compilation of a method-neutral crystal transform.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SymmetryTransformError {
    #[error("cell has {positions} positions but {atomic_numbers} atomic numbers")]
    SiteCountMismatch {
        positions: usize,
        atomic_numbers: usize,
    },
    #[error("symmetry tolerance must be finite and nonnegative, got {0}")]
    InvalidTolerance(f64),
    #[error("crystal lattice is singular")]
    SingularLattice,
    #[error("crystal cell has a non-finite fractional position")]
    NonFinitePosition,
    #[error("crystal cell has a non-finite lattice vector")]
    NonFiniteLattice,
    #[error("symmetry operation has a non-finite translation")]
    NonFiniteTranslation,
    #[error("fractional rotation is not unimodular")]
    NonUnimodularRotation,
    #[error("fractional rotation does not preserve the crystal metric")]
    NonOrthogonalCartesianRotation,
    #[error("site {source_site} has no image under the symmetry operation")]
    UnmappedSite { source_site: usize },
    #[error("site {source_site} has multiple images within the symmetry tolerance")]
    AmbiguousSite { source_site: usize },
    #[error("site map is not a permutation at target site {target}")]
    NonPermutationSiteMap { target: usize },
}

fn validate_cell(cell: &CrystalCell, tolerance: Bohr) -> Result<(), SymmetryTransformError> {
    if cell.positions.len() != cell.atomic_numbers.len() {
        return Err(SymmetryTransformError::SiteCountMismatch {
            positions: cell.positions.len(),
            atomic_numbers: cell.atomic_numbers.len(),
        });
    }
    if !tolerance.get().is_finite() || tolerance.get() < 0.0 {
        return Err(SymmetryTransformError::InvalidTolerance(tolerance.get()));
    }
    if cell
        .positions
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(SymmetryTransformError::NonFinitePosition);
    }
    if cell
        .lattice
        .iter()
        .flatten()
        .any(|value| !value.get().is_finite())
    {
        return Err(SymmetryTransformError::NonFiniteLattice);
    }
    Ok(())
}

fn validate_permutation(site_map: &[usize]) -> Result<(), SymmetryTransformError> {
    let mut seen = vec![false; site_map.len()];
    for &target in site_map {
        if target >= site_map.len() || std::mem::replace(&mut seen[target], true) {
            return Err(SymmetryTransformError::NonPermutationSiteMap { target });
        }
    }
    Ok(())
}

fn affine_fractional(operation: &SymmetryOperation, position: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        operation.translation[row]
            + (0..3)
                .map(|column| f64::from(operation.rotation[row][column]) * position[column])
                .sum::<f64>()
    })
}

fn periodic_cartesian_distance(left: [f64; 3], right: [f64; 3], lattice: [[Bohr; 3]; 3]) -> f64 {
    let fractional: [f64; 3] = std::array::from_fn(|axis| {
        let difference = left[axis] - right[axis];
        difference - difference.round()
    });
    let cartesian: [f64; 3] = std::array::from_fn(|axis| {
        (0..3)
            .map(|basis| fractional[basis] * lattice[basis][axis].get())
            .sum()
    });
    cartesian
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn cartesian_rotation(
    operation: &SymmetryOperation,
    lattice: [[Bohr; 3]; 3],
) -> Result<[[f64; 3]; 3], SymmetryTransformError> {
    if operation.translation.iter().any(|value| !value.is_finite()) {
        return Err(SymmetryTransformError::NonFiniteTranslation);
    }
    if integer_determinant(operation.rotation).abs() != 1 {
        return Err(SymmetryTransformError::NonUnimodularRotation);
    }
    let w = operation.rotation.map(|row| row.map(f64::from));
    let a_t: [[f64; 3]; 3] =
        std::array::from_fn(|row| std::array::from_fn(|column| lattice[column][row].get()));
    let inverse = inverse(a_t).ok_or(SymmetryTransformError::SingularLattice)?;
    let rotation = multiply(multiply(a_t, w), inverse);
    let scale = rotation
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    let tolerance = 2048.0 * f64::EPSILON * scale;
    for row in 0..3 {
        for column in 0..3 {
            let dot = (0..3)
                .map(|axis| rotation[axis][row] * rotation[axis][column])
                .sum::<f64>();
            let expected = if row == column { 1.0 } else { 0.0 };
            if (dot - expected).abs() > tolerance {
                return Err(SymmetryTransformError::NonOrthogonalCartesianRotation);
            }
        }
    }
    Ok(rotation)
}

pub(crate) fn integer_determinant(matrix: [[i32; 3]; 3]) -> i128 {
    let matrix = matrix.map(|row| row.map(i128::from));
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn determinant(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn inverse(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let determinant = determinant(matrix);
    let scale = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    if determinant.abs() <= 128.0 * f64::EPSILON * scale.powi(3) {
        return None;
    }
    Some([
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) / determinant,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) / determinant,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) / determinant,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) / determinant,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) / determinant,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) / determinant,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) / determinant,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) / determinant,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) / determinant,
        ],
    ])
}

fn multiply(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..3)
                .map(|axis| left[row][axis] * right[axis][column])
                .sum()
        })
    })
}
