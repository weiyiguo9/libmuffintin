//! Method-neutral crystal symmetry over the input cell.
//!
//! The root types are the symmetry IR consumed by the rest of the workspace.
//! `moyo_backend` detects them from a crystal structure; external codes such
//! as SPEX populate the same IR through importers instead of re-detection, so
//! downstream consumers never see a backend-native type.

#[cfg(feature = "backend-moyo")]
pub mod moyo_backend;
pub mod spex;

use muffintin_core::Bohr;

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
