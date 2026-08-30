//! Symmetry detection through the moyo crystal-symmetry engine.

use moyo::MoyoDataset;
use moyo::base::{Cell, Lattice};
use muffintin_core::Bohr;
use nalgebra::{Matrix3, Vector3};
use thiserror::Error;

use crate::{CrystalCell, SymmetryDataset, SymmetryOperation, SymmetryProvenance};

/// Failures while detecting symmetry with moyo.
#[derive(Debug, Error)]
pub enum MoyoDetectionError {
    /// The site arrays disagree in length.
    #[error("cell has {positions} positions but {atomic_numbers} atomic numbers")]
    SiteCountMismatch {
        positions: usize,
        atomic_numbers: usize,
    },
    /// The moyo iterative symmetry search failed.
    #[error("moyo symmetry search failed: {0}")]
    Search(String),
}

/// Detect the space-group symmetry of `cell` at Cartesian tolerance `symprec`.
pub fn detect(cell: &CrystalCell, symprec: Bohr) -> Result<SymmetryDataset, MoyoDetectionError> {
    if cell.positions.len() != cell.atomic_numbers.len() {
        return Err(MoyoDetectionError::SiteCountMismatch {
            positions: cell.positions.len(),
            atomic_numbers: cell.atomic_numbers.len(),
        });
    }
    let rows = cell.lattice.map(|v| v.map(Bohr::get));
    let lattice = Lattice::new(Matrix3::from_fn(|i, j| rows[i][j]));
    let positions = cell.positions.iter().copied().map(Vector3::from).collect();
    let numbers = cell.atomic_numbers.iter().map(|&z| i32::from(z)).collect();
    let moyo_cell = Cell::new(lattice, positions, numbers);
    let dataset = MoyoDataset::with_default(&moyo_cell, symprec.get())
        .map_err(|error| MoyoDetectionError::Search(error.to_string()))?;
    let operations = dataset
        .operations
        .iter()
        .map(|operation| SymmetryOperation {
            rotation: operation.rotation_as_array(),
            translation: operation.translation.into(),
            time_reversal: false,
        })
        .collect();
    Ok(SymmetryDataset {
        operations,
        equivalent_atoms: dataset.orbits,
        spacegroup_number: Some(dataset.number),
        hermann_mauguin: Some(dataset.hm_symbol),
        provenance: SymmetryProvenance::Moyo {
            symprec: Bohr(dataset.symprec),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fcc_primitive_cell_is_fm3m() {
        let half = Bohr(3.6);
        let zero = Bohr(0.0);
        let cell = CrystalCell {
            lattice: [[zero, half, half], [half, zero, half], [half, half, zero]],
            positions: vec![[0.0, 0.0, 0.0]],
            atomic_numbers: vec![62],
        };
        let dataset = detect(&cell, Bohr(1.0e-5)).unwrap();
        assert_eq!(dataset.spacegroup_number, Some(225));
        assert_eq!(dataset.hermann_mauguin.as_deref(), Some("F m -3 m"));
        assert_eq!(dataset.operations.len(), 48);
        assert_eq!(dataset.equivalent_atoms, vec![0]);
        let identity = SymmetryOperation {
            rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation: [0.0, 0.0, 0.0],
            time_reversal: false,
        };
        assert!(dataset.operations.contains(&identity));
    }

    #[test]
    fn rutile_orbits_split_by_species() {
        let x = 0.3046;
        let cell = CrystalCell {
            lattice: [
                [Bohr(8.7), Bohr(0.0), Bohr(0.0)],
                [Bohr(0.0), Bohr(8.7), Bohr(0.0)],
                [Bohr(0.0), Bohr(0.0), Bohr(5.6)],
            ],
            positions: vec![
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [x, x, 0.0],
                [-x, -x, 0.0],
                [-x + 0.5, x + 0.5, 0.5],
                [x + 0.5, -x + 0.5, 0.5],
            ],
            atomic_numbers: vec![22, 22, 8, 8, 8, 8],
        };
        let dataset = detect(&cell, Bohr(1.0e-4)).unwrap();
        assert_eq!(dataset.spacegroup_number, Some(136));
        assert_eq!(dataset.equivalent_atoms, vec![0, 0, 2, 2, 2, 2]);
    }

    #[test]
    fn mismatched_site_arrays_are_rejected() {
        let cell = CrystalCell {
            lattice: [
                [Bohr(1.0), Bohr(0.0), Bohr(0.0)],
                [Bohr(0.0), Bohr(1.0), Bohr(0.0)],
                [Bohr(0.0), Bohr(0.0), Bohr(1.0)],
            ],
            positions: vec![[0.0, 0.0, 0.0]],
            atomic_numbers: vec![],
        };
        assert!(matches!(
            detect(&cell, Bohr(1.0e-5)),
            Err(MoyoDetectionError::SiteCountMismatch { .. })
        ));
    }
}
