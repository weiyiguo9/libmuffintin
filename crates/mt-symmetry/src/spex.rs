//! SPEX compatibility import: symmetry operations, little-group tables, and
//! degenerate-subspace irreps taken verbatim from a SPEX dump instead of
//! re-detection, so downstream consumers reuse SPEX's own symmetry analysis.
//!
//! All indices are 0-based and all coordinates fractional; the versioned HDF5
//! schema and the SPEX-side conventions are recorded in
//! `doc/22_crystal_symmetry_and_spex_irrep_import.md`.

use num_complex::Complex64;

use crate::{SymmetryDataset, SymmetryOperation, SymmetryProvenance};

/// Irreps of one degenerate band block over the little group of one k-point.
#[derive(Clone, Debug, PartialEq)]
pub struct SubspaceIrreps {
    /// First band (0-based) of the degenerate block.
    pub first_band: usize,
    /// Block dimension `d`.
    pub dimension: usize,
    /// One row-major `d x d` unitary matrix per little-group operation,
    /// ordered like [`KpointIrreps::little_group`]. SPEX `irrep_sub`.
    pub matrices: Vec<Vec<Complex64>>,
}

/// Little group and subspace irreps at one irreducible k-point and spin.
#[derive(Clone, Debug, PartialEq)]
pub struct KpointIrreps {
    /// Index into [`SpexSymmetryImport::kpoints`].
    pub kpoint_index: usize,
    pub spin: usize,
    /// Indices into [`SpexSymmetryImport::operations`] whose reciprocal
    /// action fixes the k-point modulo a reciprocal lattice vector.
    pub little_group: Vec<usize>,
    pub subspaces: Vec<SubspaceIrreps>,
}

/// SPEX symmetry content mapped into libmuffintin conventions.
#[derive(Clone, Debug, PartialEq)]
pub struct SpexSymmetryImport {
    /// Full operation list in SPEX order, including the time-reversal-doubled
    /// tail SPEX appends when the crystal lacks inversion symmetry.
    pub operations: Vec<SymmetryOperation>,
    /// `inverse[i]` is the index of the operation inverse to operation `i`.
    pub inverse: Vec<usize>,
    /// `atom_map[i][a]` is the site operation `i` sends site `a` to (SPEX `pcent`).
    pub atom_map: Vec<Vec<usize>>,
    /// Full-BZ fractional k-points with the irreducible wedge first.
    pub kpoints: Vec<[f64; 3]>,
    /// Count of irreducible k-points at the head of `kpoints` (SPEX `nkpti`).
    pub irreducible_count: usize,
    /// `parent[i]` is the irreducible parent of k-point `i` (SPEX `kptp`).
    pub parent: Vec<usize>,
    /// `parent_operation[i]` maps the parent onto k-point `i` (SPEX `symkpt`).
    pub parent_operation: Vec<usize>,
    pub irreps: Vec<KpointIrreps>,
}

impl SpexSymmetryImport {
    /// View the imported operations as the method-neutral dataset.
    ///
    /// Orbit representatives come from the imported atom map; space-group
    /// classification stays absent because SPEX does not re-derive it.
    pub fn dataset(&self) -> SymmetryDataset {
        let equivalent_atoms = match self.atom_map.first() {
            Some(first_map) => (0..first_map.len())
                .map(|site| {
                    self.atom_map[1..]
                        .iter()
                        .fold(first_map[site], |representative, map| {
                            representative.min(map[site])
                        })
                })
                .collect(),
            None => Vec::new(),
        };
        SymmetryDataset {
            operations: self.operations.clone(),
            equivalent_atoms,
            spacegroup_number: None,
            hermann_mauguin: None,
            provenance: SymmetryProvenance::Imported {
                code: "spex".to_owned(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_representatives_follow_the_atom_map() {
        let identity = SymmetryOperation {
            rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation: [0.0, 0.0, 0.0],
            time_reversal: false,
        };
        let swap = SymmetryOperation {
            rotation: [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            translation: [0.0, 0.0, 0.0],
            time_reversal: false,
        };
        let import = SpexSymmetryImport {
            operations: vec![identity, swap],
            inverse: vec![0, 1],
            atom_map: vec![vec![0, 1, 2], vec![1, 0, 2]],
            kpoints: vec![[0.0, 0.0, 0.0]],
            irreducible_count: 1,
            parent: vec![0],
            parent_operation: vec![0],
            irreps: Vec::new(),
        };
        let dataset = import.dataset();
        assert_eq!(dataset.equivalent_atoms, vec![0, 0, 2]);
        assert_eq!(
            dataset.provenance,
            SymmetryProvenance::Imported {
                code: "spex".to_owned()
            }
        );
        assert_eq!(dataset.spacegroup_number, None);
    }
}
