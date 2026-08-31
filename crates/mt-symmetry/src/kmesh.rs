//! Exact orbit reduction of regular reciprocal-space meshes.

use thiserror::Error;

use crate::SymmetryDataset;

/// A regular reciprocal-space mesh with points `(i + shift) / divisions`.
///
/// Each shift component must be either `0.0` (Gamma centered) or `0.5`
/// (half-grid shifted). Fractional coordinates emitted by the reduction are
/// normalized to `[0, 1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegularKMesh {
    pub divisions: [usize; 3],
    pub shift: [f64; 3],
}

/// One symmetry action mapping an irreducible representative onto a full-mesh point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KPointOperation {
    /// Index into [`SymmetryDataset::operations`].
    pub operation_index: usize,
    /// Whether an additional time-reversal action was appended to the dataset operation.
    /// The total reciprocal-space sign is negative when this differs from the
    /// dataset operation's own `time_reversal` flag.
    pub time_reversal: bool,
}

/// One point in the complete regular mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct FullKPoint {
    /// Linear index with the first reciprocal coordinate varying fastest.
    pub full_index: usize,
    /// Integer coordinates in the regular mesh.
    pub mesh_index: [usize; 3],
    /// Fractional reciprocal coordinates normalized to `[0, 1)`.
    pub fractional: [f64; 3],
    /// Index into [`KMeshReduction::irreducible_points`].
    pub parent: usize,
    /// Linear full-mesh index of the orbit representative.
    pub representative: usize,
    /// Operation mapping `representative` onto this point.
    pub parent_operation: KPointOperation,
}

/// One representative of a regular-mesh orbit.
#[derive(Clone, Debug, PartialEq)]
pub struct IrreducibleKPoint {
    /// Linear index of this representative in the full mesh.
    pub representative: usize,
    pub mesh_index: [usize; 3],
    /// Fractional reciprocal coordinates normalized to `[0, 1)`.
    pub fractional: [f64; 3],
    pub multiplicity: usize,
    /// `multiplicity / full_mesh_size`; all irreducible weights sum to one.
    pub weight: f64,
}

/// Full-to-irreducible orbit decomposition of a regular reciprocal-space mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct KMeshReduction {
    pub mesh: RegularKMesh,
    pub full_points: Vec<FullKPoint>,
    pub irreducible_points: Vec<IrreducibleKPoint>,
    pub active_operations: Vec<KPointOperation>,
}

/// Invalid meshes, rotations, or incomplete operation tables encountered during reduction.
#[derive(Debug, Error, PartialEq)]
pub enum KMeshReductionError {
    #[error("regular k-mesh division {axis} is zero")]
    ZeroDivision { axis: usize },
    #[error("regular k-mesh shift {axis} is {shift}; only 0.0 and 0.5 are supported")]
    UnsupportedShift { axis: usize, shift: f64 },
    #[error("the symmetry dataset has no operations")]
    EmptyOperations,
    #[error(
        "symmetry operation {operation_index} has non-unimodular rotation determinant {determinant}"
    )]
    NonUnimodularRotation {
        operation_index: usize,
        determinant: i128,
    },
    #[error(
        "symmetry operation {operation_index} with additional time reversal {time_reversal} does not preserve regular k-mesh point {mesh_index:?}"
    )]
    IncompatibleShift {
        operation_index: usize,
        time_reversal: bool,
        mesh_index: [usize; 3],
    },
    #[error("no symmetry operation preserves the requested regular k mesh")]
    NoCompatibleOperations,
    #[error(
        "no dataset operation maps representative {representative} onto full k-mesh point {full_index}"
    )]
    UnmappedPoint {
        full_index: usize,
        representative: usize,
    },
    #[error("regular k-mesh integer arithmetic overflowed")]
    ArithmeticOverflow,
}

#[derive(Clone, Debug)]
struct CandidateAction {
    operation: KPointOperation,
    mapping: Vec<usize>,
}

/// Reduce a regular reciprocal-space mesh under all dataset operations.
///
/// A unitary operation with direct-space rotation `W` acts on fractional
/// reciprocal coordinates as `W^-T`. An antiunitary dataset operation adds a
/// minus sign. When `include_time_reversal` is true, the reducer also appends
/// an independent time-reversal action to every dataset operation.
pub fn reduce_regular_mesh(
    dataset: &SymmetryDataset,
    mesh: RegularKMesh,
    include_time_reversal: bool,
) -> Result<KMeshReduction, KMeshReductionError> {
    let shift = validate_mesh(mesh)?;
    if dataset.operations.is_empty() {
        return Err(KMeshReductionError::EmptyOperations);
    }
    let point_count = mesh
        .divisions
        .into_iter()
        .try_fold(1_usize, usize::checked_mul)
        .ok_or(KMeshReductionError::ArithmeticOverflow)?;
    let common_denominator = mesh
        .divisions
        .into_iter()
        .try_fold(1_i128, checked_lcm)
        .ok_or(KMeshReductionError::ArithmeticOverflow)?;

    let mut actions =
        Vec::with_capacity(dataset.operations.len() * if include_time_reversal { 2 } else { 1 });
    for (operation_index, operation) in dataset.operations.iter().enumerate() {
        let reciprocal = reciprocal_rotation(operation.rotation, operation_index)?;
        for time_reversal in 0..=usize::from(include_time_reversal) {
            let appended_time_reversal = time_reversal == 1;
            let negative = operation.time_reversal ^ appended_time_reversal;
            let operation = KPointOperation {
                operation_index,
                time_reversal: appended_time_reversal,
            };
            let mut mapping = Vec::with_capacity(point_count);
            let mut compatible = true;
            for full_index in 0..point_count {
                let mesh_index = unflatten(full_index, mesh.divisions);
                let Some(mapped) = map_mesh_index(
                    mesh_index,
                    mesh.divisions,
                    shift,
                    reciprocal,
                    negative,
                    common_denominator,
                ) else {
                    compatible = false;
                    break;
                };
                mapping.push(flatten(mapped, mesh.divisions));
            }
            if compatible {
                actions.push(CandidateAction { operation, mapping });
            }
        }
    }
    if actions.is_empty() {
        return Err(KMeshReductionError::NoCompatibleOperations);
    }

    let mut orbits = DisjointSets::new(point_count);
    for action in &actions {
        for (source, &target) in action.mapping.iter().enumerate() {
            orbits.join(source, target);
        }
    }
    let mut representatives = vec![usize::MAX; point_count];
    for full_index in 0..point_count {
        let root = orbits.root(full_index);
        representatives[root] = representatives[root].min(full_index);
    }
    let representative_for = (0..point_count)
        .map(|full_index| representatives[orbits.root(full_index)])
        .collect::<Vec<_>>();

    let mut irreducible_points = Vec::new();
    let mut parent_for_representative = vec![usize::MAX; point_count];
    for (full_index, &representative) in representative_for.iter().enumerate() {
        if full_index != representative {
            continue;
        }
        let parent = irreducible_points.len();
        parent_for_representative[representative] = parent;
        let multiplicity = representative_for
            .iter()
            .filter(|&&candidate| candidate == representative)
            .count();
        let mesh_index = unflatten(representative, mesh.divisions);
        irreducible_points.push(IrreducibleKPoint {
            representative,
            mesh_index,
            fractional: fractional(mesh_index, mesh, shift),
            multiplicity,
            weight: multiplicity as f64 / point_count as f64,
        });
    }

    let mut full_points = Vec::with_capacity(point_count);
    for (full_index, &representative) in representative_for.iter().enumerate() {
        let parent_operation = actions
            .iter()
            .find(|action| action.mapping[representative] == full_index)
            .map(|action| action.operation)
            .ok_or(KMeshReductionError::UnmappedPoint {
                full_index,
                representative,
            })?;
        let mesh_index = unflatten(full_index, mesh.divisions);
        full_points.push(FullKPoint {
            full_index,
            mesh_index,
            fractional: fractional(mesh_index, mesh, shift),
            parent: parent_for_representative[representative],
            representative,
            parent_operation,
        });
    }

    let active_operations = actions.iter().map(|action| action.operation).collect();
    Ok(KMeshReduction {
        mesh,
        full_points,
        irreducible_points,
        active_operations,
    })
}

fn validate_mesh(mesh: RegularKMesh) -> Result<[i128; 3], KMeshReductionError> {
    let mut shift = [0_i128; 3];
    for axis in 0..3 {
        if mesh.divisions[axis] == 0 {
            return Err(KMeshReductionError::ZeroDivision { axis });
        }
        shift[axis] = if mesh.shift[axis] == 0.0 {
            0
        } else if mesh.shift[axis] == 0.5 {
            1
        } else {
            return Err(KMeshReductionError::UnsupportedShift {
                axis,
                shift: mesh.shift[axis],
            });
        };
    }
    Ok(shift)
}

fn reciprocal_rotation(
    rotation: [[i32; 3]; 3],
    operation_index: usize,
) -> Result<[[i128; 3]; 3], KMeshReductionError> {
    let w = rotation.map(|row| row.map(i128::from));
    let determinant = w[0][0] * (w[1][1] * w[2][2] - w[1][2] * w[2][1])
        - w[0][1] * (w[1][0] * w[2][2] - w[1][2] * w[2][0])
        + w[0][2] * (w[1][0] * w[2][1] - w[1][1] * w[2][0]);
    if determinant.abs() != 1 {
        return Err(KMeshReductionError::NonUnimodularRotation {
            operation_index,
            determinant,
        });
    }
    let cofactor = [
        [
            w[1][1] * w[2][2] - w[1][2] * w[2][1],
            -(w[1][0] * w[2][2] - w[1][2] * w[2][0]),
            w[1][0] * w[2][1] - w[1][1] * w[2][0],
        ],
        [
            -(w[0][1] * w[2][2] - w[0][2] * w[2][1]),
            w[0][0] * w[2][2] - w[0][2] * w[2][0],
            -(w[0][0] * w[2][1] - w[0][1] * w[2][0]),
        ],
        [
            w[0][1] * w[1][2] - w[0][2] * w[1][1],
            -(w[0][0] * w[1][2] - w[0][2] * w[1][0]),
            w[0][0] * w[1][1] - w[0][1] * w[1][0],
        ],
    ];
    Ok(cofactor.map(|row| row.map(|value| value / determinant)))
}

fn map_mesh_index(
    mesh_index: [usize; 3],
    divisions: [usize; 3],
    shift: [i128; 3],
    reciprocal: [[i128; 3]; 3],
    negative: bool,
    common_denominator: i128,
) -> Option<[usize; 3]> {
    let divisions_i128 = [
        i128::try_from(divisions[0]).ok()?,
        i128::try_from(divisions[1]).ok()?,
        i128::try_from(divisions[2]).ok()?,
    ];
    let mesh_index_i128 = [
        i128::try_from(mesh_index[0]).ok()?,
        i128::try_from(mesh_index[1]).ok()?,
        i128::try_from(mesh_index[2]).ok()?,
    ];
    let sign = if negative { -1_i128 } else { 1_i128 };
    let mut mapped = [0_usize; 3];
    for target_axis in 0..3 {
        let mut numerator = 0_i128;
        for source_axis in 0..3 {
            let doubled_index = mesh_index_i128[source_axis]
                .checked_mul(2)?
                .checked_add(shift[source_axis])?;
            let term = reciprocal[target_axis][source_axis]
                .checked_mul(doubled_index)?
                .checked_mul(common_denominator / divisions_i128[source_axis])?
                .checked_mul(divisions_i128[target_axis])?;
            numerator = numerator.checked_add(term)?;
        }
        numerator = numerator.checked_mul(sign)?;
        if numerator.rem_euclid(common_denominator) != 0 {
            return None;
        }
        let doubled_division = divisions_i128[target_axis].checked_mul(2)?;
        let residue = (numerator / common_denominator).rem_euclid(doubled_division);
        if residue.rem_euclid(2) != shift[target_axis] {
            return None;
        }
        mapped[target_axis] = usize::try_from((residue - shift[target_axis]) / 2).ok()?;
    }
    Some(mapped)
}

fn fractional(mesh_index: [usize; 3], mesh: RegularKMesh, shift: [i128; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| {
        (mesh_index[axis] as f64 + 0.5 * shift[axis] as f64) / mesh.divisions[axis] as f64
    })
}

const fn flatten(index: [usize; 3], divisions: [usize; 3]) -> usize {
    index[0] + divisions[0] * (index[1] + divisions[1] * index[2])
}

const fn unflatten(index: usize, divisions: [usize; 3]) -> [usize; 3] {
    let first = index % divisions[0];
    let rest = index / divisions[0];
    [first, rest % divisions[1], rest / divisions[1]]
}

fn checked_lcm(left: i128, right: usize) -> Option<i128> {
    let right = i128::try_from(right).ok()?;
    left.checked_div(gcd(left, right))?.checked_mul(right)
}

const fn gcd(mut left: i128, mut right: i128) -> i128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[derive(Debug)]
struct DisjointSets {
    parent: Vec<usize>,
}

impl DisjointSets {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
        }
    }

    fn root(&mut self, index: usize) -> usize {
        let mut root = index;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut current = index;
        while self.parent[current] != current {
            let next = self.parent[current];
            self.parent[current] = root;
            current = next;
        }
        root
    }

    fn join(&mut self, left: usize, right: usize) {
        let left_root = self.root(left);
        let right_root = self.root(right);
        if left_root != right_root {
            self.parent[right_root] = left_root;
        }
    }
}

#[cfg(test)]
mod tests {
    use muffintin_core::Bohr;

    use super::*;
    use crate::{CrystalCell, SymmetryOperation, SymmetryProvenance, moyo_backend};

    fn dataset(operations: Vec<SymmetryOperation>) -> SymmetryDataset {
        SymmetryDataset {
            operations,
            equivalent_atoms: vec![0],
            spacegroup_number: None,
            hermann_mauguin: None,
            provenance: SymmetryProvenance::Imported {
                code: "test".to_owned(),
            },
        }
    }

    fn operation(rotation: [[i32; 3]; 3]) -> SymmetryOperation {
        SymmetryOperation {
            rotation,
            translation: [0.0; 3],
            time_reversal: false,
        }
    }

    #[test]
    fn diamond_three_cube_has_four_normalized_orbits() {
        let half = Bohr(3.6);
        let zero = Bohr(0.0);
        let diamond = CrystalCell {
            lattice: [[zero, half, half], [half, zero, half], [half, half, zero]],
            positions: vec![[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            atomic_numbers: vec![14, 14],
        };
        let symmetry = moyo_backend::detect(&diamond, Bohr(1.0e-5)).unwrap();
        let reduced = reduce_regular_mesh(
            &symmetry,
            RegularKMesh {
                divisions: [3, 3, 3],
                shift: [0.0; 3],
            },
            true,
        )
        .unwrap();

        assert_eq!(reduced.full_points.len(), 27);
        assert_eq!(reduced.irreducible_points.len(), 4);
        let mut multiplicities = reduced
            .irreducible_points
            .iter()
            .map(|point| point.multiplicity)
            .collect::<Vec<_>>();
        multiplicities.sort_unstable();
        assert_eq!(multiplicities, vec![1, 6, 8, 12]);
        let weight_sum = reduced
            .irreducible_points
            .iter()
            .map(|point| point.weight)
            .sum::<f64>();
        assert!((weight_sum - 1.0).abs() < 1.0e-15);
        for point in &reduced.irreducible_points {
            assert_eq!(point.weight, point.multiplicity as f64 / 27.0);
        }
        for point in &reduced.full_points {
            let parent = &reduced.irreducible_points[point.parent];
            assert_eq!(point.representative, parent.representative);
        }
    }

    #[test]
    fn half_shift_keeps_only_compatible_operations() {
        let identity = operation([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        let swap_xy = operation([[0, 1, 0], [1, 0, 0], [0, 0, 1]]);
        let reduced = reduce_regular_mesh(
            &dataset(vec![identity, swap_xy]),
            RegularKMesh {
                divisions: [2, 2, 1],
                shift: [0.5, 0.0, 0.0],
            },
            false,
        )
        .unwrap();
        assert_eq!(reduced.active_operations.len(), 1);
        assert_eq!(reduced.irreducible_points.len(), 4);
    }

    #[test]
    fn optional_time_reversal_reduces_a_general_mesh() {
        let identity = operation([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        let mesh = RegularKMesh {
            divisions: [3, 1, 1],
            shift: [0.0; 3],
        };
        let without = reduce_regular_mesh(&dataset(vec![identity.clone()]), mesh, false).unwrap();
        let with = reduce_regular_mesh(&dataset(vec![identity]), mesh, true).unwrap();

        assert_eq!(without.irreducible_points.len(), 3);
        assert_eq!(with.irreducible_points.len(), 2);
        assert_eq!(with.full_points[2].representative, 1);
        assert_eq!(with.full_points[2].parent_operation.time_reversal, true);
    }

    #[test]
    fn dataset_antiunitary_operation_is_kept_without_extra_time_reversal() {
        let mut antiunitary_identity = operation([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        antiunitary_identity.time_reversal = true;
        let unitary_identity = operation([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        let reduced = reduce_regular_mesh(
            &dataset(vec![unitary_identity, antiunitary_identity]),
            RegularKMesh {
                divisions: [3, 1, 1],
                shift: [0.0; 3],
            },
            false,
        )
        .unwrap();

        assert_eq!(reduced.irreducible_points.len(), 2);
        assert_eq!(reduced.active_operations.len(), 2);
        assert!(
            reduced
                .active_operations
                .iter()
                .any(|operation| operation.operation_index == 1)
        );
    }

    #[test]
    fn non_unimodular_rotations_are_rejected() {
        let doubled_x = operation([[2, 0, 0], [0, 1, 0], [0, 0, 1]]);
        let error = reduce_regular_mesh(
            &dataset(vec![doubled_x]),
            RegularKMesh {
                divisions: [1; 3],
                shift: [0.0; 3],
            },
            false,
        )
        .unwrap_err();
        assert_eq!(
            error,
            KMeshReductionError::NonUnimodularRotation {
                operation_index: 0,
                determinant: 2,
            }
        );
    }

    #[test]
    fn incomplete_operation_tables_report_unmapped_points() {
        let cycle = operation([[0, 1, 0], [0, 0, 1], [1, 0, 0]]);
        let error = reduce_regular_mesh(
            &dataset(vec![cycle]),
            RegularKMesh {
                divisions: [2; 3],
                shift: [0.0; 3],
            },
            false,
        )
        .unwrap_err();
        assert!(matches!(error, KMeshReductionError::UnmappedPoint { .. }));
    }
}
