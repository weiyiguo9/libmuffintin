//! Versioned SPEX symmetry+irrep dump (`libmuffintin.spexsym` v1).
//!
//! The producing side is a SPEX quantity dump; this module holds the schema
//! contract plus a reader and a reference writer, so the Fortran writer can
//! be diffed against files produced here. All indices are 0-based, all
//! coordinates fractional, and complex data uses a trailing `[re, im]` axis.
//! The schema and SPEX-side conventions live in
//! `doc/22_crystal_symmetry_and_spex_irrep_import.md`.

use std::path::Path;

use hdf5_metno::File;
use muffintin_symmetry::SymmetryOperation;
use muffintin_symmetry::spex::{KpointIrreps, SpexSymmetryImport, SubspaceIrreps};
use num_complex::Complex64;

use crate::error::{IoError, ValidationError};
use crate::mldump::{
    read_f64_dataset, read_i32_dataset, read_i64_attr, read_numeric_attr, read_str_attr,
    write_f64_dataset, write_i32_dataset, write_i64_attr, write_str_attr,
};

/// Stable schema name written on every SPEX symmetry dump.
pub const SPEX_SYMMETRY_SCHEMA_NAME: &str = "libmuffintin.spexsym";
/// Schema version implemented by this crate.
pub const SPEX_SYMMETRY_SCHEMA_VERSION: u32 = 1;

/// One read SPEX symmetry dump: producer identity plus the typed import.
#[derive(Clone, Debug, PartialEq)]
pub struct SpexSymmetryFileV1 {
    /// Producer program version string (for example the SPEX revision).
    pub producer_version: String,
    pub import: SpexSymmetryImport,
}

/// Write `import` as a `libmuffintin.spexsym` v1 file.
pub fn write_spex_symmetry_v1(
    path: impl AsRef<Path>,
    producer_version: &str,
    import: &SpexSymmetryImport,
) -> Result<(), IoError> {
    let n_ops = import.operations.len();
    let n_kpt = import.kpoints.len();
    require_len("operations.inverse", import.inverse.len(), n_ops)?;
    require_len("operations.atom_map", import.atom_map.len(), n_ops)?;
    let n_sites = import.atom_map.first().map_or(0, Vec::len);
    if n_sites == 0 {
        return Err(ValidationError::InvalidValue {
            path: "operations.atom_map".to_owned(),
            expected: "at least one site".to_owned(),
            actual: "zero sites".to_owned(),
        }
        .into());
    }
    require_len("kpoints.parent", import.parent.len(), n_kpt)?;
    require_len(
        "kpoints.parent_operation",
        import.parent_operation.len(),
        n_kpt,
    )?;

    let file = File::create(path)?;
    write_str_attr(&file, "schema_name", SPEX_SYMMETRY_SCHEMA_NAME)?;
    file.new_attr::<u32>()
        .create("schema_version")?
        .write_scalar(&SPEX_SYMMETRY_SCHEMA_VERSION)?;
    write_str_attr(&file, "producer_version", producer_version)?;

    let symmetry = file.create_group("symmetry")?;
    let mut rotations = Vec::with_capacity(n_ops * 9);
    let mut translations = Vec::with_capacity(n_ops * 3);
    let mut time_reversal = Vec::with_capacity(n_ops);
    for operation in &import.operations {
        rotations.extend(operation.rotation.iter().flatten().copied());
        translations.extend_from_slice(&operation.translation);
        time_reversal.push(i32::from(operation.time_reversal));
    }
    write_i32_dataset(
        &symmetry,
        "rotations",
        &[n_ops, 3, 3],
        &rotations,
        &["operation", "row", "column"],
    )?;
    write_f64_dataset(
        &symmetry,
        "translations",
        &[n_ops, 3],
        &translations,
        &["operation", "fractional"],
    )?;
    write_i32_dataset(
        &symmetry,
        "time_reversal",
        &[n_ops],
        &time_reversal,
        &["operation"],
    )?;
    write_i32_dataset(
        &symmetry,
        "inverse",
        &[n_ops],
        &to_i32("operations.inverse", &import.inverse)?,
        &["operation"],
    )?;
    let mut atom_map = Vec::with_capacity(n_ops * n_sites);
    for (operation, map) in import.atom_map.iter().enumerate() {
        require_len(
            &format!("operations.atom_map[{operation}]"),
            map.len(),
            n_sites,
        )?;
        atom_map.extend(to_i32("operations.atom_map", map)?);
    }
    write_i32_dataset(
        &symmetry,
        "atom_map",
        &[n_ops, n_sites],
        &atom_map,
        &["operation", "site"],
    )?;

    let kpoints = file.create_group("kpoints")?;
    write_i64_attr(
        &kpoints,
        "irreducible_count",
        import.irreducible_count as i64,
    )?;
    write_f64_dataset(
        &kpoints,
        "fractional",
        &[n_kpt, 3],
        &import.kpoints.iter().flatten().copied().collect::<Vec<_>>(),
        &["kpoint", "fractional"],
    )?;
    write_i32_dataset(
        &kpoints,
        "parent",
        &[n_kpt],
        &to_i32("kpoints.parent", &import.parent)?,
        &["kpoint"],
    )?;
    write_i32_dataset(
        &kpoints,
        "parent_operation",
        &[n_kpt],
        &to_i32("kpoints.parent_operation", &import.parent_operation)?,
        &["kpoint"],
    )?;

    let irreps = file.create_group("irreps")?;
    write_i64_attr(&irreps, "block_count", import.irreps.len() as i64)?;
    for (index, block) in import.irreps.iter().enumerate() {
        let group = irreps.create_group(&format!("block{index}"))?;
        let path = format!("irreps[{index}]");
        write_i64_attr(&group, "kpoint_index", block.kpoint_index as i64)?;
        write_i64_attr(&group, "spin", block.spin as i64)?;
        write_i64_attr(&group, "subspace_count", block.subspaces.len() as i64)?;
        let n_little = block.little_group.len();
        write_i32_dataset(
            &group,
            "little_group",
            &[n_little],
            &to_i32(&format!("{path}.little_group"), &block.little_group)?,
            &["little_group_operation"],
        )?;
        for (sub_index, subspace) in block.subspaces.iter().enumerate() {
            let sub_path = format!("{path}.subspaces[{sub_index}]");
            require_len(
                &format!("{sub_path}.matrices"),
                subspace.matrices.len(),
                n_little,
            )?;
            let d = subspace.dimension;
            let mut packed = Vec::with_capacity(n_little * d * d * 2);
            for matrix in &subspace.matrices {
                require_len(&format!("{sub_path}.matrices[..]"), matrix.len(), d * d)?;
                for entry in matrix {
                    packed.push(entry.re);
                    packed.push(entry.im);
                }
            }
            write_f64_dataset(
                &group,
                &format!("subspace{sub_index}"),
                &[n_little, d, d, 2],
                &packed,
                &["little_group_operation", "row", "column", "complex"],
            )?;
            let dataset = group.dataset(&format!("subspace{sub_index}"))?;
            write_i64_attr(&dataset, "first_band", subspace.first_band as i64)?;
        }
    }
    Ok(())
}

/// Read a `libmuffintin.spexsym` v1 file into the typed import.
pub fn read_spex_symmetry_v1(path: impl AsRef<Path>) -> Result<SpexSymmetryFileV1, IoError> {
    let file = File::open(path)?;
    let schema_name = read_str_attr(&file, "schema_name")?;
    if schema_name != SPEX_SYMMETRY_SCHEMA_NAME {
        return Err(IoError::InvalidFormat {
            expected: SPEX_SYMMETRY_SCHEMA_NAME,
            found: schema_name,
        });
    }
    let schema_version =
        read_numeric_attr::<u32>(&file, "schema_version", "/@schema_version/dtype")?;
    if schema_version != SPEX_SYMMETRY_SCHEMA_VERSION {
        return Err(IoError::UnsupportedVersion {
            format: SPEX_SYMMETRY_SCHEMA_NAME,
            supported: SPEX_SYMMETRY_SCHEMA_VERSION,
            found: schema_version,
        });
    }
    let producer_version = read_str_attr(&file, "producer_version")?;

    let symmetry = file.group("symmetry")?;
    let n_ops = leading_extent(&symmetry, "rotations")?;
    let n_sites = symmetry
        .dataset("atom_map")?
        .shape()
        .get(1)
        .copied()
        .ok_or_else(|| ValidationError::InvalidValue {
            path: "/symmetry/atom_map/shape".to_owned(),
            expected: "[operation, site]".to_owned(),
            actual: "rank < 2".to_owned(),
        })?;
    if n_sites == 0 {
        return Err(ValidationError::InvalidValue {
            path: "/symmetry/atom_map/shape".to_owned(),
            expected: "nonzero site extent".to_owned(),
            actual: "[operation, 0]".to_owned(),
        }
        .into());
    }
    let rotations = read_i32_dataset(
        &symmetry,
        "rotations",
        &[n_ops, 3, 3],
        &["operation", "row", "column"],
    )?;
    let translations = read_f64_dataset(
        &symmetry,
        "translations",
        &[n_ops, 3],
        &["operation", "fractional"],
    )?;
    let time_reversal = read_i32_dataset(&symmetry, "time_reversal", &[n_ops], &["operation"])?;
    let inverse = read_i32_dataset(&symmetry, "inverse", &[n_ops], &["operation"])?;
    let atom_map = read_i32_dataset(
        &symmetry,
        "atom_map",
        &[n_ops, n_sites],
        &["operation", "site"],
    )?;

    let mut operations = Vec::with_capacity(n_ops);
    for op in 0..n_ops {
        let mut rotation = [[0_i32; 3]; 3];
        for (row, target) in rotation.iter_mut().enumerate() {
            target.copy_from_slice(&rotations[op * 9 + row * 3..op * 9 + row * 3 + 3]);
        }
        let time_reversal = match time_reversal[op] {
            0 => false,
            1 => true,
            other => {
                return Err(ValidationError::InvalidValue {
                    path: format!("/symmetry/time_reversal[{op}]"),
                    expected: "0 or 1".to_owned(),
                    actual: other.to_string(),
                }
                .into());
            }
        };
        operations.push(SymmetryOperation {
            rotation,
            translation: [
                translations[op * 3],
                translations[op * 3 + 1],
                translations[op * 3 + 2],
            ],
            time_reversal,
        });
    }
    let inverse = indices("/symmetry/inverse", &inverse, n_ops)?;
    let atom_map = atom_map
        .chunks_exact(n_sites)
        .map(|chunk| indices("/symmetry/atom_map", chunk, n_sites))
        .collect::<Result<Vec<_>, _>>()?;

    let kpoints_group = file.group("kpoints")?;
    let n_kpt = leading_extent(&kpoints_group, "fractional")?;
    let irreducible_count = index(
        "/kpoints/@irreducible_count",
        read_i64_attr(
            &kpoints_group,
            "irreducible_count",
            "/kpoints/@irreducible_count",
        )?,
        n_kpt + 1,
    )?;
    if n_kpt > 0 && irreducible_count == 0 {
        return Err(ValidationError::InvalidValue {
            path: "/kpoints/@irreducible_count".to_owned(),
            expected: "positive when k-points are present".to_owned(),
            actual: "0".to_owned(),
        }
        .into());
    }
    let fractional = read_f64_dataset(
        &kpoints_group,
        "fractional",
        &[n_kpt, 3],
        &["kpoint", "fractional"],
    )?;
    let kpoints = fractional
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();
    let parent = indices(
        "/kpoints/parent",
        &read_i32_dataset(&kpoints_group, "parent", &[n_kpt], &["kpoint"])?,
        irreducible_count,
    )?;
    let parent_operation = indices(
        "/kpoints/parent_operation",
        &read_i32_dataset(&kpoints_group, "parent_operation", &[n_kpt], &["kpoint"])?,
        n_ops,
    )?;

    let irreps_group = file.group("irreps")?;
    let block_count = nonnegative(
        "/irreps/@block_count",
        read_i64_attr(&irreps_group, "block_count", "/irreps/@block_count")?,
    )?;
    let mut irreps = Vec::with_capacity(block_count);
    for block in 0..block_count {
        let group = irreps_group.group(&format!("block{block}"))?;
        let path = format!("/irreps/block{block}");
        let kpoint_index = index(
            &format!("{path}/@kpoint_index"),
            read_i64_attr(&group, "kpoint_index", &format!("{path}/@kpoint_index"))?,
            n_kpt,
        )?;
        let spin = nonnegative(
            &format!("{path}/@spin"),
            read_i64_attr(&group, "spin", &format!("{path}/@spin"))?,
        )?;
        let subspace_count = nonnegative(
            &format!("{path}/@subspace_count"),
            read_i64_attr(&group, "subspace_count", &format!("{path}/@subspace_count"))?,
        )?;
        let n_little = leading_extent(&group, "little_group")?;
        let little_group = indices(
            &format!("{path}/little_group"),
            &read_i32_dataset(
                &group,
                "little_group",
                &[n_little],
                &["little_group_operation"],
            )?,
            n_ops,
        )?;
        let mut subspaces = Vec::with_capacity(subspace_count);
        for sub in 0..subspace_count {
            let name = format!("subspace{sub}");
            let sub_path = format!("{path}/{name}");
            let dataset = group.dataset(&name)?;
            let shape = dataset.shape();
            let dimension = match shape.as_slice() {
                [ops, d, d2, 2] if *ops == n_little && d == d2 => *d,
                _ => {
                    return Err(ValidationError::InvalidValue {
                        path: format!("{sub_path}/shape"),
                        expected: format!("[{n_little}, d, d, 2]"),
                        actual: format!("{shape:?}"),
                    }
                    .into());
                }
            };
            let first_band = nonnegative(
                &format!("{sub_path}/@first_band"),
                read_i64_attr(&dataset, "first_band", &format!("{sub_path}/@first_band"))?,
            )?;
            let packed = read_f64_dataset(
                &group,
                &name,
                &[n_little, dimension, dimension, 2],
                &["little_group_operation", "row", "column", "complex"],
            )?;
            let matrices = packed
                .chunks_exact(dimension * dimension * 2)
                .map(|matrix| {
                    matrix
                        .chunks_exact(2)
                        .map(|entry| Complex64::new(entry[0], entry[1]))
                        .collect()
                })
                .collect();
            subspaces.push(SubspaceIrreps {
                first_band,
                dimension,
                matrices,
            });
        }
        irreps.push(KpointIrreps {
            kpoint_index,
            spin,
            little_group,
            subspaces,
        });
    }

    Ok(SpexSymmetryFileV1 {
        producer_version,
        import: SpexSymmetryImport {
            operations,
            inverse,
            atom_map,
            kpoints,
            irreducible_count,
            parent,
            parent_operation,
            irreps,
        },
    })
}

fn leading_extent(group: &hdf5_metno::Group, name: &str) -> Result<usize, IoError> {
    group
        .dataset(name)?
        .shape()
        .first()
        .copied()
        .ok_or_else(|| {
            ValidationError::InvalidValue {
                path: format!("{}/{name}/shape", group.name()),
                expected: "leading extent".to_owned(),
                actual: "scalar".to_owned(),
            }
            .into()
        })
}

fn require_len(path: &str, actual: usize, expected: usize) -> Result<(), ValidationError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ValidationError::LengthMismatch {
            path: path.to_owned(),
            actual,
            expected,
        })
    }
}

fn to_i32(path: &str, values: &[usize]) -> Result<Vec<i32>, ValidationError> {
    values
        .iter()
        .map(|&value| {
            i32::try_from(value).map_err(|_| ValidationError::InvalidValue {
                path: path.to_owned(),
                expected: "index representable as i32".to_owned(),
                actual: value.to_string(),
            })
        })
        .collect()
}

fn index(path: &str, value: i64, bound: usize) -> Result<usize, IoError> {
    let converted = nonnegative(path, value)?;
    if converted < bound {
        Ok(converted)
    } else {
        Err(ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: format!("index below {bound}"),
            actual: value.to_string(),
        }
        .into())
    }
}

fn nonnegative(path: &str, value: i64) -> Result<usize, IoError> {
    usize::try_from(value).map_err(|_| {
        ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "nonnegative value representable as usize".to_owned(),
            actual: value.to_string(),
        }
        .into()
    })
}

fn indices(path: &str, values: &[i32], bound: usize) -> Result<Vec<usize>, IoError> {
    values
        .iter()
        .map(|&value| index(path, i64::from(value), bound))
        .collect()
}
