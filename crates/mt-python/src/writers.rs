use std::path::PathBuf;

use muffintin_io::{
    MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON,
    MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION, MldumpGeometryV1, MldumpHeaderV1,
    MldumpKMinusQV1, MldumpKPointV1, MldumpMeshV1, MldumpMetaV1, MldumpQEntryV1,
    MldumpRadialMeshV1, MldumpSiteV1,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::checkpoint::ScalarProductSlice;
use crate::coulomb::ScalarCoulombResult;
use crate::spinor::{SpinorCoulombResult, SpinorProductSlice, SpinorThcResult};
use crate::thc::ScalarThcResult;

struct HeaderMetadata {
    producer_name: String,
    producer_version: String,
    source_revision: String,
    site_species: Vec<Option<String>>,
    site_labels: Vec<Option<String>>,
}

fn require_site_metadata(
    n_sites: usize,
    species: &[Option<String>],
    labels: &[Option<String>],
) -> PyResult<()> {
    if species.len() != n_sites || labels.len() != n_sites {
        return Err(PyValueError::new_err(format!(
            "site_species and site_labels must each contain {n_sites} entries"
        )));
    }
    Ok(())
}

fn scalar_header(
    slice: &ScalarProductSlice,
    coulomb: &ScalarCoulombResult,
    metadata: HeaderMetadata,
) -> PyResult<MldumpHeaderV1> {
    let first = slice
        .inner
        .first()
        .ok_or_else(|| PyValueError::new_err("scalar product slice is empty"))?;
    let n_sites = first.source.partition.site_count();
    require_site_metadata(n_sites, &metadata.site_species, &metadata.site_labels)?;
    let sites = first
        .source
        .partition
        .sites()
        .iter()
        .zip(&first.source.radials)
        .zip(metadata.site_species.into_iter().zip(metadata.site_labels))
        .map(|((site, radials), (species, label))| MldumpSiteV1 {
            species,
            label,
            position_bohr: site.position.map(|component| component.get()),
            radius_bohr: site.radius.get(),
            radial_mesh: MldumpRadialMeshV1 {
                first_bohr: radials.mesh.first().get(),
                log_increment: radials.mesh.increment(),
                point_count: radials.mesh.len(),
            },
        })
        .collect();
    Ok(header(
        metadata.producer_name,
        metadata.producer_version,
        metadata.source_revision,
        MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON,
        coulomb
            .spec
            .request
            .cell()
            .basis()
            .map(|row| row.map(|component| component.get())),
        first
            .reciprocal
            .basis()
            .map(|row| row.map(|component| component.get())),
        coulomb.spec.request.cell().volume().get(),
        sites,
        &first.orbitals.k_fractional,
        slice.inner.as_slice().iter().map(|input| {
            (
                input.source.q.umklapp.index,
                input
                    .k_minus_q
                    .iter()
                    .map(|mapped| MldumpKMinusQV1 {
                        k_index: mapped.k_index,
                        mapped_index: mapped.kq_index,
                        g_wrap: mapped.umklapp.index,
                    })
                    .collect(),
            )
        }),
    ))
}

fn spinor_header(
    slice: &SpinorProductSlice,
    coulomb: &SpinorCoulombResult,
    metadata: HeaderMetadata,
) -> PyResult<MldumpHeaderV1> {
    let first = slice
        .inner
        .first()
        .ok_or_else(|| PyValueError::new_err("spinor product slice is empty"))?;
    let n_sites = first.source.partition.site_count();
    require_site_metadata(n_sites, &metadata.site_species, &metadata.site_labels)?;
    let sites = first
        .source
        .partition
        .sites()
        .iter()
        .zip(&first.source.radials)
        .zip(metadata.site_species.into_iter().zip(metadata.site_labels))
        .map(|((site, radials), (species, label))| MldumpSiteV1 {
            species,
            label,
            position_bohr: site.position.map(|component| component.get()),
            radius_bohr: site.radius.get(),
            radial_mesh: MldumpRadialMeshV1 {
                first_bohr: radials.mesh.first().get(),
                log_increment: radials.mesh.increment(),
                point_count: radials.mesh.len(),
            },
        })
        .collect();
    Ok(header(
        metadata.producer_name,
        metadata.producer_version,
        metadata.source_revision,
        MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION,
        coulomb
            .spec
            .request
            .cell()
            .basis()
            .map(|row| row.map(|component| component.get())),
        first
            .reciprocal
            .basis()
            .map(|row| row.map(|component| component.get())),
        coulomb.spec.request.cell().volume().get(),
        sites,
        &first.orbitals.k_fractional,
        slice.inner.as_slice().iter().map(|input| {
            (
                input.source.q.umklapp.index,
                input
                    .k_minus_q
                    .iter()
                    .map(|mapped| MldumpKMinusQV1 {
                        k_index: mapped.k_index,
                        mapped_index: mapped.kq_index,
                        g_wrap: mapped.umklapp.index,
                    })
                    .collect(),
            )
        }),
    ))
}

#[allow(clippy::too_many_arguments)]
fn header(
    producer_name: String,
    producer_version: String,
    source_revision: String,
    feature_representation: &str,
    direct_basis_bohr: [[f64; 3]; 3],
    reciprocal_basis_inv_bohr: [[f64; 3]; 3],
    cell_volume_bohr3: f64,
    sites: Vec<MldumpSiteV1>,
    k_fractional: &[[f64; 3]],
    q_maps: impl IntoIterator<Item = ([i32; 3], Vec<MldumpKMinusQV1>)>,
) -> MldumpHeaderV1 {
    let n_k = k_fractional.len();
    let weight = 1.0 / n_k as f64;
    let q_entries = q_maps
        .into_iter()
        .enumerate()
        .map(|(q_index, (global_umklapp, k_minus_q))| {
            let canonical_fractional = k_fractional[q_index];
            MldumpQEntryV1 {
                input_fractional: std::array::from_fn(|axis| {
                    canonical_fractional[axis] + f64::from(global_umklapp[axis])
                }),
                canonical_fractional,
                global_umklapp,
                k_minus_q,
            }
        })
        .collect();
    MldumpHeaderV1::new(
        MldumpMetaV1 {
            producer_name,
            producer_version,
            source_revision,
            feature_representation: feature_representation.to_owned(),
        },
        MldumpGeometryV1 {
            direct_basis_bohr,
            reciprocal_basis_inv_bohr,
            cell_volume_bohr3,
            sites,
        },
        MldumpMeshV1 {
            k_points: k_fractional
                .iter()
                .map(|fractional| MldumpKPointV1 {
                    fractional: *fractional,
                    weight,
                })
                .collect(),
            q_entries,
        },
    )
}

#[pyfunction]
#[pyo3(signature = (path, slice, thc, coulomb, producer_name, producer_version, source_revision, site_species, site_labels))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_scalar_mldump(
    path: PathBuf,
    slice: PyRef<'_, ScalarProductSlice>,
    thc: PyRef<'_, ScalarThcResult>,
    coulomb: PyRef<'_, ScalarCoulombResult>,
    producer_name: String,
    producer_version: String,
    source_revision: String,
    site_species: Vec<Option<String>>,
    site_labels: Vec<Option<String>>,
) -> PyResult<()> {
    let metadata = HeaderMetadata {
        producer_name,
        producer_version,
        source_revision,
        site_species,
        site_labels,
    };
    let header = scalar_header(&slice, &coulomb, metadata)?;
    muffintin::write_scalar_mldump(
        path,
        &header,
        slice.inner.as_slice(),
        thc.inner.as_ref(),
        coulomb.inner.as_ref(),
        &coulomb.spec,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (path, slice, thc, coulomb, producer_name, producer_version, source_revision, site_species, site_labels))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_spinor_mldump(
    path: PathBuf,
    slice: PyRef<'_, SpinorProductSlice>,
    thc: PyRef<'_, SpinorThcResult>,
    coulomb: PyRef<'_, SpinorCoulombResult>,
    producer_name: String,
    producer_version: String,
    source_revision: String,
    site_species: Vec<Option<String>>,
    site_labels: Vec<Option<String>>,
) -> PyResult<()> {
    let metadata = HeaderMetadata {
        producer_name,
        producer_version,
        source_revision,
        site_species,
        site_labels,
    };
    let header = spinor_header(&slice, &coulomb, metadata)?;
    muffintin::write_spinor_mldump(
        path,
        &header,
        slice.inner.as_slice(),
        thc.inner.as_ref(),
        coulomb.inner.as_ref(),
        &coulomb.spec,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
pub(crate) fn write_scalar_coqui_cholesky(
    path: PathBuf,
    slice: PyRef<'_, ScalarProductSlice>,
    thc: PyRef<'_, ScalarThcResult>,
    coulomb: PyRef<'_, ScalarCoulombResult>,
    tolerance: f64,
) -> PyResult<()> {
    muffintin::write_scalar_coqui_cholesky(
        &path,
        slice.inner.as_slice(),
        thc.inner.as_ref(),
        coulomb.inner.as_ref(),
        &coulomb.spec,
        muffintin::ScalarCoquiCholeskySpec { tolerance },
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(write_scalar_mldump, module)?)?;
    module.add_function(wrap_pyfunction!(write_spinor_mldump, module)?)?;
    module.add_function(wrap_pyfunction!(write_scalar_coqui_cholesky, module)?)?;
    Ok(())
}
