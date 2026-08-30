use std::sync::Arc;

use muffintin_core::Bohr;
use muffintin_operators::lapw::Provenance;
use muffintin_prodbasis::thc::{
    PairBlock, cholesky_pivots_from_pair_blocks, pivots_from_pair_blocks,
};
use num_complex::Complex64;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::checkpoint::{ScalarProductInput, ScalarProductSlice};

#[pyclass(name = "ScalarThcResult", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct ScalarThcResult {
    pub(crate) _checkpoint: Arc<muffintin_io::CheckpointV2>,
    pub(crate) _slice: ScalarProductSlice,
    pub(crate) inner: Arc<muffintin::ScalarThcResult>,
    pub(crate) grid: muffintin::ThcParentGrid,
    pub(crate) spec: muffintin::ScalarThcSpec,
    pub(crate) pair_blocks: Arc<Vec<PairBlock>>,
    pub(crate) candidates: Vec<usize>,
    pub(crate) selection_diagonal: Vec<f64>,
}

pub(crate) fn parse_parent_grid(
    input: &muffintin::ScalarProductInput,
    coordinates: PyReadonlyArray2<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    regions: PyReadonlyArray2<'_, i64>,
) -> PyResult<muffintin::ThcParentGrid> {
    let coordinates = coordinates.as_array();
    let weights = weights.as_array();
    let regions = regions.as_array();
    let n_points = coordinates.shape()[0];
    if coordinates.shape()[1] != 3 || regions.shape() != [n_points, 3] || weights.len() != n_points
    {
        return Err(PyValueError::new_err(
            "coordinates, weights, and regions must have shapes (P,3), (P,), and (P,3)",
        ));
    }
    let mut points = Vec::with_capacity(n_points);
    for point in 0..n_points {
        let region = match (
            regions[[point, 0]],
            regions[[point, 1]],
            regions[[point, 2]],
        ) {
            (0, site, radial) => muffintin::ThcRegion::MuffinTin {
                site: usize::try_from(site)
                    .map_err(|_| PyValueError::new_err("muffin-tin site must be nonnegative"))?,
                radial_index: usize::try_from(radial).map_err(|_| {
                    PyValueError::new_err("muffin-tin radial index must be nonnegative")
                })?,
            },
            (1, -1, -1) => muffintin::ThcRegion::Interstitial,
            _ => {
                return Err(PyValueError::new_err(
                    "regions rows must be (0, site, radial) or (1, -1, -1)",
                ));
            }
        };
        points.push(muffintin::ThcPoint {
            coordinate: [
                Bohr(coordinates[[point, 0]]),
                Bohr(coordinates[[point, 1]]),
                Bohr(coordinates[[point, 2]]),
            ],
            weight: weights[point],
            region,
        });
    }
    muffintin::ThcParentGrid::new(
        input.source.partition.clone(),
        Provenance::default(),
        points,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (slice, coordinates, weights, regions, spin, rank, engine, candidates=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_scalar_thc(
    slice: PyRef<'_, ScalarProductSlice>,
    coordinates: PyReadonlyArray2<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    regions: PyReadonlyArray2<'_, i64>,
    spin: u8,
    rank: usize,
    engine: &str,
    candidates: Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<ScalarThcResult> {
    let first = slice
        .inner
        .first()
        .ok_or_else(|| PyValueError::new_err("scalar product slice is empty"))?;
    let grid = parse_parent_grid(first, coordinates, weights, regions)?;
    let (candidates, candidate_policy) = match candidates {
        Some(values) => {
            let candidates = values
                .as_array()
                .iter()
                .map(|&value| {
                    usize::try_from(value)
                        .map_err(|_| PyValueError::new_err("candidate indices must be nonnegative"))
                })
                .collect::<PyResult<Vec<_>>>()?;
            (
                candidates.clone(),
                muffintin::ThcCandidates::Indices(candidates),
            )
        }
        None => (
            grid.points()
                .iter()
                .enumerate()
                .filter_map(|(index, point)| (point.weight > 0.0).then_some(index))
                .collect(),
            muffintin::ThcCandidates::All,
        ),
    };
    let engine = match engine {
        "qrcp" => muffintin::ThcEngine::FullColumnPivotedQr,
        "pivoted-cholesky" => muffintin::ThcEngine::FullPivotedCholesky,
        _ => {
            return Err(PyValueError::new_err(
                "engine must be 'qrcp' or 'pivoted-cholesky'",
            ));
        }
    };
    let spec = muffintin::ScalarThcSpec {
        spin,
        rank: muffintin::RankPolicy::Exact { n_mu: rank },
        candidates: candidate_policy,
        engine,
    };
    let result = muffintin::build_scalar_thc(slice.inner.as_slice(), &grid, &spec)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let samples = muffintin::sample_scalar_orbitals(first, &grid, spin)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let pair_blocks = build_pair_blocks(slice.inner.as_slice(), &grid, &samples)?;
    let restricted = restrict_blocks(&pair_blocks, &candidates)?;
    let candidate_weights = candidates
        .iter()
        .map(|&index| grid.points()[index].weight)
        .collect::<Vec<_>>();
    let (_, selection_diagonal) = match engine {
        muffintin::ThcEngine::FullColumnPivotedQr => {
            pivots_from_pair_blocks(&restricted, &candidate_weights, rank)
        }
        muffintin::ThcEngine::FullPivotedCholesky => {
            cholesky_pivots_from_pair_blocks(&restricted, &candidate_weights, rank)
        }
    }
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(ScalarThcResult {
        _checkpoint: Arc::clone(&slice.checkpoint),
        _slice: ScalarProductSlice::clone(&*slice),
        inner: Arc::new(result),
        grid,
        spec,
        pair_blocks: Arc::new(pair_blocks),
        candidates,
        selection_diagonal,
    })
}

#[pyfunction]
pub(crate) fn sample_scalar_orbitals(
    py: Python<'_>,
    input: PyRef<'_, ScalarProductInput>,
    coordinates: PyReadonlyArray2<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    regions: PyReadonlyArray2<'_, i64>,
    spin: u8,
) -> PyResult<Py<PyDict>> {
    let grid = parse_parent_grid(input.inner.as_ref(), coordinates, weights, regions)?;
    let samples = muffintin::sample_scalar_orbitals(input.inner.as_ref(), &grid, spin)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    crate::export::export_orbital_samples(py, &samples)
}

fn build_pair_blocks(
    inputs: &[muffintin::ScalarProductInput],
    grid: &muffintin::ThcParentGrid,
    samples: &muffintin::ScalarOrbitalSamples,
) -> PyResult<Vec<PairBlock>> {
    let layout = inputs[0].pair_columns;
    let n_col = layout
        .n_columns()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut blocks = Vec::with_capacity(inputs.len());
    for (q_index, input) in inputs.iter().enumerate() {
        let mut values = vec![Complex64::default(); grid.points().len() * n_col];
        for mapped in &input.k_minus_q {
            for (point_index, point) in grid.points().iter().enumerate() {
                let argument = mapped
                    .umklapp
                    .cartesian
                    .iter()
                    .zip(point.coordinate)
                    .map(|(g, r)| g.get() * r.get())
                    .sum();
                let phase = Complex64::from_polar(1.0, argument);
                for left in 0..layout.n_orb {
                    let left_index =
                        (point_index * samples.n_k + mapped.kq_index) * samples.n_orb + left;
                    for right in 0..layout.n_orb {
                        let right_index =
                            (point_index * samples.n_k + mapped.k_index) * samples.n_orb + right;
                        let column = layout.encode(mapped.k_index, left, right);
                        values[point_index * n_col + column] = phase
                            * (samples.large[left_index].conj() * samples.large[right_index]
                                + samples.small[left_index].conj() * samples.small[right_index]);
                    }
                }
            }
        }
        blocks.push(
            PairBlock::new(q_index, grid.points().len(), layout, values)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        );
    }
    Ok(blocks)
}

fn restrict_blocks(blocks: &[PairBlock], candidates: &[usize]) -> PyResult<Vec<PairBlock>> {
    blocks
        .iter()
        .map(|block| {
            let n_columns = block.n_columns();
            let mut values = Vec::with_capacity(candidates.len() * n_columns);
            for &point in candidates {
                if point >= block.n_points {
                    return Err(PyValueError::new_err("candidate index is outside the grid"));
                }
                let offset = point * n_columns;
                values.extend_from_slice(&block.values()[offset..offset + n_columns]);
            }
            PairBlock::new(block.q_index, candidates.len(), block.layout, values)
                .map_err(|error| PyValueError::new_err(error.to_string()))
        })
        .collect()
}
