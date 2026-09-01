//! Frozen-orbital Hartree–Fock exchange from sampled-ζ Weinert Coulomb records.

use crate::scalar_coulomb::{ScalarCoulombQRecord, ScalarCoulombResult};
use crate::scalar_product::{
    ScalarKMinusQ, ScalarProductInput, ScalarQSliceError, require_scalar_q_slice,
};
use crate::spinor_coulomb::{SpinorCoulombQRecord, SpinorCoulombResult};
use crate::spinor_mpb::SpinorMpbResult;
use crate::spinor_product::{
    SpinorKMinusQ, SpinorProductInput, SpinorQSliceError, require_spinor_q_slice,
};
use muffintin_core::Hartree;
use muffintin_coulomb::{
    AuxiliaryKind, CoulombError, CoulombOperator, CoulombRequest, assemble_coulomb,
};
use muffintin_prodbasis::{
    AuxiliaryPartition, CompiledAuxiliaryBasis, ExchangePairLayout, ExchangeSpace, OrbitalPair,
    PairColumnLayout, PairVertex, TransferQ,
};
use num_complex::Complex64;
use thiserror::Error;

const WEIGHT_TOLERANCE: f64 = 1.0e-12;
const HERMITICITY_TOLERANCE: f64 = 1.0e-8;

/// Explicit treatment of the periodic Gamma singularity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GammaExchangeTreatment {
    /// Use the finite Weinert/SPEX Gamma body only.
    ///
    /// This is the molecule-in-box neutralizing-background convention. Cell
    /// size convergence must be checked; no divergent head is silently added.
    FiniteBody,
    /// Reject any q slice containing the separated Gamma head.
    Reject,
}

/// Occupations and k weights for one scalar spin channel or one spinor band manifold.
#[derive(Clone, Debug, PartialEq)]
pub struct IsdfExchangeSpec {
    /// Positive k weights in production k order; they must sum to one.
    pub k_weights: Vec<f64>,
    /// Fractional occupations `[k][band]`, each in `[0,1]`.
    pub occupations: Vec<Vec<f64>>,
    pub gamma: GammaExchangeTreatment,
}

/// Exchange matrix in the frozen orbital band basis at one k point.
#[derive(Clone, Debug, PartialEq)]
pub struct IsdfExchangeBandMatrix {
    k_index: usize,
    n_bands: usize,
    /// Row-major Hermitian matrix in Hartree.
    values: Vec<Complex64>,
}

impl IsdfExchangeBandMatrix {
    pub const fn k_index(&self) -> usize {
        self.k_index
    }

    pub const fn n_bands(&self) -> usize {
        self.n_bands
    }

    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    pub fn element(&self, row: usize, column: usize) -> Option<Complex64> {
        if row >= self.n_bands || column >= self.n_bands {
            return None;
        }
        let index = row.checked_mul(self.n_bands)?.checked_add(column)?;
        self.values.get(index).copied()
    }
}

/// Frozen-orbital exact-exchange energy and band-space Fock contribution.
#[derive(Clone, Debug, PartialEq)]
pub struct IsdfExchangeResult {
    pub exchange_energy: Hartree,
    pub band_matrices: Vec<IsdfExchangeBandMatrix>,
    pub maximum_antihermitian_residual: f64,
}

/// ISDF exchange contraction failure.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum IsdfExchangeError {
    #[error(transparent)]
    Coulomb(#[from] CoulombError),
    #[error("ISDF exchange q slice must be nonempty")]
    EmptySlice,
    #[error("ISDF exchange requires one canonical q record per k point: q={q_count}, k={k_count}")]
    QCount { q_count: usize, k_count: usize },
    #[error("ISDF exchange q record {index} does not match its frozen product input")]
    QContext { index: usize },
    #[error("ISDF exchange inputs do not match the frozen context sealed by Coulomb construction")]
    FrozenInputContext,
    #[error("exact MPB exchange received {actual} q results for {expected} frozen inputs")]
    MpbCount { actual: usize, expected: usize },
    #[error("exact MPB exchange result at q index {index} does not match its frozen input")]
    MpbContext { index: usize },
    #[error(
        "exact MPB exchange q index {q_index} contains {actual} vertices, expected every one of {expected} VV columns"
    )]
    MpbVertexCount {
        q_index: usize,
        actual: usize,
        expected: usize,
    },
    #[error("exact MPB exchange q index {q_index} repeats VV column {column}")]
    MpbDuplicateColumn { q_index: usize, column: usize },
    #[error("exact MPB exchange q index {q_index} is missing VV column {column}")]
    MpbMissingColumn { q_index: usize, column: usize },
    #[error("ISDF exchange k-minus-q map is invalid at q={q_index}, k={k_index}")]
    KMinusQ { q_index: usize, k_index: usize },
    #[error("ISDF exchange received {actual} k weights for {expected} k points")]
    KWeightCount { actual: usize, expected: usize },
    #[error("ISDF exchange k weight {index} is invalid: {value}")]
    KWeight { index: usize, value: f64 },
    #[error("ISDF exchange k weights sum to {sum}, expected one")]
    KWeightSum { sum: f64 },
    #[error("ISDF exchange received {actual} occupation rows for {expected} k points")]
    OccupationKCount { actual: usize, expected: usize },
    #[error("ISDF exchange occupation row {k_index} has {actual} bands, expected {expected}")]
    OccupationBandCount {
        k_index: usize,
        actual: usize,
        expected: usize,
    },
    #[error("ISDF exchange occupation at k={k_index}, band={band} is invalid: {value}")]
    Occupation {
        k_index: usize,
        band: usize,
        value: f64,
    },
    #[error("ISDF exchange scalar spin {spin} is absent from the frozen orbitals")]
    ScalarSpin { spin: u8 },
    #[error("ISDF exchange rejected the separated Gamma Coulomb head at q index {q_index}")]
    GammaHead { q_index: usize },
    #[error(
        "ISDF exchange band matrix is not Hermitian; maximum residual {residual} exceeds {tolerance}"
    )]
    NonHermitian { residual: f64, tolerance: f64 },
}

pub(crate) struct RectangularExchangeRecord<'a> {
    pub(crate) layout: ExchangePairLayout,
    pub(crate) vertices: &'a [PairVertex],
    pub(crate) operator: &'a CoulombOperator,
}

pub(crate) struct RectangularExchangeResult {
    pub(crate) target_matrices: Vec<IsdfExchangeBandMatrix>,
    pub(crate) maximum_antihermitian_residual: f64,
}

/// Build one collinear-spin frozen-orbital exchange contribution.
pub fn build_scalar_isdf_exchange(
    inputs: &[ScalarProductInput],
    coulomb: &ScalarCoulombResult,
    spec: &IsdfExchangeSpec,
) -> Result<IsdfExchangeResult, IsdfExchangeError> {
    let first = require_scalar_q_slice(inputs).map_err(scalar_q_slice_error)?;
    if !coulomb.frozen_inputs_match(inputs) {
        return Err(IsdfExchangeError::FrozenInputContext);
    }
    if !first
        .orbitals
        .channels
        .iter()
        .any(|channel| channel.spin == coulomb.spin)
    {
        return Err(IsdfExchangeError::ScalarSpin { spin: coulomb.spin });
    }
    require_scalar_record_context(inputs, coulomb)?;
    let maps = scalar_maps(coulomb.frozen_k_minus_q(), first.pair_columns.n_k)?;
    let records = coulomb
        .records
        .iter()
        .map(scalar_record)
        .collect::<Vec<_>>();
    contract_exchange(first.pair_columns, &maps, &records, spec)
}

/// Build one full-first-variation spinor frozen-orbital exchange contribution.
pub fn build_spinor_isdf_exchange(
    inputs: &[SpinorProductInput],
    coulomb: &SpinorCoulombResult,
    spec: &IsdfExchangeSpec,
) -> Result<IsdfExchangeResult, IsdfExchangeError> {
    let first = require_spinor_q_slice(inputs).map_err(spinor_q_slice_error)?;
    if !coulomb.frozen_inputs_match(inputs) {
        return Err(IsdfExchangeError::FrozenInputContext);
    }
    require_spinor_record_context(inputs, coulomb)?;
    let maps = spinor_maps(coulomb.frozen_k_minus_q(), first.pair_columns.n_k)?;
    let records = coulomb
        .records()
        .iter()
        .map(spinor_record)
        .collect::<Vec<_>>();
    contract_exchange(first.pair_columns, &maps, &records, spec)
}

/// Build the canonical exact mixed-product VV exchange contribution.
///
/// Every [`SpinorMpbResult`] must contain every square-layout band-pair column
/// exactly once. The Coulomb body is assembled directly on the retained MPB;
/// a separated Gamma head is handled only through [`GammaExchangeTreatment`].
pub fn build_spinor_mpb_exchange(
    inputs: &[SpinorProductInput],
    mpb: &[SpinorMpbResult],
    request: &CoulombRequest,
    spec: &IsdfExchangeSpec,
) -> Result<IsdfExchangeResult, IsdfExchangeError> {
    let first = require_spinor_q_slice(inputs).map_err(spinor_q_slice_error)?;
    if mpb.len() != inputs.len() {
        return Err(IsdfExchangeError::MpbCount {
            actual: mpb.len(),
            expected: inputs.len(),
        });
    }
    if request.reciprocal() != &first.reciprocal {
        return Err(IsdfExchangeError::MpbContext { index: 0 });
    }
    let expected = first
        .pair_columns
        .n_columns()
        .map_err(|_| IsdfExchangeError::MpbContext { index: 0 })?;
    let mut ordered_vertices = Vec::with_capacity(mpb.len());
    let mut operators = Vec::with_capacity(mpb.len());
    for (q_index, (input, result)) in inputs.iter().zip(mpb).enumerate() {
        if !result.frozen_input_identity().matches(input)
            || result.reciprocal != input.reciprocal
            || result.pair_columns != input.pair_columns
            || result.auxiliary.q != input.source.q
            || result.auxiliary.partition != input.source.partition
        {
            return Err(IsdfExchangeError::MpbContext { index: q_index });
        }
        if result.vertices.len() != expected {
            return Err(IsdfExchangeError::MpbVertexCount {
                q_index,
                actual: result.vertices.len(),
                expected,
            });
        }
        let mut columns = vec![None; expected];
        for selected in &result.vertices {
            if selected.column >= expected
                || input.pair_columns.decode(selected.column)
                    != (selected.k, selected.left_band, selected.right_band)
                || selected.vertex.pair()
                    != (OrbitalPair::Bloch {
                        k_index: selected.k,
                        left: selected.left_band,
                        right: selected.right_band,
                    })
            {
                return Err(IsdfExchangeError::MpbContext { index: q_index });
            }
            if columns[selected.column]
                .replace(selected.vertex.clone())
                .is_some()
            {
                return Err(IsdfExchangeError::MpbDuplicateColumn {
                    q_index,
                    column: selected.column,
                });
            }
        }
        let columns = columns
            .into_iter()
            .enumerate()
            .map(|(column, vertex)| {
                vertex.ok_or(IsdfExchangeError::MpbMissingColumn { q_index, column })
            })
            .collect::<Result<Vec<_>, _>>()?;
        ordered_vertices.push(columns);
        operators.push(assemble_coulomb(&result.auxiliary, request)?);
    }
    let records = ordered_vertices
        .iter()
        .zip(&operators)
        .map(|(vertices, operator)| RectangularExchangeRecord {
            layout: valence_layout(first.pair_columns),
            vertices,
            operator,
        })
        .collect::<Vec<_>>();
    let maps = inputs
        .iter()
        .map(|input| input.k_minus_q.clone())
        .collect::<Vec<_>>();
    let maps = spinor_maps(&maps, first.pair_columns.n_k)?;
    contract_exchange(first.pair_columns, &maps, &records, spec)
}

fn scalar_record(record: &ScalarCoulombQRecord) -> RectangularExchangeRecord<'_> {
    RectangularExchangeRecord {
        layout: valence_layout(record.layout),
        vertices: &record.vertices,
        operator: &record.operator,
    }
}

fn spinor_record(record: &SpinorCoulombQRecord) -> RectangularExchangeRecord<'_> {
    RectangularExchangeRecord {
        layout: valence_layout(record.layout),
        vertices: &record.vertices,
        operator: &record.operator,
    }
}

fn require_scalar_record_context(
    inputs: &[ScalarProductInput],
    coulomb: &ScalarCoulombResult,
) -> Result<(), IsdfExchangeError> {
    if inputs.len() != coulomb.records.len() {
        return Err(IsdfExchangeError::QCount {
            q_count: coulomb.records.len(),
            k_count: inputs.len(),
        });
    }
    for (index, (input, record)) in inputs.iter().zip(&coulomb.records).enumerate() {
        if record.spin != coulomb.spin {
            return Err(IsdfExchangeError::QContext { index });
        }
        require_record_semantics(
            index,
            input.source.q,
            input.pair_columns,
            &input.source.partition,
            record.q_index,
            record.q,
            record.layout,
            &record.auxiliary,
            &record.vertices,
            &record.operator,
            &coulomb.sealed_spec().request,
        )?;
    }
    Ok(())
}

fn require_spinor_record_context(
    inputs: &[SpinorProductInput],
    coulomb: &SpinorCoulombResult,
) -> Result<(), IsdfExchangeError> {
    if inputs.len() != coulomb.records().len() {
        return Err(IsdfExchangeError::QCount {
            q_count: coulomb.records().len(),
            k_count: inputs.len(),
        });
    }
    for (index, (input, record)) in inputs.iter().zip(coulomb.records()).enumerate() {
        require_record_semantics(
            index,
            input.source.q,
            input.pair_columns,
            &input.source.partition,
            record.q_index,
            record.q,
            record.layout,
            &record.auxiliary,
            &record.vertices,
            &record.operator,
            &coulomb.sealed_spec().request,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn require_record_semantics(
    index: usize,
    input_q: TransferQ,
    input_layout: PairColumnLayout,
    partition: &AuxiliaryPartition,
    record_q_index: usize,
    record_q: TransferQ,
    record_layout: PairColumnLayout,
    auxiliary: &CompiledAuxiliaryBasis,
    vertices: &[PairVertex],
    operator: &CoulombOperator,
    request: &CoulombRequest,
) -> Result<(), IsdfExchangeError> {
    let n_columns = record_layout
        .n_columns()
        .map_err(|_| IsdfExchangeError::QContext { index })?;
    let valid = record_q_index == index
        && record_q == input_q
        && record_layout == input_layout
        && auxiliary.q == record_q
        && &auxiliary.partition == partition
        && vertices.len() == n_columns
        && operator.dimension() == auxiliary.dimension()
        && operator.q() == record_q
        && operator.cell() == request.cell()
        && operator.reciprocal() == request.reciprocal()
        && operator.layout() == &auxiliary.layout()
        && operator.kind() == AuxiliaryKind::InterpolationPoints;
    if !valid {
        return Err(IsdfExchangeError::QContext { index });
    }
    let auxiliary_layout = auxiliary.layout();
    for (column, vertex) in vertices.iter().enumerate() {
        match vertex.pair() {
            OrbitalPair::Bloch {
                k_index,
                left,
                right,
            } if vertex.layout() == &auxiliary_layout
                && record_layout.decode(column) == (k_index, left, right)
                && vertex.provenance() == &auxiliary.provenance => {}
            _ => return Err(IsdfExchangeError::QContext { index }),
        }
    }
    Ok(())
}

fn scalar_maps(
    maps: &[Vec<ScalarKMinusQ>],
    n_k: usize,
) -> Result<Vec<Vec<usize>>, IsdfExchangeError> {
    maps.iter()
        .enumerate()
        .map(|(q_index, map)| {
            map.iter()
                .enumerate()
                .map(|(k_index, mapped)| {
                    if mapped.k_index != k_index || mapped.kq_index >= n_k {
                        return Err(IsdfExchangeError::KMinusQ { q_index, k_index });
                    }
                    Ok(mapped.kq_index)
                })
                .collect()
        })
        .collect()
}

fn spinor_maps(
    maps: &[Vec<SpinorKMinusQ>],
    n_k: usize,
) -> Result<Vec<Vec<usize>>, IsdfExchangeError> {
    maps.iter()
        .enumerate()
        .map(|(q_index, map)| {
            map.iter()
                .enumerate()
                .map(|(k_index, mapped)| {
                    if mapped.k_index != k_index || mapped.kq_index >= n_k {
                        return Err(IsdfExchangeError::KMinusQ { q_index, k_index });
                    }
                    Ok(mapped.kq_index)
                })
                .collect()
        })
        .collect()
}

fn scalar_q_slice_error(error: ScalarQSliceError) -> IsdfExchangeError {
    match error {
        ScalarQSliceError::EmptySlice => IsdfExchangeError::EmptySlice,
        ScalarQSliceError::IncompleteQSlice { actual, expected } => IsdfExchangeError::QCount {
            q_count: actual,
            k_count: expected,
        },
        ScalarQSliceError::IncompatibleInputs | ScalarQSliceError::NonFiniteQSlice => {
            IsdfExchangeError::QContext { index: 0 }
        }
        ScalarQSliceError::CanonicalQMismatch { q_index } => {
            IsdfExchangeError::QContext { index: q_index }
        }
        ScalarQSliceError::KMinusQWrap { q_index, k_index } => {
            IsdfExchangeError::KMinusQ { q_index, k_index }
        }
    }
}

fn spinor_q_slice_error(error: SpinorQSliceError) -> IsdfExchangeError {
    match error {
        SpinorQSliceError::EmptySlice => IsdfExchangeError::EmptySlice,
        SpinorQSliceError::IncompleteQSlice { actual, expected } => IsdfExchangeError::QCount {
            q_count: actual,
            k_count: expected,
        },
        SpinorQSliceError::IncompatibleInputs | SpinorQSliceError::NonFiniteQSlice => {
            IsdfExchangeError::QContext { index: 0 }
        }
        SpinorQSliceError::CanonicalQMismatch { q_index } => {
            IsdfExchangeError::QContext { index: q_index }
        }
        SpinorQSliceError::KMinusQWrap { q_index, k_index } => {
            IsdfExchangeError::KMinusQ { q_index, k_index }
        }
    }
}

fn contract_exchange(
    layout: PairColumnLayout,
    maps: &[Vec<usize>],
    records: &[RectangularExchangeRecord<'_>],
    spec: &IsdfExchangeSpec,
) -> Result<IsdfExchangeResult, IsdfExchangeError> {
    validate_spec(spec, layout.n_k, layout.n_orb)?;
    let contracted = contract_rectangular_exchange(
        valence_layout(layout),
        maps,
        records,
        &spec.k_weights,
        &spec.occupations,
        spec.gamma,
    )?;
    let mut exchange_energy = 0.0;
    for (k, matrix) in contracted.target_matrices.iter().enumerate() {
        for band in 0..layout.n_orb {
            exchange_energy += 0.5
                * spec.k_weights[k]
                * spec.occupations[k][band]
                * matrix.values[band * layout.n_orb + band].re;
        }
    }
    Ok(IsdfExchangeResult {
        exchange_energy: Hartree(exchange_energy),
        band_matrices: contracted.target_matrices,
        maximum_antihermitian_residual: contracted.maximum_antihermitian_residual,
    })
}

pub(crate) fn contract_rectangular_exchange(
    layout: ExchangePairLayout,
    maps: &[Vec<usize>],
    records: &[RectangularExchangeRecord<'_>],
    k_weights: &[f64],
    occupied_occupations: &[Vec<f64>],
    gamma: GammaExchangeTreatment,
) -> Result<RectangularExchangeResult, IsdfExchangeError> {
    let n_k = layout.n_k;
    let n_occupied = layout.n_occupied;
    if records.len() != n_k || maps.len() != n_k {
        return Err(IsdfExchangeError::QCount {
            q_count: records.len(),
            k_count: n_k,
        });
    }
    validate_k_weights(k_weights, n_k)?;
    validate_occupations(occupied_occupations, n_k, n_occupied)?;
    let expected_vertices = layout
        .n_columns()
        .map_err(|_| IsdfExchangeError::QContext { index: 0 })?;
    for (q_index, (record, map)) in records.iter().zip(maps).enumerate() {
        if record.layout != layout || record.vertices.len() != expected_vertices {
            return Err(IsdfExchangeError::QContext { index: q_index });
        }
        if map.len() != n_k {
            return Err(IsdfExchangeError::KMinusQ {
                q_index,
                k_index: map.len(),
            });
        }
        if record.operator.gamma().is_some() && gamma == GammaExchangeTreatment::Reject {
            return Err(IsdfExchangeError::GammaHead { q_index });
        }
    }

    contract_rectangular_values(
        layout,
        maps,
        k_weights,
        occupied_occupations,
        |q_index, left_column, right_column| {
            records[q_index]
                .operator
                .quadratic_form(
                    &records[q_index].vertices[left_column],
                    &records[q_index].vertices[right_column],
                )
                .map_err(IsdfExchangeError::from)
        },
    )
}

fn contract_rectangular_values(
    layout: ExchangePairLayout,
    maps: &[Vec<usize>],
    k_weights: &[f64],
    occupied_occupations: &[Vec<f64>],
    mut integral: impl FnMut(usize, usize, usize) -> Result<Complex64, IsdfExchangeError>,
) -> Result<RectangularExchangeResult, IsdfExchangeError> {
    let n_k = layout.n_k;
    let n_occupied = layout.n_occupied;
    let n_target = layout.n_target;
    let mut band_matrices = Vec::with_capacity(n_k);
    let mut maximum_antihermitian_residual = 0.0_f64;
    for k in 0..n_k {
        let mut values = vec![Complex64::default(); n_target * n_target];
        for right_band in 0..n_target {
            for right_band_prime in 0..n_target {
                let mut value = Complex64::default();
                for (q_index, map) in maps.iter().enumerate() {
                    let kq = map[k];
                    for left_band in 0..n_occupied {
                        let weight = k_weights[kq] * occupied_occupations[kq][left_band];
                        if weight == 0.0 {
                            continue;
                        }
                        let left_column = layout
                            .encode(k, left_band, right_band)
                            .map_err(|_| IsdfExchangeError::QContext { index: 0 })?;
                        let right_column = layout
                            .encode(k, left_band, right_band_prime)
                            .map_err(|_| IsdfExchangeError::QContext { index: 0 })?;
                        value -= weight * integral(q_index, left_column, right_column)?;
                    }
                }
                values[right_band * n_target + right_band_prime] = value;
            }
        }
        for row in 0..n_target {
            for column in 0..n_target {
                let residual =
                    (values[row * n_target + column] - values[column * n_target + row].conj()).norm();
                maximum_antihermitian_residual = maximum_antihermitian_residual.max(residual);
            }
        }
        band_matrices.push(IsdfExchangeBandMatrix {
            k_index: k,
            n_bands: n_target,
            values,
        });
    }
    if maximum_antihermitian_residual > HERMITICITY_TOLERANCE {
        return Err(IsdfExchangeError::NonHermitian {
            residual: maximum_antihermitian_residual,
            tolerance: HERMITICITY_TOLERANCE,
        });
    }

    Ok(RectangularExchangeResult {
        target_matrices: band_matrices,
        maximum_antihermitian_residual,
    })
}

pub(crate) fn target_trace(
    layout: ExchangePairLayout,
    matrices: &[IsdfExchangeBandMatrix],
    k_weights: &[f64],
    target_occupations: &[Vec<f64>],
) -> f64 {
    matrices
        .iter()
        .enumerate()
        .map(|(k, matrix)| {
            (0..layout.n_target)
                .map(|target| {
                    k_weights[k]
                        * target_occupations[k][target]
                        * matrix.values[target * layout.n_target + target].re
                })
                .sum::<f64>()
        })
        .sum()
}

const fn valence_layout(layout: PairColumnLayout) -> ExchangePairLayout {
    ExchangePairLayout::new(
        ExchangeSpace::Valence,
        ExchangeSpace::Valence,
        layout.n_k,
        layout.n_orb,
        layout.n_orb,
    )
}

fn validate_spec(
    spec: &IsdfExchangeSpec,
    n_k: usize,
    n_bands: usize,
) -> Result<(), IsdfExchangeError> {
    validate_k_weights(&spec.k_weights, n_k)?;
    validate_occupations(&spec.occupations, n_k, n_bands)
}

pub(crate) fn validate_k_weights(
    k_weights: &[f64],
    n_k: usize,
) -> Result<(), IsdfExchangeError> {
    if k_weights.len() != n_k {
        return Err(IsdfExchangeError::KWeightCount {
            actual: k_weights.len(),
            expected: n_k,
        });
    }
    let mut sum = 0.0;
    for (index, &value) in k_weights.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(IsdfExchangeError::KWeight { index, value });
        }
        sum += value;
    }
    if (sum - 1.0).abs() > WEIGHT_TOLERANCE {
        return Err(IsdfExchangeError::KWeightSum { sum });
    }
    Ok(())
}

pub(crate) fn validate_occupations(
    occupations: &[Vec<f64>],
    n_k: usize,
    n_bands: usize,
) -> Result<(), IsdfExchangeError> {
    if occupations.len() != n_k {
        return Err(IsdfExchangeError::OccupationKCount {
            actual: occupations.len(),
            expected: n_k,
        });
    }
    for (k_index, row) in occupations.iter().enumerate() {
        if row.len() != n_bands {
            return Err(IsdfExchangeError::OccupationBandCount {
                k_index,
                actual: row.len(),
                expected: n_bands,
            });
        }
        for (band, &value) in row.iter().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(IsdfExchangeError::Occupation {
                    k_index,
                    band,
                    value,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_kernel_applies_each_density_factor_once() {
        let layout = ExchangePairLayout::new(
            ExchangeSpace::Core,
            ExchangeSpace::Valence,
            2,
            2,
            1,
        );
        let maps = vec![vec![0, 1], vec![1, 0]];
        let weights = vec![0.25, 0.75];
        let occupied = vec![vec![0.2, 0.4], vec![0.5, 0.8]];
        let contracted = contract_rectangular_values(
            layout,
            &maps,
            &weights,
            &occupied,
            |q, column, other| {
                assert_eq!(column, other);
                let (_, orbital, _) = layout.decode(column).unwrap();
                Ok(Complex64::new((q + 1) as f64 * (orbital + 2) as f64, 0.0))
            },
        )
        .unwrap();
        for k in 0..2 {
            let mut expected = 0.0;
            for q in 0..2 {
                let kq = maps[q][k];
                for orbital in 0..2 {
                    expected -= weights[kq]
                        * occupied[kq][orbital]
                        * (q + 1) as f64
                        * (orbital + 2) as f64;
                }
            }
            assert_eq!(contracted.target_matrices[k].values()[0].re, expected);
        }
        let target = vec![vec![0.3], vec![0.6]];
        let trace = target_trace(layout, &contracted.target_matrices, &weights, &target);
        let direct = (0..2)
            .map(|k| weights[k] * target[k][0] * contracted.target_matrices[k].values()[0].re)
            .sum::<f64>();
        assert_eq!(trace, direct);
    }

    #[test]
    fn square_vv_energy_is_half_the_rectangular_target_trace() {
        let layout = ExchangePairLayout::new(
            ExchangeSpace::Valence,
            ExchangeSpace::Valence,
            1,
            2,
            2,
        );
        let weights = vec![1.0];
        let occupations = vec![vec![1.0, 0.25]];
        let contracted = contract_rectangular_values(
            layout,
            &[vec![0]],
            &weights,
            &occupations,
            |_, left, right| {
                let (_, occupied_left, target_left) = layout.decode(left).unwrap();
                let (_, occupied_right, target_right) = layout.decode(right).unwrap();
                assert_eq!(occupied_left, occupied_right);
                Ok(if target_left == target_right {
                    Complex64::new((occupied_left + target_left + 1) as f64, 0.0)
                } else {
                    Complex64::default()
                })
            },
        )
        .unwrap();
        let trace = target_trace(layout, &contracted.target_matrices, &weights, &occupations);
        let adapter_energy = 0.5
            * occupations[0]
                .iter()
                .enumerate()
                .map(|(band, occupation)| {
                    occupation
                        * contracted.target_matrices[0].values()[band * layout.n_target + band].re
                })
                .sum::<f64>();
        assert_eq!(adapter_energy, 0.5 * trace);
    }

    #[test]
    fn cv_and_vc_traces_are_contracted_independently() {
        let weights = vec![1.0];
        let core = vec![vec![0.5]];
        let valence = vec![vec![0.25, 0.75]];
        let cv_layout = ExchangePairLayout::new(
            ExchangeSpace::Core,
            ExchangeSpace::Valence,
            1,
            1,
            2,
        );
        let vc_layout = ExchangePairLayout::new(
            ExchangeSpace::Valence,
            ExchangeSpace::Core,
            1,
            2,
            1,
        );
        let cv = contract_rectangular_values(
            cv_layout,
            &[vec![0]],
            &weights,
            &core,
            |_, left, right| {
                let (_, _, left_target) = cv_layout.decode(left).unwrap();
                let (_, _, right_target) = cv_layout.decode(right).unwrap();
                Ok((left_target == right_target)
                    .then(|| Complex64::new((left_target + 1) as f64, 0.0))
                    .unwrap_or_default())
            },
        )
        .unwrap();
        let vc = contract_rectangular_values(
            vc_layout,
            &[vec![0]],
            &weights,
            &valence,
            |_, _, _| Ok(Complex64::new(3.0, 0.0)),
        )
        .unwrap();
        let t_cv = target_trace(cv_layout, &cv.target_matrices, &weights, &valence);
        let t_vc = target_trace(vc_layout, &vc.target_matrices, &weights, &core);
        assert_eq!(t_cv, -0.875);
        assert_eq!(t_vc, -1.5);
        assert_ne!(t_cv, t_vc);
    }
}
