//! Spinor sampled-$\zeta$ Coulomb bridge from M-L5d THC and optional M-L5c pairs.

use crate::scalar_coulomb::{
    CoulombBridgeError, bind_interpolation_request, quadratic_discrepancy, require_thc_q_record,
    sampled_from_thc_record, vertex_action_norm,
};
use crate::spinor_mpb::SpinorMpbResult;
use crate::spinor_product::{SpinorProductInput, SpinorQSliceError, require_spinor_q_slice};
use crate::spinor_thc::SpinorThcResult;
use crate::thc_grid::{ThcQRecord, records_match_parent_grid};
use muffintin_auxiliary_ir::{OrbitalPair, PairColumnLayout, PairVertex, TransferQ};
use muffintin_core::ExponentialMesh;
use muffintin_coulomb::{
    AuxiliaryKind, CoulombError, CoulombOperator, CoulombRequest, InterpolationProjection,
    SampledAuxiliaryFunctions, assemble_coulomb, assemble_sampled_coulomb,
};
use muffintin_thc::{L2Engine, SelectorStrategy};
use num_complex::Complex64;
use thiserror::Error;

/// Absolute/relative discrepancy floor for the quadratic $c^\dagger V c$ diagnostic.
pub const SPINOR_COULOMB_EXACTNESS_FLOOR: f64 = 1.0e-12;

/// Explicit Coulomb request and interpolation projection for one M-L5d run.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorCoulombSpec {
    /// Existing Weinert assembly request (direct cell, `LEXP`, reciprocal).
    pub request: CoulombRequest,
    /// Existing interpolation-point projection ($|q+G|$ cutoff and $L_{\max}$).
    pub projection: InterpolationProjection,
}

/// One matched M-L5c vertex to compare against the THC pair at the same $q$.
#[derive(Clone, Copy, Debug)]
pub struct SpinorCoulombPairMatch<'a> {
    /// Production $q$-index in the ordered slice / [`SpinorThcResult::records`].
    pub q_index: usize,
    /// Matching M-L5c mixed-product result at that transfer.
    pub mpb: &'a SpinorMpbResult,
    /// Index into [`SpinorMpbResult::vertices`].
    pub mpb_vertex: usize,
}

/// Sampled-$\zeta$ Coulomb operator for one $q$.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorCoulombQRecord {
    pub q_index: usize,
    pub q: TransferQ,
    pub layout: PairColumnLayout,
    /// Interpolation-point auxiliary copied from the matching M-L5d record.
    pub auxiliary: muffintin_auxiliary_ir::CompiledAuxiliaryBasis,
    /// Semantic pair vertices in [`PairColumnLayout`] order, copied from M-L5d.
    pub vertices: Vec<PairVertex>,
    /// Parent-grid sampled $\zeta$ used to assemble this operator.
    pub sampled: SampledAuxiliaryFunctions,
    /// Sampled-$\zeta$ $V^q$ from [`assemble_sampled_coulomb`].
    pub operator: CoulombOperator,
}

/// Absolute and relative discrepancy of two spinor Coulomb observables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinorCoulombDiscrepancy {
    pub absolute: f64,
    pub relative: f64,
}

/// MPB versus THC Coulomb observables for one spinor pair.
///
/// The representation-neutral comparison is the quadratic form $c^\dagger V c$
/// in each auxiliary representation. Per-side action norms $\|Vc\|$ are debug
/// diagnostics in their own bases and are not compared across representations.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorCoulombPairDiagnostic {
    pub q_index: usize,
    pub pair: OrbitalPair,
    pub column: usize,
    pub mpb_quadratic: Complex64,
    pub thc_quadratic: Complex64,
    /// Mixed-product debug action norm; not a cross-representation observable.
    pub mpb_action_norm: f64,
    /// Interpolation-point debug action norm; not a cross-representation observable.
    pub thc_action_norm: f64,
    pub quadratic_discrepancy: SpinorCoulombDiscrepancy,
}

/// Spinor M-L5d sampled-$\zeta$ Coulomb result.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorCoulombResult {
    pub(crate) records: Vec<SpinorCoulombQRecord>,
    pub diagnostics: Vec<SpinorCoulombPairDiagnostic>,
    /// Effective request and projection used by [`build_spinor_coulomb`].
    ///
    /// Not caller-forgeable through a public struct literal.
    pub(crate) context: SpinorCoulombSpec,
}

impl SpinorCoulombResult {
    /// Sealed sampled-$\zeta$ $V^q$ records in production $q$ order.
    ///
    /// Populated only by [`build_spinor_coulomb`]. The slice is read-only;
    /// replacement operators cannot be installed through this accessor.
    pub fn records(&self) -> &[SpinorCoulombQRecord] {
        &self.records
    }
}

/// Spinor Coulomb stage-boundary error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SpinorCoulombError {
    #[error(transparent)]
    Coulomb(#[from] CoulombError),
    #[error(transparent)]
    Product(#[from] muffintin_auxiliary_ir::ProductError),
    #[error("spinor Coulomb q-slice must be nonempty")]
    EmptySlice,
    #[error("spinor Coulomb q-slice has {actual} bundles, expected {expected} k-mesh transfers")]
    IncompleteQSlice { actual: usize, expected: usize },
    #[error("spinor Coulomb inputs do not share one frozen orbital window, layout, and partition")]
    IncompatibleInputs,
    #[error("spinor Coulomb q-slice contains a non-finite k, q, or wrap component")]
    NonFiniteQSlice,
    #[error(
        "spinor Coulomb canonical q at index {q_index} is not the complete-slice k-mesh transfer"
    )]
    CanonicalQMismatch { q_index: usize },
    #[error("spinor Coulomb k-minus-q wrap at q-index {q_index} k-index {k_index} is inconsistent")]
    KMinusQWrap { q_index: usize, k_index: usize },
    #[error("spinor Coulomb THC result has {actual} q-records, expected {expected}")]
    RecordCount { actual: usize, expected: usize },
    #[error("spinor Coulomb THC result is not bound to the frozen product partition")]
    Partition,
    #[error("spinor Coulomb THC record {index} does not match the frozen q-slice context")]
    ThcRecord { index: usize },
    #[error("spinor Coulomb THC record {index} is not bound to the parent grid used to fit zeta")]
    GridIdentity { index: usize },
    #[error(
        "spinor Coulomb THC record {index} vertex {column} layout, Bloch order, or provenance does not match the compiled auxiliary"
    )]
    VertexIdentity { index: usize, column: usize },
    #[error("spinor Coulomb request reciprocal lattice does not match the frozen product source")]
    ReciprocalMismatch,
    #[error(
        "spinor Coulomb matched MPB at q-index {q_index} was not built from the frozen product input at that transfer"
    )]
    FrozenInputMismatch { q_index: usize },
    #[error(
        "spinor Coulomb matched MPB reciprocal lattice at q-index {q_index} does not match the frozen product input"
    )]
    MpbReciprocalMismatch { q_index: usize },
    #[error(
        "spinor Coulomb matched MPB pair-column layout at q-index {q_index} does not match the frozen product input"
    )]
    MpbPairLayoutMismatch { q_index: usize },
    #[error("spinor Coulomb interpolation projection does not match the Coulomb request")]
    InterpolationProjection,
    #[error(
        "spinor Coulomb spec does not match the effective request/projection used to construct the result"
    )]
    SpecMismatch,
    #[error(
        "spinor Coulomb record {index} does not match inputs[q] or the accepted THC q/layout/auxiliary/vertices"
    )]
    CoulombRecord { index: usize },
    #[error("spinor Coulomb cannot export THC selection strategy")]
    UnsupportedStrategy,
    #[error("spinor Coulomb cannot export THC engine {0:?}")]
    UnsupportedEngine(L2Engine),
    #[error("spinor Coulomb comparison q-index {0} is outside the THC q-slice")]
    ComparisonQIndex(usize),
    #[error(
        "spinor Coulomb comparison vertex {index} is outside M-L5c vertices at q-index {q_index}"
    )]
    ComparisonVertex { q_index: usize, index: usize },
    #[error(
        "spinor Coulomb comparison at q-index {q_index} does not match the THC pair, layout, or transfer q"
    )]
    ComparisonContext { q_index: usize },
}

impl From<CoulombBridgeError> for SpinorCoulombError {
    fn from(error: CoulombBridgeError) -> Self {
        match error {
            CoulombBridgeError::Coulomb(error) => Self::Coulomb(error),
            CoulombBridgeError::InterpolationProjection => Self::InterpolationProjection,
            CoulombBridgeError::GridIdentity { index } => Self::GridIdentity { index },
            CoulombBridgeError::ThcRecord { index } => Self::ThcRecord { index },
            CoulombBridgeError::VertexIdentity { index, column } => {
                Self::VertexIdentity { index, column }
            }
        }
    }
}

/// Assemble sampled-$\zeta$ $V^q$ on the full parent grid for a complete $q$ slice.
///
/// [`CoulombRequest`] must carry the frozen [`SpinorProductInput::reciprocal`].
/// Each THC record's $\zeta$ is collocated on the full [`crate::ThcParentGrid`]
/// in original order, including zero-weight rows. Interpolation *nodes* are not
/// the $\zeta$ grid. Gamma retains the finite body plus
/// [`muffintin_coulomb::GammaHead`] metadata; the singular head is not inserted.
/// Matched M-L5c/M-L5d pairs compare $c^\dagger V c$ only.
/// Each match must originate from `inputs[q_index]`: the sealed frozen-input
/// identity is required, then the public reciprocal lattice and pair-column
/// layout, before mixed-product Coulomb assembly.
pub fn build_spinor_coulomb(
    inputs: &[SpinorProductInput],
    thc: &SpinorThcResult,
    spec: &SpinorCoulombSpec,
    comparisons: &[SpinorCoulombPairMatch<'_>],
) -> Result<SpinorCoulombResult, SpinorCoulombError> {
    let first = require_spinor_q_slice(inputs)?;
    if thc.records.len() != inputs.len() {
        return Err(SpinorCoulombError::RecordCount {
            actual: thc.records.len(),
            expected: inputs.len(),
        });
    }
    if thc.grid.partition() != &first.source.partition {
        return Err(SpinorCoulombError::Partition);
    }
    if spec.request.reciprocal() != &first.reciprocal {
        return Err(SpinorCoulombError::ReciprocalMismatch);
    }
    for (index, (input, record)) in inputs.iter().zip(&thc.records).enumerate() {
        require_thc_q_record(input.source.q, input.pair_columns, &thc.grid, record, index)?;
    }
    for comparison in comparisons {
        require_matched_mpb_origin(inputs, comparison)?;
    }
    let request = bind_interpolation_request(&spec.request, spec.projection)?;
    let site_meshes = first
        .source
        .radials
        .iter()
        .map(|radials| radials.mesh.clone())
        .collect::<Vec<ExponentialMesh>>();
    let mut records = Vec::with_capacity(thc.records.len());
    for record in &thc.records {
        let sampled = sampled_from_thc_record(record, &thc.grid, site_meshes.clone())?;
        let operator = assemble_sampled_coulomb(&record.auxiliary, &request, &sampled)?;
        records.push(SpinorCoulombQRecord {
            q_index: record.q_index,
            q: record.q,
            layout: record.layout,
            auxiliary: record.auxiliary.clone(),
            vertices: record.vertices.clone(),
            sampled,
            operator,
        });
    }
    let mut diagnostics = Vec::with_capacity(comparisons.len());
    for comparison in comparisons {
        diagnostics.push(pair_diagnostic(&records, comparison, &request)?);
    }
    Ok(SpinorCoulombResult {
        records,
        diagnostics,
        context: spec.clone(),
    })
}

impl From<SpinorQSliceError> for SpinorCoulombError {
    fn from(error: SpinorQSliceError) -> Self {
        match error {
            SpinorQSliceError::EmptySlice => Self::EmptySlice,
            SpinorQSliceError::IncompleteQSlice { actual, expected } => {
                Self::IncompleteQSlice { actual, expected }
            }
            SpinorQSliceError::IncompatibleInputs => Self::IncompatibleInputs,
            SpinorQSliceError::NonFiniteQSlice => Self::NonFiniteQSlice,
            SpinorQSliceError::CanonicalQMismatch { q_index } => {
                Self::CanonicalQMismatch { q_index }
            }
            SpinorQSliceError::KMinusQWrap { q_index, k_index } => {
                Self::KMinusQWrap { q_index, k_index }
            }
        }
    }
}

fn require_matched_mpb_origin(
    inputs: &[SpinorProductInput],
    comparison: &SpinorCoulombPairMatch<'_>,
) -> Result<(), SpinorCoulombError> {
    let q_index = comparison.q_index;
    let input = inputs
        .get(q_index)
        .ok_or(SpinorCoulombError::ComparisonQIndex(q_index))?;
    if !comparison.mpb.frozen_input_identity().matches(input) {
        return Err(SpinorCoulombError::FrozenInputMismatch { q_index });
    }
    if comparison.mpb.reciprocal != input.reciprocal {
        return Err(SpinorCoulombError::MpbReciprocalMismatch { q_index });
    }
    if comparison.mpb.pair_columns != input.pair_columns {
        return Err(SpinorCoulombError::MpbPairLayoutMismatch { q_index });
    }
    Ok(())
}

fn pair_diagnostic(
    records: &[SpinorCoulombQRecord],
    comparison: &SpinorCoulombPairMatch<'_>,
    request: &CoulombRequest,
) -> Result<SpinorCoulombPairDiagnostic, SpinorCoulombError> {
    let q_index = comparison.q_index;
    let record = records
        .get(q_index)
        .ok_or(SpinorCoulombError::ComparisonQIndex(q_index))?;
    let mpb_vertex = comparison.mpb.vertices.get(comparison.mpb_vertex).ok_or(
        SpinorCoulombError::ComparisonVertex {
            q_index,
            index: comparison.mpb_vertex,
        },
    )?;
    let n_col = record.layout.n_columns()?;
    let pair = match mpb_vertex.vertex.pair() {
        OrbitalPair::Bloch {
            k_index,
            left,
            right,
        } if mpb_vertex.k == k_index
            && mpb_vertex.left_band == left
            && mpb_vertex.right_band == right
            && mpb_vertex.column < n_col
            && record.layout.decode(mpb_vertex.column) == (k_index, left, right)
            && comparison.mpb.auxiliary.q == record.q
            && comparison.mpb.auxiliary.partition == record.auxiliary.partition
            && comparison.mpb.pair_columns == record.layout =>
        {
            OrbitalPair::Bloch {
                k_index,
                left,
                right,
            }
        }
        _ => return Err(SpinorCoulombError::ComparisonContext { q_index }),
    };
    let thc_vertex = record
        .vertices
        .get(mpb_vertex.column)
        .ok_or(SpinorCoulombError::ComparisonContext { q_index })?;
    if thc_vertex.pair() != pair {
        return Err(SpinorCoulombError::ComparisonContext { q_index });
    }
    let mpb_operator = assemble_coulomb(&comparison.mpb.auxiliary, request)?;
    let mpb_quadratic = mpb_operator.quadratic_form(&mpb_vertex.vertex, &mpb_vertex.vertex)?;
    let thc_quadratic = record.operator.quadratic_form(thc_vertex, thc_vertex)?;
    let mpb_action_norm = vertex_action_norm(&mpb_operator, &mpb_vertex.vertex)?;
    let thc_action_norm = vertex_action_norm(&record.operator, thc_vertex)?;
    let (absolute, relative) =
        quadratic_discrepancy(mpb_quadratic, thc_quadratic, SPINOR_COULOMB_EXACTNESS_FLOOR);
    Ok(SpinorCoulombPairDiagnostic {
        q_index,
        pair,
        column: mpb_vertex.column,
        mpb_quadratic,
        thc_quadratic,
        mpb_action_norm,
        thc_action_norm,
        quadratic_discrepancy: SpinorCoulombDiscrepancy { absolute, relative },
    })
}

/// Exact Coulomb export context: sealed spec, ordered $q$ records, THC
/// q/layout/auxiliary/vertices, and serializable strategy/engine/projection.
pub(crate) fn require_spinor_coulomb_export_context(
    inputs: &[SpinorProductInput],
    thc: &SpinorThcResult,
    coulomb: &SpinorCoulombResult,
    spec: &SpinorCoulombSpec,
) -> Result<(), SpinorCoulombError> {
    let first = require_spinor_q_slice(inputs)?;
    if spec != &coulomb.context {
        return Err(SpinorCoulombError::SpecMismatch);
    }
    bind_interpolation_request(&spec.request, spec.projection)?;
    if spec.request.reciprocal() != &first.reciprocal {
        return Err(SpinorCoulombError::ReciprocalMismatch);
    }
    if thc.records.len() != inputs.len() {
        return Err(SpinorCoulombError::RecordCount {
            actual: thc.records.len(),
            expected: inputs.len(),
        });
    }
    if coulomb.records.len() != inputs.len() {
        return Err(SpinorCoulombError::RecordCount {
            actual: coulomb.records.len(),
            expected: inputs.len(),
        });
    }
    if thc.grid.partition() != &first.source.partition {
        return Err(SpinorCoulombError::Partition);
    }
    if !records_match_parent_grid(&thc.grid, &thc.records) {
        return Err(SpinorCoulombError::GridIdentity { index: 0 });
    }
    if thc.selection.provenance.strategy != SelectorStrategy::AllQL2 {
        return Err(SpinorCoulombError::UnsupportedStrategy);
    }
    match thc.selection.provenance.engine {
        L2Engine::FullColumnPivotedQr | L2Engine::FullPivotedCholesky => {}
        other => return Err(SpinorCoulombError::UnsupportedEngine(other)),
    }
    for (index, ((input, thc_record), coulomb_record)) in inputs
        .iter()
        .zip(&thc.records)
        .zip(&coulomb.records)
        .enumerate()
    {
        require_thc_q_record(
            input.source.q,
            input.pair_columns,
            &thc.grid,
            thc_record,
            index,
        )?;
        require_coulomb_q_record(input, thc_record, coulomb_record, spec, index)?;
    }
    Ok(())
}

fn require_coulomb_q_record(
    input: &SpinorProductInput,
    thc: &ThcQRecord,
    record: &SpinorCoulombQRecord,
    spec: &SpinorCoulombSpec,
    index: usize,
) -> Result<(), SpinorCoulombError> {
    let aux_dimension = record.auxiliary.dimension();
    let ok = record.q_index == index
        && thc.q_index == index
        && record.q == input.source.q
        && record.q == thc.q
        && record.layout == input.pair_columns
        && record.layout == thc.layout
        && record.auxiliary == thc.auxiliary
        && record.vertices == thc.vertices
        && record.operator.dimension() == aux_dimension
        && record.operator.dimension() == thc.fit.n_mu
        && record.operator.q() == record.q
        && record.operator.cell() == spec.request.cell()
        && record.operator.reciprocal() == spec.request.reciprocal()
        && record.operator.layout() == &record.auxiliary.layout()
        && record.operator.kind() == AuxiliaryKind::InterpolationPoints;
    if ok {
        Ok(())
    } else {
        Err(SpinorCoulombError::CoulombRecord { index })
    }
}
