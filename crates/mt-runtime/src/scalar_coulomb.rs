//! Scalar sampled-$\zeta$ Coulomb bridge from M-L3 THC and optional M-L2 pairs.

use crate::scalar_mpb::ScalarMpbResult;
use crate::scalar_product::ScalarProductInput;
use crate::scalar_thc::ScalarThcResult;
use crate::thc_grid::{ThcParentGrid, ThcQRecord, ThcRegion, is_gamma_fractional};
use muffintin_auxiliary_ir::{OrbitalPair, PairColumnLayout, PairVertex, ProductSource, TransferQ};
use muffintin_core::{ExponentialMesh, VolumeBohr3};
use muffintin_coulomb::{
    CoulombError, CoulombOperator, CoulombRequest, InterpolationProjection,
    SampledAuxiliaryFunctions, SampledPointSupport, assemble_coulomb, assemble_sampled_coulomb,
};
use num_complex::Complex64;
use thiserror::Error;

/// Absolute/relative discrepancy floor for the quadratic $c^\dagger V c$ diagnostic.
pub const SCALAR_COULOMB_EXACTNESS_FLOOR: f64 = 1.0e-12;

/// Explicit Coulomb request and interpolation projection for one M-L4 run.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarCoulombSpec {
    /// Existing Weinert assembly request (direct cell, `LEXP`, reciprocal).
    pub request: CoulombRequest,
    /// Existing interpolation-point projection ($|q+G|$ cutoff and $L_{\max}$).
    pub projection: InterpolationProjection,
}

/// One matched M-L2 vertex to compare against the THC pair at the same $q$.
#[derive(Clone, Copy, Debug)]
pub struct ScalarCoulombPairMatch<'a> {
    /// Production $q$-index in the ordered slice / [`ScalarThcResult::records`].
    pub q_index: usize,
    /// Matching M-L2 mixed-product result at that transfer.
    pub mpb: &'a ScalarMpbResult,
    /// Index into [`ScalarMpbResult::vertices`].
    pub mpb_vertex: usize,
}

/// Sampled-$\zeta$ Coulomb operator for one $q$.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarCoulombQRecord {
    pub q_index: usize,
    pub q: TransferQ,
    pub spin: u8,
    pub layout: PairColumnLayout,
    /// Interpolation-point auxiliary copied from the matching M-L3 record.
    pub auxiliary: muffintin_auxiliary_ir::CompiledAuxiliaryBasis,
    /// Semantic pair vertices in [`PairColumnLayout`] order, copied from M-L3.
    pub vertices: Vec<PairVertex>,
    /// Parent-grid sampled $\zeta$ used to assemble this operator.
    pub sampled: SampledAuxiliaryFunctions,
    /// Sampled-$\zeta$ $V^q$ from [`assemble_sampled_coulomb`].
    pub operator: CoulombOperator,
}

/// Absolute and relative discrepancy of two scalar Coulomb observables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarCoulombDiscrepancy {
    pub absolute: f64,
    pub relative: f64,
}

/// MPB versus THC Coulomb observables for one pair.
///
/// The representation-neutral comparison is the quadratic form $c^\dagger V c$
/// in each auxiliary representation. Per-side action norms $\|Vc\|$ are debug
/// diagnostics in their own bases and are not compared across representations.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarCoulombPairDiagnostic {
    pub q_index: usize,
    pub spin: u8,
    pub pair: OrbitalPair,
    pub column: usize,
    pub mpb_quadratic: Complex64,
    pub thc_quadratic: Complex64,
    /// Mixed-product debug action norm; not a cross-representation observable.
    pub mpb_action_norm: f64,
    /// Interpolation-point debug action norm; not a cross-representation observable.
    pub thc_action_norm: f64,
    pub quadratic_discrepancy: ScalarCoulombDiscrepancy,
}

/// Scalar M-L4 sampled-$\zeta$ Coulomb result.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarCoulombResult {
    pub spin: u8,
    pub records: Vec<ScalarCoulombQRecord>,
    pub diagnostics: Vec<ScalarCoulombPairDiagnostic>,
}

/// Scalar Coulomb stage-boundary error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ScalarCoulombError {
    #[error(transparent)]
    Coulomb(#[from] CoulombError),
    #[error(transparent)]
    Product(#[from] muffintin_auxiliary_ir::ProductError),
    #[error("scalar Coulomb q-slice must be nonempty")]
    EmptySlice,
    #[error("scalar Coulomb q-slice has {actual} bundles, expected {expected} k-mesh transfers")]
    IncompleteQSlice { actual: usize, expected: usize },
    #[error("scalar Coulomb inputs do not share one frozen orbital window, layout, and partition")]
    IncompatibleInputs,
    #[error("scalar Coulomb THC result has {actual} q-records, expected {expected}")]
    RecordCount { actual: usize, expected: usize },
    #[error("scalar Coulomb THC spin {0} is not present in the frozen orbitals")]
    InvalidSpin(u8),
    #[error("scalar Coulomb THC result is not bound to the frozen product partition")]
    Partition,
    #[error("scalar Coulomb THC record {index} does not match the frozen q-slice context")]
    ThcRecord { index: usize },
    #[error("scalar Coulomb THC record {index} is not bound to the parent grid used to fit zeta")]
    GridIdentity { index: usize },
    #[error(
        "scalar Coulomb THC record {index} vertex {column} layout, Bloch order, or provenance does not match the compiled auxiliary"
    )]
    VertexIdentity { index: usize, column: usize },
    #[error("scalar Coulomb request reciprocal lattice does not match the frozen product source")]
    ReciprocalMismatch,
    #[error("scalar Coulomb interpolation projection does not match the Coulomb request")]
    InterpolationProjection,
    #[error("scalar Coulomb comparison q-index {0} is outside the THC q-slice")]
    ComparisonQIndex(usize),
    #[error(
        "scalar Coulomb comparison vertex {index} is outside M-L2 vertices at q-index {q_index}"
    )]
    ComparisonVertex { q_index: usize, index: usize },
    #[error(
        "scalar Coulomb comparison at q-index {q_index} does not match the THC pair, layout, spin, or transfer q"
    )]
    ComparisonContext { q_index: usize },
}

impl From<CoulombBridgeError> for ScalarCoulombError {
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

/// Assemble sampled-$\zeta$ $V^q$ on the M-L3 parent grid for a complete $q$ slice.
///
/// [`CoulombRequest`] must carry the frozen [`ScalarProductInput::reciprocal`].
/// Each THC record's $\zeta$ is collocated on the full parent grid in original
/// order, including zero-weight rows. The parent-grid construction identity
/// must match every $q$ record. Interpolation *nodes* are not the $\zeta$
/// grid. Gauge is unchanged: no extra pair rephasing and no second global
/// Umklapp insertion. Gamma retains the finite body plus [`muffintin_coulomb::GammaHead`]
/// metadata; the singular head is not inserted.
pub fn build_scalar_coulomb(
    inputs: &[ScalarProductInput],
    thc: &ScalarThcResult,
    spec: &ScalarCoulombSpec,
    comparisons: &[ScalarCoulombPairMatch<'_>],
) -> Result<ScalarCoulombResult, ScalarCoulombError> {
    let first = inputs.first().ok_or(ScalarCoulombError::EmptySlice)?;
    require_compatible_slice(inputs)?;
    if thc.records.len() != inputs.len() {
        return Err(ScalarCoulombError::RecordCount {
            actual: thc.records.len(),
            expected: inputs.len(),
        });
    }
    if !first
        .orbitals
        .channels
        .iter()
        .any(|channel| channel.spin == thc.spin)
    {
        return Err(ScalarCoulombError::InvalidSpin(thc.spin));
    }
    if thc.grid.partition() != &first.source.partition {
        return Err(ScalarCoulombError::Partition);
    }
    if spec.request.reciprocal() != &first.reciprocal {
        return Err(ScalarCoulombError::ReciprocalMismatch);
    }
    for (index, (input, record)) in inputs.iter().zip(&thc.records).enumerate() {
        require_thc_q_record(input.source.q, input.pair_columns, &thc.grid, record, index)?;
    }
    let request = bind_interpolation_request(&spec.request, spec.projection)?;
    let site_meshes = site_meshes(&first.source);
    let mut records = Vec::with_capacity(thc.records.len());
    for record in &thc.records {
        let sampled = sampled_from_thc_record(record, &thc.grid, site_meshes.clone())?;
        let operator = assemble_sampled_coulomb(&record.auxiliary, &request, &sampled)?;
        records.push(ScalarCoulombQRecord {
            q_index: record.q_index,
            q: record.q,
            spin: thc.spin,
            layout: record.layout,
            auxiliary: record.auxiliary.clone(),
            vertices: record.vertices.clone(),
            sampled,
            operator,
        });
    }
    let mut diagnostics = Vec::with_capacity(comparisons.len());
    for comparison in comparisons {
        diagnostics.push(pair_diagnostic(thc, &records, comparison, &request)?);
    }
    Ok(ScalarCoulombResult {
        spin: thc.spin,
        records,
        diagnostics,
    })
}

fn site_meshes(source: &ProductSource) -> Vec<ExponentialMesh> {
    source
        .radials
        .iter()
        .map(|radials| radials.mesh.clone())
        .collect()
}

fn require_compatible_slice(inputs: &[ScalarProductInput]) -> Result<(), ScalarCoulombError> {
    let first = &inputs[0];
    let n_k = first.orbitals.k_fractional.len();
    if inputs.len() != n_k {
        return Err(ScalarCoulombError::IncompleteQSlice {
            actual: inputs.len(),
            expected: n_k,
        });
    }
    for (iq, input) in inputs.iter().enumerate() {
        if input.orbitals != first.orbitals
            || input.pair_columns != first.pair_columns
            || input.source.partition != first.source.partition
            || input.source.radials != first.source.radials
            || input.reciprocal != first.reciprocal
            || input.k_minus_q.len() != n_k
        {
            return Err(ScalarCoulombError::IncompatibleInputs);
        }
        let mapped = input
            .k_minus_q
            .iter()
            .find(|mapped| mapped.k_index == iq)
            .ok_or(ScalarCoulombError::IncompatibleInputs)?;
        if !is_gamma_fractional(first.orbitals.k_fractional[mapped.kq_index]) {
            return Err(ScalarCoulombError::IncompleteQSlice {
                actual: iq,
                expected: n_k,
            });
        }
    }
    Ok(())
}

fn pair_diagnostic(
    thc: &ScalarThcResult,
    records: &[ScalarCoulombQRecord],
    comparison: &ScalarCoulombPairMatch<'_>,
    request: &CoulombRequest,
) -> Result<ScalarCoulombPairDiagnostic, ScalarCoulombError> {
    let q_index = comparison.q_index;
    let record = records
        .get(q_index)
        .ok_or(ScalarCoulombError::ComparisonQIndex(q_index))?;
    let mpb_vertex = comparison.mpb.vertices.get(comparison.mpb_vertex).ok_or(
        ScalarCoulombError::ComparisonVertex {
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
        } if mpb_vertex.spin == thc.spin
            && mpb_vertex.k == k_index
            && mpb_vertex.left_band == left
            && mpb_vertex.right_band == right
            && mpb_vertex.column < n_col
            && record.layout.decode(mpb_vertex.column) == (k_index, left, right)
            && comparison.mpb.auxiliary.q == record.q
            && comparison.mpb.auxiliary.partition == record.auxiliary.partition =>
        {
            OrbitalPair::Bloch {
                k_index,
                left,
                right,
            }
        }
        _ => return Err(ScalarCoulombError::ComparisonContext { q_index }),
    };
    let thc_vertex = record
        .vertices
        .get(mpb_vertex.column)
        .ok_or(ScalarCoulombError::ComparisonContext { q_index })?;
    if thc_vertex.pair() != pair {
        return Err(ScalarCoulombError::ComparisonContext { q_index });
    }
    let mpb_operator = assemble_coulomb(&comparison.mpb.auxiliary, request)?;
    let mpb_quadratic = mpb_operator.quadratic_form(&mpb_vertex.vertex, &mpb_vertex.vertex)?;
    let thc_quadratic = record.operator.quadratic_form(thc_vertex, thc_vertex)?;
    let mpb_action_norm = vertex_action_norm(&mpb_operator, &mpb_vertex.vertex)?;
    let thc_action_norm = vertex_action_norm(&record.operator, thc_vertex)?;
    let (absolute, relative) =
        quadratic_discrepancy(mpb_quadratic, thc_quadratic, SCALAR_COULOMB_EXACTNESS_FLOOR);
    Ok(ScalarCoulombPairDiagnostic {
        q_index,
        spin: thc.spin,
        pair,
        column: mpb_vertex.column,
        mpb_quadratic,
        thc_quadratic,
        mpb_action_norm,
        thc_action_norm,
        quadratic_discrepancy: ScalarCoulombDiscrepancy { absolute, relative },
    })
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CoulombBridgeError {
    Coulomb(CoulombError),
    InterpolationProjection,
    GridIdentity { index: usize },
    ThcRecord { index: usize },
    VertexIdentity { index: usize, column: usize },
}

pub(crate) fn bind_interpolation_request(
    request: &CoulombRequest,
    projection: InterpolationProjection,
) -> Result<CoulombRequest, CoulombBridgeError> {
    match request.interpolation() {
        None => request
            .clone()
            .with_interpolation(projection)
            .map_err(CoulombBridgeError::Coulomb),
        Some(existing) if existing == projection => Ok(request.clone()),
        Some(_) => Err(CoulombBridgeError::InterpolationProjection),
    }
}

pub(crate) fn require_thc_q_record(
    input_q: TransferQ,
    input_layout: PairColumnLayout,
    grid: &ThcParentGrid,
    record: &ThcQRecord,
    index: usize,
) -> Result<(), CoulombBridgeError> {
    if record.grid_identity() != grid.identity() {
        return Err(CoulombBridgeError::GridIdentity { index });
    }
    let n_col = record
        .layout
        .n_columns()
        .map_err(|_| CoulombBridgeError::ThcRecord { index })?;
    let n_points = grid.points().len();
    let n_mu = record.auxiliary.dimension();
    let ok = record.q_index == index
        && record.q == input_q
        && record.layout == input_layout
        && record.auxiliary.q == record.q
        && record.auxiliary.partition == *grid.partition()
        && record.fit.q == record.q
        && record.fit.q_index == index
        && record.fit.n_points == n_points
        && record.fit.n_mu == n_mu
        && record.fit.zeta.len() == n_points.saturating_mul(n_mu)
        && record.vertices.len() == n_col;
    if !ok {
        return Err(CoulombBridgeError::ThcRecord { index });
    }
    let expected_layout = record.auxiliary.layout();
    for (column, vertex) in record.vertices.iter().enumerate() {
        match vertex.pair() {
            OrbitalPair::Bloch {
                k_index,
                left,
                right,
            } if vertex.layout() == &expected_layout
                && record.layout.decode(column) == (k_index, left, right)
                && vertex.provenance() == &record.auxiliary.provenance => {}
            _ => {
                return Err(CoulombBridgeError::VertexIdentity { index, column });
            }
        }
    }
    Ok(())
}

pub(crate) fn sampled_from_thc_record(
    record: &ThcQRecord,
    grid: &ThcParentGrid,
    site_meshes: Vec<ExponentialMesh>,
) -> Result<SampledAuxiliaryFunctions, CoulombError> {
    let points = grid
        .points()
        .iter()
        .map(|point| point.coordinate)
        .collect::<Vec<_>>();
    let weights = grid
        .points()
        .iter()
        .map(|point| VolumeBohr3(point.weight))
        .collect::<Vec<_>>();
    let supports = grid
        .points()
        .iter()
        .map(|point| sampled_support(point.region))
        .collect::<Vec<_>>();
    SampledAuxiliaryFunctions::new(
        record.auxiliary.layout(),
        site_meshes,
        points,
        weights,
        supports,
        record.fit.zeta.clone(),
    )
}

fn sampled_support(region: ThcRegion) -> SampledPointSupport {
    match region {
        ThcRegion::MuffinTin { site, radial_index } => {
            SampledPointSupport::MuffinTin { site, radial_index }
        }
        ThcRegion::Interstitial => SampledPointSupport::Interstitial,
    }
}

pub(crate) fn vertex_action_norm(
    operator: &CoulombOperator,
    vertex: &PairVertex,
) -> Result<f64, CoulombError> {
    let applied = operator.apply(vertex)?;
    Ok(applied.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt())
}

pub(crate) fn quadratic_discrepancy(left: Complex64, right: Complex64, floor: f64) -> (f64, f64) {
    let absolute = (left - right).norm();
    let scale = left.norm().max(right.norm()).max(floor);
    (absolute, absolute / scale)
}
