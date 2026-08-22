//! THC construction and validation errors.

use libmuffintin_core::LmError;
use libmuffintin_product::ProductError;
use thiserror::Error;

/// k-point ISDF/THC error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ThcError {
    #[error(transparent)]
    Product(#[from] ProductError),
    #[error(transparent)]
    Harmonic(#[from] LmError),
    #[error("k-mesh divisions must be positive, got {0:?}")]
    InvalidKMeshDivisions([u32; 3]),
    #[error("k-mesh index {index} is out of range for {count} points")]
    KMeshIndex { index: usize, count: usize },
    #[error("storage length overflow for dimensions {dimensions:?}")]
    DimensionOverflow { dimensions: Vec<usize> },
    #[error("orbital array has {actual} values, expected {expected}")]
    OrbitalCount { expected: usize, actual: usize },
    #[error("orbitals have {orbitals} grid points, expected {points}")]
    OrbitalPointCount { orbitals: usize, points: usize },
    #[error("orbitals have {orbitals} k-points, expected mesh length {mesh}")]
    OrbitalKCount { orbitals: usize, mesh: usize },
    #[error("grid has {points} points but {weights} weights")]
    GridWeightCount { points: usize, weights: usize },
    #[error("grid has {points} points but {regions} region tags")]
    GridRegionCount { points: usize, regions: usize },
    #[error("quadrature weight {index} is negative or non-finite ({value})")]
    InvalidWeight { index: usize, value: f64 },
    #[error("quadrature weights have no strictly positive entry")]
    NoPositiveWeight,
    #[error("core orbital index {index} is outside n_orb={n_orb}")]
    InvalidCoreOrbital { index: usize, n_orb: usize },
    #[error("interpolation grid has no points")]
    EmptyGrid,
    #[error("pair block has {actual} entries, expected {expected}")]
    PairBlockLength { expected: usize, actual: usize },
    #[error("requested Nμ={n_mu} exceeds the {n_points}-point grid")]
    RankExceedsGrid { n_mu: usize, n_points: usize },
    #[error("Nμ must be positive")]
    EmptyRank,
    #[error(
        "L2 selector leading residual is absent, non-finite, or non-positive; threshold selection cannot choose a rank"
    )]
    DegenerateRank,
    #[error("pool factor must be an integer >= 1, got {0}")]
    InvalidPoolFactor(usize),
    #[error("threshold {0} must be finite and positive")]
    InvalidThreshold(f64),
    #[error("allq_coulomb_pool requires an explicit Nμ; threshold termination is L2-only")]
    CoulombPoolRequiresExactRank,
    #[error("allq_coulomb_pool requires injected Coulomb grams covering every canonical q")]
    MissingCoulombGrams,
    #[error("Coulomb gram for q-index {index} is not Hermitian (relative {relative:.3e})")]
    GramNotHermitian { index: usize, relative: f64 },
    #[error("Coulomb gram for q-index {index} has {actual_len} entries, expected {expected_len}")]
    GramShape {
        index: usize,
        expected_len: usize,
        actual_len: usize,
    },
    #[error("Coulomb gram q-index {actual} does not match requested {expected}")]
    GramQIndex { expected: usize, actual: usize },
    #[error("Coulomb gram transfer q does not match the pair block at q-index {0}")]
    GramTransferQ(usize),
    #[error("Coulomb gram column order does not match the pair layout at q-index {0}")]
    GramColumnOrder(usize),
    #[error("pair block {index} has {actual} points, expected {expected}")]
    PairBlockPointCount {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("pair block {index} layout does not match the first block")]
    PairBlockLayout { index: usize },
    #[error("Coulomb gram entry is not finite at q-index {0}")]
    GramNonFinite(usize),
    #[error(
        "Coulomb gram at q-index {index} is not positive semidefinite (λmin={min}, λmax={max})"
    )]
    GramIndefinite { index: usize, min: f64, max: f64 },
    #[error("linear algebra failed: {0}")]
    LinearAlgebra(&'static str),
    #[error("sketch row count must be positive")]
    EmptySketch,
    #[error("structured sketch has {rows} rows but exact selection requires {required}")]
    SketchRankExceedsRows { rows: usize, required: usize },
    #[error("lattice constant must be finite and positive, got {0}")]
    InvalidLattice(f64),
    #[error("interpolation point {0} is outside the parent grid")]
    PointIndex(usize),
}

/// Product of axis lengths that must fit in one contiguous storage buffer.
pub fn checked_storage_len(dimensions: &[usize]) -> Result<usize, ThcError> {
    dimensions.iter().try_fold(1_usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| ThcError::DimensionOverflow {
                dimensions: dimensions.to_vec(),
            })
    })
}

/// Reject negative/non-finite weights; require at least one strictly positive entry.
///
/// Individual zeros are allowed. Negative values are not clamped to zero.
pub fn validate_quadrature_weights(weights: &[f64]) -> Result<(), ThcError> {
    let mut any_positive = false;
    for (index, &value) in weights.iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(ThcError::InvalidWeight { index, value });
        }
        if value > 0.0 {
            any_positive = true;
        }
    }
    if !any_positive {
        return Err(ThcError::NoPositiveWeight);
    }
    Ok(())
}
