//! Coulomb construction, assembly, and context errors.

use muffintin_core::GridError;
use muffintin_core::{LatticeError, LmError, MeshError, StepFunctionError};
use muffintin_prodbasis::AuxiliaryIrError;
use muffintin_tensor::TensorError;
use thiserror::Error;

/// Coulomb operator construction or application error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum CoulombError {
    #[error(transparent)]
    Product(#[from] AuxiliaryIrError),
    #[error(transparent)]
    Lattice(#[from] LatticeError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    StepFunction(#[from] StepFunctionError),
    #[error(transparent)]
    Grid(#[from] GridError),
    #[error(transparent)]
    Angular(#[from] LmError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("Weinert angular cutoff must be at most {max}, got {0}", max = crate::MAX_LEXP)]
    InvalidLexp(u32),
    #[error("interpolation-point Coulomb assembly requires sampled zeta functions")]
    MissingSampledFunctions,
    #[error("sampled zeta functions are not used for mixed-product Coulomb assembly")]
    UnexpectedSampledFunctions,
    #[error("interpolation-point Coulomb assembly requires an interpolation projection spec")]
    MissingInterpolationProjection,
    #[error("interpolation PW cutoff must be finite and nonnegative, got {0}")]
    InvalidPwCutoff(f64),
    #[error("interpolation angular cutoff exceeds Weinert LEXP ({l_max} > {lexp})")]
    InterpolationLmax { l_max: u32, lexp: u32 },
    #[error("Spencer-Alavi truncation needs a positive full-BZ k-point count, got {0}")]
    InvalidTruncationKPointCount(usize),
    #[error("Spencer-Alavi reciprocal cutoff must be finite and positive, got {0}")]
    InvalidTruncationReciprocalCutoff(f64),
    #[error("smoothed spherical truncation needs a finite positive omega, got {0}")]
    InvalidTruncationSmoothing(f64),
    #[error("cell volume {cell} does not match the product partition volume {partition}")]
    CellVolumeMismatch { cell: f64, partition: f64 },
    #[error("Coulomb request reciprocal lattice does not match the direct cell")]
    ReciprocalMismatch,
    #[error("auxiliary interstitial G={index:?} does not match the request reciprocal lattice")]
    WaveLatticeMismatch { index: [i32; 3] },
    #[error("auxiliary interstitial G={index:?} has zero |q+G| outside the Gamma head")]
    ZeroQPlusG { index: [i32; 3] },
    #[error("auxiliary muffin-tin L={l} exceeds Weinert LEXP={lexp}")]
    AuxiliaryLExceedsLexp { l: u32, lexp: u32 },
    #[error("factorial table overflow at n={0} (reduce LEXP)")]
    FactorialOverflow(usize),
    #[error("structure-constant angular index {index} is outside 0..{count}")]
    StructureIndex { index: usize, count: usize },
    #[error("Ewald splitting parameter must be finite and positive, got {0}")]
    InvalidEwaldEta(f64),
    #[error("Ewald real-space cutoff must be finite and nonnegative, got {0}")]
    InvalidEwaldRealCutoff(f64),
    #[error("Ewald reciprocal cutoff must be finite and nonnegative, got {0}")]
    InvalidEwaldRecipCutoff(f64),
    #[error("Ewald convergence tolerance must be finite and positive, got {0}")]
    InvalidEwaldTolerance(f64),
    #[error("Ewald convergence scan needs at least two steps, got {0}")]
    InvalidEwaldSteps(usize),
    #[error(
        "Ewald kernel did not reach successive residual {tolerance} in {steps} steps (last residual {residual})"
    )]
    EwaldNotConverged {
        residual: f64,
        tolerance: f64,
        steps: usize,
    },
    #[error("real-space structure-constant cutoff search did not converge")]
    RealSpaceCutoffNotConverged,
    #[error("reciprocal-space structure-constant cutoff search did not converge")]
    ReciprocalSpaceCutoffNotConverged,
    #[error("pair vertex transfer q does not match the Coulomb operator")]
    VertexTransferQ,
    #[error(
        "pair vertex dimensions mt={vertex_mt}+I={vertex_interstitial} do not match the Coulomb operator mt={operator_mt}+I={operator_interstitial}"
    )]
    VertexDimension {
        vertex_mt: usize,
        vertex_interstitial: usize,
        operator_mt: usize,
        operator_interstitial: usize,
    },
    #[error("pair vertex auxiliary layout does not match the Coulomb operator")]
    VertexLayout,
    #[error("Coulomb vertex block dimension overflow")]
    DimensionOverflow,
    #[error(
        "Coulomb vertex block has {vertices} columns, expected {occupied} occupied states times {targets} targets"
    )]
    VertexBlockDimension {
        vertices: usize,
        occupied: usize,
        targets: usize,
    },
    #[error("sampled zeta layout does not match the compiled auxiliary")]
    SampledLayoutMismatch,
    #[error(
        "sampled zeta has {actual} values, expected {n_grid} grid points times {n_mu} functions"
    )]
    SampledZetaLength {
        actual: usize,
        n_grid: usize,
        n_mu: usize,
    },
    #[error("sampled zeta has {n_mu} functions, expected auxiliary dimension {expected}")]
    SampledZetaDimension { n_mu: usize, expected: usize },
    #[error("sampled grid has {points} points, {weights} weights, {supports} support labels")]
    SampledGridLength {
        points: usize,
        weights: usize,
        supports: usize,
    },
    #[error("sampled grid is empty")]
    EmptySampledGrid,
    #[error("sampled weight {0} is not finite")]
    NonFiniteSampledWeight(usize),
    #[error("sampled weight {0} is negative")]
    NegativeSampledWeight(usize),
    #[error("sampled grid has no strictly positive quadrature weight")]
    NoPositiveSampledWeight,
    #[error("sampled point {0} has a non-finite coordinate")]
    NonFiniteSampledPoint(usize),
    #[error("sampled zeta entry {0} is not finite")]
    NonFiniteZeta(usize),
    #[error("Coulomb matrix index {index} is outside 0..{dimension}")]
    MatrixIndex { index: usize, dimension: usize },
    #[error("radial sample at index {index} is not finite")]
    NonFiniteRadial { index: usize },
    #[error("Coulomb matrix entry ({row}, {column}) is not finite")]
    NonFiniteMatrix { row: usize, column: usize },
    #[error("interpolation point {0} lies outside its tagged muffin-tin sphere")]
    InterpolationPointOutsideSphere(usize),
    #[error("sampled muffin-tin site {site} is outside the partition")]
    SampledPointSite { site: usize },
    #[error("sampled radial mesh {site} is not outward (increment {increment})")]
    SampledMeshNotOutward { site: usize, increment: f64 },
    #[error("sampled input has {actual} site meshes, expected {expected}")]
    SampledMeshCount { expected: usize, actual: usize },
    #[error("sampled mesh {site} ends at {mesh}, but the partition sphere ends at {sphere}")]
    SampledMeshRadius { site: usize, mesh: f64, sphere: f64 },
    #[error(
        "sampled point {point} names radial index {index} outside site {site} mesh of length {count}"
    )]
    SampledRadialIndex {
        point: usize,
        site: usize,
        index: usize,
        count: usize,
    },
    #[error(
        "sampled point {point} radius {coordinate} does not match site {site} shell radius {shell}"
    )]
    SampledCoordinateShellMismatch {
        point: usize,
        site: usize,
        coordinate: f64,
        shell: f64,
    },
    #[error("site {0} is missing from the product partition")]
    MissingSite(usize),
    #[error("compiled auxiliary basis has dimension 0")]
    EmptyAuxiliary,
    #[error("point-charge oracle requires an interpolation-point auxiliary")]
    ExpectedInterpolationPoints,
}
