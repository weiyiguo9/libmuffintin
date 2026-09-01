//! Representation-neutral parent grid, candidates, engines, and q-record context.

use muffintin_core::{
    AngularGrid, Bohr, Cell, ExponentialMesh, Grid, GridError, InterstitialGrid, LatticeError,
    ReciprocalLattice, UniformGrid,
};
use muffintin_operators::lapw::Provenance;
use muffintin_prodbasis::thc::{L2Engine, PerQFit, ThcError, validate_quadrature_weights};
use muffintin_prodbasis::{
    AuxiliaryPartition, CompiledAuxiliaryBasis, InterpolationRegion, PairColumnLayout, PairVertex,
    TransferQ,
};
use thiserror::Error;

const RADIAL_SHELL_TOLERANCE: f64 = 1.0e-10;

/// Candidate-point policy for AllQL2 L2 selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ThcCandidates {
    /// Every strictly positive-weight parent-grid point, in parent order.
    All,
    /// Explicit parent-grid indices, in caller order.
    ///
    /// Zero-weight indices are rejected rather than dropped.
    Indices(Vec<usize>),
}

impl ThcCandidates {
    pub(crate) fn as_fit_indices(&self) -> Option<&[usize]> {
        match self {
            Self::All => None,
            Self::Indices(indices) => Some(indices.as_slice()),
        }
    }
}

/// Production AllQL2 full L2 engines. Structured sketches are not in this type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThcEngine {
    /// Full weighted column-pivoted QR.
    FullColumnPivotedQr,
    /// Pivoted Cholesky of the weighted point Gram.
    ///
    /// The dense Gram is not formed. The stacked weighted pair matrix is still
    /// materialized.
    FullPivotedCholesky,
}

impl From<ThcEngine> for L2Engine {
    fn from(engine: ThcEngine) -> Self {
        match engine {
            ThcEngine::FullColumnPivotedQr => Self::FullColumnPivotedQr,
            ThcEngine::FullPivotedCholesky => Self::FullPivotedCholesky,
        }
    }
}

/// Typed muffin-tin or interstitial parent-grid region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThcRegion {
    /// Stored exponential-mesh sample on muffin-tin `site`.
    MuffinTin { site: usize, radial_index: usize },
    /// Partitioned interstitial.
    Interstitial,
}

impl ThcRegion {
    pub(crate) const fn interpolation_region(self) -> InterpolationRegion {
        match self {
            Self::MuffinTin { site, .. } => InterpolationRegion::MuffinTin { site },
            Self::Interstitial => InterpolationRegion::Interstitial,
        }
    }
}

/// One immutable parent-grid point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThcPoint {
    pub coordinate: [Bohr; 3],
    /// True quadrature weight. Zeros are allowed; they are not clamped.
    pub weight: f64,
    pub region: ThcRegion,
}

/// Content fingerprint binding a parent grid to the per-$q$ $\zeta$ fits.
///
/// The mixer is a deterministic splitmix-style 64-bit fold of ordered
/// lengths, type tags, and `f64` bit patterns. It is an internal binding
/// stamp, not scientific provenance or a cryptographic digest. Distinct
/// grids can collide at a residual of one part in $2^{64}$ per comparison.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ParentGridIdentity(u64);

/// Parent-grid construction error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ThcGridError {
    #[error(transparent)]
    Thc(#[from] ThcError),
    #[error("THC parent-grid point {index} is outside the frozen product geometry")]
    GridPoint { index: usize },
    #[error(
        "THC parent-grid point {index} is not on muffin-tin site {site} radial sample {radial_index}"
    )]
    RadialShellMismatch {
        index: usize,
        site: usize,
        radial_index: usize,
    },
}

/// Natural muffin-tin/interstitial parent-grid construction parameters.
///
/// Muffin-tin points use every stored exponential-mesh radial shell crossed
/// with a deterministic Fibonacci angular rule. Interstitial points are the
/// midpoint-rule uniform-cell grid after exact periodic sphere masking. The
/// later AllQL2 QRCP or pivoted-Cholesky stage adaptively selects interpolation
/// points from this parent support.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NaturalThcGridSpec {
    pub angular_points_per_shell: usize,
    pub interstitial_divisions: [usize; 3],
}

/// Failure while constructing a natural-partition THC parent grid.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum NaturalThcGridError {
    #[error(transparent)]
    Grid(#[from] GridError),
    #[error(transparent)]
    Lattice(#[from] LatticeError),
    #[error(transparent)]
    Thc(#[from] ThcGridError),
    #[error("natural THC grid reciprocal lattice does not match its direct cell")]
    ReciprocalMismatch,
    #[error("natural THC grid received {actual} radial meshes for {expected} muffin-tin sites")]
    SiteMeshCount { actual: usize, expected: usize },
    #[error(
        "natural THC grid site {site} mesh ends at {mesh_radius} Bohr, expected muffin-tin radius {sphere_radius} Bohr"
    )]
    SiteRadius {
        site: usize,
        mesh_radius: f64,
        sphere_radius: f64,
    },
    #[error(
        "natural THC grid cell volume {cell_volume} Bohr^3 does not match partition volume {partition_volume} Bohr^3"
    )]
    CellVolume {
        cell_volume: f64,
        partition_volume: f64,
    },
}

/// Externally supplied parent support for adaptive THC.
///
/// Construction fingerprints the ordered points, weights, regions, partition,
/// and provenance so a later permutation cannot keep the original $\zeta$ fits.
#[derive(Clone, Debug, PartialEq)]
pub struct ThcParentGrid {
    partition: AuxiliaryPartition,
    provenance: Provenance,
    points: Vec<ThcPoint>,
    identity: ParentGridIdentity,
    natural_lattice: Option<NaturalThcLatticeIdentity>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct NaturalThcLatticeIdentity {
    cell: Cell,
    reciprocal: ReciprocalLattice,
}

impl ThcParentGrid {
    /// Construct after checking finite coordinates, site indices, and weights.
    pub fn new(
        partition: AuxiliaryPartition,
        provenance: Provenance,
        points: Vec<ThcPoint>,
    ) -> Result<Self, ThcGridError> {
        if points.is_empty() {
            return Err(ThcError::EmptyGrid.into());
        }
        let n_sites = partition.site_count();
        for (index, point) in points.iter().enumerate() {
            if point
                .coordinate
                .iter()
                .any(|component| !component.get().is_finite())
            {
                return Err(ThcGridError::GridPoint { index });
            }
            if let ThcRegion::MuffinTin { site, .. } = point.region
                && site >= n_sites
            {
                return Err(ThcGridError::GridPoint { index });
            }
        }
        validate_quadrature_weights(&points.iter().map(|point| point.weight).collect::<Vec<_>>())?;
        let identity = parent_grid_identity(&partition, &provenance, &points);
        Ok(Self {
            partition,
            provenance,
            points,
            identity,
            natural_lattice: None,
        })
    }

    /// Partition bound to this grid.
    pub const fn partition(&self) -> &AuxiliaryPartition {
        &self.partition
    }

    /// Construction provenance.
    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// Ordered parent-grid points.
    pub fn points(&self) -> &[ThcPoint] {
        &self.points
    }

    pub(crate) const fn identity(&self) -> ParentGridIdentity {
        self.identity
    }

    pub(crate) fn natural_reciprocal_matches(&self, reciprocal: &ReciprocalLattice) -> bool {
        self.natural_lattice
            .as_ref()
            .is_none_or(|identity| &identity.reciprocal == reciprocal)
    }

    pub(crate) fn cartesian(&self) -> Vec<[f64; 3]> {
        self.points
            .iter()
            .map(|point| point.coordinate.map(Bohr::get))
            .collect()
    }

    pub(crate) fn weights(&self) -> Vec<f64> {
        self.points.iter().map(|point| point.weight).collect()
    }

    pub(crate) fn interpolation_regions(&self) -> Vec<InterpolationRegion> {
        self.points
            .iter()
            .map(|point| point.region.interpolation_region())
            .collect()
    }
}

/// Build the Weinert-natural muffin-tin plus interstitial parent grid.
///
/// This function constructs the integration support only. Adaptive ISDF point
/// selection remains owned by [`crate::build_scalar_thc`] and
/// [`crate::build_spinor_thc`] through their explicit rank policy and engine.
pub fn build_natural_thc_parent_grid(
    partition: AuxiliaryPartition,
    cell: Cell,
    reciprocal: ReciprocalLattice,
    site_meshes: &[ExponentialMesh],
    provenance: Provenance,
    spec: NaturalThcGridSpec,
) -> Result<ThcParentGrid, NaturalThcGridError> {
    if ReciprocalLattice::from_direct(*cell.basis())? != reciprocal {
        return Err(NaturalThcGridError::ReciprocalMismatch);
    }
    let partition_volume = partition.interstitial().cell_volume().get();
    let cell_volume = cell.volume().get();
    let volume_scale = partition_volume.abs().max(cell_volume.abs()).max(1.0);
    if (partition_volume - cell_volume).abs() > 1.0e-12 * volume_scale {
        return Err(NaturalThcGridError::CellVolume {
            cell_volume,
            partition_volume,
        });
    }
    if site_meshes.len() != partition.site_count() {
        return Err(NaturalThcGridError::SiteMeshCount {
            actual: site_meshes.len(),
            expected: partition.site_count(),
        });
    }
    let angular = AngularGrid::fibonacci(spec.angular_points_per_shell)?;
    let mut points = Vec::new();
    for (site, mesh) in partition.sites().iter().zip(site_meshes) {
        let mesh_radius = mesh.last().get();
        let sphere_radius = site.radius.get();
        let radius_scale = mesh_radius.abs().max(sphere_radius.abs()).max(1.0);
        if (mesh_radius - sphere_radius).abs() > RADIAL_SHELL_TOLERANCE * radius_scale {
            return Err(NaturalThcGridError::SiteRadius {
                site: site.index,
                mesh_radius,
                sphere_radius,
            });
        }
        for (radial_index, (&radius, &radial_weight)) in
            mesh.radii().iter().zip(mesh.weights()).enumerate()
        {
            for angular_point in angular.points() {
                let coordinate = std::array::from_fn(|axis| {
                    Bohr(site.position[axis].get() + radius.get() * angular_point.direction[axis])
                });
                points.push(ThcPoint {
                    coordinate,
                    weight: radial_weight * radius.get().powi(2) * angular_point.weight,
                    region: ThcRegion::MuffinTin {
                        site: site.index,
                        radial_index,
                    },
                });
            }
        }
    }

    let uniform = UniformGrid::new(cell, spec.interstitial_divisions)?;
    let interstitial = InterstitialGrid::new(&uniform, partition.interstitial().spheres())?;
    points.extend(interstitial.points().iter().map(|point| ThcPoint {
        coordinate: point.position,
        weight: point.weight.get(),
        region: ThcRegion::Interstitial,
    }));
    let mut grid = ThcParentGrid::new(partition, provenance, points)?;
    grid.natural_lattice = Some(NaturalThcLatticeIdentity { cell, reciprocal });
    Ok(grid)
}

/// Per-$q$ interpolation-point auxiliary, $\zeta$, and pair vertices.
#[derive(Clone, Debug, PartialEq)]
pub struct ThcQRecord {
    pub q_index: usize,
    pub q: TransferQ,
    pub layout: PairColumnLayout,
    pub auxiliary: CompiledAuxiliaryBasis,
    pub fit: PerQFit,
    pub vertices: Vec<PairVertex>,
    grid_identity: ParentGridIdentity,
}

impl ThcQRecord {
    pub(crate) fn new(
        q_index: usize,
        q: TransferQ,
        layout: PairColumnLayout,
        auxiliary: CompiledAuxiliaryBasis,
        fit: PerQFit,
        vertices: Vec<PairVertex>,
        grid: &ThcParentGrid,
    ) -> Self {
        Self {
            q_index,
            q,
            layout,
            auxiliary,
            fit,
            vertices,
            grid_identity: grid.identity,
        }
    }

    pub(crate) const fn grid_identity(&self) -> ParentGridIdentity {
        self.grid_identity
    }
}

pub(crate) fn records_match_parent_grid(grid: &ThcParentGrid, records: &[ThcQRecord]) -> bool {
    records
        .iter()
        .all(|record| record.grid_identity == grid.identity)
}

pub(crate) fn require_parent_grid_radials<'a>(
    grid: &ThcParentGrid,
    mesh_at: impl Fn(usize) -> Option<&'a ExponentialMesh>,
) -> Result<(), ThcGridError> {
    for (index, point) in grid.points.iter().enumerate() {
        if let ThcRegion::MuffinTin { site, radial_index } = point.region {
            let Some(mesh) = mesh_at(site) else {
                return Err(ThcGridError::GridPoint { index });
            };
            if radial_index >= mesh.radii().len() {
                return Err(ThcGridError::GridPoint { index });
            }
            let origin = grid.partition.sites()[site].position;
            let observed = cartesian_distance(point.coordinate, origin);
            let expected = mesh.radii()[radial_index].get();
            let scale = observed.abs().max(expected.abs()).max(1.0);
            if (observed - expected).abs() > RADIAL_SHELL_TOLERANCE * scale {
                return Err(ThcGridError::RadialShellMismatch {
                    index,
                    site,
                    radial_index,
                });
            }
        }
    }
    Ok(())
}

fn parent_grid_identity(
    partition: &AuxiliaryPartition,
    provenance: &Provenance,
    points: &[ThcPoint],
) -> ParentGridIdentity {
    let mut hash = mix(0xA11C_941D_0001_0001, points.len() as u64);
    hash = mix(hash, partition.site_count() as u64);
    hash = mix(hash, partition.interstitial().cell_volume().get().to_bits());
    for site in partition.sites() {
        hash = mix(hash, site.index as u64);
        for coordinate in site.position {
            hash = mix(hash, coordinate.get().to_bits());
        }
        hash = mix(hash, site.radius.get().to_bits());
    }
    hash = mix_opt_str(hash, partition.provenance().recipe.as_deref());
    hash = mix_opt_str(hash, partition.provenance().reference.as_deref());
    hash = mix_opt_str(hash, provenance.recipe.as_deref());
    hash = mix_opt_str(hash, provenance.reference.as_deref());
    for (index, point) in points.iter().enumerate() {
        hash = mix(hash, index as u64);
        for coordinate in point.coordinate {
            hash = mix(hash, coordinate.get().to_bits());
        }
        hash = mix(hash, point.weight.to_bits());
        match point.region {
            ThcRegion::MuffinTin { site, radial_index } => {
                hash = mix(hash, 1);
                hash = mix(hash, site as u64);
                hash = mix(hash, radial_index as u64);
            }
            ThcRegion::Interstitial => {
                hash = mix(hash, 2);
            }
        }
    }
    ParentGridIdentity(hash)
}

fn mix(hash: u64, lane: u64) -> u64 {
    hash.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(lane)
}

fn mix_opt_str(hash: u64, value: Option<&str>) -> u64 {
    match value {
        None => mix(hash, 0),
        Some(text) => {
            let mut hash = mix(hash, 1);
            hash = mix(hash, text.len() as u64);
            for &byte in text.as_bytes() {
                hash = mix(hash, u64::from(byte));
            }
            hash
        }
    }
}

fn cartesian_distance(point: [Bohr; 3], origin: [Bohr; 3]) -> f64 {
    point
        .iter()
        .zip(origin)
        .map(|(component, center)| {
            let delta = component.get() - center.get();
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}
