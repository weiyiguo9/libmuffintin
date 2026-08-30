//! Scalar adaptive THC from frozen [`ScalarProductInput`] on an external grid.

use crate::scalar_product::{
    ScalarProductInput, ScalarQSliceError, ScalarSpinChannel, require_scalar_q_slice,
};
use crate::site_coords::site_coordinate;
use crate::thc_grid::{
    ThcCandidates, ThcEngine, ThcGridError, ThcParentGrid, ThcQRecord, ThcRegion,
    records_match_parent_grid, require_parent_grid_radials,
};
use muffintin_prodbasis::{ProductOrbitalKind, ProductRadial, ProductRadialId, SiteRadialSet};
use muffintin_core::{Bohr, GVector, InverseBohr, complex_spherical_harmonics, lm_index};
use muffintin_operators::lapw::{CompiledBasis, Provenance};
use muffintin_operators::{CompiledSiteProjection, OperatorError, SiteOrbitalCoefficients};
use muffintin_tensor::DenseEigenvectors;
use muffintin_prodbasis::thc::{PairBlock, RankPolicy, Selection, ThcError, fit_allq_l2_pair_blocks};
use num_complex::Complex64;
use thiserror::Error;

type OrbitalSample = (Complex64, Complex64);
type OrbitalGrid = Vec<Vec<Vec<OrbitalSample>>>;

/// Production AllQL2 request for one collinear spin.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarThcSpec {
    /// Collinear spin of the frozen orbitals (`0` up, `1` down).
    pub spin: u8,
    /// Existing THC requested/effective rank policy.
    pub rank: RankPolicy,
    /// Deterministic candidate subset used by the selected L2 engine.
    pub candidates: ThcCandidates,
    /// Full production L2 engine. Callers must choose explicitly.
    pub engine: ThcEngine,
}

/// Scalar AllQL2 result carrying the sampled-$\zeta$ Coulomb interpolation-point seam.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarThcResult {
    pub grid: ThcParentGrid,
    pub selection: Selection,
    pub requested_rank: RankPolicy,
    pub effective_rank: usize,
    /// Collinear spin selected by [`ScalarThcSpec`] (`0` up, `1` down).
    ///
    /// Every per-$q$ record in this result uses this spin. Bloch pair vertices
    /// ([`muffintin_prodbasis::OrbitalPair::Bloch`]) do not carry a spin
    /// label.
    pub spin: u8,
    pub records: Vec<ThcQRecord>,
}

impl ScalarThcResult {
    /// Whether every per-$q$ $\zeta$ record was fitted on the stored parent grid.
    pub fn records_match_parent_grid(&self) -> bool {
        records_match_parent_grid(&self.grid, &self.records)
    }
}

/// Scalar adaptive-THC stage-boundary error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ScalarThcError {
    #[error(transparent)]
    Thc(#[from] ThcError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error("scalar THC q-slice must be nonempty")]
    EmptySlice,
    #[error("scalar THC q-slice has {actual} bundles, expected {expected} k-mesh transfers")]
    IncompleteQSlice { actual: usize, expected: usize },
    #[error("scalar THC inputs do not share one frozen orbital window, layout, and partition")]
    IncompatibleInputs,
    #[error("scalar THC spin {0} is not present in the frozen orbitals")]
    InvalidSpin(u8),
    #[error("scalar THC grid is not bound to the frozen product partition")]
    GridPartitionMismatch,
    #[error("scalar THC grid point {index} is outside the frozen product geometry")]
    GridPoint { index: usize },
    #[error(
        "scalar THC grid point {index} is not on muffin-tin site {site} radial sample {radial_index}"
    )]
    RadialShellMismatch {
        index: usize,
        site: usize,
        radial_index: usize,
    },
}

impl From<ThcGridError> for ScalarThcError {
    fn from(error: ThcGridError) -> Self {
        match error {
            ThcGridError::Thc(error) => Self::Thc(error),
            ThcGridError::GridPoint { index } => Self::GridPoint { index },
            ThcGridError::RadialShellMismatch {
                index,
                site,
                radial_index,
            } => Self::RadialShellMismatch {
                index,
                site,
                radial_index,
            },
        }
    }
}

/// Build AllQL2 interpolation points, $\zeta$, and pair vertices on a parent grid.
///
/// `inputs` is the complete k-mesh $q$ slice in production $q$-index order:
/// `inputs[iq]` is the scalar product-input bundle whose canonical $q$ is the $iq$-th mesh
/// point. Muffin-tin $P$ and $Q$ are reconstructed Bloch samples converted to
/// the cell-periodic representation by $\exp(-i k\cdot r)$ at the Cartesian
/// point, using the stored plane-wave Cartesian $k$. Pair density is
/// $\exp(+i G_{\mathrm{wrap}}\cdot r)\,(P_{k-q}^*P_k+Q_{k-q}^*Q_k)$ with the
/// stored per-column wrap. Global [`muffintin_prodbasis::TransferQ::umklapp`]
/// is not applied again. Auxiliaries are created with scalar THC provenance
/// before Bloch pair vertices.
pub fn build_scalar_thc(
    inputs: &[ScalarProductInput],
    grid: &ThcParentGrid,
    spec: &ScalarThcSpec,
) -> Result<ScalarThcResult, ScalarThcError> {
    let first = require_scalar_q_slice(inputs)?;
    if grid.partition() != &first.source.partition {
        return Err(ScalarThcError::GridPartitionMismatch);
    }
    require_parent_grid_radials(grid, |site| {
        first.source.radials.get(site).map(|set| &set.mesh)
    })?;
    let channel = spin_channel(first, spec.spin)?;
    let samples = evaluate_orbitals(first, grid, channel)?;
    let blocks = pair_blocks(inputs, grid, &samples)?;
    let cartesian = grid.cartesian();
    let weights = grid.weights();
    let regions = grid.interpolation_regions();
    let transfers = inputs
        .iter()
        .map(|input| input.source.q)
        .collect::<Vec<_>>();
    let fitted = fit_allq_l2_pair_blocks(
        &blocks,
        &cartesian,
        &weights,
        &regions,
        first.source.partition.clone(),
        &transfers,
        spec.rank,
        spec.engine.into(),
        spec.candidates.as_fit_indices(),
        Provenance {
            recipe: Some("scalar-thc-allq-l2".to_owned()),
            reference: Some("snapshot-dft-frozen-scalar-thc".to_owned()),
        },
    )?;
    let layout = first.pair_columns;
    let records = fitted
        .fits
        .into_iter()
        .zip(fitted.auxiliaries)
        .zip(fitted.vertices)
        .zip(transfers)
        .map(|(((fit, auxiliary), vertices), q)| {
            ThcQRecord::new(fit.q_index, q, layout, auxiliary, fit, vertices, grid)
        })
        .collect::<Vec<_>>();
    Ok(ScalarThcResult {
        grid: grid.clone(),
        selection: fitted.selection,
        requested_rank: spec.rank,
        effective_rank: records.first().map(|record| record.fit.n_mu).unwrap_or(0),
        spin: spec.spin,
        records,
    })
}

fn spin_channel(
    input: &ScalarProductInput,
    spin: u8,
) -> Result<&ScalarSpinChannel, ScalarThcError> {
    input
        .orbitals
        .channels
        .iter()
        .find(|channel| channel.spin == spin)
        .ok_or(ScalarThcError::InvalidSpin(spin))
}

impl From<ScalarQSliceError> for ScalarThcError {
    fn from(error: ScalarQSliceError) -> Self {
        match error {
            ScalarQSliceError::EmptySlice => Self::EmptySlice,
            ScalarQSliceError::IncompleteQSlice { actual, expected } => {
                Self::IncompleteQSlice { actual, expected }
            }
            ScalarQSliceError::IncompatibleInputs => Self::IncompatibleInputs,
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn evaluate_orbitals(
    input: &ScalarProductInput,
    grid: &ThcParentGrid,
    channel: &ScalarSpinChannel,
) -> Result<OrbitalGrid, ScalarThcError> {
    let n_k = channel.eigenvectors.len();
    let n_orb = input.orbitals.band_window.count;
    let mut samples = vec![
        vec![vec![(Complex64::default(), Complex64::default()); n_orb]; n_k];
        grid.points().len()
    ];
    let volume = input
        .source
        .partition
        .interstitial()
        .cell_volume()
        .get()
        .sqrt();
    let mut site_proj = Vec::with_capacity(n_k);
    for k in 0..n_k {
        let mut per_site = Vec::with_capacity(input.source.partition.site_count());
        for site in 0..input.source.partition.site_count() {
            let projection = CompiledSiteProjection::scalar(&channel.bases[k], site)?;
            per_site.push(projection.project_eigenvectors(&channel.eigenvectors[k])?);
        }
        site_proj.push(per_site);
    }
    for (p, point) in grid.points().iter().enumerate() {
        match point.region {
            ThcRegion::MuffinTin { site, radial_index } => {
                let radials = &input.source.radials[site];
                let origin = input.source.partition.sites()[site].position;
                let direction = [
                    point.coordinate[0].get() - origin[0].get(),
                    point.coordinate[1].get() - origin[1].get(),
                    point.coordinate[2].get() - origin[2].get(),
                ];
                let radius = radials.mesh.radii()[radial_index].get();
                for k in 0..n_k {
                    let k_phase =
                        minus_i_k_dot_r(compiled_cartesian_k(&channel.bases[k])?, point.coordinate);
                    let projected = &site_proj[k][site];
                    for band in 0..n_orb {
                        let (large, small) = muffin_tin_orbital(
                            &channel.bases[k],
                            site,
                            channel.spin,
                            projected,
                            band,
                            radials,
                            radial_index,
                            radius,
                            direction,
                        )?;
                        samples[p][k][band] = (large * k_phase, small * k_phase);
                    }
                }
            }
            ThcRegion::Interstitial => {
                for k in 0..n_k {
                    for band in 0..n_orb {
                        samples[p][k][band] = (
                            interstitial_orbital(
                                &channel.bases[k],
                                &channel.eigenvectors[k],
                                band,
                                point.coordinate,
                                volume,
                            ),
                            Complex64::default(),
                        );
                    }
                }
            }
        }
    }
    Ok(samples)
}

#[allow(clippy::too_many_arguments)]
fn muffin_tin_orbital(
    compiled: &CompiledBasis,
    site: usize,
    spin: u8,
    projected: &SiteOrbitalCoefficients,
    band: usize,
    radials: &SiteRadialSet,
    radial_index: usize,
    radius: f64,
    direction: [f64; 3],
) -> Result<(Complex64, Complex64), ScalarThcError> {
    let inv_r = if radius > 0.0 { 1.0 / radius } else { 0.0 };
    let l_max = (0..projected.coordinate_count())
        .filter_map(|coord| site_coordinate(compiled, site, spin, coord).map(|(id, _)| id.l))
        .max()
        .unwrap_or(0);
    let harmonics = complex_spherical_harmonics(l_max, direction);
    let mut large = Complex64::default();
    let mut small = Complex64::default();
    for coord in 0..projected.coordinate_count() {
        let Some((id, m)) = site_coordinate(compiled, site, spin, coord) else {
            return Err(ScalarThcError::IncompatibleInputs);
        };
        let radial = find_radial(radials, id).ok_or(ScalarThcError::IncompatibleInputs)?;
        let y = harmonics[lm_index(id.l, m).map_err(ThcError::from)?];
        let amplitude = projected.at(coord, band) * y * inv_r;
        large += amplitude * radial.samples.large[radial_index];
        if let Some(q) = radial.samples.small.as_ref() {
            small += amplitude * q[radial_index];
        }
    }
    Ok((large, small))
}

fn find_radial(radials: &SiteRadialSet, id: ProductRadialId) -> Option<&ProductRadial> {
    let pool = match id.kind {
        ProductOrbitalKind::Valence => radials.valence.as_slice(),
        ProductOrbitalKind::Core => radials.cores.as_slice(),
    };
    pool.iter()
        .find(|radial| radial.l == id.l && radial.n == id.n && radial.spin == id.spin)
}

fn interstitial_orbital(
    compiled: &CompiledBasis,
    eigenvectors: &DenseEigenvectors,
    band: usize,
    coordinate: [Bohr; 3],
    sqrt_volume: f64,
) -> Complex64 {
    let mut value = Complex64::default();
    for (row, wave) in compiled.plane_waves.iter().enumerate() {
        value += eigenvectors.at(row, band) * plus_i_g_dot_r(wave.g, coordinate) / sqrt_volume;
    }
    value
}

fn plus_i_g_dot_r(g: GVector, coordinate: [Bohr; 3]) -> Complex64 {
    let argument = g
        .cartesian
        .iter()
        .zip(coordinate)
        .map(|(component, point)| component.get() * point.get())
        .sum();
    Complex64::from_polar(1.0, argument)
}

fn compiled_cartesian_k(compiled: &CompiledBasis) -> Result<[InverseBohr; 3], ScalarThcError> {
    compiled
        .plane_waves
        .first()
        .map(|wave| wave.k)
        .ok_or(ScalarThcError::IncompatibleInputs)
}

fn minus_i_k_dot_r(k: [InverseBohr; 3], coordinate: [Bohr; 3]) -> Complex64 {
    let argument: f64 = k
        .iter()
        .zip(coordinate)
        .map(|(component, point)| component.get() * point.get())
        .sum();
    Complex64::from_polar(1.0, -argument)
}

#[allow(clippy::needless_range_loop)]
fn pair_blocks(
    inputs: &[ScalarProductInput],
    grid: &ThcParentGrid,
    samples: &[Vec<Vec<OrbitalSample>>],
) -> Result<Vec<PairBlock>, ScalarThcError> {
    let layout = inputs[0].pair_columns;
    let n_orb = layout.n_orb;
    let n_col = layout.n_columns().map_err(ThcError::from)?;
    let n_points = grid.points().len();
    let mut blocks = Vec::with_capacity(inputs.len());
    for (iq, input) in inputs.iter().enumerate() {
        let mut values = vec![Complex64::default(); n_points * n_col];
        for mapped in &input.k_minus_q {
            for (p, point) in grid.points().iter().enumerate() {
                let phase = plus_i_g_dot_r(mapped.umklapp, point.coordinate);
                for left_band in 0..n_orb {
                    let (p_left, q_left) = samples[p][mapped.kq_index][left_band];
                    for right_band in 0..n_orb {
                        let (p_right, q_right) = samples[p][mapped.k_index][right_band];
                        let column = layout.encode(mapped.k_index, left_band, right_band);
                        values[p * n_col + column] =
                            phase * (p_left.conj() * p_right + q_left.conj() * q_right);
                    }
                }
            }
        }
        blocks.push(PairBlock::new(iq, n_points, layout, values)?);
    }
    Ok(blocks)
}
