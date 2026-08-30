//! Spinor AllQL2 THC from frozen [`SpinorProductInput`] on a [`ThcParentGrid`].

use crate::spinor_product::{SpinorProductInput, SpinorQSliceError, require_spinor_q_slice};
use crate::thc_grid::{
    ThcCandidates, ThcEngine, ThcGridError, ThcParentGrid, ThcQRecord, ThcRegion,
    records_match_parent_grid, require_parent_grid_radials,
};
use muffintin_core::{
    Bohr, GVector, InverseBohr, RelativisticChannel, SpinProjection, complex_spherical_harmonics,
    lm_index,
};
use muffintin_operators::lapw::{Provenance, SpinorCompiledBasis};
use muffintin_operators::{CompiledSiteProjection, OperatorError, SiteOrbitalCoefficients};
use muffintin_prodbasis::thc::{
    PairBlock, RankPolicy, Selection, ThcError, fit_allq_l2_pair_blocks,
};
use muffintin_prodbasis::{DiracRadial, DiracRadialId, DiracSiteRadialSet, ProductOrbitalKind};
use muffintin_tensor::DenseEigenvectors;
use num_complex::Complex64;
use thiserror::Error;

#[derive(Clone, Copy, Default)]
struct SpinorOrbitalSample {
    large: [Complex64; 2],
    small: [Complex64; 2],
}

type OrbitalGrid = Vec<Vec<Vec<SpinorOrbitalSample>>>;

/// Production AllQL2 request for one spinor band manifold.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorThcSpec {
    /// Existing THC requested/effective rank policy.
    pub rank: RankPolicy,
    /// Deterministic candidate subset used by the selected L2 engine.
    pub candidates: ThcCandidates,
    /// Full production L2 engine. Callers must choose explicitly.
    pub engine: ThcEngine,
}

/// Spinor AllQL2 result carrying the sampled-$\zeta$ interpolation-point seam.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorThcResult {
    pub grid: ThcParentGrid,
    pub selection: Selection,
    pub requested_rank: RankPolicy,
    pub effective_rank: usize,
    pub records: Vec<ThcQRecord>,
}

impl SpinorThcResult {
    /// Whether every per-$q$ $\zeta$ record was fitted on the stored parent grid.
    pub fn records_match_parent_grid(&self) -> bool {
        records_match_parent_grid(&self.grid, &self.records)
    }
}

/// Spinor adaptive-THC stage-boundary error.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SpinorThcError {
    #[error(transparent)]
    Thc(#[from] ThcError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error("spinor THC q-slice must be nonempty")]
    EmptySlice,
    #[error("spinor THC q-slice has {actual} bundles, expected {expected} k-mesh transfers")]
    IncompleteQSlice { actual: usize, expected: usize },
    #[error("spinor THC inputs do not share one frozen orbital window, layout, and partition")]
    IncompatibleInputs,
    #[error("spinor THC q-slice contains a non-finite k, q, or wrap component")]
    NonFiniteQSlice,
    #[error("spinor THC canonical q at index {q_index} is not the complete-slice k-mesh transfer")]
    CanonicalQMismatch { q_index: usize },
    #[error("spinor THC k-minus-q wrap at q-index {q_index} k-index {k_index} is inconsistent")]
    KMinusQWrap { q_index: usize, k_index: usize },
    #[error("spinor THC grid is not bound to the frozen product partition")]
    GridPartitionMismatch,
    #[error("spinor THC grid point {index} is outside the frozen product geometry")]
    GridPoint { index: usize },
    #[error(
        "spinor THC grid point {index} is not on muffin-tin site {site} radial sample {radial_index}"
    )]
    RadialShellMismatch {
        index: usize,
        site: usize,
        radial_index: usize,
    },
}

impl From<ThcGridError> for SpinorThcError {
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

impl From<SpinorQSliceError> for SpinorThcError {
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

/// Build AllQL2 interpolation points, $\zeta$, and pair vertices on a parent grid.
///
/// `inputs` is the complete k-mesh $q$ slice in production $q$-index order.
/// There is one spinor band manifold; Pauli components are summed inside each
/// physical density. Muffin-tin reconstruction uses exact
/// [`CompiledSiteProjection::spinor`] and
/// [`SpinorProductInput::site_projection_identity`]: large $P/\dot P$/LO-RLO
/// with $\Omega_{\kappa\mu}$ and physical small $Q/\dot Q$/LO-RLO with
/// $\Omega_{-\kappa\mu}$. Pair density is the same-Pauli PP plus QQ sum; there
/// is no PQ/QP and no $cQ$. Bloch samples are converted to cell-periodic form
/// by one $\exp(-i k\cdot r)$, then the stored per-$k$ $+G_{\mathrm{wrap}}$
/// pair phase. Global [`muffintin_prodbasis::TransferQ::umklapp`] is not
/// applied again. Interstitial evaluation uses the two Pauli plane-wave blocks
/// with G-only cell-periodic phases. Auxiliaries are created with spinor THC
/// provenance before Bloch pair vertices.
pub fn build_spinor_thc(
    inputs: &[SpinorProductInput],
    grid: &ThcParentGrid,
    spec: &SpinorThcSpec,
) -> Result<SpinorThcResult, SpinorThcError> {
    let first = require_spinor_q_slice(inputs)?;
    if grid.partition() != &first.source.partition {
        return Err(SpinorThcError::GridPartitionMismatch);
    }
    require_parent_grid_radials(grid, |site| {
        first.source.radials.get(site).map(|set| &set.mesh)
    })?;
    let samples = evaluate_orbitals(first, grid)?;
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
            recipe: Some("spinor-thc-allq-l2".to_owned()),
            reference: Some("checkpoint-dft-frozen-spinor-thc".to_owned()),
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
    Ok(SpinorThcResult {
        grid: grid.clone(),
        selection: fitted.selection,
        requested_rank: spec.rank,
        effective_rank: records.first().map(|record| record.fit.n_mu).unwrap_or(0),
        records,
    })
}

#[allow(clippy::needless_range_loop)]
fn evaluate_orbitals(
    input: &SpinorProductInput,
    grid: &ThcParentGrid,
) -> Result<OrbitalGrid, SpinorThcError> {
    let n_k = input.orbitals.k_fractional.len();
    let n_orb = input.orbitals.band_window.count;
    let mut samples =
        vec![vec![vec![SpinorOrbitalSample::default(); n_orb]; n_k]; grid.points().len()];
    let volume = input
        .source
        .partition
        .interstitial()
        .cell_volume()
        .get()
        .sqrt();
    let mut site_proj = Vec::with_capacity(n_k);
    for k in 0..n_k {
        let compiled = &input.orbitals.bases[k];
        let mut per_site = Vec::with_capacity(input.source.partition.site_count());
        for site in 0..input.source.partition.site_count() {
            let channels = site_channels(compiled, site)?;
            let projection = CompiledSiteProjection::spinor(compiled, site, channels)?;
            per_site.push(projection.project_eigenvectors(&input.orbitals.eigenvectors[k])?);
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
                    let k_phase = minus_i_k_dot_r(
                        compiled_cartesian_k(&input.orbitals.bases[k])?,
                        point.coordinate,
                    );
                    let projected = &site_proj[k][site];
                    for band in 0..n_orb {
                        let mut sample = muffin_tin_orbital(
                            input,
                            site,
                            projected,
                            band,
                            radials,
                            radial_index,
                            radius,
                            direction,
                        )?;
                        for component in sample.large.iter_mut().chain(sample.small.iter_mut()) {
                            *component *= k_phase;
                        }
                        samples[p][k][band] = sample;
                    }
                }
            }
            ThcRegion::Interstitial => {
                for k in 0..n_k {
                    for band in 0..n_orb {
                        samples[p][k][band] = SpinorOrbitalSample {
                            large: interstitial_orbital(
                                &input.orbitals.bases[k],
                                &input.orbitals.eigenvectors[k],
                                band,
                                point.coordinate,
                                volume,
                            )?,
                            small: [Complex64::default(); 2],
                        };
                    }
                }
            }
        }
    }
    Ok(samples)
}

#[allow(clippy::too_many_arguments)]
fn muffin_tin_orbital(
    input: &SpinorProductInput,
    site: usize,
    projected: &SiteOrbitalCoefficients,
    band: usize,
    radials: &DiracSiteRadialSet,
    radial_index: usize,
    radius: f64,
    direction: [f64; 3],
) -> Result<SpinorOrbitalSample, SpinorThcError> {
    let inv_r = if radius > 0.0 { 1.0 / radius } else { 0.0 };
    let mut l_max = 0;
    for coord in 0..projected.coordinate_count() {
        let Some((id, _)) = input.site_projection_identity(site, coord) else {
            return Err(SpinorThcError::IncompatibleInputs);
        };
        l_max = l_max.max(id.kappa.large_l()).max(id.kappa.small_l());
    }
    let harmonics = complex_spherical_harmonics(l_max, direction);
    let mut sample = SpinorOrbitalSample::default();
    for coord in 0..projected.coordinate_count() {
        let Some((id, twice_mu)) = input.site_projection_identity(site, coord) else {
            return Err(SpinorThcError::IncompatibleInputs);
        };
        let radial = find_dirac_radial(radials, id).ok_or(SpinorThcError::IncompatibleInputs)?;
        let channel = RelativisticChannel::new(id.kappa, twice_mu)
            .map_err(|_| SpinorThcError::IncompatibleInputs)?;
        let large_omega = pauli_omega(channel, &harmonics)?;
        let small_omega = pauli_omega(channel.opposite_kappa(), &harmonics)?;
        let amplitude = projected.at(coord, band) * inv_r;
        let p = amplitude * radial.samples.large[radial_index];
        let q = amplitude * radial.samples.small[radial_index];
        for ((large, small), (w_large, w_small)) in sample
            .large
            .iter_mut()
            .zip(sample.small.iter_mut())
            .zip(large_omega.iter().zip(small_omega.iter()))
        {
            *large += p * w_large;
            *small += q * w_small;
        }
    }
    Ok(sample)
}

fn pauli_omega(
    channel: RelativisticChannel,
    harmonics: &[Complex64],
) -> Result<[Complex64; 2], SpinorThcError> {
    let mut pauli = [Complex64::default(); 2];
    for term in channel.spinor_harmonic_terms().into_iter().flatten() {
        let y = harmonics[lm_index(term.orbital.l, term.orbital.m).map_err(ThcError::from)?];
        let spin = match term.spin {
            SpinProjection::Up => 0,
            SpinProjection::Down => 1,
        };
        pauli[spin] += Complex64::from(term.coefficient) * y;
    }
    Ok(pauli)
}

fn find_dirac_radial(radials: &DiracSiteRadialSet, id: DiracRadialId) -> Option<&DiracRadial> {
    let pool = match id.kind {
        ProductOrbitalKind::Valence => radials.valence.as_slice(),
        ProductOrbitalKind::Core => radials.cores.as_slice(),
    };
    pool.iter()
        .find(|radial| radial.kappa == id.kappa && radial.n == id.n)
}

fn interstitial_orbital(
    compiled: &SpinorCompiledBasis,
    eigenvectors: &DenseEigenvectors,
    band: usize,
    coordinate: [Bohr; 3],
    sqrt_volume: f64,
) -> Result<[Complex64; 2], SpinorThcError> {
    let mut pauli = [Complex64::default(); 2];
    for (g, wave) in compiled.plane_waves.iter().enumerate() {
        let phase = plus_i_g_dot_r(wave.g, coordinate);
        for (spin, value) in pauli.iter_mut().enumerate() {
            let row = compiled
                .layout
                .plane_wave_index(spin, g)
                .ok_or(SpinorThcError::IncompatibleInputs)?;
            *value += eigenvectors.at(row, band) * phase / sqrt_volume;
        }
    }
    Ok(pauli)
}

fn site_channels(
    compiled: &SpinorCompiledBasis,
    site: usize,
) -> Result<&[RelativisticChannel], SpinorThcError> {
    compiled
        .site_augmentations
        .get(site)
        .and_then(|waves| waves.first())
        .map(|wave| wave.channels.as_slice())
        .ok_or(SpinorThcError::IncompatibleInputs)
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

fn compiled_cartesian_k(
    compiled: &SpinorCompiledBasis,
) -> Result<[InverseBohr; 3], SpinorThcError> {
    compiled
        .plane_waves
        .first()
        .map(|wave| wave.k)
        .ok_or(SpinorThcError::IncompatibleInputs)
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
    inputs: &[SpinorProductInput],
    grid: &ThcParentGrid,
    samples: &[Vec<Vec<SpinorOrbitalSample>>],
) -> Result<Vec<PairBlock>, SpinorThcError> {
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
                    let left = samples[p][mapped.kq_index][left_band];
                    for right_band in 0..n_orb {
                        let right = samples[p][mapped.k_index][right_band];
                        let density = left
                            .large
                            .iter()
                            .zip(right.large)
                            .zip(left.small.iter().zip(right.small))
                            .map(|((l_large, r_large), (l_small, r_small))| {
                                l_large.conj() * r_large + l_small.conj() * r_small
                            })
                            .sum::<Complex64>();
                        let column = layout.encode(mapped.k_index, left_band, right_band);
                        values[p * n_col + column] = phase * density;
                    }
                }
            }
        }
        blocks.push(PairBlock::new(iq, n_points, layout, values)?);
    }
    Ok(blocks)
}
