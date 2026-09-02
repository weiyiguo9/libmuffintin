//! Shared-selector spinor THC over exact VV, CV, VC, and CC exchange sectors.

use crate::scalar_coulomb::quadratic_discrepancy;
use crate::spinor_coulomb::{SPINOR_COULOMB_EXACTNESS_FLOOR, SpinorCoulombSpec};
use crate::spinor_exchange_mpb::{SpinorExchangeMpbResult, SpinorExchangeMpbSector};
use crate::spinor_mpb::SpinorMpbResult;
use crate::spinor_product::{SpinorProductInput, SpinorQSliceError, require_spinor_q_slice};
use crate::spinor_thc::SpinorThcSpec;
use crate::thc_grid::{
    ParentGridIdentity, ThcGridError, ThcParentGrid, ThcRegion, require_parent_grid_radials,
};
use muffintin_core::{
    Bohr, ExponentialMesh, GVector, InverseBohr, RelativisticChannel, SpinProjection, VolumeBohr3,
    complex_spherical_harmonics, lm_index,
};
use muffintin_coulomb::{
    CoulombError, SampledAuxiliaryFunctions, SampledPointSupport, assemble_coulomb,
    assemble_sampled_coulomb,
};
use muffintin_operators::lapw::{Provenance, SpinorCompiledBasis};
use muffintin_operators::{CompiledSiteProjection, OperatorError, SiteOrbitalCoefficients};
use muffintin_prodbasis::thc::{
    ExchangePairBlock, RankPolicy, ThcError, WeightedResidual, fit_allq_l2_exchange_pair_blocks,
    reconstruct_pairs,
};
use muffintin_prodbasis::{
    CompiledAuxiliaryBasis, DiracRadial, DiracRadialId, DiracSiteRadialSet, ExchangePairLayout,
    ExchangeSpace, OrbitalPair, PairVertex, ProductOrbitalKind, TransferQ,
};
use muffintin_tensor::DenseEigenvectors;
use num_complex::Complex64;
use thiserror::Error;

#[derive(Clone, Copy, Default)]
struct SpinorOrbitalSample {
    large: [Complex64; 2],
    small: [Complex64; 2],
}

type OrbitalGrid = Vec<Vec<Vec<SpinorOrbitalSample>>>;

/// Pair vertices for one exact rectangular exchange sector at one transfer.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorSectorThcVertices {
    pub layout: ExchangePairLayout,
    pub vertices: Vec<PairVertex>,
}

/// Separate weighted L2 residuals of the four exact exchange sectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinorSectorThcDiagnostics {
    pub vv: WeightedResidual,
    pub cv: WeightedResidual,
    pub vc: WeightedResidual,
    pub cc: WeightedResidual,
}

/// Explicit column and selector-row accounting for one shared four-sector fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpinorSectorThcRankScaling {
    pub n_k: usize,
    pub n_valence: usize,
    pub n_core: usize,
    pub n_candidates: usize,
    pub effective_rank: usize,
    pub vv_columns_per_q: usize,
    pub cv_columns_per_q: usize,
    pub vc_columns_per_q: usize,
    pub cc_columns_per_q: usize,
    pub pooled_columns_per_q: usize,
    pub selector_rows: usize,
}

/// One shared-zeta four-sector THC record at a canonical transfer.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorSectorThcQRecord {
    pub q_index: usize,
    pub q: TransferQ,
    pub auxiliary: CompiledAuxiliaryBasis,
    /// Zeta on the full parent grid, row-major `n_points * n_mu`.
    pub zeta: Vec<Complex64>,
    pub n_points: usize,
    pub n_mu: usize,
    pub residuals: SpinorSectorThcDiagnostics,
    pub vv: SpinorSectorThcVertices,
    pub cv: SpinorSectorThcVertices,
    pub vc: SpinorSectorThcVertices,
    pub cc: SpinorSectorThcVertices,
    grid_identity: ParentGridIdentity,
}

/// Core-aware spinor AllQL2 result with one selector and one zeta per q.
#[derive(Clone, Debug)]
pub struct SpinorSectorThcResult {
    pub grid: ThcParentGrid,
    pub selection: muffintin_prodbasis::thc::ExchangeSelection,
    pub requested_rank: RankPolicy,
    pub effective_rank: usize,
    /// Four-sector residuals aggregated over all q, points, and exact columns.
    pub diagnostics: SpinorSectorThcDiagnostics,
    pub rank_scaling: SpinorSectorThcRankScaling,
    pub records: Vec<SpinorSectorThcQRecord>,
    sealed_inputs: Vec<SpinorProductInput>,
}

impl PartialEq for SpinorSectorThcResult {
    fn eq(&self, other: &Self) -> bool {
        self.grid == other.grid
            && self.selection == other.selection
            && self.requested_rank == other.requested_rank
            && self.effective_rank == other.effective_rank
            && self.diagnostics == other.diagnostics
            && self.rank_scaling == other.rank_scaling
            && self.records == other.records
    }
}

impl SpinorSectorThcResult {
    /// Whether every per-q zeta record remains bound to the stored parent grid.
    pub fn records_match_parent_grid(&self) -> bool {
        self.records
            .iter()
            .all(|record| record.grid_identity == self.grid.identity())
    }

    /// Whether this fit was constructed from exactly this frozen orbital/core frame.
    pub fn frozen_context_matches(&self, inputs: &[SpinorProductInput]) -> bool {
        self.sealed_inputs == inputs
    }
}

/// One representation-neutral MPB-versus-THC quadratic comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorSectorThcMpbPairDiagnostic {
    pub q_index: usize,
    pub column: usize,
    pub pair: OrbitalPair,
    pub mpb_quadratic: Complex64,
    pub thc_quadratic: Complex64,
    pub absolute: f64,
    pub relative: f64,
}

/// Complete per-sector quadratic diagnostics and maxima.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorSectorThcMpbSectorComparison {
    pub pairs: Vec<SpinorSectorThcMpbPairDiagnostic>,
    pub maximum_absolute: f64,
    pub maximum_relative: f64,
    pub worst_absolute_q_index: usize,
    pub worst_absolute_column: usize,
    pub worst_relative_q_index: usize,
    pub worst_relative_column: usize,
}

/// Full four-sector MPB oracle comparison.
#[derive(Clone, Debug)]
pub struct SpinorSectorThcMpbComparison {
    pub vv: SpinorSectorThcMpbSectorComparison,
    pub cv: SpinorSectorThcMpbSectorComparison,
    pub vc: SpinorSectorThcMpbSectorComparison,
    pub cc: SpinorSectorThcMpbSectorComparison,
    sealed_inputs: Vec<SpinorProductInput>,
}

impl PartialEq for SpinorSectorThcMpbComparison {
    fn eq(&self, other: &Self) -> bool {
        self.vv == other.vv && self.cv == other.cv && self.vc == other.vc && self.cc == other.cc
    }
}

impl SpinorSectorThcMpbComparison {
    /// Whether this MPB comparison was evaluated on exactly this frozen frame.
    pub fn frozen_context_matches(&self, inputs: &[SpinorProductInput]) -> bool {
        self.sealed_inputs == inputs
    }
}

/// Core-aware spinor THC construction and MPB-comparison failure.
#[derive(Debug, Error)]
pub enum SpinorSectorThcError {
    #[error(transparent)]
    Thc(#[from] ThcError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Coulomb(#[from] CoulombError),
    #[error("spinor sector THC q-slice must be nonempty")]
    EmptySlice,
    #[error("spinor sector THC q-slice has {actual} bundles, expected {expected} k-mesh transfers")]
    IncompleteQSlice { actual: usize, expected: usize },
    #[error("spinor sector THC inputs do not share one frozen orbital/core frame")]
    IncompatibleInputs,
    #[error("spinor sector THC q-slice contains a non-finite k, q, or wrap component")]
    NonFiniteQSlice,
    #[error("spinor sector THC canonical q at index {q_index} is not the complete-slice transfer")]
    CanonicalQMismatch { q_index: usize },
    #[error(
        "spinor sector THC k-minus-q wrap at q-index {q_index} k-index {k_index} is inconsistent"
    )]
    KMinusQWrap { q_index: usize, k_index: usize },
    #[error("spinor sector THC requires at least one attached core spin orbital")]
    EmptyCore,
    #[error("spinor sector THC grid is not bound to the frozen product partition")]
    GridPartitionMismatch,
    #[error("spinor sector THC natural grid reciprocal lattice does not match the frozen input")]
    GridReciprocalMismatch,
    #[error("spinor sector THC grid point {index} is outside the frozen product geometry")]
    GridPoint { index: usize },
    #[error(
        "spinor sector THC grid point {index} is not on muffin-tin site {site} radial sample {radial_index}"
    )]
    RadialShellMismatch {
        index: usize,
        site: usize,
        radial_index: usize,
    },
    #[error("spinor sector THC result has {actual} q-records, expected {expected}")]
    RecordCount { actual: usize, expected: usize },
    #[error("spinor sector THC Coulomb request does not match the frozen reciprocal lattice")]
    CoulombReciprocalMismatch,
    #[error("spinor sector THC interpolation projection does not match the Coulomb request")]
    InterpolationProjection,
    #[error(
        "spinor sector THC {kind} MPB record at q-index {q_index} does not match the frozen input"
    )]
    MpbContext { kind: &'static str, q_index: usize },
    #[error(
        "spinor sector THC MPB sector {occupied_space:?}->{target_space:?} at q-index {q_index} is not a complete exact column set"
    )]
    MpbCoverage {
        q_index: usize,
        occupied_space: ExchangeSpace,
        target_space: ExchangeSpace,
    },
    #[error(
        "spinor sector THC record {q_index} has inconsistent q, layout, auxiliary, or vertices"
    )]
    ThcRecord { q_index: usize },
}

impl From<ThcGridError> for SpinorSectorThcError {
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

impl From<SpinorQSliceError> for SpinorSectorThcError {
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

/// Fit one shared interpolation-point selector and one zeta per q over VV/CV/VC/CC.
///
/// Valence sampling is identical to [`crate::build_spinor_thc`]. An attached core
/// spin orbital is sampled only inside its owning muffin tin as physical P/Q with
/// `Omega(kappa,mu)` and `Omega(-kappa,mu)` and the cell-periodic
/// `exp(-i k*x)` phase. It is identically zero on other sites and in the
/// interstitial. Every sector then applies the stored per-k `+G_wrap` phase.
/// Core occupations do not enter sampling, selection, zeta, or pair vertices.
pub fn build_spinor_sector_thc(
    inputs: &[SpinorProductInput],
    grid: &ThcParentGrid,
    spec: &SpinorThcSpec,
) -> Result<SpinorSectorThcResult, SpinorSectorThcError> {
    let first = require_spinor_q_slice(inputs)?;
    if first.core.orbitals.is_empty() {
        return Err(SpinorSectorThcError::EmptyCore);
    }
    if !grid.natural_reciprocal_matches(&first.reciprocal) {
        return Err(SpinorSectorThcError::GridReciprocalMismatch);
    }
    if grid.partition() != &first.source.partition {
        return Err(SpinorSectorThcError::GridPartitionMismatch);
    }
    require_parent_grid_radials(grid, |site| {
        first.source.radials.get(site).map(|set| &set.mesh)
    })?;

    let valence = evaluate_valence_orbitals(first, grid)?;
    let core = evaluate_core_orbitals(first, grid)?;
    let blocks = exchange_pair_blocks(inputs, grid, &valence, &core)?;
    let transfers = inputs
        .iter()
        .map(|input| input.source.q)
        .collect::<Vec<_>>();
    let fitted = fit_allq_l2_exchange_pair_blocks(
        &blocks,
        &grid.cartesian(),
        &grid.weights(),
        &grid.interpolation_regions(),
        first.source.partition.clone(),
        &transfers,
        spec.rank,
        spec.engine.into(),
        spec.candidates.as_fit_indices(),
        Provenance {
            recipe: Some("spinor-sector-thc-allq-l2".to_owned()),
            reference: Some("checkpoint-dft-frozen-spinor-sector-thc".to_owned()),
        },
    )?;

    let selected = fitted
        .selection
        .points
        .iter()
        .map(|point| point.id)
        .collect::<Vec<_>>();
    let diagnostics = aggregate_residuals(&blocks, &fitted.fits, &selected, &grid.weights())?;
    let effective_rank = fitted.fits.first().map(|fit| fit.n_mu).unwrap_or(0);
    let fitted_rank = fitted.rank_scaling;
    let rank_scaling = SpinorSectorThcRankScaling {
        n_k: fitted_rank.n_k,
        n_valence: fitted_rank.n_valence,
        n_core: fitted_rank.n_core,
        n_candidates: fitted_rank.n_candidates,
        effective_rank: fitted_rank.effective_rank,
        vv_columns_per_q: fitted_rank.vv_columns,
        cv_columns_per_q: fitted_rank.cv_columns,
        vc_columns_per_q: fitted_rank.vc_columns,
        cc_columns_per_q: fitted_rank.cc_columns,
        pooled_columns_per_q: fitted_rank.pooled_columns_per_q,
        selector_rows: fitted_rank.selector_rows,
    };

    let records = fitted
        .fits
        .into_iter()
        .zip(fitted.auxiliaries)
        .zip(fitted.vertices)
        .zip(transfers)
        .map(|(((fit, auxiliary), vertices), q)| SpinorSectorThcQRecord {
            q_index: fit.q_index,
            q,
            auxiliary,
            zeta: fit.zeta,
            n_points: fit.n_points,
            n_mu: fit.n_mu,
            residuals: SpinorSectorThcDiagnostics {
                vv: fit.residuals.vv,
                cv: fit.residuals.cv,
                vc: fit.residuals.vc,
                cc: fit.residuals.cc,
            },
            vv: SpinorSectorThcVertices {
                layout: vertices.vv.layout,
                vertices: vertices.vv.vertices,
            },
            cv: SpinorSectorThcVertices {
                layout: vertices.cv.layout,
                vertices: vertices.cv.vertices,
            },
            vc: SpinorSectorThcVertices {
                layout: vertices.vc.layout,
                vertices: vertices.vc.vertices,
            },
            cc: SpinorSectorThcVertices {
                layout: vertices.cc.layout,
                vertices: vertices.cc.vertices,
            },
            grid_identity: grid.identity(),
        })
        .collect();
    Ok(SpinorSectorThcResult {
        grid: grid.clone(),
        selection: fitted.selection,
        requested_rank: spec.rank,
        effective_rank,
        diagnostics,
        rank_scaling,
        records,
        sealed_inputs: inputs.to_vec(),
    })
}

fn exchange_layouts(n_k: usize, n_valence: usize, n_core: usize) -> [ExchangePairLayout; 4] {
    [
        ExchangePairLayout::new(
            ExchangeSpace::Valence,
            ExchangeSpace::Valence,
            n_k,
            n_valence,
            n_valence,
        ),
        ExchangePairLayout::new(
            ExchangeSpace::Core,
            ExchangeSpace::Valence,
            n_k,
            n_core,
            n_valence,
        ),
        ExchangePairLayout::new(
            ExchangeSpace::Valence,
            ExchangeSpace::Core,
            n_k,
            n_valence,
            n_core,
        ),
        ExchangePairLayout::new(
            ExchangeSpace::Core,
            ExchangeSpace::Core,
            n_k,
            n_core,
            n_core,
        ),
    ]
}

#[allow(clippy::needless_range_loop)]
fn evaluate_valence_orbitals(
    input: &SpinorProductInput,
    grid: &ThcParentGrid,
) -> Result<OrbitalGrid, SpinorSectorThcError> {
    let n_k = input.orbitals.k_fractional.len();
    let n_orb = input.orbitals.band_window.count;
    let mut samples =
        vec![vec![vec![SpinorOrbitalSample::default(); n_orb]; n_k]; grid.points().len()];
    let sqrt_volume = input
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
    for (point_index, point) in grid.points().iter().enumerate() {
        match point.region {
            ThcRegion::MuffinTin { site, radial_index } => {
                let radials = &input.source.radials[site];
                let origin = input.source.partition.sites()[site].position;
                let displacement = subtract(point.coordinate, origin);
                let radius = radials.mesh.radii()[radial_index].get();
                for k in 0..n_k {
                    let phase = minus_i_dot(
                        compiled_cartesian_k(&input.orbitals.bases[k])?,
                        point.coordinate,
                    );
                    for band in 0..n_orb {
                        let mut sample = muffin_tin_valence_orbital(
                            input,
                            site,
                            &site_proj[k][site],
                            band,
                            radials,
                            radial_index,
                            radius,
                            displacement.map(Bohr::get),
                        )?;
                        scale_sample(&mut sample, phase);
                        samples[point_index][k][band] = sample;
                    }
                }
            }
            ThcRegion::Interstitial => {
                for k in 0..n_k {
                    for band in 0..n_orb {
                        samples[point_index][k][band] = SpinorOrbitalSample {
                            large: interstitial_valence_orbital(
                                &input.orbitals.bases[k],
                                &input.orbitals.eigenvectors[k],
                                band,
                                point.coordinate,
                                sqrt_volume,
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

#[allow(clippy::needless_range_loop)]
fn evaluate_core_orbitals(
    input: &SpinorProductInput,
    grid: &ThcParentGrid,
) -> Result<OrbitalGrid, SpinorSectorThcError> {
    let n_k = input.orbitals.k_fractional.len();
    let n_core = input.core.orbitals.len();
    let mut samples =
        vec![vec![vec![SpinorOrbitalSample::default(); n_core]; n_k]; grid.points().len()];
    for (point_index, point) in grid.points().iter().enumerate() {
        let ThcRegion::MuffinTin { site, radial_index } = point.region else {
            continue;
        };
        let radials = &input.source.radials[site];
        let origin = input.source.partition.sites()[site].position;
        let displacement = subtract(point.coordinate, origin);
        let radius = radials.mesh.radii()[radial_index].get();
        let inv_r = if radius > 0.0 { 1.0 / radius } else { 0.0 };
        for (core_index, core) in input.core.orbitals.iter().enumerate() {
            if core.site_index != site {
                continue;
            }
            let radial = input
                .source
                .find_radial(core.radial)
                .ok_or(SpinorSectorThcError::IncompatibleInputs)?;
            let channel = RelativisticChannel::new(core.kappa, core.twice_mu)
                .map_err(|_| SpinorSectorThcError::IncompatibleInputs)?;
            let l_max = core.kappa.large_l().max(core.kappa.small_l());
            let harmonics = complex_spherical_harmonics(l_max, displacement.map(Bohr::get));
            let large_omega = pauli_omega(channel, &harmonics)?;
            let small_omega = pauli_omega(channel.opposite_kappa(), &harmonics)?;
            for k in 0..n_k {
                let phase = minus_i_dot(
                    compiled_cartesian_k(&input.orbitals.bases[k])?,
                    displacement,
                );
                let p = radial.samples.large[radial_index] * inv_r * phase;
                let q = radial.samples.small[radial_index] * inv_r * phase;
                samples[point_index][k][core_index] = SpinorOrbitalSample {
                    large: large_omega.map(|omega| p * omega),
                    small: small_omega.map(|omega| q * omega),
                };
            }
        }
    }
    Ok(samples)
}

#[allow(clippy::too_many_arguments)]
fn muffin_tin_valence_orbital(
    input: &SpinorProductInput,
    site: usize,
    projected: &SiteOrbitalCoefficients,
    band: usize,
    radials: &DiracSiteRadialSet,
    radial_index: usize,
    radius: f64,
    direction: [f64; 3],
) -> Result<SpinorOrbitalSample, SpinorSectorThcError> {
    let inv_r = if radius > 0.0 { 1.0 / radius } else { 0.0 };
    let mut l_max = 0;
    for coordinate in 0..projected.coordinate_count() {
        let (id, _) = input
            .site_projection_identity(site, coordinate)
            .ok_or(SpinorSectorThcError::IncompatibleInputs)?;
        l_max = l_max.max(id.kappa.large_l()).max(id.kappa.small_l());
    }
    let harmonics = complex_spherical_harmonics(l_max, direction);
    let mut sample = SpinorOrbitalSample::default();
    for coordinate in 0..projected.coordinate_count() {
        let (id, twice_mu) = input
            .site_projection_identity(site, coordinate)
            .ok_or(SpinorSectorThcError::IncompatibleInputs)?;
        let radial =
            find_dirac_radial(radials, id).ok_or(SpinorSectorThcError::IncompatibleInputs)?;
        let channel = RelativisticChannel::new(id.kappa, twice_mu)
            .map_err(|_| SpinorSectorThcError::IncompatibleInputs)?;
        let large_omega = pauli_omega(channel, &harmonics)?;
        let small_omega = pauli_omega(channel.opposite_kappa(), &harmonics)?;
        let amplitude = projected.at(coordinate, band) * inv_r;
        let p = amplitude * radial.samples.large[radial_index];
        let q = amplitude * radial.samples.small[radial_index];
        for spin in 0..2 {
            sample.large[spin] += p * large_omega[spin];
            sample.small[spin] += q * small_omega[spin];
        }
    }
    Ok(sample)
}

fn pauli_omega(
    channel: RelativisticChannel,
    harmonics: &[Complex64],
) -> Result<[Complex64; 2], SpinorSectorThcError> {
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

fn interstitial_valence_orbital(
    compiled: &SpinorCompiledBasis,
    eigenvectors: &DenseEigenvectors,
    band: usize,
    coordinate: [Bohr; 3],
    sqrt_volume: f64,
) -> Result<[Complex64; 2], SpinorSectorThcError> {
    let mut pauli = [Complex64::default(); 2];
    for (g, wave) in compiled.plane_waves.iter().enumerate() {
        let phase = plus_i_g_dot_r(wave.g, coordinate);
        for (spin, value) in pauli.iter_mut().enumerate() {
            let row = compiled
                .layout
                .plane_wave_index(spin, g)
                .ok_or(SpinorSectorThcError::IncompatibleInputs)?;
            *value += eigenvectors.at(row, band) * phase / sqrt_volume;
        }
    }
    Ok(pauli)
}

fn site_channels(
    compiled: &SpinorCompiledBasis,
    site: usize,
) -> Result<&[RelativisticChannel], SpinorSectorThcError> {
    compiled
        .site_augmentations
        .get(site)
        .and_then(|waves| waves.first())
        .map(|wave| wave.channels.as_slice())
        .ok_or(SpinorSectorThcError::IncompatibleInputs)
}

fn compiled_cartesian_k(
    compiled: &SpinorCompiledBasis,
) -> Result<[InverseBohr; 3], SpinorSectorThcError> {
    compiled
        .plane_waves
        .first()
        .map(|wave| wave.k)
        .ok_or(SpinorSectorThcError::IncompatibleInputs)
}

fn subtract(left: [Bohr; 3], right: [Bohr; 3]) -> [Bohr; 3] {
    std::array::from_fn(|axis| Bohr(left[axis].get() - right[axis].get()))
}

fn scale_sample(sample: &mut SpinorOrbitalSample, scale: Complex64) {
    for component in sample.large.iter_mut().chain(sample.small.iter_mut()) {
        *component *= scale;
    }
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

fn minus_i_dot(k: [InverseBohr; 3], coordinate: [Bohr; 3]) -> Complex64 {
    let argument: f64 = k
        .iter()
        .zip(coordinate)
        .map(|(component, point)| component.get() * point.get())
        .sum();
    Complex64::from_polar(1.0, -argument)
}

fn pair_density(left: SpinorOrbitalSample, right: SpinorOrbitalSample) -> Complex64 {
    left.large
        .iter()
        .zip(right.large)
        .zip(left.small.iter().zip(right.small))
        .map(|((left_p, right_p), (left_q, right_q))| {
            left_p.conj() * right_p + left_q.conj() * right_q
        })
        .sum()
}

#[allow(clippy::needless_range_loop)]
fn exchange_pair_blocks(
    inputs: &[SpinorProductInput],
    grid: &ThcParentGrid,
    valence: &OrbitalGrid,
    core: &OrbitalGrid,
) -> Result<Vec<ExchangePairBlock>, SpinorSectorThcError> {
    let n_k = inputs[0].orbitals.k_fractional.len();
    let n_valence = inputs[0].orbitals.band_window.count;
    let n_core = inputs[0].core.orbitals.len();
    let layouts = exchange_layouts(n_k, n_valence, n_core);
    let n_points = grid.points().len();
    let mut blocks = Vec::with_capacity(4 * inputs.len());
    for (q_index, input) in inputs.iter().enumerate() {
        let mut values = layouts
            .iter()
            .map(|layout| {
                let n_columns = layout.n_columns().map_err(ThcError::from)?;
                Ok(vec![Complex64::default(); n_points * n_columns])
            })
            .collect::<Result<Vec<_>, ThcError>>()?;
        for mapped in &input.k_minus_q {
            for (point_index, point) in grid.points().iter().enumerate() {
                let wrap = plus_i_g_dot_r(mapped.umklapp, point.coordinate);
                fill_sector(
                    &mut values[0],
                    layouts[0],
                    point_index,
                    mapped.k_index,
                    &valence[point_index][mapped.kq_index],
                    &valence[point_index][mapped.k_index],
                    wrap,
                )?;
                fill_sector(
                    &mut values[1],
                    layouts[1],
                    point_index,
                    mapped.k_index,
                    &core[point_index][mapped.kq_index],
                    &valence[point_index][mapped.k_index],
                    wrap,
                )?;
                fill_sector(
                    &mut values[2],
                    layouts[2],
                    point_index,
                    mapped.k_index,
                    &valence[point_index][mapped.kq_index],
                    &core[point_index][mapped.k_index],
                    wrap,
                )?;
                fill_sector(
                    &mut values[3],
                    layouts[3],
                    point_index,
                    mapped.k_index,
                    &core[point_index][mapped.kq_index],
                    &core[point_index][mapped.k_index],
                    wrap,
                )?;
            }
        }
        for (layout, values) in layouts.into_iter().zip(values) {
            blocks.push(ExchangePairBlock::new(q_index, n_points, layout, values)?);
        }
    }
    Ok(blocks)
}

fn fill_sector(
    values: &mut [Complex64],
    layout: ExchangePairLayout,
    point: usize,
    k: usize,
    occupied: &[SpinorOrbitalSample],
    target: &[SpinorOrbitalSample],
    wrap: Complex64,
) -> Result<(), SpinorSectorThcError> {
    let n_columns = layout.n_columns().map_err(ThcError::from)?;
    for occupied_index in 0..layout.n_occupied {
        for target_index in 0..layout.n_target {
            let column = layout
                .encode(k, occupied_index, target_index)
                .map_err(ThcError::from)?;
            values[point * n_columns + column] =
                wrap * pair_density(occupied[occupied_index], target[target_index]);
        }
    }
    Ok(())
}

fn aggregate_residuals(
    blocks: &[ExchangePairBlock],
    fits: &[muffintin_prodbasis::thc::ExchangePerQFit],
    selected: &[usize],
    weights: &[f64],
) -> Result<SpinorSectorThcDiagnostics, SpinorSectorThcError> {
    let residual = |sector: usize| -> Result<WeightedResidual, SpinorSectorThcError> {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        let mut column_max = 0.0_f64;
        for (q_index, fit) in fits.iter().enumerate() {
            let block = &blocks[4 * q_index + sector];
            let selected_rows = block.selected_rows(selected)?;
            let reconstructed = reconstruct_pairs(
                &selected_rows,
                fit.n_mu,
                block.n_columns(),
                &fit.zeta,
                block.n_points,
            );
            let mut column_numerator = vec![0.0; block.n_columns()];
            let mut column_denominator = vec![0.0; block.n_columns()];
            for point in 0..block.n_points {
                for column in 0..block.n_columns() {
                    let exact = block.at(point, column);
                    let difference = exact - reconstructed[point * block.n_columns() + column];
                    let weighted_difference = weights[point] * difference.norm_sqr();
                    let weighted_exact = weights[point] * exact.norm_sqr();
                    numerator += weighted_difference;
                    denominator += weighted_exact;
                    column_numerator[column] += weighted_difference;
                    column_denominator[column] += weighted_exact;
                }
            }
            let scale = column_denominator.iter().copied().fold(0.0_f64, f64::max);
            let floor = f64::EPSILON * scale.max(1.0);
            for column in 0..block.n_columns() {
                if column_denominator[column] > floor {
                    column_max = column_max
                        .max((column_numerator[column] / column_denominator[column]).sqrt());
                }
            }
        }
        Ok(WeightedResidual {
            frobenius: if denominator > 0.0 {
                (numerator / denominator).sqrt()
            } else {
                0.0
            },
            column_max,
        })
    };
    Ok(SpinorSectorThcDiagnostics {
        vv: residual(0)?,
        cv: residual(1)?,
        vc: residual(2)?,
        cc: residual(3)?,
    })
}

/// Compare every exact VV/CV/VC/CC pair through its representation-neutral
/// Coulomb quadratic form.
///
/// `vv_mpb[q]` must contain every VV column exactly once and `core_mpb[q]`
/// must contain the complete CV, VC, and CC sectors. No coefficient norm or
/// action norm is compared across the two auxiliary representations.
pub fn compare_spinor_sector_thc_mpb(
    inputs: &[SpinorProductInput],
    thc: &SpinorSectorThcResult,
    vv_mpb: &[SpinorMpbResult],
    core_mpb: &[SpinorExchangeMpbResult],
    spec: &SpinorCoulombSpec,
) -> Result<SpinorSectorThcMpbComparison, SpinorSectorThcError> {
    let first = require_spinor_q_slice(inputs)?;
    if first.core.orbitals.is_empty() {
        return Err(SpinorSectorThcError::EmptyCore);
    }
    for actual in [thc.records.len(), vv_mpb.len(), core_mpb.len()] {
        if actual != inputs.len() {
            return Err(SpinorSectorThcError::RecordCount {
                actual,
                expected: inputs.len(),
            });
        }
    }
    if spec.request.reciprocal() != &first.reciprocal {
        return Err(SpinorSectorThcError::CoulombReciprocalMismatch);
    }
    if thc.grid.partition() != &first.source.partition {
        return Err(SpinorSectorThcError::GridPartitionMismatch);
    }
    if !thc.records_match_parent_grid() {
        return Err(SpinorSectorThcError::ThcRecord { q_index: 0 });
    }
    if !thc.frozen_context_matches(inputs) {
        return Err(SpinorSectorThcError::IncompatibleInputs);
    }
    let sampled_request = match spec.request.interpolation() {
        None => spec.request.clone().with_interpolation(spec.projection)?,
        Some(existing) if existing == spec.projection => spec.request.clone(),
        Some(_) => return Err(SpinorSectorThcError::InterpolationProjection),
    };
    let site_meshes = first
        .source
        .radials
        .iter()
        .map(|radials| radials.mesh.clone())
        .collect::<Vec<ExponentialMesh>>();
    let layouts = exchange_layouts(
        first.orbitals.k_fractional.len(),
        first.orbitals.band_window.count,
        first.core.orbitals.len(),
    );
    let mut vv_pairs = Vec::new();
    let mut cv_pairs = Vec::new();
    let mut vc_pairs = Vec::new();
    let mut cc_pairs = Vec::new();
    for q_index in 0..inputs.len() {
        let input = &inputs[q_index];
        let record = &thc.records[q_index];
        require_thc_record(record, input, q_index, layouts)?;
        if !vv_mpb[q_index].frozen_input_identity().matches(input)
            || vv_mpb[q_index].auxiliary.q != input.source.q
            || vv_mpb[q_index].auxiliary.partition != input.source.partition
        {
            return Err(SpinorSectorThcError::MpbContext {
                kind: "VV",
                q_index,
            });
        }
        if !core_mpb[q_index].frozen_input_identity().matches(input)
            || core_mpb[q_index].auxiliary.q != input.source.q
            || core_mpb[q_index].auxiliary.partition != input.source.partition
        {
            return Err(SpinorSectorThcError::MpbContext {
                kind: "core-member",
                q_index,
            });
        }
        let vv_exact = complete_vv_mpb_vertices(&vv_mpb[q_index], layouts[0], q_index)?;
        let cv_exact = complete_exchange_mpb_vertices(&core_mpb[q_index].cv, q_index)?;
        let vc_exact = complete_exchange_mpb_vertices(&core_mpb[q_index].vc, q_index)?;
        let cc_exact = complete_exchange_mpb_vertices(&core_mpb[q_index].cc, q_index)?;

        let sampled = sampled_sector_auxiliary(record, &thc.grid, site_meshes.clone())?;
        let thc_operator = assemble_sampled_coulomb(&record.auxiliary, &sampled_request, &sampled)?;
        let vv_operator = assemble_coulomb(&vv_mpb[q_index].auxiliary, &spec.request)?;
        let core_operator = assemble_coulomb(&core_mpb[q_index].auxiliary, &spec.request)?;
        compare_sector_pairs(
            q_index,
            &record.vv,
            &vv_exact,
            &thc_operator,
            &vv_operator,
            &mut vv_pairs,
        )?;
        compare_sector_pairs(
            q_index,
            &record.cv,
            &cv_exact,
            &thc_operator,
            &core_operator,
            &mut cv_pairs,
        )?;
        compare_sector_pairs(
            q_index,
            &record.vc,
            &vc_exact,
            &thc_operator,
            &core_operator,
            &mut vc_pairs,
        )?;
        compare_sector_pairs(
            q_index,
            &record.cc,
            &cc_exact,
            &thc_operator,
            &core_operator,
            &mut cc_pairs,
        )?;
    }
    Ok(SpinorSectorThcMpbComparison {
        vv: summarize_mpb_sector(vv_pairs, layouts[0])?,
        cv: summarize_mpb_sector(cv_pairs, layouts[1])?,
        vc: summarize_mpb_sector(vc_pairs, layouts[2])?,
        cc: summarize_mpb_sector(cc_pairs, layouts[3])?,
        sealed_inputs: inputs.to_vec(),
    })
}

fn require_thc_record(
    record: &SpinorSectorThcQRecord,
    input: &SpinorProductInput,
    q_index: usize,
    layouts: [ExchangePairLayout; 4],
) -> Result<(), SpinorSectorThcError> {
    let sectors = [&record.vv, &record.cv, &record.vc, &record.cc];
    let base_ok = record.q_index == q_index
        && record.q == input.source.q
        && record.auxiliary.q == record.q
        && record.auxiliary.partition == input.source.partition
        && record.n_mu == record.auxiliary.dimension()
        && record.n_points > 0
        && record.zeta.len() == record.n_points * record.n_mu;
    if !base_ok {
        return Err(SpinorSectorThcError::ThcRecord { q_index });
    }
    for (sector, layout) in sectors.into_iter().zip(layouts) {
        if sector.layout != layout
            || sector.vertices.len() != layout.n_columns().map_err(ThcError::from)?
        {
            return Err(SpinorSectorThcError::ThcRecord { q_index });
        }
        for (column, vertex) in sector.vertices.iter().enumerate() {
            let (k_index, occupied, target) = layout.decode(column).map_err(ThcError::from)?;
            let pair = OrbitalPair::Exchange {
                k_index,
                occupied_space: layout.occupied_space,
                occupied,
                target_space: layout.target_space,
                target,
            };
            if vertex.layout() != &record.auxiliary.layout()
                || vertex.pair() != pair
                || vertex.provenance() != &record.auxiliary.provenance
            {
                return Err(SpinorSectorThcError::ThcRecord { q_index });
            }
        }
    }
    Ok(())
}

fn complete_vv_mpb_vertices<'a>(
    result: &'a SpinorMpbResult,
    layout: ExchangePairLayout,
    q_index: usize,
) -> Result<Vec<&'a PairVertex>, SpinorSectorThcError> {
    let n_columns = layout.n_columns().map_err(ThcError::from)?;
    let mut ordered = vec![None; n_columns];
    if result.vertices.len() != n_columns {
        return Err(coverage_error(q_index, layout));
    }
    for record in &result.vertices {
        if record.column >= n_columns || ordered[record.column].is_some() {
            return Err(coverage_error(q_index, layout));
        }
        let (k_index, occupied, target) = layout.decode(record.column).map_err(ThcError::from)?;
        if record.k != k_index
            || record.left_band != occupied
            || record.right_band != target
            || record.vertex.layout() != &result.auxiliary.layout()
            || record.vertex.pair()
                != (OrbitalPair::Bloch {
                    k_index,
                    left: occupied,
                    right: target,
                })
        {
            return Err(coverage_error(q_index, layout));
        }
        ordered[record.column] = Some(&record.vertex);
    }
    ordered
        .into_iter()
        .map(|vertex| vertex.ok_or_else(|| coverage_error(q_index, layout)))
        .collect()
}

fn complete_exchange_mpb_vertices<'a>(
    sector: &'a SpinorExchangeMpbSector,
    q_index: usize,
) -> Result<Vec<&'a PairVertex>, SpinorSectorThcError> {
    let layout = sector.layout;
    let n_columns = layout.n_columns().map_err(ThcError::from)?;
    let mut ordered = vec![None; n_columns];
    if sector.vertices.len() != n_columns {
        return Err(coverage_error(q_index, layout));
    }
    for record in &sector.vertices {
        if record.column >= n_columns || ordered[record.column].is_some() {
            return Err(coverage_error(q_index, layout));
        }
        let (k_index, occupied, target) = layout.decode(record.column).map_err(ThcError::from)?;
        let pair = OrbitalPair::Exchange {
            k_index,
            occupied_space: layout.occupied_space,
            occupied,
            target_space: layout.target_space,
            target,
        };
        if record.k != k_index
            || record.occupied != occupied
            || record.target != target
            || record.vertex.pair() != pair
        {
            return Err(coverage_error(q_index, layout));
        }
        ordered[record.column] = Some(&record.vertex);
    }
    ordered
        .into_iter()
        .map(|vertex| vertex.ok_or_else(|| coverage_error(q_index, layout)))
        .collect()
}

fn coverage_error(q_index: usize, layout: ExchangePairLayout) -> SpinorSectorThcError {
    SpinorSectorThcError::MpbCoverage {
        q_index,
        occupied_space: layout.occupied_space,
        target_space: layout.target_space,
    }
}

fn sampled_sector_auxiliary(
    record: &SpinorSectorThcQRecord,
    grid: &ThcParentGrid,
    site_meshes: Vec<ExponentialMesh>,
) -> Result<SampledAuxiliaryFunctions, CoulombError> {
    SampledAuxiliaryFunctions::new(
        record.auxiliary.layout(),
        site_meshes,
        grid.points().iter().map(|point| point.coordinate).collect(),
        grid.points()
            .iter()
            .map(|point| VolumeBohr3(point.weight))
            .collect(),
        grid.points()
            .iter()
            .map(|point| match point.region {
                ThcRegion::MuffinTin { site, radial_index } => {
                    SampledPointSupport::MuffinTin { site, radial_index }
                }
                ThcRegion::Interstitial => SampledPointSupport::Interstitial,
            })
            .collect(),
        record.zeta.clone(),
    )
}

fn compare_sector_pairs(
    q_index: usize,
    thc: &SpinorSectorThcVertices,
    mpb: &[&PairVertex],
    thc_operator: &muffintin_coulomb::CoulombOperator,
    mpb_operator: &muffintin_coulomb::CoulombOperator,
    diagnostics: &mut Vec<SpinorSectorThcMpbPairDiagnostic>,
) -> Result<(), SpinorSectorThcError> {
    if thc.vertices.len() != mpb.len() {
        return Err(coverage_error(q_index, thc.layout));
    }
    for (column, (thc_vertex, mpb_vertex)) in thc.vertices.iter().zip(mpb).enumerate() {
        let mpb_quadratic = mpb_operator.quadratic_form(mpb_vertex, mpb_vertex)?;
        let thc_quadratic = thc_operator.quadratic_form(thc_vertex, thc_vertex)?;
        let (absolute, relative) =
            quadratic_discrepancy(mpb_quadratic, thc_quadratic, SPINOR_COULOMB_EXACTNESS_FLOOR);
        diagnostics.push(SpinorSectorThcMpbPairDiagnostic {
            q_index,
            column,
            pair: thc_vertex.pair(),
            mpb_quadratic,
            thc_quadratic,
            absolute,
            relative,
        });
    }
    Ok(())
}

fn summarize_mpb_sector(
    pairs: Vec<SpinorSectorThcMpbPairDiagnostic>,
    layout: ExchangePairLayout,
) -> Result<SpinorSectorThcMpbSectorComparison, SpinorSectorThcError> {
    let first = pairs.first().ok_or_else(|| coverage_error(0, layout))?;
    let mut maximum_absolute = first.absolute;
    let mut maximum_relative = first.relative;
    let mut worst_absolute_q_index = first.q_index;
    let mut worst_absolute_column = first.column;
    let mut worst_relative_q_index = first.q_index;
    let mut worst_relative_column = first.column;
    for pair in pairs.iter().skip(1) {
        if pair.absolute > maximum_absolute {
            maximum_absolute = pair.absolute;
            worst_absolute_q_index = pair.q_index;
            worst_absolute_column = pair.column;
        }
        if pair.relative > maximum_relative {
            maximum_relative = pair.relative;
            worst_relative_q_index = pair.q_index;
            worst_relative_column = pair.column;
        }
    }
    Ok(SpinorSectorThcMpbSectorComparison {
        pairs,
        maximum_absolute,
        maximum_relative,
        worst_absolute_q_index,
        worst_absolute_column,
        worst_relative_q_index,
        worst_relative_column,
    })
}
