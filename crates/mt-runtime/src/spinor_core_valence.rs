//! Frozen site-valence densities and radial/exact-MPB core-valence diagnostics.

use crate::isdf_exchange::{IsdfExchangeError, validate_k_weights, validate_occupations};
use crate::spinor_product::{SpinorProductInput, SpinorQSliceError, require_spinor_q_slice};
use crate::spinor_sector_exchange::{
    CoreShellSpillDiagnostic, FrozenCoreValenceExchange, SectorOccupations,
};
use muffintin_core::{Hartree, RelativisticChannel, TwiceMu};
use muffintin_coulomb::{
    BorrowedCoreShell, BorrowedValenceRadial, ClosedCoreOccupations, PreweightedSiteValenceDensity,
    RadialSlaterCvTraces, RadialSlaterError, RadialSlaterSite, RadialValenceCoreActions,
    RadialValenceCoreError, radial_slater_cv_traces, radial_valence_core_actions,
};
use muffintin_dft::{
    CoreFixedPotentialResult, CoreFixedPotentialSpec, CoreRelaxationError, CoreShellOccupations,
    CoreShellOrbitals, FixedSiteValenceDensity, relax_core_at_fixed_potential,
};
use muffintin_operators::{CompiledSiteProjection, OperatorError};
use muffintin_prodbasis::{DiracRadialId, DiracRadialNormalization};
use num_complex::Complex64;
use thiserror::Error;

/// One normalized site-projection coordinate in the frozen valence density.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenSiteValenceOrbital {
    pub radial: DiracRadialId,
    pub twice_mu: TwiceMu,
    pub channel: RelativisticChannel,
    pub p: Vec<f64>,
    pub q: Vec<f64>,
    pub normalization: f64,
}

/// Complete occupied valence density in one site's projection coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenSiteValenceDensity {
    pub site_index: usize,
    pub orbitals: Vec<FrozenSiteValenceOrbital>,
    /// Row-major density in normalized radial coordinates:
    /// $D_{ab}=\sum_{kn}w_k f_{kn}\hat d^*_{an}\hat d_{bn}$,
    /// with $\hat d_a=\sqrt{N_a}d_a$ for the original LAPW coefficients.
    pub matrix: Vec<Complex64>,
}

/// Frozen site densities sealed against the complete q slice and occupations.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenSiteValenceDensities {
    pub sites: Vec<FrozenSiteValenceDensity>,
    sealed: FrozenSiteValenceContext,
}

/// Production radial actions sealed against their complete frozen density frame.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenRadialValenceCoreActions {
    pub actions: RadialValenceCoreActions,
    sealed: FrozenSiteValenceContext,
}

#[derive(Clone, Debug, PartialEq)]
struct FrozenSiteValenceContext {
    inputs: Vec<SpinorProductInput>,
    occupations: SectorOccupations,
}

impl FrozenSiteValenceDensities {
    /// Exact freshness check over orbitals, bases, radials, q maps, weights, and occupations.
    pub fn frozen_context_matches(
        &self,
        inputs: &[SpinorProductInput],
        occupations: &SectorOccupations,
    ) -> bool {
        self.sealed.inputs == inputs && &self.sealed.occupations == occupations
    }

    pub(crate) fn frozen_context(&self) -> (&[SpinorProductInput], &SectorOccupations) {
        (&self.sealed.inputs, &self.sealed.occupations)
    }
}

impl FrozenRadialValenceCoreActions {
    pub fn frozen_context_matches(
        &self,
        inputs: &[SpinorProductInput],
        occupations: &SectorOccupations,
    ) -> bool {
        self.sealed.inputs == inputs && &self.sealed.occupations == occupations
    }
}

/// Numerical and physical-spill gates for M3b radial actions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreValenceComparisonSpec {
    pub numerical_tolerance: Hartree,
    pub maximum_shell_spill: f64,
}

/// Exact versus spherical CV expectation for one flat core spin orbital.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreValenceDeltaDiagnostic {
    pub core_index: usize,
    pub site_index: usize,
    pub n: u32,
    pub kappa: muffintin_core::Kappa,
    pub twice_mu: TwiceMu,
    pub occupation: f64,
    pub exact_vc: Hartree,
    pub spherical_vc: Hartree,
    pub delta_c: Hartree,
}

/// Gated production-action, independent-radial, and exact-MPB CV comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenCoreValenceComparison {
    pub actions: RadialValenceCoreActions,
    pub radial_oracle: RadialSlaterCvTraces,
    /// Production VC action versus the legacy `radial.cv_mt` oracle field.
    pub vc_action_legacy_radial_residual: Hartree,
    /// Full finite-body MPB CV trace minus the spherical on-site VC action trace.
    pub vc_action_cross_cv_mpb_difference: Hartree,
    /// Full finite-body MPB VC trace minus the spherical on-site VC action trace.
    pub vc_action_mpb_difference: Hartree,
    pub mpb_cross_trace_residual: Hartree,
    pub deltas: Vec<CoreValenceDeltaDiagnostic>,
    pub weighted_delta: Hartree,
    /// $T_{vc}^{\mathrm{MPB}}-T_{vc}^{\mathrm{radial}}$.
    pub weighted_delta_target: Hartree,
    pub weighted_delta_closure_residual: Hartree,
    pub shell_spill: Vec<CoreShellSpillDiagnostic>,
    pub maximum_measured_shell_spill: f64,
    pub shell_spill_threshold: f64,
}

#[derive(Debug, Error)]
pub enum FrozenCoreValenceError {
    #[error(transparent)]
    Mesh(#[from] muffintin_core::MeshError),
    #[error(transparent)]
    Exchange(#[from] IsdfExchangeError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Action(#[from] RadialValenceCoreError),
    #[error(transparent)]
    RadialOracle(#[from] RadialSlaterError),
    #[error(transparent)]
    CoreRelaxation(#[from] CoreRelaxationError),
    #[error("frozen core-valence density requires a complete compatible spinor q slice")]
    FrozenContext,
    #[error("frozen core-valence occupations do not match the flat core table")]
    CoreOccupations,
    #[error("frozen core-valence site-projection identities changed across k points")]
    SiteProjectionIdentity,
    #[error("frozen core-valence radial identity is absent from its source")]
    RadialIdentity,
    #[error("frozen core-valence comparison tolerance is invalid")]
    Tolerance,
    #[error("frozen core-valence comparison contexts are stale or inconsistent")]
    ComparisonContext,
    #[error("frozen VC relaxation core sidecar does not match its sealed valence-density frame")]
    CoreRelaxationContext,
    #[error("frozen core-valence comparison is missing a radial shell action")]
    MissingShellAction,
    #[error("frozen core-valence {quantity} numerical residual {residual} exceeds {tolerance}")]
    Numerical {
        quantity: &'static str,
        residual: f64,
        tolerance: f64,
    },
    #[error("frozen core-valence weighted delta closure residual {residual} exceeds {tolerance}")]
    DeltaClosure { residual: f64, tolerance: f64 },
    #[error(
        "core shell site={site} n={n} kappa={kappa} spill {spill} exceeds the dimensionless threshold {threshold}"
    )]
    CoreSpill {
        site: usize,
        n: u32,
        kappa: i32,
        spill: f64,
        threshold: f64,
    },
}

/// Form each site's occupied valence density exactly once from one complete q slice.
///
/// This inserts $w_k f_{kn}$ and changes to the individually normalized radial
/// coordinates used by the radial exchange oracle/action. It inserts no $q$
/// weight, spin multiplier, or full overlap matrix.
pub fn build_frozen_site_valence_densities(
    inputs: &[SpinorProductInput],
    occupations: &SectorOccupations,
) -> Result<FrozenSiteValenceDensities, FrozenCoreValenceError> {
    let first = require_spinor_q_slice(inputs).map_err(q_slice_error)?;
    let n_k = first.pair_columns.n_k;
    let n_valence = first.pair_columns.n_orb;
    validate_k_weights(&occupations.k_weights, n_k)?;
    validate_occupations(&occupations.valence, n_k, n_valence)?;
    let core_rows = vec![occupations.core.clone(); n_k];
    validate_occupations(&core_rows, n_k, first.core.orbitals.len())?;
    if occupations.core
        != first
            .core
            .orbitals
            .iter()
            .map(|orbital| orbital.occupation)
            .collect::<Vec<_>>()
    {
        return Err(FrozenCoreValenceError::CoreOccupations);
    }

    let mut sites = Vec::with_capacity(first.source.partition.site_count());
    for site in 0..first.source.partition.site_count() {
        let first_projection = site_projection(first, 0, site)?;
        let coordinate_count = first_projection.coordinate_count();
        let mut orbitals = Vec::with_capacity(coordinate_count);
        for coordinate in 0..coordinate_count {
            let (radial, twice_mu) = first
                .site_projection_identity(site, coordinate)
                .ok_or(FrozenCoreValenceError::SiteProjectionIdentity)?;
            let source = first
                .source
                .find_radial(radial)
                .ok_or(FrozenCoreValenceError::RadialIdentity)?;
            let normalization = match source.normalization {
                DiracRadialNormalization::Explicit(value) => value,
                DiracRadialNormalization::OnMesh => first.source.radials[site].mesh.integrate(
                    &source
                        .samples
                        .large
                        .iter()
                        .zip(&source.samples.small)
                        .map(|(p, q)| p * p + q * q)
                        .collect::<Vec<_>>(),
                )?,
            };
            orbitals.push(FrozenSiteValenceOrbital {
                radial,
                twice_mu,
                channel: RelativisticChannel::new(radial.kappa, twice_mu)
                    .expect("site projection identity carries a valid channel"),
                p: source.samples.large.clone(),
                q: source.samples.small.clone(),
                normalization,
            });
        }
        let mut matrix = vec![Complex64::new(0.0, 0.0); coordinate_count * coordinate_count];
        for k in 0..n_k {
            let projected = if k == 0 {
                first_projection.project_eigenvectors(&first.orbitals.eigenvectors[k])?
            } else {
                site_projection(first, k, site)?
                    .project_eigenvectors(&first.orbitals.eigenvectors[k])?
            };
            if projected.coordinate_count() != coordinate_count
                || (0..coordinate_count).any(|coordinate| {
                    first.site_projection_identity(site, coordinate)
                        != orbitals
                            .get(coordinate)
                            .map(|orbital| (orbital.radial, orbital.twice_mu))
                })
            {
                return Err(FrozenCoreValenceError::SiteProjectionIdentity);
            }
            for band in 0..n_valence {
                let weight = occupations.k_weights[k] * occupations.valence[k][band];
                for left in 0..coordinate_count {
                    for right in 0..coordinate_count {
                        matrix[left * coordinate_count + right] += weight
                            * projected.at(left, band).conj()
                            * projected.at(right, band)
                            * (orbitals[left].normalization * orbitals[right].normalization).sqrt();
                    }
                }
            }
        }
        sites.push(FrozenSiteValenceDensity {
            site_index: site,
            orbitals,
            matrix,
        });
    }
    Ok(FrozenSiteValenceDensities {
        sites,
        sealed: FrozenSiteValenceContext {
            inputs: inputs.to_vec(),
            occupations: occupations.clone(),
        },
    })
}

/// Build only the production radial core-valence action from a sealed site density.
///
/// The independent radial Slater oracle and exact MPB diagnostics are not
/// evaluated on this path.
pub fn build_frozen_radial_valence_core_actions(
    densities: &FrozenSiteValenceDensities,
) -> Result<FrozenRadialValenceCoreActions, FrozenCoreValenceError> {
    let actions = with_radial_sites(densities, |sites| Ok(radial_valence_core_actions(sites)?))?;
    Ok(FrozenRadialValenceCoreActions {
        actions,
        sealed: densities.sealed.clone(),
    })
}

/// Relax one core sidecar against fresh CC and VC actions from a sealed valence density.
///
/// The valence density remains fixed while the VC action is rebuilt from each
/// latest core Picard iterate inside the shared DFT loop.
pub fn relax_frozen_core_at_fixed_potential(
    densities: &FrozenSiteValenceDensities,
    initial: &CoreShellOrbitals,
    spec: CoreFixedPotentialSpec,
) -> Result<CoreFixedPotentialResult, FrozenCoreValenceError> {
    let density = densities
        .sites
        .iter()
        .find(|density| density.site_index == initial.site_index)
        .ok_or(FrozenCoreValenceError::FrozenContext)?;
    let first = densities
        .sealed
        .inputs
        .first()
        .ok_or(FrozenCoreValenceError::FrozenContext)?;
    let sealed_sidecar = first
        .core
        .sidecars
        .iter()
        .find(|sidecar| sidecar.site_index == initial.site_index)
        .ok_or(FrozenCoreValenceError::CoreRelaxationContext)?;
    if sealed_sidecar != initial {
        return Err(FrozenCoreValenceError::CoreRelaxationContext);
    }
    let orbitals = density
        .orbitals
        .iter()
        .map(|orbital| BorrowedValenceRadial {
            channel: orbital.channel,
            p: &orbital.p,
            q: &orbital.q,
            normalization: orbital.normalization,
        })
        .collect::<Vec<_>>();
    Ok(relax_core_at_fixed_potential(
        initial,
        FixedSiteValenceDensity {
            site_index: density.site_index,
            muffin_tin_mesh: &first.source.radials[density.site_index].mesh,
            valence: PreweightedSiteValenceDensity {
                orbitals: &orbitals,
                matrix: &density.matrix,
            },
        },
        spec,
    )?)
}

/// Gate the production radial action against both independent radial and exact-MPB traces.
pub fn compare_frozen_core_valence(
    exchange: &FrozenCoreValenceExchange,
    densities: &FrozenSiteValenceDensities,
    spec: CoreValenceComparisonSpec,
) -> Result<FrozenCoreValenceComparison, FrozenCoreValenceError> {
    let tolerance = spec.numerical_tolerance.get();
    if !tolerance.is_finite()
        || tolerance < 0.0
        || !spec.maximum_shell_spill.is_finite()
        || spec.maximum_shell_spill < 0.0
    {
        return Err(FrozenCoreValenceError::Tolerance);
    }
    let (inputs, occupations) = densities.frozen_context();
    if !exchange.frozen_inputs_occupations_match(inputs, occupations) {
        return Err(FrozenCoreValenceError::ComparisonContext);
    }
    let actions = build_frozen_radial_valence_core_actions(densities)?.actions;
    let radial_oracle = with_radial_sites(densities, |sites| Ok(radial_slater_cv_traces(sites)?))?;
    let action_radial = (actions.action_trace.get() - radial_oracle.cv_mt.total.get()).abs();
    let action_cv_difference = exchange.cv.trace.get() - actions.action_trace.get();
    let action_vc_difference = exchange.vc.trace.get() - actions.action_trace.get();
    let mpb_cross_trace = (exchange.cv.trace.get() - exchange.vc.trace.get()).abs();
    for (quantity, residual) in [
        ("VC action/legacy radial CV trace", action_radial),
        ("MPB CV/VC cross trace", mpb_cross_trace),
        ("VC action imaginary trace", actions.imaginary_residual),
        (
            "radial imaginary trace",
            radial_oracle.cv_imaginary_residual,
        ),
    ] {
        if residual > tolerance {
            return Err(FrozenCoreValenceError::Numerical {
                quantity,
                residual,
                tolerance,
            });
        }
    }

    let first = inputs
        .first()
        .ok_or(FrozenCoreValenceError::ComparisonContext)?;
    if exchange.exact_vc_diagonal.len() != first.core.orbitals.len() {
        return Err(FrozenCoreValenceError::ComparisonContext);
    }
    let mut deltas = Vec::with_capacity(first.core.orbitals.len());
    for (core_index, (core, exact_vc)) in first
        .core
        .orbitals
        .iter()
        .zip(&exchange.exact_vc_diagonal)
        .enumerate()
    {
        let sidecar = first
            .core
            .sidecars
            .iter()
            .find(|sidecar| sidecar.site_index == core.site_index)
            .ok_or(FrozenCoreValenceError::MissingShellAction)?;
        let shell_index = sidecar
            .shells
            .iter()
            .position(|shell| shell.state.n == core.n && shell.state.kappa == core.kappa)
            .ok_or(FrozenCoreValenceError::MissingShellAction)?;
        let spherical_vc = actions
            .shells
            .iter()
            .find(|action| {
                action.site_index == core.site_index && action.shell_index == shell_index
            })
            .ok_or(FrozenCoreValenceError::MissingShellAction)?
            .spherical_expectation;
        deltas.push(CoreValenceDeltaDiagnostic {
            core_index,
            site_index: core.site_index,
            n: core.n,
            kappa: core.kappa,
            twice_mu: core.twice_mu,
            occupation: core.occupation,
            exact_vc: *exact_vc,
            spherical_vc,
            delta_c: Hartree(exact_vc.get() - spherical_vc.get()),
        });
    }
    let weighted_delta = Hartree(
        deltas
            .iter()
            .map(|diagnostic| diagnostic.occupation * diagnostic.delta_c.get())
            .sum(),
    );
    let weighted_delta_target = Hartree(exchange.vc.trace.get() - radial_oracle.cv_mt.total.get());
    let weighted_delta_closure = (weighted_delta.get() - weighted_delta_target.get()).abs();
    if weighted_delta_closure > tolerance {
        return Err(FrozenCoreValenceError::DeltaClosure {
            residual: weighted_delta_closure,
            tolerance,
        });
    }

    let mut shell_spill = Vec::new();
    for core in &first.core.orbitals {
        if shell_spill.iter().any(|item: &CoreShellSpillDiagnostic| {
            item.site_index == core.site_index && item.n == core.n && item.kappa == core.kappa
        }) {
            continue;
        }
        if core.spill > spec.maximum_shell_spill {
            return Err(FrozenCoreValenceError::CoreSpill {
                site: core.site_index,
                n: core.n,
                kappa: core.kappa.get(),
                spill: core.spill,
                threshold: spec.maximum_shell_spill,
            });
        }
        shell_spill.push(CoreShellSpillDiagnostic {
            site_index: core.site_index,
            n: core.n,
            kappa: core.kappa,
            spill: core.spill,
            threshold: spec.maximum_shell_spill,
        });
    }
    let maximum_measured_shell_spill = shell_spill
        .iter()
        .map(|diagnostic| diagnostic.spill)
        .fold(0.0_f64, f64::max);
    Ok(FrozenCoreValenceComparison {
        actions,
        radial_oracle,
        vc_action_legacy_radial_residual: Hartree(action_radial),
        vc_action_cross_cv_mpb_difference: Hartree(action_cv_difference),
        vc_action_mpb_difference: Hartree(action_vc_difference),
        mpb_cross_trace_residual: Hartree(mpb_cross_trace),
        deltas,
        weighted_delta,
        weighted_delta_target,
        weighted_delta_closure_residual: Hartree(weighted_delta_closure),
        shell_spill,
        maximum_measured_shell_spill,
        shell_spill_threshold: spec.maximum_shell_spill,
    })
}

fn with_radial_sites<T>(
    densities: &FrozenSiteValenceDensities,
    evaluate: impl FnOnce(&[RadialSlaterSite<'_>]) -> Result<T, FrozenCoreValenceError>,
) -> Result<T, FrozenCoreValenceError> {
    let first = densities
        .sealed
        .inputs
        .first()
        .ok_or(FrozenCoreValenceError::FrozenContext)?;
    let entries = first
        .core
        .sidecars
        .iter()
        .map(|sidecar| {
            let density = densities
                .sites
                .iter()
                .find(|density| density.site_index == sidecar.site_index)
                .ok_or(FrozenCoreValenceError::FrozenContext)?;
            Ok((sidecar, density))
        })
        .collect::<Result<Vec<_>, FrozenCoreValenceError>>()?;
    let cores = entries
        .iter()
        .map(|(sidecar, _)| {
            sidecar
                .shells
                .iter()
                .map(|shell| BorrowedCoreShell {
                    kappa: shell.state.kappa,
                    p: &shell.p,
                    q: &shell.q,
                    normalization: shell.norm_total,
                    occupations: match &shell.occupations {
                        CoreShellOccupations::MuResolved(occupations) => {
                            ClosedCoreOccupations::MuResolved(occupations)
                        }
                        CoreShellOccupations::ExplicitCollinear { up, down } => {
                            ClosedCoreOccupations::ExplicitCollinear {
                                up: *up,
                                down: *down,
                            }
                        }
                    },
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let valence = entries
        .iter()
        .map(|(_, density)| {
            density
                .orbitals
                .iter()
                .map(|orbital| BorrowedValenceRadial {
                    channel: orbital.channel,
                    p: &orbital.p,
                    q: &orbital.q,
                    normalization: orbital.normalization,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let sites = entries
        .iter()
        .enumerate()
        .map(|(index, (sidecar, density))| RadialSlaterSite {
            site_index: sidecar.site_index,
            mt_mesh: &first.source.radials[sidecar.site_index].mesh,
            extended_mesh: &sidecar.extended_mesh,
            cores: &cores[index],
            valence: PreweightedSiteValenceDensity {
                orbitals: &valence[index],
                matrix: &density.matrix,
            },
        })
        .collect::<Vec<_>>();
    evaluate(&sites)
}

fn site_projection(
    input: &SpinorProductInput,
    k: usize,
    site: usize,
) -> Result<CompiledSiteProjection, FrozenCoreValenceError> {
    let basis = input
        .orbitals
        .bases
        .get(k)
        .ok_or(FrozenCoreValenceError::FrozenContext)?;
    let channels = basis
        .site_augmentations
        .get(site)
        .and_then(|waves| waves.first())
        .map(|wave| wave.channels.as_slice())
        .ok_or(FrozenCoreValenceError::SiteProjectionIdentity)?;
    Ok(CompiledSiteProjection::spinor(basis, site, channels)?)
}

fn q_slice_error(_: SpinorQSliceError) -> FrozenCoreValenceError {
    FrozenCoreValenceError::FrozenContext
}
