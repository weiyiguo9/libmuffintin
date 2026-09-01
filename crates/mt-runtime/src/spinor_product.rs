//! Frozen full-first-variation spinor product input for one requested transfer $q$.

use muffintin_core::{Hartree, Kappa, ReciprocalLattice, TwiceMu};
use muffintin_dft::{
    CheckpointBandSolution, CheckpointKPointSolution, CoreShellOccupations, CoreShellOrbitals,
    MaterialKernelError, ScfConfig, ScfRelativity, SpinorIterationBasis, SpinorRadialSite,
    build_extended_checkpoint_core_potentials, regular_k_points,
};
use muffintin_envelope::site_translation_phase;
use muffintin_operators::lapw::{Provenance, SpinorCompiledBasis};
use muffintin_prodbasis::{
    AuxiliaryPartition, DiracProductSource, DiracRadial, DiracRadialId, DiracRadialNormalization,
    DiracRadialSamples, DiracSiteRadialSet, PairColumnLayout, ProductOrbitalKind,
    RawInterstitialPairSupport, TransferQ,
};
use muffintin_sphere::CorePotentialContinuationSpec;
use muffintin_tensor::DenseEigenvectors;
use num_complex::Complex64;
use std::collections::{BTreeSet, HashSet};
use thiserror::Error;

use crate::checkpoint_physics::{CheckpointPhysics, CheckpointPhysicsError};
use crate::q_mesh::{
    CanonicalQMapError, canonical_transfer_q, map_k_minus_q, validate_canonical_q_map,
};
use crate::scalar_product::leading_bands;

/// `DiracRadialId.n` for the APW base $(P,Q)$.
pub const SPINOR_RADIAL_P: usize = 0;
/// `DiracRadialId.n` for the analytic energy derivative $(\dot P,\dot Q)$.
pub const SPINOR_RADIAL_PDOT: usize = 1;
/// First signed-$\kappa$ LO/RLO `DiracRadialId.n`; later requests use `SPINOR_RADIAL_LO0 + ordinal`.
pub const SPINOR_RADIAL_LO0: usize = 2;

const CORE_DIAGNOSTIC_TOLERANCE: f64 = 1.0e-12;

/// Per-k map of $k-q_{\mathrm{canonical}}$ onto the regular mesh.
///
/// The integer wrap $G_{\mathrm{wrap}}$ satisfies
/// $k_{\mathrm{frac}}-q_{\mathrm{canonical,frac}}
/// =(k-q)_{\mathrm{frac}}+G_{\mathrm{wrap,index}}$
/// in primitive reciprocal coordinates. Pair phases use
/// $\exp(+i G_{\mathrm{wrap}}\cdot r)$. This wrap is not
/// [`TransferQ::umklapp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinorKMinusQ {
    pub k_index: usize,
    pub kq_index: usize,
    pub umklapp: muffintin_core::GVector,
}

/// Common leading band window retained for pair columns.
///
/// The spinor product-input window keeps the lowest `count` eigenpairs starting at `start` (always 0).
/// Per-$k$ available counts remain on [`SpinorFrozenOrbitals::available_bands`].
/// Eigenvector **rows** are never truncated to a common basis dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpinorBandWindow {
    pub start: usize,
    pub count: usize,
}

/// Frozen full-first-variation Bloch data retained for later spinor MPB stages.
///
/// There is no collinear `spin=0/1` field. Eigenvectors are column-major
/// `[basis, band]` in the live [`SpinorCompiledBasis`] order:
/// two Pauli interstitial plane-wave blocks $\mathrm{spin}\,N_G+G$ (spin slow,
/// shared spatial $G$ labels), then each site's confined LO/RLO rows in
/// canonical $(\kappa, 2\mu, n)$ order with $n$ fastest, $\kappa$ ascending,
/// $2\mu$ ascending. APW $(P,\dot P)$ columns are matching coefficients on
/// those plane-wave rows, not extra eigenbasis rows.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorFrozenOrbitals {
    pub k_fractional: Vec<[f64; 3]>,
    pub eigenvectors: Vec<DenseEigenvectors>,
    pub energies: Vec<Vec<Hartree>>,
    pub bases: Vec<SpinorCompiledBasis>,
    pub available_bands: Vec<usize>,
    pub band_window: SpinorBandWindow,
}

/// One caller-retained core spin orbital in canonical flat order.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorCoreOrbital {
    pub site_index: usize,
    pub n: u32,
    pub kappa: Kappa,
    pub twice_mu: TwiceMu,
    pub occupation: f64,
    pub radial: DiracRadialId,
    pub energy: Hartree,
    pub norm_total: f64,
    pub norm_mt: f64,
    pub spill: f64,
}

/// Preserved M0 sidecars plus the canonical `site -> n -> kappa -> twice_mu` table.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpinorCoreTable {
    pub sidecars: Vec<CoreShellOrbitals>,
    pub orbitals: Vec<SpinorCoreOrbital>,
}

/// Frozen spinor first-variation solve plus Dirac product input at one $q$.
///
/// `source` is the method-neutral [`DiracProductSource`]. Valence radials use
/// [`SPINOR_RADIAL_P`], [`SPINOR_RADIAL_PDOT`], then signed-$\kappa$ LO/RLO
/// from [`SPINOR_RADIAL_LO0`] in each shell's request order. Radial identity
/// $(site, kind, \kappa, n)$ is $\mu$-degenerate. Pair columns use
/// [`PairColumnLayout`] indexing $k\cdot N_{\mathrm{orb}}^2+i\cdot N_{\mathrm{orb}}+j$
/// with left band at $k-q$ and right band at $k$. Caller-provided core
/// sidecars remain in [`SpinorCoreTable`].
/// `reciprocal` is the exact lattice used to fold $q_{\mathrm{in}}$ and
/// $G_{\mathrm{wrap}}$.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorProductInput {
    pub source: DiracProductSource,
    pub orbitals: SpinorFrozenOrbitals,
    pub k_minus_q: Vec<SpinorKMinusQ>,
    pub pair_columns: PairColumnLayout,
    pub reciprocal: ReciprocalLattice,
    pub core: SpinorCoreTable,
}

/// Core-sidecar rejection at the spinor product boundary.
#[derive(Debug, Error)]
pub enum SpinorCoreInputError {
    #[error(transparent)]
    Checkpoint(#[from] CheckpointPhysicsError),
    #[error("core sidecar site index {site} is outside 0..{site_count}")]
    SiteIndex { site: usize, site_count: usize },
    #[error("core sidecar site index {0} appears more than once")]
    DuplicateSite(usize),
    #[error(
        "core sidecar site {site} extended mesh is not an exact prefix extension of its muffin-tin mesh"
    )]
    MeshPrefix { site: usize },
    #[error(
        "core sidecar site {site} n={n} kappa={kappa} has inconsistent P/Q and extended-mesh lengths"
    )]
    RadialLength { site: usize, n: u32, kappa: i32 },
    #[error(
        "core sidecar site {site} n={n} kappa={kappa} has invalid norm/spill diagnostics: norm_total={norm_total}, norm_mt={norm_mt}, spill={spill}"
    )]
    RadialDiagnostics {
        site: usize,
        n: u32,
        kappa: i32,
        norm_total: f64,
        norm_mt: f64,
        spill: f64,
    },
    #[error("core sidecar site {site} duplicates n={n} kappa={kappa}")]
    DuplicateShell { site: usize, n: u32, kappa: i32 },
    #[error(
        "core sidecar site {site} n={n} kappa={kappa} uses ExplicitCollinear occupations; rectangular spinor exchange requires MuResolved"
    )]
    ExplicitCollinear { site: usize, n: u32, kappa: i32 },
    #[error(
        "core sidecar site {site} n={n} kappa={kappa} does not contain each magnetic channel exactly once"
    )]
    MagneticChannels { site: usize, n: u32, kappa: i32 },
    #[error(
        "core sidecar site {site} n={n} kappa={kappa} 2mu={twice_mu} has invalid occupation {occupation}"
    )]
    Occupation {
        site: usize,
        n: u32,
        kappa: i32,
        twice_mu: i64,
        occupation: f64,
    },
    #[error(transparent)]
    Product(#[from] muffintin_prodbasis::DiracProductError),
}

impl CheckpointPhysics {
    /// Frozen full-first-variation one-particle solve and Dirac product input.
    ///
    /// `q_fractional` is the requested primitive-cell transfer $q_{\mathrm{in}}$.
    /// The emitted [`TransferQ`] stores $q_{\mathrm{canonical}}$ in $[0,1)^3$
    /// with $q_{\mathrm{in}}=q_{\mathrm{canonical}}+G_{\mathrm{transfer}}$.
    /// Each k-point is mapped with $q_{\mathrm{canonical}}$; off-mesh folded
    /// targets are rejected. Relativity must be
    /// [`ScfRelativity::SpinorFirstVariation`].
    pub fn spinor_product_input(
        &self,
        config: &ScfConfig,
        q_fractional: [f64; 3],
    ) -> Result<SpinorProductInput, CheckpointPhysicsError> {
        match config.relativity {
            ScfRelativity::SpinorFirstVariation => {}
            ScfRelativity::Scalar => {
                return Err(CheckpointPhysicsError::SpinorProductRejectsScalarRelativity);
            }
            ScfRelativity::SocSecondVariation { .. } => {
                return Err(CheckpointPhysicsError::SpinorProductRejectsSocSecondVariation);
            }
        }
        let meshes = self.kernel.channel_meshes(&config.basis)?;
        let extended = build_extended_checkpoint_core_potentials(
            self.frozen_potential(),
            self.geometry(),
            self.nuclear_charges(),
            &meshes,
            CorePotentialContinuationSpec::default(),
        )
        .map_err(MaterialKernelError::from)?;
        let basis = self.kernel.materialize_nonspectral_basis(
            self.frozen_potential(),
            &config.basis,
            &extended,
        )?;
        let k_fractional = regular_k_points(config.k_mesh)?;
        let bands = self.kernel.solve_points(
            self.frozen_potential(),
            &basis,
            &k_fractional,
            ScfRelativity::SpinorFirstVariation,
        )?;
        self.spinor_product_input_from_bands(&bands, &k_fractional, q_fractional)
    }

    /// Build a Dirac product input from one current live spinor band solution.
    ///
    /// The supplied k points are the exact ordered points used to create
    /// `bands`; no frozen-potential solve occurs here. This is the feedback
    /// seam used by self-consistent Fock iterations so every orbital update
    /// produces fresh pair vertices.
    pub fn spinor_product_input_from_bands(
        &self,
        bands: &CheckpointBandSolution,
        k_fractional: &[[f64; 3]],
        q_fractional: [f64; 3],
    ) -> Result<SpinorProductInput, CheckpointPhysicsError> {
        if bands.points().len() != k_fractional.len() || k_fractional.is_empty() {
            return Err(CheckpointPhysicsError::SpinorProductKSliceMismatch);
        }
        let transfer = canonical_transfer_q(q_fractional, *self.reciprocal())?;
        let mut k_minus_q = Vec::with_capacity(k_fractional.len());
        for (k_index, &k_frac) in k_fractional.iter().enumerate() {
            let mapped =
                map_k_minus_q(k_index, k_frac, transfer, k_fractional, *self.reciprocal())?;
            k_minus_q.push(SpinorKMinusQ {
                k_index: mapped.k_index,
                kq_index: mapped.kq_index,
                umklapp: mapped.umklapp,
            });
        }
        emit_spinor_product_input(self, bands, k_fractional, transfer.q, k_minus_q)
    }

    /// Build the spinor product input and attach caller-owned M0 core sidecars.
    pub fn spinor_product_input_with_cores(
        &self,
        config: &ScfConfig,
        q_fractional: [f64; 3],
        sidecars: &[CoreShellOrbitals],
    ) -> Result<SpinorProductInput, SpinorCoreInputError> {
        Ok(self
            .spinor_product_input(config, q_fractional)?
            .with_core_sidecars(sidecars)?)
    }
}

impl SpinorProductInput {
    /// Attach exact M0 core sidecars without resampling or MT renormalization.
    pub fn with_core_sidecars(
        mut self,
        sidecars: &[CoreShellOrbitals],
    ) -> Result<Self, SpinorCoreInputError> {
        let site_count = self.source.radials.len();
        let mut seen_sites = HashSet::new();
        let mut seen_shells = HashSet::new();
        let mut shell_refs = Vec::new();
        for sidecar in sidecars {
            let site = sidecar.site_index;
            if site >= site_count {
                return Err(SpinorCoreInputError::SiteIndex { site, site_count });
            }
            if !seen_sites.insert(site) {
                return Err(SpinorCoreInputError::DuplicateSite(site));
            }
            let mesh = &self.source.radials[site].mesh;
            if sidecar.extended_mesh.len() < mesh.len()
                || sidecar.extended_mesh.radii()[..mesh.len()] != mesh.radii()[..]
            {
                return Err(SpinorCoreInputError::MeshPrefix { site });
            }
            for shell in &sidecar.shells {
                if shell.p.len() != sidecar.extended_mesh.len()
                    || shell.q.len() != sidecar.extended_mesh.len()
                {
                    return Err(SpinorCoreInputError::RadialLength {
                        site,
                        n: shell.state.n,
                        kappa: shell.state.kappa.get(),
                    });
                }
                let diagnostic_scale = shell
                    .norm_total
                    .abs()
                    .max(shell.norm_mt.abs())
                    .max(shell.spill.abs())
                    .max(1.0);
                let diagnostic_tolerance = CORE_DIAGNOSTIC_TOLERANCE * diagnostic_scale;
                if !shell.norm_total.is_finite()
                    || shell.norm_total <= 0.0
                    || !shell.norm_mt.is_finite()
                    || shell.norm_mt < 0.0
                    || shell.norm_mt > shell.norm_total + diagnostic_tolerance
                    || !shell.spill.is_finite()
                    || shell.spill < -diagnostic_tolerance
                    || (shell.norm_total - shell.norm_mt - shell.spill).abs()
                        > diagnostic_tolerance
                {
                    return Err(SpinorCoreInputError::RadialDiagnostics {
                        site,
                        n: shell.state.n,
                        kappa: shell.state.kappa.get(),
                        norm_total: shell.norm_total,
                        norm_mt: shell.norm_mt,
                        spill: shell.spill,
                    });
                }
                if !seen_shells.insert((site, shell.state.n, shell.state.kappa)) {
                    return Err(SpinorCoreInputError::DuplicateShell {
                        site,
                        n: shell.state.n,
                        kappa: shell.state.kappa.get(),
                    });
                }
                let CoreShellOccupations::MuResolved(occupations) = &shell.occupations else {
                    return Err(SpinorCoreInputError::ExplicitCollinear {
                        site,
                        n: shell.state.n,
                        kappa: shell.state.kappa.get(),
                    });
                };
                let mut occupations = occupations.clone();
                occupations.sort_by_key(|(twice_mu, _)| *twice_mu);
                let expected = shell.state.kappa.twice_mu_values().collect::<Vec<_>>();
                if occupations.len() != expected.len()
                    || occupations
                        .iter()
                        .map(|(twice_mu, _)| *twice_mu)
                        .ne(expected.iter().copied())
                {
                    return Err(SpinorCoreInputError::MagneticChannels {
                        site,
                        n: shell.state.n,
                        kappa: shell.state.kappa.get(),
                    });
                }
                for &(twice_mu, occupation) in &occupations {
                    if !occupation.is_finite() || !(0.0..=1.0).contains(&occupation) {
                        return Err(SpinorCoreInputError::Occupation {
                            site,
                            n: shell.state.n,
                            kappa: shell.state.kappa.get(),
                            twice_mu: twice_mu.get(),
                            occupation,
                        });
                    }
                }
                shell_refs.push((site, shell, occupations));
            }
        }
        shell_refs.sort_by_key(|(site, shell, _)| (*site, shell.state.n, shell.state.kappa));
        let mut flat = Vec::new();
        for (site, shell, occupations) in shell_refs {
            let mt_len = self.source.radials[site].mesh.len();
            let radial = DiracRadialId {
                site,
                kind: ProductOrbitalKind::Core,
                kappa: shell.state.kappa,
                n: shell.state.n as usize,
            };
            self.source.radials[site].cores.push(DiracRadial {
                kappa: shell.state.kappa,
                n: radial.n,
                samples: DiracRadialSamples {
                    large: shell.p[..mt_len].to_vec(),
                    small: shell.q[..mt_len].to_vec(),
                },
                normalization: DiracRadialNormalization::Explicit(shell.norm_total),
            });
            for (twice_mu, occupation) in occupations {
                flat.push(SpinorCoreOrbital {
                    site_index: site,
                    n: shell.state.n,
                    kappa: shell.state.kappa,
                    twice_mu,
                    occupation,
                    radial,
                    energy: shell.energy,
                    norm_total: shell.norm_total,
                    norm_mt: shell.norm_mt,
                    spill: shell.spill,
                });
            }
        }
        self.core = SpinorCoreTable {
            sidecars: sidecars.to_vec(),
            orbitals: flat,
        };
        self.source.validate()?;
        Ok(self)
    }

    /// Pauli plane-wave eigenbasis row $\mathrm{spin}\,N_G+G$ at k-point `k`.
    ///
    /// Both spin blocks share [`SpinorCompiledBasis::plane_waves`] labels.
    /// The spinor mixed-product bridge sums same-component up/down products; there is no spin cross term.
    pub fn pauli_plane_wave_row(&self, k: usize, spin: u8, g: usize) -> Option<usize> {
        self.orbitals
            .bases
            .get(k)?
            .layout
            .plane_wave_index(usize::from(spin), g)
    }

    /// Confined LO/RLO eigenbasis row for `radial_n = 2 + ordinal`.
    ///
    /// APW $n=0,1$ are matching columns on plane-wave rows and return `None`.
    pub fn compiled_lo_row(
        &self,
        k: usize,
        site: usize,
        kappa: Kappa,
        twice_mu: TwiceMu,
        radial_n: usize,
    ) -> Option<usize> {
        let lo_n = radial_n.checked_sub(SPINOR_RADIAL_LO0)?;
        self.orbitals
            .bases
            .get(k)?
            .layout
            .site_spinor_index(site, kappa, twice_mu, lo_n)
    }

    /// Invert a compiled LO/RLO eigenbasis row to a [`DiracRadialId`] and $2\mu$.
    pub fn compiled_lo_identity(&self, k: usize, row: usize) -> Option<(DiracRadialId, TwiceMu)> {
        let layout = &self.orbitals.bases.get(k)?.layout;
        if row < layout.plane_wave_count() {
            return None;
        }
        for site in 0..layout.site_count() {
            let range = layout.site_spinor_range(site)?;
            if !range.contains(&row) {
                continue;
            }
            let site_layout = layout.site_layout(site)?;
            for &(kappa, count) in site_layout.counts_by_kappa() {
                for twice_mu in kappa.twice_mu_values() {
                    for n in 0..count {
                        if layout.site_spinor_index(site, kappa, twice_mu, n) == Some(row) {
                            return Some((
                                DiracRadialId {
                                    site,
                                    kind: ProductOrbitalKind::Valence,
                                    kappa,
                                    n: SPINOR_RADIAL_LO0 + n,
                                },
                                twice_mu,
                            ));
                        }
                    }
                }
            }
        }
        None
    }

    /// Invert a site-projection coordinate to [`DiracRadialId`] and $2\mu$.
    ///
    /// Coordinates follow the live APW-then-LO layout used by
    /// [`Self::site_projection_row`]. Large and small radial samples share that
    /// coordinate; they are not a collinear spin index.
    pub fn site_projection_identity(
        &self,
        site: usize,
        coordinate: usize,
    ) -> Option<(DiracRadialId, TwiceMu)> {
        let radials = self.source.radials.get(site)?;
        for shell in shells(radials) {
            for twice_mu in shell.kappa.twice_mu_values() {
                for radial_n in 0..(SPINOR_RADIAL_LO0 + shell.lo_count) {
                    if self.site_projection_row(site, shell.kappa, twice_mu, radial_n)
                        == Some(coordinate)
                    {
                        return Some((
                            DiracRadialId {
                                site,
                                kind: ProductOrbitalKind::Valence,
                                kappa: shell.kappa,
                                n: radial_n,
                            },
                            twice_mu,
                        ));
                    }
                }
            }
        }
        None
    }

    /// Site-projection coordinate of $(site,\kappa,2\mu,n)$ in the live
    /// APW-then-LO order: radial column fastest inside the APW $(P,\dot P)$
    /// block, then confined LO/RLO $n$ fastest. This is not an eigenbasis row.
    pub fn site_projection_row(
        &self,
        site: usize,
        kappa: Kappa,
        twice_mu: TwiceMu,
        radial_n: usize,
    ) -> Option<usize> {
        let radials = self.source.radials.get(site)?;
        let mu = mu_index(kappa, twice_mu)?;
        let shells = shells(radials);
        let shell = shells.iter().position(|item| item.kappa == kappa)?;
        if radial_n < SPINOR_RADIAL_LO0 {
            if radial_n > SPINOR_RADIAL_PDOT {
                return None;
            }
            let preceding_channels = shells[..shell]
                .iter()
                .map(|item| item.kappa.degeneracy() as usize)
                .sum::<usize>();
            return Some(2 * (preceding_channels + mu) + radial_n);
        }
        if radial_n - SPINOR_RADIAL_LO0 >= shells[shell].lo_count {
            return None;
        }
        let augmented_count = shells
            .iter()
            .map(|item| 2 * item.kappa.degeneracy() as usize)
            .sum::<usize>();
        let preceding_locals = shells[..shell]
            .iter()
            .map(|item| item.kappa.degeneracy() as usize * item.lo_count)
            .sum::<usize>();
        Some(
            augmented_count + preceding_locals + mu * shells[shell].lo_count + radial_n
                - SPINOR_RADIAL_LO0,
        )
    }

    pub(crate) fn validate(&self) -> Result<(), CheckpointPhysicsError> {
        self.source.validate()?;
        if self.source.q != self.source.interstitial_pair_support.q {
            return Err(CheckpointPhysicsError::SpinorProductTransferQMismatch);
        }
        let n_k = self.orbitals.k_fractional.len();
        if n_k == 0
            || self.orbitals.eigenvectors.len() != n_k
            || self.orbitals.energies.len() != n_k
            || self.orbitals.bases.len() != n_k
            || self.orbitals.available_bands.len() != n_k
            || self.k_minus_q.len() != n_k
            || self.pair_columns.n_k != n_k
        {
            return Err(CheckpointPhysicsError::SpinorProductKSliceMismatch);
        }
        if self.orbitals.band_window.start != 0
            || self.pair_columns.n_orb != self.orbitals.band_window.count
        {
            return Err(CheckpointPhysicsError::InconsistentBandCount);
        }
        let n_orb = self.orbitals.band_window.count;
        if n_orb == 0 {
            return Err(CheckpointPhysicsError::EmptyKPointSet);
        }
        let _ = self.pair_columns.n_columns()?;
        for k in 0..n_k {
            let eigenvectors = &self.orbitals.eigenvectors[k];
            let basis = &self.orbitals.bases[k];
            if eigenvectors.rows() != basis.layout.dimension()
                || eigenvectors.columns() != n_orb
                || self.orbitals.energies[k].len() != n_orb
                || self.orbitals.available_bands[k] < n_orb
                || self.k_minus_q[k].k_index != k
                || self.k_minus_q[k].kq_index >= n_k
            {
                return Err(CheckpointPhysicsError::SpinorProductKSliceMismatch);
            }
        }
        Ok(())
    }
}

/// Shared q-slice contract used by spinor THC, Coulomb, and MLDUMP.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpinorQSliceError {
    EmptySlice,
    IncompleteQSlice { actual: usize, expected: usize },
    IncompatibleInputs,
    NonFiniteQSlice,
    CanonicalQMismatch { q_index: usize },
    KMinusQWrap { q_index: usize, k_index: usize },
}

pub(crate) fn require_spinor_q_slice(
    inputs: &[SpinorProductInput],
) -> Result<&SpinorProductInput, SpinorQSliceError> {
    let first = inputs.first().ok_or(SpinorQSliceError::EmptySlice)?;
    first
        .validate()
        .map_err(|_| SpinorQSliceError::IncompatibleInputs)?;
    let n_k = first.orbitals.k_fractional.len();
    if inputs.len() != n_k {
        return Err(SpinorQSliceError::IncompleteQSlice {
            actual: inputs.len(),
            expected: n_k,
        });
    }
    for (iq, input) in inputs.iter().enumerate() {
        if iq > 0 {
            input
                .validate()
                .map_err(|_| SpinorQSliceError::IncompatibleInputs)?;
        }
        if input.orbitals != first.orbitals
            || input.pair_columns != first.pair_columns
            || input.source.partition != first.source.partition
            || input.source.radials != first.source.radials
            || input.core != first.core
            || input.reciprocal != first.reciprocal
            || input.k_minus_q.len() != n_k
        {
            return Err(SpinorQSliceError::IncompatibleInputs);
        }
        let k_fractional = first.orbitals.k_fractional.as_slice();
        validate_canonical_q_map(
            k_fractional,
            first.reciprocal,
            input.source.q.cartesian,
            iq,
            input
                .k_minus_q
                .iter()
                .map(|mapped| (mapped.k_index, mapped.kq_index, mapped.umklapp.index)),
        )
        .map_err(|error| match error {
            CanonicalQMapError::NonFiniteQSlice => SpinorQSliceError::NonFiniteQSlice,
            CanonicalQMapError::CanonicalQMismatch => {
                SpinorQSliceError::CanonicalQMismatch { q_index: iq }
            }
            CanonicalQMapError::IncompatibleMap => SpinorQSliceError::IncompatibleInputs,
            CanonicalQMapError::KMinusQWrap { k_index } => SpinorQSliceError::KMinusQWrap {
                q_index: iq,
                k_index,
            },
        })?;
    }
    Ok(first)
}

fn emit_spinor_product_input(
    physics: &CheckpointPhysics,
    bands: &CheckpointBandSolution,
    k_fractional: &[[f64; 3]],
    q: TransferQ,
    k_minus_q: Vec<SpinorKMinusQ>,
) -> Result<SpinorProductInput, CheckpointPhysicsError> {
    let n_k = k_fractional.len();
    let mut available_bands = Vec::with_capacity(n_k);
    let mut n_orb = None;
    let mut radials = None;
    for point in bands.points() {
        match &point.solution {
            CheckpointKPointSolution::Spinor {
                basis, solution, ..
            } => {
                match &radials {
                    None => radials = Some(site_radials(basis)?),
                    Some(expected) => require_same_radial_identity(expected, basis)?,
                }
                let bands_here = solution.eigenvectors.columns();
                available_bands.push(bands_here);
                n_orb = Some(n_orb.unwrap_or(bands_here).min(bands_here));
            }
            CheckpointKPointSolution::Collinear { .. } => {
                return Err(CheckpointPhysicsError::InconsistentRelativityRoute);
            }
        }
    }
    let n_orb = n_orb
        .filter(|&count| count > 0)
        .ok_or(CheckpointPhysicsError::EmptyKPointSet)?;
    let pair_columns = PairColumnLayout::new(n_k, n_orb, None);
    let _ = pair_columns.n_columns()?;

    let mut eigenvectors = Vec::with_capacity(n_k);
    let mut energies = Vec::with_capacity(n_k);
    let mut bases = Vec::with_capacity(n_k);
    for point in bands.points() {
        let CheckpointKPointSolution::Spinor {
            basis, solution, ..
        } = &point.solution
        else {
            return Err(CheckpointPhysicsError::InconsistentRelativityRoute);
        };
        eigenvectors.push(leading_bands(&solution.eigenvectors, n_orb)?);
        let mut values = solution.eigenvalues.clone();
        values.truncate(n_orb);
        energies.push(values);
        bases.push(basis.compiled.clone());
    }

    let interstitial_pair_support = raw_pair_support(q, *physics.reciprocal(), &bases, &k_minus_q)?;
    let source = DiracProductSource::new(
        AuxiliaryPartition::from_interstitial(physics.geometry().clone()),
        radials.ok_or(CheckpointPhysicsError::EmptyKPointSet)?,
        q,
        interstitial_pair_support,
        Provenance {
            recipe: None,
            reference: Some("checkpoint-dft-frozen-spinor-product-input".to_owned()),
        },
    )?;
    let input = SpinorProductInput {
        source,
        orbitals: SpinorFrozenOrbitals {
            k_fractional: k_fractional.to_vec(),
            eigenvectors,
            energies,
            bases,
            available_bands,
            band_window: SpinorBandWindow {
                start: 0,
                count: n_orb,
            },
        },
        k_minus_q,
        pair_columns,
        reciprocal: *physics.reciprocal(),
        core: SpinorCoreTable::default(),
    };
    input.validate()?;
    Ok(input)
}

pub(crate) struct SpinorPairSitePhases {
    pub left_bloch: Complex64,
    pub right_bloch: Complex64,
    pub auxiliary_compensation: Complex64,
}

pub(crate) fn spinor_pair_site_phases(
    input: &SpinorProductInput,
    mapped: SpinorKMinusQ,
    site: usize,
) -> Option<SpinorPairSitePhases> {
    let position = input.source.partition.sites().get(site)?.position;
    let left_k = fractional_reciprocal(
        *input.orbitals.k_fractional.get(mapped.kq_index)?,
        input.reciprocal,
    );
    let right_k = fractional_reciprocal(
        *input.orbitals.k_fractional.get(mapped.k_index)?,
        input.reciprocal,
    );
    Some(SpinorPairSitePhases {
        left_bloch: site_translation_phase(left_k, position),
        right_bloch: site_translation_phase(right_k, position),
        auxiliary_compensation: site_translation_phase(input.source.q.cartesian, position).conj(),
    })
}

fn fractional_reciprocal(
    fractional: [f64; 3],
    reciprocal: ReciprocalLattice,
) -> [muffintin_core::InverseBohr; 3] {
    std::array::from_fn(|axis| {
        muffintin_core::InverseBohr(
            fractional
                .iter()
                .zip(reciprocal.basis())
                .map(|(&coefficient, basis)| coefficient * basis[axis].get())
                .sum(),
        )
    })
}

fn raw_pair_support(
    q: TransferQ,
    reciprocal: ReciprocalLattice,
    bases: &[SpinorCompiledBasis],
    k_minus_q: &[SpinorKMinusQ],
) -> Result<RawInterstitialPairSupport, CheckpointPhysicsError> {
    let mut indices = BTreeSet::new();
    for mapped in k_minus_q {
        let right = &bases[mapped.k_index].plane_waves;
        let left = &bases[mapped.kq_index].plane_waves;
        let wrap = mapped.umklapp.index;
        for g_k in right {
            for g_kmq in left {
                indices.insert([
                    g_k.g.index[0] - g_kmq.g.index[0] + wrap[0],
                    g_k.g.index[1] - g_kmq.g.index[1] + wrap[1],
                    g_k.g.index[2] - g_kmq.g.index[2] + wrap[2],
                ]);
            }
        }
    }
    Ok(RawInterstitialPairSupport::from_relative_indices(
        q, reciprocal, indices,
    )?)
}

fn site_radials(
    basis: &SpinorIterationBasis,
) -> Result<Vec<DiracSiteRadialSet>, CheckpointPhysicsError> {
    basis
        .radial_sites
        .iter()
        .zip(&basis.density_sites)
        .map(|(radials, density)| {
            Ok(DiracSiteRadialSet {
                mesh: density.mesh.clone(),
                valence: valence_radials(radials),
                cores: Vec::new(),
            })
        })
        .collect()
}

fn valence_radials(site: &SpinorRadialSite) -> Vec<DiracRadial> {
    let mut valence = Vec::new();
    for (solution, locals) in site.solutions.iter().zip(&site.local_orbitals) {
        valence.push(DiracRadial {
            kappa: solution.kappa,
            n: SPINOR_RADIAL_P,
            samples: DiracRadialSamples {
                large: solution.p.clone(),
                small: solution.q.clone(),
            },
            normalization: DiracRadialNormalization::OnMesh,
        });
        valence.push(DiracRadial {
            kappa: solution.kappa,
            n: SPINOR_RADIAL_PDOT,
            samples: DiracRadialSamples {
                large: solution.energy_derivative.p.clone(),
                small: solution.energy_derivative.q.clone(),
            },
            normalization: DiracRadialNormalization::OnMesh,
        });
        for (ordinal, local) in locals.iter().enumerate() {
            valence.push(DiracRadial {
                kappa: solution.kappa,
                n: SPINOR_RADIAL_LO0 + ordinal,
                samples: DiracRadialSamples {
                    large: local.orbital.p.clone(),
                    small: local.orbital.q.clone(),
                },
                normalization: DiracRadialNormalization::OnMesh,
            });
        }
    }
    valence
}

fn require_same_radial_identity(
    expected: &[DiracSiteRadialSet],
    basis: &SpinorIterationBasis,
) -> Result<(), CheckpointPhysicsError> {
    let actual = site_radials(basis)?;
    if expected.len() != actual.len() {
        return Err(CheckpointPhysicsError::InconsistentRelativityRoute);
    }
    for (left, right) in expected.iter().zip(&actual) {
        if left.valence.len() != right.valence.len() {
            return Err(CheckpointPhysicsError::InconsistentRelativityRoute);
        }
        for (lhs, rhs) in left.valence.iter().zip(&right.valence) {
            if lhs.kappa != rhs.kappa || lhs.n != rhs.n {
                return Err(CheckpointPhysicsError::InconsistentRelativityRoute);
            }
        }
    }
    Ok(())
}

struct RadialShell {
    kappa: Kappa,
    lo_count: usize,
}

fn shells(radials: &DiracSiteRadialSet) -> Vec<RadialShell> {
    let mut shells: Vec<RadialShell> = Vec::new();
    for radial in &radials.valence {
        match shells.iter_mut().find(|shell| shell.kappa == radial.kappa) {
            Some(shell) => {
                if radial.n >= SPINOR_RADIAL_LO0 {
                    shell.lo_count += 1;
                }
            }
            None => shells.push(RadialShell {
                kappa: radial.kappa,
                lo_count: usize::from(radial.n >= SPINOR_RADIAL_LO0),
            }),
        }
    }
    shells
}

fn mu_index(kappa: Kappa, twice_mu: TwiceMu) -> Option<usize> {
    let twice_j = i64::from(kappa.twice_j());
    let twice_mu = twice_mu.get();
    if twice_mu < -twice_j || twice_mu > twice_j || (twice_mu + twice_j) % 2 != 0 {
        return None;
    }
    Some(((twice_mu + twice_j) / 2) as usize)
}
