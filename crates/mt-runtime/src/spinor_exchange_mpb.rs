//! Full rectangular core-sector spinor MPB vertices for one transfer q.

use crate::checkpoint_physics::CheckpointPhysicsError;
use crate::spinor_mpb::{SpinorFrozenInputIdentity, spinor_frozen_input_identity};
use crate::spinor_product::{
    SpinorCoreOrbital, SpinorKMinusQ, SpinorProductInput, spinor_pair_site_phases,
};
use muffintin_core::{InverseBohr, RelativisticChannel};
use muffintin_operators::{CompiledSiteProjection, OperatorError, SiteOrbitalCoefficients};
use muffintin_prodbasis::mpb::{
    DiracBlochVertexAccumulator, MpbError, apply_dirac_overlap_cutoff,
    untruncated_dirac_product_space,
};
use muffintin_prodbasis::{
    AuxiliaryIrError, CompiledAuxiliaryBasis, DiracChargeSector, DiracMtPairSpec,
    DiracProductSource, DiracRadial, DiracRadialNormalization, DiracRawProductSpace,
    ExchangePairLayout, ExchangeSpace, OrbitalPair, PairVertex, RawInterstitialPairSupport,
};
use num_complex::Complex64;
use std::collections::HashSet;
use std::f64::consts::PI;
use thiserror::Error;

/// SPEX mixed-product controls for all three core-member rectangular sectors.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorExchangeMpbSpec {
    pub product_l_max: u32,
    pub product_g_max: InverseBohr,
    pub overlap_tolerance: f64,
}

/// One vertex at a checked rectangular column.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorExchangeMpbPairVertex {
    pub column: usize,
    pub k: usize,
    pub occupied: usize,
    pub target: usize,
    pub vertex: PairVertex,
}

/// Complete column-major rectangular sector at one q.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorExchangeMpbSector {
    pub layout: ExchangePairLayout,
    pub vertices: Vec<SpinorExchangeMpbPairVertex>,
}

/// Gamma constant-mode measurement for one CV or VC column.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorGammaConstantModeDiagnostic {
    pub column: usize,
    /// `None` at every finite q.
    pub coupling: Option<Complex64>,
    /// Direct normalized MT overlap, independent of the auxiliary expansion.
    pub direct_overlap: Option<Complex64>,
    /// Absolute difference between `coupling` and `direct_overlap`.
    pub residual: Option<f64>,
}

/// CV/VC Gamma diagnostics. No sector energy or SCF contraction is published here.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorExchangeMpbDiagnostics {
    pub cv: Vec<SpinorGammaConstantModeDiagnostic>,
    pub vc: Vec<SpinorGammaConstantModeDiagnostic>,
    /// Maximum direct-overlap error and paired CV/VC conjugacy error at Gamma.
    pub max_residual: Option<f64>,
}

/// Exact MPB core-sector oracle for one q.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorExchangeMpbResult {
    pub raw: DiracRawProductSpace,
    pub auxiliary: CompiledAuxiliaryBasis,
    pub cv: SpinorExchangeMpbSector,
    pub vc: SpinorExchangeMpbSector,
    pub cc: SpinorExchangeMpbSector,
    pub diagnostics: SpinorExchangeMpbDiagnostics,
    frozen_input: SpinorFrozenInputIdentity,
}

impl SpinorExchangeMpbResult {
    pub(crate) const fn frozen_input_identity(&self) -> &SpinorFrozenInputIdentity {
        &self.frozen_input
    }
}

#[derive(Debug, Error)]
pub enum SpinorExchangeMpbError {
    #[error(transparent)]
    Input(#[from] CheckpointPhysicsError),
    #[error(transparent)]
    Product(#[from] AuxiliaryIrError),
    #[error(transparent)]
    Mpb(#[from] MpbError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error("rectangular core-sector MPB requires at least one MuResolved core spin orbital")]
    EmptyCore,
    #[error("spinor core table and Dirac core-radial source are inconsistent")]
    IncompatibleCoreTable,
    #[error("spinor rectangular pair context is inconsistent with the frozen q slice")]
    IncompatiblePairContext,
}

/// Build every CV, VC, and CC column over one common core-member MPB.
///
/// Occupations are copied only in [`crate::SpinorCoreTable`] and never enter
/// radial products or vertices. The core-sector raw interstitial support is
/// empty, so every returned vertex has an identically zero interstitial block.
pub fn build_spinor_exchange_mpb(
    input: &SpinorProductInput,
    spec: &SpinorExchangeMpbSpec,
) -> Result<SpinorExchangeMpbResult, SpinorExchangeMpbError> {
    input.validate()?;
    if input.core.orbitals.is_empty() {
        return Err(SpinorExchangeMpbError::EmptyCore);
    }
    require_core_table(input)?;

    let mut source = input.source.clone();
    source.interstitial_pair_support = RawInterstitialPairSupport::empty(source.q);
    source.validate().map_err(MpbError::from)?;
    let raw = untruncated_dirac_product_space(&source, spec.product_l_max)?;
    let auxiliary = apply_dirac_overlap_cutoff(
        &raw,
        &source,
        spec.overlap_tolerance,
        1.0,
        &input.reciprocal,
        spec.product_g_max,
    )?;

    let n_k = input.orbitals.k_fractional.len();
    let n_valence = input.orbitals.band_window.count;
    let n_core = input.core.orbitals.len();
    let cv_layout = ExchangePairLayout::new(
        ExchangeSpace::Core,
        ExchangeSpace::Valence,
        n_k,
        n_core,
        n_valence,
    );
    let vc_layout = ExchangePairLayout::new(
        ExchangeSpace::Valence,
        ExchangeSpace::Core,
        n_k,
        n_valence,
        n_core,
    );
    let cc_layout = ExchangePairLayout::new(
        ExchangeSpace::Core,
        ExchangeSpace::Core,
        n_k,
        n_core,
        n_core,
    );
    let _ = cv_layout.n_columns()?;
    let _ = vc_layout.n_columns()?;
    let _ = cc_layout.n_columns()?;

    let known_pp = raw_mt_pairs(&raw, DiracChargeSector::LargeLarge);
    let known_qq = raw_mt_pairs(&raw, DiracChargeSector::SmallSmall);
    let gamma = is_gamma(&source);
    let mut cv_vertices = Vec::with_capacity(cv_layout.n_columns()?);
    let mut vc_vertices = Vec::with_capacity(vc_layout.n_columns()?);
    let mut cc_vertices = Vec::with_capacity(cc_layout.n_columns()?);
    let mut cv_diagnostics = Vec::with_capacity(cv_layout.n_columns()?);
    let mut vc_diagnostics = Vec::with_capacity(vc_layout.n_columns()?);

    for mapped in input.k_minus_q.iter().copied() {
        let projected = project_k_sites(input, mapped)?;
        for (core_index, core) in input.core.orbitals.iter().enumerate() {
            for target in 0..n_valence {
                let column = cv_layout.encode(mapped.k_index, core_index, target)?;
                let pair = exchange_pair(cv_layout, mapped.k_index, core_index, target);
                let mut acc = DiracBlochVertexAccumulator::new(&source, &raw, &auxiliary, pair)?;
                let direct = add_cv(
                    &mut acc,
                    &source,
                    input,
                    mapped,
                    core,
                    &projected[core.site_index].right,
                    target,
                    &known_pp,
                    &known_qq,
                )?;
                let vertex = acc.finish()?;
                cv_diagnostics.push(gamma_diagnostic(
                    gamma, column, &auxiliary, &vertex, direct,
                )?);
                cv_vertices.push(SpinorExchangeMpbPairVertex {
                    column,
                    k: mapped.k_index,
                    occupied: core_index,
                    target,
                    vertex,
                });
            }
        }
        for occupied in 0..n_valence {
            for (core_index, core) in input.core.orbitals.iter().enumerate() {
                let column = vc_layout.encode(mapped.k_index, occupied, core_index)?;
                let pair = exchange_pair(vc_layout, mapped.k_index, occupied, core_index);
                let mut acc = DiracBlochVertexAccumulator::new(&source, &raw, &auxiliary, pair)?;
                let direct = add_vc(
                    &mut acc,
                    &source,
                    input,
                    mapped,
                    &projected[core.site_index].left,
                    occupied,
                    core,
                    &known_pp,
                    &known_qq,
                )?;
                let vertex = acc.finish()?;
                vc_diagnostics.push(gamma_diagnostic(
                    gamma, column, &auxiliary, &vertex, direct,
                )?);
                vc_vertices.push(SpinorExchangeMpbPairVertex {
                    column,
                    k: mapped.k_index,
                    occupied,
                    target: core_index,
                    vertex,
                });
            }
        }
        for (occupied, left) in input.core.orbitals.iter().enumerate() {
            for (target, right) in input.core.orbitals.iter().enumerate() {
                let column = cc_layout.encode(mapped.k_index, occupied, target)?;
                let pair = exchange_pair(cc_layout, mapped.k_index, occupied, target);
                let mut acc = DiracBlochVertexAccumulator::new(&source, &raw, &auxiliary, pair)?;
                add_cc(&mut acc, input, mapped, left, right, &known_pp, &known_qq)?;
                cc_vertices.push(SpinorExchangeMpbPairVertex {
                    column,
                    k: mapped.k_index,
                    occupied,
                    target,
                    vertex: acc.finish()?,
                });
            }
        }
    }

    let max_residual =
        gamma.then(|| gamma_max_residual(&cv_layout, &vc_layout, &cv_diagnostics, &vc_diagnostics));
    Ok(SpinorExchangeMpbResult {
        raw,
        auxiliary,
        cv: SpinorExchangeMpbSector {
            layout: cv_layout,
            vertices: cv_vertices,
        },
        vc: SpinorExchangeMpbSector {
            layout: vc_layout,
            vertices: vc_vertices,
        },
        cc: SpinorExchangeMpbSector {
            layout: cc_layout,
            vertices: cc_vertices,
        },
        diagnostics: SpinorExchangeMpbDiagnostics {
            cv: cv_diagnostics,
            vc: vc_diagnostics,
            max_residual,
        },
        frozen_input: spinor_frozen_input_identity(input),
    })
}

struct ProjectedSitePair {
    left: SiteOrbitalCoefficients,
    right: SiteOrbitalCoefficients,
}

fn project_k_sites(
    input: &SpinorProductInput,
    mapped: SpinorKMinusQ,
) -> Result<Vec<ProjectedSitePair>, SpinorExchangeMpbError> {
    let left_basis = &input.orbitals.bases[mapped.kq_index];
    let right_basis = &input.orbitals.bases[mapped.k_index];
    let left_ev = &input.orbitals.eigenvectors[mapped.kq_index];
    let right_ev = &input.orbitals.eigenvectors[mapped.k_index];
    let mut sites = Vec::with_capacity(input.source.partition.site_count());
    for site in 0..input.source.partition.site_count() {
        let left_channels = site_channels(left_basis, site)?;
        let right_channels = site_channels(right_basis, site)?;
        let left = CompiledSiteProjection::spinor(left_basis, site, left_channels)?
            .project_eigenvectors(left_ev)?;
        let right = CompiledSiteProjection::spinor(right_basis, site, right_channels)?
            .project_eigenvectors(right_ev)?;
        sites.push(ProjectedSitePair { left, right });
    }
    Ok(sites)
}

fn site_channels(
    compiled: &muffintin_operators::lapw::SpinorCompiledBasis,
    site: usize,
) -> Result<&[RelativisticChannel], SpinorExchangeMpbError> {
    compiled
        .site_augmentations
        .get(site)
        .and_then(|waves| waves.first())
        .map(|wave| wave.channels.as_slice())
        .ok_or(SpinorExchangeMpbError::IncompatiblePairContext)
}

fn exchange_pair(
    layout: ExchangePairLayout,
    k_index: usize,
    occupied: usize,
    target: usize,
) -> OrbitalPair {
    OrbitalPair::Exchange {
        k_index,
        occupied_space: layout.occupied_space,
        occupied,
        target_space: layout.target_space,
        target,
    }
}

#[allow(clippy::too_many_arguments)]
fn add_cv(
    acc: &mut DiracBlochVertexAccumulator<'_>,
    source: &DiracProductSource,
    input: &SpinorProductInput,
    mapped: SpinorKMinusQ,
    core: &SpinorCoreOrbital,
    right: &SiteOrbitalCoefficients,
    target: usize,
    known_pp: &HashSet<(
        muffintin_prodbasis::DiracRadialId,
        muffintin_prodbasis::DiracRadialId,
    )>,
    known_qq: &HashSet<(
        muffintin_prodbasis::DiracRadialId,
        muffintin_prodbasis::DiracRadialId,
    )>,
) -> Result<Complex64, SpinorExchangeMpbError> {
    let phases = spinor_pair_site_phases(input, mapped, core.site_index)
        .ok_or(SpinorExchangeMpbError::IncompatiblePairContext)?;
    let mut direct = Complex64::default();
    for coordinate in 0..right.coordinate_count() {
        let (right_id, right_mu) = input
            .site_projection_identity(core.site_index, coordinate)
            .ok_or(SpinorExchangeMpbError::IncompatiblePairContext)?;
        let pair = DiracMtPairSpec {
            left: core.radial,
            left_twice_mu: core.twice_mu,
            right: right_id,
            right_twice_mu: right_mu,
        };
        let amplitude =
            phases.left_bloch.conj() * right.at(coordinate, target) * phases.auxiliary_compensation;
        add_pair(acc, pair, amplitude, known_pp, known_qq)?;
        direct += amplitude * direct_overlap(source, pair)?;
    }
    Ok(direct)
}

#[allow(clippy::too_many_arguments)]
fn add_vc(
    acc: &mut DiracBlochVertexAccumulator<'_>,
    source: &DiracProductSource,
    input: &SpinorProductInput,
    mapped: SpinorKMinusQ,
    left: &SiteOrbitalCoefficients,
    occupied: usize,
    core: &SpinorCoreOrbital,
    known_pp: &HashSet<(
        muffintin_prodbasis::DiracRadialId,
        muffintin_prodbasis::DiracRadialId,
    )>,
    known_qq: &HashSet<(
        muffintin_prodbasis::DiracRadialId,
        muffintin_prodbasis::DiracRadialId,
    )>,
) -> Result<Complex64, SpinorExchangeMpbError> {
    let phases = spinor_pair_site_phases(input, mapped, core.site_index)
        .ok_or(SpinorExchangeMpbError::IncompatiblePairContext)?;
    let mut direct = Complex64::default();
    for coordinate in 0..left.coordinate_count() {
        let (left_id, left_mu) = input
            .site_projection_identity(core.site_index, coordinate)
            .ok_or(SpinorExchangeMpbError::IncompatiblePairContext)?;
        let pair = DiracMtPairSpec {
            left: left_id,
            left_twice_mu: left_mu,
            right: core.radial,
            right_twice_mu: core.twice_mu,
        };
        let amplitude = left.at(coordinate, occupied).conj()
            * phases.right_bloch
            * phases.auxiliary_compensation;
        add_pair(acc, pair, amplitude, known_pp, known_qq)?;
        direct += amplitude * direct_overlap(source, pair)?;
    }
    Ok(direct)
}

fn add_cc(
    acc: &mut DiracBlochVertexAccumulator<'_>,
    input: &SpinorProductInput,
    mapped: SpinorKMinusQ,
    left: &SpinorCoreOrbital,
    right: &SpinorCoreOrbital,
    known_pp: &HashSet<(
        muffintin_prodbasis::DiracRadialId,
        muffintin_prodbasis::DiracRadialId,
    )>,
    known_qq: &HashSet<(
        muffintin_prodbasis::DiracRadialId,
        muffintin_prodbasis::DiracRadialId,
    )>,
) -> Result<(), SpinorExchangeMpbError> {
    let Some((pair, amplitude)) = cc_pair_term(left, right, |site| {
        let phases = spinor_pair_site_phases(input, mapped, site)
            .ok_or(SpinorExchangeMpbError::IncompatiblePairContext)?;
        Ok((
            phases.left_bloch,
            phases.right_bloch,
            phases.auxiliary_compensation,
        ))
    })?
    else {
        return Ok(());
    };
    add_pair(acc, pair, amplitude, known_pp, known_qq)
}

fn cc_pair_term(
    left: &SpinorCoreOrbital,
    right: &SpinorCoreOrbital,
    phases: impl FnOnce(usize) -> Result<(Complex64, Complex64, Complex64), SpinorExchangeMpbError>,
) -> Result<Option<(DiracMtPairSpec, Complex64)>, SpinorExchangeMpbError> {
    if left.site_index != right.site_index {
        return Ok(None);
    }
    let (left_bloch, right_bloch, auxiliary_compensation) = phases(left.site_index)?;
    Ok(Some((
        DiracMtPairSpec {
            left: left.radial,
            left_twice_mu: left.twice_mu,
            right: right.radial,
            right_twice_mu: right.twice_mu,
        },
        left_bloch.conj() * right_bloch * auxiliary_compensation,
    )))
}

fn add_pair(
    acc: &mut DiracBlochVertexAccumulator<'_>,
    pair: DiracMtPairSpec,
    amplitude: Complex64,
    known_pp: &HashSet<(
        muffintin_prodbasis::DiracRadialId,
        muffintin_prodbasis::DiracRadialId,
    )>,
    known_qq: &HashSet<(
        muffintin_prodbasis::DiracRadialId,
        muffintin_prodbasis::DiracRadialId,
    )>,
) -> Result<(), SpinorExchangeMpbError> {
    if known_pp.contains(&(pair.left, pair.right)) {
        acc.add_pp(pair, amplitude)?;
    }
    if known_qq.contains(&(pair.left, pair.right)) {
        acc.add_qq(pair, amplitude)?;
    }
    Ok(())
}

fn direct_overlap(
    source: &DiracProductSource,
    pair: DiracMtPairSpec,
) -> Result<Complex64, SpinorExchangeMpbError> {
    if pair.left.kappa != pair.right.kappa || pair.left_twice_mu != pair.right_twice_mu {
        return Ok(Complex64::default());
    }
    let left = source
        .find_radial(pair.left)
        .ok_or(SpinorExchangeMpbError::IncompatiblePairContext)?;
    let right = source
        .find_radial(pair.right)
        .ok_or(SpinorExchangeMpbError::IncompatiblePairContext)?;
    let mesh = &source.radials[pair.left.site].mesh;
    let integrand = left
        .samples
        .large
        .iter()
        .zip(&left.samples.small)
        .zip(&right.samples.large)
        .zip(&right.samples.small)
        .map(|(((lp, lq), rp), rq)| lp * rp + lq * rq)
        .collect::<Vec<_>>();
    let scale = (radial_norm(mesh, left)? * radial_norm(mesh, right)?).sqrt();
    let overlap = mesh.integrate(&integrand).map_err(MpbError::from)?;
    Ok(Complex64::new(overlap / scale, 0.0))
}

fn radial_norm(
    mesh: &muffintin_core::ExponentialMesh,
    radial: &DiracRadial,
) -> Result<f64, SpinorExchangeMpbError> {
    match radial.normalization {
        DiracRadialNormalization::Explicit(value) => Ok(value),
        DiracRadialNormalization::OnMesh => {
            let integrand = radial
                .samples
                .large
                .iter()
                .zip(&radial.samples.small)
                .map(|(large, small)| large * large + small * small)
                .collect::<Vec<_>>();
            Ok(mesh.integrate(&integrand).map_err(MpbError::from)?)
        }
    }
}

fn raw_mt_pairs(
    raw: &DiracRawProductSpace,
    sector: DiracChargeSector,
) -> HashSet<(
    muffintin_prodbasis::DiracRadialId,
    muffintin_prodbasis::DiracRadialId,
)> {
    raw.radial_products
        .iter()
        .filter(|product| product.channel.sector == sector)
        .flat_map(|product| {
            [
                (product.channel.left, product.channel.right),
                (product.channel.right, product.channel.left),
            ]
        })
        .collect()
}

fn gamma_diagnostic(
    gamma: bool,
    column: usize,
    auxiliary: &CompiledAuxiliaryBasis,
    vertex: &PairVertex,
    direct: Complex64,
) -> Result<SpinorGammaConstantModeDiagnostic, SpinorExchangeMpbError> {
    if !gamma {
        return Ok(SpinorGammaConstantModeDiagnostic {
            column,
            coupling: None,
            direct_overlap: None,
            residual: None,
        });
    }
    let coupling = constant_mode_coupling(auxiliary, vertex)?;
    Ok(SpinorGammaConstantModeDiagnostic {
        column,
        coupling: Some(coupling),
        direct_overlap: Some(direct),
        residual: Some((coupling - direct).norm()),
    })
}

fn constant_mode_coupling(
    auxiliary: &CompiledAuxiliaryBasis,
    vertex: &PairVertex,
) -> Result<Complex64, SpinorExchangeMpbError> {
    let payload = auxiliary.require_mixed_product()?;
    let mut coupling = Complex64::default();
    for block in &payload.sites {
        for mode in block.modes.iter().filter(|mode| mode.l == 0) {
            let index = auxiliary
                .mt_index(block.site, 0, 0, mode.n)
                .ok_or(SpinorExchangeMpbError::IncompatiblePairContext)?;
            let integrand = mode
                .radial
                .iter()
                .zip(block.mesh.radii())
                .map(|(radial, radius)| radial * radius.get())
                .collect::<Vec<_>>();
            let constant_projection =
                (4.0 * PI).sqrt() * block.mesh.integrate(&integrand).map_err(MpbError::from)?;
            coupling += vertex.coefficients()[index] * constant_projection;
        }
    }
    Ok(coupling)
}

fn gamma_max_residual(
    cv_layout: &ExchangePairLayout,
    vc_layout: &ExchangePairLayout,
    cv: &[SpinorGammaConstantModeDiagnostic],
    vc: &[SpinorGammaConstantModeDiagnostic],
) -> f64 {
    let mut maximum = cv
        .iter()
        .chain(vc)
        .filter_map(|diagnostic| diagnostic.residual)
        .fold(0.0, f64::max);
    for diagnostic in cv {
        let Ok((k, core, valence)) = cv_layout.decode(diagnostic.column) else {
            continue;
        };
        let Ok(reverse) = vc_layout.encode(k, valence, core) else {
            continue;
        };
        let Some(reverse) = vc.iter().find(|diagnostic| diagnostic.column == reverse) else {
            continue;
        };
        if let (Some(forward), Some(reverse)) = (diagnostic.coupling, reverse.coupling) {
            maximum = maximum.max((forward - reverse.conj()).norm());
        }
    }
    maximum
}

fn is_gamma(source: &DiracProductSource) -> bool {
    source.q.cartesian.iter().all(|value| value.get() == 0.0) && source.q.umklapp.index == [0; 3]
}

fn require_core_table(input: &SpinorProductInput) -> Result<(), SpinorExchangeMpbError> {
    for core in &input.core.orbitals {
        if core.site_index != core.radial.site
            || input.source.find_radial(core.radial).is_none()
            || core.radial.kind != muffintin_prodbasis::ProductOrbitalKind::Core
        {
            return Err(SpinorExchangeMpbError::IncompatibleCoreTable);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::{
        Bohr, ExponentialMesh, Hartree, InterstitialGeometry, Kappa, Sphere, TwiceMu, VolumeBohr3,
    };
    use muffintin_envelope::Provenance;
    use muffintin_prodbasis::{
        AuxiliaryInterstitialSupport, AuxiliaryPartition, AuxiliaryRepresentation, DiracRadialId,
        ExchangeSpace, MixedProductAuxiliary, MtAuxiliaryMode, ProductOrbitalKind,
        SiteAuxiliaryBlock, TransferQ,
    };

    fn core(site: usize) -> SpinorCoreOrbital {
        SpinorCoreOrbital {
            site_index: site,
            n: 1,
            kappa: Kappa::new(-1).unwrap(),
            twice_mu: TwiceMu::new(-1).unwrap(),
            occupation: 1.0,
            radial: DiracRadialId {
                site,
                kind: ProductOrbitalKind::Core,
                kappa: Kappa::new(-1).unwrap(),
                n: 1,
            },
            energy: Hartree(-1.0),
            norm_total: 1.0,
            norm_mt: 1.0,
            spill: 0.0,
        }
    }

    #[test]
    fn different_site_cc_emits_no_pair_term_for_the_zero_vertex() {
        let term = cc_pair_term(&core(0), &core(1), |_| {
            panic!("cross-site CC must not request a Bloch/site phase")
        })
        .unwrap();
        assert!(term.is_none());
    }

    #[test]
    fn gamma_constant_mode_sums_every_retained_l0_radial() {
        let first = 1.0e-4_f64;
        let point_count = 31;
        let increment = (0.8 / first).ln() / (point_count - 1) as f64;
        let mesh = ExponentialMesh::new(Bohr(first), increment, point_count).unwrap();
        let radius = mesh.last().get();
        let constant_norm = (radius.powi(3) / 3.0).sqrt();
        let constant = mesh
            .radii()
            .iter()
            .map(|radius| radius.get() / constant_norm)
            .collect::<Vec<_>>();
        let second = mesh
            .radii()
            .iter()
            .map(|sample| sample.get() * (1.0 - 0.4 * sample.get() / radius))
            .collect::<Vec<_>>();
        let q = TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
        let auxiliary = CompiledAuxiliaryBasis {
            partition: AuxiliaryPartition::from_interstitial(
                InterstitialGeometry::new(
                    VolumeBohr3(64.0),
                    vec![Sphere {
                        center: [Bohr(0.0); 3],
                        radius: Bohr(radius),
                    }],
                )
                .unwrap(),
            ),
            q,
            representation: AuxiliaryRepresentation::MixedProduct(MixedProductAuxiliary {
                sites: vec![SiteAuxiliaryBlock {
                    site: 0,
                    mesh: mesh.clone(),
                    modes: vec![
                        MtAuxiliaryMode {
                            l: 0,
                            n: 0,
                            radial: constant,
                        },
                        MtAuxiliaryMode {
                            l: 0,
                            n: 1,
                            radial: second,
                        },
                    ],
                }],
                interstitial: AuxiliaryInterstitialSupport {
                    q,
                    g_cut: InverseBohr(0.0),
                    waves: Vec::new(),
                },
                cutoff: None,
            }),
            provenance: Provenance::default(),
        };
        auxiliary.validate().unwrap();
        let coefficients = [0.3, -0.7];
        let vertex = PairVertex::from_auxiliary(
            &auxiliary,
            OrbitalPair::Exchange {
                k_index: 0,
                occupied_space: ExchangeSpace::Core,
                occupied: 0,
                target_space: ExchangeSpace::Valence,
                target: 0,
            },
            coefficients
                .into_iter()
                .map(|value| Complex64::new(value, 0.0))
                .collect(),
        )
        .unwrap();
        let modes = &auxiliary.require_mixed_product().unwrap().sites[0].modes;
        let reconstructed = (0..mesh.len())
            .map(|index| {
                coefficients[0] * modes[0].radial[index] + coefficients[1] * modes[1].radial[index]
            })
            .collect::<Vec<_>>();
        let direct_integrand = reconstructed
            .iter()
            .zip(mesh.radii())
            .map(|(radial, radius)| radial * radius.get())
            .collect::<Vec<_>>();
        let direct = (4.0 * PI).sqrt() * mesh.integrate(&direct_integrand).unwrap();
        let complete = constant_mode_coupling(&auxiliary, &vertex).unwrap();
        let old_n0_only = coefficients[0] * (4.0 * PI * radius.powi(3) / 3.0).sqrt();
        assert!((complete - Complex64::new(direct, 0.0)).norm() < 1.0e-12);
        assert!((complete.re - old_n0_only).abs() > 1.0e-6);
    }
}
