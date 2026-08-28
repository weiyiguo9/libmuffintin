//! Dirac PP/QQ muffin-tin vertices on an existing scalar-charge auxiliary.
//!
//! The stored coefficient at auxiliary $(L,M)$ is the complex-harmonic
//! density expansion
//! $(-1)^M\langle\Omega|Y_{L,-M}|\Omega\rangle$ times the radial overlap
//! and the site phase $\exp(+i q\cdot R_a)$ once. PP uses $\Omega_\kappa$
//! on both bra and ket; QQ uses $\Omega_{-\kappa}$ on both. The matrix
//! element of $Y_{LM}$ is not stored in slot $M$.

use crate::MpbError;
use crate::dirac_construct::require_matching_dirac_source_and_raw;
use muffintin_auxiliary_ir::{
    CompiledAuxiliaryBasis, DiracChargeSector, DiracMtPairSpec, DiracPairVertex,
    DiracProductSource, DiracRadialId, DiracRawProductSpace,
};
use muffintin_core::{Lm, RelativisticChannel, RelativisticChannelError, spinor_gaunt};
use muffintin_envelope::site_translation_phase;
use num_complex::Complex64;

/// Accumulator that adds separate PP and QQ muffin-tin terms onto one layout.
///
/// Interstitial spinor plane-wave contraction is not implemented here.
#[derive(Debug)]
pub struct DiracPairVertexAccumulator<'a> {
    source: &'a DiracProductSource,
    raw: &'a DiracRawProductSpace,
    auxiliary: &'a CompiledAuxiliaryBasis,
    pair: DiracMtPairSpec,
    coefficients: Vec<Complex64>,
}

impl<'a> DiracPairVertexAccumulator<'a> {
    /// Start an empty vertex on a matching Dirac source / raw / auxiliary context.
    pub fn new(
        source: &'a DiracProductSource,
        raw: &'a DiracRawProductSpace,
        auxiliary: &'a CompiledAuxiliaryBasis,
        pair: DiracMtPairSpec,
    ) -> Result<Self, MpbError> {
        require_matching_dirac_context(source, raw, auxiliary)?;
        require_pair_orbitals(source, pair)?;
        Ok(Self {
            source,
            raw,
            auxiliary,
            pair,
            coefficients: vec![Complex64::default(); auxiliary.dimension()],
        })
    }

    /// Add PP ($\Omega_\kappa$) and QQ ($\Omega_{-\kappa}$) muffin-tin terms.
    pub fn add_muffin_tin(&mut self, amplitude: Complex64) -> Result<(), MpbError> {
        self.add_sector(DiracChargeSector::LargeLarge, amplitude)?;
        self.add_sector(DiracChargeSector::SmallSmall, amplitude)
    }

    /// Add only the large-large PP sector.
    pub fn add_pp(&mut self, amplitude: Complex64) -> Result<(), MpbError> {
        self.add_sector(DiracChargeSector::LargeLarge, amplitude)
    }

    /// Add only the small-small QQ sector.
    pub fn add_qq(&mut self, amplitude: Complex64) -> Result<(), MpbError> {
        self.add_sector(DiracChargeSector::SmallSmall, amplitude)
    }

    /// Seal the accumulated coefficients as a checked [`DiracPairVertex`].
    pub fn finish(self) -> Result<DiracPairVertex, MpbError> {
        Ok(DiracPairVertex::from_auxiliary(
            self.auxiliary,
            self.pair,
            self.coefficients,
        )?)
    }

    fn add_sector(
        &mut self,
        sector: DiracChargeSector,
        amplitude: Complex64,
    ) -> Result<(), MpbError> {
        add_sector(
            self.source,
            self.raw,
            self.auxiliary,
            self.pair,
            sector,
            amplitude,
            &mut self.coefficients,
        )
    }
}

/// Expand one Dirac muffin-tin pair with both PP and QQ sectors.
pub fn dirac_mt_pair_vertex(
    source: &DiracProductSource,
    raw: &DiracRawProductSpace,
    auxiliary: &CompiledAuxiliaryBasis,
    spec: DiracMtPairSpec,
) -> Result<DiracPairVertex, MpbError> {
    let mut acc = DiracPairVertexAccumulator::new(source, raw, auxiliary, spec)?;
    acc.add_muffin_tin(Complex64::new(1.0, 0.0))?;
    acc.finish()
}

/// Exact source/raw/auxiliary $q$, partition, pair support, and site meshes.
pub fn require_matching_dirac_context(
    source: &DiracProductSource,
    raw: &DiracRawProductSpace,
    auxiliary: &CompiledAuxiliaryBasis,
) -> Result<(), MpbError> {
    require_matching_dirac_source_and_raw(source, raw)?;
    auxiliary.validate()?;
    auxiliary.require_mixed_product()?;
    if auxiliary.q != source.q {
        return Err(MpbError::TransferQMismatch);
    }
    if auxiliary.partition != source.partition || auxiliary.partition != raw.partition {
        return Err(MpbError::PartitionMismatch);
    }
    let payload = auxiliary.require_mixed_product()?;
    if payload.interstitial.q != source.q {
        return Err(MpbError::TransferQMismatch);
    }
    if payload.sites.len() != source.radials.len() {
        return Err(MpbError::Product(
            muffintin_auxiliary_ir::ProductError::AuxiliarySiteCount {
                expected: source.radials.len(),
                actual: payload.sites.len(),
            },
        ));
    }
    for (site, (block, radials)) in payload.sites.iter().zip(&source.radials).enumerate() {
        if block.mesh != radials.mesh {
            return Err(MpbError::Product(
                muffintin_auxiliary_ir::ProductError::AuxiliaryMeshMismatch { site },
            ));
        }
    }
    Ok(())
}

fn require_pair_orbitals(
    source: &DiracProductSource,
    spec: DiracMtPairSpec,
) -> Result<(), MpbError> {
    if spec.left.site != spec.right.site {
        return Err(MpbError::DiracCrossSitePair);
    }
    find_orbital(source, spec.left)?;
    find_orbital(source, spec.right)?;
    pair_channels(spec)?;
    Ok(())
}

fn pair_channels(
    spec: DiracMtPairSpec,
) -> Result<(RelativisticChannel, RelativisticChannel), MpbError> {
    Ok((
        RelativisticChannel::new(spec.left.kappa, spec.left_twice_mu)
            .map_err(dirac_magnetic_error)?,
        RelativisticChannel::new(spec.right.kappa, spec.right_twice_mu)
            .map_err(dirac_magnetic_error)?,
    ))
}

fn dirac_magnetic_error(error: RelativisticChannelError) -> MpbError {
    match error {
        RelativisticChannelError::MuOutsideChannel {
            kappa,
            twice_mu,
            twice_j,
        } => MpbError::DiracMagneticQuantumNumber {
            kappa,
            twice_mu,
            twice_j,
        },
    }
}

fn find_orbital(
    source: &DiracProductSource,
    id: DiracRadialId,
) -> Result<&muffintin_auxiliary_ir::DiracRadial, MpbError> {
    source.find_radial(id).ok_or(MpbError::UnknownDiracOrbital {
        site: id.site,
        kind: id.kind,
        kappa: id.kappa.get(),
        n: id.n,
    })
}

fn add_sector(
    source: &DiracProductSource,
    raw: &DiracRawProductSpace,
    auxiliary: &CompiledAuxiliaryBasis,
    spec: DiracMtPairSpec,
    sector: DiracChargeSector,
    amplitude: Complex64,
    coefficients: &mut [Complex64],
) -> Result<(), MpbError> {
    if spec.left.site != spec.right.site {
        return Err(MpbError::DiracCrossSitePair);
    }
    let site = spec.left.site;
    find_orbital(source, spec.left)?;
    find_orbital(source, spec.right)?;
    if !raw
        .radial_products
        .iter()
        .any(|product| product.channel.sector == sector && pair_matches(product.channel, spec))
    {
        return Err(MpbError::UnknownDiracMtPair {
            left: spec.left,
            right: spec.right,
            sector,
        });
    }
    let mesh = auxiliary
        .site_mesh(site)
        .ok_or(MpbError::UnknownDiracOrbital {
            site,
            kind: spec.left.kind,
            kappa: spec.left.kappa.get(),
            n: spec.left.n,
        })?;
    let (left_channel, right_channel) = pair_channels(spec)?;
    let left_omega = sector.omega_channel(left_channel);
    let right_omega = sector.omega_channel(right_channel);
    let position = source.partition.sites()[site].position;
    let phase = site_translation_phase(source.q.cartesian, position);
    let block = auxiliary
        .require_mixed_product()?
        .sites
        .iter()
        .find(|block| block.site == site)
        .ok_or(MpbError::UnknownDiracOrbital {
            site,
            kind: spec.left.kind,
            kappa: spec.left.kappa.get(),
            n: spec.left.n,
        })?;
    for mode in &block.modes {
        let Some(product) = raw.radial_products.iter().find(|product| {
            product.channel.coupled_l == mode.l
                && product.channel.sector == sector
                && pair_matches(product.channel, spec)
        }) else {
            continue;
        };
        let integrand = product
            .samples
            .iter()
            .zip(&mode.radial)
            .map(|(sample, mode)| sample * mode)
            .collect::<Vec<_>>();
        let radial_overlap = mesh.integrate(&integrand)?;
        let l_i = mode.l as i32;
        for m in -l_i..=l_i {
            let Some(index) = auxiliary.mt_index(site, mode.l, m, mode.n) else {
                continue;
            };
            let angular = density_angular(left_omega, mode.l, m, right_omega);
            coefficients[index] +=
                amplitude * phase * Complex64::new(angular * radial_overlap, 0.0);
        }
    }
    Ok(())
}

fn pair_matches(stored: muffintin_auxiliary_ir::DiracPairChannel, spec: DiracMtPairSpec) -> bool {
    // Uniqueness of the unordered `(left,right)×sector×L` product is a
    // `DiracRawProductSpace::validate_internal` invariant.
    (stored.left == spec.left && stored.right == spec.right)
        || (stored.left == spec.right && stored.right == spec.left)
}

/// Stored density coefficient: $(-1)^M\langle\Omega|Y_{L,-M}|\Omega\rangle$.
fn density_angular(left: RelativisticChannel, l: u32, m: i32, right: RelativisticChannel) -> f64 {
    let field = Lm::new(l, -m).expect("auxiliary M lies in [-L, L]");
    magnetic_phase(m) * spinor_gaunt(left, field, right)
}

fn magnetic_phase(m: i32) -> f64 {
    if m.unsigned_abs() % 2 == 0 { 1.0 } else { -1.0 }
}
