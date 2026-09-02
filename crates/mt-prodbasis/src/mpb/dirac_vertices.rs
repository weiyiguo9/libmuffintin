//! Dirac PP/QQ muffin-tin vertices on an existing scalar-charge auxiliary.
//!
//! The stored coefficient at auxiliary $(L,M)$ is the complex-harmonic
//! density expansion
//! $(-1)^M\langle\Omega|Y_{L,-M}|\Omega\rangle$ times the radial overlap
//! and the site phase $\exp(+i q\cdot R_a)$ once. PP uses $\Omega_\kappa$
//! on both bra and ket; QQ uses $\Omega_{-\kappa}$ on both. The matrix
//! element of $Y_{LM}$ is not stored in slot $M$.

use crate::mpb::MpbError;
use crate::mpb::dirac_construct::require_matching_dirac_source_and_raw;
use crate::mpb::interstitial::add_raw_support_theta_i;
use crate::{
    CompiledAuxiliaryBasis, DiracChargeSector, DiracMtPairSpec, DiracPairVertex,
    DiracProductSource, DiracRadialId, DiracRawProductSpace, InterstitialPairSpec, OrbitalPair,
    PairVertex,
};
use muffintin_core::{Lm, RelativisticChannel, RelativisticChannelError, spinor_gaunt};
use muffintin_envelope::site_translation_phase;
use num_complex::Complex64;
use std::collections::HashMap;

/// Accumulator that adds separate PP and QQ muffin-tin terms onto one layout.
///
/// Interstitial spinor plane-wave contraction is not implemented here.
#[derive(Debug)]
pub struct DiracPairVertexAccumulator<'a> {
    source: &'a DiracProductSource,
    auxiliary: &'a CompiledAuxiliaryBasis,
    table: DiracMtSectorTable<'a>,
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
            auxiliary,
            table: DiracMtSectorTable::new(raw, auxiliary),
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
            &mut self.table,
            self.pair,
            sector,
            amplitude,
            &mut self.coefficients,
        )
    }
}

/// Accumulator that finishes a Dirac muffin-tin plus interstitial expansion
/// into a generic checked [`PairVertex`].
///
/// Muffin-tin PP/QQ terms reuse the Dirac mixed-product primitive, including the site phase
/// $\exp(+i q\cdot R_a)$ once. Interstitial terms reuse the shared raw-support
/// $\Theta_I$ helper. Construction accepts [`OrbitalPair::Bloch`] and
/// rectangular [`OrbitalPair::Exchange`] identities.
#[derive(Debug)]
pub struct DiracBlochVertexAccumulator<'a> {
    source: &'a DiracProductSource,
    raw: &'a DiracRawProductSpace,
    auxiliary: &'a CompiledAuxiliaryBasis,
    pair: OrbitalPair,
    coefficients: Vec<Complex64>,
}

impl<'a> DiracBlochVertexAccumulator<'a> {
    /// Radial-overlap table this accumulator's PP and QQ terms expect.
    ///
    /// Build it once per q slice and pass it to every column; the table is
    /// independent of the band pair.
    pub fn sector_table(
        raw: &'a DiracRawProductSpace,
        auxiliary: &'a CompiledAuxiliaryBasis,
    ) -> DiracMtSectorTable<'a> {
        DiracMtSectorTable::new(raw, auxiliary)
    }
}

impl<'a> DiracBlochVertexAccumulator<'a> {
    /// Start an empty Bloch vertex on a matching Dirac source / raw / auxiliary context.
    pub fn new(
        source: &'a DiracProductSource,
        raw: &'a DiracRawProductSpace,
        auxiliary: &'a CompiledAuxiliaryBasis,
        pair: OrbitalPair,
    ) -> Result<Self, MpbError> {
        require_matching_dirac_context(source, raw, auxiliary)?;
        if !matches!(
            pair,
            OrbitalPair::Bloch { .. } | OrbitalPair::Exchange { .. }
        ) {
            return Err(MpbError::ExpectedDiracBlochPair);
        }
        Ok(Self {
            source,
            raw,
            auxiliary,
            pair,
            coefficients: vec![Complex64::default(); auxiliary.dimension()],
        })
    }

    /// Add PP ($\Omega_\kappa$) muffin-tin terms for one radial-factor pair.
    ///
    /// `table` must be bound to this accumulator's raw product space and
    /// auxiliary; see [`Self::sector_table`].
    pub fn add_pp(
        &mut self,
        table: &mut DiracMtSectorTable<'_>,
        spec: DiracMtPairSpec,
        amplitude: Complex64,
    ) -> Result<(), MpbError> {
        require_pair_orbitals(self.source, spec)?;
        self.require_table(table)?;
        add_sector(
            self.source,
            table,
            spec,
            DiracChargeSector::LargeLarge,
            amplitude,
            &mut self.coefficients,
        )
    }

    /// Add QQ ($\Omega_{-\kappa}$) muffin-tin terms for one radial-factor pair.
    ///
    /// `table` must be bound to this accumulator's raw product space and
    /// auxiliary; see [`Self::sector_table`].
    pub fn add_qq(
        &mut self,
        table: &mut DiracMtSectorTable<'_>,
        spec: DiracMtPairSpec,
        amplitude: Complex64,
    ) -> Result<(), MpbError> {
        require_pair_orbitals(self.source, spec)?;
        self.require_table(table)?;
        add_sector(
            self.source,
            table,
            spec,
            DiracChargeSector::SmallSmall,
            amplitude,
            &mut self.coefficients,
        )
    }

    fn require_table(&self, table: &DiracMtSectorTable<'_>) -> Result<(), MpbError> {
        if table.auxiliary.q != self.auxiliary.q || table.raw.q != self.raw.q {
            return Err(MpbError::TransferQMismatch);
        }
        Ok(())
    }

    /// Add one interstitial pair expansion through shared $\Theta_I$.
    ///
    /// Coefficients are
    /// $A\Theta_I(G_{\mathrm{aux}}-G_{\mathrm{wrap}}-G_{\mathrm{rel}})$.
    /// `g_relative` must exist on the raw pair support. Global
    /// [`crate::TransferQ::umklapp`] enters the $\Theta_I$
    /// argument only.
    pub fn add_interstitial(&mut self, spec: InterstitialPairSpec) -> Result<(), MpbError> {
        add_raw_support_theta_i(
            &self.raw.interstitial_pair_support,
            self.auxiliary,
            spec,
            &mut self.coefficients,
        )
    }

    /// Seal the accumulated coefficients as a checked [`PairVertex`].
    pub fn finish(self) -> Result<PairVertex, MpbError> {
        Ok(PairVertex::from_auxiliary(
            self.auxiliary,
            self.pair,
            self.coefficients,
        )?)
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
            crate::AuxiliaryIrError::AuxiliarySiteCount {
                expected: source.radials.len(),
                actual: payload.sites.len(),
            },
        ));
    }
    for (site, (block, radials)) in payload.sites.iter().zip(&source.radials).enumerate() {
        if block.mesh != radials.mesh {
            return Err(MpbError::Product(
                crate::AuxiliaryIrError::AuxiliaryMeshMismatch { site },
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
) -> Result<&crate::DiracRadial, MpbError> {
    source.find_radial(id).ok_or(MpbError::UnknownDiracOrbital {
        site: id.site,
        kind: id.kind,
        kappa: id.kappa.get(),
        n: id.n,
    })
}

/// One retained auxiliary mode that receives a muffin-tin sector contribution.
///
/// `overlap` and `indices` depend only on the radial pair channel and the
/// compiled auxiliary, never on the band pair or the magnetic projections.
#[derive(Clone, Debug, PartialEq)]
struct DiracMtModeTerm {
    l: u32,
    overlap: f64,
    /// Auxiliary index per $M=-L,\ldots,L$; `None` where the mode is absent.
    indices: Vec<Option<usize>>,
}

/// Radial overlaps and auxiliary indices shared by every band pair at one $q$.
///
/// The stored coefficient of one PP or QQ term is
/// `amplitude * phase * angular * radial_overlap`. Only `amplitude` and the
/// magnetic projections vary between band pairs and between the spinor
/// coordinates of one pair, so the radial quadrature and the auxiliary index
/// lookup are a function of `(sector, left, right)` alone. Reusing one table
/// across a whole q slice removes that quadrature from the inner loop, which
/// otherwise repeats it once per band-pair column.
///
/// Entries are filled on first use, so a caller that expands a single pair
/// pays exactly the quadrature it needs.
#[derive(Debug)]
pub struct DiracMtSectorTable<'a> {
    raw: &'a DiracRawProductSpace,
    auxiliary: &'a CompiledAuxiliaryBasis,
    terms: HashMap<(DiracChargeSector, DiracRadialId, DiracRadialId), Vec<DiracMtModeTerm>>,
}

impl<'a> DiracMtSectorTable<'a> {
    /// Empty table bound to one raw product space and one compiled auxiliary.
    pub fn new(raw: &'a DiracRawProductSpace, auxiliary: &'a CompiledAuxiliaryBasis) -> Self {
        Self {
            raw,
            auxiliary,
            terms: HashMap::new(),
        }
    }

    fn terms(
        &mut self,
        sector: DiracChargeSector,
        spec: DiracMtPairSpec,
    ) -> Result<&[DiracMtModeTerm], MpbError> {
        let key = (sector, spec.left, spec.right);
        if !self.terms.contains_key(&key) {
            let built = self.build(sector, spec)?;
            self.terms.insert(key, built);
        }
        Ok(&self.terms[&key])
    }

    fn build(
        &self,
        sector: DiracChargeSector,
        spec: DiracMtPairSpec,
    ) -> Result<Vec<DiracMtModeTerm>, MpbError> {
        let site = spec.left.site;
        if !self
            .raw
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
        let unknown_orbital = || MpbError::UnknownDiracOrbital {
            site,
            kind: spec.left.kind,
            kappa: spec.left.kappa.get(),
            n: spec.left.n,
        };
        let mesh = self.auxiliary.site_mesh(site).ok_or_else(unknown_orbital)?;
        let block = self
            .auxiliary
            .require_mixed_product()?
            .sites
            .iter()
            .find(|block| block.site == site)
            .ok_or_else(unknown_orbital)?;
        let mut terms = Vec::new();
        for mode in &block.modes {
            let Some(product) = self.raw.radial_products.iter().find(|product| {
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
            let l_i = mode.l as i32;
            terms.push(DiracMtModeTerm {
                l: mode.l,
                overlap: mesh.integrate(&integrand)?,
                indices: (-l_i..=l_i)
                    .map(|m| self.auxiliary.mt_index(site, mode.l, m, mode.n))
                    .collect(),
            });
        }
        Ok(terms)
    }
}

fn add_sector(
    source: &DiracProductSource,
    table: &mut DiracMtSectorTable<'_>,
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
    let (left_channel, right_channel) = pair_channels(spec)?;
    let left_omega = sector.omega_channel(left_channel);
    let right_omega = sector.omega_channel(right_channel);
    let position = source.partition.sites()[site].position;
    let phase = site_translation_phase(source.q.cartesian, position);
    for term in table.terms(sector, spec)? {
        let l_i = term.l as i32;
        for (slot, m) in (-l_i..=l_i).enumerate() {
            let Some(index) = term.indices[slot] else {
                continue;
            };
            let angular = density_angular(left_omega, term.l, m, right_omega);
            coefficients[index] += amplitude * phase * Complex64::new(angular * term.overlap, 0.0);
        }
    }
    Ok(())
}

fn pair_matches(stored: crate::DiracPairChannel, spec: DiracMtPairSpec) -> bool {
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
    if m.unsigned_abs().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}
