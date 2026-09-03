//! Pair vertices onto a SPEX mixed product basis.

use crate::mpb::InterstitialThetaTable;
use crate::mpb::MpbError;
use crate::mpb::construct::require_matching_context;
use crate::mpb::interstitial::add_raw_support_theta_i;
use crate::{
    AuxiliaryLayout, AuxiliarySource, CompiledAuxiliaryBasis, InterstitialPairSpec, MtPairSpec,
    OrbitalPair, PairVertex, PairVertexSpec, ProductOrbitalKind, ProductRadial, ProductRadialId,
    RawProductSpace,
};
use muffintin_core::gaunt;
use muffintin_envelope::site_translation_phase;
use num_complex::Complex64;
use std::collections::HashMap;

/// Scalar source, raw product space, and compiled auxiliary validated together.
///
/// The matching checks walk the static radial and interstitial context. A
/// band-pair batch validates that context once and starts every checked vertex
/// from this handle.
#[derive(Clone, Copy, Debug)]
pub struct ScalarVertexContext<'a> {
    source: &'a AuxiliarySource,
    raw: &'a RawProductSpace,
    auxiliary: &'a CompiledAuxiliaryBasis,
}

impl<'a> ScalarVertexContext<'a> {
    pub fn new(
        source: &'a AuxiliarySource,
        raw: &'a RawProductSpace,
        auxiliary: &'a CompiledAuxiliaryBasis,
    ) -> Result<Self, MpbError> {
        require_matching_context(source, raw, auxiliary)?;
        Ok(Self {
            source,
            raw,
            auxiliary,
        })
    }

    /// Empty radial-overlap table shared by all scalar band-pair columns.
    pub fn muffin_tin_table(&self) -> ScalarMtPairTable<'a> {
        ScalarMtPairTable::new(self.raw, self.auxiliary)
    }

    /// Precompute every raw-pair interstitial step-function row once.
    pub fn interstitial_table(&self) -> Result<InterstitialThetaTable, MpbError> {
        InterstitialThetaTable::new(&self.raw.interstitial_pair_support, self.auxiliary)
    }

    /// Start an empty checked vertex without repeating the static context walk.
    pub fn accumulator(&self, pair: OrbitalPair) -> PairVertexAccumulator<'a> {
        PairVertexAccumulator {
            source: self.source,
            raw: self.raw,
            auxiliary: self.auxiliary,
            pair,
            coefficients: vec![Complex64::default(); self.auxiliary.dimension()],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ScalarMtModeTerm {
    l: u32,
    n: usize,
    overlap: f64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ScalarRadialPair(ProductRadialId, ProductRadialId);

impl ScalarRadialPair {
    fn new(left: ProductRadialId, right: ProductRadialId) -> Self {
        if radial_order_key(left) <= radial_order_key(right) {
            Self(left, right)
        } else {
            Self(right, left)
        }
    }
}

fn radial_order_key(id: ProductRadialId) -> (usize, u8, u32, usize, u8) {
    let kind = match id.kind {
        ProductOrbitalKind::Valence => 0,
        ProductOrbitalKind::Core => 1,
    };
    (id.site, kind, id.l, id.n, id.spin)
}

/// Radial overlaps shared by every magnetic coordinate and band-pair column.
///
/// Entries are populated on first use. The expensive radial quadrature is a
/// function of the unordered radial pair and retained auxiliary mode, while
/// Gaunt factors are compiled once for each requested magnetic pair.
#[derive(Debug)]
pub struct ScalarMtPairTable<'a> {
    raw: &'a RawProductSpace,
    auxiliary: &'a CompiledAuxiliaryBasis,
    terms: HashMap<ScalarRadialPair, Vec<ScalarMtModeTerm>>,
    expanded: HashMap<(ProductRadialId, i32, ProductRadialId, i32), Vec<(usize, Complex64)>>,
}

/// Precompiled contribution of one scalar spin-angular coordinate pair.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarMtCompiledPair {
    layout: AuxiliaryLayout,
    coefficients: Vec<(usize, Complex64)>,
}

impl ScalarMtCompiledPair {
    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    pub fn coefficients(&self) -> &[(usize, Complex64)] {
        &self.coefficients
    }
}

impl<'a> ScalarMtPairTable<'a> {
    pub fn new(raw: &'a RawProductSpace, auxiliary: &'a CompiledAuxiliaryBasis) -> Self {
        Self {
            raw,
            auxiliary,
            terms: HashMap::new(),
            expanded: HashMap::new(),
        }
    }

    pub fn auxiliary_dimension(&self) -> usize {
        self.auxiliary.dimension()
    }

    /// Compile the retained auxiliary coefficients for one coordinate pair.
    pub fn compile_pair(
        &mut self,
        source: &AuxiliarySource,
        spec: MtPairSpec,
    ) -> Result<ScalarMtCompiledPair, MpbError> {
        let coefficients = self.expanded(source, spec)?.to_vec();
        Ok(ScalarMtCompiledPair {
            layout: self.auxiliary.layout(),
            coefficients,
        })
    }

    fn expanded(
        &mut self,
        source: &AuxiliarySource,
        spec: MtPairSpec,
    ) -> Result<&[(usize, Complex64)], MpbError> {
        let key = (spec.left, spec.left_m, spec.right, spec.right_m);
        if !self.expanded.contains_key(&key) {
            if spec.left.site != spec.right.site {
                return Err(MpbError::CrossSitePair);
            }
            if spec.left_m.unsigned_abs() > spec.left.l {
                return Err(MpbError::MagneticQuantumNumber {
                    l: spec.left.l,
                    m: spec.left_m,
                });
            }
            if spec.right_m.unsigned_abs() > spec.right.l {
                return Err(MpbError::MagneticQuantumNumber {
                    l: spec.right.l,
                    m: spec.right_m,
                });
            }
            let site = spec.left.site;
            let radials = source.radials.get(site).ok_or(MpbError::UnknownOrbital {
                site,
                kind: spec.left.kind,
                l: spec.left.l,
                n: spec.left.n,
                spin: spec.left.spin,
            })?;
            find_radial(radials, spec.left)?;
            find_radial(radials, spec.right)?;
            let position = source.partition.sites()[site].position;
            let phase = site_translation_phase(source.q.cartesian, position);
            let m = spec.right_m - spec.left_m;
            let terms = self.terms(spec)?.to_vec();
            let mut coefficients = Vec::with_capacity(terms.len());
            for term in terms {
                if m.unsigned_abs() > term.l {
                    continue;
                }
                let angular = gaunt(
                    spec.left.l,
                    spec.right.l,
                    term.l,
                    spec.left_m,
                    spec.right_m,
                    m,
                );
                if let Some(index) = self.auxiliary.mt_index(site, term.l, m, term.n) {
                    coefficients.push((index, phase * Complex64::new(angular * term.overlap, 0.0)));
                }
            }
            self.expanded.insert(key, coefficients);
        }
        Ok(&self.expanded[&key])
    }

    fn terms(&mut self, spec: MtPairSpec) -> Result<&[ScalarMtModeTerm], MpbError> {
        let key = ScalarRadialPair::new(spec.left, spec.right);
        if !self.terms.contains_key(&key) {
            let built = self.build(spec)?;
            self.terms.insert(key, built);
        }
        Ok(&self.terms[&key])
    }

    fn build(&self, spec: MtPairSpec) -> Result<Vec<ScalarMtModeTerm>, MpbError> {
        let site = spec.left.site;
        if !self.raw.radial_products.iter().any(|product| {
            product.channel.left.site == site
                && pair_matches(
                    product.channel.left,
                    product.channel.right,
                    spec.left,
                    spec.right,
                )
        }) {
            return Err(MpbError::UnknownMtPair {
                left: spec.left,
                right: spec.right,
            });
        }
        let mesh = self
            .auxiliary
            .site_mesh(site)
            .ok_or(MpbError::UnknownOrbital {
                site,
                kind: spec.left.kind,
                l: spec.left.l,
                n: spec.left.n,
                spin: spec.left.spin,
            })?;
        let block = self
            .auxiliary
            .require_mixed_product()?
            .sites
            .iter()
            .find(|block| block.site == site)
            .ok_or(MpbError::UnknownOrbital {
                site,
                kind: spec.left.kind,
                l: spec.left.l,
                n: spec.left.n,
                spin: spec.left.spin,
            })?;
        let mut terms = Vec::new();
        for mode in &block.modes {
            let Some(product) = self.raw.radial_products.iter().find(|product| {
                product.channel.coupled_l == mode.l
                    && product.channel.left.site == site
                    && pair_matches(
                        product.channel.left,
                        product.channel.right,
                        spec.left,
                        spec.right,
                    )
            }) else {
                continue;
            };
            let integrand = product
                .samples
                .iter()
                .zip(&mode.radial)
                .map(|(sample, mode)| sample * mode)
                .collect::<Vec<_>>();
            terms.push(ScalarMtModeTerm {
                l: mode.l,
                n: mode.n,
                overlap: mesh.integrate(&integrand)?,
            });
        }
        Ok(terms)
    }
}

/// Accumulator that sums primitive muffin-tin and interstitial pair terms
/// onto one checked auxiliary vertex.
///
/// Each [`Self::add_muffin_tin`] / [`Self::add_interstitial`] call reuses the
/// same Gaunt, radial-overlap, site-phase, and $\Theta_I$ algebra as
/// [`pair_vertex`]. Terms add into the shared coefficient vector; they do not
/// replace earlier contributions. This is the MPB-side contraction surface
/// for a band-orbital sum. It is not a Coulomb kernel.
#[derive(Debug)]
pub struct PairVertexAccumulator<'a> {
    source: &'a AuxiliarySource,
    raw: &'a RawProductSpace,
    auxiliary: &'a CompiledAuxiliaryBasis,
    pair: OrbitalPair,
    coefficients: Vec<Complex64>,
}

impl<'a> PairVertexAccumulator<'a> {
    /// Start an empty vertex on a matching source / raw / auxiliary context.
    pub fn new(
        source: &'a AuxiliarySource,
        raw: &'a RawProductSpace,
        auxiliary: &'a CompiledAuxiliaryBasis,
        pair: OrbitalPair,
    ) -> Result<Self, MpbError> {
        Ok(ScalarVertexContext::new(source, raw, auxiliary)?.accumulator(pair))
    }

    /// Add one muffin-tin radial-factor pair scaled by `amplitude`.
    ///
    /// The kernel is Gaunt-weighted radial overlap times
    /// $\exp(+i q\cdot R_a)$. The contribution is added into every matching
    /// auxiliary muffin-tin index.
    pub fn add_muffin_tin(
        &mut self,
        spec: MtPairSpec,
        amplitude: Complex64,
    ) -> Result<(), MpbError> {
        add_muffin_tin(
            self.source,
            self.raw,
            self.auxiliary,
            spec,
            amplitude,
            &mut self.coefficients,
        )
    }

    /// Add one coordinate-pair contribution compiled for this auxiliary basis.
    pub fn add_compiled_muffin_tin(
        &mut self,
        pair: &ScalarMtCompiledPair,
        amplitude: Complex64,
    ) -> Result<(), MpbError> {
        if pair.layout != self.auxiliary.layout() {
            return Err(MpbError::CompiledScalarMtContext);
        }
        for &(index, factor) in &pair.coefficients {
            self.coefficients[index] += amplitude * factor;
        }
        Ok(())
    }

    /// Add one already contracted auxiliary coefficient vector.
    pub fn add_auxiliary_coefficients(
        &mut self,
        coefficients: &[Complex64],
    ) -> Result<(), MpbError> {
        if coefficients.len() != self.coefficients.len() {
            return Err(MpbError::CompiledScalarMtContext);
        }
        for (target, &coefficient) in self.coefficients.iter_mut().zip(coefficients) {
            *target += coefficient;
        }
        Ok(())
    }

    /// Add one interstitial pair expansion.
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

    /// Add one interstitial pair through a precomputed step-function table.
    pub fn add_interstitial_from_table(
        &mut self,
        table: &InterstitialThetaTable,
        spec: InterstitialPairSpec,
    ) -> Result<(), MpbError> {
        table.add(self.auxiliary, spec, &mut self.coefficients)
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

/// Expand an explicit MT and/or interstitial pair onto the auxiliary basis.
///
/// Muffin-tin coefficients are Gaunt-weighted radial overlaps times
/// $\exp(+i q\cdot R_a)$. Interstitial coefficients are
/// $A\Theta_I(G_{\mathrm{aux}}-G_{\mathrm{wrap}}-G_{\mathrm{rel}})$ using the partition step
/// function. The interstitial G label must exist on the raw pair support.
/// Missing spec arms stay zero. This is not a Coulomb kernel.
pub fn pair_vertex(
    source: &AuxiliarySource,
    raw: &RawProductSpace,
    auxiliary: &CompiledAuxiliaryBasis,
    spec: PairVertexSpec,
) -> Result<PairVertex, MpbError> {
    let pair = spec.pair_identity().ok_or(MpbError::EmptyPairSpec)?;
    let mut acc = PairVertexAccumulator::new(source, raw, auxiliary, pair)?;
    if let Some(mt) = spec.muffin_tin {
        acc.add_muffin_tin(mt, Complex64::new(1.0, 0.0))?;
    }
    if let Some(interstitial) = spec.interstitial {
        acc.add_interstitial(interstitial)?;
    }
    acc.finish()
}

fn add_muffin_tin(
    source: &AuxiliarySource,
    raw: &RawProductSpace,
    auxiliary: &CompiledAuxiliaryBasis,
    spec: MtPairSpec,
    amplitude: Complex64,
    coefficients: &mut [Complex64],
) -> Result<(), MpbError> {
    if spec.left.site != spec.right.site {
        return Err(MpbError::CrossSitePair);
    }
    if spec.left_m.unsigned_abs() > spec.left.l {
        return Err(MpbError::MagneticQuantumNumber {
            l: spec.left.l,
            m: spec.left_m,
        });
    }
    if spec.right_m.unsigned_abs() > spec.right.l {
        return Err(MpbError::MagneticQuantumNumber {
            l: spec.right.l,
            m: spec.right_m,
        });
    }
    let site = spec.left.site;
    let radials = source.radials.get(site).ok_or(MpbError::UnknownOrbital {
        site,
        kind: spec.left.kind,
        l: spec.left.l,
        n: spec.left.n,
        spin: spec.left.spin,
    })?;
    find_radial(radials, spec.left)?;
    find_radial(radials, spec.right)?;
    if !raw.radial_products.iter().any(|product| {
        product.channel.left.site == site
            && pair_matches(
                product.channel.left,
                product.channel.right,
                spec.left,
                spec.right,
            )
    }) {
        return Err(MpbError::UnknownMtPair {
            left: spec.left,
            right: spec.right,
        });
    }
    let mesh = auxiliary.site_mesh(site).ok_or(MpbError::UnknownOrbital {
        site,
        kind: spec.left.kind,
        l: spec.left.l,
        n: spec.left.n,
        spin: spec.left.spin,
    })?;
    let position = source.partition.sites()[site].position;
    let phase = site_translation_phase(source.q.cartesian, position);
    let m = spec.right_m - spec.left_m;
    let block = auxiliary
        .require_mixed_product()?
        .sites
        .iter()
        .find(|block| block.site == site)
        .ok_or(MpbError::UnknownOrbital {
            site,
            kind: spec.left.kind,
            l: spec.left.l,
            n: spec.left.n,
            spin: spec.left.spin,
        })?;
    for mode in &block.modes {
        if m.unsigned_abs() > mode.l {
            continue;
        }
        let Some(product) = raw.radial_products.iter().find(|product| {
            product.channel.coupled_l == mode.l
                && product.channel.left.site == site
                && pair_matches(
                    product.channel.left,
                    product.channel.right,
                    spec.left,
                    spec.right,
                )
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
        let angular = gaunt(
            spec.left.l,
            spec.right.l,
            mode.l,
            spec.left_m,
            spec.right_m,
            m,
        );
        if let Some(index) = auxiliary.mt_index(site, mode.l, m, mode.n) {
            coefficients[index] +=
                amplitude * phase * Complex64::new(angular * radial_overlap, 0.0);
        }
    }
    Ok(())
}

fn pair_matches(
    stored_left: ProductRadialId,
    stored_right: ProductRadialId,
    left: ProductRadialId,
    right: ProductRadialId,
) -> bool {
    (stored_left == left && stored_right == right) || (stored_left == right && stored_right == left)
}

fn find_radial(
    radials: &crate::SiteRadialSet,
    id: ProductRadialId,
) -> Result<&ProductRadial, MpbError> {
    let pool = match id.kind {
        ProductOrbitalKind::Valence => radials.valence.as_slice(),
        ProductOrbitalKind::Core => radials.cores.as_slice(),
    };
    pool.iter()
        .find(|radial| radial.l == id.l && radial.n == id.n && radial.spin == id.spin)
        .ok_or(MpbError::UnknownOrbital {
            site: id.site,
            kind: id.kind,
            l: id.l,
            n: id.n,
            spin: id.spin,
        })
}
