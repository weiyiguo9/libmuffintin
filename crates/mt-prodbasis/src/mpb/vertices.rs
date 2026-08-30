//! Pair vertices onto a SPEX mixed product basis.

use crate::mpb::MpbError;
use crate::mpb::construct::require_matching_context;
use crate::mpb::interstitial::add_raw_support_theta_i;
use crate::{
    AuxiliarySource, CompiledAuxiliaryBasis, InterstitialPairSpec, MtPairSpec, OrbitalPair,
    PairVertex, PairVertexSpec, ProductOrbitalKind, ProductRadial, ProductRadialId,
    RawProductSpace,
};
use muffintin_core::gaunt;
use muffintin_envelope::site_translation_phase;
use num_complex::Complex64;

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
        require_matching_context(source, raw, auxiliary)?;
        Ok(Self {
            source,
            raw,
            auxiliary,
            pair,
            coefficients: vec![Complex64::default(); auxiliary.dimension()],
        })
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
