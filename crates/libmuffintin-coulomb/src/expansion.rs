//! One internal Weinert charge expansion for mixed-product and sampled zeta functions.

use crate::CoulombError;
use crate::spec::{CoulombRequest, InterpolationProjection};
use libmuffintin_core::{
    Bohr, ExponentialMesh, InverseBohr, VolumeBohr3, complex_spherical_harmonics, lm_index,
};
use libmuffintin_product::{
    AuxiliaryInterstitialWave, AuxiliaryLayout, AuxiliaryRegion, CompiledAuxiliaryBasis,
    InterpolationRegion, MixedProductAuxiliary, TransferQ,
};
use num_complex::Complex64;

/// Sampled interpolation functions $\zeta_\mu^q(r)$ on a parent quadrature grid.
///
/// Storage is row-major `n_grid × n_mu`. This is the production interpolation
/// input to the Coulomb assembler. Interpolation *nodes* are not $\zeta$.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledAuxiliaryFunctions {
    layout: AuxiliaryLayout,
    points: Vec<[Bohr; 3]>,
    weights: Vec<VolumeBohr3>,
    regions: Vec<InterpolationRegion>,
    zeta: Vec<Complex64>,
}

impl SampledAuxiliaryFunctions {
    /// Construct after checking layout, grid lengths, weights, and $\zeta$ shape.
    pub fn new(
        layout: AuxiliaryLayout,
        points: Vec<[Bohr; 3]>,
        weights: Vec<VolumeBohr3>,
        regions: Vec<InterpolationRegion>,
        zeta: Vec<Complex64>,
    ) -> Result<Self, CoulombError> {
        let n_grid = points.len();
        if n_grid == 0 {
            return Err(CoulombError::EmptySampledGrid);
        }
        if weights.len() != n_grid || regions.len() != n_grid {
            return Err(CoulombError::SampledGridLength {
                points: n_grid,
                weights: weights.len(),
                regions: regions.len(),
            });
        }
        let n_mu = layout.dimension();
        if n_mu == 0 {
            return Err(CoulombError::EmptyAuxiliary);
        }
        let expected = n_grid
            .checked_mul(n_mu)
            .ok_or(CoulombError::SampledZetaLength {
                actual: zeta.len(),
                n_grid,
                n_mu,
            })?;
        if zeta.len() != expected {
            return Err(CoulombError::SampledZetaLength {
                actual: zeta.len(),
                n_grid,
                n_mu,
            });
        }
        let mut any_positive = false;
        for (index, point) in points.iter().enumerate() {
            if point.iter().any(|component| !component.get().is_finite()) {
                return Err(CoulombError::NonFiniteSampledPoint(index));
            }
            let weight = weights[index].get();
            if !weight.is_finite() {
                return Err(CoulombError::NonFiniteSampledWeight(index));
            }
            if weight < 0.0 {
                return Err(CoulombError::NegativeSampledWeight(index));
            }
            if weight > 0.0 {
                any_positive = true;
            }
        }
        if !any_positive {
            return Err(CoulombError::NoPositiveSampledWeight);
        }
        for (index, value) in zeta.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(CoulombError::NonFiniteZeta(index));
            }
        }
        Ok(Self {
            layout,
            points,
            weights,
            regions,
            zeta,
        })
    }

    /// Auxiliary layout ($q$, regions, split) of the $\zeta$ columns.
    pub const fn layout(&self) -> &AuxiliaryLayout {
        &self.layout
    }

    /// Parent-grid point count.
    pub fn n_grid(&self) -> usize {
        self.points.len()
    }

    /// Number of interpolation functions, equal to the auxiliary dimension.
    pub fn n_mu(&self) -> usize {
        self.layout.dimension()
    }

    /// Row-major $\zeta$, `n_grid × n_mu`.
    pub fn zeta(&self) -> &[Complex64] {
        &self.zeta
    }

    /// Parent-grid coordinates.
    pub fn points(&self) -> &[[Bohr; 3]] {
        &self.points
    }

    /// Parent-grid quadrature weights.
    pub fn weights(&self) -> &[VolumeBohr3] {
        &self.weights
    }

    /// Parent-grid region tags.
    pub fn regions(&self) -> &[InterpolationRegion] {
        &self.regions
    }
}

/// One muffin-tin radial $\times Y_{LM}$ contribution to a charge expansion.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MtPiece {
    pub site: usize,
    pub l: u32,
    pub m: i32,
    /// SPEX `basm` radial samples.
    pub radial: Vec<f64>,
    /// Complex amplitude in front of the real radial.
    pub amplitude: Complex64,
}

/// Auxiliary function as MT pieces plus interstitial PW coefficients.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ChargeDensity {
    pub mt: Vec<MtPiece>,
    pub pw: Vec<Complex64>,
}

/// Shared geometry of the Weinert expansion.
#[derive(Clone, Debug)]
pub(crate) struct ExpansionSupport {
    pub volume: f64,
    pub sites: Vec<SiteSupport>,
    pub waves: Vec<AuxiliaryInterstitialWave>,
}

#[derive(Clone, Debug)]
pub(crate) struct SiteSupport {
    pub position: [Bohr; 3],
    pub radius: Bohr,
    pub mesh: ExponentialMesh,
}

pub(crate) fn mixed_product_support(
    auxiliary: &CompiledAuxiliaryBasis,
    payload: &MixedProductAuxiliary,
) -> Result<ExpansionSupport, CoulombError> {
    let mut sites = Vec::with_capacity(payload.sites.len());
    for (index, block) in payload.sites.iter().enumerate() {
        let partition = auxiliary
            .partition
            .sites
            .get(index)
            .ok_or(CoulombError::MissingSite(index))?;
        sites.push(SiteSupport {
            position: partition.position,
            radius: partition.radius,
            mesh: block.mesh.clone(),
        });
    }
    Ok(ExpansionSupport {
        volume: auxiliary.partition.interstitial.cell_volume().get(),
        sites,
        waves: payload.interstitial.waves.clone(),
    })
}

pub(crate) fn mixed_product_densities(
    auxiliary: &CompiledAuxiliaryBasis,
    payload: &MixedProductAuxiliary,
) -> Result<Vec<ChargeDensity>, CoulombError> {
    let n_pw = payload.interstitial.waves.len();
    let mut densities = Vec::with_capacity(auxiliary.dimension());
    for region in auxiliary.regions() {
        match region {
            AuxiliaryRegion::MuffinTin { site, l, m, n } => {
                let block = payload
                    .sites
                    .get(site)
                    .ok_or(CoulombError::MissingSite(site))?;
                let mode = block
                    .modes
                    .iter()
                    .find(|mode| mode.l == l && mode.n == n)
                    .ok_or(CoulombError::MissingSite(site))?;
                densities.push(ChargeDensity {
                    mt: vec![MtPiece {
                        site,
                        l,
                        m,
                        radial: mode.radial.clone(),
                        amplitude: Complex64::new(1.0, 0.0),
                    }],
                    pw: vec![Complex64::default(); n_pw],
                });
            }
            AuxiliaryRegion::Interstitial { g } => {
                let mut pw = vec![Complex64::default(); n_pw];
                if let Some(index) = payload
                    .interstitial
                    .waves
                    .iter()
                    .position(|wave| wave.g.index == g.index)
                {
                    pw[index] = Complex64::new(1.0, 0.0);
                }
                densities.push(ChargeDensity { mt: Vec::new(), pw });
            }
            AuxiliaryRegion::InterpolationPoint { .. } => {
                return Err(CoulombError::Product(
                    libmuffintin_product::ProductError::ExpectedMixedProduct,
                ));
            }
        }
    }
    Ok(densities)
}

pub(crate) fn interpolation_support(
    auxiliary: &CompiledAuxiliaryBasis,
    request: &CoulombRequest,
    projection: InterpolationProjection,
) -> Result<ExpansionSupport, CoulombError> {
    let mut sites = Vec::with_capacity(auxiliary.partition.site_count());
    for site in &auxiliary.partition.sites {
        sites.push(SiteSupport {
            position: site.position,
            radius: site.radius,
            mesh: site_mesh(site.radius)?,
        });
    }
    let waves = auxiliary_waves(request, auxiliary.q, projection.pw_cutoff)?;
    Ok(ExpansionSupport {
        volume: auxiliary.partition.interstitial.cell_volume().get(),
        sites,
        waves,
    })
}

pub(crate) fn sampled_zeta_densities(
    sampled: &SampledAuxiliaryFunctions,
    support: &ExpansionSupport,
    projection: InterpolationProjection,
) -> Result<Vec<ChargeDensity>, CoulombError> {
    let n_mu = sampled.n_mu();
    let n_grid = sampled.n_grid();
    let n_pw = support.waves.len();
    let mut densities = vec![
        ChargeDensity {
            mt: Vec::new(),
            pw: vec![Complex64::default(); n_pw],
        };
        n_mu
    ];
    for p in 0..n_grid {
        let weight = sampled.weights[p].get();
        if weight == 0.0 {
            continue;
        }
        for (mu, density) in densities.iter_mut().enumerate() {
            let zeta = sampled.zeta[p * n_mu + mu];
            if zeta.norm() == 0.0 {
                continue;
            }
            accumulate_sample(
                density,
                support,
                sampled.points[p],
                sampled.regions[p],
                zeta * weight,
                projection.l_max,
                p,
            )?;
        }
    }
    Ok(densities)
}

/// Toy Ewald path: each interpolation *node* is a quadrature-weighted point charge.
///
/// This is not the production $\zeta$ metric.
pub(crate) fn point_charge_densities(
    auxiliary: &CompiledAuxiliaryBasis,
    support: &ExpansionSupport,
    projection: InterpolationProjection,
) -> Result<Vec<ChargeDensity>, CoulombError> {
    let points = auxiliary.require_interpolation_points()?;
    let mut densities = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        let mut density = ChargeDensity {
            mt: Vec::new(),
            pw: vec![Complex64::default(); support.waves.len()],
        };
        accumulate_sample(
            &mut density,
            support,
            point.coordinate,
            point.region,
            Complex64::new(point.weight.get(), 0.0),
            projection.l_max,
            index,
        )?;
        densities.push(density);
    }
    Ok(densities)
}

fn accumulate_sample(
    density: &mut ChargeDensity,
    support: &ExpansionSupport,
    coordinate: [Bohr; 3],
    region: InterpolationRegion,
    charge: Complex64,
    l_max: u32,
    index: usize,
) -> Result<(), CoulombError> {
    match region {
        InterpolationRegion::MuffinTin { site } => {
            if site >= support.sites.len() {
                return Err(CoulombError::SampledPointSite { site });
            }
            let site_support = support
                .sites
                .get(site)
                .ok_or(CoulombError::MissingSite(site))?;
            let rel: [f64; 3] = std::array::from_fn(|axis| {
                coordinate[axis].get() - site_support.position[axis].get()
            });
            let r0 = rel.iter().map(|c| c * c).sum::<f64>().sqrt();
            if r0 > site_support.radius.get() * (1.0 + 1.0e-8) {
                return Err(CoulombError::InterpolationPointOutsideSphere(index));
            }
            let harmonics = complex_spherical_harmonics(l_max, rel);
            let spike = radial_delta(&site_support.mesh, r0);
            for l in 0..=l_max {
                for m in -(l as i32)..=l as i32 {
                    let y = harmonics[lm_index(l, m)?].conj();
                    density.mt.push(MtPiece {
                        site,
                        l,
                        m,
                        radial: spike.clone(),
                        amplitude: y * charge,
                    });
                }
            }
        }
        InterpolationRegion::Interstitial | InterpolationRegion::Uniform => {
            let svol = support.volume.sqrt();
            for (local, wave) in support.waves.iter().enumerate() {
                let phase = wave
                    .q_plus_g
                    .iter()
                    .zip(coordinate)
                    .map(|(q, r)| q.get() * r.get())
                    .sum::<f64>();
                density.pw[local] += charge * Complex64::from_polar(1.0, -phase) / svol;
            }
        }
    }
    Ok(())
}

fn radial_delta(mesh: &ExponentialMesh, r0: f64) -> Vec<f64> {
    let mut best = 0usize;
    let mut best_d = f64::MAX;
    for (index, radius) in mesh.radii().iter().enumerate() {
        let distance = (radius.get() - r0).abs();
        if distance < best_d {
            best_d = distance;
            best = index;
        }
    }
    let mut samples = vec![0.0; mesh.len()];
    let weight = mesh.weights()[best];
    let radius = mesh.radii()[best].get();
    if weight.abs() > 0.0 && radius.abs() > 0.0 {
        samples[best] = 1.0 / (weight * radius);
    }
    samples
}

fn site_mesh(radius: Bohr) -> Result<ExponentialMesh, CoulombError> {
    let first = 1.0e-5;
    let number = 73;
    let increment = (radius.get() / first).ln() / (number as f64 - 1.0);
    Ok(ExponentialMesh::new(Bohr(first), increment, number)?)
}

pub(crate) fn auxiliary_waves(
    request: &CoulombRequest,
    q: TransferQ,
    g_cut: InverseBohr,
) -> Result<Vec<AuxiliaryInterstitialWave>, CoulombError> {
    let bound = InverseBohr(g_cut.get() + crate::math::inverse_norm(q.cartesian));
    let cutoff_squared = g_cut.get() * g_cut.get();
    let tolerance = 64.0 * f64::EPSILON * cutoff_squared.max(1.0);
    let mut waves = Vec::new();
    for g in request.reciprocal().enumerate(bound)? {
        let q_plus_g: [InverseBohr; 3] = std::array::from_fn(|axis| {
            InverseBohr(q.cartesian[axis].get() + g.cartesian[axis].get())
        });
        let qg_squared = q_plus_g
            .iter()
            .map(|component| component.get().powi(2))
            .sum::<f64>();
        if qg_squared <= cutoff_squared + tolerance {
            waves.push(AuxiliaryInterstitialWave {
                g,
                q_plus_g,
                q_plus_g_norm: InverseBohr(qg_squared.sqrt()),
            });
        }
    }
    Ok(waves)
}
