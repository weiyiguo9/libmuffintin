//! One internal Weinert charge expansion for mixed-product and sampled zeta functions.

use crate::CoulombError;
use crate::spec::{CoulombRequest, InterpolationProjection};
use muffintin_core::{
    Bohr, Cell, ExponentialMesh, InverseBohr, ReciprocalLattice, UniformGrid, VolumeBohr3,
    complex_spherical_harmonics, lm_count, lm_index,
};
use muffintin_prodbasis::{
    AuxiliaryInterstitialWave, AuxiliaryLayout, AuxiliaryRegion, CompiledAuxiliaryBasis,
    InterpolationRegion, MixedProductAuxiliary, TransferQ,
};
use num_complex::Complex64;
#[cfg(feature = "fft-fftw")]
use std::f64::consts::PI;

#[derive(Clone, Debug, PartialEq)]
struct UniformInterstitialGrid {
    cell: Cell,
    dimensions: [usize; 3],
    sampled_to_uniform: Vec<Option<usize>>,
}

/// Sampled interpolation functions $\zeta_\mu^q(r)$ on a parent quadrature grid.
///
/// Storage is row-major `n_grid × n_mu`. This is the production interpolation
/// input to the Coulomb assembler. Interpolation *nodes* are not $\zeta$.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledAuxiliaryFunctions {
    layout: AuxiliaryLayout,
    site_meshes: Vec<ExponentialMesh>,
    points: Vec<[Bohr; 3]>,
    weights: Vec<VolumeBohr3>,
    supports: Vec<SampledPointSupport>,
    zeta: Vec<Complex64>,
    uniform_interstitial_grid: Option<UniformInterstitialGrid>,
}

/// Exact radial/interstitial support of one sampled parent-grid point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampledPointSupport {
    /// Point on an explicit radial shell of muffin-tin `site`.
    MuffinTin { site: usize, radial_index: usize },
    /// Point in the partitioned interstitial.
    Interstitial,
    /// Point on an unpartitioned uniform grid.
    Uniform,
}

impl SampledAuxiliaryFunctions {
    /// Construct after checking layout, grid lengths, weights, and $\zeta$ shape.
    pub fn new(
        layout: AuxiliaryLayout,
        site_meshes: Vec<ExponentialMesh>,
        points: Vec<[Bohr; 3]>,
        weights: Vec<VolumeBohr3>,
        supports: Vec<SampledPointSupport>,
        zeta: Vec<Complex64>,
    ) -> Result<Self, CoulombError> {
        let n_grid = points.len();
        if n_grid == 0 {
            return Err(CoulombError::EmptySampledGrid);
        }
        if weights.len() != n_grid || supports.len() != n_grid {
            return Err(CoulombError::SampledGridLength {
                points: n_grid,
                weights: weights.len(),
                supports: supports.len(),
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
        for (site, mesh) in site_meshes.iter().enumerate() {
            if mesh.increment() <= 0.0 {
                return Err(CoulombError::SampledMeshNotOutward {
                    site,
                    increment: mesh.increment(),
                });
            }
        }
        for (point, support) in supports.iter().enumerate() {
            if let SampledPointSupport::MuffinTin { site, radial_index } = *support {
                let mesh = site_meshes
                    .get(site)
                    .ok_or(CoulombError::SampledPointSite { site })?;
                if radial_index >= mesh.len() {
                    return Err(CoulombError::SampledRadialIndex {
                        point,
                        site,
                        index: radial_index,
                        count: mesh.len(),
                    });
                }
            }
        }
        for (index, value) in zeta.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(CoulombError::NonFiniteZeta(index));
            }
        }
        Ok(Self {
            layout,
            site_meshes,
            points,
            weights,
            supports,
            zeta,
            uniform_interstitial_grid: None,
        })
    }

    /// Mark interstitial and uniform samples as a subset of one natural
    /// midpoint grid, enabling FFT projection when the FFTW feature is active.
    pub fn with_uniform_interstitial_grid(
        mut self,
        grid: &UniformGrid,
    ) -> Result<Self, CoulombError> {
        let cell = grid.cell();
        let dimensions = grid.divisions();
        let reciprocal = ReciprocalLattice::from_direct(*cell.basis())?;
        let mut sampled_to_uniform = Vec::with_capacity(self.points.len());
        for (point, (coordinate, support)) in self.points.iter().zip(&self.supports).enumerate() {
            if matches!(
                support,
                SampledPointSupport::Interstitial | SampledPointSupport::Uniform
            ) {
                sampled_to_uniform.push(Some(natural_grid_index(
                    cell,
                    &reciprocal,
                    dimensions,
                    *coordinate,
                    point,
                )?));
            } else {
                sampled_to_uniform.push(None);
            }
        }
        self.uniform_interstitial_grid = Some(UniformInterstitialGrid {
            cell,
            dimensions,
            sampled_to_uniform,
        });
        Ok(self)
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

    /// Per-site radial meshes in site-index order.
    pub fn site_meshes(&self) -> &[ExponentialMesh] {
        &self.site_meshes
    }

    /// Exact support of every parent-grid point.
    pub fn supports(&self) -> &[SampledPointSupport] {
        &self.supports
    }
}

/// One muffin-tin radial $\times Y_{LM}$ contribution to a charge expansion.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MtPiece {
    pub site: usize,
    pub l: u32,
    pub m: i32,
    /// Retained MPB radial mode, independent of m; absent for sampled functions.
    pub mpb_mode: Option<usize>,
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

#[derive(Clone)]
struct SampledMtChannel {
    site: usize,
    l: u32,
    m: i32,
    radial: Vec<Complex64>,
}

pub(crate) fn mixed_product_support(
    auxiliary: &CompiledAuxiliaryBasis,
    payload: &MixedProductAuxiliary,
) -> Result<ExpansionSupport, CoulombError> {
    let mut sites = Vec::with_capacity(payload.sites.len());
    for (index, block) in payload.sites.iter().enumerate() {
        let partition = auxiliary
            .partition
            .sites()
            .get(index)
            .ok_or(CoulombError::MissingSite(index))?;
        sites.push(SiteSupport {
            position: partition.position,
            radius: partition.radius,
            mesh: block.mesh.clone(),
        });
    }
    Ok(ExpansionSupport {
        volume: auxiliary.partition.interstitial().cell_volume().get(),
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
    let mut next_pw = 0;
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
                        mpb_mode: Some(n),
                        radial: mode.radial.clone(),
                        amplitude: Complex64::new(1.0, 0.0),
                    }],
                    pw: vec![Complex64::default(); n_pw],
                });
            }
            AuxiliaryRegion::Interstitial { .. } => {
                let mut pw = vec![Complex64::default(); n_pw];
                pw[next_pw] = Complex64::new(1.0, 0.0);
                next_pw += 1;
                densities.push(ChargeDensity { mt: Vec::new(), pw });
            }
            AuxiliaryRegion::InterpolationPoint { .. } => {
                return Err(CoulombError::Product(
                    muffintin_prodbasis::AuxiliaryIrError::ExpectedMixedProduct,
                ));
            }
        }
    }
    Ok(densities)
}

pub(crate) fn sampled_interpolation_support(
    auxiliary: &CompiledAuxiliaryBasis,
    request: &CoulombRequest,
    projection: &InterpolationProjection,
    sampled: &SampledAuxiliaryFunctions,
) -> Result<ExpansionSupport, CoulombError> {
    if sampled
        .uniform_interstitial_grid
        .as_ref()
        .is_some_and(|uniform| uniform.cell != *request.cell())
    {
        return Err(CoulombError::SampledUniformGridCellMismatch);
    }
    if sampled.site_meshes.len() != auxiliary.partition.site_count() {
        return Err(CoulombError::SampledMeshCount {
            expected: auxiliary.partition.site_count(),
            actual: sampled.site_meshes.len(),
        });
    }
    let mut sites = Vec::with_capacity(auxiliary.partition.site_count());
    for (site_index, (site, mesh)) in auxiliary
        .partition
        .sites()
        .iter()
        .zip(&sampled.site_meshes)
        .enumerate()
    {
        let mesh_radius = mesh.last().get();
        let sphere_radius = site.radius.get();
        if (mesh_radius - sphere_radius).abs() > 1.0e-10 * sphere_radius.max(1.0) {
            return Err(CoulombError::SampledMeshRadius {
                site: site_index,
                mesh: mesh_radius,
                sphere: sphere_radius,
            });
        }
        sites.push(SiteSupport {
            position: site.position,
            radius: site.radius,
            mesh: mesh.clone(),
        });
    }
    for (point, support) in sampled.supports.iter().enumerate() {
        if let SampledPointSupport::MuffinTin { site, radial_index } = *support {
            let site_support = &sites[site];
            let rel: [f64; 3] = std::array::from_fn(|axis| {
                sampled.points[point][axis].get() - site_support.position[axis].get()
            });
            let coordinate_radius = rel.iter().map(|value| value * value).sum::<f64>().sqrt();
            let shell_radius = site_support.mesh.radii()[radial_index].get();
            if (coordinate_radius - shell_radius).abs() > 1.0e-10 * shell_radius.max(1.0) {
                return Err(CoulombError::SampledCoordinateShellMismatch {
                    point,
                    site,
                    coordinate: coordinate_radius,
                    shell: shell_radius,
                });
            }
        }
    }
    let waves = auxiliary_waves(request, auxiliary.q, projection.pw_cutoff)?;
    Ok(ExpansionSupport {
        volume: auxiliary.partition.interstitial().cell_volume().get(),
        sites,
        waves,
    })
}

pub(crate) fn point_charge_support(
    auxiliary: &CompiledAuxiliaryBasis,
    request: &CoulombRequest,
    projection: &InterpolationProjection,
) -> Result<ExpansionSupport, CoulombError> {
    let mut sites = Vec::with_capacity(auxiliary.partition.site_count());
    for site in auxiliary.partition.sites() {
        sites.push(SiteSupport {
            position: site.position,
            radius: site.radius,
            mesh: point_charge_mesh(site.radius)?,
        });
    }
    Ok(ExpansionSupport {
        volume: auxiliary.partition.interstitial().cell_volume().get(),
        sites,
        waves: auxiliary_waves(request, auxiliary.q, projection.pw_cutoff)?,
    })
}

pub(crate) fn sampled_zeta_densities(
    sampled: &SampledAuxiliaryFunctions,
    support: &ExpansionSupport,
    projection: &InterpolationProjection,
) -> Result<Vec<ChargeDensity>, CoulombError> {
    let n_mu = sampled.n_mu();
    let n_grid = sampled.n_grid();
    let n_pw = support.waves.len();
    let channels_per_site = lm_count(projection.l_max);
    let mut densities = vec![
        ChargeDensity {
            mt: Vec::new(),
            pw: vec![Complex64::default(); n_pw],
        };
        n_mu
    ];
    let channel_template = support
        .sites
        .iter()
        .enumerate()
        .flat_map(|(site, site_support)| {
            (0..=projection.l_max).flat_map(move |l| {
                (-(l as i32)..=l as i32).map(move |m| SampledMtChannel {
                    site,
                    l,
                    m,
                    radial: vec![Complex64::default(); site_support.mesh.len()],
                })
            })
        })
        .collect::<Vec<_>>();
    let mut mt_channels = vec![channel_template; n_mu];
    for p in 0..n_grid {
        #[cfg(feature = "fft-fftw")]
        if sampled.uniform_interstitial_grid.is_some()
            && matches!(
                sampled.supports[p],
                SampledPointSupport::Interstitial | SampledPointSupport::Uniform
            )
        {
            continue;
        }
        let weight = sampled.weights[p].get();
        if weight == 0.0 {
            continue;
        }
        for (mu, density) in densities.iter_mut().enumerate() {
            let zeta = sampled.zeta[p * n_mu + mu];
            if zeta.norm() == 0.0 {
                continue;
            }
            let charge = zeta * weight;
            match sampled.supports[p] {
                SampledPointSupport::MuffinTin { site, radial_index } => {
                    let site_support = &support.sites[site];
                    let rel: [f64; 3] = std::array::from_fn(|axis| {
                        sampled.points[p][axis].get() - site_support.position[axis].get()
                    });
                    let harmonics = complex_spherical_harmonics(projection.l_max, rel);
                    let radial_scale = 1.0
                        / (site_support.mesh.weights()[radial_index]
                            * site_support.mesh.radii()[radial_index].get());
                    for l in 0..=projection.l_max {
                        for m in -(l as i32)..=l as i32 {
                            let lm = lm_index(l, m)?;
                            mt_channels[mu][site * channels_per_site + lm].radial[radial_index] +=
                                harmonics[lm].conj() * charge * radial_scale;
                        }
                    }
                }
                SampledPointSupport::Interstitial | SampledPointSupport::Uniform => {
                    accumulate_interstitial(density, support, sampled.points[p], charge);
                }
            }
        }
    }
    #[cfg(feature = "fft-fftw")]
    if let Some(uniform) = &sampled.uniform_interstitial_grid {
        accumulate_uniform_interstitial_fft(sampled, support, uniform, &mut densities)?;
    }
    for (density, channels) in densities.iter_mut().zip(mt_channels) {
        for channel in channels {
            let real = channel
                .radial
                .iter()
                .map(|value| value.re)
                .collect::<Vec<_>>();
            if real.iter().any(|value| *value != 0.0) {
                density.mt.push(MtPiece {
                    site: channel.site,
                    l: channel.l,
                    m: channel.m,
                    mpb_mode: None,
                    radial: real,
                    amplitude: Complex64::new(1.0, 0.0),
                });
            }
            let imaginary = channel
                .radial
                .iter()
                .map(|value| value.im)
                .collect::<Vec<_>>();
            if imaginary.iter().any(|value| *value != 0.0) {
                density.mt.push(MtPiece {
                    site: channel.site,
                    l: channel.l,
                    m: channel.m,
                    mpb_mode: None,
                    radial: imaginary,
                    amplitude: Complex64::new(0.0, 1.0),
                });
            }
        }
    }
    Ok(densities)
}

fn natural_grid_index(
    cell: Cell,
    reciprocal: &ReciprocalLattice,
    dimensions: [usize; 3],
    coordinate: [Bohr; 3],
    point: usize,
) -> Result<usize, CoulombError> {
    let mut indices = [0usize; 3];
    for axis in 0..3 {
        let fractional = coordinate
            .iter()
            .zip(&reciprocal.basis()[axis])
            .map(|(r, b)| r.get() * b.get())
            .sum::<f64>()
            / std::f64::consts::TAU;
        let index = (fractional * dimensions[axis] as f64 - 0.5).round();
        if index < 0.0 || index >= dimensions[axis] as f64 {
            return Err(CoulombError::SampledPointNotOnUniformGrid(point));
        }
        indices[axis] = index as usize;
    }
    let reconstructed = cell.cartesian(std::array::from_fn(|axis| {
        (indices[axis] as f64 + 0.5) / dimensions[axis] as f64
    }));
    let scale = cell
        .basis()
        .iter()
        .flatten()
        .map(|component| component.get().abs())
        .fold(1.0, f64::max);
    if reconstructed
        .iter()
        .zip(coordinate)
        .any(|(expected, actual)| (expected.get() - actual.get()).abs() > 1.0e-10 * scale)
    {
        return Err(CoulombError::SampledPointNotOnUniformGrid(point));
    }
    Ok((indices[0] * dimensions[1] + indices[1]) * dimensions[2] + indices[2])
}

#[cfg(feature = "fft-fftw")]
fn accumulate_uniform_interstitial_fft(
    sampled: &SampledAuxiliaryFunctions,
    support: &ExpansionSupport,
    uniform: &UniformInterstitialGrid,
    densities: &mut [ChargeDensity],
) -> Result<(), CoulombError> {
    use muffintin_tensor::fft::{FftGrid, FftPlan};

    let grid = FftGrid::new(uniform.dimensions)?;
    let mut plan = FftPlan::new(grid)?;
    let mut input = vec![Complex64::default(); grid.len()];
    let q = sampled.layout.q().cartesian;
    let inverse_sqrt_volume = 1.0 / support.volume.sqrt();
    for (mu, density) in densities.iter_mut().enumerate() {
        input.fill(Complex64::default());
        for (point, mapped) in uniform.sampled_to_uniform.iter().enumerate() {
            let Some(index) = mapped else { continue };
            let phase = q
                .iter()
                .zip(sampled.points[point])
                .map(|(q_axis, r_axis)| q_axis.get() * r_axis.get())
                .sum::<f64>();
            input[*index] += sampled.zeta[point * sampled.n_mu() + mu]
                * sampled.weights[point].get()
                * Complex64::from_polar(1.0, -phase);
        }
        let spectrum = plan.forward(&input)?;
        for (local, wave) in support.waves.iter().enumerate() {
            let midpoint_phase = -PI
                * wave
                    .g
                    .index
                    .iter()
                    .zip(uniform.dimensions)
                    .map(|(&mode, count)| f64::from(mode) / count as f64)
                    .sum::<f64>();
            density.pw[local] += spectrum[grid.index(wave.g.index)]
                * Complex64::from_polar(1.0, midpoint_phase)
                * inverse_sqrt_volume;
        }
    }
    Ok(())
}

/// Toy Ewald path: each interpolation *node* is a quadrature-weighted point charge.
///
/// This is not the production $\zeta$ metric.
pub(crate) fn point_charge_densities(
    auxiliary: &CompiledAuxiliaryBasis,
    support: &ExpansionSupport,
    projection: &InterpolationProjection,
) -> Result<Vec<ChargeDensity>, CoulombError> {
    let points = auxiliary.require_interpolation_points()?;
    let mut densities = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        let mut density = ChargeDensity {
            mt: Vec::new(),
            pw: vec![Complex64::default(); support.waves.len()],
        };
        accumulate_point_charge(
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

fn accumulate_point_charge(
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
            let site_support = support
                .sites
                .get(site)
                .ok_or(CoulombError::SampledPointSite { site })?;
            let rel: [f64; 3] = std::array::from_fn(|axis| {
                coordinate[axis].get() - site_support.position[axis].get()
            });
            let radius = rel.iter().map(|value| value * value).sum::<f64>().sqrt();
            let spike = nearest_radial_delta(&site_support.mesh, radius);
            accumulate_muffin_tin(
                density,
                site_support,
                site,
                coordinate,
                spike,
                charge,
                l_max,
                index,
            )?;
        }
        InterpolationRegion::Interstitial | InterpolationRegion::Uniform => {
            accumulate_interstitial(density, support, coordinate, charge);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn accumulate_muffin_tin(
    density: &mut ChargeDensity,
    site_support: &SiteSupport,
    site: usize,
    coordinate: [Bohr; 3],
    spike: Vec<f64>,
    charge: Complex64,
    l_max: u32,
    index: usize,
) -> Result<(), CoulombError> {
    let rel: [f64; 3] =
        std::array::from_fn(|axis| coordinate[axis].get() - site_support.position[axis].get());
    let radius = rel.iter().map(|value| value * value).sum::<f64>().sqrt();
    if radius > site_support.radius.get() * (1.0 + 1.0e-8) {
        return Err(CoulombError::InterpolationPointOutsideSphere(index));
    }
    let harmonics = complex_spherical_harmonics(l_max, rel);
    for l in 0..=l_max {
        for m in -(l as i32)..=l as i32 {
            density.mt.push(MtPiece {
                site,
                l,
                m,
                mpb_mode: None,
                radial: spike.clone(),
                amplitude: harmonics[lm_index(l, m)?].conj() * charge,
            });
        }
    }
    Ok(())
}

fn accumulate_interstitial(
    density: &mut ChargeDensity,
    support: &ExpansionSupport,
    coordinate: [Bohr; 3],
    charge: Complex64,
) {
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

fn nearest_radial_delta(mesh: &ExponentialMesh, r0: f64) -> Vec<f64> {
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

fn point_charge_mesh(radius: Bohr) -> Result<ExponentialMesh, CoulombError> {
    let first = if radius.get() > 1.0e-5 {
        1.0e-5
    } else {
        radius.get() * 1.0e-5
    };
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "fft-fftw")]
    use muffintin_core::{GVector, Grid, InterstitialGrid, Sphere};
    use muffintin_prodbasis::{AuxiliaryRegion, InterpolationRegion};

    fn interstitial_layout(q: TransferQ, n_mu: usize) -> AuxiliaryLayout {
        AuxiliaryLayout::from_regions(
            q,
            (0..n_mu)
                .map(|id| AuxiliaryRegion::InterpolationPoint {
                    id,
                    region: InterpolationRegion::Interstitial,
                })
                .collect(),
        )
    }

    #[test]
    fn uniform_grid_tag_rejects_non_midpoint_sample() {
        let cell = Cell::new([
            [Bohr(4.0), Bohr(0.0), Bohr(0.0)],
            [Bohr(0.0), Bohr(4.0), Bohr(0.0)],
            [Bohr(0.0), Bohr(0.0), Bohr(4.0)],
        ])
        .unwrap();
        let uniform = UniformGrid::new(cell, [2, 2, 2]).unwrap();
        let q = TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
        let sampled = SampledAuxiliaryFunctions::new(
            interstitial_layout(q, 1),
            Vec::new(),
            vec![[Bohr(1.1), Bohr(1.0), Bohr(1.0)]],
            vec![VolumeBohr3(1.0)],
            vec![SampledPointSupport::Interstitial],
            vec![Complex64::new(1.0, 0.0)],
        )
        .unwrap();
        assert!(matches!(
            sampled.with_uniform_interstitial_grid(&uniform),
            Err(CoulombError::SampledPointNotOnUniformGrid(0))
        ));
    }

    #[cfg(feature = "fft-fftw")]
    #[test]
    fn fft_projection_matches_direct_sum_for_finite_q_offcentre_mask() {
        let cell = Cell::new([
            [Bohr(5.0), Bohr(0.0), Bohr(0.0)],
            [Bohr(0.4), Bohr(4.5), Bohr(0.0)],
            [Bohr(0.2), Bohr(0.3), Bohr(4.0)],
        ])
        .unwrap();
        let dimensions = [4, 5, 6];
        let uniform = UniformGrid::new(cell, dimensions).unwrap();
        let masked = InterstitialGrid::new(
            &uniform,
            &[Sphere {
                center: cell.cartesian([0.23, 0.41, 0.37]),
                radius: Bohr(1.05),
            }],
        )
        .unwrap();
        let reciprocal = ReciprocalLattice::from_direct(*cell.basis()).unwrap();
        let q_cartesian = std::array::from_fn(|cartesian_axis| {
            InverseBohr(
                [0.27, -0.19, 0.11]
                    .into_iter()
                    .zip(reciprocal.basis())
                    .map(|(coefficient, basis)| coefficient * basis[cartesian_axis].get())
                    .sum(),
            )
        });
        let q = TransferQ::from_cartesian(q_cartesian).unwrap();
        let n_mu = 2;
        let points = masked
            .points()
            .iter()
            .map(|point| point.position)
            .collect::<Vec<_>>();
        let weights = masked
            .points()
            .iter()
            .map(|point| point.weight)
            .collect::<Vec<_>>();
        let supports = vec![SampledPointSupport::Interstitial; points.len()];
        let zeta = points
            .iter()
            .enumerate()
            .flat_map(|(point, coordinate)| {
                (0..n_mu).map(move |mu| {
                    let x = coordinate[0].get();
                    let y = coordinate[1].get();
                    Complex64::new(
                        (0.13 * point as f64 + 0.7 * mu as f64 + x).sin(),
                        (0.21 * point as f64 - 0.4 * mu as f64 + y).cos(),
                    )
                })
            })
            .collect::<Vec<_>>();
        let direct = SampledAuxiliaryFunctions::new(
            interstitial_layout(q, n_mu),
            Vec::new(),
            points,
            weights,
            supports,
            zeta,
        )
        .unwrap();
        let fft = direct
            .clone()
            .with_uniform_interstitial_grid(&uniform)
            .unwrap();
        let waves = [[0, 0, 0], [1, -1, 2], [5, -6, 8], [-7, 4, -9]]
            .into_iter()
            .map(|index| {
                let cartesian = reciprocal.cartesian(index);
                let g = GVector {
                    index,
                    cartesian,
                    norm: InverseBohr(
                        cartesian
                            .iter()
                            .map(|component| component.get().powi(2))
                            .sum::<f64>()
                            .sqrt(),
                    ),
                };
                let q_plus_g = std::array::from_fn(|axis| {
                    InverseBohr(q.cartesian[axis].get() + cartesian[axis].get())
                });
                AuxiliaryInterstitialWave {
                    g,
                    q_plus_g,
                    q_plus_g_norm: InverseBohr(
                        q_plus_g
                            .iter()
                            .map(|component| component.get().powi(2))
                            .sum::<f64>()
                            .sqrt(),
                    ),
                }
            })
            .collect();
        let support = ExpansionSupport {
            volume: cell.volume().get(),
            sites: Vec::new(),
            waves,
        };
        let projection = InterpolationProjection::new(InverseBohr(20.0), 0).unwrap();
        let direct_densities = sampled_zeta_densities(&direct, &support, &projection).unwrap();
        let fft_densities = sampled_zeta_densities(&fft, &support, &projection).unwrap();
        for (mu, (direct, fft)) in direct_densities.iter().zip(&fft_densities).enumerate() {
            for (mode, (direct, fft)) in direct.pw.iter().zip(&fft.pw).enumerate() {
                assert!(
                    (direct - fft).norm() < 3.0e-12,
                    "mu {mu}, mode {mode}: direct={direct:?}, fft={fft:?}"
                );
            }
        }
    }
}
