//! Deterministic regional transforms around the pointwise LDA/PBE kernel.

use crate::{
    DensityJet2, InterstitialField, MuffinTinField, RegionalDensity, RegionalError,
    RegionalPotential, XcError, XcFunctional, evaluate_xc_point,
};
use muffintin_core::{
    Bohr, FourierFieldError, FourierLayout, Hartree, Lm, MeshError, complex_spherical_harmonics,
    lm_count, real_spherical_harmonics,
};
use muffintin_grid::{AngularGrid, Cell, Grid, GridError, InterstitialGrid, UniformGrid};
use muffintin_lapw::Collinear;
use muffintin_sphere::{HarmonicConvention, SphereField, SphereFieldError};
use num_complex::Complex64;
use std::collections::BTreeMap;
use std::f64::consts::TAU;
use thiserror::Error;

const REAL_TOLERANCE: f64 = 4096.0 * f64::EPSILON;
const DERIVATIVE_RADIUS_FRACTION: f64 = 0.2;
const DERIVATIVE_SPACING_FRACTION: f64 = 0.25;

/// Deterministic transform controls for regional exchange-correlation fields.
///
/// The interstitial midpoint grid is a convergence parameter for the nonlinear
/// direct/inverse Fourier transform. The seedless Fibonacci rule controls the
/// angular projection. Muffin-tin Cartesian derivatives use fourth-order
/// symmetric differences with a step equal to one quarter of the local radial
/// spacing, capped at one fifth of the shell radius; five neighboring
/// exponential-mesh samples provide the local quartic radial interpolant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct XcFieldSpec {
    pub interstitial_divisions: [usize; 3],
    pub angular_point_count: usize,
    pub output_l_max: u32,
}

impl XcFieldSpec {
    fn validate(&self, layout: &FourierLayout) -> Result<(), RegionalXcError> {
        if self.interstitial_divisions.contains(&0) {
            return Err(RegionalXcError::ZeroInterstitialDivision(
                self.interstitial_divisions,
            ));
        }
        if self.angular_point_count == 0 {
            return Err(RegionalXcError::ZeroAngularPointCount);
        }
        if self.output_l_max > i32::MAX as u32 {
            return Err(RegionalXcError::OutputLMaxTooLarge(self.output_l_max));
        }
        let channel_count = u64::from(self.output_l_max + 1).pow(2);
        if channel_count > usize::MAX as u64 {
            return Err(RegionalXcError::OutputLMaxTooLarge(self.output_l_max));
        }
        if self.angular_point_count < channel_count as usize {
            return Err(RegionalXcError::UndersampledAngularGrid {
                points: self.angular_point_count,
                channels: channel_count as usize,
            });
        }
        for axis in 0..3 {
            let maximum = layout
                .vectors()
                .iter()
                .map(|vector| i64::from(vector.index[axis]).abs())
                .max()
                .unwrap_or(0);
            let required = 2_i64
                .checked_mul(maximum)
                .and_then(|value| value.checked_add(1))
                .ok_or(RegionalXcError::FourierIndexRange)?;
            if self.interstitial_divisions[axis] < required as usize {
                return Err(RegionalXcError::UndersampledFourierAxis {
                    axis,
                    divisions: self.interstitial_divisions[axis],
                    required: required as usize,
                });
            }
        }
        Ok(())
    }
}

/// Regional XC potential and the two energy-functional contractions needed by SCF.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionalXcResult {
    pub potential: RegionalPotential,
    /// $E_{xc} = \int rho\,epsilon_{xc}$.
    pub exchange_correlation_energy: Hartree,
    /// $\int \sum_sigma rho_sigma v_{xc,sigma}$.
    pub density_potential_integral: Hartree,
}

/// Evaluate LDA/PW92 or PBE over both physical regions.
pub fn evaluate_regional_xc(
    functional: XcFunctional,
    density: &RegionalDensity,
    spec: XcFieldSpec,
) -> Result<RegionalXcResult, RegionalXcError> {
    let layout = density.interstitial().up.layout();
    spec.validate(layout)?;
    let angular = AngularGrid::fibonacci(spec.angular_point_count)?;

    let interstitial = transform_interstitial(functional, density, spec.interstitial_divisions)?;
    let muffin_tin = transform_muffin_tins(functional, density, &angular, spec.output_l_max)?;
    let potential = RegionalPotential::new(muffin_tin.fields, interstitial.fields)?;
    Ok(RegionalXcResult {
        potential,
        exchange_correlation_energy: Hartree(
            muffin_tin.exchange_correlation_energy + interstitial.exchange_correlation_energy,
        ),
        density_potential_integral: Hartree(
            muffin_tin.density_potential_integral + interstitial.density_potential_integral,
        ),
    })
}

struct RegionTransform {
    fields: Collinear<Vec<MuffinTinField>>,
    exchange_correlation_energy: f64,
    density_potential_integral: f64,
}

struct InterstitialTransform {
    fields: Collinear<InterstitialField>,
    exchange_correlation_energy: f64,
    density_potential_integral: f64,
}

fn transform_interstitial(
    functional: XcFunctional,
    density: &RegionalDensity,
    divisions: [usize; 3],
) -> Result<InterstitialTransform, RegionalXcError> {
    let layout = density.interstitial().up.layout();
    let cell = direct_cell(layout)?;
    let uniform = UniformGrid::new(cell, divisions)?;
    let interstitial_grid = InterstitialGrid::new(&uniform, density.geometry().spheres())?;
    let volume = density.geometry().cell_volume().get();
    let mut coefficients = Collinear::new(
        vec![Complex64::new(0.0, 0.0); layout.len()],
        vec![Complex64::new(0.0, 0.0); layout.len()],
    );
    let mut exchange_correlation_energy = 0.0;
    let mut density_potential_integral = 0.0;

    for point in interstitial_grid.points() {
        let jet = interstitial_density_jet(density.interstitial(), point.position)?;
        let xc = evaluate_xc_point(functional, jet)?;
        exchange_correlation_energy += point.weight.get() * xc.energy_density;
        density_potential_integral += point.weight.get()
            * (jet.rho[0] * xc.potential[0].get() + jet.rho[1] * xc.potential[1].get());
        let normalized_weight = point.weight.get() / volume;
        for (position, vector) in layout.vectors().iter().enumerate() {
            let phase = -dot_g_r(vector.cartesian, point.position);
            let transform = Complex64::from_polar(normalized_weight, phase);
            coefficients.up[position] += xc.potential[0].get() * transform;
            coefficients.down[position] += xc.potential[1].get() * transform;
        }
    }
    enforce_fourier_reality(layout, &mut coefficients.up)?;
    enforce_fourier_reality(layout, &mut coefficients.down)?;
    let fields = Collinear::new(
        interstitial_from_ordered(layout.clone(), coefficients.up)?,
        interstitial_from_ordered(layout.clone(), coefficients.down)?,
    );
    Ok(InterstitialTransform {
        fields,
        exchange_correlation_energy,
        density_potential_integral,
    })
}

fn transform_muffin_tins(
    functional: XcFunctional,
    density: &RegionalDensity,
    angular: &AngularGrid,
    output_l_max: u32,
) -> Result<RegionTransform, RegionalXcError> {
    let mut fields = Collinear::new(Vec::new(), Vec::new());
    let mut exchange_correlation_energy = 0.0;
    let mut density_potential_integral = 0.0;
    for (up, down) in density
        .muffin_tins()
        .up
        .iter()
        .zip(&density.muffin_tins().down)
    {
        let transformed = transform_muffin_tin(functional, up, down, angular, output_l_max)?;
        fields.up.push(transformed.fields.up);
        fields.down.push(transformed.fields.down);
        exchange_correlation_energy += transformed.exchange_correlation_energy;
        density_potential_integral += transformed.density_potential_integral;
    }
    Ok(RegionTransform {
        fields,
        exchange_correlation_energy,
        density_potential_integral,
    })
}

struct MuffinTinTransform {
    fields: Collinear<MuffinTinField>,
    exchange_correlation_energy: f64,
    density_potential_integral: f64,
}

fn transform_muffin_tin(
    functional: XcFunctional,
    up: &MuffinTinField,
    down: &MuffinTinField,
    angular: &AngularGrid,
    output_l_max: u32,
) -> Result<MuffinTinTransform, RegionalXcError> {
    if up.mesh() != down.mesh() {
        return Err(RegionalXcError::MuffinTinSpinMeshMismatch);
    }
    let mesh = up.mesh();
    let channel_count = lm_count(output_l_max);
    let mut projected = Collinear::new(
        vec![vec![Complex64::new(0.0, 0.0); mesh.len()]; channel_count],
        vec![vec![Complex64::new(0.0, 0.0); mesh.len()]; channel_count],
    );
    let mut radial_energy = vec![0.0; mesh.len()];
    let mut radial_density_potential = vec![0.0; mesh.len()];
    let convention = up.field().convention();

    for radial_index in 0..mesh.len() {
        let radius = mesh.radius(radial_index).unwrap().get();
        let derivative_step = derivative_step(mesh.radii(), radial_index);
        for point in angular.points() {
            let position = point.direction.map(|component| component * radius);
            let jet = muffin_tin_density_jet(up, down, radial_index, position, derivative_step)?;
            let xc = evaluate_xc_point(functional, jet)?;
            radial_energy[radial_index] += point.weight * xc.energy_density;
            radial_density_potential[radial_index] += point.weight
                * (jet.rho[0] * xc.potential[0].get() + jet.rho[1] * xc.potential[1].get());
            project_angular_value(
                convention,
                output_l_max,
                point.direction,
                point.weight,
                xc.potential[0].get(),
                radial_index,
                &mut projected.up,
            );
            project_angular_value(
                convention,
                output_l_max,
                point.direction,
                point.weight,
                xc.potential[1].get(),
                radial_index,
                &mut projected.down,
            );
        }
        let radius_squared = radius * radius;
        radial_energy[radial_index] *= radius_squared;
        radial_density_potential[radial_index] *= radius_squared;
    }

    let up_field = finish_sphere_projection(convention, output_l_max, projected.up)?;
    let down_field = finish_sphere_projection(convention, output_l_max, projected.down)?;
    Ok(MuffinTinTransform {
        fields: Collinear::new(
            MuffinTinField::new(mesh.clone(), up_field)?,
            MuffinTinField::new(mesh.clone(), down_field)?,
        ),
        exchange_correlation_energy: mesh.integrate(&radial_energy)?,
        density_potential_integral: mesh.integrate(&radial_density_potential)?,
    })
}

pub(crate) fn interstitial_density_jet(
    fields: &Collinear<InterstitialField>,
    position: [Bohr; 3],
) -> Result<DensityJet2, RegionalXcError> {
    let up = interstitial_spin_jet(&fields.up, position)?;
    let down = interstitial_spin_jet(&fields.down, position)?;
    Ok(DensityJet2 {
        rho: [up.0, down.0],
        gradient: [up.1, down.1],
        hessian: [up.2, down.2],
    })
}

fn interstitial_spin_jet(
    field: &InterstitialField,
    position: [Bohr; 3],
) -> Result<(f64, [f64; 3], [f64; 6]), RegionalXcError> {
    let mut value = Complex64::new(0.0, 0.0);
    let mut gradient = [Complex64::new(0.0, 0.0); 3];
    let mut hessian = [Complex64::new(0.0, 0.0); 6];
    let mut scale = 0.0;
    for (vector, &coefficient) in field.field().iter() {
        let g = vector.cartesian.map(|component| component.get());
        let phase = Complex64::from_polar(1.0, dot_raw(g, position.map(Bohr::get)));
        let term = coefficient * phase;
        value += term;
        for axis in 0..3 {
            gradient[axis] += Complex64::new(0.0, g[axis]) * term;
        }
        let products = [
            g[0] * g[0],
            g[1] * g[1],
            g[2] * g[2],
            g[0] * g[1],
            g[0] * g[2],
            g[1] * g[2],
        ];
        for (entry, product) in hessian.iter_mut().zip(products) {
            *entry -= product * term;
        }
        scale += term.norm()
            * (1.0
                + g.iter().map(|component| component.abs()).sum::<f64>()
                + products.iter().map(|product| product.abs()).sum::<f64>());
    }
    Ok((
        checked_real(value, scale, "interstitial density")?,
        checked_real_array(gradient, scale, "interstitial gradient")?,
        checked_real_array(hessian, scale, "interstitial Hessian")?,
    ))
}

fn muffin_tin_density_jet(
    up: &MuffinTinField,
    down: &MuffinTinField,
    radial_index: usize,
    position: [f64; 3],
    step: f64,
) -> Result<DensityJet2, RegionalXcError> {
    let up = muffin_tin_spin_jet(up, radial_index, position, step)?;
    let down = muffin_tin_spin_jet(down, radial_index, position, step)?;
    Ok(DensityJet2 {
        rho: [up.0, down.0],
        gradient: [up.1, down.1],
        hessian: [up.2, down.2],
    })
}

fn muffin_tin_spin_jet(
    field: &MuffinTinField,
    radial_index: usize,
    position: [f64; 3],
    step: f64,
) -> Result<(f64, [f64; 3], [f64; 6]), RegionalXcError> {
    const OFFSETS: [i32; 4] = [-2, -1, 1, 2];
    const FIRST_WEIGHTS: [f64; 4] = [1.0, -8.0, 8.0, -1.0];
    let center = evaluate_muffin_tin_field(field, radial_index, position)?;
    let mut axial = [[0.0; 4]; 3];
    for axis in 0..3 {
        for (slot, offset) in OFFSETS.into_iter().enumerate() {
            let mut displaced = position;
            displaced[axis] += f64::from(offset) * step;
            axial[axis][slot] = evaluate_muffin_tin_field(field, radial_index, displaced)?;
        }
    }
    let mut gradient = [0.0; 3];
    let mut hessian = [0.0; 6];
    for axis in 0..3 {
        gradient[axis] = axial[axis]
            .iter()
            .zip(FIRST_WEIGHTS)
            .map(|(&value, weight)| value * weight)
            .sum::<f64>()
            / (12.0 * step);
        hessian[axis] = (-axial[axis][3] + 16.0 * axial[axis][2] - 30.0 * center
            + 16.0 * axial[axis][1]
            - axial[axis][0])
            / (12.0 * step * step);
    }
    for (entry, (first_axis, second_axis)) in [(3, (0, 1)), (4, (0, 2)), (5, (1, 2))] {
        let mut derivative = 0.0;
        for (first_slot, first_offset) in OFFSETS.into_iter().enumerate() {
            for (second_slot, second_offset) in OFFSETS.into_iter().enumerate() {
                let mut displaced = position;
                displaced[first_axis] += f64::from(first_offset) * step;
                displaced[second_axis] += f64::from(second_offset) * step;
                derivative += FIRST_WEIGHTS[first_slot]
                    * FIRST_WEIGHTS[second_slot]
                    * evaluate_muffin_tin_field(field, radial_index, displaced)?;
            }
        }
        hessian[entry] = derivative / (144.0 * step * step);
    }
    Ok((center, gradient, hessian))
}

fn evaluate_muffin_tin_field(
    field: &MuffinTinField,
    radial_index: usize,
    position: [f64; 3],
) -> Result<f64, RegionalXcError> {
    let radius = dot_raw(position, position).sqrt();
    let l_max = field
        .field()
        .channels()
        .map(|(channel, _)| channel.l)
        .max()
        .unwrap_or(0);
    let mut value = Complex64::new(0.0, 0.0);
    match field.field().convention() {
        HarmonicConvention::Complex => {
            let harmonics = complex_spherical_harmonics(l_max, position);
            for (channel, samples) in field.field().channels() {
                value += interpolate_radial(field.mesh().radii(), samples, radial_index, radius)
                    * harmonics[channel.index()];
            }
        }
        HarmonicConvention::Real => {
            let harmonics = real_spherical_harmonics(l_max, position);
            for (channel, samples) in field.field().channels() {
                value += interpolate_radial(field.mesh().radii(), samples, radial_index, radius)
                    * harmonics[channel.index()];
            }
        }
    }
    checked_real(value, value.norm(), "muffin-tin density")
}

fn interpolate_radial(
    radii: &[Bohr],
    samples: &[Complex64],
    center: usize,
    radius: f64,
) -> Complex64 {
    let start = center.saturating_sub(2).min(radii.len() - 5);
    let mut value = Complex64::new(0.0, 0.0);
    for local in 0..5 {
        let point = start + local;
        let mut basis = 1.0;
        for other_local in 0..5 {
            if local != other_local {
                let other = start + other_local;
                basis *= (radius - radii[other].get()) / (radii[point].get() - radii[other].get());
            }
        }
        value += basis * samples[point];
    }
    value
}

fn derivative_step(radii: &[Bohr], index: usize) -> f64 {
    let radial_spacing = if index == 0 {
        radii[1].get() - radii[0].get()
    } else if index + 1 == radii.len() {
        radii[index].get() - radii[index - 1].get()
    } else {
        (radii[index + 1].get() - radii[index - 1].get()) / 2.0
    };
    (DERIVATIVE_SPACING_FRACTION * radial_spacing.abs())
        .min(DERIVATIVE_RADIUS_FRACTION * radii[index].get())
}

fn project_angular_value(
    convention: HarmonicConvention,
    l_max: u32,
    direction: [f64; 3],
    weight: f64,
    value: f64,
    radial_index: usize,
    projected: &mut [Vec<Complex64>],
) {
    match convention {
        HarmonicConvention::Complex => {
            for (target, harmonic) in projected
                .iter_mut()
                .zip(complex_spherical_harmonics(l_max, direction))
            {
                target[radial_index] += weight * value * harmonic.conj();
            }
        }
        HarmonicConvention::Real => {
            for (target, harmonic) in projected
                .iter_mut()
                .zip(real_spherical_harmonics(l_max, direction))
            {
                target[radial_index] += weight * value * harmonic;
            }
        }
    }
}

fn finish_sphere_projection(
    convention: HarmonicConvention,
    l_max: u32,
    projected: Vec<Vec<Complex64>>,
) -> Result<SphereField, RegionalXcError> {
    let mut channels = BTreeMap::new();
    for l in 0..=l_max {
        for m in -(l as i32)..=l as i32 {
            let channel = Lm::new(l, m).expect("loop bounds validate channel");
            channels.insert(channel, projected[channel.index()].clone());
        }
    }
    enforce_sphere_reality(convention, &mut channels);
    Ok(SphereField::new(
        convention,
        channels
            .into_iter()
            .map(|(channel, values)| ((channel.l, channel.m), values)),
    )?)
}

fn enforce_sphere_reality(
    convention: HarmonicConvention,
    channels: &mut BTreeMap<Lm, Vec<Complex64>>,
) {
    if convention == HarmonicConvention::Real {
        for values in channels.values_mut() {
            for value in values {
                value.im = 0.0;
            }
        }
        return;
    }
    let l_max = channels.keys().map(|channel| channel.l).max().unwrap_or(0);
    for l in 0..=l_max {
        let zero = Lm::new(l, 0).unwrap();
        if let Some(values) = channels.get_mut(&zero) {
            for value in values {
                value.im = 0.0;
            }
        }
        for m in 1..=l as i32 {
            let positive = Lm::new(l, m).unwrap();
            let negative = Lm::new(l, -m).unwrap();
            let phase = if m % 2 == 0 { 1.0 } else { -1.0 };
            let negative_values = channels[&negative].clone();
            let positive_values = channels.get_mut(&positive).unwrap();
            for (positive_value, negative_value) in positive_values.iter_mut().zip(&negative_values)
            {
                *positive_value = (*positive_value + phase * negative_value.conj()) / 2.0;
            }
            let positive_values = channels[&positive].clone();
            let negative_values = channels.get_mut(&negative).unwrap();
            for (negative_value, positive_value) in negative_values.iter_mut().zip(&positive_values)
            {
                *negative_value = phase * positive_value.conj();
            }
        }
    }
}

fn interstitial_from_ordered(
    layout: FourierLayout,
    coefficients: Vec<Complex64>,
) -> Result<InterstitialField, RegionalXcError> {
    let map = layout
        .vectors()
        .iter()
        .zip(coefficients)
        .map(|(vector, coefficient)| (vector.index, coefficient))
        .collect();
    Ok(InterstitialField::new(layout, map)?)
}

fn enforce_fourier_reality(
    layout: &FourierLayout,
    coefficients: &mut [Complex64],
) -> Result<(), RegionalXcError> {
    for vector in layout.vectors() {
        let position = layout.index(vector.index).unwrap();
        let opposite_index = [
            vector.index[0]
                .checked_neg()
                .ok_or(RegionalXcError::FourierIndexRange)?,
            vector.index[1]
                .checked_neg()
                .ok_or(RegionalXcError::FourierIndexRange)?,
            vector.index[2]
                .checked_neg()
                .ok_or(RegionalXcError::FourierIndexRange)?,
        ];
        let opposite = layout
            .index(opposite_index)
            .ok_or(RegionalXcError::Fourier(
                FourierFieldError::MissingConjugate {
                    index: vector.index,
                },
            ))?;
        if position == opposite {
            coefficients[position].im = 0.0;
        } else if position < opposite {
            let average = (coefficients[position] + coefficients[opposite].conj()) / 2.0;
            coefficients[position] = average;
            coefficients[opposite] = average.conj();
        }
    }
    Ok(())
}

fn direct_cell(layout: &FourierLayout) -> Result<Cell, RegionalXcError> {
    let reciprocal = layout.reciprocal().basis();
    let b = reciprocal.map(|vector| vector.map(|component| component.get()));
    let determinant = dot_raw(b[0], cross(b[1], b[2]));
    let direct = [
        scale(cross(b[1], b[2]), TAU / determinant),
        scale(cross(b[2], b[0]), TAU / determinant),
        scale(cross(b[0], b[1]), TAU / determinant),
    ]
    .map(|vector| vector.map(Bohr));
    Ok(Cell::new(direct)?)
}

fn checked_real(
    value: Complex64,
    scale: f64,
    quantity: &'static str,
) -> Result<f64, RegionalXcError> {
    let tolerance = REAL_TOLERANCE * scale.max(value.re.abs()).max(1.0);
    if value.im.abs() > tolerance {
        Err(RegionalXcError::NonRealTransform {
            quantity,
            imaginary: value.im,
            tolerance,
        })
    } else {
        Ok(value.re)
    }
}

fn checked_real_array<const N: usize>(
    values: [Complex64; N],
    scale: f64,
    quantity: &'static str,
) -> Result<[f64; N], RegionalXcError> {
    let mut result = [0.0; N];
    for (target, value) in result.iter_mut().zip(values) {
        *target = checked_real(value, scale, quantity)?;
    }
    Ok(result)
}

fn dot_g_r(g: [muffintin_core::InverseBohr; 3], r: [Bohr; 3]) -> f64 {
    g.into_iter()
        .zip(r)
        .map(|(left, right)| left.get() * right.get())
        .sum()
}

fn dot_raw(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(x, y)| x * y).sum()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    vector.map(|component| factor * component)
}

/// Invalid transform controls, representation, or regional XC evaluation.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RegionalXcError {
    #[error("interstitial grid divisions must be nonzero, got {0:?}")]
    ZeroInterstitialDivision([usize; 3]),
    #[error("angular point count must be nonzero")]
    ZeroAngularPointCount,
    #[error("output l_max is too large: {0}")]
    OutputLMaxTooLarge(u32),
    #[error("reciprocal integer coordinate cannot be negated or bounded safely")]
    FourierIndexRange,
    #[error(
        "interstitial grid axis {axis} has {divisions} points, needs at least {required} to represent the input Fourier layout"
    )]
    UndersampledFourierAxis {
        axis: usize,
        divisions: usize,
        required: usize,
    },
    #[error("angular grid has {points} points, fewer than the {channels} output harmonic channels")]
    UndersampledAngularGrid { points: usize, channels: usize },
    #[error("spin channels use different muffin-tin radial meshes")]
    MuffinTinSpinMeshMismatch,
    #[error("{quantity} has imaginary part {imaginary}, tolerance {tolerance}")]
    NonRealTransform {
        quantity: &'static str,
        imaginary: f64,
        tolerance: f64,
    },
    #[error(transparent)]
    Xc(#[from] XcError),
    #[error(transparent)]
    Regional(#[from] RegionalError),
    #[error(transparent)]
    Fourier(#[from] FourierFieldError),
    #[error(transparent)]
    Sphere(#[from] SphereFieldError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    Grid(#[from] GridError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::{
        GVector, InterstitialGeometry, InverseBohr, ReciprocalLattice, Sphere, VolumeBohr3,
    };
    use std::f64::consts::PI;

    fn reciprocal() -> ReciprocalLattice {
        ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap()
    }

    fn layout(indices: &[[i32; 3]]) -> FourierLayout {
        let reciprocal = reciprocal();
        FourierLayout::new(
            reciprocal,
            indices
                .iter()
                .map(|&index| {
                    let cartesian = reciprocal.cartesian(index);
                    let norm = cartesian
                        .iter()
                        .map(|component| component.get().powi(2))
                        .sum::<f64>()
                        .sqrt();
                    GVector {
                        index,
                        cartesian,
                        norm: InverseBohr(norm),
                    }
                })
                .collect(),
        )
        .unwrap()
    }

    fn interstitial_field(
        layout: FourierLayout,
        coefficients: impl IntoIterator<Item = ([i32; 3], Complex64)>,
    ) -> InterstitialField {
        InterstitialField::new(layout, coefficients.into_iter().collect()).unwrap()
    }

    fn interstitial_density(
        indices: &[[i32; 3]],
        up: impl IntoIterator<Item = ([i32; 3], Complex64)>,
        down: impl IntoIterator<Item = ([i32; 3], Complex64)>,
    ) -> RegionalDensity {
        let layout = layout(indices);
        RegionalDensity::new(
            InterstitialGeometry::new(VolumeBohr3(TAU.powi(3)), Vec::new()).unwrap(),
            Collinear::new(Vec::new(), Vec::new()),
            Collinear::new(
                interstitial_field(layout.clone(), up),
                interstitial_field(layout, down),
            ),
        )
        .unwrap()
    }

    fn uniform_density(up: f64, down: f64) -> RegionalDensity {
        interstitial_density(
            &[[0; 3]],
            [([0; 3], Complex64::new(up, 0.0))],
            [([0; 3], Complex64::new(down, 0.0))],
        )
    }

    fn spec(divisions_x: usize) -> XcFieldSpec {
        XcFieldSpec {
            interstitial_divisions: [divisions_x, 3, 3],
            angular_point_count: 50,
            output_l_max: 0,
        }
    }

    #[test]
    fn uniform_lda_and_pbe_match_constant_point_kernel_and_direct_integrals() {
        let density = uniform_density(0.2, 0.1);
        let lda = evaluate_regional_xc(XcFunctional::LdaPw92, &density, spec(4)).unwrap();
        let pbe = evaluate_regional_xc(XcFunctional::Pbe, &density, spec(4)).unwrap();
        let point = evaluate_xc_point(
            XcFunctional::LdaPw92,
            DensityJet2 {
                rho: [0.2, 0.1],
                gradient: [[0.0; 3]; 2],
                hessian: [[0.0; 6]; 2],
            },
        )
        .unwrap();
        for spin in 0..2 {
            let lda_field = if spin == 0 {
                &lda.potential.interstitial().up
            } else {
                &lda.potential.interstitial().down
            };
            let pbe_field = if spin == 0 {
                &pbe.potential.interstitial().up
            } else {
                &pbe.potential.interstitial().down
            };
            assert!(
                (lda_field.coefficient([0; 3]).unwrap().re - point.potential[spin].get()).abs()
                    < 2.0e-14
            );
            assert!(
                (pbe_field.coefficient([0; 3]).unwrap().re - point.potential[spin].get()).abs()
                    < 2.0e-14
            );
        }
        let expected_energy = TAU.powi(3) * point.energy_density;
        let expected_density_potential =
            TAU.powi(3) * (0.2 * point.potential[0].get() + 0.1 * point.potential[1].get());
        assert!((lda.exchange_correlation_energy.get() - expected_energy).abs() < 2.0e-12);
        assert!(
            (lda.density_potential_integral.get() - expected_density_potential).abs() < 2.0e-12
        );
        assert!(
            (pbe.exchange_correlation_energy.get() - lda.exchange_correlation_energy.get()).abs()
                < 2.0e-12
        );
        assert!(
            (pbe.density_potential_integral.get() - lda.density_potential_integral.get()).abs()
                < 2.0e-12
        );
    }

    #[test]
    fn one_cosine_has_analytic_jet_and_nonlinear_transform_converges() {
        let indices = [[-1, 0, 0], [0; 3], [1, 0, 0]];
        let density = interstitial_density(
            &indices,
            [
                ([-1, 0, 0], Complex64::new(0.05, 0.0)),
                ([0; 3], Complex64::new(0.5, 0.0)),
                ([1, 0, 0], Complex64::new(0.05, 0.0)),
            ],
            [
                ([-1, 0, 0], Complex64::new(0.02, 0.0)),
                ([0; 3], Complex64::new(0.3, 0.0)),
                ([1, 0, 0], Complex64::new(0.02, 0.0)),
            ],
        );
        let x = 0.73;
        let jet =
            interstitial_density_jet(density.interstitial(), [Bohr(x), Bohr(0.2), Bohr(-0.4)])
                .unwrap();
        assert!((jet.rho[0] - (0.5 + 0.1 * x.cos())).abs() < 2.0e-15);
        assert!((jet.gradient[0][0] + 0.1 * x.sin()).abs() < 2.0e-15);
        assert!((jet.hessian[0][0] + 0.1 * x.cos()).abs() < 2.0e-15);
        assert_eq!(jet.gradient[0][1], 0.0);
        assert_eq!(jet.hessian[0][3], 0.0);

        let coarse = evaluate_regional_xc(XcFunctional::LdaPw92, &density, spec(4)).unwrap();
        let fine = evaluate_regional_xc(XcFunctional::LdaPw92, &density, spec(8)).unwrap();
        let reference = evaluate_regional_xc(XcFunctional::LdaPw92, &density, spec(64)).unwrap();
        let coefficient = |result: &RegionalXcResult| {
            result
                .potential
                .interstitial()
                .up
                .coefficient([1, 0, 0])
                .unwrap()
                .re
        };
        let coarse_error = (coefficient(&coarse) - coefficient(&reference)).abs();
        let fine_error = (coefficient(&fine) - coefficient(&reference)).abs();
        assert!(fine_error < coarse_error);

        let pbe = evaluate_regional_xc(XcFunctional::Pbe, &density, spec(16)).unwrap();
        let lda = evaluate_regional_xc(XcFunctional::LdaPw92, &density, spec(16)).unwrap();
        assert!(
            (pbe.exchange_correlation_energy.get() - lda.exchange_correlation_energy.get()).abs()
                > 1.0e-8
        );
        let plus = pbe
            .potential
            .interstitial()
            .up
            .coefficient([1, 0, 0])
            .unwrap();
        let minus = pbe
            .potential
            .interstitial()
            .up
            .coefficient([-1, 0, 0])
            .unwrap();
        assert_eq!(minus, plus.conj());
    }

    #[test]
    fn radial_monopole_cartesian_derivatives_match_quadratic_density() {
        let mesh = muffintin_core::ExponentialMesh::new(Bohr(0.08), 0.15, 13).unwrap();
        let base = 0.4;
        let quadratic = 0.3;
        let monopole: Vec<_> = mesh
            .radii()
            .iter()
            .map(|radius| {
                Complex64::new(
                    (4.0 * PI).sqrt() * (base + quadratic * radius.get().powi(2)),
                    0.0,
                )
            })
            .collect();
        let field = MuffinTinField::new(
            mesh.clone(),
            SphereField::new(HarmonicConvention::Real, [((0, 0), monopole)]).unwrap(),
        )
        .unwrap();
        let radial_index = 6;
        let radius = mesh.radius(radial_index).unwrap().get();
        let inverse_norm = 1.0 / 14.0_f64.sqrt();
        let position = [inverse_norm, 2.0 * inverse_norm, 3.0 * inverse_norm]
            .map(|direction| direction * radius);
        let step = derivative_step(mesh.radii(), radial_index);
        let jet = muffin_tin_spin_jet(&field, radial_index, position, step).unwrap();
        assert!((jet.0 - (base + quadratic * radius.powi(2))).abs() < 2.0e-13);
        for ((&gradient, &coordinate), &hessian) in jet.1.iter().zip(&position).zip(&jet.2[..3]) {
            assert!((gradient - 2.0 * quadratic * coordinate).abs() < 2.0e-10);
            assert!((hessian - 2.0 * quadratic).abs() < 2.0e-8);
        }
        for &mixed in &jet.2[3..] {
            assert!(mixed.abs() < 2.0e-8);
        }
    }

    #[test]
    fn spin_swap_swaps_potentials_and_preserves_both_integrals() {
        let first =
            evaluate_regional_xc(XcFunctional::Pbe, &uniform_density(0.23, 0.11), spec(4)).unwrap();
        let swapped =
            evaluate_regional_xc(XcFunctional::Pbe, &uniform_density(0.11, 0.23), spec(4)).unwrap();
        assert!(
            (first.exchange_correlation_energy.get() - swapped.exchange_correlation_energy.get())
                .abs()
                < 1.0e-13
        );
        assert!(
            (first.density_potential_integral.get() - swapped.density_potential_integral.get())
                .abs()
                < 1.0e-13
        );
        assert_eq!(
            first.potential.interstitial().up.coefficient([0; 3]),
            swapped.potential.interstitial().down.coefficient([0; 3])
        );
        assert_eq!(
            first.potential.interstitial().down.coefficient([0; 3]),
            swapped.potential.interstitial().up.coefficient([0; 3])
        );
    }

    #[test]
    fn radial_monopole_closes_to_a_real_muffin_tin_potential() {
        let mesh = muffintin_core::ExponentialMesh::new(Bohr(0.02), 0.2, 13).unwrap();
        let make_field = |density: f64| {
            MuffinTinField::new(
                mesh.clone(),
                SphereField::new(
                    HarmonicConvention::Real,
                    [(
                        (0, 0),
                        vec![Complex64::new((4.0 * PI).sqrt() * density, 0.0); mesh.len()],
                    )],
                )
                .unwrap(),
            )
            .unwrap()
        };
        let fourier_layout = layout(&[[0; 3]]);
        let zero = interstitial_field(fourier_layout.clone(), [([0; 3], Complex64::new(0.0, 0.0))]);
        let density = RegionalDensity::new(
            InterstitialGeometry::new(
                VolumeBohr3(TAU.powi(3)),
                vec![Sphere {
                    center: [Bohr(0.0); 3],
                    radius: Bohr(0.5),
                }],
            )
            .unwrap(),
            Collinear::new(vec![make_field(0.2)], vec![make_field(0.1)]),
            Collinear::new(zero.clone(), zero),
        )
        .unwrap();
        let result = evaluate_regional_xc(
            XcFunctional::LdaPw92,
            &density,
            XcFieldSpec {
                interstitial_divisions: [8, 8, 8],
                angular_point_count: 100,
                output_l_max: 0,
            },
        )
        .unwrap();
        for field in [
            &result.potential.muffin_tins().up[0],
            &result.potential.muffin_tins().down[0],
        ] {
            assert_eq!(field.field().channel_count(), 1);
            assert!(
                field
                    .field()
                    .channel(0, 0)
                    .unwrap()
                    .iter()
                    .all(|value| value.re.is_finite() && value.im == 0.0)
            );
        }
        assert!(result.exchange_correlation_energy.get().is_finite());
        assert!(result.density_potential_integral.get().is_finite());
    }

    #[test]
    fn transform_spec_rejects_aliasing_and_empty_quadratures() {
        let density = uniform_density(0.2, 0.1);
        assert!(matches!(
            evaluate_regional_xc(
                XcFunctional::LdaPw92,
                &density,
                XcFieldSpec {
                    interstitial_divisions: [0, 1, 1],
                    angular_point_count: 1,
                    output_l_max: 0,
                }
            ),
            Err(RegionalXcError::ZeroInterstitialDivision(_))
        ));
        assert!(matches!(
            evaluate_regional_xc(
                XcFunctional::LdaPw92,
                &density,
                XcFieldSpec {
                    interstitial_divisions: [1; 3],
                    angular_point_count: 0,
                    output_l_max: 0,
                }
            ),
            Err(RegionalXcError::ZeroAngularPointCount)
        ));

        let cosine = interstitial_density(
            &[[-1, 0, 0], [0; 3], [1, 0, 0]],
            [
                ([-1, 0, 0], Complex64::new(0.05, 0.0)),
                ([0; 3], Complex64::new(0.2, 0.0)),
                ([1, 0, 0], Complex64::new(0.05, 0.0)),
            ],
            [
                ([-1, 0, 0], Complex64::new(0.0, 0.0)),
                ([0; 3], Complex64::new(0.1, 0.0)),
                ([1, 0, 0], Complex64::new(0.0, 0.0)),
            ],
        );
        assert!(matches!(
            evaluate_regional_xc(XcFunctional::LdaPw92, &cosine, spec(2)),
            Err(RegionalXcError::UndersampledFourierAxis {
                axis: 0,
                divisions: 2,
                required: 3,
            })
        ));
    }
}
