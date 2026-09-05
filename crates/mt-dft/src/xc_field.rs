//! Deterministic regional transforms around the pointwise LDA/PBE kernel.

use crate::{
    DensityJet2, InterstitialField, MuffinTinField, RegionalDensity, RegionalError,
    RegionalPotential, RegionalScalarField, XcError, XcFunctional, evaluate_xc_point,
};
use muffintin_core::{AngularGrid, Cell, Grid, GridError, InterstitialGrid, UniformGrid};
use muffintin_core::{
    Bohr, FourierFieldError, FourierLayout, Hartree, Lm, MeshError, complex_spherical_harmonics,
    lm_count, real_spherical_harmonics,
};
use muffintin_sphere::{HarmonicConvention, SphereField, SphereFieldError};
use num_complex::Complex64;
use std::collections::BTreeMap;
use std::f64::consts::TAU;
use thiserror::Error;

const REAL_TOLERANCE: f64 = 4096.0 * f64::EPSILON;
const DERIVATIVE_RADIUS_FRACTION: f64 = 0.2;
const DERIVATIVE_SPACING_FRACTION: f64 = 0.25;
const LOCAL_FRAME_MAGNETIZATION_THRESHOLD: f64 = 1.0e-12;

/// How noncollinear PBE derivatives are reduced to two local spin channels.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NoncollinearXcRoute {
    /// Away from magnetization nodes, follow SPEX 06.00pre38 `potential.f`:
    /// project the density, gradient, and Hessian independently on the
    /// magnetization direction at the point.
    #[default]
    LocalSpinFrame,
    /// Treat the local eigenvalue fields `(n +/- |m|)/2` as scalar fields,
    /// including the complete Hessian of `|m|` in PBE.
    MagnetizationField,
}

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
    pub noncollinear_route: NoncollinearXcRoute,
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
    /// Exchange-correlation energy integrated over both physical regions.
    pub exchange_correlation_energy: Hartree,
    /// Integral of `n Vxc + m . Bxc` in the Pauli convention below.
    pub density_potential_integral: Hartree,
}

/// Evaluate LDA/PW92 or PBE with the selected noncollinear derivative route.
///
/// Away from magnetization nodes, [`NoncollinearXcRoute::LocalSpinFrame`]
/// follows SPEX 06.00pre38 `potential.f`: it projects charge/magnetization
/// values and their first and second derivatives onto the instantaneous
/// magnetization direction before invoking the collinear point kernel.
/// [`NoncollinearXcRoute::MagnetizationField`] instead differentiates the
/// eigenvalue fields containing `|m|`, including its transverse Hessian term.
/// With this crate's Pauli Hamiltonian `Vxc I + Bxc . sigma`, both routes use
/// `Vxc=(v_up+v_down)/2` and `Bxc=(v_up-v_down) m/(2|m|)`. Below the
/// `|m| < 1e-12` stability threshold, the direction is undefined: the local
/// polarization jet and magnetic field are set to zero rather than selecting
/// an arbitrary global axis. The shared point kernel applies its declared
/// gradient threshold in both regions; it does not reproduce the muffin-tin
/// `drhon >= drhon` typo in that SPEX snapshot.
pub fn evaluate_regional_xc(
    functional: XcFunctional,
    density: &RegionalDensity,
    spec: XcFieldSpec,
) -> Result<RegionalXcResult, RegionalXcError> {
    let layout = density.charge().interstitial().layout();
    spec.validate(layout)?;
    let angular = AngularGrid::fibonacci(spec.angular_point_count)?;

    let interstitial = transform_interstitial(
        functional,
        density,
        spec.interstitial_divisions,
        spec.noncollinear_route,
    )?;
    let muffin_tin = transform_muffin_tins(
        functional,
        density,
        &angular,
        spec.output_l_max,
        spec.noncollinear_route,
    )?;
    let [
        scalar_muffin_tins,
        bx_muffin_tins,
        by_muffin_tins,
        bz_muffin_tins,
    ] = muffin_tin.fields;
    let [
        scalar_interstitial,
        bx_interstitial,
        by_interstitial,
        bz_interstitial,
    ] = interstitial.fields;
    let scalar = RegionalScalarField::new(
        density.geometry().clone(),
        scalar_muffin_tins,
        scalar_interstitial,
    )?;
    let magnetic = [
        RegionalScalarField::new(density.geometry().clone(), bx_muffin_tins, bx_interstitial)?,
        RegionalScalarField::new(density.geometry().clone(), by_muffin_tins, by_interstitial)?,
        RegionalScalarField::new(density.geometry().clone(), bz_muffin_tins, bz_interstitial)?,
    ];
    let potential = RegionalPotential::new(scalar, magnetic)?;
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
    fields: [Vec<MuffinTinField>; 4],
    exchange_correlation_energy: f64,
    density_potential_integral: f64,
}

struct InterstitialTransform {
    fields: [InterstitialField; 4],
    exchange_correlation_energy: f64,
    density_potential_integral: f64,
}

fn transform_interstitial(
    functional: XcFunctional,
    density: &RegionalDensity,
    divisions: [usize; 3],
    route: NoncollinearXcRoute,
) -> Result<InterstitialTransform, RegionalXcError> {
    let layout = density.charge().interstitial().layout();
    let cell = direct_cell(layout)?;
    let uniform = UniformGrid::new(cell, divisions)?;
    let interstitial_grid = InterstitialGrid::new(&uniform, density.geometry().spheres())?;
    let volume = density.geometry().cell_volume().get();
    let mut coefficients: [Vec<Complex64>; 4] =
        std::array::from_fn(|_| vec![Complex64::new(0.0, 0.0); layout.len()]);
    let mut exchange_correlation_energy = 0.0;
    let mut density_potential_integral = 0.0;
    let density_fields = [
        density.charge().interstitial(),
        density.magnetization()[0].interstitial(),
        density.magnetization()[1].interstitial(),
        density.magnetization()[2].interstitial(),
    ];
    let zero_fields = density_fields.map(|field| {
        field
            .field()
            .coefficients()
            .iter()
            .all(|&coefficient| coefficient == Complex64::default())
    });

    for point in interstitial_grid.points() {
        let [charge, mx, my, mz] = [0, 1, 2, 3].map(|component| {
            if zero_fields[component] {
                Ok(FieldJet::value(0.0))
            } else if functional == XcFunctional::LdaPw92 {
                interstitial_field_value(density_fields[component], point.position)
                    .map(FieldJet::value)
            } else {
                interstitial_field_jet(density_fields[component], point.position)
            }
        });
        let xc = evaluate_noncollinear_xc_point(functional, route, charge?, [mx?, my?, mz?])?;
        exchange_correlation_energy += point.weight.get() * xc.energy_density;
        density_potential_integral += point.weight.get() * xc.density_potential;
        let normalized_weight = point.weight.get() / volume;
        for (position, vector) in layout.vectors().iter().enumerate() {
            let phase = -dot_g_r(vector.cartesian, point.position);
            let transform = Complex64::from_polar(normalized_weight, phase);
            for (target, value) in coefficients.iter_mut().zip(xc.potential) {
                if value != 0.0 {
                    target[position] += value * transform;
                }
            }
        }
    }
    for component in &mut coefficients {
        enforce_fourier_reality(layout, component)?;
    }
    let [scalar, bx, by, bz] = coefficients;
    let fields = [
        interstitial_from_ordered(layout.clone(), scalar)?,
        interstitial_from_ordered(layout.clone(), bx)?,
        interstitial_from_ordered(layout.clone(), by)?,
        interstitial_from_ordered(layout.clone(), bz)?,
    ];
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
    route: NoncollinearXcRoute,
) -> Result<RegionTransform, RegionalXcError> {
    let mut fields: [Vec<MuffinTinField>; 4] = std::array::from_fn(|_| Vec::new());
    let mut exchange_correlation_energy = 0.0;
    let mut density_potential_integral = 0.0;
    for site in 0..density.charge().muffin_tins().len() {
        let transformed = transform_muffin_tin(
            functional,
            route,
            [
                &density.charge().muffin_tins()[site],
                &density.magnetization()[0].muffin_tins()[site],
                &density.magnetization()[1].muffin_tins()[site],
                &density.magnetization()[2].muffin_tins()[site],
            ],
            angular,
            output_l_max,
        )?;
        for (target, component) in fields.iter_mut().zip(transformed.fields) {
            target.push(component);
        }
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
    fields: [MuffinTinField; 4],
    exchange_correlation_energy: f64,
    density_potential_integral: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FieldJet {
    value: f64,
    gradient: [f64; 3],
    hessian: [f64; 6],
}

impl FieldJet {
    const fn value(value: f64) -> Self {
        Self {
            value,
            gradient: [0.0; 3],
            hessian: [0.0; 6],
        }
    }
}

struct NoncollinearXcPoint {
    /// `[Vxc, Bxc_x, Bxc_y, Bxc_z]` in Hartree.
    potential: [f64; 4],
    energy_density: f64,
    density_potential: f64,
}

fn evaluate_noncollinear_xc_point(
    functional: XcFunctional,
    route: NoncollinearXcRoute,
    charge: FieldJet,
    magnetization: [FieldJet; 3],
) -> Result<NoncollinearXcPoint, RegionalXcError> {
    let magnitude = magnetization
        .iter()
        .fold(0.0_f64, |norm, component| norm.hypot(component.value));
    let magnetization_node = magnitude < LOCAL_FRAME_MAGNETIZATION_THRESHOLD;
    let direction = if magnetization_node {
        [0.0; 3]
    } else {
        magnetization.map(|component| component.value / magnitude)
    };
    let projected = if magnetization_node {
        FieldJet {
            value: 0.0,
            gradient: [0.0; 3],
            hessian: [0.0; 6],
        }
    } else if route == NoncollinearXcRoute::MagnetizationField {
        magnetization_modulus_jet(magnetization, magnitude)
    } else {
        project_magnetization_jet(magnetization, direction)
    };
    let jet = split_local_spin_jet(charge, projected);
    let xc = evaluate_xc_point(functional, jet)?;
    let scalar = 0.5 * (xc.potential[0].get() + xc.potential[1].get());
    let splitting = 0.5 * (xc.potential[0].get() - xc.potential[1].get());
    let magnetic = direction.map(|component| splitting * component);
    Ok(NoncollinearXcPoint {
        potential: [scalar, magnetic[0], magnetic[1], magnetic[2]],
        energy_density: xc.energy_density,
        density_potential: jet.rho[0] * xc.potential[0].get() + jet.rho[1] * xc.potential[1].get(),
    })
}

fn project_magnetization_jet(magnetization: [FieldJet; 3], direction: [f64; 3]) -> FieldJet {
    FieldJet {
        value: magnetization
            .iter()
            .zip(direction)
            .map(|(component, axis)| component.value * axis)
            .sum(),
        gradient: std::array::from_fn(|coordinate| {
            magnetization
                .iter()
                .zip(direction)
                .map(|(component, axis)| component.gradient[coordinate] * axis)
                .sum()
        }),
        hessian: std::array::from_fn(|coordinate| {
            magnetization
                .iter()
                .zip(direction)
                .map(|(component, axis)| component.hessian[coordinate] * axis)
                .sum()
        }),
    }
}

fn magnetization_modulus_jet(magnetization: [FieldJet; 3], magnitude: f64) -> FieldJet {
    let gradient = std::array::from_fn(|coordinate| {
        magnetization
            .iter()
            .map(|component| component.value * component.gradient[coordinate])
            .sum::<f64>()
            / magnitude
    });
    let axes = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
    let hessian = std::array::from_fn(|coordinate| {
        let (left, right) = axes[coordinate];
        let numerator = magnetization
            .iter()
            .map(|component| {
                component.gradient[left] * component.gradient[right]
                    + component.value * component.hessian[coordinate]
            })
            .sum::<f64>()
            - gradient[left] * gradient[right];
        numerator / magnitude
    });
    FieldJet {
        value: magnitude,
        gradient,
        hessian,
    }
}

fn split_local_spin_jet(charge: FieldJet, polarization: FieldJet) -> DensityJet2 {
    DensityJet2 {
        rho: [
            0.5 * (charge.value + polarization.value),
            0.5 * (charge.value - polarization.value),
        ],
        gradient: [
            std::array::from_fn(|axis| 0.5 * (charge.gradient[axis] + polarization.gradient[axis])),
            std::array::from_fn(|axis| 0.5 * (charge.gradient[axis] - polarization.gradient[axis])),
        ],
        hessian: [
            std::array::from_fn(|axis| 0.5 * (charge.hessian[axis] + polarization.hessian[axis])),
            std::array::from_fn(|axis| 0.5 * (charge.hessian[axis] - polarization.hessian[axis])),
        ],
    }
}

fn transform_muffin_tin(
    functional: XcFunctional,
    route: NoncollinearXcRoute,
    fields: [&MuffinTinField; 4],
    angular: &AngularGrid,
    output_l_max: u32,
) -> Result<MuffinTinTransform, RegionalXcError> {
    let charge = fields[0];
    let mesh = charge.mesh();
    let channel_count = lm_count(output_l_max);
    let mut projected: [Vec<Vec<Complex64>>; 4] =
        std::array::from_fn(|_| vec![vec![Complex64::new(0.0, 0.0); mesh.len()]; channel_count]);
    let mut radial_energy = vec![0.0; mesh.len()];
    let mut radial_density_potential = vec![0.0; mesh.len()];
    let convention = charge.field().convention();
    let zero_fields = fields.map(|field| {
        field
            .field()
            .channels()
            .all(|(_, values)| values.iter().all(|&value| value == Complex64::default()))
    });
    let angular_harmonics = angular
        .points()
        .iter()
        .map(|point| match convention {
            HarmonicConvention::Complex => {
                complex_spherical_harmonics(output_l_max, point.direction)
            }
            HarmonicConvention::Real => real_spherical_harmonics(output_l_max, point.direction)
                .into_iter()
                .map(|value| Complex64::new(value, 0.0))
                .collect(),
        })
        .collect::<Vec<_>>();

    for radial_index in 0..mesh.len() {
        let radius = mesh.radius(radial_index).unwrap().get();
        let derivative_step = derivative_step(mesh.radii(), radial_index);
        for (point, harmonics) in angular.points().iter().zip(&angular_harmonics) {
            let position = point.direction.map(|component| component * radius);
            let [charge, mx, my, mz] = [0, 1, 2, 3].map(|component| {
                if zero_fields[component] {
                    Ok(FieldJet::value(0.0))
                } else if functional == XcFunctional::LdaPw92 {
                    evaluate_muffin_tin_shell(fields[component], radial_index, harmonics)
                        .map(FieldJet::value)
                } else {
                    muffin_tin_field_jet(fields[component], radial_index, position, derivative_step)
                }
            });
            let xc = evaluate_noncollinear_xc_point(functional, route, charge?, [mx?, my?, mz?])?;
            radial_energy[radial_index] += point.weight * xc.energy_density;
            radial_density_potential[radial_index] += point.weight * xc.density_potential;
            for (target, value) in projected.iter_mut().zip(xc.potential) {
                if value == 0.0 {
                    continue;
                }
                project_angular_value(
                    convention,
                    harmonics,
                    point.weight,
                    value,
                    radial_index,
                    target,
                );
            }
        }
        let radius_squared = radius * radius;
        radial_energy[radial_index] *= radius_squared;
        radial_density_potential[radial_index] *= radius_squared;
    }

    let [scalar, bx, by, bz] = projected;
    Ok(MuffinTinTransform {
        fields: [
            MuffinTinField::new(
                mesh.clone(),
                finish_sphere_projection(convention, output_l_max, scalar)?,
            )?,
            MuffinTinField::new(
                mesh.clone(),
                finish_sphere_projection(convention, output_l_max, bx)?,
            )?,
            MuffinTinField::new(
                mesh.clone(),
                finish_sphere_projection(convention, output_l_max, by)?,
            )?,
            MuffinTinField::new(
                mesh.clone(),
                finish_sphere_projection(convention, output_l_max, bz)?,
            )?,
        ],
        exchange_correlation_energy: mesh.integrate(&radial_energy)?,
        density_potential_integral: mesh.integrate(&radial_density_potential)?,
    })
}

fn interstitial_pauli_jets(
    density: &RegionalDensity,
    position: [Bohr; 3],
) -> Result<(FieldJet, [FieldJet; 3]), RegionalXcError> {
    Ok((
        interstitial_field_jet(density.charge().interstitial(), position)?,
        [
            interstitial_field_jet(density.magnetization()[0].interstitial(), position)?,
            interstitial_field_jet(density.magnetization()[1].interstitial(), position)?,
            interstitial_field_jet(density.magnetization()[2].interstitial(), position)?,
        ],
    ))
}

pub(crate) fn evaluate_interstitial_noncollinear_xc_potential(
    functional: XcFunctional,
    route: NoncollinearXcRoute,
    density: &RegionalDensity,
    position: [Bohr; 3],
) -> Result<(Hartree, [Hartree; 3]), RegionalXcError> {
    let (charge, magnetization) = interstitial_pauli_jets(density, position)?;
    let point = evaluate_noncollinear_xc_point(functional, route, charge, magnetization)?;
    Ok((
        Hartree(point.potential[0]),
        point.potential[1..]
            .iter()
            .copied()
            .map(Hartree)
            .collect::<Vec<_>>()
            .try_into()
            .expect("three magnetic components remain three components"),
    ))
}

fn interstitial_field_jet(
    field: &InterstitialField,
    position: [Bohr; 3],
) -> Result<FieldJet, RegionalXcError> {
    let mut value = Complex64::new(0.0, 0.0);
    let mut gradient = [Complex64::new(0.0, 0.0); 3];
    let mut hessian = [Complex64::new(0.0, 0.0); 6];
    let mut scale = 0.0;
    for (vector, &coefficient) in field.field().iter() {
        if coefficient == Complex64::default() {
            continue;
        }
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
    Ok(FieldJet {
        value: checked_real(value, scale, "interstitial field")?,
        gradient: checked_real_array(gradient, scale, "interstitial gradient")?,
        hessian: checked_real_array(hessian, scale, "interstitial Hessian")?,
    })
}

fn interstitial_field_value(
    field: &InterstitialField,
    position: [Bohr; 3],
) -> Result<f64, RegionalXcError> {
    let mut value = Complex64::new(0.0, 0.0);
    let mut scale = 0.0;
    for (vector, &coefficient) in field.field().iter() {
        if coefficient == Complex64::default() {
            continue;
        }
        let phase = Complex64::from_polar(1.0, dot_g_r(vector.cartesian, position));
        let term = coefficient * phase;
        value += term;
        scale += term.norm();
    }
    checked_real(value, scale, "interstitial field")
}

fn muffin_tin_field_jet(
    field: &MuffinTinField,
    radial_index: usize,
    position: [f64; 3],
    step: f64,
) -> Result<FieldJet, RegionalXcError> {
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
    Ok(FieldJet {
        value: center,
        gradient,
        hessian,
    })
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

fn evaluate_muffin_tin_shell(
    field: &MuffinTinField,
    radial_index: usize,
    harmonics: &[Complex64],
) -> Result<f64, RegionalXcError> {
    let mut value = Complex64::new(0.0, 0.0);
    for (channel, samples) in field.field().channels() {
        value += samples[radial_index] * harmonics[channel.index()];
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
    harmonics: &[Complex64],
    weight: f64,
    value: f64,
    radial_index: usize,
    projected: &mut [Vec<Complex64>],
) {
    match convention {
        HarmonicConvention::Complex => {
            for (target, &harmonic) in projected.iter_mut().zip(harmonics) {
                target[radial_index] += weight * value * harmonic.conj();
            }
        }
        HarmonicConvention::Real => {
            for (target, harmonic) in projected.iter_mut().zip(harmonics) {
                target[radial_index] += weight * value * harmonic.re;
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

/// XC field controls derived from a density: interstitial divisions covering
/// the stored reciprocal support, an angular rule for the requested output
/// `l_max`, and the caller's noncollinear route.
pub fn xc_spec_for_density(
    density: &crate::RegionalDensity,
    output_l_max: u32,
    noncollinear_route: NoncollinearXcRoute,
) -> XcFieldSpec {
    let layout = density.charge().interstitial().layout();
    let divisions = std::array::from_fn(|axis| {
        let maximum = layout
            .vectors()
            .iter()
            .map(|vector| vector.index[axis].unsigned_abs() as usize)
            .max()
            .unwrap_or(0);
        (2 * maximum + 1).max(4)
    });
    let angular_point_count = ((output_l_max as usize + 1).pow(2) * 2).max(50);
    XcFieldSpec {
        interstitial_divisions: divisions,
        angular_point_count,
        output_l_max,
        noncollinear_route,
    }
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
        let geometry = InterstitialGeometry::new(VolumeBohr3(TAU.powi(3)), Vec::new()).unwrap();
        let up: BTreeMap<_, _> = up.into_iter().collect();
        let down: BTreeMap<_, _> = down.into_iter().collect();
        let combine = |sign: f64| -> Vec<_> {
            indices
                .iter()
                .map(|&index| (index, up[&index] + sign * down[&index]))
                .collect()
        };
        let charge = RegionalScalarField::new(
            geometry.clone(),
            Vec::new(),
            interstitial_field(layout.clone(), combine(1.0)),
        )
        .unwrap();
        let mz = RegionalScalarField::new(
            geometry,
            Vec::new(),
            interstitial_field(layout, combine(-1.0)),
        )
        .unwrap();
        let zero = charge.zero_like();
        RegionalDensity::new(charge, [zero.clone(), zero, mz]).unwrap()
    }

    fn uniform_density(up: f64, down: f64) -> RegionalDensity {
        interstitial_density(
            &[[0; 3]],
            [([0; 3], Complex64::new(up, 0.0))],
            [([0; 3], Complex64::new(down, 0.0))],
        )
    }

    fn uniform_pauli_density(charge: f64, magnetization: [f64; 3]) -> RegionalDensity {
        let layout = layout(&[[0; 3]]);
        let geometry = InterstitialGeometry::new(VolumeBohr3(TAU.powi(3)), Vec::new()).unwrap();
        let scalar = |value: f64| {
            RegionalScalarField::new(
                geometry.clone(),
                Vec::new(),
                interstitial_field(layout.clone(), [([0; 3], Complex64::new(value, 0.0))]),
            )
            .unwrap()
        };
        RegionalDensity::new(scalar(charge), magnetization.map(scalar)).unwrap()
    }

    fn line_pauli_density(coefficients: [[Complex64; 3]; 4]) -> RegionalDensity {
        let indices = [[-1, 0, 0], [0; 3], [1, 0, 0]];
        let layout = layout(&indices);
        let geometry = InterstitialGeometry::new(VolumeBohr3(TAU.powi(3)), Vec::new()).unwrap();
        let fields = coefficients.map(|values| {
            RegionalScalarField::new(
                geometry.clone(),
                Vec::new(),
                interstitial_field(layout.clone(), indices.into_iter().zip(values)),
            )
            .unwrap()
        });
        let [charge, mx, my, mz] = fields;
        RegionalDensity::new(charge, [mx, my, mz]).unwrap()
    }

    fn spec(divisions_x: usize) -> XcFieldSpec {
        XcFieldSpec {
            interstitial_divisions: [divisions_x, 3, 3],
            angular_point_count: 50,
            output_l_max: 0,
            noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
        }
    }

    fn route_spec(divisions_x: usize, route: NoncollinearXcRoute) -> XcFieldSpec {
        XcFieldSpec {
            noncollinear_route: route,
            ..spec(divisions_x)
        }
    }

    fn spin_potential_coefficient(
        result: &RegionalXcResult,
        spin: usize,
        index: [i32; 3],
    ) -> Complex64 {
        let scalar = result
            .potential
            .scalar()
            .interstitial()
            .coefficient(index)
            .unwrap();
        let bz = result.potential.magnetic()[2]
            .interstitial()
            .coefficient(index)
            .unwrap();
        if spin == 0 { scalar + bz } else { scalar - bz }
    }

    fn pauli_potential_coefficient(result: &RegionalXcResult, index: [i32; 3]) -> [Complex64; 4] {
        [
            result.potential.scalar(),
            &result.potential.magnetic()[0],
            &result.potential.magnetic()[1],
            &result.potential.magnetic()[2],
        ]
        .map(|field| field.interstitial().coefficient(index).unwrap())
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
            assert!(
                (spin_potential_coefficient(&lda, spin, [0; 3]).re - point.potential[spin].get())
                    .abs()
                    < 2.0e-14
            );
            assert!(
                (spin_potential_coefficient(&pbe, spin, [0; 3]).re - point.potential[spin].get())
                    .abs()
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
    fn uniform_global_spin_rotation_covariance_and_collinear_reduction() {
        for route in [
            NoncollinearXcRoute::LocalSpinFrame,
            NoncollinearXcRoute::MagnetizationField,
        ] {
            let tilted = uniform_pauli_density(0.3, [0.03, 0.04, 0.0]);
            let collinear = uniform_pauli_density(0.3, [0.0, 0.0, 0.05]);
            let tilted =
                evaluate_regional_xc(XcFunctional::Pbe, &tilted, route_spec(4, route)).unwrap();
            let collinear =
                evaluate_regional_xc(XcFunctional::Pbe, &collinear, route_spec(4, route)).unwrap();
            let tilted_v = pauli_potential_coefficient(&tilted, [0; 3]).map(|value| value.re);
            let collinear_v = pauli_potential_coefficient(&collinear, [0; 3]).map(|value| value.re);
            assert!((tilted_v[0] - collinear_v[0]).abs() < 2.0e-14);
            assert!((tilted_v[1] - 0.6 * collinear_v[3]).abs() < 2.0e-14);
            assert!((tilted_v[2] - 0.8 * collinear_v[3]).abs() < 2.0e-14);
            assert!(tilted_v[3].abs() < 2.0e-14);
            assert!(
                (tilted.exchange_correlation_energy.get()
                    - collinear.exchange_correlation_energy.get())
                .abs()
                    < 2.0e-12
            );
        }
    }

    #[test]
    fn nonuniform_global_spin_rotation_is_covariant_for_both_routes() {
        let zero = [Complex64::new(0.0, 0.0); 3];
        let charge = [
            Complex64::new(0.02, 0.0),
            Complex64::new(0.4, 0.0),
            Complex64::new(0.02, 0.0),
        ];
        let magnetization = [
            Complex64::new(0.01, 0.0),
            Complex64::new(0.08, 0.0),
            Complex64::new(0.01, 0.0),
        ];
        for route in [
            NoncollinearXcRoute::LocalSpinFrame,
            NoncollinearXcRoute::MagnetizationField,
        ] {
            let along_x = line_pauli_density([charge, magnetization, zero, zero]);
            let along_z = line_pauli_density([charge, zero, zero, magnetization]);
            let along_x =
                evaluate_regional_xc(XcFunctional::Pbe, &along_x, route_spec(24, route)).unwrap();
            let along_z =
                evaluate_regional_xc(XcFunctional::Pbe, &along_z, route_spec(24, route)).unwrap();
            for index in [[-1, 0, 0], [0; 3], [1, 0, 0]] {
                let x = pauli_potential_coefficient(&along_x, index);
                let z = pauli_potential_coefficient(&along_z, index);
                assert!((x[0] - z[0]).norm() < 2.0e-13);
                assert!((x[1] - z[3]).norm() < 2.0e-13);
                assert!(x[2].norm() < 2.0e-13 && x[3].norm() < 2.0e-13);
                assert!(z[1].norm() < 2.0e-13 && z[2].norm() < 2.0e-13);
            }
        }
    }

    #[test]
    fn textured_pbe_distinguishes_the_two_derivative_routes() {
        let charge = FieldJet {
            value: 0.3,
            gradient: [0.0; 3],
            hessian: [0.0; 6],
        };
        let amplitude = 0.08;
        // At x=0 for m=a(cos x, sin x, 0): the projected Hessian is -a,
        // while the complete Hessian of |m| is exactly zero.
        let magnetization = [
            FieldJet {
                value: amplitude,
                gradient: [0.0; 3],
                hessian: [-amplitude, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            FieldJet {
                value: 0.0,
                gradient: [amplitude, 0.0, 0.0],
                hessian: [0.0; 6],
            },
            FieldJet {
                value: 0.0,
                gradient: [0.0; 3],
                hessian: [0.0; 6],
            },
        ];
        let local = evaluate_noncollinear_xc_point(
            XcFunctional::Pbe,
            NoncollinearXcRoute::LocalSpinFrame,
            charge,
            magnetization,
        )
        .unwrap();
        let field = evaluate_noncollinear_xc_point(
            XcFunctional::Pbe,
            NoncollinearXcRoute::MagnetizationField,
            charge,
            magnetization,
        )
        .unwrap();
        let expected_local = evaluate_xc_point(
            XcFunctional::Pbe,
            split_local_spin_jet(
                charge,
                FieldJet {
                    value: amplitude,
                    gradient: [0.0; 3],
                    hessian: [-amplitude, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
            ),
        )
        .unwrap();
        let expected_field = evaluate_xc_point(
            XcFunctional::Pbe,
            split_local_spin_jet(
                charge,
                FieldJet {
                    value: amplitude,
                    gradient: [0.0; 3],
                    hessian: [0.0; 6],
                },
            ),
        )
        .unwrap();
        assert!(
            (local.potential[0]
                - 0.5 * (expected_local.potential[0].get() + expected_local.potential[1].get()))
            .abs()
                < 2.0e-14
        );
        assert!(
            (field.potential[0]
                - 0.5 * (expected_field.potential[0].get() + expected_field.potential[1].get()))
            .abs()
                < 2.0e-14
        );
        assert!((local.potential[0] - field.potential[0]).abs() > 1.0e-8);

        let zero = [Complex64::new(0.0, 0.0); 3];
        let charge_coefficients = [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.3, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let mx = [
            Complex64::new(amplitude / 2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(amplitude / 2.0, 0.0),
        ];
        let my = [
            Complex64::new(0.0, amplitude / 2.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -amplitude / 2.0),
        ];
        let texture = line_pauli_density([charge_coefficients, mx, my, zero]);
        let local = evaluate_regional_xc(
            XcFunctional::Pbe,
            &texture,
            route_spec(32, NoncollinearXcRoute::LocalSpinFrame),
        )
        .unwrap();
        let field = evaluate_regional_xc(
            XcFunctional::Pbe,
            &texture,
            route_spec(32, NoncollinearXcRoute::MagnetizationField),
        )
        .unwrap();
        assert!(
            (pauli_potential_coefficient(&local, [0; 3])[0]
                - pauli_potential_coefficient(&field, [0; 3])[0])
                .norm()
                > 1.0e-8
        );

        let local_lda = evaluate_regional_xc(
            XcFunctional::LdaPw92,
            &texture,
            route_spec(32, NoncollinearXcRoute::LocalSpinFrame),
        )
        .unwrap();
        let field_lda = evaluate_regional_xc(
            XcFunctional::LdaPw92,
            &texture,
            route_spec(32, NoncollinearXcRoute::MagnetizationField),
        )
        .unwrap();
        for index in [[-1, 0, 0], [0; 3], [1, 0, 0]] {
            let left = pauli_potential_coefficient(&local_lda, index);
            let right = pauli_potential_coefficient(&field_lda, index);
            for (left, right) in left.into_iter().zip(right) {
                assert!((left - right).norm() < 2.0e-13);
            }
        }
    }

    #[test]
    fn magnetization_node_with_derivatives_is_spin_rotation_covariant() {
        let charge = FieldJet {
            value: 0.2,
            gradient: [0.01, 0.0, 0.0],
            hessian: [0.002, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let mx = FieldJet {
            value: 0.0,
            gradient: [0.03, -0.02, 0.01],
            hessian: [0.006, -0.004, 0.002, 0.003, -0.001, 0.005],
        };
        let my = FieldJet {
            value: 0.0,
            gradient: [-0.01, 0.04, 0.02],
            hessian: [-0.003, 0.007, -0.002, 0.001, 0.004, -0.005],
        };
        let mz = FieldJet {
            value: 0.0,
            gradient: [0.02, 0.01, -0.03],
            hessian: [0.005, 0.001, -0.006, -0.002, 0.003, 0.004],
        };
        // A 90-degree spin rotation around z: (mx,my,mz) -> (-my,mx,mz).
        let rotated_mx = FieldJet {
            value: -my.value,
            gradient: my.gradient.map(|value| -value),
            hessian: my.hessian.map(|value| -value),
        };
        let rotated_magnetization = [rotated_mx, mx, mz];
        for route in [
            NoncollinearXcRoute::LocalSpinFrame,
            NoncollinearXcRoute::MagnetizationField,
        ] {
            let point =
                evaluate_noncollinear_xc_point(XcFunctional::Pbe, route, charge, [mx, my, mz])
                    .unwrap();
            let rotated_point = evaluate_noncollinear_xc_point(
                XcFunctional::Pbe,
                route,
                charge,
                rotated_magnetization,
            )
            .unwrap();
            assert!(point.potential.into_iter().all(f64::is_finite));
            assert_eq!(point.energy_density, rotated_point.energy_density);
            assert_eq!(point.potential[0], rotated_point.potential[0]);
            assert_eq!(point.potential[1..], [0.0; 3]);
            assert_eq!(rotated_point.potential[1..], [0.0; 3]);
        }
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
        let (charge, magnetization) =
            interstitial_pauli_jets(&density, [Bohr(x), Bohr(0.2), Bohr(-0.4)]).unwrap();
        let jet = split_local_spin_jet(
            charge,
            project_magnetization_jet(magnetization, [0.0, 0.0, 1.0]),
        );
        assert!((jet.rho[0] - (0.5 + 0.1 * x.cos())).abs() < 2.0e-15);
        assert!((jet.gradient[0][0] + 0.1 * x.sin()).abs() < 2.0e-15);
        assert!((jet.hessian[0][0] + 0.1 * x.cos()).abs() < 2.0e-15);
        assert_eq!(jet.gradient[0][1], 0.0);
        assert_eq!(jet.hessian[0][3], 0.0);

        let coarse = evaluate_regional_xc(XcFunctional::LdaPw92, &density, spec(4)).unwrap();
        let fine = evaluate_regional_xc(XcFunctional::LdaPw92, &density, spec(8)).unwrap();
        let reference = evaluate_regional_xc(XcFunctional::LdaPw92, &density, spec(64)).unwrap();
        let coefficient =
            |result: &RegionalXcResult| spin_potential_coefficient(result, 0, [1, 0, 0]).re;
        let coarse_error = (coefficient(&coarse) - coefficient(&reference)).abs();
        let fine_error = (coefficient(&fine) - coefficient(&reference)).abs();
        assert!(fine_error < coarse_error);

        let pbe = evaluate_regional_xc(XcFunctional::Pbe, &density, spec(16)).unwrap();
        let lda = evaluate_regional_xc(XcFunctional::LdaPw92, &density, spec(16)).unwrap();
        assert!(
            (pbe.exchange_correlation_energy.get() - lda.exchange_correlation_energy.get()).abs()
                > 1.0e-8
        );
        let plus = spin_potential_coefficient(&pbe, 0, [1, 0, 0]);
        let minus = spin_potential_coefficient(&pbe, 0, [-1, 0, 0]);
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
        let jet = muffin_tin_field_jet(&field, radial_index, position, step).unwrap();
        assert!((jet.value - (base + quadratic * radius.powi(2))).abs() < 2.0e-13);
        for ((&gradient, &coordinate), &hessian) in
            jet.gradient.iter().zip(&position).zip(&jet.hessian[..3])
        {
            assert!((gradient - 2.0 * quadratic * coordinate).abs() < 2.0e-10);
            assert!((hessian - 2.0 * quadratic).abs() < 2.0e-8);
        }
        for &mixed in &jet.hessian[3..] {
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
            spin_potential_coefficient(&first, 0, [0; 3]),
            spin_potential_coefficient(&swapped, 1, [0; 3])
        );
        assert_eq!(
            spin_potential_coefficient(&first, 1, [0; 3]),
            spin_potential_coefficient(&swapped, 0, [0; 3])
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
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(TAU.powi(3)),
            vec![Sphere {
                center: [Bohr(0.0); 3],
                radius: Bohr(0.5),
            }],
        )
        .unwrap();
        let charge =
            RegionalScalarField::new(geometry.clone(), vec![make_field(0.3)], zero.clone())
                .unwrap();
        let mz = RegionalScalarField::new(geometry, vec![make_field(0.1)], zero).unwrap();
        let zero = charge.zero_like();
        let density = RegionalDensity::new(charge, [zero.clone(), zero, mz]).unwrap();
        let result = evaluate_regional_xc(
            XcFunctional::LdaPw92,
            &density,
            XcFieldSpec {
                interstitial_divisions: [8, 8, 8],
                angular_point_count: 100,
                output_l_max: 0,
                noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
            },
        )
        .unwrap();
        for field in [
            &result.potential.scalar().muffin_tins()[0],
            &result.potential.magnetic()[2].muffin_tins()[0],
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
                    noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
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
                    noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
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
