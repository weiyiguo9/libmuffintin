//! Four-component core density over the exact muffin-tin/interstitial partition.

use crate::density::{DensityError, electron_count, scalar_field_integral};
use crate::regional::{
    InterstitialField, MuffinTinField, RegionalDensity, RegionalError, RegionalScalarField,
};
use crate::scf::CoreContribution;
use muffintin_core::{
    ExponentialMesh, FourierFieldError, Hartree, HermitianFourierField, InterstitialGeometry,
    InverseBohr, Lm, StepFunctionError, spherical_bessel_j,
};
use muffintin_operators::Collinear;
use muffintin_radial::{CoreDiracSolution, CoreState};
use muffintin_sphere::{SphereField, SphereFieldError};
use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

const MATCH_TOLERANCE: f64 = 4096.0 * f64::EPSILON;
const ZERO_MODE_FRACTION_TOLERANCE: f64 = 1.0e-12;
const CHARGE_CLOSURE_TOLERANCE: f64 = 65536.0 * f64::EPSILON;

/// Explicit distribution of one core-shell occupation over collinear outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CoreSpinPartition {
    /// A closed shell contributes half of its total occupation to each channel.
    ClosedShellAverage,
    /// Caller-specified explicit spin occupations; their sum must equal the
    /// shell's total occupation.
    ExplicitCollinear { up: f64, down: f64 },
}

/// One extended four-component core shell entering a regional site density.
#[derive(Clone, Copy, Debug)]
pub struct RegionalCoreShellInput<'a> {
    pub mesh: &'a ExponentialMesh,
    pub solution: &'a CoreDiracSolution,
    pub occupation: f64,
    pub spin: CoreSpinPartition,
}

/// Boundary evidence for the smooth pseudocharge continuation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PseudochargeBoundaryDiagnostic {
    pub value: f64,
    pub derivative: f64,
    pub continued_value: f64,
    pub continued_derivative: f64,
}

/// Charge accounting for one physical core shell.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreShellDensityDiagnostic {
    pub state: CoreState,
    pub occupation: f64,
    pub muffin_tin_charge: f64,
    pub spill_charge: f64,
    pub smooth_charge: f64,
    pub pseudocharge_boundary: PseudochargeBoundaryDiagnostic,
}

/// Aggregate regional charge and Fourier evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreDensityDiagnostics {
    pub shells: Vec<CoreShellDensityDiagnostic>,
    pub requested_charge: f64,
    pub represented_charge: f64,
    pub muffin_tin_charge: f64,
    pub spill_charge: f64,
    pub finite_g_norm: f64,
    pub zero_mode_adjustment: PseudochargeZeroModeAdjustment,
}

/// Finite-layout pseudocharge closure applied only to the constant mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PseudochargeZeroModeAdjustment {
    pub interstitial_fraction: f64,
    pub response_volume: f64,
    pub requested_spin_charge: [f64; 2],
    pub uncorrected_spin_charge: [f64; 2],
    pub coefficient_correction: [f64; 2],
}

/// SCF core contribution plus explicit regional-partition diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct BuiltRegionalCoreContribution {
    pub contribution: CoreContribution,
    pub diagnostics: CoreDensityDiagnostics,
}

/// Build one site's true-MT plus smooth-tail regional core contribution.
///
/// The physical muffin-tin density always uses the true `P^2 + Q^2` samples.
/// For the plane-wave representation, the true tail is retained outside the
/// muffin-tin radius and continued inside by the SPEX smooth pseudocharge
/// matching its boundary value and derivative. The transform is
/// `4 pi / Omega integral r^2 rho_s(r) j0(Gr) dr exp(-i G.R)`.
pub fn build_regional_core_contribution(
    site_id: String,
    geometry: &InterstitialGeometry,
    site_index: usize,
    muffin_tin_mesh: &ExponentialMesh,
    shells: &[RegionalCoreShellInput<'_>],
    zero_like_template: &RegionalDensity,
) -> Result<BuiltRegionalCoreContribution, CoreDensityError> {
    if geometry != zero_like_template.geometry() {
        return Err(CoreDensityError::GeometryMismatch);
    }
    let sphere = geometry
        .spheres()
        .get(site_index)
        .ok_or(CoreDensityError::SiteIndex {
            site: site_index,
            site_count: geometry.spheres().len(),
        })?;
    if muffin_tin_mesh.last() != sphere.radius {
        return Err(CoreDensityError::MuffinTinRadius {
            expected: sphere.radius.get(),
            actual: muffin_tin_mesh.last().get(),
        });
    }
    let template_up =
        zero_like_template
            .muffin_tins()
            .up
            .get(site_index)
            .ok_or(CoreDensityError::SiteIndex {
                site: site_index,
                site_count: zero_like_template.muffin_tins().up.len(),
            })?;
    let template_down = &zero_like_template.muffin_tins().down[site_index];
    if template_up.mesh() != muffin_tin_mesh || template_down.mesh() != muffin_tin_mesh {
        return Err(CoreDensityError::TemplateMuffinTinMesh);
    }
    let layout = zero_like_template.interstitial().up.layout();
    if zero_like_template.interstitial().down.layout() != layout {
        return Err(CoreDensityError::Fourier(FourierFieldError::LayoutMismatch));
    }

    let mut mt_up = vec![0.0; muffin_tin_mesh.len()];
    let mut mt_down = mt_up.clone();
    let mut fourier_up = vec![Complex64::new(0.0, 0.0); layout.len()];
    let mut fourier_down = fourier_up.clone();
    let mut diagnostics = Vec::with_capacity(shells.len());
    let mut eigenvalue_sum = Hartree(0.0);
    let mut requested_spin_charge = [0.0; 2];

    for shell in shells {
        validate_shell(muffin_tin_mesh, shell)?;
        let [up_occupation, down_occupation] = spin_occupations(shell)?;
        requested_spin_charge[0] += up_occupation;
        requested_spin_charge[1] += down_occupation;
        let transform = smooth_shell_transform(
            geometry,
            site_index,
            muffin_tin_mesh,
            shell.mesh,
            shell.solution,
            layout,
        )?;
        let norm_tolerance = 1.0e-10 * shell.solution.norm_mt.abs().max(1.0);
        if (transform.muffin_tin_charge - shell.solution.norm_mt).abs() > norm_tolerance {
            return Err(CoreDensityError::MuffinTinNormMismatch {
                solution: shell.solution.norm_mt,
                integrated: transform.muffin_tin_charge,
            });
        }
        let probability = shell
            .solution
            .p
            .iter()
            .zip(&shell.solution.q)
            .take(muffin_tin_mesh.len())
            .map(|(&p, &q)| p * p + q * q);
        for (index, value) in probability.enumerate() {
            mt_up[index] += up_occupation * value;
            mt_down[index] += down_occupation * value;
        }
        for position in 0..layout.len() {
            fourier_up[position] += up_occupation * transform.coefficients[position];
            fourier_down[position] += down_occupation * transform.coefficients[position];
        }
        let muffin_tin_charge = shell.occupation * transform.muffin_tin_charge;
        diagnostics.push(CoreShellDensityDiagnostic {
            state: shell.solution.state,
            occupation: shell.occupation,
            muffin_tin_charge,
            spill_charge: shell.occupation * shell.solution.spill,
            smooth_charge: shell.occupation * transform.smooth_charge,
            pseudocharge_boundary: PseudochargeBoundaryDiagnostic {
                value: shell.occupation * transform.boundary.value,
                derivative: shell.occupation * transform.boundary.derivative,
                continued_value: shell.occupation * transform.boundary.continued_value,
                continued_derivative: shell.occupation * transform.boundary.continued_derivative,
            },
        });
        eigenvalue_sum += shell.solution.energy * shell.occupation;
    }

    enforce_fourier_reality(layout, &mut fourier_up)?;
    enforce_fourier_reality(layout, &mut fourier_down)?;
    let mut muffin_tins = zero_like_template.zero_like().muffin_tins().clone();
    muffin_tins.up[site_index] = replace_monopole(template_up, &mt_up)?;
    muffin_tins.down[site_index] = replace_monopole(template_down, &mt_down)?;
    let zero_mode_adjustment = close_pseudocharge_zero_mode(
        geometry,
        layout,
        &muffin_tins,
        requested_spin_charge,
        &mut fourier_up,
        &mut fourier_down,
    )?;
    let interstitial = Collinear::new(
        InterstitialField::from_fourier_field(HermitianFourierField::new(
            layout.clone(),
            fourier_up,
        )?),
        InterstitialField::from_fourier_field(HermitianFourierField::new(
            layout.clone(),
            fourier_down,
        )?),
    );
    let density = RegionalDensity::new(geometry.clone(), muffin_tins, interstitial)?;
    let represented_charge = electron_count(&density)?;
    let requested_charge: f64 = requested_spin_charge.into_iter().sum();
    let closure_tolerance = CHARGE_CLOSURE_TOLERANCE * requested_charge.abs().max(1.0);
    if (represented_charge - requested_charge).abs() > closure_tolerance {
        return Err(CoreDensityError::ChargeClosure {
            requested: requested_charge,
            represented: represented_charge,
            tolerance: closure_tolerance,
        });
    }
    let muffin_tin_charge = diagnostics
        .iter()
        .map(|diagnostic| diagnostic.muffin_tin_charge)
        .sum();
    let spill_charge = diagnostics
        .iter()
        .map(|diagnostic| diagnostic.spill_charge)
        .sum();
    let finite_g_norm = density
        .interstitial()
        .up
        .field()
        .iter()
        .zip(density.interstitial().down.field().iter())
        .filter(|((vector, _), _)| vector.index != [0; 3])
        .map(|((_, up), (_, down))| (*up + *down).norm_sqr())
        .sum::<f64>()
        .sqrt();
    Ok(BuiltRegionalCoreContribution {
        contribution: CoreContribution {
            site_id,
            density,
            eigenvalue_sum,
        },
        diagnostics: CoreDensityDiagnostics {
            shells: diagnostics,
            requested_charge,
            represented_charge,
            muffin_tin_charge,
            spill_charge,
            finite_g_norm,
            zero_mode_adjustment,
        },
    })
}

fn close_pseudocharge_zero_mode(
    geometry: &InterstitialGeometry,
    layout: &muffintin_core::FourierLayout,
    muffin_tins: &Collinear<Vec<MuffinTinField>>,
    requested_spin_charge: [f64; 2],
    fourier_up: &mut [Complex64],
    fourier_down: &mut [Complex64],
) -> Result<PseudochargeZeroModeAdjustment, CoreDensityError> {
    let zero = layout
        .index([0, 0, 0])
        .ok_or(CoreDensityError::MissingZeroMode)?;
    let theta_zero = geometry.coefficient([InverseBohr(0.0); 3])?;
    let reality_tolerance = MATCH_TOLERANCE * theta_zero.re.abs().max(1.0);
    if theta_zero.im.abs() > reality_tolerance {
        return Err(CoreDensityError::ComplexZeroModeResponse {
            real: theta_zero.re,
            imaginary: theta_zero.im,
        });
    }
    let interstitial_fraction = theta_zero.re;
    let response_volume = geometry.cell_volume().get() * interstitial_fraction;
    if !response_volume.is_finite() || interstitial_fraction <= ZERO_MODE_FRACTION_TOLERANCE {
        return Err(CoreDensityError::IllConditionedZeroMode {
            interstitial_fraction,
            response_volume,
        });
    }

    let preview = |muffin_tins: &[MuffinTinField], coefficients: &[Complex64]| {
        let interstitial = InterstitialField::from_fourier_field(HermitianFourierField::new(
            layout.clone(),
            coefficients.to_vec(),
        )?);
        let field = RegionalScalarField::new(geometry.clone(), muffin_tins.to_vec(), interstitial)?;
        scalar_field_integral(&field).map_err(CoreDensityError::from)
    };
    let uncorrected_spin_charge = [
        preview(&muffin_tins.up, fourier_up)?,
        preview(&muffin_tins.down, fourier_down)?,
    ];
    let coefficient_correction = [
        (requested_spin_charge[0] - uncorrected_spin_charge[0]) / response_volume,
        (requested_spin_charge[1] - uncorrected_spin_charge[1]) / response_volume,
    ];
    if coefficient_correction
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(CoreDensityError::NonFiniteZeroModeCorrection {
            correction: coefficient_correction,
        });
    }
    fourier_up[zero].re += coefficient_correction[0];
    fourier_down[zero].re += coefficient_correction[1];
    Ok(PseudochargeZeroModeAdjustment {
        interstitial_fraction,
        response_volume,
        requested_spin_charge,
        uncorrected_spin_charge,
        coefficient_correction,
    })
}

#[derive(Clone, Debug)]
struct SmoothShellTransform {
    coefficients: Vec<Complex64>,
    muffin_tin_charge: f64,
    smooth_charge: f64,
    boundary: PseudochargeBoundaryDiagnostic,
}

fn smooth_shell_transform(
    geometry: &InterstitialGeometry,
    site_index: usize,
    muffin_tin_mesh: &ExponentialMesh,
    extended_mesh: &ExponentialMesh,
    solution: &CoreDiracSolution,
    layout: &muffintin_core::FourierLayout,
) -> Result<SmoothShellTransform, CoreDensityError> {
    let mt_len = muffin_tin_mesh.len();
    let probability = solution
        .p
        .iter()
        .zip(&solution.q)
        .map(|(&p, &q)| p * p + q * q)
        .collect::<Vec<_>>();
    let actual_density = probability
        .iter()
        .zip(extended_mesh.radii())
        .map(|(&value, radius)| value / (4.0 * PI * radius.get().powi(2)))
        .collect::<Vec<_>>();
    let radius = muffin_tin_mesh.last().get();
    let value = actual_density[mt_len - 1];
    let next_radius = extended_mesh.radii()[mt_len].get();
    let mut derivative = (actual_density[mt_len] - value) / (next_radius - radius);
    let derivative_tolerance = MATCH_TOLERANCE * value.abs().max(1.0) / radius;
    if derivative > derivative_tolerance {
        return Err(CoreDensityError::PositiveBoundaryDerivative { derivative });
    }
    if derivative > 0.0 {
        derivative = 0.0;
    }
    let pseudo = Pseudocharge::new(radius, value, derivative)?;
    let mut smooth_density = actual_density;
    for (index, mesh_radius) in extended_mesh.radii()[..mt_len - 1].iter().enumerate() {
        smooth_density[index] = pseudo.value(mesh_radius.get())?;
    }
    let radial_charge = extended_mesh
        .radii()
        .iter()
        .zip(&smooth_density)
        .map(|(radius, &density)| 4.0 * PI * radius.get().powi(2) * density)
        .collect::<Vec<_>>();
    let smooth_charge = extended_mesh.integrate(&radial_charge)?;
    let muffin_tin_charge = muffin_tin_mesh.integrate(&probability[..mt_len])?;
    let center = geometry.spheres()[site_index].center;
    let inverse_volume = 1.0 / geometry.cell_volume().get();
    let mut coefficients = Vec::with_capacity(layout.len());
    for vector in layout.vectors() {
        let integrand = radial_charge
            .iter()
            .zip(extended_mesh.radii())
            .map(|(&charge, radius)| {
                charge * spherical_bessel_j(0, vector.norm.get() * radius.get())
            })
            .collect::<Vec<_>>();
        let radial = extended_mesh.integrate(&integrand)?;
        let phase = -vector
            .cartesian
            .iter()
            .zip(center)
            .map(|(g, r)| g.get() * r.get())
            .sum::<f64>();
        coefficients.push(Complex64::from_polar(radial * inverse_volume, phase));
    }
    let continued_value = pseudo.value(radius)?;
    let continued_derivative = pseudo.derivative(radius)?;
    Ok(SmoothShellTransform {
        coefficients,
        muffin_tin_charge,
        smooth_charge,
        boundary: PseudochargeBoundaryDiagnostic {
            value,
            derivative,
            continued_value,
            continued_derivative,
        },
    })
}

#[derive(Clone, Copy, Debug)]
struct Pseudocharge {
    radius: f64,
    value_at_radius: f64,
    derivative_at_radius: f64,
}

impl Pseudocharge {
    fn new(
        radius: f64,
        value_at_radius: f64,
        derivative_at_radius: f64,
    ) -> Result<Self, CoreDensityError> {
        if !value_at_radius.is_finite() || value_at_radius < 0.0 {
            return Err(CoreDensityError::InvalidBoundaryValue(value_at_radius));
        }
        if !derivative_at_radius.is_finite() {
            return Err(CoreDensityError::InvalidBoundaryDerivative(
                derivative_at_radius,
            ));
        }
        if value_at_radius == 0.0 && derivative_at_radius != 0.0 {
            return Err(CoreDensityError::ZeroBoundaryWithDerivative(
                derivative_at_radius,
            ));
        }
        Ok(Self {
            radius,
            value_at_radius,
            derivative_at_radius,
        })
    }

    fn value(self, radius: f64) -> Result<f64, CoreDensityError> {
        if self.value_at_radius == 0.0 || self.derivative_at_radius == 0.0 {
            return Ok(self.value_at_radius);
        }
        let exponent = self.derivative_at_radius / (2.0 * self.value_at_radius * self.radius)
            * (radius * radius - self.radius * self.radius);
        let value = self.value_at_radius * exponent.exp();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(CoreDensityError::PseudochargeOverflow { radius })
        }
    }

    fn derivative(self, radius: f64) -> Result<f64, CoreDensityError> {
        if self.value_at_radius == 0.0 || self.derivative_at_radius == 0.0 {
            return Ok(0.0);
        }
        Ok(self.value(radius)? * self.derivative_at_radius * radius
            / (self.value_at_radius * self.radius))
    }
}

fn validate_shell(
    muffin_tin_mesh: &ExponentialMesh,
    shell: &RegionalCoreShellInput<'_>,
) -> Result<(), CoreDensityError> {
    let capacity = f64::from(shell.solution.state.kappa.degeneracy());
    if !shell.occupation.is_finite() || shell.occupation < 0.0 || shell.occupation > capacity {
        return Err(CoreDensityError::InvalidOccupation {
            occupation: shell.occupation,
            capacity,
        });
    }
    if shell.solution.p.len() != shell.mesh.len() || shell.solution.q.len() != shell.mesh.len() {
        return Err(CoreDensityError::SolutionMeshLength {
            mesh: shell.mesh.len(),
            p: shell.solution.p.len(),
            q: shell.solution.q.len(),
        });
    }
    if shell.mesh.len() <= muffin_tin_mesh.len() {
        return Err(CoreDensityError::MissingTailPoint {
            muffin_tin: muffin_tin_mesh.len(),
            extended: shell.mesh.len(),
        });
    }
    if shell.mesh.radii()[..muffin_tin_mesh.len()] != *muffin_tin_mesh.radii() {
        return Err(CoreDensityError::MeshPrefixMismatch);
    }
    Ok(())
}

fn spin_occupations(shell: &RegionalCoreShellInput<'_>) -> Result<[f64; 2], CoreDensityError> {
    let occupations = match shell.spin {
        CoreSpinPartition::ClosedShellAverage => [0.5 * shell.occupation; 2],
        CoreSpinPartition::ExplicitCollinear { up, down } => {
            if !up.is_finite() || !down.is_finite() || up < 0.0 || down < 0.0 {
                return Err(CoreDensityError::InvalidSpinOccupations { up, down });
            }
            [up, down]
        }
    };
    let sum = occupations[0] + occupations[1];
    let tolerance = MATCH_TOLERANCE * shell.occupation.max(1.0);
    if (sum - shell.occupation).abs() > tolerance {
        return Err(CoreDensityError::SpinOccupationSum {
            occupation: shell.occupation,
            up: occupations[0],
            down: occupations[1],
        });
    }
    Ok([occupations[0], shell.occupation - occupations[0]])
}

fn replace_monopole(
    template: &MuffinTinField,
    reduced_probability: &[f64],
) -> Result<MuffinTinField, CoreDensityError> {
    let monopole = Lm::new(0, 0).expect("monopole is valid");
    let mut found = false;
    let channels = template
        .field()
        .channels()
        .map(|(channel, values)| {
            let mut values = vec![Complex64::new(0.0, 0.0); values.len()];
            if channel == monopole {
                found = true;
                for (index, radius) in template.mesh().radii().iter().enumerate() {
                    values[index].re =
                        reduced_probability[index] / ((4.0 * PI).sqrt() * radius.get().powi(2));
                }
            }
            ((channel.l, channel.m), values)
        })
        .collect::<Vec<_>>();
    if !found {
        return Err(CoreDensityError::MissingTemplateMonopole);
    }
    Ok(MuffinTinField::new(
        template.mesh().clone(),
        SphereField::new(template.field().convention(), channels)?,
    )?)
}

fn enforce_fourier_reality(
    layout: &muffintin_core::FourierLayout,
    coefficients: &mut [Complex64],
) -> Result<(), CoreDensityError> {
    for vector in layout.vectors() {
        let position = layout
            .index(vector.index)
            .expect("layout contains its stored vector");
        let opposite_index = [
            vector.index[0].checked_neg(),
            vector.index[1].checked_neg(),
            vector.index[2].checked_neg(),
        ];
        let [Some(g0), Some(g1), Some(g2)] = opposite_index else {
            return Err(CoreDensityError::MissingOpposite(vector.index));
        };
        let opposite = layout
            .index([g0, g1, g2])
            .ok_or(CoreDensityError::MissingOpposite(vector.index))?;
        if position == opposite {
            coefficients[position].im = 0.0;
        } else if position < opposite {
            let average = 0.5 * (coefficients[position] + coefficients[opposite].conj());
            coefficients[position] = average;
            coefficients[opposite] = average.conj();
        }
    }
    Ok(())
}

/// Invalid regional core-density construction.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum CoreDensityError {
    #[error("input geometry differs from the regional template")]
    GeometryMismatch,
    #[error("site {site} is outside geometry with {site_count} sites")]
    SiteIndex { site: usize, site_count: usize },
    #[error("muffin-tin mesh radius {actual} does not match geometry radius {expected}")]
    MuffinTinRadius { expected: f64, actual: f64 },
    #[error("regional template muffin-tin mesh differs from the requested mesh")]
    TemplateMuffinTinMesh,
    #[error("core occupation {occupation} is outside [0,{capacity}]")]
    InvalidOccupation { occupation: f64, capacity: f64 },
    #[error("explicit core spin occupations are invalid: up={up}, down={down}")]
    InvalidSpinOccupations { up: f64, down: f64 },
    #[error("core spin occupations up={up}, down={down} do not sum to {occupation}")]
    SpinOccupationSum { occupation: f64, up: f64, down: f64 },
    #[error("extended core mesh has {mesh} points but solution arrays have P={p}, Q={q}")]
    SolutionMeshLength { mesh: usize, p: usize, q: usize },
    #[error("extended core mesh has {extended} points but MT prefix has {muffin_tin}")]
    MissingTailPoint { muffin_tin: usize, extended: usize },
    #[error("muffin-tin mesh is not an exact prefix of the extended core mesh")]
    MeshPrefixMismatch,
    #[error(
        "core solution reports MT norm {solution}, but the requested MT prefix integrates to {integrated}"
    )]
    MuffinTinNormMismatch { solution: f64, integrated: f64 },
    #[error("core density derivative at the muffin-tin boundary is positive: {derivative}")]
    PositiveBoundaryDerivative { derivative: f64 },
    #[error("core density boundary value is invalid: {0}")]
    InvalidBoundaryValue(f64),
    #[error("core density boundary derivative is invalid: {0}")]
    InvalidBoundaryDerivative(f64),
    #[error("zero boundary density has nonzero derivative {0}")]
    ZeroBoundaryWithDerivative(f64),
    #[error("pseudocharge continuation overflowed at r={radius}")]
    PseudochargeOverflow { radius: f64 },
    #[error("regional template has no normalized Y00 channel")]
    MissingTemplateMonopole,
    #[error("Fourier layout has no opposite vector for {0:?}")]
    MissingOpposite([i32; 3]),
    #[error("core-density Fourier layout has no G=0 mode")]
    MissingZeroMode,
    #[error("core-density zero-mode response is complex: {real} + i {imaginary}")]
    ComplexZeroModeResponse { real: f64, imaginary: f64 },
    #[error(
        "core-density G=0 response is ill-conditioned: interstitial fraction {interstitial_fraction}, response volume {response_volume}"
    )]
    IllConditionedZeroMode {
        interstitial_fraction: f64,
        response_volume: f64,
    },
    #[error("core-density G=0 coefficient correction is non-finite: {correction:?}")]
    NonFiniteZeroModeCorrection { correction: [f64; 2] },
    #[error(
        "closed core density represents {represented} electrons, requested {requested} within {tolerance}"
    )]
    ChargeClosure {
        requested: f64,
        represented: f64,
        tolerance: f64,
    },
    #[error(transparent)]
    Mesh(#[from] muffintin_core::MeshError),
    #[error(transparent)]
    Sphere(#[from] SphereFieldError),
    #[error(transparent)]
    Regional(#[from] RegionalError),
    #[error(transparent)]
    Fourier(#[from] FourierFieldError),
    #[error(transparent)]
    Density(#[from] DensityError),
    #[error(transparent)]
    StepFunction(#[from] StepFunctionError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::{
        Bohr, DiracAngularContract, FourierLayout, InverseBohr, Kappa, ReciprocalLattice, Sphere,
        VolumeBohr3,
    };
    use muffintin_radial::{CoreState, RelativisticRole};
    use muffintin_sphere::HarmonicConvention;

    const CELL_LENGTH: f64 = 8.0;

    fn meshes() -> (ExponentialMesh, ExponentialMesh) {
        let first = Bohr(1.0e-2);
        let increment = 0.02;
        (
            ExponentialMesh::new(first, increment, 231).unwrap(),
            ExponentialMesh::new(first, increment, 401).unwrap(),
        )
    }

    fn core_solution(
        muffin_tin_mesh: &ExponentialMesh,
        extended_mesh: &ExponentialMesh,
        pure_q: bool,
        compact: bool,
    ) -> CoreDiracSolution {
        let radius = muffin_tin_mesh.last().get();
        let mut p = Vec::with_capacity(extended_mesh.len());
        let mut q = Vec::with_capacity(extended_mesh.len());
        for r in extended_mesh.radii() {
            let radial = if compact {
                if r.get() < radius {
                    r.get() * (1.0 - r.get() / radius)
                } else {
                    0.0
                }
            } else {
                r.get() * (-1.3 * r.get()).exp()
            };
            if pure_q {
                p.push(0.0);
                q.push(radial);
            } else {
                p.push(radial);
                q.push(if compact { 0.0 } else { 0.25 * radial });
            }
        }
        let norm_values = p
            .iter()
            .zip(&q)
            .map(|(&p, &q)| p * p + q * q)
            .collect::<Vec<_>>();
        let norm = extended_mesh.integrate(&norm_values).unwrap();
        let scale = norm.sqrt().recip();
        p.iter_mut().for_each(|value| *value *= scale);
        q.iter_mut().for_each(|value| *value *= scale);
        let normalized = p
            .iter()
            .zip(&q)
            .map(|(&p, &q)| p * p + q * q)
            .collect::<Vec<_>>();
        let norm_total = extended_mesh.integrate(&normalized).unwrap();
        let norm_mt = muffin_tin_mesh
            .integrate(&normalized[..muffin_tin_mesh.len()])
            .unwrap();
        let spill = (norm_total - norm_mt).max(0.0);
        let kappa = Kappa::new(-1).unwrap();
        CoreDiracSolution {
            role: RelativisticRole::Core,
            state: CoreState::new(1, kappa).unwrap(),
            angular: DiracAngularContract::from(kappa),
            energy: Hartree(-0.5),
            p,
            q,
            norm_total,
            norm_mt,
            norm_outside: spill,
            spill,
            nodes: 0,
            match_radius: muffin_tin_mesh.last(),
            matching_residual: 0.0,
        }
    }

    fn reciprocal() -> ReciprocalLattice {
        ReciprocalLattice::from_direct([
            [Bohr(CELL_LENGTH), Bohr(0.0), Bohr(0.0)],
            [Bohr(0.0), Bohr(CELL_LENGTH), Bohr(0.0)],
            [Bohr(0.0), Bohr(0.0), Bohr(CELL_LENGTH)],
        ])
        .unwrap()
    }

    fn template(
        center: [Bohr; 3],
        muffin_tin_mesh: &ExponentialMesh,
        cutoff: f64,
    ) -> RegionalDensity {
        let reciprocal = reciprocal();
        let vectors = reciprocal.enumerate(InverseBohr(cutoff)).unwrap();
        let layout = FourierLayout::new(reciprocal, vectors).unwrap();
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(CELL_LENGTH.powi(3)),
            vec![Sphere {
                center,
                radius: muffin_tin_mesh.last(),
            }],
        )
        .unwrap();
        let muffin_tin = MuffinTinField::new(
            muffin_tin_mesh.clone(),
            SphereField::new(
                HarmonicConvention::Complex,
                [(
                    (0, 0),
                    vec![Complex64::new(0.0, 0.0); muffin_tin_mesh.len()],
                )],
            )
            .unwrap(),
        )
        .unwrap();
        let interstitial = InterstitialField::from_fourier_field(
            HermitianFourierField::new(
                layout.clone(),
                vec![Complex64::new(0.0, 0.0); layout.len()],
            )
            .unwrap(),
        );
        RegionalDensity::new(
            geometry,
            Collinear::new(vec![muffin_tin.clone()], vec![muffin_tin]),
            Collinear::new(interstitial.clone(), interstitial),
        )
        .unwrap()
    }

    fn build(
        template: &RegionalDensity,
        muffin_tin_mesh: &ExponentialMesh,
        extended_mesh: &ExponentialMesh,
        solution: &CoreDiracSolution,
        occupation: f64,
    ) -> BuiltRegionalCoreContribution {
        build_regional_core_contribution(
            "X".to_owned(),
            template.geometry(),
            0,
            muffin_tin_mesh,
            &[RegionalCoreShellInput {
                mesh: extended_mesh,
                solution,
                occupation,
                spin: CoreSpinPartition::ClosedShellAverage,
            }],
            template,
        )
        .unwrap()
    }

    #[test]
    fn pure_q_shell_and_closed_shell_spin_partition_conserve_total_occupation() {
        let (mt_mesh, extended_mesh) = meshes();
        let solution = core_solution(&mt_mesh, &extended_mesh, true, false);
        let template = template([Bohr(0.0); 3], &mt_mesh, 4.0);
        let result = build(&template, &mt_mesh, &extended_mesh, &solution, 2.0);
        let up = result.contribution.density.muffin_tins().up[0]
            .field()
            .channel(0, 0)
            .unwrap();
        assert!(up.iter().any(|value| value.re > 0.0));
        assert_eq!(
            result.contribution.density.muffin_tins().up[0],
            result.contribution.density.muffin_tins().down[0]
        );
        assert!((result.diagnostics.requested_charge - 2.0).abs() < 1.0e-14);
        assert!((result.diagnostics.represented_charge - 2.0).abs() < 2.0e-12);
        assert_eq!(
            result
                .diagnostics
                .zero_mode_adjustment
                .requested_spin_charge,
            [1.0, 1.0]
        );
        assert!((result.contribution.eigenvalue_sum.get() + 1.0).abs() < 1.0e-14);

        let polarized = build_regional_core_contribution(
            "X".to_owned(),
            template.geometry(),
            0,
            &mt_mesh,
            &[RegionalCoreShellInput {
                mesh: &extended_mesh,
                solution: &solution,
                occupation: 2.0,
                spin: CoreSpinPartition::ExplicitCollinear { up: 2.0, down: 0.0 },
            }],
            &template,
        )
        .unwrap();
        assert!(
            polarized.contribution.density.muffin_tins().down[0]
                .field()
                .channel(0, 0)
                .unwrap()
                .iter()
                .all(|value| *value == Complex64::new(0.0, 0.0))
        );
        assert_eq!(
            polarized
                .diagnostics
                .zero_mode_adjustment
                .requested_spin_charge,
            [2.0, 0.0]
        );
        assert_eq!(
            polarized
                .diagnostics
                .zero_mode_adjustment
                .coefficient_correction[1],
            0.0
        );
    }

    #[test]
    fn g0_only_pseudocharge_closes_exact_core_occupation() {
        let (mt_mesh, extended_mesh) = meshes();
        let solution = core_solution(&mt_mesh, &extended_mesh, false, false);
        let template = template([Bohr(0.0); 3], &mt_mesh, 0.0);
        let result = build(&template, &mt_mesh, &extended_mesh, &solution, 2.0);
        let adjustment = result.diagnostics.zero_mode_adjustment;
        let uncorrected = adjustment.uncorrected_spin_charge.into_iter().sum::<f64>();

        assert!((uncorrected - 2.0).abs() > 1.0e-3);
        assert!(adjustment.coefficient_correction[0].abs() > 1.0e-8);
        assert_eq!(
            adjustment.coefficient_correction[0],
            adjustment.coefficient_correction[1]
        );
        assert!((result.diagnostics.represented_charge - 2.0).abs() < 2.0e-12);
        assert!((electron_count(&result.contribution.density).unwrap() - 2.0).abs() < 2.0e-12);
        assert_eq!(result.diagnostics.finite_g_norm, 0.0);
    }

    #[test]
    fn core_tail_has_finite_g_and_charge_converges_with_fourier_cutoff() {
        let (mt_mesh, extended_mesh) = meshes();
        let solution = core_solution(&mt_mesh, &extended_mesh, false, false);
        assert!(solution.spill > 0.0);
        let low_template = template([Bohr(0.0); 3], &mt_mesh, 2.0);
        let high_template = template([Bohr(0.0); 3], &mt_mesh, 4.0);
        let low = build(&low_template, &mt_mesh, &extended_mesh, &solution, 2.0);
        let high = build(&high_template, &mt_mesh, &extended_mesh, &solution, 2.0);
        let uncorrected_error = |density: &BuiltRegionalCoreContribution| {
            (density
                .diagnostics
                .zero_mode_adjustment
                .uncorrected_spin_charge
                .into_iter()
                .sum::<f64>()
                - 2.0)
                .abs()
        };
        let low_error = uncorrected_error(&low);
        let high_error = uncorrected_error(&high);
        assert!(high_error < low_error);
        assert!((low.diagnostics.represented_charge - 2.0).abs() < 2.0e-12);
        assert!((high.diagnostics.represented_charge - 2.0).abs() < 2.0e-12);
        assert!(high.diagnostics.spill_charge > 0.0);
        assert!(high.diagnostics.finite_g_norm > 1.0e-8);
        let layout = high.contribution.density.interstitial().up.layout();
        let g0 = layout.index([0, 0, 0]).unwrap();
        assert!(
            high.contribution
                .density
                .interstitial()
                .up
                .field()
                .coefficients()[g0]
                .re
                < high.diagnostics.requested_charge
        );
        let transform = smooth_shell_transform(
            high_template.geometry(),
            0,
            &mt_mesh,
            &extended_mesh,
            &solution,
            layout,
        )
        .unwrap();
        for (position, vector) in layout.vectors().iter().enumerate() {
            if vector.index == [0; 3] {
                continue;
            }
            let actual = high
                .contribution
                .density
                .interstitial()
                .up
                .field()
                .coefficients()[position]
                + high
                    .contribution
                    .density
                    .interstitial()
                    .down
                    .field()
                    .coefficients()[position];
            assert!((actual - 2.0 * transform.coefficients[position]).norm() < 2.0e-14);
        }
    }

    #[test]
    fn translation_phase_and_hermiticity_are_exact() {
        let (mt_mesh, extended_mesh) = meshes();
        let solution = core_solution(&mt_mesh, &extended_mesh, false, false);
        let origin_template = template([Bohr(0.0); 3], &mt_mesh, 2.0);
        let shifted_center = [Bohr(0.37), Bohr(-0.21), Bohr(0.13)];
        let shifted_template = template(shifted_center, &mt_mesh, 2.0);
        let origin = build(&origin_template, &mt_mesh, &extended_mesh, &solution, 2.0);
        let shifted = build(&shifted_template, &mt_mesh, &extended_mesh, &solution, 2.0);
        let index = [1, 0, 0];
        let origin_value = origin
            .contribution
            .density
            .interstitial()
            .up
            .coefficient(index)
            .unwrap();
        let shifted_value = shifted
            .contribution
            .density
            .interstitial()
            .up
            .coefficient(index)
            .unwrap();
        let g = reciprocal().cartesian(index);
        let phase = -g
            .iter()
            .zip(shifted_center)
            .map(|(g, r)| g.get() * r.get())
            .sum::<f64>();
        assert!((shifted_value - origin_value * Complex64::from_polar(1.0, phase)).norm() < 1e-15);
        assert_eq!(
            shifted
                .contribution
                .density
                .interstitial()
                .up
                .coefficient([-1, 0, 0])
                .unwrap(),
            shifted_value.conj()
        );
    }

    #[test]
    fn pseudocharge_matches_value_and_slope_and_zero_spill_has_no_finite_g_tail() {
        let (mt_mesh, extended_mesh) = meshes();
        let tailed = core_solution(&mt_mesh, &extended_mesh, false, false);
        let template = template([Bohr(0.0); 3], &mt_mesh, 3.0);
        let result = build(&template, &mt_mesh, &extended_mesh, &tailed, 2.0);
        let boundary = result.diagnostics.shells[0].pseudocharge_boundary;
        assert!((boundary.value - boundary.continued_value).abs() < 2.0e-15);
        assert!((boundary.derivative - boundary.continued_derivative).abs() < 2.0e-15);

        let compact = core_solution(&mt_mesh, &extended_mesh, false, true);
        assert!(compact.spill < 2.0e-14);
        let compact_result = build(&template, &mt_mesh, &extended_mesh, &compact, 2.0);
        assert!(compact_result.diagnostics.finite_g_norm < 1.0e-14);
        let compact_interstitial = compact_result.contribution.density.interstitial();
        for (vector, coefficient) in compact_interstitial.up.field().iter() {
            if vector.index != [0; 3] {
                assert!(coefficient.norm() < 1.0e-14);
            }
        }
        assert_eq!(
            compact_interstitial.up.coefficient([0; 3]).unwrap().re,
            compact_result
                .diagnostics
                .zero_mode_adjustment
                .coefficient_correction[0]
        );
        assert!(
            (compact_result.diagnostics.represented_charge - 2.0).abs() < 2.0e-12,
            "represented compact charge = {}",
            compact_result.diagnostics.represented_charge
        );
    }
}
