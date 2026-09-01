//! Four-component core density over the exact muffin-tin/interstitial partition.

use crate::density::{DensityError, scalar_field_integral};
use crate::regional::{
    InterstitialField, MuffinTinField, RegionalDensity, RegionalError, RegionalScalarField,
};
use crate::scf::CoreContribution;
use muffintin_core::{
    ExponentialMesh, FourierFieldError, Hartree, HermitianFourierField, InterstitialGeometry,
    InverseBohr, Lm, StepFunctionError, spherical_bessel_j,
};
use muffintin_sphere::CoreState;
use muffintin_sphere::{SphereField, SphereFieldError};
use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

const MATCH_TOLERANCE: f64 = 4096.0 * f64::EPSILON;
const ZERO_MODE_FRACTION_TOLERANCE: f64 = 1.0e-12;
const CHARGE_CLOSURE_TOLERANCE: f64 = 65536.0 * f64::EPSILON;

/// Explicit distribution of one core-shell occupation into charge and $m_z$.
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
    pub state: CoreState,
    pub energy: Hartree,
    pub p: &'a [f64],
    pub q: &'a [f64],
    pub norm_mt: f64,
    pub spill: f64,
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
    pub requested_charge: f64,
    pub requested_magnetization_z: f64,
    pub uncorrected_charge: f64,
    pub uncorrected_magnetization_z: f64,
    /// Constant Fourier-coefficient correction for the charge component.
    pub charge_coefficient_correction: f64,
    /// Constant Fourier-coefficient correction for the $m_z$ component.
    pub magnetization_z_coefficient_correction: f64,
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
/// Closed shells produce zero magnetization; explicit collinear occupations
/// produce $m_z = \rho_\uparrow - \rho_\down$ with $m_x=m_y=0$.
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
    let template_charge = zero_like_template
        .charge()
        .muffin_tins()
        .get(site_index)
        .ok_or(CoreDensityError::SiteIndex {
            site: site_index,
            site_count: zero_like_template.charge().muffin_tins().len(),
        })?;
    if template_charge.mesh() != muffin_tin_mesh {
        return Err(CoreDensityError::TemplateMuffinTinMesh);
    }
    let layout = zero_like_template.charge().interstitial().layout();

    let mut mt_charge = vec![0.0; muffin_tin_mesh.len()];
    let mut mt_magnetization_z = mt_charge.clone();
    let mut fourier_charge = vec![Complex64::new(0.0, 0.0); layout.len()];
    let mut fourier_magnetization_z = fourier_charge.clone();
    let mut diagnostics = Vec::with_capacity(shells.len());
    let mut eigenvalue_sum = Hartree(0.0);
    let mut requested_charge = 0.0;
    let mut requested_magnetization_z = 0.0;

    for shell in shells {
        validate_shell(muffin_tin_mesh, shell)?;
        let [up_occupation, down_occupation] = spin_occupations(shell)?;
        let magnetization_z_occupation = up_occupation - down_occupation;
        requested_charge += shell.occupation;
        requested_magnetization_z += magnetization_z_occupation;
        let transform = smooth_shell_transform(
            geometry,
            site_index,
            muffin_tin_mesh,
            shell.mesh,
            shell,
            layout,
        )?;
        let norm_tolerance = 1.0e-10 * shell.norm_mt.abs().max(1.0);
        if (transform.muffin_tin_charge - shell.norm_mt).abs() > norm_tolerance {
            return Err(CoreDensityError::MuffinTinNormMismatch {
                solution: shell.norm_mt,
                integrated: transform.muffin_tin_charge,
            });
        }
        let probability = shell
            .p
            .iter()
            .zip(shell.q)
            .take(muffin_tin_mesh.len())
            .map(|(&p, &q)| p * p + q * q);
        for (index, value) in probability.enumerate() {
            mt_charge[index] += shell.occupation * value;
            mt_magnetization_z[index] += magnetization_z_occupation * value;
        }
        for position in 0..layout.len() {
            fourier_charge[position] += shell.occupation * transform.coefficients[position];
            fourier_magnetization_z[position] +=
                magnetization_z_occupation * transform.coefficients[position];
        }
        let muffin_tin_charge = shell.occupation * transform.muffin_tin_charge;
        diagnostics.push(CoreShellDensityDiagnostic {
            state: shell.state,
            occupation: shell.occupation,
            muffin_tin_charge,
            spill_charge: shell.occupation * shell.spill,
            smooth_charge: shell.occupation * transform.smooth_charge,
            pseudocharge_boundary: PseudochargeBoundaryDiagnostic {
                value: shell.occupation * transform.boundary.value,
                derivative: shell.occupation * transform.boundary.derivative,
                continued_value: shell.occupation * transform.boundary.continued_value,
                continued_derivative: shell.occupation * transform.boundary.continued_derivative,
            },
        });
        eigenvalue_sum += shell.energy * shell.occupation;
    }

    enforce_fourier_reality(layout, &mut fourier_charge)?;
    enforce_fourier_reality(layout, &mut fourier_magnetization_z)?;
    let mut charge_muffin_tins = zero_like_template
        .charge()
        .zero_like()
        .muffin_tins()
        .to_vec();
    charge_muffin_tins[site_index] = replace_monopole(template_charge, &mt_charge)?;
    let mut magnetization_z_muffin_tins = zero_like_template
        .charge()
        .zero_like()
        .muffin_tins()
        .to_vec();
    magnetization_z_muffin_tins[site_index] =
        replace_monopole(template_charge, &mt_magnetization_z)?;
    let zero_mode_adjustment = close_finite_layout_zero_mode(
        geometry,
        layout,
        FiniteLayoutClosureComponent {
            muffin_tins: &charge_muffin_tins,
            requested_integral: requested_charge,
            fourier: &mut fourier_charge,
        },
        FiniteLayoutClosureComponent {
            muffin_tins: &magnetization_z_muffin_tins,
            requested_integral: requested_magnetization_z,
            fourier: &mut fourier_magnetization_z,
        },
    )?;
    let charge = RegionalScalarField::new(
        geometry.clone(),
        charge_muffin_tins,
        InterstitialField::from_fourier_field(HermitianFourierField::new(
            layout.clone(),
            fourier_charge,
        )?),
    )?;
    let magnetization_z = RegionalScalarField::new(
        geometry.clone(),
        magnetization_z_muffin_tins,
        InterstitialField::from_fourier_field(HermitianFourierField::new(
            layout.clone(),
            fourier_magnetization_z,
        )?),
    )?;
    let zero_magnetization = charge.zero_like();
    let density = RegionalDensity::new(
        charge,
        [
            zero_magnetization.clone(),
            zero_magnetization,
            magnetization_z,
        ],
    )?;
    let represented_charge = scalar_field_integral(density.charge())?;
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
        .charge()
        .interstitial()
        .field()
        .iter()
        .filter(|(vector, _)| vector.index != [0; 3])
        .map(|(_, charge)| charge.norm_sqr())
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

pub(crate) struct FiniteLayoutClosureComponent<'a> {
    pub(crate) muffin_tins: &'a [MuffinTinField],
    pub(crate) requested_integral: f64,
    pub(crate) fourier: &'a mut [Complex64],
}

pub(crate) fn close_finite_layout_zero_mode(
    geometry: &InterstitialGeometry,
    layout: &muffintin_core::FourierLayout,
    charge: FiniteLayoutClosureComponent<'_>,
    magnetization_z: FiniteLayoutClosureComponent<'_>,
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
    let uncorrected_charge = preview(charge.muffin_tins, &*charge.fourier)?;
    let uncorrected_magnetization_z =
        preview(magnetization_z.muffin_tins, &*magnetization_z.fourier)?;
    let coefficient_correction = [
        (charge.requested_integral - uncorrected_charge) / response_volume,
        (magnetization_z.requested_integral - uncorrected_magnetization_z) / response_volume,
    ];
    if coefficient_correction
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(CoreDensityError::NonFiniteZeroModeCorrection {
            correction: coefficient_correction,
        });
    }
    charge.fourier[zero].re += coefficient_correction[0];
    magnetization_z.fourier[zero].re += coefficient_correction[1];
    Ok(PseudochargeZeroModeAdjustment {
        interstitial_fraction,
        response_volume,
        requested_charge: charge.requested_integral,
        requested_magnetization_z: magnetization_z.requested_integral,
        uncorrected_charge,
        uncorrected_magnetization_z,
        charge_coefficient_correction: coefficient_correction[0],
        magnetization_z_coefficient_correction: coefficient_correction[1],
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
    shell: &RegionalCoreShellInput<'_>,
    layout: &muffintin_core::FourierLayout,
) -> Result<SmoothShellTransform, CoreDensityError> {
    let mt_len = muffin_tin_mesh.len();
    let probability = shell
        .p
        .iter()
        .zip(shell.q)
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
    let capacity = f64::from(shell.state.kappa.degeneracy());
    if !shell.occupation.is_finite() || shell.occupation < 0.0 || shell.occupation > capacity {
        return Err(CoreDensityError::InvalidOccupation {
            occupation: shell.occupation,
            capacity,
        });
    }
    if shell.p.len() != shell.mesh.len() || shell.q.len() != shell.mesh.len() {
        return Err(CoreDensityError::SolutionMeshLength {
            mesh: shell.mesh.len(),
            p: shell.p.len(),
            q: shell.q.len(),
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
    #[error("core sidecar shell {shell} does not contain every magnetic channel exactly once")]
    SidecarMagneticChannels { shell: usize },
    #[error("core sidecar shell {shell} is not uniform over magnetic channels")]
    SidecarOpenShell { shell: usize },
    #[error("core sidecar shell {shell} has invalid occupation {occupation}")]
    SidecarOccupation { shell: usize, occupation: f64 },
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
    use muffintin_sphere::HarmonicConvention;
    use muffintin_sphere::{CoreDiracSolution, CoreState, RelativisticRole};

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
        let charge = RegionalScalarField::new(geometry, vec![muffin_tin], interstitial).unwrap();
        let zero = charge.zero_like();
        RegionalDensity::new(charge, [zero.clone(), zero.clone(), zero]).unwrap()
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
                state: solution.state,
                energy: solution.energy,
                p: &solution.p,
                q: &solution.q,
                norm_mt: solution.norm_mt,
                spill: solution.spill,
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
        let charge = result.contribution.density.charge().muffin_tins()[0]
            .field()
            .channel(0, 0)
            .unwrap();
        assert!(charge.iter().any(|value| value.re > 0.0));
        for component in result.contribution.density.magnetization() {
            assert_eq!(component, &result.contribution.density.charge().zero_like());
        }
        assert!((result.diagnostics.requested_charge - 2.0).abs() < 1.0e-14);
        assert!((result.diagnostics.represented_charge - 2.0).abs() < 2.0e-12);
        assert_eq!(
            result.diagnostics.zero_mode_adjustment.requested_charge,
            2.0
        );
        assert_eq!(
            result
                .diagnostics
                .zero_mode_adjustment
                .requested_magnetization_z,
            0.0
        );
        assert!((result.contribution.eigenvalue_sum.get() + 1.0).abs() < 1.0e-14);

        let polarized = build_regional_core_contribution(
            "X".to_owned(),
            template.geometry(),
            0,
            &mt_mesh,
            &[RegionalCoreShellInput {
                mesh: &extended_mesh,
                state: solution.state,
                energy: solution.energy,
                p: &solution.p,
                q: &solution.q,
                norm_mt: solution.norm_mt,
                spill: solution.spill,
                occupation: 2.0,
                spin: CoreSpinPartition::ExplicitCollinear { up: 2.0, down: 0.0 },
            }],
            &template,
        )
        .unwrap();
        assert_eq!(
            polarized.contribution.density.magnetization()[2],
            *polarized.contribution.density.charge()
        );
        assert_eq!(
            polarized.contribution.density.magnetization()[0],
            polarized.contribution.density.charge().zero_like()
        );
        assert_eq!(
            polarized.contribution.density.magnetization()[1],
            polarized.contribution.density.charge().zero_like()
        );
        assert_eq!(
            polarized
                .diagnostics
                .zero_mode_adjustment
                .requested_magnetization_z,
            2.0
        );
        assert_eq!(
            polarized
                .diagnostics
                .zero_mode_adjustment
                .magnetization_z_coefficient_correction,
            polarized
                .diagnostics
                .zero_mode_adjustment
                .charge_coefficient_correction
        );
    }

    #[test]
    fn g0_only_pseudocharge_closes_exact_core_occupation() {
        let (mt_mesh, extended_mesh) = meshes();
        let solution = core_solution(&mt_mesh, &extended_mesh, false, false);
        let template = template([Bohr(0.0); 3], &mt_mesh, 0.0);
        let result = build(&template, &mt_mesh, &extended_mesh, &solution, 2.0);
        let adjustment = result.diagnostics.zero_mode_adjustment;
        let uncorrected = adjustment.uncorrected_charge;

        assert!((uncorrected - 2.0).abs() > 1.0e-3);
        assert!(adjustment.charge_coefficient_correction.abs() > 1.0e-8);
        assert_eq!(adjustment.magnetization_z_coefficient_correction, 0.0);
        assert!((result.diagnostics.represented_charge - 2.0).abs() < 2.0e-12);
        assert!(
            (scalar_field_integral(result.contribution.density.charge()).unwrap() - 2.0).abs()
                < 2.0e-12
        );
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
            (density.diagnostics.zero_mode_adjustment.uncorrected_charge - 2.0).abs()
        };
        let low_error = uncorrected_error(&low);
        let high_error = uncorrected_error(&high);
        assert!(high_error < low_error);
        assert!((low.diagnostics.represented_charge - 2.0).abs() < 2.0e-12);
        assert!((high.diagnostics.represented_charge - 2.0).abs() < 2.0e-12);
        assert!(high.diagnostics.spill_charge > 0.0);
        assert!(high.diagnostics.finite_g_norm > 1.0e-8);
        let layout = high.contribution.density.charge().interstitial().layout();
        let g0 = layout.index([0, 0, 0]).unwrap();
        assert!(
            high.contribution
                .density
                .charge()
                .interstitial()
                .field()
                .coefficients()[g0]
                .re
                < high.diagnostics.requested_charge
        );
        let shell = RegionalCoreShellInput {
            mesh: &extended_mesh,
            state: solution.state,
            energy: solution.energy,
            p: &solution.p,
            q: &solution.q,
            norm_mt: solution.norm_mt,
            spill: solution.spill,
            occupation: 2.0,
            spin: CoreSpinPartition::ClosedShellAverage,
        };
        let transform = smooth_shell_transform(
            high_template.geometry(),
            0,
            &mt_mesh,
            &extended_mesh,
            &shell,
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
                .charge()
                .interstitial()
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
            .charge()
            .interstitial()
            .coefficient(index)
            .unwrap();
        let shifted_value = shifted
            .contribution
            .density
            .charge()
            .interstitial()
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
                .charge()
                .interstitial()
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
        let compact_interstitial = compact_result.contribution.density.charge().interstitial();
        for (vector, coefficient) in compact_interstitial.field().iter() {
            if vector.index != [0; 3] {
                assert!(coefficient.norm() < 1.0e-14);
            }
        }
        assert_eq!(
            compact_interstitial.coefficient([0; 3]).unwrap().re,
            compact_result
                .diagnostics
                .zero_mode_adjustment
                .charge_coefficient_correction
        );
        assert!(
            (compact_result.diagnostics.represented_charge - 2.0).abs() < 2.0e-12,
            "represented compact charge = {}",
            compact_result.diagnostics.represented_charge
        );
    }
}
