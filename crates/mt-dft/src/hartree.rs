//! Charge-only regional adapter for full-potential electrostatics.

use crate::{InterstitialField, MuffinTinField, RegionalError, RegionalScalarField};
use muffintin_core::{
    FourierFieldError, FourierLayout, Hartree, HermitianFourierField, InterstitialGeometry, Lm,
    MeshError, StepFunctionError, lm_count, lm_index,
};
use muffintin_coulomb::{
    HartreeError, InterstitialHartreePotential, MuffinTinChargeDensity, MuffinTinHartreePotential,
    PeriodicChargeTreatment, RawElectrostaticPotential, RawHartreePotential, RawNuclearPotential,
    WeinertChargeDensity, WeinertHartreeSpec, solve_periodic_nuclear_potential,
    solve_weinert_hartree,
};
use muffintin_sphere::{HarmonicConvention, SphereField, SphereFieldError};
use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

const REALITY_TOLERANCE: f64 = 4096.0 * f64::EPSILON;
const TOTAL_CHARGE_TOLERANCE: f64 = 1.0e-8;

/// Full-potential electronic-Hartree and periodic-nuclear specification.
#[derive(Clone, Debug, PartialEq)]
pub struct ElectrostaticSpec {
    weinert: WeinertHartreeSpec,
    nuclear_charges: Vec<f64>,
}

impl ElectrostaticSpec {
    /// Construct an electronic-Hartree specification with explicit nuclei.
    pub fn new(
        weinert: WeinertHartreeSpec,
        nuclear_charges: Vec<f64>,
    ) -> Result<Self, RegionalElectrostaticError> {
        require_electronic_treatment(weinert)?;
        Ok(Self {
            weinert,
            nuclear_charges,
        })
    }

    /// Weinert order and electronic constant-mode treatment.
    pub fn weinert(&self) -> WeinertHartreeSpec {
        self.weinert
    }

    /// Positive nuclear charges in geometry site order.
    pub fn nuclear_charges(&self) -> &[f64] {
        &self.nuclear_charges
    }
}

/// Electrostatic potential and the energy terms consumed by an SCF step.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionalElectrostaticResult {
    /// Scalar `V_H + V_nuclear`. Its interstitial coefficients are
    /// step-function masked and ready for the LAPW operator boundary.
    pub potential: RegionalScalarField,
    /// SPEX Coulomb integral `C = integral n (V_H + V_nuc)`.
    pub coulomb: Hartree,
    /// SPEX Madelung term `M = E_en + 2 E_II`.
    pub madelung: Hartree,
    /// Conventional electronic Hartree energy `E_H = 1/2 integral n V_H`.
    pub electron_hartree: Hartree,
    /// Electron-nuclear energy `E_en = integral n V_nuc`.
    pub electron_nuclear: Hartree,
    /// Periodic ion-ion energy in the common zero-mean gauge.
    pub nuclear_nuclear: Hartree,
    /// Dense complex-harmonic charge density passed to Weinert.
    pub weinert_density: WeinertChargeDensity,
    /// Raw unmasked electronic Hartree potential.
    pub raw_hartree: RawHartreePotential,
    /// Raw unmasked periodic nuclear external potential.
    pub raw_nuclear: RawNuclearPotential,
    /// Raw unmasked sum of electronic and nuclear potentials.
    pub raw_electrostatic: RawElectrostaticPotential,
}

/// Convert the regional electronic charge component and solve its electrostatics.
///
/// The API accepts only [`RegionalScalarField`], so no magnetization component
/// can enter Poisson's equation. The electronic and nuclear
/// subsystems each omit their constant Fourier source through compensating
/// backgrounds; this adapter additionally requires their physical source
/// charges to cancel. The resulting raw interstitial potential is convolved
/// with the interstitial step function before it is exposed as a
/// [`RegionalScalarField`].
pub fn evaluate_regional_electrostatics(
    charge: &RegionalScalarField,
    spec: &ElectrostaticSpec,
) -> Result<RegionalElectrostaticResult, RegionalElectrostaticError> {
    let weinert_density = weinert_density(charge)?;
    let raw_hartree = solve_weinert_hartree(&weinert_density, spec.weinert())?;
    let raw_nuclear = solve_periodic_nuclear_potential(&weinert_density, spec.nuclear_charges())?;
    let raw_electrostatic = raw_hartree.add_nuclear_external(&raw_nuclear)?;

    let charge_scale = raw_hartree
        .source_charge()
        .abs()
        .max(raw_nuclear.source_charge().abs())
        .max(1.0);
    let net_charge = raw_electrostatic.source_charge();
    let tolerance = TOTAL_CHARGE_TOLERANCE * charge_scale;
    if net_charge.abs() > tolerance {
        return Err(RegionalElectrostaticError::NonNeutralElectronNuclear {
            charge: net_charge,
            tolerance,
        });
    }

    let hartree_integral = density_potential_integral(
        &weinert_density,
        raw_hartree.muffin_tins(),
        raw_hartree.interstitial(),
        "electronic Hartree energy",
    )?;
    let electron_nuclear = density_potential_integral(
        &weinert_density,
        raw_nuclear.muffin_tins(),
        raw_nuclear.interstitial(),
        "electron-nuclear energy",
    )?;
    let coulomb = density_potential_integral(
        &weinert_density,
        raw_electrostatic.muffin_tins(),
        raw_electrostatic.interstitial(),
        "Coulomb integral",
    )?;
    let electron_hartree = Hartree(0.5 * hartree_integral.get());
    let nuclear_nuclear = nuclear_nuclear_energy(&raw_nuclear)?;
    let madelung = Hartree(electron_nuclear.get() + 2.0 * nuclear_nuclear.get());
    let potential = regional_potential(charge, &raw_electrostatic)?;

    Ok(RegionalElectrostaticResult {
        potential,
        coulomb,
        madelung,
        electron_hartree,
        electron_nuclear,
        nuclear_nuclear,
        weinert_density,
        raw_hartree,
        raw_nuclear,
        raw_electrostatic,
    })
}

fn require_electronic_treatment(
    spec: WeinertHartreeSpec,
) -> Result<(), RegionalElectrostaticError> {
    if spec.charge_treatment() != PeriodicChargeTreatment::ElectronicWithUniformBackground {
        Err(RegionalElectrostaticError::NonElectronicChargeTreatment)
    } else {
        Ok(())
    }
}

fn weinert_density(
    charge: &RegionalScalarField,
) -> Result<WeinertChargeDensity, RegionalElectrostaticError> {
    let muffin_tins = charge
        .muffin_tins()
        .iter()
        .map(muffin_tin_charge)
        .collect::<Result<Vec<_>, _>>()?;

    let interstitial = charge.interstitial().field().clone();
    Ok(WeinertChargeDensity::new(
        charge.geometry().clone(),
        muffin_tins,
        interstitial,
    )?)
}

fn muffin_tin_charge(
    charge: &MuffinTinField,
) -> Result<MuffinTinChargeDensity, RegionalElectrostaticError> {
    let field = charge.field();
    let l_max = field
        .channels()
        .map(|(channel, _)| channel.l)
        .max()
        .unwrap_or(0);
    let n_radial = charge.mesh().len();
    let mut coefficients = vec![Complex64::default(); lm_count(l_max) * n_radial];
    for (channel, values) in field.channels() {
        match field.convention() {
            HarmonicConvention::Complex => {
                let start = channel.index() * n_radial;
                coefficients[start..start + n_radial].copy_from_slice(values);
            }
            HarmonicConvention::Real => {
                accumulate_real_channel(
                    &mut coefficients,
                    n_radial,
                    channel,
                    values.iter().copied(),
                );
            }
        }
    }
    Ok(MuffinTinChargeDensity::new(
        charge.mesh().clone(),
        l_max,
        coefficients,
    )?)
}

fn accumulate_real_channel(
    coefficients: &mut [Complex64],
    n_radial: usize,
    channel: Lm,
    values: impl Iterator<Item = Complex64>,
) {
    if channel.m == 0 {
        let start = channel.index() * n_radial;
        for (target, value) in coefficients[start..start + n_radial].iter_mut().zip(values) {
            *target += value;
        }
        return;
    }
    let q = channel.m.unsigned_abs() as i32;
    let positive = lm_index(channel.l, q).expect("validated real harmonic") * n_radial;
    let negative = lm_index(channel.l, -q).expect("validated real harmonic") * n_radial;
    let inverse_sqrt_two = 1.0 / 2.0_f64.sqrt();
    let phase = if q % 2 == 0 { 1.0 } else { -1.0 };
    for (radial, value) in values.enumerate() {
        if channel.m > 0 {
            coefficients[positive + radial] += phase * inverse_sqrt_two * value;
            coefficients[negative + radial] += inverse_sqrt_two * value;
        } else {
            coefficients[positive + radial] += Complex64::new(0.0, inverse_sqrt_two) * value;
            coefficients[negative + radial] +=
                Complex64::new(0.0, -phase * inverse_sqrt_two) * value;
        }
    }
}

fn regional_potential(
    charge: &RegionalScalarField,
    raw: &RawElectrostaticPotential,
) -> Result<RegionalScalarField, RegionalElectrostaticError> {
    let muffin_tins = raw
        .muffin_tins()
        .iter()
        .map(raw_muffin_tin_field)
        .collect::<Result<Vec<_>, _>>()?;
    let interstitial = masked_interstitial(charge, raw.interstitial())?;
    RegionalScalarField::new(charge.geometry().clone(), muffin_tins, interstitial)
        .map_err(Into::into)
}

fn raw_muffin_tin_field(
    raw: &MuffinTinHartreePotential,
) -> Result<MuffinTinField, RegionalElectrostaticError> {
    let mut channels = Vec::with_capacity(lm_count(raw.l_max()));
    for l in 0..=raw.l_max() {
        for m in -(l as i32)..=l as i32 {
            let values = raw
                .channel(l, m)
                .expect("raw potential stores every channel through l_max")
                .iter()
                .map(|value| value.as_complex())
                .collect();
            channels.push(((l, m), values));
        }
    }
    Ok(MuffinTinField::new(
        raw.mesh().clone(),
        SphereField::new(HarmonicConvention::Complex, channels)?,
    )?)
}

fn masked_interstitial(
    charge: &RegionalScalarField,
    raw: &InterstitialHartreePotential,
) -> Result<InterstitialField, RegionalElectrostaticError> {
    let layout = raw.layout();
    let raw_coefficients = raw
        .coefficients()
        .map(|value| value.as_complex())
        .collect::<Vec<_>>();
    let masked = mask_fourier_coefficients(charge.geometry(), layout, &raw_coefficients)?;
    Ok(InterstitialField::from_fourier_field(
        HermitianFourierField::new(layout.clone(), masked)?,
    ))
}

fn mask_fourier_coefficients(
    geometry: &InterstitialGeometry,
    layout: &FourierLayout,
    raw_coefficients: &[Complex64],
) -> Result<Vec<Complex64>, RegionalElectrostaticError> {
    if raw_coefficients.len() != layout.len() {
        return Err(RegionalElectrostaticError::RawLayoutMismatch);
    }
    let reciprocal = layout.reciprocal();
    let mut masked = Vec::with_capacity(layout.len());
    for target in layout.vectors() {
        let mut value = Complex64::default();
        for (source, &coefficient) in layout.vectors().iter().zip(raw_coefficients) {
            let difference = reciprocal_difference(target.index, source.index)?;
            value += geometry.coefficient(reciprocal.cartesian(difference))? * coefficient;
        }
        masked.push(value);
    }
    canonicalize_fourier(layout, &mut masked)?;
    Ok(masked)
}

fn density_potential_integral(
    density: &WeinertChargeDensity,
    muffin_tins: &[MuffinTinHartreePotential],
    interstitial: &InterstitialHartreePotential,
    quantity: &'static str,
) -> Result<Hartree, RegionalElectrostaticError> {
    if density.muffin_tins().len() != muffin_tins.len()
        || density.interstitial().layout() != interstitial.layout()
    {
        return Err(RegionalElectrostaticError::RawLayoutMismatch);
    }
    let mut total = Complex64::default();
    let mut scale = 0.0;
    for (charge, potential) in density.muffin_tins().iter().zip(muffin_tins) {
        if charge.mesh() != potential.mesh() || charge.l_max() != potential.l_max() {
            return Err(RegionalElectrostaticError::RawLayoutMismatch);
        }
        let radii = charge.mesh().radii();
        for l in 0..=charge.l_max() {
            for m in -(l as i32)..=l as i32 {
                let products = charge
                    .channel(l, m)
                    .expect("dense charge channel")
                    .iter()
                    .zip(potential.channel(l, m).expect("dense potential channel"))
                    .zip(radii)
                    .map(|((&rho, &value), radius)| {
                        rho.conj() * value.as_complex() * radius.get().powi(2)
                    })
                    .collect::<Vec<_>>();
                let real = products.iter().map(|value| value.re).collect::<Vec<_>>();
                let imaginary = products.iter().map(|value| value.im).collect::<Vec<_>>();
                let contribution = Complex64::new(
                    charge.mesh().integrate(&real)?,
                    charge.mesh().integrate(&imaginary)?,
                );
                total += contribution;
                scale += contribution.norm();
            }
        }
    }

    let volume = density.geometry().cell_volume().get();
    let reciprocal = density.interstitial().layout().reciprocal();
    let potential_coefficients = interstitial
        .coefficients()
        .map(|value| value.as_complex())
        .collect::<Vec<_>>();
    for (left, &rho) in density.interstitial().iter() {
        for (right, &potential) in density
            .interstitial()
            .layout()
            .vectors()
            .iter()
            .zip(&potential_coefficients)
        {
            let difference = reciprocal_difference(left.index, right.index)?;
            let theta = density
                .geometry()
                .coefficient(reciprocal.cartesian(difference))?;
            let term = volume * rho.conj() * theta * potential;
            total += term;
            scale += term.norm();
        }
    }
    checked_real_hartree(total, scale, quantity)
}

fn nuclear_nuclear_energy(
    nuclear: &RawNuclearPotential,
) -> Result<Hartree, RegionalElectrostaticError> {
    let mut energy = 0.0;
    for (site, (&charge, potential)) in nuclear
        .nuclear_charges()
        .iter()
        .zip(nuclear.muffin_tins())
        .enumerate()
    {
        let monopole = potential
            .channel(0, 0)
            .ok_or(RegionalElectrostaticError::MissingNuclearMonopole { site })?;
        let surface = monopole
            .last()
            .ok_or(RegionalElectrostaticError::MissingNuclearMonopole { site })?
            .as_complex();
        let regular = surface / (4.0 * PI).sqrt() + charge / potential.mesh().last().get();
        if regular.im.abs() > REALITY_TOLERANCE * regular.norm().max(1.0) {
            return Err(RegionalElectrostaticError::NonRealEnergy {
                quantity: "regular nuclear potential",
                imaginary: regular.im,
                tolerance: REALITY_TOLERANCE * regular.norm().max(1.0),
            });
        }
        energy -= 0.5 * charge * regular.re;
    }
    Hartree::checked(energy).ok_or(RegionalElectrostaticError::NonFiniteEnergy {
        quantity: "nuclear-nuclear energy",
    })
}

fn checked_real_hartree(
    value: Complex64,
    scale: f64,
    quantity: &'static str,
) -> Result<Hartree, RegionalElectrostaticError> {
    let tolerance = REALITY_TOLERANCE * scale.max(value.norm()).max(1.0);
    if value.im.abs() > tolerance {
        return Err(RegionalElectrostaticError::NonRealEnergy {
            quantity,
            imaginary: value.im,
            tolerance,
        });
    }
    Hartree::checked(value.re).ok_or(RegionalElectrostaticError::NonFiniteEnergy { quantity })
}

fn canonicalize_fourier(
    layout: &FourierLayout,
    coefficients: &mut [Complex64],
) -> Result<(), RegionalElectrostaticError> {
    for vector in layout.vectors() {
        let position = layout
            .index(vector.index)
            .expect("layout contains its own vector");
        let opposite_index = vector
            .index
            .map(|component| component.checked_neg())
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .and_then(|values| values.try_into().ok())
            .ok_or(RegionalElectrostaticError::ReciprocalIndexOverflow {
                left: [0; 3],
                right: vector.index,
            })?;
        let opposite = layout
            .index(opposite_index)
            .ok_or(FourierFieldError::MissingConjugate {
                index: vector.index,
            })?;
        if vector.index == [0; 3] {
            let residual = coefficients[position].im.abs();
            let tolerance = REALITY_TOLERANCE * coefficients[position].norm().max(1.0);
            if residual > tolerance {
                return Err(RegionalElectrostaticError::NonHermitianMaskedPotential {
                    g: vector.index,
                    residual,
                    tolerance,
                });
            }
            coefficients[position].im = 0.0;
        } else if vector.index < opposite_index {
            let residual = (coefficients[opposite] - coefficients[position].conj()).norm();
            let tolerance = REALITY_TOLERANCE
                * coefficients[position]
                    .norm()
                    .max(coefficients[opposite].norm())
                    .max(1.0);
            if residual > tolerance {
                return Err(RegionalElectrostaticError::NonHermitianMaskedPotential {
                    g: vector.index,
                    residual,
                    tolerance,
                });
            }
            let average = (coefficients[position] + coefficients[opposite].conj()) * 0.5;
            coefficients[position] = average;
            coefficients[opposite] = average.conj();
        }
    }
    Ok(())
}

fn reciprocal_difference(
    left: [i32; 3],
    right: [i32; 3],
) -> Result<[i32; 3], RegionalElectrostaticError> {
    Ok([
        left[0]
            .checked_sub(right[0])
            .ok_or(RegionalElectrostaticError::ReciprocalIndexOverflow { left, right })?,
        left[1]
            .checked_sub(right[1])
            .ok_or(RegionalElectrostaticError::ReciprocalIndexOverflow { left, right })?,
        left[2]
            .checked_sub(right[2])
            .ok_or(RegionalElectrostaticError::ReciprocalIndexOverflow { left, right })?,
    ])
}

/// Invalid regional electrostatic input or failed full-potential solve.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RegionalElectrostaticError {
    #[error(transparent)]
    Hartree(#[from] HartreeError),
    #[error(transparent)]
    Regional(#[from] RegionalError),
    #[error(transparent)]
    Sphere(#[from] SphereFieldError),
    #[error(transparent)]
    Fourier(#[from] FourierFieldError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    StepFunction(#[from] StepFunctionError),
    #[error("regional electrostatics requires ElectronicWithUniformBackground")]
    NonElectronicChargeTreatment,
    #[error("raw charge and potential layouts differ")]
    RawLayoutMismatch,
    #[error("electron-plus-nuclear source is not neutral: charge {charge}, tolerance {tolerance}")]
    NonNeutralElectronNuclear { charge: f64, tolerance: f64 },
    #[error("reciprocal index overflows in {left:?} - {right:?}")]
    ReciprocalIndexOverflow { left: [i32; 3], right: [i32; 3] },
    #[error(
        "masked interstitial potential is not Hermitian at G={g:?}: residual {residual}, tolerance {tolerance}"
    )]
    NonHermitianMaskedPotential {
        g: [i32; 3],
        residual: f64,
        tolerance: f64,
    },
    #[error("site {site} has no nuclear monopole sample")]
    MissingNuclearMonopole { site: usize },
    #[error("{quantity} has imaginary residual {imaginary}, tolerance {tolerance}")]
    NonRealEnergy {
        quantity: &'static str,
        imaginary: f64,
        tolerance: f64,
    },
    #[error("{quantity} is not finite")]
    NonFiniteEnergy { quantity: &'static str },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RegionalDensity;
    use muffintin_core::{
        Bohr, ExponentialMesh, GVector, InverseBohr, ReciprocalLattice, Sphere, VolumeBohr3,
        spherical_bessel_j,
    };
    use std::f64::consts::TAU;

    const LATTICE: f64 = 8.0;
    const RADIUS: f64 = 1.0;
    const CENTER: [Bohr; 3] = [Bohr(4.0), Bohr(4.0), Bohr(4.0)];

    fn reciprocal() -> ReciprocalLattice {
        ReciprocalLattice::from_direct([
            [Bohr(LATTICE), Bohr(0.0), Bohr(0.0)],
            [Bohr(0.0), Bohr(LATTICE), Bohr(0.0)],
            [Bohr(0.0), Bohr(0.0), Bohr(LATTICE)],
        ])
        .unwrap()
    }

    fn shell_layout(shells: f64) -> FourierLayout {
        let reciprocal = reciprocal();
        let cutoff = InverseBohr(shells * TAU / LATTICE);
        FourierLayout::new(reciprocal, reciprocal.enumerate(cutoff).unwrap()).unwrap()
    }

    fn line_layout() -> FourierLayout {
        let reciprocal = reciprocal();
        let vectors = [[-1, 0, 0], [0, 0, 0], [1, 0, 0]]
            .into_iter()
            .map(|index| {
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
            .collect();
        FourierLayout::new(reciprocal, vectors).unwrap()
    }

    fn mesh() -> ExponentialMesh {
        let first: f64 = 1.0e-5;
        let number = 101;
        let increment = (RADIUS / first).ln() / (number - 1) as f64;
        ExponentialMesh::new(Bohr(first), increment, number).unwrap()
    }

    fn sphere_field(mesh: &ExponentialMesh, value: f64) -> MuffinTinField {
        MuffinTinField::new(
            mesh.clone(),
            SphereField::new(
                HarmonicConvention::Real,
                [(
                    (0, 0),
                    vec![Complex64::new((4.0 * PI).sqrt() * value, 0.0); mesh.len()],
                )],
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn regional_charge() -> RegionalScalarField {
        let layout = shell_layout(3.0);
        let interstitial = |mean: f64, mode: Complex64| {
            let coefficients = layout
                .vectors()
                .iter()
                .map(|vector| match vector.index {
                    [0, 0, 0] => Complex64::new(mean, 0.0),
                    [1, 0, 0] => mode,
                    [-1, 0, 0] => mode.conj(),
                    _ => Complex64::default(),
                })
                .collect();
            InterstitialField::from_fourier_field(
                HermitianFourierField::new(layout.clone(), coefficients).unwrap(),
            )
        };
        let radial_mesh = mesh();
        RegionalScalarField::new(
            InterstitialGeometry::new(
                VolumeBohr3(LATTICE.powi(3)),
                vec![Sphere {
                    center: CENTER,
                    radius: Bohr(RADIUS),
                }],
            )
            .unwrap(),
            vec![sphere_field(&radial_mesh, 0.03)],
            interstitial(0.03, Complex64::new(0.003, 0.0005)),
        )
        .unwrap()
    }

    #[test]
    fn sparse_real_charge_becomes_dense_complex_harmonics() {
        let radial_mesh = ExponentialMesh::new(Bohr(0.01), 0.2, 7).unwrap();
        let make = |positive: f64, negative: f64| {
            MuffinTinField::new(
                radial_mesh.clone(),
                SphereField::new(
                    HarmonicConvention::Real,
                    [
                        (
                            (1, -1),
                            vec![Complex64::new(negative, 0.0); radial_mesh.len()],
                        ),
                        (
                            (1, 1),
                            vec![Complex64::new(positive, 0.0); radial_mesh.len()],
                        ),
                    ],
                )
                .unwrap(),
            )
            .unwrap()
        };
        let dense = muffin_tin_charge(&make(2.0, 3.0)).unwrap();
        assert_eq!(dense.l_max(), 1);
        assert_eq!(dense.channel(0, 0).unwrap(), vec![Complex64::default(); 7]);
        let inverse_sqrt_two = 1.0 / 2.0_f64.sqrt();
        let positive = Complex64::new(-2.0 * inverse_sqrt_two, 3.0 * inverse_sqrt_two);
        let negative = Complex64::new(2.0 * inverse_sqrt_two, 3.0 * inverse_sqrt_two);
        assert!(
            dense
                .channel(1, 1)
                .unwrap()
                .iter()
                .all(|&value| value == positive)
        );
        assert!(
            dense
                .channel(1, -1)
                .unwrap()
                .iter()
                .all(|&value| value == negative)
        );
        assert_eq!(negative, -positive.conj());
        assert_eq!(dense.channel(1, 0).unwrap(), vec![Complex64::default(); 7]);
    }

    #[test]
    fn mask_convolution_handles_uniform_and_finite_fourier_fields() {
        let layout = line_layout();
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(LATTICE.powi(3)),
            vec![Sphere {
                center: [Bohr(0.0); 3],
                radius: Bohr(0.5),
            }],
        )
        .unwrap();
        let uniform = [
            Complex64::default(),
            Complex64::new(2.0, 0.0),
            Complex64::default(),
        ];
        let masked_uniform = mask_fourier_coefficients(&geometry, &layout, &uniform).unwrap();
        for (position, target) in layout.vectors().iter().enumerate() {
            let expected = 2.0 * geometry.coefficient(target.cartesian).unwrap();
            assert!((masked_uniform[position] - expected).norm() < 1.0e-14);
        }

        let mode = Complex64::new(0.7, 0.2);
        let finite = [mode.conj(), Complex64::default(), mode];
        let masked_finite = mask_fourier_coefficients(&geometry, &layout, &finite).unwrap();
        for (position, target) in layout.vectors().iter().enumerate() {
            let minus = reciprocal_difference(target.index, [-1, 0, 0]).unwrap();
            let plus = reciprocal_difference(target.index, [1, 0, 0]).unwrap();
            let expected = geometry
                .coefficient(layout.reciprocal().cartesian(minus))
                .unwrap()
                * mode.conj()
                + geometry
                    .coefficient(layout.reciprocal().cartesian(plus))
                    .unwrap()
                    * mode;
            assert!((masked_finite[position] - expected).norm() < 1.0e-14);
        }
        HermitianFourierField::new(layout, masked_finite).unwrap();
    }

    #[test]
    fn regional_electrostatics_closes_boundary_energy_spin_and_layout() {
        let charge = regional_charge();
        let converted = weinert_density(&charge).unwrap();
        let electronic =
            solve_weinert_hartree(&converted, WeinertHartreeSpec::electronic(4).unwrap()).unwrap();
        let spec = ElectrostaticSpec::new(
            WeinertHartreeSpec::electronic(4).unwrap(),
            vec![electronic.source_charge()],
        )
        .unwrap();
        let result = evaluate_regional_electrostatics(&charge, &spec).unwrap();

        assert!(result.raw_electrostatic.source_charge().abs() < 1.0e-12);
        assert!(
            result
                .raw_electrostatic
                .neutralizing_background_density()
                .abs()
                < 1.0e-14
        );
        assert_eq!(
            result.potential.interstitial().layout(),
            charge.interstitial().layout()
        );
        assert_eq!(
            result.potential.muffin_tins()[0].mesh(),
            charge.muffin_tins()[0].mesh()
        );
        assert_eq!(
            result.potential.muffin_tins()[0].field().convention(),
            HarmonicConvention::Complex
        );

        let raw_coefficients = result
            .raw_electrostatic
            .interstitial()
            .coefficients()
            .map(|value| value.as_complex())
            .collect::<Vec<_>>();
        let expected_masked = mask_fourier_coefficients(
            charge.geometry(),
            charge.interstitial().layout(),
            &raw_coefficients,
        )
        .unwrap();
        assert_eq!(
            result.potential.interstitial().field().coefficients(),
            expected_masked
        );

        let raw_boundary = result.raw_electrostatic.muffin_tins()[0]
            .channel(0, 0)
            .unwrap()
            .last()
            .unwrap()
            .as_complex();
        let y00 = 1.0 / (4.0 * PI).sqrt();
        let mut fourier_boundary = Complex64::default();
        for vector in result.raw_electrostatic.interstitial().layout().vectors() {
            let phase = vector
                .cartesian
                .iter()
                .zip(CENTER)
                .map(|(component, coordinate)| component.get() * coordinate.get())
                .sum::<f64>();
            fourier_boundary += 4.0
                * PI
                * spherical_bessel_j(0, vector.norm.get() * RADIUS)
                * Complex64::from_polar(1.0, phase)
                * y00
                * result
                    .raw_electrostatic
                    .interstitial()
                    .coefficient(vector.index)
                    .unwrap()
                    .as_complex();
        }
        assert!(
            (raw_boundary - fourier_boundary).norm() < 3.0e-10,
            "raw boundary {raw_boundary}, Fourier boundary {fourier_boundary}, residual {}",
            (raw_boundary - fourier_boundary).norm()
        );

        assert!(
            (result.coulomb.get()
                - (2.0 * result.electron_hartree.get() + result.electron_nuclear.get()))
            .abs()
                < 2.0e-10
        );
        assert!(
            ((result.madelung.get() - result.coulomb.get()) / 2.0
                - (result.nuclear_nuclear.get() - result.electron_hartree.get()))
            .abs()
                < 2.0e-10
        );
    }

    #[test]
    fn regional_electrostatics_rejects_non_neutral_electron_nuclear_cell() {
        let charge = regional_charge();
        let converted = weinert_density(&charge).unwrap();
        let electronic =
            solve_weinert_hartree(&converted, WeinertHartreeSpec::electronic(4).unwrap()).unwrap();
        let spec = ElectrostaticSpec::new(
            WeinertHartreeSpec::electronic(4).unwrap(),
            vec![electronic.source_charge() + 0.01],
        )
        .unwrap();
        assert!(matches!(
            evaluate_regional_electrostatics(&charge, &spec),
            Err(RegionalElectrostaticError::NonNeutralElectronNuclear { .. })
        ));
    }

    #[test]
    fn hartree_is_independent_of_magnetization_orientation() {
        let charge = regional_charge();
        let zero = charge.zero_like();
        let mut finite = charge.zero_like();
        finite.add_scaled(0.2, &charge).unwrap();
        let along_x =
            RegionalDensity::new(charge.clone(), [finite.clone(), zero.clone(), zero.clone()])
                .unwrap();
        let along_z = RegionalDensity::new(charge, [zero.clone(), zero, finite]).unwrap();
        let converted = weinert_density(along_x.charge()).unwrap();
        let spec = ElectrostaticSpec::new(
            WeinertHartreeSpec::electronic(4).unwrap(),
            vec![
                solve_weinert_hartree(&converted, WeinertHartreeSpec::electronic(4).unwrap())
                    .unwrap()
                    .source_charge(),
            ],
        )
        .unwrap();
        let x = evaluate_regional_electrostatics(along_x.charge(), &spec).unwrap();
        let z = evaluate_regional_electrostatics(along_z.charge(), &spec).unwrap();
        assert_eq!(x.potential, z.potential);
        assert_eq!(x.coulomb, z.coulomb);
    }
}

/// One SCF iteration's assembled potential and derived controls.
#[derive(Clone, Debug)]
pub struct ScfPotentialBuild {
    pub potential: crate::RegionalPotential,
    pub electrostatic: RegionalElectrostaticResult,
    pub exchange_correlation: crate::RegionalXcResult,
    pub core_spec: crate::CorePotentialBuildSpec,
    pub energy_terms: crate::ScfEnergyTerms,
}

/// Failure assembling the iteration potential from a density.
#[derive(Debug, Error)]
pub enum ScfPotentialBuildError {
    #[error(transparent)]
    Hartree(#[from] muffintin_coulomb::HartreeError),
    #[error(transparent)]
    Electrostatic(#[from] RegionalElectrostaticError),
    #[error(transparent)]
    Xc(#[from] crate::RegionalXcError),
    #[error(transparent)]
    Regional(#[from] RegionalError),
}

/// Electronic Hartree plus periodic nuclei plus XC on a regional density.
pub fn build_scf_potential(
    density: &crate::RegionalDensity,
    nuclear_charges: &[f64],
    exchange_correlation: crate::ScfExchangeCorrelation,
) -> Result<ScfPotentialBuild, ScfPotentialBuildError> {
    let electrostatic = evaluate_regional_electrostatics(
        density.charge(),
        &ElectrostaticSpec::new(
            muffintin_coulomb::WeinertHartreeSpec::electronic(4)?,
            nuclear_charges.to_vec(),
        )?,
    )?;
    let output_l_max = std::iter::once(density.charge())
        .chain(density.magnetization())
        .flat_map(RegionalScalarField::muffin_tins)
        .flat_map(|field| field.field().channels().map(|(channel, _)| channel.l))
        .max()
        .unwrap_or(0);
    let xc_field_spec = crate::xc_spec_for_density(
        density,
        output_l_max,
        exchange_correlation.noncollinear_route,
    );
    let exchange_correlation_result = crate::evaluate_regional_xc(
        exchange_correlation.functional,
        density,
        xc_field_spec,
    )?;
    let mut scalar = electrostatic.potential.clone();
    scalar.add_scaled(1.0, exchange_correlation_result.potential.scalar())?;
    let potential = crate::RegionalPotential::new(
        scalar,
        exchange_correlation_result.potential.magnetic().clone(),
    )?;
    Ok(ScfPotentialBuild {
        potential,
        core_spec: crate::CorePotentialBuildSpec {
            continuation: muffintin_sphere::CorePotentialContinuationSpec::default(),
            xc_functional: exchange_correlation.functional,
            xc_noncollinear_route: exchange_correlation.noncollinear_route,
            xc_angular_point_count: xc_field_spec.angular_point_count,
        },
        energy_terms: crate::ScfEnergyTerms {
            madelung: electrostatic.madelung,
            coulomb: electrostatic.coulomb,
            exchange_correlation: exchange_correlation_result.exchange_correlation_energy,
            exchange_correlation_potential: exchange_correlation_result.density_potential_integral,
        },
        electrostatic,
        exchange_correlation: exchange_correlation_result,
    })
}
