//! Basis-neutral full-potential Hartree solve with Weinert pseudocharges.

use crate::CoulombError;
use crate::math::{i_pow, parity, plane_wave_phase};
use crate::moments::{multipole_moment, spherical_bessel_moment};
use crate::primitive::radial_primitive;
use muffintin_core::{
    ExponentialMesh, FourierFieldError, FourierLayout, Hartree, HermitianFourierField,
    InterstitialGeometry, MeshError, StepFunctionError, complex_spherical_harmonics, lm_count,
    lm_index, spherical_bessel_j,
};
use num_complex::Complex64;
use std::f64::consts::{PI, TAU};
use thiserror::Error;

const REALITY_TOLERANCE: f64 = 1.0e-10;
const GEOMETRY_TOLERANCE: f64 = 1.0e-10;
const MAX_ANGULAR_CUTOFF: u32 = 32;
const MAX_PSEUDOCHARGE_ORDER: u32 = 32;

/// Complex potential coefficient with both components explicitly in Hartree.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ComplexHartree {
    real: Hartree,
    imaginary: Hartree,
}

impl ComplexHartree {
    /// Construct an explicitly unit-labelled complex potential coefficient.
    pub const fn new(real: Hartree, imaginary: Hartree) -> Self {
        Self { real, imaginary }
    }

    /// Real component in Hartree.
    pub const fn real(self) -> Hartree {
        self.real
    }

    /// Imaginary component in Hartree.
    pub const fn imaginary(self) -> Hartree {
        self.imaginary
    }

    /// Convert to a raw complex number for numerical consumers that already
    /// carry the Hartree convention at their boundary.
    pub const fn as_complex(self) -> Complex64 {
        Complex64::new(self.real.get(), self.imaginary.get())
    }

    fn from_raw(value: Complex64) -> Self {
        Self::new(Hartree(value.re), Hartree(value.im))
    }
}

/// Physical charge density in one muffin-tin sphere.
///
/// The dense storage is `lm_index(l,m) * mesh.len() + radial_index` and
/// represents
/// $\rho(r)=\sum_{lm}\rho_{lm}(r)Y_{lm}(\hat r)$. Samples are physical charge
/// per cubic Bohr, not SPEX `basm = r rho`. Reality is enforced as
/// $\rho_{l,-m}=(-1)^m\rho_{lm}^*$. In particular, a spherical scalar density
/// is $\rho_{00}(r)/\sqrt{4\pi}$.
#[derive(Clone, Debug, PartialEq)]
pub struct MuffinTinChargeDensity {
    mesh: ExponentialMesh,
    l_max: u32,
    coefficients: Vec<Complex64>,
}

impl MuffinTinChargeDensity {
    /// Construct a dense physical multipole density on an exact radial mesh.
    pub fn new(
        mesh: ExponentialMesh,
        l_max: u32,
        mut coefficients: Vec<Complex64>,
    ) -> Result<Self, HartreeError> {
        if l_max > MAX_ANGULAR_CUTOFF {
            return Err(HartreeError::AngularCutoffTooLarge(l_max));
        }
        let expected = lm_count(l_max).checked_mul(mesh.len()).ok_or(
            HartreeError::MuffinTinCoefficientCount {
                expected: usize::MAX,
                actual: coefficients.len(),
            },
        )?;
        if coefficients.len() != expected {
            return Err(HartreeError::MuffinTinCoefficientCount {
                expected,
                actual: coefficients.len(),
            });
        }
        for (index, value) in coefficients.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(HartreeError::NonFiniteMuffinTinDensity { index });
            }
        }
        canonicalize_muffin_tin_reality(&mut coefficients, l_max, mesh.len())?;
        Ok(Self {
            mesh,
            l_max,
            coefficients,
        })
    }

    /// Exact radial mesh of this sphere.
    pub const fn mesh(&self) -> &ExponentialMesh {
        &self.mesh
    }

    /// Maximum angular momentum represented densely by this sphere.
    pub const fn l_max(&self) -> u32 {
        self.l_max
    }

    /// Physical radial samples for one normalized complex-harmonic channel.
    pub fn channel(&self, l: u32, m: i32) -> Option<&[Complex64]> {
        let lm = lm_index(l, m).ok()?;
        if l > self.l_max {
            return None;
        }
        let start = lm * self.mesh.len();
        Some(&self.coefficients[start..start + self.mesh.len()])
    }

    fn channel_unchecked(&self, l: u32, m: i32) -> &[Complex64] {
        let lm = lm_index(l, m).expect("validated angular loop");
        let start = lm * self.mesh.len();
        &self.coefficients[start..start + self.mesh.len()]
    }
}

/// Full regional charge density consumed by the Weinert Hartree solver.
///
/// Interstitial coefficients obey
/// $\rho^I(r)=\sum_G\rho^I_G\exp(iG\cdot r)$ and therefore have units of
/// charge per cubic Bohr. The Fourier layout is also the finite pseudocharge
/// and raw-potential layout; it must contain `G=0` and every conjugate partner.
#[derive(Clone, Debug, PartialEq)]
pub struct WeinertChargeDensity {
    geometry: InterstitialGeometry,
    muffin_tins: Vec<MuffinTinChargeDensity>,
    interstitial: HermitianFourierField,
}

impl WeinertChargeDensity {
    /// Bind regional density fields to one exact muffin-tin geometry.
    pub fn new(
        geometry: InterstitialGeometry,
        muffin_tins: Vec<MuffinTinChargeDensity>,
        interstitial: HermitianFourierField,
    ) -> Result<Self, HartreeError> {
        if muffin_tins.len() != geometry.spheres().len() {
            return Err(HartreeError::MuffinTinCount {
                expected: geometry.spheres().len(),
                actual: muffin_tins.len(),
            });
        }
        if interstitial.layout().index([0; 3]).is_none() {
            return Err(HartreeError::MissingZeroVector);
        }
        validate_reciprocal_volume(&geometry, interstitial.layout())?;
        for (site, (density, sphere)) in muffin_tins.iter().zip(geometry.spheres()).enumerate() {
            let mesh_radius = density.mesh.last().get();
            let sphere_radius = sphere.radius.get();
            if (mesh_radius - sphere_radius).abs() > GEOMETRY_TOLERANCE * sphere_radius.max(1.0) {
                return Err(HartreeError::MuffinTinRadius {
                    site,
                    mesh: mesh_radius,
                    geometry: sphere_radius,
                });
            }
        }
        Ok(Self {
            geometry,
            muffin_tins,
            interstitial,
        })
    }

    /// Validated cell volume and muffin-tin spheres.
    pub const fn geometry(&self) -> &InterstitialGeometry {
        &self.geometry
    }

    /// Physical muffin-tin multipole densities in geometry site order.
    pub fn muffin_tins(&self) -> &[MuffinTinChargeDensity] {
        &self.muffin_tins
    }

    /// Hermitian interstitial Fourier density and its exact reciprocal layout.
    pub const fn interstitial(&self) -> &HermitianFourierField {
        &self.interstitial
    }
}

/// Treatment of the periodic source's constant-charge mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PeriodicChargeTreatment {
    /// The regional source is a complete physical charge distribution and
    /// must be neutral within the stated absolute charge tolerance.
    RequireNeutral { tolerance: f64 },
    /// The source is a positive electronic number density. Its nonzero cell
    /// charge is cancelled by a uniform background in both the interstitial
    /// and muffin-tin Poisson equations. This returns the electronic Hartree potential;
    /// the attractive nuclear external potential is deliberately separate and
    /// must be added by the DFT Hamiltonian.
    ElectronicWithUniformBackground,
}

/// Accuracy and constant-charge contract for the Weinert pseudocharge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeinertHartreeSpec {
    pseudocharge_order: u32,
    charge_treatment: PeriodicChargeTreatment,
}

impl WeinertHartreeSpec {
    /// Construct the strict periodic-total-charge specification.
    ///
    /// `pseudocharge_order` is Weinert's polynomial $N$. The neutrality
    /// tolerance is an absolute charge per cell in elementary-charge units.
    pub fn neutral(
        pseudocharge_order: u32,
        neutrality_tolerance: f64,
    ) -> Result<Self, HartreeError> {
        validate_pseudocharge_order(pseudocharge_order)?;
        if !neutrality_tolerance.is_finite() || neutrality_tolerance < 0.0 {
            return Err(HartreeError::InvalidNeutralityTolerance(
                neutrality_tolerance,
            ));
        }
        Ok(Self {
            pseudocharge_order,
            charge_treatment: PeriodicChargeTreatment::RequireNeutral {
                tolerance: neutrality_tolerance,
            },
        })
    }

    /// Construct an electronic-Hartree specification.
    ///
    /// Positive electron number density need not be neutral. A uniform
    /// background removes its constant source mode, the physical regional
    /// potential has zero cell mean, and nuclear attraction remains a separate external
    /// potential owned by the DFT Hamiltonian.
    pub fn electronic(pseudocharge_order: u32) -> Result<Self, HartreeError> {
        validate_pseudocharge_order(pseudocharge_order)?;
        Ok(Self {
            pseudocharge_order,
            charge_treatment: PeriodicChargeTreatment::ElectronicWithUniformBackground,
        })
    }

    /// Weinert polynomial order $N$.
    pub const fn pseudocharge_order(self) -> u32 {
        self.pseudocharge_order
    }

    /// Explicit treatment of the periodic constant-charge mode.
    pub const fn charge_treatment(self) -> PeriodicChargeTreatment {
        self.charge_treatment
    }
}

impl Default for WeinertHartreeSpec {
    fn default() -> Self {
        Self {
            pseudocharge_order: 4,
            charge_treatment: PeriodicChargeTreatment::RequireNeutral { tolerance: 1.0e-10 },
        }
    }
}

/// Gauge applied to the raw periodic Hartree potential.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HartreeGauge {
    /// The physical potential, integrated over the muffin tins and the
    /// step-function-restricted interstitial, has zero cell mean. The Fourier
    /// continuation's `G=0` coefficient need not vanish.
    ZeroCellMean,
}

/// Angular radial Hartree potential for one muffin-tin sphere.
#[derive(Clone, Debug, PartialEq)]
pub struct MuffinTinHartreePotential {
    mesh: ExponentialMesh,
    l_max: u32,
    coefficients: Vec<ComplexHartree>,
}

impl MuffinTinHartreePotential {
    /// Exact radial mesh inherited from the physical density.
    pub const fn mesh(&self) -> &ExponentialMesh {
        &self.mesh
    }

    /// Maximum represented angular momentum.
    pub const fn l_max(&self) -> u32 {
        self.l_max
    }

    /// Normalized-harmonic $V_{lm}(r)$ samples in Hartree, not `r V`.
    /// A spherical radial solver consumes $V_{00}(r)/\sqrt{4\pi}$.
    pub fn channel(&self, l: u32, m: i32) -> Option<&[ComplexHartree]> {
        let lm = lm_index(l, m).ok()?;
        if l > self.l_max {
            return None;
        }
        let start = lm * self.mesh.len();
        Some(&self.coefficients[start..start + self.mesh.len()])
    }
}

/// Raw, unmasked interstitial Fourier Hartree potential.
#[derive(Clone, Debug, PartialEq)]
pub struct InterstitialHartreePotential {
    // Raw values in this core container are numerically Hartree. Keeping the
    // field private prevents an unlabelled potential from escaping the API.
    field: HermitianFourierField,
}

impl InterstitialHartreePotential {
    /// Exact reciprocal layout inherited from the density.
    pub const fn layout(&self) -> &FourierLayout {
        self.field.layout()
    }

    /// One raw Fourier coefficient in Hartree.
    pub fn coefficient(&self, reciprocal_index: [i32; 3]) -> Option<ComplexHartree> {
        self.field
            .coefficient(reciprocal_index)
            .map(ComplexHartree::from_raw)
    }

    /// Raw coefficients in exact layout order, with explicit Hartree units.
    pub fn coefficients(&self) -> impl Iterator<Item = ComplexHartree> + '_ {
        self.field
            .coefficients()
            .iter()
            .copied()
            .map(ComplexHartree::from_raw)
    }
}

/// Raw full-potential solution of the periodic Poisson equation.
///
/// This is neither an auxiliary-basis Coulomb matrix nor a Gamma head nor a
/// step-function-masked LAPW potential. Muffin-tin and interstitial values are
/// two representations of one boundary-continuous potential.
#[derive(Clone, Debug, PartialEq)]
pub struct RawHartreePotential {
    muffin_tins: Vec<MuffinTinHartreePotential>,
    interstitial: InterstitialHartreePotential,
    gauge: HartreeGauge,
    charge_treatment: PeriodicChargeTreatment,
    source_charge: f64,
    neutralizing_background_density: f64,
}

impl RawHartreePotential {
    /// Muffin-tin angular radial potentials in geometry site order.
    pub fn muffin_tins(&self) -> &[MuffinTinHartreePotential] {
        &self.muffin_tins
    }

    /// Raw unmasked interstitial Fourier potential.
    pub const fn interstitial(&self) -> &InterstitialHartreePotential {
        &self.interstitial
    }

    /// Explicit energy-zero convention.
    pub const fn gauge(&self) -> HartreeGauge {
        self.gauge
    }

    /// Treatment used for the periodic constant-charge mode.
    pub const fn charge_treatment(&self) -> PeriodicChargeTreatment {
        self.charge_treatment
    }

    /// Integrated regional source charge before any uniform compensation.
    pub const fn source_charge(&self) -> f64 {
        self.source_charge
    }

    /// Uniform compensating charge per cubic Bohr.
    ///
    /// This is exactly zero in strict-neutral mode and `-source_charge/volume`
    /// in electronic mode.
    pub const fn neutralizing_background_density(&self) -> f64 {
        self.neutralizing_background_density
    }

    /// Add a separately constructed periodic nuclear external potential.
    pub fn add_nuclear_external(
        &self,
        nuclear: &RawNuclearPotential,
    ) -> Result<RawElectrostaticPotential, HartreeError> {
        add_regional_potentials(self, nuclear)
    }
}

/// Periodic attractive potential of point nuclei in the Hartree gauge.
///
/// Interstitial coefficients are the raw finite-layout point-source Fourier
/// potential. Inside each owning sphere the central `-Z/r` singularity is
/// restored analytically and the remaining harmonic solution is matched to
/// the same Fourier potential on the sphere boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct RawNuclearPotential {
    muffin_tins: Vec<MuffinTinHartreePotential>,
    interstitial: InterstitialHartreePotential,
    gauge: HartreeGauge,
    nuclear_charges: Vec<f64>,
    source_charge: f64,
    neutralizing_background_density: f64,
}

impl RawNuclearPotential {
    /// Muffin-tin angular radial external potentials in geometry site order.
    pub fn muffin_tins(&self) -> &[MuffinTinHartreePotential] {
        &self.muffin_tins
    }

    /// Raw unmasked interstitial Fourier external potential.
    pub const fn interstitial(&self) -> &InterstitialHartreePotential {
        &self.interstitial
    }

    /// The same zero physical cell-mean gauge used by the electronic Hartree solve.
    pub const fn gauge(&self) -> HartreeGauge {
        self.gauge
    }

    /// Positive nuclear charges `Z` in geometry site order.
    pub fn nuclear_charges(&self) -> &[f64] {
        &self.nuclear_charges
    }

    /// Integrated attractive point-nuclear source charge, `-sum(Z)`.
    pub const fn source_charge(&self) -> f64 {
        self.source_charge
    }

    /// Uniform positive background implicit in omitting the nuclear `G=0`
    /// coefficient.
    pub const fn neutralizing_background_density(&self) -> f64 {
        self.neutralizing_background_density
    }
}

/// Sum of electronic Hartree and periodic nuclear external potentials.
#[derive(Clone, Debug, PartialEq)]
pub struct RawElectrostaticPotential {
    muffin_tins: Vec<MuffinTinHartreePotential>,
    interstitial: InterstitialHartreePotential,
    gauge: HartreeGauge,
    source_charge: f64,
    neutralizing_background_density: f64,
}

impl RawElectrostaticPotential {
    /// Summed muffin-tin angular radial potentials.
    pub fn muffin_tins(&self) -> &[MuffinTinHartreePotential] {
        &self.muffin_tins
    }

    /// Summed raw unmasked interstitial Fourier potential.
    pub const fn interstitial(&self) -> &InterstitialHartreePotential {
        &self.interstitial
    }

    /// Common zero physical cell-mean gauge.
    pub const fn gauge(&self) -> HartreeGauge {
        self.gauge
    }

    /// Electron-plus-nuclear source charge before constant-mode compensation.
    pub const fn source_charge(&self) -> f64 {
        self.source_charge
    }

    /// Sum of the electronic and nuclear constant-mode backgrounds.
    pub const fn neutralizing_background_density(&self) -> f64 {
        self.neutralizing_background_density
    }
}

/// Invalid regional density or failed Weinert Hartree solve.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum HartreeError {
    #[error(transparent)]
    Fourier(#[from] FourierFieldError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    StepFunction(#[from] StepFunctionError),
    #[error(transparent)]
    Coulomb(#[from] CoulombError),
    #[error("muffin-tin angular cutoff {0} exceeds the supported maximum 32")]
    AngularCutoffTooLarge(u32),
    #[error("Weinert pseudocharge order {0} exceeds the supported maximum 32")]
    PseudochargeOrderTooLarge(u32),
    #[error("neutrality tolerance must be finite and nonnegative, got {0}")]
    InvalidNeutralityTolerance(f64),
    #[error("muffin-tin density has {actual} coefficients, expected {expected}")]
    MuffinTinCoefficientCount { expected: usize, actual: usize },
    #[error("muffin-tin density coefficient {index} is not finite")]
    NonFiniteMuffinTinDensity { index: usize },
    #[error(
        "muffin-tin density violates physical reality at L={l}, M={m}, radial index {radial_index}"
    )]
    NonRealMuffinTinDensity { l: u32, m: i32, radial_index: usize },
    #[error("regional density has {actual} muffin tins, expected {expected}")]
    MuffinTinCount { expected: usize, actual: usize },
    #[error("site {site} radial mesh ends at {mesh}, but the geometry radius is {geometry}")]
    MuffinTinRadius {
        site: usize,
        mesh: f64,
        geometry: f64,
    },
    #[error("interstitial Fourier layout must contain G=0")]
    MissingZeroVector,
    #[error("reciprocal-cell volume {reciprocal} does not match geometry volume {geometry}")]
    ReciprocalVolumeMismatch { reciprocal: f64, geometry: f64 },
    #[error("periodic Hartree source is not neutral: charge {charge}, tolerance {tolerance}")]
    NonNeutral { charge: f64, tolerance: f64 },
    #[error("nuclear potential has {actual} site charges, expected {expected}")]
    NuclearChargeCount { expected: usize, actual: usize },
    #[error("nuclear charge Z[{site}] must be finite and nonnegative, got {charge}")]
    InvalidNuclearCharge { site: usize, charge: f64 },
    #[error("regional potentials have different Fourier or muffin-tin layouts")]
    PotentialLayoutMismatch,
    #[error("integrated cell charge is not real: imaginary residual {imaginary}")]
    NonRealTotalCharge { imaginary: f64 },
    #[error("{stage} produced a non-finite coefficient")]
    NonFiniteCoefficient { stage: &'static str },
    #[error("{stage} violates Hermiticity by {residual}, tolerance {tolerance}")]
    NonHermitianOutput {
        stage: &'static str,
        residual: f64,
        tolerance: f64,
    },
}

/// Solve periodic Poisson for a physical regional charge density by Weinert's
/// pseudocharge construction.
///
/// Strict mode rejects a nonneutral total source. Electronic mode instead adds
/// a uniform compensating background and returns only the repulsive electronic
/// Hartree potential; it does not manufacture a nuclear external potential.
/// In both modes the physical regional potential has zero cell mean. The
/// Fourier continuation's constant is fixed after the MT reconstruction, not
/// independently of its density-dependent cell integral. Finite `G` coefficients are
/// $4\pi\tilde\rho_G/G^2$ in Hartree, and each muffin-tin Green-function
/// solution is matched to that Fourier potential on its sphere boundary.
pub fn solve_weinert_hartree(
    density: &WeinertChargeDensity,
    spec: WeinertHartreeSpec,
) -> Result<RawHartreePotential, HartreeError> {
    validate_pseudocharge_order(spec.pseudocharge_order)?;
    let multipoles = muffin_tin_multipoles(density)?;
    let total_charge = integrated_charge(density, &multipoles)?;
    let background_density = match spec.charge_treatment {
        PeriodicChargeTreatment::RequireNeutral { tolerance } => {
            if total_charge.abs() > tolerance {
                return Err(HartreeError::NonNeutral {
                    charge: total_charge,
                    tolerance,
                });
            }
            0.0
        }
        PeriodicChargeTreatment::ElectronicWithUniformBackground => {
            -total_charge / density.geometry.cell_volume().get()
        }
    };
    let continuation = continuation_multipoles(density)?;
    let mut pseudo_density = pseudocharge_fourier(density, spec, &multipoles, &continuation)?;
    let zero = density
        .interstitial
        .layout()
        .index([0; 3])
        .expect("regional density requires G=0");
    pseudo_density[zero] += background_density;
    // The compensated constant source has no periodic Poisson inverse. Set it
    // exactly to zero after recording the physical source and compensation.
    pseudo_density[zero] = Complex64::default();
    let mut potential_coefficients =
        hartree_fourier(density.interstitial.layout(), &pseudo_density)?;
    let mut muffin_tins = muffin_tin_potentials(density, &potential_coefficients)?;
    complete_periodic_potential(
        density,
        &mut muffin_tins,
        &mut potential_coefficients,
        background_density,
    )?;
    let potential_field = HermitianFourierField::new(
        density.interstitial.layout().clone(),
        potential_coefficients.clone(),
    )?;
    Ok(RawHartreePotential {
        muffin_tins,
        interstitial: InterstitialHartreePotential {
            field: potential_field,
        },
        gauge: HartreeGauge::ZeroCellMean,
        charge_treatment: spec.charge_treatment,
        source_charge: total_charge,
        neutralizing_background_density: background_density,
    })
}

/// Construct the periodic external potential of point nuclei.
///
/// `nuclear_charges` are positive atomic charges $Z_a$ in the site order of
/// `template.geometry()`. The interstitial field uses each nucleus's normalized
/// Weinert pseudocharge with the same polynomial order as the electronic
/// Hartree solve. The exact point singularity is restored in its MT sphere.
/// The positive uniform background is retained inside MT spheres, and the
/// reconstructed physical potential has zero cell mean.
/// The template supplies only basis-neutral geometry,
/// radial meshes/angular cutoffs, and the exact Hermitian Fourier layout; its
/// density values are not used.
pub fn solve_periodic_nuclear_potential(
    template: &WeinertChargeDensity,
    nuclear_charges: &[f64],
    spec: WeinertHartreeSpec,
) -> Result<RawNuclearPotential, HartreeError> {
    if nuclear_charges.len() != template.geometry.spheres().len() {
        return Err(HartreeError::NuclearChargeCount {
            expected: template.geometry.spheres().len(),
            actual: nuclear_charges.len(),
        });
    }
    for (site, &charge) in nuclear_charges.iter().enumerate() {
        if !charge.is_finite() || charge < 0.0 {
            return Err(HartreeError::InvalidNuclearCharge { site, charge });
        }
    }
    let mut coefficients = nuclear_fourier_coefficients(template, nuclear_charges, spec)?;
    let source_charge = -nuclear_charges.iter().sum::<f64>();
    let background_density = -source_charge / template.geometry.cell_volume().get();
    let mut muffin_tins = nuclear_muffin_tin_potentials(template, nuclear_charges, &coefficients)?;
    complete_periodic_potential(
        template,
        &mut muffin_tins,
        &mut coefficients,
        background_density,
    )?;
    let field =
        HermitianFourierField::new(template.interstitial.layout().clone(), coefficients.clone())?;
    Ok(RawNuclearPotential {
        muffin_tins,
        interstitial: InterstitialHartreePotential { field },
        gauge: HartreeGauge::ZeroCellMean,
        nuclear_charges: nuclear_charges.to_vec(),
        source_charge,
        neutralizing_background_density: background_density,
    })
}

/// Complete the background Poisson solution inside each sphere, then fix one
/// density-independent gauge for the physical (not pseudocharge) potential.
fn complete_periodic_potential(
    template: &WeinertChargeDensity,
    muffin_tins: &mut [MuffinTinHartreePotential],
    fourier: &mut [Complex64],
    background_density: f64,
) -> Result<(), HartreeError> {
    let sqrt_four_pi = (4.0 * PI).sqrt();
    let volume = template.geometry.cell_volume().get();
    let mut integral = 0.0;
    for site in muffin_tins.iter_mut() {
        let radius_squared = site.mesh.last().get().powi(2);
        let mut samples = Vec::with_capacity(site.mesh.len());
        for (value, radius) in site.coefficients[..site.mesh.len()]
            .iter_mut()
            .zip(site.mesh.radii())
        {
            let r_squared = radius.get().powi(2);
            value.real += Hartree(
                sqrt_four_pi * 2.0 * PI / 3.0 * background_density * (radius_squared - r_squared),
            );
            samples.push(sqrt_four_pi * value.real.get() * r_squared);
        }
        integral += site.mesh.integrate(&samples)?;
    }
    for (g, value) in template
        .interstitial
        .layout()
        .vectors()
        .iter()
        .zip(fourier.iter())
    {
        integral += volume * (template.geometry.coefficient_for_g(g)?.conj() * value).re;
    }
    let mean = integral / volume;
    let zero = template
        .interstitial
        .layout()
        .index([0; 3])
        .expect("regional density requires G=0");
    fourier[zero] -= Complex64::new(mean, 0.0);
    for site in muffin_tins {
        for value in &mut site.coefficients[..site.mesh.len()] {
            value.real -= Hartree(sqrt_four_pi * mean);
        }
    }
    Ok(())
}

fn nuclear_fourier_coefficients(
    template: &WeinertChargeDensity,
    nuclear_charges: &[f64],
    spec: WeinertHartreeSpec,
) -> Result<Vec<Complex64>, HartreeError> {
    let volume = template.geometry.cell_volume().get();
    let order = spec.pseudocharge_order;
    let normalization = pseudocharge_normalization(0, order)?;
    let mut coefficients = Vec::with_capacity(template.interstitial.layout().len());
    for g in template.interstitial.layout().vectors() {
        let value = if g.index == [0; 3] {
            Complex64::default()
        } else {
            let structure_factor = template
                .geometry
                .spheres()
                .iter()
                .zip(nuclear_charges)
                .map(|(sphere, charge)| {
                    charge
                        * normalization
                        * pseudocharge_bessel_ratio(0, order, g.norm.get() * sphere.radius.get())
                        * plane_wave_phase(g.cartesian, sphere.center).conj()
                })
                .sum::<Complex64>();
            -4.0 * PI / (volume * g.norm.get().powi(2)) * structure_factor
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(HartreeError::NonFiniteCoefficient {
                stage: "periodic nuclear Fourier potential",
            });
        }
        coefficients.push(value);
    }
    canonicalize_fourier_reality(
        template.interstitial.layout(),
        &mut coefficients,
        "periodic nuclear Fourier potential",
    )?;
    Ok(coefficients)
}

fn nuclear_muffin_tin_potentials(
    template: &WeinertChargeDensity,
    nuclear_charges: &[f64],
    fourier_potential: &[Complex64],
) -> Result<Vec<MuffinTinHartreePotential>, HartreeError> {
    let mut output = Vec::with_capacity(template.muffin_tins.len());
    for ((site, sphere), &charge) in template
        .muffin_tins
        .iter()
        .zip(template.geometry.spheres())
        .zip(nuclear_charges)
    {
        let n_radial = site.mesh.len();
        let surface = surface_potential(
            site.l_max,
            sphere.center,
            sphere.radius.get(),
            template.interstitial.layout(),
            fourier_potential,
        );
        let mesh_radius = site.mesh.last().get();
        let mut raw = vec![Complex64::default(); lm_count(site.l_max) * n_radial];
        for l in 0..=site.l_max {
            for m in -(l as i32)..=l as i32 {
                let lm = lm_index(l, m).expect("angular loop is valid");
                let start = lm * n_radial;
                for (radial_index, radius) in site.mesh.radii().iter().enumerate() {
                    raw[start + radial_index] =
                        surface[lm] * (radius.get() / mesh_radius).powi(l as i32);
                }
            }
        }
        let monopole = lm_index(0, 0).expect("monopole is valid") * n_radial;
        let point_coefficient = -(4.0 * PI).sqrt() * charge;
        for (radial_index, radius) in site.mesh.radii().iter().enumerate() {
            raw[monopole + radial_index] +=
                point_coefficient * (1.0 / radius.get() - 1.0 / mesh_radius);
        }
        canonicalize_potential_reality(&mut raw, site.l_max, n_radial)?;
        output.push(MuffinTinHartreePotential {
            mesh: site.mesh.clone(),
            l_max: site.l_max,
            coefficients: raw.into_iter().map(ComplexHartree::from_raw).collect(),
        });
    }
    Ok(output)
}

fn add_regional_potentials(
    hartree: &RawHartreePotential,
    nuclear: &RawNuclearPotential,
) -> Result<RawElectrostaticPotential, HartreeError> {
    if hartree.gauge != nuclear.gauge
        || hartree.interstitial.layout() != nuclear.interstitial.layout()
        || hartree.muffin_tins.len() != nuclear.muffin_tins.len()
    {
        return Err(HartreeError::PotentialLayoutMismatch);
    }
    let fourier = hartree
        .interstitial
        .field
        .coefficients()
        .iter()
        .zip(nuclear.interstitial.field.coefficients())
        .map(|(&electronic, &external)| electronic + external)
        .collect::<Vec<_>>();
    let interstitial = InterstitialHartreePotential {
        field: HermitianFourierField::new(hartree.interstitial.layout().clone(), fourier)?,
    };
    let mut muffin_tins = Vec::with_capacity(hartree.muffin_tins.len());
    for (electronic, external) in hartree.muffin_tins.iter().zip(&nuclear.muffin_tins) {
        if electronic.mesh != external.mesh
            || electronic.l_max != external.l_max
            || electronic.coefficients.len() != external.coefficients.len()
        {
            return Err(HartreeError::PotentialLayoutMismatch);
        }
        muffin_tins.push(MuffinTinHartreePotential {
            mesh: electronic.mesh.clone(),
            l_max: electronic.l_max,
            coefficients: electronic
                .coefficients
                .iter()
                .zip(&external.coefficients)
                .map(|(&left, &right)| {
                    ComplexHartree::from_raw(left.as_complex() + right.as_complex())
                })
                .collect(),
        });
    }
    Ok(RawElectrostaticPotential {
        muffin_tins,
        interstitial,
        gauge: hartree.gauge,
        source_charge: hartree.source_charge + nuclear.source_charge,
        neutralizing_background_density: hartree.neutralizing_background_density
            + nuclear.neutralizing_background_density,
    })
}

fn validate_pseudocharge_order(pseudocharge_order: u32) -> Result<(), HartreeError> {
    if pseudocharge_order > MAX_PSEUDOCHARGE_ORDER {
        Err(HartreeError::PseudochargeOrderTooLarge(pseudocharge_order))
    } else {
        Ok(())
    }
}

fn validate_reciprocal_volume(
    geometry: &InterstitialGeometry,
    layout: &FourierLayout,
) -> Result<(), HartreeError> {
    let basis = layout
        .reciprocal()
        .basis()
        .map(|row| row.map(|value| value.get()));
    let determinant = dot(basis[0], cross(basis[1], basis[2])).abs();
    let reciprocal_volume = TAU.powi(3) / determinant;
    let geometry_volume = geometry.cell_volume().get();
    if (reciprocal_volume - geometry_volume).abs()
        > GEOMETRY_TOLERANCE * reciprocal_volume.max(geometry_volume)
    {
        return Err(HartreeError::ReciprocalVolumeMismatch {
            reciprocal: reciprocal_volume,
            geometry: geometry_volume,
        });
    }
    Ok(())
}

fn canonicalize_muffin_tin_reality(
    coefficients: &mut [Complex64],
    l_max: u32,
    n_radial: usize,
) -> Result<(), HartreeError> {
    for l in 0..=l_max {
        let zero = lm_index(l, 0).expect("m=0 is valid") * n_radial;
        for radial_index in 0..n_radial {
            let value = coefficients[zero + radial_index];
            let tolerance = REALITY_TOLERANCE * value.norm().max(1.0);
            if value.im.abs() > tolerance {
                return Err(HartreeError::NonRealMuffinTinDensity {
                    l,
                    m: 0,
                    radial_index,
                });
            }
            coefficients[zero + radial_index].im = 0.0;
        }
        for m in 1..=l as i32 {
            let positive = lm_index(l, m).expect("angular loop is valid") * n_radial;
            let negative = lm_index(l, -m).expect("angular loop is valid") * n_radial;
            let sign = parity(m);
            for radial_index in 0..n_radial {
                let pos = coefficients[positive + radial_index];
                let neg = coefficients[negative + radial_index];
                let expected = sign * pos.conj();
                let tolerance = REALITY_TOLERANCE * pos.norm().max(neg.norm()).max(1.0);
                if (neg - expected).norm() > tolerance {
                    return Err(HartreeError::NonRealMuffinTinDensity { l, m, radial_index });
                }
                let average = (pos + sign * neg.conj()) * 0.5;
                coefficients[positive + radial_index] = average;
                coefficients[negative + radial_index] = sign * average.conj();
            }
        }
    }
    Ok(())
}

fn muffin_tin_multipoles(
    density: &WeinertChargeDensity,
) -> Result<Vec<Vec<Complex64>>, HartreeError> {
    density
        .muffin_tins
        .iter()
        .map(|site| {
            let mut values = vec![Complex64::default(); lm_count(site.l_max)];
            for l in 0..=site.l_max {
                for m in -(l as i32)..=l as i32 {
                    let lm = lm_index(l, m).expect("angular loop is valid");
                    values[lm] = physical_multipole(l, &site.mesh, site.channel_unchecked(l, m))?;
                }
            }
            Ok(values)
        })
        .collect()
}

fn physical_multipole(
    l: u32,
    mesh: &ExponentialMesh,
    density: &[Complex64],
) -> Result<Complex64, HartreeError> {
    let basm_real = mesh
        .radii()
        .iter()
        .zip(density)
        .map(|(radius, value)| radius.get() * value.re)
        .collect::<Vec<_>>();
    let basm_imaginary = mesh
        .radii()
        .iter()
        .zip(density)
        .map(|(radius, value)| radius.get() * value.im)
        .collect::<Vec<_>>();
    Ok(Complex64::new(
        multipole_moment(l, mesh, &basm_real)?,
        multipole_moment(l, mesh, &basm_imaginary)?,
    ))
}

fn integrated_charge(
    density: &WeinertChargeDensity,
    multipoles: &[Vec<Complex64>],
) -> Result<f64, HartreeError> {
    let mut charge = multipoles
        .iter()
        .map(|site| (4.0 * PI).sqrt() * site[0])
        .sum::<Complex64>();
    let mut interstitial = Complex64::default();
    for (g, coefficient) in density.interstitial.iter() {
        // Integral rho_I Theta_I = Omega sum_G rho_G Theta_{-G}.
        let theta_minus_g = density.geometry.coefficient(g.cartesian)?.conj();
        interstitial += *coefficient * theta_minus_g;
    }
    charge += density.geometry.cell_volume().get() * interstitial;
    let tolerance = REALITY_TOLERANCE * charge.re.abs().max(charge.im.abs()).max(1.0);
    if charge.im.abs() > tolerance {
        return Err(HartreeError::NonRealTotalCharge {
            imaginary: charge.im,
        });
    }
    Ok(charge.re)
}

fn continuation_multipoles(
    density: &WeinertChargeDensity,
) -> Result<Vec<Vec<Complex64>>, HartreeError> {
    let mut result = Vec::with_capacity(density.muffin_tins.len());
    for (site, sphere) in density.muffin_tins.iter().zip(density.geometry.spheres()) {
        let mut values = vec![Complex64::default(); lm_count(site.l_max)];
        for (g, coefficient) in density.interstitial.iter() {
            let harmonics = complex_spherical_harmonics(
                site.l_max,
                g.cartesian.map(|component| component.get()),
            );
            let phase = plane_wave_phase(g.cartesian, sphere.center);
            for l in 0..=site.l_max {
                let radial = spherical_bessel_moment(l, g.norm.get(), sphere.radius.get());
                let common = 4.0 * PI * i_pow(l) * radial * phase * *coefficient;
                for m in -(l as i32)..=l as i32 {
                    let lm = lm_index(l, m).expect("angular loop is valid");
                    values[lm] += common * harmonics[lm].conj();
                }
            }
        }
        result.push(values);
    }
    Ok(result)
}

fn pseudocharge_fourier(
    density: &WeinertChargeDensity,
    spec: WeinertHartreeSpec,
    muffin_tin: &[Vec<Complex64>],
    continuation: &[Vec<Complex64>],
) -> Result<Vec<Complex64>, HartreeError> {
    let volume = density.geometry.cell_volume().get();
    let order = spec.pseudocharge_order;
    let mut result = density.interstitial.coefficients().to_vec();
    for (g_position, g) in density.interstitial.layout().vectors().iter().enumerate() {
        let mut correction = Complex64::default();
        for (site_index, (site, sphere)) in density
            .muffin_tins
            .iter()
            .zip(density.geometry.spheres())
            .enumerate()
        {
            let harmonics = complex_spherical_harmonics(
                site.l_max,
                g.cartesian.map(|component| component.get()),
            );
            let phase = plane_wave_phase(g.cartesian, sphere.center).conj();
            let x = g.norm.get() * sphere.radius.get();
            for l in 0..=site.l_max {
                let bessel = pseudocharge_bessel_ratio(l, order, x);
                let normalization = pseudocharge_normalization(l, order)?;
                let common = 4.0 * PI / volume * i_pow(l).conj() * phase * bessel * normalization
                    / sphere.radius.get().powi(l as i32);
                for m in -(l as i32)..=l as i32 {
                    let lm = lm_index(l, m).expect("angular loop is valid");
                    let difference = muffin_tin[site_index][lm] - continuation[site_index][lm];
                    correction += common * difference * harmonics[lm];
                }
            }
        }
        result[g_position] += correction;
        if !result[g_position].re.is_finite() || !result[g_position].im.is_finite() {
            return Err(HartreeError::NonFiniteCoefficient {
                stage: "Weinert pseudocharge",
            });
        }
    }
    canonicalize_fourier_reality(
        density.interstitial.layout(),
        &mut result,
        "Weinert pseudocharge",
    )?;
    Ok(result)
}

fn pseudocharge_normalization(l: u32, order: u32) -> Result<f64, HartreeError> {
    // (2l + 2N + 3)!! / (2l + 1)!!.
    let mut value = 1.0;
    for step in 0..=order {
        value *= f64::from(2 * l + 3 + 2 * step);
    }
    if value.is_finite() {
        Ok(value)
    } else {
        Err(HartreeError::NonFiniteCoefficient {
            stage: "Weinert pseudocharge normalization",
        })
    }
}

fn pseudocharge_bessel_ratio(l: u32, order: u32, x: f64) -> f64 {
    let power = order + 1;
    let rank = l + power;
    if x < 0.25 {
        // Direct series for j_rank(x) / x^power avoids the removable G=0
        // singularity and cancellation for small reciprocal vectors.
        let mut leading = x.powi(l as i32);
        for k in 1..=rank {
            leading /= f64::from(2 * k + 1);
        }
        let mut sum = 1.0;
        let mut term = 1.0;
        for k in 1..=512_u32 {
            term *= -x * x / (2.0 * f64::from(k) * f64::from(2 * rank + 2 * k + 1));
            sum += term;
            if term.abs() <= 2.0 * f64::EPSILON * sum.abs().max(1.0) {
                break;
            }
        }
        leading * sum
    } else {
        spherical_bessel_j(rank, x) / x.powi(power as i32)
    }
}

fn hartree_fourier(
    layout: &FourierLayout,
    pseudo_density: &[Complex64],
) -> Result<Vec<Complex64>, HartreeError> {
    let mut potential = Vec::with_capacity(layout.len());
    for (g, density) in layout.vectors().iter().zip(pseudo_density) {
        let value = if g.index == [0; 3] {
            Complex64::default()
        } else {
            4.0 * PI / g.norm.get().powi(2) * *density
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(HartreeError::NonFiniteCoefficient {
                stage: "Hartree Fourier potential",
            });
        }
        potential.push(value);
    }
    canonicalize_fourier_reality(layout, &mut potential, "Hartree Fourier potential")?;
    Ok(potential)
}

fn canonicalize_fourier_reality(
    layout: &FourierLayout,
    coefficients: &mut [Complex64],
    stage: &'static str,
) -> Result<(), HartreeError> {
    for g in layout.vectors() {
        let position = layout
            .index(g.index)
            .expect("layout contains its own vector");
        let opposite_index = g.index.map(|component| -component);
        let opposite = layout
            .index(opposite_index)
            .expect("Hermitian input layout contains every opposite vector");
        if g.index == [0; 3] {
            let tolerance = REALITY_TOLERANCE * coefficients[position].norm().max(1.0);
            if coefficients[position].im.abs() > tolerance {
                return Err(HartreeError::NonHermitianOutput {
                    stage,
                    residual: coefficients[position].im.abs(),
                    tolerance,
                });
            }
            coefficients[position].im = 0.0;
        } else if g.index < opposite_index {
            let left = coefficients[position];
            let right = coefficients[opposite];
            let residual = (right - left.conj()).norm();
            let tolerance = REALITY_TOLERANCE * left.norm().max(right.norm()).max(1.0);
            if residual > tolerance {
                return Err(HartreeError::NonHermitianOutput {
                    stage,
                    residual,
                    tolerance,
                });
            }
            let average = (left + right.conj()) * 0.5;
            coefficients[position] = average;
            coefficients[opposite] = average.conj();
        }
    }
    Ok(())
}

fn muffin_tin_potentials(
    density: &WeinertChargeDensity,
    fourier_potential: &[Complex64],
) -> Result<Vec<MuffinTinHartreePotential>, HartreeError> {
    let mut output = Vec::with_capacity(density.muffin_tins.len());
    for (site, sphere) in density.muffin_tins.iter().zip(density.geometry.spheres()) {
        let n_radial = site.mesh.len();
        let mesh_radius = site.mesh.last().get();
        let mut raw = vec![Complex64::default(); lm_count(site.l_max) * n_radial];
        let surface = surface_potential(
            site.l_max,
            sphere.center,
            sphere.radius.get(),
            density.interstitial.layout(),
            fourier_potential,
        );
        for l in 0..=site.l_max {
            for m in -(l as i32)..=l as i32 {
                let lm = lm_index(l, m).expect("angular loop is valid");
                let isolated =
                    isolated_muffin_tin_potential(l, &site.mesh, site.channel_unchecked(l, m))?;
                // The analytical isolated boundary is the corresponding
                // multipole, but computing it through a second quadrature can
                // differ slightly from `radial_primitive`. Match the actual
                // sampled Poisson solution so the discrete MT/Fourier
                // boundary is continuous to machine precision.
                let isolated_boundary = isolated[n_radial - 1];
                let start = lm * n_radial;
                for (radial_index, radius) in site.mesh.radii().iter().enumerate() {
                    let homogeneous = (surface[lm] - isolated_boundary)
                        * (radius.get() / mesh_radius).powi(l as i32);
                    raw[start + radial_index] = isolated[radial_index] + homogeneous;
                }
            }
        }
        canonicalize_potential_reality(&mut raw, site.l_max, n_radial)?;
        if raw
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(HartreeError::NonFiniteCoefficient {
                stage: "muffin-tin Hartree potential",
            });
        }
        output.push(MuffinTinHartreePotential {
            mesh: site.mesh.clone(),
            l_max: site.l_max,
            coefficients: raw.into_iter().map(ComplexHartree::from_raw).collect(),
        });
    }
    Ok(output)
}

fn surface_potential(
    l_max: u32,
    center: [muffintin_core::Bohr; 3],
    radius: f64,
    layout: &FourierLayout,
    potential: &[Complex64],
) -> Vec<Complex64> {
    let mut surface = vec![Complex64::default(); lm_count(l_max)];
    for (g, coefficient) in layout.vectors().iter().zip(potential) {
        let harmonics =
            complex_spherical_harmonics(l_max, g.cartesian.map(|component| component.get()));
        let phase = plane_wave_phase(g.cartesian, center);
        for l in 0..=l_max {
            let common = 4.0
                * PI
                * i_pow(l)
                * spherical_bessel_j(l, g.norm.get() * radius)
                * phase
                * *coefficient;
            for m in -(l as i32)..=l as i32 {
                let lm = lm_index(l, m).expect("angular loop is valid");
                surface[lm] += common * harmonics[lm].conj();
            }
        }
    }
    surface
}

fn isolated_muffin_tin_potential(
    l: u32,
    mesh: &ExponentialMesh,
    density: &[Complex64],
) -> Result<Vec<Complex64>, HartreeError> {
    // Use exactly the `basm = r rho` and radial primitives of the existing
    // intra-sphere Poisson kernel; only the uncontracted potential is retained.
    let n = mesh.len();
    let radii = mesh.radii();
    let basm = radii
        .iter()
        .zip(density)
        .map(|(radius, value)| *value * radius.get())
        .collect::<Vec<_>>();
    let mut out_real = Vec::with_capacity(n);
    let mut out_imaginary = Vec::with_capacity(n);
    let mut in_real = Vec::with_capacity(n);
    let mut in_imaginary = Vec::with_capacity(n);
    for (radius, value) in radii.iter().zip(&basm) {
        let power = radius.get().powi(l as i32);
        let out = *value * power * radius.get();
        let inward = if l == 0 { *value } else { *value / power };
        out_real.push(out.re);
        out_imaginary.push(out.im);
        in_real.push(inward.re);
        in_imaginary.push(inward.im);
    }
    let primitive_out_real = radial_primitive(mesh, &out_real, false)?;
    let primitive_out_imaginary = radial_primitive(mesh, &out_imaginary, false)?;
    let primitive_in_real = radial_primitive(mesh, &in_real, true)?;
    let primitive_in_imaginary = radial_primitive(mesh, &in_imaginary, true)?;
    let factor = 4.0 * PI / f64::from(2 * l + 1);
    let mut potential = Vec::with_capacity(n);
    for index in 0..n {
        let radius = radii[index].get();
        let power = radius.powi(l as i32);
        let outward =
            Complex64::new(primitive_out_real[index], primitive_out_imaginary[index]) / power;
        let inward = Complex64::new(primitive_in_real[index], primitive_in_imaginary[index])
            * power
            * radius;
        potential.push(factor * (outward + inward) / radius);
    }
    Ok(potential)
}

fn canonicalize_potential_reality(
    coefficients: &mut [Complex64],
    l_max: u32,
    n_radial: usize,
) -> Result<(), HartreeError> {
    for l in 0..=l_max {
        let zero = lm_index(l, 0).expect("m=0 is valid") * n_radial;
        for radial_index in 0..n_radial {
            let value = coefficients[zero + radial_index];
            let tolerance = REALITY_TOLERANCE * value.norm().max(1.0);
            if value.im.abs() > tolerance {
                return Err(HartreeError::NonHermitianOutput {
                    stage: "muffin-tin Hartree potential",
                    residual: value.im.abs(),
                    tolerance,
                });
            }
            coefficients[zero + radial_index].im = 0.0;
        }
        for m in 1..=l as i32 {
            let positive = lm_index(l, m).expect("angular loop is valid") * n_radial;
            let negative = lm_index(l, -m).expect("angular loop is valid") * n_radial;
            let sign = parity(m);
            for radial_index in 0..n_radial {
                let pos = coefficients[positive + radial_index];
                let neg = coefficients[negative + radial_index];
                let residual = (neg - sign * pos.conj()).norm();
                let tolerance = REALITY_TOLERANCE * pos.norm().max(neg.norm()).max(1.0);
                if residual > tolerance {
                    return Err(HartreeError::NonHermitianOutput {
                        stage: "muffin-tin Hartree potential",
                        residual,
                        tolerance,
                    });
                }
                let average = (pos + sign * neg.conj()) * 0.5;
                coefficients[positive + radial_index] = average;
                coefficients[negative + radial_index] = sign * average.conj();
            }
        }
    }
    Ok(())
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}
