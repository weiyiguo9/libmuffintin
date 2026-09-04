//! Regional muffin-tin/interstitial scalar fields and their physical metric.

use muffintin_core::{
    ExponentialMesh, FourierFieldError, FourierLayout, HermitianFourierField, InterstitialGeometry,
    MeshError, StepFunctionError, complex_spherical_harmonics, real_spherical_harmonics,
};
use muffintin_operators::lapw::{InterstitialPauliPotential, InterstitialPotential, LapwError};
use muffintin_sphere::{HarmonicConvention, SphereField, SphereFieldError};
use muffintin_symmetry::CrystalSymmetryTransform;
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet, HashMap, hash_map::Entry};
use std::f64::consts::{PI, TAU};
use thiserror::Error;

const REALITY_TOLERANCE: f64 = 1024.0 * f64::EPSILON;

/// One angularly resolved field on one exact radial mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct MuffinTinField {
    mesh: ExponentialMesh,
    field: SphereField,
}

impl MuffinTinField {
    pub fn new(mesh: ExponentialMesh, field: SphereField) -> Result<Self, RegionalError> {
        if let Some(actual) = field.sample_count()
            && actual != mesh.len()
        {
            return Err(RegionalError::MuffinTinSampleCount {
                expected: mesh.len(),
                actual,
            });
        }
        field.validate_physical_reality(REALITY_TOLERANCE)?;
        Ok(Self { mesh, field })
    }

    pub const fn mesh(&self) -> &ExponentialMesh {
        &self.mesh
    }

    pub const fn field(&self) -> &SphereField {
        &self.field
    }

    pub fn zero_like(&self) -> Self {
        Self {
            mesh: self.mesh.clone(),
            field: self.field.zero_like(),
        }
    }

    pub fn add_scaled(&mut self, scale: f64, other: &Self) -> Result<(), RegionalError> {
        self.require_same_layout(other)?;
        self.field
            .add_scaled(Complex64::new(scale, 0.0), &other.field)?;
        Ok(())
    }

    pub fn difference(&self, other: &Self) -> Result<Self, RegionalError> {
        self.require_same_layout(other)?;
        Ok(Self {
            mesh: self.mesh.clone(),
            field: self.field.difference(&other.field)?,
        })
    }

    fn require_same_layout(&self, other: &Self) -> Result<(), RegionalError> {
        if self.mesh != other.mesh {
            return Err(RegionalError::MuffinTinMeshMismatch);
        }
        // A zero-scale checked accumulation compares convention and channels
        // without changing either operand.
        let mut probe = self.field.clone();
        probe.add_scaled(Complex64::new(0.0, 0.0), &other.field)?;
        Ok(())
    }
}

/// A real periodic scalar field on an exact reciprocal layout.
#[derive(Clone, Debug, PartialEq)]
pub struct InterstitialField {
    field: HermitianFourierField,
}

impl InterstitialField {
    /// Build from coefficients keyed by integer reciprocal coordinates.
    ///
    /// The map must cover the layout exactly. The wrapped core field then
    /// enforces finite coefficients, `f(-G) = conj(f(G))`, and real `G=0`.
    pub fn new(
        layout: FourierLayout,
        mut coefficients: BTreeMap<[i32; 3], Complex64>,
    ) -> Result<Self, RegionalError> {
        let mut ordered = Vec::with_capacity(layout.len());
        for vector in layout.vectors() {
            let value = coefficients
                .remove(&vector.index)
                .ok_or(RegionalError::MissingInterstitialCoefficient { g: vector.index })?;
            ordered.push(value);
        }
        if let Some((&g, _)) = coefficients.first_key_value() {
            return Err(RegionalError::ExtraInterstitialCoefficient { g });
        }
        Ok(Self {
            field: HermitianFourierField::new(layout, ordered)?,
        })
    }

    pub const fn from_fourier_field(field: HermitianFourierField) -> Self {
        Self { field }
    }

    pub const fn field(&self) -> &HermitianFourierField {
        &self.field
    }

    pub const fn layout(&self) -> &FourierLayout {
        self.field.layout()
    }

    pub fn coefficient(&self, g: [i32; 3]) -> Option<Complex64> {
        self.field.coefficient(g)
    }

    pub fn coefficients(&self) -> impl Iterator<Item = ([i32; 3], Complex64)> + '_ {
        self.field
            .iter()
            .map(|(vector, coefficient)| (vector.index, *coefficient))
    }

    pub fn zero_like(&self) -> Self {
        Self {
            field: self.field.zero_like(),
        }
    }

    pub fn add_scaled(&mut self, scale: f64, other: &Self) -> Result<(), RegionalError> {
        self.field.add_scaled(scale, &other.field)?;
        Ok(())
    }

    pub fn difference(&self, other: &Self) -> Result<Self, RegionalError> {
        Ok(Self {
            field: self.field.difference(&other.field)?,
        })
    }
}

impl TryFrom<&InterstitialField> for InterstitialPotential {
    type Error = LapwError;

    fn try_from(field: &InterstitialField) -> Result<Self, Self::Error> {
        Self::new(field.coefficients())
    }
}

/// One physically real scalar field over the muffin-tin/interstitial partition.
///
/// This is the component type used by noncollinear densities: charge and each
/// Cartesian magnetization component share the same exact regional layout,
/// without pretending that signed transverse magnetization is a collinear
/// occupation channel.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionalScalarField {
    geometry: InterstitialGeometry,
    muffin_tins: Vec<MuffinTinField>,
    interstitial: InterstitialField,
}

impl RegionalScalarField {
    pub fn new(
        geometry: InterstitialGeometry,
        muffin_tins: Vec<MuffinTinField>,
        interstitial: InterstitialField,
    ) -> Result<Self, RegionalError> {
        if muffin_tins.len() != geometry.spheres().len() {
            return Err(RegionalError::GeometryMuffinTinCountMismatch {
                geometry: geometry.spheres().len(),
                fields: muffin_tins.len(),
            });
        }
        validate_reciprocal_volume(&geometry, interstitial.layout())?;
        Ok(Self {
            geometry,
            muffin_tins,
            interstitial,
        })
    }

    pub const fn geometry(&self) -> &InterstitialGeometry {
        &self.geometry
    }

    pub fn muffin_tins(&self) -> &[MuffinTinField] {
        &self.muffin_tins
    }

    pub const fn interstitial(&self) -> &InterstitialField {
        &self.interstitial
    }

    pub fn zero_like(&self) -> Self {
        Self {
            geometry: self.geometry.clone(),
            muffin_tins: self
                .muffin_tins
                .iter()
                .map(MuffinTinField::zero_like)
                .collect(),
            interstitial: self.interstitial.zero_like(),
        }
    }

    pub fn add_scaled(&mut self, scale: f64, other: &Self) -> Result<(), RegionalError> {
        self.require_same_layout(other)?;
        for (target, source) in self.muffin_tins.iter_mut().zip(&other.muffin_tins) {
            target.add_scaled(scale, source)?;
        }
        self.interstitial.add_scaled(scale, &other.interstitial)
    }

    pub fn difference(&self, other: &Self) -> Result<Self, RegionalError> {
        self.require_same_layout(other)?;
        let mut result = self.clone();
        result.add_scaled(-1.0, other)?;
        Ok(result)
    }

    /// Physical regional inner product for one scalar component.
    pub fn physical_inner_product(&self, other: &Self) -> Result<f64, RegionalError> {
        self.require_same_layout(other)?;
        let mut total = Complex64::new(0.0, 0.0);
        let mut absolute_scale = 0.0;
        for (left, right) in self.muffin_tins.iter().zip(&other.muffin_tins) {
            let contribution = muffin_tin_inner_product(left, right)?;
            total += contribution;
            absolute_scale += contribution.norm();
        }
        let (contribution, term_scale) =
            interstitial_inner_product(&self.geometry, &self.interstitial, &other.interstitial)?;
        total += contribution;
        absolute_scale += term_scale;
        real_metric_value(total, absolute_scale)
    }

    fn physical_inner_product_by_region(&self, other: &Self) -> Result<(f64, f64), RegionalError> {
        self.require_same_layout(other)?;
        let mut muffin_tin = Complex64::new(0.0, 0.0);
        let mut muffin_tin_scale = 0.0;
        for (left, right) in self.muffin_tins.iter().zip(&other.muffin_tins) {
            let contribution = muffin_tin_inner_product(left, right)?;
            muffin_tin += contribution;
            muffin_tin_scale += contribution.norm();
        }
        let (interstitial, interstitial_scale) =
            interstitial_inner_product(&self.geometry, &self.interstitial, &other.interstitial)?;
        Ok((
            real_metric_value(muffin_tin, muffin_tin_scale)?,
            real_metric_value(interstitial, interstitial_scale)?,
        ))
    }

    /// Root-mean-square magnitude of this component per cell volume.
    pub fn residual_rms(&self) -> Result<f64, RegionalError> {
        let norm_squared = self.physical_inner_product(self)?;
        let tolerance = REALITY_TOLERANCE * norm_squared.abs().max(1.0);
        if norm_squared < -tolerance {
            return Err(RegionalError::NegativeNorm(norm_squared));
        }
        Ok((norm_squared.max(0.0) / self.geometry.cell_volume().get()).sqrt())
    }

    pub fn difference_rms(&self, other: &Self) -> Result<f64, RegionalError> {
        self.difference(other)?.residual_rms()
    }

    /// Apply one active crystal operation to this physical scalar field.
    ///
    /// Interstitial coefficients use the input-cell affine operation, sites
    /// follow the transform's explicit permutation, and muffin-tin channels
    /// are rotated in their declared normalized-harmonic convention.  Time
    /// reversal has no extra action on a physically real, time-even scalar.
    pub fn transformed(&self, transform: &CrystalSymmetryTransform) -> Result<Self, RegionalError> {
        self.validate_symmetry_transform(transform)?;
        let operation = transform.operation();
        let interstitial_coefficients = self
            .interstitial
            .layout()
            .vectors()
            .iter()
            .map(|vector| {
                let source = transpose_integer_action(operation.rotation, vector.index)?;
                let coefficient = self.interstitial.coefficient(source).ok_or(
                    RegionalError::SymmetryFourierLayoutNotClosed {
                        target: vector.index,
                        source_g: source,
                    },
                )?;
                let phase = -TAU
                    * vector
                        .index
                        .iter()
                        .zip(operation.translation)
                        .map(|(&index, translation)| f64::from(index) * translation)
                        .sum::<f64>();
                Ok((
                    vector.index,
                    coefficient * Complex64::from_polar(1.0, phase),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, RegionalError>>()?;
        let interstitial = InterstitialField::new(
            self.interstitial.layout().clone(),
            interstitial_coefficients,
        )?;

        let mut muffin_tins = vec![None; self.muffin_tins.len()];
        for (source, (&target, field)) in transform
            .site_map()
            .iter()
            .zip(&self.muffin_tins)
            .enumerate()
        {
            debug_assert!(muffin_tins[target].is_none(), "validated site permutation");
            muffin_tins[target] = Some(rotate_muffin_tin(field, *transform.cartesian_rotation())?);
            debug_assert!(source < self.muffin_tins.len());
        }
        Ok(Self {
            geometry: self.geometry.clone(),
            muffin_tins: muffin_tins
                .into_iter()
                .map(|field| field.expect("validated site permutation is complete"))
                .collect(),
            interstitial,
        })
    }

    /// Project this scalar field onto the invariant subspace of `transforms`.
    pub fn symmetry_average(
        &self,
        transforms: &[CrystalSymmetryTransform],
    ) -> Result<Self, RegionalError> {
        let (first, rest) = transforms
            .split_first()
            .ok_or(RegionalError::EmptySymmetryGroup)?;
        let first = self.transformed(first)?;
        let mut average = first.zero_like();
        let scale = 1.0 / transforms.len() as f64;
        average.add_scaled(scale, &first)?;
        for transform in rest {
            average.add_scaled(scale, &self.transformed(transform)?)?;
        }
        Ok(average)
    }

    fn require_same_layout(&self, other: &Self) -> Result<(), RegionalError> {
        if self.geometry != other.geometry {
            return Err(RegionalError::InterstitialGeometryMismatch);
        }
        if self.muffin_tins.len() != other.muffin_tins.len() {
            return Err(RegionalError::RegionalMuffinTinCountMismatch);
        }
        for (left, right) in self.muffin_tins.iter().zip(&other.muffin_tins) {
            left.require_same_layout(right)?;
        }
        if self.interstitial.layout() != other.interstitial.layout() {
            return Err(RegionalError::Fourier(FourierFieldError::LayoutMismatch));
        }
        Ok(())
    }

    fn validate_symmetry_transform(
        &self,
        transform: &CrystalSymmetryTransform,
    ) -> Result<(), RegionalError> {
        if transform.site_map().len() != self.muffin_tins.len() {
            return Err(RegionalError::SymmetrySiteCountMismatch {
                transform: transform.site_map().len(),
                field: self.muffin_tins.len(),
            });
        }
        validate_direct_reciprocal_duality(transform.direct_lattice(), self.interstitial.layout())?;
        let tolerance = transform.tolerance().get();
        for (source, &target) in transform.site_map().iter().enumerate() {
            let source_sphere = self.geometry.spheres()[source];
            let target_sphere = self.geometry.spheres()[target];
            if (source_sphere.radius.get() - target_sphere.radius.get()).abs() > tolerance {
                return Err(RegionalError::SymmetryMuffinTinRadiusMismatch {
                    source_site: source,
                    target,
                });
            }
            if self.muffin_tins[source].mesh != self.muffin_tins[target].mesh {
                return Err(RegionalError::SymmetryMuffinTinMeshMismatch {
                    source_site: source,
                    target,
                });
            }
        }
        Ok(())
    }
}

/// Charge and Cartesian magnetization over muffin-tin and interstitial regions.
///
/// The density matrix convention is
/// `rho = (charge * I + magnetization . sigma) / 2`. Therefore a collinear
/// density has `charge = rho_up + rho_down` and `magnetization[2] = rho_up -
/// rho_down`; transverse magnetization components are ordinary signed scalar
/// fields, not occupation channels.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionalDensity {
    charge: RegionalScalarField,
    magnetization: [RegionalScalarField; 3],
}

/// Cell-volume-normalized RMS contributions from the muffin tins and
/// interstitial region. Their squared values add to `total_rms^2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionalDensityResidualRms {
    pub muffin_tin_rms: f64,
    pub interstitial_rms: f64,
    pub total_rms: f64,
}

impl RegionalDensity {
    pub fn new(
        charge: RegionalScalarField,
        magnetization: [RegionalScalarField; 3],
    ) -> Result<Self, RegionalError> {
        for component in &magnetization {
            charge.require_same_layout(component)?;
        }
        Ok(Self {
            charge,
            magnetization,
        })
    }

    pub const fn geometry(&self) -> &InterstitialGeometry {
        self.charge.geometry()
    }

    pub const fn charge(&self) -> &RegionalScalarField {
        &self.charge
    }

    /// Cartesian magnetization fields in `[mx, my, mz]` order.
    pub const fn magnetization(&self) -> &[RegionalScalarField; 3] {
        &self.magnetization
    }

    pub fn zero_like(&self) -> Self {
        Self {
            charge: self.charge.zero_like(),
            magnetization: self.magnetization.each_ref().map(|field| field.zero_like()),
        }
    }

    pub fn add_scaled(&mut self, scale: f64, other: &Self) -> Result<(), RegionalError> {
        self.require_same_layout(other)?;
        self.charge.add_scaled(scale, &other.charge)?;
        for (target, source) in self.magnetization.iter_mut().zip(&other.magnetization) {
            target.add_scaled(scale, source)?;
        }
        Ok(())
    }

    /// Form `self - other` after exact regional-layout validation.
    pub fn difference(&self, other: &Self) -> Result<Self, RegionalError> {
        self.require_same_layout(other)?;
        let mut difference = self.clone();
        difference.add_scaled(-1.0, other)?;
        Ok(difference)
    }

    /// Physical density-matrix inner product.
    ///
    /// The factor one half is `Tr(rho_a rho_b)` in the Pauli decomposition and
    /// makes the result reduce exactly to the sum of explicit up/down metrics
    /// for a collinear density.
    pub fn physical_inner_product(&self, other: &Self) -> Result<f64, RegionalError> {
        self.require_same_layout(other)?;
        let mut total = self.charge.physical_inner_product(&other.charge)?;
        for (left, right) in self.magnetization.iter().zip(&other.magnetization) {
            total += left.physical_inner_product(right)?;
        }
        let total = 0.5 * total;
        if !total.is_finite() {
            return Err(RegionalError::NonFiniteMetric);
        }
        Ok(total)
    }

    /// Root-mean-square magnitude of this regional residual per cell volume.
    pub fn residual_rms(&self) -> Result<f64, RegionalError> {
        let norm_squared = self.physical_inner_product(self)?;
        let tolerance = REALITY_TOLERANCE * norm_squared.abs().max(1.0);
        if norm_squared < -tolerance {
            return Err(RegionalError::NegativeNorm(norm_squared));
        }
        Ok((norm_squared.max(0.0) / self.geometry().cell_volume().get()).sqrt())
    }

    /// Decompose the physical Pauli-density residual norm by spatial region.
    pub fn residual_rms_by_region(&self) -> Result<RegionalDensityResidualRms, RegionalError> {
        let (mut muffin_tin, mut interstitial) =
            self.charge.physical_inner_product_by_region(&self.charge)?;
        for component in &self.magnetization {
            let (component_mt, component_interstitial) =
                component.physical_inner_product_by_region(component)?;
            muffin_tin += component_mt;
            interstitial += component_interstitial;
        }
        muffin_tin *= 0.5;
        interstitial *= 0.5;
        let scale = muffin_tin.abs().max(interstitial.abs()).max(1.0);
        let tolerance = REALITY_TOLERANCE * scale;
        if muffin_tin < -tolerance {
            return Err(RegionalError::NegativeNorm(muffin_tin));
        }
        if interstitial < -tolerance {
            return Err(RegionalError::NegativeNorm(interstitial));
        }
        let inverse_volume = 1.0 / self.geometry().cell_volume().get();
        let muffin_tin_rms = (muffin_tin.max(0.0) * inverse_volume).sqrt();
        let interstitial_rms = (interstitial.max(0.0) * inverse_volume).sqrt();
        Ok(RegionalDensityResidualRms {
            muffin_tin_rms,
            interstitial_rms,
            total_rms: muffin_tin_rms.hypot(interstitial_rms),
        })
    }

    /// RMS of `self - other`, using the same physical metric.
    pub fn difference_rms(&self, other: &Self) -> Result<f64, RegionalError> {
        self.difference(other)?.residual_rms()
    }

    /// Apply one crystal operation to charge and axial magnetization.
    ///
    /// Charge is time-even.  Magnetization transforms as an axial Cartesian
    /// vector, with an additional sign under an antiunitary operation.
    pub fn transformed(&self, transform: &CrystalSymmetryTransform) -> Result<Self, RegionalError> {
        let charge = self.charge.transformed(transform)?;
        let spatial = self
            .magnetization
            .each_ref()
            .map(|component| component.transformed(transform))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let rotation = transform.cartesian_rotation();
        let determinant = determinant(*rotation);
        let time_reversal = if transform.operation().time_reversal {
            -1.0
        } else {
            1.0
        };
        let axial_sign = determinant * time_reversal;
        let mut magnetization = Vec::with_capacity(3);
        for row in rotation {
            let mut component = spatial[0].zero_like();
            for (weight, source) in row.iter().zip(&spatial) {
                component.add_scaled(axial_sign * weight, source)?;
            }
            magnetization.push(component);
        }
        Self::new(
            charge,
            magnetization
                .try_into()
                .expect("three Cartesian rows produce three components"),
        )
    }

    /// Project charge and magnetization onto the crystal-symmetry invariants.
    pub fn symmetry_average(
        &self,
        transforms: &[CrystalSymmetryTransform],
    ) -> Result<Self, RegionalError> {
        let (first, rest) = transforms
            .split_first()
            .ok_or(RegionalError::EmptySymmetryGroup)?;
        let first = self.transformed(first)?;
        let mut average = first.zero_like();
        let scale = 1.0 / transforms.len() as f64;
        average.add_scaled(scale, &first)?;
        for transform in rest {
            average.add_scaled(scale, &self.transformed(transform)?)?;
        }
        Ok(average)
    }

    fn require_same_layout(&self, other: &Self) -> Result<(), RegionalError> {
        self.charge.require_same_layout(&other.charge)?;
        for (left, right) in self.magnetization.iter().zip(&other.magnetization) {
            left.require_same_layout(right)?;
        }
        Ok(())
    }
}

/// Noncollinear regional potential `scalar * I + magnetic . sigma`.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionalPotential {
    scalar: RegionalScalarField,
    magnetic: [RegionalScalarField; 3],
}

impl RegionalPotential {
    pub fn new(
        scalar: RegionalScalarField,
        magnetic: [RegionalScalarField; 3],
    ) -> Result<Self, RegionalError> {
        for component in &magnetic {
            scalar.require_same_layout(component)?;
        }
        Ok(Self { scalar, magnetic })
    }

    pub const fn scalar(&self) -> &RegionalScalarField {
        &self.scalar
    }

    /// Cartesian magnetic fields in `[Bx, By, Bz]` order.
    pub const fn magnetic(&self) -> &[RegionalScalarField; 3] {
        &self.magnetic
    }

    pub fn zero_like(&self) -> Self {
        Self {
            scalar: self.scalar.zero_like(),
            magnetic: self.magnetic.each_ref().map(|field| field.zero_like()),
        }
    }

    pub fn add_scaled(&mut self, scale: f64, other: &Self) -> Result<(), RegionalError> {
        self.require_same_layout(other)?;
        self.scalar.add_scaled(scale, &other.scalar)?;
        for (target, source) in self.magnetic.iter_mut().zip(&other.magnetic) {
            target.add_scaled(scale, source)?;
        }
        Ok(())
    }

    pub fn difference(&self, other: &Self) -> Result<Self, RegionalError> {
        self.require_same_layout(other)?;
        let mut difference = self.clone();
        difference.add_scaled(-1.0, other)?;
        Ok(difference)
    }

    /// Apply one crystal operation to the scalar and axial magnetic fields.
    pub fn transformed(&self, transform: &CrystalSymmetryTransform) -> Result<Self, RegionalError> {
        let scalar = self.scalar.transformed(transform)?;
        let spatial = self
            .magnetic
            .each_ref()
            .map(|component| component.transformed(transform))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let rotation = transform.cartesian_rotation();
        let time_reversal = if transform.operation().time_reversal {
            -1.0
        } else {
            1.0
        };
        let axial_sign = determinant(*rotation) * time_reversal;
        let mut magnetic = Vec::with_capacity(3);
        for row in rotation {
            let mut component = spatial[0].zero_like();
            for (weight, source) in row.iter().zip(&spatial) {
                component.add_scaled(axial_sign * weight, source)?;
            }
            magnetic.push(component);
        }
        Self::new(
            scalar,
            magnetic
                .try_into()
                .expect("three Cartesian rows produce three components"),
        )
    }

    /// Project scalar and magnetic fields onto the crystal-symmetry invariants.
    pub fn symmetry_average(
        &self,
        transforms: &[CrystalSymmetryTransform],
    ) -> Result<Self, RegionalError> {
        let (first, rest) = transforms
            .split_first()
            .ok_or(RegionalError::EmptySymmetryGroup)?;
        let first = self.transformed(first)?;
        let mut average = first.zero_like();
        let scale = 1.0 / transforms.len() as f64;
        average.add_scaled(scale, &first)?;
        for transform in rest {
            average.add_scaled(scale, &self.transformed(transform)?)?;
        }
        Ok(average)
    }

    fn require_same_layout(&self, other: &Self) -> Result<(), RegionalError> {
        self.scalar.require_same_layout(&other.scalar)?;
        for (left, right) in self.magnetic.iter().zip(&other.magnetic) {
            left.require_same_layout(right)?;
        }
        Ok(())
    }

    /// Convert the Pauli components to the LAPW interstitial boundary.
    pub fn to_lapw_interstitial(&self) -> Result<InterstitialPauliPotential, RegionalError> {
        Ok(InterstitialPauliPotential::new(
            InterstitialPotential::try_from(self.scalar.interstitial())?,
            InterstitialPotential::try_from(self.magnetic[0].interstitial())?,
            InterstitialPotential::try_from(self.magnetic[1].interstitial())?,
            InterstitialPotential::try_from(self.magnetic[2].interstitial())?,
        ))
    }
}

fn transpose_integer_action(
    rotation: [[i32; 3]; 3],
    target: [i32; 3],
) -> Result<[i32; 3], RegionalError> {
    let component = |row| {
        let value = (0..3).try_fold(0_i64, |sum, column| {
            sum.checked_add(i64::from(rotation[column][row]) * i64::from(target[column]))
        });
        value
            .and_then(|value| i32::try_from(value).ok())
            .ok_or(RegionalError::SymmetryReciprocalIndexOverflow { target })
    };
    Ok([component(0)?, component(1)?, component(2)?])
}

fn rotate_muffin_tin(
    source: &MuffinTinField,
    rotation: [[f64; 3]; 3],
) -> Result<MuffinTinField, RegionalError> {
    let angular_momenta = source
        .field
        .channels()
        .map(|(channel, _)| channel.l)
        .collect::<BTreeSet<_>>();
    let sample_count = source.field.sample_count().unwrap_or(0);
    let mut channels = Vec::new();
    for l in angular_momenta {
        let dimension = usize::try_from(2 * l + 1).expect("u32 angular dimension fits usize");
        let projection = harmonic_rotation(source.field.convention(), l, rotation);
        for target_offset in 0..dimension {
            let target_m = i32::try_from(target_offset).expect("angular offset fits i32")
                - i32::try_from(l).expect("l fits i32");
            let mut values = vec![Complex64::new(0.0, 0.0); sample_count];
            for source_offset in 0..dimension {
                let source_m = i32::try_from(source_offset).expect("angular offset fits i32")
                    - i32::try_from(l).expect("l fits i32");
                let Some(source_values) = source.field.channel(l, source_m) else {
                    continue;
                };
                let weight = projection[target_offset * dimension + source_offset];
                for (value, &source_value) in values.iter_mut().zip(source_values) {
                    *value += weight * source_value;
                }
            }
            channels.push(((l, target_m), values));
        }
    }
    MuffinTinField::new(
        source.mesh.clone(),
        SphereField::new(source.field.convention(), channels)?,
    )
}

/// Matrix of the active scalar action
/// `Y_lm(R^-1 n) = sum_M D[M,m] Y_lM(n)`.
fn harmonic_rotation(
    convention: HarmonicConvention,
    l: u32,
    rotation: [[f64; 3]; 3],
) -> Vec<Complex64> {
    let dimension = usize::try_from(2 * l + 1).expect("u32 angular dimension fits usize");
    let theta_order = usize::try_from(l + 1).expect("u32 quadrature order fits usize");
    let phi_order = usize::try_from(4 * l + 2).expect("u32 quadrature order fits usize");
    let phi_weight = TAU / phi_order as f64;
    let mut matrix = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for (z, z_weight) in gauss_legendre(theta_order) {
        let radial = (1.0 - z * z).max(0.0).sqrt();
        for phi_index in 0..phi_order {
            let phi = TAU * (phi_index as f64 + 0.5) / phi_order as f64;
            let target_direction = [radial * phi.cos(), radial * phi.sin(), z];
            let source_direction: [f64; 3] = std::array::from_fn(|axis| {
                (0..3)
                    .map(|row| rotation[row][axis] * target_direction[row])
                    .sum()
            });
            match convention {
                HarmonicConvention::Complex => {
                    let target = complex_spherical_harmonics(l, target_direction);
                    let source = complex_spherical_harmonics(l, source_direction);
                    let offset = usize::try_from(l * l).expect("u32 harmonic offset fits usize");
                    for target_index in 0..dimension {
                        for source_index in 0..dimension {
                            matrix[target_index * dimension + source_index] += z_weight
                                * phi_weight
                                * target[offset + target_index].conj()
                                * source[offset + source_index];
                        }
                    }
                }
                HarmonicConvention::Real => {
                    let target = real_spherical_harmonics(l, target_direction);
                    let source = real_spherical_harmonics(l, source_direction);
                    let offset = usize::try_from(l * l).expect("u32 harmonic offset fits usize");
                    for target_index in 0..dimension {
                        for source_index in 0..dimension {
                            matrix[target_index * dimension + source_index].re += z_weight
                                * phi_weight
                                * target[offset + target_index]
                                * source[offset + source_index];
                        }
                    }
                }
            }
        }
    }
    matrix
}

fn gauss_legendre(order: usize) -> Vec<(f64, f64)> {
    let mut result = vec![(0.0, 0.0); order];
    let half = order.div_ceil(2);
    for index in 0..half {
        let mut node = (PI * (index as f64 + 0.75) / (order as f64 + 0.5)).cos();
        let node = loop {
            let (polynomial, previous) = legendre_pair(order, node);
            let derivative = order as f64 * (node * polynomial - previous) / (node * node - 1.0);
            let next = node - polynomial / derivative;
            if (next - node).abs() <= 8.0 * f64::EPSILON {
                break next;
            }
            node = next;
        };
        let (_, previous) = legendre_pair(order, node);
        let polynomial = legendre_pair(order, node).0;
        let derivative = order as f64 * (node * polynomial - previous) / (node * node - 1.0);
        let weight = 2.0 / ((1.0 - node * node) * derivative * derivative);
        result[index] = (node, weight);
        result[order - 1 - index] = (-node, weight);
    }
    result
}

fn legendre_pair(order: usize, x: f64) -> (f64, f64) {
    let mut previous = 1.0;
    if order == 0 {
        return (previous, 0.0);
    }
    let mut current = x;
    for degree in 2..=order {
        let next = ((2 * degree - 1) as f64 * x * current - (degree - 1) as f64 * previous)
            / degree as f64;
        previous = current;
        current = next;
    }
    (current, previous)
}

fn validate_direct_reciprocal_duality(
    direct: &[[muffintin_core::Bohr; 3]; 3],
    layout: &FourierLayout,
) -> Result<(), RegionalError> {
    let reciprocal = layout.reciprocal().basis();
    let scale = direct
        .iter()
        .flatten()
        .map(|value| value.get().abs())
        .fold(1.0_f64, f64::max)
        * reciprocal
            .iter()
            .flatten()
            .map(|value| value.get().abs())
            .fold(1.0_f64, f64::max);
    let tolerance = 2048.0 * f64::EPSILON * scale;
    for (direct_index, direct_vector) in direct.iter().enumerate() {
        for (reciprocal_index, reciprocal_vector) in reciprocal.iter().enumerate() {
            let dot = direct_vector
                .iter()
                .zip(reciprocal_vector)
                .map(|(left, right)| left.get() * right.get())
                .sum::<f64>();
            let expected = if direct_index == reciprocal_index {
                TAU
            } else {
                0.0
            };
            if (dot - expected).abs() > tolerance {
                return Err(RegionalError::SymmetryLatticeMismatch);
            }
        }
    }
    Ok(())
}

fn determinant(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn muffin_tin_inner_product(
    left: &MuffinTinField,
    right: &MuffinTinField,
) -> Result<Complex64, RegionalError> {
    left.require_same_layout(right)?;
    let radii = left.mesh.radii();
    let mut total = Complex64::new(0.0, 0.0);
    for ((left_lm, left_values), (right_lm, right_values)) in
        left.field.channels().zip(right.field.channels())
    {
        debug_assert_eq!(left_lm, right_lm);
        let products: Vec<_> = left_values
            .iter()
            .zip(right_values)
            .zip(radii)
            .map(|((&a, &b), radius)| a.conj() * b * radius.get().powi(2))
            .collect();
        let real: Vec<_> = products.iter().map(|value| value.re).collect();
        let imaginary: Vec<_> = products.iter().map(|value| value.im).collect();
        total += Complex64::new(
            left.mesh.integrate(&real)?,
            left.mesh.integrate(&imaginary)?,
        );
    }
    Ok(total)
}

fn interstitial_inner_product(
    geometry: &InterstitialGeometry,
    left: &InterstitialField,
    right: &InterstitialField,
) -> Result<(Complex64, f64), RegionalError> {
    if left.layout() != right.layout() {
        return Err(RegionalError::Fourier(FourierFieldError::LayoutMismatch));
    }
    let reciprocal = left.layout().reciprocal();
    let volume = geometry.cell_volume().get();
    let mut total = Complex64::new(0.0, 0.0);
    let mut absolute_scale = 0.0;
    let mut step_coefficients = HashMap::new();
    for (left_vector, &left_value) in left.field.iter() {
        for (right_vector, &right_value) in right.field.iter() {
            let difference = [
                left_vector.index[0].checked_sub(right_vector.index[0]),
                left_vector.index[1].checked_sub(right_vector.index[1]),
                left_vector.index[2].checked_sub(right_vector.index[2]),
            ];
            let difference = match difference {
                [Some(g0), Some(g1), Some(g2)] => [g0, g1, g2],
                _ => {
                    return Err(RegionalError::ReciprocalDifferenceOverflow {
                        left: left_vector.index,
                        right: right_vector.index,
                    });
                }
            };
            let theta = match step_coefficients.entry(difference) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    *entry.insert(geometry.coefficient(reciprocal.cartesian(difference))?)
                }
            };
            let term = volume * left_value.conj() * theta * right_value;
            total += term;
            absolute_scale += term.norm();
        }
    }
    Ok((total, absolute_scale))
}

fn real_metric_value(value: Complex64, absolute_scale: f64) -> Result<f64, RegionalError> {
    if !value.re.is_finite() || !value.im.is_finite() || !absolute_scale.is_finite() {
        return Err(RegionalError::NonFiniteMetric);
    }
    let tolerance = REALITY_TOLERANCE * absolute_scale.max(value.re.abs()).max(1.0);
    if value.im.abs() > tolerance {
        Err(RegionalError::NonRealMetric {
            imaginary: value.im,
            tolerance,
        })
    } else {
        Ok(value.re)
    }
}

fn validate_reciprocal_volume(
    geometry: &InterstitialGeometry,
    layout: &FourierLayout,
) -> Result<(), RegionalError> {
    let basis = layout.reciprocal().basis();
    let b = basis.map(|vector| vector.map(|component| component.get()));
    let determinant = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);
    let reciprocal_volume = TAU.powi(3) / determinant.abs();
    let geometry_volume = geometry.cell_volume().get();
    let tolerance = REALITY_TOLERANCE * reciprocal_volume.max(geometry_volume).max(1.0);
    if (reciprocal_volume - geometry_volume).abs() > tolerance {
        Err(RegionalError::ReciprocalVolumeMismatch {
            geometry: geometry_volume,
            reciprocal: reciprocal_volume,
        })
    } else {
        Ok(())
    }
}

/// Invalid regional field, layout, or physical-metric operation.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RegionalError {
    #[error("muffin-tin field has {actual} radial samples, expected {expected}")]
    MuffinTinSampleCount { expected: usize, actual: usize },
    #[error("muffin-tin radial meshes differ")]
    MuffinTinMeshMismatch,
    #[error("regional fields have different muffin-tin counts")]
    RegionalMuffinTinCountMismatch,
    #[error("geometry has {geometry} spheres but regional field has {fields} muffin tins")]
    GeometryMuffinTinCountMismatch { geometry: usize, fields: usize },
    #[error("interstitial geometries differ")]
    InterstitialGeometryMismatch,
    #[error("interstitial coefficient map is missing G={g:?}")]
    MissingInterstitialCoefficient { g: [i32; 3] },
    #[error("interstitial coefficient map contains G={g:?} outside its layout")]
    ExtraInterstitialCoefficient { g: [i32; 3] },
    #[error("G-vector difference overflows i32: {left:?} - {right:?}")]
    ReciprocalDifferenceOverflow { left: [i32; 3], right: [i32; 3] },
    #[error(
        "reciprocal lattice implies volume {reciprocal}, but interstitial geometry has {geometry}"
    )]
    ReciprocalVolumeMismatch { geometry: f64, reciprocal: f64 },
    #[error("physical inner product has imaginary part {imaginary}, tolerance {tolerance}")]
    NonRealMetric { imaginary: f64, tolerance: f64 },
    #[error("physical inner product is not finite")]
    NonFiniteMetric,
    #[error("physical squared norm is negative: {0}")]
    NegativeNorm(f64),
    #[error("cannot average an empty symmetry-operation set")]
    EmptySymmetryGroup,
    #[error("symmetry transform has {transform} sites but regional field has {field}")]
    SymmetrySiteCountMismatch { transform: usize, field: usize },
    #[error("symmetry direct lattice is not dual to the field reciprocal lattice")]
    SymmetryLatticeMismatch,
    #[error("symmetry maps muffin tin {source_site} to {target} with a different radius")]
    SymmetryMuffinTinRadiusMismatch { source_site: usize, target: usize },
    #[error("symmetry maps muffin tin {source_site} to {target} with a different radial mesh")]
    SymmetryMuffinTinMeshMismatch { source_site: usize, target: usize },
    #[error("symmetry reciprocal action overflows i32 at target G={target:?}")]
    SymmetryReciprocalIndexOverflow { target: [i32; 3] },
    #[error(
        "Fourier layout is not symmetry closed: target G={target:?} needs source G={source_g:?}"
    )]
    SymmetryFourierLayoutNotClosed {
        target: [i32; 3],
        source_g: [i32; 3],
    },
    #[error(transparent)]
    Fourier(#[from] FourierFieldError),
    #[error(transparent)]
    Sphere(#[from] SphereFieldError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    StepFunction(#[from] StepFunctionError),
    #[error(transparent)]
    Lapw(#[from] LapwError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::{Bohr, GVector, InverseBohr, ReciprocalLattice, Sphere, VolumeBohr3};
    use muffintin_sphere::HarmonicConvention;
    use muffintin_symmetry::{CrystalCell, SymmetryOperation};

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
        let vectors = indices
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
            .collect();
        FourierLayout::new(reciprocal, vectors).unwrap()
    }

    fn interstitial(
        layout: FourierLayout,
        values: impl IntoIterator<Item = ([i32; 3], Complex64)>,
    ) -> InterstitialField {
        InterstitialField::new(layout, values.into_iter().collect()).unwrap()
    }

    fn geometry(spheres: Vec<Sphere>) -> InterstitialGeometry {
        InterstitialGeometry::new(VolumeBohr3(TAU.powi(3)), spheres).unwrap()
    }

    fn direct_lattice() -> [[Bohr; 3]; 3] {
        [
            [Bohr(TAU), Bohr(0.0), Bohr(0.0)],
            [Bohr(0.0), Bohr(TAU), Bohr(0.0)],
            [Bohr(0.0), Bohr(0.0), Bohr(TAU)],
        ]
    }

    fn regional_scalar(
        geometry: InterstitialGeometry,
        interstitial: InterstitialField,
    ) -> RegionalScalarField {
        let empty_muffin_tin = MuffinTinField::new(
            ExponentialMesh::new(Bohr(0.01), 0.1, 7).unwrap(),
            SphereField::new(HarmonicConvention::Real, []).unwrap(),
        )
        .unwrap();
        let muffin_tins = vec![empty_muffin_tin; geometry.spheres().len()];
        RegionalScalarField::new(geometry, muffin_tins, interstitial).unwrap()
    }

    fn g0_scalar(
        geometry: &InterstitialGeometry,
        layout: &FourierLayout,
        value: f64,
    ) -> RegionalScalarField {
        regional_scalar(
            geometry.clone(),
            interstitial(layout.clone(), [([0; 3], Complex64::new(value, 0.0))]),
        )
    }

    #[test]
    fn pauli_metric_reduces_to_explicit_collinear_spin_metric() {
        let layout = layout(&[[0; 3]]);
        let geometry = geometry(Vec::new());
        // rho_up=2 and rho_down=3 imply n=5 and mz=-1.
        let charge = g0_scalar(&geometry, &layout, 5.0);
        let zero = charge.zero_like();
        let density = RegionalDensity::new(
            charge,
            [zero.clone(), zero, g0_scalar(&geometry, &layout, -1.0)],
        )
        .unwrap();
        let norm = density.physical_inner_product(&density).unwrap();
        let regional = density.residual_rms_by_region().unwrap();
        assert!((norm - TAU.powi(3) * 13.0).abs() < 1.0e-11);
        assert!((density.residual_rms().unwrap() - 13.0_f64.sqrt()).abs() < 1.0e-13);
        assert_eq!(regional.muffin_tin_rms, 0.0);
        assert!((regional.interstitial_rms - 13.0_f64.sqrt()).abs() < 1.0e-13);
        assert!((regional.total_rms - density.residual_rms().unwrap()).abs() < 1.0e-13);
        assert_eq!(density.difference_rms(&density).unwrap(), 0.0);
    }

    #[test]
    fn non_finite_physical_metric_is_rejected_before_reality_check() {
        let layout = layout(&[[0; 3]]);
        let geometry = geometry(Vec::new());
        let huge = g0_scalar(&geometry, &layout, 1.0e200);
        let zero = huge.zero_like();
        assert_eq!(
            huge.physical_inner_product(&huge),
            Err(RegionalError::NonFiniteMetric)
        );
        assert_eq!(huge.residual_rms(), Err(RegionalError::NonFiniteMetric));
        let density = RegionalDensity::new(huge, [zero.clone(), zero.clone(), zero]).unwrap();
        assert_eq!(
            density.physical_inner_product(&density),
            Err(RegionalError::NonFiniteMetric)
        );
        assert_eq!(density.residual_rms(), Err(RegionalError::NonFiniteMetric));
    }

    #[test]
    fn pauli_metric_is_invariant_under_global_spin_rotation() {
        let layout = layout(&[[0; 3]]);
        let geometry = geometry(Vec::new());
        let charge = g0_scalar(&geometry, &layout, 6.0);
        let first = RegionalDensity::new(
            charge.clone(),
            [
                g0_scalar(&geometry, &layout, 3.0),
                g0_scalar(&geometry, &layout, 4.0),
                charge.zero_like(),
            ],
        )
        .unwrap();
        let second = RegionalDensity::new(
            charge.clone(),
            [
                charge.zero_like(),
                charge.zero_like(),
                g0_scalar(&geometry, &layout, 5.0),
            ],
        )
        .unwrap();
        assert!(
            (first.physical_inner_product(&first).unwrap()
                - second.physical_inner_product(&second).unwrap())
            .abs()
                < 1.0e-11
        );
    }

    #[test]
    fn plus_minus_g_metric_uses_step_function_convolution() {
        let indices = [[-1, 0, 0], [1, 0, 0]];
        let layout = layout(&indices);
        let values = [
            ([-1, 0, 0], Complex64::new(1.0, 0.0)),
            ([1, 0, 0], Complex64::new(1.0, 0.0)),
        ];
        let sphere = Sphere {
            center: [Bohr(0.0); 3],
            radius: Bohr(0.5),
        };
        let geometry = geometry(vec![sphere]);
        let field = regional_scalar(geometry.clone(), interstitial(layout.clone(), values));
        let theta_zero = geometry.coefficient([InverseBohr(0.0); 3]).unwrap().re;
        let theta_two = geometry
            .coefficient([InverseBohr(2.0), InverseBohr(0.0), InverseBohr(0.0)])
            .unwrap()
            .re;
        let expected = TAU.powi(3) * (2.0 * theta_zero + 2.0 * theta_two);
        let diagonal_only = TAU.powi(3) * 2.0 * theta_zero;
        let actual = field.physical_inner_product(&field).unwrap();
        assert!((actual - expected).abs() < 1.0e-12);
        assert!((actual - diagonal_only).abs() > 1.0e-4);
        assert!(actual >= 0.0);
    }

    #[test]
    fn pure_muffin_tin_metric_matches_radial_quadrature() {
        let mesh = ExponentialMesh::new(Bohr(0.01), 2.0_f64.ln(), 7).unwrap();
        let left = MuffinTinField::new(
            mesh.clone(),
            SphereField::new(
                HarmonicConvention::Real,
                [((0, 0), vec![Complex64::new(2.0, 0.0); mesh.len()])],
            )
            .unwrap(),
        )
        .unwrap();
        let right = MuffinTinField::new(
            mesh.clone(),
            SphereField::new(
                HarmonicConvention::Real,
                [((0, 0), vec![Complex64::new(3.0, 0.0); mesh.len()])],
            )
            .unwrap(),
        )
        .unwrap();
        let reciprocal_layout = layout(&[[0; 3]]);
        let zero_interstitial =
            interstitial(reciprocal_layout, [([0; 3], Complex64::new(0.0, 0.0))]);
        let geometry = geometry(vec![Sphere {
            center: [Bohr(0.0); 3],
            radius: Bohr(1.0),
        }]);
        let left_field =
            RegionalScalarField::new(geometry.clone(), vec![left], zero_interstitial.clone())
                .unwrap();
        let right_field =
            RegionalScalarField::new(geometry, vec![right], zero_interstitial).unwrap();
        let r_squared: Vec<_> = mesh
            .radii()
            .iter()
            .map(|radius| radius.get().powi(2))
            .collect();
        let expected = 6.0 * mesh.integrate(&r_squared).unwrap();
        let actual = left_field.physical_inner_product(&right_field).unwrap();
        assert!((actual - expected).abs() < 1.0e-14);
    }

    #[test]
    fn interstitial_rejects_nonhermitian_and_layout_mismatch() {
        let three_g = layout(&[[-1, 0, 0], [0; 3], [1, 0, 0]]);
        let nonhermitian = InterstitialField::new(
            three_g,
            [
                ([-1, 0, 0], Complex64::new(1.0, 1.0)),
                ([0; 3], Complex64::new(0.0, 0.0)),
                ([1, 0, 0], Complex64::new(1.0, 1.0)),
            ]
            .into_iter()
            .collect(),
        );
        assert!(matches!(
            nonhermitian,
            Err(RegionalError::Fourier(
                FourierFieldError::NonHermitianPair { .. }
            ))
        ));

        let g0_layout = layout(&[[0; 3]]);
        let g0 = interstitial(g0_layout.clone(), [([0; 3], Complex64::new(0.0, 0.0))]);
        let three_layout = layout(&[[-1, 0, 0], [0; 3], [1, 0, 0]]);
        let three = interstitial(
            three_layout,
            [
                ([-1, 0, 0], Complex64::new(0.0, 0.0)),
                ([0; 3], Complex64::new(0.0, 0.0)),
                ([1, 0, 0], Complex64::new(0.0, 0.0)),
            ],
        );
        let first = regional_scalar(geometry(Vec::new()), g0);
        let second = regional_scalar(geometry(Vec::new()), three);
        assert!(matches!(
            first.difference(&second),
            Err(RegionalError::Fourier(FourierFieldError::LayoutMismatch))
        ));
    }

    #[test]
    fn regional_potential_converts_all_pauli_components_to_lapw() {
        let layout = layout(&[[0; 3]]);
        let geometry = geometry(Vec::new());
        let scalar = g0_scalar(&geometry, &layout, 0.25);
        let potential = RegionalPotential::new(
            scalar.clone(),
            [
                g0_scalar(&geometry, &layout, -0.5),
                g0_scalar(&geometry, &layout, 0.75),
                scalar.zero_like(),
            ],
        )
        .unwrap();
        let converted = potential.to_lapw_interstitial().unwrap();
        assert_eq!(converted.v0.coefficient([0; 3]), Complex64::new(0.25, 0.0));
        assert_eq!(converted.bx.coefficient([0; 3]), Complex64::new(-0.5, 0.0));
        assert_eq!(converted.by.coefficient([0; 3]), Complex64::new(0.75, 0.0));
    }

    #[test]
    fn crystal_transform_rotates_fourier_and_complex_muffin_tin_channels() {
        let operation = SymmetryOperation {
            rotation: [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            translation: [0.25, 0.25, 0.0],
            time_reversal: false,
        };
        let cell = CrystalCell {
            lattice: direct_lattice(),
            positions: vec![[0.0, 0.25, 0.0]],
            atomic_numbers: vec![6],
        };
        let transform =
            CrystalSymmetryTransform::from_cell(operation, &cell, Bohr(1.0e-12)).unwrap();
        assert_eq!(transform.site_map(), &[0]);

        let mesh = ExponentialMesh::new(Bohr(0.01), 0.1, 7).unwrap();
        let radial = vec![Complex64::new(1.0, 0.0); mesh.len()];
        let muffin_tin = MuffinTinField::new(
            mesh,
            SphereField::new(
                HarmonicConvention::Complex,
                [
                    ((1, -1), radial.iter().map(|value| -*value).collect()),
                    ((1, 1), radial),
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let layout = layout(&[[-1, 0, 0], [0, -1, 0], [0; 3], [0, 1, 0], [1, 0, 0]]);
        let x_coefficient = Complex64::new(2.0, 3.0);
        let interstitial = interstitial(
            layout,
            [
                ([-1, 0, 0], x_coefficient.conj()),
                ([0, -1, 0], Complex64::new(0.0, 0.0)),
                ([0; 3], Complex64::new(4.0, 0.0)),
                ([0, 1, 0], Complex64::new(0.0, 0.0)),
                ([1, 0, 0], x_coefficient),
            ],
        );
        let sphere = Sphere {
            center: [Bohr(0.0), Bohr(0.25 * TAU), Bohr(0.0)],
            radius: Bohr(0.2),
        };
        let field =
            RegionalScalarField::new(geometry(vec![sphere]), vec![muffin_tin], interstitial)
                .unwrap();
        let rotated = field.transformed(&transform).unwrap();

        let expected_fourier = x_coefficient * Complex64::new(0.0, -1.0);
        assert!(
            (rotated.interstitial().coefficient([0, 1, 0]).unwrap() - expected_fourier).norm()
                < 1.0e-13
        );
        let expected_m1 = Complex64::new(0.0, -1.0);
        for &value in rotated.muffin_tins()[0].field().channel(1, 1).unwrap() {
            assert!((value - expected_m1).norm() < 1.0e-13);
        }
        for &value in rotated.muffin_tins()[0].field().channel(1, -1).unwrap() {
            assert!((value - expected_m1).norm() < 1.0e-13);
        }
    }

    #[test]
    fn site_permutation_enters_the_scalar_group_average() {
        let cell = CrystalCell {
            lattice: direct_lattice(),
            positions: vec![[0.1, 0.0, 0.0], [0.4, 0.0, 0.0]],
            atomic_numbers: vec![6, 6],
        };
        let identity = CrystalSymmetryTransform::from_cell(
            SymmetryOperation {
                rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                translation: [0.0; 3],
                time_reversal: false,
            },
            &cell,
            Bohr(1.0e-12),
        )
        .unwrap();
        let inversion = CrystalSymmetryTransform::from_cell(
            SymmetryOperation {
                rotation: [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                translation: [0.5, 0.0, 0.0],
                time_reversal: false,
            },
            &cell,
            Bohr(1.0e-12),
        )
        .unwrap();
        assert_eq!(inversion.site_map(), &[1, 0]);

        let mesh = ExponentialMesh::new(Bohr(0.01), 0.1, 7).unwrap();
        let muffin_tin = |value| {
            MuffinTinField::new(
                mesh.clone(),
                SphereField::new(
                    HarmonicConvention::Complex,
                    [((0, 0), vec![Complex64::new(value, 0.0); mesh.len()])],
                )
                .unwrap(),
            )
            .unwrap()
        };
        let spheres = cell
            .positions
            .iter()
            .map(|position| Sphere {
                center: [Bohr(position[0] * TAU), Bohr(0.0), Bohr(0.0)],
                radius: Bohr(0.2),
            })
            .collect();
        let field = RegionalScalarField::new(
            geometry(spheres),
            vec![muffin_tin(1.0), muffin_tin(2.0)],
            interstitial(layout(&[[0; 3]]), [([0; 3], Complex64::new(0.0, 0.0))]),
        )
        .unwrap();
        let average = field.symmetry_average(&[identity, inversion]).unwrap();
        for muffin_tin in average.muffin_tins() {
            for &value in muffin_tin.field().channel(0, 0).unwrap() {
                assert!((value.re - 1.5).abs() < 1.0e-14);
                assert!(value.im.abs() < 1.0e-14);
            }
        }
    }

    #[test]
    fn antiunitary_identity_flips_axial_magnetization_only() {
        let cell = CrystalCell {
            lattice: direct_lattice(),
            positions: Vec::new(),
            atomic_numbers: Vec::new(),
        };
        let transform = CrystalSymmetryTransform::from_cell(
            SymmetryOperation {
                rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                translation: [0.0; 3],
                time_reversal: true,
            },
            &cell,
            Bohr(1.0e-12),
        )
        .unwrap();
        let layout = layout(&[[0; 3]]);
        let geometry = geometry(Vec::new());
        let charge = g0_scalar(&geometry, &layout, 4.0);
        let zero = charge.zero_like();
        let density = RegionalDensity::new(
            charge,
            [g0_scalar(&geometry, &layout, 2.0), zero.clone(), zero],
        )
        .unwrap();
        let transformed = density.transformed(&transform).unwrap();
        assert_eq!(
            transformed
                .charge()
                .interstitial()
                .coefficient([0; 3])
                .unwrap()
                .re,
            4.0
        );
        assert!(
            (transformed.magnetization()[0]
                .interstitial()
                .coefficient([0; 3])
                .unwrap()
                .re
                + 2.0)
                .abs()
                < 1.0e-14
        );
    }
}
