//! Regional muffin-tin/interstitial scalar fields and their physical metric.

use muffintin_core::{
    ExponentialMesh, FourierFieldError, FourierLayout, HermitianFourierField, InterstitialGeometry,
    MeshError, StepFunctionError,
};
use muffintin_lapw::{Collinear, InterstitialPotential, LapwError};
use muffintin_sphere::{SphereField, SphereFieldError};
use num_complex::Complex64;
use std::collections::BTreeMap;
use std::f64::consts::TAU;
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
        if let Some(actual) = field.sample_count() {
            if actual != mesh.len() {
                return Err(RegionalError::MuffinTinSampleCount {
                    expected: mesh.len(),
                    actual,
                });
            }
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
}

#[derive(Clone, Debug, PartialEq)]
struct RegionalFields {
    muffin_tins: Collinear<Vec<MuffinTinField>>,
    interstitial: Collinear<InterstitialField>,
}

impl RegionalFields {
    fn new(
        muffin_tins: Collinear<Vec<MuffinTinField>>,
        interstitial: Collinear<InterstitialField>,
    ) -> Result<Self, RegionalError> {
        if muffin_tins.up.len() != muffin_tins.down.len() {
            return Err(RegionalError::SpinMuffinTinCountMismatch {
                up: muffin_tins.up.len(),
                down: muffin_tins.down.len(),
            });
        }
        for (up, down) in muffin_tins.up.iter().zip(&muffin_tins.down) {
            up.require_same_layout(down)?;
        }
        if interstitial.up.layout() != interstitial.down.layout() {
            return Err(RegionalError::Fourier(FourierFieldError::LayoutMismatch));
        }
        Ok(Self {
            muffin_tins,
            interstitial,
        })
    }

    fn zero_like(&self) -> Self {
        Self {
            muffin_tins: Collinear::new(
                self.muffin_tins
                    .up
                    .iter()
                    .map(MuffinTinField::zero_like)
                    .collect(),
                self.muffin_tins
                    .down
                    .iter()
                    .map(MuffinTinField::zero_like)
                    .collect(),
            ),
            interstitial: Collinear::new(
                self.interstitial.up.zero_like(),
                self.interstitial.down.zero_like(),
            ),
        }
    }

    fn add_scaled(&mut self, scale: f64, other: &Self) -> Result<(), RegionalError> {
        self.require_same_layout(other)?;
        for (target, source) in self.muffin_tins.up.iter_mut().zip(&other.muffin_tins.up) {
            target.add_scaled(scale, source)?;
        }
        for (target, source) in self
            .muffin_tins
            .down
            .iter_mut()
            .zip(&other.muffin_tins.down)
        {
            target.add_scaled(scale, source)?;
        }
        self.interstitial
            .up
            .add_scaled(scale, &other.interstitial.up)?;
        self.interstitial
            .down
            .add_scaled(scale, &other.interstitial.down)?;
        Ok(())
    }

    fn difference(&self, other: &Self) -> Result<Self, RegionalError> {
        self.require_same_layout(other)?;
        Ok(Self {
            muffin_tins: Collinear::new(
                self.muffin_tins
                    .up
                    .iter()
                    .zip(&other.muffin_tins.up)
                    .map(|(left, right)| left.difference(right))
                    .collect::<Result<_, _>>()?,
                self.muffin_tins
                    .down
                    .iter()
                    .zip(&other.muffin_tins.down)
                    .map(|(left, right)| left.difference(right))
                    .collect::<Result<_, _>>()?,
            ),
            interstitial: Collinear::new(
                self.interstitial.up.difference(&other.interstitial.up)?,
                self.interstitial
                    .down
                    .difference(&other.interstitial.down)?,
            ),
        })
    }

    fn require_same_layout(&self, other: &Self) -> Result<(), RegionalError> {
        if self.muffin_tins.up.len() != other.muffin_tins.up.len()
            || self.muffin_tins.down.len() != other.muffin_tins.down.len()
        {
            return Err(RegionalError::RegionalMuffinTinCountMismatch);
        }
        for (left, right) in self.muffin_tins.up.iter().zip(&other.muffin_tins.up) {
            left.require_same_layout(right)?;
        }
        for (left, right) in self.muffin_tins.down.iter().zip(&other.muffin_tins.down) {
            left.require_same_layout(right)?;
        }
        if self.interstitial.up.layout() != other.interstitial.up.layout()
            || self.interstitial.down.layout() != other.interstitial.down.layout()
        {
            return Err(RegionalError::Fourier(FourierFieldError::LayoutMismatch));
        }
        Ok(())
    }
}

/// Explicit spin-up/spin-down density over muffin-tin and interstitial regions.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionalDensity {
    geometry: InterstitialGeometry,
    fields: RegionalFields,
}

impl RegionalDensity {
    pub fn new(
        geometry: InterstitialGeometry,
        muffin_tins: Collinear<Vec<MuffinTinField>>,
        interstitial: Collinear<InterstitialField>,
    ) -> Result<Self, RegionalError> {
        let fields = RegionalFields::new(muffin_tins, interstitial)?;
        if fields.muffin_tins.up.len() != geometry.spheres().len() {
            return Err(RegionalError::GeometryMuffinTinCountMismatch {
                geometry: geometry.spheres().len(),
                fields: fields.muffin_tins.up.len(),
            });
        }
        validate_reciprocal_volume(&geometry, fields.interstitial.up.layout())?;
        Ok(Self { geometry, fields })
    }

    pub const fn geometry(&self) -> &InterstitialGeometry {
        &self.geometry
    }

    pub const fn muffin_tins(&self) -> &Collinear<Vec<MuffinTinField>> {
        &self.fields.muffin_tins
    }

    pub const fn interstitial(&self) -> &Collinear<InterstitialField> {
        &self.fields.interstitial
    }

    pub fn zero_like(&self) -> Self {
        Self {
            geometry: self.geometry.clone(),
            fields: self.fields.zero_like(),
        }
    }

    pub fn add_scaled(&mut self, scale: f64, other: &Self) -> Result<(), RegionalError> {
        self.require_same_layout(other)?;
        self.fields.add_scaled(scale, &other.fields)
    }

    /// Form `self - other` after exact regional-layout validation.
    pub fn difference(&self, other: &Self) -> Result<Self, RegionalError> {
        self.require_same_layout(other)?;
        Ok(Self {
            geometry: self.geometry.clone(),
            fields: self.fields.difference(&other.fields)?,
        })
    }

    /// Physical collinear inner product, summed over both explicit spins.
    ///
    /// Muffin-tin terms are `sum_LM integral r^2 a*_LM b_LM dr`.
    /// Interstitial terms are the step-function convolution
    /// `Omega sum_GG' a*_G theta_(G-G') b_G'`.
    pub fn physical_inner_product(&self, other: &Self) -> Result<f64, RegionalError> {
        self.require_same_layout(other)?;
        let mut total = Complex64::new(0.0, 0.0);
        let mut absolute_scale = 0.0;

        for (left, right) in self
            .fields
            .muffin_tins
            .up
            .iter()
            .zip(&other.fields.muffin_tins.up)
            .chain(
                self.fields
                    .muffin_tins
                    .down
                    .iter()
                    .zip(&other.fields.muffin_tins.down),
            )
        {
            let contribution = muffin_tin_inner_product(left, right)?;
            total += contribution;
            absolute_scale += contribution.norm();
        }
        for (left, right) in [
            (&self.fields.interstitial.up, &other.fields.interstitial.up),
            (
                &self.fields.interstitial.down,
                &other.fields.interstitial.down,
            ),
        ] {
            let (contribution, term_scale) =
                interstitial_inner_product(&self.geometry, left, right)?;
            total += contribution;
            absolute_scale += term_scale;
        }

        real_metric_value(total, absolute_scale)
    }

    /// Root-mean-square magnitude of this regional residual per cell volume.
    pub fn residual_rms(&self) -> Result<f64, RegionalError> {
        let norm_squared = self.physical_inner_product(self)?;
        let tolerance = REALITY_TOLERANCE * norm_squared.abs().max(1.0);
        if norm_squared < -tolerance {
            return Err(RegionalError::NegativeNorm(norm_squared));
        }
        Ok((norm_squared.max(0.0) / self.geometry.cell_volume().get()).sqrt())
    }

    /// RMS of `self - other`, using the same physical metric.
    pub fn difference_rms(&self, other: &Self) -> Result<f64, RegionalError> {
        self.difference(other)?.residual_rms()
    }

    fn require_same_layout(&self, other: &Self) -> Result<(), RegionalError> {
        if self.geometry != other.geometry {
            return Err(RegionalError::InterstitialGeometryMismatch);
        }
        self.fields.require_same_layout(&other.fields)
    }
}

/// Explicit spin-up/spin-down regional potential.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionalPotential {
    fields: RegionalFields,
}

impl RegionalPotential {
    pub fn new(
        muffin_tins: Collinear<Vec<MuffinTinField>>,
        interstitial: Collinear<InterstitialField>,
    ) -> Result<Self, RegionalError> {
        Ok(Self {
            fields: RegionalFields::new(muffin_tins, interstitial)?,
        })
    }

    pub const fn muffin_tins(&self) -> &Collinear<Vec<MuffinTinField>> {
        &self.fields.muffin_tins
    }

    pub const fn interstitial(&self) -> &Collinear<InterstitialField> {
        &self.fields.interstitial
    }

    pub fn zero_like(&self) -> Self {
        Self {
            fields: self.fields.zero_like(),
        }
    }

    pub fn add_scaled(&mut self, scale: f64, other: &Self) -> Result<(), RegionalError> {
        self.fields.add_scaled(scale, &other.fields)
    }

    pub fn difference(&self, other: &Self) -> Result<Self, RegionalError> {
        Ok(Self {
            fields: self.fields.difference(&other.fields)?,
        })
    }

    /// Convert both explicit spin channels to the LAPW interstitial boundary.
    pub fn to_lapw_interstitial(&self) -> Result<Collinear<InterstitialPotential>, RegionalError> {
        Ok(Collinear::new(
            InterstitialPotential::try_from(&self.fields.interstitial.up)?,
            InterstitialPotential::try_from(&self.fields.interstitial.down)?,
        ))
    }
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
            let theta = geometry.coefficient(reciprocal.cartesian(difference))?;
            let term = volume * left_value.conj() * theta * right_value;
            total += term;
            absolute_scale += term.norm();
        }
    }
    Ok((total, absolute_scale))
}

fn real_metric_value(value: Complex64, absolute_scale: f64) -> Result<f64, RegionalError> {
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
    #[error("spin channels have different muffin-tin counts: up={up}, down={down}")]
    SpinMuffinTinCountMismatch { up: usize, down: usize },
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
    #[error("physical squared norm is negative: {0}")]
    NegativeNorm(f64),
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

    fn interstitial_only_density(
        geometry: InterstitialGeometry,
        up: InterstitialField,
        down: InterstitialField,
    ) -> RegionalDensity {
        let empty_muffin_tin = MuffinTinField::new(
            ExponentialMesh::new(Bohr(0.01), 0.1, 7).unwrap(),
            SphereField::new(HarmonicConvention::Real, []).unwrap(),
        )
        .unwrap();
        let muffin_tins = vec![empty_muffin_tin; geometry.spheres().len()];
        RegionalDensity::new(
            geometry,
            Collinear::new(muffin_tins.clone(), muffin_tins),
            Collinear::new(up, down),
        )
        .unwrap()
    }

    #[test]
    fn g0_metric_sums_explicit_spin_channels_and_rms_identity() {
        let layout = layout(&[[0; 3]]);
        let density = interstitial_only_density(
            geometry(Vec::new()),
            interstitial(layout.clone(), [([0; 3], Complex64::new(2.0, 0.0))]),
            interstitial(layout, [([0; 3], Complex64::new(3.0, 0.0))]),
        );
        let norm = density.physical_inner_product(&density).unwrap();
        assert!((norm - TAU.powi(3) * 13.0).abs() < 1.0e-11);
        assert!((density.residual_rms().unwrap() - 13.0_f64.sqrt()).abs() < 1.0e-13);
        assert_eq!(density.difference_rms(&density).unwrap(), 0.0);
    }

    #[test]
    fn plus_minus_g_metric_uses_step_function_convolution() {
        let indices = [[-1, 0, 0], [1, 0, 0]];
        let layout = layout(&indices);
        let values = [
            ([-1, 0, 0], Complex64::new(1.0, 0.0)),
            ([1, 0, 0], Complex64::new(1.0, 0.0)),
        ];
        let zeros = indices.map(|g| (g, Complex64::new(0.0, 0.0)));
        let sphere = Sphere {
            center: [Bohr(0.0); 3],
            radius: Bohr(0.5),
        };
        let geometry = geometry(vec![sphere]);
        let density = interstitial_only_density(
            geometry.clone(),
            interstitial(layout.clone(), values),
            interstitial(layout, zeros),
        );
        let theta_zero = geometry.coefficient([InverseBohr(0.0); 3]).unwrap().re;
        let theta_two = geometry
            .coefficient([InverseBohr(2.0), InverseBohr(0.0), InverseBohr(0.0)])
            .unwrap()
            .re;
        let expected = TAU.powi(3) * (2.0 * theta_zero + 2.0 * theta_two);
        let diagonal_only = TAU.powi(3) * 2.0 * theta_zero;
        let actual = density.physical_inner_product(&density).unwrap();
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
        let zero_mt = left.zero_like();
        let reciprocal_layout = layout(&[[0; 3]]);
        let zero_interstitial =
            interstitial(reciprocal_layout, [([0; 3], Complex64::new(0.0, 0.0))]);
        let geometry = geometry(vec![Sphere {
            center: [Bohr(0.0); 3],
            radius: Bohr(1.0),
        }]);
        let left_density = RegionalDensity::new(
            geometry.clone(),
            Collinear::new(vec![left], vec![zero_mt.clone()]),
            Collinear::new(zero_interstitial.clone(), zero_interstitial.clone()),
        )
        .unwrap();
        let right_density = RegionalDensity::new(
            geometry,
            Collinear::new(vec![right], vec![zero_mt]),
            Collinear::new(zero_interstitial.clone(), zero_interstitial),
        )
        .unwrap();
        let r_squared: Vec<_> = mesh
            .radii()
            .iter()
            .map(|radius| radius.get().powi(2))
            .collect();
        let expected = 6.0 * mesh.integrate(&r_squared).unwrap();
        let actual = left_density.physical_inner_product(&right_density).unwrap();
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
        let first = interstitial_only_density(geometry(Vec::new()), g0.clone(), g0);
        let second = interstitial_only_density(geometry(Vec::new()), three.clone(), three);
        assert!(matches!(
            first.difference(&second),
            Err(RegionalError::Fourier(FourierFieldError::LayoutMismatch))
        ));
    }

    #[test]
    fn regional_potential_converts_both_spin_channels_to_lapw() {
        let layout = layout(&[[0; 3]]);
        let potential = RegionalPotential::new(
            Collinear::new(Vec::new(), Vec::new()),
            Collinear::new(
                interstitial(layout.clone(), [([0; 3], Complex64::new(0.25, 0.0))]),
                interstitial(layout, [([0; 3], Complex64::new(-0.5, 0.0))]),
            ),
        )
        .unwrap();
        let converted = potential.to_lapw_interstitial().unwrap();
        assert_eq!(converted.up.coefficient([0; 3]), Complex64::new(0.25, 0.0));
        assert_eq!(
            converted.down.coefficient([0; 3]),
            Complex64::new(-0.5, 0.0)
        );
    }
}
