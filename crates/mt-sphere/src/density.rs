//! Scalar pair-density projection in normalized spherical harmonics.

use super::{
    HarmonicConvention, SphereField, SphereFieldError, SphereOrbital, SpinorSphereOrbital,
    complex_matrix_gaunt, magnetic_phase,
};
use muffintin_core::{
    ExponentialMesh, Lm, RelativisticChannel, SpinProjection, real_gaunt, spinor_gaunt,
};
use muffintin_radial::RadialComponents;
use num_complex::Complex64;
use thiserror::Error;

impl SphereField {
    /// Iterate over stored channels in deterministic `(L,M)` order.
    pub fn channels(&self) -> impl Iterator<Item = (Lm, &[Complex64])> {
        self.channels
            .iter()
            .map(|(&channel, values)| (channel, values.as_slice()))
    }

    /// A zero field with the same convention, radial size, and channel set.
    pub fn zero_like(&self) -> Self {
        Self {
            convention: self.convention,
            sample_count: self.sample_count,
            channels: self
                .channels
                .keys()
                .copied()
                .map(|channel| {
                    (
                        channel,
                        vec![Complex64::new(0.0, 0.0); self.sample_count.unwrap_or(0)],
                    )
                })
                .collect(),
        }
    }

    /// Accumulate another field after checking its exact representation.
    ///
    /// Complex scaling is useful when conjugate orbital-pair contributions
    /// are assembled.  A real-tesseral result still passes through the normal
    /// constructor and therefore rejects an imaginary coefficient.
    pub fn add_scaled(&mut self, scale: Complex64, other: &Self) -> Result<(), SphereFieldError> {
        if !scale.re.is_finite() || !scale.im.is_finite() {
            return Err(SphereFieldError::InvalidScale(scale));
        }
        self.require_same_representation(other)?;
        let channels = self.channels.iter().zip(&other.channels).map(
            |((&channel, left), (&right_channel, right))| {
                debug_assert_eq!(channel, right_channel);
                (
                    (channel.l, channel.m),
                    left.iter()
                        .zip(right)
                        .map(|(&left, &right)| left + scale * right)
                        .collect::<Vec<_>>(),
                )
            },
        );
        let updated = Self::new(self.convention, channels)?;
        self.channels = updated.channels;
        Ok(())
    }

    /// Form `self - other` on the same harmonic and radial representation.
    pub fn difference(&self, other: &Self) -> Result<Self, SphereFieldError> {
        self.require_same_representation(other)?;
        let mut result = self.clone();
        result.add_scaled(Complex64::new(-1.0, 0.0), other)?;
        Ok(result)
    }

    /// Check that the expansion represents a physically real scalar field.
    ///
    /// Complex harmonics require
    /// `f_(L,-M) = (-1)^M conj(f_(L,M))` sample by sample.  Omitted channels
    /// are interpreted as zero.  Real-tesseral fields satisfy the condition by
    /// construction.
    pub fn validate_physical_reality(&self, tolerance: f64) -> Result<(), SphereFieldError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(SphereFieldError::InvalidRealityTolerance(tolerance));
        }
        if self.convention == HarmonicConvention::Real {
            return Ok(());
        }
        for (&channel, values) in &self.channels {
            let partner = Lm::new(channel.l, -channel.m)
                .expect("negating a validated magnetic channel remains valid");
            let partner_values = self.channels.get(&partner);
            for (index, &value) in values.iter().enumerate() {
                let opposite =
                    partner_values.map_or(Complex64::new(0.0, 0.0), |samples| samples[index]);
                let expected = magnetic_phase(channel.m) * value.conj();
                if (opposite - expected).norm() > tolerance * (1.0 + expected.norm()) {
                    return Err(SphereFieldError::PhysicalReality {
                        l: channel.l,
                        m: channel.m,
                        index,
                        expected,
                        actual: opposite,
                    });
                }
            }
        }
        Ok(())
    }

    fn require_same_representation(&self, other: &Self) -> Result<(), SphereFieldError> {
        if self.convention != other.convention {
            return Err(SphereFieldError::ConventionMismatch {
                left: self.convention,
                right: other.convention,
            });
        }
        if self.sample_count != other.sample_count {
            return Err(SphereFieldError::SampleCountMismatch {
                left: self.sample_count,
                right: other.sample_count,
            });
        }
        if !self.channels.keys().eq(other.channels.keys()) {
            return Err(SphereFieldError::ChannelLayoutMismatch);
        }
        Ok(())
    }
}

/// Project `conj(left) * right` into normalized complex harmonics.
///
/// [`SphereOrbital`] components are reduced radial functions, so the returned
/// physical density contains `(p_left p_right + Q_left Q_right) / r^2`.
pub fn project_orbital_pair_density(
    mesh: &ExponentialMesh,
    left: &SphereOrbital,
    right: &SphereOrbital,
) -> Result<SphereField, DensityProjectionError> {
    project_orbital_pair_density_with_convention(mesh, left, right, HarmonicConvention::Complex)
}

/// Project an orbital pair using an explicit complex or real harmonic basis.
///
/// [`project_orbital_pair_density`] is the complex-harmonic convenience path
/// used by LAPW eigenstates. This form also covers real tesseral orbitals and
/// uses the crate's matching [`real_gaunt`] convention throughout.
pub fn project_orbital_pair_density_with_convention(
    mesh: &ExponentialMesh,
    left: &SphereOrbital,
    right: &SphereOrbital,
    convention: HarmonicConvention,
) -> Result<SphereField, DensityProjectionError> {
    validate_scalar_lengths(mesh, left, right)?;
    let large_left = left.large_component();
    let large_right = right.large_component();
    let small_left = left.small_component();
    let small_right = right.small_component();
    let radial = mesh
        .radii()
        .iter()
        .enumerate()
        .map(|(index, radius)| {
            let mut product = large_left[index] * large_right[index];
            if let (Some(left), Some(right)) = (small_left, small_right) {
                product += left[index] * right[index];
            }
            product / (radius.get() * radius.get())
        })
        .collect::<Vec<_>>();

    let left_angular = left.angular();
    let right_angular = right.angular();
    let l_max = left_angular.l + right_angular.l;
    let mut channels = Vec::new();
    for l in 0..=l_max {
        for m in -(l as i32)..=l as i32 {
            let angular = match convention {
                HarmonicConvention::Complex => {
                    let field = Lm::new(l, -m).expect("loop bounds validate magnetic channel");
                    magnetic_phase(m) * complex_matrix_gaunt(left_angular, field, right_angular)
                }
                HarmonicConvention::Real => real_gaunt(
                    left_angular.l,
                    l,
                    right_angular.l,
                    left_angular.m,
                    m,
                    right_angular.m,
                ),
            };
            channels.push((
                (l, m),
                radial
                    .iter()
                    .map(|&value| Complex64::new(angular * value, 0.0))
                    .collect(),
            ));
        }
    }
    SphereField::new(convention, channels).map_err(Into::into)
}

/// Project a scalar Dirac-spinor pair density into normalized harmonics.
///
/// The large `P` product is projected through `Omega_kappa`, and the small
/// `Q` product through `Omega_-kappa`.  No `PQ` or `QP` term enters a scalar
/// density.
pub fn project_spinor_pair_density(
    mesh: &ExponentialMesh,
    left: &SpinorSphereOrbital,
    right: &SpinorSphereOrbital,
) -> Result<SphereField, DensityProjectionError> {
    validate_spinor_lengths(mesh, left, right)?;
    let left_channel = left.channel();
    let right_channel = right.channel();
    let large_l_max = left_channel.kappa().large_l() + right_channel.kappa().large_l();
    let small_l_max = left_channel.kappa().small_l() + right_channel.kappa().small_l();
    let l_max = large_l_max.max(small_l_max);
    let mut channels = Vec::new();
    for l in 0..=l_max {
        for m in -(l as i32)..=l as i32 {
            let expansion_channel = Lm::new(l, -m).expect("loop bounds validate magnetic channel");
            let phase = magnetic_phase(m);
            let pp_angular = phase * spinor_gaunt(left_channel, expansion_channel, right_channel);
            let qq_angular = phase
                * spinor_gaunt(
                    left_channel.opposite_kappa(),
                    expansion_channel,
                    right_channel.opposite_kappa(),
                );
            let values = mesh
                .radii()
                .iter()
                .enumerate()
                .map(|(index, radius)| {
                    let value = (pp_angular * left.p()[index] * right.p()[index]
                        + qq_angular * left.q()[index] * right.q()[index])
                        / (radius.get() * radius.get());
                    Complex64::new(value, 0.0)
                })
                .collect();
            channels.push(((l, m), values));
        }
    }
    SphereField::new(HarmonicConvention::Complex, channels).map_err(Into::into)
}

/// Charge and Cartesian Pauli-spin fields of one four-component orbital pair.
///
/// Every component is the normalized-harmonic projection of
/// `left^dagger (I, sigma_x, sigma_y, sigma_z) right`. The large `P` and small
/// `Q` sectors are both retained, with `Omega_kappa` and `Omega_-kappa`
/// respectively; a local Pauli operator never introduces `PQ` or `QP` terms.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorPairDensity {
    charge: SphereField,
    spin: [SphereField; 3],
}

impl SpinorPairDensity {
    pub const fn charge(&self) -> &SphereField {
        &self.charge
    }

    /// Cartesian spin fields in `[x, y, z]` order.
    pub const fn spin(&self) -> &[SphereField; 3] {
        &self.spin
    }
}

/// Project the full charge/Pauli-spin density of a four-component pair.
pub fn project_spinor_pair_density_components(
    mesh: &ExponentialMesh,
    left: &SpinorSphereOrbital,
    right: &SpinorSphereOrbital,
) -> Result<SpinorPairDensity, DensityProjectionError> {
    validate_spinor_lengths(mesh, left, right)?;
    Ok(SpinorPairDensity {
        charge: project_spinor_pair_density(mesh, left, right)?,
        spin: [
            project_spinor_pair_pauli_density(mesh, left, right, 0)?,
            project_spinor_pair_pauli_density(mesh, left, right, 1)?,
            project_spinor_pair_pauli_density(mesh, left, right, 2)?,
        ],
    })
}

fn project_spinor_pair_pauli_density(
    mesh: &ExponentialMesh,
    left: &SpinorSphereOrbital,
    right: &SpinorSphereOrbital,
    axis: usize,
) -> Result<SphereField, DensityProjectionError> {
    let left_channel = left.channel();
    let right_channel = right.channel();
    let large_l_max = left_channel.kappa().large_l() + right_channel.kappa().large_l();
    let small_l_max = left_channel.kappa().small_l() + right_channel.kappa().small_l();
    let l_max = large_l_max.max(small_l_max);
    let mut channels = Vec::new();
    for l in 0..=l_max {
        for m in -(l as i32)..=l as i32 {
            let expansion_channel = Lm::new(l, -m).expect("loop bounds validate magnetic channel");
            let phase = magnetic_phase(m);
            let pp_angular =
                phase * pauli_angular(left_channel, expansion_channel, right_channel, axis);
            let qq_angular = phase
                * pauli_angular(
                    left_channel.opposite_kappa(),
                    expansion_channel,
                    right_channel.opposite_kappa(),
                    axis,
                );
            let values = mesh
                .radii()
                .iter()
                .enumerate()
                .map(|(index, radius)| {
                    (pp_angular * left.p()[index] * right.p()[index]
                        + qq_angular * left.q()[index] * right.q()[index])
                        / (radius.get() * radius.get())
                })
                .collect();
            channels.push(((l, m), values));
        }
    }
    SphereField::new(HarmonicConvention::Complex, channels).map_err(Into::into)
}

fn pauli_angular(
    left: RelativisticChannel,
    field: Lm,
    right: RelativisticChannel,
    axis: usize,
) -> Complex64 {
    let mut value = Complex64::new(0.0, 0.0);
    for left_term in left.spinor_harmonic_terms().into_iter().flatten() {
        for right_term in right.spinor_harmonic_terms().into_iter().flatten() {
            value += left_term.coefficient
                * right_term.coefficient
                * pauli(axis, left_term.spin, right_term.spin)
                * complex_matrix_gaunt(left_term.orbital, field, right_term.orbital);
        }
    }
    value
}

fn pauli(axis: usize, left: SpinProjection, right: SpinProjection) -> Complex64 {
    use SpinProjection::{Down, Up};
    match (axis, left, right) {
        (0, Up, Down) | (0, Down, Up) => Complex64::new(1.0, 0.0),
        (1, Up, Down) => Complex64::new(0.0, -1.0),
        (1, Down, Up) => Complex64::new(0.0, 1.0),
        (2, Up, Up) => Complex64::new(1.0, 0.0),
        (2, Down, Down) => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, 0.0),
    }
}

fn validate_scalar_lengths(
    mesh: &ExponentialMesh,
    left: &SphereOrbital,
    right: &SphereOrbital,
) -> Result<(), DensityProjectionError> {
    validate_length(
        mesh,
        left.large_component(),
        DensityOperand::Left,
        DensityComponent::Large,
    )?;
    validate_length(
        mesh,
        right.large_component(),
        DensityOperand::Right,
        DensityComponent::Large,
    )?;
    if let Some(values) = left.small_component() {
        validate_length(mesh, values, DensityOperand::Left, DensityComponent::Small)?;
    }
    if let Some(values) = right.small_component() {
        validate_length(mesh, values, DensityOperand::Right, DensityComponent::Small)?;
    }
    Ok(())
}

fn validate_spinor_lengths(
    mesh: &ExponentialMesh,
    left: &SpinorSphereOrbital,
    right: &SpinorSphereOrbital,
) -> Result<(), DensityProjectionError> {
    validate_length(
        mesh,
        left.p(),
        DensityOperand::Left,
        DensityComponent::Large,
    )?;
    validate_length(
        mesh,
        left.q(),
        DensityOperand::Left,
        DensityComponent::Small,
    )?;
    validate_length(
        mesh,
        right.p(),
        DensityOperand::Right,
        DensityComponent::Large,
    )?;
    validate_length(
        mesh,
        right.q(),
        DensityOperand::Right,
        DensityComponent::Small,
    )
}

fn validate_length(
    mesh: &ExponentialMesh,
    values: &[f64],
    operand: DensityOperand,
    component: DensityComponent,
) -> Result<(), DensityProjectionError> {
    if values.len() != mesh.len() {
        Err(DensityProjectionError::MeshLength {
            operand,
            component,
            expected: mesh.len(),
            actual: values.len(),
        })
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DensityOperand {
    Left,
    Right,
}

impl std::fmt::Display for DensityOperand {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Left => formatter.write_str("left"),
            Self::Right => formatter.write_str("right"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DensityComponent {
    Large,
    Small,
}

impl std::fmt::Display for DensityComponent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Large => formatter.write_str("large"),
            Self::Small => formatter.write_str("small"),
        }
    }
}

/// Invalid orbital-pair density request.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum DensityProjectionError {
    #[error(
        "{operand} orbital {component} component has {actual} samples, but mesh has {expected}"
    )]
    MeshLength {
        operand: DensityOperand,
        component: DensityComponent,
        expected: usize,
        actual: usize,
    },
    #[error(transparent)]
    Field(#[from] SphereFieldError),
}
