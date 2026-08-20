//! Angular and radial algebra inside a muffin-tin sphere.
//!
//! A [`SphereField`] stores radial expansion coefficients by `(L,M)`
//! channel.  A [`SphereOrbital`] combines an angular channel with reduced
//! large and optional small radial components.  [`matrix_element`] composes
//! the Gaunt coefficients from `mt-core` with the radial quadrature from
//! `mt-radial`.

#![forbid(unsafe_code)]

use mt_core::{ExponentialMesh, Lm, gaunt, real_gaunt};
use mt_radial::{RadialComponents, RadialIntegralError, RadialIntegralKernel, radial_integral};
use num_complex::Complex64;
use std::collections::BTreeMap;
use thiserror::Error;

/// Spherical-harmonic basis used for a field and its orbital labels.
///
/// There is deliberately no default: callers must state whether signed `m`
/// denotes complex Condon--Shortley harmonics or real tesseral harmonics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HarmonicConvention {
    /// Complex Condon--Shortley `Y_lm` harmonics.
    Complex,
    /// Real tesseral `R_lm` harmonics in the `mt-core` convention.
    Real,
}

/// An `(L,M)`-resolved scalar field sampled on one radial mesh.
///
/// Coefficients multiply normalized harmonics.  In particular, a constant
/// physical value `v` is represented by the `(0,0)` coefficient
/// `sqrt(4 pi) v`.  Complex coefficients are supported for the complex
/// convention; real-convention coefficients are required to have zero
/// imaginary part.
#[derive(Clone, Debug, PartialEq)]
pub struct SphereField {
    convention: HarmonicConvention,
    sample_count: Option<usize>,
    channels: BTreeMap<Lm, Vec<Complex64>>,
}

impl SphereField {
    /// Construct a field from normalized-harmonic expansion coefficients.
    ///
    /// Every channel must have the same number of radial samples.  For the
    /// real convention, use [`Self::from_real_channels`] unless complex-valued
    /// input is already convenient.
    pub fn new<I>(convention: HarmonicConvention, channels: I) -> Result<Self, SphereFieldError>
    where
        I: IntoIterator<Item = ((u32, i32), Vec<Complex64>)>,
    {
        let mut result = BTreeMap::new();
        let mut sample_count = None;
        for ((l, m), values) in channels {
            let channel = Lm::new(l, m).map_err(|_| SphereFieldError::InvalidChannel { l, m })?;
            if result.contains_key(&channel) {
                return Err(SphereFieldError::DuplicateChannel { l, m });
            }
            let expected = *sample_count.get_or_insert(values.len());
            if values.len() != expected {
                return Err(SphereFieldError::ChannelLength {
                    l,
                    m,
                    expected,
                    actual: values.len(),
                });
            }
            for (index, &value) in values.iter().enumerate() {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(SphereFieldError::NonFiniteSample { l, m, index, value });
                }
                if convention == HarmonicConvention::Real && value.im != 0.0 {
                    return Err(SphereFieldError::ComplexSampleInRealConvention {
                        l,
                        m,
                        index,
                        imaginary: value.im,
                    });
                }
            }
            result.insert(channel, values);
        }
        Ok(Self {
            convention,
            sample_count,
            channels: result,
        })
    }

    /// Construct a real-tesseral field from real radial coefficients.
    pub fn from_real_channels<I>(channels: I) -> Result<Self, SphereFieldError>
    where
        I: IntoIterator<Item = ((u32, i32), Vec<f64>)>,
    {
        Self::new(
            HarmonicConvention::Real,
            channels.into_iter().map(|(lm, values)| {
                (
                    lm,
                    values
                        .into_iter()
                        .map(|value| Complex64::new(value, 0.0))
                        .collect(),
                )
            }),
        )
    }

    /// Harmonic basis used by all channels.
    pub const fn convention(&self) -> HarmonicConvention {
        self.convention
    }

    /// Common number of radial samples, or `None` for an empty field.
    pub const fn sample_count(&self) -> Option<usize> {
        self.sample_count
    }

    /// Number of angular channels stored in the field.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Borrow a channel's radial coefficients.
    pub fn channel(&self, l: u32, m: i32) -> Option<&[Complex64]> {
        Lm::new(l, m)
            .ok()
            .and_then(|channel| self.channels.get(&channel).map(Vec::as_slice))
    }
}

/// An angular orbital carrying owned reduced radial components.
#[derive(Clone, Debug, PartialEq)]
pub struct SphereOrbital {
    angular: Lm,
    large: Vec<f64>,
    small: Option<Vec<f64>>,
}

impl SphereOrbital {
    /// Construct an orbital and validate its angular label and component sizes.
    pub fn new(
        l: u32,
        m: i32,
        large: Vec<f64>,
        small: Option<Vec<f64>>,
    ) -> Result<Self, SphereOrbitalError> {
        let angular =
            Lm::new(l, m).map_err(|_| SphereOrbitalError::InvalidAngularChannel { l, m })?;
        if let Some(values) = &small {
            if values.len() != large.len() {
                return Err(SphereOrbitalError::SmallComponentLength {
                    expected: large.len(),
                    actual: values.len(),
                });
            }
        }
        Ok(Self {
            angular,
            large,
            small,
        })
    }

    /// Validated `(l,m)` orbital label.
    pub const fn angular(&self) -> Lm {
        self.angular
    }
}

impl RadialComponents for SphereOrbital {
    fn large_component(&self) -> &[f64] {
        &self.large
    }

    fn small_component(&self) -> Option<&[f64]> {
        self.small.as_deref()
    }
}

/// Compose angular Gaunt factors and radial integrals for a sphere matrix element.
///
/// The field and both orbitals must be sampled on `mesh`.  Complex-harmonic
/// coefficients produce a complex result; the real-tesseral path returns a
/// [`Complex64`] whose imaginary part is exactly zero.
pub fn matrix_element(
    mesh: &ExponentialMesh,
    left: &SphereOrbital,
    field: &SphereField,
    right: &SphereOrbital,
) -> Result<Complex64, MatrixElementError> {
    validate_orbital_length(mesh, left, Operand::Left)?;
    validate_orbital_length(mesh, right, Operand::Right)?;
    if let Some(actual) = field.sample_count {
        if actual != mesh.len() {
            return Err(MatrixElementError::FieldMeshLength {
                expected: mesh.len(),
                actual,
            });
        }
    }

    let mut result = Complex64::new(0.0, 0.0);
    for (&channel, values) in &field.channels {
        let angular = match field.convention {
            HarmonicConvention::Complex => {
                complex_matrix_gaunt(left.angular, channel, right.angular)
            }
            HarmonicConvention::Real => real_gaunt(
                left.angular.l,
                channel.l,
                right.angular.l,
                left.angular.m,
                channel.m,
                right.angular.m,
            ),
        };
        if angular == 0.0 {
            continue;
        }

        let real_values: Vec<f64> = values.iter().map(|value| value.re).collect();
        let real = channel_integral(mesh, left, right, channel, &real_values)?;
        if field.convention == HarmonicConvention::Real {
            result.re += angular * real;
            continue;
        }

        let imaginary_values: Vec<f64> = values.iter().map(|value| value.im).collect();
        let imaginary = channel_integral(mesh, left, right, channel, &imaginary_values)?;
        result += Complex64::new(real, imaginary) * angular;
    }
    Ok(result)
}

fn complex_matrix_gaunt(left: Lm, field: Lm, right: Lm) -> f64 {
    // mt-core's SPEX coefficient is
    //   integral conj(Y_left) Y_field conj(Y_third).
    // conj(Y_l,-m) = (-1)^m Y_lm, hence the phase below converts its third
    // argument into the unconjugated right orbital required by <left|V|right>.
    magnetic_phase(right.m) * gaunt(left.l, field.l, right.l, left.m, field.m, -right.m)
}

fn magnetic_phase(m: i32) -> f64 {
    if m.unsigned_abs() % 2 == 0 { 1.0 } else { -1.0 }
}

fn channel_integral(
    mesh: &ExponentialMesh,
    left: &SphereOrbital,
    right: &SphereOrbital,
    channel: Lm,
    values: &[f64],
) -> Result<f64, MatrixElementError> {
    radial_integral(
        mesh,
        left,
        right,
        RadialIntegralKernel::PotentialMultipole {
            angular_l: channel.l,
            angular_m: channel.m,
            values,
        },
    )
    .map_err(|source| MatrixElementError::RadialIntegral {
        l: channel.l,
        m: channel.m,
        source,
    })
}

fn validate_orbital_length(
    mesh: &ExponentialMesh,
    orbital: &SphereOrbital,
    operand: Operand,
) -> Result<(), MatrixElementError> {
    let expected = mesh.len();
    if orbital.large.len() != expected {
        return Err(MatrixElementError::OrbitalMeshLength {
            operand,
            component: Component::Large,
            expected,
            actual: orbital.large.len(),
        });
    }
    if let Some(small) = &orbital.small {
        if small.len() != expected {
            return Err(MatrixElementError::OrbitalMeshLength {
                operand,
                component: Component::Small,
                expected,
                actual: small.len(),
            });
        }
    }
    Ok(())
}

/// Invalid field construction.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SphereFieldError {
    /// A channel has `|M| > L`.
    #[error("invalid field channel (L={l}, M={m})")]
    InvalidChannel { l: u32, m: i32 },
    /// A channel label occurs more than once.
    #[error("duplicate field channel (L={l}, M={m})")]
    DuplicateChannel { l: u32, m: i32 },
    /// Radial channel lengths are inconsistent.
    #[error("field channel (L={l}, M={m}) has {actual} radial samples, expected {expected}")]
    ChannelLength {
        l: u32,
        m: i32,
        expected: usize,
        actual: usize,
    },
    /// A complex coefficient was supplied for the real-tesseral convention.
    #[error(
        "real-harmonic field channel (L={l}, M={m}) has imaginary sample {imaginary} at index {index}"
    )]
    ComplexSampleInRealConvention {
        l: u32,
        m: i32,
        index: usize,
        imaginary: f64,
    },
    /// A radial expansion coefficient is not finite.
    #[error("field channel (L={l}, M={m}) has non-finite sample {value} at index {index}")]
    NonFiniteSample {
        l: u32,
        m: i32,
        index: usize,
        value: Complex64,
    },
}

/// Invalid orbital construction.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SphereOrbitalError {
    /// The orbital has `|m| > l`.
    #[error("invalid orbital channel (l={l}, m={m})")]
    InvalidAngularChannel { l: u32, m: i32 },
    /// The optional small component does not match the large component.
    #[error("small component has {actual} radial samples, expected {expected}")]
    SmallComponentLength { expected: usize, actual: usize },
}

/// Side of a matrix element whose data failed validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operand {
    Left,
    Right,
}

impl std::fmt::Display for Operand {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Left => formatter.write_str("left"),
            Self::Right => formatter.write_str("right"),
        }
    }
}

/// Reduced radial component whose data failed validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Component {
    Large,
    Small,
}

impl std::fmt::Display for Component {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Large => formatter.write_str("large"),
            Self::Small => formatter.write_str("small"),
        }
    }
}

/// Invalid sphere matrix-element request.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum MatrixElementError {
    /// Field samples do not lie on the supplied mesh.
    #[error("field channels have {actual} radial samples, but mesh has {expected}")]
    FieldMeshLength { expected: usize, actual: usize },
    /// An orbital component does not lie on the supplied mesh.
    #[error(
        "{operand} orbital {component} component has {actual} radial samples, but mesh has {expected}"
    )]
    OrbitalMeshLength {
        operand: Operand,
        component: Component,
        expected: usize,
        actual: usize,
    },
    /// The radial primitive rejected one field channel.
    #[error("radial integral failed for field channel (L={l}, M={m}): {source}")]
    RadialIntegral {
        l: u32,
        m: i32,
        #[source]
        source: RadialIntegralError,
    },
}
