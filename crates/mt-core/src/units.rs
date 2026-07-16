//! Strong scalar types for the internal Hartree atomic-unit convention.

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// One Hartree in electron-volts (CODATA 2018 exact-to-shown-digits value).
pub const HARTREE_TO_EV: f64 = 27.211_386_245_988;
/// One Bohr radius in Angstrom.
pub const BOHR_TO_ANGSTROM: f64 = 0.529_177_210_903;

macro_rules! scalar_unit {
    ($name:ident, $label:literal) => {
        #[doc = concat!("A scalar measured in ", $label, ".")]
        #[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(pub f64);

        impl $name {
            /// Construct a value after checking that it is finite.
            pub fn checked(value: f64) -> Option<Self> {
                value.is_finite().then_some(Self(value))
            }

            /// Return the raw value in the named internal unit.
            pub const fn get(self) -> f64 {
                self.0
            }
        }

        impl Add for $name {
            type Output = Self;
            fn add(self, rhs: Self) -> Self::Output {
                Self(self.0 + rhs.0)
            }
        }

        impl AddAssign for $name {
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }

        impl Sub for $name {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self::Output {
                Self(self.0 - rhs.0)
            }
        }

        impl SubAssign for $name {
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }

        impl Mul<f64> for $name {
            type Output = Self;
            fn mul(self, rhs: f64) -> Self::Output {
                Self(self.0 * rhs)
            }
        }

        impl Mul<$name> for f64 {
            type Output = $name;
            fn mul(self, rhs: $name) -> Self::Output {
                $name(self * rhs.0)
            }
        }

        impl Div<f64> for $name {
            type Output = Self;
            fn div(self, rhs: f64) -> Self::Output {
                Self(self.0 / rhs)
            }
        }

        impl Div for $name {
            type Output = f64;
            fn div(self, rhs: Self) -> Self::Output {
                self.0 / rhs.0
            }
        }

        impl Neg for $name {
            type Output = Self;
            fn neg(self) -> Self::Output {
                Self(-self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{} {}", self.0, $label)
            }
        }
    };
}

scalar_unit!(Hartree, "Ha");
scalar_unit!(Bohr, "bohr");
scalar_unit!(InverseBohr, "bohr^-1");
scalar_unit!(VolumeBohr3, "bohr^3");

impl Hartree {
    /// Convert an explicitly labelled electron-volt value to Hartree.
    pub fn from_ev(ev: f64) -> Self {
        Self(ev / HARTREE_TO_EV)
    }

    /// Convert to electron-volts for output.
    pub fn to_ev(self) -> f64 {
        self.0 * HARTREE_TO_EV
    }

    /// Convert an explicitly labelled Rydberg value to Hartree.
    pub fn from_rydberg(rydberg: f64) -> Self {
        Self(rydberg * 0.5)
    }

    /// Convert to Rydberg for output.
    pub fn to_rydberg(self) -> f64 {
        self.0 * 2.0
    }
}

impl Bohr {
    /// Convert an explicitly labelled Angstrom value to Bohr.
    pub fn from_angstrom(angstrom: f64) -> Self {
        Self(angstrom / BOHR_TO_ANGSTROM)
    }

    /// Convert to Angstrom for output.
    pub fn to_angstrom(self) -> f64 {
        self.0 * BOHR_TO_ANGSTROM
    }

    /// Cube this length to obtain a volume.
    pub fn cubed(self) -> VolumeBohr3 {
        VolumeBohr3(self.0.powi(3))
    }
}

impl InverseBohr {
    /// Square the reciprocal length, useful for Cartesian norm comparisons.
    pub fn squared(self) -> f64 {
        self.0 * self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_conversions_round_trip() {
        let e = Hartree::from_ev(HARTREE_TO_EV);
        assert_eq!(e, Hartree(1.0));
        assert_eq!(e.to_rydberg(), 2.0);
        let r = Bohr::from_angstrom(BOHR_TO_ANGSTROM);
        assert_eq!(r, Bohr(1.0));
    }
}
