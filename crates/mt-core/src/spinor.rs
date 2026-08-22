//! Dirac spin-angular quantum numbers and spinor spherical harmonics.
//!
//! The phase convention is the Condon--Shortley Clebsch--Gordan expansion
//!
//! `Omega_(kappa,mu) = sum_s <l,mu-s;1/2,s|j,mu> Y_(l,mu-s) chi_s`.

use thiserror::Error;

use crate::{Lm, gaunt};

/// A validated, nonzero Dirac angular quantum number.
///
/// `kappa = -(l + 1)` labels `j = l + 1/2`, while `kappa = l` labels
/// `j = l - 1/2`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Kappa(i32);

impl Kappa {
    pub fn new(value: i32) -> Result<Self, KappaError> {
        match value {
            0 => Err(KappaError::Zero),
            i32::MIN => Err(KappaError::OutOfRange(value)),
            _ => Ok(Self(value)),
        }
    }

    pub const fn get(self) -> i32 {
        self.0
    }

    /// Orbital angular momentum of `Omega_kappa` (the large component).
    pub const fn large_l(self) -> u32 {
        if self.0 < 0 {
            self.0.unsigned_abs() - 1
        } else {
            self.0 as u32
        }
    }

    /// Orbital angular momentum of `Omega_-kappa` (the small component).
    pub const fn small_l(self) -> u32 {
        if self.0 < 0 {
            self.0.unsigned_abs()
        } else {
            self.0 as u32 - 1
        }
    }

    /// `2j`, represented exactly as an odd integer.
    pub const fn twice_j(self) -> u32 {
        2 * self.0.unsigned_abs() - 1
    }

    /// Magnetic degeneracy `2j + 1 = 2 |kappa|`.
    pub const fn degeneracy(self) -> u32 {
        2 * self.0.unsigned_abs()
    }

    pub const fn angular_contract(self) -> DiracAngularContract {
        DiracAngularContract {
            kappa: self,
            large_l: self.large_l(),
            small_l: self.small_l(),
            twice_j: self.twice_j(),
            degeneracy: self.degeneracy(),
        }
    }

    /// The channel with `-kappa`, which has the same `j` and mu range.
    pub const fn opposite(self) -> Self {
        Self(-self.0)
    }

    /// Enumerate allowed `2mu` values in ascending order.
    pub fn twice_mu_values(self) -> impl Iterator<Item = TwiceMu> {
        let twice_j = i64::from(self.twice_j());
        (-twice_j..=twice_j).step_by(2).map(TwiceMu)
    }

    /// Enumerate complete `(kappa,mu)` channels in ascending `mu` order.
    pub fn channels(self) -> impl Iterator<Item = RelativisticChannel> {
        self.twice_mu_values()
            .map(move |twice_mu| RelativisticChannel {
                kappa: self,
                twice_mu,
            })
    }
}

impl TryFrom<i32> for Kappa {
    type Error = KappaError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Kappa> for i32 {
    fn from(value: Kappa) -> Self {
        value.get()
    }
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum KappaError {
    #[error("Dirac kappa cannot be zero")]
    Zero,
    #[error("Dirac kappa is outside the supported range: {0}")]
    OutOfRange(i32),
}

/// Exact integer representation of the half-integer magnetic number `mu`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct TwiceMu(i64);

impl TwiceMu {
    /// Construct `2mu`; Dirac spin-angular channels require an odd integer.
    pub fn new(value: i64) -> Result<Self, TwiceMuError> {
        if value & 1 == 0 {
            Err(TwiceMuError::Even(value))
        } else {
            Ok(Self(value))
        }
    }

    pub const fn get(self) -> i64 {
        self.0
    }
}

impl TryFrom<i64> for TwiceMu {
    type Error = TwiceMuError;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<TwiceMu> for i64 {
    fn from(value: TwiceMu) -> Self {
        value.get()
    }
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum TwiceMuError {
    #[error("twice the half-integer mu must be odd, got {0}")]
    Even(i64),
}

/// Explicit mapping between `kappa` and both spinor spherical harmonics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiracAngularContract {
    pub kappa: Kappa,
    pub large_l: u32,
    pub small_l: u32,
    pub twice_j: u32,
    pub degeneracy: u32,
}

impl From<Kappa> for DiracAngularContract {
    fn from(kappa: Kappa) -> Self {
        kappa.angular_contract()
    }
}

/// A validated spinor spherical-harmonic channel `(kappa,mu)`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RelativisticChannel {
    kappa: Kappa,
    twice_mu: TwiceMu,
}

impl RelativisticChannel {
    pub fn new(kappa: Kappa, twice_mu: TwiceMu) -> Result<Self, RelativisticChannelError> {
        if twice_mu.get().unsigned_abs() <= u64::from(kappa.twice_j()) {
            Ok(Self { kappa, twice_mu })
        } else {
            Err(RelativisticChannelError::MuOutsideChannel {
                kappa: kappa.get(),
                twice_mu: twice_mu.get(),
                twice_j: kappa.twice_j(),
            })
        }
    }

    pub const fn kappa(self) -> Kappa {
        self.kappa
    }

    pub const fn twice_mu(self) -> TwiceMu {
        self.twice_mu
    }

    /// Keep `mu` and replace `kappa` by `-kappa` (large/small counterpart).
    pub const fn opposite_kappa(self) -> Self {
        Self {
            kappa: self.kappa.opposite(),
            twice_mu: self.twice_mu,
        }
    }

    /// Clebsch--Gordan expansion, ordered spin-up then spin-down.
    ///
    /// Zero-coefficient terms at the extremal `j=l+1/2` channels are absent.
    pub fn spinor_harmonic_terms(self) -> [Option<SpinorHarmonicTerm>; 2] {
        let l = self.kappa.large_l();
        let twice_mu = self.twice_mu.get();
        let denominator = 2.0 * f64::from(2 * l + 1);
        let plus = ((f64::from(2 * l + 1) + twice_mu as f64) / denominator).sqrt();
        let minus = ((f64::from(2 * l + 1) - twice_mu as f64) / denominator).sqrt();
        let (up_coefficient, down_coefficient) = if self.kappa.get() < 0 {
            (plus, minus)
        } else {
            (-minus, plus)
        };
        [
            spinor_term(l, twice_mu, SpinProjection::Up, up_coefficient),
            spinor_term(l, twice_mu, SpinProjection::Down, down_coefficient),
        ]
    }
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum RelativisticChannelError {
    #[error("2mu={twice_mu} is outside [-{twice_j}, {twice_j}] for kappa={kappa}")]
    MuOutsideChannel {
        kappa: i32,
        twice_mu: i64,
        twice_j: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SpinProjection {
    Up,
    Down,
}

impl SpinProjection {
    pub const fn twice_ms(self) -> i32 {
        match self {
            Self::Up => 1,
            Self::Down => -1,
        }
    }
}

/// One nonzero `C Y_lm chi_s` term in `Omega_(kappa,mu)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinorHarmonicTerm {
    pub spin: SpinProjection,
    pub orbital: Lm,
    pub coefficient: f64,
}

/// Scalar-harmonic matrix element
/// `<Omega_left | Y_(field.l,field.m) | Omega_right>`.
///
/// Passing `left.opposite_kappa()` and `right.opposite_kappa()` gives the
/// corresponding small-component (`Omega_-kappa`) reduction without mixing it
/// with the large-component radial integral.
pub fn spinor_gaunt(left: RelativisticChannel, field: Lm, right: RelativisticChannel) -> f64 {
    let mut value = 0.0;
    for left_term in left.spinor_harmonic_terms().into_iter().flatten() {
        for right_term in right.spinor_harmonic_terms().into_iter().flatten() {
            if left_term.spin != right_term.spin {
                continue;
            }
            // gaunt uses conj(Y_left) Y_field conj(Y_third). Convert the
            // physical ket Y_right with Y_lm = (-1)^m conj(Y_l,-m).
            let ket_phase = if right_term.orbital.m & 1 == 0 {
                1.0
            } else {
                -1.0
            };
            value += left_term.coefficient
                * right_term.coefficient
                * ket_phase
                * gaunt(
                    left_term.orbital.l,
                    field.l,
                    right_term.orbital.l,
                    left_term.orbital.m,
                    field.m,
                    -right_term.orbital.m,
                );
        }
    }
    value
}

fn spinor_term(
    l: u32,
    twice_mu: i64,
    spin: SpinProjection,
    coefficient: f64,
) -> Option<SpinorHarmonicTerm> {
    if coefficient == 0.0 {
        return None;
    }
    let twice_ms = i64::from(spin.twice_ms());
    let m = i32::try_from((twice_mu - twice_ms) / 2)
        .expect("validated Dirac channel orbital projection fits i32");
    Some(SpinorHarmonicTerm {
        spin,
        orbital: Lm::new(l, m).expect("nonzero Clebsch--Gordan term has valid orbital projection"),
        coefficient,
    })
}
