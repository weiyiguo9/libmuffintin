//! Typed atomic core/valence defaults used by FLEUR's input generator.
//!
//! The catalogue is transcribed from FLEUR `develop` commit
//! `904b9b9707b375e2300082f89ddd0070447635a0`,
//! `src/tools/inpgen3/Profiles/default.econfig`.

/// A supported nuclear charge (hydrogen through lawrencium).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AtomicNumber(u8);

impl AtomicNumber {
    pub const MIN: u8 = 1;
    pub const MAX: u8 = 103;

    /// Constructs an atomic number when `z` is covered by the FLEUR catalogue.
    pub const fn new(z: u8) -> Option<Self> {
        if z >= Self::MIN && z <= Self::MAX {
            Some(Self(z))
        } else {
            None
        }
    }

    pub const fn get(self) -> u8 {
        self.0
    }

    /// Returns the canonical, case-sensitive chemical symbol.
    pub const fn symbol(self) -> &'static str {
        ELEMENT_SYMBOLS[(self.0 - Self::MIN) as usize]
    }

    /// Parses a canonical, case-sensitive chemical symbol.
    pub fn from_symbol(symbol: &str) -> Option<Self> {
        ELEMENT_SYMBOLS
            .iter()
            .position(|&candidate| candidate == symbol)
            .map(|index| Self(index as u8 + Self::MIN))
    }
}

/// A relativistic orbital identified by principal quantum number and signed
/// Dirac angular quantum number.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RelativisticOrbital {
    principal_quantum_number: u8,
    kappa: i8,
}

impl RelativisticOrbital {
    /// Constructs a physically admissible `(n, kappa)` orbital.
    pub const fn new(principal_quantum_number: u8, kappa: i8) -> Option<Self> {
        let angular_momentum = if kappa > 0 {
            kappa as u8
        } else if kappa < 0 {
            (-kappa - 1) as u8
        } else {
            return None;
        };
        if principal_quantum_number > angular_momentum {
            Some(Self {
                principal_quantum_number,
                kappa,
            })
        } else {
            None
        }
    }

    pub const fn principal_quantum_number(self) -> u8 {
        self.principal_quantum_number
    }

    pub const fn kappa(self) -> i8 {
        self.kappa
    }
}

/// How an occupied relativistic channel enters the radial basis treatment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AtomicChannelTreatment {
    Core,
    Valence,
    RelativisticLocalOrbital,
}

/// One occupied signed-kappa channel in an atomic reference configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomicOccupation {
    pub orbital: RelativisticOrbital,
    pub occupation: f64,
    pub treatment: AtomicChannelTreatment,
}

/// The occupied relativistic channels for one neutral atom.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomicElectronicConfiguration {
    atomic_number: AtomicNumber,
    occupations: Vec<AtomicOccupation>,
}

impl AtomicElectronicConfiguration {
    pub const fn atomic_number(&self) -> AtomicNumber {
        self.atomic_number
    }

    pub fn occupations(&self) -> &[AtomicOccupation] {
        &self.occupations
    }

    pub fn total_occupation(&self) -> f64 {
        self.occupations
            .iter()
            .map(|channel| channel.occupation)
            .sum()
    }

    /// Overrides the treatment of one occupied channel.
    ///
    /// Returns `false` when the requested channel is absent from this atom.
    pub fn set_treatment(
        &mut self,
        orbital: RelativisticOrbital,
        treatment: AtomicChannelTreatment,
    ) -> bool {
        let Some(channel) = self
            .occupations
            .iter_mut()
            .find(|channel| channel.orbital == orbital)
        else {
            return false;
        };
        channel.treatment = treatment;
        true
    }
}

/// Returns FLEUR's default neutral-atom configuration, expanded into signed
/// kappa channels.
///
/// In addition to FLEUR's core/valence split, the `5p1/2` channel is assigned
/// relativistic-local-orbital treatment for Z=55..86 and `6p1/2` for Z=87..103.
pub fn fleur_default_atomic_configuration(
    atomic_number: AtomicNumber,
) -> AtomicElectronicConfiguration {
    let source = FLEUR_DEFAULT_CONFIGURATIONS[(atomic_number.get() - 1) as usize];
    let (core, valence) = source
        .split_once('|')
        .expect("embedded FLEUR configuration must contain a core/valence separator");
    let mut occupations = Vec::new();
    append_shells(core, AtomicChannelTreatment::Core, &mut occupations);
    append_shells(valence, AtomicChannelTreatment::Valence, &mut occupations);

    let relativistic_lo = match atomic_number.get() {
        55..=86 => RelativisticOrbital::new(5, 1),
        87..=103 => RelativisticOrbital::new(6, 1),
        _ => None,
    };
    if let Some(orbital) = relativistic_lo {
        let channel = occupations
            .iter_mut()
            .find(|channel| channel.orbital == orbital)
            .expect("period-six and period-seven defaults must contain the semicore p1/2 channel");
        channel.treatment = AtomicChannelTreatment::RelativisticLocalOrbital;
    }

    AtomicElectronicConfiguration {
        atomic_number,
        occupations,
    }
}

fn append_shells(
    source: &str,
    treatment: AtomicChannelTreatment,
    occupations: &mut Vec<AtomicOccupation>,
) {
    for shell in source.split_whitespace() {
        let bytes = shell.as_bytes();
        let principal_quantum_number = bytes[0] - b'0';
        let occupation = shell[2..]
            .parse::<f64>()
            .expect("embedded FLEUR shell must have a numeric occupation");
        let channels: &[(i8, f64)] = match bytes[1] {
            b's' => &[(-1, 1.0)],
            b'p' => &[(1, 1.0 / 3.0), (-2, 2.0 / 3.0)],
            b'd' => &[(2, 2.0 / 5.0), (-3, 3.0 / 5.0)],
            b'f' => &[(3, 3.0 / 7.0), (-4, 4.0 / 7.0)],
            _ => panic!("embedded FLEUR shell must use s, p, d, or f"),
        };
        occupations.extend(channels.iter().map(|&(kappa, fraction)| {
            AtomicOccupation {
                orbital: RelativisticOrbital::new(principal_quantum_number, kappa)
                    .expect("embedded FLEUR shell must be physically admissible"),
                occupation: occupation * fraction,
                treatment,
            }
        }));
    }
}

// The `|` is FLEUR's exact core/valence boundary. Local-orbital declarations
// from the source file are basis hints rather than electronic occupations and
// are therefore not duplicated here.
const ELEMENT_SYMBOLS: [&str; 103] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr",
];

const FLEUR_DEFAULT_CONFIGURATIONS: [&str; 103] = [
    "|1s1",
    "|1s2",
    "1s2|2s1",
    "1s2|2s2",
    "1s2|2s2 2p1",
    "1s2|2s2 2p2",
    "1s2|2s2 2p3",
    "1s2|2s2 2p4",
    "1s2|2s2 2p5",
    "1s2|2s2 2p6",
    "1s2|2s2 2p6 3s1",
    "1s2|2s2 2p6 3s2",
    "1s2 2s2 2p6|3s2 3p1",
    "1s2 2s2 2p6|3s2 3p2",
    "1s2 2s2 2p6|3s2 3p3",
    "1s2 2s2 2p6|3s2 3p4",
    "1s2 2s2 2p6|3s2 3p5",
    "1s2 2s2 2p6|3s2 3p6",
    "1s2 2s2 2p6|3s2 3p6 4s1",
    "1s2 2s2 2p6|3s2 3p6 4s2",
    "1s2 2s2 2p6|3s2 3p6 4s2 3d1",
    "1s2 2s2 2p6|3s2 3p6 4s2 3d2",
    "1s2 2s2 2p6|3s2 3p6 4s2 3d3",
    "1s2 2s2 2p6|3s2 3p6 4s1 3d5",
    "1s2 2s2 2p6|3s2 3p6 4s2 3d5",
    "1s2 2s2 2p6|3s2 3p6 4s2 3d6",
    "1s2 2s2 2p6 3s2|3p6 4s2 3d7",
    "1s2 2s2 2p6 3s2|3p6 4s2 3d8",
    "1s2 2s2 2p6 3s2 3p6|4s1 3d10",
    "1s2 2s2 2p6 3s2 3p6|4s2 3d10",
    "1s2 2s2 2p6 3s2 3p6|4s2 3d10 4p1",
    "1s2 2s2 2p6 3s2 3p6|4s2 3d10 4p2",
    "1s2 2s2 2p6 3s2 3p6|4s2 3d10 4p3",
    "1s2 2s2 2p6 3s2 3p6|4s2 3d10 4p4",
    "1s2 2s2 2p6 3s2 3p6 3d10|4s2 4p5",
    "1s2 2s2 2p6 3s2 3p6 3d10|4s2 4p6",
    "1s2 2s2 2p6 3s2 3p6 3d10|4s2 4p6 5s1",
    "1s2 2s2 2p6 3s2 3p6 3d10|4s2 4p6 5s2",
    "1s2 2s2 2p6 3s2 3p6 3d10|4s2 4p6 5s2 4d1",
    "1s2 2s2 2p6 3s2 3p6 3d10|4s2 4p6 5s2 4d2",
    "1s2 2s2 2p6 3s2 3p6 3d10|4s2 4p6 5s1 4d4",
    "1s2 2s2 2p6 3s2 3p6 3d10|4s2 4p6 5s1 4d5",
    "1s2 2s2 2p6 3s2 3p6 3d10|4s2 4p6 5s2 4d5",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10|4p6 5s1 4d7",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10|4p6 5s1 4d8",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10|4p6 5s1 4d9",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10|4p6 5s1 4d10",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10|4p6 5s2 4d10",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6|5s2 4d10 5p1",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6|5s2 4d10 5p2",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6|5s2 4d10 5p3",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6|5s2 4d10 5p4",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6|5s2 4d10 5p5",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6|5s2 4d10 5p6",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s1",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 5d1",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 5d1 4f1",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f3",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f4",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f5",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f6",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f7",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f7 5d1",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f9",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f10",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f11",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f12",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f13",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f14",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f14 5d1",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f14 5d2",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10|5s2 5p6 6s2 4f14 5d3",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10 4f14|5s2 5p6 6s2 5d4",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 4d10 4f14|5s2 5p6 6s2 5d5",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 4f14|5p6 6s2 5d6",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 4f14|5p6 6s2 5d7",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 4f14|5p6 6s1 5d9",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 4f14|5p6 6s1 5d10",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 4f14|5p6 6s2 5d10",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 4f14 5p6|6s2 5d10 6p1",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 4f14 5p6|6s2 5d10 6p2",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14|6s2 5d10 6p3",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14|6s2 5d10 6p4",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14|6s2 5d10 6p5",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14|6s2 5d10 6p6",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s1",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 6d1",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 6d2",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 5f2 6d1",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 5f3 6d1",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 5f4 6d1",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 5f6",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 5f7",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 5f7 6d1",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 5f9",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 5f10",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 5f11",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 5f12",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 5f13",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 5f14",
    "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d10 5p6 4f14 5d10|6s2 6p6 7s2 5f14 6d1",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn orbital(n: u8, kappa: i8) -> RelativisticOrbital {
        RelativisticOrbital::new(n, kappa).unwrap()
    }

    fn occupation(
        configuration: &AtomicElectronicConfiguration,
        n: u8,
        kappa: i8,
    ) -> AtomicOccupation {
        *configuration
            .occupations()
            .iter()
            .find(|channel| channel.orbital == orbital(n, kappa))
            .unwrap()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn every_neutral_default_sums_to_its_atomic_number() {
        for z in AtomicNumber::MIN..=AtomicNumber::MAX {
            let configuration = fleur_default_atomic_configuration(AtomicNumber::new(z).unwrap());
            assert!(
                (configuration.total_occupation() - f64::from(z)).abs() < 1.0e-12,
                "occupation mismatch for Z={z}"
            );
        }
    }

    #[test]
    fn atomic_symbols_round_trip_for_every_supported_element() {
        for z in AtomicNumber::MIN..=AtomicNumber::MAX {
            let atomic_number = AtomicNumber::new(z).unwrap();
            assert_eq!(
                AtomicNumber::from_symbol(atomic_number.symbol()),
                Some(atomic_number)
            );
        }

        assert_eq!(AtomicNumber::new(AtomicNumber::MIN).unwrap().symbol(), "H");
        assert_eq!(AtomicNumber::new(AtomicNumber::MAX).unwrap().symbol(), "Lr");
    }

    #[test]
    fn atomic_symbol_parser_rejects_noncanonical_inputs() {
        for symbol in ["", "h", "HE", "Xx"] {
            assert_eq!(AtomicNumber::from_symbol(symbol), None);
        }
    }

    #[test]
    fn iron_preserves_fleur_core_valence_boundary_and_kappa_ratios() {
        let iron = fleur_default_atomic_configuration(AtomicNumber::new(26).unwrap());
        assert_eq!(
            occupation(&iron, 2, 1).treatment,
            AtomicChannelTreatment::Core
        );
        assert_eq!(
            occupation(&iron, 3, 1).treatment,
            AtomicChannelTreatment::Valence
        );
        assert_close(occupation(&iron, 3, 2).occupation, 12.0 / 5.0);
        assert_close(occupation(&iron, 3, -3).occupation, 18.0 / 5.0);
    }

    #[test]
    fn exceptional_ground_state_occupations_match_fleur() {
        let chromium = fleur_default_atomic_configuration(AtomicNumber::new(24).unwrap());
        assert_close(occupation(&chromium, 4, -1).occupation, 1.0);
        assert_close(occupation(&chromium, 3, 2).occupation, 2.0);
        assert_close(occupation(&chromium, 3, -3).occupation, 3.0);

        let palladium = fleur_default_atomic_configuration(AtomicNumber::new(46).unwrap());
        assert_close(occupation(&palladium, 5, -1).occupation, 1.0);
        assert_close(occupation(&palladium, 4, 2).occupation, 18.0 / 5.0);
        assert_close(occupation(&palladium, 4, -3).occupation, 27.0 / 5.0);
    }

    #[test]
    fn period_semicore_p_half_channels_are_relativistic_local_orbitals() {
        for z in 55..=86 {
            let configuration = fleur_default_atomic_configuration(AtomicNumber::new(z).unwrap());
            assert_eq!(
                occupation(&configuration, 5, 1).treatment,
                AtomicChannelTreatment::RelativisticLocalOrbital
            );
            assert_ne!(
                occupation(&configuration, 5, -2).treatment,
                AtomicChannelTreatment::RelativisticLocalOrbital
            );
        }
        for z in 87..=103 {
            let configuration = fleur_default_atomic_configuration(AtomicNumber::new(z).unwrap());
            assert_eq!(
                occupation(&configuration, 6, 1).treatment,
                AtomicChannelTreatment::RelativisticLocalOrbital
            );
            assert_ne!(
                occupation(&configuration, 6, -2).treatment,
                AtomicChannelTreatment::RelativisticLocalOrbital
            );
        }
    }

    #[test]
    fn caller_can_override_an_occupied_channel_treatment() {
        let mut carbon = fleur_default_atomic_configuration(AtomicNumber::new(6).unwrap());
        assert!(carbon.set_treatment(
            orbital(2, 1),
            AtomicChannelTreatment::RelativisticLocalOrbital
        ));
        assert_eq!(
            occupation(&carbon, 2, 1).treatment,
            AtomicChannelTreatment::RelativisticLocalOrbital
        );
        assert!(!carbon.set_treatment(orbital(3, -1), AtomicChannelTreatment::Core));
    }
}
