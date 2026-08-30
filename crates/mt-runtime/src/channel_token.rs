//! Spectroscopic channel tokens and the normalized recipe record they produce.

use muffintin_core::Hartree;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The species or concrete site to which a normalized channel belongs.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ChannelScope {
    Species { name: String },
    Site { name: String },
}

/// Route-independent channel identity.
///
/// A bare token retains its scalar `l` identity. Only an explicit `j` tag is
/// normalized to signed kappa; expansion into spinor partners belongs to the
/// later route-specific compilation step.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ChannelIdentity {
    ScalarL { n: u32, l: u32 },
    Kappa { n: u32, kappa: i32 },
}

/// Physical role assigned to a radial channel.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ChannelTreatment {
    Core,
    Valence,
    Lo,
    Hdlo,
}

/// Stable generator names shared by recipe artifacts and token suffixes.
#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum ChannelEnergyGenerator {
    Explicit,
    #[default]
    Atomic,
    BandCenter,
    LogDerivative,
    BandCog,
    FermiOffset,
    FrozenCheckpoint,
}

impl ChannelEnergyGenerator {
    fn from_suffix(suffix: &str) -> Option<Self> {
        match suffix {
            "explicit" => Some(Self::Explicit),
            "atomic" => Some(Self::Atomic),
            "band-center" => Some(Self::BandCenter),
            "log-derivative" => Some(Self::LogDerivative),
            "band-cog" => Some(Self::BandCog),
            "fermi-offset" => Some(Self::FermiOffset),
            "frozen-checkpoint" | "frozen" => Some(Self::FrozenCheckpoint),
            _ => None,
        }
    }
}

/// Stable origin categories retained when a normalized record is emitted.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ChannelProvenance {
    BuiltIn,
    ExternalRecipe {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
    },
    TaskDefault,
    Species,
    Site,
}

/// One normalized channel record; this is also the recipe artifact IR.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct ChannelRecipeRecord {
    pub scope: ChannelScope,
    pub identity: ChannelIdentity,
    pub treatment: ChannelTreatment,
    /// Arbitrary nonnegative derivative order. Execution support is checked later.
    pub derivative_order: u32,
    pub generator: ChannelEnergyGenerator,
    /// Generator restart value in Hartree, when one is available.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_hartree"
    )]
    pub seed: Option<Hartree>,
    pub provenance: ChannelProvenance,
}

/// Non-token fields inherited while one compact token is normalized.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelTokenContext {
    pub scope: ChannelScope,
    pub treatment: ChannelTreatment,
    pub derivative_order: u32,
    pub generator: ChannelEnergyGenerator,
    pub seed: Option<Hartree>,
    pub provenance: ChannelProvenance,
}

/// A species definition or one site-level edit.
#[derive(Clone, Debug, PartialEq)]
pub enum ParsedChannelToken {
    Define(ChannelRecipeRecord),
    Add(ChannelRecipeRecord),
    Remove {
        scope: ChannelScope,
        identity: ChannelIdentity,
        treatment: ChannelTreatment,
        derivative_order: u32,
    },
}

/// A typed failure to normalize one spectroscopic channel token.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ChannelTokenError {
    #[error("channel token is empty")]
    Empty,
    #[error("channel token `{token}` contains whitespace")]
    Whitespace { token: String },
    #[error("channel token `{token}` has no principal quantum number")]
    MissingPrincipalQuantumNumber { token: String },
    #[error("principal quantum number in `{token}` is zero")]
    ZeroPrincipalQuantumNumber { token: String },
    #[error("principal quantum number in `{token}` exceeds u32")]
    PrincipalQuantumNumberOverflow { token: String },
    #[error("channel token `{token}` has no orbital symbol")]
    MissingOrbitalSymbol { token: String },
    #[error("`{symbol}` is not a supported spectroscopic orbital symbol in `{token}`")]
    InvalidOrbitalSymbol { token: String, symbol: char },
    #[error("invalid quantum numbers in `{token}`: n={n} must be greater than l={l}")]
    InvalidQuantumNumbers { token: String, n: u32, l: u32 },
    #[error("j tag `{tag}` is incompatible with l={l} in `{token}`")]
    InvalidJTag { token: String, tag: String, l: u32 },
    #[error("trailing text `{trailing}` is not a j tag in `{token}`")]
    TrailingGarbage { token: String, trailing: String },
    #[error("species channel `{token}` cannot use a site edit prefix")]
    SpeciesEdit { token: String },
    #[error("site channel `{token}` must begin with `+` or `-`")]
    MissingSiteEdit { token: String },
    #[error("removal token `{token}` cannot carry an `@` suffix")]
    SuffixOnRemove { token: String },
    #[error("channel token `{token}` has an empty `@` suffix")]
    EmptySuffix { token: String },
    #[error("unknown channel suffix `{suffix}` in `{token}`")]
    UnknownSuffix { token: String, suffix: String },
    #[error("explicit energy `{suffix}` in `{token}` is not a number")]
    InvalidExplicitEnergy { token: String, suffix: String },
    #[error("explicit energy `{suffix}` in `{token}` is not finite")]
    NonFiniteExplicitEnergy { token: String, suffix: String },
}

/// Parse and normalize one token using its enclosing treatment and scope.
pub fn parse_channel_token(
    token: &str,
    context: &ChannelTokenContext,
) -> Result<ParsedChannelToken, ChannelTokenError> {
    if token.is_empty() {
        return Err(ChannelTokenError::Empty);
    }
    if token.chars().any(char::is_whitespace) {
        return Err(ChannelTokenError::Whitespace {
            token: token.to_owned(),
        });
    }

    let (edit, body) = match token.as_bytes()[0] {
        b'+' => (TokenEdit::Add, &token[1..]),
        b'-' => (TokenEdit::Remove, &token[1..]),
        _ => (TokenEdit::Define, token),
    };
    validate_edit(token, edit, &context.scope)?;

    let (channel, suffix) = match body.split_once('@') {
        Some((_channel, "")) => {
            return Err(ChannelTokenError::EmptySuffix {
                token: token.to_owned(),
            });
        }
        Some((channel, suffix)) => (channel, Some(suffix)),
        None => (body, None),
    };
    if edit == TokenEdit::Remove && suffix.is_some() {
        return Err(ChannelTokenError::SuffixOnRemove {
            token: token.to_owned(),
        });
    }

    let identity = parse_identity(token, channel)?;
    if edit == TokenEdit::Remove {
        return Ok(ParsedChannelToken::Remove {
            scope: context.scope.clone(),
            identity,
            treatment: context.treatment,
            derivative_order: context.derivative_order,
        });
    }

    let (generator, seed) = match suffix {
        Some(suffix) => parse_suffix(token, suffix, context.seed)?,
        None => (context.generator, context.seed),
    };
    let record = ChannelRecipeRecord {
        scope: context.scope.clone(),
        identity,
        treatment: context.treatment,
        derivative_order: context.derivative_order,
        generator,
        seed,
        provenance: context.provenance.clone(),
    };
    Ok(match edit {
        TokenEdit::Define => ParsedChannelToken::Define(record),
        TokenEdit::Add => ParsedChannelToken::Add(record),
        TokenEdit::Remove => unreachable!("removal returned before record construction"),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenEdit {
    Define,
    Add,
    Remove,
}

fn validate_edit(
    token: &str,
    edit: TokenEdit,
    scope: &ChannelScope,
) -> Result<(), ChannelTokenError> {
    match (scope, edit) {
        (ChannelScope::Species { .. }, TokenEdit::Add | TokenEdit::Remove) => {
            Err(ChannelTokenError::SpeciesEdit {
                token: token.to_owned(),
            })
        }
        (ChannelScope::Site { .. }, TokenEdit::Define) => Err(ChannelTokenError::MissingSiteEdit {
            token: token.to_owned(),
        }),
        _ => Ok(()),
    }
}

fn parse_identity(token: &str, channel: &str) -> Result<ChannelIdentity, ChannelTokenError> {
    let digit_count = channel.bytes().take_while(u8::is_ascii_digit).count();
    if digit_count == 0 {
        return Err(ChannelTokenError::MissingPrincipalQuantumNumber {
            token: token.to_owned(),
        });
    }
    let n_text = &channel[..digit_count];
    let n =
        n_text
            .parse::<u32>()
            .map_err(|_| ChannelTokenError::PrincipalQuantumNumberOverflow {
                token: token.to_owned(),
            })?;
    if n == 0 {
        return Err(ChannelTokenError::ZeroPrincipalQuantumNumber {
            token: token.to_owned(),
        });
    }

    let rest = &channel[digit_count..];
    let Some(symbol) = rest.chars().next() else {
        return Err(ChannelTokenError::MissingOrbitalSymbol {
            token: token.to_owned(),
        });
    };
    let l = orbital_angular_momentum(symbol).ok_or_else(|| {
        ChannelTokenError::InvalidOrbitalSymbol {
            token: token.to_owned(),
            symbol,
        }
    })?;
    if n <= l {
        return Err(ChannelTokenError::InvalidQuantumNumbers {
            token: token.to_owned(),
            n,
            l,
        });
    }

    let j_tag = &rest[symbol.len_utf8()..];
    if j_tag.is_empty() {
        return Ok(ChannelIdentity::ScalarL { n, l });
    }
    let kappa = parse_j_tag(token, j_tag, l)?;
    Ok(ChannelIdentity::Kappa { n, kappa })
}

fn orbital_angular_momentum(symbol: char) -> Option<u32> {
    "spdfghiklmnoqrtuvwxyz"
        .chars()
        .position(|candidate| candidate == symbol)
        .map(|l| l as u32)
}

fn parse_j_tag(token: &str, tag: &str, l: u32) -> Result<i32, ChannelTokenError> {
    match tag {
        "-" if l > 0 => return Ok(l as i32),
        "+" => return Ok(-((l + 1) as i32)),
        "-" => return Err(invalid_j_tag(token, tag, l)),
        _ => {}
    }

    let Some((numerator, denominator)) = tag.split_once('/') else {
        return Err(ChannelTokenError::TrailingGarbage {
            token: token.to_owned(),
            trailing: tag.to_owned(),
        });
    };
    let Ok(numerator) = numerator.parse::<u32>() else {
        return Err(ChannelTokenError::TrailingGarbage {
            token: token.to_owned(),
            trailing: tag.to_owned(),
        });
    };
    if denominator != "2" {
        return Err(invalid_j_tag(token, tag, l));
    }
    if l > 0 && numerator == 2 * l - 1 {
        Ok(l as i32)
    } else if numerator == 2 * l + 1 {
        Ok(-((l + 1) as i32))
    } else {
        Err(invalid_j_tag(token, tag, l))
    }
}

fn invalid_j_tag(token: &str, tag: &str, l: u32) -> ChannelTokenError {
    ChannelTokenError::InvalidJTag {
        token: token.to_owned(),
        tag: tag.to_owned(),
        l,
    }
}

fn parse_suffix(
    token: &str,
    suffix: &str,
    inherited_seed: Option<Hartree>,
) -> Result<(ChannelEnergyGenerator, Option<Hartree>), ChannelTokenError> {
    if let Some(generator) = ChannelEnergyGenerator::from_suffix(suffix) {
        return Ok((generator, inherited_seed));
    }

    let (number, electron_volts) = match suffix.strip_suffix("ev") {
        Some(number) => (number, true),
        None => match suffix.strip_suffix("eV") {
            Some(number) => (number, true),
            None => match suffix.strip_suffix("EV") {
                Some(number) => (number, true),
                None => (suffix, false),
            },
        },
    };
    let value = number
        .parse::<f64>()
        .map_err(|_| classify_bad_suffix(token, suffix))?;
    if !value.is_finite() {
        return Err(ChannelTokenError::NonFiniteExplicitEnergy {
            token: token.to_owned(),
            suffix: suffix.to_owned(),
        });
    }
    let seed = if electron_volts {
        Hartree::from_ev(value)
    } else {
        Hartree(value)
    };
    if !seed.get().is_finite() {
        return Err(ChannelTokenError::NonFiniteExplicitEnergy {
            token: token.to_owned(),
            suffix: suffix.to_owned(),
        });
    }
    Ok((ChannelEnergyGenerator::Explicit, Some(seed)))
}

fn classify_bad_suffix(token: &str, suffix: &str) -> ChannelTokenError {
    let numeric_start = matches!(
        suffix.as_bytes().first(),
        Some(b'+' | b'-' | b'.' | b'0'..=b'9')
    );
    if numeric_start {
        ChannelTokenError::InvalidExplicitEnergy {
            token: token.to_owned(),
            suffix: suffix.to_owned(),
        }
    } else {
        ChannelTokenError::UnknownSuffix {
            token: token.to_owned(),
            suffix: suffix.to_owned(),
        }
    }
}

mod optional_hartree {
    use muffintin_core::Hartree;
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

    pub fn serialize<S>(value: &Option<Hartree>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        value.map(Hartree::get).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Hartree>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<f64>::deserialize(deserializer)?.map_or(Ok(None), |value| {
            Hartree::checked(value)
                .map(Some)
                .ok_or_else(|| D::Error::custom("channel seed must be finite Hartree"))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};

    use muffintin_core::Hartree;

    use super::*;

    fn species_context() -> ChannelTokenContext {
        ChannelTokenContext {
            scope: ChannelScope::Species {
                name: "Pb".to_owned(),
            },
            treatment: ChannelTreatment::Lo,
            derivative_order: 0,
            generator: ChannelEnergyGenerator::Atomic,
            seed: Some(Hartree(-0.25)),
            provenance: ChannelProvenance::Species,
        }
    }

    fn site_context() -> ChannelTokenContext {
        ChannelTokenContext {
            scope: ChannelScope::Site {
                name: "Pb-1".to_owned(),
            },
            treatment: ChannelTreatment::Lo,
            derivative_order: 0,
            generator: ChannelEnergyGenerator::Atomic,
            seed: None,
            provenance: ChannelProvenance::Site,
        }
    }

    fn defined_record(token: &str) -> ChannelRecipeRecord {
        match parse_channel_token(token, &species_context()).unwrap() {
            ParsedChannelToken::Define(record) => record,
            other => panic!("expected definition, got {other:?}"),
        }
    }

    #[test]
    fn shorthand_and_full_j_tags_are_synonyms() {
        assert_eq!(
            defined_record("5p-").identity,
            defined_record("5p1/2").identity
        );
        assert_eq!(
            defined_record("5p+").identity,
            defined_record("5p3/2").identity
        );
        assert_eq!(
            defined_record("5p-").identity,
            ChannelIdentity::Kappa { n: 5, kappa: 1 }
        );
        assert_eq!(
            defined_record("5p+").identity,
            ChannelIdentity::Kappa { n: 5, kappa: -2 }
        );
        assert_eq!(
            defined_record("5p").identity,
            ChannelIdentity::ScalarL { n: 5, l: 1 }
        );
    }

    #[test]
    fn explicit_ev_is_normalized_to_hartree() {
        for token in ["4f@-4.08ev", "4f@-4.08eV", "4f@-4.08EV"] {
            let record = defined_record(token);
            assert_eq!(record.generator, ChannelEnergyGenerator::Explicit);
            let actual = record.seed.unwrap().get();
            assert!((actual - Hartree::from_ev(-4.08).get()).abs() < 1.0e-15);
        }
        assert_eq!(defined_record("4f@-0.15").seed, Some(Hartree(-0.15)));
    }

    #[test]
    fn every_generator_suffix_is_supported() {
        assert_eq!(
            ChannelEnergyGenerator::default(),
            ChannelEnergyGenerator::Atomic
        );
        let cases = [
            ("explicit", ChannelEnergyGenerator::Explicit),
            ("atomic", ChannelEnergyGenerator::Atomic),
            ("band-center", ChannelEnergyGenerator::BandCenter),
            ("log-derivative", ChannelEnergyGenerator::LogDerivative),
            ("band-cog", ChannelEnergyGenerator::BandCog),
            ("fermi-offset", ChannelEnergyGenerator::FermiOffset),
            ("frozen-checkpoint", ChannelEnergyGenerator::FrozenCheckpoint),
            ("frozen", ChannelEnergyGenerator::FrozenCheckpoint),
        ];
        for (suffix, expected) in cases {
            let record = defined_record(&format!("5d@{suffix}"));
            assert_eq!(record.generator, expected);
            assert_eq!(record.seed, Some(Hartree(-0.25)));
        }
    }

    #[test]
    fn site_tokens_are_typed_edits() {
        match parse_channel_token("+5f", &site_context()).unwrap() {
            ParsedChannelToken::Add(record) => {
                assert_eq!(record.scope, site_context().scope);
                assert_eq!(record.identity, ChannelIdentity::ScalarL { n: 5, l: 3 });
            }
            other => panic!("expected addition, got {other:?}"),
        }
        assert_eq!(
            parse_channel_token("-4f", &site_context()).unwrap(),
            ParsedChannelToken::Remove {
                scope: site_context().scope,
                identity: ChannelIdentity::ScalarL { n: 4, l: 3 },
                treatment: ChannelTreatment::Lo,
                derivative_order: 0,
            }
        );

        let mut hdlo = site_context();
        hdlo.treatment = ChannelTreatment::Hdlo;
        hdlo.derivative_order = 2;
        assert_eq!(
            parse_channel_token("-4f", &hdlo).unwrap(),
            ParsedChannelToken::Remove {
                scope: hdlo.scope,
                identity: ChannelIdentity::ScalarL { n: 4, l: 3 },
                treatment: ChannelTreatment::Hdlo,
                derivative_order: 2,
            }
        );
    }

    #[test]
    fn invalid_quantum_numbers_edits_and_trailing_text_are_typed() {
        assert!(matches!(
            parse_channel_token("1s-", &species_context()),
            Err(ChannelTokenError::InvalidJTag { .. })
        ));
        assert!(matches!(
            parse_channel_token("0s", &species_context()),
            Err(ChannelTokenError::ZeroPrincipalQuantumNumber { .. })
        ));
        assert!(matches!(
            parse_channel_token("4294967296s", &species_context()),
            Err(ChannelTokenError::PrincipalQuantumNumberOverflow { .. })
        ));
        assert!(matches!(
            parse_channel_token("5pwat", &species_context()),
            Err(ChannelTokenError::TrailingGarbage { .. })
        ));
        assert!(matches!(
            parse_channel_token("+5p", &species_context()),
            Err(ChannelTokenError::SpeciesEdit { .. })
        ));
        assert!(matches!(
            parse_channel_token("5p", &site_context()),
            Err(ChannelTokenError::MissingSiteEdit { .. })
        ));
        assert!(matches!(
            parse_channel_token("-5p@atomic", &site_context()),
            Err(ChannelTokenError::SuffixOnRemove { .. })
        ));
    }

    #[test]
    fn identity_is_reusable_for_duplicate_detection() {
        let identity = defined_record("5p-").identity;
        let synonym = defined_record("5p1/2").identity;
        let mut ordered = BTreeSet::new();
        let mut hashed = HashSet::new();
        assert!(ordered.insert(identity));
        assert!(!ordered.insert(synonym));
        assert!(hashed.insert(identity));
        assert!(!hashed.insert(synonym));
    }

    #[test]
    fn recipe_ir_round_trips_all_derivative_orders() {
        for derivative_order in [0, 1, 2, 3, u32::MAX] {
            let mut context = species_context();
            context.derivative_order = derivative_order;
            context.provenance = ChannelProvenance::ExternalRecipe {
                source: Some("pb.toml".to_owned()),
            };
            let record = match parse_channel_token("5p@frozen", &context).unwrap() {
                ParsedChannelToken::Define(record) => record,
                other => panic!("expected definition, got {other:?}"),
            };
            let encoded = toml::to_string(&record).unwrap();
            let decoded: ChannelRecipeRecord = toml::from_str(&encoded).unwrap();
            assert_eq!(decoded, record);
        }
    }
}
