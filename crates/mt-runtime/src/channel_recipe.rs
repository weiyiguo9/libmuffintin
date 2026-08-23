//! Pure channel-recipe artifact handling and deterministic layer compilation.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};

use muffintin_dft::{AtomicChannelTreatment, AtomicNumber, fleur_default_atomic_configuration};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::channel_token::{
    ChannelEnergyGenerator, ChannelIdentity, ChannelProvenance, ChannelRecipeRecord, ChannelScope,
    ChannelTokenContext, ChannelTokenError, ChannelTreatment, ParsedChannelToken,
    parse_channel_token,
};

/// A standalone, normalized channel-recipe document.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct ChannelRecipeArtifact {
    pub channels: Vec<ChannelRecipeRecord>,
}

/// One concrete site for which the layered recipe is compiled.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecipeSite {
    pub id: String,
    pub atomic_number: AtomicNumber,
}

/// An external normalized recipe together with its displayed source.
#[derive(Clone, Copy, Debug)]
pub struct ExternalChannelRecipe<'a> {
    pub artifact: &'a ChannelRecipeArtifact,
    pub source: &'a str,
}

/// The final normalized records for one concrete site.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledSiteRecipe {
    pub site: String,
    pub atomic_number: AtomicNumber,
    pub channels: Vec<ChannelRecipeRecord>,
}

/// The channel recipe compiled for all sites, in input site order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompiledChannelRecipe {
    pub sites: Vec<CompiledSiteRecipe>,
}

impl CompiledChannelRecipe {
    /// Finds one compiled site by its exact site identifier.
    pub fn site(&self, site: &str) -> Option<&CompiledSiteRecipe> {
        self.sites.iter().find(|candidate| candidate.site == site)
    }
}

impl IntoIterator for CompiledChannelRecipe {
    type Item = CompiledSiteRecipe;
    type IntoIter = std::vec::IntoIter<CompiledSiteRecipe>;

    fn into_iter(self) -> Self::IntoIter {
        self.sites.into_iter()
    }
}

/// A typed channel-artifact or layer-compilation failure.
#[derive(Debug, Error)]
pub enum ChannelRecipeError {
    #[error("could not decode channel recipe TOML: {0}")]
    Decode(#[from] toml::de::Error),
    #[error("could not encode channel recipe TOML: {0}")]
    Encode(#[from] toml::ser::Error),
    #[error("recipe sites contain duplicate site id {site:?}")]
    DuplicateSite { site: String },
    #[error("external recipe refers to species {species:?}, which is absent from the sites")]
    UnknownRecipeSpecies { species: String },
    #[error("external recipe refers to site {site:?}, which is absent from the sites")]
    UnknownRecipeSite { site: String },
    #[error("channel key {key:?} matches both a present canonical species and a site id")]
    AmbiguousInputScope { key: String },
    #[error("channel key {key:?} matches neither a present canonical species nor a site id")]
    UnknownInputScope { key: String },
    #[error("invalid {treatment:?} token {token:?} in channel row {scope:?}: {source}")]
    Token {
        scope: String,
        treatment: ChannelTreatment,
        token: String,
        #[source]
        source: ChannelTokenError,
    },
    #[error(
        "{layer} layer for {scope:?} contains duplicate normalized {treatment:?} instance {identity:?} at derivative order {derivative_order}"
    )]
    DuplicateInstance {
        layer: &'static str,
        scope: String,
        treatment: ChannelTreatment,
        identity: ChannelIdentity,
        derivative_order: u32,
    },
    #[error(
        "{layer} layer for {scope:?} assigns {identity:?} at derivative order {derivative_order} to both {first:?} and {second:?}"
    )]
    TreatmentConflict {
        layer: &'static str,
        scope: String,
        identity: ChannelIdentity,
        derivative_order: u32,
        first: ChannelTreatment,
        second: ChannelTreatment,
    },
    #[error(
        "site {site:?} cannot remove absent {treatment:?} channel {identity:?} at derivative order {derivative_order}"
    )]
    DeleteMiss {
        site: String,
        treatment: ChannelTreatment,
        identity: ChannelIdentity,
        derivative_order: u32,
    },
    #[error(
        "compiled site {site:?} has an explicit channel {identity:?} at derivative order {derivative_order} without a Hartree seed"
    )]
    MissingExplicitSeed {
        site: String,
        identity: ChannelIdentity,
        derivative_order: u32,
    },
}

/// Parses a normalized recipe document and canonicalizes its record order.
pub fn parse_channel_recipe_toml(text: &str) -> Result<ChannelRecipeArtifact, ChannelRecipeError> {
    let mut artifact: ChannelRecipeArtifact = toml::from_str(text)?;
    sort_records(&mut artifact.channels);
    Ok(artifact)
}

/// Serializes a normalized recipe document in canonical record order.
pub fn channel_recipe_to_toml(
    artifact: &ChannelRecipeArtifact,
) -> Result<String, ChannelRecipeError> {
    let mut canonical = artifact.clone();
    sort_records(&mut canonical.channels);
    let mut text = toml::to_string_pretty(&canonical)?;
    if !text.ends_with('\n') {
        text.push('\n');
    }
    Ok(text)
}

/// Compiles the built-in, external, task, species, site, and token-suffix layers.
///
/// The input map uses canonical species symbols or exact site identifiers as
/// keys. Species rows replace each treatment key they contain, while site rows
/// contain only `+` and `-` edits.
pub fn compile_channel_recipe(
    sites: &[RecipeSite],
    external: Option<ExternalChannelRecipe<'_>>,
    task_generator: Option<ChannelEnergyGenerator>,
    input: &BTreeMap<String, BTreeMap<ChannelTreatment, Vec<String>>>,
) -> Result<CompiledChannelRecipe, ChannelRecipeError> {
    let index = SiteIndex::new(sites)?;
    let mut compiled = built_in_recipes(sites);

    if let Some(external) = external {
        apply_external_recipe(&mut compiled, &index, external)?;
    }
    if let Some(generator) = task_generator {
        apply_task_generator(&mut compiled, generator)?;
    }

    let rows = resolve_input_rows(input, &index)?;
    for row in rows
        .iter()
        .filter(|row| matches!(&row.scope, ResolvedScope::Species(_)))
    {
        apply_species_row(&mut compiled, row, task_generator)?;
    }
    for row in rows
        .iter()
        .filter(|row| matches!(&row.scope, ResolvedScope::Site(_)))
    {
        apply_site_row(&mut compiled, row, task_generator)?;
    }

    for site in &mut compiled.sites {
        validate_explicit_seeds(site)?;
        sort_records(&mut site.channels);
    }
    Ok(compiled)
}

struct SiteIndex {
    site_positions: BTreeMap<String, usize>,
    species: BTreeSet<String>,
}

impl SiteIndex {
    fn new(sites: &[RecipeSite]) -> Result<Self, ChannelRecipeError> {
        let mut site_positions = BTreeMap::new();
        let mut species = BTreeSet::new();
        for (position, site) in sites.iter().enumerate() {
            if site_positions.insert(site.id.clone(), position).is_some() {
                return Err(ChannelRecipeError::DuplicateSite {
                    site: site.id.clone(),
                });
            }
            species.insert(site.atomic_number.symbol().to_owned());
        }
        Ok(Self {
            site_positions,
            species,
        })
    }
}

fn built_in_recipes(sites: &[RecipeSite]) -> CompiledChannelRecipe {
    CompiledChannelRecipe {
        sites: sites
            .iter()
            .map(|site| {
                let scope = ChannelScope::Site {
                    name: site.id.clone(),
                };
                let configuration = fleur_default_atomic_configuration(site.atomic_number);
                let channels = configuration
                    .occupations()
                    .iter()
                    .map(|occupation| ChannelRecipeRecord {
                        scope: scope.clone(),
                        identity: ChannelIdentity::Kappa {
                            n: u32::from(occupation.orbital.principal_quantum_number()),
                            kappa: i32::from(occupation.orbital.kappa()),
                        },
                        treatment: match occupation.treatment {
                            AtomicChannelTreatment::Core => ChannelTreatment::Core,
                            AtomicChannelTreatment::Valence => ChannelTreatment::Valence,
                            AtomicChannelTreatment::RelativisticLocalOrbital => {
                                ChannelTreatment::Lo
                            }
                        },
                        derivative_order: 0,
                        generator: ChannelEnergyGenerator::Atomic,
                        seed: None,
                        provenance: ChannelProvenance::BuiltIn,
                    })
                    .collect();
                CompiledSiteRecipe {
                    site: site.id.clone(),
                    atomic_number: site.atomic_number,
                    channels,
                }
            })
            .collect(),
    }
}

fn apply_external_recipe(
    compiled: &mut CompiledChannelRecipe,
    index: &SiteIndex,
    external: ExternalChannelRecipe<'_>,
) -> Result<(), ChannelRecipeError> {
    let mut species_records: BTreeMap<String, Vec<ChannelRecipeRecord>> = BTreeMap::new();
    let mut site_records: BTreeMap<String, Vec<ChannelRecipeRecord>> = BTreeMap::new();
    for record in &external.artifact.channels {
        match &record.scope {
            ChannelScope::Species { name } => {
                if !index.species.contains(name) {
                    return Err(ChannelRecipeError::UnknownRecipeSpecies {
                        species: name.clone(),
                    });
                }
                species_records
                    .entry(name.clone())
                    .or_default()
                    .push(record.clone());
            }
            ChannelScope::Site { name } => {
                if !index.site_positions.contains_key(name) {
                    return Err(ChannelRecipeError::UnknownRecipeSite { site: name.clone() });
                }
                site_records
                    .entry(name.clone())
                    .or_default()
                    .push(record.clone());
            }
        }
    }

    for (species, records) in species_records {
        validate_record_layer("external recipe", &species, &records)?;
        for site in compiled
            .sites
            .iter_mut()
            .filter(|site| site.atomic_number.symbol() == species.as_str())
        {
            site.channels = external_records_for_site(&records, &site.site, external.source);
        }
    }
    for (site_name, records) in site_records {
        validate_record_layer("external recipe", &site_name, &records)?;
        let site = &mut compiled.sites[index.site_positions[&site_name]];
        site.channels = external_records_for_site(&records, &site.site, external.source);
    }
    Ok(())
}

fn external_records_for_site(
    records: &[ChannelRecipeRecord],
    site: &str,
    source: &str,
) -> Vec<ChannelRecipeRecord> {
    records
        .iter()
        .cloned()
        .map(|mut record| {
            record.scope = ChannelScope::Site {
                name: site.to_owned(),
            };
            record.provenance = ChannelProvenance::ExternalRecipe {
                source: Some(source.to_owned()),
            };
            record
        })
        .collect()
}

fn apply_task_generator(
    compiled: &mut CompiledChannelRecipe,
    generator: ChannelEnergyGenerator,
) -> Result<(), ChannelRecipeError> {
    for site in &mut compiled.sites {
        for record in &mut site.channels {
            record.generator = generator;
        }
        validate_record_layer("task generator", &site.site, &site.channels)?;
    }
    Ok(())
}

enum ResolvedScope {
    Species(String),
    Site(String),
}

struct ResolvedInputRow<'a> {
    key: &'a str,
    scope: ResolvedScope,
    treatments: &'a BTreeMap<ChannelTreatment, Vec<String>>,
}

fn resolve_input_rows<'a>(
    input: &'a BTreeMap<String, BTreeMap<ChannelTreatment, Vec<String>>>,
    index: &SiteIndex,
) -> Result<Vec<ResolvedInputRow<'a>>, ChannelRecipeError> {
    input
        .iter()
        .map(|(key, treatments)| {
            let is_species = index.species.contains(key);
            let is_site = index.site_positions.contains_key(key);
            let scope = match (is_species, is_site) {
                (true, true) => {
                    return Err(ChannelRecipeError::AmbiguousInputScope { key: key.clone() });
                }
                (true, false) => ResolvedScope::Species(key.clone()),
                (false, true) => ResolvedScope::Site(key.clone()),
                (false, false) => {
                    return Err(ChannelRecipeError::UnknownInputScope { key: key.clone() });
                }
            };
            Ok(ResolvedInputRow {
                key,
                scope,
                treatments,
            })
        })
        .collect()
}

fn apply_species_row(
    compiled: &mut CompiledChannelRecipe,
    row: &ResolvedInputRow<'_>,
    task_generator: Option<ChannelEnergyGenerator>,
) -> Result<(), ChannelRecipeError> {
    let ResolvedScope::Species(species) = &row.scope else {
        unreachable!("species pass contains only species rows");
    };
    let scope = ChannelScope::Species {
        name: species.clone(),
    };
    let definitions = parse_species_definitions(row, scope, task_generator)?;
    let flattened: Vec<_> = definitions.values().flatten().cloned().collect();
    validate_record_layer("species", row.key, &flattened)?;

    for site in compiled
        .sites
        .iter_mut()
        .filter(|site| site.atomic_number.symbol() == species.as_str())
    {
        for treatment in definitions.keys() {
            site.channels
                .retain(|record| record.treatment != *treatment);
        }
        for records in definitions.values() {
            for record in records {
                site.channels.retain(|lower| {
                    lower.treatment == record.treatment || !same_selector(lower, record)
                });
                let mut record = record.clone();
                record.scope = ChannelScope::Site {
                    name: site.site.clone(),
                };
                site.channels.push(record);
            }
        }
    }
    Ok(())
}

fn parse_species_definitions(
    row: &ResolvedInputRow<'_>,
    scope: ChannelScope,
    task_generator: Option<ChannelEnergyGenerator>,
) -> Result<BTreeMap<ChannelTreatment, Vec<ChannelRecipeRecord>>, ChannelRecipeError> {
    row.treatments
        .iter()
        .map(|(&treatment, tokens)| {
            let context = token_context(
                scope.clone(),
                treatment,
                task_generator,
                ChannelProvenance::Species,
            );
            let records = tokens
                .iter()
                .map(
                    |token| match parse_input_token(row.key, treatment, token, &context)? {
                        ParsedChannelToken::Define(record) => Ok(record),
                        ParsedChannelToken::Add(_) | ParsedChannelToken::Remove { .. } => {
                            unreachable!("the token parser enforces species definitions")
                        }
                    },
                )
                .collect::<Result<_, ChannelRecipeError>>()?;
            Ok((treatment, records))
        })
        .collect()
}

fn apply_site_row(
    compiled: &mut CompiledChannelRecipe,
    row: &ResolvedInputRow<'_>,
    task_generator: Option<ChannelEnergyGenerator>,
) -> Result<(), ChannelRecipeError> {
    let ResolvedScope::Site(site_name) = &row.scope else {
        unreachable!("site pass contains only site rows");
    };
    let scope = ChannelScope::Site {
        name: site_name.clone(),
    };
    let edits = parse_site_edits(row, scope, task_generator)?;
    validate_edit_layer(row.key, &edits)?;

    let site = compiled
        .sites
        .iter_mut()
        .find(|site| site.site == *site_name)
        .expect("resolved site row must retain a compiled site");
    for edit in edits {
        match edit {
            ParsedChannelToken::Add(record) => {
                site.channels.retain(|lower| {
                    if lower.treatment != record.treatment && same_selector(lower, &record) {
                        return false;
                    }
                    !same_instance(lower, &record)
                });
                site.channels.push(record);
            }
            ParsedChannelToken::Remove {
                identity,
                treatment,
                derivative_order,
                ..
            } => {
                let before = site.channels.len();
                site.channels.retain(|record| {
                    !(record.identity == identity
                        && record.treatment == treatment
                        && record.derivative_order == derivative_order)
                });
                if site.channels.len() == before {
                    return Err(ChannelRecipeError::DeleteMiss {
                        site: site_name.clone(),
                        treatment,
                        identity,
                        derivative_order,
                    });
                }
            }
            ParsedChannelToken::Define(_) => {
                unreachable!("the token parser enforces site edits")
            }
        }
    }
    Ok(())
}

fn parse_site_edits(
    row: &ResolvedInputRow<'_>,
    scope: ChannelScope,
    task_generator: Option<ChannelEnergyGenerator>,
) -> Result<Vec<ParsedChannelToken>, ChannelRecipeError> {
    let mut edits = Vec::new();
    for (&treatment, tokens) in row.treatments {
        let context = token_context(
            scope.clone(),
            treatment,
            task_generator,
            ChannelProvenance::Site,
        );
        for token in tokens {
            edits.push(parse_input_token(row.key, treatment, token, &context)?);
        }
    }
    Ok(edits)
}

fn parse_input_token(
    scope: &str,
    treatment: ChannelTreatment,
    token: &str,
    context: &ChannelTokenContext,
) -> Result<ParsedChannelToken, ChannelRecipeError> {
    parse_channel_token(token, context).map_err(|source| ChannelRecipeError::Token {
        scope: scope.to_owned(),
        treatment,
        token: token.to_owned(),
        source,
    })
}

fn token_context(
    scope: ChannelScope,
    treatment: ChannelTreatment,
    task_generator: Option<ChannelEnergyGenerator>,
    provenance: ChannelProvenance,
) -> ChannelTokenContext {
    ChannelTokenContext {
        scope,
        treatment,
        derivative_order: default_derivative_order(treatment),
        generator: task_generator.unwrap_or(ChannelEnergyGenerator::Atomic),
        seed: None,
        provenance,
    }
}

fn default_derivative_order(treatment: ChannelTreatment) -> u32 {
    match treatment {
        ChannelTreatment::Core | ChannelTreatment::Valence | ChannelTreatment::Lo => 0,
        ChannelTreatment::Hdlo => 2,
    }
}

fn validate_record_layer(
    layer: &'static str,
    scope: &str,
    records: &[ChannelRecipeRecord],
) -> Result<(), ChannelRecipeError> {
    for (index, record) in records.iter().enumerate() {
        for prior in &records[..index] {
            if same_instance(prior, record) {
                return Err(ChannelRecipeError::DuplicateInstance {
                    layer,
                    scope: scope.to_owned(),
                    treatment: record.treatment,
                    identity: record.identity,
                    derivative_order: record.derivative_order,
                });
            }
            if same_selector(prior, record) && prior.treatment != record.treatment {
                return Err(ChannelRecipeError::TreatmentConflict {
                    layer,
                    scope: scope.to_owned(),
                    identity: record.identity,
                    derivative_order: record.derivative_order,
                    first: prior.treatment,
                    second: record.treatment,
                });
            }
        }
    }
    Ok(())
}

fn validate_edit_layer(
    scope: &str,
    edits: &[ParsedChannelToken],
) -> Result<(), ChannelRecipeError> {
    for (index, edit) in edits.iter().enumerate() {
        let (identity, treatment, derivative_order) = edit_selector(edit);
        for prior in &edits[..index] {
            let (prior_identity, prior_treatment, prior_order) = edit_selector(prior);
            if identity == prior_identity
                && derivative_order == prior_order
                && treatment != prior_treatment
            {
                return Err(ChannelRecipeError::TreatmentConflict {
                    layer: "site",
                    scope: scope.to_owned(),
                    identity,
                    derivative_order,
                    first: prior_treatment,
                    second: treatment,
                });
            }
            let duplicate = match (prior, edit) {
                (ParsedChannelToken::Add(prior), ParsedChannelToken::Add(record)) => {
                    same_instance(prior, record)
                }
                (
                    ParsedChannelToken::Remove {
                        identity: prior_identity,
                        treatment: prior_treatment,
                        derivative_order: prior_order,
                        ..
                    },
                    ParsedChannelToken::Remove { .. },
                ) => {
                    *prior_identity == identity
                        && *prior_treatment == treatment
                        && *prior_order == derivative_order
                }
                _ => false,
            };
            if duplicate {
                return Err(ChannelRecipeError::DuplicateInstance {
                    layer: "site",
                    scope: scope.to_owned(),
                    treatment,
                    identity,
                    derivative_order,
                });
            }
        }
    }
    Ok(())
}

fn edit_selector(edit: &ParsedChannelToken) -> (ChannelIdentity, ChannelTreatment, u32) {
    match edit {
        ParsedChannelToken::Add(record) | ParsedChannelToken::Define(record) => {
            (record.identity, record.treatment, record.derivative_order)
        }
        ParsedChannelToken::Remove {
            identity,
            treatment,
            derivative_order,
            ..
        } => (*identity, *treatment, *derivative_order),
    }
}

fn validate_explicit_seeds(site: &CompiledSiteRecipe) -> Result<(), ChannelRecipeError> {
    for record in &site.channels {
        if record.generator == ChannelEnergyGenerator::Explicit && record.seed.is_none() {
            return Err(ChannelRecipeError::MissingExplicitSeed {
                site: site.site.clone(),
                identity: record.identity,
                derivative_order: record.derivative_order,
            });
        }
    }
    Ok(())
}

fn same_selector(left: &ChannelRecipeRecord, right: &ChannelRecipeRecord) -> bool {
    left.identity == right.identity && left.derivative_order == right.derivative_order
}

fn same_instance(left: &ChannelRecipeRecord, right: &ChannelRecipeRecord) -> bool {
    same_selector(left, right)
        && left.treatment == right.treatment
        && left.generator == right.generator
        && left.seed == right.seed
}

fn sort_records(records: &mut [ChannelRecipeRecord]) {
    records.sort_by(compare_records);
}

fn compare_records(left: &ChannelRecipeRecord, right: &ChannelRecipeRecord) -> Ordering {
    left.scope
        .cmp(&right.scope)
        .then_with(|| left.treatment.cmp(&right.treatment))
        .then_with(|| left.identity.cmp(&right.identity))
        .then_with(|| left.derivative_order.cmp(&right.derivative_order))
        .then_with(|| left.generator.cmp(&right.generator))
        .then_with(|| compare_seeds(left.seed, right.seed))
        .then_with(|| left.provenance.cmp(&right.provenance))
}

fn compare_seeds(
    left: Option<muffintin_core::Hartree>,
    right: Option<muffintin_core::Hartree>,
) -> Ordering {
    match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left), Some(right)) => left.get().total_cmp(&right.get()),
    }
}

#[cfg(test)]
mod tests {
    use muffintin_core::Hartree;

    use super::*;

    fn site(id: &str, symbol: &str) -> RecipeSite {
        RecipeSite {
            id: id.to_owned(),
            atomic_number: AtomicNumber::from_symbol(symbol).unwrap(),
        }
    }

    fn record(
        scope: ChannelScope,
        identity: ChannelIdentity,
        treatment: ChannelTreatment,
        derivative_order: u32,
        generator: ChannelEnergyGenerator,
        seed: Option<f64>,
    ) -> ChannelRecipeRecord {
        ChannelRecipeRecord {
            scope,
            identity,
            treatment,
            derivative_order,
            generator,
            seed: seed.map(Hartree),
            provenance: ChannelProvenance::BuiltIn,
        }
    }

    fn row(
        treatment: ChannelTreatment,
        tokens: &[&str],
    ) -> BTreeMap<ChannelTreatment, Vec<String>> {
        BTreeMap::from([(
            treatment,
            tokens.iter().map(|token| (*token).to_owned()).collect(),
        )])
    }

    #[test]
    fn artifact_round_trip_is_canonical_and_accepts_derivative_three() {
        let artifact = ChannelRecipeArtifact {
            channels: vec![record(
                ChannelScope::Species {
                    name: "Pb".to_owned(),
                },
                ChannelIdentity::ScalarL { n: 5, l: 2 },
                ChannelTreatment::Hdlo,
                3,
                ChannelEnergyGenerator::BandCenter,
                Some(-0.2),
            )],
        };
        let encoded = channel_recipe_to_toml(&artifact).unwrap();
        let decoded = parse_channel_recipe_toml(&encoded).unwrap();
        assert_eq!(decoded, artifact);
        assert!(parse_channel_recipe_toml(&("unknown = true\n".to_owned() + &encoded)).is_err());
    }

    #[test]
    fn all_five_merge_layers_and_token_suffix_have_one_way_precedence() {
        let sites = [site("Pb-1", "Pb")];
        let artifact = ChannelRecipeArtifact {
            channels: vec![record(
                ChannelScope::Species {
                    name: "Pb".to_owned(),
                },
                ChannelIdentity::ScalarL { n: 6, l: 0 },
                ChannelTreatment::Valence,
                0,
                ChannelEnergyGenerator::BandCenter,
                Some(-0.4),
            )],
        };
        let input = BTreeMap::from([
            ("Pb".to_owned(), row(ChannelTreatment::Lo, &["6s"])),
            (
                "Pb-1".to_owned(),
                row(ChannelTreatment::Lo, &["-6s", "+6s@0.25"]),
            ),
        ]);
        let compiled = compile_channel_recipe(
            &sites,
            Some(ExternalChannelRecipe {
                artifact: &artifact,
                source: "pb.toml",
            }),
            Some(ChannelEnergyGenerator::LogDerivative),
            &input,
        )
        .unwrap();
        let channels = &compiled.site("Pb-1").unwrap().channels;
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].treatment, ChannelTreatment::Lo);
        assert_eq!(channels[0].generator, ChannelEnergyGenerator::Explicit);
        assert_eq!(channels[0].seed, Some(Hartree(0.25)));
        assert_eq!(channels[0].provenance, ChannelProvenance::Site);
        assert_eq!(
            channels[0].scope,
            ChannelScope::Site {
                name: "Pb-1".to_owned()
            }
        );
    }

    #[test]
    fn explicit_empty_species_treatment_clears_the_builtin_row() {
        let sites = [site("Pb-1", "Pb")];
        let input = BTreeMap::from([("Pb".to_owned(), row(ChannelTreatment::Lo, &[]))]);
        let compiled = compile_channel_recipe(&sites, None, None, &input).unwrap();
        assert!(
            compiled
                .site("Pb-1")
                .unwrap()
                .channels
                .iter()
                .all(|record| record.treatment != ChannelTreatment::Lo)
        );
    }

    #[test]
    fn site_addition_reclassifies_the_automatic_relativistic_lo() {
        let sites = [site("Pb-1", "Pb")];
        let remove = BTreeMap::from([("Pb-1".to_owned(), row(ChannelTreatment::Lo, &["-5p-"]))]);
        let removed = compile_channel_recipe(&sites, None, None, &remove).unwrap();
        assert!(
            removed
                .site("Pb-1")
                .unwrap()
                .channels
                .iter()
                .all(|record| { record.identity != (ChannelIdentity::Kappa { n: 5, kappa: 1 }) })
        );

        let input =
            BTreeMap::from([("Pb-1".to_owned(), row(ChannelTreatment::Valence, &["+5p-"]))]);
        let compiled = compile_channel_recipe(&sites, None, None, &input).unwrap();
        let matching: Vec<_> = compiled
            .site("Pb-1")
            .unwrap()
            .channels
            .iter()
            .filter(|record| record.identity == (ChannelIdentity::Kappa { n: 5, kappa: 1 }))
            .collect();
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].treatment, ChannelTreatment::Valence);
        assert_eq!(matching[0].provenance, ChannelProvenance::Site);
    }

    #[test]
    fn site_removal_deletes_automatic_relativistic_los() {
        for (site_id, symbol, n) in [("Pb-1", "Pb", 5), ("Fr-1", "Fr", 6)] {
            let sites = [site(site_id, symbol)];
            let input = BTreeMap::from([(
                site_id.to_owned(),
                row(ChannelTreatment::Lo, &[&format!("-{n}p-")]),
            )]);
            let compiled = compile_channel_recipe(&sites, None, None, &input).unwrap();
            assert!(
                compiled
                    .site(site_id)
                    .unwrap()
                    .channels
                    .iter()
                    .all(|record| record.identity != ChannelIdentity::Kappa { n, kappa: 1 })
            );
        }
    }

    #[test]
    fn synonymous_j_tags_are_duplicate_instances() {
        let sites = [site("Pb-1", "Pb")];
        let input = BTreeMap::from([(
            "Pb".to_owned(),
            row(ChannelTreatment::Lo, &["5p-", "5p1/2"]),
        )]);
        assert!(matches!(
            compile_channel_recipe(&sites, None, None, &input),
            Err(ChannelRecipeError::DuplicateInstance { .. })
        ));
    }

    #[test]
    fn distinct_explicit_energies_are_preserved_as_a_multimap() {
        let sites = [site("Ti-1", "Ti")];
        let input = BTreeMap::from([(
            "Ti".to_owned(),
            row(ChannelTreatment::Lo, &["3d@-0.1", "3d@0.5"]),
        )]);
        let compiled = compile_channel_recipe(&sites, None, None, &input).unwrap();
        let matching: Vec<_> = compiled
            .site("Ti-1")
            .unwrap()
            .channels
            .iter()
            .filter(|record| {
                record.identity == (ChannelIdentity::ScalarL { n: 3, l: 2 })
                    && record.treatment == ChannelTreatment::Lo
            })
            .collect();
        assert_eq!(matching.len(), 2);
        assert_eq!(matching[0].seed, Some(Hartree(-0.1)));
        assert_eq!(matching[1].seed, Some(Hartree(0.5)));
    }

    #[test]
    fn unknown_and_ambiguous_dynamic_scopes_are_rejected() {
        let sites = [site("Pb-1", "Pb")];
        let unknown = BTreeMap::from([("Xe".to_owned(), row(ChannelTreatment::Valence, &["5p"]))]);
        assert!(matches!(
            compile_channel_recipe(&sites, None, None, &unknown),
            Err(ChannelRecipeError::UnknownInputScope { .. })
        ));

        let ambiguous_sites = [site("Pb", "Pb")];
        let ambiguous =
            BTreeMap::from([("Pb".to_owned(), row(ChannelTreatment::Valence, &["6s"]))]);
        assert!(matches!(
            compile_channel_recipe(&ambiguous_sites, None, None, &ambiguous),
            Err(ChannelRecipeError::AmbiguousInputScope { .. })
        ));
    }

    #[test]
    fn unknown_external_scope_and_delete_miss_are_rejected() {
        let sites = [site("Pb-1", "Pb")];
        let artifact = ChannelRecipeArtifact {
            channels: vec![record(
                ChannelScope::Site {
                    name: "missing".to_owned(),
                },
                ChannelIdentity::ScalarL { n: 6, l: 0 },
                ChannelTreatment::Valence,
                0,
                ChannelEnergyGenerator::Atomic,
                None,
            )],
        };
        assert!(matches!(
            compile_channel_recipe(
                &sites,
                Some(ExternalChannelRecipe {
                    artifact: &artifact,
                    source: "bad.toml"
                }),
                None,
                &BTreeMap::new()
            ),
            Err(ChannelRecipeError::UnknownRecipeSite { .. })
        ));

        let input = BTreeMap::from([("Pb-1".to_owned(), row(ChannelTreatment::Lo, &["-9s"]))]);
        assert!(matches!(
            compile_channel_recipe(&sites, None, None, &input),
            Err(ChannelRecipeError::DeleteMiss { .. })
        ));
    }

    #[test]
    fn absent_task_generator_preserves_external_generator_and_source() {
        let sites = [site("Pb-1", "Pb")];
        let artifact = ChannelRecipeArtifact {
            channels: vec![record(
                ChannelScope::Species {
                    name: "Pb".to_owned(),
                },
                ChannelIdentity::ScalarL { n: 6, l: 0 },
                ChannelTreatment::Valence,
                0,
                ChannelEnergyGenerator::BandCog,
                Some(-0.3),
            )],
        };
        let compiled = compile_channel_recipe(
            &sites,
            Some(ExternalChannelRecipe {
                artifact: &artifact,
                source: "pb.toml",
            }),
            None,
            &BTreeMap::new(),
        )
        .unwrap();
        let record = &compiled.site("Pb-1").unwrap().channels[0];
        assert_eq!(record.generator, ChannelEnergyGenerator::BandCog);
        assert_eq!(
            record.provenance,
            ChannelProvenance::ExternalRecipe {
                source: Some("pb.toml".to_owned())
            }
        );

        let overridden = compile_channel_recipe(
            &sites,
            Some(ExternalChannelRecipe {
                artifact: &artifact,
                source: "pb.toml",
            }),
            Some(ChannelEnergyGenerator::FermiOffset),
            &BTreeMap::new(),
        )
        .unwrap();
        let record = &overridden.site("Pb-1").unwrap().channels[0];
        assert_eq!(record.generator, ChannelEnergyGenerator::FermiOffset);
        assert_eq!(
            record.provenance,
            ChannelProvenance::ExternalRecipe {
                source: Some("pb.toml".to_owned())
            }
        );
    }
}
