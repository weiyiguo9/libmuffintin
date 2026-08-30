//! Dy bcc catalogue and producer-blocker checks.
//!
//! These tests do **not** run a Dy material MPB/THC/`c^\dagger V c` lane.
//! No honest libmuffintin-consumable Dy bcc Checkpoint V2 was found. They only
//! record the FLEUR `default.econfig` signed-$\kappa$ catalogue for $Z=66$
//! and the compiled built-in recipe. HDLO is a FLEUR inpgen basis hint, not
//! an occupation, and is therefore absent from the built-in records.

use std::collections::BTreeMap;

use muffintin::{ChannelIdentity, ChannelTreatment, RecipeSite, compile_channel_recipe};
use muffintin_dft::{
    AtomicChannelTreatment, AtomicNumber, RelativisticOrbital, fleur_default_atomic_configuration,
};

const DY_Z: u8 = 66;

fn dy_number() -> AtomicNumber {
    let atomic_number = AtomicNumber::new(DY_Z).expect("Dy is in the FLEUR catalogue");
    assert_eq!(atomic_number.symbol(), "Dy");
    assert_eq!(AtomicNumber::from_symbol("Dy"), Some(atomic_number));
    atomic_number
}

fn occupation(
    configuration: &muffintin_dft::AtomicElectronicConfiguration,
    n: u8,
    kappa: i8,
) -> muffintin_dft::AtomicOccupation {
    let orbital = RelativisticOrbital::new(n, kappa).expect("admissible (n, kappa)");
    *configuration
        .occupations()
        .iter()
        .find(|channel| channel.orbital == orbital)
        .expect("requested Dy channel must be occupied")
}

#[test]
fn dy_fleur_catalogue_has_4f_valence_and_5p12_relativistic_lo() {
    let configuration = fleur_default_atomic_configuration(dy_number());
    assert!((configuration.total_occupation() - f64::from(DY_Z)).abs() < 1.0e-12);

    let four_f_minus = occupation(&configuration, 4, 3);
    let four_f_plus = occupation(&configuration, 4, -4);
    assert_eq!(four_f_minus.treatment, AtomicChannelTreatment::Valence);
    assert_eq!(four_f_plus.treatment, AtomicChannelTreatment::Valence);
    assert!((four_f_minus.occupation - 10.0 * 3.0 / 7.0).abs() < 1.0e-12);
    assert!((four_f_plus.occupation - 10.0 * 4.0 / 7.0).abs() < 1.0e-12);

    let five_p12 = occupation(&configuration, 5, 1);
    let five_p32 = occupation(&configuration, 5, -2);
    assert_eq!(
        five_p12.treatment,
        AtomicChannelTreatment::RelativisticLocalOrbital
    );
    assert_eq!(five_p32.treatment, AtomicChannelTreatment::Valence);
    assert!((five_p12.occupation - 2.0).abs() < 1.0e-12);
    assert!((five_p32.occupation - 4.0).abs() < 1.0e-12);

    assert!(
        !configuration
            .occupations()
            .iter()
            .any(|channel| channel.treatment == AtomicChannelTreatment::Core
                && channel.orbital.principal_quantum_number() == 4
                && matches!(channel.orbital.kappa(), 3 | -4)),
        "Dy 4f must remain valence in the FLEUR default split"
    );
}

#[test]
fn dy_built_in_recipe_keeps_5p12_as_lo_and_does_not_invent_hdlo() {
    let compiled = compile_channel_recipe(
        &[RecipeSite {
            id: "Dy-1".to_owned(),
            atomic_number: dy_number(),
        }],
        None,
        None,
        &BTreeMap::new(),
    )
    .expect("built-in Dy recipe must compile");
    let site = compiled.site("Dy-1").expect("Dy-1 site");
    assert_eq!(site.atomic_number.get(), DY_Z);

    let has_4f = site.channels.iter().any(|record| {
        record.identity == ChannelIdentity::Kappa { n: 4, kappa: 3 }
            && record.treatment == ChannelTreatment::Valence
    }) && site.channels.iter().any(|record| {
        record.identity == ChannelIdentity::Kappa { n: 4, kappa: -4 }
            && record.treatment == ChannelTreatment::Valence
    });
    assert!(
        has_4f,
        "compiled Dy recipe must retain both 4f kappa partners"
    );

    let five_p12 = site
        .channels
        .iter()
        .find(|record| record.identity == ChannelIdentity::Kappa { n: 5, kappa: 1 })
        .expect("5p1/2 must be present");
    assert_eq!(five_p12.treatment, ChannelTreatment::Lo);

    assert!(
        !site
            .channels
            .iter()
            .any(|record| record.treatment == ChannelTreatment::Hdlo),
        "FLEUR default.econfig does not encode HDLO; the built-in recipe must not invent one"
    );
}
