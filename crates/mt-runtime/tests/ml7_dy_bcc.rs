//! M-L7 Dy bcc catalogue and producer-blocker checks.
//!
//! These tests do **not** run a Dy material MPB/THC/`c^\dagger V c` lane.
//! No honest libmuffintin-consumable Dy bcc Snapshot V2 was found. They only
//! record the FLEUR `default.econfig` signed-$\kappa$ catalogue for $Z=66$
//! and the compiled built-in recipe. HDLO is a FLEUR inpgen basis hint, not
//! an occupation, and is therefore absent from the built-in records.

use std::collections::BTreeMap;

use muffintin::{ChannelIdentity, ChannelTreatment, RecipeSite, compile_channel_recipe};
use muffintin_core::Hartree;
use muffintin_dft::{
    AtomicChannelTreatment, AtomicNumber, NoncollinearXcRoute, RelativisticOrbital, ScfBasis,
    ScfConfig, ScfConvergence, ScfExchangeCorrelation, ScfKMesh, ScfMixing, ScfOccupations,
    ScfRelativity, XcFunctional, fleur_default_atomic_configuration,
};

#[path = "ml7_material_common.rs"]
mod ml7_material_common;

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

fn spinor_config() -> ScfConfig {
    ScfConfig {
        electron_count: f64::from(DY_Z),
        k_mesh: ScfKMesh {
            divisions: [1, 1, 1],
            shift: [0.0; 3],
        },
        basis: ScfBasis {
            plane_wave_cutoff: 1.0,
            l_max: 3,
            channels: Vec::new(),
            resolved_channels: Vec::new(),
        },
        occupations: ScfOccupations::FermiDirac {
            temperature: Hartree(0.02),
        },
        exchange_correlation: ScfExchangeCorrelation {
            functional: XcFunctional::LdaPw92,
            noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
        },
        mixing: ScfMixing::Linear { alpha: 1.0 },
        relativity: ScfRelativity::SpinorFirstVariation,
        convergence: ScfConvergence {
            energy_tolerance: Hartree(1.0),
            density_tolerance: 1.0,
            max_iterations: 1,
        },
        core_sites: Vec::new(),
    }
}

#[test]
fn dy_bcc_material_snapshot_is_absent() {
    // Consume the p6 helper load path. A later material lane would then call
    // ordered_q_slice, compare_qrcp_cholesky, MPB/THC c†Vc, and MLDUMP.
    // That Snapshot V2 does not exist: WSL <thc-experiment> has only scalar-collinear
    // Sm FCC SPEX dumps, the FLEUR converter is frozen, and there is no
    // SPEX→Snapshot V2 importer. Do not substitute synthetic atomic data.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ml7-dy/absent.toml");
    assert!(
        !path.exists()
            && !std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../ml7-dy")
                .exists(),
        "ml7-dy prefixed snapshot paths must stay absent until a real producer exists"
    );
    let provenance = ml7_material_common::Ml7Provenance {
        snapshot_path: path.clone(),
        snapshot_sha256: String::new(),
        producer: "absent".to_owned(),
    };
    match ml7_material_common::load_spinor_snapshot_v2(&path, spinor_config(), provenance) {
        Err(ml7_material_common::Ml7CommonError::MissingSnapshot(missing)) => {
            assert_eq!(missing, path);
        }
        Err(_) => panic!("expected missing Snapshot V2 from shared helper"),
        Ok(_) => panic!("absent ml7-dy snapshot must not load"),
    }
}
