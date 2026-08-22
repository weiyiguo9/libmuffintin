use std::f64::consts::PI;

use muffintin_core::{
    Kappa, KappaError, Lm, RelativisticChannel, RelativisticChannelError, SpinProjection,
    SpinorHarmonicTerm, TwiceMu, TwiceMuError, spinor_gaunt,
};

#[test]
fn kappa_maps_exactly_to_large_and_small_orbital_channels() {
    let cases = [
        (-1, 0, 1, 1, 2),
        (1, 1, 0, 1, 2),
        (-2, 1, 2, 3, 4),
        (2, 2, 1, 3, 4),
        (-3, 2, 3, 5, 6),
        (3, 3, 2, 5, 6),
    ];
    for (raw, large_l, small_l, twice_j, degeneracy) in cases {
        let kappa = Kappa::new(raw).unwrap();
        assert_eq!(kappa.get(), raw);
        assert_eq!(kappa.large_l(), large_l);
        assert_eq!(kappa.small_l(), small_l);
        assert_eq!(kappa.twice_j(), twice_j);
        assert_eq!(kappa.degeneracy(), degeneracy);
        let contract = kappa.angular_contract();
        assert_eq!(contract.kappa, kappa);
        assert_eq!(contract.large_l, large_l);
        assert_eq!(contract.small_l, small_l);
        assert_eq!(contract.twice_j, twice_j);
        assert_eq!(contract.degeneracy, degeneracy);
        assert_eq!(kappa.opposite().large_l(), small_l);
        assert_eq!(kappa.opposite().opposite(), kappa);
    }
    assert_eq!(Kappa::new(0), Err(KappaError::Zero));
    assert_eq!(Kappa::new(i32::MIN), Err(KappaError::OutOfRange(i32::MIN)));
}

#[test]
fn twice_mu_and_channel_validation_are_exact() {
    assert_eq!(TwiceMu::new(0), Err(TwiceMuError::Even(0)));
    assert_eq!(TwiceMu::new(2), Err(TwiceMuError::Even(2)));
    assert_eq!(TwiceMu::new(-3).unwrap().get(), -3);

    let kappa = Kappa::new(-2).unwrap();
    let enumerated: Vec<_> = kappa.twice_mu_values().map(TwiceMu::get).collect();
    assert_eq!(enumerated, [-3, -1, 1, 3]);
    assert_eq!(kappa.channels().count(), kappa.degeneracy() as usize);

    let edge = RelativisticChannel::new(kappa, TwiceMu::new(3).unwrap()).unwrap();
    assert_eq!(edge.kappa(), kappa);
    assert_eq!(edge.twice_mu().get(), 3);
    assert_eq!(edge.opposite_kappa().kappa(), Kappa::new(2).unwrap());
    assert_eq!(edge.opposite_kappa().twice_mu(), edge.twice_mu());

    assert_eq!(
        RelativisticChannel::new(kappa, TwiceMu::new(5).unwrap()),
        Err(RelativisticChannelError::MuOutsideChannel {
            kappa: -2,
            twice_mu: 5,
            twice_j: 3,
        })
    );
}

#[test]
fn clebsch_gordan_phase_normalization_and_orthogonality_are_fixed() {
    let s_up = channel(-1, 1).spinor_harmonic_terms();
    assert_eq!(s_up, [Some(term(SpinProjection::Up, 0, 0, 1.0)), None,]);
    let s_down = channel(-1, -1).spinor_harmonic_terms();
    assert_eq!(s_down, [None, Some(term(SpinProjection::Down, 0, 0, 1.0)),]);

    let p_half_up = channel(1, 1).spinor_harmonic_terms();
    assert_term(
        p_half_up[0],
        SpinProjection::Up,
        1,
        0,
        -1.0 / 3.0_f64.sqrt(),
    );
    assert_term(
        p_half_up[1],
        SpinProjection::Down,
        1,
        1,
        (2.0 / 3.0_f64).sqrt(),
    );

    for raw_kappa in -5..=5 {
        if raw_kappa == 0 {
            continue;
        }
        let kappa = Kappa::new(raw_kappa).unwrap();
        let channels: Vec<_> = kappa.channels().collect();
        for (i, &left) in channels.iter().enumerate() {
            assert!((overlap(left, left) - 1.0).abs() < 3e-15);
            for &right in &channels[..i] {
                assert!(overlap(left, right).abs() < 3e-15);
            }
        }
    }

    // The two Condon--Shortley coupled states with l=1 and mu=1/2 are
    // orthogonal only with the documented minus sign in the kappa=+l branch.
    assert!(overlap(channel(-2, 1), channel(1, 1)).abs() < 3e-15);
    assert!(overlap(channel(-2, -1), channel(1, -1)).abs() < 3e-15);
}

#[test]
fn low_order_spinor_gaunt_values_match_hand_reduction() {
    let y00 = 1.0 / (4.0 * PI).sqrt();
    let scalar = Lm::new(0, 0).unwrap();
    let dipole_z = Lm::new(1, 0).unwrap();
    let dipole_minus = Lm::new(1, -1).unwrap();

    // A scalar constant sees a normalized Omega as one, for both the large
    // kappa channel and its explicit small-component -kappa counterpart.
    let s_up = channel(-1, 1);
    assert!((spinor_gaunt(s_up, scalar, s_up) - y00).abs() < 2e-15);
    let p_up = s_up.opposite_kappa();
    assert!((spinor_gaunt(p_up, scalar, p_up) - y00).abs() < 2e-15);
    assert_eq!(spinor_gaunt(s_up, scalar, channel(-1, -1)), 0.0);

    // <Omega_-1,+1/2 | Y_10 | Omega_+1,+1/2>
    // = -1/sqrt(3) integral(Y_00 Y_10 Y_10).
    let expected_z = -y00 / 3.0_f64.sqrt();
    assert!((spinor_gaunt(s_up, dipole_z, channel(1, 1)) - expected_z).abs() < 2e-15);
    assert!((spinor_gaunt(channel(1, 1), dipole_z, s_up) - expected_z).abs() < 2e-15);

    // The M=-1 spin-flip-in-total-j channel retains the complex-harmonic
    // Condon--Shortley sign: sqrt(2/3) integral(Y_00 Y_1,-1 Y_11).
    let expected_minus = -(2.0 / 3.0_f64).sqrt() * y00;
    assert!(
        (spinor_gaunt(channel(-1, -1), dipole_minus, channel(1, 1)) - expected_minus).abs() < 2e-15
    );
}

fn channel(kappa: i32, twice_mu: i64) -> RelativisticChannel {
    RelativisticChannel::new(Kappa::new(kappa).unwrap(), TwiceMu::new(twice_mu).unwrap()).unwrap()
}

fn term(spin: SpinProjection, l: u32, m: i32, coefficient: f64) -> SpinorHarmonicTerm {
    SpinorHarmonicTerm {
        spin,
        orbital: Lm::new(l, m).unwrap(),
        coefficient,
    }
}

fn assert_term(
    actual: Option<SpinorHarmonicTerm>,
    spin: SpinProjection,
    l: u32,
    m: i32,
    coefficient: f64,
) {
    let actual = actual.unwrap();
    assert_eq!(actual.spin, spin);
    assert_eq!(actual.orbital, Lm::new(l, m).unwrap());
    assert!((actual.coefficient - coefficient).abs() < 2e-15);
}

fn overlap(left: RelativisticChannel, right: RelativisticChannel) -> f64 {
    let mut value = 0.0;
    for left_term in left.spinor_harmonic_terms().into_iter().flatten() {
        for right_term in right.spinor_harmonic_terms().into_iter().flatten() {
            if left_term.spin == right_term.spin && left_term.orbital == right_term.orbital {
                value += left_term.coefficient * right_term.coefficient;
            }
        }
    }
    value
}
