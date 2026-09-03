//! Production density-to-Hartree gates for the basis-neutral Weinert path.

use muffintin_core::{
    Bohr, ExponentialMesh, FourierLayout, HermitianFourierField, InterstitialGeometry, InverseBohr,
    ReciprocalLattice, Sphere, VolumeBohr3, complex_spherical_harmonics, lm_count, lm_index,
    spherical_bessel_j,
};
use muffintin_coulomb::{
    CoulombRequest, EwaldScan, HartreeError, HartreeGauge, MuffinTinChargeDensity,
    PeriodicChargeTreatment, WeinertChargeDensity, WeinertHartreeSpec,
    converged_ewald_point_kernel, intra_sphere_poisson, multipole_moment,
    solve_periodic_nuclear_potential, solve_weinert_hartree,
};
use muffintin_prodbasis::TransferQ;
use num_complex::Complex64;
use std::f64::consts::PI;

const LATTICE: f64 = 8.0;
const RADIUS: f64 = 1.0;
const CENTER: [Bohr; 3] = [Bohr(4.0), Bohr(4.0), Bohr(4.0)];

fn reciprocal() -> ReciprocalLattice {
    ReciprocalLattice::from_direct([
        [Bohr(LATTICE), Bohr(0.0), Bohr(0.0)],
        [Bohr(0.0), Bohr(LATTICE), Bohr(0.0)],
        [Bohr(0.0), Bohr(0.0), Bohr(LATTICE)],
    ])
    .unwrap()
}

fn layout(shells: f64) -> FourierLayout {
    let reciprocal = reciprocal();
    let cutoff = InverseBohr(shells * 2.0 * PI / LATTICE);
    FourierLayout::new(reciprocal, reciprocal.enumerate(cutoff).unwrap()).unwrap()
}

fn mesh() -> ExponentialMesh {
    let first: f64 = 1.0e-5;
    let number = 121;
    let increment = (RADIUS / first).ln() / (number - 1) as f64;
    ExponentialMesh::new(Bohr(first), increment, number).unwrap()
}

fn geometry(with_sphere: bool) -> InterstitialGeometry {
    let spheres = if with_sphere {
        vec![Sphere {
            center: CENTER,
            radius: Bohr(RADIUS),
        }]
    } else {
        Vec::new()
    };
    InterstitialGeometry::new(VolumeBohr3(LATTICE.powi(3)), spheres).unwrap()
}

fn geometry_at(center: [Bohr; 3]) -> InterstitialGeometry {
    InterstitialGeometry::new(
        VolumeBohr3(LATTICE.powi(3)),
        vec![Sphere {
            center,
            radius: Bohr(RADIUS),
        }],
    )
    .unwrap()
}

fn field(
    layout: FourierLayout,
    coefficient: impl Fn([i32; 3]) -> Complex64,
) -> HermitianFourierField {
    let coefficients = layout
        .vectors()
        .iter()
        .map(|g| coefficient(g.index))
        .collect();
    HermitianFourierField::new(layout, coefficients).unwrap()
}

fn spherical_density(value: f64) -> MuffinTinChargeDensity {
    let mesh = mesh();
    MuffinTinChargeDensity::new(
        mesh.clone(),
        0,
        vec![Complex64::new(value, 0.0); mesh.len()],
    )
    .unwrap()
}

fn zero_template(center: [Bohr; 3], layout: FourierLayout, l_max: u32) -> WeinertChargeDensity {
    let mesh = mesh();
    let muffin_tin = MuffinTinChargeDensity::new(
        mesh.clone(),
        l_max,
        vec![Complex64::default(); lm_count(l_max) * mesh.len()],
    )
    .unwrap();
    let interstitial = field(layout, |_| Complex64::default());
    WeinertChargeDensity::new(geometry_at(center), vec![muffin_tin], interstitial).unwrap()
}

#[test]
fn empty_sphere_fourier_mode_is_raw_four_pi_over_g_squared() {
    let layout = layout(1.0);
    let rho = Complex64::new(0.031, -0.017);
    let interstitial = field(layout.clone(), |g| match g {
        [1, 0, 0] => rho,
        [-1, 0, 0] => rho.conj(),
        _ => Complex64::default(),
    });
    let density = WeinertChargeDensity::new(geometry(false), Vec::new(), interstitial).unwrap();
    let potential = solve_weinert_hartree(&density, WeinertHartreeSpec::default()).unwrap();

    let gnorm = layout
        .vectors()
        .iter()
        .find(|g| g.index == [1, 0, 0])
        .unwrap()
        .norm
        .get();
    let expected = 4.0 * PI / gnorm.powi(2) * rho;
    let positive = potential
        .interstitial()
        .coefficient([1, 0, 0])
        .unwrap()
        .as_complex();
    let negative = potential
        .interstitial()
        .coefficient([-1, 0, 0])
        .unwrap()
        .as_complex();
    assert!((positive - expected).norm() < 1.0e-13);
    assert_eq!(negative, positive.conj());
    assert_eq!(
        potential
            .interstitial()
            .coefficient([0, 0, 0])
            .unwrap()
            .as_complex(),
        Complex64::default()
    );
    assert_eq!(potential.gauge(), HartreeGauge::ZeroInterstitialFourierMean);
}

#[test]
fn nonneutral_periodic_source_is_rejected_before_gauge_fixing() {
    let layout = layout(1.0);
    let interstitial = field(layout, |g| {
        if g == [0; 3] {
            Complex64::new(0.01, 0.0)
        } else {
            Complex64::default()
        }
    });
    let density = WeinertChargeDensity::new(geometry(false), Vec::new(), interstitial).unwrap();
    let error = solve_weinert_hartree(&density, WeinertHartreeSpec::neutral(4, 1.0e-12).unwrap())
        .unwrap_err();
    match error {
        HartreeError::NonNeutral { charge, tolerance } => {
            assert!((charge - 0.01 * LATTICE.powi(3)).abs() < 1.0e-12);
            assert_eq!(tolerance, 1.0e-12);
        }
        other => panic!("expected nonneutral error, got {other}"),
    }
}

#[test]
fn electronic_density_uses_explicit_uniform_background_and_zero_gauge() {
    let layout = layout(1.0);
    let mean_density = 0.01;
    let mode = Complex64::new(0.031, -0.017);
    let interstitial = field(layout.clone(), |g| match g {
        [0, 0, 0] => Complex64::new(mean_density, 0.0),
        [1, 0, 0] => mode,
        [-1, 0, 0] => mode.conj(),
        _ => Complex64::default(),
    });
    let density = WeinertChargeDensity::new(geometry(false), Vec::new(), interstitial).unwrap();
    let potential =
        solve_weinert_hartree(&density, WeinertHartreeSpec::electronic(4).unwrap()).unwrap();

    let source_charge = mean_density * LATTICE.powi(3);
    assert!((potential.source_charge() - source_charge).abs() < 1.0e-12);
    assert!((potential.neutralizing_background_density() + mean_density).abs() < 1.0e-15);
    assert_eq!(
        potential.charge_treatment(),
        PeriodicChargeTreatment::ElectronicWithUniformBackground
    );
    assert_eq!(
        potential
            .interstitial()
            .coefficient([0, 0, 0])
            .unwrap()
            .as_complex(),
        Complex64::default()
    );
    let gnorm = layout
        .vectors()
        .iter()
        .find(|g| g.index == [1, 0, 0])
        .unwrap()
        .norm
        .get();
    let expected = 4.0 * PI / gnorm.powi(2) * mode;
    let actual = potential
        .interstitial()
        .coefficient([1, 0, 0])
        .unwrap()
        .as_complex();
    assert!((actual - expected).norm() < 1.0e-13);
}

#[test]
fn periodic_nucleus_has_weinert_fourier_form_and_minus_z_over_r_core() {
    let layout = layout(4.0);
    let template = zero_template(CENTER, layout.clone(), 1);
    let charge = 2.3;
    let nuclear = solve_periodic_nuclear_potential(
        &template,
        &[charge],
        WeinertHartreeSpec::electronic(4).unwrap(),
    )
    .unwrap();

    assert_eq!(nuclear.gauge(), HartreeGauge::ZeroInterstitialFourierMean);
    assert_eq!(nuclear.nuclear_charges(), &[charge]);
    assert_eq!(nuclear.source_charge(), -charge);
    assert!((nuclear.neutralizing_background_density() - charge / LATTICE.powi(3)).abs() < 1.0e-15);
    assert_eq!(
        nuclear
            .interstitial()
            .coefficient([0, 0, 0])
            .unwrap()
            .as_complex(),
        Complex64::default()
    );
    let g = layout
        .vectors()
        .iter()
        .find(|g| g.index == [1, 0, 0])
        .unwrap();
    let phase = g
        .cartesian
        .iter()
        .zip(CENTER)
        .map(|(component, coordinate)| component.get() * coordinate.get())
        .sum::<f64>();
    // The Fourier transform of normalized (R^2-r^2)^4 is 11!! j5(GR)/(GR)^5.
    let x = g.norm.get() * RADIUS;
    let form = 10_395.0 * spherical_bessel_j(5, x) / x.powi(5);
    let expected = -4.0 * PI * charge * form / (LATTICE.powi(3) * g.norm.get().powi(2))
        * Complex64::from_polar(1.0, -phase);
    let actual = nuclear
        .interstitial()
        .coefficient(g.index)
        .unwrap()
        .as_complex();
    assert!((actual - expected).norm() < 1.0e-14);

    let monopole = nuclear.muffin_tins()[0].channel(0, 0).unwrap();
    let radii = nuclear.muffin_tins()[0].mesh().radii();
    let first = 5;
    let second = 35;
    let physical_difference =
        (monopole[first].real().get() - monopole[second].real().get()) / (4.0 * PI).sqrt();
    let expected_difference = -charge * (1.0 / radii[first].get() - 1.0 / radii[second].get());
    assert!((physical_difference - expected_difference).abs() < 1.0e-9);
}

#[test]
fn nuclear_translation_covariance_and_mt_local_potential_are_preserved() {
    let layout = layout(4.0);
    let shifted = [Bohr(4.3), Bohr(3.8), Bohr(4.1)];
    let displacement = [0.3, -0.2, 0.1];
    let original = solve_periodic_nuclear_potential(
        &zero_template(CENTER, layout.clone(), 1),
        &[1.7],
        WeinertHartreeSpec::electronic(4).unwrap(),
    )
    .unwrap();
    let translated = solve_periodic_nuclear_potential(
        &zero_template(shifted, layout.clone(), 1),
        &[1.7],
        WeinertHartreeSpec::electronic(4).unwrap(),
    )
    .unwrap();

    for g in layout.vectors() {
        let left = original
            .interstitial()
            .coefficient(g.index)
            .unwrap()
            .as_complex();
        let right = translated
            .interstitial()
            .coefficient(g.index)
            .unwrap()
            .as_complex();
        let phase = g
            .cartesian
            .iter()
            .zip(displacement)
            .map(|(component, delta)| component.get() * delta)
            .sum::<f64>();
        assert!((right - left * Complex64::from_polar(1.0, -phase)).norm() < 2.0e-13);
    }
    for l in 0..=1 {
        for m in -(l as i32)..=l as i32 {
            for (&left, &right) in original.muffin_tins()[0]
                .channel(l, m)
                .unwrap()
                .iter()
                .zip(translated.muffin_tins()[0].channel(l, m).unwrap())
            {
                assert!((left.as_complex() - right.as_complex()).norm() < 3.0e-12);
            }
        }
    }
}

#[test]
fn electronic_and_nuclear_parts_add_to_a_neutral_total_source() {
    let layout = layout(4.0);
    let charge = 2.0;
    let mesh = mesh();
    let unit_basm = mesh
        .radii()
        .iter()
        .map(|radius| (4.0 * PI).sqrt() * radius.get())
        .collect::<Vec<_>>();
    let represented_mt_volume = (4.0 * PI).sqrt() * multipole_moment(0, &mesh, &unit_basm).unwrap();
    let interstitial_volume = LATTICE.powi(3) - 4.0 * PI * RADIUS.powi(3) / 3.0;
    let mean_density = charge / (represented_mt_volume + interstitial_volume);
    let electron_mt = MuffinTinChargeDensity::new(
        mesh.clone(),
        0,
        vec![Complex64::new((4.0 * PI).sqrt() * mean_density, 0.0); mesh.len()],
    )
    .unwrap();
    let electron_i = field(layout.clone(), |g| {
        if g == [0; 3] {
            Complex64::new(mean_density, 0.0)
        } else {
            Complex64::default()
        }
    });
    let density = WeinertChargeDensity::new(geometry(true), vec![electron_mt], electron_i).unwrap();
    let electronic =
        solve_weinert_hartree(&density, WeinertHartreeSpec::electronic(4).unwrap()).unwrap();
    let nuclear = solve_periodic_nuclear_potential(
        &density,
        &[charge],
        WeinertHartreeSpec::electronic(4).unwrap(),
    )
    .unwrap();
    let total = electronic.add_nuclear_external(&nuclear).unwrap();

    assert!((electronic.source_charge() - charge).abs() < 2.0e-12);
    assert!(
        (electronic.source_charge() - nuclear.nuclear_charges().iter().sum::<f64>()).abs()
            < 2.0e-12
    );
    assert_eq!(total.gauge(), HartreeGauge::ZeroInterstitialFourierMean);
    assert!(total.source_charge().abs() < 2.0e-12);
    assert!(total.neutralizing_background_density().abs() < 2.0e-15);
    for g in layout.vectors() {
        let sum = electronic
            .interstitial()
            .coefficient(g.index)
            .unwrap()
            .as_complex()
            + nuclear
                .interstitial()
                .coefficient(g.index)
                .unwrap()
                .as_complex();
        assert_eq!(
            total
                .interstitial()
                .coefficient(g.index)
                .unwrap()
                .as_complex(),
            sum
        );
    }
}

#[test]
fn weinert_nuclear_sum_matches_ewald_in_the_pseudocharge_gauge() {
    let layout = layout(18.0);
    let template = zero_template(CENTER, layout.clone(), 0);
    let charge = 1.0;
    let nuclear = solve_periodic_nuclear_potential(
        &template,
        &[charge],
        WeinertHartreeSpec::electronic(4).unwrap(),
    )
    .unwrap();
    let evaluation = [Bohr(1.3), Bohr(2.1), Bohr(3.2)];
    let mut fourier_value = Complex64::default();
    for g in layout.vectors() {
        let phase = g
            .cartesian
            .iter()
            .zip(evaluation)
            .map(|(component, coordinate)| component.get() * coordinate.get())
            .sum::<f64>();
        fourier_value += nuclear
            .interstitial()
            .coefficient(g.index)
            .unwrap()
            .as_complex()
            * Complex64::from_polar(1.0, phase);
    }
    let request = CoulombRequest::cubic(LATTICE, 0).unwrap();
    let q0 = TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
    let ewald = converged_ewald_point_kernel(
        request.cell(),
        request.reciprocal(),
        q0,
        CENTER,
        evaluation,
        EwaldScan {
            tolerance: 1.0e-6,
            max_steps: 8,
        },
    )
    .unwrap();
    // Removing the pseudocharge G=0 potential shifts the outside potential by
    // -2*pi*Z*<r^2>/(3*Omega), with <r^2>=3*R^2/(2*N+5).
    let gauge_shift = -2.0 * PI * charge * RADIUS.powi(2) / (13.0 * LATTICE.powi(3));
    assert!(
        (fourier_value + charge * ewald.value - gauge_shift).norm() < 1.0e-6,
        "finite-G {fourier_value}, Ewald {}",
        -charge * ewald.value
    );
}

#[test]
fn neutral_spherical_sphere_matches_poisson_and_fourier_boundary() {
    let site_density = spherical_density(0.2);
    let radial = site_density.channel(0, 0).unwrap();
    let basm = site_density
        .mesh()
        .radii()
        .iter()
        .zip(radial)
        .map(|(radius, value)| radius.get() * value.re)
        .collect::<Vec<_>>();
    let q00 = multipole_moment(0, site_density.mesh(), &basm).unwrap();
    let sphere_charge = (4.0 * PI).sqrt() * q00;
    let interstitial_volume = LATTICE.powi(3) - 4.0 * PI * RADIUS.powi(3) / 3.0;
    let background = -sphere_charge / interstitial_volume;

    let layout = layout(4.0);
    let interstitial = field(layout.clone(), |g| {
        if g == [0; 3] {
            Complex64::new(background, 0.0)
        } else {
            Complex64::default()
        }
    });
    let density =
        WeinertChargeDensity::new(geometry(true), vec![site_density.clone()], interstitial)
            .unwrap();
    let potential =
        solve_weinert_hartree(&density, WeinertHartreeSpec::neutral(4, 1.0e-10).unwrap()).unwrap();
    assert!(potential.source_charge().abs() < 1.0e-12);
    assert_eq!(potential.neutralizing_background_density(), 0.0);

    let channel = potential.muffin_tins()[0].channel(0, 0).unwrap();
    let boundary = channel.last().unwrap().as_complex();
    let y00 = complex_spherical_harmonics(0, [0.0; 3])[0];
    let mut fourier_boundary = Complex64::default();
    for g in layout.vectors() {
        let coefficient = potential
            .interstitial()
            .coefficient(g.index)
            .unwrap()
            .as_complex();
        let phase = g
            .cartesian
            .iter()
            .zip(CENTER)
            .map(|(component, coordinate)| component.get() * coordinate.get())
            .sum::<f64>();
        fourier_boundary += 4.0
            * PI
            * spherical_bessel_j(0, g.norm.get() * RADIUS)
            * Complex64::from_polar(1.0, phase)
            * y00.conj()
            * coefficient;
    }
    assert!((boundary - fourier_boundary).norm() < 2.0e-10);

    let sample = site_density.mesh().len() / 2;
    let radius = site_density.mesh().radii()[sample].get();
    let expected_difference = 2.0 * PI / 3.0 * 0.2 * (RADIUS.powi(2) - radius.powi(2));
    let actual_difference = channel[sample].real().get() - channel.last().unwrap().real().get();
    assert!(
        (actual_difference - expected_difference).abs() < 2.0e-6,
        "radial Poisson difference {actual_difference}, expected {expected_difference}"
    );

    // Recover the isolated-sphere contribution by removing the constant
    // homogeneous match, then compare its contraction with the established
    // intra-sphere Poisson kernel.
    let isolated_boundary = 4.0 * PI * q00 / RADIUS;
    let homogeneous = channel.last().unwrap().real().get() - isolated_boundary;
    let integrand = site_density
        .mesh()
        .radii()
        .iter()
        .zip(radial)
        .zip(channel)
        .map(|((radius, rho), value)| {
            rho.re * (value.real().get() - homogeneous) * radius.get().powi(2)
        })
        .collect::<Vec<_>>();
    let contracted = site_density.mesh().integrate(&integrand).unwrap();
    let established = intra_sphere_poisson(0, site_density.mesh(), &basm, &basm).unwrap();
    assert!((contracted - established).abs() < 2.0e-8);
}

#[test]
fn neutral_dipole_matches_known_multipole_green_function() {
    let mesh = mesh();
    let l_max = 1;
    let mut coefficients = vec![Complex64::default(); lm_count(l_max) * mesh.len()];
    let amplitude = 0.37;
    let lm10 = lm_index(1, 0).unwrap();
    for (index, radius) in mesh.radii().iter().enumerate() {
        coefficients[lm10 * mesh.len() + index] = Complex64::new(amplitude * radius.get(), 0.0);
    }
    let site = MuffinTinChargeDensity::new(mesh.clone(), l_max, coefficients).unwrap();
    let layout = layout(4.0);
    let interstitial = field(layout, |_| Complex64::default());
    let density = WeinertChargeDensity::new(geometry(true), vec![site], interstitial).unwrap();
    let potential =
        solve_weinert_hartree(&density, WeinertHartreeSpec::neutral(4, 1.0e-12).unwrap()).unwrap();

    let channel = potential.muffin_tins()[0].channel(1, 0).unwrap();
    let boundary = channel.last().unwrap().real().get();
    for index in [20, 50, 80] {
        let radius = mesh.radii()[index].get();
        let homogeneous_surface = boundary * radius / RADIUS;
        let expected = 2.0 * PI * amplitude * radius * (RADIUS.powi(2) - radius.powi(2)) / 5.0;
        let actual = channel[index].real().get() - homogeneous_surface;
        assert!(
            (actual - expected).abs() < 3.0e-6,
            "r={radius}: multipole Green function {actual}, expected {expected}"
        );
    }
    assert!(
        potential.muffin_tins()[0]
            .channel(1, 1)
            .unwrap()
            .iter()
            .all(|value| value.as_complex().norm() < 1.0e-12)
    );
}

#[test]
fn muffin_tin_reality_relation_is_enforced() {
    let mesh = mesh();
    let mut coefficients = vec![Complex64::default(); lm_count(1) * mesh.len()];
    coefficients[lm_index(1, 1).unwrap() * mesh.len()] = Complex64::new(0.2, 0.1);
    coefficients[lm_index(1, -1).unwrap() * mesh.len()] = Complex64::new(0.2, -0.1);
    assert!(matches!(
        MuffinTinChargeDensity::new(mesh, 1, coefficients),
        Err(HartreeError::NonRealMuffinTinDensity { l: 1, m: 1, .. })
    ));
}
