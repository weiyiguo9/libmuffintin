use muffintin_core::{Bohr, ExponentialMesh, Kappa};
use muffintin_sphere::{
    HarmonicConvention, SphereOrbital, SpinorSphereOrbital, project_orbital_pair_density,
    project_orbital_pair_density_with_convention, project_spinor_pair_density,
    project_spinor_pair_density_components,
};
use num_complex::Complex64;
use std::f64::consts::PI;

fn mesh() -> ExponentialMesh {
    ExponentialMesh::new(Bohr(1.0e-4), 0.02, 301).unwrap()
}

fn reduced(mesh: &ExponentialMesh, scale: f64) -> Vec<f64> {
    mesh.radii()
        .iter()
        .map(|radius| scale * radius.get() * (-radius.get()).exp())
        .collect()
}

#[test]
fn real_tesseral_pair_uses_the_real_gaunt_convention() {
    let mesh = mesh();
    let s = SphereOrbital::new(0, 0, reduced(&mesh, 0.8), None).unwrap();
    let p_x = SphereOrbital::new(1, 1, reduced(&mesh, 1.1), None).unwrap();
    let pair =
        project_orbital_pair_density_with_convention(&mesh, &s, &p_x, HarmonicConvention::Real)
            .unwrap();
    assert_eq!(pair.convention(), HarmonicConvention::Real);
    assert!(
        pair.channel(1, 1)
            .unwrap()
            .iter()
            .any(|value| value.re != 0.0)
    );
    let expected = muffintin_core::Lm::new(1, 1).unwrap();
    for (channel, values) in pair.channels() {
        if channel != expected {
            assert!(
                values
                    .iter()
                    .all(|value| *value == Complex64::new(0.0, 0.0))
            );
        }
    }
}

#[test]
fn s_orbital_density_integrates_to_the_reduced_radial_norm_and_is_real() {
    let mesh = mesh();
    let p = reduced(&mesh, 1.3);
    let orbital = SphereOrbital::new(0, 0, p.clone(), None).unwrap();
    let density = project_orbital_pair_density(&mesh, &orbital, &orbital).unwrap();
    density.validate_physical_reality(1.0e-14).unwrap();

    let y00_integral = (4.0 * PI).sqrt();
    let physical_radial = density
        .channel(0, 0)
        .unwrap()
        .iter()
        .zip(mesh.radii())
        .map(|(coefficient, radius)| coefficient.re * radius.get().powi(2) * y00_integral)
        .collect::<Vec<_>>();
    let projected_norm = mesh.integrate(&physical_radial).unwrap();
    let radial_norm = mesh
        .integrate(&p.iter().map(|value| value * value).collect::<Vec<_>>())
        .unwrap();
    assert!((projected_norm - radial_norm).abs() < 2.0e-14 * (1.0 + radial_norm));
}

#[test]
fn reversing_a_complex_orbital_pair_gives_the_conjugate_physical_field() {
    let mesh = mesh();
    let s = SphereOrbital::new(0, 0, reduced(&mesh, 0.8), None).unwrap();
    let p_plus = SphereOrbital::new(1, 1, reduced(&mesh, 1.1), None).unwrap();
    let sp = project_orbital_pair_density(&mesh, &s, &p_plus).unwrap();
    let ps = project_orbital_pair_density(&mesh, &p_plus, &s).unwrap();

    for l in 0..=1 {
        for m in -(l as i32)..=l as i32 {
            let phase = if m.unsigned_abs() % 2 == 0 { 1.0 } else { -1.0 };
            let sp_values = sp.channel(l, m);
            let ps_partner = ps.channel(l, -m);
            for index in 0..mesh.len() {
                let value = sp_values.map_or(Complex64::new(0.0, 0.0), |values| values[index]);
                let reverse = ps_partner.map_or(Complex64::new(0.0, 0.0), |values| values[index]);
                assert!((reverse - phase * value.conj()).norm() < 2.0e-15);
            }
        }
    }

    let mut real_combination = sp.clone();
    real_combination
        .add_scaled(Complex64::new(1.0, 0.0), &ps)
        .unwrap();
    real_combination.validate_physical_reality(2.0e-14).unwrap();
}

#[test]
fn closed_kappa_shell_is_isotropic_and_contains_p_squared_plus_q_squared() {
    let mesh = mesh();
    let kappa = Kappa::new(-2).unwrap();
    let p = reduced(&mesh, 0.9);
    let q = reduced(&mesh, 0.35);
    let mut shell = None;
    for channel in kappa.channels() {
        let orbital = SpinorSphereOrbital::new(channel, p.clone(), q.clone()).unwrap();
        let pair = project_spinor_pair_density(&mesh, &orbital, &orbital).unwrap();
        match &mut shell {
            None => shell = Some(pair),
            Some(total) => total.add_scaled(Complex64::new(1.0, 0.0), &pair).unwrap(),
        }
    }
    let shell = shell.unwrap();
    shell.validate_physical_reality(3.0e-14).unwrap();

    for m in -2..=2 {
        if let Some(values) = shell.channel(2, m) {
            assert!(values.iter().all(|value| value.norm() < 3.0e-14));
        }
    }
    let coefficient = shell.channel(0, 0).unwrap();
    let expected_factor = f64::from(kappa.degeneracy()) / (4.0 * PI).sqrt();
    for (index, radius) in mesh.radii().iter().enumerate() {
        let expected =
            expected_factor * (p[index] * p[index] + q[index] * q[index]) / radius.get().powi(2);
        assert!((coefficient[index].re - expected).abs() < 4.0e-14 * (1.0 + expected));
        assert_eq!(coefficient[index].im, 0.0);
    }
}

#[test]
fn full_spinor_pair_projects_cartesian_pauli_density_and_reverses_hermitianly() {
    let mesh = mesh();
    let mut channels = Kappa::new(-1).unwrap().channels();
    let down = channels.next().unwrap();
    let up = channels.next().unwrap();
    let radial = reduced(&mesh, 0.9);
    let zero = vec![0.0; mesh.len()];
    let up_orbital = SpinorSphereOrbital::new(up, radial.clone(), zero.clone()).unwrap();
    let down_orbital = SpinorSphereOrbital::new(down, radial, zero).unwrap();

    let diagonal = project_spinor_pair_density_components(&mesh, &up_orbital, &up_orbital).unwrap();
    for index in 0..mesh.len() {
        assert_eq!(
            diagonal.charge().channel(0, 0).unwrap()[index],
            diagonal.spin()[2].channel(0, 0).unwrap()[index]
        );
        assert_eq!(
            diagonal.spin()[0].channel(0, 0).unwrap()[index],
            Complex64::new(0.0, 0.0)
        );
        assert_eq!(
            diagonal.spin()[1].channel(0, 0).unwrap()[index],
            Complex64::new(0.0, 0.0)
        );
    }

    let up_down =
        project_spinor_pair_density_components(&mesh, &up_orbital, &down_orbital).unwrap();
    let down_up =
        project_spinor_pair_density_components(&mesh, &down_orbital, &up_orbital).unwrap();
    for axis in 0..3 {
        for (channel, values) in up_down.spin()[axis].channels() {
            let phase = if channel.m.unsigned_abs() % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            let reverse = down_up.spin()[axis].channel(channel.l, -channel.m).unwrap();
            for (&value, &reverse) in values.iter().zip(reverse) {
                assert!((reverse - phase * value.conj()).norm() < 2.0e-15);
            }
        }
    }
}
