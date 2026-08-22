use muffintin_core::{
    Bohr, ExponentialMesh, Kappa, Lm, RelativisticChannel, TwiceMu, spinor_gaunt,
};
use muffintin_radial::{RadialComponents, RadialIntegralKernel, radial_integral};
use muffintin_sphere::{
    HarmonicConvention, MatrixElementError, SphereField, SphereFieldError, SphereOrbital,
    SphereOrbitalError, SpinorSphereOrbital, matrix_element, spinor_matrix_element,
};
use num_complex::Complex64;
use std::f64::consts::PI;

fn mesh() -> ExponentialMesh {
    ExponentialMesh::new(Bohr(1.0e-4), 0.02, 301).unwrap()
}

fn orbital(l: u32, m: i32, mesh: &ExponentialMesh, scale: f64) -> SphereOrbital {
    let large = mesh
        .radii()
        .iter()
        .map(|radius| scale * radius.get() * (-radius.get()).exp())
        .collect();
    SphereOrbital::new(l, m, large, None).unwrap()
}

fn channel(kappa: i32, twice_mu: i64) -> RelativisticChannel {
    RelativisticChannel::new(Kappa::new(kappa).unwrap(), TwiceMu::new(twice_mu).unwrap()).unwrap()
}

fn radial_component(mesh: &ExponentialMesh, scale: f64) -> Vec<f64> {
    mesh.radii()
        .iter()
        .map(|radius| scale * radius.get() * (-radius.get()).exp())
        .collect()
}

fn component_overlap(mesh: &ExponentialMesh, left: &[f64], right: &[f64]) -> f64 {
    let values: Vec<f64> = left
        .iter()
        .zip(right)
        .map(|(&left, &right)| left * right)
        .collect();
    mesh.integrate(&values).unwrap()
}

#[test]
fn constant_field_obeys_selection_rules_and_hermiticity() {
    let mesh = mesh();
    let coefficient = (4.0 * PI).sqrt();
    let field = SphereField::new(
        HarmonicConvention::Complex,
        [((0, 0), vec![Complex64::new(coefficient, 0.0); mesh.len()])],
    )
    .unwrap();
    let s = orbital(0, 0, &mesh, 0.8);
    let p0 = orbital(1, 0, &mesh, 1.1);
    let p1_left = orbital(1, 1, &mesh, 0.7);
    let p1_right = orbital(1, 1, &mesh, 1.3);

    assert_eq!(
        matrix_element(&mesh, &s, &field, &p0).unwrap(),
        Complex64::new(0.0, 0.0)
    );
    assert_eq!(
        matrix_element(&mesh, &p0, &field, &s).unwrap(),
        Complex64::new(0.0, 0.0)
    );

    let radial =
        radial_integral(&mesh, &p1_left, &p1_right, RadialIntegralKernel::Overlap).unwrap();
    let left_right = matrix_element(&mesh, &p1_left, &field, &p1_right).unwrap();
    let right_left = matrix_element(&mesh, &p1_right, &field, &p1_left).unwrap();
    assert!((left_right.re - radial).abs() < 2.0e-14 * (1.0 + radial.abs()));
    assert_eq!(left_right, right_left.conj());
}

#[test]
fn real_and_complex_low_order_channels_match_reference_values() {
    let mesh = mesh();
    let s = orbital(0, 0, &mesh, 1.0);
    let px = orbital(1, 1, &mesh, 1.0);
    let radial = radial_integral(&mesh, &s, &px, RadialIntegralKernel::Overlap).unwrap();
    let y00 = 1.0 / (4.0 * PI).sqrt();

    let real = SphereField::from_real_channels([((1, 1), vec![2.0; mesh.len()])]).unwrap();
    let real_value = matrix_element(&mesh, &s, &real, &px).unwrap();
    assert_eq!(real_value.im, 0.0);
    assert!((real_value.re - 2.0 * y00 * radial).abs() < 2.0e-14 * (1.0 + radial.abs()));

    // A real physical l=1 field in the complex basis obeys
    // V_{1,-1} = -conj(V_{1,1}).  This pair also exercises complex-valued
    // off-diagonal Hermiticity, not just a real m=0 special case.
    let v11 = Complex64::new(0.6, -0.25);
    let v1n1 = -v11.conj();
    let complex = SphereField::new(
        HarmonicConvention::Complex,
        [
            ((1, -1), vec![v1n1; mesh.len()]),
            ((1, 1), vec![v11; mesh.len()]),
        ],
    )
    .unwrap();
    let sp = matrix_element(&mesh, &s, &complex, &px).unwrap();
    let ps = matrix_element(&mesh, &px, &complex, &s).unwrap();
    let expected = v11.conj() * y00 * radial;
    assert!((sp - expected).norm() < 2.0e-14 * (1.0 + radial.abs()));
    assert!((ps - sp.conj()).norm() < 2.0e-14 * (1.0 + radial.abs()));
}

#[test]
fn length_and_convention_errors_identify_the_bad_input() {
    let channel_error =
        SphereField::from_real_channels([((0, 0), vec![1.0; 7]), ((1, 0), vec![1.0; 8])])
            .unwrap_err();
    assert_eq!(
        channel_error,
        SphereFieldError::ChannelLength {
            l: 1,
            m: 0,
            expected: 7,
            actual: 8,
        }
    );

    let convention_error = SphereField::new(
        HarmonicConvention::Real,
        [((0, 0), vec![Complex64::new(1.0, 0.2); 7])],
    )
    .unwrap_err();
    assert!(matches!(
        convention_error,
        SphereFieldError::ComplexSampleInRealConvention { index: 0, .. }
    ));

    assert_eq!(
        SphereOrbital::new(0, 0, vec![1.0; 8], Some(vec![0.0; 7])).unwrap_err(),
        SphereOrbitalError::SmallComponentLength {
            expected: 8,
            actual: 7,
        }
    );

    let mesh = mesh();
    let field = SphereField::from_real_channels([((0, 0), vec![1.0; mesh.len()])]).unwrap();
    let short = SphereOrbital::new(0, 0, vec![1.0; mesh.len() - 1], None).unwrap();
    let valid = orbital(0, 0, &mesh, 1.0);
    assert!(matches!(
        matrix_element(&mesh, &short, &field, &valid),
        Err(MatrixElementError::OrbitalMeshLength { actual, .. }) if actual == mesh.len() - 1
    ));

    // Ensure the public orbital is directly usable by libmuffintin-radial consumers.
    assert_eq!(valid.large_component().len(), mesh.len());
}

#[test]
fn spinor_low_order_oracles_resolve_p_and_q_angular_channels() {
    let mesh = mesh();
    let field = SphereField::new(
        HarmonicConvention::Complex,
        [((1, 0), vec![Complex64::new(1.0, 0.0); mesh.len()])],
    )
    .unwrap();
    let zero = vec![0.0; mesh.len()];
    let left_radial = radial_component(&mesh, 0.8);
    let right_radial = radial_component(&mesh, 1.3);
    let radial = component_overlap(&mesh, &left_radial, &right_radial);
    let expected_angular = (2.0_f64 / 3.0).sqrt() / (4.0 * PI).sqrt();

    // Omega_(-1,1/2) = Y_00 chi_up and the matching term of
    // Omega_(-2,1/2) is sqrt(2/3) Y_10 chi_up.
    let large_left =
        SpinorSphereOrbital::new(channel(-1, 1), left_radial.clone(), zero.clone()).unwrap();
    let large_right =
        SpinorSphereOrbital::new(channel(-2, 1), right_radial.clone(), zero.clone()).unwrap();
    let large = spinor_matrix_element(&mesh, &large_left, &field, &large_right).unwrap();
    assert!((large.re - expected_angular * radial).abs() < 2.0e-14 * (1.0 + radial.abs()));
    assert_eq!(large.im, 0.0);

    // For kappa=+1,+2 the small harmonics are the same Omega_-1/Omega_-2
    // pair.  A zero P component therefore exercises the independent QQ path.
    let small_left = SpinorSphereOrbital::new(channel(1, 1), zero.clone(), left_radial).unwrap();
    let small_right = SpinorSphereOrbital::new(channel(2, 1), zero, right_radial).unwrap();
    let small = spinor_matrix_element(&mesh, &small_left, &field, &small_right).unwrap();
    assert!((small.re - expected_angular * radial).abs() < 2.0e-14 * (1.0 + radial.abs()));
    assert_eq!(small.im, 0.0);
}

#[test]
fn spinor_pp_and_qq_integrals_are_assembled_separately() {
    let mesh = mesh();
    let field_channel = Lm::new(1, 0).unwrap();
    let field = SphereField::new(
        HarmonicConvention::Complex,
        [((1, 0), vec![Complex64::new(0.7, -0.2); mesh.len()])],
    )
    .unwrap();
    let left_channel = channel(-1, 1);
    let right_channel = channel(-2, 1);
    let left_p = radial_component(&mesh, 0.6);
    let left_q = radial_component(&mesh, 1.1);
    let right_p = radial_component(&mesh, 1.4);
    let right_q = radial_component(&mesh, 0.9);
    let pp = component_overlap(&mesh, &left_p, &right_p);
    let qq = component_overlap(&mesh, &left_q, &right_q);
    let left = SpinorSphereOrbital::new(left_channel, left_p, left_q).unwrap();
    let right = SpinorSphereOrbital::new(right_channel, right_p, right_q).unwrap();

    let large_angular = spinor_gaunt(left_channel, field_channel, right_channel);
    let small_angular = spinor_gaunt(
        left_channel.opposite_kappa(),
        field_channel,
        right_channel.opposite_kappa(),
    );
    let expected = Complex64::new(0.7, -0.2) * (large_angular * pp + small_angular * qq);
    let actual = spinor_matrix_element(&mesh, &left, &field, &right).unwrap();
    assert!((actual - expected).norm() < 3.0e-14 * (1.0 + expected.norm()));
}

#[test]
fn spinor_real_tesseral_blocks_are_hermitian() {
    let mesh = mesh();
    let field = SphereField::from_real_channels([((1, -1), vec![0.9; mesh.len()])]).unwrap();
    let left = SpinorSphereOrbital::new(
        channel(-1, 1),
        radial_component(&mesh, 0.7),
        radial_component(&mesh, 0.2),
    )
    .unwrap();
    let right = SpinorSphereOrbital::new(
        channel(-2, -1),
        radial_component(&mesh, 1.3),
        radial_component(&mesh, 0.5),
    )
    .unwrap();

    let left_right = spinor_matrix_element(&mesh, &left, &field, &right).unwrap();
    let right_left = spinor_matrix_element(&mesh, &right, &field, &left).unwrap();
    assert!(left_right.im.abs() > 1.0e-12);
    assert!((left_right - right_left.conj()).norm() < 3.0e-14 * (1.0 + left_right.norm()));
}

#[test]
fn spinor_length_errors_identify_component_and_operand() {
    assert_eq!(
        SpinorSphereOrbital::new(channel(-1, 1), vec![1.0; 8], vec![0.0; 7]).unwrap_err(),
        SphereOrbitalError::SmallComponentLength {
            expected: 8,
            actual: 7,
        }
    );

    let mesh = mesh();
    let field = SphereField::from_real_channels([((0, 0), vec![1.0; mesh.len()])]).unwrap();
    let short = SpinorSphereOrbital::new(
        channel(-1, 1),
        vec![1.0; mesh.len() - 1],
        vec![0.0; mesh.len() - 1],
    )
    .unwrap();
    let valid =
        SpinorSphereOrbital::new(channel(-1, 1), vec![1.0; mesh.len()], vec![0.0; mesh.len()])
            .unwrap();
    assert!(matches!(
        spinor_matrix_element(&mesh, &valid, &field, &short),
        Err(MatrixElementError::OrbitalMeshLength {
            operand: muffintin_sphere::Operand::Right,
            component: muffintin_sphere::Component::Large,
            actual,
            ..
        }) if actual == mesh.len() - 1
    ));

    let short_field =
        SphereField::from_real_channels([((0, 0), vec![1.0; mesh.len() - 1])]).unwrap();
    assert_eq!(
        spinor_matrix_element(&mesh, &valid, &short_field, &valid),
        Err(MatrixElementError::FieldMeshLength {
            expected: mesh.len(),
            actual: mesh.len() - 1,
        })
    );
}
