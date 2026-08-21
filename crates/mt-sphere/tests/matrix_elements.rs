use libmuffintin_core::{Bohr, ExponentialMesh};
use libmuffintin_radial::{RadialComponents, RadialIntegralKernel, radial_integral};
use libmuffintin_sphere::{
    HarmonicConvention, MatrixElementError, SphereField, SphereFieldError, SphereOrbital,
    SphereOrbitalError, matrix_element,
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
