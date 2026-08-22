use muffintin_basis::{
    ApwSiteGeometry, BasisLayout, CompiledBasis, LocalOrbitalLayout, PlaneWaveAugmentation,
    Provenance, SpinorBasisLayout, SpinorCompiledBasis, SpinorPlaneWaveAugmentation,
    SpinorSiteLayout,
};
use muffintin_core::{Bohr, GVector, InverseBohr, Kappa, RelativisticChannel, TwiceMu};
use muffintin_envelope::PlaneWave;
use muffintin_operators::{
    CompiledSiteProjection, SiteOperatorBlocks, add_site_contributions,
    project_eigenvectors_to_site, project_spinor_eigenvectors_to_site,
};
use muffintin_tensor::{Axis, DenseEigenvectors, DenseHermitianMatrix};
use num_complex::Complex64;

fn wave(index: [i32; 3]) -> PlaneWave {
    let cartesian = index.map(|value| InverseBohr(f64::from(value)));
    let norm = InverseBohr(
        cartesian
            .iter()
            .map(|value| value.get().powi(2))
            .sum::<f64>()
            .sqrt(),
    );
    PlaneWave::new(
        [InverseBohr(0.0); 3],
        GVector {
            index,
            cartesian,
            norm,
        },
    )
}

fn geometry() -> ApwSiteGeometry {
    ApwSiteGeometry {
        position: [Bohr(0.0); 3],
        radius: Bohr(2.0),
    }
}

fn identity_site(dimension: usize) -> DenseHermitianMatrix {
    DenseHermitianMatrix::from_upper_triangle(dimension, Axis::SiteCoordinate, |row, column| {
        Complex64::new(if row == column { 1.0 } else { 0.0 }, 0.0)
    })
    .unwrap()
}

fn scalar_fixture() -> CompiledBasis {
    let phase = Complex64::new(0.6, 0.8);
    CompiledBasis {
        layout: BasisLayout::new(2, vec![LocalOrbitalLayout::new(vec![1])]),
        plane_waves: vec![wave([0, 0, 0]), wave([1, 0, 0])],
        site_augmentations: vec![vec![
            PlaneWaveAugmentation {
                coefficients: vec![[phase * 1.0, phase * 2.0], [phase * 3.0, phase * 4.0]],
            },
            PlaneWaveAugmentation {
                coefficients: vec![[phase * 5.0, phase * 6.0], [phase * 7.0, phase * 8.0]],
            },
        ]],
        site_geometry: vec![geometry()],
        provenance: Provenance::default(),
    }
}

#[test]
fn scalar_projection_has_one_apw_phase_and_lo_identity_rows() {
    let compiled = scalar_fixture();
    let phase = Complex64::new(0.6, 0.8);
    let projection = CompiledSiteProjection::scalar(&compiled, 0).unwrap();
    assert_eq!(projection.global_indices(), &[0, 1, 2]);
    assert_eq!(projection.coordinate_count(), 5);
    assert_eq!(projection.matrix().at(&[0, 0]), phase);
    assert_eq!(projection.matrix().at(&[1, 0]), phase * 2.0);
    assert_eq!(projection.matrix().at(&[2, 1]), phase * 7.0);
    assert_eq!(projection.matrix().at(&[3, 1]), phase * 8.0);
    assert_eq!(projection.matrix().at(&[4, 2]), Complex64::new(1.0, 0.0));
    assert_eq!(projection.matrix().at(&[4, 0]), Complex64::new(0.0, 0.0));

    let basis_unit = DenseEigenvectors::from_host_column_major(
        3,
        1,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let coefficients = project_eigenvectors_to_site(&compiled, 0, &basis_unit).unwrap();
    assert_eq!(coefficients.at(0, 0), phase);
    assert_eq!(coefficients.at(1, 0), phase * 2.0);
    assert_eq!(coefficients.at(4, 0), Complex64::new(0.0, 0.0));
}

#[test]
fn operator_congruence_and_orbital_projection_use_the_same_p() {
    let compiled = scalar_fixture();
    let eigenvectors = DenseEigenvectors::from_host_column_major(
        3,
        1,
        vec![
            Complex64::new(0.3, -0.1),
            Complex64::new(-0.2, 0.4),
            Complex64::new(0.7, 0.2),
        ],
    )
    .unwrap();
    let site_coefficients = project_eigenvectors_to_site(&compiled, 0, &eigenvectors).unwrap();
    let local_norm = (0..site_coefficients.coordinate_count())
        .map(|coordinate| site_coefficients.at(coordinate, 0).norm_sqr())
        .sum::<f64>();

    let block = identity_site(5);
    let site = SiteOperatorBlocks {
        overlap: block.clone(),
        hamiltonian: block,
    };
    let mut overlap = vec![Complex64::new(0.0, 0.0); 9];
    let mut hamiltonian = overlap.clone();
    let operators =
        add_site_contributions(&mut overlap, &mut hamiltonian, 3, &compiled, &[site]).unwrap();
    let mut global = Complex64::new(0.0, 0.0);
    for row in 0..3 {
        for column in 0..3 {
            global += eigenvectors.at(row, 0).conj()
                * operators.overlap.at(row, column)
                * eigenvectors.at(column, 0);
        }
    }
    assert!((global.re - local_norm).abs() < 2.0e-13 * (1.0 + local_norm));
    assert!(global.im.abs() < 2.0e-13 * (1.0 + local_norm));
}

#[test]
fn spinor_projection_preserves_channel_radial_then_lo_coordinate_order() {
    let kappa = Kappa::new(-1).unwrap();
    let channels = vec![
        RelativisticChannel::new(kappa, TwiceMu::new(-1).unwrap()).unwrap(),
        RelativisticChannel::new(kappa, TwiceMu::new(1).unwrap()).unwrap(),
    ];
    let site_layout = SpinorSiteLayout::new(vec![(kappa, 1)]).unwrap();
    let compiled = SpinorCompiledBasis {
        layout: SpinorBasisLayout::new(1, vec![site_layout]),
        plane_waves: vec![wave([0, 0, 0])],
        site_augmentations: vec![vec![SpinorPlaneWaveAugmentation {
            channels: channels.clone(),
            coefficients: [
                vec![
                    [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
                    [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
                ],
                vec![
                    [Complex64::new(10.0, 0.0), Complex64::new(20.0, 0.0)],
                    [Complex64::new(30.0, 0.0), Complex64::new(40.0, 0.0)],
                ],
            ],
        }]],
        site_geometry: vec![geometry()],
        provenance: Provenance::default(),
    };
    let eigenvectors = DenseEigenvectors::from_host_column_major(
        4,
        1,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let projected =
        project_spinor_eigenvectors_to_site(&compiled, 0, &channels, &eigenvectors).unwrap();
    assert_eq!(projected.coordinate_count(), 6);
    assert_eq!(projected.at(0, 0), Complex64::new(21.0, 0.0));
    assert_eq!(projected.at(1, 0), Complex64::new(42.0, 0.0));
    assert_eq!(projected.at(2, 0), Complex64::new(63.0, 0.0));
    assert_eq!(projected.at(3, 0), Complex64::new(84.0, 0.0));
    assert_eq!(projected.at(4, 0), Complex64::new(3.0, 0.0));
    assert_eq!(projected.at(5, 0), Complex64::new(4.0, 0.0));
}
