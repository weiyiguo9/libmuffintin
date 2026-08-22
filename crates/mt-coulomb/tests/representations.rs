//! Both published auxiliary representations go through the public assembler.

mod common;

use muffintin_core::{Bohr, ExponentialMesh, InverseBohr, VolumeBohr3};
use muffintin_coulomb::{
    AuxiliaryKind, CoulombError, CoulombRequest, InterpolationProjection,
    SampledAuxiliaryFunctions, SampledPointSupport, assemble_coulomb, assemble_point_charge_oracle,
    assemble_sampled_coulomb,
};
use muffintin_product::{AuxiliaryLayout, InterpolationRegion, TransferQ};
use muffintin_thc::toy::{
    mt_adaptive_grid, mt_bloch_orbitals, mt_kmesh, mt_orbital_norms, mt_partition,
    mt_reference_grid,
};
use muffintin_thc::{
    GridPath, HEADLINE_SEED, L2Engine, RankPolicy, SelectionRequest, SelectorStrategy, run_thc,
};
use num_complex::Complex64;
use std::f64::consts::PI;

#[test]
fn mixed_product_and_sampled_zeta_share_no_privileged_input_type() {
    let q = common::transfer_q([0.5, 0.0, 0.0]);
    let mpb_request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let (_source, mpb) = common::mixed_product_auxiliary(q);
    let mpb_operator = assemble_coulomb(&mpb, &mpb_request).unwrap();
    assert_eq!(mpb_operator.kind(), AuxiliaryKind::MixedProduct);
    assert_eq!(mpb_operator.dimension(), mpb.dimension());
    assert_eq!(mpb_operator.layout(), &mpb.layout());

    let interpolation = common::interpolation_auxiliary(q);
    let sampled_request = CoulombRequest::cubic(common::LATTICE, 2)
        .unwrap()
        .with_interpolation(InterpolationProjection::new(InverseBohr(1.6), 2).unwrap())
        .unwrap();
    assert!(matches!(
        assemble_coulomb(&interpolation, &sampled_request).unwrap_err(),
        CoulombError::MissingSampledFunctions
    ));
    let sampled = common::identity_zeta(&interpolation);
    let interpolation_operator =
        assemble_sampled_coulomb(&interpolation, &sampled_request, &sampled).unwrap();
    assert_eq!(
        interpolation_operator.kind(),
        AuxiliaryKind::InterpolationPoints
    );
    assert_eq!(
        interpolation_operator.dimension(),
        interpolation.dimension()
    );
    assert_ne!(mpb_operator.dimension(), interpolation_operator.dimension());
}

#[test]
fn identity_zeta_and_point_charge_routes_share_expansion_plumbing() {
    let q = common::transfer_q([0.5, 0.0, 0.0]);
    let request = CoulombRequest::cubic(common::LATTICE, 2)
        .unwrap()
        .with_interpolation(InterpolationProjection::new(InverseBohr(1.6), 2).unwrap())
        .unwrap();
    let auxiliary = common::interpolation_auxiliary(q);
    let sampled = common::identity_zeta(&auxiliary);
    let zeta = assemble_sampled_coulomb(&auxiliary, &request, &sampled).unwrap();
    let oracle = assemble_point_charge_oracle(&auxiliary, &request).unwrap();
    assert_eq!(zeta.dimension(), oracle.dimension());
    let n = zeta.dimension();
    let mut worst: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            worst = worst.max((zeta.element(i, j).unwrap() - oracle.element(i, j).unwrap()).norm());
        }
    }
    assert!(
        worst < 1.0e-10,
        "the two explicit plumbing routes disagree, worst {worst}"
    );
}

#[test]
fn sampled_mt_projection_uses_declared_shells_without_coordinate_snapping() {
    let q = common::transfer_q([0.5, 0.0, 0.0]);
    let auxiliary = common::interpolation_auxiliary(q);
    let request = CoulombRequest::cubic(common::LATTICE, 2)
        .unwrap()
        .with_interpolation(InterpolationProjection::new(InverseBohr(1.6), 0).unwrap())
        .unwrap();
    let first: f64 = 2.0e-2;
    let count = 13;
    let increment = (common::RADIUS / first).ln() / (count - 1) as f64;
    let radial_mesh = ExponentialMesh::new(Bohr(first), increment, count).unwrap();
    let make_sampled = |radial_index: usize, radius: f64| {
        let mut zeta = vec![Complex64::default(); auxiliary.dimension()];
        zeta[0] = Complex64::new(1.0, 0.0);
        SampledAuxiliaryFunctions::new(
            auxiliary.layout(),
            vec![radial_mesh.clone()],
            vec![[
                Bohr(common::POSITION[0].get() + radius),
                common::POSITION[1],
                common::POSITION[2],
            ]],
            vec![VolumeBohr3(0.05)],
            vec![SampledPointSupport::MuffinTin {
                site: 0,
                radial_index,
            }],
            zeta,
        )
        .unwrap()
    };
    let left_index = 5;
    let right_index = 6;
    let left = make_sampled(left_index, radial_mesh.radii()[left_index].get());
    let right = make_sampled(right_index, radial_mesh.radii()[right_index].get());
    let left_value = assemble_sampled_coulomb(&auxiliary, &request, &left)
        .unwrap()
        .element(0, 0)
        .unwrap();
    let right_value = assemble_sampled_coulomb(&auxiliary, &request, &right)
        .unwrap()
        .element(0, 0)
        .unwrap();
    assert!(
        (left_value - right_value).norm() > 1.0e-6,
        "different declared shells must not collapse to one radial node"
    );

    let mismatched = make_sampled(left_index, radial_mesh.radii()[right_index].get());
    assert!(matches!(
        assemble_sampled_coulomb(&auxiliary, &request, &mismatched),
        Err(CoulombError::SampledCoordinateShellMismatch { .. })
    ));
    let centre = make_sampled(left_index, 0.0);
    assert!(matches!(
        assemble_sampled_coulomb(&auxiliary, &request, &centre),
        Err(CoulombError::SampledCoordinateShellMismatch { .. })
    ));
}

#[test]
fn thc_zeta_assembles_and_rejects_layout_mismatch() {
    assert_eq!(muffintin_thc::DEFAULT_SELECTOR, SelectorStrategy::AllQL2);
    let mesh = mt_kmesh();
    let partition = mt_partition();
    let nrad = 73;
    let nang = 8;
    let grid = mt_adaptive_grid(nrad, nang, 4);
    let reference = mt_reference_grid();
    let norms = mt_orbital_norms(&reference);
    let orbitals = mt_bloch_orbitals(&grid, &norms, &mesh).unwrap();
    let result = run_thc(
        &orbitals,
        &grid,
        &mesh,
        &partition,
        &SelectionRequest {
            strategy: SelectorStrategy::AllQL2,
            rank: RankPolicy::Exact { n_mu: 6 },
            seed: HEADLINE_SEED,
            pool_factor: 2,
            engine: L2Engine::StructuredSketch { rows: 32 },
            grid_path: GridPath::Adaptive {
                nrad,
                nang,
                ninter: 4,
            },
        },
        None,
        Some(0),
        None,
    )
    .unwrap();
    let iq = 1;
    let auxiliary = &result.auxiliaries[iq];
    let fit = &result.fits[iq];
    assert!(auxiliary.mixed_product().is_none());
    assert_eq!(fit.n_mu, auxiliary.dimension());
    assert_eq!(fit.n_points, grid.len());
    assert_eq!(fit.q, auxiliary.q);
    let first_radius: f64 = 2.0e-3;
    let increment = (muffintin_thc::toy::MT_RADIUS / first_radius).ln() / (nrad - 1) as f64;
    let radial_mesh = ExponentialMesh::new(Bohr(first_radius), increment, nrad).unwrap();
    let mut mt_counts = vec![0usize; partition.site_count()];
    let sampled_supports = grid
        .regions
        .iter()
        .map(|region| match *region {
            InterpolationRegion::MuffinTin { site } => {
                let radial_index = mt_counts[site] / nang;
                mt_counts[site] += 1;
                SampledPointSupport::MuffinTin { site, radial_index }
            }
            InterpolationRegion::Interstitial => SampledPointSupport::Interstitial,
            InterpolationRegion::Uniform => SampledPointSupport::Uniform,
        })
        .collect();
    let sampled = SampledAuxiliaryFunctions::new(
        auxiliary.layout(),
        vec![radial_mesh],
        grid.points
            .iter()
            .map(|point| [Bohr(point[0]), Bohr(point[1]), Bohr(point[2])])
            .collect(),
        grid.weights
            .iter()
            .map(|weight| VolumeBohr3(*weight))
            .collect(),
        sampled_supports,
        fit.zeta.clone(),
    )
    .unwrap();
    let request = CoulombRequest::cubic(muffintin_thc::toy::MT_LATTICE, 2)
        .unwrap()
        .with_interpolation(InterpolationProjection::new(InverseBohr(1.8), 1).unwrap())
        .unwrap();
    let operator = assemble_sampled_coulomb(auxiliary, &request, &sampled).unwrap();
    assert_eq!(operator.kind(), AuxiliaryKind::InterpolationPoints);
    assert_eq!(operator.q(), auxiliary.q);
    assert_eq!(operator.layout(), &auxiliary.layout());
    let min_eig = {
        let n = operator.dimension();
        let mat = faer::Mat::from_fn(n, n, |row, column| operator.matrix()[row * n + column]);
        let eigen = mat
            .self_adjoint_eigen(faer::Side::Lower)
            .expect("Hermitian EVD");
        (0..n)
            .map(|index| eigen.S()[index].re)
            .fold(f64::INFINITY, f64::min)
    };
    assert!(
        min_eig > -1.0e-4,
        "THC zeta Coulomb min eigenvalue {min_eig}"
    );
    let vertex = &result.vertices[iq][0];
    let action = operator.quadratic_form(vertex, vertex).unwrap();
    assert!(action.re.is_finite() && action.im.is_finite());

    let mut regions = auxiliary.regions();
    regions.reverse();
    let bad = SampledAuxiliaryFunctions::new(
        AuxiliaryLayout::from_regions(auxiliary.q, regions),
        sampled.site_meshes().to_vec(),
        sampled.points().to_vec(),
        sampled.weights().to_vec(),
        sampled.supports().to_vec(),
        sampled.zeta().to_vec(),
    )
    .unwrap();
    assert!(matches!(
        assemble_sampled_coulomb(auxiliary, &request, &bad).unwrap_err(),
        CoulombError::SampledLayoutMismatch
    ));
    let other_q =
        TransferQ::from_cartesian([InverseBohr(0.1), InverseBohr(0.0), InverseBohr(0.0)]).unwrap();
    let bad_q = SampledAuxiliaryFunctions::new(
        AuxiliaryLayout::from_regions(other_q, auxiliary.regions()),
        sampled.site_meshes().to_vec(),
        sampled.points().to_vec(),
        sampled.weights().to_vec(),
        sampled.supports().to_vec(),
        sampled.zeta().to_vec(),
    )
    .unwrap();
    assert!(matches!(
        assemble_sampled_coulomb(auxiliary, &request, &bad_q).unwrap_err(),
        CoulombError::SampledLayoutMismatch
    ));
    let short = auxiliary.regions()[..1].to_vec();
    let n_mu = 1;
    let n_grid = sampled.n_grid();
    let zeta_short = vec![Complex64::new(1.0, 0.0); n_grid * n_mu];
    let bad_dim = SampledAuxiliaryFunctions::new(
        AuxiliaryLayout::from_regions(auxiliary.q, short),
        sampled.site_meshes().to_vec(),
        sampled.points().to_vec(),
        sampled.weights().to_vec(),
        sampled.supports().to_vec(),
        zeta_short,
    )
    .unwrap();
    assert!(matches!(
        assemble_sampled_coulomb(auxiliary, &request, &bad_dim).unwrap_err(),
        CoulombError::SampledZetaDimension { n_mu: 1, expected }
            if expected == auxiliary.dimension()
    ));
    let _ = PI;
}
