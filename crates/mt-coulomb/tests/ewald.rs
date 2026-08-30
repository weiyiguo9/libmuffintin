//! Direct Ewald-summed point-charge oracle versus assembled Weinert $V^q$.

mod common;

use muffintin_auxiliary_ir::{
    AuxiliaryPartition, AuxiliaryRepresentation, CompiledAuxiliaryBasis,
    InterpolationAuxiliaryPoint, InterpolationPointAuxiliary, InterpolationRegion,
};
use muffintin_envelope::Provenance;
use muffintin_core::{Bohr, InterstitialGeometry, InverseBohr, Sphere, VolumeBohr3};
use muffintin_coulomb::{
    CoulombError, CoulombRequest, EwaldScan, EwaldSummation, InterpolationProjection,
    assemble_point_charge_oracle, converged_ewald_point_kernel, ewald_point_kernel,
};
use std::f64::consts::PI;

/// Successive-residual gate for cutoff stability, not absolute `erfc` accuracy.
const EWALD_SUCCESSIVE_TOLERANCE: f64 = 1.0e-6;
/// Two muffin-tin unit charges, Weinert (1b) versus the independent Ewald kernel.
/// Recorded relative error $1.7\times 10^{-7}$ on cubic $a=8$, $q=2\pi\hat y/a$.
const WEINERT_VS_EWALD_TOLERANCE: f64 = 1.0e-6;

fn two_site_unit_charges(q: muffintin_auxiliary_ir::TransferQ) -> CompiledAuxiliaryBasis {
    let r1 = [Bohr(2.0), Bohr(2.0), Bohr(2.0)];
    let r2 = [Bohr(6.0), Bohr(2.0), Bohr(2.0)];
    let partition = AuxiliaryPartition::from_interstitial(
        InterstitialGeometry::new(
            VolumeBohr3(common::LATTICE.powi(3)),
            vec![
                Sphere {
                    center: r1,
                    radius: Bohr(common::RADIUS),
                },
                Sphere {
                    center: r2,
                    radius: Bohr(common::RADIUS),
                },
            ],
        )
        .unwrap(),
    );
    let mut points = vec![
        InterpolationAuxiliaryPoint {
            id: 0,
            coordinate: r1,
            weight: VolumeBohr3(1.0),
            region: InterpolationRegion::MuffinTin { site: 0 },
        },
        InterpolationAuxiliaryPoint {
            id: 1,
            coordinate: r2,
            weight: VolumeBohr3(1.0),
            region: InterpolationRegion::MuffinTin { site: 1 },
        },
    ];
    muffintin_auxiliary_ir::sort_interpolation_points(&mut points);
    let auxiliary = CompiledAuxiliaryBasis {
        partition,
        q,
        representation: AuxiliaryRepresentation::InterpolationPoints(InterpolationPointAuxiliary {
            points,
        }),
        provenance: Provenance {
            recipe: Some("two-site-unit-charges".to_owned()),
            reference: Some("Ewald oracle".to_owned()),
        },
    };
    auxiliary.validate().unwrap();
    auxiliary
}

#[test]
fn ewald_kernel_reports_successive_residual_and_rejects_tight_unmet_tolerance() {
    let q = common::transfer_q([0.0, 0.5, 0.0]);
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let r1 = [Bohr(2.0), Bohr(2.0), Bohr(2.0)];
    let r2 = [Bohr(6.0), Bohr(2.0), Bohr(2.0)];
    let conv = converged_ewald_point_kernel(
        request.cell(),
        request.reciprocal(),
        q,
        r1,
        r2,
        EwaldScan {
            tolerance: EWALD_SUCCESSIVE_TOLERANCE,
            max_steps: 8,
        },
    )
    .unwrap();
    assert!(conv.value.re.is_finite() && conv.value.im.is_finite());
    assert!(conv.successive_residual < EWALD_SUCCESSIVE_TOLERANCE);
    assert!(conv.steps >= 2);
    let larger = ewald_point_kernel(
        request.cell(),
        request.reciprocal(),
        q,
        r1,
        r2,
        EwaldSummation {
            eta: conv.eta,
            real_cutoff: Bohr(conv.real_cutoff.get() * 1.5),
            recip_cutoff: InverseBohr(conv.recip_cutoff.get() * 1.5),
        },
    )
    .unwrap();
    assert!((larger - conv.value).norm() < 10.0 * EWALD_SUCCESSIVE_TOLERANCE.max(1.0e-8));
    let other_eta = ewald_point_kernel(
        request.cell(),
        request.reciprocal(),
        q,
        r1,
        r2,
        EwaldSummation {
            eta: conv.eta * 1.4,
            real_cutoff: conv.real_cutoff,
            recip_cutoff: conv.recip_cutoff,
        },
    )
    .unwrap();
    assert!(
        (other_eta - conv.value).norm() < 5.0e-6,
        "second eta drifted by {}",
        (other_eta - conv.value).norm()
    );
    let too_tight = converged_ewald_point_kernel(
        request.cell(),
        request.reciprocal(),
        q,
        r1,
        r2,
        EwaldScan {
            tolerance: 1.0e-30,
            max_steps: 2,
        },
    );
    assert!(matches!(
        too_tight,
        Err(CoulombError::EwaldNotConverged { .. })
    ));
    assert!(matches!(
        converged_ewald_point_kernel(
            request.cell(),
            request.reciprocal(),
            q,
            r1,
            r2,
            EwaldScan {
                tolerance: EWALD_SUCCESSIVE_TOLERANCE,
                max_steps: 1,
            },
        ),
        Err(CoulombError::InvalidEwaldSteps(1))
    ));
}

#[test]
fn coincident_gamma_kernel_includes_the_regular_ewald_self_limit() {
    let request = CoulombRequest::cubic(1.0, 0).unwrap();
    let q = muffintin_auxiliary_ir::TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
    let origin = [Bohr(0.0); 3];
    let value = ewald_point_kernel(
        request.cell(),
        request.reciprocal(),
        q,
        origin,
        origin,
        EwaldSummation {
            eta: PI,
            real_cutoff: Bohr(8.0),
            recip_cutoff: InverseBohr(16.0 * PI),
        },
    )
    .unwrap();
    let simple_cubic_madelung = -2.837_297_479_480_619;
    assert!(value.im.abs() < 1.0e-12);
    assert!(
        (value.re - simple_cubic_madelung).abs() < 2.0e-6,
        "regular self value {} vs {simple_cubic_madelung}",
        value.re
    );
}

#[test]
fn two_site_monopoles_track_ewald_kernel() {
    let q = common::transfer_q([0.0, 0.5, 0.0]);
    let request = CoulombRequest::cubic(common::LATTICE, 2)
        .unwrap()
        .with_interpolation(InterpolationProjection::new(InverseBohr(1.2), 0).unwrap())
        .unwrap();
    let auxiliary = two_site_unit_charges(q);
    let operator = assemble_point_charge_oracle(&auxiliary, &request).unwrap();
    assert_eq!(
        operator.kind(),
        muffintin_coulomb::AuxiliaryKind::PointChargeOracle
    );
    let points = auxiliary.require_interpolation_points().unwrap();
    let assembled = operator.element(0, 1).unwrap();
    let conv = converged_ewald_point_kernel(
        request.cell(),
        request.reciprocal(),
        q,
        points[0].coordinate,
        points[1].coordinate,
        EwaldScan {
            tolerance: EWALD_SUCCESSIVE_TOLERANCE,
            max_steps: 8,
        },
    )
    .unwrap();
    assert!(conv.value.norm() > 0.01);
    let relative = (assembled - conv.value).norm() / conv.value.norm();
    assert!(
        relative < WEINERT_VS_EWALD_TOLERANCE,
        "Weinert {assembled} Ewald {} relative {relative} successive {}",
        conv.value,
        conv.successive_residual
    );
    let _ = PI;
}
