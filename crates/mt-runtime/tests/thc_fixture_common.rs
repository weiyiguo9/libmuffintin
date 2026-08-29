//! Shared THC parent-grid fixture for runtime tests.
//!
//! Include with `#[path = "thc_fixture_common.rs"]`. This file is not a
//! standalone test suite. `bounded_parent_grid` in `ml7_sm_fcc.rs` is a
//! different interstitial layout and stays there; it reuses [`on_shell`].

#![allow(dead_code)]

use muffintin::{ScalarProductInput, SpinorProductInput, ThcParentGrid, ThcPoint, ThcRegion};
use muffintin_auxiliary_ir::ProductPartition;
use muffintin_core::{Bohr, ExponentialMesh};
use muffintin_lapw::Provenance;

pub fn on_shell(origin: [Bohr; 3], radius: f64, direction: [f64; 3]) -> [Bohr; 3] {
    let norm = direction
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    [
        Bohr(origin[0].get() + radius * direction[0] / norm),
        Bohr(origin[1].get() + radius * direction[1] / norm),
        Bohr(origin[2].get() + radius * direction[2] / norm),
    ]
}

fn parent_grid_from_mesh(partition: ProductPartition, mesh: &ExponentialMesh) -> ThcParentGrid {
    let origin = partition.sites()[0].position;
    let mid = mesh.radii().len() / 2;
    let r_mid = mesh.radii()[mid].get();
    let r0 = mesh.radii()[0].get();
    ThcParentGrid::new(
        partition,
        Provenance::default(),
        vec![
            ThcPoint {
                coordinate: on_shell(origin, r0, [0.4, -0.3, 0.2]),
                weight: 0.35,
                region: ThcRegion::MuffinTin {
                    site: 0,
                    radial_index: 0,
                },
            },
            ThcPoint {
                coordinate: on_shell(origin, r_mid, [1.0, 0.0, 0.0]),
                weight: 0.0,
                region: ThcRegion::MuffinTin {
                    site: 0,
                    radial_index: mid,
                },
            },
            ThcPoint {
                coordinate: on_shell(origin, r_mid, [0.0, 1.0, 0.0]),
                weight: 0.45,
                region: ThcRegion::MuffinTin {
                    site: 0,
                    radial_index: mid,
                },
            },
            ThcPoint {
                coordinate: [Bohr(0.2), Bohr(0.2), Bohr(0.2)],
                weight: 0.8,
                region: ThcRegion::Interstitial,
            },
            ThcPoint {
                coordinate: [Bohr(5.0), Bohr(4.0), Bohr(4.0)],
                weight: 0.15,
                region: ThcRegion::Interstitial,
            },
            ThcPoint {
                coordinate: [Bohr(2.0), Bohr(6.5), Bohr(4.0)],
                weight: 0.25,
                region: ThcRegion::Interstitial,
            },
        ],
    )
    .unwrap()
}

pub fn scalar_parent_grid(input: &ScalarProductInput) -> ThcParentGrid {
    parent_grid_from_mesh(
        input.source.partition.clone(),
        &input.source.radials[0].mesh,
    )
}

pub fn spinor_parent_grid(input: &SpinorProductInput) -> ThcParentGrid {
    parent_grid_from_mesh(
        input.source.partition.clone(),
        &input.source.radials[0].mesh,
    )
}
