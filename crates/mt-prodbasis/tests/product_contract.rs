//! Interpolation-point auxiliaries stay on the product-space IR.

use muffintin_prodbasis::{AuxiliaryRegion, InterpolationRegion, OrbitalPair};
use muffintin_prodbasis::thc::toy::{
    mt_adaptive_grid, mt_bloch_orbitals, mt_kmesh, mt_orbital_norms, mt_partition,
    mt_reference_grid,
};
use muffintin_prodbasis::thc::{
    GridPath, HEADLINE_SEED, L2Engine, RankPolicy, SelectionRequest, SelectorStrategy, run_thc,
};

#[test]
fn interpolation_auxiliary_and_bloch_vertices_match_product_regions() {
    let mesh = mt_kmesh();
    let partition = mt_partition();
    let grid = mt_adaptive_grid(8, 12, 6);
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
            rank: RankPolicy::Exact { n_mu: 10 },
            seed: HEADLINE_SEED,
            pool_factor: 2,
            engine: L2Engine::StructuredSketch { rows: 48 },
            grid_path: GridPath::Adaptive {
                nrad: 8,
                nang: 12,
                ninter: 6,
            },
        },
        None,
        Some(0),
        None,
    )
    .unwrap();
    let auxiliary = &result.auxiliaries[0];
    assert!(auxiliary.mixed_product().is_none());
    assert_eq!(auxiliary.dimension(), result.selection.points.len());
    assert_eq!(auxiliary.partition, partition);
    let regions = auxiliary.regions();
    assert_eq!(regions.len(), auxiliary.dimension());
    let mt = regions
        .iter()
        .filter(|region| {
            matches!(
                region,
                AuxiliaryRegion::InterpolationPoint {
                    region: InterpolationRegion::MuffinTin { .. },
                    ..
                }
            )
        })
        .count();
    assert_eq!(mt, auxiliary.mt_dimension());
    assert_eq!(
        auxiliary.interstitial_dimension(),
        auxiliary.dimension() - mt
    );
    let vertex = &result.vertices[0][0];
    assert_eq!(vertex.mt().len(), auxiliary.mt_dimension());
    assert_eq!(
        vertex.interstitial().len(),
        auxiliary.interstitial_dimension()
    );
    assert!(matches!(vertex.pair(), OrbitalPair::Bloch { .. }));
    assert_eq!(vertex.q(), auxiliary.q);
}
