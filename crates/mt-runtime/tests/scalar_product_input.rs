//! Public scalar product-input tests on a frozen checkpoint solve.

use std::collections::{BTreeMap, BTreeSet};

use muffintin::{CheckpointPhysics, CheckpointPhysicsError, SCALAR_RADIAL_U, SCALAR_RADIAL_UDOT};
use muffintin_core::{Bohr, Hartree, InverseBohr, ReciprocalLattice};
use muffintin_dft::{
    FirstVariationWindow, LinearizationEnergyGenerator, NoncollinearXcRoute, ScfBasis,
    ScfChannelIdentity, ScfChannelProvenance, ScfChannelRecipe, ScfChannelTreatment, ScfConfig,
    ScfConvergence, ScfCoreSite, ScfExchangeCorrelation, ScfKMesh, ScfMixing, ScfOccupations,
    ScfRelativity, XcFunctional,
};
use muffintin_io::{
    AngularBasis, BasisHints, CheckpointMeta, CheckpointV1, CheckpointV2, Complex64V1,
    EnergyParameterV1, EnergyUnit, ExponentialMeshSpec, FourierCoefficientV1, FourierNormalization,
    FourierPhase, GeometryV1, InterstitialV1, InverseLengthUnit, LatticeV1, LengthUnit,
    LinearizationV1, PotentialChannelV1, PotentialConventionV1, PotentialRadialQuantityV1,
    RadialEquationTag, SiteSpinV1, SiteV1, SphericalChannelConvention, SpinTag,
};
use num_complex::Complex64;

fn hydrogen_checkpoint() -> CheckpointV2 {
    let point_count = 61;
    let first: f64 = 1.0e-4;
    let radius: f64 = 1.0;
    let increment = (radius / first).ln() / (point_count - 1) as f64;
    let radii = (0..point_count)
        .map(|index| first * (index as f64 * increment).exp())
        .collect::<Vec<_>>();
    CheckpointV1::new(
        CheckpointMeta {
            title: "scalar product-input hydrogen smoke".to_owned(),
            producer: "mt-runtime test".to_owned(),
            producer_version: None,
            energy_zero: "zero interstitial Fourier mean".to_owned(),
            potential_convention: PotentialConventionV1 {
                angular_basis: AngularBasis::ComplexCondonShortley,
                radial_quantity: PotentialRadialQuantityV1::Potential,
                spherical_channel: SphericalChannelConvention::PhysicalValue,
            },
            annotations: BTreeMap::new(),
        },
        GeometryV1 {
            lattice: LatticeV1 {
                unit: LengthUnit::Bohr,
                vectors: [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            },
            sites: vec![SiteV1 {
                id: "H-1".to_owned(),
                atomic_number: 1,
                fractional_position: [1.25, -0.5, 0.5],
                muffin_tin_radius_unit: LengthUnit::Bohr,
                muffin_tin_radius: radius,
                spins: vec![SiteSpinV1 {
                    spin: SpinTag::Scalar,
                    mesh: ExponentialMeshSpec {
                        radius_unit: LengthUnit::Bohr,
                        first,
                        log_increment: increment,
                        point_count,
                        last: first * ((point_count - 1) as f64 * increment).exp(),
                        consistency_tolerance: 1.0e-12,
                    },
                    radial_equation: RadialEquationTag::ScalarKoellingHarmon,
                    potential_unit: EnergyUnit::Hartree,
                    potential_channels: vec![PotentialChannelV1 {
                        l: 0,
                        m: 0,
                        real: radii.iter().map(|radius| -1.0 / radius).collect(),
                        imaginary: Vec::new(),
                    }],
                    linearization: LinearizationV1 {
                        energy_unit: EnergyUnit::Hartree,
                        linearization_energies: vec![
                            EnergyParameterV1 { l: 0, energy: -0.3 },
                            EnergyParameterV1 {
                                l: 1,
                                energy: -0.15,
                            },
                        ],
                        local_orbital_energies: Vec::new(),
                    },
                }],
            }],
        },
        InterstitialV1 {
            coefficient_unit: EnergyUnit::Hartree,
            coefficients: vec![FourierCoefficientV1 {
                g: [0; 3],
                value: Complex64V1 {
                    real: 0.0,
                    imaginary: 0.0,
                },
            }],
            basis_hints: BasisHints {
                reciprocal_length_unit: InverseLengthUnit::BohrInverse,
                plane_wave_cutoff: Some(0.5),
                coefficient_cutoff: Some(1.0),
                normalization: FourierNormalization::CellNormalized,
                phase: FourierPhase::NegativeExponent,
            },
        },
    )
    .normalize_v2()
    .unwrap()
}

fn scalar_config(divisions: [usize; 3], cutoff: f64) -> ScfConfig {
    ScfConfig {
        electron_count: 1.0,
        k_mesh: ScfKMesh {
            divisions,
            shift: [0.0; 3],
            reduction: muffintin_dft::ScfKReduction::Full,
        },
        basis: ScfBasis {
            plane_wave_cutoff: InverseBohr(cutoff),
            l_max: 1,
            channels: vec![
                ScfChannelRecipe {
                    site: "H-1".to_owned(),
                    identity: ScfChannelIdentity::ScalarL { n: 1, l: 0 },
                    treatment: ScfChannelTreatment::Valence,
                    derivative_order: 0,
                    generator: LinearizationEnergyGenerator::FrozenCheckpoint,
                    seed: None,
                    provenance: ScfChannelProvenance::BuiltIn,
                },
                ScfChannelRecipe {
                    site: "H-1".to_owned(),
                    identity: ScfChannelIdentity::ScalarL { n: 2, l: 1 },
                    treatment: ScfChannelTreatment::Valence,
                    derivative_order: 0,
                    generator: LinearizationEnergyGenerator::FrozenCheckpoint,
                    seed: None,
                    provenance: ScfChannelProvenance::BuiltIn,
                },
            ],
            resolved_channels: Vec::new(),
        },
        occupations: ScfOccupations::FermiDirac {
            temperature: Hartree(0.02),
        },
        exchange_correlation: ScfExchangeCorrelation {
            functional: XcFunctional::LdaPw92,
            noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
        },
        mixing: ScfMixing::Linear { alpha: 1.0 },
        relativity: ScfRelativity::Scalar,
        convergence: ScfConvergence {
            energy_tolerance: Hartree(1.0e100),
            density_tolerance: 1.0e100,
            max_iterations: 2,
        },
        core_sites: vec![ScfCoreSite {
            id: "H-1".to_owned(),
            states: Vec::new(),
        }],
    }
}

fn cartesian_from_fractional(
    reciprocal: ReciprocalLattice,
    fractional: [f64; 3],
) -> [InverseBohr; 3] {
    std::array::from_fn(|axis| {
        InverseBohr(
            fractional
                .iter()
                .zip(reciprocal.basis())
                .map(|(&coefficient, vector)| coefficient * vector[axis].get())
                .sum(),
        )
    })
}

fn pair_phase(reciprocal: ReciprocalLattice, wrap: [i32; 3], site: [Bohr; 3]) -> Complex64 {
    let g = reciprocal.cartesian(wrap);
    let argument = g
        .iter()
        .zip(site)
        .map(|(component, coord)| component.get() * coord.get())
        .sum();
    Complex64::from_polar(1.0, argument)
}

fn expected_relative_g(input: &muffintin::ScalarProductInput) -> BTreeSet<[i32; 3]> {
    let mut indices = BTreeSet::new();
    for channel in &input.orbitals.channels {
        for mapped in &input.k_minus_q {
            let g_k = &channel.bases[mapped.k_index].plane_waves;
            let g_kmq = &channel.bases[mapped.kq_index].plane_waves;
            let wrap = mapped.umklapp.index;
            for right in g_k {
                for left in g_kmq {
                    indices.insert([
                        right.g.index[0] - left.g.index[0] + wrap[0],
                        right.g.index[1] - left.g.index[1] + wrap[1],
                        right.g.index[2] - left.g.index[2] + wrap[2],
                    ]);
                }
            }
        }
    }
    indices
}

fn support_indices(input: &muffintin::ScalarProductInput) -> Vec<[i32; 3]> {
    input
        .source
        .interstitial_pair_support
        .components
        .iter()
        .map(|component| component.g_relative.index)
        .collect()
}

#[test]
fn scalar_product_input_rejects_full_spinor_relativity() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let mut config = scalar_config([1, 1, 1], 0.5);
    config.relativity = ScfRelativity::SpinorFirstVariation;
    let error = physics.scalar_product_input(&config, [0.0; 3]).unwrap_err();
    assert!(matches!(
        error,
        CheckpointPhysicsError::ScalarProductRejectsSpinorFirstVariation
    ));
}

#[test]
fn q0_second_variation_product_input_retains_pauli_components() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let mut config = scalar_config([1, 1, 1], 0.5);
    config.relativity = ScfRelativity::SocSecondVariation {
        window: FirstVariationWindow::new(0, 1).unwrap(),
    };
    let input = physics.scalar_product_input(&config, [0.0; 3]).unwrap();

    assert_eq!(input.orbitals.relativity, config.relativity);
    assert_eq!(input.orbitals.channels.len(), 2);
    assert_eq!(input.orbitals.band_window.count, 2);
    assert!(
        input
            .orbitals
            .channels
            .iter()
            .all(|channel| channel.eigenvectors[0].columns() == 2)
    );
    assert_eq!(
        input.source.provenance.reference.as_deref(),
        Some("checkpoint-dft-frozen-second-variation-product-input")
    );
}

#[test]
fn q0_frozen_scalar_product_input_emits_neutral_source_and_orbitals() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let config = scalar_config([1, 1, 1], 0.5);
    let input = physics.scalar_product_input(&config, [0.0; 3]).unwrap();

    assert_eq!(input.reciprocal, *physics.reciprocal());
    assert_eq!(input.source.q.umklapp.index, [0; 3]);
    assert!(
        input
            .source
            .q
            .cartesian
            .iter()
            .all(|component| component.get().abs() <= 64.0 * f64::EPSILON)
    );
    assert_eq!(input.source.partition.site_count(), 1);
    assert_eq!(
        input.source.partition.interstitial().spheres()[0].center,
        [Bohr(2.0), Bohr(4.0), Bohr(4.0)]
    );
    assert_eq!(input.source.radials.len(), 1);
    assert!(input.source.radials[0].cores.is_empty());
    assert!(
        input.source.radials[0]
            .valence
            .iter()
            .any(|radial| radial.spin == 0 && radial.n == SCALAR_RADIAL_U)
    );
    assert!(
        input.source.radials[0]
            .valence
            .iter()
            .any(|radial| radial.spin == 0 && radial.n == SCALAR_RADIAL_UDOT)
    );
    assert!(
        input.source.radials[0]
            .valence
            .iter()
            .any(|radial| radial.spin == 1 && radial.n == SCALAR_RADIAL_U)
    );
    assert_eq!(
        input.source.provenance.reference.as_deref(),
        Some("checkpoint-dft-frozen-scalar-product-input")
    );

    assert_eq!(input.k_minus_q.len(), 1);
    assert_eq!(input.k_minus_q[0].k_index, 0);
    assert_eq!(input.k_minus_q[0].kq_index, 0);
    assert_eq!(input.k_minus_q[0].umklapp.index, [0; 3]);
    assert_eq!(input.source.interstitial_pair_support.q, input.source.q);
    assert_eq!(
        support_indices(&input).into_iter().collect::<BTreeSet<_>>(),
        expected_relative_g(&input)
    );

    assert_eq!(input.orbitals.k_fractional, vec![[0.0; 3]]);
    assert_eq!(input.orbitals.channels.len(), 2);
    assert_eq!(input.orbitals.channels[0].spin, 0);
    assert_eq!(input.orbitals.channels[1].spin, 1);
    assert_eq!(input.orbitals.band_window.start, 0);
    assert_eq!(input.pair_columns.n_k, 1);
    assert!(input.pair_columns.core_orbital.is_none());
    let n_orb = input.pair_columns.n_orb;
    assert_eq!(input.orbitals.band_window.count, n_orb);
    assert!(n_orb > 0);
    assert_eq!(input.pair_columns.encode(0, 1, 0), n_orb);
    for channel in &input.orbitals.channels {
        assert_eq!(channel.eigenvectors.len(), 1);
        assert_eq!(channel.eigenvectors[0].columns(), n_orb);
        assert_eq!(channel.energies[0].len(), n_orb);
        assert_eq!(channel.bases.len(), 1);
        assert_eq!(
            channel.eigenvectors[0].rows(),
            channel.bases[0].layout.dimension()
        );
        assert_eq!(channel.available_bands.len(), 1);
        assert!(channel.available_bands[0] >= n_orb);
    }
}

#[test]
fn q0_multi_plane_wave_support_and_row_basis_are_self_contained() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let config = scalar_config([1, 1, 1], 1.0);
    let input = physics.scalar_product_input(&config, [0.0; 3]).unwrap();
    let labels = support_indices(&input);
    assert_eq!(
        labels.iter().copied().collect::<BTreeSet<_>>(),
        expected_relative_g(&input)
    );
    assert!(
        labels.iter().any(|index| *index != [0, 0, 0]),
        "q=0 multi-G support must not collapse to {{0}}; got {labels:?}"
    );

    let channel = &input.orbitals.channels[0];
    let compiled = &channel.bases[0];
    let row = compiled
        .plane_waves
        .iter()
        .position(|wave| wave.g.index != [0, 0, 0])
        .expect("cutoff 1.0 must retain a nonzero plane-wave G");
    assert_eq!(channel.eigenvectors[0].rows(), compiled.layout.dimension());
    assert!(row < compiled.layout.plane_wave_count());
    assert_eq!(
        compiled.site_augmentations[0].len(),
        compiled.plane_waves.len()
    );
    let g = compiled.plane_waves[row].g.index;
    assert_ne!(g, [0, 0, 0]);
    let aug = &compiled.site_augmentations[0][row];
    assert!(
        !aug.coefficients.is_empty(),
        "APW row {row} G={g:?} must carry (u, udot) matching coefficients"
    );
    let [a, b] = aug.coefficients[0];
    assert!(a.norm() + b.norm() > 0.0);
    assert!(
        input.source.radials[0]
            .valence
            .iter()
            .any(|radial| radial.n == SCALAR_RADIAL_U && radial.l == 0 && radial.spin == 0)
    );
    assert!(
        input.source.radials[0]
            .valence
            .iter()
            .any(|radial| radial.n == SCALAR_RADIAL_UDOT && radial.l == 0 && radial.spin == 0)
    );
    if let Some(lo_range) = compiled.layout.site_local_orbital_range(0)
        && !lo_range.is_empty()
    {
        let lo_row = lo_range.start;
        assert!(lo_row >= compiled.layout.plane_wave_count());
        assert!(compiled.layout.local_orbital_index(0, 0, 0, 0).is_some());
    }
}

#[test]
fn finite_q_wrap_uses_reciprocal_lattice_umklapp_and_k_minus_q_phase() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let config = scalar_config([2, 1, 1], 0.5);
    let q_fractional = [1.5, 0.0, 0.0];
    let input = physics.scalar_product_input(&config, q_fractional).unwrap();

    let reciprocal = *physics.reciprocal();
    assert_eq!(input.reciprocal, reciprocal);
    let q_input = cartesian_from_fractional(reciprocal, q_fractional);
    let transfer_g = reciprocal.cartesian([1, 0, 0]);
    assert_eq!(input.source.q.umklapp.index, [1, 0, 0]);
    assert_eq!(input.source.q.umklapp.cartesian, transfer_g);
    for axis in 0..3 {
        let canonical = q_input[axis].get() - transfer_g[axis].get();
        assert!(
            (input.source.q.cartesian[axis].get() - canonical).abs() <= 1.0e-12,
            "q_in = q_canonical + G_transfer, axis {axis}"
        );
    }

    assert_eq!(
        input.orbitals.k_fractional,
        vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    );
    assert_eq!(input.k_minus_q.len(), 2);
    assert_eq!(input.k_minus_q[0].k_index, 0);
    assert_eq!(input.k_minus_q[0].kq_index, 1);
    assert_eq!(input.k_minus_q[0].umklapp.index, [-1, 0, 0]);
    assert_eq!(
        input.k_minus_q[0].umklapp.cartesian,
        reciprocal.cartesian([-1, 0, 0])
    );
    assert_eq!(input.k_minus_q[1].k_index, 1);
    assert_eq!(input.k_minus_q[1].kq_index, 0);
    assert_eq!(input.k_minus_q[1].umklapp.index, [0, 0, 0]);

    let site = physics.geometry().spheres()[0].center;
    assert_eq!(site, [Bohr(2.0), Bohr(4.0), Bohr(4.0)]);
    let phase = pair_phase(reciprocal, input.k_minus_q[0].umklapp.index, site);
    assert!(
        (phase + Complex64::i()).norm() < 1.0e-12,
        "G_wrap · R must be -π/2 at the wrapped muffin-tin site; got {phase}"
    );
    let flipped = pair_phase(reciprocal, [1, 0, 0], site);
    assert!(
        (flipped - Complex64::i()).norm() < 1.0e-12,
        "a G_wrap sign error would yield +i, got {flipped}"
    );
    assert!((phase - flipped).norm() > 1.0);

    let labels = support_indices(&input);
    assert_eq!(
        labels.iter().copied().collect::<BTreeSet<_>>(),
        expected_relative_g(&input)
    );
    assert!(labels.contains(&[-1, 0, 0]));
    assert_eq!(input.pair_columns.n_k, 2);
    let n_orb = input.pair_columns.n_orb;
    assert_eq!(input.orbitals.band_window.count, n_orb);
    assert_eq!(input.pair_columns.encode(1, 0, 1), n_orb * n_orb + 1);
    for channel in &input.orbitals.channels {
        assert_eq!(channel.eigenvectors.len(), 2);
        assert!(channel.eigenvectors.iter().all(|ev| ev.columns() == n_orb));
        assert_eq!(channel.available_bands.len(), 2);
        assert!(channel.available_bands.iter().all(|&count| count >= n_orb));
        assert_ne!(
            channel.available_bands[0], channel.available_bands[1],
            "2x1x1 with cutoff 0.5 must expose differing per-k band counts"
        );
    }
}

#[test]
fn off_mesh_canonical_q_is_rejected_without_rounding_onto_the_mesh() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let config = scalar_config([2, 1, 1], 0.5);
    let error = physics
        .scalar_product_input(&config, [0.25, 0.0, 0.0])
        .unwrap_err();
    assert!(
        matches!(
            error,
            CheckpointPhysicsError::OffMeshTransfer {
                q_in: [q0, 0.0, 0.0],
                q_canonical: [qc, 0.0, 0.0],
                folded: [folded, 0.0, 0.0],
                ..
            } if (q0 - 0.25).abs() < 1.0e-15
                && (qc - 0.25).abs() < 1.0e-15
                && (folded - 0.75).abs() < 1.0e-12
        ),
        "{error:?}"
    );
    let ok = physics
        .scalar_product_input(&config, [1.5, 0.0, 0.0])
        .unwrap();
    assert_eq!(ok.source.q.umklapp.index, [1, 0, 0]);
    assert!(
        (ok.source.q.cartesian[0].get()
            - cartesian_from_fractional(*physics.reciprocal(), [0.5, 0.0, 0.0])[0].get())
        .abs()
            < 1.0e-12
    );
}
