//! Public scalar mixed-product bridge on frozen scalar product input.

use std::collections::BTreeMap;

use muffintin::{
    CheckpointPhysics, SCALAR_MPB_NSPIN, SCALAR_RADIAL_U, SCALAR_RADIAL_UDOT, ScalarMpbError,
    ScalarMpbSelection, ScalarMpbSpec, build_scalar_mpb,
};
use muffintin_core::{Hartree, InverseBohr, lm_from_index};
use muffintin_dft::{
    LinearizationEnergyGenerator, NoncollinearXcRoute, ScfBasis, ScfChannelIdentity,
    ScfChannelProvenance, ScfChannelRecipe, ScfChannelTreatment, ScfConfig, ScfConvergence,
    ScfCoreSite, ScfExchangeCorrelation, ScfKMesh, ScfMixing, ScfOccupations, ScfRelativity,
    XcFunctional,
};
use muffintin_io::{
    AngularBasis, BasisHints, CheckpointMeta, CheckpointV1, CheckpointV2, Complex64V1,
    EnergyParameterV1, EnergyUnit, ExponentialMeshSpec, FourierCoefficientV1, FourierNormalization,
    FourierPhase, GeometryV1, InterstitialV1, InverseLengthUnit, LatticeV1, LengthUnit,
    LinearizationV1, PotentialChannelV1, PotentialConventionV1, PotentialRadialQuantityV1,
    RadialEquationTag, SiteSpinV1, SiteV1, SphericalChannelConvention, SpinTag,
};
use muffintin_operators::CompiledSiteProjection;
use muffintin_prodbasis::mpb::DEFAULT_TOLERANCE;
use muffintin_prodbasis::{CompiledAuxiliaryBasis, OrbitalPair};
use num_complex::Complex64;

fn hydrogen_checkpoint(point_count: usize) -> CheckpointV2 {
    let first: f64 = 1.0e-4;
    let radius: f64 = 1.0;
    let increment = (radius / first).ln() / (point_count - 1) as f64;
    let radii = (0..point_count)
        .map(|index| first * (index as f64 * increment).exp())
        .collect::<Vec<_>>();
    CheckpointV1::new(
        CheckpointMeta {
            title: "scalar MPB hydrogen smoke".to_owned(),
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

fn ground_selection() -> ScalarMpbSelection {
    ScalarMpbSelection {
        spin: 0,
        k: 0,
        left_band: 0,
        right_band: 0,
    }
}

fn mpb_spec(lattice: muffintin_core::ReciprocalLattice, k: usize) -> ScalarMpbSpec {
    ScalarMpbSpec {
        lattice,
        product_l_max: 2,
        product_g_max: InverseBohr(1.5),
        overlap_tolerance: DEFAULT_TOLERANCE,
        selections: vec![ScalarMpbSelection {
            spin: 0,
            k,
            left_band: 0,
            right_band: 0,
        }],
    }
}

fn independent_pw_theta(
    input: &muffintin::ScalarProductInput,
    auxiliary: &CompiledAuxiliaryBasis,
    selection: ScalarMpbSelection,
    conjugate_left: bool,
    reverse_g: bool,
    wrap_sign: i32,
    fold_global_into_rel: bool,
) -> Vec<Complex64> {
    let mapped = input
        .k_minus_q
        .iter()
        .find(|mapped| mapped.k_index == selection.k)
        .expect("k-q map");
    let channel = input
        .orbitals
        .channels
        .iter()
        .find(|channel| channel.spin == selection.spin)
        .expect("spin channel");
    let left_basis = &channel.bases[mapped.kq_index];
    let right_basis = &channel.bases[mapped.k_index];
    let left_ev = &channel.eigenvectors[mapped.kq_index];
    let right_ev = &channel.eigenvectors[mapped.k_index];
    let volume = input.source.partition.interstitial().cell_volume().get();
    let wrap = mapped.umklapp.index.map(|component| wrap_sign * component);
    let global = input.source.q.umklapp;
    let payload = auxiliary.mixed_product().expect("mixed-product auxiliary");
    let mut values = vec![Complex64::default(); payload.interstitial.waves.len()];
    for (left_row, left_wave) in left_basis.plane_waves.iter().enumerate() {
        for (right_row, right_wave) in right_basis.plane_waves.iter().enumerate() {
            let (g_left, g_right) = if reverse_g {
                (right_wave.g.index, left_wave.g.index)
            } else {
                (left_wave.g.index, right_wave.g.index)
            };
            let mut index = [
                g_right[0] - g_left[0] + wrap[0],
                g_right[1] - g_left[1] + wrap[1],
                g_right[2] - g_left[2] + wrap[2],
            ];
            if fold_global_into_rel {
                index[0] += global.index[0];
                index[1] += global.index[1];
                index[2] += global.index[2];
            }
            let g_relative = input
                .source
                .interstitial_pair_support
                .components
                .iter()
                .find(|component| component.g_relative.index == index)
                .map(|component| component.g_relative);
            let Some(g_relative) = g_relative else {
                continue;
            };
            let left_c = left_ev.at(left_row, selection.left_band);
            let right_c = right_ev.at(right_row, selection.right_band);
            let left_factor = if conjugate_left {
                left_c.conj()
            } else {
                left_c
            };
            let amplitude = left_factor * right_c / volume;
            for (local, wave) in payload.interstitial.waves.iter().enumerate() {
                let argument = std::array::from_fn(|axis| {
                    InverseBohr(
                        wave.g.cartesian[axis].get()
                            - global.cartesian[axis].get()
                            - g_relative.cartesian[axis].get(),
                    )
                });
                values[local] += amplitude
                    * auxiliary
                        .partition
                        .interstitial()
                        .coefficient(argument)
                        .unwrap();
            }
        }
    }
    values
}

fn max_abs_diff(left: &[Complex64], right: &[Complex64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(got, want)| (*got - *want).norm())
        .fold(0.0, f64::max)
}

#[test]
fn q0_scalar_mpb_bridge_emits_raw_retained_and_real_vertex() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint(241)).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let mut spec = mpb_spec(*physics.reciprocal(), 0);
    spec.overlap_tolerance = 1.0e-12;
    let result = build_scalar_mpb(&input, &spec).unwrap();

    assert_eq!(result.raw.q, input.source.q);
    assert_eq!(result.auxiliary.q, input.source.q);
    assert_eq!(
        result.raw.interstitial_pair_support,
        input.source.interstitial_pair_support
    );
    let cutoff = result
        .auxiliary
        .mixed_product()
        .and_then(|payload| payload.cutoff)
        .expect("TOL record");
    assert_eq!(cutoff.nspin_factor, SCALAR_MPB_NSPIN);
    assert_eq!(cutoff.value, spec.overlap_tolerance);
    assert!(
        input.source.radials[0]
            .valence
            .iter()
            .any(|radial| radial.n == SCALAR_RADIAL_U)
    );
    assert!(
        input.source.radials[0]
            .valence
            .iter()
            .any(|radial| radial.n == SCALAR_RADIAL_UDOT)
    );

    assert_eq!(result.vertices.len(), 1);
    let record = &result.vertices[0];
    assert_eq!(record.spin, 0);
    assert_eq!(record.k, 0);
    assert_eq!(record.left_band, 0);
    assert_eq!(record.right_band, 0);
    assert_eq!(record.column, input.pair_columns.encode(0, 0, 0));
    assert_eq!(
        record.vertex.pair(),
        OrbitalPair::Bloch {
            k_index: 0,
            left: 0,
            right: 0
        }
    );
    assert_eq!(record.vertex.layout(), &result.auxiliary.layout());
    assert_eq!(record.vertex.q(), input.source.q);
    assert_eq!(
        record.vertex.coefficients().len(),
        result.auxiliary.dimension()
    );
    assert_eq!(record.vertex.mt().len(), result.auxiliary.mt_dimension());
    // This fixture has APW u/udot coordinates only. Compare the MPB integral
    // with raw LAPW coefficients and physical radials, not normalized products.
    let channel = &input.orbitals.channels[0];
    let projected = CompiledSiteProjection::scalar(&channel.bases[0], 0)
        .unwrap()
        .project_eigenvectors(&channel.eigenvectors[0])
        .unwrap();
    let site = &input.source.radials[0];
    let mut physical_mt = Complex64::default();
    for left in 0..projected.coordinate_count() {
        let left_lm = lm_from_index(left / 2);
        let a = &site
            .valence
            .iter()
            .find(|radial| {
                radial.l == left_lm.l && radial.n == left % 2 && radial.spin == channel.spin
            })
            .unwrap()
            .samples;
        for right in 0..projected.coordinate_count() {
            if left / 2 != right / 2 {
                continue;
            }
            let b = &site
                .valence
                .iter()
                .find(|radial| {
                    radial.l == left_lm.l && radial.n == right % 2 && radial.spin == channel.spin
                })
                .unwrap()
                .samples;
            let overlap = site
                .mesh
                .integrate(
                    &a.large
                        .iter()
                        .zip(&b.large)
                        .enumerate()
                        .map(|(i, (ap, bp))| {
                            ap * bp
                                + match (&a.small, &b.small) {
                                    (Some(aq), Some(bq)) => aq[i] * bq[i],
                                    _ => 0.0,
                                }
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            physical_mt += projected.at(left, 0).conj() * projected.at(right, 0) * overlap;
        }
    }
    let mut mpb_mt = Complex64::default();
    for block in &result.auxiliary.mixed_product().unwrap().sites {
        for mode in block.modes.iter().filter(|mode| mode.l == 0) {
            let index = result.auxiliary.mt_index(block.site, 0, 0, mode.n).unwrap();
            let integral = block
                .mesh
                .integrate(
                    &mode
                        .radial
                        .iter()
                        .zip(block.mesh.radii())
                        .map(|(radial, r)| radial * r.get())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
                * (4.0 * std::f64::consts::PI).sqrt();
            mpb_mt += record.vertex.coefficients()[index] * integral;
        }
    }
    assert!(
        (mpb_mt - physical_mt).norm() < 1.0e-8,
        "MPB {mpb_mt}, physical {physical_mt}"
    );
    assert_eq!(
        record.vertex.interstitial().len(),
        result.auxiliary.interstitial_dimension()
    );
    assert!(
        record
            .vertex
            .mt()
            .iter()
            .any(|value| value.norm() > 1.0e-12),
        "q=0 hydrogen vertex must carry muffin-tin signal"
    );
    assert!(
        record
            .vertex
            .interstitial()
            .iter()
            .any(|value| value.norm() > 1.0e-12),
        "q=0 multi-PW vertex must carry interstitial signal"
    );
}

#[test]
fn finite_q_scalar_mpb_uses_canonical_q_and_wraps() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint(61)).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([2, 1, 1], 0.5), [1.5, 0.0, 0.0])
        .unwrap();
    assert_eq!(input.source.q.umklapp.index, [1, 0, 0]);
    assert_eq!(input.k_minus_q[0].k_index, 0);
    assert_eq!(input.k_minus_q[0].kq_index, 1);
    assert_eq!(input.k_minus_q[0].umklapp.index, [-1, 0, 0]);

    let result = build_scalar_mpb(&input, &mpb_spec(*physics.reciprocal(), 0)).unwrap();
    let record = &result.vertices[0];
    assert_eq!(record.vertex.q(), input.source.q);
    assert_eq!(record.vertex.q().umklapp.index, [1, 0, 0]);
    assert_eq!(record.k, 0);
    assert_eq!(record.column, input.pair_columns.encode(0, 0, 0));
    assert!(
        record
            .vertex
            .mt()
            .iter()
            .any(|value| value.norm() > 1.0e-12)
    );
    assert!(
        record
            .vertex
            .interstitial()
            .iter()
            .any(|value| value.norm() > 1.0e-12)
    );

    let expected = independent_pw_theta(
        &input,
        &result.auxiliary,
        ground_selection(),
        true,
        false,
        1,
        false,
    );
    assert!(max_abs_diff(record.vertex.interstitial(), &expected) < 1.0e-10);
    let flipped_wrap = independent_pw_theta(
        &input,
        &result.auxiliary,
        ground_selection(),
        true,
        false,
        -1,
        false,
    );
    assert!(
        max_abs_diff(record.vertex.interstitial(), &flipped_wrap) > 1.0e-8,
        "per-column wrap sign must change the interstitial vertex"
    );
}

#[test]
fn selected_vertex_plane_wave_theta_sum_matches_independent_oracle() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint(61)).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([2, 1, 1], 1.0), [1.5, 0.0, 0.0])
        .unwrap();
    let result = build_scalar_mpb(&input, &mpb_spec(*physics.reciprocal(), 0)).unwrap();
    let interstitial = result.vertices[0].vertex.interstitial();
    let expected = independent_pw_theta(
        &input,
        &result.auxiliary,
        ground_selection(),
        true,
        false,
        1,
        false,
    );
    assert_eq!(interstitial.len(), expected.len());
    assert!(max_abs_diff(interstitial, &expected) < 1.0e-10);
    assert!(expected.iter().any(|value| value.norm() > 1.0e-12));

    let no_conj = independent_pw_theta(
        &input,
        &result.auxiliary,
        ground_selection(),
        false,
        false,
        1,
        false,
    );
    let reversed_g = independent_pw_theta(
        &input,
        &result.auxiliary,
        ground_selection(),
        true,
        true,
        1,
        false,
    );
    let flipped_wrap = independent_pw_theta(
        &input,
        &result.auxiliary,
        ground_selection(),
        true,
        false,
        -1,
        false,
    );
    let folded_global = independent_pw_theta(
        &input,
        &result.auxiliary,
        ground_selection(),
        true,
        false,
        1,
        true,
    );
    assert!(max_abs_diff(interstitial, &no_conj) > 1.0e-8);
    assert!(max_abs_diff(interstitial, &reversed_g) > 1.0e-8);
    assert!(max_abs_diff(interstitial, &flipped_wrap) > 1.0e-8);
    assert!(max_abs_diff(interstitial, &folded_global) > 1.0e-8);
}

#[test]
fn scalar_mpb_rejects_empty_or_incompatible_selection() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint(61)).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([1, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let lattice = *physics.reciprocal();
    let spec = |selections| ScalarMpbSpec {
        lattice,
        product_l_max: 2,
        product_g_max: InverseBohr(1.5),
        overlap_tolerance: DEFAULT_TOLERANCE,
        selections,
    };
    assert!(matches!(
        build_scalar_mpb(&input, &spec(Vec::new())),
        Err(ScalarMpbError::EmptySelection)
    ));
    let n_orb = input.orbitals.band_window.count;
    assert!(matches!(
        build_scalar_mpb(
            &input,
            &spec(vec![ScalarMpbSelection {
                spin: 0,
                k: 0,
                left_band: 0,
                right_band: n_orb,
            }])
        ),
        Err(ScalarMpbError::InvalidSelection {
            spin: 0,
            k: 0,
            left_band: 0,
            right_band,
        }) if right_band == n_orb
    ));
    let mut bad_layout = input.clone();
    bad_layout.pair_columns.n_orb = n_orb + 1;
    assert!(matches!(
        build_scalar_mpb(&bad_layout, &spec(vec![ground_selection()])),
        Err(ScalarMpbError::IncompatiblePairLayout)
    ));
}

#[test]
fn batched_scalar_mpb_preserves_selection_order_and_coefficients() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint(61)).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    assert!(input.orbitals.band_window.count >= 2);
    let selections = vec![
        ScalarMpbSelection {
            spin: 1,
            k: 0,
            left_band: 1,
            right_band: 0,
        },
        ScalarMpbSelection {
            spin: 0,
            k: 0,
            left_band: 0,
            right_band: 1,
        },
        ScalarMpbSelection {
            spin: 1,
            k: 0,
            left_band: 0,
            right_band: 0,
        },
    ];
    let spec = |selections| ScalarMpbSpec {
        lattice: *physics.reciprocal(),
        product_l_max: 2,
        product_g_max: InverseBohr(1.5),
        overlap_tolerance: DEFAULT_TOLERANCE,
        selections,
    };
    let batched = build_scalar_mpb(&input, &spec(selections.clone())).unwrap();
    for (position, selection) in selections.into_iter().enumerate() {
        let single = build_scalar_mpb(&input, &spec(vec![selection])).unwrap();
        let record = &batched.vertices[position];
        assert_eq!(record.spin, selection.spin);
        assert_eq!(record.left_band, selection.left_band);
        assert_eq!(record.right_band, selection.right_band);
        assert!(
            max_abs_diff(
                record.vertex.coefficients(),
                single.vertices[0].vertex.coefficients()
            ) < 1.0e-10
        );
    }
}
