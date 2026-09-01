//! Public selected-band spinor mixed-product bridge.

use std::collections::BTreeMap;

use muffintin::{
    CheckpointPhysics, SPINOR_MPB_NSPIN, SPINOR_RADIAL_LO0, SPINOR_RADIAL_P, SpinorExchangeMpbSpec,
    SpinorMpbError, SpinorMpbSelection, SpinorMpbSpec, build_spinor_exchange_mpb, build_spinor_mpb,
};
use muffintin_core::{ExponentialMesh, Hartree, InverseBohr, Kappa, ReciprocalLattice, TwiceMu};
use muffintin_dft::{
    CoreShellOccupations, CoreShellOrbital, CoreShellOrbitals, CoreShellOrbitalsProvenance,
    LinearizationEnergyGenerator, NoncollinearXcRoute, ScfBasis, ScfChannelIdentity,
    ScfChannelProvenance, ScfChannelRecipe, ScfChannelTreatment, ScfConfig, ScfConvergence,
    ScfCoreSite, ScfExchangeCorrelation, ScfKMesh, ScfMixing, ScfOccupations, ScfRelativity,
    XcFunctional,
};
use muffintin_envelope::site_translation_phase;
use muffintin_io::{
    AngularBasis, BasisHints, CheckpointMeta, CheckpointV1, CheckpointV2, Complex64V1,
    EnergyParameterV1, EnergyUnit, ExponentialMeshSpec, FourierCoefficientV1, FourierNormalization,
    FourierPhase, GeometryV1, InterstitialV1, InverseLengthUnit, LatticeV1, LengthUnit,
    LinearizationV1, PotentialChannelV1, PotentialConventionV1, PotentialRadialQuantityV1,
    RadialEquationTag, SiteSpinV1, SiteV1, SphericalChannelConvention, SpinTag,
};
use muffintin_operators::CompiledSiteProjection;
use muffintin_prodbasis::mpb::{DEFAULT_TOLERANCE, DiracBlochVertexAccumulator};
use muffintin_prodbasis::{
    CompiledAuxiliaryBasis, DiracChargeSector, DiracMtPairSpec, ExchangeSpace, OrbitalPair,
    ProductOrbitalKind,
};
use muffintin_sphere::CoreState;
use num_complex::Complex64;

fn hydrogen_spinor_checkpoint() -> CheckpointV2 {
    let point_count = 61;
    let first: f64 = 1.0e-4;
    let radius: f64 = 1.0;
    let increment = (radius / first).ln() / (point_count - 1) as f64;
    let radii = (0..point_count)
        .map(|index| first * (index as f64 * increment).exp())
        .collect::<Vec<_>>();
    CheckpointV1::new(
        CheckpointMeta {
            title: "spinor MPB hydrogen smoke".to_owned(),
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
                    radial_equation: RadialEquationTag::FullyRelativisticDirac,
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

fn spinor_config(divisions: [usize; 3], cutoff: f64) -> ScfConfig {
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
                ScfChannelRecipe {
                    site: "H-1".to_owned(),
                    identity: ScfChannelIdentity::Kappa { n: 2, kappa: 1 },
                    treatment: ScfChannelTreatment::Hdlo,
                    derivative_order: 2,
                    generator: LinearizationEnergyGenerator::FrozenCheckpoint,
                    seed: None,
                    provenance: ScfChannelProvenance::Site,
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
        relativity: ScfRelativity::SpinorFirstVariation,
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

fn ground_selection() -> SpinorMpbSelection {
    SpinorMpbSelection {
        k: 0,
        left_band: 0,
        right_band: 0,
    }
}

fn mpb_spec(k: usize) -> SpinorMpbSpec {
    SpinorMpbSpec {
        product_l_max: 2,
        product_g_max: InverseBohr(1.5),
        overlap_tolerance: DEFAULT_TOLERANCE,
        selections: vec![SpinorMpbSelection {
            k,
            left_band: 0,
            right_band: 0,
        }],
    }
}

fn core_sidecar(input: &muffintin::SpinorProductInput, occupation: f64) -> CoreShellOrbitals {
    let mesh = &input.source.radials[0].mesh;
    let extended_mesh =
        ExponentialMesh::new(mesh.first(), mesh.increment(), mesh.len() + 7).unwrap();
    let radial = input.source.radials[0]
        .valence
        .iter()
        .find(|radial| radial.kappa == Kappa::new(-1).unwrap() && radial.n == SPINOR_RADIAL_P)
        .unwrap();
    let mut p = radial.samples.large.clone();
    let mut q = radial.samples.small.clone();
    p.resize(extended_mesh.len(), 0.0);
    q.resize(extended_mesh.len(), 0.0);
    let norm_mt = mesh
        .integrate(
            &radial
                .samples
                .large
                .iter()
                .zip(&radial.samples.small)
                .map(|(p, q)| p * p + q * q)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    CoreShellOrbitals {
        site_index: 0,
        site_id: "H-1".to_owned(),
        extended_mesh,
        shells: vec![CoreShellOrbital {
            state: CoreState::new(1, Kappa::new(-1).unwrap()).unwrap(),
            energy: Hartree(-0.5),
            p,
            q,
            norm_total: norm_mt + 0.01,
            norm_mt,
            spill: 0.01,
            occupations: CoreShellOccupations::MuResolved(vec![
                (TwiceMu::new(-1).unwrap(), occupation),
                (TwiceMu::new(1).unwrap(), occupation),
            ]),
        }],
        provenance: CoreShellOrbitalsProvenance {
            extended_potential: Vec::new(),
            solve_specs: Vec::new(),
        },
    }
}

fn exchange_spec() -> SpinorExchangeMpbSpec {
    SpinorExchangeMpbSpec {
        product_l_max: 2,
        product_g_max: InverseBohr(1.5),
        overlap_tolerance: DEFAULT_TOLERANCE,
    }
}

fn fractional_cartesian(reciprocal: ReciprocalLattice, fractional: [f64; 3]) -> [InverseBohr; 3] {
    std::array::from_fn(|axis| {
        InverseBohr(
            fractional
                .iter()
                .zip(reciprocal.basis())
                .map(|(&coefficient, basis)| coefficient * basis[axis].get())
                .sum(),
        )
    })
}

fn minimal_cv_trace(
    input: &muffintin::SpinorProductInput,
    result: &muffintin::SpinorExchangeMpbResult,
) -> f64 {
    result
        .cv
        .vertices
        .iter()
        .map(|record| {
            input.core.orbitals[record.occupied].occupation
                * record
                    .vertex
                    .coefficients()
                    .iter()
                    .map(Complex64::norm_sqr)
                    .sum::<f64>()
        })
        .sum()
}

fn max_abs_diff(left: &[Complex64], right: &[Complex64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(got, want)| (*got - *want).norm())
        .fold(0.0, f64::max)
}

fn independent_pauli_theta(
    input: &muffintin::SpinorProductInput,
    auxiliary: &CompiledAuxiliaryBasis,
    selection: SpinorMpbSelection,
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
    let left_basis = &input.orbitals.bases[mapped.kq_index];
    let right_basis = &input.orbitals.bases[mapped.k_index];
    let left_ev = &input.orbitals.eigenvectors[mapped.kq_index];
    let right_ev = &input.orbitals.eigenvectors[mapped.k_index];
    let volume = input.source.partition.interstitial().cell_volume().get();
    let wrap = mapped.umklapp.index.map(|component| wrap_sign * component);
    let global = input.source.q.umklapp;
    let payload = auxiliary.mixed_product().expect("mixed-product auxiliary");
    let mut values = vec![Complex64::default(); payload.interstitial.waves.len()];
    for (left_g, left_wave) in left_basis.plane_waves.iter().enumerate() {
        for (right_g, right_wave) in right_basis.plane_waves.iter().enumerate() {
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
            let mut amplitude = Complex64::default();
            for spin in 0..2 {
                let left_row = left_basis.layout.plane_wave_index(spin, left_g).unwrap();
                let right_row = right_basis.layout.plane_wave_index(spin, right_g).unwrap();
                let left_c = left_ev.at(left_row, selection.left_band);
                let right_c = right_ev.at(right_row, selection.right_band);
                let left_factor = if conjugate_left {
                    left_c.conj()
                } else {
                    left_c
                };
                amplitude += left_factor * right_c;
            }
            amplitude /= volume;
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

fn independent_mt_sector(
    input: &muffintin::SpinorProductInput,
    raw: &muffintin_prodbasis::DiracRawProductSpace,
    auxiliary: &CompiledAuxiliaryBasis,
    selection: SpinorMpbSelection,
    sector: DiracChargeSector,
) -> Vec<Complex64> {
    let mapped = input
        .k_minus_q
        .iter()
        .find(|mapped| mapped.k_index == selection.k)
        .expect("k-q map");
    let left_basis = &input.orbitals.bases[mapped.kq_index];
    let right_basis = &input.orbitals.bases[mapped.k_index];
    let left_ev = &input.orbitals.eigenvectors[mapped.kq_index];
    let right_ev = &input.orbitals.eigenvectors[mapped.k_index];
    let mut acc = DiracBlochVertexAccumulator::new(
        &input.source,
        raw,
        auxiliary,
        OrbitalPair::Bloch {
            k_index: selection.k,
            left: selection.left_band,
            right: selection.right_band,
        },
    )
    .unwrap();
    for (site, region) in input.source.partition.sites().iter().enumerate() {
        let left_channels = left_basis.site_augmentations[site][0].channels.as_slice();
        let right_channels = right_basis.site_augmentations[site][0].channels.as_slice();
        let left_proj = CompiledSiteProjection::spinor(left_basis, site, left_channels).unwrap();
        let right_proj = CompiledSiteProjection::spinor(right_basis, site, right_channels).unwrap();
        let left_site = left_proj.project_eigenvectors(left_ev).unwrap();
        let right_site = right_proj.project_eigenvectors(right_ev).unwrap();
        let phase = site_translation_phase(input.source.q.cartesian, region.position).conj();
        let known: std::collections::HashSet<_> = raw
            .radial_products
            .iter()
            .filter(|product| product.channel.sector == sector)
            .flat_map(|product| {
                [
                    (product.channel.left, product.channel.right),
                    (product.channel.right, product.channel.left),
                ]
            })
            .collect();
        for left_coord in 0..left_site.coordinate_count() {
            let (left_id, left_mu) = input.site_projection_identity(site, left_coord).unwrap();
            for right_coord in 0..right_site.coordinate_count() {
                let (right_id, right_mu) =
                    input.site_projection_identity(site, right_coord).unwrap();
                if !known.contains(&(left_id, right_id)) {
                    continue;
                }
                let spec = DiracMtPairSpec {
                    left: left_id,
                    left_twice_mu: left_mu,
                    right: right_id,
                    right_twice_mu: right_mu,
                };
                let amplitude = left_site.at(left_coord, selection.left_band).conj()
                    * right_site.at(right_coord, selection.right_band)
                    * phase;
                match sector {
                    DiracChargeSector::LargeLarge => acc.add_pp(spec, amplitude).unwrap(),
                    DiracChargeSector::SmallSmall => acc.add_qq(spec, amplitude).unwrap(),
                }
            }
        }
    }
    acc.finish().unwrap().mt().to_vec()
}

#[test]
fn spinor_mpb_rejects_empty_selection() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let input = physics
        .spinor_product_input(&spinor_config([1, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let error = build_spinor_mpb(
        &input,
        &SpinorMpbSpec {
            product_l_max: 2,
            product_g_max: InverseBohr(1.5),
            overlap_tolerance: DEFAULT_TOLERANCE,
            selections: Vec::new(),
        },
    )
    .unwrap_err();
    assert!(matches!(error, SpinorMpbError::EmptySelection));
}

#[test]
fn spinor_mpb_rejects_band_outside_leading_window() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let input = physics
        .spinor_product_input(&spinor_config([1, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let n_orb = input.orbitals.band_window.count;
    let error = build_spinor_mpb(
        &input,
        &SpinorMpbSpec {
            product_l_max: 2,
            product_g_max: InverseBohr(1.5),
            overlap_tolerance: DEFAULT_TOLERANCE,
            selections: vec![SpinorMpbSelection {
                k: 0,
                left_band: n_orb,
                right_band: 0,
            }],
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        SpinorMpbError::InvalidSelection {
            k: 0,
            left_band,
            right_band: 0
        } if left_band == n_orb
    ));
}

#[test]
fn q0_signed_kappa_hdlo_has_site_identity_and_pp_qq_signals() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let input = physics
        .spinor_product_input(&spinor_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let kappa = Kappa::new(1).unwrap();
    let mut found_lo = false;
    let channels = input.orbitals.bases[0].site_augmentations[0][0]
        .channels
        .as_slice();
    let projection = CompiledSiteProjection::spinor(&input.orbitals.bases[0], 0, channels).unwrap();
    for coord in 0..projection.coordinate_count() {
        let (id, twice_mu) = input
            .site_projection_identity(0, coord)
            .expect("every projection coordinate inverts");
        assert_eq!(
            input.site_projection_row(0, id.kappa, twice_mu, id.n),
            Some(coord)
        );
        if id.kappa == kappa && id.n == SPINOR_RADIAL_LO0 {
            found_lo = true;
        }
    }
    assert!(
        found_lo,
        "kappa=+1 HDLO must occupy a site-projection coordinate"
    );

    let result = build_spinor_mpb(&input, &mpb_spec(0)).unwrap();
    assert_eq!(result.reciprocal, input.reciprocal);
    assert_eq!(result.pair_columns, input.pair_columns);
    assert_eq!(result.raw.q, input.source.q);
    assert_eq!(result.auxiliary.q, input.source.q);
    let cutoff = result
        .auxiliary
        .mixed_product()
        .and_then(|payload| payload.cutoff)
        .expect("TOL record");
    assert_eq!(cutoff.nspin_factor, SPINOR_MPB_NSPIN);
    assert_eq!(cutoff.value, DEFAULT_TOLERANCE);

    let record = &result.vertices[0];
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

    let pp = independent_mt_sector(
        &input,
        &result.raw,
        &result.auxiliary,
        ground_selection(),
        DiracChargeSector::LargeLarge,
    );
    let qq = independent_mt_sector(
        &input,
        &result.raw,
        &result.auxiliary,
        ground_selection(),
        DiracChargeSector::SmallSmall,
    );
    assert!(
        pp.iter().any(|value| value.norm() > 1.0e-12),
        "q=0 hydrogen PP muffin-tin sector must be supported"
    );
    assert!(
        qq.iter().any(|value| value.norm() > 1.0e-12),
        "q=0 hydrogen QQ muffin-tin sector must be supported"
    );
    let mut summed = pp.clone();
    for (total, part) in summed.iter_mut().zip(&qq) {
        *total += *part;
    }
    assert!(max_abs_diff(record.vertex.mt(), &summed) < 1.0e-10);
    assert!(max_abs_diff(record.vertex.mt(), &pp) > 1.0e-8);
    assert!(max_abs_diff(record.vertex.mt(), &qq) > 1.0e-8);
    assert!(
        record
            .vertex
            .interstitial()
            .iter()
            .any(|value| value.norm() > 1.0e-12)
    );
}

#[test]
fn finite_q_two_pauli_interstitial_matches_independent_theta_oracle() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let input = physics
        .spinor_product_input(&spinor_config([2, 1, 1], 1.0), [1.5, 0.0, 0.0])
        .unwrap();
    assert_eq!(input.source.q.umklapp.index, [1, 0, 0]);
    assert_eq!(input.k_minus_q[0].umklapp.index, [-1, 0, 0]);
    let result = build_spinor_mpb(&input, &mpb_spec(0)).unwrap();
    let interstitial = result.vertices[0].vertex.interstitial();
    let expected = independent_pauli_theta(
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

    let no_conj = independent_pauli_theta(
        &input,
        &result.auxiliary,
        ground_selection(),
        false,
        false,
        1,
        false,
    );
    let reversed_g = independent_pauli_theta(
        &input,
        &result.auxiliary,
        ground_selection(),
        true,
        true,
        1,
        false,
    );
    let flipped_wrap = independent_pauli_theta(
        &input,
        &result.auxiliary,
        ground_selection(),
        true,
        false,
        -1,
        false,
    );
    let double_umklapp = independent_pauli_theta(
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
    assert!(max_abs_diff(interstitial, &double_umklapp) > 1.0e-8);
}

#[test]
fn rectangular_core_vertices_are_mt_only_pp_qq_and_occupation_free() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let plain = physics
        .spinor_product_input(&spinor_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let low = plain
        .clone()
        .with_core_sidecars(&[core_sidecar(&plain, 0.25)])
        .unwrap();
    let high = plain
        .clone()
        .with_core_sidecars(&[core_sidecar(&plain, 0.5)])
        .unwrap();
    let low_result = build_spinor_exchange_mpb(&low, &exchange_spec()).unwrap();
    let high_result = build_spinor_exchange_mpb(&high, &exchange_spec()).unwrap();
    assert_eq!(low_result.cv.layout.occupied_space, ExchangeSpace::Core);
    assert_eq!(low_result.cv.layout.target_space, ExchangeSpace::Valence);
    assert_eq!(low_result.vc.layout.occupied_space, ExchangeSpace::Valence);
    assert_eq!(low_result.vc.layout.target_space, ExchangeSpace::Core);
    assert_eq!(low_result.cc.layout.occupied_space, ExchangeSpace::Core);
    assert_eq!(low_result.cc.layout.target_space, ExchangeSpace::Core);
    assert!(
        low_result
            .raw
            .interstitial_pair_support
            .components
            .is_empty()
    );
    assert!(
        low_result
            .raw
            .radial_products
            .iter()
            .all(|product| matches!(
                product.channel.sector,
                DiracChargeSector::LargeLarge | DiracChargeSector::SmallSmall
            ))
    );
    assert!(low_result.raw.radial_products.iter().any(|product| {
        product.channel.left.kind == ProductOrbitalKind::Core
            && product.channel.right.kind == ProductOrbitalKind::Core
    }));
    for sector in [&low_result.cv, &low_result.vc, &low_result.cc] {
        assert!(sector.vertices.iter().all(|record| {
            record
                .vertex
                .interstitial()
                .iter()
                .all(|value| value.norm() == 0.0)
        }));
    }
    assert_eq!(low_result.cv.vertices, high_result.cv.vertices);
    assert_eq!(low_result.vc.vertices, high_result.vc.vertices);
    assert_eq!(low_result.cc.vertices, high_result.cc.vertices);
    let low_trace = minimal_cv_trace(&low, &low_result);
    let high_trace = minimal_cv_trace(&high, &high_result);
    assert!((high_trace - 2.0 * low_trace).abs() < 1.0e-12 * high_trace.max(1.0));
    assert!(
        low_result
            .diagnostics
            .cv
            .iter()
            .chain(&low_result.diagnostics.vc)
            .all(|diagnostic| diagnostic.coupling.is_some()
                && diagnostic.direct_overlap.is_some()
                && diagnostic.residual.is_some())
    );
    assert!(low_result.diagnostics.max_residual.unwrap() < 1.0e-9);
}

#[test]
fn finite_q_core_phase_uses_k_minus_q_wrap_and_gamma_couplings_are_conjugate() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let gamma_plain = physics
        .spinor_product_input(&spinor_config([2, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let sidecar = core_sidecar(&gamma_plain, 1.0);
    let gamma = gamma_plain
        .with_core_sidecars(std::slice::from_ref(&sidecar))
        .unwrap();
    let finite = physics
        .spinor_product_input(&spinor_config([2, 1, 1], 1.0), [1.5, 0.0, 0.0])
        .unwrap()
        .with_core_sidecars(&[sidecar])
        .unwrap();
    assert_eq!(finite.k_minus_q[0].umklapp.index, [-1, 0, 0]);
    let gamma_result = build_spinor_exchange_mpb(&gamma, &exchange_spec()).unwrap();
    let finite_result = build_spinor_exchange_mpb(&finite, &exchange_spec()).unwrap();
    assert!(
        finite_result
            .diagnostics
            .cv
            .iter()
            .chain(&finite_result.diagnostics.vc)
            .all(|diagnostic| diagnostic.coupling.is_none()
                && diagnostic.direct_overlap.is_none()
                && diagnostic.residual.is_none())
    );
    assert_eq!(finite_result.diagnostics.max_residual, None);

    let core = 0;
    let gamma_column = gamma_result.cc.layout.encode(0, core, core).unwrap();
    let finite_column = finite_result.cc.layout.encode(0, core, core).unwrap();
    let gamma_vertex = &gamma_result.cc.vertices[gamma_column].vertex;
    let finite_vertex = &finite_result.cc.vertices[finite_column].vertex;
    let k = finite.orbitals.k_fractional[0];
    let kq = finite.orbitals.k_fractional[finite.k_minus_q[0].kq_index];
    let delta = std::array::from_fn(|axis| k[axis] - kq[axis]);
    let expected = site_translation_phase(
        fractional_cartesian(finite.reciprocal, delta),
        finite.source.partition.sites()[0].position,
    );
    assert!(gamma_vertex.mt().iter().any(|value| value.norm() > 1.0e-12));
    assert!(
        finite_vertex
            .mt()
            .iter()
            .zip(gamma_vertex.mt())
            .all(|(got, reference)| (*got - expected * reference).norm() < 1.0e-9)
    );
    for cv in &gamma_result.diagnostics.cv {
        let (k, core, valence) = gamma_result.cv.layout.decode(cv.column).unwrap();
        let reverse = gamma_result.vc.layout.encode(k, valence, core).unwrap();
        let vc = &gamma_result.diagnostics.vc[reverse];
        assert!((cv.coupling.unwrap() - vc.coupling.unwrap().conj()).norm() < 1.0e-9);
    }
}
