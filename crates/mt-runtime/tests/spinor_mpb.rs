//! Public M-L5c selected-band spinor mixed-product bridge.

use std::collections::BTreeMap;

use muffintin::{
    SPINOR_MPB_NSPIN, SPINOR_RADIAL_LO0, SnapshotDftPhysics, SpinorMpbError, SpinorMpbSelection,
    SpinorMpbSpec, build_spinor_mpb,
};
use muffintin_auxiliary_ir::{
    CompiledAuxiliaryBasis, DiracChargeSector, DiracMtPairSpec, OrbitalPair,
};
use muffintin_core::{Hartree, InverseBohr, Kappa};
use muffintin_dft::{
    LinearizationEnergyGenerator, NoncollinearXcRoute, ScfBasis, ScfChannelIdentity,
    ScfChannelProvenance, ScfChannelRecipe, ScfChannelTreatment, ScfConfig, ScfConvergence,
    ScfCoreSite, ScfExchangeCorrelation, ScfKMesh, ScfMixing, ScfOccupations, ScfRelativity,
    XcFunctional,
};
use muffintin_envelope::site_translation_phase;
use muffintin_io::{
    AngularBasisV1, BasisHintsV1, Complex64V1, EnergyParameterV1, EnergyUnitV1,
    ExponentialMeshSpecV1, FourierCoefficientV1, FourierNormalizationV1, FourierPhaseV1,
    GeometryV1, InterstitialV1, InverseLengthUnitV1, LatticeV1, LengthUnitV1, LinearizationV1,
    MetaV1, PotentialChannelV1, PotentialConventionV1, PotentialRadialQuantityV1,
    RadialEquationTagV1, SiteSpinV1, SiteV1, SnapshotV1, SnapshotV2, SphericalChannelConventionV1,
    SpinTagV1,
};
use muffintin_mpb::{DEFAULT_TOLERANCE, DiracBlochVertexAccumulator};
use muffintin_operators::CompiledSiteProjection;
use num_complex::Complex64;

fn hydrogen_spinor_snapshot() -> SnapshotV2 {
    let point_count = 61;
    let first: f64 = 1.0e-4;
    let radius: f64 = 1.0;
    let increment = (radius / first).ln() / (point_count - 1) as f64;
    let radii = (0..point_count)
        .map(|index| first * (index as f64 * increment).exp())
        .collect::<Vec<_>>();
    SnapshotV1::new(
        MetaV1 {
            title: "spinor MPB hydrogen smoke".to_owned(),
            producer: "mt-runtime test".to_owned(),
            producer_version: None,
            energy_zero: "zero interstitial Fourier mean".to_owned(),
            potential_convention: PotentialConventionV1 {
                angular_basis: AngularBasisV1::ComplexCondonShortley,
                radial_quantity: PotentialRadialQuantityV1::Potential,
                spherical_channel: SphericalChannelConventionV1::PhysicalValue,
            },
            annotations: BTreeMap::new(),
        },
        GeometryV1 {
            lattice: LatticeV1 {
                unit: LengthUnitV1::Bohr,
                vectors: [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            },
            sites: vec![SiteV1 {
                id: "H-1".to_owned(),
                atomic_number: 1,
                fractional_position: [1.25, -0.5, 0.5],
                muffin_tin_radius_unit: LengthUnitV1::Bohr,
                muffin_tin_radius: radius,
                spins: vec![SiteSpinV1 {
                    spin: SpinTagV1::Scalar,
                    mesh: ExponentialMeshSpecV1 {
                        radius_unit: LengthUnitV1::Bohr,
                        first,
                        log_increment: increment,
                        point_count,
                        last: first * ((point_count - 1) as f64 * increment).exp(),
                        consistency_tolerance: 1.0e-12,
                    },
                    radial_equation: RadialEquationTagV1::FullyRelativisticDirac,
                    potential_unit: EnergyUnitV1::Hartree,
                    potential_channels: vec![PotentialChannelV1 {
                        l: 0,
                        m: 0,
                        real: radii.iter().map(|radius| -1.0 / radius).collect(),
                        imaginary: Vec::new(),
                    }],
                    linearization: LinearizationV1 {
                        energy_unit: EnergyUnitV1::Hartree,
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
            coefficient_unit: EnergyUnitV1::Hartree,
            coefficients: vec![FourierCoefficientV1 {
                g: [0; 3],
                value: Complex64V1 {
                    real: 0.0,
                    imaginary: 0.0,
                },
            }],
            basis_hints: BasisHintsV1 {
                reciprocal_length_unit: InverseLengthUnitV1::BohrInverse,
                plane_wave_cutoff: Some(0.5),
                coefficient_cutoff: Some(1.0),
                normalization: FourierNormalizationV1::CellNormalized,
                phase: FourierPhaseV1::NegativeExponent,
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
        },
        basis: ScfBasis {
            plane_wave_cutoff: cutoff,
            l_max: 1,
            channels: vec![
                ScfChannelRecipe {
                    site: "H-1".to_owned(),
                    identity: ScfChannelIdentity::ScalarL { n: 1, l: 0 },
                    treatment: ScfChannelTreatment::Valence,
                    derivative_order: 0,
                    generator: LinearizationEnergyGenerator::FrozenSnapshot,
                    seed: None,
                    provenance: ScfChannelProvenance::BuiltIn,
                },
                ScfChannelRecipe {
                    site: "H-1".to_owned(),
                    identity: ScfChannelIdentity::ScalarL { n: 2, l: 1 },
                    treatment: ScfChannelTreatment::Valence,
                    derivative_order: 0,
                    generator: LinearizationEnergyGenerator::FrozenSnapshot,
                    seed: None,
                    provenance: ScfChannelProvenance::BuiltIn,
                },
                ScfChannelRecipe {
                    site: "H-1".to_owned(),
                    identity: ScfChannelIdentity::Kappa { n: 2, kappa: 1 },
                    treatment: ScfChannelTreatment::Hdlo,
                    derivative_order: 2,
                    generator: LinearizationEnergyGenerator::FrozenSnapshot,
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
    raw: &muffintin_auxiliary_ir::DiracRawProductSpace,
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
    let physics = SnapshotDftPhysics::new(&hydrogen_spinor_snapshot()).unwrap();
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
    let physics = SnapshotDftPhysics::new(&hydrogen_spinor_snapshot()).unwrap();
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
    let physics = SnapshotDftPhysics::new(&hydrogen_spinor_snapshot()).unwrap();
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
    let physics = SnapshotDftPhysics::new(&hydrogen_spinor_snapshot()).unwrap();
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
