//! Public frozen full-first-variation spinor product-input tests.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;

use muffintin::{
    SPINOR_RADIAL_LO0, SPINOR_RADIAL_P, SPINOR_RADIAL_PDOT, CheckpointPhysicsError, CheckpointPhysics,
};
use muffintin_prodbasis::{DiracRadial, ProductOrbitalKind};
use muffintin_core::{Bohr, Hartree, InverseBohr, Kappa, ReciprocalLattice, TwiceMu};
use muffintin_dft::{
    FirstVariationWindow, LinearizationEnergyGenerator, NoncollinearXcRoute, ScfBasis,
    ScfChannelIdentity, ScfChannelProvenance, ScfChannelRecipe, ScfChannelTreatment, ScfConfig,
    ScfConvergence, ScfCoreSite, ScfExchangeCorrelation, ScfKMesh, ScfMixing, ScfOccupations,
    ScfRelativity, XcFunctional,
};
use muffintin_io::{
    AngularBasisV1, BasisHintsV1, Complex64V1, EnergyParameterV1, EnergyUnitV1,
    ExponentialMeshSpecV1, FourierCoefficientV1, FourierNormalizationV1, FourierPhaseV1,
    GeometryV1, InterstitialV1, InverseLengthUnitV1, LatticeV1, LengthUnitV1, LinearizationV1,
    MetaV1, PotentialChannelV1, PotentialConventionV1, PotentialRadialQuantityV1,
    RadialEquationTagV1, SiteSpinV1, SiteV1, CheckpointV1, CheckpointV2, SphericalChannelConventionV1,
    SpinTagV1,
};
use muffintin_sphere::{SPEX_SPEED_OF_LIGHT, ValenceDiracSpec, solve_valence_dirac};
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
        MetaV1 {
            title: "spinor product-input hydrogen smoke".to_owned(),
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

fn expected_relative_g(input: &muffintin::SpinorProductInput) -> BTreeSet<[i32; 3]> {
    let mut indices = BTreeSet::new();
    for mapped in &input.k_minus_q {
        let g_k = &input.orbitals.bases[mapped.k_index].plane_waves;
        let g_kmq = &input.orbitals.bases[mapped.kq_index].plane_waves;
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
    indices
}

fn support_indices(input: &muffintin::SpinorProductInput) -> Vec<[i32; 3]> {
    input
        .source
        .interstitial_pair_support
        .components
        .iter()
        .map(|component| component.g_relative.index)
        .collect()
}

fn find_radial(input: &muffintin::SpinorProductInput, kappa: i32, n: usize) -> &DiracRadial {
    let kappa = Kappa::new(kappa).unwrap();
    input.source.radials[0]
        .valence
        .iter()
        .find(|radial| radial.kappa == kappa && radial.n == n)
        .unwrap()
}

fn spherical_potential(physics: &CheckpointPhysics) -> Vec<f64> {
    let monopole = physics.frozen_potential().scalar().muffin_tins()[0]
        .field()
        .channel(0, 0)
        .expect("checkpoint monopole");
    monopole
        .iter()
        .map(|value| value.re / (4.0 * PI).sqrt())
        .collect()
}

fn samples_close(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| (a - b).abs() <= 1.0e-12 * (1.0 + a.abs() + b.abs()))
}

#[test]
fn spinor_product_input_rejects_scalar_relativity() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let mut config = spinor_config([1, 1, 1], 0.5);
    config.relativity = ScfRelativity::Scalar;
    let error = physics.spinor_product_input(&config, [0.0; 3]).unwrap_err();
    assert!(matches!(
        error,
        CheckpointPhysicsError::SpinorProductRejectsScalarRelativity
    ));
}

#[test]
fn spinor_product_input_rejects_soc_second_variation() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let mut config = spinor_config([1, 1, 1], 0.5);
    config.relativity = ScfRelativity::SocSecondVariation {
        window: FirstVariationWindow::new(0, 1).unwrap(),
    };
    let error = physics.spinor_product_input(&config, [0.0; 3]).unwrap_err();
    assert!(matches!(
        error,
        CheckpointPhysicsError::SpinorProductRejectsSocSecondVariation
    ));
}

#[test]
fn q0_signed_kappa_lo_row_maps_to_dirac_radial_and_samples() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let config = spinor_config([1, 1, 1], 0.5);
    let input = physics.spinor_product_input(&config, [0.0; 3]).unwrap();

    assert_eq!(
        input.source.provenance.reference.as_deref(),
        Some("checkpoint-dft-frozen-spinor-product-input")
    );
    assert!(input.source.radials[0].cores.is_empty());
    assert_eq!(input.orbitals.band_window.start, 0);

    let kappa = Kappa::new(1).unwrap();
    let twice_mu = TwiceMu::new(-1).unwrap();
    assert!(
        input
            .source
            .find_radial(muffintin_prodbasis::DiracRadialId {
                site: 0,
                kind: ProductOrbitalKind::Valence,
                kappa,
                n: SPINOR_RADIAL_LO0,
            })
            .is_some()
    );
    let row = input
        .compiled_lo_row(0, 0, kappa, twice_mu, SPINOR_RADIAL_LO0)
        .expect("first +kappa LO is a compiled row");
    let compiled = &input.orbitals.bases[0];
    assert!(row >= compiled.layout.plane_wave_count());
    assert_eq!(
        compiled.layout.site_spinor_index(0, kappa, twice_mu, 0),
        Some(row)
    );
    let (id, mapped_mu) = input.compiled_lo_identity(0, row).unwrap();
    assert_eq!(id.site, 0);
    assert_eq!(id.kind, ProductOrbitalKind::Valence);
    assert_eq!(id.kappa, kappa);
    assert_eq!(id.n, SPINOR_RADIAL_LO0);
    assert_eq!(mapped_mu, twice_mu);
    assert!(
        input
            .compiled_lo_row(0, 0, kappa, twice_mu, SPINOR_RADIAL_P)
            .is_none(),
        "APW n=0 is matching, not a compiled LO row"
    );

    let lo = find_radial(&input, 1, SPINOR_RADIAL_LO0);
    assert_eq!(lo.samples.large.len(), input.source.radials[0].mesh.len());
    assert_eq!(lo.samples.small.len(), lo.samples.large.len());
    assert!(lo.samples.large.iter().any(|value| value.abs() > 0.0));
    assert!(lo.samples.small.iter().any(|value| value.abs() > 0.0));

    let host = input.orbitals.eigenvectors[0].to_host_column_major();
    let rows = input.orbitals.eigenvectors[0].rows();
    let cols = input.orbitals.eigenvectors[0].columns();
    let peak = (0..cols)
        .map(|band| host[band * rows + row].norm())
        .fold(0.0_f64, f64::max);
    assert!(
        peak > 0.0,
        "compiled LO row {row} must participate in some band; peak={peak}"
    );
}

#[test]
fn q0_row_count_equals_two_pauli_pw_blocks_plus_site_lo() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let config = spinor_config([1, 1, 1], 1.0);
    let input = physics.spinor_product_input(&config, [0.0; 3]).unwrap();
    let compiled = &input.orbitals.bases[0];
    let n_g = compiled.plane_waves.len();
    assert!(n_g > 1, "cutoff 1.0 must retain more than Gamma");
    assert_eq!(compiled.layout.spatial_plane_wave_count(), n_g);
    assert_eq!(compiled.layout.plane_wave_count(), 2 * n_g);
    assert_eq!(input.pauli_plane_wave_row(0, 0, 0), Some(0));
    assert_eq!(input.pauli_plane_wave_row(0, 1, 0), Some(n_g));
    let labels: BTreeSet<_> = compiled
        .plane_waves
        .iter()
        .map(|wave| wave.g.index)
        .collect();
    assert_eq!(labels.len(), n_g, "Pauli blocks share spatial G labels");

    let lo_dim = compiled
        .layout
        .site_layout(0)
        .map(|site| site.len())
        .unwrap_or(0);
    assert!(lo_dim > 0, "kappa=+1 LOs must occupy compiled site rows");
    assert_eq!(compiled.layout.dimension(), 2 * n_g + lo_dim);
    assert_eq!(
        input.orbitals.eigenvectors[0].rows(),
        compiled.layout.dimension()
    );
    assert_eq!(
        input.orbitals.eigenvectors[0].columns(),
        input.orbitals.band_window.count
    );
}

#[test]
fn physical_pq_matches_independent_dirac_materialization() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let config = spinor_config([1, 1, 1], 0.5);
    let input = physics.spinor_product_input(&config, [0.0; 3]).unwrap();
    let mesh = &input.source.radials[0].mesh;
    let spherical = spherical_potential(&physics);
    let kappa = Kappa::new(1).unwrap();
    let base = solve_valence_dirac(
        mesh,
        &spherical,
        ValenceDiracSpec::new(kappa, Hartree(-0.15)).unwrap(),
    )
    .unwrap();
    let emitted_p = find_radial(&input, 1, SPINOR_RADIAL_P);
    let emitted_pdot = find_radial(&input, 1, SPINOR_RADIAL_PDOT);
    assert!(samples_close(&emitted_p.samples.large, &base.p));
    assert!(samples_close(&emitted_p.samples.small, &base.q));
    assert!(samples_close(
        &emitted_pdot.samples.large,
        &base.energy_derivative.p
    ));
    assert!(samples_close(
        &emitted_pdot.samples.small,
        &base.energy_derivative.q
    ));

    let cq = base
        .q
        .iter()
        .map(|value| value * SPEX_SPEED_OF_LIGHT)
        .collect::<Vec<_>>();
    assert!(
        !samples_close(&emitted_p.samples.small, &cq),
        "physical Q must not equal a hidden cQ rescaling"
    );
    let omitted = vec![0.0; base.q.len()];
    assert!(
        !samples_close(&emitted_p.samples.small, &omitted),
        "physical Q must not be omitted"
    );

    let independent_lo = base.sra_hdlo(mesh).unwrap();
    let emitted_lo = find_radial(&input, 1, SPINOR_RADIAL_LO0);
    assert!(samples_close(&emitted_lo.samples.large, &independent_lo.p));
    assert!(samples_close(&emitted_lo.samples.small, &independent_lo.q));
    assert_eq!(emitted_lo.n, SPINOR_RADIAL_LO0);
}

#[test]
fn finite_q_wrap_and_off_mesh_transfer() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let config = spinor_config([2, 1, 1], 0.5);
    let q_fractional = [1.5, 0.0, 0.0];
    let input = physics.spinor_product_input(&config, q_fractional).unwrap();

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
    assert_eq!(input.k_minus_q[0].k_index, 0);
    assert_eq!(input.k_minus_q[0].kq_index, 1);
    assert_eq!(input.k_minus_q[0].umklapp.index, [-1, 0, 0]);
    assert_eq!(input.k_minus_q[1].k_index, 1);
    assert_eq!(input.k_minus_q[1].kq_index, 0);
    assert_eq!(input.k_minus_q[1].umklapp.index, [0, 0, 0]);

    let site = physics.geometry().spheres()[0].center;
    let phase = pair_phase(reciprocal, input.k_minus_q[0].umklapp.index, site);
    assert!((phase + Complex64::i()).norm() < 1.0e-12);
    let flipped = pair_phase(reciprocal, [1, 0, 0], site);
    assert!((flipped - Complex64::i()).norm() < 1.0e-12);

    let labels = support_indices(&input);
    assert_eq!(
        labels.iter().copied().collect::<BTreeSet<_>>(),
        expected_relative_g(&input)
    );
    assert!(labels.contains(&[-1, 0, 0]));

    let error = physics
        .spinor_product_input(&config, [0.25, 0.0, 0.0])
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
}

#[test]
fn differing_per_k_counts_keep_full_eigenvector_rows() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let config = spinor_config([2, 1, 1], 0.5);
    let input = physics.spinor_product_input(&config, [0.0; 3]).unwrap();
    assert_eq!(input.orbitals.band_window.start, 0);
    let n_orb = input.orbitals.band_window.count;
    assert_eq!(input.pair_columns.n_orb, n_orb);
    assert_eq!(input.pair_columns.n_k, 2);
    assert_ne!(
        input.orbitals.available_bands[0], input.orbitals.available_bands[1],
        "2x1x1 with cutoff 0.5 must expose differing per-k band counts"
    );
    assert_ne!(
        input.orbitals.bases[0].layout.dimension(),
        input.orbitals.bases[1].layout.dimension(),
        "per-k plane-wave counts must differ"
    );
    for k in 0..2 {
        assert_eq!(input.orbitals.eigenvectors[k].columns(), n_orb);
        assert_eq!(
            input.orbitals.eigenvectors[k].rows(),
            input.orbitals.bases[k].layout.dimension()
        );
        assert!(input.orbitals.available_bands[k] >= n_orb);
        assert_eq!(
            input.orbitals.eigenvectors[k].rows(),
            2 * input.orbitals.bases[k].plane_waves.len()
                + input.orbitals.bases[k]
                    .layout
                    .site_layout(0)
                    .map(|site| site.len())
                    .unwrap_or(0)
        );
    }
}
