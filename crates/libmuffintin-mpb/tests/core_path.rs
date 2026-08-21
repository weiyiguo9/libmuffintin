//! SPEX-convention mixed-product core path without a live SPEX dump.

use libmuffintin_basis::Provenance;
use libmuffintin_core::{
    Bohr, GVector, InterstitialGeometry, InverseBohr, ReciprocalLattice, Sphere, VolumeBohr3, gaunt,
};
use libmuffintin_envelope::site_translation_phase;
use libmuffintin_mpb::{
    DEFAULT_TOLERANCE, apply_overlap_cutoff, interstitial_plane_waves, pair_vertex,
    spex_mixed_product_basis,
};
use libmuffintin_operators::solve_real_symmetric;
use libmuffintin_product::{
    AuxiliaryRegion, CompiledAuxiliaryBasis, InterstitialPairSpec, MixedProductAuxiliary,
    MtPairSpec, OrbitalPair, PairVertexSpec, ProductError, ProductOrbitalKind, ProductPartition,
    ProductRadial, ProductRadialId, ProductSource, RadialSamples, RawInterstitialPairComponent,
    RawInterstitialPairSupport, SiteRadialSet, TransferQ,
};
use num_complex::Complex64;

const RADIUS: f64 = 0.8;
const POSITION: [Bohr; 3] = [Bohr(0.25), Bohr(0.0), Bohr(0.0)];

fn mesh() -> libmuffintin_core::ExponentialMesh {
    let first = 1.0e-5;
    let number = 73;
    let increment = (RADIUS / first).ln() / (number - 1) as f64;
    libmuffintin_core::ExponentialMesh::new(Bohr(first), increment, number).unwrap()
}

fn samples(kind: u8) -> RadialSamples {
    let mesh = mesh();
    let large = mesh
        .radii()
        .iter()
        .map(|radius| {
            let r = radius.get();
            match kind {
                0 => r * (-2.0 * r).exp(),
                1 => r * (1.0 - 0.4 * r) * (-2.0 * r).exp(),
                2 => r * (-4.5 * r).exp(),
                _ => r * r * (-2.2 * r).exp(),
            }
        })
        .collect();
    RadialSamples { large, small: None }
}

fn partition() -> ProductPartition {
    ProductPartition::from_interstitial(
        InterstitialGeometry::new(
            VolumeBohr3(512.0),
            vec![Sphere {
                center: POSITION,
                radius: Bohr(RADIUS),
            }],
        )
        .unwrap(),
    )
}

fn cubic_lattice() -> ReciprocalLattice {
    ReciprocalLattice::from_direct([
        [Bohr(8.0), Bohr(0.0), Bohr(0.0)],
        [Bohr(0.0), Bohr(8.0), Bohr(0.0)],
        [Bohr(0.0), Bohr(0.0), Bohr(8.0)],
    ])
    .unwrap()
}

fn g_vector(lattice: &ReciprocalLattice, index: [i32; 3]) -> GVector {
    let cartesian = lattice.cartesian(index);
    let norm = InverseBohr(
        cartesian
            .iter()
            .map(|component| component.get().powi(2))
            .sum::<f64>()
            .sqrt(),
    );
    GVector {
        index,
        cartesian,
        norm,
    }
}

fn convention_pair_support(
    q: TransferQ,
    lattice: &ReciprocalLattice,
) -> RawInterstitialPairSupport {
    let labels = [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [2, 0, 0],
    ];
    RawInterstitialPairSupport::from_components(
        q,
        labels
            .into_iter()
            .map(|index| RawInterstitialPairComponent {
                g_relative: g_vector(lattice, index),
            })
            .collect(),
    )
    .unwrap()
}

fn source_with_support(
    include_core: bool,
    include_p: bool,
    q: TransferQ,
    support: RawInterstitialPairSupport,
) -> ProductSource {
    let mut valence = vec![
        ProductRadial {
            l: 0,
            n: 0,
            spin: 0,
            samples: samples(0),
        },
        ProductRadial {
            l: 0,
            n: 1,
            spin: 0,
            samples: samples(1),
        },
    ];
    if include_p {
        valence.push(ProductRadial {
            l: 1,
            n: 0,
            spin: 0,
            samples: samples(3),
        });
    }
    let mut cores = Vec::new();
    if include_core {
        cores.push(ProductRadial {
            l: 0,
            n: 0,
            spin: 0,
            samples: samples(2),
        });
    }
    ProductSource::new(
        partition(),
        vec![SiteRadialSet {
            mesh: mesh(),
            valence,
            cores,
        }],
        q,
        support,
        Provenance::default(),
    )
    .unwrap()
}

fn source_vv_cv(include_core: bool, include_p: bool) -> ProductSource {
    let q = TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
    let lattice = cubic_lattice();
    source_with_support(
        include_core,
        include_p,
        q,
        convention_pair_support(q, &lattice),
    )
}

fn radial_id(kind: ProductOrbitalKind, l: u32, n: usize) -> ProductRadialId {
    ProductRadialId {
        site: 0,
        kind,
        l,
        n,
        spin: 0,
    }
}

fn gamma_interstitial_spec(lattice: &ReciprocalLattice) -> InterstitialPairSpec {
    InterstitialPairSpec {
        g_relative: g_vector(lattice, [0; 3]),
        amplitude: Complex64::new(1.0, 0.0),
    }
}

fn mixed_product(auxiliary: &CompiledAuxiliaryBasis) -> &MixedProductAuxiliary {
    auxiliary.mixed_product().expect("mixed-product auxiliary")
}

fn theta_i_oracle(
    auxiliary: &CompiledAuxiliaryBasis,
    spec: InterstitialPairSpec,
) -> Vec<Complex64> {
    let wrap = auxiliary.q.umklapp;
    mixed_product(auxiliary)
        .interstitial
        .waves
        .iter()
        .map(|wave| {
            let argument = std::array::from_fn(|axis| {
                InverseBohr(
                    spec.g_relative.cartesian[axis].get() + wrap.cartesian[axis].get()
                        - wave.g.cartesian[axis].get(),
                )
            });
            spec.amplitude
                * auxiliary
                    .partition
                    .interstitial
                    .coefficient(argument)
                    .unwrap()
        })
        .collect()
}

#[test]
fn valence_valence_unordered_count_is_independent_of_tol() {
    let source = source_vv_cv(false, false);
    let lattice = cubic_lattice();
    let g_cut = InverseBohr(0.8);
    let (raw, auxiliary) = spex_mixed_product_basis(&source, 0, g_cut, &lattice).unwrap();
    assert_eq!(raw.radial_product_count(), 3);
    assert_eq!(
        raw.interstitial_pair_support,
        source.interstitial_pair_support
    );
    assert!(!mixed_product(&auxiliary).interstitial.waves.is_empty());
    let spectrum = raw.spectrum(0, 0).unwrap().eigenvalues.clone();
    let _retained =
        apply_overlap_cutoff(&raw, &source, DEFAULT_TOLERANCE, 1.0, &lattice, g_cut).unwrap();
    assert_eq!(raw.spectrum(0, 0).unwrap().eigenvalues, spectrum);
}

#[test]
fn triangle_and_parity_forbid_odd_sum_for_two_s_functions() {
    let source = source_vv_cv(false, false);
    let (raw, _) =
        spex_mixed_product_basis(&source, 1, InverseBohr(0.5), &cubic_lattice()).unwrap();
    assert!(raw.spectrum(0, 1).is_none());
    assert!(raw.spectrum(0, 0).is_some());
}

#[test]
fn selected_core_valence_adds_cv_products() {
    let vv = source_vv_cv(false, false);
    let cv = source_vv_cv(true, false);
    let lattice = cubic_lattice();
    let (raw_vv, _) = spex_mixed_product_basis(&vv, 0, InverseBohr(0.5), &lattice).unwrap();
    let (raw_cv, _) = spex_mixed_product_basis(&cv, 0, InverseBohr(0.5), &lattice).unwrap();
    assert_eq!(
        raw_cv.radial_product_count(),
        raw_vv.radial_product_count() + 2
    );
    assert!(raw_cv.radial_products.iter().any(|product| {
        product.channel.left.kind == ProductOrbitalKind::Core
            && product.channel.right.kind == ProductOrbitalKind::Valence
    }));
}

#[test]
fn l0_constant_is_the_first_retained_mode() {
    let source = source_vv_cv(false, false);
    let (_, auxiliary) =
        spex_mixed_product_basis(&source, 0, InverseBohr(0.5), &cubic_lattice()).unwrap();
    let mode = &mixed_product(&auxiliary).sites[0].modes[0];
    assert_eq!(mode.l, 0);
    assert_eq!(mode.n, 0);
    let mesh = auxiliary.site_mesh(0).unwrap();
    let radius = mesh.last().get();
    let scale = (radius.powi(3) / 3.0).sqrt();
    for (sample, r) in mode.radial.iter().zip(mesh.radii()) {
        assert!((sample - r.get() / scale).abs() < 1.0e-12);
    }
}

#[test]
fn compiled_auxiliary_mesh_integrates_the_constant_mode() {
    let source = source_vv_cv(false, false);
    let (_, auxiliary) =
        spex_mixed_product_basis(&source, 0, InverseBohr(0.5), &cubic_lattice()).unwrap();
    let mesh = auxiliary.site_mesh(0).unwrap();
    let mode = auxiliary.mt_mode(0, 0, 0).unwrap();
    let integrand = mode
        .radial
        .iter()
        .map(|value| value * value)
        .collect::<Vec<_>>();
    let norm = mesh.integrate(&integrand).unwrap();
    let r2 = mesh
        .radii()
        .iter()
        .map(|radius| radius.get() * radius.get())
        .collect::<Vec<_>>();
    let volume = mesh.integrate(&r2).unwrap();
    let analytic = mesh.last().get().powi(3) / 3.0;
    assert!(
        (norm - volume / analytic).abs() < 1.0e-12,
        "compiled mesh and constant mode must pair: {norm} vs {}",
        volume / analytic
    );
}

#[test]
fn mt_flatten_is_site_l_m_then_n() {
    let source = source_vv_cv(false, true);
    let (_, auxiliary) =
        spex_mixed_product_basis(&source, 1, InverseBohr(0.5), &cubic_lattice()).unwrap();
    let l1: Vec<_> = mixed_product(&auxiliary).sites[0]
        .modes
        .iter()
        .filter(|mode| mode.l == 1)
        .collect();
    assert!(l1.len() >= 2);
    let regions = auxiliary.regions();
    let mt: Vec<_> = regions
        .into_iter()
        .filter_map(|region| match region {
            AuxiliaryRegion::MuffinTin { l: 1, m, n, .. } => Some((m, n)),
            _ => None,
        })
        .collect();
    let mut expected = Vec::new();
    let ns = {
        let mut values = l1.iter().map(|mode| mode.n).collect::<Vec<_>>();
        values.sort_unstable();
        values
    };
    for m in -1..=1 {
        for &n in &ns {
            expected.push((m, n));
        }
    }
    assert_eq!(mt, expected);
    for &(m, n) in &expected {
        let index = auxiliary.mt_index(0, 1, m, n).unwrap();
        assert!(matches!(
            auxiliary.regions()[index],
            AuxiliaryRegion::MuffinTin {
                site: 0,
                l: 1,
                m: mm,
                n: nn,
            } if mm == m && nn == n
        ));
    }
}

#[test]
fn nonzero_cutoff_matches_independent_retained_projector() {
    let source = source_vv_cv(false, false);
    let lattice = cubic_lattice();
    let g_cut = InverseBohr(0.5);
    let (raw, _) = spex_mixed_product_basis(&source, 0, g_cut, &lattice).unwrap();
    let spectrum = raw.spectrum(0, 0).unwrap();
    let products: Vec<_> = raw
        .radial_products
        .iter()
        .filter(|product| product.channel.coupled_l == 0)
        .cloned()
        .collect();
    let mesh = mesh();
    let radius = mesh.last().get();
    let constant_norm = (radius.powi(3) / 3.0).sqrt();
    let mut functions = Vec::new();
    for product in &products {
        let mut samples = product.samples.clone();
        let projection_integrand = mesh
            .radii()
            .iter()
            .zip(&samples)
            .map(|(r, sample)| sample * r.get())
            .collect::<Vec<_>>();
        let projection = mesh.integrate(&projection_integrand).unwrap() / constant_norm;
        for (sample, r) in samples.iter_mut().zip(mesh.radii()) {
            *sample -= projection * r.get() / constant_norm;
        }
        let norm_sq = mesh
            .integrate(&samples.iter().map(|v| v * v).collect::<Vec<_>>())
            .unwrap();
        let scale = norm_sq.max(0.0).sqrt();
        for sample in &mut samples {
            *sample /= scale;
        }
        functions.push(samples);
    }
    let n = functions.len();
    let independent = solve_real_symmetric(n, |row, column| {
        let integrand = functions[row]
            .iter()
            .zip(&functions[column])
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>();
        mesh.integrate(&integrand).unwrap()
    })
    .unwrap();
    for (left, right) in independent.eigenvalues.iter().zip(&spectrum.eigenvalues) {
        assert!((left - right).abs() < 1.0e-10);
    }
    let threshold = 0.05;
    let mut kept_fns = Vec::new();
    for (index, &eigenvalue) in independent.eigenvalues.iter().enumerate() {
        if eigenvalue > 0.0 && eigenvalue >= threshold {
            let scale = 1.0 / eigenvalue.sqrt();
            let mut radial = vec![0.0; functions[0].len()];
            for (basis, function) in functions.iter().enumerate() {
                let coefficient = independent.eigenvectors[basis + index * n] * scale;
                for (sample, value) in radial.iter_mut().zip(function) {
                    *sample += coefficient * value;
                }
            }
            kept_fns.push(radial);
        }
    }
    let auxiliary = apply_overlap_cutoff(&raw, &source, threshold, 1.0, &lattice, g_cut).unwrap();
    let compiled: Vec<Vec<f64>> = mixed_product(&auxiliary).sites[0]
        .modes
        .iter()
        .filter(|mode| mode.l == 0 && mode.n > 0)
        .map(|mode| mode.radial.clone())
        .collect();
    assert_eq!(compiled.len(), kept_fns.len());
    assert_eq!(
        mixed_product(&auxiliary).sites[0]
            .modes
            .iter()
            .filter(|mode| mode.l == 0)
            .count(),
        kept_fns.len() + 1
    );
    let k = kept_fns.len();
    assert!(k > 0);
    let mut gram = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            let integrand = kept_fns[i]
                .iter()
                .zip(&compiled[j])
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>();
            gram[i * k + j] = mesh.integrate(&integrand).unwrap();
        }
    }
    for i in 0..k {
        for j in 0..k {
            let mut overlap = 0.0;
            for p in 0..k {
                overlap += gram[i * k + p] * gram[j * k + p];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (overlap - expected).abs() < 1.0e-8,
                "retained projector mismatch at ({i},{j}): {overlap} vs {expected}"
            );
        }
    }
    assert!(matches!(
        mixed_product(&auxiliary).cutoff,
        Some(record) if record.value == threshold
    ));
}

#[test]
fn finite_q_interstitial_support_uses_q_plus_g() {
    let lattice = cubic_lattice();
    let q =
        TransferQ::from_cartesian([InverseBohr(0.15), InverseBohr(0.0), InverseBohr(0.0)]).unwrap();
    let g_cut = InverseBohr(0.9);
    let waves = interstitial_plane_waves(&lattice, q, g_cut).unwrap();
    let zero = TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
    let gamma = interstitial_plane_waves(&lattice, zero, g_cut).unwrap();
    assert_eq!(
        gamma.iter().map(|wave| wave.g.index).collect::<Vec<_>>(),
        lattice
            .enumerate(g_cut)
            .unwrap()
            .iter()
            .map(|g| g.index)
            .collect::<Vec<_>>()
    );
    for wave in &waves {
        assert!(wave.q_plus_g_norm.get() <= g_cut.get() + 1.0e-12);
    }
    assert_ne!(waves.len(), gamma.len());
}

#[test]
fn interstitial_pair_vertex_matches_independent_theta_i_oracle_including_umklapp() {
    let source = source_vv_cv(false, false);
    let lattice = cubic_lattice();
    let (raw, auxiliary) =
        spex_mixed_product_basis(&source, 0, InverseBohr(0.8), &lattice).unwrap();
    let spec = PairVertexSpec {
        muffin_tin: None,
        interstitial: Some(gamma_interstitial_spec(&lattice)),
    };
    let vertex = pair_vertex(&source, &raw, &auxiliary, spec).unwrap();
    assert_eq!(vertex.coefficients().len(), auxiliary.dimension());
    assert!(vertex.mt().iter().all(|value| value.norm() == 0.0));
    let expected = theta_i_oracle(&auxiliary, spec.interstitial.unwrap());
    assert_eq!(vertex.interstitial().len(), expected.len());
    for (got, want) in vertex.interstitial().iter().zip(&expected) {
        assert!((got - want).norm() < 1.0e-12);
    }
    assert!(expected.iter().any(|value| value.norm() > 1.0e-8));
    let wrap = lattice.enumerate(InverseBohr(1.0)).unwrap()[1];
    let folded = TransferQ::fold_by_reciprocal_vector(source.q.cartesian, wrap).unwrap();
    let folded_source = ProductSource::new(
        source.partition.clone(),
        source.radials.clone(),
        folded,
        source.interstitial_pair_support.with_q(folded).unwrap(),
        Provenance::default(),
    )
    .unwrap();
    let (raw_u, auxiliary_u) =
        spex_mixed_product_basis(&folded_source, 0, InverseBohr(0.8), &lattice).unwrap();
    let shifted = pair_vertex(&folded_source, &raw_u, &auxiliary_u, spec).unwrap();
    let expected_u = theta_i_oracle(&auxiliary_u, spec.interstitial.unwrap());
    assert_eq!(shifted.interstitial().len(), expected_u.len());
    for (got, want) in shifted.interstitial().iter().zip(&expected_u) {
        assert!((got - want).norm() < 1.0e-12);
    }
    assert_ne!(vertex.interstitial(), shifted.interstitial());
}

#[test]
fn muffin_tin_pair_vertex_carries_site_phase() {
    let source = source_vv_cv(false, false);
    let lattice = cubic_lattice();
    let (raw, auxiliary) =
        spex_mixed_product_basis(&source, 0, InverseBohr(0.6), &lattice).unwrap();
    let left = radial_id(ProductOrbitalKind::Valence, 0, 0);
    let spec = PairVertexSpec {
        muffin_tin: Some(MtPairSpec {
            left,
            left_m: 0,
            right: left,
            right_m: 0,
        }),
        interstitial: None,
    };
    let vertex = pair_vertex(&source, &raw, &auxiliary, spec).unwrap();
    assert_eq!(vertex.mt().len(), auxiliary.mt_dimension());
    assert!(matches!(vertex.pair(), OrbitalPair::MuffinTin { .. }));
    let sss = gaunt(0, 0, 0, 0, 0, 0);
    assert!(sss.abs() > 0.2);
    let q = TransferQ::from_cartesian([InverseBohr(0.3), InverseBohr(-0.1), InverseBohr(0.05)])
        .unwrap();
    let phased_source = ProductSource::new(
        source.partition.clone(),
        source.radials.clone(),
        q,
        source.interstitial_pair_support.with_q(q).unwrap(),
        Provenance::default(),
    )
    .unwrap();
    let (raw_q, auxiliary_q) =
        spex_mixed_product_basis(&phased_source, 0, InverseBohr(0.6), &lattice).unwrap();
    let phased = pair_vertex(&phased_source, &raw_q, &auxiliary_q, spec).unwrap();
    let expected_phase = site_translation_phase(q.cartesian, POSITION);
    let index = auxiliary_q.mt_index(0, 0, 0, 0).unwrap();
    if vertex.coefficients()[index].norm() > 1.0e-14 {
        let ratio = phased.coefficients()[index] / vertex.coefficients()[index];
        assert!((ratio - expected_phase).norm() < 1.0e-10);
    }
}

#[test]
fn dual_arm_pair_vertex_keeps_interstitial_identity() {
    let source = source_vv_cv(false, false);
    let lattice = cubic_lattice();
    let (raw, auxiliary) =
        spex_mixed_product_basis(&source, 0, InverseBohr(0.8), &lattice).unwrap();
    let left = radial_id(ProductOrbitalKind::Valence, 0, 0);
    let spec = PairVertexSpec {
        muffin_tin: Some(MtPairSpec {
            left,
            left_m: 0,
            right: left,
            right_m: 0,
        }),
        interstitial: Some(gamma_interstitial_spec(&lattice)),
    };
    let vertex = pair_vertex(&source, &raw, &auxiliary, spec).unwrap();
    assert!(matches!(
        vertex.pair(),
        OrbitalPair::Composite { interstitial, .. } if interstitial.index == [0, 0, 0]
    ));
    assert!(vertex.mt().iter().any(|value| value.norm() > 0.0));
    assert!(
        vertex
            .interstitial()
            .iter()
            .any(|value| value.norm() > 1.0e-8)
    );
}

#[test]
fn pair_vertex_rejects_mismatched_transfer_q() {
    let source = source_vv_cv(false, false);
    let lattice = cubic_lattice();
    let (raw, auxiliary) =
        spex_mixed_product_basis(&source, 0, InverseBohr(0.5), &lattice).unwrap();
    let q =
        TransferQ::from_cartesian([InverseBohr(0.2), InverseBohr(0.0), InverseBohr(0.0)]).unwrap();
    let other = ProductSource::new(
        source.partition.clone(),
        source.radials.clone(),
        q,
        source.interstitial_pair_support.with_q(q).unwrap(),
        Provenance::default(),
    )
    .unwrap();
    let spec = PairVertexSpec {
        muffin_tin: Some(MtPairSpec {
            left: radial_id(ProductOrbitalKind::Valence, 0, 0),
            left_m: 0,
            right: radial_id(ProductOrbitalKind::Valence, 0, 0),
            right_m: 0,
        }),
        interstitial: None,
    };
    assert!(matches!(
        pair_vertex(&other, &raw, &auxiliary, spec),
        Err(libmuffintin_mpb::MpbError::TransferQMismatch)
    ));
}

#[test]
fn pair_vertex_rejects_same_sites_with_different_cell_volume() {
    let source = source_vv_cv(false, false);
    let lattice = cubic_lattice();
    let (raw, auxiliary) =
        spex_mixed_product_basis(&source, 0, InverseBohr(0.8), &lattice).unwrap();
    let volume_source = ProductSource::new(
        ProductPartition::from_interstitial(
            InterstitialGeometry::new(
                VolumeBohr3(1024.0),
                vec![Sphere {
                    center: POSITION,
                    radius: Bohr(RADIUS),
                }],
            )
            .unwrap(),
        ),
        source.radials.clone(),
        source.q,
        source.interstitial_pair_support.clone(),
        Provenance::default(),
    )
    .unwrap();
    let spec = PairVertexSpec {
        muffin_tin: None,
        interstitial: Some(gamma_interstitial_spec(&lattice)),
    };
    assert!(matches!(
        pair_vertex(&volume_source, &raw, &auxiliary, spec),
        Err(libmuffintin_mpb::MpbError::PartitionMismatch)
    ));
}

#[test]
fn pair_vertex_rejects_permuted_or_relabelled_raw_pair_support() {
    let source = source_vv_cv(false, false);
    let lattice = cubic_lattice();
    let (raw, auxiliary) =
        spex_mixed_product_basis(&source, 0, InverseBohr(0.8), &lattice).unwrap();
    let spec = PairVertexSpec {
        muffin_tin: None,
        interstitial: Some(gamma_interstitial_spec(&lattice)),
    };
    let mut permuted = raw.clone();
    permuted.interstitial_pair_support.components.swap(0, 1);
    assert!(matches!(
        pair_vertex(&source, &permuted, &auxiliary, spec),
        Err(libmuffintin_mpb::MpbError::InterstitialPairSupportMismatch)
    ));
    let mut relabelled = raw.clone();
    relabelled.interstitial_pair_support.components[0].g_relative = g_vector(&lattice, [3, 0, 0]);
    assert!(matches!(
        pair_vertex(&source, &relabelled, &auxiliary, spec),
        Err(libmuffintin_mpb::MpbError::InterstitialPairSupportMismatch)
    ));
    let mut auxiliary_perm = auxiliary.clone();
    auxiliary_perm
        .mixed_product_mut()
        .unwrap()
        .interstitial
        .waves
        .swap(0, 1);
    assert!(matches!(
        pair_vertex(&source, &raw, &auxiliary_perm, spec),
        Err(libmuffintin_mpb::MpbError::Product(
            ProductError::AuxiliaryWaveOrder
        ))
    ));
}

#[test]
fn pair_vertex_rejects_mismatched_mesh_and_mode_length() {
    let source = source_vv_cv(false, false);
    let lattice = cubic_lattice();
    let (raw, mut auxiliary) =
        spex_mixed_product_basis(&source, 0, InverseBohr(0.5), &lattice).unwrap();
    auxiliary.mixed_product_mut().unwrap().sites[0].modes[0]
        .radial
        .pop();
    let spec = PairVertexSpec {
        muffin_tin: Some(MtPairSpec {
            left: radial_id(ProductOrbitalKind::Valence, 0, 0),
            left_m: 0,
            right: radial_id(ProductOrbitalKind::Valence, 0, 0),
            right_m: 0,
        }),
        interstitial: None,
    };
    assert!(matches!(
        pair_vertex(&source, &raw, &auxiliary, spec),
        Err(libmuffintin_mpb::MpbError::Product(
            ProductError::AuxiliaryModeLength { .. }
        ))
    ));
}

#[test]
fn pair_vertex_rejects_absent_mt_pair() {
    let source = source_vv_cv(true, false);
    let lattice = cubic_lattice();
    let (raw, auxiliary) =
        spex_mixed_product_basis(&source, 0, InverseBohr(0.5), &lattice).unwrap();
    let core = radial_id(ProductOrbitalKind::Core, 0, 0);
    let spec = PairVertexSpec {
        muffin_tin: Some(MtPairSpec {
            left: core,
            left_m: 0,
            right: core,
            right_m: 0,
        }),
        interstitial: None,
    };
    assert!(matches!(
        pair_vertex(&source, &raw, &auxiliary, spec),
        Err(libmuffintin_mpb::MpbError::UnknownMtPair { .. })
    ));
}

#[test]
fn raw_pair_support_survives_independent_auxiliary_g_cut() {
    let lattice = cubic_lattice();
    let q = TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
    let g0 = g_vector(&lattice, [0, 0, 0]);
    let g_out = g_vector(&lattice, [2, 0, 0]);
    let g_aux = g_vector(&lattice, [1, 0, 0]);
    let support = RawInterstitialPairSupport::from_components(
        q,
        vec![
            RawInterstitialPairComponent { g_relative: g0 },
            RawInterstitialPairComponent { g_relative: g_out },
        ],
    )
    .unwrap();
    let source = source_with_support(false, false, q, support);
    let g_cut = InverseBohr(0.8);
    let (raw, auxiliary) = spex_mixed_product_basis(&source, 0, g_cut, &lattice).unwrap();
    assert_eq!(
        raw.interstitial_pair_support,
        source.interstitial_pair_support
    );
    assert!(raw.interstitial_pair_support.find(&g_out).is_some());
    assert!(
        mixed_product(&auxiliary)
            .interstitial
            .waves
            .iter()
            .all(|wave| wave.g.index != [2, 0, 0])
    );
    assert!(
        mixed_product(&auxiliary)
            .interstitial
            .waves
            .iter()
            .any(|wave| wave.g.index == [1, 0, 0])
    );
    let outside = pair_vertex(
        &source,
        &raw,
        &auxiliary,
        PairVertexSpec {
            muffin_tin: None,
            interstitial: Some(InterstitialPairSpec {
                g_relative: g_out,
                amplitude: Complex64::new(1.0, 0.0),
            }),
        },
    )
    .unwrap();
    let expected = theta_i_oracle(
        &auxiliary,
        InterstitialPairSpec {
            g_relative: g_out,
            amplitude: Complex64::new(1.0, 0.0),
        },
    );
    for (got, want) in outside.interstitial().iter().zip(&expected) {
        assert!((got - want).norm() < 1.0e-12);
    }
    assert!(matches!(
        pair_vertex(
            &source,
            &raw,
            &auxiliary,
            PairVertexSpec {
                muffin_tin: None,
                interstitial: Some(InterstitialPairSpec {
                    g_relative: g_aux,
                    amplitude: Complex64::new(1.0, 0.0),
                }),
            }
        ),
        Err(libmuffintin_mpb::MpbError::UnknownInterstitialPair { g }) if g == [1, 0, 0]
    ));
}
