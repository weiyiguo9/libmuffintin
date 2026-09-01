//! Dirac PP/QQ union overlap spectra, retained cutoff, and Bloch $\Theta_I$.

use muffintin_core::{
    Bohr, ExponentialMesh, GVector, InterstitialGeometry, InverseBohr, Kappa, ReciprocalLattice,
    Sphere, VolumeBohr3,
};
use muffintin_envelope::Provenance;
use muffintin_operators::solve_real_symmetric;
use muffintin_prodbasis::mpb::{
    DEFAULT_TOLERANCE, DiracBlochVertexAccumulator, MpbError, apply_dirac_overlap_cutoff,
    auxiliary_interstitial_support, untruncated_dirac_product_space,
};
use muffintin_prodbasis::{
    AuxiliaryPartition, AuxiliaryRegion, ChannelSpectrum, CompiledAuxiliaryBasis,
    DiracChargeSector, DiracProductSource, DiracRadial, DiracRadialId, DiracRadialNormalization,
    DiracRadialSamples, DiracSiteRadialSet, InterstitialPairSpec, MixedProductAuxiliary,
    OrbitalPair, ProductOrbitalKind, RawInterstitialPairComponent, RawInterstitialPairSupport,
    TransferQ,
};
use num_complex::Complex64;

const RADIUS: f64 = 0.8;
const POSITION: [Bohr; 3] = [Bohr(0.25), Bohr(0.0), Bohr(0.0)];

fn mesh() -> ExponentialMesh {
    let first = 1.0e-5;
    let number = 31;
    let increment = (RADIUS / first).ln() / (number - 1) as f64;
    ExponentialMesh::new(Bohr(first), increment, number).unwrap()
}

fn partition() -> AuxiliaryPartition {
    AuxiliaryPartition::from_interstitial(
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

fn kappa(value: i32) -> Kappa {
    Kappa::new(value).unwrap()
}

fn samples(kind: u8, scale_q: f64) -> DiracRadialSamples {
    let mesh = mesh();
    let large = mesh
        .radii()
        .iter()
        .map(|radius| {
            let r = radius.get();
            match kind {
                0 => r * (-2.0 * r).exp(),
                1 => r * (1.0 - 0.35 * r) * (-1.7 * r).exp(),
                _ => r * r * (-2.2 * r).exp(),
            }
        })
        .collect::<Vec<_>>();
    let small = large.iter().map(|value| scale_q * *value).collect();
    DiracRadialSamples { large, small }
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

fn pair_support(q: TransferQ, lattice: &ReciprocalLattice) -> RawInterstitialPairSupport {
    let labels = [[0, 0, 0], [1, 0, 0], [-1, 0, 0]];
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

fn union_source() -> DiracProductSource {
    let q = TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
    let lattice = cubic_lattice();
    DiracProductSource::new(
        partition(),
        vec![DiracSiteRadialSet {
            mesh: mesh(),
            valence: vec![
                DiracRadial {
                    kappa: kappa(-1),
                    n: 0,
                    samples: samples(0, 0.35),
                    normalization: DiracRadialNormalization::OnMesh,
                },
                DiracRadial {
                    kappa: kappa(-1),
                    n: 1,
                    samples: samples(1, 0.28),
                    normalization: DiracRadialNormalization::OnMesh,
                },
                DiracRadial {
                    kappa: kappa(1),
                    n: 0,
                    samples: samples(2, 0.40),
                    normalization: DiracRadialNormalization::OnMesh,
                },
            ],
            cores: Vec::new(),
        }],
        q,
        pair_support(q, &lattice),
        Provenance::default(),
    )
    .unwrap()
}

fn mixed_product(auxiliary: &CompiledAuxiliaryBasis) -> &MixedProductAuxiliary {
    auxiliary.mixed_product().expect("mixed-product auxiliary")
}

fn independent_channel_functions(l: u32, samples: &[&[f64]]) -> Vec<Vec<f64>> {
    let mesh = mesh();
    let radius = mesh.last().get();
    let constant_norm = (radius.powi(3) / 3.0).sqrt();
    let mut functions = Vec::new();
    for product in samples {
        let mut values = product.to_vec();
        if l == 0 {
            let projection_integrand = mesh
                .radii()
                .iter()
                .zip(&values)
                .map(|(r, sample)| sample * r.get())
                .collect::<Vec<_>>();
            let projection = mesh.integrate(&projection_integrand).unwrap() / constant_norm;
            for (sample, r) in values.iter_mut().zip(mesh.radii()) {
                *sample -= projection * r.get() / constant_norm;
            }
        }
        let norm_sq = mesh
            .integrate(&values.iter().map(|value| value * value).collect::<Vec<_>>())
            .unwrap();
        let scale = norm_sq.max(0.0).sqrt();
        if scale > 0.0 {
            for sample in &mut values {
                *sample /= scale;
            }
        }
        functions.push(values);
    }
    functions
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
                wave.g.cartesian[axis].get()
                    - wrap.cartesian[axis].get()
                    - spec.g_relative.cartesian[axis].get()
            });
            spec.amplitude * analytic_interstitial_coefficient(auxiliary, argument)
        })
        .collect()
}

fn analytic_interstitial_coefficient(auxiliary: &CompiledAuxiliaryBasis, k: [f64; 3]) -> Complex64 {
    let volume = auxiliary.partition.interstitial().cell_volume().get();
    let norm = k.iter().map(|value| value * value).sum::<f64>().sqrt();
    let mut value = if norm <= 1.0e-14 {
        Complex64::new(1.0, 0.0)
    } else {
        Complex64::default()
    };
    for sphere in auxiliary.partition.interstitial().spheres() {
        let radius = sphere.radius.get();
        let sphere_volume = 4.0 * std::f64::consts::PI * radius.powi(3) / 3.0;
        let radial = if norm <= 1.0e-14 {
            1.0
        } else {
            let x = norm * radius;
            3.0 * (x.sin() - x * x.cos()) / x.powi(3)
        };
        let phase = -k
            .iter()
            .zip(sphere.center)
            .map(|(component, coordinate)| component * coordinate.get())
            .sum::<f64>();
        value -= Complex64::from_polar(sphere_volume / volume * radial, phase);
    }
    value
}

#[test]
fn dirac_union_spectra_keep_separate_pp_and_qq_and_match_independent_lowdin() {
    let source = union_source();
    let lattice = cubic_lattice();
    let g_cut = InverseBohr(0.8);
    let raw = untruncated_dirac_product_space(&source, 1).unwrap();
    let l0: Vec<_> = raw
        .radial_products
        .iter()
        .filter(|product| product.channel.left.site == 0 && product.channel.coupled_l == 0)
        .collect();
    let n_pp = l0
        .iter()
        .filter(|product| product.channel.sector == DiracChargeSector::LargeLarge)
        .count();
    let n_qq = l0
        .iter()
        .filter(|product| product.channel.sector == DiracChargeSector::SmallSmall)
        .count();
    assert!(n_pp > 0 && n_qq > 0, "L=0 union must contain both sectors");
    assert_eq!(l0.len(), n_pp + n_qq);
    let spectrum = raw.spectrum(0, 0).expect("L=0 union spectrum");
    assert_eq!(spectrum.eigenvalues.len(), l0.len());
    assert!(
        l0[..n_pp]
            .iter()
            .all(|product| product.channel.sector == DiracChargeSector::LargeLarge)
    );
    assert!(
        l0[n_pp..]
            .iter()
            .all(|product| product.channel.sector == DiracChargeSector::SmallSmall)
    );
    let pp_ids = l0[..n_pp]
        .iter()
        .map(|product| (product.channel.left, product.channel.right))
        .collect::<Vec<_>>();
    let qq_ids = l0[n_pp..]
        .iter()
        .map(|product| (product.channel.left, product.channel.right))
        .collect::<Vec<_>>();
    let radial_id = |kappa_value, n| DiracRadialId {
        site: 0,
        kind: ProductOrbitalKind::Valence,
        kappa: kappa(kappa_value),
        n,
    };
    let s0 = radial_id(-1, 0);
    let s1 = radial_id(-1, 1);
    let p0 = radial_id(1, 0);
    let canonical_l0_pairs = vec![(s0, s0), (s0, s1), (s1, s1), (p0, p0)];
    assert_eq!(pp_ids, canonical_l0_pairs, "canonical PP pair order");
    assert_eq!(qq_ids, canonical_l0_pairs, "canonical QQ pair order");

    let functions = independent_channel_functions(
        0,
        &l0.iter()
            .map(|product| product.samples.as_slice())
            .collect::<Vec<_>>(),
    );
    let n = functions.len();
    let mesh = mesh();
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
    let auxiliary =
        apply_dirac_overlap_cutoff(&raw, &source, threshold, 1.0, &lattice, g_cut).unwrap();
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
    let cutoff = mixed_product(&auxiliary).cutoff.expect("TOL record");
    assert_eq!(cutoff.value, threshold);
    assert_eq!(cutoff.nspin_factor, 1.0);
}

#[test]
fn dirac_cutoff_keeps_eigenvalues_equal_to_threshold() {
    let source = union_source();
    let lattice = cubic_lattice();
    let raw = untruncated_dirac_product_space(&source, 0).unwrap();
    let spectrum = raw.spectrum(0, 0).unwrap();
    let threshold = spectrum
        .eigenvalues
        .iter()
        .copied()
        .filter(|&value| value > 0.0)
        .fold(f64::INFINITY, f64::min);
    assert!(threshold.is_finite() && threshold > 0.0);
    let auxiliary =
        apply_dirac_overlap_cutoff(&raw, &source, threshold, 1.0, &lattice, InverseBohr(0.5))
            .unwrap();
    let independent_kept = spectrum
        .eigenvalues
        .iter()
        .filter(|&&value| value > 0.0 && value >= threshold)
        .count();
    let retained = mixed_product(&auxiliary).sites[0]
        .modes
        .iter()
        .filter(|mode| mode.l == 0 && mode.n > 0)
        .count();
    assert_eq!(retained, independent_kept);
    assert!(independent_kept > 0);
}

#[test]
fn dirac_retained_layout_is_site_l_m_n_then_interstitial() {
    let source = union_source();
    let lattice = cubic_lattice();
    let raw = untruncated_dirac_product_space(&source, 1).unwrap();
    let auxiliary = apply_dirac_overlap_cutoff(
        &raw,
        &source,
        DEFAULT_TOLERANCE,
        1.0,
        &lattice,
        InverseBohr(0.8),
    )
    .unwrap();
    let l1: Vec<_> = mixed_product(&auxiliary).sites[0]
        .modes
        .iter()
        .filter(|mode| mode.l == 1)
        .collect();
    assert!(!l1.is_empty());
    let mt: Vec<_> = auxiliary
        .regions()
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
    let interstitial = auxiliary
        .regions()
        .into_iter()
        .skip_while(|region| matches!(region, AuxiliaryRegion::MuffinTin { .. }))
        .all(|region| matches!(region, AuxiliaryRegion::Interstitial { .. }));
    assert!(interstitial);
    assert!(!mixed_product(&auxiliary).interstitial.waves.is_empty());
}

#[test]
fn dirac_bloch_interstitial_matches_independent_theta_including_umklapp() {
    let source = union_source();
    let lattice = cubic_lattice();
    let raw = untruncated_dirac_product_space(&source, 0).unwrap();
    let auxiliary = apply_dirac_overlap_cutoff(
        &raw,
        &source,
        DEFAULT_TOLERANCE,
        1.0,
        &lattice,
        InverseBohr(0.8),
    )
    .unwrap();
    let spec = InterstitialPairSpec {
        g_relative: g_vector(&lattice, [0; 3]),
        amplitude: Complex64::new(1.0, 0.0),
    };
    let mut acc = DiracBlochVertexAccumulator::new(
        &source,
        &raw,
        &auxiliary,
        OrbitalPair::Bloch {
            k_index: 0,
            left: 0,
            right: 0,
        },
    )
    .unwrap();
    acc.add_interstitial(spec).unwrap();
    let vertex = acc.finish().unwrap();
    assert_eq!(
        vertex.pair(),
        OrbitalPair::Bloch {
            k_index: 0,
            left: 0,
            right: 0
        }
    );
    assert!(vertex.mt().iter().all(|value| value.norm() == 0.0));
    let expected = theta_i_oracle(&auxiliary, spec);
    assert_eq!(vertex.interstitial().len(), expected.len());
    for (got, want) in vertex.interstitial().iter().zip(&expected) {
        assert!((got - want).norm() < 1.0e-12);
    }
    assert!(expected.iter().any(|value| value.norm() > 1.0e-8));

    let wrap = lattice.enumerate(InverseBohr(1.0)).unwrap()[1];
    let folded = TransferQ::fold_by_reciprocal_vector(source.q.cartesian, wrap).unwrap();
    let folded_source = DiracProductSource::new(
        source.partition.clone(),
        source.radials.clone(),
        folded,
        source.interstitial_pair_support.with_q(folded).unwrap(),
        Provenance::default(),
    )
    .unwrap();
    let raw_u = untruncated_dirac_product_space(&folded_source, 0).unwrap();
    let auxiliary_u = apply_dirac_overlap_cutoff(
        &raw_u,
        &folded_source,
        DEFAULT_TOLERANCE,
        1.0,
        &lattice,
        InverseBohr(0.8),
    )
    .unwrap();
    let mut acc_u = DiracBlochVertexAccumulator::new(
        &folded_source,
        &raw_u,
        &auxiliary_u,
        OrbitalPair::Bloch {
            k_index: 0,
            left: 0,
            right: 0,
        },
    )
    .unwrap();
    acc_u.add_interstitial(spec).unwrap();
    let shifted = acc_u.finish().unwrap();
    let expected_u = theta_i_oracle(&auxiliary_u, spec);
    for (got, want) in shifted.interstitial().iter().zip(&expected_u) {
        assert!((got - want).norm() < 1.0e-12);
    }
    assert_ne!(auxiliary.q.umklapp.index, auxiliary_u.q.umklapp.index);
    let waves = auxiliary_interstitial_support(&lattice, folded, InverseBohr(0.8))
        .unwrap()
        .waves;
    assert_eq!(
        mixed_product(&auxiliary_u).interstitial.waves.len(),
        waves.len()
    );
}

#[test]
fn apply_dirac_overlap_cutoff_rejects_missing_extra_and_malformed_spectra() {
    let source = union_source();
    let lattice = cubic_lattice();
    let g_cut = InverseBohr(0.5);
    let raw = untruncated_dirac_product_space(&source, 1).unwrap();
    assert!(
        apply_dirac_overlap_cutoff(&raw, &source, DEFAULT_TOLERANCE, 1.0, &lattice, g_cut).is_ok()
    );

    let mut missing = raw.clone();
    missing.overlap_spectra.clear();
    let error =
        apply_dirac_overlap_cutoff(&missing, &source, DEFAULT_TOLERANCE, 1.0, &lattice, g_cut)
            .unwrap_err();
    assert!(
        matches!(
            error,
            MpbError::MissingDiracOverlapSpectrum { site: 0, l: 0 }
        ),
        "{error:?}"
    );

    let mut extra = raw.clone();
    extra.overlap_spectra.push(ChannelSpectrum {
        site: 0,
        l: 99,
        eigenvalues: vec![1.0],
        eigenvectors: vec![1.0],
    });
    let error =
        apply_dirac_overlap_cutoff(&extra, &source, DEFAULT_TOLERANCE, 1.0, &lattice, g_cut)
            .unwrap_err();
    assert!(
        matches!(
            error,
            MpbError::UnmatchedDiracOverlapSpectrum { site: 0, l: 99 }
        ),
        "{error:?}"
    );

    let mut malformed = raw.clone();
    let n = malformed.overlap_spectra[0].eigenvalues.len();
    assert!(n > 1);
    malformed.overlap_spectra[0].eigenvectors = vec![0.0; n];
    let error =
        apply_dirac_overlap_cutoff(&malformed, &source, DEFAULT_TOLERANCE, 1.0, &lattice, g_cut)
            .unwrap_err();
    assert!(
        matches!(
            error,
            MpbError::DiracOverlapSpectrumDimension {
                site: 0,
                l: 0,
                n_products,
                n_eigenvalues,
                n_eigenvectors,
            } if n_products == n && n_eigenvalues == n && n_eigenvectors == n
        ),
        "{error:?}"
    );
}

#[test]
fn dirac_bloch_accumulator_rejects_non_bloch_identity() {
    let source = union_source();
    let lattice = cubic_lattice();
    let raw = untruncated_dirac_product_space(&source, 0).unwrap();
    let auxiliary = apply_dirac_overlap_cutoff(
        &raw,
        &source,
        DEFAULT_TOLERANCE,
        1.0,
        &lattice,
        InverseBohr(0.8),
    )
    .unwrap();
    let pair = OrbitalPair::Interstitial {
        g_relative: g_vector(&lattice, [0; 3]),
    };
    let error = DiracBlochVertexAccumulator::new(&source, &raw, &auxiliary, pair).unwrap_err();
    assert!(
        matches!(error, MpbError::ExpectedDiracBlochPair),
        "{error:?}"
    );
}
