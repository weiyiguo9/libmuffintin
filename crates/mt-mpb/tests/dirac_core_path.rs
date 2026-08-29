//! Dirac PP/QQ muffin-tin products and checked vertices.

use muffintin_auxiliary_ir::{
    AuxiliaryInterstitialSupport, AuxiliaryPartition, AuxiliaryRepresentation,
    CompiledAuxiliaryBasis, DiracChargeSector, DiracMtPairSpec, DiracProductSource, DiracRadial,
    DiracRadialId, DiracRadialSamples, DiracSiteRadialSet, MixedProductAuxiliary, MtAuxiliaryMode,
    ProductOrbitalKind, RawInterstitialPairSupport, SiteAuxiliaryBlock, TransferQ,
};
use muffintin_basis::Provenance;
use muffintin_core::{
    Bohr, ExponentialMesh, InterstitialGeometry, InverseBohr, Kappa, Lm, RelativisticChannel,
    Sphere, TwiceMu, VolumeBohr3, gaunt,
};
use muffintin_mpb::{
    DiracPairVertexAccumulator, dirac_mt_pair_vertex, untruncated_dirac_product_space,
};
use num_complex::Complex64;
use std::collections::BTreeSet;

fn mesh() -> ExponentialMesh {
    let first = 1.0e-5;
    let number = 31;
    let increment = (0.8_f64 / first).ln() / (number - 1) as f64;
    ExponentialMesh::new(Bohr(first), increment, number).unwrap()
}

fn partition() -> AuxiliaryPartition {
    AuxiliaryPartition::from_interstitial(
        InterstitialGeometry::new(
            VolumeBohr3(512.0),
            vec![Sphere {
                center: [Bohr(0.0); 3],
                radius: Bohr(0.8),
            }],
        )
        .unwrap(),
    )
}

fn q_gamma() -> TransferQ {
    TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap()
}

fn kappa(value: i32) -> Kappa {
    Kappa::new(value).unwrap()
}

fn twice_mu(value: i64) -> TwiceMu {
    TwiceMu::new(value).unwrap()
}

fn channel(raw_kappa: i32, raw_mu: i64) -> RelativisticChannel {
    RelativisticChannel::new(kappa(raw_kappa), twice_mu(raw_mu)).unwrap()
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
                _ => r * (1.0 - 0.35 * r) * (-1.7 * r).exp(),
            }
        })
        .collect::<Vec<_>>();
    let small = large.iter().map(|value| scale_q * *value).collect();
    DiracRadialSamples { large, small }
}

fn source(radials: Vec<DiracRadial>) -> DiracProductSource {
    let q = q_gamma();
    DiracProductSource::new(
        partition(),
        vec![DiracSiteRadialSet {
            mesh: mesh(),
            valence: radials,
            cores: Vec::new(),
        }],
        q,
        RawInterstitialPairSupport::empty(q),
        Provenance::default(),
    )
    .unwrap()
}

fn id(raw_kappa: i32, n: usize) -> DiracRadialId {
    DiracRadialId {
        site: 0,
        kind: ProductOrbitalKind::Valence,
        kappa: kappa(raw_kappa),
        n,
    }
}

fn spec(raw_kappa: i32, n: usize, raw_mu: i64) -> DiracMtPairSpec {
    DiracMtPairSpec {
        left: id(raw_kappa, n),
        left_twice_mu: twice_mu(raw_mu),
        right: id(raw_kappa, n),
        right_twice_mu: twice_mu(raw_mu),
    }
}

fn auxiliary(source: &DiracProductSource, angular: &[u32]) -> CompiledAuxiliaryBasis {
    let n = source.radials[0].mesh.len();
    let modes = angular
        .iter()
        .map(|&l| MtAuxiliaryMode {
            l,
            n: 0,
            radial: vec![1.0; n],
        })
        .collect();
    let auxiliary = CompiledAuxiliaryBasis {
        partition: source.partition.clone(),
        q: source.q,
        representation: AuxiliaryRepresentation::MixedProduct(MixedProductAuxiliary {
            sites: vec![SiteAuxiliaryBlock {
                site: 0,
                mesh: source.radials[0].mesh.clone(),
                modes,
            }],
            interstitial: AuxiliaryInterstitialSupport {
                q: source.q,
                g_cut: InverseBohr(0.0),
                waves: Vec::new(),
            },
            cutoff: None,
        }),
        provenance: Provenance::default(),
    };
    auxiliary.validate().unwrap();
    auxiliary
}

fn magnetic_phase(m: i32) -> f64 {
    if m.unsigned_abs() % 2 == 0 { 1.0 } else { -1.0 }
}

/// Independent $\langle\Omega|Y_{LM}|\Omega\rangle$ from terms, SPEX Gaunt,
/// bra conjugation, and ket conversion.
///
/// SPEX `gaunt` is $\int Y_{l_1 m_1}^* Y_{LM} Y_{l_3 m_3}^*$. The first
/// harmonic is therefore conjugated (bra). The ket is converted with
/// $Y_{lm}=(-1)^m Y_{l,-m}^*$. This does not call
/// [`muffintin_core::spinor_gaunt`].
fn independent_spinor_y(left: RelativisticChannel, field: Lm, right: RelativisticChannel) -> f64 {
    let mut value = 0.0;
    for left_term in left.spinor_harmonic_terms().into_iter().flatten() {
        for right_term in right.spinor_harmonic_terms().into_iter().flatten() {
            if left_term.spin != right_term.spin {
                continue;
            }
            let ket_phase = magnetic_phase(right_term.orbital.m);
            value += left_term.coefficient
                * right_term.coefficient
                * ket_phase
                * gaunt(
                    left_term.orbital.l,
                    field.l,
                    right_term.orbital.l,
                    left_term.orbital.m,
                    field.m,
                    -right_term.orbital.m,
                );
        }
    }
    value
}

/// Independent stored density coefficient $(-1)^M\langle\Omega|Y_{L,-M}|\Omega\rangle$.
fn independent_density_coefficient(
    left: RelativisticChannel,
    l: u32,
    m: i32,
    right: RelativisticChannel,
) -> f64 {
    magnetic_phase(m) * independent_spinor_y(left, Lm::new(l, -m).unwrap(), right)
}

/// Same-$\kappa$ angular path that still applies ket conversion and the
/// stored-$M$ magnetic phase. The only defect is using $\Omega_\kappa$
/// where QQ needs $\Omega_{-\kappa}$.
fn same_kappa_only_density(
    left: RelativisticChannel,
    l: u32,
    m: i32,
    right: RelativisticChannel,
) -> f64 {
    independent_density_coefficient(left, l, m, right)
}

/// Ket conversion omitted: third Gaunt magnetic index is $m$ not $-m$, and
/// the ket phase $(-1)^m$ is dropped. Bra conjugation is kept.
fn missing_ket_conversion(left: RelativisticChannel, field: Lm, right: RelativisticChannel) -> f64 {
    let mut value = 0.0;
    for left_term in left.spinor_harmonic_terms().into_iter().flatten() {
        for right_term in right.spinor_harmonic_terms().into_iter().flatten() {
            if left_term.spin != right_term.spin {
                continue;
            }
            value += left_term.coefficient
                * right_term.coefficient
                * gaunt(
                    left_term.orbital.l,
                    field.l,
                    right_term.orbital.l,
                    left_term.orbital.m,
                    field.m,
                    right_term.orbital.m,
                );
        }
    }
    value
}

fn overlap(source: &DiracProductSource, samples: &[f64], mode: &[f64]) -> f64 {
    let integrand = samples
        .iter()
        .zip(mode)
        .map(|(sample, mode)| sample * mode)
        .collect::<Vec<_>>();
    source.radials[0].mesh.integrate(&integrand).unwrap()
}

fn product_samples(
    raw: &muffintin_auxiliary_ir::DiracRawProductSpace,
    sector: DiracChargeSector,
    coupled_l: u32,
) -> Option<&[f64]> {
    raw.radial_products
        .iter()
        .find(|product| product.channel.sector == sector && product.channel.coupled_l == coupled_l)
        .map(|product| product.samples.as_slice())
}

fn mixed_mu_spec() -> DiracMtPairSpec {
    DiracMtPairSpec {
        left: id(-2, 0),
        left_twice_mu: twice_mu(3),
        right: id(-2, 0),
        right_twice_mu: twice_mu(1),
    }
}

#[test]
fn pp_omega_kappa_and_qq_omega_minus_kappa_coefficients_differ() {
    let source = source(vec![DiracRadial {
        kappa: kappa(-2),
        n: 0,
        samples: samples(0, 0.45),
    }]);
    let raw = untruncated_dirac_product_space(&source, 2).unwrap();
    let auxiliary = auxiliary(&source, &[2]);
    let spec = spec(-2, 0, 3);
    let left = channel(-2, 3);
    let field = Lm::new(2, 0).unwrap();
    let g_pp = independent_spinor_y(left, field, left);
    let g_qq = independent_spinor_y(left.opposite_kappa(), field, left.opposite_kappa());
    let g_same_kappa_only = same_kappa_only_density(left, 2, 0, left);
    let g_missing_ket = missing_ket_conversion(left, field, left);
    let g_missing_ket_qq =
        missing_ket_conversion(left.opposite_kappa(), field, left.opposite_kappa());
    assert!(g_pp.abs() > 1.0e-8 && g_qq.abs() > 1.0e-8);
    assert!(
        (g_same_kappa_only - g_missing_ket).abs() > 1.0e-8,
        "same-kappa-only with ket conversion must differ from missing ket conversion: {g_same_kappa_only} vs {g_missing_ket}"
    );
    assert!(
        (g_missing_ket - g_qq).abs() > 1.0e-8,
        "ket-phase negative control must differ from QQ: {g_missing_ket} vs {g_qq}"
    );
    assert!(
        (g_missing_ket_qq - g_qq).abs() > 1.0e-8,
        "ket-phase on Omega_-kappa must differ from QQ: {g_missing_ket_qq} vs {g_qq}"
    );

    let mut pp_only = DiracPairVertexAccumulator::new(&source, &raw, &auxiliary, spec).unwrap();
    pp_only.add_pp(Complex64::new(1.0, 0.0)).unwrap();
    let pp_vertex = pp_only.finish().unwrap();
    let mut qq_only = DiracPairVertexAccumulator::new(&source, &raw, &auxiliary, spec).unwrap();
    qq_only.add_qq(Complex64::new(1.0, 0.0)).unwrap();
    let qq_vertex = qq_only.finish().unwrap();
    let index = auxiliary.mt_index(0, 2, 0, 0).unwrap();
    let mode = vec![1.0; mesh().len()];
    let i_pp = overlap(
        &source,
        product_samples(&raw, DiracChargeSector::LargeLarge, 2).unwrap(),
        &mode,
    );
    let i_qq = overlap(
        &source,
        product_samples(&raw, DiracChargeSector::SmallSmall, 2).unwrap(),
        &mode,
    );
    assert!(
        (pp_vertex.coefficients()[index] - Complex64::new(g_pp * i_pp, 0.0)).norm() < 1.0e-10,
        "PP coefficient {} vs independent {}",
        pp_vertex.coefficients()[index],
        g_pp * i_pp
    );
    assert!(
        (qq_vertex.coefficients()[index] - Complex64::new(g_qq * i_qq, 0.0)).norm() < 1.0e-10,
        "QQ coefficient {} vs independent {}",
        qq_vertex.coefficients()[index],
        g_qq * i_qq
    );
    assert!(
        (pp_vertex.coefficients()[index] - qq_vertex.coefficients()[index]).norm() > 1.0e-8,
        "PP and QQ vertex coefficients must differ"
    );
}

#[test]
fn pp_and_qq_accumulate_separately_and_merged_angular_differs() {
    let source = source(vec![DiracRadial {
        kappa: kappa(-2),
        n: 0,
        samples: samples(0, 0.45),
    }]);
    let raw = untruncated_dirac_product_space(&source, 2).unwrap();
    let auxiliary = auxiliary(&source, &[2]);
    let spec = spec(-2, 0, 3);
    let left = channel(-2, 3);
    let field = Lm::new(2, 0).unwrap();
    let g_pp = independent_spinor_y(left, field, left);
    let g_qq = independent_spinor_y(left.opposite_kappa(), field, left.opposite_kappa());
    let mode = vec![1.0; mesh().len()];
    let i_pp = overlap(
        &source,
        product_samples(&raw, DiracChargeSector::LargeLarge, 2).unwrap(),
        &mode,
    );
    let i_qq = overlap(
        &source,
        product_samples(&raw, DiracChargeSector::SmallSmall, 2).unwrap(),
        &mode,
    );
    let both = dirac_mt_pair_vertex(&source, &raw, &auxiliary, spec).unwrap();
    let mut dropped = DiracPairVertexAccumulator::new(&source, &raw, &auxiliary, spec).unwrap();
    dropped.add_pp(Complex64::new(1.0, 0.0)).unwrap();
    let dropped = dropped.finish().unwrap();
    let index = auxiliary.mt_index(0, 2, 0, 0).unwrap();
    let separate = g_pp * i_pp + g_qq * i_qq;
    let merged = missing_ket_conversion(left, field, left) * (i_pp + i_qq);
    assert!((both.coefficients()[index] - Complex64::new(separate, 0.0)).norm() < 1.0e-10);
    assert!((dropped.coefficients()[index] - Complex64::new(g_pp * i_pp, 0.0)).norm() < 1.0e-10);
    assert!((both.coefficients()[index] - dropped.coefficients()[index]).norm() > 1.0e-10);
    assert!(
        (separate - merged).abs() > 1.0e-10,
        "merged (PP+QQ)*one-angular-factor must differ: separate={separate} merged={merged}"
    );
}

#[test]
fn same_orbital_diagonal_scalar_charge_is_real_and_nonnegative() {
    let source = source(vec![DiracRadial {
        kappa: kappa(-1),
        n: 0,
        samples: samples(0, 0.3),
    }]);
    let raw = untruncated_dirac_product_space(&source, 2).unwrap();
    assert!(raw.radial_products.iter().all(|product| {
        matches!(
            product.channel.sector,
            DiracChargeSector::LargeLarge | DiracChargeSector::SmallSmall
        )
    }));
    let auxiliary = auxiliary(&source, &[0]);
    let vertex = dirac_mt_pair_vertex(&source, &raw, &auxiliary, spec(-1, 0, 1)).unwrap();
    let index = auxiliary.mt_index(0, 0, 0, 0).unwrap();
    let value = vertex.coefficients()[index];
    assert!(
        value.im.abs() < 1.0e-12,
        "diagonal charge must be real: {value}"
    );
    assert!(
        value.re >= 0.0,
        "diagonal charge must be nonnegative: {value}"
    );
}

#[test]
fn dirac_context_rejects_auxiliary_partition_mismatch() {
    let source = source(vec![DiracRadial {
        kappa: kappa(-1),
        n: 0,
        samples: samples(0, 0.3),
    }]);
    let raw = untruncated_dirac_product_space(&source, 0).unwrap();
    let mut auxiliary = auxiliary(&source, &[0]);
    auxiliary.partition = AuxiliaryPartition::from_interstitial(
        InterstitialGeometry::new(
            VolumeBohr3(1000.0),
            vec![Sphere {
                center: [Bohr(0.0); 3],
                radius: Bohr(0.8),
            }],
        )
        .unwrap(),
    );
    let error = DiracPairVertexAccumulator::new(&source, &raw, &auxiliary, spec(-1, 0, 1));
    assert!(matches!(
        error,
        Err(muffintin_mpb::MpbError::PartitionMismatch)
    ));
}

#[test]
fn scaling_physical_q_suppresses_qq_relative_to_pp() {
    let full = source(vec![DiracRadial {
        kappa: kappa(-1),
        n: 0,
        samples: samples(0, 0.4),
    }]);
    let small = source(vec![DiracRadial {
        kappa: kappa(-1),
        n: 0,
        samples: samples(0, 0.04),
    }]);
    let raw_full = untruncated_dirac_product_space(&full, 0).unwrap();
    let raw_small = untruncated_dirac_product_space(&small, 0).unwrap();
    let pp_full = product_samples(&raw_full, DiracChargeSector::LargeLarge, 0).unwrap();
    let qq_full = product_samples(&raw_full, DiracChargeSector::SmallSmall, 0).unwrap();
    let pp_small = product_samples(&raw_small, DiracChargeSector::LargeLarge, 0).unwrap();
    let qq_small = product_samples(&raw_small, DiracChargeSector::SmallSmall, 0).unwrap();
    let ratio_full = qq_full
        .iter()
        .zip(pp_full)
        .map(|(qq, pp)| qq.abs() / pp.abs().max(1.0e-18))
        .fold(0.0_f64, f64::max);
    let ratio_small = qq_small
        .iter()
        .zip(pp_small)
        .map(|(qq, pp)| qq.abs() / pp.abs().max(1.0e-18))
        .fold(0.0_f64, f64::max);
    assert!(
        ratio_small < 0.05 * ratio_full,
        "scaling Q by 0.1 must suppress QQ/PP: small={ratio_small} full={ratio_full}"
    );
}

#[test]
fn mixed_mu_nonzero_m_uses_density_coefficient_not_ylm_slot() {
    let source = source(vec![DiracRadial {
        kappa: kappa(-2),
        n: 0,
        samples: samples(0, 0.45),
    }]);
    let raw = untruncated_dirac_product_space(&source, 2).unwrap();
    let auxiliary = auxiliary(&source, &[2]);
    let spec = mixed_mu_spec();
    let left = channel(-2, 3);
    let right = channel(-2, 1);
    let l = 2;
    let stored_m = -1;
    let g_pp = independent_density_coefficient(left, l, stored_m, right);
    let g_qq =
        independent_density_coefficient(left.opposite_kappa(), l, stored_m, right.opposite_kappa());
    let g_old_pp = independent_spinor_y(left, Lm::new(l, stored_m).unwrap(), right);
    let g_old_qq = independent_spinor_y(
        left.opposite_kappa(),
        Lm::new(l, stored_m).unwrap(),
        right.opposite_kappa(),
    );
    let g_same_kappa_only = same_kappa_only_density(left, l, stored_m, right);
    let g_missing_ket_qq = magnetic_phase(stored_m)
        * missing_ket_conversion(
            left.opposite_kappa(),
            Lm::new(l, -stored_m).unwrap(),
            right.opposite_kappa(),
        );
    assert!(
        g_pp.abs() > 1.0e-8 && g_qq.abs() > 1.0e-8,
        "PP and QQ density coefficients must be nonzero at M={stored_m}: pp={g_pp} qq={g_qq}"
    );
    assert!(
        (g_old_pp - g_pp).abs() > 1.0e-8 && (g_old_qq - g_qq).abs() > 1.0e-8,
        "old Y_{{L,M}} into slot M must fail: old_pp={g_old_pp} pp={g_pp} old_qq={g_old_qq} qq={g_qq}"
    );
    assert!(
        ((-g_pp) - g_pp).abs() > 1.0e-8,
        "opposite sign must differ from the density coefficient"
    );
    assert!(
        (g_same_kappa_only - g_missing_ket_qq).abs() > 1.0e-8,
        "same-kappa-only with ket conversion must differ from QQ missing-ket: {g_same_kappa_only} vs {g_missing_ket_qq}"
    );

    let mut pp_only = DiracPairVertexAccumulator::new(&source, &raw, &auxiliary, spec).unwrap();
    pp_only.add_pp(Complex64::new(1.0, 0.0)).unwrap();
    let pp_vertex = pp_only.finish().unwrap();
    let mut qq_only = DiracPairVertexAccumulator::new(&source, &raw, &auxiliary, spec).unwrap();
    qq_only.add_qq(Complex64::new(1.0, 0.0)).unwrap();
    let qq_vertex = qq_only.finish().unwrap();
    let index = auxiliary.mt_index(0, l, stored_m, 0).unwrap();
    let opposite_index = auxiliary.mt_index(0, l, -stored_m, 0).unwrap();
    let mode = vec![1.0; mesh().len()];
    let i_pp = overlap(
        &source,
        product_samples(&raw, DiracChargeSector::LargeLarge, l).unwrap(),
        &mode,
    );
    let i_qq = overlap(
        &source,
        product_samples(&raw, DiracChargeSector::SmallSmall, l).unwrap(),
        &mode,
    );
    let expected_pp = Complex64::new(g_pp * i_pp, 0.0);
    let expected_qq = Complex64::new(g_qq * i_qq, 0.0);
    assert!(
        (pp_vertex.coefficients()[index] - expected_pp).norm() < 1.0e-10,
        "PP M={stored_m} {} vs independent {}",
        pp_vertex.coefficients()[index],
        expected_pp
    );
    assert!(
        (qq_vertex.coefficients()[index] - expected_qq).norm() < 1.0e-10,
        "QQ M={stored_m} {} vs independent {}",
        qq_vertex.coefficients()[index],
        expected_qq
    );
    assert!((pp_vertex.coefficients()[index] - qq_vertex.coefficients()[index]).norm() > 1.0e-8);
    assert!(
        (pp_vertex.coefficients()[index] - Complex64::new(g_old_pp * i_pp, 0.0)).norm() > 1.0e-8,
        "production must not match old opposite-M PP projection"
    );
    assert!(
        (qq_vertex.coefficients()[index] - Complex64::new(g_old_qq * i_qq, 0.0)).norm() > 1.0e-8,
        "production must not match old opposite-M QQ projection"
    );
    assert!(
        (pp_vertex.coefficients()[index] + expected_pp).norm() > 1.0e-8,
        "production must not match the opposite-sign PP coefficient"
    );
    assert!(
        (qq_vertex.coefficients()[index] + expected_qq).norm() > 1.0e-8,
        "production must not match the opposite-sign QQ coefficient"
    );
    assert!(
        pp_vertex.coefficients()[opposite_index].norm() < 1.0e-10,
        "old slot M=+1 must stay empty, got {}",
        pp_vertex.coefficients()[opposite_index]
    );
    assert!(
        qq_vertex.coefficients()[opposite_index].norm() < 1.0e-10,
        "old slot M=+1 must stay empty, got {}",
        qq_vertex.coefficients()[opposite_index]
    );
}

#[test]
fn coupled_channel_radial_index_restarts_in_each_site_l_block() {
    let q = q_gamma();
    let partition = AuxiliaryPartition::from_interstitial(
        InterstitialGeometry::new(
            VolumeBohr3(512.0),
            vec![
                Sphere {
                    center: [Bohr(0.0); 3],
                    radius: Bohr(0.8),
                },
                Sphere {
                    center: [Bohr(2.0), Bohr(0.0), Bohr(0.0)],
                    radius: Bohr(0.8),
                },
            ],
        )
        .unwrap(),
    );
    let site_set = |n0: u8, n1: u8| DiracSiteRadialSet {
        mesh: mesh(),
        valence: vec![
            DiracRadial {
                kappa: kappa(-2),
                n: 0,
                samples: samples(n0, 0.4),
            },
            DiracRadial {
                kappa: kappa(-2),
                n: 1,
                samples: samples(n1, 0.35),
            },
        ],
        cores: Vec::new(),
    };
    let source = DiracProductSource::new(
        partition,
        vec![site_set(0, 1), site_set(1, 0)],
        q,
        RawInterstitialPairSupport::empty(q),
        Provenance::default(),
    )
    .unwrap();
    let raw = untruncated_dirac_product_space(&source, 2).unwrap();
    let mut global_offset = 0usize;
    let mut seen_blocks = 0usize;
    for site in 0..2 {
        for l in 0..=2 {
            let n_products = raw
                .radial_products
                .iter()
                .filter(|product| {
                    product.channel.left.site == site && product.channel.coupled_l == l
                })
                .count();
            let indices = raw
                .channels
                .iter()
                .filter(|channel| channel.site == site && channel.l == l)
                .map(|channel| channel.radial_index)
                .collect::<BTreeSet<_>>();
            if n_products == 0 {
                assert!(
                    indices.is_empty(),
                    "empty (site {site}, L={l}) must not emit channels"
                );
                continue;
            }
            seen_blocks += 1;
            let expected = (0..n_products).collect::<BTreeSet<_>>();
            assert_eq!(
                indices, expected,
                "block (site {site}, L={l}) must use local 0..{n_products}"
            );
            if global_offset >= n_products {
                assert!(
                    !indices.contains(&global_offset),
                    "global offset {global_offset} leaked into site {site} L={l}"
                );
            }
            global_offset += n_products;
        }
    }
    assert!(
        seen_blocks >= 4,
        "need multiple site/L blocks, got {seen_blocks}"
    );
    assert!(global_offset > 0);
}
