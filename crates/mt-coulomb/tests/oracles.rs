//! Independent MT-PW, PW-PW, Gamma, and idum-parity oracles transcribed from
//! SPEX `coulombmatrix.f`, not from `assemble.rs`.

mod common;

use muffintin_core::{
    Bohr, GVector, InterstitialGeometry, InverseBohr, ReciprocalLattice, VolumeBohr3,
    complex_spherical_harmonics, lm_index,
};
use muffintin_coulomb::{
    CoulombRequest, StructureConstants, assemble_coulomb, bessel_overlap, bessel_weinert_integral,
    multipole_moment, second_moment, sphbessel_pw_integral, spherical_bessel_moment,
    structure_constants, weinert_gmat,
};
use muffintin_envelope::Provenance;
use muffintin_prodbasis::{
    AuxiliaryInterstitialSupport, AuxiliaryInterstitialWave, AuxiliaryPartition,
    AuxiliaryRepresentation, CompiledAuxiliaryBasis, MixedProductAuxiliary, SiteAuxiliaryBlock,
    TransferQ,
};
use num_complex::Complex64;
use std::f64::consts::PI;

const RECONSTRUCTION: f64 = 1.0e-8;

fn empty_partition() -> AuxiliaryPartition {
    AuxiliaryPartition::from_interstitial(
        InterstitialGeometry::new(VolumeBohr3(common::LATTICE.powi(3)), Vec::new()).unwrap(),
    )
}

fn wave(lattice: &ReciprocalLattice, q: TransferQ, index: [i32; 3]) -> AuxiliaryInterstitialWave {
    let cartesian = lattice.cartesian(index);
    let norm = InverseBohr(
        cartesian
            .iter()
            .map(|c| c.get().powi(2))
            .sum::<f64>()
            .sqrt(),
    );
    let g = GVector {
        index,
        cartesian,
        norm,
    };
    let q_plus_g: [InverseBohr; 3] =
        std::array::from_fn(|axis| InverseBohr(q.cartesian[axis].get() + cartesian[axis].get()));
    let q_plus_g_norm = InverseBohr(q_plus_g.iter().map(|c| c.get().powi(2)).sum::<f64>().sqrt());
    AuxiliaryInterstitialWave {
        g,
        q_plus_g,
        q_plus_g_norm,
    }
}

fn sorted_waves(
    lattice: &ReciprocalLattice,
    q: TransferQ,
    indices: &[[i32; 3]],
) -> Vec<AuxiliaryInterstitialWave> {
    let mut waves: Vec<_> = indices
        .iter()
        .map(|index| wave(lattice, q, *index))
        .collect();
    waves.sort_by(|left, right| {
        left.g
            .norm
            .get()
            .total_cmp(&right.g.norm.get())
            .then_with(|| left.g.index.cmp(&right.g.index))
    });
    waves
}

fn empty_sphere_pw_auxiliary(q: TransferQ, indices: &[[i32; 3]]) -> CompiledAuxiliaryBasis {
    let lattice = common::cubic_lattice();
    CompiledAuxiliaryBasis {
        partition: empty_partition(),
        q,
        representation: AuxiliaryRepresentation::MixedProduct(MixedProductAuxiliary {
            sites: Vec::new(),
            interstitial: AuxiliaryInterstitialSupport {
                q,
                g_cut: InverseBohr(1.6),
                waves: sorted_waves(&lattice, q, indices),
            },
            cutoff: None,
        }),
        provenance: Provenance::default(),
    }
}

fn one_sphere_pw_auxiliary(q: TransferQ, indices: &[[i32; 3]]) -> CompiledAuxiliaryBasis {
    let lattice = common::cubic_lattice();
    CompiledAuxiliaryBasis {
        partition: common::partition(),
        q,
        representation: AuxiliaryRepresentation::MixedProduct(MixedProductAuxiliary {
            sites: vec![SiteAuxiliaryBlock {
                site: 0,
                mesh: common::mesh(),
                modes: Vec::new(),
            }],
            interstitial: AuxiliaryInterstitialSupport {
                q,
                g_cut: InverseBohr(1.6),
                waves: sorted_waves(&lattice, q, indices),
            },
            cutoff: None,
        }),
        provenance: Provenance::default(),
    }
}

fn i_pow(l: u32) -> Complex64 {
    match l % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

fn parity(n: i32) -> f64 {
    if n.rem_euclid(2) == 0 { 1.0 } else { -1.0 }
}

fn phase(q: [InverseBohr; 3], r: [Bohr; 3]) -> Complex64 {
    let arg = q
        .iter()
        .zip(r)
        .map(|(component, coordinate)| component.get() * coordinate.get())
        .sum();
    Complex64::from_polar(1.0, arg)
}

fn sfac_table(max_n: usize) -> Vec<f64> {
    let mut table = vec![1.0; max_n + 1];
    for n in 1..=max_n {
        table[n] = table[n - 1] * (n as f64).sqrt();
    }
    table
}

fn structure_and_sfac(
    request: &CoulombRequest,
    auxiliary: &CompiledAuxiliaryBasis,
) -> (StructureConstants, Vec<f64>) {
    let structure = structure_constants(
        request.cell(),
        request.reciprocal(),
        &auxiliary.partition,
        auxiliary.q,
        request.lexp(),
    )
    .unwrap();
    (structure, sfac_table((4 * request.lexp() + 4) as usize))
}

fn pw_kernel(qnorm: f64) -> f64 {
    if qnorm <= 1.0e-12 {
        0.0
    } else {
        4.0 * PI / (qnorm * qnorm)
    }
}

fn sphere_form(gnorm: f64, radius: f64) -> f64 {
    if gnorm.abs() <= 1.0e-14 {
        4.0 * PI * radius.powi(3) / 3.0
    } else {
        let x = radius * gnorm;
        4.0 * PI * (x.sin() - x * x.cos()) / gnorm.powi(3)
    }
}

fn close(got: Complex64, expected: Complex64) -> bool {
    (got - expected).norm() <= RECONSTRUCTION * expected.norm().max(1.0)
}

fn idum_toggle(lexp: u32, reset_each_l: bool) -> Vec<(u32, i32, f64)> {
    let mut values = Vec::new();
    let mut idum: f64 = 1.0;
    for l in 0..=lexp {
        if reset_each_l {
            idum = 1.0;
        }
        for m in -(l as i32)..=l as i32 {
            values.push((l, m, idum));
            idum = -idum;
        }
    }
    values
}

#[test]
fn empty_sphere_pw_pw_is_diagonal_four_pi_over_qg_squared() {
    let q = common::transfer_q([0.5, 0.0, 0.0]);
    let auxiliary = empty_sphere_pw_auxiliary(q, &[[0, 0, 0], [1, 0, 0], [0, 1, 0]]);
    auxiliary.validate().unwrap();
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let operator = assemble_coulomb(&auxiliary, &request).unwrap();
    let payload = auxiliary.mixed_product().unwrap();
    for (i, wave_i) in payload.interstitial.waves.iter().enumerate() {
        for (j, _) in payload.interstitial.waves.iter().enumerate() {
            let got = operator.element(i, j).unwrap();
            if i == j {
                let qg = wave_i.q_plus_g_norm.get();
                let expected = Complex64::new(4.0 * PI / (qg * qg), 0.0);
                assert!(
                    close(got, expected),
                    "empty-sphere diagonal {i}: {got} vs {expected}"
                );
            } else {
                assert!(
                    got.norm() < 1.0e-10,
                    "empty-sphere off-diagonal {i},{j} = {got}"
                );
            }
        }
    }
}

#[test]
fn empty_sphere_gamma_omits_g0_head_and_keeps_finite_g_coulomb() {
    let q = TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
    let auxiliary = empty_sphere_pw_auxiliary(q, &[[0, 0, 0], [1, 0, 0], [0, 1, 0]]);
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let operator = assemble_coulomb(&auxiliary, &request).unwrap();
    let head = operator.gamma().expect("Gamma head metadata");
    assert!(head.spherical_average_subtracted);
    assert!((head.head_prefactor - 4.0 * PI).abs() < 1.0e-12);
    let payload = auxiliary.mixed_product().unwrap();
    let mut g0 = None;
    for (index, wave) in payload.interstitial.waves.iter().enumerate() {
        let got = operator.element(index, index).unwrap();
        if wave.g.index == [0, 0, 0] {
            g0 = Some(index);
            assert!(
                got.norm() < 1.0e-10,
                "Gamma G=0 body must omit 4π/|q|², got {got}"
            );
            assert!(head.constant_coefficients[index].norm() > 0.5);
        } else {
            let expected = Complex64::new(4.0 * PI / wave.q_plus_g_norm.get().powi(2), 0.0);
            assert!(
                close(got, expected),
                "Gamma empty-sphere finite G {got} vs {expected}"
            );
        }
    }
    let g0 = g0.expect("G=0 wave");
    for (index, wave) in payload.interstitial.waves.iter().enumerate() {
        if index == g0 || wave.g.index == [0, 0, 0] {
            continue;
        }
        assert!(operator.element(g0, index).unwrap().norm() < 1.0e-10);
    }
}

#[test]
fn mt_pw_and_pw_pw_idum_reset_at_each_l_not_continued() {
    let reset = idum_toggle(2, true);
    let continued = idum_toggle(2, false);
    assert_eq!(reset[0], (0, 0, 1.0));
    let l1_start = reset
        .iter()
        .position(|&(l, m, _)| l == 1 && m == -1)
        .unwrap();
    assert_eq!(reset[l1_start].2, 1.0, "SPEX resets idum at each L");
    assert_eq!(
        continued[l1_start].2, -1.0,
        "continued toggle from L=0 must differ so the oracle fails on the old placement"
    );
    for &(l, m, value) in &reset {
        assert!(
            (value - parity(l as i32 + m)).abs() < 1.0e-15,
            "reset idum must equal (-1)^{{L+M}}"
        );
    }
    let l2_start = reset
        .iter()
        .position(|&(l, m, _)| l == 2 && m == -2)
        .unwrap();
    assert_eq!(reset[l2_start].2, 1.0);
    // Each $2L+1$ is odd, so a continued toggle returns to $+1$ at even $L$.
    // The SPEX reset is visible at odd $L$ (L=1 start is $+1$ vs $-1$).
    assert_eq!(continued[l2_start].2, 1.0);
}

fn independent_mt_pw(
    request: &CoulombRequest,
    auxiliary: &CompiledAuxiliaryBasis,
    structure: &StructureConstants,
    sfac: &[f64],
    wave: &AuxiliaryInterstitialWave,
    reset_idum: bool,
) -> Complex64 {
    let payload = auxiliary.mixed_product().unwrap();
    let block = &payload.sites[0];
    let mode = block
        .modes
        .iter()
        .find(|mode| mode.l == 0 && mode.n == 0)
        .unwrap();
    let lexp = request.lexp();
    let mesh = &block.mesh;
    let pos = auxiliary.partition.sites()[0].position;
    let svol = auxiliary
        .partition
        .interstitial()
        .cell_volume()
        .get()
        .sqrt();
    let vol = auxiliary.partition.interstitial().cell_volume().get();
    let qnorm = wave.q_plus_g_norm.get();
    let y = complex_spherical_harmonics(lexp, wave.q_plus_g.map(InverseBohr::get));
    let l = 0u32;
    let m = 0i32;
    let lm = lm_index(l, m).unwrap();
    let moment = multipole_moment(l, mesh, &mode.radial).unwrap();
    let moment2 = second_moment(mesh, &mode.radial).unwrap();
    let olap = bessel_overlap(l, qnorm, mesh, &mode.radial).unwrap();
    let integral = bessel_weinert_integral(l, qnorm, mesh, &mode.radial).unwrap();
    let is_gamma = auxiliary
        .q
        .cartesian
        .iter()
        .all(|c| c.get().abs() <= 1.0e-12);
    let g_is_zero = wave.g.index == [0; 3];
    let mut csum = Complex64::default();
    for (site1, support1) in auxiliary.partition.sites().iter().enumerate() {
        let cexp = 4.0
            * PI
            * phase(wave.q_plus_g, support1.position)
            * phase(auxiliary.q.cartesian, pos).conj();
        // Old bug: `idum = 1` outside the L1 loop (continues across L). SPEX resets every L1.
        let mut idum_state: f64 = 1.0;
        for l1 in 0..=lexp {
            if reset_idum {
                idum_state = 1.0;
            }
            let sph = spherical_bessel_moment(l1, qnorm, support1.radius.get());
            let cdum = sph * i_pow(l1) * cexp;
            for m1 in -(l1 as i32)..=l1 as i32 {
                let idum = idum_state;
                let lm1 = lm_index(l1, m1).unwrap();
                let l2 = l + l1;
                let m2 = m - m1;
                let struc = if l2 <= 2 * lexp && m2.unsigned_abs() <= l2 {
                    structure.get(0, site1, l2, m2).unwrap()
                } else {
                    Complex64::default()
                };
                let g = weinert_gmat(l1, m1, l, m, sfac).unwrap();
                csum -= idum * g * y[lm1].conj() * cdum * struc;
                idum_state = -idum_state;
            }
        }
        if is_gamma && l <= 2 {
            let cexp_g = phase(wave.g.cartesian, support1.position)
                * weinert_gmat(l, m, 0, 0, sfac).unwrap()
                * 4.0
                * PI
                / vol;
            let radius = support1.radius.get();
            if g_is_zero {
                csum -= cexp_g * radius.powi(5) / 30.0;
            } else {
                let m0 = spherical_bessel_moment(0, qnorm, radius);
                let m2 = spherical_bessel_moment(2, qnorm, radius);
                csum -= cexp_g * (m0 * radius * radius - m2 * 2.0 / 3.0) / 10.0;
            }
        }
    }
    let cdum = (4.0 * PI).powi(2) * i_pow(l) * y[lm].conj() * phase(wave.g.cartesian, pos);
    if is_gamma && g_is_zero {
        -cdum * moment2 / 6.0 / svol
            + (-cdum / (2.0 * f64::from(l) + 1.0) * integral + csum * moment) / svol
    } else if qnorm <= 1.0e-12 {
        Complex64::default()
    } else {
        (cdum * olap / (qnorm * qnorm) - cdum / (2.0 * f64::from(l) + 1.0) * integral
            + csum * moment)
            / svol
    }
}

#[test]
fn mt_pw_matches_independent_2a_2b_2c_and_rejects_continued_idum() {
    // $q$ along $z$ so $Y_{10}(q+G)$ feeds the L1=1 idum-odd channel.
    let q = common::transfer_q([0.0, 0.0, 0.5]);
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let (_source, auxiliary) = common::mixed_product_auxiliary(q);
    let operator = assemble_coulomb(&auxiliary, &request).unwrap();
    let payload = auxiliary.mixed_product().unwrap();
    let mt_index = auxiliary.mt_index(0, 0, 0, 0).unwrap();
    let (structure, sfac) = structure_and_sfac(&request, &auxiliary);
    let mut saw_pw = false;
    let mut saw_idum_split = false;
    for (local, wave) in payload.interstitial.waves.iter().enumerate() {
        if wave.g.index == [0, 0, 0] {
            continue;
        }
        let got = operator
            .element(mt_index, auxiliary.mt_dimension() + local)
            .unwrap();
        let expected = independent_mt_pw(&request, &auxiliary, &structure, &sfac, wave, true);
        assert!(
            close(got, expected),
            "MT-PW 2a+2b+2c {got} vs independent {expected}"
        );
        let wrong = independent_mt_pw(&request, &auxiliary, &structure, &sfac, wave, false);
        if (expected - wrong).norm() > 1.0e-12 {
            saw_idum_split = true;
            assert!(
                (got - wrong).norm() > (got - expected).norm(),
                "continued-idum oracle must miss the assembled MT-PW element"
            );
        }
        saw_pw = true;
    }
    assert!(saw_pw, "need a finite-G PW channel");
    assert!(
        saw_idum_split,
        "correct vs continued idum must differ on this fixture"
    );
}

fn independent_pw_pw(
    request: &CoulombRequest,
    auxiliary: &CompiledAuxiliaryBasis,
    structure: &StructureConstants,
    sfac: &[f64],
    wave1: &AuxiliaryInterstitialWave,
    wave2: &AuxiliaryInterstitialWave,
    reset_idum: bool,
) -> Complex64 {
    let lexp = request.lexp();
    let vol = auxiliary.partition.interstitial().cell_volume().get();
    let q1n = wave1.q_plus_g_norm.get();
    let q2n = wave2.q_plus_g_norm.get();
    let v1 = pw_kernel(q1n);
    let v2 = pw_kernel(q2n);
    let gdiff: [InverseBohr; 3] = std::array::from_fn(|axis| {
        InverseBohr(wave2.g.cartesian[axis].get() - wave1.g.cartesian[axis].get())
    });
    let gnorm = gdiff.iter().map(|c| c.get().powi(2)).sum::<f64>().sqrt();
    let mut cint = Complex64::default();
    for site in auxiliary.partition.sites() {
        let form = sphere_form(gnorm, site.radius.get());
        cint += if gnorm <= 1.0e-14 {
            Complex64::new(form, 0.0)
        } else {
            form * phase(gdiff, site.position)
        };
    }
    let is_gamma = auxiliary
        .q
        .cartesian
        .iter()
        .all(|c| c.get().abs() <= 1.0e-12);
    let g1_zero = wave1.g.index == [0; 3];
    let g2_zero = wave2.g.index == [0; 3];
    let mut value = Complex64::default();
    if is_gamma {
        if !g1_zero {
            value -= cint * v1 / vol;
        }
        if !g2_zero {
            value -= cint * v2 / vol;
        }
        if wave1.g.index == wave2.g.index && !g2_zero {
            value += v2;
        }
    } else {
        value -= cint * (v1 + v2) / vol;
        if wave1.g.index == wave2.g.index {
            value += v2;
        }
    }

    let y1 = complex_spherical_harmonics(lexp, wave1.q_plus_g.map(InverseBohr::get));
    let y2 = complex_spherical_harmonics(lexp, wave2.q_plus_g.map(InverseBohr::get));
    let mut csum = Complex64::default();
    for (ic2, site2) in auxiliary.partition.sites().iter().enumerate() {
        let cexp2 = phase(wave2.q_plus_g, site2.position);
        // Old bug: `idum = 1` outside the L2 loop (continues across L). SPEX resets every L2.
        let mut idum_state: f64 = 1.0;
        for l2 in 0..=lexp {
            if reset_idum {
                idum_state = 1.0;
            }
            let sph2 = spherical_bessel_moment(l2, q2n, site2.radius.get());
            for m2 in -(l2 as i32)..=l2 as i32 {
                let idum = idum_state;
                let lm2 = lm_index(l2, m2).unwrap();
                let cdum = idum * sph2 * cexp2 * 4.0 * PI * i_pow(l2) * y2[lm2].conj();
                if cdum.norm() > 0.0 {
                    for (ic1, site1) in auxiliary.partition.sites().iter().enumerate() {
                        for l1 in 0..=lexp {
                            let sph1 = spherical_bessel_moment(l1, q1n, site1.radius.get());
                            for m1 in -(l1 as i32)..=l1 as i32 {
                                let lm1 = lm_index(l1, m1).unwrap();
                                let l = l1 + l2;
                                let m = m1 - m2;
                                if l > 2 * lexp || m.unsigned_abs() > l {
                                    continue;
                                }
                                let g = weinert_gmat(l1, m1, l2, m2, sfac).unwrap();
                                let struc = structure.get(ic1, ic2, l, m).unwrap();
                                let left = phase(wave1.q_plus_g, site1.position).conj()
                                    * sph1
                                    * 4.0
                                    * PI
                                    * i_pow(l1).conj()
                                    * y1[lm1];
                                csum += left * cdum * g * struc;
                            }
                        }
                    }
                }
                idum_state = -idum_state;
            }
        }
    }
    value += csum / vol;

    let mut cdum = Complex64::default();
    for l in 0..=lexp {
        let mut cdum1 = Complex64::default();
        for site in auxiliary.partition.sites() {
            cdum1 += phase(gdiff, site.position)
                * sphbessel_pw_integral(l, q1n, q2n, site.radius.get())
                / (2.0 * f64::from(l) + 1.0);
        }
        for m in -(l as i32)..=l as i32 {
            let lm = lm_index(l, m).unwrap();
            cdum += cdum1 * y1[lm] * y2[lm].conj();
        }
    }
    value += (4.0 * PI).powi(3) * cdum / vol;

    if is_gamma {
        let g00 = weinert_gmat(0, 0, 0, 0, sfac).unwrap();
        let rdum = (4.0 * PI).powf(1.5) / vol.powi(2) * g00;
        if !g1_zero && !g2_zero {
            let q1v = wave1.q_plus_g.map(InverseBohr::get);
            let q2v = wave2.q_plus_g.map(InverseBohr::get);
            let rdum1 = (q1v[0] * q2v[0] + q1v[1] * q2v[1] + q1v[2] * q2v[2]) / (q1n * q2n);
            for site1 in auxiliary.partition.sites() {
                for site2 in auxiliary.partition.sites() {
                    let cdum = phase(
                        [
                            InverseBohr(-wave1.g.cartesian[0].get()),
                            InverseBohr(-wave1.g.cartesian[1].get()),
                            InverseBohr(-wave1.g.cartesian[2].get()),
                        ],
                        site1.position,
                    ) * phase(wave2.g.cartesian, site2.position);
                    let m0a = spherical_bessel_moment(0, q1n, site1.radius.get());
                    let m1a = spherical_bessel_moment(1, q1n, site1.radius.get());
                    let m2a = spherical_bessel_moment(2, q1n, site1.radius.get());
                    let m0b = spherical_bessel_moment(0, q2n, site2.radius.get());
                    let m1b = spherical_bessel_moment(1, q2n, site2.radius.get());
                    let m2b = spherical_bessel_moment(2, q2n, site2.radius.get());
                    value += rdum
                        * cdum
                        * (-m1a * m1b * rdum1 / 3.0 - m0a * m2b / 6.0 - m2a * m0b / 6.0
                            + m0a * m1b / q2n / 2.0
                            + m1a * m0b / q1n / 2.0);
                }
            }
        } else if g1_zero && !g2_zero {
            for site1 in auxiliary.partition.sites() {
                let r1 = site1.radius.get();
                for site2 in auxiliary.partition.sites() {
                    let cdum = phase(wave2.g.cartesian, site2.position);
                    let m0b = spherical_bessel_moment(0, q2n, site2.radius.get());
                    let m1b = spherical_bessel_moment(1, q2n, site2.radius.get());
                    let m2b = spherical_bessel_moment(2, q2n, site2.radius.get());
                    value += rdum
                        * cdum
                        * r1.powi(3)
                        * (m0b / 30.0 * r1 * r1 - m2b / 18.0 + m1b / 6.0 / q2n);
                }
            }
        } else if !g1_zero && g2_zero {
            for site2 in auxiliary.partition.sites() {
                let r2 = site2.radius.get();
                for site1 in auxiliary.partition.sites() {
                    let cdum = phase(
                        [
                            InverseBohr(-wave1.g.cartesian[0].get()),
                            InverseBohr(-wave1.g.cartesian[1].get()),
                            InverseBohr(-wave1.g.cartesian[2].get()),
                        ],
                        site1.position,
                    );
                    let m0a = spherical_bessel_moment(0, q1n, site1.radius.get());
                    let m1a = spherical_bessel_moment(1, q1n, site1.radius.get());
                    let m2a = spherical_bessel_moment(2, q1n, site1.radius.get());
                    value += rdum
                        * cdum
                        * r2.powi(3)
                        * (m0a / 30.0 * r2 * r2 - m2a / 18.0 + m1a / 6.0 / q1n);
                }
            }
        } else {
            for site1 in auxiliary.partition.sites() {
                let r1 = site1.radius.get();
                for site2 in auxiliary.partition.sites() {
                    let r2 = site2.radius.get();
                    value += rdum * r1.powi(3) * r2.powi(3) * (r1 * r1 + r2 * r2) / 90.0;
                }
            }
        }
    }
    value
}

#[test]
fn one_sphere_pw_pw_matches_step_function_identity_and_idum_reset() {
    let q = common::transfer_q([0.5, 0.0, 0.0]);
    let auxiliary = one_sphere_pw_auxiliary(q, &[[0, 0, 0], [1, 0, 0], [0, 1, 0]]);
    auxiliary.validate().unwrap();
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let operator = assemble_coulomb(&auxiliary, &request).unwrap();
    let payload = auxiliary.mixed_product().unwrap();
    let (structure, sfac) = structure_and_sfac(&request, &auxiliary);
    let mut saw_idum_split = false;
    for (i, wave_i) in payload.interstitial.waves.iter().enumerate() {
        for (j, wave_j) in payload.interstitial.waves.iter().enumerate() {
            let got = operator.element(i, j).unwrap();
            let expected = independent_pw_pw(
                &request, &auxiliary, &structure, &sfac, wave_i, wave_j, true,
            );
            assert!(
                close(got, expected),
                "one-sphere PW-PW ({i},{j}) {got} vs independent {expected}"
            );
            let wrong = independent_pw_pw(
                &request, &auxiliary, &structure, &sfac, wave_i, wave_j, false,
            );
            if (expected - wrong).norm() > 1.0e-8 * expected.norm().max(1.0e-8) {
                saw_idum_split = true;
            }
        }
    }
    assert!(
        saw_idum_split,
        "PW-PW intersite continued idum must differ from SPEX reset"
    );
}

#[test]
fn gamma_pw_pw_zero_and_finite_g_match_independent_taylor() {
    let q = TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
    let auxiliary = one_sphere_pw_auxiliary(q, &[[0, 0, 0], [1, 0, 0]]);
    auxiliary.validate().unwrap();
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let operator = assemble_coulomb(&auxiliary, &request).unwrap();
    assert!(operator.gamma().is_some());
    let payload = auxiliary.mixed_product().unwrap();
    let residual = {
        let n = operator.dimension();
        let mut worst: f64 = 0.0;
        for i in 0..n {
            for j in 0..n {
                worst = worst.max(
                    (operator.element(i, j).unwrap() - operator.element(j, i).unwrap().conj())
                        .norm(),
                );
            }
        }
        worst
    };
    assert!(
        residual < 1.0e-10,
        "Gamma PW-PW Hermitian residual {residual}"
    );
    let (structure, sfac) = structure_and_sfac(&request, &auxiliary);
    for (i, wave_i) in payload.interstitial.waves.iter().enumerate() {
        for (j, wave_j) in payload.interstitial.waves.iter().enumerate() {
            let got = operator.element(i, j).unwrap();
            let expected = independent_pw_pw(
                &request, &auxiliary, &structure, &sfac, wave_i, wave_j, true,
            );
            assert!(
                close(got, expected),
                "Gamma PW-PW ({i},{j}) G1={:?} G2={:?}: {got} vs {expected}",
                wave_i.g.index,
                wave_j.g.index
            );
        }
    }
    let g0 = payload
        .interstitial
        .waves
        .iter()
        .position(|wave| wave.g.index == [0; 3])
        .unwrap();
    let body00 = operator.element(g0, g0).unwrap();
    assert!(body00.re.is_finite() && body00.im.abs() < 1.0e-10);
}
