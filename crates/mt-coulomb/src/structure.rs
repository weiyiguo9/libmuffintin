//! Andersen/SPEX Ewald structure constants (`coulombmatrix.f:2287-2583`).

use crate::CoulombError;
use crate::math::{parity, plane_wave_phase, structure_lm};
use muffintin_auxiliary_ir::{ProductPartition, TransferQ};
use muffintin_core::{
    Bohr, InverseBohr, ReciprocalLattice, complex_spherical_harmonics, lm_count, lm_index,
};
use muffintin_grid::Cell;
use num_complex::Complex64;
use std::f64::consts::PI;

/// SPEX `CONVPARAM1` (`coulombmatrix.f:2298`).
const CONVPARAM1: f64 = 1.0e-10;
/// SPEX `CONVPARAM2 = CONVPARAM1*10` for $L\ge 8$.
const CONVPARAM2: f64 = CONVPARAM1 * 10.0;
/// Fixed SPEX `EWALD_SCALE` used by this implementation.
const EWALD_SCALE: f64 = 1.0;

/// Structure constants $S_{LM}(a,b;q)$ in `lm_index` order through $2 L_{\mathrm{exp}}$.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureConstants {
    l_max: u32,
    n_sites: usize,
    values: Vec<Complex64>,
}

impl StructureConstants {
    /// $S_{LM}(a,b)$.
    pub fn get(
        &self,
        site_a: usize,
        site_b: usize,
        l: u32,
        m: i32,
    ) -> Result<Complex64, CoulombError> {
        let lm = structure_lm(l, m, self.l_max)?;
        Ok(self.raw(site_a, site_b, lm))
    }

    fn raw(&self, site_a: usize, site_b: usize, lm: usize) -> Complex64 {
        let nlm = lm_count(self.l_max);
        self.values[(site_a * self.n_sites + site_b) * nlm + lm]
    }

    fn set(&mut self, site_a: usize, site_b: usize, lm: usize, value: Complex64) {
        let nlm = lm_count(self.l_max);
        self.values[(site_a * self.n_sites + site_b) * nlm + lm] = value;
    }

    /// Angular cutoff $2 L_{\mathrm{exp}}$.
    pub const fn l_max(&self) -> u32 {
        self.l_max
    }
}

/// SPEX `structureconstant` at one transfer $q$.
pub fn structure_constants(
    cell: &Cell,
    reciprocal: &ReciprocalLattice,
    partition: &ProductPartition,
    q: TransferQ,
    lexp: u32,
) -> Result<StructureConstants, CoulombError> {
    let l_max = 2 * lexp;
    let n_sites = partition.site_count();
    let nlm = lm_count(l_max);
    let mut values = vec![Complex64::default(); n_sites * n_sites * nlm];
    if n_sites == 0 {
        return Ok(StructureConstants {
            l_max,
            n_sites,
            values,
        });
    }
    let vol = cell.volume().get();
    let latcon = vol.cbrt();
    let ewald_scale = EWALD_SCALE;
    let scale = ewald_scale / latcon;
    let factor_real = 4.0 * PI / scale.powi(2) / latcon.powi(2);
    let real_cut = real_space_cutoff(scale, l_max, latcon, ewald_scale, factor_real)?;
    let recip_cut = reciprocal_cutoff(scale, vol, latcon, ewald_scale)?;
    let positions: Vec<[Bohr; 3]> = partition.sites().iter().map(|site| site.position).collect();

    let real_points = enumerate_direct(cell, Bohr(real_cut))?;
    let recip_points = reciprocal.enumerate(InverseBohr(recip_cut))?;

    for site_b in 0..n_sites {
        let site_a_max = if site_b == 0 { 1 } else { site_b };
        for site_a in 0..site_a_max {
            let delta = [
                positions[site_b][0].get() - positions[site_a][0].get(),
                positions[site_b][1].get() - positions[site_a][1].get(),
                positions[site_b][2].get() - positions[site_a][2].get(),
            ];
            let mut shlp = vec![Complex64::default(); nlm];
            for (t_index, t_cart) in &real_points {
                let ra = [
                    t_cart[0] + delta[0],
                    t_cart[1] + delta[1],
                    t_cart[2] + delta[2],
                ];
                let r = ra.iter().map(|c| c * c).sum::<f64>().sqrt();
                let a = scale * r;
                if a <= 1.0e-14 {
                    continue;
                }
                let harmonics = complex_spherical_harmonics(l_max, ra);
                let phase = plane_wave_phase(q.cartesian, index_to_bohr(*t_index, cell));
                for l in 0..=l_max {
                    let g = spex_real_g(l, a);
                    if g.abs() * factor_real * a * a < convpar(l, ewald_scale) {
                        continue;
                    }
                    for m in -(l as i32)..=l as i32 {
                        let lm = lm_index(l, m)?;
                        shlp[lm] += g * phase * harmonics[lm].conj();
                    }
                }
            }
            for lm in 0..nlm {
                values[(site_a * n_sites + site_b) * nlm + lm] = shlp[lm];
            }
        }
    }

    let pref = 4.0 * PI / (scale.powi(3) * vol);
    let q_cart = q.cartesian.map(InverseBohr::get);
    for g in &recip_points {
        let k = [
            q_cart[0] + g.cartesian[0].get(),
            q_cart[1] + g.cartesian[1].get(),
            q_cart[2] + g.cartesian[2].get(),
        ];
        let knorm = k.iter().map(|c| c * c).sum::<f64>().sqrt();
        let a = knorm / scale;
        let harmonics = complex_spherical_harmonics(l_max, k);
        let mut y = vec![Complex64::default(); nlm];
        let mut cdum = Complex64::new(1.0, 0.0);
        for l in 0..=l_max.min(7) {
            let gl = recip_g(l, a, pref);
            for m in -(l as i32)..=l as i32 {
                let lm = lm_index(l, m)?;
                y[lm] = harmonics[lm].conj() * gl * cdum;
            }
            cdum *= Complex64::new(0.0, 1.0);
        }
        for site_b in 0..n_sites {
            let site_a_max = if site_b == 0 { 1 } else { site_b };
            for site_a in 0..site_a_max {
                let delta_r = [
                    Bohr(positions[site_a][0].get() - positions[site_b][0].get()),
                    Bohr(positions[site_a][1].get() - positions[site_b][1].get()),
                    Bohr(positions[site_a][2].get() - positions[site_b][2].get()),
                ];
                let k_inv = [InverseBohr(k[0]), InverseBohr(k[1]), InverseBohr(k[2])];
                let cexp = plane_wave_phase(k_inv, delta_r);
                for lm in 0..nlm {
                    values[(site_a * n_sites + site_b) * nlm + lm] += cexp * y[lm];
                }
            }
        }
    }

    let mut constants = StructureConstants {
        l_max,
        n_sites,
        values,
    };
    if nlm > 0 {
        let correction = Complex64::new(-5.0 / 16.0 / (4.0 * PI).sqrt(), 0.0);
        let current = constants.raw(0, 0, 0);
        constants.set(0, 0, 0, current + correction);
    }
    for site_b in 0..n_sites {
        for site_a in 0..site_b {
            for l in 0..=l_max {
                for m in -(l as i32)..=l as i32 {
                    let lm = lm_index(l, m)?;
                    let lm_neg = lm_index(l, -m)?;
                    let source = constants.raw(site_a, site_b, lm_neg);
                    constants.set(site_b, site_a, lm, parity(l as i32 + m) * source.conj());
                }
            }
        }
    }
    for site in 1..n_sites {
        for lm in 0..nlm {
            let value = constants.raw(0, 0, lm);
            constants.set(site, site, lm, value);
        }
    }
    for l in 0..=l_max {
        let factor = scale.powi(l as i32 + 1);
        for site_a in 0..n_sites {
            for site_b in 0..n_sites {
                for m in -(l as i32)..=l as i32 {
                    let lm = lm_index(l, m)?;
                    let value = constants.raw(site_a, site_b, lm) * factor;
                    constants.set(site_a, site_b, lm, value);
                }
            }
        }
    }
    Ok(constants)
}

fn convpar(l: u32, ew: f64) -> f64 {
    let param = if l <= 7 { CONVPARAM1 } else { CONVPARAM2 };
    param / ew.powi(l as i32 + 1) * 4.0 * PI
}

fn real_space_cutoff(
    scale: f64,
    l_max: u32,
    latcon: f64,
    ew: f64,
    factor: f64,
) -> Result<f64, CoulombError> {
    let mut a = 1.0;
    let mut da = 1.0;
    let mut converged = false;
    for _ in 0..20_000 {
        let mut all_small = true;
        for l in 0..=7u32.min(l_max) {
            if spex_real_g(l, a).abs() * factor * a * a >= convpar(l, ew) {
                all_small = false;
                break;
            }
        }
        if all_small {
            if da < 5.0e-5 {
                converged = true;
                break;
            }
            a -= da;
            da /= 10.0;
        }
        a += da;
        if a > 120.0 {
            break;
        }
    }
    if !converged {
        return Err(CoulombError::RealSpaceCutoffNotConverged);
    }
    let mut rad = a / scale;
    if l_max >= 8 {
        let high_l = (1.0 / CONVPARAM2).powf(1.0 / 7.0) * latcon;
        rad = rad.max(high_l);
    }
    Ok(rad)
}

fn reciprocal_cutoff(scale: f64, vol: f64, latcon: f64, ew: f64) -> Result<f64, CoulombError> {
    let pref = 4.0 * PI / (scale.powi(3) * vol);
    let factor = 4.0 * PI * scale.powi(2) / latcon.powi(2);
    let mut a = 1.0;
    let mut da = 1.0;
    let mut converged = false;
    for _ in 0..20_000 {
        let g0 = recip_g(0, a, pref);
        if (g0 * factor * a * a).abs() < convpar(0, ew) {
            if da < 5.0e-5 {
                converged = true;
                break;
            }
            a -= da;
            da /= 10.0;
        }
        a += da;
        if a > 120.0 {
            break;
        }
    }
    if !converged {
        return Err(CoulombError::ReciprocalSpaceCutoffNotConverged);
    }
    Ok(a * scale)
}

/// SPEX real-space $g_L(a)$ (`coulombmatrix.f:2430-2442`).
///
/// $L=4$ skips $a/9$ and ends at $a/10$. $L\ge 8$ is $a^{-(L+1)}$ (real space only).
pub fn spex_real_g(l: u32, a: f64) -> f64 {
    if a == 0.0 {
        return 0.0;
    }
    let rexp = (-a).exp();
    match l {
        0 => rexp / a * (1.0 + a * 11.0 / 16.0 * (1.0 + a * 3.0 / 11.0 * (1.0 + a / 9.0))),
        1 => {
            rexp / a.powi(2)
                * (1.0 + a * (1.0 + a / 2.0 * (1.0 + a * 7.0 / 24.0 * (1.0 + a / 7.0))))
        }
        2 => {
            rexp / a.powi(3)
                * (1.0
                    + a * (1.0
                        + a / 2.0
                            * (1.0
                                + a / 3.0
                                    * (1.0 + a / 4.0 * (1.0 + a * 3.0 / 16.0 * (1.0 + a / 9.0))))))
        }
        3 => {
            rexp / a.powi(4)
                * (1.0
                    + a * (1.0
                        + a / 2.0
                            * (1.0
                                + a / 3.0
                                    * (1.0
                                        + a / 4.0
                                            * (1.0
                                                + a / 5.0 * (1.0 + a / 6.0 * (1.0 + a / 8.0)))))))
        }
        4 => {
            rexp / a.powi(5)
                * (1.0
                    + a * (1.0
                        + a / 2.0
                            * (1.0
                                + a / 3.0
                                    * (1.0
                                        + a / 4.0
                                            * (1.0
                                                + a / 5.0
                                                    * (1.0
                                                        + a / 6.0
                                                            * (1.0
                                                                + a / 7.0
                                                                    * (1.0
                                                                        + a / 8.0
                                                                            * (1.0
                                                                                + a / 10.0)))))))))
        }
        5 => rexp / a.powi(6) * hlp9(a, 10),
        6 => rexp / a.powi(7) * hlp9(a, 12),
        7 => rexp / a.powi(8) * hlp9(a, 13),
        _ => a.powi(-(l as i32 + 1)),
    }
}

/// SPEX `HLP9` prefix through $a/9$, then $a/10,\ldots,a/n_{\mathrm{last}}$.
fn hlp9(a: f64, last: i32) -> f64 {
    let mut acc = 1.0;
    for k in (10..=last).rev() {
        acc = 1.0 + a / f64::from(k) * acc;
    }
    for k in (1..=9).rev() {
        acc = 1.0 + a / f64::from(k) * acc;
    }
    acc
}

fn recip_g(l: u32, a: f64, pref: f64) -> f64 {
    if a == 0.0 {
        return match l {
            0 => pref * (-4.0),
            1 => 0.0,
            _ => 0.0,
        };
    }
    let aa = 1.0 / (1.0 + a * a);
    match l {
        0 => pref * aa.powi(4) / a.powi(2),
        1 => pref * aa.powi(4) / a,
        2 => pref * aa.powi(5) / 3.0,
        3 => pref * aa.powi(5) * a / 15.0,
        4 => pref * aa.powi(6) * a.powi(2) / 105.0,
        5 => pref * aa.powi(6) * a.powi(3) / 945.0,
        6 => pref * aa.powi(7) * a.powi(4) / 10_395.0,
        7 => pref * aa.powi(7) * a.powi(5) / 135_135.0,
        _ => 0.0,
    }
}

type DirectImage = ([i32; 3], [f64; 3]);

fn enumerate_direct(cell: &Cell, cutoff: Bohr) -> Result<Vec<DirectImage>, CoulombError> {
    let basis = cell
        .basis()
        .map(|vector| vector.map(|component| InverseBohr(component.get())));
    let fake = ReciprocalLattice::new(basis)?;
    let vectors = fake.enumerate(InverseBohr(cutoff.get()))?;
    Ok(vectors
        .into_iter()
        .map(|g| (g.index, g.cartesian.map(InverseBohr::get)))
        .collect())
}

fn index_to_bohr(index: [i32; 3], cell: &Cell) -> [Bohr; 3] {
    cell.cartesian([
        f64::from(index[0]),
        f64::from(index[1]),
        f64::from(index[2]),
    ])
}

/// Independent real-space sum of $Y_{LM}^*(\mathbf{R})/R^{L+1}$ used as a convention oracle.
///
/// The $T=0$ image is omitted when $|\Delta R|=0$. This is absolutely convergent for $L\ge 1$.
pub fn brute_force_structure_constant(
    cell: &Cell,
    q: TransferQ,
    delta: [f64; 3],
    l: u32,
    m: i32,
    n_max: i32,
) -> Result<Complex64, CoulombError> {
    let mut acc = Complex64::default();
    let a = cell.basis();
    for n0 in -n_max..=n_max {
        for n1 in -n_max..=n_max {
            for n2 in -n_max..=n_max {
                let t = [
                    f64::from(n0) * a[0][0].get()
                        + f64::from(n1) * a[1][0].get()
                        + f64::from(n2) * a[2][0].get(),
                    f64::from(n0) * a[0][1].get()
                        + f64::from(n1) * a[1][1].get()
                        + f64::from(n2) * a[2][1].get(),
                    f64::from(n0) * a[0][2].get()
                        + f64::from(n1) * a[1][2].get()
                        + f64::from(n2) * a[2][2].get(),
                ];
                let ra = [t[0] + delta[0], t[1] + delta[1], t[2] + delta[2]];
                let r = ra.iter().map(|c| c * c).sum::<f64>().sqrt();
                if r <= 1.0e-14 {
                    continue;
                }
                let harmonics = complex_spherical_harmonics(l, ra);
                let phase = plane_wave_phase(q.cartesian, [Bohr(t[0]), Bohr(t[1]), Bohr(t[2])]);
                acc += phase * harmonics[lm_index(l, m)?].conj() / r.powi(l as i32 + 1);
            }
        }
    }
    Ok(acc)
}
