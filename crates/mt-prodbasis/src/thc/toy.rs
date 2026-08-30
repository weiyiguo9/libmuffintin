//! Finite deterministic periodic toy bases from the local scratch scripts.
//!
//! The $4\pi/|q+G|^2$ helper is a data-only finite-cutoff candidate oracle.
//! It is not Weinert/SPEX `coulombmatrix` and is not a production Coulomb
//! assembler.

use crate::thc::ThcError;
use crate::thc::gram::InjectedCoulombGram;
use crate::thc::kmesh::KMesh;
use crate::thc::pair::{BlochOrbitals, PairBlock};
use crate::{AuxiliaryPartition, InterpolationRegion, PairColumnLayout};
use muffintin_core::{Bohr, InterstitialGeometry, Sphere, VolumeBohr3, complex_spherical_harmonic};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Cubic lattice constant of `thc_mt_kpoint_test.py`.
pub const MT_LATTICE: f64 = 6.0;
/// Muffin-tin radius of the MT-like toy.
pub const MT_RADIUS: f64 = 2.0;
/// Localization cutoff of the MT-like toy.
pub const MT_RCUT: f64 = 2.9;
/// Orbital count of the MT-like toy.
pub const MT_NORB: usize = 6;

/// Cubic lattice constant of `thc_lapw_end_to_end_test.py`.
pub const LAPW_LATTICE: f64 = 5.0;
/// Muffin-tin radius of the synthetic LAPW toy.
pub const LAPW_RADIUS: f64 = 0.82;

/// Parent-grid points, weights, and region tags.
#[derive(Clone, Debug, PartialEq)]
pub struct ToyGrid {
    pub name: String,
    pub points: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub regions: Vec<InterpolationRegion>,
}

impl ToyGrid {
    /// Number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the grid is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

fn fold_cubic(point: [f64; 3], lattice: f64) -> [f64; 3] {
    std::array::from_fn(|axis| point[axis] - lattice * (point[axis] / lattice).round())
}

fn fib_sphere(n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|index| {
            let i = index as f64 + 0.5;
            let ct = 1.0 - 2.0 * i / n as f64;
            let st = (1.0 - ct * ct).max(0.0).sqrt();
            let th = PI * (1.0 + 5.0_f64.sqrt()) * i;
            [st * th.cos(), st * th.sin(), ct]
        })
        .collect()
}

fn log_radial_trap(r0: f64, r1: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let h = (r1 / r0).ln() / (n as f64 - 1.0);
    let r: Vec<f64> = (0..n).map(|j| r0 * (h * j as f64).exp()).collect();
    let mut w: Vec<f64> = r.iter().map(|radius| radius.powi(3) * h).collect();
    if let Some(first) = w.first_mut() {
        *first *= 0.5;
    }
    if let Some(last) = w.last_mut() {
        *last *= 0.5;
    }
    (r, w)
}

fn log_radial_shells(r0: f64, r1: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let h = (r1 / r0).ln() / (n as f64 - 1.0);
    let r: Vec<f64> = (0..n).map(|j| r0 * (h * j as f64).exp()).collect();
    let mut boundaries = vec![0.0; n + 1];
    for i in 1..n {
        boundaries[i] = (r[i - 1] * r[i]).sqrt();
    }
    boundaries[n] = r1;
    let w = (0..n)
        .map(|i| (boundaries[i + 1].powi(3) - boundaries[i].powi(3)) / 3.0)
        .collect();
    (r, w)
}

/// Uniform cubic grid matching `thc_mt_kpoint_test.py:103-119`.
pub fn mt_uniform_grid(n: usize, shift: crate::thc::select::UniformShift) -> ToyGrid {
    let offset = match shift {
        crate::thc::select::UniformShift::Origin => [0.0; 3],
        crate::thc::select::UniformShift::Half => [0.5; 3],
        crate::thc::select::UniformShift::Random { seed } => {
            let mut rng = SplitMix(seed);
            [rng.unit(), rng.unit(), rng.unit()]
        }
    };
    let mut points = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let raw = [
                    (i as f64 + offset[0]) / n as f64 * MT_LATTICE,
                    (j as f64 + offset[1]) / n as f64 * MT_LATTICE,
                    (k as f64 + offset[2]) / n as f64 * MT_LATTICE,
                ];
                points.push(fold_cubic(raw, MT_LATTICE));
            }
        }
    }
    let weight = MT_LATTICE.powi(3) / (n * n * n) as f64;
    let npts = points.len();
    ToyGrid {
        name: format!("uniform {n} {:?}", shift.as_str()),
        points,
        weights: vec![weight; npts],
        regions: vec![InterpolationRegion::Uniform; npts],
    }
}

/// MT-adaptive grid matching `thc_mt_kpoint_test.py:121-127`.
pub fn mt_adaptive_grid(nrad: usize, nang: usize, ninter: usize) -> ToyGrid {
    let (radii, wr) = log_radial_trap(2.0e-3, MT_RADIUS, nrad);
    let ang = fib_sphere(nang);
    let mut points = Vec::new();
    let mut weights = Vec::new();
    let mut regions = Vec::new();
    for (radius, radial_w) in radii.iter().zip(&wr) {
        for direction in &ang {
            points.push([
                radius * direction[0],
                radius * direction[1],
                radius * direction[2],
            ]);
            weights.push(radial_w * 4.0 * PI / nang as f64);
            regions.push(InterpolationRegion::MuffinTin { site: 0 });
        }
    }
    let uniform = mt_uniform_grid(ninter, crate::thc::select::UniformShift::Half);
    for (point, weight) in uniform.points.iter().zip(&uniform.weights) {
        let distance = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
        if distance > MT_RADIUS && distance <= MT_RCUT {
            points.push(*point);
            weights.push(*weight);
            regions.push(InterpolationRegion::Interstitial);
        }
    }
    ToyGrid {
        name: format!("MT-adaptive nrad={nrad}"),
        points,
        weights,
        regions,
    }
}

/// Dense MT-like reference grid from `thc_mt_kpoint_test.py:278`.
pub fn mt_reference_grid() -> ToyGrid {
    let (radii, wr) = log_radial_trap(5.0e-4, MT_RCUT, 72);
    let ang = fib_sphere(78);
    let mut points = Vec::new();
    let mut weights = Vec::new();
    for (radius, radial_w) in radii.iter().zip(&wr) {
        for direction in &ang {
            points.push([
                radius * direction[0],
                radius * direction[1],
                radius * direction[2],
            ]);
            weights.push(radial_w * 4.0 * PI / 78.0);
        }
    }
    let npts = points.len();
    ToyGrid {
        name: "mt-reference".to_owned(),
        points,
        weights,
        regions: vec![InterpolationRegion::MuffinTin { site: 0 }; npts],
    }
}

fn chi_all(disp: [f64; 3]) -> [f64; MT_NORB] {
    let r = (disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]).sqrt();
    let fc = if r < MT_RCUT {
        let x = (r / MT_RCUT).min(1.0);
        (1.0 - x * x).powi(2)
    } else {
        0.0
    };
    let p = (-2.0 * r).exp();
    [
        (-20.0 * r).exp() * fc,
        r * p * fc,
        disp[0] * p * fc,
        disp[1] * p * fc,
        disp[2] * p * fc,
        (-0.25 * r * r).exp() * fc,
    ]
}

/// Orbital $L^2$ norms on the MT-like reference grid.
pub fn mt_orbital_norms(reference: &ToyGrid) -> [f64; MT_NORB] {
    let mut acc = [0.0; MT_NORB];
    for (point, weight) in reference.points.iter().zip(&reference.weights) {
        let chi = chi_all(*point);
        for (index, value) in chi.iter().enumerate() {
            acc[index] += value * value * *weight;
        }
    }
    acc.map(f64::sqrt)
}

/// Cell-periodic MT-like Bloch parts, `thc_mt_kpoint_test.py:143-151`.
pub fn mt_bloch_orbitals(
    grid: &ToyGrid,
    norms: &[f64; MT_NORB],
    mesh: &KMesh,
) -> Result<BlochOrbitals, ThcError> {
    let n_k = mesh.len();
    let mut values = vec![Complex64::default(); grid.len() * n_k * MT_NORB];
    for tx in -1..=1 {
        for ty in -1..=1 {
            for tz in -1..=1 {
                let lattice = [f64::from(tx), f64::from(ty), f64::from(tz)];
                for (p, point) in grid.points.iter().enumerate() {
                    let disp = [
                        point[0] - MT_LATTICE * lattice[0],
                        point[1] - MT_LATTICE * lattice[1],
                        point[2] - MT_LATTICE * lattice[2],
                    ];
                    let chi = chi_all(disp);
                    for (ik, kfrac) in mesh.fractional().iter().enumerate() {
                        let phase = Complex64::from_polar(
                            1.0,
                            2.0 * PI
                                * (kfrac[0] * lattice[0]
                                    + kfrac[1] * lattice[1]
                                    + kfrac[2] * lattice[2]),
                        );
                        for orb in 0..MT_NORB {
                            let index = (p * n_k + ik) * MT_NORB + orb;
                            values[index] += phase * chi[orb] / norms[orb];
                        }
                    }
                }
            }
        }
    }
    for (p, point) in grid.points.iter().enumerate() {
        for (ik, kfrac) in mesh.fractional().iter().enumerate() {
            let phase = Complex64::from_polar(
                1.0,
                -2.0 * PI / MT_LATTICE
                    * (point[0] * kfrac[0] + point[1] * kfrac[1] + point[2] * kfrac[2]),
            );
            for orb in 0..MT_NORB {
                let index = (p * n_k + ik) * MT_NORB + orb;
                values[index] *= phase;
            }
        }
    }
    BlochOrbitals::new(grid.len(), n_k, MT_NORB, values)
}

/// Product partition of the MT-like toy.
pub fn mt_partition() -> AuxiliaryPartition {
    AuxiliaryPartition::from_interstitial(
        InterstitialGeometry::new(
            VolumeBohr3(MT_LATTICE.powi(3)),
            vec![Sphere {
                center: [Bohr(0.0); 3],
                radius: Bohr(MT_RADIUS),
            }],
        )
        .expect("MT-like partition"),
    )
}

/// $2\times2\times2$ mesh of the MT-like toy.
pub fn mt_kmesh() -> KMesh {
    KMesh::gamma_centred([2, 2, 2], MT_LATTICE).expect("MT-like k-mesh")
}

struct SplitMix(u64);

impl SplitMix {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z >> 11) as f64 / ((1_u64 << 53) as f64)
    }
}

const ATOM_POS: [[f64; 3]; 2] = [[-1.18, -0.31, 0.17], [1.07, 0.43, -0.29]];
const ATOM_SCALE: [f64; 2] = [1.0, 0.73];
const G_APW: [[i32; 3]; 4] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]];
const LO_SPECS: [(usize, u32, i32, f64); 2] = [(0, 0, 0, 18.0), (1, 3, 2, 3.0)];
/// Synthetic LAPW orbital count (`thc_lapw_end_to_end_test.py` `NORB`).
pub const LAPW_NORB: usize = 6;
const LMAX: u32 = 3;

fn fold_displacement(displacement: [f64; 3]) -> ([f64; 3], [i32; 3]) {
    let image = [
        (displacement[0] / LAPW_LATTICE).round() as i32,
        (displacement[1] / LAPW_LATTICE).round() as i32,
        (displacement[2] / LAPW_LATTICE).round() as i32,
    ];
    (
        [
            displacement[0] - LAPW_LATTICE * f64::from(image[0]),
            displacement[1] - LAPW_LATTICE * f64::from(image[1]),
            displacement[2] - LAPW_LATTICE * f64::from(image[2]),
        ],
        image,
    )
}

fn augmentation_radial(radius: f64, l: u32, atom_scale: f64) -> f64 {
    let x = radius / LAPW_RADIUS;
    let amplitude = [3.2, 2.3, 1.55, 1.05][l as usize];
    let sharpness = [10.0, 7.0, 4.5, 2.5][l as usize];
    atom_scale * amplitude * x.powi(l as i32) * (1.0 - x).powi(2) * (-sharpness * x).exp()
}

fn local_orbital_radial(radius: f64, l: u32, sharpness: f64) -> f64 {
    let x = radius / LAPW_RADIUS;
    x.powi(l as i32) * (1.0 - x).powi(2) * (-sharpness * x).exp()
}

fn i_pow_l(l: u32) -> Complex64 {
    match l % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

fn outside_spheres(point: [f64; 3]) -> bool {
    ATOM_POS.iter().all(|position| {
        let (disp, _) = fold_displacement([
            point[0] - position[0],
            point[1] - position[1],
            point[2] - position[2],
        ]);
        (disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]).sqrt() >= LAPW_RADIUS
    })
}

/// Uniform cubic grid of the synthetic LAPW toy (`thc_lapw_end_to_end_test.py:131-136`).
pub fn lapw_uniform_grid(n: usize) -> ToyGrid {
    let volume = LAPW_LATTICE.powi(3);
    let mut points = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                points.push([
                    ((i as f64 + 0.5) / n as f64 - 0.5) * LAPW_LATTICE,
                    ((j as f64 + 0.5) / n as f64 - 0.5) * LAPW_LATTICE,
                    ((k as f64 + 0.5) / n as f64 - 0.5) * LAPW_LATTICE,
                ]);
            }
        }
    }
    let npts = points.len();
    ToyGrid {
        name: format!("uniform {n}^3"),
        points,
        weights: vec![volume / npts as f64; npts],
        regions: vec![InterpolationRegion::Uniform; npts],
    }
}

/// Composite adaptive grid of the synthetic LAPW toy.
pub fn lapw_composite_grid(name: &str, nrad: usize, nang: usize, ninter: usize) -> ToyGrid {
    let (radii, wr) = log_radial_shells(2.0e-4, LAPW_RADIUS, nrad);
    let ang = fib_sphere(nang);
    let mut points = Vec::new();
    let mut weights = Vec::new();
    let mut regions = Vec::new();
    for (atom, position) in ATOM_POS.iter().enumerate() {
        for (radius, radial_w) in radii.iter().zip(&wr) {
            for direction in &ang {
                points.push([
                    position[0] + radius * direction[0],
                    position[1] + radius * direction[1],
                    position[2] + radius * direction[2],
                ]);
                weights.push(radial_w * 4.0 * PI / nang as f64);
                regions.push(InterpolationRegion::MuffinTin { site: atom });
            }
        }
    }
    let interstitial = lapw_uniform_grid(ninter);
    let mut inter_points = Vec::new();
    let mut inter_weights = Vec::new();
    for (point, weight) in interstitial.points.iter().zip(&interstitial.weights) {
        if outside_spheres(*point) {
            inter_points.push(*point);
            inter_weights.push(*weight);
        }
    }
    let exact_interstitial = LAPW_LATTICE.powi(3) - 2.0 * 4.0 * PI * LAPW_RADIUS.powi(3) / 3.0;
    let sum: f64 = inter_weights.iter().sum();
    if sum > 0.0 {
        let scale = exact_interstitial / sum;
        for weight in &mut inter_weights {
            *weight *= scale;
        }
    }
    points.extend(inter_points);
    weights.extend(inter_weights);
    regions.extend(std::iter::repeat_n(
        InterpolationRegion::Interstitial,
        weights.len() - regions.len(),
    ));
    ToyGrid {
        name: name.to_owned(),
        points,
        weights,
        regions,
    }
}

fn raw_cell_periodic_orbitals(grid: &ToyGrid, mesh: &KMesh) -> Result<Vec<Complex64>, ThcError> {
    let n_k = mesh.len();
    let n_orb = LAPW_NORB;
    let mut result = vec![Complex64::default(); grid.len() * n_k * n_orb];
    let reciprocal = 2.0 * PI / LAPW_LATTICE;
    for (ik, k_fractional) in mesh.fractional().iter().enumerate() {
        let k_vector = [
            reciprocal * k_fractional[0],
            reciprocal * k_fractional[1],
            reciprocal * k_fractional[2],
        ];
        for (ib, g_integer) in G_APW.iter().enumerate() {
            let wave_vector = [
                reciprocal * (k_fractional[0] + f64::from(g_integer[0])),
                reciprocal * (k_fractional[1] + f64::from(g_integer[1])),
                reciprocal * (k_fractional[2] + f64::from(g_integer[2])),
            ];
            let wave_norm = (wave_vector[0] * wave_vector[0]
                + wave_vector[1] * wave_vector[1]
                + wave_vector[2] * wave_vector[2])
                .sqrt();
            let (wave_theta, wave_phi) = if wave_norm > 0.0 {
                (
                    (wave_vector[2] / wave_norm).clamp(-1.0, 1.0).acos(),
                    wave_vector[1].atan2(wave_vector[0]),
                )
            } else {
                (0.0, 0.0)
            };
            for (p, point) in grid.points.iter().enumerate() {
                let mut psi = Complex64::from_polar(
                    1.0,
                    point[0] * wave_vector[0]
                        + point[1] * wave_vector[1]
                        + point[2] * wave_vector[2],
                );
                for (atom, (position, atom_scale)) in ATOM_POS.iter().zip(&ATOM_SCALE).enumerate() {
                    let _ = atom;
                    let (disp, image) = fold_displacement([
                        point[0] - position[0],
                        point[1] - position[1],
                        point[2] - position[2],
                    ]);
                    let radius = (disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]).sqrt();
                    if radius >= LAPW_RADIUS {
                        continue;
                    }
                    let theta = if radius > 0.0 {
                        (disp[2] / radius).clamp(-1.0, 1.0).acos()
                    } else {
                        0.0
                    };
                    let phi = disp[1].atan2(disp[0]);
                    let image_center = [
                        position[0] + LAPW_LATTICE * f64::from(image[0]),
                        position[1] + LAPW_LATTICE * f64::from(image[1]),
                        position[2] + LAPW_LATTICE * f64::from(image[2]),
                    ];
                    let center_phase = Complex64::from_polar(
                        1.0,
                        image_center[0] * wave_vector[0]
                            + image_center[1] * wave_vector[1]
                            + image_center[2] * wave_vector[2],
                    );
                    let mut correction = Complex64::default();
                    for l in 0..=LMAX {
                        if wave_norm == 0.0 && l > 0 {
                            continue;
                        }
                        let radial = augmentation_radial(radius, l, *atom_scale);
                        let l_i = l as i32;
                        for m in -l_i..=l_i {
                            let ylm_wave = complex_spherical_harmonic(l, m, wave_theta, wave_phi)?;
                            let ylm_r = complex_spherical_harmonic(l, m, theta, phi)?;
                            correction += 4.0 * PI * i_pow_l(l) * ylm_wave.conj() * radial * ylm_r;
                        }
                    }
                    psi += center_phase * correction;
                }
                let bloch = Complex64::from_polar(
                    1.0,
                    -(point[0] * k_vector[0] + point[1] * k_vector[1] + point[2] * k_vector[2]),
                );
                result[(p * n_k + ik) * n_orb + ib] = psi * bloch;
            }
        }
        for (offset, (atom, l, m, sharpness)) in LO_SPECS.iter().enumerate() {
            let ib = G_APW.len() + offset;
            let position = ATOM_POS[*atom];
            for (p, point) in grid.points.iter().enumerate() {
                let (disp, image) = fold_displacement([
                    point[0] - position[0],
                    point[1] - position[1],
                    point[2] - position[2],
                ]);
                let radius = (disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]).sqrt();
                let mut psi = Complex64::default();
                if radius < LAPW_RADIUS {
                    let theta = if radius > 0.0 {
                        (disp[2] / radius).clamp(-1.0, 1.0).acos()
                    } else {
                        0.0
                    };
                    let phi = disp[1].atan2(disp[0]);
                    let lattice_phase = Complex64::from_polar(
                        1.0,
                        LAPW_LATTICE
                            * (f64::from(image[0]) * k_vector[0]
                                + f64::from(image[1]) * k_vector[1]
                                + f64::from(image[2]) * k_vector[2]),
                    );
                    psi = lattice_phase
                        * local_orbital_radial(radius, *l, *sharpness)
                        * complex_spherical_harmonic(*l, *m, theta, phi)?;
                }
                let bloch = Complex64::from_polar(
                    1.0,
                    -(point[0] * k_vector[0] + point[1] * k_vector[1] + point[2] * k_vector[2]),
                );
                result[(p * n_k + ik) * n_orb + ib] = psi * bloch;
            }
        }
    }
    Ok(result)
}

fn orthonormalize(
    raw: &[Complex64],
    n_points: usize,
    n_k: usize,
    n_orb: usize,
    weights: &[f64],
) -> Result<Vec<Complex64>, ThcError> {
    let mut out = vec![Complex64::default(); raw.len()];
    for ik in 0..n_k {
        let mut overlap = vec![Complex64::default(); n_orb * n_orb];
        for p in 0..n_points {
            for i in 0..n_orb {
                let vi = raw[(p * n_k + ik) * n_orb + i].conj() * weights[p];
                for j in 0..n_orb {
                    overlap[i * n_orb + j] += vi * raw[(p * n_k + ik) * n_orb + j];
                }
            }
        }
        let (values, vectors) = crate::thc::linalg::hermitian_eigensystem(&overlap, n_orb)?;
        if values[0] < 1.0e-10 {
            return Err(ThcError::LinearAlgebra(
                "ill-conditioned synthetic APW overlap",
            ));
        }
        let mut transform = vec![Complex64::default(); n_orb * n_orb];
        for i in 0..n_orb {
            for j in 0..n_orb {
                let mut acc = Complex64::default();
                for k in 0..n_orb {
                    acc += vectors[i * n_orb + k]
                        * (1.0 / values[k].sqrt())
                        * vectors[j * n_orb + k].conj();
                }
                transform[i * n_orb + j] = acc;
            }
        }
        for p in 0..n_points {
            for j in 0..n_orb {
                let mut acc = Complex64::default();
                for i in 0..n_orb {
                    acc += raw[(p * n_k + ik) * n_orb + i] * transform[i * n_orb + j];
                }
                out[(p * n_k + ik) * n_orb + j] = acc;
            }
        }
    }
    Ok(out)
}

/// Orthonormal synthetic LAPW Bloch orbitals on `grid`.
pub fn lapw_bloch_orbitals(
    grid: &ToyGrid,
    mesh: &KMesh,
    transforms_from: Option<&ToyGrid>,
) -> Result<BlochOrbitals, ThcError> {
    let source = transforms_from.unwrap_or(grid);
    let raw_ref = raw_cell_periodic_orbitals(source, mesh)?;
    let raw = if transforms_from.is_some() {
        raw_cell_periodic_orbitals(grid, mesh)?
    } else {
        raw_ref.clone()
    };
    let n_k = mesh.len();
    let ortho_source = if transforms_from.is_some() {
        orthonormalize(&raw_ref, source.len(), n_k, LAPW_NORB, &source.weights)?
    } else {
        orthonormalize(&raw, grid.len(), n_k, LAPW_NORB, &grid.weights)?
    };
    let values = if transforms_from.is_some() {
        apply_saved_transform(
            &raw,
            grid.len(),
            n_k,
            &raw_ref,
            source.len(),
            &source.weights,
        )?
    } else {
        ortho_source
    };
    BlochOrbitals::new(grid.len(), n_k, LAPW_NORB, values)
}

fn apply_saved_transform(
    raw: &[Complex64],
    n_points: usize,
    n_k: usize,
    raw_ref: &[Complex64],
    n_ref: usize,
    ref_weights: &[f64],
) -> Result<Vec<Complex64>, ThcError> {
    let n_orb = LAPW_NORB;
    let mut out = vec![Complex64::default(); raw.len()];
    for ik in 0..n_k {
        let mut overlap = vec![Complex64::default(); n_orb * n_orb];
        for p in 0..n_ref {
            for i in 0..n_orb {
                let vi = raw_ref[(p * n_k + ik) * n_orb + i].conj() * ref_weights[p];
                for j in 0..n_orb {
                    overlap[i * n_orb + j] += vi * raw_ref[(p * n_k + ik) * n_orb + j];
                }
            }
        }
        let (values, vectors) = crate::thc::linalg::hermitian_eigensystem(&overlap, n_orb)?;
        let mut transform = vec![Complex64::default(); n_orb * n_orb];
        for i in 0..n_orb {
            for j in 0..n_orb {
                let mut acc = Complex64::default();
                for k in 0..n_orb {
                    acc += vectors[i * n_orb + k]
                        * (1.0 / values[k].sqrt())
                        * vectors[j * n_orb + k].conj();
                }
                transform[i * n_orb + j] = acc;
            }
        }
        for p in 0..n_points {
            for j in 0..n_orb {
                let mut acc = Complex64::default();
                for i in 0..n_orb {
                    acc += raw[(p * n_k + ik) * n_orb + i] * transform[i * n_orb + j];
                }
                out[(p * n_k + ik) * n_orb + j] = acc;
            }
        }
    }
    Ok(out)
}

/// $2\times2\times1$ mesh of the synthetic LAPW toy.
pub fn lapw_kmesh() -> KMesh {
    KMesh::gamma_centred([2, 2, 1], LAPW_LATTICE).expect("LAPW k-mesh")
}

/// Product partition of the synthetic LAPW toy.
pub fn lapw_partition() -> AuxiliaryPartition {
    AuxiliaryPartition::from_interstitial(
        InterstitialGeometry::new(
            VolumeBohr3(LAPW_LATTICE.powi(3)),
            ATOM_POS
                .iter()
                .map(|position| Sphere {
                    center: [Bohr(position[0]), Bohr(position[1]), Bohr(position[2])],
                    radius: Bohr(LAPW_RADIUS),
                })
                .collect::<Vec<_>>(),
        )
        .expect("LAPW partition"),
    )
}

/// Integer reciprocal vectors with $|G|^2\le$ `cutoff_squared`.
pub fn reciprocal_vectors(cutoff_squared: i32) -> Vec<[i32; 3]> {
    let limit = (cutoff_squared as f64).sqrt() as i32 + 1;
    let mut out = Vec::new();
    for i in -limit..=limit {
        for j in -limit..=limit {
            for k in -limit..=limit {
                if i * i + j * j + k * k <= cutoff_squared {
                    out.push([i, j, k]);
                }
            }
        }
    }
    out
}

/// Finite-cutoff $4\pi/|q+G|^2$ factors with the $q+G=0$ head omitted.
///
/// Candidate-oracle only: not Weinert, not SPEX `coulombmatrix`.
pub fn toy_coulomb_factors(
    q_fractional: [f64; 3],
    g_integer: &[[i32; 3]],
    lattice: f64,
) -> Vec<f64> {
    let scale = 2.0 * PI / lattice;
    g_integer
        .iter()
        .map(|g| {
            let qg = [
                scale * (f64::from(g[0]) + q_fractional[0]),
                scale * (f64::from(g[1]) + q_fractional[1]),
                scale * (f64::from(g[2]) + q_fractional[2]),
            ];
            let squared = qg[0] * qg[0] + qg[1] * qg[1] + qg[2] * qg[2];
            if squared > 1.0e-14 {
                4.0 * PI / squared
            } else {
                0.0
            }
        })
        .collect()
}

/// Pair Fourier coefficients on a toy grid.
pub fn pair_fourier(
    block: &PairBlock,
    grid: &ToyGrid,
    g_cart: &[[f64; 3]],
    volume: f64,
) -> Vec<Complex64> {
    let n_g = g_cart.len();
    let n_col = block.n_columns();
    let mut out = vec![Complex64::default(); n_g * n_col];
    for (g_index, g) in g_cart.iter().enumerate() {
        for p in 0..grid.len() {
            let phase = Complex64::from_polar(
                1.0,
                -(grid.points[p][0] * g[0] + grid.points[p][1] * g[1] + grid.points[p][2] * g[2]),
            );
            let w = grid.weights[p];
            for col in 0..n_col {
                out[g_index * n_col + col] += phase * w * block.at(p, col);
            }
        }
    }
    for value in &mut out {
        *value /= volume;
    }
    out
}

/// Injected pair-pair Gram from the finite-cutoff toy Coulomb oracle.
pub fn toy_coulomb_gram(
    q_index: usize,
    q: crate::TransferQ,
    layout: PairColumnLayout,
    pair_fourier: &[Complex64],
    factors: &[f64],
    volume: f64,
) -> Result<InjectedCoulombGram, ThcError> {
    let n_g = factors.len();
    let n = layout.n_columns()?;
    let expected = crate::thc::error::checked_storage_len(&[n_g, n])?;
    if pair_fourier.len() != expected {
        return Err(ThcError::PairBlockLength {
            expected,
            actual: pair_fourier.len(),
        });
    }
    let gram_len = crate::thc::error::checked_storage_len(&[n, n])?;
    let mut gram = vec![Complex64::default(); gram_len];
    for g in 0..n_g {
        let factor = factors[g];
        for i in 0..n {
            let left = pair_fourier[g * n + i].conj();
            for j in 0..n {
                gram[i * n + j] += left * factor * pair_fourier[g * n + j];
            }
        }
    }
    for value in &mut gram {
        *value *= volume;
    }
    InjectedCoulombGram::from_dense(q_index, q, layout, gram)
}

/// Recorded finite-cutoff ERI/action gates from `thc_lapw_end_to_end_test.py:604-611`.
pub const RECORDED_FINE_ERI_FROBENIUS: f64 = 4.932e-2;
/// Recorded fine-grid pair-action error.
pub const RECORDED_FINE_ACTION: f64 = 6.230e-2;
/// Deterministic ERI/action gate used by the Python smoke test.
pub const RECORDED_ERI_ACTION_GATE: f64 = 8.0e-2;
/// Independent-reference ERI convergence recorded in the Python output.
pub const RECORDED_REFERENCE_CONVERGENCE: f64 = 2.498e-2;
/// Independent-reference convergence gate.
pub const RECORDED_REFERENCE_GATE: f64 = 5.0e-2;
/// Seed of the eight action vectors in `thc_lapw_end_to_end_test.py:35,453`.
pub const ACTION_VECTOR_SEED: u64 = 19;
/// Number of action probe vectors.
pub const ACTION_VECTOR_COUNT: usize = 8;

/// Finite-cutoff $4\pi/|q+G|^2$ kernel used by the toy ERI/action oracle.
#[derive(Clone, Debug, PartialEq)]
pub struct ToyFiniteCutoffKernel {
    pub g_integer: Vec<[i32; 3]>,
    pub lattice: f64,
    pub volume: f64,
}

impl ToyFiniteCutoffKernel {
    /// Cartesian $G$ for the stored integer labels.
    pub fn g_cartesian(&self) -> Vec<[f64; 3]> {
        let scale = 2.0 * PI / self.lattice;
        self.g_integer
            .iter()
            .map(|g| {
                [
                    scale * f64::from(g[0]),
                    scale * f64::from(g[1]),
                    scale * f64::from(g[2]),
                ]
            })
            .collect()
    }
}

/// Per-$q$ toy ERI/action comparison following
/// `thc_lapw_end_to_end_test.py:420-460`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToyEriActionMetrics {
    pub pair_fourier: f64,
    pub eri_frobenius: f64,
    pub eri_max_element: f64,
    pub action: f64,
}

impl ToyEriActionMetrics {
    /// Component-wise maximum, matching Python `max` over $q$.
    pub fn max_with(self, other: Self) -> Self {
        Self {
            pair_fourier: self.pair_fourier.max(other.pair_fourier),
            eri_frobenius: self.eri_frobenius.max(other.eri_frobenius),
            eri_max_element: self.eri_max_element.max(other.eri_max_element),
            action: self.action.max(other.action),
        }
    }
}

/// Fourier coefficients of a row-major `n_points × n_col` collocation.
///
/// Matches `thc_lapw_end_to_end_test.py:364-370`.
pub fn values_fourier(
    values: &[Complex64],
    n_points: usize,
    n_col: usize,
    grid: &ToyGrid,
    g_cart: &[[f64; 3]],
    volume: f64,
) -> Result<Vec<Complex64>, ThcError> {
    let expected = crate::thc::error::checked_storage_len(&[n_points, n_col])?;
    if values.len() != expected {
        return Err(ThcError::PairBlockLength {
            expected,
            actual: values.len(),
        });
    }
    if grid.len() != n_points {
        return Err(ThcError::OrbitalPointCount {
            orbitals: n_points,
            points: grid.len(),
        });
    }
    crate::thc::error::validate_quadrature_weights(&grid.weights)?;
    let n_g = g_cart.len();
    let mut out = vec![Complex64::default(); crate::thc::error::checked_storage_len(&[n_g, n_col])?];
    for (g_index, g) in g_cart.iter().enumerate() {
        for p in 0..n_points {
            let phase = Complex64::from_polar(
                1.0,
                -(grid.points[p][0] * g[0] + grid.points[p][1] * g[1] + grid.points[p][2] * g[2]),
            );
            let w = grid.weights[p];
            for col in 0..n_col {
                out[g_index * n_col + col] += phase * w * values[p * n_col + col];
            }
        }
    }
    for value in &mut out {
        *value /= volume;
    }
    Ok(out)
}

/// Approximate pair Fourier $\tilde\rho_G = \zeta_G R_\mu$ on the candidate grid.
pub fn approximate_pair_fourier(
    zeta_fourier: &[Complex64],
    n_g: usize,
    n_mu: usize,
    selected_rows: &[Complex64],
    n_col: usize,
) -> Result<Vec<Complex64>, ThcError> {
    let expected_zeta = crate::thc::error::checked_storage_len(&[n_g, n_mu])?;
    if zeta_fourier.len() != expected_zeta {
        return Err(ThcError::PairBlockLength {
            expected: expected_zeta,
            actual: zeta_fourier.len(),
        });
    }
    let expected_rows = crate::thc::error::checked_storage_len(&[n_mu, n_col])?;
    if selected_rows.len() != expected_rows {
        return Err(ThcError::PairBlockLength {
            expected: expected_rows,
            actual: selected_rows.len(),
        });
    }
    Ok(crate::thc::select::matmul(
        zeta_fourier,
        n_g,
        n_mu,
        selected_rows,
        n_col,
    ))
}

/// Relative Gram Frobenius $\lVert A-B\rVert_F/\lVert A\rVert_F$.
pub fn relative_gram_frobenius(
    reference: &InjectedCoulombGram,
    other: &InjectedCoulombGram,
) -> Result<f64, ThcError> {
    if reference.dimension() != other.dimension() || reference.data().len() != other.data().len() {
        return Err(ThcError::GramShape {
            index: other.q_index,
            expected_len: reference.data().len(),
            actual_len: other.data().len(),
        });
    }
    let mut diff = Vec::with_capacity(reference.data().len());
    for (left, right) in reference.data().iter().zip(other.data()) {
        diff.push(*left - *right);
    }
    let denom = crate::thc::linalg::frobenius(reference.data());
    if denom <= 1.0e-30 {
        return Ok(0.0);
    }
    Ok(crate::thc::linalg::frobenius(&diff) / denom)
}

/// Candidate-grid ζ Fourier → approximate pair Fourier/Gram vs a reference Gram.
///
/// Candidate-oracle only. Uses $4\pi/|q+G|^2$ with the $q+G=0$ head omitted.
/// Action probes are eight complex Gaussians from SplitMix64 seed 19, not
/// NumPy PCG64 bit-identity; the count, seed, and $\lVert V_{\mathrm{ref}} x\rVert$
/// normalisation match `thc_lapw_end_to_end_test.py:453-460`.
#[allow(clippy::too_many_arguments)]
pub fn compare_candidate_eri_action(
    zeta: &[Complex64],
    n_mu: usize,
    selected_rows: &[Complex64],
    n_col: usize,
    candidate: &ToyGrid,
    q_fractional: [f64; 3],
    kernel: &ToyFiniteCutoffKernel,
    reference_pair_fourier: &[Complex64],
    q_index: usize,
    q: crate::TransferQ,
    layout: PairColumnLayout,
) -> Result<ToyEriActionMetrics, ThcError> {
    let g_cart = kernel.g_cartesian();
    let n_g = g_cart.len();
    let n_points = candidate.len();
    let zeta_fourier = values_fourier(zeta, n_points, n_mu, candidate, &g_cart, kernel.volume)?;
    let approx_pair = approximate_pair_fourier(&zeta_fourier, n_g, n_mu, selected_rows, n_col)?;
    let factors = toy_coulomb_factors(q_fractional, &kernel.g_integer, kernel.lattice);
    let pair_denom = crate::thc::linalg::frobenius(reference_pair_fourier).max(1.0e-30);
    let mut pair_diff = Vec::with_capacity(approx_pair.len());
    if approx_pair.len() != reference_pair_fourier.len() {
        return Err(ThcError::PairBlockLength {
            expected: reference_pair_fourier.len(),
            actual: approx_pair.len(),
        });
    }
    for (left, right) in approx_pair.iter().zip(reference_pair_fourier) {
        pair_diff.push(*left - *right);
    }
    let pair_fourier = crate::thc::linalg::frobenius(&pair_diff) / pair_denom;
    let approximate_gram =
        toy_coulomb_gram(q_index, q, layout, &approx_pair, &factors, kernel.volume)?;
    let exact_gram = toy_coulomb_gram(
        q_index,
        q,
        layout,
        reference_pair_fourier,
        &factors,
        kernel.volume,
    )?;
    let n = layout.n_columns()?;
    let mut diff = Vec::with_capacity(n * n);
    let mut max_diff = 0.0_f64;
    let mut max_exact = 0.0_f64;
    for (left, right) in approximate_gram.data().iter().zip(exact_gram.data()) {
        let delta = *left - *right;
        max_diff = max_diff.max(delta.norm());
        max_exact = max_exact.max(right.norm());
        diff.push(delta);
    }
    let exact_frob = crate::thc::linalg::frobenius(exact_gram.data()).max(1.0e-30);
    let eri_frobenius = crate::thc::linalg::frobenius(&diff) / exact_frob;
    let eri_max_element = max_diff / max_exact.max(1.0e-30);
    let action = action_error(&diff, exact_gram.data(), n, ACTION_VECTOR_SEED);
    Ok(ToyEriActionMetrics {
        pair_fourier,
        eri_frobenius,
        eri_max_element,
        action,
    })
}

fn action_error(diff: &[Complex64], exact: &[Complex64], n: usize, seed: u64) -> f64 {
    let mut rng = SplitMix(seed);
    let mut worst = 0.0_f64;
    for _ in 0..ACTION_VECTOR_COUNT {
        let mut vector = vec![Complex64::default(); n];
        for entry in &mut vector {
            let (re, im) = rng.normal_pair();
            *entry = Complex64::new(re, im);
        }
        let denom = gemv_norm(exact, n, &vector);
        if denom > 1.0e-14 {
            let numer = gemv_norm(diff, n, &vector);
            worst = worst.max(numer / denom);
        }
    }
    worst
}

fn gemv_norm(matrix: &[Complex64], n: usize, vector: &[Complex64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..n {
        let mut sum = Complex64::default();
        for j in 0..n {
            sum += matrix[i * n + j] * vector[j];
        }
        acc += sum.norm_sqr();
    }
    acc.sqrt()
}

impl SplitMix {
    fn normal_pair(&mut self) -> (f64, f64) {
        let u = self.unit().max(f64::MIN_POSITIVE);
        let v = self.unit();
        let radius = (-2.0 * u.ln()).sqrt();
        let theta = 2.0 * PI * v;
        (radius * theta.cos(), radius * theta.sin())
    }
}
