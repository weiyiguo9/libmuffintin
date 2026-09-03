//! Exact muffin-tin core exchange on scalar/KH valence radial shells.

use muffintin_core::{ExponentialMesh, Kappa, RelativisticChannel, SpinProjection, gaunt};
use muffintin_tensor::{Axis, DenseHermitianMatrix, TensorError};
use num_complex::Complex64;
use thiserror::Error;

use crate::{CoulombError, intra_sphere_poisson};

/// One scalar/KH radial function in a fixed orbital-angular-momentum shell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarCoreExchangeRadial<'a> {
    pub p: &'a [f64],
    pub q: Option<&'a [f64]>,
}

/// One closed physical Dirac core shell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticCoreExchangeShell<'a> {
    pub kappa: Kappa,
    pub p: &'a [f64],
    pub q: &'a [f64],
    /// Explicit full-mesh normalization of the physical core `P/Q` pair.
    pub normalization: f64,
    /// Occupation of every magnetic channel in this closed shell.
    pub occupation_per_mu: f64,
}

/// Angular treatment of a closed relativistic core shell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StaticCoreExchangeMode {
    /// Keep only the spherical, spin-independent part used in scalar KH.
    ScalarAverage,
    /// Retain the complete spin-diagonal and spin-off-diagonal operator.
    SpinOrbitResolved,
}

/// Exact core exchange for one scalar valence `l` shell.
///
/// Coordinates are ordered as `(spin, m, radial)`, with spin up first and
/// `m=-l..l`. The matrix already contains the negative Fock sign.
#[derive(Clone, Debug, PartialEq)]
pub struct StaticCoreExchangeBlock {
    pub l: u32,
    pub radial_count: usize,
    pub matrix: DenseHermitianMatrix,
}

impl StaticCoreExchangeBlock {
    pub fn coordinate(&self, spin: SpinProjection, m: i32, radial: usize) -> Option<usize> {
        if m.unsigned_abs() > self.l || radial >= self.radial_count {
            return None;
        }
        let spin = match spin {
            SpinProjection::Up => 0,
            SpinProjection::Down => 1,
        };
        let magnetic = usize::try_from(i64::from(m) + i64::from(self.l)).ok()?;
        Some((spin * (2 * self.l as usize + 1) + magnetic) * self.radial_count + radial)
    }
}

/// Invalid radial input for exact static core exchange.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum StaticCoreExchangeError {
    #[error("static core exchange requires at least one scalar radial")]
    EmptyValenceRadials,
    #[error("static core exchange valence radial {radial} has an inconsistent P/Q length")]
    ValenceRadialLength { radial: usize },
    #[error("static core exchange shell {shell} has an inconsistent P/Q length")]
    CoreRadialLength { shell: usize },
    #[error("static core exchange shell {shell} has invalid normalization {value}")]
    Normalization { shell: usize, value: f64 },
    #[error("static core exchange shell {shell} has invalid per-mu occupation {value}")]
    Occupation { shell: usize, value: f64 },
    #[error(transparent)]
    Coulomb(#[from] CoulombError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

/// Build the exact SPEX-style static core-exchange block for one valence `l`.
///
/// Every core-valence product is retained; no mixed-product cutoff or
/// auxiliary-space projection is applied. [`StaticCoreExchangeMode::ScalarAverage`]
/// is the spherical operator used during scalar KH iteration. The resolved
/// mode retains the closed-shell `L.S` structure required when SOC is applied
/// to the valence states.
pub fn static_core_exchange_block(
    mesh: &ExponentialMesh,
    l: u32,
    valence: &[ScalarCoreExchangeRadial<'_>],
    cores: &[StaticCoreExchangeShell<'_>],
    mode: StaticCoreExchangeMode,
) -> Result<StaticCoreExchangeBlock, StaticCoreExchangeError> {
    validate_inputs(mesh, valence, cores)?;
    let radial_count = valence.len();
    let magnetic_count = 2 * l as usize + 1;
    let dimension = 2 * magnetic_count * radial_count;
    let mut values = vec![0.0; dimension * dimension];

    for core in cores {
        let core_scale = core.normalization.sqrt().recip();
        let core_l = core.kappa.large_l();
        let l_max = l + core_l;
        for channel in core.kappa.channels() {
            for multipole_l in 0..=l_max {
                let radial = radial_integrals(mesh, multipole_l, core, core_scale, valence)?;
                for multipole_m in -(multipole_l as i32)..=(multipole_l as i32) {
                    for left_spin in [SpinProjection::Up, SpinProjection::Down] {
                        for left_m in -(l as i32)..=(l as i32) {
                            let left_angular = scalar_spinor_gaunt(
                                l,
                                left_m,
                                left_spin,
                                multipole_l,
                                multipole_m,
                                channel,
                            );
                            if left_angular == 0.0 {
                                continue;
                            }
                            for right_spin in [SpinProjection::Up, SpinProjection::Down] {
                                for right_m in -(l as i32)..=(l as i32) {
                                    let right_angular = scalar_spinor_gaunt(
                                        l,
                                        right_m,
                                        right_spin,
                                        multipole_l,
                                        multipole_m,
                                        channel,
                                    );
                                    if right_angular == 0.0 {
                                        continue;
                                    }
                                    let angular =
                                        -core.occupation_per_mu * left_angular * right_angular;
                                    for left_radial in 0..radial_count {
                                        let left = coordinate(
                                            l,
                                            radial_count,
                                            left_spin,
                                            left_m,
                                            left_radial,
                                        );
                                        for right_radial in 0..radial_count {
                                            let right = coordinate(
                                                l,
                                                radial_count,
                                                right_spin,
                                                right_m,
                                                right_radial,
                                            );
                                            values[left * dimension + right] += angular
                                                * radial[left_radial * radial_count + right_radial];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if mode == StaticCoreExchangeMode::ScalarAverage {
        values = scalar_average(l, radial_count, &values);
    }
    let matrix = DenseHermitianMatrix::from_host_row_major(
        dimension,
        Axis::SiteCoordinate,
        values
            .into_iter()
            .map(|value| Complex64::new(value, 0.0))
            .collect(),
    )?;
    Ok(StaticCoreExchangeBlock {
        l,
        radial_count,
        matrix,
    })
}

fn validate_inputs(
    mesh: &ExponentialMesh,
    valence: &[ScalarCoreExchangeRadial<'_>],
    cores: &[StaticCoreExchangeShell<'_>],
) -> Result<(), StaticCoreExchangeError> {
    if valence.is_empty() {
        return Err(StaticCoreExchangeError::EmptyValenceRadials);
    }
    for (radial, input) in valence.iter().enumerate() {
        if input.p.len() != mesh.len()
            || input.q.is_some_and(|q| q.len() != mesh.len())
            || input
                .p
                .iter()
                .chain(input.q.into_iter().flatten())
                .any(|value| !value.is_finite())
        {
            return Err(StaticCoreExchangeError::ValenceRadialLength { radial });
        }
    }
    for (shell, input) in cores.iter().enumerate() {
        if input.p.len() < mesh.len()
            || input.q.len() < mesh.len()
            || input
                .p
                .iter()
                .take(mesh.len())
                .chain(input.q.iter().take(mesh.len()))
                .any(|value| !value.is_finite())
        {
            return Err(StaticCoreExchangeError::CoreRadialLength { shell });
        }
        if !input.normalization.is_finite() || input.normalization <= 0.0 {
            return Err(StaticCoreExchangeError::Normalization {
                shell,
                value: input.normalization,
            });
        }
        if !input.occupation_per_mu.is_finite() || !(0.0..=1.0).contains(&input.occupation_per_mu) {
            return Err(StaticCoreExchangeError::Occupation {
                shell,
                value: input.occupation_per_mu,
            });
        }
    }
    Ok(())
}

fn radial_integrals(
    mesh: &ExponentialMesh,
    multipole_l: u32,
    core: &StaticCoreExchangeShell<'_>,
    core_scale: f64,
    valence: &[ScalarCoreExchangeRadial<'_>],
) -> Result<Vec<f64>, StaticCoreExchangeError> {
    let products = valence
        .iter()
        .map(|radial| {
            mesh.radii()
                .iter()
                .enumerate()
                .map(|(index, radius)| {
                    let q = radial.q.map_or(0.0, |q| q[index]);
                    core_scale * (core.p[index] * radial.p[index] + core.q[index] * q)
                        / radius.get()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let radial_count = valence.len();
    let mut values = vec![0.0; radial_count * radial_count];
    for left in 0..radial_count {
        for right in left..radial_count {
            let value = intra_sphere_poisson(multipole_l, mesh, &products[left], &products[right])?;
            values[left * radial_count + right] = value;
            values[right * radial_count + left] = value;
        }
    }
    Ok(values)
}

fn scalar_spinor_gaunt(
    l: u32,
    m: i32,
    spin: SpinProjection,
    field_l: u32,
    field_m: i32,
    core: RelativisticChannel,
) -> f64 {
    let Some(term) = core
        .spinor_harmonic_terms()
        .into_iter()
        .flatten()
        .find(|term| term.spin == spin)
    else {
        return 0.0;
    };
    let ket_phase = if term.orbital.m & 1 == 0 { 1.0 } else { -1.0 };
    term.coefficient * ket_phase * gaunt(l, field_l, term.orbital.l, m, field_m, -term.orbital.m)
}

fn coordinate(l: u32, radial_count: usize, spin: SpinProjection, m: i32, radial: usize) -> usize {
    let spin = match spin {
        SpinProjection::Up => 0,
        SpinProjection::Down => 1,
    };
    let magnetic = usize::try_from(i64::from(m) + i64::from(l))
        .expect("m in the enumerated shell produces a nonnegative coordinate");
    (spin * (2 * l as usize + 1) + magnetic) * radial_count + radial
}

fn scalar_average(l: u32, radial_count: usize, resolved: &[f64]) -> Vec<f64> {
    let magnetic_count = 2 * l as usize + 1;
    let angular_count = 2 * magnetic_count;
    let dimension = angular_count * radial_count;
    let mut radial = vec![0.0; radial_count * radial_count];
    for left_radial in 0..radial_count {
        for right_radial in 0..radial_count {
            for angular in 0..angular_count {
                let left = angular * radial_count + left_radial;
                let right = angular * radial_count + right_radial;
                radial[left_radial * radial_count + right_radial] +=
                    resolved[left * dimension + right] / angular_count as f64;
            }
        }
    }
    let mut averaged = vec![0.0; dimension * dimension];
    for angular in 0..angular_count {
        for left_radial in 0..radial_count {
            let left = angular * radial_count + left_radial;
            for right_radial in 0..radial_count {
                let right = angular * radial_count + right_radial;
                averaged[left * dimension + right] =
                    radial[left_radial * radial_count + right_radial];
            }
        }
    }
    averaged
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::Bohr;

    const TOLERANCE: f64 = 2.0e-12;

    fn mesh() -> ExponentialMesh {
        ExponentialMesh::new(Bohr(1.0e-4), 0.16, 58).unwrap()
    }

    fn radial(mesh: &ExponentialMesh, l: i32, decay: f64, q_scale: f64) -> (Vec<f64>, Vec<f64>) {
        let p = mesh
            .radii()
            .iter()
            .map(|radius| radius.get().powi(l + 1) * (-decay * radius.get()).exp())
            .collect::<Vec<_>>();
        let q = mesh
            .radii()
            .iter()
            .map(|radius| q_scale * radius.get().powi(l + 2) * (-decay * radius.get()).exp())
            .collect::<Vec<_>>();
        (p, q)
    }

    fn normalization(mesh: &ExponentialMesh, p: &[f64], q: &[f64]) -> f64 {
        mesh.integrate(
            &p.iter()
                .zip(q)
                .map(|(p, q)| p * p + q * q)
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    #[test]
    fn scalar_average_matches_spex_qfac0_reduction() {
        let mesh = mesh();
        let (core_p, core_q) = radial(&mesh, 1, 1.2, 0.08);
        let (u_p, u_q) = radial(&mesh, 1, 0.7, 0.04);
        let core = StaticCoreExchangeShell {
            kappa: Kappa::new(-2).unwrap(),
            p: &core_p,
            q: &core_q,
            normalization: normalization(&mesh, &core_p, &core_q),
            occupation_per_mu: 1.0,
        };
        let valence = [ScalarCoreExchangeRadial {
            p: &u_p,
            q: Some(&u_q),
        }];
        let block = static_core_exchange_block(
            &mesh,
            1,
            &valence,
            &[core],
            StaticCoreExchangeMode::ScalarAverage,
        )
        .unwrap();

        let product = mesh
            .radii()
            .iter()
            .enumerate()
            .map(|(index, radius)| {
                (core_p[index] * u_p[index] + core_q[index] * u_q[index])
                    / (radius.get() * core.normalization.sqrt())
            })
            .collect::<Vec<_>>();
        let mut expected = 0.0;
        for multipole_l in (0..=2).step_by(2) {
            let radial = intra_sphere_poisson(multipole_l, &mesh, &product, &product).unwrap();
            let angular = gaunt(1, 1, multipole_l, 0, 0, 0)
                * (((2 * 1 + 1) as f64 * (2 * multipole_l + 1) as f64)
                    / (4.0 * std::f64::consts::PI * (2 * 1 + 1) as f64))
                    .sqrt()
                * 2.0
                / (2 * 1 + 1) as f64;
            expected -= angular * radial;
        }
        for spin in [SpinProjection::Up, SpinProjection::Down] {
            for m in -1..=1 {
                let coordinate = block.coordinate(spin, m, 0).unwrap();
                assert!((block.matrix.at(coordinate, coordinate).re - expected).abs() < TOLERANCE);
            }
        }
    }

    #[test]
    fn complete_identical_p_core_is_spin_and_orbital_scalar() {
        let mesh = mesh();
        let (core_p, core_q) = radial(&mesh, 1, 1.1, 0.06);
        let (u_p, u_q) = radial(&mesh, 1, 0.8, 0.03);
        let normalization = normalization(&mesh, &core_p, &core_q);
        let cores = [
            StaticCoreExchangeShell {
                kappa: Kappa::new(-2).unwrap(),
                p: &core_p,
                q: &core_q,
                normalization,
                occupation_per_mu: 1.0,
            },
            StaticCoreExchangeShell {
                kappa: Kappa::new(1).unwrap(),
                p: &core_p,
                q: &core_q,
                normalization,
                occupation_per_mu: 1.0,
            },
        ];
        let valence = [ScalarCoreExchangeRadial {
            p: &u_p,
            q: Some(&u_q),
        }];
        let resolved = static_core_exchange_block(
            &mesh,
            1,
            &valence,
            &cores,
            StaticCoreExchangeMode::SpinOrbitResolved,
        )
        .unwrap();
        let averaged = static_core_exchange_block(
            &mesh,
            1,
            &valence,
            &cores,
            StaticCoreExchangeMode::ScalarAverage,
        )
        .unwrap();
        for row in 0..resolved.matrix.dimension() {
            for column in 0..resolved.matrix.dimension() {
                assert!(
                    (resolved.matrix.at(row, column) - averaged.matrix.at(row, column)).norm()
                        < TOLERANCE
                );
            }
        }
    }

    #[test]
    fn split_p_core_retains_soc_operator_only_in_resolved_mode() {
        let mesh = mesh();
        let (core_p, core_q) = radial(&mesh, 1, 1.1, 0.06);
        let (u_p, u_q) = radial(&mesh, 1, 0.8, 0.03);
        let core = StaticCoreExchangeShell {
            kappa: Kappa::new(-2).unwrap(),
            p: &core_p,
            q: &core_q,
            normalization: normalization(&mesh, &core_p, &core_q),
            occupation_per_mu: 1.0,
        };
        let valence = [ScalarCoreExchangeRadial {
            p: &u_p,
            q: Some(&u_q),
        }];
        let resolved = static_core_exchange_block(
            &mesh,
            1,
            &valence,
            &[core],
            StaticCoreExchangeMode::SpinOrbitResolved,
        )
        .unwrap();
        let averaged = static_core_exchange_block(
            &mesh,
            1,
            &valence,
            &[core],
            StaticCoreExchangeMode::ScalarAverage,
        )
        .unwrap();
        let up_m0 = resolved.coordinate(SpinProjection::Up, 0, 0).unwrap();
        let down_m1 = resolved.coordinate(SpinProjection::Down, 1, 0).unwrap();
        assert!(resolved.matrix.at(up_m0, down_m1).norm() > 1.0e-8);
        assert_eq!(averaged.matrix.at(up_m0, down_m1), Complex64::new(0.0, 0.0));
    }
}
