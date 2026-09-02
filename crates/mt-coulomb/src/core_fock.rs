//! Channel-reduced radial core-core Fock actions for spherical shells.

use std::f64::consts::PI;

use muffintin_core::{ExponentialMesh, Hartree, Kappa, Lm, RelativisticChannel, spinor_gaunt};
use thiserror::Error;

use crate::{CoulombError, radial_primitive};

/// One physical core shell with one occupation shared by every magnetic channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreCoreFockShell<'a> {
    pub kappa: Kappa,
    pub p: &'a [f64],
    pub q: &'a [f64],
    /// Explicit full-mesh normalization of the physical `P/Q` pair.
    pub normalization: f64,
    /// Occupation of each magnetic channel in this shell.
    pub occupation_per_mu: f64,
}

/// Already-signed physical radial action `(K psi)_P/(K psi)_Q` for one shell.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreCoreFockAction {
    pub p: Vec<f64>,
    pub q: Vec<f64>,
}

/// PP, QQ, and Coulomb interference contributions to the occupied CC trace.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CoreCoreFockTrace {
    pub pp: Hartree,
    pub qq: Hartree,
    pub pp_qq_interference: Hartree,
    pub total: Hartree,
}

/// Channel-reduced actions in input-shell order and their occupied trace.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CoreCoreFockResult {
    pub actions: Vec<CoreCoreFockAction>,
    pub trace: CoreCoreFockTrace,
}

/// Invalid radial action input or Coulomb primitive failure.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum CoreCoreFockError {
    #[error("core-core Fock shell {shell} has an inconsistent P/Q radial length")]
    RadialLength { shell: usize },
    #[error("core-core Fock shell {shell} has invalid full normalization {value}")]
    Normalization { shell: usize, value: f64 },
    #[error("core-core Fock shell {shell} has invalid per-mu occupation {value}")]
    Occupation { shell: usize, value: f64 },
    #[error(transparent)]
    Coulomb(#[from] CoulombError),
}

#[derive(Clone)]
struct SplitAction {
    p_from_pp: Vec<f64>,
    p_from_qq: Vec<f64>,
    q_from_pp: Vec<f64>,
    q_from_qq: Vec<f64>,
}

impl SplitAction {
    fn zeros(length: usize) -> Self {
        Self {
            p_from_pp: vec![0.0; length],
            p_from_qq: vec![0.0; length],
            q_from_pp: vec![0.0; length],
            q_from_qq: vec![0.0; length],
        }
    }

    fn total(&self) -> CoreCoreFockAction {
        CoreCoreFockAction {
            p: self
                .p_from_pp
                .iter()
                .zip(&self.p_from_qq)
                .map(|(pp, qq)| pp + qq)
                .collect(),
            q: self
                .q_from_pp
                .iter()
                .zip(&self.q_from_qq)
                .map(|(pp, qq)| pp + qq)
                .collect(),
        }
    }
}

/// Build closed-shell radial CC Fock actions on one complete extended mesh.
///
/// The source occupation is applied here exactly once. The returned action
/// already carries the exchange minus sign. A uniform average over the target
/// magnetic channels produces the one-dimensional action shared by the shell;
/// the target occupation enters only when the returned trace is formed.
pub fn core_core_fock_actions(
    mesh: &ExponentialMesh,
    shells: &[CoreCoreFockShell<'_>],
) -> Result<CoreCoreFockResult, CoreCoreFockError> {
    validate_shells(mesh, shells)?;

    let split = shells
        .iter()
        .map(|target| build_shell_action(mesh, shells, *target))
        .collect::<Result<Vec<_>, _>>()?;
    let trace = action_trace(mesh, shells, &split)?;
    let actions = split.iter().map(SplitAction::total).collect();
    Ok(CoreCoreFockResult { actions, trace })
}

fn validate_shells(
    mesh: &ExponentialMesh,
    shells: &[CoreCoreFockShell<'_>],
) -> Result<(), CoreCoreFockError> {
    for (shell, input) in shells.iter().enumerate() {
        if input.p.len() != mesh.len()
            || input.q.len() != mesh.len()
            || input
                .p
                .iter()
                .chain(input.q)
                .any(|value| !value.is_finite())
        {
            return Err(CoreCoreFockError::RadialLength { shell });
        }
        if !input.normalization.is_finite() || input.normalization <= 0.0 {
            return Err(CoreCoreFockError::Normalization {
                shell,
                value: input.normalization,
            });
        }
        if !input.occupation_per_mu.is_finite() || !(0.0..=1.0).contains(&input.occupation_per_mu) {
            return Err(CoreCoreFockError::Occupation {
                shell,
                value: input.occupation_per_mu,
            });
        }
    }
    Ok(())
}

fn build_shell_action(
    mesh: &ExponentialMesh,
    shells: &[CoreCoreFockShell<'_>],
    target: CoreCoreFockShell<'_>,
) -> Result<SplitAction, CoreCoreFockError> {
    let mut action = SplitAction::zeros(mesh.len());
    let target_channels = target.kappa.twice_mu_values().collect::<Vec<_>>();
    let target_average = 1.0 / target_channels.len() as f64;
    let target_scale = target.normalization.sqrt().recip();

    for target_mu in target_channels {
        let target_channel = RelativisticChannel::new(target.kappa, target_mu)
            .expect("kappa supplies its own valid magnetic channels");
        for source in shells {
            let source_scale = source.normalization.sqrt().recip();
            let action_scale = -source.occupation_per_mu * target_average;
            for source_mu in source.kappa.twice_mu_values() {
                let source_channel = RelativisticChannel::new(source.kappa, source_mu)
                    .expect("kappa supplies its own valid magnetic channels");
                let l_max = pair_l_max(target_channel, source_channel);
                for l in 0..=l_max {
                    for m in -(l as i32)..=(l as i32) {
                        let pp_angular = density_angular(target_channel, l, m, source_channel);
                        let qq_angular = density_angular(
                            target_channel.opposite_kappa(),
                            l,
                            m,
                            source_channel.opposite_kappa(),
                        );
                        if pp_angular == 0.0 && qq_angular == 0.0 {
                            continue;
                        }
                        let mut pair_pp = Vec::with_capacity(mesh.len());
                        let mut pair_qq = Vec::with_capacity(mesh.len());
                        for index in 0..mesh.len() {
                            let inverse_radius = mesh.radii()[index].get().recip();
                            pair_pp.push(
                                pp_angular
                                    * target.p[index]
                                    * source.p[index]
                                    * target_scale
                                    * source_scale
                                    * inverse_radius,
                            );
                            pair_qq.push(
                                qq_angular
                                    * target.q[index]
                                    * source.q[index]
                                    * target_scale
                                    * source_scale
                                    * inverse_radius,
                            );
                        }
                        let potential_pp = multipole_potential(l, mesh, &pair_pp)?;
                        let potential_qq = multipole_potential(l, mesh, &pair_qq)?;
                        for index in 0..mesh.len() {
                            let inverse_radius = mesh.radii()[index].get().recip();
                            let source_p = source.p[index] * source_scale * inverse_radius;
                            let source_q = source.q[index] * source_scale * inverse_radius;
                            action.p_from_pp[index] +=
                                action_scale * pp_angular * source_p * potential_pp[index];
                            action.p_from_qq[index] +=
                                action_scale * pp_angular * source_p * potential_qq[index];
                            action.q_from_pp[index] +=
                                action_scale * qq_angular * source_q * potential_pp[index];
                            action.q_from_qq[index] +=
                                action_scale * qq_angular * source_q * potential_qq[index];
                        }
                    }
                }
            }
        }
    }
    Ok(action)
}

fn action_trace(
    mesh: &ExponentialMesh,
    shells: &[CoreCoreFockShell<'_>],
    actions: &[SplitAction],
) -> Result<CoreCoreFockTrace, CoreCoreFockError> {
    let mut pp = 0.0;
    let mut qq = 0.0;
    let mut total = 0.0;
    for (shell, action) in shells.iter().zip(actions) {
        let target_weight = shell.occupation_per_mu * f64::from(shell.kappa.degeneracy());
        let target_scale = shell.normalization.sqrt().recip();
        let pp_integrand = shell
            .p
            .iter()
            .zip(&action.p_from_pp)
            .map(|(target, value)| target_scale * target * value)
            .collect::<Vec<_>>();
        let qq_integrand = shell
            .q
            .iter()
            .zip(&action.q_from_qq)
            .map(|(target, value)| target_scale * target * value)
            .collect::<Vec<_>>();
        let total_integrand = (0..mesh.len())
            .map(|index| {
                target_scale
                    * (shell.p[index] * (action.p_from_pp[index] + action.p_from_qq[index])
                        + shell.q[index] * (action.q_from_pp[index] + action.q_from_qq[index]))
            })
            .collect::<Vec<_>>();
        pp += target_weight * mesh.integrate(&pp_integrand).map_err(CoulombError::from)?;
        qq += target_weight * mesh.integrate(&qq_integrand).map_err(CoulombError::from)?;
        total += target_weight
            * mesh
                .integrate(&total_integrand)
                .map_err(CoulombError::from)?;
    }
    Ok(CoreCoreFockTrace {
        pp: Hartree(pp),
        qq: Hartree(qq),
        pp_qq_interference: Hartree(total - pp - qq),
        total: Hartree(total),
    })
}

fn multipole_potential(
    l: u32,
    mesh: &ExponentialMesh,
    radial: &[f64],
) -> Result<Vec<f64>, CoreCoreFockError> {
    let mut outward_integrand = Vec::with_capacity(mesh.len());
    let mut inward_integrand = Vec::with_capacity(mesh.len());
    for (radius, value) in mesh.radii().iter().zip(radial) {
        let r = radius.get();
        let power = r.powi(l as i32);
        outward_integrand.push(value * power * r);
        inward_integrand.push(if l == 0 { *value } else { value / power });
    }
    let outward = radial_primitive(mesh, &outward_integrand, false)?;
    let inward = radial_primitive(mesh, &inward_integrand, true)?;
    let angular_scale = 4.0 * PI / (2.0 * f64::from(l) + 1.0);
    Ok(mesh
        .radii()
        .iter()
        .enumerate()
        .map(|(index, radius)| {
            let r = radius.get();
            let power = r.powi(l as i32);
            let inner = if l == 0 {
                outward[index]
            } else {
                outward[index] / power
            };
            angular_scale * (inner + inward[index] * power * r)
        })
        .collect())
}

fn pair_l_max(left: RelativisticChannel, right: RelativisticChannel) -> u32 {
    (left.kappa().large_l() + right.kappa().large_l())
        .max(left.kappa().small_l() + right.kappa().small_l())
}

/// Stored density coefficient `(-1)^M <Omega_left|Y_(L,-M)|Omega_right>`.
fn density_angular(left: RelativisticChannel, l: u32, m: i32, right: RelativisticChannel) -> f64 {
    let field = Lm::new(l, -m).expect("magnetic channel lies in [-L, L]");
    let phase = if m.unsigned_abs().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    phase * spinor_gaunt(left, field, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BorrowedCoreShell, ClosedCoreOccupations, PreweightedSiteValenceDensity, RadialSlaterSite,
        radial_slater_traces,
    };
    use muffintin_core::{Bohr, TwiceMu};

    #[test]
    fn multi_kappa_nonzero_q_action_trace_matches_independent_slater_oracle() {
        // kappa -2 and 2 carry small-component l of 2 and 3, so pair_l_max
        // reaches 6 here. That is the only coverage of the high-l spinor
        // Gaunt channels the radial Slater oracle skips when they vanish.
        let mesh = ExponentialMesh::new(Bohr(1.0e-4), 0.18, 55).unwrap();
        let p_s = mesh
            .radii()
            .iter()
            .map(|radius| radius.get() * (-1.1 * radius.get()).exp())
            .collect::<Vec<_>>();
        let q_s = mesh
            .radii()
            .iter()
            .map(|radius| 0.22 * radius.get().powi(2) * (-1.1 * radius.get()).exp())
            .collect::<Vec<_>>();
        let p_p = mesh
            .radii()
            .iter()
            .map(|radius| radius.get().powi(2) * (-0.75 * radius.get()).exp())
            .collect::<Vec<_>>();
        let q_p = mesh
            .radii()
            .iter()
            .map(|radius| -0.17 * radius.get() * (-0.75 * radius.get()).exp())
            .collect::<Vec<_>>();
        let normalization = |p: &[f64], q: &[f64]| {
            mesh.integrate(
                &p.iter()
                    .zip(q)
                    .map(|(large, small)| large * large + small * small)
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let p_p32 = mesh
            .radii()
            .iter()
            .map(|radius| radius.get().powi(2) * (-0.9 * radius.get()).exp())
            .collect::<Vec<_>>();
        let q_p32 = mesh
            .radii()
            .iter()
            .map(|radius| 0.13 * radius.get().powi(3) * (-0.9 * radius.get()).exp())
            .collect::<Vec<_>>();
        let p_d32 = mesh
            .radii()
            .iter()
            .map(|radius| radius.get().powi(3) * (-0.6 * radius.get()).exp())
            .collect::<Vec<_>>();
        let q_d32 = mesh
            .radii()
            .iter()
            .map(|radius| -0.11 * radius.get().powi(4) * (-0.6 * radius.get()).exp())
            .collect::<Vec<_>>();
        let kappa_s = Kappa::new(-1).unwrap();
        let kappa_p = Kappa::new(1).unwrap();
        let kappa_p32 = Kappa::new(-2).unwrap();
        let kappa_d32 = Kappa::new(2).unwrap();
        let shell_inputs = [
            CoreCoreFockShell {
                kappa: kappa_s,
                p: &p_s,
                q: &q_s,
                normalization: normalization(&p_s, &q_s),
                occupation_per_mu: 0.8,
            },
            CoreCoreFockShell {
                kappa: kappa_p,
                p: &p_p,
                q: &q_p,
                normalization: normalization(&p_p, &q_p),
                occupation_per_mu: 0.65,
            },
            CoreCoreFockShell {
                kappa: kappa_p32,
                p: &p_p32,
                q: &q_p32,
                normalization: normalization(&p_p32, &q_p32),
                occupation_per_mu: 0.55,
            },
            CoreCoreFockShell {
                kappa: kappa_d32,
                p: &p_d32,
                q: &q_d32,
                normalization: normalization(&p_d32, &q_d32),
                occupation_per_mu: 0.45,
            },
        ];
        let actions = core_core_fock_actions(&mesh, &shell_inputs).unwrap();

        let occupations_s = kappa_s
            .twice_mu_values()
            .map(|mu| (mu, 0.8))
            .collect::<Vec<(TwiceMu, f64)>>();
        let occupations_p = kappa_p
            .twice_mu_values()
            .map(|mu| (mu, 0.65))
            .collect::<Vec<(TwiceMu, f64)>>();
        let occupations_p32 = kappa_p32
            .twice_mu_values()
            .map(|mu| (mu, 0.55))
            .collect::<Vec<(TwiceMu, f64)>>();
        let occupations_d32 = kappa_d32
            .twice_mu_values()
            .map(|mu| (mu, 0.45))
            .collect::<Vec<(TwiceMu, f64)>>();
        let oracle_shells = [
            BorrowedCoreShell {
                kappa: kappa_s,
                p: &p_s,
                q: &q_s,
                normalization: shell_inputs[0].normalization,
                occupations: ClosedCoreOccupations::MuResolved(&occupations_s),
            },
            BorrowedCoreShell {
                kappa: kappa_p,
                p: &p_p,
                q: &q_p,
                normalization: shell_inputs[1].normalization,
                occupations: ClosedCoreOccupations::MuResolved(&occupations_p),
            },
            BorrowedCoreShell {
                kappa: kappa_p32,
                p: &p_p32,
                q: &q_p32,
                normalization: shell_inputs[2].normalization,
                occupations: ClosedCoreOccupations::MuResolved(&occupations_p32),
            },
            BorrowedCoreShell {
                kappa: kappa_d32,
                p: &p_d32,
                q: &q_d32,
                normalization: shell_inputs[3].normalization,
                occupations: ClosedCoreOccupations::MuResolved(&occupations_d32),
            },
        ];
        let oracle = radial_slater_traces(&[RadialSlaterSite {
            site_index: 0,
            mt_mesh: &mesh,
            extended_mesh: &mesh,
            cores: &oracle_shells,
            valence: PreweightedSiteValenceDensity {
                orbitals: &[],
                matrix: &[],
            },
        }])
        .unwrap();

        assert_eq!(actions.actions.len(), 4);
        assert!(actions.actions.iter().any(|action| {
            action
                .q
                .iter()
                .any(|value| value.abs() > 64.0 * f64::EPSILON)
        }));
        assert!((actions.trace.pp.get() - oracle.cc_extended.pp.get()).abs() < 2.0e-12);
        assert!((actions.trace.qq.get() - oracle.cc_extended.qq.get()).abs() < 2.0e-12);
        assert!(
            (actions.trace.pp_qq_interference.get() - oracle.cc_extended.pp_qq_interference.get())
                .abs()
                < 2.0e-12
        );
        assert!((actions.trace.total.get() - oracle.cc_extended.total.get()).abs() < 2.0e-12);
        assert!(actions.trace.total.get() < 0.0);
    }
}
