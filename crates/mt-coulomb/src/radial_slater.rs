//! Independent trace-only relativistic radial Slater oracle.

use crate::{CoulombError, intra_sphere_poisson};
use muffintin_core::{
    ExponentialMesh, Hartree, Kappa, Lm, RelativisticChannel, TwiceMu, spinor_gaunt,
};
use num_complex::Complex64;
use thiserror::Error;

const HERMITICITY_TOLERANCE: f64 = 1.0e-10;

/// Explicit magnetic occupation representation for one core shell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClosedCoreOccupations<'a> {
    MuResolved(&'a [(TwiceMu, f64)]),
    ExplicitCollinear { up: f64, down: f64 },
}

/// Borrowed physical core P/Q on the site's extended solve mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BorrowedCoreShell<'a> {
    pub kappa: Kappa,
    pub p: &'a [f64],
    pub q: &'a [f64],
    /// The fixed full-mesh norm used by both MT and extended traces.
    pub normalization: f64,
    pub occupations: ClosedCoreOccupations<'a>,
}

/// Borrowed physical valence P/Q in one site-projection coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BorrowedValenceRadial<'a> {
    pub channel: RelativisticChannel,
    pub p: &'a [f64],
    pub q: &'a [f64],
    pub normalization: f64,
}

/// Site valence density after all k weights and valence occupations are applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreweightedSiteValenceDensity<'a> {
    pub orbitals: &'a [BorrowedValenceRadial<'a>],
    /// Row-major Hermitian density matrix in `orbitals` order.
    pub matrix: &'a [Complex64],
}

/// One site's borrowed physical radial data for the independent trace oracle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialSlaterSite<'a> {
    pub site_index: usize,
    pub mt_mesh: &'a ExponentialMesh,
    pub extended_mesh: &'a ExponentialMesh,
    pub cores: &'a [BorrowedCoreShell<'a>],
    pub valence: PreweightedSiteValenceDensity<'a>,
}

/// PP, QQ, and their Coulomb interference, all in Hartree trace units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RadialSlaterComponents {
    pub pp: Hartree,
    pub qq: Hartree,
    pub pp_qq_interference: Hartree,
    pub total: Hartree,
}

/// Independent core-valence radial exchange trace.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RadialSlaterCvTraces {
    pub cv_mt: RadialSlaterComponents,
    pub cv_imaginary_residual: f64,
}

/// Independent core-core and core-valence radial exchange traces.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RadialSlaterTraces {
    pub cc_mt: RadialSlaterComponents,
    pub cc_extended: RadialSlaterComponents,
    /// Measured `|T_cc(extended)-T_cc(MT)|`; never a numerical tolerance.
    pub cc_spill_allowance: Hartree,
    pub cv_mt: RadialSlaterComponents,
    pub cv_imaginary_residual: f64,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum RadialSlaterError {
    #[error(transparent)]
    Coulomb(#[from] CoulombError),
    #[error(
        "radial Slater site {site} extended mesh is not an exact prefix extension of its MT mesh"
    )]
    MeshPrefix { site: usize },
    #[error("radial Slater site {site} has an inconsistent P/Q radial length")]
    RadialLength { site: usize },
    #[error("radial Slater site {site} has invalid explicit radial normalization {value}")]
    Normalization { site: usize, value: f64 },
    #[error("radial Slater site {site} core shell uses ExplicitCollinear occupations")]
    ExplicitCollinear { site: usize },
    #[error(
        "radial Slater site {site} core shell does not contain every magnetic channel exactly once"
    )]
    MagneticChannels { site: usize },
    #[error("radial Slater site {site} core shell is not closed: magnetic occupations differ")]
    OpenShell { site: usize },
    #[error("radial Slater site {site} has an invalid core occupation {value}")]
    Occupation { site: usize, value: f64 },
    #[error(
        "radial Slater site {site} valence density has dimension {actual}, expected {expected}"
    )]
    DensityDimension {
        site: usize,
        actual: usize,
        expected: usize,
    },
    #[error("radial Slater site {site} valence density is not finite Hermitian")]
    DensityNotHermitian { site: usize },
}

#[derive(Clone, Copy)]
struct OrbitalRef<'a> {
    channel: RelativisticChannel,
    p: &'a [f64],
    q: &'a [f64],
    normalization: f64,
}

#[derive(Clone, Copy, Default)]
struct Components {
    pp: f64,
    qq: f64,
    total: f64,
}

#[derive(Clone, Copy, Default)]
struct ComplexComponents {
    pp: Complex64,
    qq: Complex64,
    total: Complex64,
}

impl Components {
    fn scaled_add(&mut self, scale: f64, value: Self) {
        self.pp += scale * value.pp;
        self.qq += scale * value.qq;
        self.total += scale * value.total;
    }

    fn into_public(self) -> RadialSlaterComponents {
        RadialSlaterComponents {
            pp: Hartree(self.pp),
            qq: Hartree(self.qq),
            pp_qq_interference: Hartree(self.total - self.pp - self.qq),
            total: Hartree(self.total),
        }
    }
}

impl ComplexComponents {
    fn scaled_add(&mut self, scale: Complex64, value: Components) {
        self.pp += scale * value.pp;
        self.qq += scale * value.qq;
        self.total += scale * value.total;
    }

    fn real(self) -> Components {
        Components {
            pp: self.pp.re,
            qq: self.qq.re,
            total: self.total.re,
        }
    }
}

/// Evaluate trace-only CC and CV radial Slater exchange without MPB products.
pub fn radial_slater_traces(
    sites: &[RadialSlaterSite<'_>],
) -> Result<RadialSlaterTraces, RadialSlaterError> {
    let mut cc_mt = Components::default();
    let mut cc_extended = Components::default();
    let mut cv_mt = ComplexComponents::default();
    for site in sites {
        validate_site(site)?;
        let cores = expand_cores(site)?;
        for &(left, left_occupation) in &cores {
            for &(right, right_occupation) in &cores {
                let weight = -left_occupation * right_occupation;
                let left_mt = truncate(left, site.mt_mesh.len());
                let right_mt = truncate(right, site.mt_mesh.len());
                cc_mt.scaled_add(
                    weight,
                    slater_integral(site.mt_mesh, left_mt, right_mt, left_mt, right_mt)?,
                );
                cc_extended.scaled_add(
                    weight,
                    slater_integral(site.extended_mesh, left, right, left, right)?,
                );
            }
        }

        accumulate_cv(site, &cores, &mut cv_mt)?;
    }
    let cc_mt = cc_mt.into_public();
    let cc_extended = cc_extended.into_public();
    let cv_imaginary_residual = cv_mt.total.im.abs();
    let cv_mt = cv_mt.real().into_public();
    Ok(RadialSlaterTraces {
        cc_spill_allowance: Hartree((cc_extended.total.get() - cc_mt.total.get()).abs()),
        cc_mt,
        cc_extended,
        cv_mt,
        cv_imaginary_residual,
    })
}

/// Evaluate only the core-valence half of [`radial_slater_traces`].
///
/// The core-core half is an independent reimplementation of
/// `core_core_fock_actions` over core orbitals that are fixed inside one SCF
/// step. It is cross-checked in
/// `core_fock::tests::multi_kappa_nonzero_q_action_trace_matches_independent_slater_oracle`
/// and is not evaluated on the SCF path.
pub fn radial_slater_cv_traces(
    sites: &[RadialSlaterSite<'_>],
) -> Result<RadialSlaterCvTraces, RadialSlaterError> {
    let mut cv_mt = ComplexComponents::default();
    for site in sites {
        validate_site(site)?;
        let cores = expand_cores(site)?;
        accumulate_cv(site, &cores, &mut cv_mt)?;
    }
    let cv_imaginary_residual = cv_mt.total.im.abs();
    Ok(RadialSlaterCvTraces {
        cv_mt: cv_mt.real().into_public(),
        cv_imaginary_residual,
    })
}

fn accumulate_cv(
    site: &RadialSlaterSite<'_>,
    cores: &[(OrbitalRef<'_>, f64)],
    cv_mt: &mut ComplexComponents,
) -> Result<(), RadialSlaterError> {
    let valence = site
        .valence
        .orbitals
        .iter()
        .map(|orbital| OrbitalRef {
            channel: orbital.channel,
            p: orbital.p,
            q: orbital.q,
            normalization: orbital.normalization,
        })
        .collect::<Vec<_>>();
    for &(core, core_occupation) in cores {
        let core = truncate(core, site.mt_mesh.len());
        let metric = hermitian_cv_metric(site.mt_mesh, core, &valence)?;
        for left_index in 0..valence.len() {
            for right_index in 0..valence.len() {
                let density = site.valence.matrix[right_index * valence.len() + left_index];
                cv_mt.scaled_add(
                    -core_occupation * density,
                    metric[left_index * valence.len() + right_index],
                );
            }
        }
    }
    Ok(())
}

fn hermitian_cv_metric(
    mesh: &ExponentialMesh,
    core: OrbitalRef<'_>,
    valence: &[OrbitalRef<'_>],
) -> Result<Vec<Components>, RadialSlaterError> {
    let n = valence.len();
    let mut metric = vec![Components::default(); n * n];
    for left in 0..n {
        for right in left..n {
            let forward = slater_integral(mesh, core, valence[left], core, valence[right])?;
            let value = if left == right {
                forward
            } else {
                let reverse = slater_integral(mesh, core, valence[right], core, valence[left])?;
                Components {
                    pp: 0.5 * (forward.pp + reverse.pp),
                    qq: 0.5 * (forward.qq + reverse.qq),
                    total: 0.5 * (forward.total + reverse.total),
                }
            };
            metric[left * n + right] = value;
            metric[right * n + left] = value;
        }
    }
    Ok(metric)
}

fn validate_site(site: &RadialSlaterSite<'_>) -> Result<(), RadialSlaterError> {
    if site.extended_mesh.len() < site.mt_mesh.len()
        || site.extended_mesh.radii()[..site.mt_mesh.len()] != site.mt_mesh.radii()[..]
    {
        return Err(RadialSlaterError::MeshPrefix {
            site: site.site_index,
        });
    }
    for core in site.cores {
        validate_radial(
            site.site_index,
            site.extended_mesh.len(),
            core.p,
            core.q,
            core.normalization,
        )?;
    }
    for valence in site.valence.orbitals {
        validate_radial(
            site.site_index,
            site.mt_mesh.len(),
            valence.p,
            valence.q,
            valence.normalization,
        )?;
    }
    let n = site.valence.orbitals.len();
    let expected = n.checked_mul(n).unwrap_or(usize::MAX);
    if site.valence.matrix.len() != expected {
        return Err(RadialSlaterError::DensityDimension {
            site: site.site_index,
            actual: site.valence.matrix.len(),
            expected,
        });
    }
    for row in 0..n {
        for column in 0..n {
            let value = site.valence.matrix[row * n + column];
            let reverse = site.valence.matrix[column * n + row].conj();
            if !value.re.is_finite()
                || !value.im.is_finite()
                || (value - reverse).norm() > HERMITICITY_TOLERANCE
            {
                return Err(RadialSlaterError::DensityNotHermitian {
                    site: site.site_index,
                });
            }
        }
    }
    Ok(())
}

fn validate_radial(
    site: usize,
    expected: usize,
    p: &[f64],
    q: &[f64],
    normalization: f64,
) -> Result<(), RadialSlaterError> {
    if p.len() != expected
        || q.len() != expected
        || p.iter().chain(q).any(|value| !value.is_finite())
    {
        return Err(RadialSlaterError::RadialLength { site });
    }
    if !normalization.is_finite() || normalization <= 0.0 {
        return Err(RadialSlaterError::Normalization {
            site,
            value: normalization,
        });
    }
    Ok(())
}

fn expand_cores<'a>(
    site: &'a RadialSlaterSite<'a>,
) -> Result<Vec<(OrbitalRef<'a>, f64)>, RadialSlaterError> {
    let mut expanded = Vec::new();
    for shell in site.cores {
        let ClosedCoreOccupations::MuResolved(occupations) = shell.occupations else {
            return Err(RadialSlaterError::ExplicitCollinear {
                site: site.site_index,
            });
        };
        let expected = shell.kappa.twice_mu_values().collect::<Vec<_>>();
        if occupations.len() != expected.len()
            || expected
                .iter()
                .any(|mu| occupations.iter().filter(|(found, _)| found == mu).count() != 1)
        {
            return Err(RadialSlaterError::MagneticChannels {
                site: site.site_index,
            });
        }
        let reference = occupations[0].1;
        if occupations
            .iter()
            .any(|(_, value)| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            let value = occupations
                .iter()
                .find_map(|(_, value)| {
                    (!value.is_finite() || !(0.0..=1.0).contains(value)).then_some(*value)
                })
                .expect("invalid occupation was found");
            return Err(RadialSlaterError::Occupation {
                site: site.site_index,
                value,
            });
        }
        if occupations
            .iter()
            .any(|(_, value)| value.to_bits() != reference.to_bits())
        {
            return Err(RadialSlaterError::OpenShell {
                site: site.site_index,
            });
        }
        for &mu in &expected {
            expanded.push((
                OrbitalRef {
                    channel: RelativisticChannel::new(shell.kappa, mu)
                        .expect("kappa supplies its own valid magnetic channels"),
                    p: shell.p,
                    q: shell.q,
                    normalization: shell.normalization,
                },
                reference,
            ));
        }
    }
    Ok(expanded)
}

fn truncate(orbital: OrbitalRef<'_>, length: usize) -> OrbitalRef<'_> {
    OrbitalRef {
        channel: orbital.channel,
        p: &orbital.p[..length],
        q: &orbital.q[..length],
        normalization: orbital.normalization,
    }
}

fn slater_integral(
    mesh: &ExponentialMesh,
    left_bra: OrbitalRef<'_>,
    left_ket: OrbitalRef<'_>,
    right_bra: OrbitalRef<'_>,
    right_ket: OrbitalRef<'_>,
) -> Result<Components, RadialSlaterError> {
    let l_max = pair_l_max(left_bra.channel, left_ket.channel)
        .max(pair_l_max(right_bra.channel, right_ket.channel));
    let mut value = Components::default();
    for l in 0..=l_max {
        for m in -(l as i32)..=(l as i32) {
            // The spinor Gaunt selection rule leaves one nonzero m per l.
            // Both sides enter every component bilinearly, so a vanishing
            // angular pair contributes an exact zero and the radial
            // quadrature below is skipped.
            let left_angular = pair_angular(left_bra, left_ket, l, m)?;
            let right_angular = pair_angular(right_bra, right_ket, l, m)?;
            if left_angular == (0.0, 0.0) || right_angular == (0.0, 0.0) {
                continue;
            }
            let left = pair_radials(mesh, left_bra, left_ket, left_angular);
            let right = pair_radials(mesh, right_bra, right_ket, right_angular);
            value.pp += intra_sphere_poisson(l, mesh, &left.0, &right.0)?;
            value.qq += intra_sphere_poisson(l, mesh, &left.1, &right.1)?;
            let left_total = left
                .0
                .iter()
                .zip(&left.1)
                .map(|(pp, qq)| pp + qq)
                .collect::<Vec<_>>();
            let right_total = right
                .0
                .iter()
                .zip(&right.1)
                .map(|(pp, qq)| pp + qq)
                .collect::<Vec<_>>();
            value.total += intra_sphere_poisson(l, mesh, &left_total, &right_total)?;
        }
    }
    Ok(value)
}

/// PP and QQ spinor-Gaunt weights of one density pair at `(l, m)`.
fn pair_angular(
    left: OrbitalRef<'_>,
    right: OrbitalRef<'_>,
    l: u32,
    m: i32,
) -> Result<(f64, f64), RadialSlaterError> {
    Ok((
        density_angular(left.channel, l, m, right.channel)?,
        density_angular(
            left.channel.opposite_kappa(),
            l,
            m,
            right.channel.opposite_kappa(),
        )?,
    ))
}

fn pair_radials(
    mesh: &ExponentialMesh,
    left: OrbitalRef<'_>,
    right: OrbitalRef<'_>,
    (pp_angular, qq_angular): (f64, f64),
) -> (Vec<f64>, Vec<f64>) {
    let scale = (left.normalization * right.normalization).sqrt();
    let pp = mesh
        .radii()
        .iter()
        .enumerate()
        .map(|(index, radius)| pp_angular * left.p[index] * right.p[index] / (radius.get() * scale))
        .collect();
    let qq = mesh
        .radii()
        .iter()
        .enumerate()
        .map(|(index, radius)| qq_angular * left.q[index] * right.q[index] / (radius.get() * scale))
        .collect();
    (pp, qq)
}

fn pair_l_max(left: RelativisticChannel, right: RelativisticChannel) -> u32 {
    (left.kappa().large_l() + right.kappa().large_l())
        .max(left.kappa().small_l() + right.kappa().small_l())
}

fn density_angular(
    left: RelativisticChannel,
    l: u32,
    m: i32,
    right: RelativisticChannel,
) -> Result<f64, RadialSlaterError> {
    let field = Lm::new(l, -m).map_err(CoulombError::from)?;
    let phase = if m.unsigned_abs().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    Ok(phase * spinor_gaunt(left, field, right))
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::Bohr;
    use std::f64::consts::PI;

    fn meshes() -> (ExponentialMesh, ExponentialMesh) {
        let first = Bohr(1.0e-4);
        let increment = (1.2_f64 / first.get()).ln() / 30.0;
        (
            ExponentialMesh::new(first, increment, 25).unwrap(),
            ExponentialMesh::new(first, increment, 31).unwrap(),
        )
    }

    #[test]
    fn closed_s_half_l0_matches_direct_quadrature_without_mt_renormalization() {
        let (mt_mesh, extended_mesh) = meshes();
        let p = extended_mesh
            .radii()
            .iter()
            .map(|radius| radius.get() * (-radius.get()).exp())
            .collect::<Vec<_>>();
        let q = vec![0.0; extended_mesh.len()];
        let norm_total = extended_mesh
            .integrate(&p.iter().map(|value| value * value).collect::<Vec<_>>())
            .unwrap();
        let occupations = [
            (TwiceMu::new(-1).unwrap(), 1.0),
            (TwiceMu::new(1).unwrap(), 1.0),
        ];
        let cores = [BorrowedCoreShell {
            kappa: Kappa::new(-1).unwrap(),
            p: &p,
            q: &q,
            normalization: norm_total,
            occupations: ClosedCoreOccupations::MuResolved(&occupations),
        }];
        let site = RadialSlaterSite {
            site_index: 0,
            mt_mesh: &mt_mesh,
            extended_mesh: &extended_mesh,
            cores: &cores,
            valence: PreweightedSiteValenceDensity {
                orbitals: &[],
                matrix: &[],
            },
        };
        let result = radial_slater_traces(&[site]).unwrap();
        let basm = mt_mesh
            .radii()
            .iter()
            .enumerate()
            .map(|(index, radius)| {
                p[index] * p[index] / ((4.0 * PI).sqrt() * radius.get() * norm_total)
            })
            .collect::<Vec<_>>();
        let direct = -2.0 * intra_sphere_poisson(0, &mt_mesh, &basm, &basm).unwrap();
        assert!((result.cc_mt.total.get() - direct).abs() < 1.0e-12);

        let norm_mt = mt_mesh
            .integrate(
                &p[..mt_mesh.len()]
                    .iter()
                    .map(|value| value * value)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let mt_renormalized = mt_mesh
            .radii()
            .iter()
            .enumerate()
            .map(|(index, radius)| {
                p[index] * p[index] / ((4.0 * PI).sqrt() * radius.get() * norm_mt)
            })
            .collect::<Vec<_>>();
        let old =
            -2.0 * intra_sphere_poisson(0, &mt_mesh, &mt_renormalized, &mt_renormalized).unwrap();
        assert!((result.cc_mt.total.get() - old).abs() > 1.0e-6);
        assert!(result.cc_spill_allowance.get() > 0.0);
    }

    #[test]
    fn nonzero_small_q_and_complex_hermitian_density_keep_pp_qq_separate() {
        let (mt_mesh, extended_mesh) = meshes();
        let core_p = extended_mesh
            .radii()
            .iter()
            .map(|radius| radius.get() * (-radius.get()).exp())
            .collect::<Vec<_>>();
        let core_q = extended_mesh
            .radii()
            .iter()
            .map(|radius| 0.3 * radius.get().powi(2) * (-radius.get()).exp())
            .collect::<Vec<_>>();
        let norm = extended_mesh
            .integrate(
                &core_p
                    .iter()
                    .zip(&core_q)
                    .map(|(p, q)| p * p + q * q)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let occupations = [
            (TwiceMu::new(-1).unwrap(), 0.75),
            (TwiceMu::new(1).unwrap(), 0.75),
        ];
        let cores = [BorrowedCoreShell {
            kappa: Kappa::new(-1).unwrap(),
            p: &core_p,
            q: &core_q,
            normalization: norm,
            occupations: ClosedCoreOccupations::MuResolved(&occupations),
        }];
        let valence_p0 = mt_mesh
            .radii()
            .iter()
            .map(|radius| radius.get() * (-0.7 * radius.get()).exp())
            .collect::<Vec<_>>();
        let valence_p1 = mt_mesh
            .radii()
            .iter()
            .map(|radius| radius.get().powi(2) * (-0.8 * radius.get()).exp())
            .collect::<Vec<_>>();
        let valence_q0 = valence_p0
            .iter()
            .map(|value| 0.2 * value)
            .collect::<Vec<_>>();
        let valence_q1 = valence_p1
            .iter()
            .map(|value| -0.1 * value)
            .collect::<Vec<_>>();
        let channel =
            RelativisticChannel::new(Kappa::new(-1).unwrap(), TwiceMu::new(-1).unwrap()).unwrap();
        let norm0 = mt_mesh
            .integrate(
                &valence_p0
                    .iter()
                    .zip(&valence_q0)
                    .map(|(p, q)| p * p + q * q)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let norm1 = mt_mesh
            .integrate(
                &valence_p1
                    .iter()
                    .zip(&valence_q1)
                    .map(|(p, q)| p * p + q * q)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let valence = [
            BorrowedValenceRadial {
                channel,
                p: &valence_p0,
                q: &valence_q0,
                normalization: norm0,
            },
            BorrowedValenceRadial {
                channel,
                p: &valence_p1,
                q: &valence_q1,
                normalization: norm1,
            },
        ];
        let density = [
            Complex64::new(0.7, 0.0),
            Complex64::new(0.0, 0.2),
            Complex64::new(0.0, -0.2),
            Complex64::new(0.4, 0.0),
        ];
        let core_channel =
            RelativisticChannel::new(Kappa::new(-1).unwrap(), TwiceMu::new(-1).unwrap()).unwrap();
        let core_ref = truncate(
            OrbitalRef {
                channel: core_channel,
                p: &core_p,
                q: &core_q,
                normalization: norm,
            },
            mt_mesh.len(),
        );
        let valence_refs = valence
            .iter()
            .map(|orbital| OrbitalRef {
                channel: orbital.channel,
                p: orbital.p,
                q: orbital.q,
                normalization: orbital.normalization,
            })
            .collect::<Vec<_>>();
        let forward = slater_integral(
            &mt_mesh,
            core_ref,
            valence_refs[0],
            core_ref,
            valence_refs[1],
        )
        .unwrap();
        let reverse = slater_integral(
            &mt_mesh,
            core_ref,
            valence_refs[1],
            core_ref,
            valence_refs[0],
        )
        .unwrap();
        assert!((forward.total - reverse.total).abs() > 1.0e-10);
        let metric = hermitian_cv_metric(&mt_mesh, core_ref, &valence_refs).unwrap();
        assert_eq!(metric[1].total, 0.5 * (forward.total + reverse.total));
        assert_eq!(metric[1].pp, metric[2].pp);
        assert_eq!(metric[1].qq, metric[2].qq);
        assert_eq!(metric[1].total, metric[2].total);
        let site = RadialSlaterSite {
            site_index: 0,
            mt_mesh: &mt_mesh,
            extended_mesh: &extended_mesh,
            cores: &cores,
            valence: PreweightedSiteValenceDensity {
                orbitals: &valence,
                matrix: &density,
            },
        };
        let result = radial_slater_traces(&[site]).unwrap();
        assert!(result.cc_mt.pp.get().abs() > 0.0);
        assert!(result.cc_mt.qq.get().abs() > 0.0);
        assert!(result.cv_mt.pp.get().abs() > 0.0);
        assert!(result.cv_mt.qq.get().abs() > 0.0);
        assert!(result.cv_imaginary_residual < 1.0e-12);
        assert!(
            (result.cv_mt.total.get()
                - result.cv_mt.pp.get()
                - result.cv_mt.qq.get()
                - result.cv_mt.pp_qq_interference.get())
            .abs()
                < 1.0e-14
        );
    }
}
