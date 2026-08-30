//! Independent Bloch-periodized $1/r$ Ewald kernel. Not Weinert assembly.

use crate::CoulombError;
use crate::math::plane_wave_phase;

use muffintin_prodbasis::TransferQ;
use muffintin_core::{Bohr, InverseBohr, ReciprocalLattice};
use muffintin_core::Cell;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Real/reciprocal cutoffs and splitting parameter for [`ewald_point_kernel`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EwaldSummation {
    /// Gaussian splitting parameter $\eta$ in bohr$^{-2}$.
    pub eta: f64,
    /// Real-space lattice-vector cutoff.
    pub real_cutoff: Bohr,
    /// Reciprocal-space cutoff.
    pub recip_cutoff: InverseBohr,
}

/// Successful cutoff scan of [`ewald_point_kernel`].
///
/// `successive_residual` is the change between the last two cutoff levels.
/// It is not an absolute error versus the Abramowitz–Stegun `erfc` (~$1.5\times 10^{-7}$).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EwaldConvergence {
    /// Converged kernel value.
    pub value: Complex64,
    /// Splitting parameter used for the scan.
    pub eta: f64,
    /// Final real-space cutoff.
    pub real_cutoff: Bohr,
    /// Final reciprocal cutoff.
    pub recip_cutoff: InverseBohr,
    /// $|v_n-v_{n-1}|$ at the accepted step.
    pub successive_residual: f64,
    /// Number of cutoff enlargements, including the accepted step.
    pub steps: usize,
}

/// Direct Ewald sum of the periodic Coulomb kernel between two point charges.
///
/// ```math
/// v^q(\mathbf{R}_1,\mathbf{R}_2)
/// = \frac{4\pi}{\Omega}\sum_{\mathbf{G}}
///   \frac{e^{-|q+G|^2/4\eta}}{|q+G|^2}
///   e^{i(q+G)\cdot(R_2-R_1)}
/// + \sum_{\mathbf{T}}
///   \frac{\mathrm{erfc}(\sqrt{\eta}\,|R_2-R_1+T|)}{|R_2-R_1+T|}
///   e^{i q\cdot T}
/// ```
///
/// The $q+G=0$ term is omitted. When a real-space image coincides with the
/// source, the singular $1/r$ term is replaced by the regular Ewald self limit
/// $-2\sqrt{\eta/\pi}$. This is a toy oracle, not SPEX `coulombmatrix`.
pub fn ewald_point_kernel(
    cell: &Cell,
    reciprocal: &ReciprocalLattice,
    q: TransferQ,
    r1: [Bohr; 3],
    r2: [Bohr; 3],
    summation: EwaldSummation,
) -> Result<Complex64, CoulombError> {
    let EwaldSummation {
        eta,
        real_cutoff,
        recip_cutoff,
    } = summation;
    if !eta.is_finite() || eta <= 0.0 {
        return Err(CoulombError::InvalidEwaldEta(eta));
    }
    if !real_cutoff.get().is_finite() || real_cutoff.get() < 0.0 {
        return Err(CoulombError::InvalidEwaldRealCutoff(real_cutoff.get()));
    }
    if !recip_cutoff.get().is_finite() || recip_cutoff.get() < 0.0 {
        return Err(CoulombError::InvalidEwaldRecipCutoff(recip_cutoff.get()));
    }
    let vol = cell.volume().get();
    let delta = [
        r2[0].get() - r1[0].get(),
        r2[1].get() - r1[1].get(),
        r2[2].get() - r1[2].get(),
    ];
    let mut acc = Complex64::default();

    for g in reciprocal.enumerate(recip_cutoff)? {
        let qg: [InverseBohr; 3] = std::array::from_fn(|axis| {
            InverseBohr(q.cartesian[axis].get() + g.cartesian[axis].get())
        });
        let qg2 = qg.iter().map(|c| c.get().powi(2)).sum::<f64>();
        if qg2 <= 1.0e-20 {
            continue;
        }
        let damping = (-qg2 / (4.0 * eta)).exp();
        let phase = plane_wave_phase(qg, [Bohr(delta[0]), Bohr(delta[1]), Bohr(delta[2])]);
        acc += phase * (4.0 * PI / vol * damping / qg2);
    }

    let basis = cell
        .basis()
        .map(|vector| vector.map(|c| InverseBohr(c.get())));
    let fake = ReciprocalLattice::new(basis)?;
    let mut self_correction = Complex64::default();
    for t in fake.enumerate(InverseBohr(real_cutoff.get()))? {
        let t_cart = t.cartesian.map(InverseBohr::get);
        let r = [
            delta[0] + t_cart[0],
            delta[1] + t_cart[1],
            delta[2] + t_cart[2],
        ];
        let dist = r.iter().map(|c| c * c).sum::<f64>().sqrt();
        if dist <= 1.0e-14 {
            self_correction += plane_wave_phase(
                q.cartesian,
                [Bohr(t_cart[0]), Bohr(t_cart[1]), Bohr(t_cart[2])],
            );
            continue;
        }
        let phase = plane_wave_phase(
            q.cartesian,
            [Bohr(t_cart[0]), Bohr(t_cart[1]), Bohr(t_cart[2])],
        );
        acc += phase * (erfc(eta.sqrt() * dist) / dist);
    }
    acc -= self_correction * (2.0 * (eta / PI).sqrt());

    if crate::math::is_gamma(q.cartesian) {
        acc -= PI / (eta * vol);
    }
    Ok(acc)
}

/// Complementary error function. Abramowitz–Stegun 7.1.26, max error ~$1.5\times 10^{-7}$.
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let value = poly * (-z * z).exp();
    if x >= 0.0 { value } else { 2.0 - value }
}

/// Successive-residual scan of [`ewald_point_kernel`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EwaldScan {
    /// $|v_n-v_{n-1}|$ required to accept the kernel.
    pub tolerance: f64,
    /// Maximum cutoff enlargements, including the first evaluation.
    pub max_steps: usize,
}

/// Scan real/reciprocal cutoffs until successive values differ by less than `scan.tolerance`.
///
/// Returns [`CoulombError::EwaldNotConverged`] if `scan.max_steps` enlargements
/// never meet the successive-residual gate. That residual is cutoff stability,
/// not absolute `erfc` accuracy.
pub fn converged_ewald_point_kernel(
    cell: &Cell,
    reciprocal: &ReciprocalLattice,
    q: TransferQ,
    r1: [Bohr; 3],
    r2: [Bohr; 3],
    scan: EwaldScan,
) -> Result<EwaldConvergence, CoulombError> {
    let vol = cell.volume().get();
    let eta = PI / vol.cbrt().powi(2);
    let mut real = vol.cbrt() * 2.0;
    let mut recip = 2.0 * PI / vol.cbrt();
    let mut previous = Complex64::default();
    let tolerance = scan.tolerance;
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(CoulombError::InvalidEwaldTolerance(tolerance));
    }
    if scan.max_steps < 2 {
        return Err(CoulombError::InvalidEwaldSteps(scan.max_steps));
    }
    for step in 0..scan.max_steps {
        real *= if step == 0 { 1.0 } else { 1.5 };
        recip *= if step == 0 { 1.0 } else { 1.5 };
        let value = ewald_point_kernel(
            cell,
            reciprocal,
            q,
            r1,
            r2,
            EwaldSummation {
                eta,
                real_cutoff: Bohr(real),
                recip_cutoff: InverseBohr(recip),
            },
        )?;
        if step > 0 {
            let residual = (value - previous).norm();
            if residual < tolerance {
                return Ok(EwaldConvergence {
                    value,
                    eta,
                    real_cutoff: Bohr(real),
                    recip_cutoff: InverseBohr(recip),
                    successive_residual: residual,
                    steps: step + 1,
                });
            }
            if step + 1 == scan.max_steps {
                return Err(CoulombError::EwaldNotConverged {
                    residual,
                    tolerance,
                    steps: scan.max_steps,
                });
            }
        }
        previous = value;
    }
    Err(CoulombError::EwaldNotConverged {
        residual: f64::INFINITY,
        tolerance,
        steps: scan.max_steps,
    })
}
