//! SPEX/Weinert block assembly over the shared charge expansion.

use crate::CoulombError;
use crate::expansion::{
    ChargeDensity, ExpansionSupport, SampledAuxiliaryFunctions, auxiliary_waves,
    mixed_product_densities, mixed_product_support, point_charge_densities, point_charge_support,
    sampled_interpolation_support, sampled_zeta_densities,
};
use crate::math::{
    i_pow, inverse_norm, is_gamma, is_zero_norm, parity, plane_wave_phase, sfac_table, weinert_gmat,
};
use crate::moments::{
    bessel_overlap, bessel_weinert_integral, multipole_moment, second_moment,
    sphbessel_pw_integral, sphere_plane_wave_integral, spherical_bessel_moment,
};
use crate::operator::{AuxiliaryKind, CoulombOperator, GammaHead, SpencerAlaviSphere};
use crate::primitive::intra_sphere_poisson;
use crate::spec::{CoulombKernel, CoulombRequest};
use crate::structure::structure_constants;
use muffintin_core::{Bohr, InverseBohr, ReciprocalLattice, complex_spherical_harmonics, lm_index};
use muffintin_envelope::Provenance;
use muffintin_prodbasis::{
    AuxiliaryInterstitialWave, AuxiliaryRegion, AuxiliaryRepresentation, CompiledAuxiliaryBasis,
};
use muffintin_tensor::{Axis, ComplexTensor, einsum};
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::f64::consts::PI;

const WAVE_TOLERANCE: f64 = 1.0e-12;

struct Prepared<'a> {
    request: &'a CoulombRequest,
    auxiliary: &'a CompiledAuxiliaryBasis,
    support: ExpansionSupport,
    structure: crate::structure::StructureConstants,
    sfac: Vec<f64>,
    is_gamma: bool,
}

/// Assemble $V^q$ for a mixed-product auxiliary.
///
/// Interpolation-point auxiliaries must use [`assemble_sampled_coulomb`].
pub fn assemble_coulomb(
    auxiliary: &CompiledAuxiliaryBasis,
    request: &CoulombRequest,
) -> Result<CoulombOperator, CoulombError> {
    match &auxiliary.representation {
        AuxiliaryRepresentation::MixedProduct(_) => {
            assemble_kind(auxiliary, request, AuxiliaryKind::MixedProduct, None)
        }
        AuxiliaryRepresentation::InterpolationPoints(_) => {
            Err(CoulombError::MissingSampledFunctions)
        }
    }
}

/// Assemble $V^q_{\mu\nu}=\langle\zeta_\mu^q|v|\zeta_\nu^q\rangle$ from sampled
/// interpolation functions on a parent quadrature grid.
pub fn assemble_sampled_coulomb(
    auxiliary: &CompiledAuxiliaryBasis,
    request: &CoulombRequest,
    sampled: &SampledAuxiliaryFunctions,
) -> Result<CoulombOperator, CoulombError> {
    if auxiliary.mixed_product().is_some() {
        return Err(CoulombError::UnexpectedSampledFunctions);
    }
    if sampled.n_mu() != auxiliary.dimension() {
        return Err(CoulombError::SampledZetaDimension {
            n_mu: sampled.n_mu(),
            expected: auxiliary.dimension(),
        });
    }
    if sampled.layout() != &auxiliary.layout() {
        return Err(CoulombError::SampledLayoutMismatch);
    }
    assemble_kind(
        auxiliary,
        request,
        AuxiliaryKind::InterpolationPoints,
        Some(sampled),
    )
}

/// Toy Ewald path: interpolation *nodes* as weighted point charges.
///
/// This is not the production $\zeta$ metric.
pub fn assemble_point_charge_oracle(
    auxiliary: &CompiledAuxiliaryBasis,
    request: &CoulombRequest,
) -> Result<CoulombOperator, CoulombError> {
    if auxiliary.interpolation_points().is_none() {
        return Err(CoulombError::ExpectedInterpolationPoints);
    }
    assemble_kind(auxiliary, request, AuxiliaryKind::PointChargeOracle, None)
}

fn assemble_kind(
    auxiliary: &CompiledAuxiliaryBasis,
    request: &CoulombRequest,
    kind: AuxiliaryKind,
    sampled: Option<&SampledAuxiliaryFunctions>,
) -> Result<CoulombOperator, CoulombError> {
    auxiliary.validate()?;
    if auxiliary.dimension() == 0 {
        return Err(CoulombError::EmptyAuxiliary);
    }
    require_cell_and_reciprocal(auxiliary, request)?;
    let (support, densities) = match kind {
        AuxiliaryKind::MixedProduct => {
            let payload = auxiliary.require_mixed_product()?;
            require_mixed_product_waves(payload, request, auxiliary.q)?;
            let support = mixed_product_support(auxiliary, payload)?;
            let densities = mixed_product_densities(auxiliary, payload)?;
            (support, densities)
        }
        AuxiliaryKind::InterpolationPoints => {
            let sampled = sampled.ok_or(CoulombError::MissingSampledFunctions)?;
            let projection = request
                .interpolation()
                .ok_or(CoulombError::MissingInterpolationProjection)?;
            let support = sampled_interpolation_support(auxiliary, request, &projection, sampled)?;
            let densities = sampled_zeta_densities(sampled, &support, &projection)?;
            (support, densities)
        }
        AuxiliaryKind::PointChargeOracle => {
            let projection = request
                .interpolation()
                .ok_or(CoulombError::MissingInterpolationProjection)?;
            let support = point_charge_support(auxiliary, request, &projection)?;
            let densities = point_charge_densities(auxiliary, &support, &projection)?;
            (support, densities)
        }
    };
    for density in &densities {
        for piece in &density.mt {
            if piece.l > request.lexp() {
                return Err(CoulombError::AuxiliaryLExceedsLexp {
                    l: piece.l,
                    lexp: request.lexp(),
                });
            }
        }
    }
    if let CoulombKernel::SpencerAlaviSphere {
        full_k_points,
        reciprocal_cutoff,
    } = request.kernel()
    {
        return assemble_spencer_alavi(
            auxiliary,
            request,
            kind,
            &support,
            &densities,
            full_k_points,
            reciprocal_cutoff,
        );
    }
    let structure = structure_constants(
        request.cell(),
        request.reciprocal(),
        &auxiliary.partition,
        auxiliary.q,
        request.lexp(),
    )?;
    let sfac = sfac_table((4 * request.lexp() + 4) as usize)?;
    let prepared = Prepared {
        request,
        auxiliary,
        support,
        structure,
        sfac,
        is_gamma: is_gamma(auxiliary.q.cartesian),
    };
    let n = densities.len();
    let mut matrix = vec![Complex64::default(); n * n];
    matrix
        .par_chunks_mut(n)
        .enumerate()
        .try_for_each(|(i, row)| -> Result<(), CoulombError> {
            for j in i..n {
                row[j] = weinert_inner(&densities[i], &densities[j], &prepared)?;
            }
            Ok(())
        })?;
    for i in 0..n {
        for j in i..n {
            if i != j {
                matrix[j * n + i] = matrix[i * n + j].conj();
            }
        }
    }
    let mut gamma = None;
    if prepared.is_gamma {
        let vectors = gamma_vectors(&densities, &prepared)?;
        subtract_spherical_average(
            &mut matrix,
            n,
            &vectors.coeff,
            &vectors.claplace,
            &vectors.cderiv,
        );
        gamma = Some(GammaHead {
            spherical_average_subtracted: true,
            head_prefactor: 4.0 * PI,
            constant_coefficients: vectors.coeff,
        });
    }
    let mut spencer_alavi = None;
    let mut provenance = Provenance {
        recipe: Some("weinert-spex-coulomb".to_owned()),
        reference: Some("SPEX coulombmatrix.f".to_owned()),
    };
    if let CoulombKernel::SmoothedSpencerAlaviSphere {
        full_k_points,
        reciprocal_cutoff,
        smoothing,
    } = request.kernel()
    {
        let correction =
            smoothed_truncation_correction(&densities, &prepared, reciprocal_cutoff, smoothing)?;
        for i in 0..n {
            for j in i..n {
                matrix[i * n + j] += correction[i * n + j];
                matrix[j * n + i] = matrix[i * n + j].conj();
            }
        }
        gamma = None;
        spencer_alavi = Some(SpencerAlaviSphere {
            radius: request.spencer_alavi_radius().expect("selected sphere"),
            full_k_points,
            reciprocal_cutoff,
            smoothing: Some(smoothing),
        });
        provenance = Provenance {
            recipe: Some("weinert-smoothed-spencer-alavi-coulomb".to_owned()),
            reference: Some("SPEX coulombmatrix.f; Yang et al. arXiv:2609.00203 Eq. 9".to_owned()),
        };
    }
    for (index, value) in matrix.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(CoulombError::NonFiniteMatrix {
                row: index / n,
                column: index % n,
            });
        }
    }
    Ok(CoulombOperator {
        layout: auxiliary.layout(),
        cell: *request.cell(),
        reciprocal: *request.reciprocal(),
        kind,
        matrix,
        gamma,
        spencer_alavi,
        provenance,
    })
}

/// Add the damped boundary correction to the full periodic finite body.
/// Its Q=0 entry is the complete finite limit 2*pi*Rc^2 + pi/omega^2,
/// since the periodic body omits that Fourier component.
fn smoothed_truncation_correction(
    densities: &[ChargeDensity],
    prepared: &Prepared<'_>,
    reciprocal_cutoff: InverseBohr,
    smoothing: InverseBohr,
) -> Result<Vec<Complex64>, CoulombError> {
    let radius = prepared
        .request
        .spencer_alavi_radius()
        .expect("selected sphere")
        .get();
    let omega = smoothing.get();
    let waves = auxiliary_waves(prepared.request, prepared.auxiliary.q, reciprocal_cutoff)?;
    let n = densities.len();
    if waves.is_empty() {
        return Ok(vec![Complex64::default(); n * n]);
    }
    let coefficients =
        auxiliary_fourier_coefficients(densities, &prepared.support, prepared.auxiliary, &waves)?;
    let kernels = waves
        .iter()
        .map(|wave| {
            let q = wave.q_plus_g_norm.get();
            if is_zero_norm(q) {
                2.0 * PI * radius * radius + PI / (omega * omega)
            } else {
                -4.0 * PI / (q * q) * (q * radius).cos() * (-(q / (2.0 * omega)).powi(2)).exp()
            }
        })
        .collect::<Vec<_>>();
    let conjugate = ComplexTensor::from_host_row_major(
        &[waves.len(), n],
        &[Axis::Auxiliary, Axis::Auxiliary],
        coefficients.iter().flatten().map(|c| c.conj()).collect(),
    )?;
    let weighted = ComplexTensor::from_host_row_major(
        &[waves.len(), n],
        &[Axis::Auxiliary, Axis::Auxiliary],
        coefficients
            .iter()
            .zip(kernels)
            .flat_map(|(row, kernel)| row.iter().map(move |c| c * kernel))
            .collect(),
    )?;
    Ok(einsum("gi,gj->ij", &[&conjugate, &weighted])?.to_host_row_major())
}

#[allow(clippy::too_many_arguments)]
fn assemble_spencer_alavi(
    auxiliary: &CompiledAuxiliaryBasis,
    request: &CoulombRequest,
    kind: AuxiliaryKind,
    support: &ExpansionSupport,
    densities: &[ChargeDensity],
    full_k_points: usize,
    reciprocal_cutoff: InverseBohr,
) -> Result<CoulombOperator, CoulombError> {
    let radius = request
        .spencer_alavi_radius()
        .expect("the caller selected the Spencer-Alavi kernel");
    let waves = auxiliary_waves(request, auxiliary.q, reciprocal_cutoff)?;
    let coefficients = auxiliary_fourier_coefficients(densities, support, auxiliary, &waves)?;
    let kernels = waves
        .iter()
        .map(|wave| spencer_alavi_kernel(wave.q_plus_g_norm.get(), radius.get()))
        .collect::<Vec<_>>();
    let n = densities.len();
    let mut matrix = vec![Complex64::default(); n * n];
    matrix
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(left, row)| {
            for right in left..n {
                row[right] = coefficients
                    .iter()
                    .zip(&kernels)
                    .map(|(wave, &kernel)| wave[left].conj() * kernel * wave[right])
                    .sum();
            }
        });
    for left in 0..n {
        for right in left + 1..n {
            matrix[right * n + left] = matrix[left * n + right].conj();
        }
    }
    for (index, value) in matrix.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(CoulombError::NonFiniteMatrix {
                row: index / n,
                column: index % n,
            });
        }
    }
    Ok(CoulombOperator {
        layout: auxiliary.layout(),
        cell: *request.cell(),
        reciprocal: *request.reciprocal(),
        kind,
        matrix,
        gamma: None,
        spencer_alavi: Some(SpencerAlaviSphere {
            radius,
            full_k_points,
            reciprocal_cutoff,
            smoothing: None,
        }),
        provenance: Provenance {
            recipe: Some("spencer-alavi-sphere-coulomb".to_owned()),
            reference: Some("VASP HFRCUT=-1; Spencer-Alavi PRB 77, 193110".to_owned()),
        },
    })
}

/// MPB radial transforms depend on |q+G|, not its direction or magnetic m.
/// Cache exact floating-point shells, with no approximate radius grouping.
fn auxiliary_fourier_coefficients(
    densities: &[ChargeDensity],
    support: &ExpansionSupport,
    auxiliary: &CompiledAuxiliaryBasis,
    waves: &[AuxiliaryInterstitialWave],
) -> Result<Vec<Vec<Complex64>>, CoulombError> {
    let Some(payload) = auxiliary.mixed_product() else {
        return waves
            .par_iter()
            .map(|wave| {
                densities
                    .iter()
                    .map(|density| truncated_fourier_coefficient(density, wave, support, auxiliary))
                    .collect()
            })
            .collect();
    };
    let mut shell_indices = BTreeMap::new();
    let mut shell_norms = Vec::new();
    let wave_shells = waves
        .iter()
        .map(|wave| {
            *shell_indices
                .entry(wave.q_plus_g_norm.get().to_bits())
                .or_insert_with(|| {
                    let index = shell_norms.len();
                    shell_norms.push(wave.q_plus_g_norm.get());
                    index
                })
        })
        .collect::<Vec<_>>();
    let mut mode_indices = BTreeMap::new();
    let mut modes = Vec::new();
    for site in &payload.sites {
        for mode in &site.modes {
            mode_indices.insert((site.site, mode.l, mode.n), modes.len());
            modes.push((site, mode));
        }
    }
    let transforms = modes
        .par_iter()
        .map(|(site, mode)| {
            shell_norms
                .iter()
                .map(|&norm| bessel_overlap(mode.l, norm, &site.mesh, &mode.radial))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let regions = auxiliary.regions();
    let l_max = modes.iter().map(|(_, mode)| mode.l).max().unwrap_or(0);
    let scale = 4.0 * PI / support.volume.sqrt();
    waves
        .par_iter()
        .zip(&wave_shells)
        .map(|(wave, &shell)| {
            let harmonics = complex_spherical_harmonics(l_max, wave.q_plus_g.map(InverseBohr::get));
            let phases = support
                .sites
                .iter()
                .map(|site| plane_wave_phase(wave.q_plus_g, site.position).conj())
                .collect::<Vec<_>>();
            let mut next_wave = 0;
            regions
                .iter()
                .map(|region| match *region {
                    AuxiliaryRegion::MuffinTin { site, l, m, n } => {
                        let mode = mode_indices[&(site, l, n)];
                        Ok(scale
                            * i_pow(l).conj()
                            * harmonics[lm_index(l, m)?]
                            * phases[site]
                            * transforms[mode][shell])
                    }
                    AuxiliaryRegion::Interstitial { .. } => {
                        let source = &support.waves[next_wave];
                        next_wave += 1;
                        let difference = std::array::from_fn(|axis| {
                            InverseBohr(
                                source.g.cartesian[axis].get() - wave.g.cartesian[axis].get(),
                            )
                        });
                        Ok(auxiliary.partition.interstitial().coefficient(difference)?)
                    }
                    AuxiliaryRegion::InterpolationPoint { .. } => {
                        unreachable!("validated mixed-product regions")
                    }
                })
                .collect()
        })
        .collect()
}

fn truncated_fourier_coefficient(
    density: &ChargeDensity,
    wave: &AuxiliaryInterstitialWave,
    support: &ExpansionSupport,
    auxiliary: &CompiledAuxiliaryBasis,
) -> Result<Complex64, CoulombError> {
    let mut coefficient = Complex64::default();
    let harmonics = complex_spherical_harmonics(
        density.mt.iter().map(|piece| piece.l).max().unwrap_or(0),
        wave.q_plus_g.map(InverseBohr::get),
    );
    let inverse_sqrt_volume = support.volume.sqrt().recip();
    for piece in &density.mt {
        let site = &support.sites[piece.site];
        let radial = bessel_overlap(piece.l, wave.q_plus_g_norm.get(), &site.mesh, &piece.radial)?;
        coefficient += piece.amplitude
            * 4.0
            * PI
            * i_pow(piece.l).conj()
            * harmonics[lm_index(piece.l, piece.m)?]
            * plane_wave_phase(wave.q_plus_g, site.position).conj()
            * radial
            * inverse_sqrt_volume;
    }
    for (source, amplitude) in support.waves.iter().zip(&density.pw) {
        if *amplitude == Complex64::default() {
            continue;
        }
        let difference = std::array::from_fn(|axis| {
            InverseBohr(source.g.cartesian[axis].get() - wave.g.cartesian[axis].get())
        });
        coefficient += *amplitude * auxiliary.partition.interstitial().coefficient(difference)?;
    }
    Ok(coefficient)
}

fn spencer_alavi_kernel(q_norm: f64, radius: f64) -> f64 {
    if is_zero_norm(q_norm) {
        2.0 * PI * radius * radius
    } else {
        8.0 * PI * (0.5 * q_norm * radius).sin().powi(2) / (q_norm * q_norm)
    }
}

fn require_cell_and_reciprocal(
    auxiliary: &CompiledAuxiliaryBasis,
    request: &CoulombRequest,
) -> Result<(), CoulombError> {
    let cell_volume = request.cell().volume().get();
    let partition_volume = auxiliary.partition.interstitial().cell_volume().get();
    if (cell_volume - partition_volume).abs() > 1.0e-8 * cell_volume.max(partition_volume) {
        return Err(CoulombError::CellVolumeMismatch {
            cell: cell_volume,
            partition: partition_volume,
        });
    }
    let from_cell = ReciprocalLattice::from_direct(*request.cell().basis())?;
    if &from_cell != request.reciprocal() {
        return Err(CoulombError::ReciprocalMismatch);
    }
    Ok(())
}

fn require_mixed_product_waves(
    payload: &muffintin_prodbasis::MixedProductAuxiliary,
    request: &CoulombRequest,
    q: muffintin_prodbasis::TransferQ,
) -> Result<(), CoulombError> {
    for wave in &payload.interstitial.waves {
        let cartesian = request.reciprocal().cartesian(wave.g.index);
        let mismatch = cartesian
            .iter()
            .zip(wave.g.cartesian.iter())
            .any(|(want, got)| (want.get() - got.get()).abs() > WAVE_TOLERANCE);
        let norm = cartesian
            .iter()
            .map(|component| component.get().powi(2))
            .sum::<f64>()
            .sqrt();
        if mismatch || (wave.g.norm.get() - norm).abs() > WAVE_TOLERANCE {
            return Err(CoulombError::WaveLatticeMismatch {
                index: wave.g.index,
            });
        }
        let q_plus_g: [InverseBohr; 3] = std::array::from_fn(|axis| {
            InverseBohr(q.cartesian[axis].get() + wave.g.cartesian[axis].get())
        });
        let qg_ok = wave
            .q_plus_g
            .iter()
            .zip(q_plus_g.iter())
            .all(|(actual, want)| (actual.get() - want.get()).abs() <= WAVE_TOLERANCE);
        let qg_norm = q_plus_g
            .iter()
            .map(|component| component.get().powi(2))
            .sum::<f64>()
            .sqrt();
        if !qg_ok || (wave.q_plus_g_norm.get() - qg_norm).abs() > WAVE_TOLERANCE {
            return Err(CoulombError::WaveLatticeMismatch {
                index: wave.g.index,
            });
        }
    }
    Ok(())
}

fn weinert_inner(
    left: &ChargeDensity,
    right: &ChargeDensity,
    prepared: &Prepared<'_>,
) -> Result<Complex64, CoulombError> {
    let mut value = Complex64::default();
    for a in &left.mt {
        for b in &right.mt {
            value += mt_mt_element(a, b, prepared)?;
        }
    }
    for a in &left.mt {
        for (g, coeff) in right.pw.iter().enumerate() {
            if coeff.norm() == 0.0 {
                continue;
            }
            value += mt_pw_element(a, &prepared.support.waves[g], prepared)? * *coeff;
        }
    }
    for b in &right.mt {
        for (g, coeff) in left.pw.iter().enumerate() {
            if coeff.norm() == 0.0 {
                continue;
            }
            value += (mt_pw_element(b, &prepared.support.waves[g], prepared)? * *coeff).conj();
        }
    }
    for (g1, c1) in left.pw.iter().enumerate() {
        if c1.norm() == 0.0 {
            continue;
        }
        for (g2, c2) in right.pw.iter().enumerate() {
            if c2.norm() == 0.0 {
                continue;
            }
            value += c1.conj() * pw_pw_element(g1, g2, prepared)? * c2;
        }
    }
    Ok(value)
}

fn mt_mt_element(
    left: &crate::expansion::MtPiece,
    right: &crate::expansion::MtPiece,
    prepared: &Prepared<'_>,
) -> Result<Complex64, CoulombError> {
    let mut value = Complex64::default();
    if left.site == right.site && left.l == right.l && left.m == right.m {
        let mesh = &prepared.support.sites[left.site].mesh;
        let radial = intra_sphere_poisson(left.l, mesh, &left.radial, &right.radial)?;
        value += left.amplitude.conj() * right.amplitude * radial;
    }
    let moment_l = multipole_moment(
        left.l,
        &prepared.support.sites[left.site].mesh,
        &left.radial,
    )?;
    let moment_r = multipole_moment(
        right.l,
        &prepared.support.sites[right.site].mesh,
        &right.radial,
    )?;
    let l = left.l + right.l;
    let m = left.m - right.m;
    if l > 2 * prepared.request.lexp() || m.unsigned_abs() > l {
        return Ok(value);
    }
    let g = weinert_gmat(left.l, left.m, right.l, right.m, &prepared.sfac)?;
    let rdum = parity(right.l as i32 + right.m) * moment_l * moment_r * g;
    let r1 = prepared.support.sites[left.site].position;
    let r2 = prepared.support.sites[right.site].position;
    let phase = plane_wave_phase(
        prepared.auxiliary.q.cartesian,
        [
            Bohr(r2[0].get() - r1[0].get()),
            Bohr(r2[1].get() - r1[1].get()),
            Bohr(r2[2].get() - r1[2].get()),
        ],
    );
    let structure = prepared.structure.get(left.site, right.site, l, m)?;
    value += left.amplitude.conj() * right.amplitude * phase * structure * rdum;
    Ok(value)
}

fn mt_pw_element(
    piece: &crate::expansion::MtPiece,
    wave: &AuxiliaryInterstitialWave,
    prepared: &Prepared<'_>,
) -> Result<Complex64, CoulombError> {
    let site = piece.site;
    let l = piece.l;
    let m = piece.m;
    let mesh = &prepared.support.sites[site].mesh;
    let pos = prepared.support.sites[site].position;
    let svol = prepared.support.volume.sqrt();
    let qnorm = wave.q_plus_g_norm.get();
    let y =
        complex_spherical_harmonics(prepared.request.lexp(), wave.q_plus_g.map(InverseBohr::get));
    let lm = lm_index(l, m)?;
    let moment = multipole_moment(l, mesh, &piece.radial)?;
    let moment2 = if l == 0 {
        second_moment(mesh, &piece.radial)?
    } else {
        0.0
    };
    let olap = bessel_overlap(l, qnorm, mesh, &piece.radial)?;
    let integral = bessel_weinert_integral(l, qnorm, mesh, &piece.radial)?;
    let mut csum = Complex64::default();
    let g_is_zero = wave.g.index == [0; 3];
    for (site1, support1) in prepared.support.sites.iter().enumerate() {
        let cexp = 4.0
            * PI
            * plane_wave_phase(wave.q_plus_g, support1.position)
            * plane_wave_phase(prepared.auxiliary.q.cartesian, pos).conj();
        for l1 in 0..=prepared.request.lexp() {
            let sph = spherical_bessel_moment(l1, qnorm, support1.radius.get());
            let cdum = sph * i_pow(l1) * cexp;
            for m1 in -(l1 as i32)..=l1 as i32 {
                // SPEX `idum = 1` inside every L1, then `(-1)^{L1+M1}` (`~484-493`).
                let idum = parity(l1 as i32 + m1);
                let lm1 = lm_index(l1, m1)?;
                let l2 = l + l1;
                let m2 = m - m1;
                let structure = if l2 <= 2 * prepared.request.lexp() && m2.unsigned_abs() <= l2 {
                    prepared.structure.get(site, site1, l2, m2)?
                } else {
                    Complex64::default()
                };
                let g = weinert_gmat(l1, m1, l, m, &prepared.sfac)?;
                csum -= idum * g * y[lm1].conj() * cdum * structure;
            }
        }
        if prepared.is_gamma && l <= 2 {
            let cexp_g = plane_wave_phase(wave.g.cartesian, support1.position)
                * weinert_gmat(l, m, 0, 0, &prepared.sfac)?
                * 4.0
                * PI
                / prepared.support.volume;
            let radius = support1.radius.get();
            if l == 0 {
                if g_is_zero {
                    csum -= cexp_g * radius.powi(5) / 30.0;
                } else {
                    let m0 = spherical_bessel_moment(0, qnorm, radius);
                    let m2 = spherical_bessel_moment(2, qnorm, radius);
                    csum -= cexp_g * (m0 * radius * radius - m2 * 2.0 / 3.0) / 10.0;
                }
            } else if l == 1 {
                let m1 = spherical_bessel_moment(1, qnorm, radius);
                // SPEX `y` is already `conjg(harmonicsr)` (`~461-462`, `~509`).
                csum +=
                    cexp_g * Complex64::new(0.0, 1.0) * (4.0 * PI).sqrt() * m1 * y[lm].conj() / 3.0;
            }
        }
    }
    let cdum =
        (4.0 * PI).powi(2) * i_pow(l) * y[lm].conj() * plane_wave_phase(wave.g.cartesian, pos);
    let carr = if prepared.is_gamma && g_is_zero {
        let mut z = Complex64::default();
        if l == 0 {
            z -= cdum * moment2 / 6.0 / svol;
        }
        z + (-cdum / (2.0 * f64::from(l) + 1.0) * integral + csum * moment) / svol
    } else if is_zero_norm(qnorm) {
        return Err(CoulombError::ZeroQPlusG {
            index: wave.g.index,
        });
    } else {
        (cdum * olap / (qnorm * qnorm) - cdum / (2.0 * f64::from(l) + 1.0) * integral
            + csum * moment)
            / svol
    };
    Ok(piece.amplitude.conj() * carr)
}

fn pw_pw_element(g1: usize, g2: usize, prepared: &Prepared<'_>) -> Result<Complex64, CoulombError> {
    let wave1 = &prepared.support.waves[g1];
    let wave2 = &prepared.support.waves[g2];
    let vol = prepared.support.volume;
    let q1 = wave1.q_plus_g_norm.get();
    let q2 = wave2.q_plus_g_norm.get();
    let v1 = if is_zero_norm(q1) {
        0.0
    } else {
        4.0 * PI / (q1 * q1)
    };
    let v2 = if is_zero_norm(q2) {
        0.0
    } else {
        4.0 * PI / (q2 * q2)
    };
    let gdiff: [InverseBohr; 3] = std::array::from_fn(|axis| {
        InverseBohr(wave2.g.cartesian[axis].get() - wave1.g.cartesian[axis].get())
    });
    let gnorm = inverse_norm(gdiff);
    let mut cint = Complex64::default();
    if is_zero_norm(gnorm) {
        for site in &prepared.support.sites {
            cint += sphere_plane_wave_integral(0.0, site.radius.get());
        }
    } else {
        for site in &prepared.support.sites {
            let form = sphere_plane_wave_integral(gnorm, site.radius.get());
            cint += form * plane_wave_phase(gdiff, site.position);
        }
    }
    let g1_zero = wave1.g.index == [0; 3];
    let g2_zero = wave2.g.index == [0; 3];
    let mut value = Complex64::default();
    if prepared.is_gamma {
        if !g1_zero {
            value -= cint * v1 / vol;
        }
        if !g2_zero {
            value -= cint * v2 / vol;
        }
        if g1 == g2 && !g2_zero {
            value += v2;
        }
    } else {
        value -= cint * (v1 + v2) / vol;
        if g1 == g2 {
            value += v2;
        }
    }
    value += pw_pw_intersite(wave1, wave2, prepared)?;
    value += pw_pw_intrasite(wave1, wave2, prepared)?;
    if prepared.is_gamma {
        value += pw_pw_gamma_correction(wave1, wave2, prepared)?;
    }
    Ok(value)
}

fn pw_pw_intersite(
    wave1: &AuxiliaryInterstitialWave,
    wave2: &AuxiliaryInterstitialWave,
    prepared: &Prepared<'_>,
) -> Result<Complex64, CoulombError> {
    let vol = prepared.support.volume;
    let q1 = wave1.q_plus_g.map(InverseBohr::get);
    let q2 = wave2.q_plus_g.map(InverseBohr::get);
    let y1 = complex_spherical_harmonics(prepared.request.lexp(), q1);
    let y2 = complex_spherical_harmonics(prepared.request.lexp(), q2);
    let mut csum = Complex64::default();
    for (ic2, site2) in prepared.support.sites.iter().enumerate() {
        let cexp2 = plane_wave_phase(wave2.q_plus_g, site2.position);
        for l2 in 0..=prepared.request.lexp() {
            let sph2 = spherical_bessel_moment(l2, wave2.q_plus_g_norm.get(), site2.radius.get());
            for m2 in -(l2 as i32)..=l2 as i32 {
                // SPEX `idum = 1` inside every L2 (`~775-776`).
                let idum = parity(l2 as i32 + m2);
                let lm2 = lm_index(l2, m2)?;
                let cdum = idum * sph2 * cexp2 * 4.0 * PI * i_pow(l2) * y2[lm2].conj();
                if cdum.norm() > 0.0 {
                    for (ic1, site1) in prepared.support.sites.iter().enumerate() {
                        for l1 in 0..=prepared.request.lexp() {
                            let sph1 = spherical_bessel_moment(
                                l1,
                                wave1.q_plus_g_norm.get(),
                                site1.radius.get(),
                            );
                            for m1 in -(l1 as i32)..=l1 as i32 {
                                let lm1 = lm_index(l1, m1)?;
                                let l = l1 + l2;
                                let m = m1 - m2;
                                if l > 2 * prepared.request.lexp() || m.unsigned_abs() > l {
                                    continue;
                                }
                                let g = weinert_gmat(l1, m1, l2, m2, &prepared.sfac)?;
                                let structure = prepared.structure.get(ic1, ic2, l, m)?;
                                let left = plane_wave_phase(wave1.q_plus_g, site1.position).conj()
                                    * sph1
                                    * 4.0
                                    * PI
                                    * i_pow(l1).conj()
                                    * y1[lm1];
                                csum += left * cdum * g * structure;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(csum / vol)
}

fn pw_pw_intrasite(
    wave1: &AuxiliaryInterstitialWave,
    wave2: &AuxiliaryInterstitialWave,
    prepared: &Prepared<'_>,
) -> Result<Complex64, CoulombError> {
    let vol = prepared.support.volume;
    let q1 = wave1.q_plus_g.map(InverseBohr::get);
    let q2 = wave2.q_plus_g.map(InverseBohr::get);
    let y1 = complex_spherical_harmonics(prepared.request.lexp(), q1);
    let y2 = complex_spherical_harmonics(prepared.request.lexp(), q2);
    let gdiff: [InverseBohr; 3] = std::array::from_fn(|axis| {
        InverseBohr(wave2.g.cartesian[axis].get() - wave1.g.cartesian[axis].get())
    });
    let mut cdum = Complex64::default();
    for l in 0..=prepared.request.lexp() {
        let mut cdum1 = Complex64::default();
        for site in &prepared.support.sites {
            let phase = plane_wave_phase(gdiff, site.position);
            cdum1 += phase
                * sphbessel_pw_integral(
                    l,
                    wave1.q_plus_g_norm.get(),
                    wave2.q_plus_g_norm.get(),
                    site.radius.get(),
                )
                / (2.0 * f64::from(l) + 1.0);
        }
        for m in -(l as i32)..=l as i32 {
            let lm = lm_index(l, m)?;
            cdum += cdum1 * y1[lm] * y2[lm].conj();
        }
    }
    Ok((4.0 * PI).powi(3) * cdum / vol)
}

fn pw_pw_gamma_correction(
    wave1: &AuxiliaryInterstitialWave,
    wave2: &AuxiliaryInterstitialWave,
    prepared: &Prepared<'_>,
) -> Result<Complex64, CoulombError> {
    let g1_zero = wave1.g.index == [0; 3];
    let g2_zero = wave2.g.index == [0; 3];
    let vol = prepared.support.volume;
    let g00 = weinert_gmat(0, 0, 0, 0, &prepared.sfac)?;
    let rdum = (4.0 * PI).powf(1.5) / vol.powi(2) * g00;
    let q1 = wave1.q_plus_g_norm.get();
    let q2 = wave2.q_plus_g_norm.get();
    let mut value = Complex64::default();
    if !g1_zero && !g2_zero {
        let q1v = wave1.q_plus_g.map(InverseBohr::get);
        let q2v = wave2.q_plus_g.map(InverseBohr::get);
        let rdum1 = (q1v[0] * q2v[0] + q1v[1] * q2v[1] + q1v[2] * q2v[2]) / (q1 * q2);
        for site1 in &prepared.support.sites {
            for site2 in &prepared.support.sites {
                let cdum = plane_wave_phase(
                    [
                        InverseBohr(-wave1.g.cartesian[0].get()),
                        InverseBohr(-wave1.g.cartesian[1].get()),
                        InverseBohr(-wave1.g.cartesian[2].get()),
                    ],
                    site1.position,
                ) * plane_wave_phase(wave2.g.cartesian, site2.position);
                let m0a = spherical_bessel_moment(0, q1, site1.radius.get());
                let m1a = spherical_bessel_moment(1, q1, site1.radius.get());
                let m2a = spherical_bessel_moment(2, q1, site1.radius.get());
                let m0b = spherical_bessel_moment(0, q2, site2.radius.get());
                let m1b = spherical_bessel_moment(1, q2, site2.radius.get());
                let m2b = spherical_bessel_moment(2, q2, site2.radius.get());
                value += rdum
                    * cdum
                    * (-m1a * m1b * rdum1 / 3.0 - m0a * m2b / 6.0 - m2a * m0b / 6.0
                        + m0a * m1b / q2 / 2.0
                        + m1a * m0b / q1 / 2.0);
            }
        }
    } else if g1_zero && !g2_zero {
        for site1 in &prepared.support.sites {
            let r1 = site1.radius.get();
            for site2 in &prepared.support.sites {
                let cdum = plane_wave_phase(wave2.g.cartesian, site2.position);
                let m0b = spherical_bessel_moment(0, q2, site2.radius.get());
                let m1b = spherical_bessel_moment(1, q2, site2.radius.get());
                let m2b = spherical_bessel_moment(2, q2, site2.radius.get());
                value +=
                    rdum * cdum * r1.powi(3) * (m0b / 30.0 * r1 * r1 - m2b / 18.0 + m1b / 6.0 / q2);
            }
        }
    } else if !g1_zero && g2_zero {
        for site2 in &prepared.support.sites {
            let r2 = site2.radius.get();
            for site1 in &prepared.support.sites {
                let cdum = plane_wave_phase(
                    [
                        InverseBohr(-wave1.g.cartesian[0].get()),
                        InverseBohr(-wave1.g.cartesian[1].get()),
                        InverseBohr(-wave1.g.cartesian[2].get()),
                    ],
                    site1.position,
                );
                let m0a = spherical_bessel_moment(0, q1, site1.radius.get());
                let m1a = spherical_bessel_moment(1, q1, site1.radius.get());
                let m2a = spherical_bessel_moment(2, q1, site1.radius.get());
                value +=
                    rdum * cdum * r2.powi(3) * (m0a / 30.0 * r2 * r2 - m2a / 18.0 + m1a / 6.0 / q1);
            }
        }
    } else {
        for site1 in &prepared.support.sites {
            let r1 = site1.radius.get();
            for site2 in &prepared.support.sites {
                let r2 = site2.radius.get();
                value += rdum * r1.powi(3) * r2.powi(3) * (r1 * r1 + r2 * r2) / 90.0;
            }
        }
    }
    Ok(value)
}

struct GammaVectors {
    coeff: Vec<Complex64>,
    claplace: Vec<Complex64>,
    cderiv: Vec<[Complex64; 3]>,
}

fn gamma_vectors(
    densities: &[ChargeDensity],
    prepared: &Prepared<'_>,
) -> Result<GammaVectors, CoulombError> {
    let n = densities.len();
    let svol = prepared.support.volume.sqrt();
    let mut coeff = vec![Complex64::default(); n];
    let mut claplace = vec![Complex64::default(); n];
    let mut cderiv = vec![[Complex64::default(); 3]; n];
    for (index, density) in densities.iter().enumerate() {
        for piece in &density.mt {
            let mesh = &prepared.support.sites[piece.site].mesh;
            if piece.l == 0 {
                let integrand_c: Vec<f64> = mesh
                    .radii()
                    .iter()
                    .zip(&piece.radial)
                    .map(|(radius, sample)| sample * radius.get())
                    .collect();
                let integrand_l: Vec<f64> = mesh
                    .radii()
                    .iter()
                    .zip(&piece.radial)
                    .map(|(radius, sample)| sample * radius.get().powi(3))
                    .collect();
                coeff[index] +=
                    piece.amplitude * (4.0 * PI).sqrt() * mesh.integrate(&integrand_c)? / svol;
                claplace[index] +=
                    piece.amplitude * -(4.0 * PI).sqrt() * mesh.integrate(&integrand_l)? / svol;
            } else if piece.l == 1 {
                let integrand: Vec<f64> = mesh
                    .radii()
                    .iter()
                    .zip(&piece.radial)
                    .map(|(radius, sample)| sample * radius.get().powi(2))
                    .collect();
                let value = -(4.0 * PI / 3.0).sqrt()
                    * Complex64::new(0.0, 1.0)
                    * mesh.integrate(&integrand)?
                    / svol;
                let slot = match piece.m {
                    -1 => 0,
                    0 => 1,
                    1 => 2,
                    _ => continue,
                };
                cderiv[index][slot] += piece.amplitude * value;
            }
        }
        for (g, c_g) in density.pw.iter().enumerate() {
            if c_g.norm() == 0.0 {
                continue;
            }
            let wave = &prepared.support.waves[g];
            let neg_g: [InverseBohr; 3] =
                std::array::from_fn(|axis| InverseBohr(-wave.g.cartesian[axis].get()));
            coeff[index] += *c_g
                * prepared
                    .auxiliary
                    .partition
                    .interstitial()
                    .coefficient(neg_g)?;
        }
    }
    Ok(GammaVectors {
        coeff,
        claplace,
        cderiv,
    })
}

fn subtract_spherical_average(
    matrix: &mut [Complex64],
    n: usize,
    coeff: &[Complex64],
    claplace: &[Complex64],
    cderiv: &[[Complex64; 3]],
) {
    let prefactor = 4.0 * PI / 3.0;
    for j in 0..n {
        for i in 0..n {
            let dipole = cderiv[i][0].conj() * cderiv[j][0]
                + cderiv[i][1].conj() * cderiv[j][1]
                + cderiv[i][2].conj() * cderiv[j][2];
            let laplace = (coeff[i].conj() * claplace[j] + claplace[i].conj() * coeff[j]) / 2.0;
            matrix[i * n + j] -= prefactor * (dipole + laplace);
        }
    }
}
