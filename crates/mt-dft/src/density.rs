//! Occupied LAPW eigenstates to muffin-tin and interstitial density fields.

use crate::{
    InterstitialField, MuffinTinField, RegionalDensity, RegionalError, RegionalScalarField,
};
use muffintin_core::{
    ExponentialMesh, FourierLayout, InterstitialGeometry, Lm, MeshError, RelativisticChannel,
    lm_index,
};
use muffintin_envelope::{CompiledBasis, SpinorCompiledBasis};
use muffintin_operators::{
    Collinear, CompiledSiteProjection, GeneralizedEigensolution, OperatorError,
};
use muffintin_sphere::{CoreDiracSolution, RadialComponents};
use muffintin_sphere::{
    DensityProjectionError, HarmonicConvention, SphereField, SphereFieldError, SphereOrbital,
    SpinorSphereOrbital, scalar_pair_density_angular, spinor_pair_density_angular,
};
use muffintin_tensor::{
    Axis, ComplexTensor, DenseEigenvectors, DenseHermitianMatrix, TensorError, einsum,
};
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::sync::Arc;
use thiserror::Error;

const WEIGHT_TOLERANCE: f64 = 4096.0 * f64::EPSILON;
const BAND_PROJECTION_TOLERANCE: f64 = 4096.0 * f64::EPSILON;
const SPINOR_COUNT_TOLERANCE: f64 = 1.0e-9;

/// One regular full-BZ scalar first-variation solution.
#[derive(Clone, Debug)]
pub struct CollinearKPoint<'a> {
    pub weight: f64,
    pub compiled: &'a CompiledBasis,
    pub solutions: Collinear<&'a GeneralizedEigensolution>,
    pub occupations: Collinear<&'a [f64]>,
}

/// One SOC second-variation solution represented by Pauli components on the
/// same scalar LAPW basis.
#[derive(Clone, Debug)]
pub struct SecondVariationKPoint<'a> {
    pub weight: f64,
    pub compiled: &'a CompiledBasis,
    pub solutions: Collinear<&'a GeneralizedEigensolution>,
    pub occupations: &'a [f64],
}

/// Radial orbitals in exactly the same order as one compiled site projection.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarSiteBasis {
    pub mesh: ExponentialMesh,
    pub orbitals: Vec<SphereOrbital>,
}

/// One regular full-BZ full-spinor first-variation solution.
///
/// Occupations are one value per spinor band and do not include `weight`.
#[derive(Clone, Debug)]
pub struct FullSpinorKPoint<'a> {
    pub weight: f64,
    pub compiled: &'a SpinorCompiledBasis,
    pub solution: &'a GeneralizedEigensolution,
    pub occupations: &'a [f64],
}

/// Four-component radial orbitals in exact compiled site-coordinate order.
#[derive(Clone, Debug, PartialEq)]
pub struct FullSpinorDensitySiteBasis {
    pub mesh: ExponentialMesh,
    pub channels: Vec<RelativisticChannel>,
    pub orbitals: Vec<SpinorSphereOrbital>,
    contraction: SpinorDensityContraction,
}

#[derive(Clone, Debug, PartialEq)]
struct SpinorDensityContraction {
    harmonics: Vec<Lm>,
    terms: [Vec<Vec<SpinorDensityTerm>>; 4],
    radial_products: Arc<Vec<Complex64>>,
    radial_pair_count: usize,
    radial_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SpinorDensityTerm {
    left: usize,
    right: usize,
    radial_pair: usize,
    pp: Complex64,
    qq: Complex64,
}

#[derive(Clone, Debug, PartialEq)]
struct ScalarDensityContraction {
    harmonics: Vec<Lm>,
    terms: Vec<Vec<ScalarDensityTerm>>,
    radial_large: Vec<f64>,
    radial_small: Vec<f64>,
    radial_count: usize,
    orbital_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScalarDensityTerm {
    left: usize,
    right: usize,
    angular: f64,
}

impl ScalarDensityContraction {
    fn compile(site: &ScalarSiteBasis) -> Self {
        let l_max = site
            .orbitals
            .iter()
            .flat_map(|left| {
                site.orbitals
                    .iter()
                    .map(move |right| left.angular().l + right.angular().l)
            })
            .max()
            .unwrap_or(0);
        let harmonics = (0..=l_max)
            .flat_map(|l| {
                (-(l as i32)..=l as i32)
                    .map(move |m| Lm::new(l, m).expect("loop bounds validate magnetic channel"))
            })
            .collect::<Vec<_>>();
        let mut terms = vec![Vec::new(); harmonics.len()];
        for (left, left_orbital) in site.orbitals.iter().enumerate() {
            for (right, right_orbital) in site.orbitals.iter().enumerate() {
                for angular in
                    scalar_pair_density_angular(left_orbital.angular(), right_orbital.angular())
                {
                    if angular.coefficient == 0.0 {
                        continue;
                    }
                    let harmonic = lm_index(angular.channel.l, angular.channel.m)
                        .expect("compiled angular channel is valid");
                    terms[harmonic].push(ScalarDensityTerm {
                        left,
                        right,
                        angular: angular.coefficient,
                    });
                }
            }
        }
        let radial_count = site.mesh.len();
        let radial_large = site
            .orbitals
            .iter()
            .flat_map(|orbital| {
                orbital
                    .large_component()
                    .iter()
                    .zip(site.mesh.radii())
                    .map(|(&value, radius)| value / radius.get())
            })
            .collect();
        let radial_small = site
            .orbitals
            .iter()
            .flat_map(|orbital| {
                (0..radial_count).map(move |radial| {
                    orbital
                        .small_component()
                        .map_or(0.0, |small| small[radial] / site.mesh.radii()[radial].get())
                })
            })
            .collect();
        Self {
            harmonics,
            terms,
            radial_large,
            radial_small,
            radial_count,
            orbital_count: site.orbitals.len(),
        }
    }
}

impl FullSpinorDensitySiteBasis {
    pub fn new(
        mesh: ExponentialMesh,
        channels: Vec<RelativisticChannel>,
        orbitals: Vec<SpinorSphereOrbital>,
    ) -> Self {
        let contraction = SpinorDensityContraction::compile(&mesh, &orbitals);
        Self {
            mesh,
            channels,
            orbitals,
            contraction,
        }
    }
}

impl SpinorDensityContraction {
    fn compile(mesh: &ExponentialMesh, orbitals: &[SpinorSphereOrbital]) -> Self {
        // Magnetic channels share radials. Identify only exact sample matches;
        // no angular channel or small component is screened out.
        let mut radial_ids = BTreeMap::new();
        let mut representatives = Vec::new();
        let orbital_radials = orbitals
            .iter()
            .enumerate()
            .map(|(index, orbital)| {
                let key = orbital
                    .p()
                    .iter()
                    .chain(orbital.q())
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>();
                let next = representatives.len();
                *radial_ids.entry(key).or_insert_with(|| {
                    representatives.push(index);
                    next
                })
            })
            .collect::<Vec<_>>();
        let radial_count = representatives.len();
        let radial_pair_count = radial_count * (radial_count + 1) / 2;
        let l_max = orbitals
            .iter()
            .flat_map(|left| {
                orbitals.iter().map(move |right| {
                    let left = left.channel().kappa();
                    let right = right.channel().kappa();
                    (left.large_l() + right.large_l()).max(left.small_l() + right.small_l())
                })
            })
            .max()
            .unwrap_or(0);
        let harmonics = (0..=l_max)
            .flat_map(|l| {
                (-(l as i32)..=l as i32)
                    .map(move |m| Lm::new(l, m).expect("loop bounds validate magnetic channel"))
            })
            .collect::<Vec<_>>();
        let mut terms: [Vec<Vec<SpinorDensityTerm>>; 4] =
            std::array::from_fn(|_| vec![Vec::new(); harmonics.len()]);
        for (left, left_orbital) in orbitals.iter().enumerate() {
            for (right, right_orbital) in orbitals.iter().enumerate() {
                for angular in
                    spinor_pair_density_angular(left_orbital.channel(), right_orbital.channel())
                {
                    let harmonic = lm_index(angular.channel.l, angular.channel.m)
                        .expect("compiled angular channel is valid");
                    for component in 0..4 {
                        if angular.pp[component] != Complex64::default()
                            || angular.qq[component] != Complex64::default()
                        {
                            terms[component][harmonic].push(SpinorDensityTerm {
                                left,
                                right,
                                radial_pair: {
                                    let a = orbital_radials[left].min(orbital_radials[right]);
                                    let b = orbital_radials[left].max(orbital_radials[right]);
                                    a * (2 * radial_count - a + 1) / 2 + b - a
                                },
                                pp: angular.pp[component],
                                qq: angular.qq[component],
                            });
                        }
                    }
                }
            }
        }
        let mut radial_products = Vec::with_capacity(2 * radial_pair_count * mesh.len());
        for small in [false, true] {
            for (a, &left) in representatives.iter().enumerate() {
                let left = if small {
                    orbitals[left].q()
                } else {
                    orbitals[left].p()
                };
                for &right in &representatives[a..] {
                    let right = if small {
                        orbitals[right].q()
                    } else {
                        orbitals[right].p()
                    };
                    radial_products.extend(left.iter().zip(right).zip(mesh.radii()).map(
                        |((&left, &right), radius)| {
                            Complex64::new((left / radius.get()) * (right / radius.get()), 0.0)
                        },
                    ));
                }
            }
        }
        Self {
            harmonics,
            terms,
            radial_products: Arc::new(radial_products),
            radial_pair_count,
            radial_count: mesh.len(),
        }
    }
}

/// Physical site projections of every eigenvector band.
///
/// The precompiled map first forms `d = P_site C`. For the selected site
/// coordinates, each returned value is the full Hermitian quadratic form
/// `d_selected^dagger S_selected d_selected`, including all overlap cross
/// terms. The operation is independent of whether `projection` came from a
/// scalar or spinor basis.
pub fn physical_site_band_projections(
    projection: &CompiledSiteProjection,
    eigenvectors: &DenseEigenvectors,
    site_overlap: &DenseHermitianMatrix,
    selected_coordinates: &[usize],
) -> Result<Vec<f64>, DensityError> {
    let coordinate_count = projection.coordinate_count();
    if site_overlap.axis() != Axis::SiteCoordinate {
        return Err(DensityError::BandProjectionOverlapAxis {
            actual: site_overlap.axis(),
        });
    }
    if site_overlap.dimension() != coordinate_count {
        return Err(DensityError::BandProjectionOverlapDimension {
            expected: coordinate_count,
            actual: site_overlap.dimension(),
        });
    }
    let mut selected = vec![false; coordinate_count];
    for &coordinate in selected_coordinates {
        if coordinate >= coordinate_count {
            return Err(DensityError::BandProjectionCoordinate {
                coordinate,
                coordinate_count,
            });
        }
        if selected[coordinate] {
            return Err(DensityError::DuplicateBandProjectionCoordinate { coordinate });
        }
        selected[coordinate] = true;
    }

    let projected = projection.project_eigenvectors(eigenvectors)?;
    let mut weights = Vec::with_capacity(projected.band_count());
    for band in 0..projected.band_count() {
        let mut value = Complex64::new(0.0, 0.0);
        let mut scale = 0.0;
        for &left in selected_coordinates {
            for &right in selected_coordinates {
                let term = projected.at(left, band).conj()
                    * site_overlap.at(left, right)
                    * projected.at(right, band);
                value += term;
                scale += term.norm();
            }
        }
        let tolerance = BAND_PROJECTION_TOLERANCE * scale.max(1.0);
        if !value.re.is_finite() {
            return Err(DensityError::NonFiniteBandProjection {
                band,
                projection: value.re,
            });
        }
        if value.re < -tolerance {
            return Err(DensityError::NegativeBandProjection {
                band,
                projection: value.re,
                tolerance,
            });
        }
        weights.push(value.re.max(0.0));
    }
    Ok(weights)
}

/// Synthesize charge and longitudinal magnetization from occupied collinear
/// LAPW states.
///
/// Muffin-tin coefficients are formed after applying the canonical compiled
/// site projection. Interstitial coefficients use the plane-wave rows
/// directly and the normalization `exp(i(k+G)r) / sqrt(Omega)`. The returned
/// Fourier field is the smooth interstitial extension; integration over the
/// physical interstitial region is therefore performed with the step
/// function carried by [`RegionalDensity`].
pub fn synthesize_collinear_valence_density(
    geometry: InterstitialGeometry,
    layout: FourierLayout,
    sites: &[ScalarSiteBasis],
    k_points: &[CollinearKPoint<'_>],
) -> Result<RegionalDensity, DensityError> {
    validate_k_points(sites, k_points)?;
    let contractions = sites
        .iter()
        .map(ScalarDensityContraction::compile)
        .collect::<Vec<_>>();
    let mut muffin_tins = Collinear::new(site_accumulators(sites), site_accumulators(sites));
    let mut interstitial = Collinear::new(
        vec![Complex64::new(0.0, 0.0); layout.len()],
        vec![Complex64::new(0.0, 0.0); layout.len()],
    );
    let inverse_volume = 1.0 / geometry.cell_volume().get();

    for k_point in k_points {
        accumulate_spin(
            k_point,
            k_point.solutions.up,
            k_point.occupations.up,
            SpinDensityAccumulator {
                sites,
                contractions: &contractions,
                muffin_tins: &mut muffin_tins.up,
                layout: &layout,
                interstitial: &mut interstitial.up,
                inverse_volume,
            },
        )?;
        accumulate_spin(
            k_point,
            k_point.solutions.down,
            k_point.occupations.down,
            SpinDensityAccumulator {
                sites,
                contractions: &contractions,
                muffin_tins: &mut muffin_tins.down,
                layout: &layout,
                interstitial: &mut interstitial.down,
                inverse_volume,
            },
        )?;
    }

    let muffin_tins = Collinear::new(
        finish_muffin_tins(sites, muffin_tins.up)?,
        finish_muffin_tins(sites, muffin_tins.down)?,
    );
    enforce_fourier_reality(&layout, &mut interstitial.up);
    enforce_fourier_reality(&layout, &mut interstitial.down);
    let interstitial = Collinear::new(
        InterstitialField::from_fourier_field(muffintin_core::HermitianFourierField::new(
            layout.clone(),
            interstitial.up,
        )?),
        InterstitialField::from_fourier_field(muffintin_core::HermitianFourierField::new(
            layout,
            interstitial.down,
        )?),
    );
    regional_density_from_collinear(geometry, muffin_tins, interstitial)
}

/// Synthesize the complete Pauli density of SOC second-variation orbitals.
///
/// The two scalar solutions are the up/down components of the same spinor
/// bands and therefore share one occupation vector. Cross-component density
/// matrices generate the transverse magnetization instead of being discarded
/// as independent collinear channels.
pub fn synthesize_second_variation_valence_density(
    geometry: InterstitialGeometry,
    layout: FourierLayout,
    sites: &[ScalarSiteBasis],
    k_points: &[SecondVariationKPoint<'_>],
) -> Result<RegionalDensity, DensityError> {
    let validation = k_points
        .iter()
        .map(|point| CollinearKPoint {
            weight: point.weight,
            compiled: point.compiled,
            solutions: point.solutions.clone(),
            occupations: Collinear::new(point.occupations, point.occupations),
        })
        .collect::<Vec<_>>();
    validate_k_points(sites, &validation)?;
    if layout.index([0, 0, 0]).is_none() {
        return Err(DensityError::MissingZeroVector);
    }
    let contractions = sites
        .iter()
        .map(ScalarDensityContraction::compile)
        .collect::<Vec<_>>();
    let mut muffin_tins: [Vec<BTreeMap<Lm, Vec<Complex64>>>; 4] =
        std::array::from_fn(|_| site_accumulators(sites));
    let mut interstitial: [Vec<Complex64>; 4] =
        std::array::from_fn(|_| vec![Complex64::default(); layout.len()]);
    let inverse_volume = 1.0 / geometry.cell_volume().get();
    let mut expected_electron_count = 0.0;

    for point in k_points {
        expected_electron_count += point.weight * point.occupations.iter().sum::<f64>();
        let state_weights = ComplexTensor::from_host_row_major(
            &[point.occupations.len()],
            &[Axis::Band],
            point
                .occupations
                .iter()
                .map(|&occupation| Complex64::new(point.weight * occupation, 0.0))
                .collect(),
        )?;
        for (site, contraction) in contractions.iter().enumerate() {
            let projection = CompiledSiteProjection::scalar(point.compiled, site)?;
            let up = projection.project_eigenvectors(&point.solutions.up.eigenvectors)?;
            let down = projection.project_eigenvectors(&point.solutions.down.eigenvectors)?;
            let components =
                pauli_density_matrices(up.as_tensor(), down.as_tensor(), &state_weights)?;
            for component in 0..4 {
                let density = synthesize_scalar_site_density(contraction, &components[component]);
                merge_sphere_accumulator(&mut muffin_tins[component][site], &density);
            }
        }

        let plane_wave_count = point.compiled.plane_waves.len();
        if plane_wave_count != 0 {
            let up = select_plane_wave_bands(
                &point.solutions.up.eigenvectors,
                plane_wave_count,
                point.occupations.len(),
            )?;
            let down = select_plane_wave_bands(
                &point.solutions.down.eigenvectors,
                plane_wave_count,
                point.occupations.len(),
            )?;
            let components = pauli_density_matrices(&up, &down, &state_weights)?;
            for (component, density) in components.iter().enumerate() {
                accumulate_scalar_interstitial_matrix(
                    &point.compiled.plane_waves,
                    &layout,
                    &density.to_host_row_major(),
                    &mut interstitial[component],
                    inverse_volume,
                )?;
            }
        }
    }

    let muffin_tins = muffin_tins
        .into_iter()
        .map(|accumulators| finish_muffin_tins(sites, accumulators))
        .collect::<Result<Vec<_>, _>>()?;
    for coefficients in &mut interstitial {
        enforce_fourier_reality(&layout, coefficients);
    }
    let interstitial = interstitial
        .into_iter()
        .map(|coefficients| {
            Ok(InterstitialField::from_fourier_field(
                muffintin_core::HermitianFourierField::new(layout.clone(), coefficients)?,
            ))
        })
        .collect::<Result<Vec<_>, DensityError>>()?;
    let mut muffin_tins = muffin_tins.into_iter();
    let mut interstitial = interstitial.into_iter();
    let charge = RegionalScalarField::new(
        geometry.clone(),
        muffin_tins
            .next()
            .expect("four components were constructed"),
        interstitial
            .next()
            .expect("four components were constructed"),
    )?;
    let mut magnetization = Vec::with_capacity(3);
    for _ in 0..3 {
        magnetization.push(RegionalScalarField::new(
            geometry.clone(),
            muffin_tins
                .next()
                .expect("four components were constructed"),
            interstitial
                .next()
                .expect("four components were constructed"),
        )?);
    }
    let magnetization = magnetization
        .try_into()
        .expect("exactly three spin components were constructed");
    let result = RegionalDensity::new(charge, magnetization)?;
    let actual = electron_count(&result)?;
    let tolerance = SPINOR_COUNT_TOLERANCE * expected_electron_count.abs().max(1.0);
    if (actual - expected_electron_count).abs() > tolerance {
        return Err(DensityError::SpinorChargeMismatch {
            expected: expected_electron_count,
            actual,
            tolerance,
        });
    }
    Ok(result)
}

/// Synthesize charge and Cartesian spin density from occupied full-spinor states.
///
/// Site coefficients use the canonical [`CompiledSiteProjection`] and retain
/// both `P` and `Q` sectors. Interstitial coefficients use the two Pauli rows
/// `spin * n_G + G` directly. Every reciprocal difference must be present in
/// `layout`; the final fields are explicitly Hermitianized before construction.
pub fn synthesize_full_spinor_valence_density(
    geometry: InterstitialGeometry,
    layout: FourierLayout,
    sites: &[FullSpinorDensitySiteBasis],
    k_points: &[FullSpinorKPoint<'_>],
) -> Result<RegionalDensity, DensityError> {
    validate_full_spinor_k_points(&geometry, &layout, sites, k_points)?;
    if layout.index([0, 0, 0]).is_none() {
        return Err(DensityError::MissingZeroVector);
    }
    let mut muffin_tins: [Vec<BTreeMap<Lm, Vec<Complex64>>>; 4] =
        std::array::from_fn(|_| site_accumulators_spinor(sites));
    let mut interstitial: [Vec<Complex64>; 4] =
        std::array::from_fn(|_| vec![Complex64::new(0.0, 0.0); layout.len()]);
    let inverse_volume = 1.0 / geometry.cell_volume().get();
    let mut expected_electron_count = 0.0;

    for k_point in k_points {
        expected_electron_count +=
            k_point.weight * k_point.occupations.iter().copied().sum::<f64>();
        for (site_index, site) in sites.iter().enumerate() {
            let projected =
                CompiledSiteProjection::spinor(k_point.compiled, site_index, &site.channels)?
                    .project_eigenvectors(&k_point.solution.eigenvectors)?;
            let state_weights = ComplexTensor::from_host_row_major(
                &[k_point.occupations.len()],
                &[Axis::Band],
                k_point
                    .occupations
                    .iter()
                    .map(|&occupation| Complex64::new(k_point.weight * occupation, 0.0))
                    .collect(),
            )?;
            let weighted = einsum("ib,b->ib", &[projected.as_tensor(), &state_weights])?;
            let density_matrix = einsum(
                "ib,jb->ij",
                &[&projected.as_tensor().conjugate(), &weighted],
            )?;
            // The pair density of two site orbitals does not depend on the
            // band, so contract the occupied bands into one site density
            // matrix element and project each orbital pair once.
            let site_density = synthesize_spinor_site_density(site, &density_matrix)?;
            for component in 0..4 {
                merge_sphere_accumulator(
                    &mut muffin_tins[component][site_index],
                    &site_density[component],
                );
            }
        }
        accumulate_spinor_interstitial(k_point, &layout, &mut interstitial, inverse_volume)?;
    }

    let muffin_tins = muffin_tins
        .into_iter()
        .map(|accumulators| finish_spinor_muffin_tins(sites, accumulators))
        .collect::<Result<Vec<_>, _>>()?;
    for coefficients in &mut interstitial {
        enforce_fourier_reality(&layout, coefficients);
    }
    let interstitial = interstitial
        .into_iter()
        .map(|coefficients| {
            Ok(InterstitialField::from_fourier_field(
                muffintin_core::HermitianFourierField::new(layout.clone(), coefficients)?,
            ))
        })
        .collect::<Result<Vec<_>, DensityError>>()?;

    let mut muffin_tins = muffin_tins.into_iter();
    let mut interstitial = interstitial.into_iter();
    let charge = RegionalScalarField::new(
        geometry.clone(),
        muffin_tins
            .next()
            .expect("four components were constructed"),
        interstitial
            .next()
            .expect("four components were constructed"),
    )?;
    let mut spin = Vec::with_capacity(3);
    for _ in 0..3 {
        spin.push(RegionalScalarField::new(
            geometry.clone(),
            muffin_tins
                .next()
                .expect("four components were constructed"),
            interstitial
                .next()
                .expect("four components were constructed"),
        )?);
    }
    let spin: [RegionalScalarField; 3] = spin
        .try_into()
        .expect("exactly three spin components were constructed");
    let result = RegionalDensity::new(charge, spin)?;
    let actual = electron_count(&result)?;
    let tolerance = SPINOR_COUNT_TOLERANCE * expected_electron_count.abs().max(1.0);
    if (actual - expected_electron_count).abs() > tolerance {
        return Err(DensityError::SpinorChargeMismatch {
            expected: expected_electron_count,
            actual,
            tolerance,
        });
    }
    Ok(result)
}

fn synthesize_spinor_site_density(
    site: &FullSpinorDensitySiteBasis,
    density_matrix: &ComplexTensor,
) -> Result<[BTreeMap<Lm, Vec<Complex64>>; 4], DensityError> {
    let contraction = &site.contraction;
    let orbital_count = site.orbitals.len();
    let density_matrix = density_matrix.to_host_row_major();
    let harmonic_count = contraction.harmonics.len();
    let pair_count = contraction.radial_pair_count;
    let coefficients = (0..4 * harmonic_count)
        .into_par_iter()
        .flat_map_iter(|block| {
            let component = block / harmonic_count;
            let harmonic = block % harmonic_count;
            let mut values = vec![Complex64::default(); 2 * pair_count];
            for term in &contraction.terms[component][harmonic] {
                let coefficient = density_matrix[term.left * orbital_count + term.right];
                if coefficient == Complex64::default() {
                    continue;
                }
                values[term.radial_pair] += coefficient * term.pp;
                values[pair_count + term.radial_pair] += coefficient * term.qq;
            }
            values
        })
        .collect::<Vec<_>>();
    let coefficients = ComplexTensor::from_host_row_major(
        &[4 * harmonic_count, 2 * pair_count],
        &[Axis::FieldChannel, Axis::PairColumn],
        coefficients,
    )?;
    let products = ComplexTensor::from_host_row_major(
        &[2 * pair_count, contraction.radial_count],
        &[Axis::PairColumn, Axis::RadialPoint],
        contraction.radial_products.as_ref().clone(),
    )?;
    let values = einsum("ap,pr->ar", &[&coefficients, &products])?.to_host_row_major();
    let mut density: [BTreeMap<Lm, Vec<Complex64>>; 4] = std::array::from_fn(|_| BTreeMap::new());
    for (block, values) in values.chunks_exact(contraction.radial_count).enumerate() {
        density[block / harmonic_count].insert(
            contraction.harmonics[block % harmonic_count],
            values.to_vec(),
        );
    }
    Ok(density)
}

/// Complete-shell four-component core density on a muffin-tin mesh.
///
/// `occupation` is the number of electrons in the `(n,kappa)` shell and may
/// range from zero through `2j+1`. Both `P^2` and `Q^2` enter the monopole.
pub fn core_shell_density(
    mesh: &ExponentialMesh,
    solution: &CoreDiracSolution,
    occupation: f64,
) -> Result<SphereField, DensityError> {
    let capacity = f64::from(solution.state.kappa.degeneracy());
    if !occupation.is_finite() || occupation < 0.0 || occupation > capacity {
        return Err(DensityError::InvalidCoreOccupation {
            occupation,
            capacity,
        });
    }
    if solution.p.len() < mesh.len() || solution.q.len() < mesh.len() {
        return Err(DensityError::CoreMeshLength {
            expected_at_least: mesh.len(),
            p: solution.p.len(),
            q: solution.q.len(),
        });
    }
    let normalization = occupation / (4.0 * PI).sqrt();
    let monopole = mesh
        .radii()
        .iter()
        .enumerate()
        .map(|(index, radius)| {
            let probability = solution.p[index].powi(2) + solution.q[index].powi(2);
            Complex64::new(normalization * probability / radius.get().powi(2), 0.0)
        })
        .collect();
    SphereField::new(HarmonicConvention::Complex, [((0, 0), monopole)]).map_err(Into::into)
}

/// Add core monopoles to an existing valence muffin-tin density.
pub fn add_core_density(
    valence: &mut MuffinTinField,
    shells: &[(CoreDiracSolution, f64)],
) -> Result<(), DensityError> {
    if shells.is_empty() {
        return Ok(());
    }
    let mut channels = valence
        .field()
        .channels()
        .map(|(lm, values)| (lm, values.to_vec()))
        .collect::<BTreeMap<_, _>>();
    let monopole = channels
        .entry(Lm::new(0, 0).expect("the monopole is valid"))
        .or_insert_with(|| vec![Complex64::new(0.0, 0.0); valence.mesh().len()]);
    for (solution, occupation) in shells {
        let core = core_shell_density(valence.mesh(), solution, *occupation)?;
        let core_monopole = core.channel(0, 0).expect("core density has a monopole");
        for (target, &source) in monopole.iter_mut().zip(core_monopole) {
            *target += source;
        }
    }
    let updated = SphereField::new(
        valence.field().convention(),
        channels
            .into_iter()
            .map(|(lm, values)| ((lm.l, lm.m), values)),
    )?;
    *valence = MuffinTinField::new(valence.mesh().clone(), updated)?;
    Ok(())
}

/// Integrate charge over the non-overlapping muffin-tin and interstitial
/// regions of one cell.
pub fn electron_count(density: &RegionalDensity) -> Result<f64, DensityError> {
    scalar_field_integral(density.charge())
}

/// Correct only roundoff-scale charge drift through the charge `G=0`
/// interstitial coefficient, preserving all magnetization and every other
/// regional coefficient.
pub fn correct_electron_count(
    density: RegionalDensity,
    target: f64,
    tolerance: f64,
) -> Result<RegionalDensity, DensityError> {
    if !target.is_finite() || target < 0.0 {
        return Err(DensityError::InvalidTargetElectronCount(target));
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(DensityError::InvalidChargeTolerance(tolerance));
    }
    let actual = electron_count(&density)?;
    let mismatch = target - actual;
    if mismatch.abs() > tolerance {
        return Err(DensityError::ChargeMismatch {
            target,
            actual,
            tolerance,
        });
    }
    if mismatch == 0.0 {
        return Ok(density);
    }
    let layout = density.charge().interstitial().layout();
    let Some(zero) = layout.index([0, 0, 0]) else {
        return Err(DensityError::MissingZeroVector);
    };
    let theta_zero = density
        .geometry()
        .coefficient([muffintin_core::InverseBohr(0.0); 3])?
        .re;
    let interstitial_volume = density.geometry().cell_volume().get() * theta_zero;
    if !interstitial_volume.is_finite() || interstitial_volume <= 0.0 {
        return Err(DensityError::EmptyInterstitial);
    }
    let mut coefficients = density
        .charge()
        .interstitial()
        .field()
        .coefficients()
        .to_vec();
    coefficients[zero].re += mismatch / interstitial_volume;
    let charge = RegionalScalarField::new(
        density.geometry().clone(),
        density.charge().muffin_tins().to_vec(),
        InterstitialField::from_fourier_field(muffintin_core::HermitianFourierField::new(
            layout.clone(),
            coefficients,
        )?),
    )?;
    RegionalDensity::new(charge, density.magnetization().clone()).map_err(Into::into)
}

fn regional_density_from_collinear(
    geometry: InterstitialGeometry,
    muffin_tins: Collinear<Vec<MuffinTinField>>,
    interstitial: Collinear<InterstitialField>,
) -> Result<RegionalDensity, DensityError> {
    let up = RegionalScalarField::new(geometry.clone(), muffin_tins.up, interstitial.up)?;
    let down = RegionalScalarField::new(geometry, muffin_tins.down, interstitial.down)?;
    let mut charge = up.clone();
    charge.add_scaled(1.0, &down)?;
    let mut mz = up;
    mz.add_scaled(-1.0, &down)?;
    let zero = charge.zero_like();
    RegionalDensity::new(charge, [zero.clone(), zero, mz]).map_err(Into::into)
}

fn validate_full_spinor_k_points(
    regional_geometry: &InterstitialGeometry,
    layout: &FourierLayout,
    sites: &[FullSpinorDensitySiteBasis],
    k_points: &[FullSpinorKPoint<'_>],
) -> Result<(), DensityError> {
    if k_points.is_empty() {
        return Err(DensityError::EmptyKMesh);
    }
    let mut weight_sum = 0.0;
    for (k_index, k_point) in k_points.iter().enumerate() {
        if !k_point.weight.is_finite() || k_point.weight < 0.0 {
            return Err(DensityError::InvalidKWeight {
                k_index,
                weight: k_point.weight,
            });
        }
        weight_sum += k_point.weight;
        if k_point.compiled.site_count() != sites.len() {
            return Err(DensityError::SiteCount {
                k_index,
                expected: sites.len(),
                actual: k_point.compiled.site_count(),
            });
        }
        for wave in &k_point.compiled.plane_waves {
            if layout.reciprocal().cartesian(wave.g.index) != wave.g.cartesian {
                return Err(DensityError::SpinorReciprocalMismatch { k_index });
            }
        }
        if k_point.solution.eigenvectors.rows() != k_point.compiled.layout.dimension() {
            return Err(DensityError::EigenvectorBasis {
                k_index,
                expected: k_point.compiled.layout.dimension(),
                actual: k_point.solution.eigenvectors.rows(),
            });
        }
        if k_point.occupations.len() != k_point.solution.eigenvectors.columns() {
            return Err(DensityError::OccupationCount {
                k_index,
                expected: k_point.solution.eigenvectors.columns(),
                actual: k_point.occupations.len(),
            });
        }
        for (band, &occupation) in k_point.occupations.iter().enumerate() {
            if !occupation.is_finite() || !(0.0..=1.0).contains(&occupation) {
                return Err(DensityError::InvalidOccupation {
                    k_index,
                    band,
                    occupation,
                });
            }
        }
        for (site_index, site) in sites.iter().enumerate() {
            let projection =
                CompiledSiteProjection::spinor(k_point.compiled, site_index, &site.channels)?;
            if projection.coordinate_count() != site.orbitals.len() {
                return Err(DensityError::SiteOrbitalCount {
                    k_index,
                    site: site_index,
                    expected: projection.coordinate_count(),
                    actual: site.orbitals.len(),
                });
            }
            let expected_channels =
                full_spinor_coordinate_channels(k_point.compiled, site_index, &site.channels)?;
            for (coordinate, (orbital, expected)) in
                site.orbitals.iter().zip(&expected_channels).enumerate()
            {
                if orbital.channel() != *expected {
                    return Err(DensityError::SpinorCoordinateChannel {
                        k_index,
                        site: site_index,
                        coordinate,
                        expected: *expected,
                        actual: orbital.channel(),
                    });
                }
                if orbital.p().len() != site.mesh.len() || orbital.q().len() != site.mesh.len() {
                    return Err(DensityError::SiteOrbitalMesh { site: site_index });
                }
            }
            let geometry = k_point
                .compiled
                .site_geometry
                .get(site_index)
                .ok_or(DensityError::SpinorSiteGeometry { site: site_index })?;
            let sphere = regional_geometry
                .spheres()
                .get(site_index)
                .ok_or(DensityError::SpinorRegionalGeometry { site: site_index })?;
            if geometry.position != sphere.center || geometry.radius != sphere.radius {
                return Err(DensityError::SpinorRegionalGeometry { site: site_index });
            }
            if site.mesh.last() != geometry.radius {
                return Err(DensityError::SpinorSiteMeshRadius {
                    k_index,
                    site: site_index,
                    expected: geometry.radius.get(),
                    actual: site.mesh.last().get(),
                });
            }
        }
    }
    if (weight_sum - 1.0).abs() > WEIGHT_TOLERANCE * k_points.len() as f64 {
        return Err(DensityError::KWeightSum(weight_sum));
    }
    Ok(())
}

fn full_spinor_coordinate_channels(
    compiled: &SpinorCompiledBasis,
    site: usize,
    apw_channels: &[RelativisticChannel],
) -> Result<Vec<RelativisticChannel>, DensityError> {
    let local_orbitals = compiled
        .layout
        .site_layout(site)
        .ok_or(DensityError::SpinorSiteLayout { site })?;
    let mut result = Vec::with_capacity(2 * apw_channels.len() + local_orbitals.len());
    for &channel in apw_channels {
        result.extend([channel, channel]);
    }
    for &(kappa, count) in local_orbitals.counts_by_kappa() {
        for channel in kappa.channels() {
            result.extend(std::iter::repeat_n(channel, count));
        }
    }
    Ok(result)
}

fn accumulate_spinor_interstitial(
    k_point: &FullSpinorKPoint<'_>,
    layout: &FourierLayout,
    interstitial: &mut [Vec<Complex64>; 4],
    inverse_volume: f64,
) -> Result<(), DensityError> {
    let n_g = k_point.compiled.layout.spatial_plane_wave_count();
    for (band, &occupation) in k_point.occupations.iter().enumerate() {
        let state_weight = k_point.weight * occupation * inverse_volume;
        if state_weight == 0.0 {
            continue;
        }
        for (left, left_wave) in k_point.compiled.plane_waves.iter().enumerate() {
            let left_up = k_point.solution.eigenvectors.at(left, band).conj();
            let left_down = k_point.solution.eigenvectors.at(n_g + left, band).conj();
            for (right, right_wave) in k_point.compiled.plane_waves.iter().enumerate() {
                let difference = reciprocal_difference(right_wave.g.index, left_wave.g.index)?;
                let position = layout
                    .index(difference)
                    .ok_or(DensityError::MissingReciprocalDifference { g: difference })?;
                let right_up = k_point.solution.eigenvectors.at(right, band);
                let right_down = k_point.solution.eigenvectors.at(n_g + right, band);
                let up_up = left_up * right_up;
                let down_down = left_down * right_down;
                let up_down = left_up * right_down;
                let down_up = left_down * right_up;
                interstitial[0][position] += state_weight * (up_up + down_down);
                interstitial[1][position] += state_weight * (up_down + down_up);
                interstitial[2][position] += state_weight
                    * (Complex64::new(0.0, -1.0) * up_down + Complex64::new(0.0, 1.0) * down_up);
                interstitial[3][position] += state_weight * (up_up - down_down);
            }
        }
    }
    Ok(())
}

fn reciprocal_difference(right: [i32; 3], left: [i32; 3]) -> Result<[i32; 3], DensityError> {
    let difference = [
        right[0].checked_sub(left[0]),
        right[1].checked_sub(left[1]),
        right[2].checked_sub(left[2]),
    ];
    let [Some(g0), Some(g1), Some(g2)] = difference else {
        return Err(DensityError::ReciprocalDifferenceOverflow);
    };
    Ok([g0, g1, g2])
}

pub(crate) fn scalar_field_integral(field: &RegionalScalarField) -> Result<f64, DensityError> {
    let mut count = Complex64::new(0.0, 0.0);
    for muffin_tin in field.muffin_tins() {
        if let Some(monopole) = muffin_tin.field().channel(0, 0) {
            let radii = muffin_tin.mesh().radii();
            let real = monopole
                .iter()
                .zip(radii)
                .map(|(&value, radius)| value.re * radius.get().powi(2))
                .collect::<Vec<_>>();
            let imaginary = monopole
                .iter()
                .zip(radii)
                .map(|(&value, radius)| value.im * radius.get().powi(2))
                .collect::<Vec<_>>();
            count += (4.0 * PI).sqrt()
                * Complex64::new(
                    muffin_tin.mesh().integrate(&real)?,
                    muffin_tin.mesh().integrate(&imaginary)?,
                );
        }
    }
    let reciprocal = field.interstitial().layout().reciprocal();
    for (vector, &coefficient) in field.interstitial().field().iter() {
        count += field.geometry().cell_volume().get()
            * field
                .geometry()
                .coefficient(reciprocal.cartesian(vector.index.map(|component| -component)))?
            * coefficient;
    }
    let tolerance = 4096.0 * f64::EPSILON * count.re.abs().max(1.0);
    if count.im.abs() > tolerance {
        Err(DensityError::ComplexElectronCount {
            real: count.re,
            imaginary: count.im,
        })
    } else {
        Ok(count.re)
    }
}

fn validate_k_points(
    sites: &[ScalarSiteBasis],
    k_points: &[CollinearKPoint<'_>],
) -> Result<(), DensityError> {
    if k_points.is_empty() {
        return Err(DensityError::EmptyKMesh);
    }
    let mut weight_sum = 0.0;
    for (k_index, k_point) in k_points.iter().enumerate() {
        if !k_point.weight.is_finite() || k_point.weight < 0.0 {
            return Err(DensityError::InvalidKWeight {
                k_index,
                weight: k_point.weight,
            });
        }
        weight_sum += k_point.weight;
        if k_point.compiled.site_count() != sites.len() {
            return Err(DensityError::SiteCount {
                k_index,
                expected: sites.len(),
                actual: k_point.compiled.site_count(),
            });
        }
        validate_spin(
            k_index,
            k_point,
            k_point.solutions.up,
            k_point.occupations.up,
        )?;
        validate_spin(
            k_index,
            k_point,
            k_point.solutions.down,
            k_point.occupations.down,
        )?;
        for (site_index, site) in sites.iter().enumerate() {
            let projection = CompiledSiteProjection::scalar(k_point.compiled, site_index)?;
            if projection.coordinate_count() != site.orbitals.len() {
                return Err(DensityError::SiteOrbitalCount {
                    k_index,
                    site: site_index,
                    expected: projection.coordinate_count(),
                    actual: site.orbitals.len(),
                });
            }
            if site.orbitals.iter().any(|orbital| {
                orbital.large_component().len() != site.mesh.len()
                    || orbital
                        .small_component()
                        .is_some_and(|small| small.len() != site.mesh.len())
            }) {
                return Err(DensityError::SiteOrbitalMesh { site: site_index });
            }
        }
    }
    if (weight_sum - 1.0).abs() > WEIGHT_TOLERANCE * k_points.len() as f64 {
        return Err(DensityError::KWeightSum(weight_sum));
    }
    Ok(())
}

fn validate_spin(
    k_index: usize,
    k_point: &CollinearKPoint<'_>,
    solution: &GeneralizedEigensolution,
    occupations: &[f64],
) -> Result<(), DensityError> {
    if solution.eigenvectors.rows() != k_point.compiled.layout.dimension() {
        return Err(DensityError::EigenvectorBasis {
            k_index,
            expected: k_point.compiled.layout.dimension(),
            actual: solution.eigenvectors.rows(),
        });
    }
    if occupations.len() != solution.eigenvectors.columns() {
        return Err(DensityError::OccupationCount {
            k_index,
            expected: solution.eigenvectors.columns(),
            actual: occupations.len(),
        });
    }
    for (band, &occupation) in occupations.iter().enumerate() {
        if !occupation.is_finite() || !(0.0..=1.0).contains(&occupation) {
            return Err(DensityError::InvalidOccupation {
                k_index,
                band,
                occupation,
            });
        }
    }
    Ok(())
}

struct SpinDensityAccumulator<'a> {
    sites: &'a [ScalarSiteBasis],
    contractions: &'a [ScalarDensityContraction],
    muffin_tins: &'a mut [BTreeMap<Lm, Vec<Complex64>>],
    layout: &'a FourierLayout,
    interstitial: &'a mut [Complex64],
    inverse_volume: f64,
}

fn accumulate_spin(
    k_point: &CollinearKPoint<'_>,
    solution: &GeneralizedEigensolution,
    occupations: &[f64],
    accumulator: SpinDensityAccumulator<'_>,
) -> Result<(), DensityError> {
    let SpinDensityAccumulator {
        sites,
        contractions,
        muffin_tins,
        layout,
        interstitial,
        inverse_volume,
    } = accumulator;
    let state_weights = ComplexTensor::from_host_row_major(
        &[occupations.len()],
        &[Axis::Band],
        occupations
            .iter()
            .map(|&occupation| Complex64::new(k_point.weight * occupation, 0.0))
            .collect(),
    )?;
    for (site_index, ((_site, contraction), muffin_tin)) in
        sites.iter().zip(contractions).zip(muffin_tins).enumerate()
    {
        let projected = CompiledSiteProjection::scalar(k_point.compiled, site_index)?
            .project_eigenvectors(&solution.eigenvectors)?;
        let weighted = einsum("ib,b->ib", &[projected.as_tensor(), &state_weights])?;
        let density_matrix = einsum(
            "ib,jb->ij",
            &[&projected.as_tensor().conjugate(), &weighted],
        )?;
        let site_density = synthesize_scalar_site_density(contraction, &density_matrix);
        merge_sphere_accumulator(muffin_tin, &site_density);
    }

    let plane_waves = &k_point.compiled.plane_waves;
    if !plane_waves.is_empty() {
        let coefficients =
            select_plane_wave_bands(&solution.eigenvectors, plane_waves.len(), occupations.len())?;
        let weighted = einsum("ib,b->ib", &[&coefficients, &state_weights])?;
        let density_matrix =
            einsum("ib,jb->ij", &[&coefficients.conjugate(), &weighted])?.to_host_row_major();
        for (left, left_wave) in plane_waves.iter().enumerate() {
            for (right, right_wave) in plane_waves.iter().enumerate() {
                let difference = [
                    right_wave.g.index[0].checked_sub(left_wave.g.index[0]),
                    right_wave.g.index[1].checked_sub(left_wave.g.index[1]),
                    right_wave.g.index[2].checked_sub(left_wave.g.index[2]),
                ];
                let [Some(g0), Some(g1), Some(g2)] = difference else {
                    return Err(DensityError::ReciprocalDifferenceOverflow);
                };
                if let Some(position) = layout.index([g0, g1, g2]) {
                    interstitial[position] +=
                        inverse_volume * density_matrix[left * plane_waves.len() + right];
                }
            }
        }
    }
    Ok(())
}

fn pauli_density_matrices(
    up: &ComplexTensor,
    down: &ComplexTensor,
    state_weights: &ComplexTensor,
) -> Result<[ComplexTensor; 4], DensityError> {
    if up.shape() != down.shape() || up.axes() != down.axes() || up.shape().len() != 2 {
        return Err(DensityError::Tensor(TensorError::Shape {
            expected: up.shape(),
            actual: down.shape(),
        }));
    }
    let shape = up.shape()[0];
    let doubled = 2 * shape;
    let mut coefficients = up.to_host_row_major();
    coefficients.extend(down.to_host_row_major());
    let spinor =
        ComplexTensor::from_host_row_major(&[doubled, up.shape()[1]], up.axes(), coefficients)?;
    // One Gram contraction forms all four Pauli spin blocks. This avoids four
    // separate small GEMMs and their native parallel synchronization overhead.
    let weighted = einsum("ib,b->ib", &[&spinor, state_weights])?;
    let gram = einsum("ib,jb->ij", &[&spinor.conjugate(), &weighted])?.to_host_row_major();
    let axes = [up.axes()[0], up.axes()[0]];
    let build = |component: usize| {
        let values = (0..shape * shape)
            .map(|index| {
                let row = index / shape;
                let column = index % shape;
                let uu = gram[row * doubled + column];
                let ud = gram[row * doubled + shape + column];
                let du = gram[(shape + row) * doubled + column];
                let dd = gram[(shape + row) * doubled + shape + column];
                match component {
                    0 => uu + dd,
                    1 => ud + du,
                    2 => Complex64::new(0.0, -1.0) * ud + Complex64::new(0.0, 1.0) * du,
                    3 => uu - dd,
                    _ => unreachable!(),
                }
            })
            .collect();
        ComplexTensor::from_host_row_major(&[shape, shape], &axes, values)
    };
    Ok([build(0)?, build(1)?, build(2)?, build(3)?])
}

fn accumulate_scalar_interstitial_matrix(
    plane_waves: &[muffintin_envelope::PlaneWave],
    layout: &FourierLayout,
    density_matrix: &[Complex64],
    interstitial: &mut [Complex64],
    inverse_volume: f64,
) -> Result<(), DensityError> {
    for (left, left_wave) in plane_waves.iter().enumerate() {
        for (right, right_wave) in plane_waves.iter().enumerate() {
            let difference = [
                right_wave.g.index[0].checked_sub(left_wave.g.index[0]),
                right_wave.g.index[1].checked_sub(left_wave.g.index[1]),
                right_wave.g.index[2].checked_sub(left_wave.g.index[2]),
            ];
            let [Some(g0), Some(g1), Some(g2)] = difference else {
                return Err(DensityError::ReciprocalDifferenceOverflow);
            };
            if let Some(position) = layout.index([g0, g1, g2]) {
                interstitial[position] +=
                    inverse_volume * density_matrix[left * plane_waves.len() + right];
            }
        }
    }
    Ok(())
}

fn synthesize_scalar_site_density(
    contraction: &ScalarDensityContraction,
    density_matrix: &ComplexTensor,
) -> BTreeMap<Lm, Vec<Complex64>> {
    let density_matrix = density_matrix.to_host_row_major();
    let blocks = contraction
        .terms
        .par_iter()
        .enumerate()
        .map(|(harmonic, terms)| {
            let mut values = vec![Complex64::default(); contraction.radial_count];
            for term in terms {
                let coefficient = density_matrix
                    [term.left * contraction.orbital_count + term.right]
                    * term.angular;
                if coefficient == Complex64::default() {
                    continue;
                }
                let left_large = &contraction.radial_large[term.left * contraction.radial_count
                    ..(term.left + 1) * contraction.radial_count];
                let right_large = &contraction.radial_large[term.right * contraction.radial_count
                    ..(term.right + 1) * contraction.radial_count];
                let left_small = &contraction.radial_small[term.left * contraction.radial_count
                    ..(term.left + 1) * contraction.radial_count];
                let right_small = &contraction.radial_small[term.right * contraction.radial_count
                    ..(term.right + 1) * contraction.radial_count];
                for radial in 0..contraction.radial_count {
                    values[radial] += coefficient
                        * (left_large[radial] * right_large[radial]
                            + left_small[radial] * right_small[radial]);
                }
            }
            (contraction.harmonics[harmonic], values)
        })
        .collect::<Vec<_>>();
    blocks.into_iter().collect()
}

fn select_plane_wave_bands(
    eigenvectors: &DenseEigenvectors,
    plane_wave_count: usize,
    band_count: usize,
) -> Result<ComplexTensor, DensityError> {
    let mut values = Vec::with_capacity(plane_wave_count * band_count);
    for plane_wave in 0..plane_wave_count {
        for band in 0..band_count {
            values.push(eigenvectors.at(plane_wave, band));
        }
    }
    Ok(ComplexTensor::from_host_row_major(
        &[plane_wave_count, band_count],
        &[Axis::GlobalBasis, Axis::Band],
        values,
    )?)
}

fn site_accumulators(sites: &[ScalarSiteBasis]) -> Vec<BTreeMap<Lm, Vec<Complex64>>> {
    sites
        .iter()
        .map(|site| {
            let l_max = site
                .orbitals
                .iter()
                .flat_map(|left| {
                    site.orbitals
                        .iter()
                        .map(move |right| left.angular().l + right.angular().l)
                })
                .max();
            l_max.map_or_else(BTreeMap::new, |l_max| {
                (0..=l_max)
                    .flat_map(|l| {
                        (-(l as i32)..=l as i32).map(move |m| {
                            (
                                Lm::new(l, m).expect("loop bounds validate magnetic channel"),
                                vec![Complex64::new(0.0, 0.0); site.mesh.len()],
                            )
                        })
                    })
                    .collect()
            })
        })
        .collect()
}

fn site_accumulators_spinor(
    sites: &[FullSpinorDensitySiteBasis],
) -> Vec<BTreeMap<Lm, Vec<Complex64>>> {
    sites.iter().map(spinor_site_accumulator).collect()
}

fn spinor_site_accumulator(site: &FullSpinorDensitySiteBasis) -> BTreeMap<Lm, Vec<Complex64>> {
    let l_max = site
        .orbitals
        .iter()
        .flat_map(|left| {
            site.orbitals.iter().map(move |right| {
                let left = left.channel().kappa();
                let right = right.channel().kappa();
                (left.large_l() + right.large_l()).max(left.small_l() + right.small_l())
            })
        })
        .max();
    l_max.map_or_else(BTreeMap::new, |l_max| {
        (0..=l_max)
            .flat_map(|l| {
                (-(l as i32)..=l as i32).map(move |m| {
                    (
                        Lm::new(l, m).expect("loop bounds validate magnetic channel"),
                        vec![Complex64::new(0.0, 0.0); site.mesh.len()],
                    )
                })
            })
            .collect()
    })
}

fn merge_sphere_accumulator(
    target: &mut BTreeMap<Lm, Vec<Complex64>>,
    source: &BTreeMap<Lm, Vec<Complex64>>,
) {
    for (&channel, values) in source {
        let destination = target
            .get_mut(&channel)
            .expect("spinor density accumulators use one canonical channel layout");
        for (destination, &value) in destination.iter_mut().zip(values) {
            *destination += value;
        }
    }
}

fn finish_muffin_tins(
    sites: &[ScalarSiteBasis],
    mut accumulators: Vec<BTreeMap<Lm, Vec<Complex64>>>,
) -> Result<Vec<MuffinTinField>, DensityError> {
    sites
        .iter()
        .zip(&mut accumulators)
        .map(|(site, channels)| {
            enforce_sphere_reality(channels);
            let field = SphereField::new(
                HarmonicConvention::Complex,
                std::mem::take(channels)
                    .into_iter()
                    .map(|(lm, values)| ((lm.l, lm.m), values)),
            )?;
            MuffinTinField::new(site.mesh.clone(), field).map_err(Into::into)
        })
        .collect()
}

fn finish_spinor_muffin_tins(
    sites: &[FullSpinorDensitySiteBasis],
    mut accumulators: Vec<BTreeMap<Lm, Vec<Complex64>>>,
) -> Result<Vec<MuffinTinField>, DensityError> {
    sites
        .iter()
        .zip(&mut accumulators)
        .map(|(site, channels)| {
            enforce_sphere_reality(channels);
            let field = SphereField::new(
                HarmonicConvention::Complex,
                std::mem::take(channels)
                    .into_iter()
                    .map(|(lm, values)| ((lm.l, lm.m), values)),
            )?;
            MuffinTinField::new(site.mesh.clone(), field).map_err(Into::into)
        })
        .collect()
}

fn enforce_sphere_reality(channels: &mut BTreeMap<Lm, Vec<Complex64>>) {
    let positive = channels
        .keys()
        .copied()
        .filter(|channel| channel.m >= 0)
        .collect::<Vec<_>>();
    for channel in positive {
        if channel.m == 0 {
            if let Some(values) = channels.get_mut(&channel) {
                for value in values {
                    value.im = 0.0;
                }
            }
            continue;
        }
        let partner = Lm::new(channel.l, -channel.m).expect("magnetic partner is valid");
        let Some(left) = channels.get(&channel).cloned() else {
            continue;
        };
        let Some(right) = channels.get(&partner).cloned() else {
            continue;
        };
        let phase = if channel.m % 2 == 0 { 1.0 } else { -1.0 };
        let averaged = left
            .into_iter()
            .zip(right)
            .map(|(left, right)| 0.5 * (left + phase * right.conj()))
            .collect::<Vec<_>>();
        channels.insert(channel, averaged.clone());
        channels.insert(
            partner,
            averaged
                .into_iter()
                .map(|value| phase * value.conj())
                .collect(),
        );
    }
}

fn enforce_fourier_reality(layout: &FourierLayout, coefficients: &mut [Complex64]) {
    for vector in layout.vectors() {
        let position = layout
            .index(vector.index)
            .expect("layout contains its vector");
        let opposite_index = vector.index.map(|value| -value);
        let opposite = layout
            .index(opposite_index)
            .expect("Hermitian layout contains every opposite vector");
        if position == opposite {
            coefficients[position].im = 0.0;
        } else if position < opposite {
            let average = 0.5 * (coefficients[position] + coefficients[opposite].conj());
            coefficients[position] = average;
            coefficients[opposite] = average.conj();
        }
    }
}

/// Invalid occupied-state density synthesis.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum DensityError {
    #[error("the regular k mesh is empty")]
    EmptyKMesh,
    #[error("k-point {k_index} has invalid weight {weight}")]
    InvalidKWeight { k_index: usize, weight: f64 },
    #[error("regular k-point weights sum to {0}, expected one")]
    KWeightSum(f64),
    #[error("k-point {k_index} has {actual} sites, expected {expected}")]
    SiteCount {
        k_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("k-point {k_index} eigenvectors have {actual} rows, expected {expected}")]
    EigenvectorBasis {
        k_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("k-point {k_index} has {actual} occupations, expected {expected}")]
    OccupationCount {
        k_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("k-point {k_index} band {band} has invalid occupation {occupation}")]
    InvalidOccupation {
        k_index: usize,
        band: usize,
        occupation: f64,
    },
    #[error("k-point {k_index} site {site} has {actual} radial orbitals, expected {expected}")]
    SiteOrbitalCount {
        k_index: usize,
        site: usize,
        expected: usize,
        actual: usize,
    },
    #[error("site {site} radial orbital samples do not match its mesh")]
    SiteOrbitalMesh { site: usize },
    #[error("spinor site {site} is absent from the compiled local-orbital layout")]
    SpinorSiteLayout { site: usize },
    #[error("spinor site {site} is absent from the compiled geometry")]
    SpinorSiteGeometry { site: usize },
    #[error("spinor site {site} geometry differs from the regional partition")]
    SpinorRegionalGeometry { site: usize },
    #[error("k-point {k_index} plane waves and density layout use different reciprocal lattices")]
    SpinorReciprocalMismatch { k_index: usize },
    #[error(
        "k-point {k_index} site {site} coordinate {coordinate} has channel {actual:?}, expected {expected:?}"
    )]
    SpinorCoordinateChannel {
        k_index: usize,
        site: usize,
        coordinate: usize,
        expected: RelativisticChannel,
        actual: RelativisticChannel,
    },
    #[error(
        "k-point {k_index} spinor site {site} mesh radius {actual} does not match compiled radius {expected}"
    )]
    SpinorSiteMeshRadius {
        k_index: usize,
        site: usize,
        expected: f64,
        actual: f64,
    },
    #[error("a reciprocal-vector difference overflowed i32")]
    ReciprocalDifferenceOverflow,
    #[error("density Fourier layout is missing reciprocal difference G={g:?}")]
    MissingReciprocalDifference { g: [i32; 3] },
    #[error("integrated electron count is complex: {real} + i {imaginary}")]
    ComplexElectronCount { real: f64, imaginary: f64 },
    #[error("target electron count must be finite and nonnegative, got {0}")]
    InvalidTargetElectronCount(f64),
    #[error("charge correction tolerance must be finite and nonnegative, got {0}")]
    InvalidChargeTolerance(f64),
    #[error("density integrates to {actual} electrons, target is {target}, tolerance {tolerance}")]
    ChargeMismatch {
        target: f64,
        actual: f64,
        tolerance: f64,
    },
    #[error(
        "full-spinor density integrates to {actual} electrons, expected {expected} within {tolerance}"
    )]
    SpinorChargeMismatch {
        expected: f64,
        actual: f64,
        tolerance: f64,
    },
    #[error("density Fourier layout has no G=0 coefficient")]
    MissingZeroVector,
    #[error("cell has no finite positive interstitial volume")]
    EmptyInterstitial,
    #[error("core occupation {occupation} is outside [0, {capacity}]")]
    InvalidCoreOccupation { occupation: f64, capacity: f64 },
    #[error("core spinor has P/Q lengths {p}/{q}, expected at least {expected_at_least}")]
    CoreMeshLength {
        expected_at_least: usize,
        p: usize,
        q: usize,
    },
    #[error("band-projection overlap axis is {actual}, expected SiteCoordinate")]
    BandProjectionOverlapAxis { actual: Axis },
    #[error("band-projection overlap has dimension {actual}, expected {expected} site coordinates")]
    BandProjectionOverlapDimension { expected: usize, actual: usize },
    #[error("selected band-projection coordinate {coordinate} is outside 0..{coordinate_count}")]
    BandProjectionCoordinate {
        coordinate: usize,
        coordinate_count: usize,
    },
    #[error("selected band-projection coordinate {coordinate} appears more than once")]
    DuplicateBandProjectionCoordinate { coordinate: usize },
    #[error("band {band} has non-finite physical site projection {projection}")]
    NonFiniteBandProjection { band: usize, projection: f64 },
    #[error(
        "band {band} has negative physical site projection {projection}, below allowed -{tolerance}"
    )]
    NegativeBandProjection {
        band: usize,
        projection: f64,
        tolerance: f64,
    },
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Projection(#[from] DensityProjectionError),
    #[error(transparent)]
    Sphere(#[from] SphereFieldError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    StepFunction(#[from] muffintin_core::StepFunctionError),
    #[error(transparent)]
    Regional(#[from] RegionalError),
    #[error(transparent)]
    Fourier(#[from] muffintin_core::FourierFieldError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::{
        Bohr, Hartree, HermitianFourierField, InverseBohr, Kappa, ReciprocalLattice, Sphere,
        VolumeBohr3,
    };
    use muffintin_envelope::PlaneWave;
    use muffintin_envelope::{
        ApwSiteGeometry, BasisLayout, LocalOrbitalLayout, Provenance, SpinorBasisLayout,
        SpinorCompiledBasis, SpinorSiteLayout,
    };
    use muffintin_sphere::{CoreState, RelativisticRole};
    use muffintin_tensor::{Axis, DenseEigenvectors, DenseHermitianMatrix};

    #[test]
    fn physical_site_band_projection_includes_nonorthogonal_cross_terms() {
        let compiled = CompiledBasis {
            layout: BasisLayout::new(0, vec![LocalOrbitalLayout::new(vec![2])]),
            plane_waves: Vec::new(),
            site_augmentations: vec![Vec::new()],
            site_geometry: vec![ApwSiteGeometry {
                position: [Bohr(0.0); 3],
                radius: Bohr(1.0),
            }],
            provenance: Provenance::default(),
        };
        let projection = CompiledSiteProjection::scalar(&compiled, 0).unwrap();
        let eigenvectors = DenseEigenvectors::from_host_column_major(
            2,
            1,
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
        )
        .unwrap();
        let overlap = DenseHermitianMatrix::from_host_row_major(
            2,
            Axis::SiteCoordinate,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .unwrap();

        let weights =
            physical_site_band_projections(&projection, &eigenvectors, &overlap, &[0, 1]).unwrap();

        assert_eq!(weights, vec![3.0]);
    }

    fn interstitial_only_density(electrons: f64) -> RegionalDensity {
        let reciprocal = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        let volume = (2.0 * PI).powi(3);
        let geometry = InterstitialGeometry::new(VolumeBohr3(volume), Vec::new()).unwrap();
        let vectors = reciprocal.enumerate(InverseBohr(0.0)).unwrap();
        let layout = FourierLayout::new(reciprocal, vectors).unwrap();
        let charge_interstitial = InterstitialField::from_fourier_field(
            HermitianFourierField::new(
                layout.clone(),
                vec![Complex64::new(electrons / volume, 0.0)],
            )
            .unwrap(),
        );
        let charge = RegionalScalarField::new(geometry, Vec::new(), charge_interstitial).unwrap();
        let zero = charge.zero_like();
        RegionalDensity::new(charge, [zero.clone(), zero.clone(), zero]).unwrap()
    }

    #[test]
    fn regional_charge_is_integrated_and_only_roundoff_is_corrected() {
        let density = interstitial_only_density(4.0 - 1.0e-12);
        assert!((electron_count(&density).unwrap() - (4.0 - 1.0e-12)).abs() < 1.0e-14);
        let corrected = correct_electron_count(density, 4.0, 2.0e-12).unwrap();
        assert!((electron_count(&corrected).unwrap() - 4.0).abs() < 1.0e-14);
        assert!(
            (corrected
                .charge()
                .interstitial()
                .coefficient([0, 0, 0])
                .unwrap()
                .re
                - 4.0 / corrected.geometry().cell_volume().get())
            .abs()
                < 1.0e-15
        );
        assert!(
            corrected
                .magnetization()
                .iter()
                .all(|field| field.residual_rms().unwrap() == 0.0)
        );

        let density = interstitial_only_density(3.0);
        assert!(matches!(
            correct_electron_count(density, 4.0, 1.0e-6),
            Err(DensityError::ChargeMismatch { .. })
        ));
    }

    #[test]
    fn plane_wave_density_uses_right_minus_left_and_inverse_cell_volume() {
        let reciprocal = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        let g_vectors = reciprocal.enumerate(InverseBohr(1.0)).unwrap();
        let zero = *g_vectors
            .iter()
            .find(|vector| vector.index == [0, 0, 0])
            .unwrap();
        let plus_x = *g_vectors
            .iter()
            .find(|vector| vector.index == [1, 0, 0])
            .unwrap();
        let compiled = CompiledBasis {
            layout: BasisLayout::new(2, Vec::new()),
            plane_waves: vec![
                PlaneWave::new([InverseBohr(0.0); 3], zero),
                PlaneWave::new([InverseBohr(0.0); 3], plus_x),
            ],
            site_augmentations: Vec::new(),
            site_geometry: Vec::new(),
            provenance: Provenance::default(),
        };
        let amplitude = 0.5_f64.sqrt();
        let eigenvectors = DenseEigenvectors::from_host_column_major(
            2,
            1,
            vec![
                Complex64::new(amplitude, 0.0),
                Complex64::new(0.0, amplitude),
            ],
        )
        .unwrap();
        let solution = GeneralizedEigensolution {
            eigenvalues: vec![Hartree(0.0)],
            eigenvectors,
            retained_dimension: 1,
            filtered_dimension: 1,
            residuals: Vec::new(),
        };
        let k_point = CollinearKPoint {
            weight: 1.0,
            compiled: &compiled,
            solutions: Collinear::new(&solution, &solution),
            occupations: Collinear::new(&[1.0], &[0.0]),
        };
        let volume = (2.0 * PI).powi(3);
        let density = synthesize_collinear_valence_density(
            InterstitialGeometry::new(VolumeBohr3(volume), Vec::new()).unwrap(),
            FourierLayout::new(reciprocal, g_vectors).unwrap(),
            &[],
            &[k_point],
        )
        .unwrap();
        let charge = density.charge().interstitial();
        assert!((charge.coefficient([0, 0, 0]).unwrap().re - 1.0 / volume).abs() < 1.0e-15);
        assert!(
            (charge.coefficient([1, 0, 0]).unwrap() - Complex64::new(0.0, 0.5 / volume)).norm()
                < 1.0e-15
        );
        assert!(
            (charge.coefficient([-1, 0, 0]).unwrap() - Complex64::new(0.0, -0.5 / volume)).norm()
                < 1.0e-15
        );
        assert_eq!(
            density.magnetization()[2].interstitial(),
            density.charge().interstitial()
        );
        assert!((electron_count(&density).unwrap() - 1.0).abs() < 1.0e-14);
    }

    #[test]
    fn collinear_valence_density_keeps_zero_orbital_pair_channels() {
        let reciprocal = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        let mesh = ExponentialMesh::new(Bohr(1.0e-3), 0.18, 12).unwrap();
        let compiled = CompiledBasis {
            layout: BasisLayout::new(0, vec![LocalOrbitalLayout::new(vec![1, 1])]),
            plane_waves: Vec::new(),
            site_augmentations: vec![Vec::new()],
            site_geometry: vec![ApwSiteGeometry {
                position: [Bohr(0.0); 3],
                radius: mesh.last(),
            }],
            provenance: Provenance::default(),
        };
        let solution = GeneralizedEigensolution {
            eigenvalues: vec![Hartree(0.0)],
            eigenvectors: DenseEigenvectors::from_host_column_major(
                4,
                1,
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .unwrap(),
            retained_dimension: 1,
            filtered_dimension: 3,
            residuals: Vec::new(),
        };
        let sites = [ScalarSiteBasis {
            mesh: mesh.clone(),
            orbitals: vec![
                SphereOrbital::new(0, 0, vec![1.0; mesh.len()], None).unwrap(),
                SphereOrbital::new(1, -1, vec![1.0; mesh.len()], None).unwrap(),
                SphereOrbital::new(1, 0, vec![1.0; mesh.len()], None).unwrap(),
                SphereOrbital::new(1, 1, vec![1.0; mesh.len()], None).unwrap(),
            ],
        }];
        let density = synthesize_collinear_valence_density(
            InterstitialGeometry::new(
                VolumeBohr3((2.0 * PI).powi(3)),
                vec![Sphere {
                    center: [Bohr(0.0); 3],
                    radius: mesh.last(),
                }],
            )
            .unwrap(),
            FourierLayout::new(reciprocal, reciprocal.enumerate(InverseBohr(0.0)).unwrap())
                .unwrap(),
            &sites,
            &[CollinearKPoint {
                weight: 1.0,
                compiled: &compiled,
                solutions: Collinear::new(&solution, &solution),
                occupations: Collinear::new(&[1.0], &[0.0]),
            }],
        )
        .unwrap();
        let expected = (0..=2)
            .flat_map(|l| {
                (-(l as i32)..=l as i32).map(move |m| Lm::new(l, m).expect("test channel is valid"))
            })
            .collect::<Vec<_>>();

        for component in std::iter::once(density.charge()).chain(density.magnetization()) {
            let field = component.muffin_tins()[0].field();
            assert_eq!(
                field
                    .channels()
                    .map(|(channel, _)| channel)
                    .collect::<Vec<_>>(),
                expected
            );
            for m in -2..=2 {
                assert!(
                    field
                        .channel(2, m)
                        .unwrap()
                        .iter()
                        .all(|&value| value == Complex64::new(0.0, 0.0))
                );
            }
        }
    }

    #[test]
    fn batched_collinear_muffin_tin_density_matches_pair_oracle() {
        let reciprocal = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        let mesh = ExponentialMesh::new(Bohr(1.0e-3), 0.18, 12).unwrap();
        let compiled = CompiledBasis {
            layout: BasisLayout::new(0, vec![LocalOrbitalLayout::new(vec![1, 1])]),
            plane_waves: Vec::new(),
            site_augmentations: vec![Vec::new()],
            site_geometry: vec![ApwSiteGeometry {
                position: [Bohr(0.0); 3],
                radius: mesh.last(),
            }],
            provenance: Provenance::default(),
        };
        let coefficients = [
            Complex64::new(0.7, 0.1),
            Complex64::new(-0.2, 0.3),
            Complex64::new(0.15, -0.25),
            Complex64::new(0.4, 0.05),
        ];
        let solution = GeneralizedEigensolution {
            eigenvalues: vec![Hartree(0.0)],
            eigenvectors: DenseEigenvectors::from_host_column_major(4, 1, coefficients.to_vec())
                .unwrap(),
            retained_dimension: 1,
            filtered_dimension: 3,
            residuals: Vec::new(),
        };
        let orbitals = vec![
            SphereOrbital::new(0, 0, vec![0.8; mesh.len()], None).unwrap(),
            SphereOrbital::new(1, -1, vec![0.9; mesh.len()], None).unwrap(),
            SphereOrbital::new(1, 0, vec![1.0; mesh.len()], None).unwrap(),
            SphereOrbital::new(1, 1, vec![1.1; mesh.len()], None).unwrap(),
        ];
        let sites = [ScalarSiteBasis {
            mesh: mesh.clone(),
            orbitals: orbitals.clone(),
        }];
        let occupation = 0.65;
        let density = synthesize_collinear_valence_density(
            InterstitialGeometry::new(
                VolumeBohr3((2.0 * PI).powi(3)),
                vec![Sphere {
                    center: [Bohr(0.0); 3],
                    radius: mesh.last(),
                }],
            )
            .unwrap(),
            FourierLayout::new(reciprocal, reciprocal.enumerate(InverseBohr(0.0)).unwrap())
                .unwrap(),
            &sites,
            &[CollinearKPoint {
                weight: 1.0,
                compiled: &compiled,
                solutions: Collinear::new(&solution, &solution),
                occupations: Collinear::new(&[occupation], &[0.0]),
            }],
        )
        .unwrap();

        let projected = CompiledSiteProjection::scalar(&compiled, 0)
            .unwrap()
            .project_eigenvectors(&solution.eigenvectors)
            .unwrap();
        let mut expected = site_accumulators(&sites).remove(0);
        for (left, left_orbital) in orbitals.iter().enumerate() {
            for (right, right_orbital) in orbitals.iter().enumerate() {
                let scale = occupation * projected.at(left, 0).conj() * projected.at(right, 0);
                let pair = muffintin_sphere::project_orbital_pair_density(
                    &mesh,
                    left_orbital,
                    right_orbital,
                )
                .unwrap();
                for (channel, values) in pair.channels() {
                    for (target, &value) in
                        expected.get_mut(&channel).unwrap().iter_mut().zip(values)
                    {
                        *target += scale * value;
                    }
                }
            }
        }
        enforce_sphere_reality(&mut expected);
        let actual = density.charge().muffin_tins()[0].field();
        for (channel, expected) in expected {
            for (&actual, expected) in actual
                .channel(channel.l, channel.m)
                .unwrap()
                .iter()
                .zip(expected)
            {
                assert!((actual - expected).norm() < 2.0e-13 * (1.0 + expected.norm()));
            }
        }
    }

    #[test]
    fn full_spinor_interstitial_uses_two_component_pauli_density() {
        let reciprocal = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        let vectors = reciprocal.enumerate(InverseBohr(0.0)).unwrap();
        let zero = vectors[0];
        let compiled = SpinorCompiledBasis {
            layout: SpinorBasisLayout::new(1, Vec::new()),
            plane_waves: vec![PlaneWave::new([InverseBohr(0.0); 3], zero)],
            site_augmentations: Vec::new(),
            site_geometry: Vec::new(),
            provenance: Provenance::default(),
        };
        let amplitude = 0.5_f64.sqrt();
        let solution = GeneralizedEigensolution {
            eigenvalues: vec![Hartree(0.0)],
            eigenvectors: DenseEigenvectors::from_host_column_major(
                2,
                1,
                vec![
                    Complex64::new(amplitude, 0.0),
                    Complex64::new(0.0, amplitude),
                ],
            )
            .unwrap(),
            retained_dimension: 1,
            filtered_dimension: 1,
            residuals: Vec::new(),
        };
        let volume = (2.0 * PI).powi(3);
        let density = synthesize_full_spinor_valence_density(
            InterstitialGeometry::new(VolumeBohr3(volume), Vec::new()).unwrap(),
            FourierLayout::new(reciprocal, vectors).unwrap(),
            &[],
            &[FullSpinorKPoint {
                weight: 1.0,
                compiled: &compiled,
                solution: &solution,
                occupations: &[1.0],
            }],
        )
        .unwrap();

        assert!((electron_count(&density).unwrap() - 1.0).abs() < 1.0e-14);
        assert!(
            (density
                .charge()
                .interstitial()
                .coefficient([0; 3])
                .unwrap()
                .re
                - 1.0 / volume)
                .abs()
                < 1.0e-15
        );
        assert!(
            density.magnetization()[0]
                .interstitial()
                .coefficient([0; 3])
                .unwrap()
                .norm()
                < 1.0e-15
        );
        assert!(
            (density.magnetization()[1]
                .interstitial()
                .coefficient([0; 3])
                .unwrap()
                .re
                - 1.0 / volume)
                .abs()
                < 1.0e-15
        );
        assert!(
            density.magnetization()[2]
                .interstitial()
                .coefficient([0; 3])
                .unwrap()
                .norm()
                < 1.0e-15
        );
        assert!(density.magnetization()[0].residual_rms().unwrap() < 1.0e-15);
        assert!(density.magnetization()[1].residual_rms().unwrap() > 0.0);
    }

    #[test]
    fn full_spinor_finite_g_components_are_exactly_hermitian() {
        let reciprocal = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        let vectors = reciprocal.enumerate(InverseBohr(1.0)).unwrap();
        let zero = *vectors
            .iter()
            .find(|vector| vector.index == [0; 3])
            .unwrap();
        let plus_x = *vectors
            .iter()
            .find(|vector| vector.index == [1, 0, 0])
            .unwrap();
        let compiled = SpinorCompiledBasis {
            layout: SpinorBasisLayout::new(2, Vec::new()),
            plane_waves: vec![
                PlaneWave::new([InverseBohr(0.0); 3], zero),
                PlaneWave::new([InverseBohr(0.0); 3], plus_x),
            ],
            site_augmentations: Vec::new(),
            site_geometry: Vec::new(),
            provenance: Provenance::default(),
        };
        let solution = GeneralizedEigensolution {
            eigenvalues: vec![Hartree(0.0)],
            eigenvectors: DenseEigenvectors::from_host_column_major(
                4,
                1,
                vec![
                    Complex64::new(0.5, 0.0),
                    Complex64::new(0.0, 0.5),
                    Complex64::new(0.5, 0.0),
                    Complex64::new(-0.5, 0.0),
                ],
            )
            .unwrap(),
            retained_dimension: 1,
            filtered_dimension: 3,
            residuals: Vec::new(),
        };
        let volume = (2.0 * PI).powi(3);
        let density = synthesize_full_spinor_valence_density(
            InterstitialGeometry::new(VolumeBohr3(volume), Vec::new()).unwrap(),
            FourierLayout::new(reciprocal, vectors).unwrap(),
            &[],
            &[FullSpinorKPoint {
                weight: 1.0,
                compiled: &compiled,
                solution: &solution,
                occupations: &[1.0],
            }],
        )
        .unwrap();
        for field in std::iter::once(density.charge()).chain(density.magnetization()) {
            let plus = field.interstitial().coefficient([1, 0, 0]).unwrap();
            let minus = field.interstitial().coefficient([-1, 0, 0]).unwrap();
            assert_eq!(minus, plus.conj());
        }
        assert!((electron_count(&density).unwrap() - 1.0).abs() < 1.0e-14);
    }

    #[test]
    fn full_spinor_local_orbital_density_retains_pure_small_component() {
        let reciprocal = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        let volume = (2.0 * PI).powi(3);
        let mesh = ExponentialMesh::new(Bohr(1.0e-3), 0.18, 32).unwrap();
        let kappa = Kappa::new(-1).unwrap();
        let channel = kappa.channels().next().unwrap();
        let mut q = mesh
            .radii()
            .iter()
            .map(|radius| radius.get() * (-2.0 * radius.get()).exp())
            .collect::<Vec<_>>();
        let norm = mesh
            .integrate(&q.iter().map(|value| value * value).collect::<Vec<_>>())
            .unwrap()
            .sqrt();
        for value in &mut q {
            *value /= norm;
        }
        let site_layout = SpinorSiteLayout::new(vec![(kappa, 1)]).unwrap();
        let compiled = SpinorCompiledBasis {
            layout: SpinorBasisLayout::new(0, vec![site_layout]),
            plane_waves: Vec::new(),
            site_augmentations: vec![Vec::new()],
            site_geometry: vec![ApwSiteGeometry {
                position: [Bohr(0.0); 3],
                radius: mesh.last(),
            }],
            provenance: Provenance::default(),
        };
        let solution = GeneralizedEigensolution {
            eigenvalues: vec![Hartree(0.0)],
            eigenvectors: DenseEigenvectors::from_host_column_major(
                2,
                1,
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            )
            .unwrap(),
            retained_dimension: 1,
            filtered_dimension: 1,
            residuals: Vec::new(),
        };
        let sites = [FullSpinorDensitySiteBasis::new(
            mesh.clone(),
            Vec::new(),
            vec![
                SpinorSphereOrbital::new(channel, vec![0.0; mesh.len()], q).unwrap(),
                SpinorSphereOrbital::new(
                    kappa.channels().nth(1).unwrap(),
                    vec![0.0; mesh.len()],
                    vec![0.0; mesh.len()],
                )
                .unwrap(),
            ],
        )];
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(volume),
            vec![Sphere {
                center: [Bohr(0.0); 3],
                radius: mesh.last(),
            }],
        )
        .unwrap();
        let layout =
            FourierLayout::new(reciprocal, reciprocal.enumerate(InverseBohr(0.0)).unwrap())
                .unwrap();
        let density = synthesize_full_spinor_valence_density(
            geometry,
            layout,
            &sites,
            &[FullSpinorKPoint {
                weight: 1.0,
                compiled: &compiled,
                solution: &solution,
                occupations: &[1.0],
            }],
        )
        .unwrap();

        assert!((electron_count(&density).unwrap() - 1.0).abs() < 2.0e-13);
        assert!(
            density.charge().muffin_tins()[0]
                .field()
                .channel(0, 0)
                .unwrap()
                .iter()
                .any(|value| value.re > 0.0)
        );
        density.charge().muffin_tins()[0]
            .field()
            .validate_physical_reality(2.0e-13)
            .unwrap();
        for component in density.magnetization() {
            component.muffin_tins()[0]
                .field()
                .validate_physical_reality(2.0e-13)
                .unwrap();
        }
    }

    #[test]
    fn core_shell_monopole_counts_the_small_component_and_shell_occupation() {
        let mesh = ExponentialMesh::new(Bohr(1.0e-4), 0.2, 12).unwrap();
        let raw_q = vec![1.0; mesh.len()];
        let norm = mesh
            .integrate(&raw_q.iter().map(|value| value * value).collect::<Vec<_>>())
            .unwrap()
            .sqrt();
        let q = raw_q
            .into_iter()
            .map(|value| value / norm)
            .collect::<Vec<_>>();
        let kappa = Kappa::new(-1).unwrap();
        let solution = CoreDiracSolution {
            role: RelativisticRole::Core,
            state: CoreState::new(1, kappa).unwrap(),
            angular: kappa.angular_contract(),
            energy: Hartree(-0.5),
            p: vec![0.0; mesh.len()],
            q,
            norm_total: 1.0,
            norm_mt: 1.0,
            norm_outside: 0.0,
            spill: 0.0,
            nodes: 0,
            match_radius: mesh.last(),
            matching_residual: 0.0,
        };
        let occupation = 1.5;
        let density = core_shell_density(&mesh, &solution, occupation).unwrap();
        let monopole = density.channel(0, 0).unwrap();
        let integrated = (4.0 * PI).sqrt()
            * mesh
                .integrate(
                    &monopole
                        .iter()
                        .zip(mesh.radii())
                        .map(|(value, radius)| value.re * radius.get().powi(2))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        assert!((integrated - occupation).abs() < 2.0e-14);
        assert!(monopole.iter().all(|value| value.re > 0.0));
    }
}
