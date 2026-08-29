//! Periodic neutral-free-atom superposition on an exact regional layout.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::f64::consts::PI;

use muffintin_core::{
    Bohr, ExponentialMesh, FourierFieldError, FourierLayout, HermitianFourierField,
    InterstitialGeometry, LatticeError, MeshError, ReciprocalLattice, Sphere, StepFunctionError,
    lm_count, lm_from_index, real_spherical_harmonics, spherical_bessel_j,
};
use muffintin_grid::{AngularGrid, Cell};
use muffintin_sphere::{SphereField, SphereFieldError};
use num_complex::Complex64;
use thiserror::Error;

use crate::atomic_configuration::AtomicNumber;
use crate::core_density::{
    CoreDensityError, FiniteLayoutClosureComponent, close_finite_layout_zero_mode,
};
use crate::density::{DensityError, scalar_field_integral};
use crate::free_atom::{FreeAtomScfError, FreeAtomScfSpec, FreeAtomState, run_free_atom_lda};
use crate::regional::{
    InterstitialField, MuffinTinField, RegionalDensity, RegionalError, RegionalScalarField,
};

const CHARGE_CLOSURE_TOLERANCE: f64 = 65536.0 * f64::EPSILON;

/// One neutral atom and the muffin-tin representation centred on it.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomicSuperpositionSite {
    pub atomic_number: AtomicNumber,
    pub position: [Bohr; 3],
    pub muffin_tin_mesh: ExponentialMesh,
}

/// Exact inputs for one periodic neutral-atom superposition.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomicSuperpositionSpec {
    pub direct_lattice: Cell,
    pub sites: Vec<AtomicSuperpositionSite>,
    pub fourier_layout: FourierLayout,
    pub muffin_tin_l_max: u32,
    pub angular_grid: AngularGrid,
    pub target_electron_count: f64,
    pub free_atom_scf: FreeAtomScfSpec,
}

/// Constant-mode charge correction on the caller's finite Fourier layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomicSuperpositionChargeClosure {
    pub interstitial_fraction: f64,
    pub response_volume: f64,
    pub target_electron_count: f64,
    pub uncorrected_electron_count: f64,
    pub zero_mode_coefficient_correction: f64,
    pub represented_electron_count: f64,
}

/// Nonmagnetic periodic density and its finite-layout charge accounting.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomicSuperpositionDensity {
    pub density: RegionalDensity,
    pub charge_closure: AtomicSuperpositionChargeClosure,
}

/// Invalid neutral-atom superposition input or construction.
#[derive(Debug, Error)]
pub enum AtomicSuperpositionError {
    #[error("atomic superposition requires at least one site")]
    EmptySites,
    #[error("neutral atomic superposition requires target electron count {neutral}, got {target}")]
    NonNeutralTarget { target: f64, neutral: f64 },
    #[error("the Fourier layout reciprocal lattice does not match the supplied direct lattice")]
    ReciprocalLayoutMismatch,
    #[error("free-atom radial mesh must run outwards")]
    NonOutwardFreeAtomMesh,
    #[error("site {site} muffin-tin radial mesh must run outwards")]
    NonOutwardMuffinTinMesh { site: usize },
    #[error(
        "site {site} muffin-tin mesh starts at {actual} bohr, inside the verified free-atom mesh start {minimum} bohr"
    )]
    MuffinTinInsideFreeAtomMesh {
        site: usize,
        actual: f64,
        minimum: f64,
    },
    #[error("neutral free-atom solve failed for {atomic_number:?}")]
    FreeAtom {
        atomic_number: AtomicNumber,
        #[source]
        source: FreeAtomScfError,
    },
    #[error(
        "periodic free-atom interpolation for source site {site} requested radius {radius} below mesh start {minimum} bohr"
    )]
    InterpolationBelowMesh {
        site: usize,
        radius: f64,
        minimum: f64,
    },
    #[error(
        "periodic free-atom interpolation for source site {site} requested radius {radius} beyond verified tail {maximum} bohr"
    )]
    InterpolationBeyondVerifiedTail {
        site: usize,
        radius: f64,
        maximum: f64,
    },
    #[error("Fourier layout has no opposite vector for {0:?}")]
    MissingOpposite([i32; 3]),
    #[error(
        "closed atomic superposition represents {represented} electrons, requested {requested} within {tolerance}"
    )]
    ChargeClosure {
        requested: f64,
        represented: f64,
        tolerance: f64,
    },
    #[error(transparent)]
    Lattice(#[from] LatticeError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    Sphere(#[from] SphereFieldError),
    #[error(transparent)]
    Regional(#[from] RegionalError),
    #[error(transparent)]
    Fourier(#[from] FourierFieldError),
    #[error(transparent)]
    Density(#[from] DensityError),
    #[error(transparent)]
    StepFunction(#[from] StepFunctionError),
    #[error(transparent)]
    FiniteLayoutClosure(#[from] CoreDensityError),
}

/// Build a periodic, nonmagnetic superposition of converged neutral atoms.
pub fn build_atomic_superposition_density(
    spec: &AtomicSuperpositionSpec,
) -> Result<AtomicSuperpositionDensity, AtomicSuperpositionError> {
    validate_spec(spec)?;
    let geometry = InterstitialGeometry::new(
        spec.direct_lattice.volume(),
        spec.sites
            .iter()
            .map(|site| Sphere {
                center: site.position,
                radius: site.muffin_tin_mesh.last(),
            })
            .collect::<Vec<_>>(),
    )?;

    let mut atoms = BTreeMap::new();
    for site in &spec.sites {
        if let Entry::Vacant(entry) = atoms.entry(site.atomic_number) {
            let state =
                run_free_atom_lda(site.atomic_number, &spec.free_atom_scf).map_err(|source| {
                    AtomicSuperpositionError::FreeAtom {
                        atomic_number: site.atomic_number,
                        source,
                    }
                })?;
            entry.insert(state);
        }
    }
    for (site_index, site) in spec.sites.iter().enumerate() {
        let atom = &atoms[&site.atomic_number];
        if site.muffin_tin_mesh.first().get() < atom.mesh.first().get() {
            return Err(AtomicSuperpositionError::MuffinTinInsideFreeAtomMesh {
                site: site_index,
                actual: site.muffin_tin_mesh.first().get(),
                minimum: atom.mesh.first().get(),
            });
        }
    }

    let muffin_tins = build_muffin_tins(spec, &atoms)?;
    let mut fourier = build_fourier_coefficients(spec, &atoms)?;
    enforce_fourier_reality(&spec.fourier_layout, &mut fourier)?;

    let zero_muffin_tins = muffin_tins
        .iter()
        .map(MuffinTinField::zero_like)
        .collect::<Vec<_>>();
    let mut zero_fourier = vec![Complex64::new(0.0, 0.0); spec.fourier_layout.len()];
    let closure = close_finite_layout_zero_mode(
        &geometry,
        &spec.fourier_layout,
        FiniteLayoutClosureComponent {
            muffin_tins: &muffin_tins,
            requested_integral: spec.target_electron_count,
            fourier: &mut fourier,
        },
        FiniteLayoutClosureComponent {
            muffin_tins: &zero_muffin_tins,
            requested_integral: 0.0,
            fourier: &mut zero_fourier,
        },
    )?;
    let charge = RegionalScalarField::new(
        geometry,
        muffin_tins,
        InterstitialField::from_fourier_field(HermitianFourierField::new(
            spec.fourier_layout.clone(),
            fourier,
        )?),
    )?;
    let zero = charge.zero_like();
    let density = RegionalDensity::new(charge, [zero.clone(), zero.clone(), zero])?;
    let represented_electron_count = scalar_field_integral(density.charge())?;
    let tolerance = CHARGE_CLOSURE_TOLERANCE * spec.target_electron_count.abs().max(1.0);
    if (represented_electron_count - spec.target_electron_count).abs() > tolerance {
        return Err(AtomicSuperpositionError::ChargeClosure {
            requested: spec.target_electron_count,
            represented: represented_electron_count,
            tolerance,
        });
    }

    Ok(AtomicSuperpositionDensity {
        density,
        charge_closure: AtomicSuperpositionChargeClosure {
            interstitial_fraction: closure.interstitial_fraction,
            response_volume: closure.response_volume,
            target_electron_count: spec.target_electron_count,
            uncorrected_electron_count: closure.uncorrected_charge,
            zero_mode_coefficient_correction: closure.charge_coefficient_correction,
            represented_electron_count,
        },
    })
}

fn validate_spec(spec: &AtomicSuperpositionSpec) -> Result<(), AtomicSuperpositionError> {
    if spec.sites.is_empty() {
        return Err(AtomicSuperpositionError::EmptySites);
    }
    let neutral = spec
        .sites
        .iter()
        .map(|site| f64::from(site.atomic_number.get()))
        .sum::<f64>();
    if spec.target_electron_count != neutral {
        return Err(AtomicSuperpositionError::NonNeutralTarget {
            target: spec.target_electron_count,
            neutral,
        });
    }
    if spec.free_atom_scf.mesh.increment() <= 0.0 {
        return Err(AtomicSuperpositionError::NonOutwardFreeAtomMesh);
    }
    for (site, input) in spec.sites.iter().enumerate() {
        if input.muffin_tin_mesh.increment() <= 0.0 {
            return Err(AtomicSuperpositionError::NonOutwardMuffinTinMesh { site });
        }
    }
    let reciprocal = ReciprocalLattice::from_direct(*spec.direct_lattice.basis())?;
    if spec.fourier_layout.reciprocal() != &reciprocal {
        return Err(AtomicSuperpositionError::ReciprocalLayoutMismatch);
    }
    Ok(())
}

fn build_muffin_tins(
    spec: &AtomicSuperpositionSpec,
    atoms: &BTreeMap<AtomicNumber, FreeAtomState>,
) -> Result<Vec<MuffinTinField>, AtomicSuperpositionError> {
    let harmonics = spec
        .angular_grid
        .points()
        .iter()
        .map(|point| real_spherical_harmonics(spec.muffin_tin_l_max, point.direction))
        .collect::<Vec<_>>();
    let channel_count = lm_count(spec.muffin_tin_l_max);
    let mut result = Vec::with_capacity(spec.sites.len());

    for target in &spec.sites {
        let mut images = Vec::new();
        for (source_index, source) in spec.sites.iter().enumerate() {
            let atom = &atoms[&source.atomic_number];
            let center_displacement = subtract(target.position, source.position);
            let cutoff = atom.mesh.last().get() + target.muffin_tin_mesh.last().get();
            for translation in
                translations_within(&spec.direct_lattice, center_displacement, cutoff)
            {
                images.push((
                    source_index,
                    atom,
                    add(source.position.map(Bohr::get), translation),
                ));
            }
        }

        let mut channels = vec![vec![0.0; target.muffin_tin_mesh.len()]; channel_count];
        for (radial_index, radius) in target.muffin_tin_mesh.radii().iter().enumerate() {
            for (angular_index, angular) in spec.angular_grid.points().iter().enumerate() {
                let point = std::array::from_fn(|axis| {
                    target.position[axis].get() + radius.get() * angular.direction[axis]
                });
                let mut density = 0.0;
                for &(source_index, atom, image_center) in &images {
                    let distance = norm(subtract_raw(point, image_center));
                    if distance <= atom.mesh.last().get() {
                        density += interpolate_atom(atom, source_index, distance)?;
                    }
                }
                for (channel, &harmonic) in channels.iter_mut().zip(&harmonics[angular_index]) {
                    channel[radial_index] += angular.weight * density * harmonic;
                }
            }
        }
        let sphere = SphereField::from_real_channels(channels.into_iter().enumerate().map(
            |(index, values)| {
                let lm = lm_from_index(index);
                ((lm.l, lm.m), values)
            },
        ))?;
        result.push(MuffinTinField::new(target.muffin_tin_mesh.clone(), sphere)?);
    }
    Ok(result)
}

fn build_fourier_coefficients(
    spec: &AtomicSuperpositionSpec,
    atoms: &BTreeMap<AtomicNumber, FreeAtomState>,
) -> Result<Vec<Complex64>, AtomicSuperpositionError> {
    let volume = spec.direct_lattice.volume().get();
    spec.fourier_layout
        .vectors()
        .iter()
        .map(|vector| {
            let mut coefficient = Complex64::new(0.0, 0.0);
            let mut transforms = BTreeMap::new();
            for site in &spec.sites {
                let transform = if let Some(&transform) = transforms.get(&site.atomic_number) {
                    transform
                } else {
                    let atom = &atoms[&site.atomic_number];
                    let integrand = atom
                        .number_density
                        .iter()
                        .zip(atom.mesh.radii())
                        .map(|(&density, radius)| {
                            density
                                * radius.get().powi(2)
                                * spherical_bessel_j(0, vector.norm.get() * radius.get())
                        })
                        .collect::<Vec<_>>();
                    let transform = 4.0 * PI * atom.mesh.integrate(&integrand)? / volume;
                    transforms.insert(site.atomic_number, transform);
                    transform
                };
                let phase = -vector
                    .cartesian
                    .iter()
                    .zip(site.position)
                    .map(|(g, r)| g.get() * r.get())
                    .sum::<f64>();
                coefficient += Complex64::from_polar(transform, phase);
            }
            Ok(coefficient)
        })
        .collect()
}

fn interpolate_atom(
    atom: &FreeAtomState,
    site: usize,
    radius: f64,
) -> Result<f64, AtomicSuperpositionError> {
    let radii = atom.mesh.radii();
    if radius < radii[0].get() {
        return Err(AtomicSuperpositionError::InterpolationBelowMesh {
            site,
            radius,
            minimum: radii[0].get(),
        });
    }
    if radius > radii[radii.len() - 1].get() {
        return Err(AtomicSuperpositionError::InterpolationBeyondVerifiedTail {
            site,
            radius,
            maximum: radii[radii.len() - 1].get(),
        });
    }
    match radii.binary_search_by(|sample| sample.get().total_cmp(&radius)) {
        Ok(index) => Ok(atom.number_density[index]),
        Err(upper) => {
            let lower = upper - 1;
            let fraction = (radius / radii[lower].get()).ln() / atom.mesh.increment();
            Ok(atom.number_density[lower]
                + fraction * (atom.number_density[upper] - atom.number_density[lower]))
        }
    }
}

fn enforce_fourier_reality(
    layout: &FourierLayout,
    coefficients: &mut [Complex64],
) -> Result<(), AtomicSuperpositionError> {
    for vector in layout.vectors() {
        let position = layout
            .index(vector.index)
            .expect("layout contains its stored vector");
        let opposite_index = vector.index.map(|component| component.checked_neg());
        let [Some(g0), Some(g1), Some(g2)] = opposite_index else {
            return Err(AtomicSuperpositionError::MissingOpposite(vector.index));
        };
        let opposite = layout
            .index([g0, g1, g2])
            .ok_or(AtomicSuperpositionError::MissingOpposite(vector.index))?;
        if position == opposite {
            coefficients[position].im = 0.0;
        } else if position < opposite {
            let average = 0.5 * (coefficients[position] + coefficients[opposite].conj());
            coefficients[position] = average;
            coefficients[opposite] = average.conj();
        }
    }
    Ok(())
}

fn translations_within(cell: &Cell, displacement: [f64; 3], cutoff: f64) -> Vec<[f64; 3]> {
    let basis = cell.basis().map(|vector| vector.map(Bohr::get));
    let determinant = dot(basis[0], cross(basis[1], basis[2]));
    let inverse = [
        scale(cross(basis[1], basis[2]), 1.0 / determinant),
        scale(cross(basis[2], basis[0]), 1.0 / determinant),
        scale(cross(basis[0], basis[1]), 1.0 / determinant),
    ];
    let fractional = inverse.map(|row| dot(row, displacement));
    let mut lower = [0_i64; 3];
    let mut upper = [0_i64; 3];
    for axis in 0..3 {
        let extent = norm(inverse[axis]) * cutoff;
        lower[axis] = (fractional[axis] - extent).ceil() as i64;
        upper[axis] = (fractional[axis] + extent).floor() as i64;
    }
    let cutoff_squared = cutoff * cutoff;
    let tolerance = 64.0 * f64::EPSILON * cutoff_squared.max(1.0);
    let mut result = Vec::new();
    for n0 in lower[0]..=upper[0] {
        for n1 in lower[1]..=upper[1] {
            for n2 in lower[2]..=upper[2] {
                let translation = add(
                    add(scale(basis[0], n0 as f64), scale(basis[1], n1 as f64)),
                    scale(basis[2], n2 as f64),
                );
                if dot(
                    subtract_raw(displacement, translation),
                    subtract_raw(displacement, translation),
                ) <= cutoff_squared + tolerance
                {
                    result.push(translation);
                }
            }
        }
    }
    result
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn subtract(left: [Bohr; 3], right: [Bohr; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis].get() - right[axis].get())
}

fn subtract_raw(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    vector.map(|component| component * factor)
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::InverseBohr;
    use muffintin_grid::AngularPoint;

    #[test]
    fn two_site_superposition_closes_charge_and_retains_neighbor_anisotropy() {
        let side = 8.0;
        let direct = [
            [Bohr(side), Bohr(0.0), Bohr(0.0)],
            [Bohr(0.0), Bohr(side), Bohr(0.0)],
            [Bohr(0.0), Bohr(0.0), Bohr(side)],
        ];
        let cell = Cell::new(direct).unwrap();
        let reciprocal = ReciprocalLattice::from_direct(direct).unwrap();
        let layout =
            FourierLayout::new(reciprocal, reciprocal.enumerate(InverseBohr(1.0)).unwrap())
                .unwrap();
        let muffin_tin_mesh = ExponentialMesh::new(Bohr(1.0e-4), 0.1, 92).unwrap();
        let angular_grid = AngularGrid::new(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
            .map(|direction| AngularPoint {
                direction,
                weight: 2.0 * PI / 3.0,
            })
            .to_vec(),
        )
        .unwrap();
        let hydrogen = AtomicNumber::new(1).unwrap();
        let spec = AtomicSuperpositionSpec {
            direct_lattice: cell,
            sites: vec![
                AtomicSuperpositionSite {
                    atomic_number: hydrogen,
                    position: [Bohr(2.75), Bohr(4.0), Bohr(4.0)],
                    muffin_tin_mesh: muffin_tin_mesh.clone(),
                },
                AtomicSuperpositionSite {
                    atomic_number: hydrogen,
                    position: [Bohr(5.25), Bohr(4.0), Bohr(4.0)],
                    muffin_tin_mesh,
                },
            ],
            fourier_layout: layout,
            muffin_tin_l_max: 1,
            angular_grid,
            target_electron_count: 2.0,
            free_atom_scf: FreeAtomScfSpec {
                mesh: ExponentialMesh::new(Bohr(1.0e-6), 0.01, 1683).unwrap(),
                mixing: 0.3,
                potential_tolerance: 2.0e-5,
                tail_tolerance: 1.0e-7,
                max_iterations: 120,
            },
        };

        let built = build_atomic_superposition_density(&spec).unwrap();
        assert!((built.charge_closure.represented_electron_count - 2.0).abs() < 1.0e-11);
        assert!(
            built
                .density
                .charge()
                .interstitial()
                .coefficients()
                .any(|(g, coefficient)| g != [0, 0, 0] && coefficient.norm() > 1.0e-12)
        );
        let neighbor_channel = built.density.charge().muffin_tins()[0]
            .field()
            .channel(1, 1)
            .unwrap();
        assert!(
            neighbor_channel
                .iter()
                .any(|value| value.re.abs() > 1.0e-10)
        );
        let zero = built.density.charge().zero_like();
        for component in built.density.magnetization() {
            assert_eq!(component, &zero);
        }
    }
}
