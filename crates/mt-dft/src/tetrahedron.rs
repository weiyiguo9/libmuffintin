//! Linear-tetrahedron density of states on a regular periodic full-BZ mesh.

use muffintin_core::{Hartree, InverseBohr};
use thiserror::Error;

/// A band spectrum on a regular periodic full-Brillouin-zone mesh.
///
/// `energies` is band-major. Within each band, the k-point index is
/// `k1 + divisions[0] * (k2 + divisions[1] * k3)`, so `k1` is fastest.
/// `state_degeneracy[band]` is the number of states represented by that band
/// (for example, two for an implicit nonmagnetic spin pair).
#[derive(Clone, Debug, PartialEq)]
pub struct RegularSpectrum {
    pub divisions: [usize; 3],
    pub reciprocal_basis: [[InverseBohr; 3]; 3],
    pub energies: Vec<Hartree>,
    pub state_degeneracy: Vec<u32>,
}

impl RegularSpectrum {
    pub fn new(
        divisions: [usize; 3],
        reciprocal_basis: [[InverseBohr; 3]; 3],
        energies: Vec<Hartree>,
        state_degeneracy: Vec<u32>,
    ) -> Result<Self, TetrahedronError> {
        let spectrum = Self {
            divisions,
            reciprocal_basis,
            energies,
            state_degeneracy,
        };
        spectrum.validate()?;
        Ok(spectrum)
    }

    /// Number of k-points on the regular mesh, `divisions[0]*divisions[1]*divisions[2]`.
    ///
    /// [`Self::new`] already checked that this product fits in `usize`.
    pub fn k_point_count(&self) -> usize {
        self.divisions[0] * self.divisions[1] * self.divisions[2]
    }

    fn validate(&self) -> Result<usize, TetrahedronError> {
        for (axis, &division) in self.divisions.iter().enumerate() {
            if division == 0 {
                return Err(TetrahedronError::ZeroDivision { axis });
            }
        }
        let k_point_count = self
            .divisions
            .into_iter()
            .try_fold(1_usize, usize::checked_mul)
            .ok_or(TetrahedronError::KPointCountOverflow)?;

        let raw_basis = self
            .reciprocal_basis
            .map(|vector| vector.map(InverseBohr::get));
        for (vector, values) in raw_basis.iter().enumerate() {
            for (component, &value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(TetrahedronError::NonFiniteReciprocalBasis {
                        vector,
                        component,
                        value,
                    });
                }
            }
        }
        if determinant(raw_basis) == 0.0 {
            return Err(TetrahedronError::SingularReciprocalBasis);
        }

        let expected_energy_count = k_point_count
            .checked_mul(self.state_degeneracy.len())
            .ok_or(TetrahedronError::EnergyCountOverflow)?;
        if self.state_degeneracy.is_empty() || self.energies.len() != expected_energy_count {
            return Err(TetrahedronError::InvalidEnergyLayout {
                energy_count: self.energies.len(),
                k_point_count,
                band_count: self.state_degeneracy.len(),
            });
        }
        for (band, &degeneracy) in self.state_degeneracy.iter().enumerate() {
            if degeneracy == 0 {
                return Err(TetrahedronError::ZeroDegeneracy { band });
            }
        }
        for (index, energy) in self.energies.iter().enumerate() {
            if !energy.get().is_finite() {
                return Err(TetrahedronError::NonFiniteEnergy {
                    index,
                    value: energy.get(),
                });
            }
        }
        Ok(k_point_count)
    }
}

/// Binned density of states derived from cumulative linear-tetrahedron counts.
#[derive(Clone, Debug, PartialEq)]
pub struct TetrahedronDosBins {
    /// Strictly increasing bin edges.
    pub edges: Vec<Hartree>,
    /// State count per Hartree in each interval between neighboring edges.
    pub density: Vec<f64>,
    /// Cumulative state count at every edge.
    pub integrated_count: Vec<f64>,
}

/// Invalid regular spectrum or DOS-bin request.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum TetrahedronError {
    #[error("regular-mesh division on axis {axis} is zero")]
    ZeroDivision { axis: usize },
    #[error("regular-mesh k-point count overflows usize")]
    KPointCountOverflow,
    #[error("regular-spectrum energy count overflows usize")]
    EnergyCountOverflow,
    #[error(
        "invalid band-major energy layout: got {energy_count} energies for {band_count} bands and {k_point_count} k points"
    )]
    InvalidEnergyLayout {
        energy_count: usize,
        k_point_count: usize,
        band_count: usize,
    },
    #[error("band {band} has zero state degeneracy")]
    ZeroDegeneracy { band: usize },
    #[error("energy {index} is non-finite: {value} Ha")]
    NonFiniteEnergy { index: usize, value: f64 },
    #[error("reciprocal-basis vector {vector} component {component} is non-finite: {value}")]
    NonFiniteReciprocalBasis {
        vector: usize,
        component: usize,
        value: f64,
    },
    #[error("reciprocal basis is singular")]
    SingularReciprocalBasis,
    #[error("DOS binning needs at least two edges, got {count}")]
    TooFewEdges { count: usize },
    #[error("DOS edge {index} is non-finite: {value} Ha")]
    NonFiniteEdge { index: usize, value: f64 },
    #[error("DOS edges {index} and {next_index} are not strictly increasing: {lower} >= {upper}")]
    NonIncreasingEdges {
        index: usize,
        next_index: usize,
        lower: f64,
        upper: f64,
    },
}

/// Integrate a regular spectrum into DOS bins with the linear tetrahedron method.
///
/// Each periodic mesh cell is split into six tetrahedra. The shortest body
/// diagonal is used; when several diagonals are tied within the SPEX tolerance,
/// their six-tetrahedron decompositions are averaged. This routine only computes
/// a DOS and cumulative state count. It does not determine occupations or a
/// chemical potential and does not apply a Blöchl correction or smearing.
pub fn tetrahedron_dos_bins(
    spectrum: &RegularSpectrum,
    edges: &[Hartree],
) -> Result<TetrahedronDosBins, TetrahedronError> {
    let k_point_count = spectrum.k_point_count();
    validate_edges(edges)?;

    let tetrahedron_sets = shortest_tetrahedron_sets(spectrum);
    let mut integrated_count = vec![0.0; edges.len()];
    let set_weight = 1.0 / tetrahedron_sets.len() as f64;
    let tetrahedron_weight = set_weight / (6 * k_point_count) as f64;
    let divisions = spectrum.divisions;

    for (band, &degeneracy) in spectrum.state_degeneracy.iter().enumerate() {
        let band_offset = band * k_point_count;
        let state_weight = f64::from(degeneracy) * tetrahedron_weight;
        for k3 in 0..divisions[2] {
            for k2 in 0..divisions[1] {
                for k1 in 0..divisions[0] {
                    let corners = cell_corners([k1, k2, k3], divisions);
                    for tetrahedron_set in &tetrahedron_sets {
                        for tetrahedron in *tetrahedron_set {
                            let mut tetrahedron_energies = tetrahedron.map(|corner| {
                                spectrum.energies[band_offset + corners[corner]].get()
                            });
                            tetrahedron_energies.sort_by(f64::total_cmp);
                            for (count, edge) in integrated_count.iter_mut().zip(edges) {
                                *count += state_weight
                                    * sorted_tetrahedron_cumulative_fraction(
                                        tetrahedron_energies,
                                        edge.get(),
                                    );
                            }
                        }
                    }
                }
            }
        }
    }

    let maximum_energy = spectrum
        .energies
        .iter()
        .map(|energy| energy.get())
        .fold(f64::NEG_INFINITY, f64::max);
    let total_states: f64 = spectrum
        .state_degeneracy
        .iter()
        .map(|&degeneracy| f64::from(degeneracy))
        .sum();
    for (count, edge) in integrated_count.iter_mut().zip(edges) {
        if edge.get() >= maximum_energy {
            *count = total_states;
        }
    }

    let density = edges
        .windows(2)
        .zip(integrated_count.windows(2))
        .map(|(edge_pair, count_pair)| {
            (count_pair[1] - count_pair[0]) / (edge_pair[1].get() - edge_pair[0].get())
        })
        .collect();

    Ok(TetrahedronDosBins {
        edges: edges.to_vec(),
        density,
        integrated_count,
    })
}

fn validate_edges(edges: &[Hartree]) -> Result<(), TetrahedronError> {
    if edges.len() < 2 {
        return Err(TetrahedronError::TooFewEdges { count: edges.len() });
    }
    for (index, edge) in edges.iter().enumerate() {
        if !edge.get().is_finite() {
            return Err(TetrahedronError::NonFiniteEdge {
                index,
                value: edge.get(),
            });
        }
    }
    for (index, pair) in edges.windows(2).enumerate() {
        if pair[1].get() <= pair[0].get() {
            return Err(TetrahedronError::NonIncreasingEdges {
                index,
                next_index: index + 1,
                lower: pair[0].get(),
                upper: pair[1].get(),
            });
        }
    }
    Ok(())
}

fn determinant(matrix: [[f64; 3]; 3]) -> f64 {
    let [a, b, c] = matrix;
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

fn cell_corners(origin: [usize; 3], divisions: [usize; 3]) -> [usize; 8] {
    let [k1, k2, k3] = origin;
    let next1 = (k1 + 1) % divisions[0];
    let next2 = (k2 + 1) % divisions[1];
    let next3 = (k3 + 1) % divisions[2];
    [
        k_index([k1, k2, k3], divisions),
        k_index([next1, k2, k3], divisions),
        k_index([k1, next2, k3], divisions),
        k_index([next1, next2, k3], divisions),
        k_index([k1, k2, next3], divisions),
        k_index([next1, k2, next3], divisions),
        k_index([k1, next2, next3], divisions),
        k_index([next1, next2, next3], divisions),
    ]
}

fn k_index(k: [usize; 3], divisions: [usize; 3]) -> usize {
    k[0] + divisions[0] * (k[1] + divisions[1] * k[2])
}

fn shortest_tetrahedron_sets(spectrum: &RegularSpectrum) -> Vec<&'static [[usize; 4]; 6]> {
    const TIE_TOLERANCE: f64 = 1.0e-8;

    let basis = spectrum
        .reciprocal_basis
        .map(|vector| vector.map(InverseBohr::get));
    let step = std::array::from_fn::<_, 3, _>(|axis| {
        basis[axis].map(|component| component / spectrum.divisions[axis] as f64)
    });
    let diagonal_squared = [
        squared_norm(add(subtract(step[0], step[1]), step[2])),
        squared_norm(add(subtract(step[1], step[0]), step[2])),
        squared_norm(add(add(step[0], step[1]), step[2])),
        squared_norm(subtract(add(step[0], step[1]), step[2])),
    ];
    let shortest = diagonal_squared.into_iter().fold(f64::INFINITY, f64::min);
    TETRAHEDRON_SETS
        .iter()
        .zip(diagonal_squared)
        .filter_map(|(set, length_squared)| {
            ((length_squared - shortest).abs() < TIE_TOLERANCE).then_some(set)
        })
        .collect()
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn squared_norm(vector: [f64; 3]) -> f64 {
    vector
        .into_iter()
        .map(|component| component * component)
        .sum()
}

fn sorted_tetrahedron_cumulative_fraction(energies: [f64; 4], edge: f64) -> f64 {
    let [e0, e1, e2, e3] = energies;

    let fraction = if edge < e0 {
        0.0
    } else if edge >= e3 {
        1.0
    } else if edge <= e0 {
        // This branch matters when two or three lower vertices are degenerate.
        0.0
    } else if edge <= e1 {
        (edge - e0).powi(3) / ((e1 - e0) * (e2 - e0) * (e3 - e0))
    } else if edge <= e2 {
        let f0_to_2 = (edge - e0) / (e2 - e0);
        let f0_to_3 = (edge - e0) / (e3 - e0);
        let f1_to_2 = (edge - e1) / (e2 - e1);
        let f1_to_3 = (edge - e1) / (e3 - e1);
        f0_to_2 * f0_to_3
            + f0_to_2 * (1.0 - f0_to_3) * f1_to_3
            + f1_to_2 * (1.0 - f0_to_2) * f1_to_3
    } else {
        let f0_to_3 = (edge - e0) / (e3 - e0);
        let f1_to_3 = (edge - e1) / (e3 - e1);
        let f2_to_3 = (edge - e2) / (e3 - e2);
        f1_to_3 + f0_to_3 * (1.0 - f1_to_3) * (1.0 - f2_to_3) + f2_to_3 * (1.0 - f1_to_3)
    };
    fraction.clamp(0.0, 1.0)
}

// SPEX cube corners, converted from one-based Fortran indices.
const TETRAHEDRON_SETS: [[[usize; 4]; 6]; 4] = [
    [
        [0, 1, 2, 5],
        [4, 6, 2, 5],
        [0, 4, 2, 5],
        [1, 3, 2, 5],
        [3, 7, 2, 5],
        [6, 7, 2, 5],
    ],
    [
        [4, 5, 1, 6],
        [0, 4, 1, 6],
        [0, 2, 1, 6],
        [7, 5, 1, 6],
        [3, 2, 1, 6],
        [7, 3, 1, 6],
    ],
    [
        [1, 5, 0, 7],
        [1, 3, 0, 7],
        [2, 3, 0, 7],
        [2, 6, 0, 7],
        [4, 6, 0, 7],
        [4, 5, 0, 7],
    ],
    [
        [1, 5, 3, 4],
        [0, 1, 3, 4],
        [0, 2, 3, 4],
        [2, 6, 3, 4],
        [6, 7, 3, 4],
        [5, 7, 3, 4],
    ],
];

#[cfg(test)]
mod tests {
    use super::*;

    fn cubic_basis() -> [[InverseBohr; 3]; 3] {
        [
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ]
    }

    fn assert_close(left: f64, right: f64, tolerance: f64) {
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }

    #[test]
    fn single_tetrahedron_cumulative_is_continuous_monotone_and_has_exact_endpoints() {
        let energies = [-1.0, 0.0, 2.0, 4.0];
        assert_eq!(sorted_tetrahedron_cumulative_fraction(energies, -2.0), 0.0);
        assert_eq!(sorted_tetrahedron_cumulative_fraction(energies, 4.0), 1.0);

        let mut previous = 0.0;
        for step in 0..=600 {
            let edge = -1.0 + 5.0 * step as f64 / 600.0;
            let current = sorted_tetrahedron_cumulative_fraction(energies, edge);
            assert!(current >= previous);
            previous = current;
        }
        for knot in [0.0, 2.0] {
            let left = sorted_tetrahedron_cumulative_fraction(energies, knot - 1.0e-8);
            let at = sorted_tetrahedron_cumulative_fraction(energies, knot);
            let right = sorted_tetrahedron_cumulative_fraction(energies, knot + 1.0e-8);
            assert!((at - left).abs() < 1.0e-7);
            assert!((right - at).abs() < 1.0e-7);
        }
    }

    #[test]
    fn flat_band_has_complete_finite_weight() {
        let spectrum =
            RegularSpectrum::new([2, 2, 2], cubic_basis(), vec![Hartree(0.0); 8], vec![2]).unwrap();
        let bins =
            tetrahedron_dos_bins(&spectrum, &[Hartree(-1.0), Hartree(0.0), Hartree(1.0)]).unwrap();
        assert_eq!(bins.integrated_count, vec![0.0, 2.0, 2.0]);
        assert_eq!(bins.density, vec![2.0, 0.0]);
        assert!(bins.density.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn each_band_integrates_to_its_state_degeneracy() {
        let first_band = [-0.8, -0.2, 0.1, 0.5, -0.4, 0.2, 0.6, 0.9];
        let second_band = first_band.map(|energy| energy + 10.0);
        let spectrum = RegularSpectrum::new(
            [2, 2, 2],
            cubic_basis(),
            first_band
                .into_iter()
                .chain(second_band)
                .map(Hartree)
                .collect(),
            vec![2, 3],
        )
        .unwrap();
        let bins = tetrahedron_dos_bins(
            &spectrum,
            &[Hartree(-2.0), Hartree(2.0), Hartree(8.0), Hartree(12.0)],
        )
        .unwrap();
        assert_close(bins.integrated_count[1], 2.0, 1.0e-13);
        assert_close(bins.integrated_count[2], 2.0, 1.0e-13);
        assert_eq!(bins.integrated_count[3], 5.0);
        assert_close(bins.density[0] * 4.0, 2.0, 1.0e-13);
        assert_close(bins.density[2] * 4.0, 3.0, 1.0e-13);
    }

    #[test]
    fn periodic_origin_shift_and_band_permutation_do_not_change_dos() {
        let divisions = [2, 3, 2];
        let k_point_count = divisions.into_iter().product();
        let first: Vec<f64> = (0..k_point_count)
            .map(|index| ((index * 7 + 3) % 17) as f64 / 5.0 - 1.0)
            .collect();
        let second: Vec<f64> = first.iter().map(|energy| energy + 4.0).collect();
        let mut shifted_first = vec![0.0; k_point_count];
        let mut shifted_second = vec![0.0; k_point_count];
        for k3 in 0..divisions[2] {
            for k2 in 0..divisions[1] {
                for k1 in 0..divisions[0] {
                    let target = k_index([k1, k2, k3], divisions);
                    let source = k_index([(k1 + 1) % 2, (k2 + 2) % 3, (k3 + 1) % 2], divisions);
                    shifted_first[target] = first[source];
                    shifted_second[target] = second[source];
                }
            }
        }
        let original = RegularSpectrum::new(
            divisions,
            cubic_basis(),
            first.iter().chain(&second).copied().map(Hartree).collect(),
            vec![1, 2],
        )
        .unwrap();
        let shifted_and_permuted = RegularSpectrum::new(
            divisions,
            cubic_basis(),
            shifted_second
                .iter()
                .chain(&shifted_first)
                .copied()
                .map(Hartree)
                .collect(),
            vec![2, 1],
        )
        .unwrap();
        let edges: Vec<_> = (-4..=16).map(|index| Hartree(index as f64 / 2.0)).collect();
        let expected = tetrahedron_dos_bins(&original, &edges).unwrap();
        let actual = tetrahedron_dos_bins(&shifted_and_permuted, &edges).unwrap();
        for (&left, &right) in expected
            .integrated_count
            .iter()
            .zip(&actual.integrated_count)
        {
            assert_close(left, right, 2.0e-13);
        }
    }

    #[test]
    fn cubic_axis_permutation_is_invariant_under_shortest_diagonal_tie_average() {
        let divisions = [2, 2, 2];
        let energies = [-0.7, 0.2, 1.6, -0.1, 2.4, 0.8, -1.2, 1.1];
        let mut permuted = [0.0; 8];
        for k3 in 0..2 {
            for k2 in 0..2 {
                for k1 in 0..2 {
                    permuted[k_index([k1, k2, k3], divisions)] =
                        energies[k_index([k2, k1, k3], divisions)];
                }
            }
        }
        let original = RegularSpectrum::new(
            divisions,
            cubic_basis(),
            energies.map(Hartree).to_vec(),
            vec![1],
        )
        .unwrap();
        let mut permuted_basis = cubic_basis();
        permuted_basis.swap(0, 1);
        let permuted = RegularSpectrum::new(
            divisions,
            permuted_basis,
            permuted.map(Hartree).to_vec(),
            vec![1],
        )
        .unwrap();
        assert_eq!(shortest_tetrahedron_sets(&original).len(), 4);
        let edges: Vec<_> = (-8..=16).map(|index| Hartree(index as f64 / 5.0)).collect();
        let expected = tetrahedron_dos_bins(&original, &edges).unwrap();
        let actual = tetrahedron_dos_bins(&permuted, &edges).unwrap();
        for (&left, &right) in expected
            .integrated_count
            .iter()
            .zip(&actual.integrated_count)
        {
            assert_close(left, right, 1.0e-13);
        }
    }

    #[test]
    fn rejects_invalid_divisions_edges_layout_degeneracy_and_non_finite_values() {
        assert!(matches!(
            RegularSpectrum::new([0, 2, 2], cubic_basis(), Vec::new(), Vec::new()),
            Err(TetrahedronError::ZeroDivision { axis: 0 })
        ));
        assert!(matches!(
            RegularSpectrum::new([2, 2, 2], cubic_basis(), vec![Hartree(0.0); 7], vec![1]),
            Err(TetrahedronError::InvalidEnergyLayout { .. })
        ));
        assert!(matches!(
            RegularSpectrum::new([1, 1, 1], cubic_basis(), vec![Hartree(0.0)], vec![0]),
            Err(TetrahedronError::ZeroDegeneracy { band: 0 })
        ));
        assert!(matches!(
            RegularSpectrum::new([1, 1, 1], cubic_basis(), vec![Hartree(f64::NAN)], vec![1]),
            Err(TetrahedronError::NonFiniteEnergy { index: 0, .. })
        ));
        let spectrum =
            RegularSpectrum::new([1, 1, 1], cubic_basis(), vec![Hartree(0.0)], vec![1]).unwrap();
        for edges in [
            vec![Hartree(0.0)],
            vec![Hartree(0.0), Hartree(0.0)],
            vec![Hartree(0.0), Hartree(f64::INFINITY)],
        ] {
            assert!(tetrahedron_dos_bins(&spectrum, &edges).is_err());
        }
        let mut invalid_basis = cubic_basis();
        invalid_basis[1][2] = InverseBohr(f64::NAN);
        assert!(matches!(
            RegularSpectrum::new([1, 1, 1], invalid_basis, vec![Hartree(0.0)], vec![1]),
            Err(TetrahedronError::NonFiniteReciprocalBasis { .. })
        ));
    }
}
