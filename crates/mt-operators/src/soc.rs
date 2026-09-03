//! Site assembly and first-variation congruence for SOC second variation.

use faer::{Mat, Side};
use muffintin_core::{Hartree, Lm, lm_count};
use muffintin_envelope::LocalOrbitalLayout;
use muffintin_sphere::SpinOrbitRadialShell;
use muffintin_tensor::{
    Axis, ComplexTensor, DenseHermitianMatrix, TensorError, hermitian_congruence,
};
use num_complex::Complex64;
use thiserror::Error;

use crate::SiteOrbitalCoefficients;

/// Hermitian sphere SOC operator on doubled scalar site coordinates.
///
/// Rows and columns are spin slow: all scalar site coordinates for spin up,
/// then the same coordinates for spin down.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteSpinOrbitBlock {
    scalar_coordinate_count: usize,
    matrix: DenseHermitianMatrix,
}

impl SiteSpinOrbitBlock {
    pub const fn scalar_coordinate_count(&self) -> usize {
        self.scalar_coordinate_count
    }

    pub const fn matrix(&self) -> &DenseHermitianMatrix {
        &self.matrix
    }

    /// Assemble the SPEX `sigma dot L` selection rules with the radial factor
    /// `dV/dr / (4 M^2 c^2 r)`.
    ///
    /// `shells[l]` must be the `(u_l, du_l/dE, LO...)` radial matrix for `l`.
    /// This product is exactly the conventional `xi(r) L dot S` operator; the
    /// factor two resides in `sigma = 2 S`.
    pub fn from_radial_shells(
        local_orbitals: &LocalOrbitalLayout,
        shells: &[SpinOrbitRadialShell],
    ) -> Result<Self, SocOperatorError> {
        if shells.is_empty() {
            return Err(SocOperatorError::EmptyRadialShells);
        }
        let l_max = u32::try_from(shells.len() - 1)
            .map_err(|_| SocOperatorError::AngularMomentumOverflow)?;
        if local_orbitals
            .counts_by_l()
            .iter()
            .skip(shells.len())
            .any(|count| *count != 0)
        {
            return Err(SocOperatorError::MissingLocalOrbitalShells);
        }
        for (l, shell) in shells.iter().enumerate() {
            let l_u32 = u32::try_from(l).map_err(|_| SocOperatorError::AngularMomentumOverflow)?;
            if shell.angular_momentum() != l_u32 {
                return Err(SocOperatorError::RadialShellOrder {
                    expected: l_u32,
                    actual: shell.angular_momentum(),
                });
            }
            let local_count = local_orbitals.counts_by_l().get(l).copied().unwrap_or(0);
            let expected = 2 + local_count;
            if shell.dimension() != expected {
                return Err(SocOperatorError::RadialShellDimension {
                    l: l_u32,
                    expected,
                    actual: shell.dimension(),
                });
            }
        }

        let augmented_count = lm_count(l_max);
        let scalar_coordinate_count = 2 * augmented_count + local_orbitals.len();
        let dimension = 2 * scalar_coordinate_count;
        let mut values = vec![Complex64::default(); dimension * dimension];

        for (l, shell) in shells.iter().enumerate().skip(1) {
            let l_u32 = u32::try_from(l).expect("shell index was validated as u32");
            let l_value = f64::from(l_u32);
            for m in -(l as i32)..=l as i32 {
                for left_radial in 0..shell.dimension() {
                    let left =
                        scalar_coordinate(local_orbitals, augmented_count, l_u32, m, left_radial);
                    for right_radial in 0..shell.dimension() {
                        let right = scalar_coordinate(
                            local_orbitals,
                            augmented_count,
                            l_u32,
                            m,
                            right_radial,
                        );
                        let radial = shell.at(left_radial, right_radial);
                        values[left * dimension + right] += radial * f64::from(m);
                        let down_left = scalar_coordinate_count + left;
                        let down_right = scalar_coordinate_count + right;
                        values[down_left * dimension + down_right] -= radial * f64::from(m);
                    }
                }

                if m < l as i32 {
                    let coefficient = (l_value * (l_value + 1.0) - f64::from(m * (m + 1))).sqrt();
                    for left_radial in 0..shell.dimension() {
                        let up = scalar_coordinate(
                            local_orbitals,
                            augmented_count,
                            l_u32,
                            m,
                            left_radial,
                        );
                        for right_radial in 0..shell.dimension() {
                            let down = scalar_coordinate_count
                                + scalar_coordinate(
                                    local_orbitals,
                                    augmented_count,
                                    l_u32,
                                    m + 1,
                                    right_radial,
                                );
                            let value = coefficient * shell.at(left_radial, right_radial);
                            values[up * dimension + down] += value;
                            values[down * dimension + up] += value;
                        }
                    }
                }
            }
        }

        Ok(Self {
            scalar_coordinate_count,
            matrix: DenseHermitianMatrix::from_host_row_major(
                dimension,
                Axis::SiteCoordinate,
                values,
            )?,
        })
    }
}

/// Project a doubled site SOC block into the doubled scalar first-variation
/// band subspace, using the already compiled `A = P C` coefficients.
pub fn project_site_soc_to_subspace(
    block: &SiteSpinOrbitBlock,
    coefficients: &SiteOrbitalCoefficients,
) -> Result<DenseHermitianMatrix, SocOperatorError> {
    project_site_spinor_operator_to_subspace(
        block.scalar_coordinate_count,
        &block.matrix,
        coefficients,
    )
}

/// Project an arbitrary doubled scalar-site operator into the doubled
/// first-variation band subspace.
pub fn project_site_spinor_operator_to_subspace(
    scalar_coordinate_count: usize,
    block: &DenseHermitianMatrix,
    coefficients: &SiteOrbitalCoefficients,
) -> Result<DenseHermitianMatrix, SocOperatorError> {
    if coefficients.coordinate_count() != scalar_coordinate_count {
        return Err(SocOperatorError::SiteCoefficientDimension {
            expected: scalar_coordinate_count,
            actual: coefficients.coordinate_count(),
        });
    }
    if block.dimension() != 2 * scalar_coordinate_count {
        return Err(SocOperatorError::DoubledSiteOperatorDimension {
            expected: 2 * scalar_coordinate_count,
            actual: block.dimension(),
        });
    }
    if block.axis() != Axis::SiteCoordinate {
        return Err(SocOperatorError::Tensor(TensorError::Axis {
            index: 0,
            expected: Axis::SiteCoordinate,
            actual: block.axis(),
        }));
    }
    let coordinates = coefficients.coordinate_count();
    let bands = coefficients.band_count();
    let mut doubled = vec![Complex64::default(); 4 * coordinates * bands];
    let doubled_bands = 2 * bands;
    for spin in 0..2 {
        for coordinate in 0..coordinates {
            for band in 0..bands {
                doubled[(spin * coordinates + coordinate) * doubled_bands + spin * bands + band] =
                    coefficients.at(coordinate, band);
            }
        }
    }
    let projection = ComplexTensor::from_host_row_major(
        &[2 * coordinates, doubled_bands],
        &[Axis::SiteCoordinate, Axis::Band],
        doubled,
    )?;
    hermitian_congruence(&projection, block).map_err(Into::into)
}

/// Residual of one ordinary Hermitian second-variation eigenpair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SocEigenpairResidual {
    pub band_index: usize,
    pub absolute: f64,
    pub relative: f64,
}

/// Mixing eigenvectors on the doubled first-variation subspace.
///
/// Rows are `(spin, source band)` with spin slow. Columns are SOC bands.
#[derive(Clone, Debug, PartialEq)]
pub struct SecondVariationMixing {
    dimension: usize,
    column_major: Vec<Complex64>,
}

impl SecondVariationMixing {
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn at(&self, row: usize, column: usize) -> Complex64 {
        self.column_major[column * self.dimension + row]
    }

    pub fn to_host_column_major(&self) -> Vec<Complex64> {
        self.column_major.clone()
    }
}

/// Ordinary Hermitian eigensolution of the second-variation Hamiltonian.
#[derive(Clone, Debug, PartialEq)]
pub struct SecondVariationSubspaceSolution {
    pub eigenvalues: Vec<Hartree>,
    pub mixing: SecondVariationMixing,
    pub residuals: Vec<SocEigenpairResidual>,
}

/// Form `diag(epsilon, epsilon) + sum_site A^H H_soc A` and diagonalize it.
pub fn solve_second_variation_subspace(
    first_variation_energies: &[Hartree],
    site_contributions: &[DenseHermitianMatrix],
) -> Result<SecondVariationSubspaceSolution, SocOperatorError> {
    let bands = first_variation_energies.len();
    if bands == 0 {
        return Err(SocOperatorError::EmptyFirstVariationSubspace);
    }
    if let Some((band, energy)) = first_variation_energies
        .iter()
        .enumerate()
        .find(|(_, energy)| !energy.get().is_finite())
    {
        return Err(SocOperatorError::NonFiniteEnergy {
            band,
            value: energy.get(),
        });
    }
    let dimension = 2 * bands;
    let mut hamiltonian = vec![Complex64::default(); dimension * dimension];
    for spin in 0..2 {
        for (band, energy) in first_variation_energies.iter().enumerate() {
            let index = spin * bands + band;
            hamiltonian[index * dimension + index] = Complex64::new(energy.get(), 0.0);
        }
    }
    let mut has_soc = false;
    for contribution in site_contributions {
        if contribution.dimension() != dimension {
            return Err(SocOperatorError::SubspaceContributionDimension {
                expected: dimension,
                actual: contribution.dimension(),
            });
        }
        if contribution.axis() != Axis::Band {
            return Err(SocOperatorError::Tensor(TensorError::Axis {
                index: 0,
                expected: Axis::Band,
                actual: contribution.axis(),
            }));
        }
        for (target, value) in hamiltonian.iter_mut().zip(contribution.to_host_row_major()) {
            has_soc |= value != Complex64::default();
            *target += value;
        }
    }

    // Preserve the exact doubled first-variation states for the important
    // SOC=0 contract instead of accepting an arbitrary rotation within every
    // degenerate spin pair from a dense eigensolver.
    if !has_soc {
        let mut order = (0..dimension).collect::<Vec<_>>();
        order.sort_by(|&left, &right| {
            let left_energy = first_variation_energies[left % bands].get();
            let right_energy = first_variation_energies[right % bands].get();
            left_energy
                .total_cmp(&right_energy)
                .then_with(|| (left / bands).cmp(&(right / bands)))
        });
        let eigenvalues = order
            .iter()
            .map(|row| first_variation_energies[row % bands])
            .collect();
        let mut mixing = vec![Complex64::default(); dimension * dimension];
        for (column, &row) in order.iter().enumerate() {
            mixing[column * dimension + row] = Complex64::new(1.0, 0.0);
        }
        return Ok(SecondVariationSubspaceSolution {
            eigenvalues,
            mixing: SecondVariationMixing {
                dimension,
                column_major: mixing,
            },
            residuals: (0..dimension)
                .map(|band_index| SocEigenpairResidual {
                    band_index,
                    absolute: 0.0,
                    relative: 0.0,
                })
                .collect(),
        });
    }

    let matrix = Mat::from_fn(dimension, dimension, |row, column| {
        hamiltonian[row * dimension + column]
    });
    let eigen = matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| SocOperatorError::Eigensolver)?;
    let eigenvalues = (0..dimension)
        .map(|band| Hartree(eigen.S()[band].re))
        .collect::<Vec<_>>();
    let mut mixing = vec![Complex64::default(); dimension * dimension];
    for column in 0..dimension {
        for row in 0..dimension {
            mixing[column * dimension + row] = eigen.U()[(row, column)];
        }
    }
    let residuals = (0..dimension)
        .map(|band_index| {
            let mut absolute_squared = 0.0;
            let mut h_norm_squared = 0.0;
            for row in 0..dimension {
                let mut h_value = Complex64::default();
                for column in 0..dimension {
                    h_value += hamiltonian[row * dimension + column]
                        * mixing[band_index * dimension + column];
                }
                let residual =
                    h_value - mixing[band_index * dimension + row] * eigenvalues[band_index].get();
                absolute_squared += residual.norm_sqr();
                h_norm_squared += h_value.norm_sqr();
            }
            let absolute = absolute_squared.sqrt();
            let scale = h_norm_squared
                .sqrt()
                .max(eigenvalues[band_index].get().abs());
            SocEigenpairResidual {
                band_index,
                absolute,
                relative: if scale == 0.0 {
                    absolute
                } else {
                    absolute / scale
                },
            }
        })
        .collect();
    Ok(SecondVariationSubspaceSolution {
        eigenvalues,
        mixing: SecondVariationMixing {
            dimension,
            column_major: mixing,
        },
        residuals,
    })
}

fn scalar_coordinate(
    local_orbitals: &LocalOrbitalLayout,
    augmented_count: usize,
    l: u32,
    m: i32,
    radial: usize,
) -> usize {
    if radial < 2 {
        2 * Lm::new(l, m).expect("assembly loop validates m").index() + radial
    } else {
        2 * augmented_count
            + local_orbitals
                .index(l, m, radial - 2)
                .expect("radial-shell dimension matches the local-orbital layout")
    }
}

/// Invalid site assembly, congruence, or SOC subspace eigensolve.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SocOperatorError {
    #[error("at least the l=0 radial shell is required")]
    EmptyRadialShells,
    #[error("angular-momentum shell index exceeds u32")]
    AngularMomentumOverflow,
    #[error("local-orbital layout has shells beyond the supplied SOC radial shells")]
    MissingLocalOrbitalShells,
    #[error("SOC radial shells must be ordered by l: expected {expected}, found {actual}")]
    RadialShellOrder { expected: u32, actual: u32 },
    #[error("SOC radial shell l={l} has dimension {actual}, expected {expected}")]
    RadialShellDimension {
        l: u32,
        expected: usize,
        actual: usize,
    },
    #[error("site coefficients have {actual} rows, expected {expected}")]
    SiteCoefficientDimension { expected: usize, actual: usize },
    #[error("doubled site operator has dimension {actual}, expected {expected}")]
    DoubledSiteOperatorDimension { expected: usize, actual: usize },
    #[error("first-variation subspace is empty")]
    EmptyFirstVariationSubspace,
    #[error("first-variation energy {band} is non-finite: {value}")]
    NonFiniteEnergy { band: usize, value: f64 },
    #[error("site SOC contribution has dimension {actual}, expected {expected}")]
    SubspaceContributionDimension { expected: usize, actual: usize },
    #[error("dense self-adjoint SOC eigendecomposition failed")]
    Eigensolver,
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::{Bohr, ExponentialMesh};
    use muffintin_sphere::{
        RadialEquation, RadialSolver, SpexSpinOrbitPotential, spex_spin_orbit_radial_shell,
    };

    fn radial_shells() -> Vec<SpinOrbitRadialShell> {
        let mesh = ExponentialMesh::new(Bohr(1.0e-5), 0.015, 401).unwrap();
        let potential = mesh
            .radii()
            .iter()
            .map(|radius| -0.8 / (radius.get() + 0.2))
            .collect::<Vec<_>>();
        let soc = SpexSpinOrbitPotential::new(&mesh, &potential).unwrap();
        let solver =
            RadialSolver::new(&mesh, &potential, RadialEquation::ScalarKoellingHarmon).unwrap();
        (0..=1)
            .map(|l| {
                let linearized = solver
                    .solve_with_energy_derivative(l, Hartree(-0.2))
                    .unwrap();
                spex_spin_orbit_radial_shell(&mesh, &soc, &linearized, &[]).unwrap()
            })
            .collect()
    }

    #[test]
    fn ordinary_subspace_solver_matches_two_by_two_reference_and_shifts_covariantly() {
        let contribution =
            DenseHermitianMatrix::from_upper_triangle(2, Axis::Band, |row, column| {
                match (row, column) {
                    (0, 0) => Complex64::new(0.2, 0.0),
                    (0, 1) => Complex64::new(0.1, -0.05),
                    (1, 1) => Complex64::new(-0.1, 0.0),
                    _ => unreachable!(),
                }
            })
            .unwrap();
        let base =
            solve_second_variation_subspace(&[Hartree(0.7)], std::slice::from_ref(&contribution))
                .unwrap();
        let trace_half = 0.75;
        let radius = (0.15_f64.powi(2) + 0.1_f64.powi(2) + 0.05_f64.powi(2)).sqrt();
        assert!((base.eigenvalues[0].get() - (trace_half - radius)).abs() < 1.0e-13);
        assert!((base.eigenvalues[1].get() - (trace_half + radius)).abs() < 1.0e-13);
        assert!(
            base.residuals
                .iter()
                .all(|residual| residual.absolute < 1.0e-13)
        );

        let shifted = solve_second_variation_subspace(&[Hartree(2.0)], &[contribution]).unwrap();
        for band in 0..2 {
            assert!(
                (shifted.eigenvalues[band].get() - base.eigenvalues[band].get() - 1.3).abs()
                    < 1.0e-13
            );
        }
    }

    #[test]
    fn zero_soc_returns_exact_spin_doubled_source_states() {
        let solution =
            solve_second_variation_subspace(&[Hartree(-0.2), Hartree(0.4)], &[]).unwrap();
        assert_eq!(
            solution.eigenvalues,
            vec![Hartree(-0.2), Hartree(-0.2), Hartree(0.4), Hartree(0.4)]
        );
        let selected_rows = [0, 2, 1, 3];
        for (column, selected) in selected_rows.into_iter().enumerate() {
            for row in 0..4 {
                let expected = if row == selected { 1.0 } else { 0.0 };
                assert_eq!(
                    solution.mixing.at(row, column),
                    Complex64::new(expected, 0.0)
                );
            }
        }
    }

    #[test]
    fn site_block_is_hermitian_and_obeys_l_dot_sigma_selection_rules() {
        let shells = radial_shells();
        let radial = shells[1].at(0, 0);
        assert_ne!(radial, 0.0);
        let block =
            SiteSpinOrbitBlock::from_radial_shells(&LocalOrbitalLayout::new(vec![0, 0]), &shells)
                .unwrap();
        assert_eq!(block.scalar_coordinate_count(), 8);
        let matrix = block.matrix();
        for row in 0..matrix.dimension() {
            for column in 0..matrix.dimension() {
                assert_eq!(matrix.at(row, column), matrix.at(column, row).conj());
            }
        }
        let up_m0 = 2 * Lm::new(1, 0).unwrap().index();
        let up_m1 = 2 * Lm::new(1, 1).unwrap().index();
        let down_m1 = block.scalar_coordinate_count() + up_m1;
        assert!((matrix.at(up_m1, up_m1).re - radial).abs() < 1.0e-15);
        assert!((matrix.at(down_m1, down_m1).re + radial).abs() < 1.0e-15);
        assert!((matrix.at(up_m0, down_m1).re - 2.0_f64.sqrt() * radial).abs() < 1.0e-15);
    }
}
