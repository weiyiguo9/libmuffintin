//! Per-q orbital-pair collocation in the canonical-q / Umklapp gauge.

use crate::ThcError;
use crate::error::checked_storage_len;
use crate::kmesh::{KMesh, umklapp_phase};
use muffintin_auxiliary_ir::PairColumnLayout;
use num_complex::Complex64;

/// How the Umklapp phase is applied to a pair column.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UmklappGauge {
    /// $\exp(+i G_{\mathrm{wrap}}\cdot r)$, the scratch convention.
    Canonical,
    /// Drop the phase (regression: omitted wrap).
    Omit,
    /// $\exp(-i G_{\mathrm{wrap}}\cdot r)$ (regression: sign flip).
    SignFlip,
    /// $\exp(+2i G_{\mathrm{wrap}}\cdot r)$ (regression: double count).
    DoubleCount,
}

/// Cell-periodic Bloch orbitals $u_{ik}(r)$ on a grid: `(point, k, orb)`.
#[derive(Clone, Debug, PartialEq)]
pub struct BlochOrbitals {
    pub n_points: usize,
    pub n_k: usize,
    pub n_orb: usize,
    values: Vec<Complex64>,
}

impl BlochOrbitals {
    /// Construct after checking `values` length.
    pub fn new(
        n_points: usize,
        n_k: usize,
        n_orb: usize,
        values: Vec<Complex64>,
    ) -> Result<Self, ThcError> {
        let expected = checked_storage_len(&[n_points, n_k, n_orb])?;
        if values.len() != expected {
            return Err(ThcError::OrbitalCount {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            n_points,
            n_k,
            n_orb,
            values,
        })
    }

    /// Value $u_{ik}(r_p)$.
    pub fn at(&self, point: usize, k: usize, orb: usize) -> Complex64 {
        self.values[(point * self.n_k + k) * self.n_orb + orb]
    }

    /// Layout implied by these orbitals.
    pub fn layout(&self, core_orbital: Option<usize>) -> PairColumnLayout {
        PairColumnLayout::new(self.n_k, self.n_orb, core_orbital)
    }
}

/// Pair-density block at one canonical $q$: `n_points × n_columns`, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct PairBlock {
    pub q_index: usize,
    pub n_points: usize,
    pub layout: PairColumnLayout,
    n_columns: usize,
    values: Vec<Complex64>,
}

impl PairBlock {
    /// Construct after checking length.
    pub fn new(
        q_index: usize,
        n_points: usize,
        layout: PairColumnLayout,
        values: Vec<Complex64>,
    ) -> Result<Self, ThcError> {
        let n_columns = layout.n_columns()?;
        let expected = checked_storage_len(&[n_points, n_columns])?;
        if values.len() != expected {
            return Err(ThcError::PairBlockLength {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            q_index,
            n_points,
            layout,
            n_columns,
            values,
        })
    }

    /// Row-major values.
    pub fn values(&self) -> &[Complex64] {
        &self.values
    }

    /// Number of pair columns.
    pub fn n_columns(&self) -> usize {
        self.n_columns
    }

    /// Entry at grid point `p` and pair column `column`.
    pub fn at(&self, point: usize, column: usize) -> Complex64 {
        self.values[point * self.n_columns() + column]
    }

    /// Copy selected rows, preserving column order.
    pub fn selected_rows(&self, points: &[usize]) -> Result<Vec<Complex64>, ThcError> {
        let n_col = self.n_columns();
        let mut out = Vec::with_capacity(points.len() * n_col);
        for &point in points {
            if point >= self.n_points {
                return Err(ThcError::PointIndex(point));
            }
            let start = point * n_col;
            out.extend_from_slice(&self.values[start..start + n_col]);
        }
        Ok(out)
    }
}

/// Evaluate $\rho^q_{k,ij}(r)=\mathrm{e}^{+i G_{\mathrm{wrap}}\cdot r}
/// u_{i,k-q}^*(r)\,u_{j,k}(r)$ for every grid point and pair column.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_pair_block(
    orbitals: &BlochOrbitals,
    points: &[[f64; 3]],
    mesh: &KMesh,
    iq: usize,
    core_orbital: Option<usize>,
    gauge: UmklappGauge,
) -> Result<PairBlock, ThcError> {
    if points.len() != orbitals.n_points {
        return Err(ThcError::OrbitalPointCount {
            orbitals: orbitals.n_points,
            points: points.len(),
        });
    }
    if orbitals.n_k != mesh.len() {
        return Err(ThcError::OrbitalKCount {
            orbitals: orbitals.n_k,
            mesh: mesh.len(),
        });
    }
    let layout = PairColumnLayout::new(orbitals.n_k, orbitals.n_orb, core_orbital);
    layout.require_core_orbital()?;
    let n_col = layout.n_columns()?;
    let expected = checked_storage_len(&[orbitals.n_points, n_col])?;
    let mut values = vec![Complex64::default(); expected];
    for ik in 0..orbitals.n_k {
        let (left, shift) = mesh.kminus(ik, iq)?;
        for (point_index, point) in points.iter().enumerate() {
            let phase = match gauge {
                UmklappGauge::Canonical => umklapp_phase(*point, shift, mesh.lattice_constant()),
                UmklappGauge::Omit => Complex64::new(1.0, 0.0),
                UmklappGauge::SignFlip => {
                    umklapp_phase(*point, shift, mesh.lattice_constant()).conj()
                }
                UmklappGauge::DoubleCount => {
                    let once = umklapp_phase(*point, shift, mesh.lattice_constant());
                    once * once
                }
            };
            for i in 0..orbitals.n_orb {
                let left_value = orbitals.at(point_index, left, i).conj();
                for j in 0..orbitals.n_orb {
                    let column = layout.encode(ik, i, j);
                    values[point_index * n_col + column] =
                        phase * left_value * orbitals.at(point_index, ik, j);
                }
            }
        }
    }
    PairBlock::new(iq, orbitals.n_points, layout, values)
}
