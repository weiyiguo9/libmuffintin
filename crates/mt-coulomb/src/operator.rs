//! Finite-$q$ Coulomb operator over a compiled auxiliary basis.

use crate::CoulombError;
use muffintin_core::{Bohr, Cell, InverseBohr, ReciprocalLattice};
use muffintin_envelope::Provenance;
use muffintin_prodbasis::{AuxiliaryLayout, AuxiliaryRegion, PairVertex, TransferQ};
use muffintin_tensor::{Axis, ComplexTensor, einsum};
use num_complex::Complex64;

/// Typed auxiliary representation stored with $V^q$. Neither mixed-product nor
/// interpolation points is a privileged public input type; both are assembled
/// from [`muffintin_prodbasis::CompiledAuxiliaryBasis`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuxiliaryKind {
    /// SPEX-style mixed product basis.
    MixedProduct,
    /// Interpolation-point / THC auxiliary assembled from sampled $\zeta$.
    InterpolationPoints,
    /// Explicit toy point-charge expansion, not production $\zeta$.
    PointChargeOracle,
}

/// Separated $q\to 0$ head metadata after SPEX `coulomb_sphaverage`.
///
/// The stored matrix is the finite body. The divergent head
/// $4\pi/|q|^2\,|\omega\rangle\langle\omega|$ is **not** inserted; $\omega$ is
/// [`Self::constant_coefficients`] and the prefactor is [`Self::head_prefactor`].
#[derive(Clone, Debug, PartialEq)]
pub struct GammaHead {
    /// Whether the $4\pi/3$ spherical-average term was subtracted from the body.
    pub spherical_average_subtracted: bool,
    /// $4\pi$ in Hartree atomic units. Consumers reconstruct $4\pi/|q|^2$ at finite $q$.
    pub head_prefactor: f64,
    /// Fourier coefficients of the cell-average/monopole channel, auxiliary order.
    pub constant_coefficients: Vec<Complex64>,
}

/// Finite spherical bare-Coulomb truncation attached to an assembled operator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpencerAlaviSphere {
    pub radius: Bohr,
    pub full_k_points: usize,
    pub reciprocal_cutoff: InverseBohr,
}

/// Hermitian Coulomb operator $V^q$ in the compiled auxiliary order.
///
/// Storage is full $n\times n$ row-major. SPEX stores packed $I\le J$; this type
/// owns the same physical matrix with the lower triangle filled by Hermitian
/// completion. Fields are private so a caller cannot forge a dimension that
/// panics on [`Self::apply`].
#[derive(Clone, Debug, PartialEq)]
pub struct CoulombOperator {
    pub(crate) layout: AuxiliaryLayout,
    pub(crate) cell: Cell,
    pub(crate) reciprocal: ReciprocalLattice,
    pub(crate) kind: AuxiliaryKind,
    pub(crate) matrix: Vec<Complex64>,
    pub(crate) gamma: Option<GammaHead>,
    pub(crate) spencer_alavi: Option<SpencerAlaviSphere>,
    pub(crate) provenance: Provenance,
}

impl CoulombOperator {
    /// Exact auxiliary layout ($q$, regions, split).
    pub const fn layout(&self) -> &AuxiliaryLayout {
        &self.layout
    }

    /// Canonical transfer $q$, including Umklapp.
    pub const fn q(&self) -> TransferQ {
        self.layout.q()
    }

    /// Direct cell used to assemble the operator.
    pub const fn cell(&self) -> &Cell {
        &self.cell
    }

    /// Reciprocal lattice used to assemble the operator.
    pub const fn reciprocal(&self) -> &ReciprocalLattice {
        &self.reciprocal
    }

    /// Total auxiliary dimension.
    pub fn dimension(&self) -> usize {
        self.layout.dimension()
    }

    /// Muffin-tin block length in the mixed-product / interpolation-point flatten.
    pub const fn mt_dimension(&self) -> usize {
        self.layout.mt_dimension()
    }

    /// Interstitial (or uniform interpolation-point) block length.
    pub const fn interstitial_dimension(&self) -> usize {
        self.layout.interstitial_dimension()
    }

    /// Combined regions copied from the compiled auxiliary.
    pub fn regions(&self) -> &[AuxiliaryRegion] {
        self.layout.regions()
    }

    /// Mixed-product, sampled-$\zeta$, or toy point-charge origin.
    pub const fn kind(&self) -> AuxiliaryKind {
        self.kind
    }

    /// Row-major Hermitian matrix, length `dimension()^2`.
    pub fn matrix(&self) -> &[Complex64] {
        &self.matrix
    }

    /// Element $V_{ij}$.
    pub fn element(&self, row: usize, column: usize) -> Result<Complex64, CoulombError> {
        let n = self.dimension();
        if row >= n {
            return Err(CoulombError::MatrixIndex {
                index: row,
                dimension: n,
            });
        }
        if column >= n {
            return Err(CoulombError::MatrixIndex {
                index: column,
                dimension: n,
            });
        }
        Ok(self.matrix[row * n + column])
    }

    /// Gamma-head metadata when $q=0$.
    pub const fn gamma(&self) -> Option<&GammaHead> {
        self.gamma.as_ref()
    }

    /// Spherical-truncation metadata; mutually exclusive with [`Self::gamma`].
    pub const fn spencer_alavi(&self) -> Option<&SpencerAlaviSphere> {
        self.spencer_alavi.as_ref()
    }

    /// Provenance of the assembly.
    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// Reject a pair vertex whose layout does not match this operator.
    pub fn require_vertex(&self, vertex: &PairVertex) -> Result<(), CoulombError> {
        if vertex.layout() == &self.layout {
            return Ok(());
        }
        if vertex.q() != self.q() {
            return Err(CoulombError::VertexTransferQ);
        }
        if vertex.mt_dimension() != self.mt_dimension()
            || vertex.interstitial_dimension() != self.interstitial_dimension()
        {
            return Err(CoulombError::VertexDimension {
                vertex_mt: vertex.mt_dimension(),
                vertex_interstitial: vertex.interstitial_dimension(),
                operator_mt: self.mt_dimension(),
                operator_interstitial: self.interstitial_dimension(),
            });
        }
        Err(CoulombError::VertexLayout)
    }

    /// $V c$ in auxiliary order.
    pub fn apply(&self, vertex: &PairVertex) -> Result<Vec<Complex64>, CoulombError> {
        self.require_vertex(vertex)?;
        let n = self.dimension();
        let coefficients = vertex.coefficients();
        let mut result = vec![Complex64::default(); n];
        for (row, slot) in result.iter_mut().enumerate() {
            let mut acc = Complex64::default();
            for (column, coefficient) in coefficients.iter().enumerate() {
                acc += self.matrix[row * n + column] * coefficient;
            }
            *slot = acc;
        }
        Ok(result)
    }

    /// $c_L^\dagger V c_R$.
    pub fn quadratic_form(
        &self,
        left: &PairVertex,
        right: &PairVertex,
    ) -> Result<Complex64, CoulombError> {
        self.require_vertex(left)?;
        self.require_vertex(right)?;
        let applied = self.apply(right)?;
        Ok(left
            .coefficients()
            .iter()
            .zip(applied)
            .map(|(coefficient, value)| coefficient.conj() * value)
            .sum())
    }

    /// Reusable contractor that keeps $V^q$ resident for many column blocks.
    pub fn contractor(&self) -> Result<CoulombVertexContractor<'_>, CoulombError> {
        CoulombVertexContractor::new(self)
    }
}

/// One resident $V^q$ prepared for batched $C_L^\dagger V C_R$ column blocks.
///
/// Building the tensor is $O(n^2)$ and independent of the columns, so a caller
/// that contracts many blocks against the same $q$ must construct this once and
/// reuse it. Each block is a single einsum on the shared tensor backend, not a
/// per-scalar matrix-vector product.
#[derive(Debug)]
pub struct CoulombVertexContractor<'a> {
    operator: &'a CoulombOperator,
    matrix: ComplexTensor,
}

impl<'a> CoulombVertexContractor<'a> {
    fn new(operator: &'a CoulombOperator) -> Result<Self, CoulombError> {
        let n = operator.dimension();
        let matrix = ComplexTensor::from_host_row_major(
            &[n, n],
            &[Axis::Auxiliary, Axis::Auxiliary],
            operator.matrix.clone(),
        )?;
        Ok(Self { operator, matrix })
    }

    /// Operator this contractor was built from.
    pub const fn operator(&self) -> &'a CoulombOperator {
        self.operator
    }

    /// $C_L^\dagger V C_R$, row-major with `left.len()` rows and `right.len()` columns.
    ///
    /// Entry $(i, j)$ is `left[i].coefficients()` conjugated against
    /// $V$ applied to `right[j].coefficients()`, identical to
    /// [`CoulombOperator::quadratic_form`] on that pair.
    pub fn quadratic_block(
        &self,
        left: &[&PairVertex],
        right: &[&PairVertex],
    ) -> Result<Vec<Complex64>, CoulombError> {
        if left.is_empty() || right.is_empty() {
            return Ok(Vec::new());
        }
        let left_block = self.column_block(left)?.conjugate();
        let right_block = self.column_block(right)?;
        let contracted = einsum("ai,ab,bj->ij", &[&left_block, &self.matrix, &right_block])?;
        Ok(contracted.to_host_row_major())
    }

    /// Weighted sum of equal-occupied quadratic blocks.
    ///
    /// `vertices` is occupied-major with `n_target` consecutive target
    /// columns per occupied state. The operator is applied to all columns in
    /// one dense contraction; only the equal-occupied target blocks are then
    /// accumulated. This avoids rebuilding and applying the same resident
    /// Coulomb tensor once per occupied state.
    pub fn weighted_occupied_quadratic_sum(
        &self,
        vertices: &[&PairVertex],
        occupied_weights: &[f64],
        n_target: usize,
    ) -> Result<Vec<Complex64>, CoulombError> {
        if occupied_weights.is_empty() || n_target == 0 {
            return Ok(Vec::new());
        }
        let expected = occupied_weights
            .len()
            .checked_mul(n_target)
            .ok_or(CoulombError::DimensionOverflow)?;
        if vertices.len() != expected {
            return Err(CoulombError::VertexBlockDimension {
                vertices: vertices.len(),
                occupied: occupied_weights.len(),
                targets: n_target,
            });
        }

        let columns = self.column_block(vertices)?;
        let applied = einsum("ab,bj->aj", &[&self.matrix, &columns])?.to_host_row_major();
        let n_auxiliary = self.operator.dimension();
        let n_columns = vertices.len();
        let mut result = vec![Complex64::default(); n_target * n_target];
        for auxiliary in 0..n_auxiliary {
            let applied_row = &applied[auxiliary * n_columns..(auxiliary + 1) * n_columns];
            for (occupied, &weight) in occupied_weights.iter().enumerate() {
                let base = occupied * n_target;
                for left_target in 0..n_target {
                    let left =
                        vertices[base + left_target].coefficients()[auxiliary].conj() * weight;
                    let output = &mut result[left_target * n_target..(left_target + 1) * n_target];
                    for right_target in 0..n_target {
                        output[right_target] += left * applied_row[base + right_target];
                    }
                }
            }
        }
        Ok(result)
    }

    /// Auxiliary-major matrix of the requested vertex coefficients.
    fn column_block(&self, vertices: &[&PairVertex]) -> Result<ComplexTensor, CoulombError> {
        let n = self.operator.dimension();
        let columns = vertices.len();
        let mut values = vec![Complex64::default(); n * columns];
        for (column, vertex) in vertices.iter().enumerate() {
            self.operator.require_vertex(vertex)?;
            for (row, coefficient) in vertex.coefficients().iter().enumerate() {
                values[row * columns + column] = *coefficient;
            }
        }
        Ok(ComplexTensor::from_host_row_major(
            &[n, columns],
            &[Axis::Auxiliary, Axis::PairColumn],
            values,
        )?)
    }
}
