//! Finite-$q$ Coulomb operator over a compiled auxiliary basis.

use crate::CoulombError;
use libmuffintin_basis::Provenance;
use libmuffintin_core::ReciprocalLattice;
use libmuffintin_grid::Cell;
use libmuffintin_product::{AuxiliaryLayout, AuxiliaryRegion, PairVertex, TransferQ};
use num_complex::Complex64;

/// Typed auxiliary representation stored with $V^q$. Neither mixed-product nor
/// interpolation points is a privileged public input type; both are assembled
/// from [`libmuffintin_product::CompiledAuxiliaryBasis`].
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

    /// Muffin-tin block length in the M-H / M-I flatten.
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
}
