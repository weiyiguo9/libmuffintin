//! Assembly request: direct cell, Weinert `LEXP`, and interpolation projection.

use crate::CoulombError;
use muffintin_core::{InverseBohr, ReciprocalLattice};
use muffintin_grid::Cell;

/// Default SPEX-style Weinert expansion cutoff used by the toy assembler.
pub const DEFAULT_LEXP: u32 = 4;

/// Projection cutoffs used when expanding sampled $\zeta$ (or the toy
/// point-charge oracle) into the Weinert charge expansion.
///
/// Production interpolation assembly uses [`crate::SampledAuxiliaryFunctions`],
/// not a delta at each interpolation node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterpolationProjection {
    /// Interstitial $|q+G|$ membership cutoff, SPEX `rdum <= gcutm**2`.
    pub pw_cutoff: InverseBohr,
    /// Angular expansion of a muffin-tin interpolation point, $L\le l_{\max}$.
    pub l_max: u32,
}

impl InterpolationProjection {
    /// Construct after checking cutoff and angular momentum.
    pub fn new(pw_cutoff: InverseBohr, l_max: u32) -> Result<Self, CoulombError> {
        if !pw_cutoff.get().is_finite() || pw_cutoff.get() < 0.0 {
            return Err(CoulombError::InvalidPwCutoff(pw_cutoff.get()));
        }
        if l_max > 12 {
            return Err(CoulombError::InvalidLexp(l_max));
        }
        Ok(Self { pw_cutoff, l_max })
    }
}

/// Finite-$q$ Coulomb assembly request.
///
/// The public assembler consumes [`muffintin_product::CompiledAuxiliaryBasis`].
/// Mixed-product auxiliaries use [`crate::assemble_coulomb`]. Interpolation-point
/// auxiliaries use [`crate::assemble_sampled_coulomb`] with sampled $\zeta$.
/// [`InterpolationProjection`] supplies the Weinert $L$ and $|q+G|$ cutoffs
/// of that expansion.
#[derive(Clone, Debug, PartialEq)]
pub struct CoulombRequest {
    cell: Cell,
    reciprocal: ReciprocalLattice,
    lexp: u32,
    interpolation: Option<InterpolationProjection>,
}

impl CoulombRequest {
    /// Build from a validated direct cell. Reciprocal vectors follow
    /// $a_i\cdot b_j = 2\pi\delta_{ij}$.
    pub fn new(cell: Cell, lexp: u32) -> Result<Self, CoulombError> {
        if lexp > 12 {
            return Err(CoulombError::InvalidLexp(lexp));
        }
        let reciprocal = ReciprocalLattice::from_direct(*cell.basis())?;
        Ok(Self {
            cell,
            reciprocal,
            lexp,
            interpolation: None,
        })
    }

    /// Cubic cell of side `lattice` Bohr and the default `LEXP`.
    pub fn cubic(lattice: f64, lexp: u32) -> Result<Self, CoulombError> {
        let cell = Cell::new([
            [
                muffintin_core::Bohr(lattice),
                muffintin_core::Bohr(0.0),
                muffintin_core::Bohr(0.0),
            ],
            [
                muffintin_core::Bohr(0.0),
                muffintin_core::Bohr(lattice),
                muffintin_core::Bohr(0.0),
            ],
            [
                muffintin_core::Bohr(0.0),
                muffintin_core::Bohr(0.0),
                muffintin_core::Bohr(lattice),
            ],
        ])?;
        Self::new(cell, lexp)
    }

    /// Attach the interpolation-point projection used when the auxiliary is
    /// [`muffintin_product::AuxiliaryRepresentation::InterpolationPoints`].
    pub fn with_interpolation(
        mut self,
        projection: InterpolationProjection,
    ) -> Result<Self, CoulombError> {
        if projection.l_max > self.lexp {
            return Err(CoulombError::InterpolationLmax {
                l_max: projection.l_max,
                lexp: self.lexp,
            });
        }
        self.interpolation = Some(projection);
        Ok(self)
    }

    /// Direct-lattice cell.
    pub const fn cell(&self) -> &Cell {
        &self.cell
    }

    /// Reciprocal lattice matching [`Self::cell`].
    pub const fn reciprocal(&self) -> &ReciprocalLattice {
        &self.reciprocal
    }

    /// Weinert / SPEX `LEXP`.
    pub const fn lexp(&self) -> u32 {
        self.lexp
    }

    /// Interpolation projection, if configured.
    pub const fn interpolation(&self) -> Option<InterpolationProjection> {
        self.interpolation
    }
}
