//! Assembly request: direct cell, Weinert `LEXP`, and interpolation projection.

use crate::CoulombError;
use muffintin_core::Cell;
use muffintin_core::{Bohr, InverseBohr, ReciprocalLattice};

/// SPEX default floor; production callers also require at least twice the MPB L cutoff.
pub const DEFAULT_LEXP: u32 = 14;

/// Supported Rayleigh cutoff; structure harmonics extend through twice this value.
/// At this bound the harmonic normalization and `gmat` factorial products fit in f64.
pub const MAX_LEXP: u32 = 32;

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

/// Real-space boundary condition of the bare Coulomb kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CoulombKernel {
    /// Periodic SPEX/Weinert body with a separated Gamma head.
    PeriodicWeinert,
    /// VASP `HFRCUT=-1`: spherical Spencer–Alavi truncation with an automatic
    /// radius and an explicit reciprocal representation cutoff.
    SpencerAlaviSphere {
        full_k_points: usize,
        reciprocal_cutoff: InverseBohr,
    },
    /// Gaussian-smoothed spherical boundary, retaining the Weinert short range.
    /// `smoothing` is omega; the sharp sphere is approached as omega increases.
    SmoothedSpencerAlaviSphere {
        full_k_points: usize,
        reciprocal_cutoff: InverseBohr,
        smoothing: InverseBohr,
    },
}

impl InterpolationProjection {
    /// Construct after checking cutoff and angular momentum.
    pub fn new(pw_cutoff: InverseBohr, l_max: u32) -> Result<Self, CoulombError> {
        if !pw_cutoff.get().is_finite() || pw_cutoff.get() < 0.0 {
            return Err(CoulombError::InvalidPwCutoff(pw_cutoff.get()));
        }
        if l_max > MAX_LEXP {
            return Err(CoulombError::InvalidLexp(l_max));
        }
        Ok(Self { pw_cutoff, l_max })
    }
}

/// Finite-$q$ Coulomb assembly request.
///
/// The public assembler consumes [`muffintin_prodbasis::CompiledAuxiliaryBasis`].
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
    kernel: CoulombKernel,
}

impl CoulombRequest {
    /// Build from a validated direct cell. Reciprocal vectors follow
    /// $a_i\cdot b_j = 2\pi\delta_{ij}$.
    pub fn new(cell: Cell, lexp: u32) -> Result<Self, CoulombError> {
        if lexp > MAX_LEXP {
            return Err(CoulombError::InvalidLexp(lexp));
        }
        let reciprocal = ReciprocalLattice::from_direct(*cell.basis())?;
        Ok(Self {
            cell,
            reciprocal,
            lexp,
            interpolation: None,
            kernel: CoulombKernel::PeriodicWeinert,
        })
    }

    /// Cubic cell of side `lattice` Bohr and an explicit `LEXP`.
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
    /// [`muffintin_prodbasis::AuxiliaryRepresentation::InterpolationPoints`].
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

    /// Select the automatic Spencer–Alavi sphere used by VASP `HFRCUT=-1`.
    pub fn with_spencer_alavi_sphere(
        mut self,
        full_k_points: usize,
        reciprocal_cutoff: InverseBohr,
    ) -> Result<Self, CoulombError> {
        if full_k_points == 0 {
            return Err(CoulombError::InvalidTruncationKPointCount(full_k_points));
        }
        if !reciprocal_cutoff.get().is_finite() || reciprocal_cutoff.get() <= 0.0 {
            return Err(CoulombError::InvalidTruncationReciprocalCutoff(
                reciprocal_cutoff.get(),
            ));
        }
        self.kernel = CoulombKernel::SpencerAlaviSphere {
            full_k_points,
            reciprocal_cutoff,
        };
        Ok(self)
    }

    /// Select a dual-space smoothed sphere, not VASP's sharp `HFRCUT=-1`.
    /// The Fourier cutoff limits only the Gaussian-damped boundary correction;
    /// the full periodic Weinert metric retains compact MT charge products.
    pub fn with_smoothed_spencer_alavi_sphere(
        mut self,
        full_k_points: usize,
        reciprocal_cutoff: InverseBohr,
        smoothing: InverseBohr,
    ) -> Result<Self, CoulombError> {
        if full_k_points == 0 {
            return Err(CoulombError::InvalidTruncationKPointCount(full_k_points));
        }
        if !reciprocal_cutoff.get().is_finite() || reciprocal_cutoff.get() <= 0.0 {
            return Err(CoulombError::InvalidTruncationReciprocalCutoff(
                reciprocal_cutoff.get(),
            ));
        }
        if !smoothing.get().is_finite() || smoothing.get() <= 0.0 {
            return Err(CoulombError::InvalidTruncationSmoothing(smoothing.get()));
        }
        self.kernel = CoulombKernel::SmoothedSpencerAlaviSphere {
            full_k_points,
            reciprocal_cutoff,
            smoothing,
        };
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

    pub const fn kernel(&self) -> CoulombKernel {
        self.kernel
    }

    /// Automatic truncation radius when the Spencer–Alavi kernel is selected.
    pub fn spencer_alavi_radius(&self) -> Option<Bohr> {
        let full_k_points = match self.kernel {
            CoulombKernel::PeriodicWeinert => return None,
            CoulombKernel::SpencerAlaviSphere { full_k_points, .. }
            | CoulombKernel::SmoothedSpencerAlaviSphere { full_k_points, .. } => full_k_points,
        };
        Some(Bohr(
            (3.0 * full_k_points as f64 * self.cell.volume().get() / (4.0 * std::f64::consts::PI))
                .cbrt(),
        ))
    }
}
