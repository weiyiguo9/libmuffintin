//! Canonical-q k-mesh and the Umklapp map `kminus`.

use crate::ThcError;
use muffintin_auxiliary_ir::TransferQ;
use muffintin_core::InverseBohr;
use std::f64::consts::PI;

/// Gamma-centred Monkhorst–Pack-style mesh used by the toy k-point tests.
///
/// Fields are private so the tensor-product occupancy of `gamma_centred`
/// cannot be bypassed. Indexing is `i` slowest, `k` fastest:
///
/// ```math
/// \mathrm{index}((i,j,k)) = ((i N_j + j) N_k + k).
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct KMesh {
    fractional: Vec<[f64; 3]>,
    divisions: [u32; 3],
    lattice_constant: f64,
}

impl KMesh {
    /// Build a Gamma-centred mesh with fractional coordinates
    /// $n_i/N_i$ for $n_i=0,\ldots,N_i-1$.
    pub fn gamma_centred(divisions: [u32; 3], lattice_constant: f64) -> Result<Self, ThcError> {
        if divisions.contains(&0) {
            return Err(ThcError::InvalidKMeshDivisions(divisions));
        }
        if !lattice_constant.is_finite() || lattice_constant <= 0.0 {
            return Err(ThcError::InvalidLattice(lattice_constant));
        }
        let mut fractional = Vec::new();
        for i in 0..divisions[0] {
            for j in 0..divisions[1] {
                for k in 0..divisions[2] {
                    fractional.push([
                        f64::from(i) / f64::from(divisions[0]),
                        f64::from(j) / f64::from(divisions[1]),
                        f64::from(k) / f64::from(divisions[2]),
                    ]);
                }
            }
        }
        Ok(Self {
            fractional,
            divisions,
            lattice_constant,
        })
    }

    /// Number of k/q points. Equal to $N_0 N_1 N_2$ after [`Self::gamma_centred`].
    pub fn len(&self) -> usize {
        self.fractional.len()
    }

    /// Whether the mesh has no points. Unreachable for a validated
    /// [`Self::gamma_centred`] mesh because every division is positive.
    pub fn is_empty(&self) -> bool {
        self.fractional.is_empty()
    }

    /// Fractional coordinates, one row per k, in generation order.
    pub fn fractional(&self) -> &[[f64; 3]] {
        &self.fractional
    }

    /// Divisions along each reciprocal axis.
    pub fn divisions(&self) -> [u32; 3] {
        self.divisions
    }

    /// Cubic lattice constant in Bohr.
    pub fn lattice_constant(&self) -> f64 {
        self.lattice_constant
    }

    /// Canonical transfer $q$ for mesh index `iq` (zero Umklapp; $q$ is on-mesh).
    pub fn transfer_q(&self, iq: usize) -> Result<TransferQ, ThcError> {
        let frac = self.fractional.get(iq).ok_or(ThcError::KMeshIndex {
            index: iq,
            count: self.len(),
        })?;
        let scale = 2.0 * PI / self.lattice_constant;
        TransferQ::from_cartesian(std::array::from_fn(|axis| InverseBohr(scale * frac[axis])))
            .map_err(|_| ThcError::InvalidLattice(self.lattice_constant))
    }

    /// Folded $k-q$ mesh index and integer reciprocal wrap.
    ///
    /// Matches `thc_mt_kpoint_test.py:134-139` and
    /// `thc_lapw_end_to_end_test.py:285-291`:
    /// `index, Gwrap` with `Gwrap = rint((k-q) - k_index)`.
    pub fn kminus(&self, ik: usize, iq: usize) -> Result<(usize, [i32; 3]), ThcError> {
        if ik >= self.len() {
            return Err(ThcError::KMeshIndex {
                index: ik,
                count: self.len(),
            });
        }
        if iq >= self.len() {
            return Err(ThcError::KMeshIndex {
                index: iq,
                count: self.len(),
            });
        }
        let unwrapped: [f64; 3] =
            std::array::from_fn(|axis| self.fractional[ik][axis] - self.fractional[iq][axis]);
        let key: [i32; 3] = std::array::from_fn(|axis| {
            let n = self.divisions[axis] as i32;
            let rounded = (unwrapped[axis] * f64::from(self.divisions[axis])).round() as i32;
            rounded.rem_euclid(n)
        });
        // `gamma_centred` enumerates i (slowest), j, k (fastest) over
        // `0..N_axis`. `key[axis]` is that same integer from rem_euclid, so
        // this index is always in `0..len` for a mesh built by that
        // constructor. Fields are private, so the occupancy cannot be
        // bypassed.
        let index = ((key[0] as usize) * self.divisions[1] as usize + key[1] as usize)
            * self.divisions[2] as usize
            + key[2] as usize;
        debug_assert!(
            index < self.len(),
            "folded key {key:?} maps outside a validated gamma-centred mesh"
        );
        let reciprocal_shift: [i32; 3] = std::array::from_fn(|axis| {
            (unwrapped[axis] - self.fractional[index][axis]).round() as i32
        });
        Ok((index, reciprocal_shift))
    }

    /// Whether mesh index `iq` is the Gamma point.
    pub fn is_gamma(&self, iq: usize) -> bool {
        self.fractional
            .get(iq)
            .is_some_and(|frac| frac.iter().all(|component| component.abs() < 1.0e-15))
    }
}

/// Positive Umklapp phase $\exp(+i G_{\mathrm{wrap}}\cdot r)$ in the canonical-$q$
/// gauge. $G_{\mathrm{wrap}}$ is in reciprocal-lattice units, so the Cartesian
/// wavevector is $2\pi G/a$ on the cubic toy lattice.
pub fn umklapp_phase(
    point: [f64; 3],
    shift: [i32; 3],
    lattice_constant: f64,
) -> num_complex::Complex64 {
    let argument = (2.0 * PI / lattice_constant)
        * (point[0] * f64::from(shift[0])
            + point[1] * f64::from(shift[1])
            + point[2] * f64::from(shift[2]));
    num_complex::Complex64::from_polar(1.0, argument)
}
