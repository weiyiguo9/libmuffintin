//! Shared canonical-$q$ fold and regular-mesh $k-q$ lookup.
//!
//! Scalar and spinor product-input boundaries use the same primitive-cell fold, transfer
//! wrap, and off-mesh rejection. This module owns those semantics; it does
//! not forward a later Coulomb or MPB $q$.

use muffintin_prodbasis::TransferQ;
use muffintin_core::{GVector, InverseBohr, ReciprocalLattice};

use crate::snapshot_dft::{SnapshotDftError, g_vector};

const MESH_COORD_TOLERANCE: f64 = 1.0e-12;

/// Requested $q_{\mathrm{in}}$, folded $q_{\mathrm{canonical}}$, and `TransferQ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CanonicalTransfer {
    pub q: TransferQ,
    pub q_in: [f64; 3],
    pub q_canonical: [f64; 3],
}

/// One regular-mesh $k \to k-q_{\mathrm{canonical}}$ record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MeshKMinusQ {
    pub k_index: usize,
    pub kq_index: usize,
    pub umklapp: GVector,
}

/// Fold $q_{\mathrm{in}}$ into $[0,1)^3$ and store $G_{\mathrm{transfer}}$.
pub(crate) fn canonical_transfer_q(
    q_fractional: [f64; 3],
    reciprocal: ReciprocalLattice,
) -> Result<CanonicalTransfer, SnapshotDftError> {
    if q_fractional.iter().any(|value| !value.is_finite()) {
        return Err(SnapshotDftError::NonFiniteKPoint(q_fractional));
    }
    let (q_canonical, q_wrap) = fold_to_unit_cell(q_fractional);
    let q_input = fractional_to_reciprocal(q_fractional, reciprocal.basis());
    let transfer_umklapp = g_vector(reciprocal, q_wrap);
    let q = TransferQ::fold_by_reciprocal_vector(q_input, transfer_umklapp)?;
    Ok(CanonicalTransfer {
        q,
        q_in: q_fractional,
        q_canonical,
    })
}

/// Map $k-q_{\mathrm{canonical}}$ onto an existing mesh point, or reject.
///
/// The wrap satisfies
/// $k_{\mathrm{frac}}-q_{\mathrm{canonical,frac}}=(k-q)_{\mathrm{frac}}+G_{\mathrm{wrap}}$.
/// Pair phases use $\exp(+i G_{\mathrm{wrap}}\cdot r)$. This wrap is not
/// [`TransferQ::umklapp`].
pub(crate) fn map_k_minus_q(
    k_index: usize,
    k_frac: [f64; 3],
    transfer: CanonicalTransfer,
    points: &[[f64; 3]],
    reciprocal: ReciprocalLattice,
) -> Result<MeshKMinusQ, SnapshotDftError> {
    let mut folded = [0.0; 3];
    for axis in 0..3 {
        folded[axis] = (k_frac[axis] - transfer.q_canonical[axis]).rem_euclid(1.0);
    }
    let kq_index = points
        .iter()
        .position(|point| coords_on_mesh(point, folded))
        .ok_or(SnapshotDftError::OffMeshTransfer {
            k: k_frac,
            q_in: transfer.q_in,
            q_canonical: transfer.q_canonical,
            folded,
        })?;
    let actual = points[kq_index];
    let wrap = std::array::from_fn(|axis| {
        (k_frac[axis] - transfer.q_canonical[axis] - actual[axis]).round() as i32
    });
    Ok(MeshKMinusQ {
        k_index,
        kq_index,
        umklapp: g_vector(reciprocal, wrap),
    })
}

fn fold_to_unit_cell(fractional: [f64; 3]) -> ([f64; 3], [i32; 3]) {
    let mut folded = [0.0; 3];
    let mut wrap = [0; 3];
    for axis in 0..3 {
        let value = fractional[axis];
        let unit = value.rem_euclid(1.0);
        wrap[axis] = (value - unit).round() as i32;
        folded[axis] = unit;
    }
    (folded, wrap)
}

fn coords_on_mesh(point: &[f64; 3], folded: [f64; 3]) -> bool {
    point
        .iter()
        .zip(folded)
        .all(|(&actual, expected)| (actual - expected).abs() <= MESH_COORD_TOLERANCE)
}

fn fractional_to_reciprocal(
    fractional: [f64; 3],
    reciprocal: &[[InverseBohr; 3]; 3],
) -> [InverseBohr; 3] {
    std::array::from_fn(|axis| {
        InverseBohr(
            fractional
                .iter()
                .zip(reciprocal)
                .map(|(&coefficient, vector)| coefficient * vector[axis].get())
                .sum(),
        )
    })
}
