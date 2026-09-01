//! Shared canonical-$q$ fold and regular-mesh $k-q$ lookup.
//!
//! Scalar and spinor product-input boundaries use the same primitive-cell fold, transfer
//! wrap, and off-mesh rejection. This module owns those semantics; it does
//! not forward a later Coulomb or MPB $q$.

use muffintin_core::{GVector, InverseBohr, ReciprocalLattice};
use muffintin_dft::g_vector;
use muffintin_prodbasis::TransferQ;

use crate::checkpoint_physics::CheckpointPhysicsError;

const MESH_COORD_TOLERANCE: f64 = 1.0e-12;

/// Failure while validating one canonical q record against its ordered k mesh.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CanonicalQMapError {
    NonFiniteQSlice,
    CanonicalQMismatch,
    IncompatibleMap,
    KMinusQWrap { k_index: usize },
    GammaTarget,
}

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
) -> Result<CanonicalTransfer, CheckpointPhysicsError> {
    if q_fractional.iter().any(|value| !value.is_finite()) {
        return Err(CheckpointPhysicsError::NonFiniteKPoint(q_fractional));
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
) -> Result<MeshKMinusQ, CheckpointPhysicsError> {
    let mut folded = [0.0; 3];
    for axis in 0..3 {
        folded[axis] = (k_frac[axis] - transfer.q_canonical[axis]).rem_euclid(1.0);
    }
    let kq_index = points
        .iter()
        .position(|point| coords_on_mesh(point, folded))
        .ok_or(CheckpointPhysicsError::OffMeshTransfer {
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

/// Validate one canonical q and its complete ordered k-minus-q map.
///
/// The q record at `q_index` must equal the matching ordered k point in
/// Cartesian reciprocal coordinates. Every map entry must retain its k index,
/// point inside the mesh, and satisfy the stored integer Umklapp relation.
pub(crate) fn validate_canonical_q_map(
    k_fractional: &[[f64; 3]],
    reciprocal: ReciprocalLattice,
    stored_q: [InverseBohr; 3],
    q_index: usize,
    maps: impl IntoIterator<Item = (usize, usize, [i32; 3])>,
) -> Result<(), CanonicalQMapError> {
    let n_k = k_fractional.len();
    let q_canonical = *k_fractional
        .get(q_index)
        .ok_or(CanonicalQMapError::IncompatibleMap)?;
    let expected_q = fractional_to_reciprocal(q_canonical, reciprocal.basis());
    if stored_q
        .iter()
        .chain(&expected_q)
        .any(|component| !component.get().is_finite())
        || q_canonical.iter().any(|component| !component.is_finite())
    {
        return Err(CanonicalQMapError::NonFiniteQSlice);
    }
    if stored_q
        .iter()
        .zip(expected_q)
        .any(|(stored, expected)| !scale_aware_eq(stored.get(), expected.get()))
    {
        return Err(CanonicalQMapError::CanonicalQMismatch);
    }

    let mut actual = 0;
    let mut q_target = None;
    for (k_index, (stored_k_index, kq_index, umklapp)) in maps.into_iter().enumerate() {
        actual += 1;
        if k_index >= n_k || stored_k_index != k_index || kq_index >= n_k {
            return Err(CanonicalQMapError::IncompatibleMap);
        }
        let k = k_fractional[k_index];
        let kq = k_fractional[kq_index];
        for axis in 0..3 {
            let residual = k[axis] - q_canonical[axis] - kq[axis] - f64::from(umklapp[axis]);
            if !k[axis].is_finite()
                || !kq[axis].is_finite()
                || !residual.is_finite()
                || !scale_aware_eq(residual, 0.0)
            {
                return Err(if !k[axis].is_finite() || !kq[axis].is_finite() {
                    CanonicalQMapError::NonFiniteQSlice
                } else {
                    CanonicalQMapError::KMinusQWrap { k_index }
                });
            }
        }
        if k_index == q_index {
            q_target = Some(kq_index);
        }
    }
    if actual != n_k {
        return Err(CanonicalQMapError::IncompatibleMap);
    }
    let q_target = q_target.ok_or(CanonicalQMapError::IncompatibleMap)?;
    if k_fractional[q_target]
        .iter()
        .any(|component| component.abs() > MESH_COORD_TOLERANCE)
    {
        return Err(CanonicalQMapError::GammaTarget);
    }
    Ok(())
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

fn scale_aware_eq(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= MESH_COORD_TOLERANCE * scale
}
