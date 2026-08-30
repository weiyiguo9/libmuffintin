//! Shared APW/LO site-coordinate identity used by scalar MPB and THC.

use crate::scalar_product::{SCALAR_RADIAL_LO0, SCALAR_RADIAL_U, SCALAR_RADIAL_UDOT};
use muffintin_auxiliary_ir::{ProductOrbitalKind, ProductRadialId};
use muffintin_core::lm_from_index;
use muffintin_operators::lapw::{CompiledBasis, LocalOrbitalLayout};

/// Map a scalar site-projection row to its [`ProductRadialId`] and $m$.
///
/// Rows are APW $u$ ($n=0$) and $\dot u$ ($n=1$) in contiguous $\mathrm{lm}$
/// order, then this site's local orbitals in `(l, m, ordinal)` order with
/// $n=2+\mathrm{ordinal}$.
pub(crate) fn site_coordinate(
    compiled: &CompiledBasis,
    site: usize,
    spin: u8,
    coord: usize,
) -> Option<(ProductRadialId, i32)> {
    let n_lm = compiled
        .site_augmentations
        .get(site)
        .and_then(|waves| waves.first())
        .map_or(0, |wave| wave.coefficients.len());
    if coord < 2 * n_lm {
        let lm = lm_from_index(coord / 2);
        let n = if coord % 2 == 0 {
            SCALAR_RADIAL_U
        } else {
            SCALAR_RADIAL_UDOT
        };
        return Some((
            ProductRadialId {
                site,
                kind: ProductOrbitalKind::Valence,
                l: lm.l,
                n,
                spin,
            },
            lm.m,
        ));
    }
    let layout = compiled.layout.site_layout(site)?;
    let (l, m, ordinal) = lo_quantum_numbers(layout, coord - 2 * n_lm)?;
    Some((
        ProductRadialId {
            site,
            kind: ProductOrbitalKind::Valence,
            l,
            n: SCALAR_RADIAL_LO0 + ordinal,
            spin,
        },
        m,
    ))
}

pub(crate) fn lo_quantum_numbers(
    layout: &LocalOrbitalLayout,
    local_index: usize,
) -> Option<(u32, i32, usize)> {
    let mut remaining = local_index;
    for (l, &count) in layout.counts_by_l().iter().enumerate() {
        if count == 0 {
            continue;
        }
        let stride = (2 * l + 1) * count;
        if remaining < stride {
            let m_block = remaining / count;
            let n = remaining % count;
            return Some((l as u32, m_block as i32 - l as i32, n));
        }
        remaining -= stride;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{lo_quantum_numbers, site_coordinate};
    use crate::scalar_product::SCALAR_RADIAL_LO0;
    use muffintin_core::Bohr;
    use muffintin_operators::lapw::{
        ApwSiteGeometry, BasisLayout, CompiledBasis, LocalOrbitalLayout, Provenance,
    };

    #[test]
    fn lo_inverse_mapping_preserves_l_m_ordinal() {
        let layout = LocalOrbitalLayout::new(vec![1, 2]);
        assert_eq!(lo_quantum_numbers(&layout, 0), Some((0, 0, 0)));
        assert_eq!(lo_quantum_numbers(&layout, 1), Some((1, -1, 0)));
        assert_eq!(lo_quantum_numbers(&layout, 2), Some((1, -1, 1)));
        assert_eq!(lo_quantum_numbers(&layout, 3), Some((1, 0, 0)));
        assert_eq!(lo_quantum_numbers(&layout, 6), Some((1, 1, 1)));
        assert_eq!(lo_quantum_numbers(&layout, 7), None);
    }

    #[test]
    fn site_coordinate_maps_lo_after_apw_u_and_udot() {
        let compiled = CompiledBasis {
            layout: BasisLayout::new(0, vec![LocalOrbitalLayout::new(vec![1, 1])]),
            plane_waves: Vec::new(),
            site_augmentations: vec![Vec::new()],
            site_geometry: vec![ApwSiteGeometry {
                position: [Bohr(0.0); 3],
                radius: Bohr(1.0),
            }],
            provenance: Provenance::default(),
        };
        let (id, m) = site_coordinate(&compiled, 0, 1, 0).expect("LO row");
        assert_eq!(id.site, 0);
        assert_eq!(id.spin, 1);
        assert_eq!(id.l, 0);
        assert_eq!(id.n, SCALAR_RADIAL_LO0);
        assert_eq!(m, 0);
        let (id, m) = site_coordinate(&compiled, 0, 1, 1).expect("p LO");
        assert_eq!(id.l, 1);
        assert_eq!(id.n, SCALAR_RADIAL_LO0);
        assert_eq!(m, -1);
    }
}
