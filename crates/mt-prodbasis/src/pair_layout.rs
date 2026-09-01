//! Neutral $(k,i,j)$ pair-column flattening used before THC selection.

use crate::AuxiliaryIrError;

/// One side of a rectangular exchange pair.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExchangeSpace {
    Valence,
    Core,
}

/// Stable rectangular $(k,i,j)$ exchange-pair flattening.
///
/// `i` belongs to `occupied_space` at $k-q$, `j` belongs to
/// `target_space` at $k$, and the column is
/// $(k N_{\mathrm{occupied}} + i)N_{\mathrm{target}} + j$.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExchangePairLayout {
    pub occupied_space: ExchangeSpace,
    pub target_space: ExchangeSpace,
    pub n_k: usize,
    pub n_occupied: usize,
    pub n_target: usize,
}

impl ExchangePairLayout {
    pub const fn new(
        occupied_space: ExchangeSpace,
        target_space: ExchangeSpace,
        n_k: usize,
        n_occupied: usize,
        n_target: usize,
    ) -> Self {
        Self {
            occupied_space,
            target_space,
            n_k,
            n_occupied,
            n_target,
        }
    }

    /// Number of rectangular pair columns per q.
    pub fn n_columns(&self) -> Result<usize, AuxiliaryIrError> {
        checked_layout_len(&[self.n_k, self.n_occupied, self.n_target])
    }

    /// Checked column index of $(k,i,j)$.
    pub fn encode(
        &self,
        k: usize,
        occupied: usize,
        target: usize,
    ) -> Result<usize, AuxiliaryIrError> {
        if k >= self.n_k || occupied >= self.n_occupied || target >= self.n_target {
            return Err(AuxiliaryIrError::ExchangePairCoordinate {
                k,
                occupied,
                target,
                n_k: self.n_k,
                n_occupied: self.n_occupied,
                n_target: self.n_target,
            });
        }
        k.checked_mul(self.n_occupied)
            .and_then(|value| value.checked_add(occupied))
            .and_then(|value| value.checked_mul(self.n_target))
            .and_then(|value| value.checked_add(target))
            .ok_or_else(|| AuxiliaryIrError::DimensionOverflow {
                dimensions: vec![self.n_k, self.n_occupied, self.n_target],
            })
    }

    /// Checked inverse of [`Self::encode`].
    pub fn decode(&self, column: usize) -> Result<(usize, usize, usize), AuxiliaryIrError> {
        let n_columns = self.n_columns()?;
        if column >= n_columns {
            return Err(AuxiliaryIrError::ExchangePairColumn { column, n_columns });
        }
        let block = self.n_occupied.checked_mul(self.n_target).ok_or_else(|| {
            AuxiliaryIrError::DimensionOverflow {
                dimensions: vec![self.n_k, self.n_occupied, self.n_target],
            }
        })?;
        let k = column / block;
        let rem = column % block;
        Ok((k, rem / self.n_target, rem % self.n_target))
    }
}

/// Stable $(k,i,j)$ flattening of pair columns.
///
/// Column index is $k\cdot N_{\mathrm{orb}}^2 + i\cdot N_{\mathrm{orb}} + j$.
/// This is method-neutral orbital-pair ordering, not a THC selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PairColumnLayout {
    pub n_k: usize,
    pub n_orb: usize,
    /// Orbital index treated as the sharp core channel, if the fixture has one.
    pub core_orbital: Option<usize>,
}

impl PairColumnLayout {
    /// Layout for `n_k` k-points and `n_orb` orbitals.
    pub const fn new(n_k: usize, n_orb: usize, core_orbital: Option<usize>) -> Self {
        Self {
            n_k,
            n_orb,
            core_orbital,
        }
    }

    /// Number of pair columns per q, $N_k N_{\mathrm{orb}}^2$.
    pub fn n_columns(&self) -> Result<usize, AuxiliaryIrError> {
        checked_layout_len(&[self.n_k, self.n_orb, self.n_orb])
    }

    /// Column index of $(k,i,j)$.
    pub fn encode(&self, k: usize, i: usize, j: usize) -> usize {
        k * self.n_orb * self.n_orb + i * self.n_orb + j
    }

    /// Inverse of [`Self::encode`].
    pub fn decode(&self, column: usize) -> (usize, usize, usize) {
        let block = self.n_orb * self.n_orb;
        let k = column / block;
        let rem = column % block;
        (k, rem / self.n_orb, rem % self.n_orb)
    }

    /// Whether the column involves the core orbital.
    pub fn is_core(&self, column: usize) -> bool {
        let Some(core) = self.core_orbital else {
            return false;
        };
        let (_, i, j) = self.decode(column);
        i == core || j == core
    }

    /// Whether the column is valence-only.
    pub fn is_valence(&self, column: usize) -> bool {
        self.core_orbital.is_some() && !self.is_core(column)
    }

    /// Reject a core index that cannot appear in any pair column.
    pub fn require_core_orbital(&self) -> Result<(), AuxiliaryIrError> {
        if let Some(core) = self.core_orbital
            && core >= self.n_orb
        {
            return Err(AuxiliaryIrError::InvalidCoreOrbital {
                index: core,
                n_orb: self.n_orb,
            });
        }
        Ok(())
    }
}

fn checked_layout_len(dimensions: &[usize]) -> Result<usize, AuxiliaryIrError> {
    dimensions.iter().try_fold(1_usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| AuxiliaryIrError::DimensionOverflow {
                dimensions: dimensions.to_vec(),
            })
    })
}

#[cfg(test)]
mod tests {
    use super::{ExchangePairLayout, ExchangeSpace, PairColumnLayout};
    use crate::AuxiliaryIrError;

    #[test]
    fn encode_is_k_major_then_i_then_j() {
        let layout = PairColumnLayout::new(3, 4, None);
        assert_eq!(layout.encode(2, 1, 3), 2 * 16 + 4 + 3);
        assert_eq!(layout.decode(layout.encode(2, 1, 3)), (2, 1, 3));
        assert_eq!(layout.n_columns().unwrap(), 48);
        assert!(!layout.is_core(0));
        assert!(!layout.is_valence(0));
    }

    #[test]
    fn n_columns_reports_dimension_overflow() {
        let error = PairColumnLayout::new(usize::MAX, 4, None)
            .n_columns()
            .unwrap_err();
        assert!(matches!(
            error,
            AuxiliaryIrError::DimensionOverflow { ref dimensions } if dimensions == &[usize::MAX, 4, 4]
        ));
    }

    #[test]
    fn core_orbital_must_lie_in_the_orbital_range() {
        let layout = PairColumnLayout::new(1, 2, Some(2));
        assert!(matches!(
            layout.require_core_orbital(),
            Err(AuxiliaryIrError::InvalidCoreOrbital { index: 2, n_orb: 2 })
        ));
        let ok = PairColumnLayout::new(1, 2, Some(1));
        ok.require_core_orbital().unwrap();
        assert!(ok.is_core(ok.encode(0, 1, 0)));
        assert!(ok.is_valence(ok.encode(0, 0, 0)));
    }

    #[test]
    fn rectangular_exchange_layout_checks_roundtrip_bounds_and_overflow() {
        let layout = ExchangePairLayout::new(ExchangeSpace::Core, ExchangeSpace::Valence, 3, 2, 5);
        let column = layout.encode(2, 1, 4).unwrap();
        assert_eq!(column, (2 * 2 + 1) * 5 + 4);
        assert_eq!(layout.decode(column).unwrap(), (2, 1, 4));
        assert_eq!(layout.n_columns().unwrap(), 30);
        assert!(matches!(
            layout.encode(3, 0, 0),
            Err(AuxiliaryIrError::ExchangePairCoordinate { .. })
        ));
        assert!(matches!(
            layout.decode(30),
            Err(AuxiliaryIrError::ExchangePairColumn {
                column: 30,
                n_columns: 30
            })
        ));
        assert!(matches!(
            ExchangePairLayout::new(ExchangeSpace::Core, ExchangeSpace::Core, usize::MAX, 2, 2,)
                .n_columns(),
            Err(AuxiliaryIrError::DimensionOverflow { .. })
        ));
    }
}
