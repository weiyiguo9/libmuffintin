//! Per-q orbital-pair collocation in the canonical-q / Umklapp gauge.

use crate::thc::ThcError;
use crate::thc::error::checked_storage_len;
use crate::{ExchangePairLayout, PairColumnLayout};
use num_complex::Complex64;

/// Pair-density block at one canonical $q$: `n_points × n_columns`, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct PairBlock {
    pub q_index: usize,
    pub n_points: usize,
    pub layout: PairColumnLayout,
    n_columns: usize,
    values: Vec<Complex64>,
}

/// Rectangular exchange-pair collocation at one canonical $q$.
///
/// Values are row-major `n_points × layout.n_columns()`. Unlike
/// [`PairBlock`], this block names the occupied and target spaces explicitly
/// and never interprets [`PairColumnLayout::core_orbital`].
#[derive(Clone, Debug, PartialEq)]
pub struct ExchangePairBlock {
    pub q_index: usize,
    pub n_points: usize,
    pub layout: ExchangePairLayout,
    n_columns: usize,
    values: Vec<Complex64>,
}

impl ExchangePairBlock {
    /// Construct after checking the rectangular storage length.
    pub fn new(
        q_index: usize,
        n_points: usize,
        layout: ExchangePairLayout,
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

    /// Number of rectangular pair columns.
    pub const fn n_columns(&self) -> usize {
        self.n_columns
    }

    /// Entry at grid point `point` and rectangular pair column `column`.
    pub fn at(&self, point: usize, column: usize) -> Complex64 {
        self.values[point * self.n_columns + column]
    }

    /// Copy selected rows, preserving the layout's column order.
    pub fn selected_rows(&self, points: &[usize]) -> Result<Vec<Complex64>, ThcError> {
        let mut out = Vec::with_capacity(checked_storage_len(&[points.len(), self.n_columns])?);
        for &point in points {
            if point >= self.n_points {
                return Err(ThcError::PointIndex(point));
            }
            let start = point * self.n_columns;
            out.extend_from_slice(&self.values[start..start + self.n_columns]);
        }
        Ok(out)
    }
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
