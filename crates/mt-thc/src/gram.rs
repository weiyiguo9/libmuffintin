//! Injected pair-pair Coulomb Gram contract.
//!
//! This is not a Coulomb assembler. Callers supply already-built Hermitian
//! pair-pair matrices (toy finite-cutoff oracles or SPEX dumps). M-J owns
//! production Weinert/SPEX $V^q$.

use crate::ThcError;
use crate::error::checked_storage_len;
use crate::linalg::{frobenius, hermitian_eigensystem};
use crate::pair::PairColumnLayout;
use muffintin_product::TransferQ;
use num_complex::Complex64;

/// Relative Hermiticity bound taken from `thc_lapw_end_to_end_test.py:439`.
pub const GRAM_HERMITIAN_TOLERANCE: f64 = 2.0e-12;
/// Relative negative-eigenvalue bound from `thc_lapw_end_to_end_test.py:444`.
pub const GRAM_PSD_TOLERANCE: f64 = 1.0e-10;

/// One injected pair-pair Coulomb Gram at a canonical $q$.
///
/// Storage is row-major `n_columns × n_columns`. The matrix multiplies pair
/// columns, not interpolation points.
#[derive(Clone, Debug, PartialEq)]
pub struct InjectedCoulombGram {
    pub q_index: usize,
    pub q: TransferQ,
    pub layout: PairColumnLayout,
    n: usize,
    data: Vec<Complex64>,
}

impl InjectedCoulombGram {
    /// Validate Hermiticity, shape, finiteness, and a weak PSD bound.
    pub fn from_dense(
        q_index: usize,
        q: TransferQ,
        layout: PairColumnLayout,
        data: Vec<Complex64>,
    ) -> Result<Self, ThcError> {
        let n = layout.n_columns()?;
        let expected_len = checked_storage_len(&[n, n])?;
        if data.len() != expected_len {
            return Err(ThcError::GramShape {
                index: q_index,
                expected_len,
                actual_len: data.len(),
            });
        }
        if data
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(ThcError::GramNonFinite(q_index));
        }
        let gram = Self {
            q_index,
            q,
            layout,
            n,
            data,
        };
        gram.validate_hermitian()?;
        gram.validate_psd()?;
        Ok(gram)
    }

    /// Row-major entries.
    pub fn data(&self) -> &[Complex64] {
        &self.data
    }

    /// Dimension (pair-column count).
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Require matching q-index, transfer $q$, pair-column layout, and dimension.
    pub fn require_context(
        &self,
        q_index: usize,
        q: TransferQ,
        layout: PairColumnLayout,
    ) -> Result<(), ThcError> {
        if self.q_index != q_index {
            return Err(ThcError::GramQIndex {
                expected: q_index,
                actual: self.q_index,
            });
        }
        if self.q != q {
            return Err(ThcError::GramTransferQ(q_index));
        }
        if self.layout != layout {
            return Err(ThcError::GramColumnOrder(q_index));
        }
        let n = layout.n_columns()?;
        if self.n != n {
            return Err(ThcError::GramShape {
                index: q_index,
                expected_len: checked_storage_len(&[n, n])?,
                actual_len: self.data.len(),
            });
        }
        Ok(())
    }

    fn validate_hermitian(&self) -> Result<(), ThcError> {
        let n = self.dimension();
        let mut diff = 0.0;
        for i in 0..n {
            for j in 0..n {
                diff += (self.data[i * n + j] - self.data[j * n + i].conj()).norm_sqr();
            }
        }
        let relative = diff.sqrt() / frobenius(&self.data).max(1.0e-30);
        if relative > GRAM_HERMITIAN_TOLERANCE {
            return Err(ThcError::GramNotHermitian {
                index: self.q_index,
                relative,
            });
        }
        Ok(())
    }

    fn validate_psd(&self) -> Result<(), ThcError> {
        let n = self.dimension();
        let (values, _) = hermitian_eigensystem(&self.data, n)?;
        let max = values.iter().fold(0.0_f64, |acc, value| acc.max(*value));
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        if min < -GRAM_PSD_TOLERANCE * max.max(1.0) {
            return Err(ThcError::GramIndefinite {
                index: self.q_index,
                min,
                max,
            });
        }
        Ok(())
    }
}

/// Injected grams for every canonical $q$, in mesh order.
#[derive(Clone, Debug, PartialEq)]
pub struct CoulombGramSet {
    grams: Vec<InjectedCoulombGram>,
}

impl CoulombGramSet {
    /// Require one gram per q-index `0..n_q` in order.
    pub fn new(
        grams: Vec<InjectedCoulombGram>,
        n_q: usize,
        layout: PairColumnLayout,
    ) -> Result<Self, ThcError> {
        if grams.len() != n_q {
            return Err(ThcError::MissingCoulombGrams);
        }
        for (index, gram) in grams.iter().enumerate() {
            if gram.q_index != index {
                return Err(ThcError::GramColumnOrder(index));
            }
            if gram.layout != layout {
                return Err(ThcError::GramColumnOrder(index));
            }
        }
        Ok(Self { grams })
    }

    /// Grams in canonical-q order.
    pub fn grams(&self) -> &[InjectedCoulombGram] {
        &self.grams
    }

    /// Gram at mesh index `iq`.
    pub fn get(&self, iq: usize) -> Result<&InjectedCoulombGram, ThcError> {
        self.grams.get(iq).ok_or(ThcError::KMeshIndex {
            index: iq,
            count: self.grams.len(),
        })
    }
}
