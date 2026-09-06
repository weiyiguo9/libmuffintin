use super::FftError;

/// Dense periodic storage, with the last axis contiguous.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FftGrid {
    pub(super) dimensions: [usize; 3],
    len: usize,
}

impl FftGrid {
    pub(crate) fn new(dimensions: [usize; 3]) -> Result<Self, FftError> {
        let len = dimensions
            .iter()
            .try_fold(1usize, |n, &d| {
                if d == 0 || d > i32::MAX as usize {
                    None
                } else {
                    n.checked_mul(d)
                }
            })
            .ok_or(FftError::InvalidDimensions(dimensions))?;
        Ok(Self { dimensions, len })
    }

    pub(crate) fn len(self) -> usize {
        self.len
    }

    pub(crate) fn index(self, index: [i32; 3]) -> usize {
        let [i, j, k] = std::array::from_fn(|axis| {
            index[axis].rem_euclid(self.dimensions[axis] as i32) as usize
        });
        (i * self.dimensions[1] + j) * self.dimensions[2] + k
    }
}
