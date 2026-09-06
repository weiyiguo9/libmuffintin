use super::FftError;

/// Dense periodic storage, with the last axis contiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FftGrid {
    pub(super) dimensions: [usize; 3],
    len: usize,
}

impl FftGrid {
    pub fn new(dimensions: [usize; 3]) -> Result<Self, FftError> {
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

    pub fn len(self) -> usize {
        self.len
    }

    pub fn dimensions(self) -> [usize; 3] {
        self.dimensions
    }

    pub fn is_empty(self) -> bool {
        false
    }

    pub fn index(self, index: [i32; 3]) -> usize {
        let [i, j, k] = std::array::from_fn(|axis| {
            index[axis].rem_euclid(self.dimensions[axis] as i32) as usize
        });
        (i * self.dimensions[1] + j) * self.dimensions[2] + k
    }
}
