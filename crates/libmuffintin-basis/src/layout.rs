//! Global `[envelope][site confined]` layout and site-local `(l, m, n)` order.

use std::ops::Range;

/// Counts and ordering of one site's local orbitals.
///
/// Orbitals are contiguous in `(l, m, n)` order: increasing `l`, then
/// `m = -l..l`, then the local-orbital number `n` for that `l`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LocalOrbitalLayout {
    counts_by_l: Vec<usize>,
}

impl LocalOrbitalLayout {
    pub fn new(counts_by_l: Vec<usize>) -> Self {
        Self { counts_by_l }
    }

    pub fn counts_by_l(&self) -> &[usize] {
        &self.counts_by_l
    }

    pub fn len(&self) -> usize {
        self.counts_by_l
            .iter()
            .enumerate()
            .map(|(l, count)| (2 * l + 1) * count)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Site-local LO index in the documented `(l, m, n)` order.
    pub fn index(&self, l: u32, m: i32, n: usize) -> Option<usize> {
        let count = *self.counts_by_l.get(l as usize)?;
        if m < -(l as i32) || m > l as i32 || n >= count {
            return None;
        }
        let preceding = self
            .counts_by_l
            .iter()
            .enumerate()
            .take(l as usize)
            .map(|(previous_l, count)| (2 * previous_l + 1) * count)
            .sum::<usize>();
        Some(preceding + (m + l as i32) as usize * count + n)
    }
}

/// Global basis order: all envelope (plane-wave) functions, followed by each
/// site's confined local orbitals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BasisLayout {
    plane_wave_count: usize,
    local_orbitals: Vec<LocalOrbitalLayout>,
}

impl BasisLayout {
    pub fn new(plane_wave_count: usize, local_orbitals: Vec<LocalOrbitalLayout>) -> Self {
        Self {
            plane_wave_count,
            local_orbitals,
        }
    }

    pub const fn plane_wave_count(&self) -> usize {
        self.plane_wave_count
    }

    pub fn plane_wave_range(&self) -> Range<usize> {
        0..self.plane_wave_count
    }

    pub fn site_count(&self) -> usize {
        self.local_orbitals.len()
    }

    pub fn site_layout(&self, site: usize) -> Option<&LocalOrbitalLayout> {
        self.local_orbitals.get(site)
    }

    pub fn site_local_orbital_range(&self, site: usize) -> Option<Range<usize>> {
        let site_layout = self.local_orbitals.get(site)?;
        let start = self.plane_wave_count
            + self.local_orbitals[..site]
                .iter()
                .map(LocalOrbitalLayout::len)
                .sum::<usize>();
        Some(start..start + site_layout.len())
    }

    pub fn local_orbital_index(&self, site: usize, l: u32, m: i32, n: usize) -> Option<usize> {
        let range = self.site_local_orbital_range(site)?;
        Some(range.start + self.local_orbitals[site].index(l, m, n)?)
    }

    pub fn dimension(&self) -> usize {
        self.plane_wave_count
            + self
                .local_orbitals
                .iter()
                .map(LocalOrbitalLayout::len)
                .sum::<usize>()
    }
}
