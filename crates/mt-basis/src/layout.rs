//! Global `[envelope][site confined]` layout and site-local `(l, m, n)` order.

use crate::BasisError;
use muffintin_core::{Kappa, TwiceMu};
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

/// Counts and ordering of one site's spinor local orbitals.
///
/// The stored shells are ordered by increasing signed `kappa`.  Within one
/// shell, orbitals are contiguous in increasing exact `twice_mu`, followed by
/// the local-orbital number `n` (fastest): `(kappa, twice_mu, n)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpinorSiteLayout {
    counts_by_kappa: Vec<(Kappa, usize)>,
}

impl SpinorSiteLayout {
    /// Construct a deterministic spinor site layout.
    ///
    /// Input order is immaterial.  Duplicate `kappa` shells are rejected
    /// because their `(kappa, twice_mu, n)` coordinates would be ambiguous.
    pub fn new(mut counts_by_kappa: Vec<(Kappa, usize)>) -> Result<Self, BasisError> {
        counts_by_kappa.sort_unstable_by_key(|(kappa, _)| kappa.get());
        for pair in counts_by_kappa.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(BasisError::DuplicateKappa {
                    kappa: pair[0].0.get(),
                });
            }
        }
        Ok(Self { counts_by_kappa })
    }

    /// `(kappa, count)` shells in their stored site-local order.
    pub fn counts_by_kappa(&self) -> &[(Kappa, usize)] {
        &self.counts_by_kappa
    }

    pub fn len(&self) -> usize {
        self.counts_by_kappa
            .iter()
            .map(|(kappa, count)| kappa.degeneracy() as usize * count)
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Site-local index in exact `(kappa, twice_mu, n)` order.
    pub fn index(&self, kappa: Kappa, twice_mu: TwiceMu, n: usize) -> Option<usize> {
        let shell = self
            .counts_by_kappa
            .binary_search_by_key(&kappa.get(), |(candidate, _)| candidate.get())
            .ok()?;
        let count = self.counts_by_kappa[shell].1;
        if n >= count {
            return None;
        }

        let twice_j = i64::from(kappa.twice_j());
        let twice_mu = twice_mu.get();
        if twice_mu < -twice_j || twice_mu > twice_j {
            return None;
        }
        let preceding = self.counts_by_kappa[..shell]
            .iter()
            .map(|(previous_kappa, count)| previous_kappa.degeneracy() as usize * count)
            .sum::<usize>();
        let mu_index = ((twice_mu + twice_j) / 2) as usize;
        Some(preceding + mu_index * count + n)
    }
}

/// Global two-component Pauli plane-wave and site-local spinor basis order.
///
/// The plane-wave coordinate is `spin * n_g + g`, with spin slow.  Both spin
/// blocks precede every site's `(kappa, twice_mu, n)` local orbitals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpinorBasisLayout {
    spatial_plane_wave_count: usize,
    site_spinors: Vec<SpinorSiteLayout>,
}

impl SpinorBasisLayout {
    pub fn new(spatial_plane_wave_count: usize, site_spinors: Vec<SpinorSiteLayout>) -> Self {
        Self {
            spatial_plane_wave_count,
            site_spinors,
        }
    }

    pub const fn spatial_plane_wave_count(&self) -> usize {
        self.spatial_plane_wave_count
    }

    pub const fn plane_wave_count(&self) -> usize {
        2 * self.spatial_plane_wave_count
    }

    pub fn plane_wave_range(&self) -> Range<usize> {
        0..self.plane_wave_count()
    }

    /// Range occupied by one Pauli spin (`0` or `1`).
    pub fn plane_wave_spin_range(&self, spin: usize) -> Option<Range<usize>> {
        if spin >= 2 {
            return None;
        }
        let start = spin * self.spatial_plane_wave_count;
        Some(start..start + self.spatial_plane_wave_count)
    }

    /// Global index `spin * n_g + g` for a Pauli plane wave.
    pub fn plane_wave_index(&self, spin: usize, g: usize) -> Option<usize> {
        if spin >= 2 || g >= self.spatial_plane_wave_count {
            return None;
        }
        Some(spin * self.spatial_plane_wave_count + g)
    }

    pub fn site_count(&self) -> usize {
        self.site_spinors.len()
    }

    pub fn site_layout(&self, site: usize) -> Option<&SpinorSiteLayout> {
        self.site_spinors.get(site)
    }

    pub fn site_spinor_range(&self, site: usize) -> Option<Range<usize>> {
        let site_layout = self.site_spinors.get(site)?;
        let start = self.plane_wave_count()
            + self.site_spinors[..site]
                .iter()
                .map(SpinorSiteLayout::len)
                .sum::<usize>();
        Some(start..start + site_layout.len())
    }

    pub fn site_spinor_index(
        &self,
        site: usize,
        kappa: Kappa,
        twice_mu: TwiceMu,
        n: usize,
    ) -> Option<usize> {
        let range = self.site_spinor_range(site)?;
        Some(range.start + self.site_spinors[site].index(kappa, twice_mu, n)?)
    }

    pub fn dimension(&self) -> usize {
        self.plane_wave_count()
            + self
                .site_spinors
                .iter()
                .map(SpinorSiteLayout::len)
                .sum::<usize>()
    }
}
