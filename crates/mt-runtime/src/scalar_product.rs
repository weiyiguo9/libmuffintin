//! Frozen scalar LAPW product input for one requested transfer $q$.

use muffintin_auxiliary_ir::{
    PairColumnLayout, ProductPartition, ProductRadial, ProductSource, RadialSamples,
    RawInterstitialPairSupport, SiteRadialSet, TransferQ,
};
use muffintin_core::{InverseBohr, ReciprocalLattice};
use muffintin_dft::{
    ScalarIterationBasis, ScalarRadialSite, ScfConfig, ScfRelativity,
    build_extended_snapshot_core_potentials,
};
use muffintin_lapw::{CompiledBasis, Provenance};
use muffintin_operators::Collinear;
use muffintin_radial::CorePotentialContinuationSpec;
use muffintin_tensor::DenseEigenvectors;
use std::collections::BTreeSet;

use crate::snapshot_dft::{
    SnapshotBandSolution, SnapshotDftError, SnapshotDftPhysics, SnapshotKPointSolution, g_vector,
    regular_k_points,
};

/// ProductRadial $n$ for the scalar linearization function $u$.
pub const SCALAR_RADIAL_U: usize = 0;
/// ProductRadial $n$ for the energy derivative $\dot u$.
pub const SCALAR_RADIAL_UDOT: usize = 1;
/// First local-orbital ProductRadial $n$; later LOs use `SCALAR_RADIAL_LO0 + ordinal`.
pub const SCALAR_RADIAL_LO0: usize = 2;

const MESH_COORD_TOLERANCE: f64 = 1.0e-12;

/// Per-k map of $k-q_{\mathrm{canonical}}$ onto the regular mesh.
///
/// The integer wrap $G_{\mathrm{wrap}}$ satisfies
/// $k_{\mathrm{frac}}-q_{\mathrm{canonical,frac}}
/// =(k-q)_{\mathrm{frac}}+G_{\mathrm{wrap,index}}$
/// in primitive reciprocal coordinates. Pair phases use
/// $\exp(+i G_{\mathrm{wrap}}\cdot r)$. This wrap is not
/// [`TransferQ::umklapp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarKMinusQ {
    pub k_index: usize,
    pub kq_index: usize,
    pub umklapp: muffintin_core::GVector,
}

/// Common leading band window retained for pair columns.
///
/// M-L1 keeps the lowest `count` eigenpairs starting at `start` (always 0).
/// Per-$k$ available counts remain on [`ScalarSpinChannel::available_bands`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScalarBandWindow {
    pub start: usize,
    pub count: usize,
}

/// One collinear spin channel of frozen scalar eigenvectors.
///
/// `eigenvectors[k]` is column-major `[basis, band]` in the k-local compiled
/// APW+LO order. `bases[k]` is the exact [`CompiledBasis`] used by that
/// solve: plane-wave $G$ labels, APW $(u,\dot u)$ matching coefficients, and
/// confined local-orbital layout. `spin` is 0 for up and 1 for down.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarSpinChannel {
    pub spin: u8,
    pub eigenvectors: Vec<DenseEigenvectors>,
    pub energies: Vec<Vec<muffintin_core::Hartree>>,
    pub bases: Vec<CompiledBasis>,
    pub available_bands: Vec<usize>,
}

/// Minimal real scalar Bloch data retained for later pair/THC stages.
///
/// Orbital coefficients and the per-$k$ [`CompiledBasis`] live here, not on
/// [`ProductSource`].
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarFrozenOrbitals {
    pub k_fractional: Vec<[f64; 3]>,
    pub channels: Vec<ScalarSpinChannel>,
    pub band_window: ScalarBandWindow,
}

/// Frozen scalar LAPW solve plus representation-neutral product input at one $q$.
///
/// `source` is the method-neutral [`ProductSource`]. Valence radials use
/// [`SCALAR_RADIAL_U`], [`SCALAR_RADIAL_UDOT`], then local orbitals from
/// [`SCALAR_RADIAL_LO0`]. Pair columns use [`PairColumnLayout`] indexing
/// $k\cdot N_{\mathrm{orb}}^2+i\cdot N_{\mathrm{orb}}+j$. Cores are empty.
/// `reciprocal` is the exact lattice used to fold $q_{\mathrm{in}}$ and
/// $G_{\mathrm{wrap}}$; it is not inferred from a later Coulomb request.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarProductInput {
    pub source: ProductSource,
    pub orbitals: ScalarFrozenOrbitals,
    pub k_minus_q: Vec<ScalarKMinusQ>,
    pub pair_columns: PairColumnLayout,
    pub reciprocal: ReciprocalLattice,
}

impl SnapshotDftPhysics {
    /// Frozen scalar one-particle solve and product input at `q_fractional`.
    ///
    /// `q_fractional` is the requested primitive-cell transfer $q_{\mathrm{in}}$.
    /// The emitted [`TransferQ`] stores $q_{\mathrm{canonical}}$ in $[0,1)^3$
    /// with $q_{\mathrm{in}}=q_{\mathrm{canonical}}+G_{\mathrm{transfer}}$.
    /// Each k-point is mapped with $q_{\mathrm{canonical}}$; off-mesh folded
    /// targets are rejected. Relativity must be scalar Koelling--Harmon.
    pub fn scalar_product_input(
        &self,
        config: &ScfConfig,
        q_fractional: [f64; 3],
    ) -> Result<ScalarProductInput, SnapshotDftError> {
        if config.relativity != ScfRelativity::Scalar {
            return Err(SnapshotDftError::ScalarProductRequiresScalarRelativity);
        }
        if q_fractional.iter().any(|value| !value.is_finite()) {
            return Err(SnapshotDftError::NonFiniteKPoint(q_fractional));
        }
        let meshes = self.channel_meshes(&config.basis)?;
        let extended = build_extended_snapshot_core_potentials(
            self.frozen_potential(),
            self.geometry(),
            self.nuclear_charges(),
            &meshes,
            CorePotentialContinuationSpec::default(),
        )?;
        let basis =
            self.materialize_nonspectral_basis(self.frozen_potential(), &config.basis, &extended)?;
        let k_fractional = regular_k_points(config.k_mesh)?;
        let (q_canonical, q_wrap) = fold_to_unit_cell(q_fractional);
        let q_input = fractional_to_reciprocal(q_fractional, self.reciprocal().basis());
        let transfer_umklapp = g_vector(*self.reciprocal(), q_wrap);
        let q = TransferQ::fold_by_reciprocal_vector(q_input, transfer_umklapp)?;
        let mut k_minus_q = Vec::with_capacity(k_fractional.len());
        for (k_index, &k_frac) in k_fractional.iter().enumerate() {
            k_minus_q.push(map_k_minus_q(
                k_index,
                k_frac,
                q_fractional,
                q_canonical,
                &k_fractional,
                *self.reciprocal(),
            )?);
        }
        let bands = self.solve_points(
            self.frozen_potential(),
            &basis,
            &k_fractional,
            ScfRelativity::Scalar,
        )?;
        emit_scalar_product_input(self, &bands, &k_fractional, q, k_minus_q)
    }
}

fn emit_scalar_product_input(
    physics: &SnapshotDftPhysics,
    bands: &SnapshotBandSolution,
    k_fractional: &[[f64; 3]],
    q: TransferQ,
    k_minus_q: Vec<ScalarKMinusQ>,
) -> Result<ScalarProductInput, SnapshotDftError> {
    let n_k = k_fractional.len();
    let mut channels = Vec::with_capacity(2);
    let mut radials = None;
    let mut n_orb = None;
    let mut available = [Vec::new(), Vec::new()];
    for point in bands.points() {
        match &point.solution {
            SnapshotKPointSolution::Collinear {
                bases, solutions, ..
            } => {
                if radials.is_none() {
                    radials = Some(site_radials(bases)?);
                }
                let up_bands = solutions.up.eigenvectors.columns();
                let down_bands = solutions.down.eigenvectors.columns();
                if up_bands != down_bands {
                    return Err(SnapshotDftError::CollinearBandCount {
                        up: up_bands,
                        down: down_bands,
                    });
                }
                available[0].push(up_bands);
                available[1].push(down_bands);
                n_orb = Some(n_orb.unwrap_or(up_bands).min(up_bands));
            }
            SnapshotKPointSolution::Spinor { .. } => {
                return Err(SnapshotDftError::InconsistentRelativityRoute);
            }
        }
    }
    let n_orb = n_orb
        .filter(|&count| count > 0)
        .ok_or(SnapshotDftError::EmptyKPointSet)?;
    let pair_columns = PairColumnLayout::new(n_k, n_orb, None);
    let _ = pair_columns.n_columns()?;

    for spin in [0_u8, 1] {
        let mut eigenvectors = Vec::with_capacity(n_k);
        let mut energies = Vec::with_capacity(n_k);
        let mut bases = Vec::with_capacity(n_k);
        for point in bands.points() {
            let SnapshotKPointSolution::Collinear {
                bases: iteration,
                solutions,
                ..
            } = &point.solution
            else {
                return Err(SnapshotDftError::InconsistentRelativityRoute);
            };
            let (solution, compiled) = if spin == 0 {
                (&solutions.up, &iteration.up.compiled)
            } else {
                (&solutions.down, &iteration.down.compiled)
            };
            eigenvectors.push(leading_bands(&solution.eigenvectors, n_orb)?);
            let mut values = solution.eigenvalues.clone();
            values.truncate(n_orb);
            energies.push(values);
            bases.push(compiled.clone());
        }
        channels.push(ScalarSpinChannel {
            spin,
            eigenvectors,
            energies,
            bases,
            available_bands: available[spin as usize].clone(),
        });
    }

    let interstitial_pair_support =
        raw_pair_support(q, *physics.reciprocal(), &channels, &k_minus_q)?;
    let source = ProductSource::new(
        ProductPartition::from_interstitial(physics.geometry().clone()),
        radials.ok_or(SnapshotDftError::EmptyKPointSet)?,
        q,
        interstitial_pair_support,
        Provenance {
            recipe: None,
            reference: Some("snapshot-dft-frozen-scalar-ml1".to_owned()),
        },
    )?;
    Ok(ScalarProductInput {
        source,
        orbitals: ScalarFrozenOrbitals {
            k_fractional: k_fractional.to_vec(),
            channels,
            band_window: ScalarBandWindow {
                start: 0,
                count: n_orb,
            },
        },
        k_minus_q,
        pair_columns,
        reciprocal: *physics.reciprocal(),
    })
}

fn raw_pair_support(
    q: TransferQ,
    reciprocal: ReciprocalLattice,
    channels: &[ScalarSpinChannel],
    k_minus_q: &[ScalarKMinusQ],
) -> Result<RawInterstitialPairSupport, SnapshotDftError> {
    let mut indices = BTreeSet::new();
    for channel in channels {
        for mapped in k_minus_q {
            let right = &channel.bases[mapped.k_index].plane_waves;
            let left = &channel.bases[mapped.kq_index].plane_waves;
            let wrap = mapped.umklapp.index;
            for g_k in right {
                for g_kmq in left {
                    indices.insert([
                        g_k.g.index[0] - g_kmq.g.index[0] + wrap[0],
                        g_k.g.index[1] - g_kmq.g.index[1] + wrap[1],
                        g_k.g.index[2] - g_kmq.g.index[2] + wrap[2],
                    ]);
                }
            }
        }
    }
    Ok(RawInterstitialPairSupport::from_relative_indices(
        q, reciprocal, indices,
    )?)
}

fn site_radials(
    bases: &Collinear<ScalarIterationBasis>,
) -> Result<Vec<SiteRadialSet>, SnapshotDftError> {
    if bases.up.radial_sites.len() != bases.down.radial_sites.len() {
        return Err(SnapshotDftError::InconsistentRelativityRoute);
    }
    bases
        .up
        .radial_sites
        .iter()
        .zip(&bases.down.radial_sites)
        .zip(&bases.up.density_sites)
        .map(|((up, down), density)| {
            let mut valence = spin_radials(0, up)?;
            valence.extend(spin_radials(1, down)?);
            Ok(SiteRadialSet {
                mesh: density.mesh.clone(),
                valence,
                cores: Vec::new(),
            })
        })
        .collect()
}

fn spin_radials(spin: u8, site: &ScalarRadialSite) -> Result<Vec<ProductRadial>, SnapshotDftError> {
    let mut valence = Vec::new();
    for (l, linearized) in site.linearized.iter().enumerate() {
        let l = u32::try_from(l).map_err(|_| SnapshotDftError::AngularMomentumOverflow)?;
        valence.push(ProductRadial {
            l,
            n: SCALAR_RADIAL_U,
            spin,
            samples: radial_samples(&linearized.solution.p, linearized.solution.q.as_deref()),
        });
        valence.push(ProductRadial {
            l,
            n: SCALAR_RADIAL_UDOT,
            spin,
            samples: radial_samples(
                &linearized.energy_derivative.p,
                linearized.energy_derivative.q.as_deref(),
            ),
        });
        let locals = site
            .local_orbitals
            .get(l as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        for (ordinal, local) in locals.iter().enumerate() {
            valence.push(ProductRadial {
                l,
                n: SCALAR_RADIAL_LO0 + ordinal,
                spin,
                samples: radial_samples(&local.orbital.p, local.orbital.q.as_deref()),
            });
        }
    }
    Ok(valence)
}

fn leading_bands(
    eigenvectors: &DenseEigenvectors,
    n_orb: usize,
) -> Result<DenseEigenvectors, SnapshotDftError> {
    let rows = eigenvectors.rows();
    let columns = eigenvectors.columns();
    if columns < n_orb {
        return Err(SnapshotDftError::InconsistentBandCount);
    }
    if columns == n_orb {
        return Ok(eigenvectors.clone());
    }
    let host = eigenvectors.to_host_column_major();
    Ok(DenseEigenvectors::from_host_column_major(
        rows,
        n_orb,
        host[..rows * n_orb].to_vec(),
    )?)
}

fn radial_samples(large: &[f64], small: Option<&[f64]>) -> RadialSamples {
    RadialSamples {
        large: large.to_vec(),
        small: small.map(<[f64]>::to_vec),
    }
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

fn map_k_minus_q(
    k_index: usize,
    k_frac: [f64; 3],
    q_in: [f64; 3],
    q_canonical: [f64; 3],
    points: &[[f64; 3]],
    reciprocal: ReciprocalLattice,
) -> Result<ScalarKMinusQ, SnapshotDftError> {
    let mut folded = [0.0; 3];
    for axis in 0..3 {
        folded[axis] = (k_frac[axis] - q_canonical[axis]).rem_euclid(1.0);
    }
    let kq_index = points
        .iter()
        .position(|point| coords_on_mesh(point, folded))
        .ok_or(SnapshotDftError::OffMeshTransfer {
            k: k_frac,
            q_in,
            q_canonical,
            folded,
        })?;
    let actual = points[kq_index];
    let wrap = std::array::from_fn(|axis| {
        (k_frac[axis] - q_canonical[axis] - actual[axis]).round() as i32
    });
    Ok(ScalarKMinusQ {
        k_index,
        kq_index,
        umklapp: g_vector(reciprocal, wrap),
    })
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
