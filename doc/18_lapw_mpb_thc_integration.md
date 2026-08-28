# 18. Scalar M-L1 product input from a frozen snapshot solve

This note records the implemented M-L1 boundary: a snapshot-backed frozen
scalar LAPW solve that emits a method-neutral [`ProductSource`] together with
the minimal real scalar Bloch coefficients and the exact per-$k$ compiled
basis needed by later pair/THC stages.
It does not construct an MPB auxiliary basis, run THC selection, assemble
Weinert $V^q$, export HDF5, include core–valence products, or accept a
SPEX/material comparison.

Product kinematics remain [13](13_product_space_and_lapw_mpb.md). The toy
canonical $q$ / Umklapp pair gauge remains [14](14_toy_kpoint_isdf_thc.md).
The DFT snapshot kernel remains [17](17_minimal_lda_scf.md).

## 1. Packages

| Directory | Package |
|---|---|
| `crates/mt-runtime` | `libmuffintin-runtime` (`muffintin`) |
| `crates/mt-auxiliary-ir` | `libmuffintin-auxiliary-ir` (`muffintin_auxiliary_ir`) |

`SnapshotDftPhysics::scalar_product_input` owns the M-L1 capability. It
depends on `libmuffintin-auxiliary-ir` for [`ProductSource`] and
[`PairColumnLayout`]. It does not depend on `libmuffintin-thc`.

## 2. Implemented boundary

The input is a validated V2 snapshot, an `ScfConfig` whose relativity is
scalar Koelling–Harmon, and a requested transfer in primitive reciprocal
coordinates `q_fractional` $=q_{\mathrm{in}}$. The kernel materializes the
frozen-snapshot iteration basis, rejects a folded $k-q$ that is not on the
regular mesh, solves the regular full-BZ scalar eigenproblem, and returns
[`ScalarProductInput`]:

- `source`: [`ProductSource`] built from the exact scalar iteration bases
  (`ProductPartition` / `InterstitialGeometry`, per-site `ExponentialMesh`,
  valence $p=ru$ samples with optional Koelling–Harmon $Q$, empty cores,
  finite raw interstitial pair support of relative $G$ labels).
- `orbitals`: per-spin, per-$k$ column-major `[basis, band]` eigenvectors
  together with the exact [`CompiledBasis`] used by `solve_points` (plane-wave
  $G$ labels, APW $(u,\dot u)$ matching coefficients, confined LO layout).
  Spin labels are `0` (up) and `1` (down). This bundle does **not** put
  coefficients or `CompiledBasis` on [`ProductSource`].
- `k_minus_q`: folded $k-q_{\mathrm{canonical}}$ mesh index and the
  per-column [`ReciprocalLattice`] wrap $G_{\mathrm{wrap}}$.
- `pair_columns`: `PairColumnLayout::new(n_k, n_orb, None)` from
  `muffintin_auxiliary_ir`, with
  $k\cdot N_{\mathrm{orb}}^2+i\cdot N_{\mathrm{orb}}+j$. The old packed
  $12\times 12$ experiment flattening is not used.
- `orbitals.band_window`: common leading window `{start: 0, count: n_orb}`.
  Each spin channel also stores `available_bands[k]`, the untruncated
  eigenpair count at that $k$.

Valence [`ProductRadial`] $n$ is stable: $n=0$ is $u$, $n=1$ is $\dot u$,
and $n=2+\mathrm{ordinal}$ are local orbitals. APW matching coefficients
multiply $(u,\dot u)$. Local-orbital rows follow
[`BasisLayout::site_local_orbital_range`]. `ProductRadialId` remains the
scalar $l$-based identifier. M-L1 does not add $\kappa$, $PP$, or $QQ$.

`ScfState` is not the orbital source. Private
`SnapshotBandSolution` / `SnapshotKPointSolution` fields stay private.

## 3. Canonical $q$ and Umklapp

The requested transfer $q_{\mathrm{in}}$ is folded into the primitive cell
$[0,1)^3$. [`TransferQ::fold_by_reciprocal_vector`] stores the canonical
Cartesian $q$ and the subtracted reciprocal vector $G_{\mathrm{transfer}}$:

```math
q_{\mathrm{in}} = q_{\mathrm{canonical}} + G_{\mathrm{transfer}}.
```

$k-q$ mapping uses $q_{\mathrm{canonical}}$ only. A folded target that is
not an existing regular-mesh coordinate (within $10^{-12}$) is
`OffMeshTransfer`; the kernel does not round onto a neighbouring mesh
point. $q_{\mathrm{in}}=(1.5,0,0)$ on a $2\times 1\times 1$ mesh is valid
because $q_{\mathrm{canonical}}=(0.5,0,0)$. $q_{\mathrm{in}}=(0.25,0,0)$
is not.

The per-column wrap $G_{\mathrm{wrap}}$ satisfies

```math
k_{\mathrm{frac}} - q_{\mathrm{canonical,frac}}
= (k-q)_{\mathrm{frac}} + G_{\mathrm{wrap,index}}
```

and the pair phase is

```math
\exp(+i G_{\mathrm{wrap}}\cdot r).
```

Raw interstitial support enumerates, for every $k\to k-q$ pair and both
spin channels, every

```math
G_{\mathrm{raw}} = G_{k} - G_{k-q} + G_{\mathrm{wrap}}
```

from the actual left/right plane-wave $G$ labels. Labels are deduplicated
and ordered by $|G|$ then G-index. That list includes per-column
$G_{\mathrm{wrap}}$ and excludes the global `TransferQ` wrap. A sign error
in $G_{\mathrm{wrap}}$ reverses $\exp(+i G\cdot R)$ at a muffin-tin site
whose $G\cdot R$ is not a multiple of $\pi$.

This is the production $k-q$ gauge. There is no second $k+q$ convention in
M-L1.

## 4. Explicit exclusions

M-L1 does not:

- run MPB construction or apply `TOL`
- run THC selection, $\zeta$ fits, or interpolation-point auxiliaries
- assemble Weinert or SPEX $V^q$
- evaluate grid-sampled `BlochOrbitals` / `PairBlock` tensors
- include selected core radials in the product window
- extend `ProductRadialId` with $\kappa$ or four-component $PP$/$QQ`
- claim SPEX or material acceptance
- complete M-L

Later M-L stages consume this scalar product-input contract rather than
reaching into snapshot solver internals.
