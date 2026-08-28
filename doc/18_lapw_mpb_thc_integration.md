# 18. Scalar M-L1 product input, M-L2 mixed-product, and M-L3 adaptive THC

This note records the implemented M-L1 boundary, the M-L2 scalar mixed-product
bridge, and the M-L3 scalar AllQL2 interpolation-point seam.
It does not assemble Weinert $V^q$, export HDF5, include core–valence
products, or accept a SPEX/material comparison.

Product kinematics remain [13](13_product_space_and_lapw_mpb.md). The toy
canonical $q$ / Umklapp pair gauge remains [14](14_toy_kpoint_isdf_thc.md).
The DFT snapshot kernel remains [17](17_minimal_lda_scf.md).

## 1. Packages

| Directory | Package |
|---|---|
| `crates/mt-runtime` | `libmuffintin-runtime` (`muffintin`) |
| `crates/mt-auxiliary-ir` | `libmuffintin-auxiliary-ir` (`muffintin_auxiliary_ir`) |
| `crates/mt-mpb` | `libmuffintin-mpb` (`muffintin_mpb`) |
| `crates/mt-thc` | `libmuffintin-thc` (`muffintin_thc`) |

`SnapshotDftPhysics::scalar_product_input` owns the M-L1 capability. It
depends on `libmuffintin-auxiliary-ir` for [`ProductSource`] and
[`PairColumnLayout`]. It does not depend on `libmuffintin-thc`.
`build_scalar_mpb` owns the M-L2 capability and depends on
`libmuffintin-mpb`. `build_scalar_thc` owns the M-L3 capability and depends
on `libmuffintin-thc`. `libmuffintin-mpb` and `libmuffintin-thc` do not
depend on runtime, and THC does not depend on MPB or Coulomb.

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

## 4. Scalar M-L2 mixed-product bridge

`build_scalar_mpb(&ScalarProductInput, &ScalarMpbSpec)` consumes the published
M-L1 bundle and an explicit spec: the reciprocal lattice required by
`spex_mixed_product_basis`, `product_l_max`, `product_g_max`,
`overlap_tolerance`, and a nonempty list of same-spin selections
`(spin, k, left_band, right_band)`. The result owns the untruncated
[`RawProductSpace`], the `TOL`-retained [`CompiledAuxiliaryBasis`] with
$n_{\mathrm{spin}}=2$, and [`ScalarMpbPairVertex`] records that keep spin,
pair-column identity $k\cdot N_{\mathrm{orb}}^2+i\cdot N_{\mathrm{orb}}+j$,
band indices, and a checked [`PairVertex`].

The left orbital is the mapped $k-q$ side; the right orbital is at $k$.
Band indices are the published M-L1 common leading window. Pair columns stay
[`PairColumnLayout`]; the old $12\times 12$ packing is not used. The vertex
identity is [`OrbitalPair::Bloch`]; spin is stored on the runtime record, not
on the shared [`OrbitalPair`] model. Empty selection, a spin/$k$/band outside
the frozen input, or an incompatible pair-column layout is a typed
stage-boundary error.

Muffin-tin contraction uses [`CompiledSiteProjection`] for every APW $u$,
APW $\dot u$, and LO site coordinate present in the exact per-$k$
[`CompiledBasis`]. The coefficient order is
$\mathrm{conj}(C_{\mathrm{left}})C_{\mathrm{right}}$ times the inverse
canonical site phase $\exp(-i q\cdot R_a)$ so the primitive MPB
$+\mathrm{i}q\cdot R_a$ kernel is not double-counted against matching phases
already stored on the projection. Terms absent from the raw radial products
(triangle/parity) are skipped; $u$, $\dot u$, and LO channels that exist in
the descriptor are not dropped.

Interstitial contraction uses PW-only rows with relative label

```math
G_{\mathrm{rel}} = G_{\mathrm{right}} - G_{\mathrm{left}} + G_{\mathrm{wrap}}
```

and amplitude $\mathrm{conj}(C_{\mathrm{left}})C_{\mathrm{right}}/\Omega$.
$G_{\mathrm{rel}}$ must exist on the M-L1 raw pair support. The global
`TransferQ` Umklapp is not folded into $G_{\mathrm{rel}}$; it remains the
MPB $\Theta_I$ argument

```math
G_{\mathrm{aux}} - G_{\mathrm{transfer}} - G_{\mathrm{rel}}.
```

The MPB accumulator [`PairVertexAccumulator`] sums those primitive terms into
one checked vertex. Runtime does not build a complete primitive vertex per
basis-pair term.

## 5. Scalar M-L3 adaptive THC

`build_scalar_thc` consumes a nonempty complete $q$ slice in production
$q$-index order: `inputs[iq]` is the M-L1 bundle whose canonical $q$ is the
$iq$-th k-mesh point. Bundles share the frozen orbital window, spins, k mesh,
partition, radials, and `PairColumnLayout`. `ScalarThcSpec` selects one
collinear spin, an existing THC `RankPolicy`, candidate points, and one
production L2 engine (`FullColumnPivotedQr` or `FullPivotedCholesky`).
`All` uses strictly positive-weight parent indices in parent order; explicit
zero-weight indices are rejected. Both engines run AllQL2 on the same ordered
pair blocks, positive-weight candidates, true weights, and rank policy. The
result type is shared; the engine is recorded on selection provenance.
Pivoted Cholesky does not form the dense point Gram, but it still
materializes the stacked weighted pair matrix. Core-orbital diagnostics stay
empty.

`ScalarThcGrid` is an externally supplied immutable parent support bound to
the exact `ProductPartition`. Muffin-tin points name `{site, radial_index}`
on the stored exponential mesh and must lie on that sample's Cartesian
radius; interstitial points are partitioned interstitial. Weights may be
zero if at least one is positive; they are never clamped and are not L2
selection candidates. There is no radial interpolation.

Muffin-tin orbitals use `CompiledSiteProjection` and the published $u$,
$\dot u$, LO map. Physical KH samples are $P=p/r$ and optional $Q=q/r$. The
reconstructed muffin-tin Bloch value is converted to the cell-periodic
representation by $\exp(-i k\cdot r)$ at the Cartesian point, using the
stored plane-wave Cartesian $k$, so muffin-tin and interstitial orbitals
share one gauge. The pair density is $\mathrm{conj}(P_{\mathrm{left}})P_{\mathrm{right}}
+\mathrm{conj}(Q_{\mathrm{left}})Q_{\mathrm{right}}$. Interstitial evaluation
uses the PW large component $C_G\exp(i G\cdot r)/\sqrt{\Omega}$. The pair
phase is the stored per-column wrap $\exp(+i G_{\mathrm{wrap}}\cdot r)$.
Global `TransferQ` Umklapp stays on the $q$ record.

The result carries the selected spin, the parent grid, selection with
requested and effective rank, and one per-$q$ interpolation-point
`CompiledAuxiliaryBasis`, parent-grid-by-selected-point $\zeta$, and
`PairVertex` columns. It does not build `SampledAuxiliaryFunctions` or
Coulomb operators.

## 6. Explicit exclusions

M-L1, M-L2, and M-L3 do not:

- assemble Weinert or SPEX Coulomb operators or `SampledAuxiliaryFunctions`
- inject Coulomb Grams or run Q0L2 / AllQCoulombPool
- include selected core radials in the product window
- extend `ProductRadialId` with $\kappa$ or four-component $PP$/$QQ`
- implement spinor or signed-$\kappa$ product vertices
- claim SPEX or material acceptance
- complete M-L

Later M-L stages consume this scalar product-input, mixed-product, and
interpolation-point contract rather than reaching into snapshot solver
internals.
