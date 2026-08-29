# 18. Scalar M-L1–L4 product/MPB/THC/Coulomb and spinor M-L5a/M-L5b/M-L5c

This note records the implemented M-L1 boundary, the M-L2 scalar mixed-product
bridge, the M-L3 scalar AllQL2 interpolation-point seam, the M-L4 sampled-$\zeta$
Coulomb bridge, the M-L5a Dirac PP/QQ IR and MPB primitive, the M-L5b
frozen full-first-variation spinor product input, the M-L5c selected-band
spinor mixed-product bridge, and the M-L6b2 runtime materialization of those
scalar objects into MLDUMP v1. Core–valence products and SPEX/material
comparison remain out of scope. The on-disk schema is [19](19_versioned_mldump_interchange.md).

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
| `crates/mt-coulomb` | `libmuffintin-coulomb` (`muffintin_coulomb`) |

`SnapshotDftPhysics::scalar_product_input` owns the M-L1 capability. It
depends on `libmuffintin-auxiliary-ir` for [`ProductSource`] and
[`PairColumnLayout`]. It does not depend on `libmuffintin-thc`.
`build_scalar_mpb` owns the M-L2 capability and depends on
`libmuffintin-mpb`. `build_scalar_thc` owns the M-L3 capability and depends
on `libmuffintin-thc`. `build_scalar_coulomb` owns the M-L4 capability and
depends on `libmuffintin-coulomb`. `SnapshotDftPhysics::spinor_product_input`
owns the M-L5b capability and consumes M-L5a `DiracProductSource` from
`libmuffintin-auxiliary-ir`. `build_spinor_mpb` owns the M-L5c capability
and depends on `libmuffintin-mpb`; MPB does not depend on runtime, THC, or
Coulomb. Production dependencies: `libmuffintin-mpb`, `libmuffintin-thc`, and
`libmuffintin-coulomb` do not depend on runtime. THC does not depend on
MPB or Coulomb, and Coulomb does not depend on MPB or THC. Coulomb's
dev-dependencies include MPB and THC for representation and vertex tests;
those are not production DAG edges.

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
- `reciprocal`: the exact [`ReciprocalLattice`] used to fold $q_{\mathrm{in}}$
  and $G_{\mathrm{wrap}}$. M-L4 requires `CoulombRequest` to carry this lattice;
  a same-volume sheared cell is rejected.
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
collinear spin, an existing THC `RankPolicy`, candidate points
([`ThcCandidates`]), and one production L2 engine ([`ThcEngine`],
`FullColumnPivotedQr` or `FullPivotedCholesky`).
`All` uses strictly positive-weight parent indices in parent order; explicit
zero-weight indices are rejected. Both engines run AllQL2 on the same ordered
pair blocks, positive-weight candidates, true weights, and rank policy. The
result type is shared; the engine is recorded on selection provenance.
Pivoted Cholesky does not form the dense point Gram, but it still
materializes the stacked weighted pair matrix. Core-orbital diagnostics stay
empty.

`ThcParentGrid` is an externally supplied immutable parent support bound to
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
`PairVertex` columns. Interpolation-point auxiliaries are created with the
scalar THC provenance before Bloch pair vertices, so auxiliary and vertex
records bind the same $q$/layout/partition/provenance at construction. It
does not build `SampledAuxiliaryFunctions` or Coulomb operators.

## 6. Scalar M-L4 sampled Coulomb

`build_scalar_coulomb` consumes the same complete ordered $q$ slice, a matching
`ScalarThcResult`, an existing `CoulombRequest` plus `InterpolationProjection`,
and an explicit bounded subset of matching M-L2 `ScalarMpbResult` vertices.
The request reciprocal lattice must equal the frozen `ScalarProductInput`
reciprocal; it is not inferred from the caller's cell recipe. For every THC
$q$ record it builds `SampledAuxiliaryFunctions` on the full M-L3 parent grid
in original order: Cartesian Bohr coordinates, true weights including
zero-weight rows, exact muffin-tin `{site, radial_index}` or interstitial
support, per-site meshes, and that record's row-major parent-grid $\zeta$.
A private construction fingerprint binds that ordered grid to every $q$
record/fit; a permuted or substituted parent grid is rejected. Interpolation
nodes are not substituted for the $\zeta$ grid. It then calls production
`assemble_sampled_coulomb`. Gauge is unchanged: no extra pair rephasing and no
second global Umklapp insertion. Gamma keeps the finite body plus `GammaHead`
metadata; the singular head is not inserted.

Each per-$q$ record returns the sampled-$\zeta$ `CoulombOperator` with $q$
index, `TransferQ`, spin, pair-column layout, interpolation-point auxiliary,
parent-grid sampled $\zeta$, and semantic pair vertices. Returned M-L3 vertices
must carry the compiled auxiliary `AuxiliaryLayout` and an
`OrbitalPair::Bloch` identity that matches the decoded column
$(k,\mathrm{left},\mathrm{right})$. Malformed public Bloch indices are
rejected without calling `PairColumnLayout::encode`. Matched M-L2/M-L3 pairs
compare quadratic forms $c^\dagger V c$ across representations, with an
absolute/relative discrepancy using the stated floor $10^{-12}$. Per-side
action norms $\|Vc\|$ are debug diagnostics in each auxiliary basis and are
not compared across representations. Elementwise MPB/THC matrices and
principal-angle spans are not compared.

## 7. Explicit exclusions

M-L1, M-L2, M-L3, and M-L4 do not:

- copy or reimplement Weinert assembly inside runtime
- inject Coulomb Grams or run Q0L2 / AllQCoulombPool
- include selected core radials in the product window
- extend `ProductRadialId` with $\kappa$ or four-component $PP/QQ$
- implement spinor or signed-$\kappa$ product vertices
- add a principal-angle engine
- claim SPEX or material acceptance
- complete M-L

Later M-L stages consume this scalar product-input, mixed-product,
interpolation-point, and sampled-$\zeta$ Coulomb contract rather than reaching
into snapshot solver internals.

## 8. M-L5a Dirac PP/QQ IR and MPB primitive

M-L5a lives in `libmuffintin-auxiliary-ir` (`DiracProductSource`, physical
$P$ and $Q$, explicit `DiracChargeSector::{LargeLarge, SmallSmall}`) and
`libmuffintin-mpb` (untruncated PP/QQ raw products and the checked muffin-tin
vertex primitive). Runtime M-L5b consumes those public types and does not
redefine them. Scalar `ProductRadialId` / `ProductSource` / `RawProductSpace`
stay unchanged. M-L5c adds Dirac overlap cutoff, retained scalar-charge
modes of the PP/QQ union, and interstitial spinor plane-wave contraction.
M-L5d adds spinor all-$q$ THC and sampled-$\zeta$ Coulomb.

## 9. M-L5b frozen spinor product input

`SnapshotDftPhysics::spinor_product_input(&ScfConfig, q_fractional)` accepts
only `ScfRelativity::SpinorFirstVariation`. Scalar Koelling–Harmon and SOC
second variation are distinct typed rejections; signed $\kappa$ is not routed
through second variation. The kernel materializes the frozen-snapshot
iteration basis, reuses the M-L1 canonical $q$ / mesh $k-q$ helper, solves
the regular full-BZ full-first-variation eigenproblem, and returns
[`SpinorProductInput`]:

- `source`: [`DiracProductSource`] with physical reduced $P$ and $Q$ on one
  site mesh, empty cores, `ProductPartition` / `TransferQ` / raw interstitial
  pair support. `DiracRadialId.n` is $n=0$ APW $(P,Q)$, $n=1$ analytic
  $(\dot P,\dot Q)$, and $n=2+\mathrm{ordinal}$ for each compiled
  signed-$\kappa$ LO/RLO in that shell's request order. Identity
  $(site,kind,\kappa,n)$ is $\mu$-degenerate. There is no $cQ$ scaling and
  no collinear `spin=0/1` field.
- `orbitals`: per-$k$ column-major `[basis, band]` eigenvectors, eigenvalues,
  and the exact [`SpinorCompiledBasis`] used by `solve_points`. Live row
  order is two Pauli interstitial blocks $\mathrm{spin}\,N_G+G$ (shared
  spatial $G$ labels), then site confined LO/RLO rows
  $(\kappa, 2\mu, n)$ with $n$ fastest. APW $(P,\dot P)$ are matching
  coefficients on plane-wave rows, not extra eigenbasis rows. Site-projection
  coordinates $(site,\kappa,2\mu,n)$ including $n=0,1$ invert through
  `SpinorProductInput::site_projection_row` and
  `SpinorProductInput::site_projection_identity`.
- `k_minus_q`: folded $k-q_{\mathrm{canonical}}$ mesh index and per-column
  $G_{\mathrm{wrap}}$ with the same positive pair-density convention as M-L1.
- `pair_columns`: `PairColumnLayout::new(n_k, n_orb, None)` with left band
  at $k-q$ and right band at $k$.
- `reciprocal`: the exact snapshot lattice. Off-mesh $q$ is `OffMeshTransfer`.
- `orbitals.band_window`: `{start: 0, count: n_orb}`. `available_bands[k]`
  keeps the untruncated eigenpair count. Eigenvector **rows** equal the
  k-local basis dimension and are not truncated to a common size.

Raw interstitial support is the deduplicated, $|G|$-then-index sorted union
of $G_{\mathrm{right}}-G_{\mathrm{left}}+G_{\mathrm{wrap}}$ over actual
spinor plane-wave $G$ labels. Global `TransferQ` Umklapp is excluded.
`ScfState` is not the orbital source.

M-L5c contracts selected spinor bands into that primitive. M-L5d adds
all-$q$ THC and sampled-$\zeta$ Coulomb.

## 10. M-L5c selected-band spinor mixed-product bridge

`build_spinor_mpb(&SpinorProductInput, &SpinorMpbSpec)` consumes the published
M-L5b bundle and an explicit spec: `product_l_max`, `product_g_max`,
`overlap_tolerance`, and a nonempty list of selections
`(k, left_band, right_band)` inside the M-L5b leading window. There is no
caller lattice and no collinear spin tag. The result owns the untruncated
[`DiracRawProductSpace`], the `TOL`-retained [`CompiledAuxiliaryBasis`] with
$n_{\mathrm{spin}}=1$, the frozen [`ReciprocalLattice`] and
[`PairColumnLayout`], and [`SpinorMpbPairVertex`] records with pair-column
identity $k\cdot N_{\mathrm{orb}}^2+i\cdot N_{\mathrm{orb}}+j$ and a checked
[`PairVertex`] whose identity is [`OrbitalPair::Bloch`]. Construction seals a
runtime-private frozen-input identity of the originating
[`SpinorProductInput`]: the complete Dirac source (partition, $q$,
provenance, site meshes, physical $P$/$Q$ identities and samples, cores, raw
interstitial support), frozen orbitals (k fractions, band window and
available counts, every ordered [`SpinorCompiledBasis`] layout / plane-wave /
site-augmentation / LO mapping, eigenvalues, and every complex eigenvector
entry), the $k-q$ map and per-$k$ wraps, [`PairColumnLayout`], and the
authoritative [`ReciprocalLattice`]. The mixer is the same splitmix-style
64-bit fold as the parent-grid construction fingerprint: an internal binding
stamp, not scientific provenance or a cryptographic digest, with a
per-comparison collision residual of one part in $2^{64}$. External struct
literals cannot forge that stamp.

Dirac overlap spectra are the real-symmetric eigensystems of the ordered
PP then QQ union at each $(site,L)$. Raw products stay sector-tagged; they
are not cast to scalar [`RawProductSpace`] and are not merged into
$(P_i P_j+Q_i Q_j)/r$. `apply_dirac_overlap_cutoff` projects that union into
the retained scalar-charge modes with $\lambda\ge\mathrm{tol}\times n_{\mathrm{spin}}$,
the $L=0$ constant first, then interstitial $|q+G|$ waves. Flatten remains
$site\to L\to M\to n$ then $G$.

Muffin-tin contraction uses [`CompiledSiteProjection::spinor`] and
`SpinorProductInput::site_projection_identity`. The coefficient is
$\mathrm{conj}(d_{\mathrm{left}})d_{\mathrm{right}}$ times the inverse
canonical site phase $\exp(-i q\cdot R_a)$ so the MPB primitive
$+\mathrm{i}q\cdot R_a$ is not double-counted. Large coordinates route only
to PP/$\Omega_\kappa$; small coordinates route only to QQ/$\Omega_{-\kappa}$.
There is no $cQ$ and no PQ/QP.

Interstitial contraction is the same-component Pauli sum

```math
\sum_{s=0,1}
\mathrm{conj}\bigl(C_{\mathrm{left}}[s,G_{\mathrm{left}}]\bigr)
C_{\mathrm{right}}[s,G_{\mathrm{right}}]/\Omega
```

at

```math
G_{\mathrm{rel}} = G_{\mathrm{right}} - G_{\mathrm{left}} + G_{\mathrm{wrap}}.
```

$G_{\mathrm{rel}}$ must exist on the M-L5b raw pair support. The global
`TransferQ` Umklapp is not folded into $G_{\mathrm{rel}}$; it remains the
MPB $\Theta_I$ argument $G_{\mathrm{aux}}-G_{\mathrm{transfer}}-G_{\mathrm{rel}}$.
The MPB accumulator [`DiracBlochVertexAccumulator`] sums those primitive
terms into one checked vertex.

## 11. M-L5d spinor all-$q$ THC and sampled-$\zeta$ Coulomb

`build_spinor_thc(&[SpinorProductInput], &ThcParentGrid, &SpinorThcSpec)`
consumes a complete ordered $q$ slice with one record for every k-mesh
point, sharing frozen orbitals, band window, partition, reciprocal lattice,
and `PairColumnLayout`. `SpinorThcSpec` requires an explicit `RankPolicy`,
[`ThcEngine`] (`FullColumnPivotedQr` or `FullPivotedCholesky`), and
[`ThcCandidates`] (`All` or explicit positive-weight parent indices). There
is no collinear spin tag: one spinor band manifold, with the two Pauli
components summed inside each physical density. The shared parent-grid
fingerprint, candidate policy, and full engines are the same objects as
scalar M-L3. Zero-weight parent rows remain in $\zeta$ and cannot be
candidates.

Muffin-tin reconstruction uses [`CompiledSiteProjection::spinor`] and
`SpinorProductInput::site_projection_identity` on the stored mesh shell.
Large $P/\dot P$/LO-RLO uses $\Omega_{\kappa\mu}$; physical small
$Q/\dot Q$/LO-RLO uses $\Omega_{-\kappa\mu}$. Pair density is the
same-Pauli PP plus QQ sum; there is no PQ/QP and no $cQ$. Each reconstructed
Bloch spinor is converted to cell-periodic form by one
$\exp(-i k\cdot r)$, then the stored per-$k$ $+G_{\mathrm{wrap}}$ pair
phase. Global `TransferQ` Umklapp is not applied again. Interstitial
evaluation uses the two Pauli plane-wave blocks

```math
\sum_{s=0,1}
\mathrm{conj}\bigl(C_{\mathrm{left}}[s,G_{\mathrm{left}}]\bigr)
C_{\mathrm{right}}[s,G_{\mathrm{right}}]/\Omega
```

with G-only cell-periodic phases; there is no interstitial small component.
Selection and $\zeta$ reuse `mt-thc::fit_allq_l2_pair_blocks`. Auxiliaries
are created with the intended provenance before Bloch pair vertices.

`build_spinor_coulomb` builds `SampledAuxiliaryFunctions` on the full
`ThcParentGrid` in original order (Cartesian Bohr coordinates, true weights
including zero, MT `{site,radial_index}` / interstitial support, per-site
meshes, row-major parent-grid $\zeta$) and calls production
`assemble_sampled_coulomb`. The request reciprocal must equal the frozen
`SpinorProductInput` reciprocal. Gamma keeps the finite body plus
`GammaHead` metadata. Explicit bounded matches to `SpinorMpbResult` must
originate from `inputs[q_index]`: the sealed frozen-input identity is
checked, then the public reciprocal lattice and pair-column layout, before
mixed-product Coulomb assembly. THC vertices must carry the compiled
auxiliary provenance at the shared record helper. Matched pairs compare
$c^\dagger V c$ with absolute/relative discrepancy; per-side $\|Vc\|$ remain
debug numbers in each auxiliary basis. This stage does not implement
core-valence products, HDF5/CoQui, material/SPEX acceptance, a
principal-angle engine, or GW/RPA.

## M-L6b2 scalar MLDUMP materialization

`write_scalar_mldump(path, header, inputs, thc, coulomb, spec)` is the
runtime-owned writer. The caller supplies `MldumpHeaderV1` because species
and labels cannot be reconstructed from `ScalarProductInput`.
`build_scalar_coulomb` seals the effective request and interpolation
projection inside `ScalarCoulombResult`. Before the HDF5 file is created,
runtime preflights the header cell/reciprocal/volume, ordered sites, radii,
radial meshes, full-BZ $k$ with uniform weights $1/n_k$, exact $q$
input/canonical/global Umklapp and per-$k$ wraps, the shared $q$-slice
contract, and the crate-private Coulomb export-context guard: the passed
spec must match that sealed request/projection, each Coulomb $q$ record must
bind `inputs[q]` and the accepted THC $q$/layout/auxiliary/vertices with a
sampled-$\zeta$ interpolation-point operator on that auxiliary layout, and the
THC strategy/engine plus Coulomb projection metadata must be serializable. A
tampered header or result is rejected with no output file.

The on-disk write uses `ScalarMldumpStreamV1`. Conversion scratch is bounded
to one $k$ eigenvector/APW-matching record, one $q$ $\zeta$/vertex record, or
one $q$ $V$/Gamma record. `/mpb` and `/exchange/{valence,core,total}` remain
`absent_not_computed`. Occupations are not invented. The owned reader and
MLDUMP v1 tree are unchanged.

## M-L6c2 spinor MLDUMP materialization

`write_spinor_mldump(path, header, inputs, thc, coulomb, spec)` is the
runtime-owned spinor writer. The caller supplies `MldumpHeaderV1` because
species and labels cannot be reconstructed from `SpinorProductInput`.
`build_spinor_coulomb` seals the effective request and interpolation
projection inside `SpinorCoulombResult`. Sampled-$\zeta$ $V^q$ records are
crate-private and exposed only through the read-only `records()` accessor.
Before the HDF5 file is created, runtime preflights the header
cell/reciprocal/volume, ordered sites, radii, radial meshes, full-BZ $k$
with uniform weights $1/n_k$, exact $q$ input/canonical/global Umklapp and
per-$k$ wraps, the shared spinor $q$-slice contract (canonical $q$ Cartesian
equals $k_{\mathrm{frac}}[q]$ Cartesian at scale-aware $10^{-12}$, and
$k_{\mathrm{frac}}[k]-q_{\mathrm{canonical}}=k_{\mathrm{frac}}[\mathrm{mapped}]+G_{\mathrm{wrap}}$
using stored fractional indices; global `TransferQ` Umklapp is not a per-$k$
label), every frozen $k$ compiled basis and site augmentation
(site count, geometry, plane-wave count, native signed-$\kappa$ then
$2\mu$ channels, complete Pauli coefficient arrays), and the crate-private
Coulomb export-context guard: the passed spec must match that sealed
request/projection, each Coulomb $q$ record must bind `inputs[q]` and the
accepted THC $q$/layout/auxiliary/vertices with a sampled-$\zeta$
interpolation-point operator on that auxiliary layout, and the THC
strategy/engine plus Coulomb projection metadata must be serializable. A
tampered header, compiled basis, wrap, or replacement operator is rejected
with no output file. There is no MPB argument; `/mpb` stays absent.

The on-disk write uses `SpinorMldumpStreamV1` and the neutral `MldumpThc*` /
`MldumpCoulomb*` DTOs. Conversion scratch is bounded to one $k$
eigenvector/site-matching record, one site radial record, one $q$
$\zeta$/vertex record, or one $q$ $V$/Gamma record. Occupations are not
invented. The MLDUMP v1 tree is unchanged.

## M-L6d1 CoQui-native scalar Cholesky adapter

`write_scalar_coqui_cholesky(path, inputs, thc, coulomb, coulomb_spec, factor)`
writes a **CoQui-native** single-file Cholesky ERI. It is not MLDUMP, not a
SPEX dump, and does not claim that q-dependent THC/MLDUMP is CoQui-compatible.
The on-disk tree follows live CoQui `chol_reader_t`
(`<coqui-inspect-checkout>`, `wg-dev` @
`a19774d03fb979bd852fae4f7f95c045a4cbca78`): `/Interaction` scalars
`Np,nspin,nspin_in_basis,nkpts,nbnd,nbnd_aux=0,tol`, Cartesian `kpts`/`qpts`,
`qk_to_kmq`, and `Vq{iq}` as native `f64` `[Np,1,nk,nbnd,nbnd,2]` with
scalar variable-length UTF-8 `__complex__="1"`. `qpts` store canonical Cartesian $q$ without a second
global Umklapp. Gamma contributes only the finite body. Each accepted
$V_q$ is factored as $V_q=B_q^\dagger B_q$ (Rust $B$ is row-major
$(\mathrm{rank},n_{\mathrm{aux}})$) with `factor.tolerance` as the rank /
small-negative roundoff policy written to `/Interaction/tol`. Semantic
vertices map as $L_{Q,0,k,i,j}=(B_q c_{k,i,j})_Q$ in k-major pair order,
matching GF2 $\sum_Q L_{Qpr}\mathrm{conj}(L_{Qsq})$. `libmuffintin-io`
owns the native HDF tree; runtime owns mapping and factorization. Scratch
is one $q$ factor and one $q$ $L$ tensor.
