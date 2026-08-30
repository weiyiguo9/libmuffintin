# 18. Scalar product/MPB/THC/Coulomb and spinor product, mixed-product, and THC paths

This note records the implemented scalar product-input boundary, the scalar
mixed-product bridge, the scalar AllQL2 interpolation-point seam, the sampled
$\zeta$ Coulomb bridge, the Dirac PP/QQ IR and MPB primitive, the frozen
full-first-variation spinor product input, the selected-band spinor
mixed-product bridge, and the runtime materialization of those scalar objects
into MLDUMP v1. Core–valence products and SPEX/material comparison remain out
of scope. The on-disk schema is [19](19_versioned_mldump_interchange.md).

Product kinematics remain [13](13_product_space_and_lapw_mpb.md). The
canonical $q$ / Umklapp pair gauge remains [14](14_kpoint_isdf_thc.md). The
checkpoint-backed DFT/SCF kernel boundary remains [17](17_minimal_dft_scf.md).

The material is organized in three layers: formulas (section 1, the mathematical
contracts), algorithms (section 2, the method-neutral procedures), and
implementation (section 3, the concrete crate/type/function binding). Section 4
collects the explicit scope exclusions.

## 1. Formulas

### 1.1 Canonical $q$ and Umklapp

The requested transfer $q_{\mathrm{in}}$ folds into the primitive cell
$[0,1)^3$. The canonical Cartesian $q$ and the subtracted reciprocal vector
$G_{\mathrm{transfer}}$ satisfy

```math
q_{\mathrm{in}} = q_{\mathrm{canonical}} + G_{\mathrm{transfer}}.
```

$k-q$ mapping uses $q_{\mathrm{canonical}}$ only. A folded target that is not
an existing regular-mesh coordinate (within $10^{-12}$) is rejected; the
kernel does not round onto a neighbouring mesh point. $q_{\mathrm{in}}=(1.5,0,0)$
on a $2\times 1\times 1$ mesh is valid because
$q_{\mathrm{canonical}}=(0.5,0,0)$. $q_{\mathrm{in}}=(0.25,0,0)$ is not.

The per-column wrap $G_{\mathrm{wrap}}$ satisfies

```math
k_{\mathrm{frac}} - q_{\mathrm{canonical,frac}}
= (k-q)_{\mathrm{frac}} + G_{\mathrm{wrap,index}},
```

and the pair phase is

```math
\exp(+i G_{\mathrm{wrap}}\cdot r).
```

Raw interstitial support enumerates, for every $k\to k-q$ pair and both spin
channels, every

```math
G_{\mathrm{raw}} = G_{k} - G_{k-q} + G_{\mathrm{wrap}}
```

from the actual left/right plane-wave $G$ labels, deduplicated and ordered
first by $|G|$ and then by $G$ index. That list includes per-column $G_{\mathrm{wrap}}$ and
excludes the global $G_{\mathrm{transfer}}$ wrap. A sign error in $G_{\mathrm{wrap}}$
reverses $\exp(+i G\cdot R)$ at a muffin-tin site whose $G\cdot R$ is not a
multiple of $\pi$.

This is the production $k-q$ gauge; there is no second $k+q$ convention
anywhere in this note.

### 1.2 Scalar pair density, phase, and interstitial kernel

Physical Koelling–Harmon samples are $P=p/r$ and optional $Q=q/r$. The pair
density is

```math
\mathrm{conj}(P_{\mathrm{left}})P_{\mathrm{right}}
+ \mathrm{conj}(Q_{\mathrm{left}})Q_{\mathrm{right}}.
```

The reconstructed muffin-tin Bloch value converts to the cell-periodic
representation by $\exp(-i k\cdot r)$ at the Cartesian point, using the stored
plane-wave Cartesian $k$, so muffin-tin and interstitial orbitals share one
gauge. Interstitial evaluation uses the plane-wave large component
$C_G\exp(i G\cdot r)/\sqrt{\Omega}$, giving interstitial contraction amplitude

```math
\mathrm{conj}(C_{\mathrm{left}})C_{\mathrm{right}}/\Omega
```

at relative label

```math
G_{\mathrm{rel}} = G_{\mathrm{right}} - G_{\mathrm{left}} + G_{\mathrm{wrap}},
```

which must exist on the scalar product-input raw pair support. The global
$G_{\mathrm{transfer}}$ Umklapp is not folded into $G_{\mathrm{rel}}$; it remains the MPB
$\Theta_I$ argument

```math
G_{\mathrm{aux}} - G_{\mathrm{transfer}} - G_{\mathrm{rel}}.
```

Muffin-tin contraction weights every coefficient pair as
$\mathrm{conj}(C_{\mathrm{left}})C_{\mathrm{right}}$ times the inverse
canonical site phase $\exp(-i q\cdot R_a)$, so the primitive MPB
$+iq\cdot R_a$ kernel is not double-counted against matching phases already
stored on the projection.

Valence radial index $n$ is stable across the scalar path: $n=0$ is $u$,
$n=1$ is $\dot u$, and $n=2+\mathrm{ordinal}$ are local orbitals. The scalar
product-input path does not add $\kappa$, $PP$, or $QQ$.

### 1.3 Spinor pair density, sectors, and cutoff

`DiracRadialId.n` follows the same stability convention: $n=0$ is the APW
$(P,Q)$ pair, $n=1$ is the analytic $(\dot P,\dot Q)$ pair, and
$n=2+\mathrm{ordinal}$ enumerates each compiled signed $\kappa$ LO/RLO in that
shell's request order. Identity $(site,kind,\kappa,n)$ is degenerate in
$\mu$; there is no $cQ$ scaling and no collinear spin field.

Dirac overlap spectra are the real-symmetric eigensystems of the ordered PP
then QQ union at each $(site,L)$. Raw products stay sector-tagged
(`DiracChargeSector::{LargeLarge, SmallSmall}`); they are not cast to the
scalar raw-product representation and are not merged into
$(P_iP_j+Q_iQ_j)/r$. The overlap cutoff keeps the modes

```math
\lambda \ge \mathrm{tol}\times n_{\mathrm{spin}},
```

with the $L=0$ constant first, then interstitial $|q+G|$ waves. Large
coordinates route only to $PP/\Omega_\kappa$; small coordinates route only to
$QQ/\Omega_{-\kappa}$. There is no $cQ$ and no $PQ/QP$ mixing.

Spinor interstitial contraction is the same-component Pauli sum

```math
\sum_{s=0,1}
\mathrm{conj}\bigl(C_{\mathrm{left}}[s,G_{\mathrm{left}}]\bigr)
C_{\mathrm{right}}[s,G_{\mathrm{right}}]/\Omega
```

at the same $G_{\mathrm{rel}} = G_{\mathrm{right}} - G_{\mathrm{left}} +
G_{\mathrm{wrap}}$, which must exist on the spinor product-input raw pair
support; the global $G_{\mathrm{transfer}}$ Umklapp again stays out of $G_{\mathrm{rel}}$
and remains the MPB $\Theta_I$ argument. Pair density for the spinor THC path
is the same-Pauli PP plus QQ sum; there is no PQ/QP and no $cQ$.

### 1.4 Coulomb comparison metric

Matched mixed-product/THC pairs (scalar or spinor) compare quadratic forms
$c^\dagger V c$ across representations, with an absolute/relative discrepancy
using the stated floor $10^{-12}$. Per-side action norms $\|Vc\|$ are debug
diagnostics in each auxiliary basis and are not compared across
representations; elementwise MPB/THC matrices and principal-angle spans are
not compared.

### 1.5 CoQui Cholesky factorization

Each accepted $V_q$ is factored as $V_q=B_q^\dagger B_q$ (Rust $B$ is
row-major $(\mathrm{rank},n_{\mathrm{aux}})$) with `factor.tolerance` as the
rank / small-negative roundoff policy. Semantic vertices map as

```math
L_{Q,0,k,i,j}=(B_q c_{k,i,j})_Q
```

in k-major pair order, matching the GF2 identity
$\sum_Q L_{Qpr}\mathrm{conj}(L_{Qsq})$.

## 2. Algorithms

### 2.1 Scalar product-input construction

The input is a validated V2 checkpoint, an `ScfConfig` whose relativity is
scalar Koelling–Harmon, and a requested transfer in primitive reciprocal
coordinates $q_{\mathrm{in}}$. The bridge materializes the frozen-checkpoint
iteration basis, solves the regular full-BZ scalar eigenproblem, folds $q$ by
section 1.1, and rejects a folded $k-q$ that is not on the regular mesh. It then
assembles one bundle carrying: the exact scalar iteration bases (partition,
per-site radial mesh, valence $P$ samples with optional $Q$, empty cores,
finite raw interstitial pair support of relative $G$ labels); per-spin,
per $k$ column-major eigenvectors together with the exact compiled basis used
by the solve (plane-wave $G$ labels, APW $(u,\dot u)$ matching coefficients,
confined LO layout) — this bundle does not carry coefficients or the compiled
basis on the orbital-pair source itself; the folded $k-q_{\mathrm{canonical}}$
mesh index and per-column $G_{\mathrm{wrap}}$; a pair-column layout keyed by
$(k,i,j)$; the exact reciprocal lattice used to fold $q$; and a common leading
band window with per $k$ available-band counts. Spin labels are $0$ (up) and
$1$ (down). SCF state is not the orbital source; the runtime bridge consumes
each solved k-point's band solution directly.

### 2.2 Scalar mixed-product assembly

The bridge consumes the published scalar product-input bundle and an
explicit spec (reciprocal lattice, product angular/reciprocal cutoffs,
overlap tolerance, a nonempty list of same-spin band selections). The left
orbital is the mapped $k-q$ side; the right orbital is at $k$, both drawn from
the published common leading window. Muffin-tin contraction visits every APW
$u$, APW $\dot u$, and LO site coordinate present in the exact per $k$
compiled basis and applies the coefficient/phase convention of section 1.2; terms
absent from the raw radial products (triangle/parity) are skipped, but $u$,
$\dot u$, and LO channels that exist in the descriptor are not dropped.
Interstitial contraction sums the PW-only rows at $G_{\mathrm{rel}}$ from
section 1.2. One accumulator sums those primitive terms into a single checked
vertex per selection; the bridge does not build a complete primitive vertex
per basis-pair term. Empty selection, a spin/$k$/band outside the frozen
input, or an incompatible pair-column layout is a typed stage-boundary
rejection.

### 2.3 Scalar adaptive THC selection and fit

The bridge consumes a nonempty complete $q$ slice in production $q$ index
order, where every entry is the scalar product-input bundle whose canonical
$q$ is that mesh point; all entries share the frozen orbital window, spins,
$k$ mesh, partition, radials, and pair-column layout. The spec selects one
collinear spin, an existing THC rank policy, a candidate-point policy (either
the full positive-weight parent set in parent order, with explicit
zero-weight indices rejected, or an explicit positive-weight subset), and one
production L2 engine (full column-pivoted QR or full pivoted Cholesky). Both
engines run AllQL2 selection on the same ordered pair blocks, positive-weight
candidates, true weights, and rank policy, sharing one result type; pivoted
Cholesky does not form the dense point Gram but still materializes the
stacked weighted pair matrix. Core-orbital diagnostics stay empty in this
path.

The parent grid is an externally supplied immutable parent support bound to
the exact product partition: muffin-tin points name a site and radial index
on the stored exponential mesh and must lie on that sample's Cartesian
radius; interstitial points are partitioned interstitial. Weights may be
zero if at least one is positive; they are never clamped and are not L2
selection candidates; there is no radial interpolation. Orbitals are
evaluated on that parent grid using the muffin-tin and interstitial kernels
of section 1.2. Interpolation-point auxiliaries are created with the scalar THC
provenance before the Bloch pair vertices are built, so auxiliary and vertex
records bind the same $q$, layout, partition, and provenance at construction.
This stage does not build sampled auxiliary functions or Coulomb operators.

### 2.4 Scalar sampled $\zeta$ Coulomb assembly

The bridge consumes the same complete ordered $q$ slice, a matching THC
result, an existing Coulomb request plus interpolation projection, and an
explicit bounded subset of matching scalar mixed-product vertices. The
request reciprocal lattice must equal the frozen product-input reciprocal;
it is not inferred from a caller-supplied cell recipe. For every THC $q$
record, the bridge builds the sampled auxiliary functions on the full THC
parent grid in original order (Cartesian coordinates, true weights including
zero-weight rows, exact muffin-tin/interstitial support, per-site meshes, and
that record's row-major parent-grid $\zeta$), binds a private construction
fingerprint to that ordered grid so a permuted or substituted parent grid is
rejected, and calls the production Weinert/SPEX Coulomb assembler. Gauge is
unchanged through this step: no extra pair rephasing and no second global
Umklapp insertion; Gamma keeps the finite body plus head metadata, and the
singular head is not inserted.

Each per $q$ record is validated before assembly: returned scalar THC
vertices must carry the compiled auxiliary layout and a Bloch identity that
matches the decoded column, and malformed public Bloch indices are rejected
without decoding them. Matched mixed-product/THC pairs are then compared by
the section 1.4 metric.

### 2.5 Frozen spinor product-input construction

Only `ScfRelativity::SpinorFirstVariation` is accepted; scalar Koelling–Harmon
and SOC second variation are distinct typed rejections, and signed $\kappa$
is not routed through second variation. The bridge materializes the
frozen-checkpoint iteration basis, solves the regular full-BZ
full-first-variation eigenproblem, reuses the scalar path's canonical $q$ /
mesh $k-q$ folding (section 1.1), and assembles a bundle carrying: the Dirac
product source with physical reduced $P$ and $Q$ on one site mesh, empty
cores, partition, $G_{\mathrm{transfer}}$, and raw interstitial pair support, and the radial
identity convention of section 1.3; per $k$ column-major eigenvectors, eigenvalues,
and the exact compiled spinor basis used by the solve, with row order two
Pauli interstitial blocks (shared spatial $G$ labels) followed by site
confined LO/RLO rows $(\kappa,2\mu,n)$ with $n$ fastest — APW $(P,\dot P)$ are
matching coefficients on plane-wave rows, not extra eigenbasis rows; the
folded $k-q_{\mathrm{canonical}}$ mesh index and per-column
$G_{\mathrm{wrap}}$ with the same positive pair-density convention as the
scalar path; a pair-column layout with left band at $k-q$ and right band at
$k$; and the exact checkpoint lattice. Off-mesh $q$ is rejected the same way
as the scalar path. Eigenvector rows equal the basis dimension local to $k$ and
are not truncated to a common size. Raw interstitial support is the
deduplicated union, sorted by $|G|$ then index, of
$G_{\mathrm{right}}-G_{\mathrm{left}}+G_{\mathrm{wrap}}$ over actual spinor
plane-wave $G$ labels; the global $G_{\mathrm{transfer}}$ Umklapp is excluded. SCF state
is not the orbital source.

### 2.6 Selected-band spinor mixed-product assembly

The bridge consumes the published spinor product-input bundle and an
explicit spec (product angular/reciprocal cutoffs, overlap tolerance, a
nonempty list of band selections inside the spinor leading window); there is
no caller lattice and no collinear spin tag. Construction seals a
runtime-private frozen-input identity of the originating spinor product
input — the complete Dirac source, frozen orbitals (k fractions, band window
and available counts, every ordered compiled-basis layout, eigenvalues, and
every complex eigenvector entry), the $k-q$ map and per $k$ wraps, the
pair-column layout, and the authoritative reciprocal lattice — using the same
splitmix-style 64-bit fold as the parent-grid construction fingerprint of
section 2.3: an internal binding stamp, not scientific provenance or a
cryptographic digest, with a per-comparison collision residual of one part
in $2^{64}$; external struct literals cannot forge that stamp.

The overlap cutoff of section 1.3 projects the ordered PP/QQ union into retained
scalar-charge modes. Muffin-tin contraction applies the coefficient/phase
convention of section 1.2 (adapted to the Dirac coefficient pair) with routing by
component sector; interstitial contraction sums the Pauli terms of section 1.3 at
$G_{\mathrm{rel}}$, which must exist on the spinor raw pair support. One
accumulator sums those primitive terms into a single checked vertex per
selection.

### 2.7 Spinor all $q$ THC and sampled $\zeta$ Coulomb

The bridge consumes a complete ordered $q$ slice with one record for every
$k$ mesh point, sharing frozen orbitals, band window, partition, reciprocal
lattice, and pair-column layout; there is no collinear spin tag; one spinor
band manifold sums the two Pauli components inside each physical density.
The spec requires an explicit rank policy, one of the two full L2 engines,
and a candidate policy (the full positive-weight parent set, or an explicit
positive-weight subset); zero-weight parent rows remain in $\zeta$ and cannot
be candidates. The shared parent-grid fingerprint, candidate policy, and full
engines are the same objects as the scalar path (section 2.3). Muffin-tin
reconstruction routes large $P/\dot P$ and LO-RLO through $\Omega_{\kappa\mu}$
and physical small $Q/\dot Q$ and LO-RLO through $\Omega_{-\kappa\mu}$; each
reconstructed Bloch spinor converts to cell-periodic form by one
$\exp(-ik\cdot r)$ then the stored per $k$ wrap phase, with the global
$G_{\mathrm{transfer}}$ Umklapp not applied again. Interstitial evaluation uses the
Pauli sum of section 1.3 with G-only cell-periodic phases; there is no
interstitial small component. Selection and $\zeta$ reuse the same
k-point-ISDF/AllQL2 fit as the scalar path ([14](14_kpoint_isdf_thc.md));
auxiliaries are created with the intended provenance before the Bloch pair
vertices.

Sampled $\zeta$ Coulomb assembly follows section 2.4: sampled auxiliary functions on
the full parent grid in original order, the frozen-request reciprocal check,
and the finite-body-plus-head Gamma treatment. Explicit bounded matches to
the spinor mixed-product result must originate from that $q$'s frozen input:
the sealed frozen-input identity of section 2.6 is checked, then the public
reciprocal lattice and pair-column layout, before mixed-product Coulomb
assembly; THC vertices must carry the compiled auxiliary provenance at the
shared record helper. Matched pairs are compared by the section 1.4 metric.

## 3. Implementation

### 3.1 Packages and kernel ownership

| Directory | Package |
|---|---|
| `crates/mt-dft` | `libmuffintin-dft` (`muffintin_dft`) |
| `crates/mt-runtime` | `libmuffintin-runtime` (`muffintin`) |
| `crates/mt-prodbasis` | `libmuffintin-prodbasis` (root IR plus `mpb::` and `thc::`) |
| `crates/mt-coulomb` | `libmuffintin-coulomb` (`muffintin_coulomb`) |

`libmuffintin-dft::MaterialKernel` owns the checkpoint-backed scalar and
full-spinor one-particle physics. Its public `solve_points` entry point
dispatches the scalar, SOC-second-variation, or full-spinor route. Runtime
`CheckpointPhysics` is the checkpoint/IO/orchestration/product-space bridge
shell and delegates those solves to its kernel. `CheckpointPhysics` remains
the runtime-owned home for `scalar_product_input` and
`spinor_product_input`, which consume [`ProductSource`] /
[`DiracProductSource`] and [`PairColumnLayout`] from the
`libmuffintin-prodbasis` root IR. `build_scalar_mpb` owns the scalar
mixed-product capability over `muffintin_prodbasis::mpb`; `build_scalar_thc`
owns the scalar AllQL2 THC capability over `muffintin_prodbasis::thc`;
`build_scalar_coulomb` owns the sampled $\zeta$ Coulomb capability and
depends on `libmuffintin-coulomb`; `build_spinor_mpb` owns the spinor
mixed-product capability. Neither `libmuffintin-prodbasis` nor
`libmuffintin-coulomb` depends on runtime, and Coulomb takes only root IR
types as public inputs; keeping `mpb::`/`thc::` types out of the Coulomb
surface is a documented convention now that the three product-space crates
share one package. The MLDUMP and CoQui writers also remain runtime-owned;
the material kernel's home in `libmuffintin-dft` does not move either writer
or either product-input bridge into DFT.

### 3.2 Scalar product-input types

`CheckpointPhysics::scalar_product_input` emits [`ScalarProductInput`]:

- `source`: [`ProductSource`] built from the exact scalar iteration bases
  (`ProductPartition` / `InterstitialGeometry`, per-site `ExponentialMesh`,
  valence $p=ru$ samples with optional Koelling–Harmon $Q$, empty cores,
  finite raw interstitial pair support of relative $G$ labels).
- `orbitals`: per-spin, per $k$ column-major `[basis, band]` eigenvectors
  together with the exact [`CompiledBasis`] used by `solve_points` (plane-wave
  $G$ labels, APW $(u,\dot u)$ matching coefficients, confined LO layout).
  Spin labels are `0` (up) and `1` (down). This bundle does not put
  coefficients or `CompiledBasis` on [`ProductSource`].
- `k_minus_q`: folded $k-q_{\mathrm{canonical}}$ mesh index and the
  per-column [`ReciprocalLattice`] wrap $G_{\mathrm{wrap}}$.
- `pair_columns`: `PairColumnLayout::new(n_k, n_orb, None)` from
  `muffintin_prodbasis`, keyed $k\cdot N_{\mathrm{orb}}^2+i\cdot
  N_{\mathrm{orb}}+j$. The old packed $12\times 12$ experiment flattening is
  not used.
- `reciprocal`: the exact [`ReciprocalLattice`] used to fold $q_{\mathrm{in}}$
  and $G_{\mathrm{wrap}}$. Sampled $\zeta$ Coulomb requires `CoulombRequest`
  to carry this lattice; a same-volume sheared cell is rejected.
- `orbitals.band_window`: common leading window `{start: 0, count: n_orb}`.
  Each spin channel also stores `available_bands[k]`, the untruncated
  eigenpair count at that $k$.

Valence [`ProductRadial`] follows the $n$ convention of section 1.2; local-orbital
rows follow [`BasisLayout::site_local_orbital_range`]. `ProductRadialId`
remains the identifier based on scalar $l$.

`ScfState` is not the orbital source. `CheckpointBandSolution` exposes its
ordered k-point slice so the runtime bridge can consume each scalar or
spinor solution payload; state, weight, and energy bookkeeping remain
encapsulated by `MaterialKernel`.

### 3.3 Scalar mixed-product types and entry point

`build_scalar_mpb(&ScalarProductInput, &ScalarMpbSpec)` takes the reciprocal
lattice required by `spex_mixed_product_basis`, `product_l_max`,
`product_g_max`, `overlap_tolerance`, and a nonempty list of same-spin
selections `(spin, k, left_band, right_band)`. The result owns the
untruncated [`RawProductSpace`], the `TOL`-retained [`CompiledAuxiliaryBasis`]
with $n_{\mathrm{spin}}=2$, and [`ScalarMpbPairVertex`] records that keep
spin, pair-column identity, band indices, and a checked [`PairVertex`]. The
vertex identity is [`OrbitalPair::Bloch`]; spin is stored on the runtime
record, not on the shared [`OrbitalPair`] model. Muffin-tin contraction uses
[`CompiledSiteProjection`]; the accumulator is [`PairVertexAccumulator`].
Empty selection, a spin/$k$/band outside the frozen input, or an incompatible
pair-column layout is a typed stage-boundary error.

### 3.4 Scalar THC types and entry point

`build_scalar_thc` takes `inputs[iq]`, the scalar product-input bundle whose
canonical $q$ is k-mesh point $iq$, and `ScalarThcSpec`, which selects one
collinear spin, an existing THC `RankPolicy`, candidate points
([`ThcCandidates`]), and one production L2 engine ([`ThcEngine`],
`FullColumnPivotedQr` or `FullPivotedCholesky`). `ThcParentGrid` is the
externally supplied immutable parent support bound to the exact
`ProductPartition`. The result carries the selected spin, the parent grid,
selection with requested and effective rank, and one per $q$
interpolation-point `CompiledAuxiliaryBasis`, parent-grid-by-selected-point
$\zeta$, and `PairVertex` columns.

### 3.5 Scalar Coulomb types and entry point

`build_scalar_coulomb` takes the complete ordered $q$ slice, a matching
`ScalarThcResult`, an existing `CoulombRequest` plus
`InterpolationProjection`, and an explicit bounded subset of matching
`ScalarMpbResult` vertices. It calls production `assemble_sampled_coulomb`.
Each per $q$ record returns the sampled $\zeta$ `CoulombOperator` with $q$
index, `TransferQ`, spin, pair-column layout, interpolation-point auxiliary,
parent-grid sampled $\zeta$, and semantic pair vertices.

### 3.6 Dirac PP/QQ IR and MPB primitive

The Dirac PP/QQ IR (`DiracProductSource`, physical $P$ and $Q$, explicit
`DiracChargeSector::{LargeLarge, SmallSmall}`) and the MPB primitive
(untruncated PP/QQ raw products and the checked muffin-tin vertex primitive)
both live in `libmuffintin-prodbasis`.
The spinor product-input path consumes those public types and does not
redefine them. Scalar `ProductRadialId` / `ProductSource` / `RawProductSpace`
stay unchanged. The spinor mixed-product bridge adds Dirac overlap cutoff,
retained scalar-charge modes of the PP/QQ union, and interstitial spinor
plane-wave contraction. The spinor THC/Coulomb bridge adds spinor all $q$
THC and sampled $\zeta$ Coulomb.

### 3.7 Spinor product-input types

`CheckpointPhysics::spinor_product_input(&ScfConfig, q_fractional)` emits
[`SpinorProductInput`]:

- `source`: [`DiracProductSource`] with physical reduced $P$ and $Q$ on one
  site mesh, empty cores, `ProductPartition` / `TransferQ` / raw interstitial
  pair support. `DiracRadialId.n` follows the section 1.3 convention.
- `orbitals`: per $k$ column-major `[basis, band]` eigenvectors, eigenvalues,
  and the exact [`SpinorCompiledBasis`] used by `solve_points`. Site-projection
  coordinates $(site,\kappa,2\mu,n)$ including $n=0,1$ invert through
  `SpinorProductInput::site_projection_row` and
  `SpinorProductInput::site_projection_identity`.
- `k_minus_q`: folded $k-q_{\mathrm{canonical}}$ mesh index and per-column
  $G_{\mathrm{wrap}}$.
- `pair_columns`: `PairColumnLayout::new(n_k, n_orb, None)` with left band
  at $k-q$ and right band at $k$.
- `reciprocal`: the exact checkpoint lattice. Off-mesh $q$ is
  `OffMeshTransfer`.
- `orbitals.band_window`: `{start: 0, count: n_orb}`. `available_bands[k]`
  keeps the untruncated eigenpair count.

### 3.8 Spinor mixed-product types and entry point

`build_spinor_mpb(&SpinorProductInput, &SpinorMpbSpec)` takes
`product_l_max`, `product_g_max`, `overlap_tolerance`, and a nonempty list of
selections `(k, left_band, right_band)`. The result owns the untruncated
[`DiracRawProductSpace`], the `TOL`-retained [`CompiledAuxiliaryBasis`] with
$n_{\mathrm{spin}}=1$, the frozen [`ReciprocalLattice`] and
[`PairColumnLayout`], and [`SpinorMpbPairVertex`] records with pair-column
identity and a checked [`PairVertex`] whose identity is [`OrbitalPair::Bloch`].
Muffin-tin contraction uses [`CompiledSiteProjection::spinor`] and
`SpinorProductInput::site_projection_identity`; the accumulator is
[`DiracBlochVertexAccumulator`].

### 3.9 Spinor THC/Coulomb types and entry points

`build_spinor_thc(&[SpinorProductInput], &ThcParentGrid, &SpinorThcSpec)`
takes an explicit `RankPolicy`, [`ThcEngine`], and [`ThcCandidates`].
Muffin-tin reconstruction uses [`CompiledSiteProjection::spinor`] and
`SpinorProductInput::site_projection_identity`; selection and $\zeta$ reuse
`muffintin_prodbasis::thc::fit_allq_l2_pair_blocks`.

`build_spinor_coulomb` builds `SampledAuxiliaryFunctions` on the full
`ThcParentGrid` and calls production `assemble_sampled_coulomb`. This stage
does not implement core-valence products, HDF5/CoQui, material/SPEX
acceptance, a principal-angle engine, or GW/RPA.

### 3.10 Scalar MLDUMP materialization

`write_scalar_mldump(path, header, inputs, thc, coulomb, spec)` is the
runtime-owned writer. The caller supplies `MldumpHeaderV1` because species
and labels cannot be reconstructed from `ScalarProductInput`.
`build_scalar_coulomb` seals the effective request and interpolation
projection inside `ScalarCoulombResult`. Before the HDF5 file is created,
runtime preflights the header cell/reciprocal/volume, ordered sites, radii,
radial meshes, full-BZ $k$ with uniform weights $1/n_k$, exact $q$
input/canonical/global Umklapp and per $k$ wraps, the shared $q$ slice
contract, and the crate-private Coulomb export-context guard: the passed spec
must match that sealed request/projection, each Coulomb $q$ record must bind
`inputs[q]` and the accepted THC $q$, layout, auxiliary, and vertices with a
sampled $\zeta$ interpolation-point operator on that auxiliary layout, and the
THC strategy/engine plus Coulomb projection metadata must be serializable. A
tampered header or result is rejected with no output file.

The on-disk write uses `ScalarMldumpStreamV1`. Conversion scratch is bounded
to one $k$ eigenvector/APW-matching record, one $q$ $\zeta$/vertex record, or
one $q$ $V$/Gamma record. `/mpb` and `/exchange/{valence,core,total}` remain
`absent_not_computed`. Occupations are not invented. The owned reader and
MLDUMP v1 tree are unchanged.

### 3.11 Spinor MLDUMP materialization

`write_spinor_mldump(path, header, inputs, thc, coulomb, spec)` is the
runtime-owned spinor writer. The caller supplies `MldumpHeaderV1` because
species and labels cannot be reconstructed from `SpinorProductInput`.
`build_spinor_coulomb` seals the effective request and interpolation
projection inside `SpinorCoulombResult`. Sampled $\zeta$ $V^q$ records are
crate-private and exposed only through the read-only `records()` accessor.
Before the HDF5 file is created, runtime preflights the header
cell/reciprocal/volume, ordered sites, radii, radial meshes, full-BZ $k$
with uniform weights $1/n_k$, exact $q$ input/canonical/global Umklapp and
per $k$ wraps, the shared spinor $q$ slice contract (canonical $q$ Cartesian
equals $k_{\mathrm{frac}}[q]$ Cartesian at scale-aware $10^{-12}$, and
$k_{\mathrm{frac}}[k]-q_{\mathrm{canonical}}=k_{\mathrm{frac}}[\mathrm{mapped}]+G_{\mathrm{wrap}}$
using stored fractional indices; global $G_{\mathrm{transfer}}$ Umklapp is not a per $k$
label), every frozen $k$ compiled basis and site augmentation (site count,
geometry, plane-wave count, native signed $\kappa$ then $2\mu$ channels,
complete Pauli coefficient arrays), and the crate-private Coulomb
export-context guard described in section 3.10. A tampered header, compiled basis,
wrap, or replacement operator is rejected with no output file. There is no
MPB argument; `/mpb` stays absent.

The on-disk write uses `SpinorMldumpStreamV1` and the neutral `MldumpThc*` /
`MldumpCoulomb*` DTOs. Conversion scratch is bounded to one $k$
eigenvector/site-matching record, one site radial record, one $q$
$\zeta$/vertex record, or one $q$ $V$/Gamma record. Occupations are not
invented. The MLDUMP v1 tree is unchanged.

### 3.12 CoQui-native scalar Cholesky adapter

`write_scalar_coqui_cholesky(path, inputs, thc, coulomb, coulomb_spec, factor)`
writes a CoQui-native single-file Cholesky ERI, using the section 1.5 factorization.
It is not MLDUMP, not a SPEX dump, and does not claim that THC/MLDUMP is
CoQui-compatible for every $q$. The on-disk tree follows live CoQui
`chol_reader_t` (the CoQui inspect checkout, `wg-dev` @
`a19774d03fb979bd852fae4f7f95c045a4cbca78`): `/Interaction` scalars
`Np,nspin,nspin_in_basis,nkpts,nbnd,nbnd_aux=0,tol`, Cartesian `kpts`/`qpts`,
`qk_to_kmq`, and `Vq{iq}` as native `f64` `[Np,1,nk,nbnd,nbnd,2]` with scalar
variable-length UTF-8 `__complex__="1"`. `qpts` store canonical Cartesian $q$
without a second global Umklapp. Gamma contributes only the finite body.
`libmuffintin-io` owns the native HDF tree; runtime owns mapping and
factorization. Scratch is one $q$ factor and one $q$ $L$ tensor.

## 4. Explicit exclusions

Scalar product-input, scalar mixed-product, scalar AllQL2 THC, and sampled
$\zeta$ Coulomb do not:

- copy or reimplement Weinert assembly inside runtime
- inject Coulomb Grams or run Q0L2 / AllQCoulombPool
- include selected core radials in the product window
- extend `ProductRadialId` with $\kappa$ or four-component $PP/QQ$
- implement spinor or signed $\kappa$ product vertices
- add a principal-angle engine
- claim SPEX or material acceptance
- complete the product, mixed-product, THC, and Coulomb sequence

Later stages consume this scalar product-input, mixed-product,
interpolation-point, and sampled $\zeta$ Coulomb contract rather than
reaching into checkpoint solver internals.

The spinor THC/Coulomb stage additionally does not implement core-valence
products, HDF5/CoQui output, material/SPEX acceptance, a principal-angle
engine, or GW/RPA.
