# 11. Tensorized numerical substrate

This note records the M-Fb local tensor contract. It does not add a mixed
product basis, THC, Coulomb operators, package renaming, or a distributed
runtime. CTF, SLATE, MPI, and manual shard types are deferred non-goals.

## 1. Objects

A dense local tensor carries:

- rank and shape;
- named axes, not anonymous integer slots;
- an in-world memory layout, C-contiguous, F-contiguous, or strided;
- dtype `Complex64`;
- a one-process [`LocalWorld`](../crates/mt-tensor/src/lib.rs);
- placement intent [`Placement::Auto`](../crates/mt-tensor/src/lib.rs).

Host snapshots remain ordinary `Vec<Complex64>` buffers with an explicit
row-major or column-major contract. General contraction tensors use row-major
host construction. [`DenseEigenvectors`](../crates/mt-tensor/src/lib.rs) has
logical axes `[GlobalBasis, Band]` and canonical column-major storage, so all
basis coefficients of one band are contiguous. Backend tensor handles stay
private to `mt-tensor`. Serialized artifacts in `mt-io` are unchanged and must
not mention RSTSR, faer, or tenferro.

## 2. Einsum layer and backends

The public contraction language is Einstein summation. Physics modules write
index statements such as

```math
M_{ij}=\sum_{cd} P_{ci}^* B_{cd} P_{dj}
```

as `einsum("ci,cd,dj->ij", [P.conjugate(), B, P])`. Nested scalar loops are
not a production implementation of that sum.

The einsum layer is backend-neutral. Two local backends are in scope:

- **RSTSR 0.7.10 + TBLIS** (`backend-rstsr`, default). Contractions call
  `rt::tblis::einsum` and therefore link `libtblis`. Host tensors live on
  RSTSR's `DeviceFaer` because that device implements the rayon bound TBLIS
  einsum requires. faer remains the Hermitian EVD, not the einsum engine.
- **tenferro-rs** (`backend-tenferro`). The same subscripts are the contract
  for `tenferro-einsum` on a CPU/faer provider, without AD, GPU, or a graph
  runtime. The workspace MSRV stays 1.85. `tenferro-einsum` 0.3.0 needs rustc
  1.96, so that feature is a local rust-version bump, not a workspace MSRV
  change.

There is no scalar fallback and no third production lowering that rewrites
the einsum as handwritten index loops. Enabling TBLIS uses
`tblis-src/build_from_source`. Set `TBLIS_SRC` to a TBLIS git repository
(URL or local clone with submodules); the crate default path in the cargo
registry is not itself a git repository.

## 3. Axis roles

LAPW site contractions use three axis roles:

- `SiteCoordinate`: every $\mathrm{lm}$ channel's $(u,\dot u)$ pair, then that
  site's local orbitals in $(l,m,n)$ order;
- `SiteBasis`: all plane waves, then that site's local orbitals;
- `GlobalBasis`: the global $[\mathrm{PW}][\text{all site LOs}]$ operator axes.

`Reduced` and `Band` are reserved for the filtered generalized eigensolver.
Labels in an einsum subscript bind to these roles. Site-local dimensions are
ragged: each site is its own tensor. Do not pad every site into one
rectangular global tensor.

## 4. Site congruence

For site $a$, form the coefficient tensor $P_a$ with axes
$[\text{SiteCoordinate},\text{SiteBasis}]$. A plane-wave column stores the two
APW matching coefficients of every $\mathrm{lm}$ channel and is zero on that
site's LO coordinates. An LO column of the same site is a unit vector in the
corresponding coordinate. Let $B_a$ be the Hermitian site block. The muffin-tin
contribution is

```math
M_a = P_a^\dagger B_a P_a,
```

evaluated as `einsum("ci,cd,dj->ij", ...)`. $M_a$ is Hermitian on
`SiteBasis`. Scattering $M_a$ into the global $[\mathrm{PW}][\text{site LO}]$
indices is an explicit index operation; it is not a second contraction.
Interstitial Fourier and step-function assembly remain ordinary physical
loops.

The same congruence supplies APW--APW, APW--LO, and LO--LO entries. Complex
conjugation is $P^*$ in the first operand, so a nonzero site phase already
stored in the APW coefficients is conjugated exactly once.

## 5. faer and later stages of this document

faer remains the local Hermitian eigensolver for the overlap spectrum and the
reduced ordinary problem

```math
H_{\mathrm{reduced}} Z = Z\varepsilon.
```

M-Fb1 does not change overlap-spectrum filtering. M-Fb2 evaluates the
reductions with the same einsum layer. faer still diagonalizes $S$ and the
reduced $H$.

```math
X = U_{\mathrm{keep}}\,\mathrm{diag}(s_{\mathrm{keep}}^{-1/2}),
\qquad
H_{\mathrm{reduced}} = X^\dagger H X,
\qquad
C = X Z.
```

In subscripts these are `ik,k->ik`, `ir,ij,js->rs`, and `ir,rb->ib`. The
residual $HC-SC\varepsilon$ is `ij,jb->ib` for $HC$ and $SC$, `ib,b->ib`
for the eigenvalue scale, and `ib,ib->b` for the squared column norms.

- M-Fb2 tensorizes $X$, $X^\dagger H X$, $C=XZ$, and the batched residual
  contraction. The filtering algorithm and faer Hermitian EVD stay in place.
- M-Fb3 stores public LAPW $H$, $S$, and eigenvector columns as tensor-native
  `DenseHermitianMatrix` and `DenseEigenvectors`. Eigenvectors use the
  column-major `[basis, band]` convention, matching SPEX, FLEUR, ELK, and QE;
  a fixed band is one contiguous basis-expansion column. "Dense" is the local
  storage kind; the old unnamed `Vec<Complex64>` buffers are gone.
- M-Fb4 adds `einsum_tenferro` behind `backend-tenferro` using
  tenferro-einsum on CPU/faer, without AD, GPU, or XLA. Compiling that
  feature requires rustc 1.96; the workspace MSRV remains 1.85. RSTSR+TBLIS
  stays the default `einsum` path.

## 6. Acceptance

M-Fb1 requires `einsum("ci,cd,dj->ij", ...)` through RSTSR+TBLIS to match
direct analytic APW--APW, APW--LO, and LO--LO values, including complex
conjugation and a nonzero $k$ site phase; a no-LO site must reduce to the
APW-only congruence; the scattered global matrices remain Hermitian; axis and
shape errors are traceable; and every existing M-F $H$, $S$, retained rank,
eigenvalue, eigenvector, and residual regression stays within its previous
tolerance. `DenseEigenvectors` must preserve logical `(basis, band)` indexing
while exporting the linear offset `basis + basis_count * band`. This document
does not claim CTF, MPI, tenferro parity, or the Cu/SPEX one-meV gate.
