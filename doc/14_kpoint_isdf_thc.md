# 14. k-point ISDF/THC kernels

This note records the production THC contract of
`muffintin_prodbasis::thc`. Product kinematics remain
[13](13_product_space_and_lapw_mpb.md). Grids remain
[07](07_grids_sphere_algebra_and_checkpoint_formats.md). The runtime
LAPW bridge that evaluates real orbitals onto parent grids is
[18](18_lapw_mpb_thc_integration.md), and production $V^q$ is
[15](15_weinert_coulomb_metric.md).

## 1. Scope

The kernels compress orbital-pair densities collocated on a parent
grid. Selection is k-point ISDF: one interpolation-point set shared
across canonical $q$, with per $q$ interpolation vectors $\zeta^q$.
Callers supply evaluated [`PairBlock`]s; the kernels select points,
fit $\zeta$, and emit `CompiledAuxiliaryBasis` interpolation-point
payloads and `PairVertex` Bloch pair vertices over the product-space
IR. There is no orbital evaluator, no k-mesh constructor, and no
Coulomb assembler here.

## 2. Canonical $q$ and Umklapp

A transfer stores the Cartesian canonical $q$ on `TransferQ` (zero
Umklapp for on-mesh $q$). Pair columns use the positive phase that
keeps $\rho^q$ in the canonical $q$ gauge:

```math
\rho^q_{k,ij}(r)
= e^{+i G_{\mathrm{wrap}}\cdot r}\,
u_{i,k-q}^*(r)\,u_{j,k}(r),
```

with the folded $k-q$ index and integer wrap $G_{\mathrm{wrap}}$ from
the mesh map. Column order is $(k,i,j)$ with $k$ slowest and $j$
fastest; $q=0$ and finite $q$ blocks share that convention. Omitting,
sign-flipping, or double-counting $G_{\mathrm{wrap}}$ changes the pair
matrix relative to an independent column oracle.

## 3. Selection and fit contract

`fit_allq_l2_pair_blocks` is the production entry: ordered per $q$
pair blocks on one parent grid, true quadrature weights (zeros allowed
but never selectable), an optional explicit positive-weight candidate
list, an explicit engine, and a [`RankPolicy`]. It selects one AllQL2
interpolation-point set, fits $\zeta^q$ by weighted least squares, and
returns a [`ThcResult`] with selection provenance, per $q$
[`PerQFit`]s, auxiliaries, and semantic vertices.

`L2Engine::FullColumnPivotedQr` and `L2Engine::FullPivotedCholesky`
operate on the same full weighted pair matrix and are the only engines
accepted on pair blocks; a structured sketch is rejected there. At
fixed rank the two full engines are equivalent selector backends in
exact arithmetic when pivots are well separated. The Cholesky path
contracts columns of the point Gram matrix-free and returns the square
root of each residual diagonal, on the same scale as QRCP $|R_{kk}|$.
Near pivot ties or the numerical-rank floor, finite-precision pivot
order may differ; results are compared by quadratic forms, never by
pivot identity.

The production selection default is `allq_l2`, the unique choice known
to be Umklapp-safe at finite $q$: a $q=0$-only selection can hide a
finite $q$ channel, demonstrated by a constructed two-channel pair
block and by the Umklapp algebra. `q0_l2` and the two-stage
`allq_coulomb_pool` remain implemented as fixture-driven strategy
sweeps in the shared test kit, not as production entries. There is no
universal conversion between MPB `TOL` and a THC rank or threshold.

Provenance records strategy, seed, uniform shift, `pool_factor`,
$N_\mu$, $q$ set, grid path, $\sqrt{w}$ weights, and the
$(N_k,N_{\mathrm{orb}})$ column window.

## 4. Coulomb boundary

Coulomb-aware residual reporting consumes `InjectedCoulombGram` /
`CoulombGramSet`. Those types validate Hermiticity, shape (buffer
lengths, not a square-root guess), finiteness, a weak PSD bound, $q$,
and column order; they do not assemble $1/r$, Weinert $V^q$, Ewald
sums, or SPEX `coulombmatrix.f`. Every injected Gram used for a
Coulomb residual is checked for q-index, exact `TransferQ`, exact
`PairColumnLayout`, and resulting dimension before whitening. The
worst finite $q$ Coulomb metric is the max Coulomb residual over
nonzero $q$, independent of the L2-worst $q$.

Production $V^q$ consumption is [15](15_weinert_coulomb_metric.md):
`assemble_sampled_coulomb` takes `ThcResult.fits[iq].zeta` on the
parent grid as `SampledAuxiliaryFunctions`. Interpolation *nodes* are
not the production $\zeta$ functions, and Weinert Coulomb does not
claim elementwise $V^{\mathrm{MPB}}=V^{\mathrm{THC}}$ when the spans
differ.

## 5. Public surface

- `PairBlock`: per $q$ pair-density block in the canonical gauge
- `L2Engine` / `RankPolicy`: full QRCP or full pivoted Cholesky with
  exact-rank or relative-residual termination
- `pivots_from_pair_blocks` / `cholesky_pivots_from_pair_blocks` /
  `truncate_rank` / `interpolation_points`: engine primitives on
  evaluated blocks
- `fit_allq_l2_pair_blocks`: AllQL2 selection, $\zeta$ fit, and
  emitted auxiliaries/vertices
- `fit_per_q`: $\zeta$ and weighted L2 / injected-Coulomb residuals
  for one $q$
- `interpolation_auxiliary` / `bloch_pair_vertices`: product IR
  emission
- `InjectedCoulombGram` / `CoulombGramSet`: explicit Coulomb data
  contract
- `linalg`: the shared dense kernels (QRCP, pivoted Cholesky,
  truncated-SVD least squares), public for the test fixture

Diagnostics report $q=0$ and worst finite $q$ separately, and core vs
valence-only column groups when the caller names a core orbital.

## 6. Toy fixture and recorded evidence

The finite deterministic toy bases, the cubic toy k-mesh, toy Bloch
orbitals with `evaluate_pair_block`, the structured-sketch selector,
the Coulomb-pool rerank, and the `run_thc` / `compare_strategies`
sweep harness live in `crates/mt-prodbasis/tests/toy_kit`, included by
path from prodbasis and coulomb tests. Two scratch fixtures are
ported, not tracked:

- MT-like localized orbitals, $a=6$, $2\times2\times2$ mesh, adaptive
  and uniform grids, seeds 7/19/43, random shift 29
  (`scratch/thc_mt_kpoint_test.py`).
- Synthetic two-region LAPW, $a=5$, $2\times2\times1$ mesh
  (`scratch/thc_lapw_end_to_end_test.py`).

Python structured sketches used NumPy PCG64. The Rust sketch and
action vectors use SplitMix64 with the same integer seeds; they are
deterministic in Rust, not bit-identical to NumPy.

| Quantity | Recorded value | Use |
|---|---|---|
| Fine adaptive ERI-F | $4.932\times10^{-2}$ | Python full-fixture table |
| Fine adaptive ERI-max | $4.560\times10^{-2}$ | Python full-fixture table |
| Fine adaptive action | $6.230\times10^{-2}$ | Python full-fixture table |
| ERI/action gate | $8\times10^{-2}$ | slow source-equivalent fine candidate only |
| Independent-reference ERI | $2.498\times10^{-2}$ | Python 38 vs 30 table |
| Reference gate | $5\times10^{-2}$ | slow source-equivalent distinct-grid refs only |
| Umklapp probe | $\lvert\rho+i\rvert<2\times10^{-14}$ | algebraic regression |

The q=0 vs all-q table at $N_\mu=48$ on adaptive nrad=20 does **not**
by itself show a finite $q$ hide (both are $\sim10^{-13}$); the hide
regressions above carry that claim. Run the slow source-equivalent
gate with:

```sh
cargo test --release -p libmuffintin-prodbasis --test end_to_end_smoke source_equivalent_python_lapw_fixture --offline -- --ignored --exact --nocapture
```

## 7. Exclusions

No Weinert assembler, no real-material LAPW/SCF/GW API, no MPI/CTF,
and no `scratch/` tracking.
