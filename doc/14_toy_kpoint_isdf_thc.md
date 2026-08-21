# 14. Toy k-point ISDF/THC selectors

This note records the M-I public contract. It does not add a production
Coulomb assembler, real-material SCF/GW, MPI, or an umbrella crate. Product
kinematics remain [13](13_product_space_and_lapw_mpb.md). Grids remain
[07](07_grids_sphere_algebra_and_snapshot_formats.md).

## 1. Scope

`libmuffintin-thc` compresses orbital-pair densities on a **finite
deterministic periodic toy basis**. Selection is k-point ISDF: one
interpolation-point set shared across canonical $q$, with per-$q$
interpolation vectors $\zeta^q$. The crate consumes the M-H product IR
(`ProductPartition`, `TransferQ`, `OrbitalPair`, `PairVertex`) and emits
`CompiledAuxiliaryBasis` interpolation-point payloads.

This is toy finite-cutoff selector evidence. It is not a physical Coulomb
assembler and not a real-material accuracy claim.

## 2. Packages

| Directory | Package |
|---|---|
| `crates/libmuffintin-thc` | `libmuffintin-thc` |

`libmuffintin-product` now stores a typed
`AuxiliaryRepresentation::{MixedProduct, InterpolationPoints}` variant on
`CompiledAuxiliaryBasis`. Mixed-product MPB tests are unchanged in
behaviour. There is no compatibility shim and no empty MPB payload for THC.

## 3. Canonical $q$ and Umklapp

A mesh $q$ is a k-mesh index plus the Cartesian $2\pi q/a$ stored on
`TransferQ` (zero Umklapp for on-mesh $q$). Pair columns use the positive
phase that keeps $\rho^q$ in the canonical-$q$ gauge:

```math
\rho^q_{k,ij}(r)
= e^{+i G_{\mathrm{wrap}}\cdot r}\,
u_{i,k-q}^*(r)\,u_{j,k}(r),
```

where `kminus(k,q)` returns the folded $k-q$ index and integer wrap
$G_{\mathrm{wrap}}$, matching `scratch/thc_mt_kpoint_test.py` lines 134–139
and 153–161 and `scratch/thc_lapw_end_to_end_test.py` lines 285–307.

Column order is $(k,i,j)$ with $k$ slowest and $j$ fastest. $q=0$ and
finite-$q$ blocks share that convention. Omitting, sign-flipping, or
double-counting $G_{\mathrm{wrap}}$ changes the pair matrix relative to an
independent column oracle.

## 4. Selectors

All three strategies are compared at identical $N_\mu$ and identical toy
inputs/seeds.

| Name | Algorithm | Rank |
|---|---|---|
| `q0_l2` | Weighted QRCP / structured sketch of the $q=0$ pair block | exact $N_\mu$ or L2 threshold |
| `allq_l2` | The same weighted selection on every canonical $q$ | exact $N_\mu$ or L2 threshold |
| `allq_coulomb_pool` | All-q L2 pool with `pool_factor=2`, then Coulomb-metric QRCP to exact $N_\mu$ | exact $N_\mu$ only |

There is no universal conversion between MPB `TOL` and a THC threshold.
Coulomb-pool mode cannot use threshold termination: the pool residual and
the rerank residual have different meanings.

The production L2 default is `allq_l2`. That is the scratch headline
(`thc_mt_kpoint_test.py` all-q, seed 7) and the unique finite-$q$/Umklapp-safe
choice supported by the q=0 hide regression. `allq_coulomb_pool` is
implemented as specified in the v0.2 plan; the Python scripts did **not**
validate that two-stage path. `q0_l2` is a CoQuí-compatibility / debug
baseline and must not be the default: a q=0-only selection can hide a
finite-$q$ channel.

Provenance records strategy, seed, uniform shift, `pool_factor`/$N_\mu$,
$q$ set, grid path, $\sqrt{w}$ weights, and the $(N_k,N_{\mathrm{orb}})$
column window.

## 5. Coulomb boundary

`allq_coulomb_pool` and Coulomb/action diagnostics consume
`InjectedCoulombGram` / `CoulombGramSet`. Those types validate Hermiticity,
shape (buffer lengths, not a square-root guess), finiteness, a weak PSD
bound, $q$, and column order. They do not assemble $1/r$, Weinert $V^q$,
Ewald sums, or SPEX `coulombmatrix.f`.

The toy helper `toy_coulomb_gram` builds a finite-cutoff
$4\pi/|q+G|^2$ pair-pair Gram with the $q+G=0$ head omitted. The ERI/action
path `compare_candidate_eri_action` follows
`thc_lapw_end_to_end_test.py:420-460`: fit $\zeta$ on the candidate grid,
Fourier that $\zeta$, form $\tilde\rho_G=\zeta_G R_\mu$, assemble the toy
Gram, and compare to an independently Fourier-transformed reference-grid
pair block. The $\zeta$ least-squares is a truncated SVD with relative
cutoff $10^{-12}$, matching NumPy `lstsq(..., rcond=1e-12)`; unpivoted QR
is not used. Action error is
$\max_i\lVert\Delta V x_i\rVert/\lVert V_{\mathrm{ref}} x_i\rVert$ over eight
complex Gaussians with seed 19. This is candidate-oracle evidence, not a
production Coulomb assembler. Production $V^q$ is M-J.

Every injected Gram used for a Coulomb residual is checked for q-index,
exact `TransferQ`, exact `PairColumnLayout`, and resulting dimension before
whitening, including `q0_l2` and `allq_l2`. The worst finite-$q$ Coulomb
metric is the max Coulomb residual over nonzero $q$, independent of the
L2-worst $q$.

## 6. Toy bases and recorded evidence

Two scratch fixtures are ported, not tracked:

- MT-like localized orbitals, $a=6$, $2\times2\times2$ mesh, adaptive and
  uniform grids, seeds 7/19/43, random shift 29
  (`scratch/thc_mt_kpoint_test.py`).
- Synthetic two-region LAPW, $a=5$, $2\times2\times1$ mesh
  (`scratch/thc_lapw_end_to_end_test.py`).

Python structured sketches used NumPy PCG64. The Rust sketch and action
vectors use SplitMix64 with the same integer seeds; they are deterministic
in Rust, not bit-identical to NumPy.

Three test layers:

1. Ordinary plumbing: rank-one interpolation identity at $10^{-12}$. This is
   not the recorded $8\times10^{-2}$ gate.
2. Ordinary bounded three-selector action comparison at identical $N_\mu$ on a
   peaked toy. Python did not implement `allq_coulomb_pool`; the $8\times10^{-2}$
   gate is not claimed for the pool.
3. Slow source-equivalent gate `source_equivalent_python_lapw_fixture`
   (ignored in ordinary workspace runs): reference 38x110+$20^3$, medium
   30x86+$18^3$, fine candidate 26x86+$18^3$, $N_\mu=96$, $|G|^2\le 12$,
   `allq_l2` full QRCP, candidate-only selection/fit. Assert distinct-grid
   reference convergence $\le 5\times10^{-2}$ and fine ERI-F / ERI-max /
   seed-19 action $\le 8\times10^{-2}$. Python table values $2.498\times10^{-2}$,
   $4.932\times10^{-2}$, $4.560\times10^{-2}$, $6.230\times10^{-2}$ are
   evidence, not bit-identity targets.

| Quantity | Recorded value | Use |
|---|---|---|
| Fine adaptive ERI-F | $4.932\times10^{-2}$ | Python full-fixture table |
| Fine adaptive ERI-max | $4.560\times10^{-2}$ | Python full-fixture table |
| Fine adaptive action | $6.230\times10^{-2}$ | Python full-fixture table |
| ERI/action gate | $8\times10^{-2}$ | slow source-equivalent fine candidate only |
| Independent-reference ERI | $2.498\times10^{-2}$ | Python 38 vs 30 table |
| Reference gate | $5\times10^{-2}$ | slow source-equivalent distinct-grid refs only |
| Umklapp probe | $\lvert\rho+i\rvert<2\times10^{-14}$ | algebraic regression |

The q=0 vs all-q table at $N_\mu=48$ on adaptive nrad=20 does **not** by
itself show a finite-$q$ hide (both are $\sim10^{-13}$). The hide is
demonstrated by a constructed two-channel pair block and by the Umklapp
algebra. That is why the default is `allq_l2`, not a taste choice between
nearly equal $10^{-14}$ numbers.

## 7. Public surface

- `evaluate_pair_block` / `pair_density_oracle` — per-$q$ pair matrices
- `select_points` / `run_thc` / `compare_strategies`
- `fit_per_q` — per-$q$ $\zeta$ and weighted L2 / injected-Coulomb residuals
- `interpolation_auxiliary` / `bloch_pair_vertices` — product IR
- `InjectedCoulombGram` — explicit Coulomb data contract
- `toy::compare_candidate_eri_action` — candidate-oracle ERI/action vs a
  reference-grid pair Fourier/Gram

Run the slow source-equivalent gate with:

```sh
cargo test --release -p libmuffintin-thc --test end_to_end_smoke source_equivalent_python_lapw_fixture --offline -- --ignored --exact --nocapture
```

Diagnostics report $q=0$ and worst finite $q$ separately, and core vs
valence-only column groups when the fixture names a core orbital.

## 8. Exclusions

No M-J Weinert assembler, no real-material LAPW/SCF/GW API, no MPI/CTF, no
CI changes, no `scratch/` tracking, no umbrella crate, no MSRV raise.
