# ctf-tiled: CTF algebra plus tile sparsity (the "tiled-ay" plan)

- Workstream ID: `ctf-tiled`
- Plan version: 1
- Approval: draft (not authorized; successor to `ctf-rs` S1 under `plans/ctf-rs/plan.v3.md`; starts after S1 closes unless the user orders it earlier)
- Supersedes: none
- Repository: rustnumgum/ctf-rs, evolved in place; no new crate
- Decision: none yet; authorizing T1 makes an ADR ("ctf-rs evolves into a distributed tiled tensor runtime") and, if the user wants it, the rename to `riir`
- Status source: false; state lives in `STATUS.md` and `ledger.md`
- Evidence: ctf-rs `docs/validation.md` section per milestone; one evd entry here per milestone close with the ctf-rs commit SHA

Status: draft, 2026-09-11, written against ctf-rs `b9990fa` from the user's
two analyses of that day (the DBT plus LibRI direction for the sparse line;
the keep-the-CTF-skeleton plan). Kept as a plan so the direction is not
lost while S1 runs.

## Objective

Evolve ctf-rs from "Rust CTF rewrite" into a distributed tiled tensor
runtime: CTF-derived distributed algebra plus TiledArray-derived tile
semantics, keeping the CTF skeleton rather than redesigning the runtime.
Two storages, each with a libmuffintin consumer: dense tiled for the mixed
product basis and GW in $q$ space; block-sparse tiles indexed by
(atom, $R$) for LRI and space-time GW, on the MTO family only.

## 1. State at plan time (ctf-rs `b9990fa`)

- `Tensor` is a `Distribution { shape, topology, mappings }` (CTF cyclic
  physical and virtual phases per axis) plus flat local data.
  `SparseTensor` is the same `Distribution` plus an element-sparse list per
  virtual block. Nothing tile-shaped exists: no tile id, no tiling, no
  block-sparse storage.
- The 2D contraction level already has panels: `ctr_2d.rs` (`Panel`,
  `execute`) for dense SUMMA-like steps and `sparse_2d.rs` (`Panel` with
  per-block variable sizes; `execute_csr`, `execute_ccsr_dense`,
  `execute_coo_dense`, `execute_csr_dense`, `execute_csr_sparse_dense`,
  `execute_pairs_dense`). A present-tile list is one more kind of panel.
- Local kernels: `LocalKernels: GemmKernel<f64>` in `linalg.rs`
  (BLAS/LAPACK, faer-replaceable by the README contract); four-scalar
  folded panels.
- S1 (element-sparse automatic planning, graph drivers) is in progress
  under plan.v3 and is not on this path; its code is not touched here.

## 2. Architecture boundary

```text
                    riir (ctf-rs, evolved in place)
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
  CTF-derived                              TiledArray-derived
  distributed algebra                      tile semantics
  Topology, Context, collectives           TileId, Tiling
  cyclic mapping, redistribution           TileState (structural shape)
  contraction planner, cost models         BlockSparse storage
  index DSL                                tile-level join
          │                                       │
          └───────────────────┬───────────────────┘
                              ▼
                  DenseTile kernels (TileKernel<T>)
                  faer │ BLAS │ TBLIS │ GPU later
```

Not a stack of runtimes (CTF over TiledArray over MADNESS). No
MADNESS-style futures or task scheduler; SPMD collectives and explicit
`close` stay as R1 left them.

## 3. Design contract

| Item | Rule |
|---|---|
| `TileId` | `TileId(Vec<usize>)`, one tile coordinate per tensor axis |
| `Tiling` | `tile_of(&[usize]) -> TileId`, `extent(&TileId) -> &[usize]`. Two forms: interval tiling (contiguous ranges; hierarchical (atom, on-site) when orbitals are numbered atom-contiguously; $R$ as an explicit axis with tile size 1) and permutation-plus-partition tiling (ISDF/THC points grouped by spatial clustering) |
| `Distribution<TileId>` | `owner(&TileId) -> Rank`. Two implementations: `Cyclic`, derived from the existing `Distribution` with tile = virtual block, so the CTF planner and redistribution stay valid; `PairList`, an atom-pair hash or balanced list (LibRPA style) for the LRI line, no CTF planner, owner computes with tile fetch. T1 uses `Cyclic` only |
| Storage | `Dense` (today's `Tensor`) or `BlockSparse { tiles: HashMap<TileId, DenseTile<T>> }`, NS only. The element-sparse `SparseTensor` is a third storage and is not replaced |
| Shape | `TileState::{Zero, Present}` per tile, structural, globally replicated; produced from geometry (a neighbor list) or from the fill; no norm in T1 |
| Planner | For the shared index $k$: `A: k -> [(i, tile)]`, `B: k -> [(j, tile)]`, cross per existing $k$. It lives inside the panel layer; the mapping layer is untouched. No triple loop over $i$, $j$, $k$ with presence checks |
| Kernel | `TileKernel<T>` extending `LocalKernels`: `contract(a, b, c)` per tile pair; dispatch by tile shape: small tiles to faer once adopted, regular GEMM to BLAS, general contractions to TBLIS; GPU later |
| Screening (T2) | `TileMeta { norm }` on input tiles; a threshold filter in the space-time style (drop $G(R,\tau)$ and $\chi_0(R,\tau)$ tiles below $\varepsilon$). The bound $\lVert C_{ij} \rVert \lesssim \sum_k \lVert A_{ik} \rVert \, \lVert B_{kj} \rVert$ may skip output tiles; it is not propagated as a shape estimate before T2 |
| $R$-join (T4) | Output $R$ from input $R$ arithmetic, which neither CTF nor TiledArray has. T1 to T3 fold $R$ into a supercell atom index (the DBT way); T4 adds explicit $R$ with a Fourier step to $q$ afterwards |
| Consumers | Dense tiled: MPB and GW in $q$ space. Block-sparse (atom, $R$): LRI and space-time GW on the MTO family only, since the LAPW interstitial plane waves are not local. The atom-centered auxiliary basis (on-site MT product basis plus a smooth-Hankel set for tail products) is libmuffintin research; it gates T4's usefulness, not T1 |

## 4. Milestones

| ID | Scope | Acceptance |
|---|---|---|
| T1 | `BlockSparse` storage in 2D and 3D; `TileId` and interval `Tiling`; `Distribution<TileId>::Cyclic` over the existing mapping; tile-level join in the panel layer; `TileKernel<T>`; README positioning paragraph; crate name unchanged | class A, ref = the dense path. Closed rows: `ik,kj->ij`, `ijk,kl->ijl`, `ijk,jkl->il`; two patterns each, banded (one tile wide) and random with 10 percent of tiles present; f64. Q1 = max abs difference against the dense contraction of the zero-padded data, bound $10^{-12}$ times the max abs of the dense result. Q2 = the present output tile set equals the structural join of the input shapes, exact. Ranks 1, 2, 4 in WSL, once |
| T2 | Norm filter on input tiles | class A: the T1 rows with a threshold $\varepsilon$ on tile Frobenius norms, fixed at authorization. Q = max abs difference against the unfiltered T1 result; bound $\varepsilon \, N_k \, \max_k \lVert B_{kj} \rVert_F$ for tiles dropped from $A$, and symmetrically for $B$ |
| T3 | `PairList` distribution and owner-computes tile fetch | the T1 rows under `PairList`; Q as T1 Q1 against the T1 `Cyclic` result, same bound; the result is independent of the rank count |
| T4 | Explicit $R$ axis and the $R$-join in the DSL; toy $\chi_0(R,\tau) = \sum C \, G(R_1,\tau) \, G(R_2,-\tau) \, C$ on a toy local basis with atom-pair distribution | class A, ref = the supercell-folded T1 computation of the same quantity; bound $10^{-12}$ times the max abs of $\chi_0$. Blocked on the libmuffintin atom-centered auxiliary basis for any real consumer |

Common: WSL Ubuntu-26.04 at 1, 2, 4 ranks once per milestone; native
compile/link once per milestone; at most the three diagnostics of section 5
per failing row, then `DIGIT / HANDOFF`; a pass is closed.

## 5. Diagnostics

1. Rank split: the failing row at 1 rank against 2 or 4 separates a kernel
   or join error from an ownership or communication error.
2. Dense twin per tile: comparing tile by tile against the dense result
   separates one wrong tile (a kernel or accumulation error) from a
   systematic offset (a wrong tile coordinate mapping).
3. Shape-only run with every tile `Present`: agreement there separates a
   join or shape error from a kernel error.

## 6. Boundaries

- No symmetry (SY, AS, SH) in block-sparse storage; no adaptive
  repartition; no futures or task scheduler; no TiledArray or MADNESS
  dependency; no norm propagation before T2; no `PairList` before T3; no
  $R$-join before T4.
- The CTF mapping planner, redistribution, and the S1 element-sparse code
  are not modified in T1.
- No gather-based fallback; no new crate; the rename to `riir` is the
  user's separate decision.
- Same executor as ctf-rs (Codex on MSI); starts after S1 closes unless the
  user orders otherwise.
