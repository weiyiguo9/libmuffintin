# ctf-rs S1 close plan

- Workstream ID: `ctf-rs`
- Plan version: 4
- Approval: accepted 2026-09-12 (immutable; direction changes make plan.v5); executor Codex on MSI, `D:/projects/ctf-rs`
- Supersedes: `plans/ctf-rs/plan.v3.md` for the S1 close only. C1 is closed under plan.v3 (evd-1015). S1a to S1c are delivered under plan.v3 (evd-1016 to evd-1018) with `G-CTF-S1` at HANDOFF, ten of thirteen targets passing; this plan carries the three open items and nothing else.
- Repository: rustnumgum/ctf-rs; the MSI checkout `D:/projects/ctf-rs` is at `0b65b90` (the S1c record commit), clean; `origin/master` is still `b9990fa` because the Mac has not published; the Mac clone `~/tmp/ctf-rs` holds `0b65b90` as `msi/master`
- Upstream: cc4s/ctf `f69cbb46` at `D:/projects/ctf-upstream-f69`, WSL reference build `/home/xylxp/ctf-rs-reference`
- Decision: ADR-0007 stands; no new ADR
- Status source: false; state lives in `STATUS.md` and `ledger.md`
- Evidence: ctf-rs `docs/validation.md` section "S1d and S1 close"; one evd entry here (evd-1019) with the ctf-rs commit SHA

## Objective

Close `G-CTF-S1`. The Mac review (evt-0068) classified the three S1
handoffs: one port gap (S1d), one fixture-size artifact (checkpoint at n=3
against the source default n=7), and one withheld rerun (AMG at WSL 2 and
4 after fix `1285771`). Each gets exactly one run under a contract fixed
here.

## 1. State at plan time (ctf-rs `0b65b90`, 2026-09-12)

| Item | Delivered state | Classification |
|---|---|---|
| `sparse_einsum_hadamard`, `sparse_scaled_expression` | `SearchCache::prepare` returns `None` (`fold_indices::Eligibility::SparseWeighIndex`); the drivers `unwrap` and panic; Q never computed at WSL or native | port gap: the pinned source removes a sparse Hadamard index before mapping (`contraction.cxx:5417-5527`), and the Rust path has no such step. `docs/upstream-known-failures.md` "S1a sparse Python ABC expressions (static source restriction)" is wrong and is withdrawn by this plan |
| `upstream_checkpoint_sparse` | n=3, Q = 3.98e-7 at 1 rank and 3.40e-7 at 2 and 4 against 1e-7·n·n·0.1·n = 2.7e-7; dense twin gives the same Q | fixture below the source default: `checkpoint_sparse.cxx:67` sets n=7 unless `-n` is given; the criterion grows as $n^3$ while the six-decimal round-trip error grows as $\sqrt{\mathrm{nnz}}$ |
| `upstream_algebraic_multigrid` WSL 2, 4 | fix `1285771` landed after the once-only WSL run; the rank-1 diagnostic and native 1, 2, 4 pass on the fixed tree | rerun withheld by the brief's once rule; Stop That Digit allows one run of the affected check after a genuine fix |

## 2. Common contract

```text
digit: Q=each driver's own metric, unchanged ref=pinned f69cbb46 tol=unchanged class=R
runs: one WSL Ubuntu-26.04 run of the thirteen S1 targets at 1, 2, 4 ranks on the
      final tree; the full scripts/acceptance-wsl.sh at 1, 2, 4 once; native
      -BuildOnly once; native 1, 2, 4 once for the three drivers this plan touches
budget: per failing driver at most the three diagnostics in section 5, then
        DIGIT / HANDOFF; a pass is closed
records: ctf-rs docs/validation.md "S1d and S1 close"; coverage, inventory and
         known-failures rows corrected; evd-1019 here with the ctf-rs SHA
```

All runs happen after the last code commit of section 3. No driver runs
twice; the one WSL run of the S1 set is at the same time the S1d acceptance,
the checkpoint gate at n=7, the AMG rerun, and the class R regression of the
other ten targets after the S1d change to the shared sparse entry.

## 3. Work, in order

| ID | Work | Gate |
|---|---|---|
| CK7 | `tests/upstream/examples/checkpoint_sparse.rs`: `N` becomes the source default 7 (`checkpoint_sparse.cxx:67`); the bound stays the source expression 1e-7·n·n·0.1·n, now 3.43e-6, computed from `N` as today; the `DIGIT` line reports n=7 and that bound. Nothing else in the driver changes: density 0.1, rank-seeded fill, six-decimal text format, world plus parity runs. Commit before any run. | `G-CTF-S1-CK7`: Q = norm2(v−u) after the text round trip; bound 3.43e-6; WSL 1, 2, 4 and native 1, 2, 4, once each |
| S1d | Sparse Hadamard-index elimination, the source step at `contraction.cxx:5417-5527`, on the Rust sparse contraction entry (section 4). After it, no contraction with a sparse operand reaches `fold_indices::select` with a label present in A, B and C. | `G-CTF-S1d`: `sparse_einsum_hadamard` (n=11, density 0.1, `ijk,jkl->ijkl`, sparse–sparse and sparse–dense) and `sparse_scaled_expression` (n=5, density 0.1, `ijl,kjl->ijk` inside the source's scaled expression); Q = sum(abs(diff)) against the dense Rust computation of the same expression; bound 1e-14, the source file's own `allclose`; WSL 1, 2, 4 and native 1, 2, 4, once each |
| AMG | No code. `upstream_algebraic_multigrid` runs inside the S1-set WSL run on the final tree. | `G-CTF-S1-AMG`: Q = V-cycle residual `rnorm` against the twice-smoothed Jacobi residual `rnorm_alt`, bound `rnorm < rnorm_alt`, WSL 1, 2, 4 once (native 1, 2, 4 already passed at evd-1018 and is not repeated) |
| Close | The S1-set WSL run (thirteen targets at 1, 2, 4, the pattern of `D:/projects/runs/ctf-rs-s1/S1c/acceptance.sh`), the full `scripts/acceptance-wsl.sh` at 1, 2, 4, `scripts/acceptance-native.ps1 -BuildOnly`, and the native 1, 2, 4 runs of `sparse_einsum_hadamard`, `sparse_scaled_expression`, `upstream_checkpoint_sparse`. | `G-CTF-S1` closes when all thirteen targets carry `DIGIT / PASS` at WSL 1, 2, 4 and the full script shows no failure line; otherwise `G-CTF-S1` stays HANDOFF with the failing targets named |

## 4. S1d specification

The source (`contraction::execute`, `contraction.cxx:5417-5527`) does the
following once per weigh index, recursing through `nc->execute()` until no
label is shared by A, B and C; `can_fold` (`contraction.cxx:538-584`) then
accepts the rewritten contraction, so the COO/CSR fold applies.

1. Find the weigh labels: with `inv_idx`, a label whose A, B and C slots
   are all set. The loop leaves `iA`, `iB` at the last such label; that one
   is eliminated in this pass.
2. Choose the operand X to expand by the source size rule: `A_sz` is
   `nnz` for sparse A (`size` for dense), raised to at least
   min(szA1, szA2), where szA1 and szA2 are the row and column extents of
   the matrix A would form, weigh labels counted on both sides; the same
   for B. X = A if A is sparse and (B is dense or `A_sz < B_sz`); otherwise
   X = B. In the three contractions of the two drivers X is always sparse
   (equal-density ties fall to B); a dense X is out of scope.
3. Build X2 of order `X.order + 1`: the axis `iX` of X is duplicated in
   place, so X2 has axes `[..., iX', iX, ...]` with `lens[iX'] = lens[iX]`,
   `sym[iX'] = NS`, and X2 is sparse. X2 holds exactly `nnz(X)` pairs, each
   pair of X placed on the diagonal `x[iX'] = x[iX]`. The source writes this
   through a summation with a repeated output label (`extract_diag`,
   `summation.cxx:1236-1243`, `:1427`); in Rust a key remap of X's pairs into
   X2's cyclic distribution is the accepted route, and the summation route
   is not required.
4. Relabel: the new axis `iX'` carries a fresh contraction label
   (`num_tot`); the original weigh label stays on axis `iX` of X2. In the
   other operand the weigh label's axis takes the fresh label
   (`nc->idx_B[iB] = num_tot` or `nc->idx_A[iA] = num_tot`). C is unchanged.
5. Execute the rewritten contraction with the same alpha, beta, output
   nonzero fraction, and function; repeat from step 1 if a weigh label
   remains (`ijk,jkl->ijkl` needs two passes, `ijl,kjl->ijk` one).

Where it lands: the Rust sparse contraction entry that today runs
`SearchCache::prepare` then `contract_sparse_from_selected`. The rewrite
precedes planning, so the planner and the leaf kernels stay as delivered in
S1a to S1c; the planner is called on the rewritten operands only. The two
drivers keep their shapes, density, seeds, expressions, alpha and beta
values, and the 1e-14 rule; they may change their call sequence to whichever
entry carries the rewrite, and the `unwrap` of a `None` selection is
replaced by the rewrite path, not by a skip. No other driver has a sparse
weigh index, so the ten passing S1 targets are unaffected by construction;
the S1-set run in section 3 is the class R record of that.

## 5. Diagnostics

At most three per failing driver, each named for the two explanations it
separates:

1. Rank-count split: run the failing driver at 1 rank; a pass there against
   a failure at 2 or 4 separates a local kernel or rewrite error from a
   communication or layout error.
2. Rewrite twin (S1d) or dense twin (CK7, AMG): for S1d, apply the same
   expansion and relabeling to the dense copies of the operands and contract
   densely; agreement with the reference separates a wrong rewrite (X
   choice, axis position, label assignment, diagonal placement) from a
   wrong sparse execution of the rewritten contraction. For CK7 and AMG the
   dense twin of plan.v3 section 5 applies.
3. Candidate pin (S1d, AMG): force the planner to the source's selected
   candidate for the rewritten contraction and run once; agreement separates
   a wrong candidate choice from a wrong kernel. For CK7: count the pairs
   written and the pairs read back; equality separates a lost or duplicated
   pair from format rounding.

## 6. Records

- ctf-rs `docs/validation.md` section "S1d and S1 close": the contract table
  above, then per target the stamp, Q, reference, bound, delta, commands, log
  paths, run count; the n=3 checkpoint history moves here from
  `docs/upstream-known-failures.md` as a note, since it was never a source
  failure.
- `docs/upstream-known-failures.md`: the section "S1a sparse Python ABC
  expressions (static source restriction)" is removed when S1d passes; if
  S1d ends in HANDOFF it is rewritten to state the port gap and the source
  location, never a source restriction. The section "Bounded S1c sparse
  checkpoint precision" is removed once the n=7 gate has run, whatever its
  outcome, and its content goes to `validation.md`.
- `docs/coverage.md` and `docs/upstream-inventory.tsv`: the rows for
  `contraction/contraction` (weigh-index elimination), `test_sparse.py`,
  `checkpoint_sparse.cxx`, `algebraic_multigrid.cxx` updated as a consequence
  of the work.
- Harness (`harness-msi`): evt-1010 at start, evd-1019 naming
  `G-CTF-S1d`, `G-CTF-S1-CK7`, `G-CTF-S1-AMG` and `G-CTF-S1` with the ctf-rs
  SHA, commands, log paths, stamps; evt-1011 at close or handoff.
- Logs under `D:/projects/runs/ctf-rs-s1/S1d/`.

## 7. Boundaries

- plan.v3 section 6 stands in full, with two named exceptions fixed here
  before any run: the checkpoint driver constant `N` becomes the source
  default 7, and the two Python drivers may change their call sequence to
  the entry that carries the S1d rewrite. No tolerance, metric, expression,
  density, or seed changes; no further fixture change.
- No driver runs more than once; a pass is closed. A failure gets at most
  the section 5 diagnostics, then `DIGIT / HANDOFF` with the value.
- No planner or kernel redesign under S1d; the rewrite is a step before
  planning. No dense X path, no sparse Hadamard support inside the leaf
  kernels.
- ctf-rs is not pushed from MSI; the Mac publishes. libmuffintin and the
  fftw fork are not edited.
