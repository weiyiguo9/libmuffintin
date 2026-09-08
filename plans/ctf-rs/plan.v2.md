# ctf-rs rsmpi binding plan

- Workstream ID: `ctf-rs`
- Plan version: 2
- Approval: proposed 2026-09-09; direction authorized by the user (one breaking change to rsmpi); acceptance on review
- Supersedes: `plans/ctf-rs/plan.v1.md` (D1 to D6 closed under it, evd-1004 to evd-1008; S1 carried over unchanged)
- Repository: rustnumgum/ctf-rs, `origin/master` at `f2039d3` (D6 close plus formatting); the MSI checkout `D:/projects/ctf-rs` executes; the Mac clone `~/tmp/ctf-rs` is stale and diverged and is not the baseline
- Decision: `decisions/ADR-0007-ctf-rs-binds-mpi-through-rsmpi.md`
- Status source: false; state lives in `STATUS.md` and `ledger.md`
- Evidence: ctf-rs `docs/validation.md` section per batch; one evd entry here per milestone close with the ctf-rs commit SHA

## Objective

Move MPI initialization and communicator ownership in ctf-rs onto rsmpi
(`mpi` 0.8 over `mpi-sys` 0.2) in one breaking change, so a host that
already holds an rsmpi `Universe` can hand ctf-rs a communicator without a
second initializer. The hand-written collective wrapper is not the defect
and is not replaced for its own sake. A second `MPI_Init`, an implicit or
foreign `MPI_Comm_free`, and a thread level the host did not grant are.

## 1. State at plan time (ctf-rs `f2039d3`, libmuffintin `main`, fftw fork `fe731a02`)

- `Runtime::initialize` calls `MPI_Init` and asserts MPI is not yet
  initialized; `finalize` calls `MPI_Finalize`. libmuffintin's `h2_hf` and
  the fftw fork's `Mpi` both expect a host-owned rsmpi `Universe`. The two
  initializers cannot share a process.
- `MPI_Init` requests `MPI_THREAD_SINGLE`; the hosts initialize at Funneled
  with Rayon threads inside each rank.
- `Comm` is `pub(crate)`; no constructor accepts a host communicator.
- The ABI is not the problem: all three resolve to `mpi-sys` 0.2.4
  (`links = "mpi"`, one copy per build), and the fftw fork re-exports
  `mpi_sys::MPI_Comm`.

## 2. Contract

| Item | Rule |
|---|---|
| FFI layer | `mpi-sys` 0.2 reached through `mpi::ffi`; ctf-rs keeps no direct `mpi-sys` dependency line |
| Dependency | `mpi = { version = "0.8", default-features = false }`, the same line as libmuffintin's workspace; `user-operations` (libffi) stays off |
| Initialization | The host owns the `Universe`. ctf-rs never calls `MPI_Init`, `MPI_Init_thread`, or `MPI_Finalize`. `Runtime::initialize` and `Runtime::finalize` are removed, with no shim |
| Entry | `Context::world(&'u Universe)` and `Context::from_communicator(&'u Universe, &'u impl Communicator)`. Every `Context<'u>` borrows the `Universe`, so tensors, sub-contexts, and MPI-IO handles cannot outlive it. A host-provided communicator is borrowed; the host frees it |
| Freeing | `close(self)` stays the only collective free, for communicators ctf-rs split itself. The wrapped split communicator must not free on Drop (`ManuallyDrop` or `into_raw`); dropping an unclosed sub-context leaks the handle, as today. The world context frees nothing |
| Thread contract | `Context` construction checks `mpi::environment::threading_support() >= Threading::Funneled` and `MPI_Is_thread_main` through `mpi::ffi` (rsmpi has no safe wrapper) and panics with the provided level otherwise. The `PhantomData<Rc<()>>` markers stay: rsmpi's handle is a pointer on Open MPI and an integer on MS-MPI, so its auto traits are not portable |
| Custom reductions | `UnsafeUserOperation::commutative` / `associative` with the existing `extern "C" fn monoid_add` and its thread-local algebra pointer; the `commutative` flag and the `Monoid` associativity and commutativity contract are unchanged. No closure `UserOperation` |
| Collectives | Wrappers coexist. A call moves to the rsmpi safe method only where the same collective exists with the same buffer semantics (section 3); everything else stays on `mpi::ffi` behind `as_raw()` |

## 3. Call mapping (from the `sys::` inventory of `src/ffi/`)

| Today | Destination |
|---|---|
| `MPI_Init`, `MPI_Initialized`, `MPI_Finalize`, `RSMPI_COMM_WORLD` | removed; `Universe` from the host; `SimpleCommunicator::world()` |
| `MPI_Comm_split`, `MPI_Comm_split_type`, `MPI_Comm_free`, rank, size, barrier | `split_by_color_with_key`, `split_shared`, `rank`, `size`, `barrier`; free only in `close` |
| `Allreduce`, `Reduce`, `Allgather`, `Allgatherv`, `Gather`, `Gatherv`, `Scatter`, `Bcast`, `Alltoall`, `Alltoallv`, `Scan`, `Sendrecv`, `Sendrecv_replace` | rsmpi `*_into`, `*_varcount_into`, `send_receive*_into`; Wire byte buffers as `u8` slices with `UserDatatype::contiguous(WIDTH, &u8::equivalent_datatype())`, native `f64`, `i32`, `i64` elsewhere |
| `MPI_Reduce_scatter` (`factor_mpi.rs`, equal blocks today) | `reduce_scatter_block_into`; a variable-count call, if one is ever needed, stays on `mpi::ffi` |
| `MPI_Op_create`, `MPI_Op_free` | `UnsafeUserOperation` |
| `MPI_Type_contiguous`, `MPI_Type_commit`, `MPI_Type_free` | `UserDatatype::contiguous` |
| `MPI_File_open`, `read_at`, `read_at_all`, `write_at`, `write_at_all`, `get_size`, `close`, `MPI_Offset`, `MPI_MODE_*` | stay on `mpi::ffi` (rsmpi 0.8.2 has no MPI-IO); the communicator argument becomes `as_raw()` |
| `RSMPI_SUM`, `RSMPI_DOUBLE`, `RSMPI_INT32_T`, and the other constants | `SystemOperation::sum()` and `Equivalence` datatypes on the safe calls; the MPI-IO path keeps its raw constants |

## 4. Milestone R1 (one breaking change, one commit series)

Order inside the series:

1. `Cargo.toml`: replace `mpi-sys = "0.2"` with `mpi = { version = "0.8", default-features = false }`.
2. `src/ffi/mpi.rs`, `allgather.rs`, `gather.rs`, `factor_mpi.rs` per section 3. `mpi_io.rs` and `binary_io.rs` switch the import to `mpi::ffi` and the communicator argument to `as_raw()`, nothing else.
3. `src/context.rs`: `Runtime` removed; `Context::world`, `Context::from_communicator`, the thread checks; `split`, `split_shared`, and `close` as in section 2.
4. Every test and example `main` (about 150 files call `Runtime::initialize` today): `let universe = mpi::initialize_with_threading(Threading::Funneled)`, check the provided level, `Context::world(&universe)`, drop order tensors, then closed contexts, then the universe. Mechanical; no test body or tolerance changes.
5. `README.md`: an "MPI ownership" paragraph under "Running the current subset". `docs/coverage.md` phase 1 row names the rsmpi entry. `docs/validation.md` gets one section "rsmpi binding".

Acceptance:

```text
digit: Q=each driver's own metric ref=pinned upstream f69cbb46 tol=upstream, unchanged class=R
runs: scripts/acceptance-wsl.sh at 1, 2, 4 ranks, once; scripts/acceptance-native.ps1 -BuildOnly once;
      the D6 native runtime set at 1, 2, 4 ranks, once
budget: per failing driver at most the three diagnostics below, then DIGIT / HANDOFF; a pass is closed
records: ctf-rs docs/validation.md section "rsmpi binding"; one evd entry here (G-CTF-R1) with the ctf-rs SHA
```

Diagnostics, at most three per failing driver, each named for the two
explanations it separates:

1. Print rank, size, and `threading_support()` at `Context` construction:
   an initialization or thread-contract failure versus a collective mismatch.
2. Trace `close` and `Universe` drop order on the failing driver: a
   free-after-finalize or missed `close` versus a wrong result.
3. Compare the Wire byte buffer before and after the collective at one rank:
   a datatype or contiguous-mapping error versus an algorithm difference.

A native compile or link failure (an MPI-3 symbol that MS-MPI lacks, for
example) is a HANDOFF naming the symbol, not a reason to drop the native
gate or to relabel WSL evidence.

## 5. After R1

- S1 (sparse automatic planning) proceeds as written in plan.v1 section 3,
  unchanged, on the rsmpi entry.
- libmuffintin and the fftw fork are not edited under this plan. Both
  consumers will then demand a `&Universe`, and libmuffintin's process-wide
  setter is the outlier; that is a separate decision (ADR-0007
  consequences).

## 6. Boundaries

- No tolerance, metric, fixture, or driver change.
- No dual API, no deprecated shim for `Runtime::initialize`, no
  `user-operations` feature, no libffi.
- Thread markers are not removed; the thread checks are not downgraded to a
  request.
- No implicit collective on Drop anywhere in ctf-rs.
- No sparse implementation before R1 closes.
