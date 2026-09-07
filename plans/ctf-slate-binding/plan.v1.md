# CTF / SLATE Rust binding — brief draft plan

- Workstream ID: `ctf-slate-binding`
- Plan version: 1
- Approval: frozen 2026-09-08 (superseded by `plans/ctf-rs/plan.v1.md`); history only, no further edits
- Supersedes: none
- Imported: 2026-09-08 from `scratch/ctf_slate_rust_binding_brief_plan.md` (last modified 2026-08-21), body unchanged
- Status source: false; state lives in `STATUS.md` and `ledger.md`

Status: local architecture draft, 2026-08-20.

This plan supports the v0.2 M-Fb tensor substrate. CTF is the first distributed
provider because it preserves global tensor rank, symmetry, index expressions,
automatic processor mapping, and redistribution. CTF's existing
ScaLAPACK-backed `Matrix` operations are the first distributed matrix solver.
SLATE is a later optional matrix provider for broader modern dense-linear-
algebra and accelerator paths.

## 1. Binding boundary

Do not bind `CTF::Tensor<T>` or SLATE templates directly. Add a small C++ shim
with explicit `double` and `std::complex<double>` instantiations and expose only
non-template opaque classes to Rust through `cxx`:

```text
CtfWorld
CtfTensorF64 / CtfTensorC64
CtfMatrixF64 / CtfMatrixC64
SlateMatrixF64 / SlateMatrixC64        later
```

The safe Rust crates own these through RAII handles. Public Rust APIs contain no
raw pointers, C++ references, `MPI_Comm`, ScaLAPACK descriptor arrays, or local
buffer addresses. `unsafe` is confined to the generated/sys bridge crate.

Exceptions thrown by the shim and recoverable CTF/SLATE APIs are caught at the
boundary and returned as structured status plus an owned error message. CTF
assertions, aborts, MPI failures, and a failure inside a collective are not
recoverable exceptions. The safe layer performs rank-synchronized preflight
validation before a collective and documents process termination as the
remaining failure mode.

## 2. Ownership and collective rules

- The first `CtfWorld` owns the CTF world object but borrows `MPI_COMM_WORLD` on
  the C++ side; the application retains MPI initialization/finalization
  ownership. Rust does not reinterpret an `MPI_Comm` value. Duplicated or
  subcommunicator-backed worlds are added only after each requested CTF matrix
  operation is capability-tested: current non-square-grid `eigh` has stricter
  `MPI_COMM_WORLD` assumptions.
- A tensor retains its world owner and cannot outlive it.
- Operations documented as collective, including global construction, I/O,
  contraction, redistribution, and matrix solves, execute in the same order on
  every rank. Destruction follows a deterministic tensor-before-world order;
  the shim does not invent barriers for operations that CTF defines as local.
- CTF handles are not `Send` or `Sync` initially. A later serialized executor
  may relax this only with evidence from the MPI thread level and CTF contract.
- Host transfer is explicit. Production code uses distributed coordinate or
  local-pair I/O; no implicit global gather is allowed.

## 3. First CTF API surface

The first binding is deliberately narrow but tensor-first:

1. create/query `World` rank and size;
2. create a global dense tensor from rank, global shape, symmetry, dtype, and
   `Placement::Auto`; no Rust-visible shard map is accepted. An expert explicit
   mapping option is deferred and, if ever added, remains backend configuration;
3. checked distributed coordinate read/write and explicit host import/export;
4. binary indexed contraction and indexed sum/transpose using ASCII labels;
5. norm and basic reductions needed by diagnostics;
6. rank-2 view/conversion to `CTF::Matrix` without moving ownership to Rust;
7. CTF/ScaLAPACK `cholesky`, triangular/SPD solve, QR, SVD, and Hermitian
   `eigh` where supported.

The complete multi-operand einsum specification is passed to the selected
backend so its native contraction-order and mapping planner can remain
effective. The CTF shim uses a supported native indexed-expression entry point
when one can be exposed safely; otherwise the CTF adapter privately plans
binary contractions while leaving every intermediate's processor mapping to
CTF. The common physics layer never freezes a binary tree. Binary contraction
is sufficient only for B0. The Rust layer validates label counts,
repeated-index rules, shapes, dtype, world identity, and legal output indices
before entering C++.

## 4. Generalized Hermitian solve

M-Fb keeps the overlap-filtering algorithm in Rust tensor code and uses CTF
matrix operations only for ordinary Hermitian eigendecomposition:

```text
S.eigh() -> retain positive overlap directions
X = U_keep * s_keep^(-1/2)
H_reduced = X^H * H * X
H_reduced.eigh()
C = X * Z
```

All products remain global CTF expressions. The solver reports retained rank,
filtered rank, overlap negativity, normalization, and residuals exactly as the
current local M-F implementation does.

## 5. Optional SLATE provider and CTF-to-SLATE bridge

SLATE is not required for the first CTF backend. Add it only after the
CTF/ScaLAPACK path passes the M-F parity suite or when a required solver/GPU
path justifies it.

Rust never obtains a CTF local-data pointer. The bridge is implemented entirely
inside C++ and returns an opaque `DistributedMatrix`/operation result:

```text
CTF Matrix -- C++ bridge --> SLATE Matrix
           <-- C++ bridge --
```

The bridge has two explicit modes:

- **borrowed view probe:** considered only when CTF exposes a stable supported
  local-buffer/descriptor interface and communicator, dtype, dimensions,
  block-cyclic descriptor, device, and lifetime are compatible. A C++ lease
  pins the CTF matrix, prevents conflicting mutation, and owns the SLATE view;
  the view cannot outlive the lease. If the required CTF interface is not
  public/stable, this mode is omitted rather than reaching into CTF internals.
- **owned redistribution:** allocate a SLATE-compatible destination and perform
  an explicit distributed copy/redistribution in C++. This is the required
  fallback and is never reported as zero-copy.

Every bridge operation returns a report containing provider, communicator size,
global shape, source/destination distribution, whether a copy or redistribution
occurred, and transferred bytes. Zero-copy is an optimization, not an API
promise.

## 6. Crate layout

```text
mt-tensor          global tensor/index/world contracts and provider traits
mt-ctf-sys         cxx bridge + C++ shim; the only FFI/unsafe boundary
mt-ctf             safe CTF tensor and CTF/ScaLAPACK provider
mt-slate-sys       optional later C++ bridge
mt-slate           optional later safe matrix provider and bridge reports
```

M-G may mechanically rename these to the `libmuffintin-*` package prefix. No
physical basis, radial, product-space, or snapshot crate imports a sys crate.

## 7. Milestones

### B0 — ABI and build probe

- pin exact CTF, MPI, BLAS, compiler, and optional ScaLAPACK versions;
- build the C++ shim and create/destroy F64/C64 worlds and tensors;
- run coordinate I/O and one complex contraction on 1 and 2 MPI ranks;
- prove recoverable shim/C++ exceptions become synchronized Rust errors;
- record assertion/abort and collective-failure cases as process-fatal.

### B1 — CTF tensor provider

- implement global tensor metadata and indexed expressions;
- verify contraction, transpose, conjugation, symmetry, and automatic mapping;
- compare against analytic fixtures and current M-F matrices, not a duplicate
  scalar backend;
- prohibit implicit gather and backend tensor types in serialized artifacts.

### B2 — CTF/ScaLAPACK matrix provider

- bind `eigh` and the minimal supporting matrix operations;
- reproduce overlap filtering, eigenvalues, subspaces, normalization, and
  residuals from M-F on supported square process grids, initially 1 and 4
  ranks; 2 ranks remains a contraction/I/O test and must reject unsupported
  `eigh` before entering CTF;
- record CTF process-grid constraints and any redistribution selected by CTF.

### B3 — optional SLATE experiment

- add opaque SLATE matrices and one Hermitian eigensolver;
- implement owned redistribution first; add a borrowed lease only if the
  supported CTF API can prove its lifetime and descriptor requirements;
- compare CTF/ScaLAPACK and SLATE accuracy, memory, communication, and GPU path;
- retain SLATE only if it supplies a measured capability or performance gain.

## 8. Acceptance and non-goals

Acceptance:

- no raw pointer or MPI handle in safe/public Rust APIs;
- no `unsafe` outside sys/generated binding code;
- deterministic RAII destruction with tensors destroyed before their world;
- F64/C64 contraction parity on 1/2/4 ranks and matrix parity on supported
  process grids;
- ASan/UBSan C++ shim tests plus Rust API and compile-fail lifetime tests;
- exact dependency/build provenance recorded with every distributed fixture;
- no silent gather, distribution change, device transfer, or zero-copy claim.

Non-goals for the first binding:

- arbitrary CTF element types, semirings, sparse tensors, or callbacks;
- runtime-loaded providers or cross-MPI-vendor ABI compatibility;
- Rust ownership of MPI initialization/finalization;
- GPU, automatic differentiation, fault tolerance, or production SLATE;
- exposing CTF or SLATE C++ template/expression types to Rust.
