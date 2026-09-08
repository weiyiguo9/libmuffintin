# ctf-rs binds MPI through rsmpi; the host owns the Universe

- Status: proposed 2026-09-09 (user direction; acceptance of `plans/ctf-rs/plan.v2.md` closes it)
- Date: 2026-09-09
- Supersedes: the ADR-0006 "MPI binding rule" lines on raw-handle crossing (`AsRaw` outward, `FromRaw` inward) and on a ctf-rs constructor that adopts a raw `MPI_Comm`

## Decision

- Initialization and communicator ownership are unified, not the wrappers.
  ctf-rs stops calling `MPI_Init` and `MPI_Finalize`; a `Context` is built
  from the host's rsmpi `Universe` and a communicator, and borrows the
  `Universe` so nothing in ctf-rs outlives it.
- Communicators cross crate boundaries as rsmpi types (`&Universe`,
  `SimpleCommunicator`, `impl Communicator`), not as raw handles. rsmpi's
  `FromRaw` is owning and forbids `MPI_COMM_WORLD`, so a raw-handle contract
  cannot express a borrowed world.
- One FFI layer stays `mpi-sys` 0.2, reached through `mpi::ffi`.
  Hand-written wrappers may coexist with the safe API. What may not coexist
  is a second initializer, an implicit or foreign `MPI_Comm_free`, or a
  thread level the host did not provide.
- The change is one breaking change to ctf-rs (`plans/ctf-rs/plan.v2.md`),
  with no dual API.

## Why

The three MPI users of this stack agree on the ABI (`mpi-sys` 0.2.4, one
copy per build) and disagree on ownership: libmuffintin and the fftw fork
expect a host `Universe` at Funneled; ctf-rs initializes at Single and
asserts nobody did before. Two ownership models in one process cannot both
be right, and rsmpi is the one layer that encodes ownership in types.

The user's review (2026-09-09) narrowed the earlier claim: rsmpi does not
cover everything (no MPI-IO, no variable-count reduce-scatter, no safe
`is_thread_main`), the closure user operations pull libffi, requesting
Funneled is not receiving it, and swapping types does not order destruction
by itself. The plan carries each of these as a rule.

## Consequences

- ctf-rs gains `mpi` 0.8 with default features off, loses its direct
  `mpi-sys` line, and keeps `mpi::ffi` for MPI-IO and any collective rsmpi
  does not wrap.
- Every ctf-rs test and example main initializes through rsmpi at Funneled
  and checks the provided level.
- After R1 both consumers demand a `&Universe`; libmuffintin's process-wide
  setter (`set_hf_mpi_communicator`) becomes the outlier. Whether the
  library takes a `&Universe` or the setter stays is a later ADR, not part
  of plan.v2.
- ADR-0006's first step (world-only setter, one Allreduce per k point) is
  unaffected.
