# MPI parallelism in Hartree–Fock is pair-level, not grid-level

- Status: accepted 2026-09-08 (user decision)
- Date: 2026-09-08

## Decision

Distributed-memory parallelism for the exchange build splits the pair axis
(occupied left band × right band, per k point) across MPI ranks. Every rank
runs the driver redundantly (SPMD); only the interstitial pair-vertex
construction is distributed, and an allgather returns the complete vertex
list to every rank, so the Coulomb contraction and the Fock loop stay
unchanged. The FFTs stay rank-local. The capability is an optional `mpi`
feature, default off, with the workspace MSRV unchanged at 1.89.

Not chosen: slab or pencil decomposition of the pair FFT grid (FFTW-MPI in
rustnumgum/fftw). That is reserved for a single large density or THC grid.

## Why

At the H₂ A0 settings the pair FFT grid is 21³ (9261 points) and the ladder
tops out near 50³; distributing a transform of that size costs more in
transposes than it saves. The cost is the number of pair transforms
(234,840 per rebuild before spectrum caching), which is embarrassingly
parallel over pairs. This is the SPEX model (`wavefproducts.f`: MPI over k
points and product columns, FFT local per rank) and the Quantum ESPRESSO
EXX practice (band groups over pairs, few ranks per transform).

## Consequences

- First step: pair-level distribution inside the fft-fftw
  `contract_interstitial_selections`, results independent of the rank count.
- Later, as a separate decision: distribute the `[pair, aux]` vertex tensor
  itself and run the exchange contraction and Coulomb assembly through
  ctf-rs.
- The redundant SPMD driver is acceptable while the exchange build
  dominates the wall time; it is not a scaling claim.

## MPI binding rule (added 2026-09-08, user decision)

`mpi` and `mpi-sys` are one project (rsmpi/rsmpi; `mpi` 0.8.2 and
`mpi-sys` 0.2.4 both released 2026-07-09). rustnumgum/fftw's `mpi` feature
uses the safe `mpi` 0.8 API on top of `mpi-sys` 0.2; ctf-rs uses raw
`mpi-sys` 0.2 with its own `Comm` wrapper and its own `Context::initialize`.
Rule for every crate in this stack:

- One FFI layer: `mpi-sys` 0.2 from rsmpi, resolved to a single copy by
  Cargo, so `MPI_Comm` is one type everywhere.
- libmuffintin binds through the safe `mpi` 0.8 crate, like the fftw fork.
- One initializer: the binary calls `mpi::initialize_with_threading` at
  `Threading::Funneled` or higher (rayon threads live inside each rank; MPI
  calls stay on the main thread). Libraries check `mpi::is_initialized()`
  and never call `MPI_Init` or `MPI_Finalize` themselves.
- Communicators cross crate boundaries as raw handles: rsmpi `AsRaw::as_raw`
  outward, `FromRaw::from_raw` inward. ctf-rs needs a constructor that
  adopts an existing `MPI_Comm` without initializing or finalizing; that is
  a small ctf-rs change recorded under its own workstream.

## First step, refined (2026-09-08, after the evd-0011 profile)

The A1 Fock iteration is 57 percent exchange contraction and 31 percent
vertex build, both sums over occupied left bands. So the first step
distributes occupied left bands across ranks and keeps everything local:
each rank builds the pair vertices of its own occupied bands, contracts
them against the redundant Coulomb operator into a partial band-space
feedback, and one `Allreduce` (sum) of that small `[n_target, n_target]`
block per k gives every rank the full feedback. No vertex allgather. The
Coulomb assembly, MPB basis compile, spinor solve, and density work stay
redundant per rank and bound the speedup. The library reads the
communicator from a process-wide setter the binary fills (like
`set_hf_verbosity`), so no public spec type depends on rsmpi when the
feature is off.
