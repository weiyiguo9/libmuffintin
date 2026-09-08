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
feature, default off, bound through `mpi` (rsmpi) or `mpi-sys` 0.2 as ctf-rs
does, with the workspace MSRV unchanged at 1.89.

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
