# libmuffintin

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/weiyiguo9/libmuffintin@main/assets/libmuffintin-logo.webp" alt="libmuffintin logo" width="300">
</p>

`libmuffintin` is a memory-safe Rust experimental library for the algebra shared by
muffin-tin electronic-structure methods.  The long-term target includes
FP-KKR, (L)APW(+lo), and the LMTO/EMTO/NMTO family; the v0.1 route is LAPW
first. Note that this library is not for production DFT calculations, and the API is subject to further changes.  The library is intended to be used as experiments to unify the foudamental muffin-tin basis in electronic structure methods and abstrct the common/heavy relyed functions. Apparantly named after libxc, libpaw, libcint.

The current M-A through M-C foundation provides:

- `mt-core`: Hartree/Bohr units, complex and real spherical harmonics,
  SPEX-convention complex Gaunt coefficients, real Gaunt coefficients,
  spherical Bessel functions, exponential radial meshes and quadrature,
  reciprocal-vector generation, and analytic interstitial step-function
  Fourier coefficients;
- `mt-radial`: nonrelativistic and scalar-relativistic valence radial
  solutions, energy derivatives, local orbitals, radial integral blocks, and
  a separate spherical four-component Dirac bound-core solver;
- an explicitly reserved valence 4c Dirac interface. Full valence 4c support
  also needs spinor augmentation and assembly and is not claimed here.
- `mt-grid`: typed atom-centred, uniform, interstitial, and stable composite
  quadrature grids, with an optional `rstsr` tensor conversion feature;
- `mt-sphere`: `(L,M)`-resolved sphere fields and Gaunt-weighted radial matrix
  elements for complex or real harmonics;
- `mt-io`: independently versioned, human-diffable TOML formats for physical
  snapshots and materialized grid artifacts. The FLEUR converter remains
  frozen.
- `mt-lapw` M-D: SPEX-convention APW boundary matching, explicit Rayleigh and
  site phases, and dense complex overlap assembly through the empty-sphere
  `S=I` regression.

All in-memory energies are Hartree and all lengths are Bohr.  Producer-specific
units and potential normalizations must be converted at an I/O boundary.
The normative convention summary is in [CONVENTIONS.md](CONVENTIONS.md), and
the numbered formula derivations are under [`doc/`](doc/).

## Build and test

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mt-grid --features rstsr
```

The implementation is cross-referenced against FLEUR conventions and
the FlapwMBPT radial formalism.  Reference paths and exact source symbols are
recorded in the numbered derivation notes.

## Scope boundary

This is not yet a self-consistent DFT code. M-D includes LAPW boundary matching
and overlap assembly, but Hamiltonian assembly, eigensolving, local-orbital
basis rows, and spin-channel drivers remain later milestones. Producer-specific
FLEUR conversion is intentionally frozen.
