# libmuffintin

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/weiyiguo9/libmuffintin@main/assets/libmuffintin-logo.png" alt="libmuffintin logo" width="300">
</p>

`libmuffintin` is a memory-safe Rust experimental library for the algebra shared by
muffin-tin electronic-structure methods.  The long-term target includes
FP-KKR, (L)APW(+lo), and the LMTO/EMTO/NMTO family; the v0.1 route is LAPW
first. Note that this library is not for production DFT calculations, and the API is subject to further changes.  The library is intended to be used as experiments to unify the foudamental muffin-tin basis in electronic structure methods and abstrct the common/heavy relyed functions.

The current M-A/M-B implementation provides:

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

All in-memory energies are Hartree and all lengths are Bohr.  Producer-specific
units and potential normalizations must be converted at an I/O boundary.
The normative convention summary is in [CONVENTIONS.md](CONVENTIONS.md), and
the numbered formula derivations are under [`doc/`](doc/).

## Build and test

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The implementation is cross-referenced against FLEUR conventions and
the FlapwMBPT radial formalism.  Reference paths and exact source symbols are
recorded in the numbered derivation notes.

## Scope boundary

This is not a self-consistent DFT code.  M-A/M-B stop at conventions and the
radial engine; LAPW matching, `(H,S)` assembly, eigensolving, snapshot I/O,
and FLEUR conversion belong to later milestones.
