# libmuffintin

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/weiyiguo9/libmuffintin@main/assets/libmuffintin-logo.webp" alt="libmuffintin logo" width="300">
</p>

`libmuffintin` is a memory-safe Rust experimental library for the algebra shared by
muffin-tin electronic-structure methods.  The long-term target includes
FP-KKR, (L)APW(+lo), and the LMTO/EMTO/NMTO family; the v0.1 route is LAPW
first. Note that this library is not for production DFT calculations, and the API is subject to further changes.  The library is intended to be used as experiments to unify the foudamental muffin-tin basis in electronic structure methods and abstrct the common/heavy relyed functions. Apparantly named after libxc, libpaw, libcint.

The current M-A through M-F implementation candidate provides:

- `mt-core`: Hartree/Bohr units, complex and real spherical harmonics,
  SPEX-convention complex Gaunt coefficients, real Gaunt coefficients,
  spherical Bessel functions, exponential radial meshes and quadrature,
  reciprocal-vector generation, and analytic interstitial step-function
  Fourier coefficients;
- `mt-radial`: nonrelativistic and scalar-relativistic valence radial
  solutions, energy derivatives, local orbitals, radial integral blocks, and
  a separate spherical four-component Dirac bound-core solver;
- an explicitly reserved valence 4c Dirac interface. Full valence 4c support
  also needs spinor augmentation and assembly and is not claimed here;
- `mt-grid`: typed atom-centred, uniform, interstitial, and stable composite
  quadrature grids, with an optional `rstsr` tensor conversion feature;
- `mt-sphere`: $(L,M)$-resolved sphere fields and Gaunt-weighted radial matrix
  elements for complex or real harmonics;
- `mt-io`: independently versioned, human-diffable TOML formats for physical
  snapshots and materialized grid artifacts. The FLEUR converter remains
  frozen;
- `mt-tensor` M-Fb1: backend-neutral `einsum` layer for dense complex tensors.
  The default backend is RSTSR 0.7.10 linked with TBLIS. tenferro-rs is the
  planned second backend behind the same subscripts. Site muffin-tin
  contributions are `einsum("ci,cd,dj->ij", [P^*, B, P])`. faer remains the
  Hermitian eigensolver. Serialized artifacts stay backend-neutral;
- `mt-lapw` M-D: SPEX-convention APW boundary matching, explicit Rayleigh and
  site phases, and dense complex overlap assembly through the empty-sphere
  $S=I$ regression;
- `mt-lapw` M-E: SPEX symmetric-Laplacian Hamiltonian assembly, full Hermitian
  muffin-tin blocks for spherical and warped potentials, a `faer`-based
  overlap-filtered generalized eigensolver, and $(k,\mathrm{band})$ reference
  reports;
- `mt-lapw` M-F: one $[\mathrm{APW}][\text{site LO}]$ basis layout and
  $P^\dagger(S/H)P$ assembly for APW--APW, APW--LO, and LO--LO blocks, plus
  independent collinear up/down channels without SOC.

All in-memory energies are Hartree and all lengths are Bohr.  Producer-specific
units and potential normalizations must be converted at an I/O boundary.
The normative convention summary is in [CONVENTIONS.md](CONVENTIONS.md), and
the numbered formula derivations are under [`doc/`](doc/).

## Build and test

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p mt-grid --features rstsr
cargo test -p mt-tensor
```

The workspace MSRV is Rust 1.85. `mt-tensor` links TBLIS through `tblis-src`.
First builds need a TBLIS git tree: set `TBLIS_SRC` to a clone with
submodules, for example `/tmp/tblis.git`, or to
`https://github.com/MatthewsResearchGroup/tblis.git`.

The optional `backend-tenferro` feature uses `tenferro-einsum` 0.3.0, which
requires rustc 1.96. Leave the workspace at 1.85 unless you enable that
feature; if you do, raise `rust-version` to 1.96 in the root `Cargo.toml` or
the tenferro backend will not compile.

The implementation is cross-referenced against FLEUR conventions and
the FlapwMBPT radial formalism.  Reference paths and exact source symbols are
recorded in the numbered derivation notes.

## Scope boundary

This is a frozen-input LAPW operator engine, not yet a self-consistent DFT code.
M-F includes local-orbital basis rows and collinear spin-channel drivers, but
not SCF, SOC, noncollinear spin, occupations, or potential construction. The
real Cu-versus-SPEX one-meV gate still needs a matching frozen
potential/basis/eigenvalue fixture; synthetic empty-lattice tests do not replace
that evidence, so the README overlay and `v0.1` release tag are not yet claimed.
Producer-specific FLEUR conversion is intentionally frozen.
