# libmuffintin

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/weiyiguo9/libmuffintin@main/assets/libmuffintin-logo.webp" alt="libmuffintin logo" width="300">
</p>

`libmuffintin` is a memory-safe experimental Rust library for algebra shared by
muffin-tin electronic-structure methods. The long-term target includes FP-KKR,
(L)APW(+lo), and the LMTO/EMTO/NMTO family; the first executable route is LAPW.
The API remains subject to breaking changes, and the DFT workflow is an
implementation candidate rather than a production-validated materials code. The
name follows libraries such as libxc, libpaw, and libcint.

## Workspace

One crate per boundary; the numbered notes under [`doc/`](doc/) carry the
exact contracts and derivations.

- `libmuffintin-core` — Hartree/Bohr units, real and complex spherical
  harmonics and Gaunt coefficients in SPEX conventions, spherical Bessel
  functions, exponential radial meshes and quadrature, reciprocal vectors,
  step-function Fourier coefficients, and the Dirac `Kappa`, `TwiceMu`, and
  spinor-harmonic contracts.
- `libmuffintin-radial` — nonrelativistic, scalar-relativistic, and
  four-component Dirac radial solutions: valence functions with energy
  derivatives and local orbitals, the bound-core solver, physical $(P,Q)$
  traces, and an explicit SRA adapter.
- `libmuffintin-grid` — typed atom-centred, uniform, interstitial, and
  composite quadrature grids, with an optional `rstsr` conversion feature.
- `libmuffintin-sphere` — sphere fields resolved in $(L,M)$ and Gaunt-weighted
  radial matrix elements, with a parallel typed spinor path.
- `libmuffintin-io` — versioned, human-diffable TOML checkpoint and grid
  formats; the MLDUMP v1 HDF5 interchange schema (`libmuffintin.mldump`,
  neither CoQui-native nor SPEX-native); the SPEX `spex.snapshot_hdf` v1
  reader; and the CoQui-native scalar Cholesky writer. The FLEUR converter
  remains frozen.
- `libmuffintin-tensor` — backend-neutral `einsum` over dense complex tensors:
  RSTSR linked with TBLIS by default, `tenferro` as an optional second engine,
  faer as the Hermitian eigensolver, and column-major `[basis, band]`
  eigenvector storage.
- `libmuffintin-envelope`, `-basis`, `-operators`, `-recipes` — plane-wave
  envelopes, method-name-free `BasisBlock` variants, `compile(&BasisSpec)` to
  a host `CompiledBasis`, generic $P^\dagger B P$ operator assembly with the
  overlap-filtered eigensolver, and the `recipes::lapw()` APW+lo preset.
- `libmuffintin-lapw` — the LAPW facade: SPEX APW matching, explicit Rayleigh
  and site phases, the SPEX symmetric-Laplacian interstitial kinetic
  convention, collinear spin without SOC, and the parallel spinor compiled
  basis with equal-spin SRA assembly for the Dirac valence route.
- `libmuffintin-auxiliary-ir`, `-mpb` — the method-neutral product-space IR
  (`AuxiliaryPartition`, `AuxiliarySource`, `CompiledAuxiliaryBasis`,
  `PairVertex` in SPEX ordering) and the SPEX mixed-product auxiliary basis
  with finite $q$ kinematics and Umklapp.
- `libmuffintin-thc` — k-point ISDF/THC selection and fitting over the
  product-space IR, with QRCP and pivoted-Cholesky engines behind one result
  type. Recorded finite-cutoff numbers are candidate-oracle evidence, not a
  real-material accuracy claim.
- `libmuffintin-coulomb` — the representation-neutral finite $q$ Weinert/SPEX
  Coulomb operator over `CompiledAuxiliaryBasis`, for mixed-product and
  sampled interpolation-point auxiliaries.
- `libmuffintin-dft` — density synthesis with noncollinear magnetization,
  four-component core densities, Weinert electrostatics with periodic nuclei,
  LDA/PW92 and PBE, occupations and mixing, total energy and SCF state
  machines, frozen-potential bands, tetrahedron DOS, channel-energy
  generators, free-atom LDA, and the periodic neutral-atom superposition.
- `libmuffintin-runtime` — the single `muffintin` binary and its library
  boundary: ordered TOML workflows with Input V2, the neutral atomic-start
  generator `materialize_atomic_checkpoint_v2`, the frozen scalar and spinor
  product-input, mixed-product, THC, and sampled-Coulomb bridges, and the
  MLDUMP and CoQui writers. The bridge contracts live in
  [`doc/17`](doc/17_minimal_lda_scf.md)–[`doc/20`](doc/20_sm_dy_full_spinor_material_demo.md).

All in-memory energies are Hartree and all lengths are Bohr; producer-specific
units and potential normalizations are converted at I/O boundaries. The
normative convention summary is in [CONVENTIONS.md](CONVENTIONS.md).

## Build and test

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p libmuffintin-grid --features rstsr
cargo test -p libmuffintin-tensor
cargo test -p libmuffintin-lapw
```

The workspace MSRV is Rust 1.85. `libmuffintin-tensor` links TBLIS through `tblis-src`.
By default, `tblis-src` builds TBLIS from source; first builds need a TBLIS git
tree, supplied with `TBLIS_SRC` as a clone with submodules or as
`https://github.com/MatthewsResearchGroup/tblis.git`. To use an existing
system installation instead, set `TBLIS_DIR` to its installation prefix. This
selection is environment-local and does not change the default source build.

The optional `backend-tenferro` feature uses `tenferro-einsum` 0.3.0, which
requires rustc 1.96. Leave the workspace at 1.85 unless you enable that
feature; if you do, raise `rust-version` to 1.96 in the root `Cargo.toml` or
the tenferro backend will not compile.

```sh
cargo test -p libmuffintin-tensor --features backend-tenferro
```

The implementation is cross-referenced against SPEX, Elk, FLEUR, and
FlapwMBPT 2106 sources; the exact source symbols are recorded in the numbered
derivation notes.

## Scope boundary

The DFT contract, the orbital-configuration V2 extension, and the neutral
atomic-start generator close the v0.2 implementation sequence across the
library, ordered TOML workflow, single executable, and versioned
checkpoint/restart boundary. The real hashed SPEX `b45d9b9` Sm fcc artifact
completes the bounded $0.5\,a_0^{-1}$ spinor lane from target Dirac $P,Q$
through MPB, both THC engines, sampled Coulomb, and MLDUMP roundtrip, as
recorded in [`doc/20`](doc/20_sm_dy_full_spinor_material_demo.md).

None of this is a release tag, cross-code acceptance, or production
validation. Still open: full source-cutoff and rank/grid convergence, honest
Dy bcc producer data, an external spinor MLDUMP consumer, and the planned
Si/SrVO3 scalar, Pt/Au second-variation SOC, magnetic, and cross-code
tetrahedron-DOS fixtures. Checkpoint V1 remains readable through exact
normalization to Checkpoint V2.

The next planned step is a thin PyO3/maturin Python binding that would expose
the frozen scalar and spinor product/THC/Coulomb boundaries as versioned NumPy
structures in the separate backend-neutral `pymuffintin` package. That binding
is not published and adds no current physics, SCF, symmetry, importer, or
wheel claim.
