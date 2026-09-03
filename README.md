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

- `libmuffintin-core`: Hartree/Bohr units, real and complex spherical
  harmonics and Gaunt coefficients in SPEX conventions, spherical Bessel
  functions, exponential radial meshes and quadrature, reciprocal vectors,
  step-function Fourier coefficients, the Dirac `Kappa`, `TwiceMu`, and
  spinor-harmonic contracts, and typed atom-centred, uniform, interstitial,
  and composite quadrature grids (optional `rstsr` conversion feature).
- `libmuffintin-symmetry`: the method-neutral crystal symmetry dataset
  (integer fractional-basis operations with a time-reversal flag, orbit
  representatives, space-group classification) with a moyo detection
  backend under `moyo_backend::` and the SPEX import contract under
  `spex::`, so external codes populate the same IR instead of re-detecting.
  The Python-side mirror lives in `pymuffintin.symmetry` over spglib and
  spgrep. The contracts live in
  [`doc/22`](doc/22_crystal_symmetry_and_spex_irrep_import.md).
- `libmuffintin-tensor`: backend-neutral `einsum` over dense complex tensors:
  RSTSR linked with TBLIS by default, `tenferro` as an optional second engine,
  faer as the Hermitian eigensolver, and column-major `[basis, band]`
  eigenvector storage.
- `libmuffintin-sphere`: everything inside the muffin-tin sphere:
  nonrelativistic, scalar-relativistic, and four-component Dirac radial
  solutions with energy derivatives and local orbitals, the bound-core
  solver, physical $(P,Q)$ traces, an explicit SRA adapter, and sphere
  fields resolved in $(L,M)$ with Gaunt-weighted radial matrix elements
  plus a parallel typed spinor path.
- `libmuffintin-envelope`: plane-wave envelopes with explicit Rayleigh and
  site phases, and the method-name-free anonymous basis: `BasisBlock`
  variants and `compile(&BasisSpec)` to a host `CompiledBasis`.
- `libmuffintin-operators`: generic $P^\dagger B P$ operator assembly with
  the overlap-filtered eigensolver; basis recipes under `recipes::` with
  the `recipes::lapw()` APW+lo preset; and the LAPW facade under `lapw::`:
  SPEX APW matching, the SPEX symmetric-Laplacian interstitial kinetic
  convention, collinear spin without SOC, and the parallel spinor compiled
  basis with equal-spin SRA assembly for the Dirac valence route.
- `libmuffintin-prodbasis`: the method-neutral product-space IR
  (`AuxiliaryPartition`, `AuxiliarySource`, `CompiledAuxiliaryBasis`,
  `PairVertex` in SPEX ordering) with two producers as modules: the SPEX
  mixed-product auxiliary basis under `mpb::` with finite $q$ kinematics
  and Umklapp, and k-point ISDF/THC selection and fitting under `thc::`
  with QRCP and pivoted-Cholesky engines behind one result type.
- `libmuffintin-coulomb`: the representation-neutral finite $q$ Weinert/SPEX
  Coulomb operator over `CompiledAuxiliaryBasis`, for mixed-product and
  sampled interpolation-point auxiliaries. It consumes only root
  product-basis IR types by documented convention.
- `libmuffintin-dft`: the checkpoint-backed `MaterialKernel` for DFT/SCF
  physics: density initialization and synthesis with noncollinear
  magnetization, four-component core densities, Weinert electrostatics with
  periodic nuclei and the per-iteration `build_scf_potential`, LDA/PW92 and
  PBE, radial and basis materialization, scalar and spinor eigensolutions,
  occupations and mixing, total energy and SCF state machines,
  frozen-potential bands, tetrahedron DOS, channel-energy generators,
  free-atom LDA, and the periodic neutral-atom superposition.
- `libmuffintin-hf`: a finite-basis closed-shell restricted Hartree–Fock
  state machine over caller-supplied overlap, one-electron, and real
  chemist-order four-index integrals. It performs genuine Fock feedback,
  density mixing, generalized eigensolves, and energy/density convergence.
  It remains the external AO oracle rather than the LAPW production driver.
- `libmuffintin-io`: versioned, human-diffable TOML checkpoint and grid
  formats; the MLDUMP v1 HDF5 interchange schema (`libmuffintin.mldump`,
  neither CoQui-native nor SPEX-native); the SPEX `spex.snapshot_hdf` v1
  reader; the `libmuffintin.spexsym` v1 symmetry+irrep dump reader and
  reference writer; and the CoQui-native scalar Cholesky writer. The FLEUR
  converter remains frozen.
- `libmuffintin-runtime`: the single `muffintin` binary and its library
  boundary: ordered TOML workflows with Input V3 and the `CheckpointPhysics`
  checkpoint/IO/orchestration shell, which delegates DFT/SCF physics to
  `MaterialKernel`; the explicit-layout neutral atomic-start generator
  `materialize_atomic_start`; the runtime-owned frozen scalar and
  spinor product-input, mixed-product, THC, sampled-Coulomb, natural-grid,
  frozen-orbital ISDF exchange bridges, exact full-VV MPB exchange, and the
  full-regular-BZ spinor-first valence-only HF SCF engine with a strict Gamma
  wrapper; retained Dirac core-orbital sidecars, rectangular VV/CV/VC/CC MPB
  vertices, frozen one-shot sector exchange, and its independent radial
  Slater-trace comparison; and the runtime-owned MLDUMP and CoQui writers. The
  bridge contracts live in
  [`doc/17`](doc/17_minimal_dft_scf.md)–[`doc/20`](doc/20_sm_dy_full_spinor_material_demo.md).

All in-memory energies are Hartree and all lengths are Bohr; producer-specific
units and potential normalizations are converted at I/O boundaries. The
normative convention summary is in [CONVENTIONS.md](CONVENTIONS.md).

## Build and test

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p libmuffintin-core --features rstsr
cargo test -p libmuffintin-tensor
cargo test -p libmuffintin-operators
cargo test -p libmuffintin-hf
```

The workspace MSRV is Rust 1.89, set by `libmuffintin-symmetry`'s moyo
dependency (which pins `nalgebra` 0.35).
`libmuffintin-tensor` links TBLIS through `tblis-src`.
Tensor storage shares one lazily initialized RSTSR device and its Rayon pool;
allocating an array does not create another worker pool. Configure the Rayon
thread budget before the first tensor allocation, for example with
`RAYON_NUM_THREADS`. RSTSR sets TBLIS's thread count for each contraction from
that device, so `TBLIS_NUM_THREADS` alone does not override this budget.

By default, `tblis-src` builds TBLIS from source; first builds need a TBLIS git
tree, supplied with `TBLIS_SRC` as a clone with submodules or as
`https://github.com/MatthewsResearchGroup/tblis.git`. To use an existing
system installation instead, set `TBLIS_DIR` to its installation prefix. This
selection is environment-local and does not change the default source build.

The optional `backend-tenferro` feature uses `tenferro-einsum` 0.3.0, which
requires rustc 1.96. Leave the workspace at 1.89 unless you enable that
feature; if you do, raise `rust-version` to 1.96 in the root `Cargo.toml` or
the tenferro backend will not compile.

`libmuffintin-python` pins `pyo3` 0.27.2 and `numpy` 0.27.1; both have MSRV
1.74, comfortably under the workspace floor. The extension uses `abi3-py310`
and is built locally with `maturin develop`; no wheel is published.

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

The local-only PyO3/maturin binding now exposes the frozen scalar and spinor
product/THC/Coulomb boundaries as versioned NumPy structures and exposes the
production DFT-SCF loop through one global entry plus linear staged handles.
The separate backend-neutral `pymuffintin` research package consumes those
exports; it is not part of this workspace. Neither package is published, and
this work adds no importer, wheel, or material-accuracy claim.

[`examples/relativistic_hf`](examples/relativistic_hf/) contains an external
PySCF NR/sf-X2C1e/X2C1e/4c-DC HF comparison and its Kr/Dyall-v2z report. It is
a Gaussian-basis quantum-chemistry diagnostic, not a Koelling–Harmon versus
FRA-LAPW acceptance test. The exact MPB route now performs valence-only
self-consistent Fock feedback on a caller-specified finite full-BZ mesh. The
Gamma API is the `1x1x1`, zero-shift wrapper over that engine. This is not a
box-size, basis, product-cutoff, or k-mesh convergence result and is not a
converged periodic HF claim. The scalar KH plus SOC driver accepts either
`ValenceOnly` or `Frozen` core treatment. Frozen core orbitals are solved once
in the immutable checkpoint potential; their density stays in the total-density
Hartree/mixing loop. Static core exchange is evaluated exactly from every
core–valence radial product: its spherical average enters scalar KH Fock
iterations and its SOC-resolved operator enters final second variation. The VV
MPB/THC space is not enlarged with CV, VC, or CC products. Relaxed core remains
available only in the full first-variation spinor HF driver; the KH plus SOC
route deliberately has no `RelaxedCore` variant until a scalar/SOC valence
density can generate the four-component VC radial action without a signed
$\kappa$ surrogate. ISDF/THC remains a frozen-orbital optional backend.

Crystal symmetry is currently detection and classification only: the
`libmuffintin-symmetry` dataset is not yet consumed by the SCF loop, and no
k-point reduction, irrep projection, or symmetrization claim is made.
