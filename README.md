# libmuffintin

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/weiyiguo9/libmuffintin@main/assets/libmuffintin-logo.webp" alt="libmuffintin logo" width="300">
</p>

`libmuffintin` is a memory-safe experimental Rust library for algebra shared by muffin-tin electronic-structure methods. The long-term target includes FP-KKR, (L)APW(+lo), and the LMTO/EMTO/NMTO family; the first executable route is LAPW. The API remains subject to breaking changes, and the new DFT workflow is an implementation candidate rather than a production-validated materials code. The name follows libraries such as libxc, libpaw, and libcint.

The current M-A through M-Kb implementation candidate provides:

- `libmuffintin-core`: Hartree/Bohr units, complex and real spherical harmonics,
  SPEX-convention complex Gaunt coefficients, real Gaunt coefficients,
  spherical Bessel functions, exponential radial meshes and quadrature,
  reciprocal-vector generation, and analytic interstitial step-function
  Fourier coefficients, and validated Dirac `Kappa`, exact `TwiceMu`,
  spinor-harmonic, and spinor-Gaunt contracts;
- `libmuffintin-radial`: nonrelativistic and scalar-relativistic valence radial
  solutions, energy derivatives, local orbitals, radial integral blocks, and
  a separate spherical four-component Dirac bound-core solver, plus regular
  fixed-energy Dirac valence functions with physical $(P,Q)$ traces and an
  explicit SRA $(U,U_r)$ adapter. The active M-Ka contract extends this route
  with analytic second energy derivatives and confined SRA HDLOs;
- `libmuffintin-grid`: typed atom-centred, uniform, interstitial, and stable composite
  quadrature grids, with an optional `rstsr` tensor conversion feature;
- `libmuffintin-sphere`: $(L,M)$-resolved sphere fields and Gaunt-weighted radial matrix
  elements for complex or real harmonics, plus a parallel typed spinor path
  that keeps the large-large and small-small radial/angular factors separate;
- `libmuffintin-io`: independently versioned, human-diffable TOML formats for physical
  snapshots and materialized grid artifacts. The FLEUR converter remains
  frozen;
- `libmuffintin-tensor` M-Fb: backend-neutral `einsum` layer for dense complex tensors.
  The default backend is RSTSR 0.7.10 linked with TBLIS. tenferro-rs is the
  optional second engine behind the same subscripts. LAPW eigenvectors use
  column-major `[basis, band]` storage, with each band column contiguous. Site muffin-tin
  contributions are `einsum("ci,cd,dj->ij", [P^*, B, P])`. faer remains the
  Hermitian eigensolver. Serialized artifacts stay backend-neutral;
- `libmuffintin-envelope`, `libmuffintin-basis`, `libmuffintin-operators`, and
  `libmuffintin-recipes` M-G: a concrete `PlaneWaveEnvelope` that owns its
  plane-wave set; historical-method-name-free v0.2 `BasisBlock` variants
  (`PlaneWaveEnvelope` with APW site augmentation plus `ConfinedSite`
  overlays, not a generic payload or trait hierarchy); `compile(&BasisSpec)`
  to a host `CompiledBasis` that retains APW site geometry; generic
  $P^\dagger B P$ operator assembly and the overlap-filtered eigensolver;
  plus the `recipes::lapw()` APW+lo preset;
- `libmuffintin-lapw` remains the LAPW facade: it builds the same M-F
  matrices by `recipes::lapw` $\to$ `compile` $\to$ `assemble_compiled`,
  keeps SPEX APW matching, explicit Rayleigh and site phases, the SPEX
  symmetric-Laplacian interstitial kinetic convention, collinear spin
  channels without SOC, and $(k,\mathrm{band})$ reference reports. An
  explicit `BasisSpec` route must not call `recipes::lapw()`. M-Ka adds a
  parallel `SpinorCompiledBasis`, `spin * n_g + g` Pauli-PW order, typed
  spinor site projection, equal-spin SRA interstitial assembly, and an
  explicit Hartree-convention Schlosser--Marcus surface bilinear form. The
  analytic second-derivative/HDLO path and a repository-local non-empty-sphere
  large-$c$ frozen-fixture reduction are internally tested; this is not yet a
  completed M-Ka acceptance or a cross-code validation against a frozen
  FlapwMBPT/source-equivalent band fixture;
- `libmuffintin-product` and `libmuffintin-mpb` M-H: a historical-method-name-free
  product-space IR (`ProductPartition`, `ProductSource` without a compiled
  one-particle basis, untruncated muffin-tin products, capability-supplied
  raw interstitial orbital-pair reciprocal support, `CompiledAuxiliaryBasis`
  with per-site meshes, and `PairVertex` in SPEX $site\to L\to M\to n$ then
  interstitial order). The MPB auxiliary $|q+G|\le g_{\mathrm{cut}}$ set is
  constructed separately in `libmuffintin-mpb` and is not the raw pair
  support. `TOL` is recorded on the retained auxiliary basis only.
  Finite-$q$ kinematics, Umklapp, and analytic interstitial pair vertices
  are included. There is no live SPEX untruncated numerical dump.
  `CompiledAuxiliaryBasis` stores a typed mixed-product or
  interpolation-point payload. Production $V^q$ is M-J;
- `libmuffintin-thc` M-I: k-point ISDF/THC on a finite periodic toy basis.
  Pair columns use the canonical-$q$ / Umklapp gauge. Selectors `q0_l2`,
  `allq_l2`, and `allq_coulomb_pool` are compared at identical $N_\mu$. The
  production L2 default is `allq_l2`; full selection can use either QRCP or
  pivoted Cholesky without materializing the weighted point Gram.
  Coulomb-aware ranking consumes injected pair-pair Grams; the crate does not
  assemble Weinert or SPEX $V^q$. Recorded Python finite-cutoff numbers are
  candidate-oracle evidence, not a real-material accuracy claim;
- `libmuffintin-coulomb` M-J: a representation-neutral finite-$q$
  Weinert/SPEX Coulomb operator over `CompiledAuxiliaryBasis`. Mixed-product
  auxiliaries use `assemble_coulomb`. Interpolation-point auxiliaries use
  `assemble_sampled_coulomb` with parent-grid $\zeta^q$ samples (not
  interpolation-node point charges). Pair vertices carry an exact
  `AuxiliaryLayout`. The public assembler does not take
  `libmuffintin-mpb` or `libmuffintin-thc` types. Direct Ewald-summed
  $1/r$ is a toy oracle only. There is no live SPEX $V^q$ dump and no
  Coulomb/THC/GW production consumer;
- `libmuffintin-dft` M-Kb: unified charge-plus-Cartesian-magnetization density synthesis with the physical step-function metric; per-iteration four-component $P^2+Q^2$ core density; general Weinert electronic Hartree plus periodic nuclei; LDA/PW92 and PBE with `LocalSpinFrame` and `MagnetizationField` noncollinear reductions; overflow-safe Fermi--Dirac and Gaussian occupations with their distinct variational corrections; linear, type-2 Broyden, and Pulay--Anderson mixing; total-energy and SCF state machines; scalar Koelling--Harmon, optional nonmagnetic SPEX-style second-variation SOC, and a noncollinear four-component first-variation route; frozen-potential bands; and regular-mesh tetrahedron DOS. State degeneracy and total-versus-core electron counting are explicit.
- `libmuffintin-runtime`: the single `muffintin` binary plus a reusable library boundary. One versioned TOML input carries an ordered `workflow.tasks` array and `[task.<id>]` blocks with nested arrays and subblocks; later tasks consume typed outputs such as `scf.state`. The registry currently executes DFT SCF, bands, and DOS and is intentionally not DFT-named so future THC tasks and a Python interface can share the same modular runtime.

All in-memory energies are Hartree and all lengths are Bohr.  Producer-specific
units and potential normalizations must be converted at an I/O boundary.
The normative convention summary is in [CONVENTIONS.md](CONVENTIONS.md), and
the numbered formula derivations are under [`doc/`](doc/).

## Build and test

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p libmuffintin-grid --features rstsr
cargo test -p libmuffintin-tensor
cargo test -p libmuffintin-lapw
```

The workspace MSRV is Rust 1.85. `libmuffintin-tensor` links TBLIS through `tblis-src`.
First builds need a TBLIS git tree: set `TBLIS_SRC` to a clone with
submodules, for example `/tmp/tblis.git`, or to
`https://github.com/MatthewsResearchGroup/tblis.git`.

The optional `backend-tenferro` feature uses `tenferro-einsum` 0.3.0, which
requires rustc 1.96. Leave the workspace at 1.85 unless you enable that
feature; if you do, raise `rust-version` to 1.96 in the root `Cargo.toml` or
the tenferro backend will not compile.

```sh
cargo test -p libmuffintin-tensor --features backend-tenferro
```

The implementation is cross-referenced against FLEUR conventions and
the FlapwMBPT radial formalism.  Reference paths and exact source symbols are
recorded in the numbered derivation notes.

## Scope boundary

The M-Kb implementation contract is closed across the library, ordered TOML workflow, single executable, and versioned snapshot/restart boundary, but it is not yet cross-code accepted or production validated. Snapshot V2 carries $n,m_x,m_y,m_z$ and $V_0,B_x,B_y,B_z$ for frozen noncollinear input and restart; Snapshot V1 remains readable through exact scalar or up/down normalization. Focused and one-site end-to-end gates do not replace the planned Si/SrVO3 scalar, Pt/Au second-variation SOC, collinear and noncollinear magnetic, and cross-code tetrahedron-DOS fixtures. The older Cu frozen-potential one-meV gate and producer-specific FLEUR conversion also remain outside the current acceptance evidence, so a `v0.1` release tag is not claimed.
