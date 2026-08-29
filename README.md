# libmuffintin

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/weiyiguo9/libmuffintin@main/assets/libmuffintin-logo.webp" alt="libmuffintin logo" width="300">
</p>

`libmuffintin` is a memory-safe experimental Rust library for algebra shared by muffin-tin electronic-structure methods. The long-term target includes FP-KKR, (L)APW(+lo), and the LMTO/EMTO/NMTO family; the first executable route is LAPW. The API remains subject to breaking changes, and the new DFT workflow is an implementation candidate rather than a production-validated materials code. The name follows libraries such as libxc, libpaw, and libcint.

The current M-A through M-Kc implementation candidate provides:

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
- `libmuffintin-sphere`: sphere fields resolved in $(L,M)$ and Gaunt-weighted radial matrix
  elements for complex or real harmonics, plus a parallel typed spinor path
  that keeps the large-large and small-small radial/angular factors separate;
- `libmuffintin-io`: independently versioned, human-diffable TOML formats for physical
  snapshots and materialized grid artifacts, plus MLDUMP v1, a
  libmuffintin-owned inspectable HDF5 schema (`schema_name=libmuffintin.mldump`)
  for later runtime materialization. MLDUMP is not CoQui-native or SPEX-native.
  The FLEUR converter remains frozen;
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
  explicit Hartree-convention Schlosser–Marcus surface bilinear form. The
  analytic second-derivative/HDLO path and a repository-local non-empty-sphere
  large $c$ frozen-fixture reduction are internally tested; this is not yet a
  completed M-Ka acceptance or a cross-code validation against a frozen
  FlapwMBPT/source-equivalent band fixture;
- `libmuffintin-auxiliary-ir` and `libmuffintin-mpb` M-H: a historical-method-name-free
  product-space IR (`ProductPartition`, `ProductSource` without a compiled
  one-particle basis, untruncated muffin-tin products, capability-supplied
  raw interstitial orbital-pair reciprocal support, `CompiledAuxiliaryBasis`
  with per-site meshes, and `PairVertex` in SPEX $site\to L\to M\to n$ then
  interstitial order). The MPB auxiliary $|q+G|\le g_{\mathrm{cut}}$ set is
  constructed separately in `libmuffintin-mpb` and is not the raw pair
  support. `TOL` is recorded on the retained auxiliary basis only.
  Finite $q$ kinematics, Umklapp, and analytic interstitial pair vertices
  are included. There is no live SPEX untruncated numerical dump.
  `CompiledAuxiliaryBasis` stores a typed mixed-product or
  interpolation-point payload. Production $V^q$ is M-J;
- `libmuffintin-thc` M-I: k-point ISDF/THC on a finite periodic toy basis.
  Pair columns use the canonical $q$ / Umklapp gauge. Selectors `q0_l2`,
  `allq_l2`, and `allq_coulomb_pool` are compared at identical $N_\mu$. The
  production L2 default is `allq_l2`; full selection can use either QRCP or
  pivoted Cholesky without materializing the weighted point Gram.
  Coulomb-aware ranking consumes injected pair-pair Grams; the crate does not
  assemble Weinert or SPEX $V^q$. Recorded Python finite-cutoff numbers are
  candidate-oracle evidence, not a real-material accuracy claim;
- `libmuffintin-coulomb` M-J: a representation-neutral finite $q$
  Weinert/SPEX Coulomb operator over `CompiledAuxiliaryBasis`. Mixed-product
  auxiliaries use `assemble_coulomb`. Interpolation-point auxiliaries use
  `assemble_sampled_coulomb` with parent-grid $\zeta^q$ samples (not
  interpolation-node point charges). Pair vertices carry an exact
  `AuxiliaryLayout`. The public assembler does not take
  `libmuffintin-mpb` or `libmuffintin-thc` types. Direct Ewald-summed
  $1/r$ is a toy oracle only. There is no live SPEX $V^q$ dump, no
  GW/RPA/self-energy consumer, and no HDF5/CoQui or material/SPEX
  acceptance path;
- `libmuffintin-dft` M-Kb/M-Kc: unified charge-plus-Cartesian-magnetization density synthesis with the physical step-function metric; per-iteration four-component $P^2+Q^2$ core density; general Weinert electronic Hartree plus periodic nuclei; LDA/PW92 and PBE with `LocalSpinFrame` and `MagnetizationField` noncollinear reductions; overflow-safe Fermi–Dirac and Gaussian occupations with their distinct variational corrections; linear, type-2 Broyden, and Pulay–Anderson mixing; total-energy and SCF state machines; scalar Koelling–Harmon, optional nonmagnetic SOC second variation, and a noncollinear four-component first-variation route; frozen-potential bands; and regular-mesh tetrahedron DOS. The M-Kc generator seam implements `explicit`, `atomic`, `band-center`, `log-derivative`, `band-cog`, `fermi-offset`, and `frozen-snapshot`, including atomic generation starting from signed $\kappa$, physical projected-DOS weights, same-iteration spectral refinement, and retained provenance.
- `libmuffintin-runtime`: the single `muffintin` binary plus a reusable library boundary. Input V2 carries an ordered `workflow.tasks` array, `[basis.envelope]`, and spectroscopic channel tables; V1 orbital fields are rejected with a migration diagnostic and have no compatibility aliases. Core/valence partitions come from the typed FLEUR `default.econfig` catalogue for $Z=1\ldots103$, then pass through external recipe, task-generator, species, site-edit, and token-suffix layers. Full first variation automatically selects sixth-period $5p_{1/2}$ and supported seventh-period $6p_{1/2}$ atomic local orbitals. Later tasks consume typed outputs such as `scf.state`, including the exact materialized basis rather than mutable kernel state. The registry currently executes DFT SCF, bands, and DOS and is intentionally not DFT-named so future THC tasks and a Python interface can share the same modular runtime. M-L1 adds `SnapshotDftPhysics::scalar_product_input`: a frozen scalar LAPW solve that emits `ProductSource` plus per-spin Bloch eigenvectors, the exact per-$k$ `CompiledBasis` (outside `ProductSource`), canonical Cartesian $q$ with $q_{\mathrm{in}}=q_{\mathrm{canonical}}+G_{\mathrm{transfer}}$, the exact snapshot `ReciprocalLattice`, $k-q$ mesh mapping that rejects off-mesh transfers, full raw interstitial $G_k-G_{k-q}+G_{\mathrm{wrap}}$ support, and semantic `PairColumnLayout` from `libmuffintin-auxiliary-ir`. M-L2 adds `build_scalar_mpb`: a runtime-owned scalar mixed-product bridge that calls `spex_mixed_product_basis`, applies `TOL` with $n_{\mathrm{spin}}=2$, and contracts explicitly selected same-spin band pairs onto checked `PairVertex` records using the M-L1 `CompiledBasis`, $k-q$ map, and `PairColumnLayout`. The muffin-tin arm uses `CompiledSiteProjection` for APW $u$, $\dot u$, and LO rows; the interstitial arm uses $G_{\mathrm{right}}-G_{\mathrm{left}}+G_{\mathrm{wrap}}$ with $\mathrm{conj}(C_{\mathrm{left}})C_{\mathrm{right}}/\Omega$ and keeps the global `TransferQ` Umklapp as an MPB $\Theta_I$ input only. M-L3 adds `build_scalar_thc`: a runtime-owned scalar AllQL2 adaptive-THC bridge on an externally supplied `ThcParentGrid`. It evaluates frozen LAPW orbitals on stored muffin-tin radial samples and interstitial points in the cell-periodic representation, builds same-spin `PairBlock`s with $\exp(+i G_{\mathrm{wrap}}\cdot r)\,(P^*P+Q^*Q)$, and fits interpolation-point $\zeta$ with true quadrature weights. Callers choose `ThcEngine::FullColumnPivotedQr` or `FullPivotedCholesky` explicitly; both engines share one result type and the same pair-block/weight/rank path, and the selected engine is recorded on selection provenance. Auxiliaries are created with intended provenance before Bloch pair vertices. The result records the selected spin together with the parent grid, selection/rank, and per-$q$ interpolation-point auxiliary, $\zeta$, and pair vertices. M-L4 adds `build_scalar_coulomb`: a runtime-owned sampled-$\zeta$ Coulomb bridge that builds `SampledAuxiliaryFunctions` on the full M-L3 parent grid and calls production `assemble_sampled_coulomb`. The request reciprocal must match the frozen M-L1 lattice, and a construction fingerprint binds the parent-grid order to each $q$ $\zeta$ record. Gauge is unchanged; Gamma keeps finite-body plus `GammaHead` metadata. Matched M-L2/M-L3 pairs compare representation-neutral quadratic forms $c^\dagger V c$ with a stated exactness floor; per-side action norms remain debug diagnostics in each auxiliary basis and are not compared across representations. M-L5a added a parallel method-neutral Dirac product IR and MPB PP/QQ muffin-tin primitive (consumed here, not owned by runtime). M-L5b adds `SnapshotDftPhysics::spinor_product_input`: a frozen full-first-variation solve that emits `DiracProductSource` with physical $P$ and $Q$ ($n=0$ APW, $n=1$ energy derivative, $n=2+\mathrm{ordinal}$ signed-$\kappa$ LO/RLO), the exact per-$k$ `SpinorCompiledBasis` (Pauli $\mathrm{spin}\,N_G+G$ then site LO rows), canonical $q$ / $k-q$ / $G_{\mathrm{wrap}}$ shared with M-L1, raw interstitial $G_k-G_{k-q}+G_{\mathrm{wrap}}$ from actual spinor PW labels, and semantic `PairColumnLayout` with left band at $k-q$ and right band at $k$. Scalar and SOC-second-variation configs are typed-rejected. M-L5c adds `build_spinor_mpb`: a runtime-owned selected-band spinor mixed-product bridge that applies Dirac overlap cutoff with $n_{\mathrm{spin}}=1$ on the ordered PP/QQ union and contracts explicitly selected spinor band pairs onto checked `PairVertex` records with `OrbitalPair::Bloch`. The result seals a runtime-private frozen-input identity (same splitmix-style 64-bit mixer as the parent-grid fingerprint; collision residual one part in $2^{64}$ per comparison; not scientific provenance). Muffin-tin terms use exact site-projection identities, $\mathrm{conj}(d_{\mathrm{left}})d_{\mathrm{right}}$ times the inverse site phase, PP for large coordinates and QQ for small; interstitial terms are the same-component Pauli sum at $G_{\mathrm{right}}-G_{\mathrm{left}}+G_{\mathrm{wrap}}$. It does not copy Weinert assembly, inject Grams, export HDF5, or include core–valence products. M-L5d adds `build_spinor_thc`: a runtime-owned full-first-variation all-$q$ AllQL2 bridge on the same `ThcParentGrid`, reconstructing physical $P$/$Q$ with $\Omega_{\kappa\mu}$/$\Omega_{-\kappa\mu}$, summing same-Pauli PP+QQ (no PQ/QP/$cQ$), and reusing `fit_allq_l2_pair_blocks`. Interstitial evaluation uses the two Pauli PW blocks with G-only cell-periodic phases. `build_spinor_coulomb` builds full-grid `SampledAuxiliaryFunctions` and calls `assemble_sampled_coulomb`; matched M-L5c results must originate from `inputs[q_index]` (identity, then reciprocal and pair layout) before mixed-product assembly, and THC vertices must carry the compiled auxiliary provenance; matched pairs compare $c^\dagger V c$ only.

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

The implementation is cross-referenced against local SPEX, Elk, FLEUR, and
FlapwMBPT 2106 sources. Reference paths and exact source symbols are recorded
in the numbered derivation notes; the FlapwMBPT reference is the
`FlapwMBPT2106_type_B.tar.gz` source archive, not only the former ComDMFT tree.

## Scope boundary

The M-Kb DFT contract and M-Kc orbital-configuration V2 extension are closed across the library, ordered TOML workflow, single executable, and versioned snapshot/restart boundary, but they are not yet cross-code accepted or production validated. M-L1 is the scalar frozen-snapshot product-input boundary, M-L2 is the scalar mixed-product/selected-vertex bridge, M-L3 is the scalar AllQL2 interpolation-point/$\zeta$ seam, M-L4 is the sampled-$\zeta$ Coulomb bridge, M-L5a is the parallel Dirac PP/QQ IR and MPB primitive, M-L5b is the frozen full-first-variation spinor product-input boundary, M-L5c is the selected-band spinor mixed-product bridge, M-L5d is the spinor all-$q$ THC plus sampled-$\zeta$ Coulomb bridge, M-L6a is the versioned MLDUMP v1 HDF5 header/geometry/mesh interchange, M-L6b2 materializes frozen scalar product/THC/Coulomb objects into that schema through a streaming writer, and M-L6c2 materializes the corresponding frozen spinor objects; M-L6d1 writes a CoQui-native scalar full-BZ Cholesky file from those scalar objects and is not MLDUMP or CoQui full compatibility; they are not SPEX/material acceptance and do not complete M-L. Snapshot V2 carries $n,m_x,m_y,m_z$ and $V_0,B_x,B_y,B_z$ for frozen noncollinear input and restart; Snapshot V1 remains readable through exact scalar or up/down normalization. Focused and one-site end-to-end gates do not replace the planned Si/SrVO3 scalar, Pt/Au second-variation SOC, collinear and noncollinear magnetic, and cross-code tetrahedron-DOS fixtures. The older Cu frozen-potential one-meV gate and producer-specific FLEUR conversion also remain outside the current acceptance evidence, so a `v0.1` release tag is not claimed.
