# 21. Python binding plan and export schema

This note is the v0.3 Python-binding contract and the home of the
`pyexport` interchange conventions. The binding adds no new physics:
it exports the existing frozen scalar and spinor product/THC/Coulomb
boundaries ([18](18_lapw_mpb_thc_integration.md),
[19](19_versioned_mldump_interchange.md)) to Python so that
auxiliary-basis experiments run without recompiling Rust. Release
sequencing and milestone status live in the `README.md` scope
boundary, not here.

## 1. Goal and non-goals

Python is the algorithm laboratory: muffin-tin local-RI metrics and
cutoffs, interstitial ISDF/THC point selection, muffin-tin/interstitial
stitching, and fixed-orbital exchange ablations. Rust remains the
stable kernel layer. The binding is a thin data-export ABI, not a
Python mirror of the Rust object model.

The laboratory itself is a separate backend-neutral package, `pymuffintin`
(basis/auxiliary/mbpt/optimize/diagnostics modules). This repository
ships only the binding, the `libmuffintin` Python package.
`libmuffintin` is the default `pymuffintin` backend and a regular
dependency: it holds the stable reference kernels. It stays behind
provider protocols (orbitals, local products, Coulomb) and is imported
lazily, only inside the `pymuffintin` muffintin adapter, so
`pymuffintin` itself still imports on a machine without the native
build and foreign-dump adapters remain first-class. The
dependency direction is strictly `pymuffintin` to `libmuffintin`, never the
reverse; future SPEX/FLEUR/Questaal/exciting/CoQui adapters are
`pymuffintin` work and never touch this repository. The contract between
the two packages is the pyexport schema of section 4 plus those
protocols, pinned on the schema version, not on Rust types or crate
versions. Protocol signatures are frozen from the first concrete
experiment and its muffintin adapter, not designed up front.

Out of scope for v0.3:

| Out of scope | Reason |
|---|---|
| Symmetry layer | Full-BZ experiments first; symmetry is a later prefactor optimization, not part of the algorithm question. |
| SPEX importer work | Already in `libmuffintin-io`; the binding loads its Checkpoint V2 output unchanged. |
| LMTO/NMTO producers | Later producers of the same export schema; the schema is method-neutral from day one. |
| Wheel or PyPI distribution | The extension links TBLIS and HDF5; local `maturin develop` only. |
| New Rust auxiliary representations | Hybrid MT-RI plus interstitial-THC compositions are assembled in Python from exported primitives. |
| Research algorithm layer in this repo | Lives in the separate backend-neutral `pymuffintin` package; this repo ships the binding only. |

## 2. Crate and package layout

```text
crates/mt-python/            # the only crate containing PyO3
  Cargo.toml                 # package libmuffintin-python, [lib] name muffintin_python, cdylib
  src/{lib.rs, checkpoint.rs, orbitals.rs, products.rs, thc.rs, coulomb.rs, export.rs}
python/
  pyproject.toml             # maturin backend, module-name = "libmuffintin._native"
  libmuffintin/
    __init__.py              # binding only: re-exports _native plus thin loaders
```

The `pymuffintin` research package is a separate repository and is not part
of this layout. Its planned shape, for reference:

```text
pymuffintin/
  pymuffintin/
    backends/muffintin.py    # the only place importing libmuffintin
    basis/{lapw,lmto,nmto,emto}.py
    auxiliary/{lri,isdf,thc,hybrid}.py
    mbpt/{hf,rpa,gw}.py
    optimize/{basis,screening,interpolation}.py
    diagnostics/{convergence,scaling,compare}.py
```

Constraints:

- Crate naming follows the repository rule: directory `mt-python`,
  package `libmuffintin-python`, library target `muffintin_python`.
- `mt-python` does not inherit the workspace lints table: PyO3 macros
  expand to `unsafe`, which the workspace `forbid(unsafe_code)` would
  reject. The crate carries a local lints table and a comment stating
  this exception.
- The workspace MSRV stays 1.85. PyO3 and rust-numpy are pinned to
  releases whose MSRV is at most 1.85; the pin is recorded in
  `README.md` the same way the tenferro exception is.
- abi3 is enabled so a venv Python upgrade does not force a rebuild.
- Build environment: the local venv, `TBLIS_DIR` for the system TBLIS,
  and the system HDF5 already required by `libmuffintin-io`.

## 3. Binding design rules

1. **Configuration enters as input V2 TOML.** The binding passes a
   path or string to the existing runtime input parser. The runtime
   `single_dft_scf_config` bridge requires exactly one `dft-scf` task;
   zero or multiple matching tasks are typed errors rather than an implicit
   task choice. No Rust configuration struct is mirrored into Python classes.
2. **Results are opaque handles.** `ScalarProductInput`,
   `ScalarMpbResult`, `ScalarThcResult`, `ScalarCoulombResult`, and
   their spinor twins are wrapped, not converted. Handles chain
   through the existing Rust bridges (THC to Coulomb to MLDUMP)
   without a Python array round trip, so the non-forgeable identity
   and fingerprint checks on those types keep working. Explicit
   `export_*` methods return NumPy structures for the experiment
   layer. The Python checkpoint, physics, and product-input handles share
   reference-counted checkpoint context. This preserves direct-lattice and
   site identity for exports without copying the checkpoint payload.
3. **Exports are versioned.** Every export dictionary carries
   `schema = "libmuffintin.pyexport"` and `version = 1`. Key-level
   schemas land in this document as each export is implemented; the
   conventions in section 4 are fixed now. Incompatible changes bump
   the version, matching the other `libmuffintin-io` interchange
   formats.

## 4. pyexport v1 conventions

- All energies are Hartree and all lengths are Bohr, as everywhere in
  memory.
- Coefficients are `complex128`. Eigenvectors keep the
  `libmuffintin-tensor` convention: column-major `[basis, band]` with
  each band column contiguous, one matrix for each $k$.
- Ragged data uses flat arrays plus explicit offset arrays. The one
  exception is eigenvectors, exported as a list of 2-d arrays for each
  $k$; experiment meshes are small and the copy is accepted.
- Auxiliary flatten order matches `AuxiliaryLayout`: the muffin-tin
  block then the interstitial block, with the exact region sequence
  exported as a structured table next to any coefficient array. Mixed
  product order is $site \to L \to M=-L..L \to n$ then $G$;
  interpolation points are muffin-tin (site, then id), interstitial
  id, then uniform id.
- Pair columns follow `PairColumnLayout`:
  $k\,N_{\mathrm{orb}}^2 + i\,N_{\mathrm{orb}} + j$.
- Radial functions are exported with their exponential-mesh
  parameters and the `ProductRadial` labels $l$, $n$, spin, using
  `SCALAR_RADIAL_U`, `SCALAR_RADIAL_UDOT`, and `SCALAR_RADIAL_LO0`
  offsets unchanged.
- THC parent grids enter from Python as arrays (Cartesian points,
  quadrature weights, region tags). Muffin-tin points must reference a
  stored site radial index, mirroring the existing parent-grid radial
  check; the binding does not relax that contract.

### 4.1 Stage 1 export dictionaries

Every dictionary in this section contains `schema =
"libmuffintin.pyexport"` and `version = 1`. Nested channel dictionaries do
not repeat that header. These tables are the complete Stage 1 key contract;
an implementation must not require consumers to consult another schema.

The field meanings and flatten orders agree with the scalar orbital and
product sections of MLDUMP v1 where the same physical data appears. This is
only a semantic cross-check: pyexport is an independent schema, is not
byte-compatible with MLDUMP, and does not depend on MLDUMP serialization.

`export_orbitals()` returns:

| Key | Python representation | Meaning |
|---|---|---|
| `k_fractional` | `float64[n_k, 3]` | Full-zone fractional $k$ points in production order. |
| `band_window_start` | `int` | First retained band, currently zero. |
| `band_window_count` | `int` | Common retained band count $n_{\mathrm{orb}}$. |
| `channels` | `list[dict]` | Stable spin-channel order; each dictionary has the fields below. |

Each `channels` element contains:

| Key | Python representation | Meaning |
|---|---|---|
| `spin` | `int` | Scalar spin channel, zero or one. |
| `energies` | `float64[n_k, n_orb]` | Retained eigenvalues in Hartree. |
| `eigenvectors` | `list[complex128[n_basis(k), n_orb]]` | One Fortran-contiguous basis-by-band matrix for each $k$. |
| `available_bands` | `int64[n_k]` | Available bands before the common leading window. |

`export_basis(k, spin)` returns:

| Key | Python representation | Meaning |
|---|---|---|
| `k_index` | `int` | Selected $k$ index. |
| `spin` | `int` | Selected scalar spin channel. |
| `basis_dimension` | `int` | Total compiled APW plus local-orbital dimension. |
| `plane_wave_count` | `int` | Number of plane-wave rows. |
| `plane_wave_g` | `int32[n_g, 3]` | Integer reciprocal labels. |
| `plane_wave_k_cartesian` | `float64[n_g, 3]` | Cartesian $k$ in inverse Bohr, repeated for every row. |
| `plane_wave_k_plus_g` | `float64[n_g, 3]` | Cartesian $k+G$ in inverse Bohr. |
| `apw_labels` | `int64[n_apw, 4]` | Rows `(site, g, l, m)` for the flattened APW coefficients. |
| `apw_coefficients` | `complex128[n_apw, 2]` | Coefficients multiplying `(u, udot)`; these include the Rayleigh factor and site phase. |
| `local_orbital_rows` | `int64[n_lo, 6]` | Rows `(global_row, site, l, m, ordinal, radial_n)`. |

`export_radials()` uses offsets so different sites and optional small
components remain ragged without object arrays:

| Key | Python representation | Meaning |
|---|---|---|
| `mesh_site` | `int64[n_site]` | Stable site index for each mesh. |
| `mesh_first` | `float64[n_site]` | First radius in Bohr. |
| `mesh_increment` | `float64[n_site]` | Exponential logarithmic increment. |
| `mesh_count` | `int64[n_site]` | Point count for each mesh. |
| `mesh_offsets` | `int64[n_site + 1]` | Offsets into `mesh_radii` and `mesh_weights`. |
| `mesh_radii` | `float64[sum(mesh_count)]` | Concatenated radii in Bohr. |
| `mesh_weights` | `float64[sum(mesh_count)]` | Concatenated radial quadrature weights. |
| `radial_labels` | `int64[n_fun, 5]` | Rows `(site, kind, l, n, spin)`; kind zero is valence and one is core. |
| `sample_offsets` | `int64[n_fun + 1]` | Offsets into `large`. |
| `large` | `float64[sum(sample_count)]` | Concatenated reduced large-component samples. |
| `small_present` | `bool[n_fun]` | Whether each radial has a small component. |
| `small_offsets` | `int64[n_fun + 1]` | Offsets into `small`; absent components have equal adjacent offsets. |
| `small` | `float64[sum(present_small_count)]` | Concatenated present reduced small-component samples. |

`export_geometry()` returns:

| Key | Python representation | Meaning |
|---|---|---|
| `site_id` | `list[str]` | Checkpoint site identifiers in stable site order. |
| `atomic_number` | `int64[n_site]` | Nuclear charge labels from the checkpoint. |
| `site_fractional` | `float64[n_site, 3]` | Stored fractional direct-lattice coordinates. |
| `site_cartesian` | `float64[n_site, 3]` | Runtime Cartesian site positions in Bohr. |
| `muffin_tin_radius` | `float64[n_site]` | Muffin-tin radii in Bohr. |
| `direct_lattice` | `float64[3, 3]` | Direct primitive vectors by row in Bohr. |
| `reciprocal_lattice` | `float64[3, 3]` | Reciprocal primitive vectors by row in inverse Bohr, including $2\pi$. |
| `cell_volume` | `float` | Unit-cell volume in cubic Bohr. |

`export_kq_map()` returns:

| Key | Python representation | Meaning |
|---|---|---|
| `k_index` | `int64[n_k]` | Source $k$ indices. |
| `kq_index` | `int64[n_k]` | Mapped $k-q$ indices. |
| `g_wrap_index` | `int32[n_k, 3]` | Integer reciprocal wraps for each mapped point. |
| `g_wrap_cartesian` | `float64[n_k, 3]` | Cartesian reciprocal wraps in inverse Bohr. |
| `transfer_cartesian` | `float64[3]` | Canonical transfer $q$ in inverse Bohr. |
| `global_transfer_index` | `int32[3]` | Global reciprocal transfer removed while canonicalizing the requested $q$. |

`export_pair_support()` returns:

| Key | Python representation | Meaning |
|---|---|---|
| `g_relative_index` | `int32[n_raw_g, 3]` | Integer relative reciprocal labels in canonical raw-support order. |
| `g_relative_cartesian` | `float64[n_raw_g, 3]` | Cartesian relative reciprocal vectors in inverse Bohr. |
| `g_relative_norm` | `float64[n_raw_g]` | Their Cartesian norms in inverse Bohr. |

`export_pair_layout()` returns:

| Key | Python representation | Meaning |
|---|---|---|
| `n_k` | `int` | Number of $k$ points. |
| `n_orb` | `int` | Common orbital count. |
| `n_columns` | `int` | Total pair-column count. |
| `core_orbital` | `int` or `None` | Optional sharp-core orbital index. |
| `pair_order` | `str` | Literal `k*n_orb^2 + i*n_orb + j`. |

## 5. Entry points

The surface is about fifteen functions and methods, scalar lane
first:

```python
import libmuffintin as mt

snap = mt.load_checkpoint("checkpoint.toml")            # Checkpoint V1/V2 via mt-io
phys = mt.CheckpointPhysics(snap)                     # CheckpointPhysics::new
inp  = phys.scalar_product_input("input.toml", q=[0.0, 0.0, 0.0])
inps = phys.scalar_q_slice("input.toml")            # complete k-mesh slice in
                                                    # production q-index order

inp.export_orbitals()      # k_fractional, per spin: energies, eigenvectors,
                           # available_bands, band_window
inp.export_basis(k, spin)  # plane-wave G labels, k+G, APW matching, LO rows
inp.export_radials()       # ProductRadial tables plus meshes
inp.export_geometry()      # sites, muffin-tin radii, direct/reciprocal lattice
inp.export_kq_map()        # kq_index and G_wrap for each k
inp.export_pair_support()  # raw interstitial pair G set
inp.export_pair_layout()   # n_k, n_orb, core_orbital, n_columns

mpb = mt.build_scalar_mpb(inp, selections, ...)     # spec fields as kwargs
mpb.export_auxiliary()     # MT modes, auxiliary |q+G| waves, CutoffRecord
mpb.export_vertices()      # (n_sel, n_aux) complex plus the region table

thc = mt.build_scalar_thc(inps, grid, spin=0, rank=..., engine="qrcp",
                          candidates=...)
vq  = mt.build_scalar_coulomb(inps, thc, request=..., projection=...)
vq.export_matrix(iq)       # (n_aux, n_aux) complex plus GammaHead metadata
vq.export_diagnostics()    # matched-pair quadratic discrepancies

mt.sample_scalar_orbitals(inp, points, spin=0)      # (P, Q) values at points

mt.write_scalar_mldump(...)
mt.write_scalar_coqui_cholesky(...)
```

`sample_scalar_orbitals` is the orbital-evaluator seam for Python-side
selectors and local-RI fits. It is the only planned change visible to
the existing kernels: the orbital evaluation currently private to the
scalar THC bridge is promoted to a public runtime function. Everything
else reuses the frozen runtime boundaries as they stand.

Stage 1 adds one narrow runtime surface, `single_dft_scf_config`: it
converts the only `dft-scf` task in a prepared Input V2 workflow to the
existing internal `ScfConfig`, returning a typed error when the count is
not exactly one. The binding calls this function and does not duplicate
runtime configuration mapping.

### 5.1 Deferred two-level DFT-SCF binding

The DFT-SCF binding follows the product, THC, and Coulomb lane rather than
interrupting Stages 1–3. It has two levels:

```python
result = mt.run_dft_scf("input.toml")
session = phys.scf_session(config)
```

`run_dft_scf` is the global production entry. It returns an opaque converged
checkpoint handle together with energy terms and convergence history. The
implementation follows the same `dft-scf` runner path as the `muffintin`
binary; the executable remains a thin wrapper.

`scf_session` is the staged entry. Its opaque `ScfSession` methods correspond
one-for-one to the eight-step chain in [17](17_minimal_dft_scf.md): initial
density, `build_scf_potential`, basis materialization and solve, occupations,
density assembly, mixing, energy terms, and convergence decision. This level
does not invent binding-specific physics APIs. Before implementation, each
step is checked against the existing public `libmuffintin-dft` and
`MaterialKernel` seams; any necessary promotion of a private helper is
recorded here individually.

The staged inventory distinguishes method-neutral algorithms from the first
LAPW solver implementation:

| Doc 17 step | Method-neutral stage contract | LAPW solver-specific contract |
|---|---|---|
| Initial density | Regional density initialization. | – |
| Build SCF potential | Density to regional potential and potential-energy terms. | – |
| Materialize basis and solve | – | LAPW radial/basis materialization, `CompiledBasis`, and eigensolve. |
| Occupations | Eigenvalue sets to chemical potential, occupations, and occupation correction. | – |
| Assemble density | – | LAPW eigenvectors and compiled projections to a regional density. |
| Mix density | Regional input/output densities to the next density and physical residual. | – |
| Evaluate energy terms | Regional fields, eigenvalue sums, and occupation correction to neutral energy terms. | – |
| Decide convergence | Energy history and physical density residual to a convergence decision. | – |

A method-neutral Python signature may accept and return only neutral types,
such as regional densities, regional potentials, eigenvalue sets, energy
terms, and convergence records. It must not expose `CompiledBasis`, LAPW
matching coefficients, or another solver-owned type. This is the acceptance
criterion for reusing the same stage unchanged from a later LMTO or FP-KKR
route.

Neutral stages may be independent opaque handles rather than methods owned by
`ScfSession`. For example, `DensityMixer.broyden2(alpha, history)` constructs
a reusable neutral mixer and `mixer.step(...)` advances it. `ScfSession` owns
only the loop ordering and connects these reusable neutral stations to the
solver-specific cells. Thus the global layer runs one LAPW `dft-scf` task,
while the staged layer is a method-neutral muffin-tin SCF toolbox whose first
solver consumer happens to be LAPW.

The global entry must be implemented on the staged layer so the two levels
have one source of truth. Within Stage 4, the global production entry lands
first; the staged API follows only after the doc 17 chain has been checked
step by step.

## 6. Stages

| Stage | Content |
|---|---|
| 1 | Scaffolding (`mt-python`, `python/`, maturin), checkpoint loading, `scalar_product_input`, `single_dft_scf_config`, and the seven `export_*` methods on the product-input handle. |
| 2 | `build_scalar_mpb` / `build_scalar_thc` / `build_scalar_coulomb` handles with exports, the parent-grid array input, and `sample_scalar_orbitals`. |
| 3 | Spinor twins of stages 1 and 2, plus the MLDUMP and CoQui Cholesky writer pass-throughs. |
| 4 | Two-level DFT-SCF binding: global `run_dft_scf` first, then the doc 17 staged `ScfSession`; the global entry is implemented on the staged layer. |
| 5 | Bootstrap of the separate `pymuffintin` package: provider protocols, the muffintin backend adapter over pyexport v1, `auxiliary/{lri,thc,hybrid}.py`, `mbpt/hf.py`, and gate 3. Gates 1 and 2 stay in this repository as binding tests. |

## 7. Acceptance gates

1. **Quadratic-form consistency.** Python recomputes
   $c^\dagger V c$ from exported vertices and $V^q$ and agrees with
   the Rust `ScalarCoulombPairMatch` within
   `SCALAR_COULOMB_EXACTNESS_FLOOR` on the hydrogen fixture.
2. **Same-engine THC reproduction.** Python QRCP on the exported pair
   blocks, with the same candidates and rank, reproduces the Rust
   `allq_l2` selection on the hydrogen fixture. This is same-algorithm
   determinism; it is not the forbidden MPB/THC or QRCP/Cholesky
   pivot comparison of [20](20_sm_dy_full_spinor_material_demo.md).
3. **Fixed-orbital exchange ablation.** `pymuffintin`, through its muffintin
   adapter, runs the MPB reference against a muffin-tin local-RI plus
   interstitial-THC composition on the small fixture and reports the
   $E_x$ and $\Sigma_x$ matrix-element differences. The number is a
   pipeline check on the fixture, not a material accuracy claim.

Gates 1 and 2 use focused tests in `mt-python` and `python/`; gate 3
is a `pymuffintin` test against the same fixture. The binding does not join
the routine workspace test scope.
