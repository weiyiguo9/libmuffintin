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
  src/{lib,checkpoint,products,thc,coulomb,spinor,writers,scf,export}.rs
python/
  pyproject.toml             # maturin backend, module-name = "libmuffintin._native"
  libmuffintin/
    __init__.py              # binding only: re-exports _native plus thin loaders
```

The `pymuffintin` research package is a separate repository and is not part
of this layout. Its planned shape, for reference:

```text
pymuffintin/
  pyproject.toml
  src/pymuffintin/
    backends/muffintin.py    # the only place importing libmuffintin
    contracts.py             # backend-neutral array DTOs
    providers.py             # orbitals/local-product/Coulomb protocols
    auxiliary/{lri,thc,hybrid}.py
    mbpt/hf.py
  tests/{test_import,test_auxiliary,test_gate3}.py
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

### 4.2 Stage 2 scalar result dictionaries

`sample_scalar_orbitals(input, coordinates, weights, regions, spin)` returns
the sampled large and small orbital components. It does not return pair
products; consumers form them with `export_kq_map()` and the pair order above.

| Key | Python representation | Meaning |
|---|---|---|
| `large` | `complex128[n_point, n_k, n_orb]` | Sampled scalar large components. |
| `small` | `complex128[n_point, n_k, n_orb]` | Sampled scalar small components. |

The following transfer fields recur in MPB, THC-record, and Coulomb exports:

| Key | Python representation | Meaning |
|---|---|---|
| `q_cartesian` | `float64[3]` | Canonical Cartesian transfer in inverse Bohr. |
| `q_umklapp_index` | `int32[3]` | Global reciprocal transfer removed by canonicalization. |
| `q_umklapp_cartesian` | `float64[3]` | Cartesian form of `q_umklapp_index` in inverse Bohr. |

`ScalarMpbResult.export_auxiliary()` returns:

| Key | Python representation | Meaning |
|---|---|---|
| transfer fields | as above | Transfer identity of the MPB. |
| `dimension` | `int` | Total retained auxiliary dimension. |
| `mt_dimension` | `int` | Muffin-tin prefix length. |
| `interstitial_dimension` | `int` | Interstitial suffix length. |
| `regions` | `int64[n_aux, 5]` | Rows `(kind, a, b, c, d)`: kind zero is `(site,l,m,n)`, kind one is `(Gx,Gy,Gz,-1)`, and kind two is interpolation metadata. |
| `mt_mesh_site` | `int64[n_site_block]` | Site index for each retained MT block. |
| `mt_mesh_first` | `float64[n_site_block]` | First radial point in Bohr. |
| `mt_mesh_increment` | `float64[n_site_block]` | Exponential-mesh logarithmic increment. |
| `mt_mesh_count` | `int64[n_site_block]` | Radial point count per block. |
| `mt_mesh_offsets` | `int64[n_site_block + 1]` | Offsets into MT radii and weights. |
| `mt_mesh_radii` | `float64[sum(mt_mesh_count)]` | Concatenated MT radii. |
| `mt_mesh_weights` | `float64[sum(mt_mesh_count)]` | Concatenated MT radial weights. |
| `mt_mode_labels` | `int64[n_mode, 3]` | Rows `(site,l,n)`. |
| `mt_mode_offsets` | `int64[n_mode + 1]` | Offsets into `mt_mode_radial`. |
| `mt_mode_radial` | `float64[sum(mode_count)]` | Retained radial-mode samples. |
| `interstitial_g_index` | `int32[n_i, 3]` | Retained interstitial reciprocal labels. |
| `interstitial_g_cartesian` | `float64[n_i, 3]` | Cartesian reciprocal vectors. |
| `interstitial_q_plus_g` | `float64[n_i, 3]` | Cartesian $q+G$ vectors. |
| `interstitial_q_plus_g_norm` | `float64[n_i]` | Norms of $q+G$. |
| `cutoff_kind` | `str` or `None` | `spectral-overlap` when a retained cutoff exists. |
| `cutoff_value` | `float` or `None` | Stored overlap cutoff. |
| `cutoff_nspin_factor` | `float` or `None` | Stored spin factor applied to the cutoff. |

`ScalarMpbResult.export_vertices()` returns:

| Key | Python representation | Meaning |
|---|---|---|
| `regions` | `int64[n_aux, 5]` | Exact auxiliary row order. |
| `labels` | `int64[n_selection, 5]` | Rows `(spin,k,left_band,right_band,column)`. |
| `coefficients` | `complex128[n_selection, n_aux]` | Pair vertices in the exported auxiliary order. |

`ScalarThcResult.export_selection()` returns:

| Key | Python representation | Meaning |
|---|---|---|
| `spin` | `int` | Selected scalar spin. |
| `requested_rank` | `int` | Requested maximum or exact rank. |
| `effective_rank` | `int` | Retained rank. |
| `point_ids` | `int64[n_mu]` | Selected parent-grid identifiers. |
| `point_regions` | `int64[n_mu, 3]` | Rows `(kind,site,radial_index)`; interstitial rows are `(1,-1,-1)`. |
| `pivots` | `int64[n_mu]` | Selector pivots in parent-grid indexing. |
| `diagonal` | `float64[n_step]` | QRCP or pivoted-Cholesky selection diagonal. |
| `candidates` | `int64[n_candidate]` | Candidate parent-grid identifiers. |
| `strategy`, `engine`, `q_set`, `weights` | `str` | Frozen selector provenance. |
| `seed` | `int` or `None` | Selector seed when the engine uses one. |
| `n_points`, `n_candidates` | `int` | Parent-grid and candidate counts. |

`ScalarThcResult.export_records()` returns the parent `coordinates`
(`float64[n_point,3]`), `weights` (`float64[n_point]`), `regions`
(`int64[n_point,3]`), and a `records` list. Each record contains:

| Key | Python representation | Meaning |
|---|---|---|
| transfer fields | as above | Transfer identity for this record. |
| `q_index`, `rank`, `n_points`, `n_mu` | `int` | Record index and dimensions. |
| `zeta` | `complex128[n_point, n_mu]` | Fitted interpolation functions on the parent grid. |
| `l2_all` | `float64[2]` | Frobenius and maximum-column residual. |
| `l2_core`, `l2_valence`, `coulomb_residual` | `float64[2]` or `None` | Optional split residuals in the same order. |
| `vertices` | `complex128[n_column, n_mu]` | Pair vertices at selected interpolation points. |
| `vertex_labels` | `int64[n_column, 4]` | Rows `(k,left,right,column)`. |
| `pair_samples` | `complex128[n_point, n_column]` | Full parent-grid pair block used by the fit. |
| `point_ids` | `int64[n_mu]` | Selected parent-grid identifiers. |
| `point_regions` | `int64[n_mu, 3]` | Selected region tags. |

Both `ScalarCoulombResult.export_matrix(q_index)` and
`ScalarMpbCoulombResult.export_matrix()` return:

| Key | Python representation | Meaning |
|---|---|---|
| `q_index` | `int` or `None` | Full-slice index; an isolated MPB operator has no slice index. |
| transfer fields | as above | Transfer identity. |
| `spin` | `int` or `None` | Scalar THC spin; an MPB operator is spin-independent. |
| `dimension`, `mt_dimension`, `interstitial_dimension` | `int` | Operator dimensions. |
| `matrix` | `complex128[n_aux, n_aux]` | Hermitian Coulomb operator. |
| `regions` | `int64[n_aux, 5]` | Exact operator auxiliary order. |
| `gamma_present` | `bool` | Whether Gamma-head metadata is present. |
| `gamma_spherical_average_subtracted` | `bool` or `None` | Gamma convention flag. |
| `gamma_head_prefactor` | `float` or `None` | Analytic head prefactor. |
| `gamma_constant_coefficients` | `complex128[n_aux]` | Constant-mode coefficients, empty away from Gamma. |

`ScalarCoulombResult.export_diagnostics()` returns one-dimensional arrays
`q_index`, `spin`, `k_index`, `left_band`, `right_band`, `column`
(`int64[n_match]`) and `mpb_quadratic`, `thc_quadratic`,
`mpb_action_norm`, `thc_action_norm`, `quadratic_absolute`,
`quadratic_relative` (`float64[n_match]`).

### 4.3 Stage 3 spinor and writer dictionaries

The spinor product handle uses the same geometry, $k-q$ map, pair-support,
and pair-layout dictionaries as the scalar handle. Its differing dictionaries
are complete below:

| Export | Keys and Python representation |
|---|---|
| `export_orbitals()` | `k_fractional: float64[n_k,3]`, `band_window_start: int`, `band_window_count: int`, `energies: float64[n_k,n_orb]`, `eigenvectors: list[complex128[n_basis(k),n_orb]]`, `available_bands: int64[n_k]`. |
| `export_basis(k)` | `k_index: int`, `basis_dimension: int`, `spatial_plane_wave_count: int`, `plane_wave_g: int32[n_g,3]`, `plane_wave_k_cartesian: float64[n_g,3]`, `plane_wave_k_plus_g: float64[n_g,3]`, `pauli_rows: int64[2*n_g,3]`, `local_orbital_rows: int64[n_lo,6]`, `projection_rows: int64[n_projection,5]`, `matching_labels: int64[n_matching,5]`, `matching_coefficients: complex128[n_matching,2]`. |
| `export_radials()` | `mesh_site: int64[n_site]`, `mesh_first: float64[n_site]`, `mesh_increment: float64[n_site]`, `mesh_count: int64[n_site]`, `mesh_offsets: int64[n_site+1]`, `mesh_radii`, `mesh_weights: float64[n_sample]`, `radial_labels: int64[n_fun,4]` as `(site,kind,kappa,n)`, `sample_offsets: int64[n_fun+1]`, `p`, `q: float64[n_sample]`. |

Spinor MPB auxiliary exports have the same keys, shapes, dtypes, and region
order as scalar MPB auxiliary exports. `SpinorMpbResult.export_vertices()`
uses `labels: int64[n_selection,4]` with rows `(k,left_band,right_band,column)`
and `coefficients: complex128[n_selection,n_aux]`. Spinor THC selection omits
the scalar `spin` and `diagonal` keys; its remaining keys match the scalar
selection table. Spinor THC records match the scalar record table except that
they do not export `pair_samples`. Spinor Coulomb matrix exports use the
common Coulomb matrix table above. Spinor diagnostics return
`pairs: int64[n_match,3]`, `columns: int64[n_match]`, and the float arrays
`mpb_quadratic`, `thc_quadratic`, `mpb_action_norm`, `thc_action_norm`,
`absolute`, and `relative`.

The writer pass-throughs accept only frozen handles produced by the same
context:

| Entry point | Required arguments | Output |
|---|---|---|
| `write_scalar_mldump` | `path, slice, thc, coulomb, producer_name, producer_version, source_revision, site_species, site_labels` | MLDUMP v1 HDF5. |
| `write_spinor_mldump` | Same metadata with spinor handles. | Spinor MLDUMP v1 HDF5. |
| `write_scalar_coqui_cholesky` | `path, slice, thc, coulomb, tolerance` | CoQui Cholesky HDF5. |

### 4.4 Regional fields and frozen scalar radial sampling

`CheckpointPhysics.export_frozen_potential()`, an available
`CheckpointPhysics.export_restart_density()`, and the staged SCF
`export_interstitial()` methods return the same additive regional dictionary.
The original interstitial keys remain unchanged; the muffin-tin keys complete
the representation:

| Key | Python representation | Meaning |
|---|---|---|
| `angular_basis` | `str` | `complex-condon-shortley` or `real-tesseral-condon-shortley`. |
| `g_vectors` | `int32[n_g,3]` | Integer reciprocal labels for the interstitial block. |
| `components` | `complex128[4,n_g]` | Interstitial charge/scalar, $x$, $y$, $z$ components. |
| `mt_mesh_site` | `int64[n_site]` | Stable site index for each muffin-tin mesh. |
| `mt_mesh_first` | `float64[n_site]` | First radial point in Bohr. |
| `mt_mesh_increment` | `float64[n_site]` | Exponential-mesh logarithmic increment. |
| `mt_mesh_count` | `int64[n_site]` | Radial point count per site. |
| `mt_mesh_offsets` | `int64[n_site+1]` | Offsets into `mt_mesh_radii` and `mt_mesh_weights`. |
| `mt_mesh_radii` | `float64[sum(mt_mesh_count)]` | Concatenated radii in Bohr. |
| `mt_mesh_weights` | `float64[sum(mt_mesh_count)]` | Concatenated radial quadrature weights. |
| `mt_channel_labels` | `int64[n_channel,3]` | Rows `(site,l,m)` in site-major angular-channel order. |
| `mt_sample_offsets` | `int64[n_channel+1]` | Offsets for the channel-major radial samples. |
| `mt_components` | `complex128[4,n_mt_sample]` | Flattened muffin-tin components in the same Pauli order. |

For a frozen-potential checkpoint, `export_restart_density()` returns `None`.
It returns the same regional dictionary only when the loaded checkpoint
contains an actual restart density.

`CheckpointPhysics.sample_frozen_scalar_radials(site_id, l, energies,
hard_radius=None)` returns:

| Key | Python representation | Meaning |
|---|---|---|
| `site_index`, `site_id`, `l` | `int`, `str`, `int` | Selected checkpoint site and angular momentum. |
| `energies` | `float64[n_E]` | Requested energies in Hartree. |
| `mesh_first`, `mesh_increment`, `mesh_count` | `float`, `float`, `int` | Native exponential-mesh parameters. |
| `mesh_radii` | `float64[n_r]` | Native muffin-tin radii in Bohr. |
| `radial_samples` | `float64[n_E,n_r]` | Normalized $u(r)$ from `RadialSolver`. |
| `boundary_radius` | `float` | Native muffin-tin radius in Bohr. |
| `boundary_radial` | `float64[n_E,2]` | Exact rows $[u(R),u'(R)]$. |
| `log_derivative` | `list[float or None]` | $u'(R)/u(R)$ in inverse Bohr. |
| `energy_derivative_boundary_radial` | `float64[n_E,2]` | Exact rows $[\partial_Eu(R),\partial_Eu'(R)]$ from `solve_with_energy_derivative`; this is the trace required by $\dot K$. |

This sampler is deliberately limited to a nonmagnetic scalar checkpoint
potential and uses the physical spherical potential
$V(r)=v_{00}(r)/\sqrt{4\pi}$ in Hartree. `hard_radius` may be omitted or must
equal the native muffin-tin radius exactly. A different hard sphere would
require a proper-potential solve and phase-shifted continuation outside the
muffin tin; the binding does not mislabel such a continuation as the native
solution.

## 5. Entry points

The surface is about fifteen functions and methods, scalar lane
first:

```python
import libmuffintin as mt

snap = mt.load_checkpoint("checkpoint.toml")            # Checkpoint V1/V2 via mt-io
phys = mt.CheckpointPhysics(snap)                     # CheckpointPhysics::new
v_mt = phys.export_frozen_potential()                 # full regional Pauli field
rho_mt = phys.export_restart_density()                # None for frozen-potential input
rad = phys.sample_frozen_scalar_radials("H-1", 0, [-0.3])
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
mpb_v = mt.build_scalar_mpb_coulomb(mpb, lexp=...)
mpb_v.export_matrix()      # full mixed-product Coulomb operator

thc = mt.build_scalar_thc(inps, coordinates, weights, regions,
                          spin=0, rank=..., engine="qrcp", candidates=...)
vq  = mt.build_scalar_coulomb(inps, thc, lexp=...,
                              interpolation_pw_cutoff=...,
                              interpolation_l_max=...)
vq.export_matrix(iq)       # (n_aux, n_aux) complex plus GammaHead metadata
vq.export_diagnostics()    # matched-pair quadratic discrepancies

mt.sample_scalar_orbitals(inp, coordinates, weights, regions, spin=0)
                                                    # large/small (P,n_k,n_orb)

mt.write_scalar_mldump(...)
mt.write_scalar_coqui_cholesky(...)
```

`sample_scalar_orbitals` is the orbital-evaluator seam for Python-side
selectors and local-RI fits. It is the only planned change visible to
the existing kernels: the orbital evaluation currently private to the
scalar THC bridge is promoted to a public runtime function. Everything
else reuses the frozen runtime boundaries as they stand.

Stage 2 also exposes `build_scalar_mpb_coulomb(mpb, lexp)`. This is a
binding-only opaque handle over the already-public `assemble_coulomb` kernel;
it is the reference metric used by the Stage 5 pair-space projection and is
not a new runtime representation or Coulomb algorithm.

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
plan = mt.prepare_dft_scf("input.toml")
session = phys.scf_session("input.toml")
```

`run_dft_scf` is the global production entry. It returns an opaque converged
checkpoint handle together with energy terms and convergence history. The
implementation follows the same `dft-scf` runner path as the `muffintin`
binary; the executable remains a thin wrapper.

`prepare_dft_scf` and both global/session entries reuse
`single_dft_scf_config`: the workflow must contain exactly one `dft-scf`
task, while unrelated task entries are not silently selected as SCF tasks.
`plan.session()` constructs an independent session from the input checkpoint;
`phys.scf_session(path)` additionally requires that the plan checkpoint equal
the opaque physics checkpoint context.

`scf_session` is the staged entry. Its opaque `ScfSession` methods correspond
one-for-one to the production order in [17](17_minimal_dft_scf.md): initial
density; potential and core; channel and LAPW operator materialization;
bands and occupations; spectral refinement; density synthesis; energy,
residual, and convergence; then mixing only for a continuing iteration. This
level does not invent binding-specific physics APIs. Before implementation,
each step is checked against the existing public `libmuffintin-dft` and
`MaterialKernel` seams; any necessary promotion of a private helper is
recorded here individually.

The staged inventory distinguishes method-neutral algorithms from the first
LAPW solver implementation:

| Production step | Method-neutral stage contract | LAPW solver-specific contract |
|---|---|---|
| Initial density | The returned regional density is neutral. | A checkpoint cold start may use LAPW-owned initialization internally and is not advertised as a neutral algorithm. |
| Potential and core | Regional density to regional potential, potential-energy terms, and site-labelled four-component core contributions. | – |
| Materialize channels and operator | – | Current-potential channels, LAPW basis, and $H/S$ assembly. |
| Bands and occupations | Eigenvalue sets to chemical potential, occupations, and occupation correction. | LAPW eigensolve produces the eigenvalue set and solver-owned eigenvectors. |
| Spectral refinement | – | Projected-spectrum channel refinement and any repeated LAPW solve at the same outer potential. |
| Assemble density | Core contributions combine through neutral regional-density algebra. | LAPW eigenvectors and compiled projections produce the valence regional density. |
| Energy and decision | Regional fields, eigenvalue sums, occupation correction, physical residual, energy history, and convergence decision. | – |
| Continue by mixing | Regional input/output densities to the next density; a converged iteration never calls the mixer. | – |

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

The concrete Python transition chain is:

```text
RegionalDensity
  → RegionalPotentialStep
  → CoreStep
  → LapwSolution
  → Occupations
  → LapwDensityAssembly
  → EnergyRecord
  → ConvergenceDecision
  ├─ converged → ScfResult
  └─ continue  → mix → RegionalDensity
```

Transition handles are linear: passing one to the next stage consumes it.
Using a consumed handle, mixing a converged decision, or passing a handle from
another session is an explicit error. `ScfResult` exposes `converged`,
`iterations`, `total_energy`, `energy_history(): float64[n_iteration]`,
`convergence_history(): float64[n_iteration,2]` ordered as density RMS then
absolute energy change, and `restart_checkpoint()`.

Method-neutral `RegionalDensity`, `RegionalPotentialStep`, and
`LapwDensityAssembly` exports return pyexport v1 dictionaries with
`g_vectors: int32[n_g,3]` and
`components: complex128[4,n_g]` in charge/scalar, $x$, $y$, $z$ order. The
same additive dictionary now includes the complete flattened muffin-tin
field described in section 4.4; it still does not accept a `CompiledBasis`.
`Occupations.values()` returns
`float64[n_state]`; the remaining staged observations are typed scalar
properties on their opaque handles.

## 6. Stages

| Stage | Content |
|---|---|
| 1 | Scaffolding (`mt-python`, `python/`, maturin), checkpoint loading, `scalar_product_input`, `single_dft_scf_config`, and the seven `export_*` methods on the product-input handle. |
| 2 | `build_scalar_mpb` / `build_scalar_thc` / `build_scalar_coulomb` handles with exports, the narrow MPB Coulomb reference handle, the parent-grid array input, and `sample_scalar_orbitals`. |
| 3 | Spinor twins of stages 1 and 2, plus the MLDUMP and CoQui Cholesky writer pass-throughs. |
| 4 | Two-level DFT-SCF binding: global `run_dft_scf` first, then the doc 17 staged `ScfSession`; the global entry is implemented on the staged layer. |
| 5 | Bootstrap of the separate `pymuffintin` package: provider protocols, the muffintin backend adapter over pyexport v1, `auxiliary/{lri,thc,hybrid}.py`, `mbpt/hf.py`, and gate 3. Gates 1 and 2 stay in this repository as binding tests. |

Stage 5 keeps the native boundary explicit. `MuffintinAdapter` reconstructs
pair samples from `sample_scalar_orbitals` plus the exported $k-q$ wraps,
builds an all-pair MPB reference, and obtains its full Coulomb operator from
`build_scalar_mpb_coulomb`. A Python auxiliary representation stores pair
vertices $C_{\mathrm{trial}}$, not forged Rust auxiliary functions. The
adapter forms the reference pair kernel and its Moore–Penrose projection:

```math
K_{\mathrm{MPB}} = C_{\mathrm{MPB}}^* V_{\mathrm{MPB}} C_{\mathrm{MPB}}^{\mathsf T},
\qquad
V_{\mathrm{trial}} = (C_{\mathrm{trial}}^*)^+
K_{\mathrm{MPB}}
(C_{\mathrm{trial}}^{\mathsf T})^+.
```

For a concatenated muffin-tin LRI plus interstitial THC vertex matrix this
produces one full Hermitian projected matrix, including the MT–interstitial
cross block. It is a pair-space ablation against the MPB reference, not a
claim that the Python DTO is a native `CompiledAuxiliaryBasis` or that Rust
assembled a new hybrid representation.

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
