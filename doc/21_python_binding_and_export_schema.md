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
| SCF binding (`run_scf`, mixing, occupations) | The frozen-checkpoint solve is the fixed-orbital entry; SCF stays behind the `muffintin` binary. |
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
   path or string to the existing runtime input parser. No Rust
   configuration struct is mirrored into Python classes.
2. **Results are opaque handles.** `ScalarProductInput`,
   `ScalarMpbResult`, `ScalarThcResult`, `ScalarCoulombResult`, and
   their spinor twins are wrapped, not converted. Handles chain
   through the existing Rust bridges (THC to Coulomb to MLDUMP)
   without a Python array round trip, so the non-forgeable identity
   and fingerprint checks on those types keep working. Explicit
   `export_*` methods return NumPy structures for the experiment
   layer.
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
inp.pair_layout()          # n_k, n_orb, core_orbital

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

## 6. Stages

| Stage | Content |
|---|---|
| 1 | Scaffolding (`mt-python`, `python/`, maturin), checkpoint loading, `scalar_product_input`, and the seven `export_*` methods on it. |
| 2 | `build_scalar_mpb` / `build_scalar_thc` / `build_scalar_coulomb` handles with exports, the parent-grid array input, and `sample_scalar_orbitals`. |
| 3 | Spinor twins of stages 1 and 2, plus the MLDUMP and CoQui Cholesky writer pass-throughs. |
| 4 | Bootstrap of the separate `pymuffintin` package: provider protocols, the muffintin backend adapter over pyexport v1, `auxiliary/{lri,thc,hybrid}.py`, `mbpt/hf.py`, and gate 3. Gates 1 and 2 stay in this repository as binding tests. |

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
