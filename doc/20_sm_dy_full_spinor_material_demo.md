# 20. Sm/Dy full-spinor material-acceptance demo

This note is the M-L7 material-acceptance record. It is not a production
release, not a `v0.1` tag, and not a SPEX or CoQui spinor consumer. The
implemented runtime path stays in
[18](18_lapw_mpb_thc_integration.md) and
[19](19_versioned_mldump_interchange.md). This document only states what a
real-material demo did, and did not, consume.

## 1. Claim classes

Keep these four classes separate.

| Class | Meaning |
|---|---|
| Implemented path | Public M-L5b/M-L5c/M-L5d/M-L6c2 APIs on HEAD. Hydrogen and other fixtures already exercise them. |
| Material demo evidence | A named producer snapshot, hashed inputs, materialized $Z=62$ or $Z=66$ basis, one bounded grid/rank, representation-neutral $c^\dagger V c$, MLDUMP roundtrip. |
| Cross-code limitation | SPEX `exchange_hdf` and CoQui scalar Cholesky do not ingest spinor full-first-variation MLDUMP. |
| Missing consumer | No spinor SPEX or CoQui full-4c reader is implemented here. |

A passing hydrogen spinor test is implemented-path evidence. It is not Sm or
Dy material evidence.

## 2. Implemented path (not a material result)

On HEAD `89ff8f8c80711eb6ded36efba688c8a7fd640bf9` the runtime already
exposes:

- `SnapshotDftPhysics::spinor_product_input` with
  `ScfRelativity::SpinorFirstVariation`, physical $P$ and $Q$, PP+QQ (no
  $PQ$/$QP$/$cQ$), and a full $q$ slice;
- `build_spinor_mpb` and `build_spinor_thc` on the same parent grid and
  candidate set, with an explicit `ThcEngine::FullColumnPivotedQr` or
  `FullPivotedCholesky` choice;
- `build_spinor_coulomb` comparing matched pairs by $c^\dagger V c$ only,
  using `SPINOR_COULOMB_EXACTNESS_FLOOR` rather than MPB `TOL` or a THC
  residual threshold;
- `write_spinor_mldump` roundtrip of that frozen path.

Those APIs must not be compared by elementwise $\zeta$ or pivot identity.
`TOL` is an overlap cutoff on a mixed-product auxiliary. A THC rank or
residual is a different number. Equating them is a documentation error.

## 3. Producer schema `spex.snapshot_hdf` v1

A separate HDF5 producer schema, not MLDUMP. SPEX does not own signed
$\kappa$. `read_spex_snapshot_hdf` returns SPEX-owned frozen fields.
`materialize_snapshot_v2` builds `SnapshotV2` only with an explicit
caller-owned recipe whose $l$ and energy match the SPEX scalar LO table.
Both SPEX hashes and `recipe_sha256` are recorded. `@external_basis_required=true`.
SPEX orbitals are $(l,E)$ plus optional $n$ when `pbas>0`; `kappa` is
forbidden in the HDF file. Scalar string attributes are Hwrapper
`H5T_NATIVE_CHARACTER` of content length (trailing spaces/NULs trimmed);
dataset `@axes` is that scalar split on whitespace, or a 1-d VL token
array. Collinear $B_x$/$B_y$ may be all-zero only
with `@zero_source` proving the spin layout; omitted $B$ is a blocker.

The SPEX writer is published at
`a073e0d5ece98e998307d78fe12ede7af421c770`
(`origin/codex/thc-dump-gfort15`, subject `feat(io): export frozen SPEX snapshots`).
Source publication is no longer blocked.

A later published producer `3b6e26f6f1cb1cc936b5916c6cee5b9bf19f1896`
ran one honest Sm fcc $3\times 3$ DFT0 job on WSL. Native-character
attributes passed; the run then died at `snapshot_hdf.f:0752` writing
VL `hashes/roles`:

> HDF5 1.14.6 Parallel IO does not support writing VL or region
> reference datatypes yet

`snapshot.h5` is $5104$ bytes, truncated, `h5dump` cannot open it,
SHA-256 `5efb1edc34fb8a11cec195b0397ece05977b15a2f9064a6c586f69db5d152d59`.
This lane did not consume that truncated file.

A later published producer `b45d9b9e1505d25236c3e78674418b011a471666`
wrote a complete `snapshot.h5` (1 548 720 B, SHA-256
`9f060f742e9078ec3dc8ee24d8945d38ec74a729e5dee85acfbffd345e132a59`).
The frozen reader loaded it. `hashes/sha256` has `STRSIZE` 248 with a
valid 64-hex prefix and Fortran control padding; the reader keeps the
leading 64 hex characters. `materialize_snapshot_v2` matched the SPEX
scalar LOs ($l=0,1$, $n=5$) with a caller-owned signed-$\kappa$ recipe.
`materialize_snapshot_v2` now ingests ULP-scale interstitial pairs with
tolerance $10^{-12}\max(|c|,|c'|,1)$ and writes conjugate-symmetric
averages so Snapshot V2 is exactly Hermitian. `SnapshotDftPhysics::new`
then succeeds as a raw frozen-field kernel, but still rejects
`SpinorFirstVariation` because SPEX `radial_equation` is honestly
`scalar-koelling-harmon`. The typed material route is
`SnapshotDftPhysics::new_spex_material`: it retains that source tag, binds
the caller-owned $(n,l,\kappa)$, treatment, derivative order, and energy to
the exact runtime `ScfBasis` request and resolved energy, then solves target
Dirac $P,Q$ from the imported $V_0$ monopole. A recipe/runtime mismatch is
`SnapshotDftError::SpexMaterialChannelMismatch`; no SPEX $P,Q$ or signed
$\kappa$ is claimed.

## 4. Dy bcc lane: blocked

The Dy writer searched, read-only, for an honest libmuffintin-consumable
bcc Dy DFT snapshot.

Located sources:

- Local `scratch/grid_budget_sm_dy.py`, SHA-256
  `9e19f3ff4ebfd4a506a2966e384d9fbacd2a4564d0baa768497649a7f8f64dab`.
  This is an order-of-magnitude MT-adaptive versus FFT budget for Sm and
  Dy fcc/bcc cells. It is not a potential.
- Local copies of the WSL experiment notes under
  `scratch/thc_smdy_experiment/` (RESULTS
  `222e9aa8dd2d98fb0ec874f918d3f4ebee4066ed169bb5aa7ec2dba42a3a7fab`).
  Those notes record fcc Sm only and state that Dy was not run.
- Live WSL tree `wsl:<thc-experiment-root>/runs`:
  `sm_fcc_3x3_dft`, `sm_fcc_3x3_ref`, `sm_fcc_4x4_dft`, `sm_fcc_4x4_ref`.
  No `dy_*` and no bcc DFT directory. The same tree has
  `codes/libmuffintin/scratch/grid_budget_sm_dy.py`.

Exact producer blocker:

1. No bcc Dy DFT run, `spex.pot`, FLEUR `cdn/pot`, or Snapshot V2 exists
   under the searched trees.
2. The frozen FLEUR converter is not an active producer.
3. `read_spex_snapshot_hdf` consumes only `spex.snapshot_hdf` v1, not
   `spex.pot`/`spex.dft`. No such hashed producer file has arrived.
   Existing Sm artifacts (`spex.inp`, `spex.pot`, `thc_orbitals*.bin`)
   remain SPEX-native, scalar-collinear, no SOC, fcc Sm.
4. Synthetic atomic data would not be Dy material and is forbidden here.

Catalogue-only facts that do not lift the blocker, checked in
`crates/mt-runtime/tests/ml7_dy_bcc.rs`:

- FLEUR `default.econfig` for $Z=66$ is
  `1s2 \ldots 4d10 | 5s2 5p6 6s2 4f10`.
- Full first variation assigns $5p_{1/2}$ ($\kappa=+1$) to relativistic
  local-orbital treatment for $Z=55$ to $86$.
- Both $4f$ $\kappa$ partners remain valence.
- Built-in compilation emits those signed $\kappa$ records. It does not
  invent HDLO: FLEUR `lo=`/`HDLO` lines are basis hints, not occupations.

A materialized spinor basis, including required $4f$, $5p_{1/2}$ rLO, and
any HDLO, can be asserted only from `ScfBasis::resolved_channels` after
`SnapshotDftPhysics` consumes a real snapshot. That step did not run.

Tracked Dy evidence: `crates/mt-runtime/tests/ml7_dy_evidence.toml`. New
artifact paths would have used the `ml7-dy` prefix only. No such path was
created.

No synthetic Dy-shaped viability fixture is retained in the material lane.
Native atomic $\to$ SCF $\to$ Snapshot V2 was not honestly converged.

## 5. Sm fcc lane: bounded material path executed

Shared harness `crates/mt-runtime/tests/ml7_material_common.rs` is
included by path. Observed contract:

- `Ml7Provenance`, `load_spinor_snapshot_v2` (missing file is
  `Ml7CommonError::MissingSnapshot`; scalar relativity is rejected
  before I/O);
- `ordered_q_slice` for a complete $q$ mesh;
- `compare_qrcp_cholesky` on one parent grid, candidate set, and rank,
  without elementwise $\zeta$ or pivot equality.

The focused `consume_b45d9b9_spex_snapshot_and_run_bounded_sm_lane` oracle
consumes the complete hashed SPEX artifact. It keeps both imported radial
bases tagged scalar Koelling–Harmon, uses one caller-owned signed-$\kappa$
recipe for the $5s$ LO and $5p_{1/2}$ rLO, and binds those records to the
target runtime basis. A deliberate derivative-order mismatch is typed-
rejected before authorization. The accepted binding produces finite,
nonzero target Dirac $P,Q$ traces. The bounded oracle caps the target
plane-wave cutoff at $0.5\,a_0^{-1}$ rather than relabelling SPEX's stored
cutoff, then executes the complete one-point $q$ slice through:

- bounded spinor MPB for one selected band pair;
- one shared six-point parent grid with exact rank one, using both full
  column-pivoted QR and full pivoted Cholesky;
- sampled spinor Coulomb on the QR-selected THC result;
- spinor MLDUMP write/read with source revision
  `b45d9b9e1505d25236c3e78674418b011a471666`.

This is bounded M-L7 Sm evidence, not convergence, production validation,
or a cross-code spinor-consumer result. Dy remains externally blocked.

## 6. Cross-code limitations

- SPEX `exchange_hdf` (scalar experiment dump) does not support
  full-first-variation four-component spinor MLDUMP.
- CoQui-native output on this HEAD is the scalar Cholesky adapter
  (`write_scalar_coqui_cholesky`). It is not MLDUMP and not a spinor
  consumer.
- Adaptive THC `TOL` versus SPEX `MBASIS LCUT` lessons from the Sm
  scalar experiment remain diagnostic notes. They are not this demo's
  acceptance numbers.

## 7. Remaining acceptance boundary

The bounded Sm lane now supplies the hashed producer, typed target binding,
Dirac $P,Q$, full one-point $q$ slice, bounded MPB/THC/Coulomb, and MLDUMP
roundtrip. It does not establish grid or rank convergence and has no SPEX or
CoQui spinor consumer. The Dy lane still needs:

1. An honest Snapshot V2 (hashed) with the physical muffin-tin potential
   from a named DFT producer.
2. `SpinorFirstVariation` materialization whose resolved channels include
   the required $4f$ and $5p_{1/2}$ rLO set from an explicit caller-owned
   recipe bound to that runtime basis.
3. Physical $P,Q$ and PP+QQ pair density on a full $q$ slice.
4. One bounded parent grid and candidate set, QRCP and pivoted Cholesky
   at the same rank, at most one rank increment, no sweep.
5. Bounded MPB/THC $c^\dagger V c$ pairs with the Coulomb exactness floor
   kept distinct from `TOL`.
6. MLDUMP roundtrip of those frozen objects.

Until those exist for Dy, M-L7 remains partially accepted: bounded Sm is
executed and Dy is blocked. Neither lane is a release claim.

## 8. Preservation

No commit, push, fetch, checkout, reset, clean, CI rerun, or stash
mutation was performed for this note. `stash@{0}` was not inspected.
`g.mod` was absent.
