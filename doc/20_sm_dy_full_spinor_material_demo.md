# 20. Sm/Dy full-spinor material-acceptance demo

This note is the material-acceptance record and the home of the
`spex.snapshot_hdf` v1 producer schema. It is not a production release
and not a SPEX or CoQui spinor consumer. The implemented runtime path
stays in [18](18_lapw_mpb_thc_integration.md) and
[19](19_versioned_mldump_interchange.md); a passing hydrogen spinor
test is implemented-path evidence, not Sm or Dy material evidence.

## 1. Comparison conventions

MPB, THC, and Coulomb results must not be compared by elementwise
$\zeta$ or pivot identity. `TOL` is an overlap cutoff on a
mixed-product auxiliary; a THC rank or residual is a different number;
`SPINOR_COULOMB_EXACTNESS_FLOOR` is a third, distinct from both.
Matched pairs compare representation-neutral quadratic forms
$c^\dagger V c$ only. Equating any of these numbers is a documentation
error.

## 2. Producer schema `spex.snapshot_hdf` v1

A separate HDF5 producer schema, not MLDUMP. SPEX does not own signed
$\kappa$: `kappa` is forbidden in the HDF file, and SPEX orbitals are
$(l,E)$ plus optional $n$ when `pbas>0`. `read_spex_snapshot_hdf`
returns SPEX-owned frozen fields. Scalar string attributes are
Hwrapper `H5T_NATIVE_CHARACTER` of content length (trailing
spaces/NULs trimmed); the reader keeps the leading 64 hex characters
of a padded `hashes/sha256`. Dataset `@axes` is that scalar split on
whitespace, or a 1-d VL token array. Collinear $B_x/B_y$ may be
all-zero only with `@zero_source` proving the spin layout; omitted $B$
is a blocker. `@external_basis_required=true`.

`materialize_checkpoint_v2` builds `CheckpointV2` only with an explicit
caller-owned recipe whose $l$ and energy match the SPEX scalar LO
table; both SPEX hashes and `recipe_sha256` are recorded. It ingests
ULP-scale interstitial pairs with tolerance
$10^{-12}\max(|c|,|c'|,1)$ and writes conjugate-symmetric averages so
Checkpoint V2 is exactly Hermitian.

`CheckpointPhysics::new` on such an import still rejects
`SpinorFirstVariation`, because SPEX `radial_equation` is honestly
`scalar-koelling-harmon`. The typed material route is
`CheckpointPhysics::new_spex_material`: it retains that source tag,
binds the caller-owned $(n,l,\kappa)$, treatment, derivative order,
and energy to the exact runtime `ScfBasis` request and resolved
energy, then solves target Dirac $P,Q$ from the imported $V_0$
monopole. A recipe/runtime mismatch is
`CheckpointPhysicsError::SpexMaterialChannelMismatch`, and the route never
claims SPEX $P,Q$ or a SPEX signed $\kappa$.

## 3. Status

The bounded Sm fcc lane is executed. The
`consume_b45d9b9_spex_snapshot_and_run_bounded_sm_lane` oracle in
`crates/mt-runtime/tests/sm_fcc_material.rs` consumes the complete
hashed SPEX artifact (source revision
`b45d9b9e1505d25236c3e78674418b011a471666`), binds a caller-owned
signed $\kappa$ recipe for the $5s$ LO and $5p_{1/2}$ rLO, caps the
target plane-wave cutoff at $0.5\,a_0^{-1}$, and runs one complete
one-point $q$ slice through bounded spinor MPB, THC with both full
column-pivoted QR and full pivoted Cholesky at exact rank one, sampled
spinor Coulomb, and a spinor MLDUMP roundtrip. This is material
evidence for Sm within those bounds; it establishes neither
convergence nor production validation, and no cross-code spinor
consumer has read the output.

The Dy bcc lane is blocked: no honest Dy bcc DFT checkpoint, `spex.pot`,
FLEUR `cdn/pot`, or Checkpoint V2 exists, and synthetic atomic data
would not be Dy material and is forbidden here. Catalogue-only facts
(the FLEUR `default.econfig` for $Z=66$, the $5p_{1/2}$ rLO assignment
for $Z=55$ to $86$, both valence $4f$ $\kappa$ partners) are locked in
`crates/mt-runtime/tests/dy_bcc_material.rs` and do not lift the
blocker.

Cross-code limitations: SPEX `exchange_hdf` and the CoQui scalar
Cholesky adapter (`write_scalar_coqui_cholesky`) do not consume spinor
full-first-variation MLDUMP, and no spinor SPEX or CoQui full-4c
reader is implemented here.

## 4. Remaining Dy acceptance boundary

1. An honest hashed Checkpoint V2 with the physical muffin-tin potential
   from a named DFT producer.
2. `SpinorFirstVariation` materialization whose resolved channels
   include the required $4f$ and $5p_{1/2}$ rLO set from an explicit
   caller-owned recipe bound to that runtime basis.
3. Physical $P,Q$ and PP+QQ pair density on a full $q$ slice.
4. One bounded parent grid and candidate set, QRCP and pivoted
   Cholesky at the same rank, at most one rank increment, no sweep.
5. Bounded MPB/THC $c^\dagger V c$ pairs with the Coulomb exactness
   floor kept distinct from `TOL`.
6. MLDUMP roundtrip of those frozen objects.

Until those exist for Dy, material acceptance remains partial: bounded
Sm is executed and Dy is blocked. Neither lane is a release claim.
