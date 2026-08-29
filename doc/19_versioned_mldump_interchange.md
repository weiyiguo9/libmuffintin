# 19. Versioned MLDUMP HDF5 interchange

This note freezes MLDUMP v1, a libmuffintin-owned, inspectable HDF5 schema
for later runtime materialization. It is not CoQui-native and not
SPEX-native. `libmuffintin-io` owns the typed DTOs, schema constants, and
the reader/writer. Runtime, mixed-product, THC, and Coulomb types stay out
of `libmuffintin-io`. A file holds the accepted header plus an optional
representation-neutral scalar or spinor payload written from borrowed
slices; runtime `write_scalar_mldump` and `write_spinor_mldump`
materialize frozen product/THC/Coulomb objects through the streaming
writers. Scalar and spinor payloads share schema version 1.
`write_scalar_coqui_cholesky` is a separate CoQui-native Cholesky ERI
adapter; it is not an MLDUMP payload and does not extend schema version 1.

## 1. Identity and ownership

Every file carries root attributes

- `schema_name = "libmuffintin.mldump"`
- `schema_version = 1`

Indices are zero-based. Floating-point payloads are IEEE-754 `f64`; integer
indices/counts use `i64`, and reciprocal labels use `i32`. Complex arrays use a final length-2 axis with index $0$
the real part and index $1$ the imaginary part. Header-only files record
that convention on `/meta` without writing a complex dataset. Scalar
and spinor payload files store eigenvectors, matching coefficients,
$\zeta$, semantic vertex coefficients, and the finite Coulomb body as
`f64` with a final `re_im` axis.

The crate does not accept serde/TOML blobs renamed `.h5`. The file must be a
real HDF5 container whose groups, datasets, and attributes are independently
inspectable with `h5ls`/`h5dump`.

## 2. Units and coordinates

`/units` is always `present` and stores conventions as attributes, not only
as Rust documentation:

| attribute | value |
| --- | --- |
| `length` | `Bohr` |
| `inverse_length` | `Bohr^-1` |
| `volume` | `Bohr^3` |
| `energy` | `Hartree` |
| `k_q_coordinates` | `fractional_reciprocal` |
| `g_umklapp` | `integer_reciprocal_lattice` |

Direct lattice vectors and site positions are Cartesian Bohr. Reciprocal
primitive vectors include the crystallographic $2\pi$ and are inverse Bohr.
Cell volume is Bohr cubed. $k$ and $q$ are fractional reciprocal coordinates.
$G$ and Umklapp labels are integer reciprocal-lattice triples. Occupations
are not exported in v1 and are never fabricated.

## 3. Status representation

Every reserved group carries a string attribute `status` whose only v1
values are `present` and `absent_not_computed`. A `present` group holds its
documented datasets. An `absent_not_computed` group holds no child member
and is never filled with zeros. A later writer may flip a reserved group
from `absent_not_computed` to `present` without changing v1 names or the
status encoding.

Exchange valence, core, and total are separate seams under `/exchange`.
`total_relation` is absent unless a real same-run valence+core payload
exists. The v1 writer writes all three exchange seams as `absent_not_computed` and
does not write `total_relation`.

## 4. v1 tree

Root attributes: `schema_name`, `schema_version`.

```text
/meta                         status=present
  @producer_name
  @producer_version
  @source_revision
  @feature_representation
  @index_origin               i64 0
  @index_convention           "zero-based"
  @numeric_dtype              "ieee754_f64"
  @complex_encoding           "final_re_im_axis"
  @complex_axis               ["re","im"]
/units                        status=present
  @length @inverse_length @volume @energy
  @k_q_coordinates @g_umklapp
/geometry                     status=present
  direct_basis                f64 [3,3]     axes=["primitive_vector","cartesian"]
  reciprocal_basis            f64 [3,3]     axes=["primitive_vector","cartesian"]
  cell_volume                 f64 scalar
  site_species                utf8 [n_site] axes=["site"]  (empty string if omitted)
  site_labels                 utf8 [n_site] axes=["site"]  (empty string if omitted)
  site_positions              f64 [n_site,3] axes=["site","cartesian"]
  site_radii                  f64 [n_site]  axes=["site"]
  radial_mesh_first           f64 [n_site]  axes=["site"]
  radial_mesh_log_increment   f64 [n_site]  axes=["site"]
  radial_mesh_point_count     i64 [n_site]  axes=["site"]
/mesh                         status=present
  k_fractional                f64 [n_k,3]   axes=["k","reciprocal_axis"]
  k_weights                   f64 [n_k]     axes=["k"]
  q_input_fractional          f64 [n_q,3]   axes=["q","reciprocal_axis"]
  q_canonical_fractional      f64 [n_q,3]   axes=["q","reciprocal_axis"]
  q_global_umklapp            i32 [n_q,3]   axes=["q","reciprocal_axis"]
  k_minus_q_k_index           i64 [n_q,n_k] axes=["q","k"]
  k_minus_q_mapped_index      i64 [n_q,n_k] axes=["q","k"]
  k_minus_q_g_wrap            i32 [n_q,n_k,3] axes=["q","k","reciprocal_axis"]
/orbitals                     status=absent_not_computed | present
/products                     status=absent_not_computed | present
/mpb                          status=absent_not_computed   (reserved; no public MPB DTO)
/thc                          status=absent_not_computed | present
/coulomb                      status=absent_not_computed | present
/exchange                     status=present   (file-level table; no numeric datasets)
  /valence                    status=absent_not_computed
  /core                       status=absent_not_computed
  /total                      status=absent_not_computed
```

Header-only files keep every payload group `absent_not_computed` with no
child member. A populated scalar or spinor file writes all four of
`/orbitals`, `/products`, `/thc`, and `/coulomb` as `present`.
`/orbitals/@representation` is the authoritative branch discriminator
(`scalar_koelling_harmon` or `spinor_full_first_variation`). The current
writer still emits the same tag on `/products`, `/thc`, and `/coulomb`.
Scalar readback accepts those three companion attrs only when they are
all absent (scalar files written before the companion tags) or all
present and equal to `scalar_koelling_harmon`. Spinor readback requires all three present and
equal to `spinor_full_first_variation`; there is no tagless spinor path.
`/meta/@feature_representation` is provenance only. Mixed presence,
absent/present mixture of companion tags, or mixed tag values is a typed
validation error. Ragged
records use only zero-padded integer names `spin_%06d`, `k_%06d`,
`site_%06d`, `q_%06d`. Scalar files keep `spin_%06d` groups; spinor
orbitals have no spin groups.

When `/orbitals` is `present`:

```text
/orbitals                     status=present
  @representation             "scalar_koelling_harmon"
  @spin_count                 i64 2
  @band_window_start          i64 0
  @band_window_count          i64
  @occupations_status         "not_exported_not_available"
  /spin_%06d
    @spin
    /k_%06d
      @spin @k @available_bands @basis_dimension
      eigenvalues             f64 [band_window_count] axes=["band"]
      eigenvectors            f64 [basis_row,band_window_count,re_im]
                              axes=["basis_row","band","re_im"]
      /basis
        plane_wave_g          i32 [plane_wave,3] axes=["plane_wave","reciprocal_axis"]
        plane_wave_k_cartesian f64 [plane_wave,3] axes=["plane_wave","cartesian"]
        plane_wave_q_cartesian f64 [plane_wave,3] axes=["plane_wave","cartesian"]
        local_orbital_*       i64 [local_orbital] axes=["local_orbital"]
                              (row_index, site, l, m, ordinal, radial_n)
        /site_%06d
          lm_l, lm_m          i32 [lm] axes=["lm"]
          matching_coefficients f64 [plane_wave,lm,radial_component,re_im]
            axes=["plane_wave","lm","radial_component","re_im"]
            @radial_component_labels ["u","udot"]
```

`available_bands` is per $k$ metadata and may exceed the common exported
window `band_window_count`. Stored eigenvalues and eigenvector columns
use `band_window_count`, not `available_bands`. Cartesian $k$ and
$q=k+G$ are stored explicitly so compiled plane-wave rows can be
reconstructed; storing only norms would not allow that. Local-orbital tables identify
every non-PW basis row. Per $k$ PW and basis counts may differ.

When `/products` is `present`, static partition/radials are stored once
and positional $q$ records bind by mesh $q$ index:

```text
/products                     status=present
  @representation             "scalar_koelling_harmon"
  @n_k @n_orb
  @pair_order                 "k,left_at_k_minus_q,right_at_k"
  @core_status                "empty_not_fitted_diagnostic_only"
  @provenance_recipe @provenance_reference
  @n_site @interstitial_volume_bohr3
  site_indices                i64 [n_site] axes=["site"]
  site_positions              f64 [n_site,3] axes=["site","cartesian"]
  site_radii                  f64 [n_site] axes=["site"]
  /site_%06d
    @site @mesh_first_bohr @mesh_log_increment @mesh_point_count
    @small_status
    kind,l,n,spin             i64 [radial] axes=["radial"]
    large                     f64 [radial,n_r] axes=["radial","radial_sample"]
    small                     optional f64 [radial,n_r]
  /q_%06d
    @q_index @provenance
    transfer_cartesian        f64 [3] axes=["cartesian"]
    global_transfer           i32 [3] axes=["reciprocal_axis"]
    raw_relative_g            i32 [n_raw,3] axes=["raw_g","reciprocal_axis"]
```

Site indices, positions, radii, and mesh identity bind to `/geometry`
rather than forming a second geometry. `/mesh` remains the owner of input and
canonical $q$ and of the per $k$ wrap map. Each product $q$ record binds
by positional $q$ index: `global_transfer` equals mesh
`q_global_umklapp`, and `transfer_cartesian` equals the mesh canonical
$q$ converted with the header reciprocal basis (already including
$2\pi$) at the accepted $10^{-12}$ scale-aware tolerance. The input $q$
is not reused and the global label is not inserted a second time.
Scalar $U/\dot U$ and LO identities stay based on $l$. Cores are empty and
never enter fitting. Radial ID tables (`kind`,`l`,`n`,`spin`) have equal
length, unique $(kind,l,n,spin)$ within a site, valence kind only, and
spin $0$ or $1$.

When `/orbitals` is spinor `present`:

```text
/orbitals                     status=present
  @representation             "spinor_full_first_variation"
  @band_window_start          i64 0
  @band_window_count          i64
  @occupations_status         "not_exported_not_available"
  /k_%06d
    @k @available_bands @basis_dimension
    eigenvalues               f64 [band_window_count] axes=["band"]
    eigenvectors              f64 [basis_row,band_window_count,re_im]
                              axes=["basis_row","band","re_im"]
    /basis
      plane_wave_g            i32 [plane_wave,3]
      plane_wave_k_cartesian  f64 [plane_wave,3]
      plane_wave_q_cartesian  f64 [plane_wave,3]
      pauli_row_index         i64 [pauli_row]
      pauli_component         i64 [pauli_row]
      pauli_plane_wave_index  i64 [pauli_row]
      local_orbital_*         i64 [local_orbital]
                              (row_index, site, signed_kappa, twice_mu, ordinal, radial_n)
      /site_%06d
        projection_*          i64 [projection_coordinate]
                              (coordinate, signed_kappa, twice_mu, radial_n)
        matching_coefficients f64 [plane_wave,pauli_component,projection_coordinate,re_im]
          axes=["plane_wave","pauli_component","projection_coordinate","re_im"]
```

There is no collinear spin field and no `spin_%06d` group. Shared spatial
$G$ labels are stored once. Pauli rows satisfy
$\mathrm{row}=$ `pauli_component` $\,N_G+$ `plane_wave_index`
with `pauli_component` $\in\{0,1\}$. Local-orbital tables identify
confined LO/RLO eigenbasis rows only; APW $P$ and $\dot P$ are matching
columns on those plane-wave rows, flattened onto the projection-coordinate
axis so the matching dataset stays rank 4. The projection table is a
strict APW prefix of the live channel order, each
$(\kappa,2\mu,n=0)$ then $(\kappa,2\mu,n=1)$, followed by the LO/RLO
tail whose ordered $(site,\kappa,2\mu,n)$ identities match that site's
local-row table. The matching third axis is that APW prefix length.
Signed $\kappa$ is nonzero; $2\mu$ belongs to $j=|\kappa|-1/2$.
Eigenvectors remain C-order
[`basis_row`, `band`, `re_im`] with width equal to
the common window; `available_bands` may be larger.

When `/products` is spinor `present`:

```text
/products                     status=present
  @representation             "spinor_full_first_variation"
  @n_k @n_orb
  @pair_order                 "k,left_at_k_minus_q,right_at_k"
  @core_status                "empty_not_fitted_diagnostic_only"
  ... partition datasets as in the scalar tree ...
  /site_%06d
    kind, signed_kappa, n     i64 [radial] axes=["radial"]
    p, q                      f64 [radial,n_r] axes=["radial","radial_sample"]
  /q_%06d                     same transfer/global/raw-G schema as scalar
```

Dirac radial identity is $(kind,\kappa,n)$ with no $\mu$ and no fake
scalar $l$ / spin fields. Physical $P$ and $Q$ are both required.
$n=0$ is $(P,Q)$, $n=1$ is $(\dot P,\dot Q)$, and $n\ge 2$ is LO/RLO.
Cores remain empty/diagnostic-only.

When `/thc` is `present`, the parent grid is stored once:

```text
/thc                          status=present
  @representation             "scalar_koelling_harmon" | "spinor_full_first_variation"
  @strategy                   "AllQL2"
  @engine                     "full_column_pivoted_qr" | "full_pivoted_cholesky"
  @requested_rank @effective_rank @n_candidates
  pivots                      i64 [rank_order] axes=["rank_order"]
  points                      i64 [aux] axes=["aux"]
  /parent_grid
    coordinates               f64 [parent_point,3] axes=["parent_point","cartesian"]
    weights                   f64 [parent_point] axes=["parent_point"]
    region_kind               i64 [parent_point]  0 muffin-tin, 1 interstitial
    site_index, radial_index  i64 [parent_point]  interstitial sentinel $-1$
  /q_%06d
    @q_index @aux_dimension @layout_provenance
    zeta                      f64 [parent_point,aux,re_im]
                              axes=["parent_point","aux","re_im"]
    fit_residual_l2_all       f64 [2] axes=["metric"]
    vertex_column             i64 [vertex] axes=["vertex"]
    vertex_k_left_right       i64 [vertex,3] axes=["vertex","k_left_right"]
    vertex_coefficients       f64 [vertex,aux,re_im] axes=["vertex","aux","re_im"]
```

`pivots` is the QRCP/Cholesky ranking. `points` is the same selected
parent-index set in canonical auxiliary/layout order. The two arrays are
distinct fields and must not be compared as vectors; both are unique, in
parent bounds, and every selected index references a strictly positive
parent weight. Zero parent-grid weights remain allowed as $\zeta$ rows but
cannot be selected. Parent weights are finite and nonnegative. Region
kind is muffin-tin or interstitial only, with interstitial sentinel
$-1$. The parent grid is not duplicated under $q$ groups. Each semantic
vertex `column` decodes as
$k\cdot n_{\mathrm{orb}}^2+\mathrm{left}\cdot n_{\mathrm{orb}}+\mathrm{right}$
in pair order `k,left_at_k_minus_q,right_at_k` and must equal the stored
`k_left_right` triple. Overflow or an out-of-range column is a
validation error.

When `/coulomb` is `present`:

```text
/coulomb                      status=present
  @representation             "scalar_koelling_harmon" | "spinor_full_first_variation"
  @lexp @interpolation_l_max @interpolation_pw_cutoff
  /q_%06d
    @q_index @aux_dimension @layout_provenance
    body                      f64 [aux_row,aux_col,re_im]
                              axes=["aux_row","aux_col","re_im"]
    /gamma                    status=present | absent_not_computed
      @spherical_average_subtracted   i64 0 or 1
      @head_prefactor         f64
      constant_coefficients   f64 [aux,re_im] axes=["aux","re_im"]
```

The stored matrix is the finite Hermitian body. The singular Gamma head
is never inserted into $V$. A present `/gamma` stores only the current
finite GammaHead metadata and is allowed only when the header canonical
$q$ is the zero vector at the same $10^{-12}$ tolerance. An absent
`/gamma` has no members, including at $q=0$. Per $q$ auxiliary dimension
and `layout_provenance` must equal the THC record at the same $q$.
$\zeta$, vertices, and the parent grid are not duplicated here.

Every multidimensional numeric dataset has an `axes` attribute. Row-major
storage matches HDF5 C order: for `direct_basis`, row $i$ is primitive
vector $\mathbf a_i$ and column $j$ is the Cartesian component. The
canonical transfer stored in `q_canonical_fractional` is the folded
$q$ used by later pair maps; `q_global_umklapp` is
$G_{\mathrm{transfer}}$ with
$q_{\mathrm{in}}=q_{\mathrm{canonical}}+G_{\mathrm{transfer}}$
within the documented scale-aware tolerance
$|a-b|\le 10^{-12}\max(|a|,|b|,1)$, the same $10^{-12}$ floor as the
product-input mesh-coordinate gate. Every canonical $q$ component lies in
$[0,1)$. Each $q$ stores exactly $n_k$ $k-q$ records in canonical $k$
order: the record at position $i_k$ has $k_{\mathrm{index}}=i_k$. The
per $k$ wrap satisfies
$k_{\mathrm{frac}}-q_{\mathrm{canonical}}=k_{\mathrm{frac}}^{\mathrm{mapped}}+G_{\mathrm{wrap}}$
with that same tolerance and does not reinsert the global transfer
Umklapp. Stored $k$ fractions are used as written; they are not
re-folded. Numeric datasets are read only when the on-disk element type
is the exact v1 dtype (`f64`, `i32`, or `i64`); HDF5 convertible
widening is rejected.

Radial meshes record site binding, first radius $r_0$, logarithmic
increment $h$ in $r_i=r_0 e^{i h}$, and point count. They do not embed
runtime mesh types.

## 5. Complex encoding

A complex array with logical shape $(d_0,\ldots,d_N)$ is stored as
`f64` with shape $(d_0,\ldots,d_N,2)$ and a final axis named `re_im`.
Index $0$ is $\mathrm{Re}$ and index $1$ is $\mathrm{Im}$. Header-only
files contain no such dataset. Scalar and spinor payload files store
eigenvectors, APW matching coefficients, $\zeta$, semantic vertex
coefficients, and the finite Coulomb body in that encoding. The Gamma
singular head is never inserted into $V$.

## 6. Public API

`MldumpWriterV1::create(path, &MldumpHeaderV1)` writes the accepted
header and reserved absent groups. Header-only files call
`MldumpWriterV1::finish`. Populated scalar files continue with
`begin_scalar()` into [`ScalarMldumpStreamV1`]. Populated spinor files
continue with `begin_spinor()` into [`SpinorMldumpStreamV1`]. Each of
`/orbitals`, `/products`, `/thc`, and `/coulomb` is opened with a
`begin_*` method, written as ordered per $(spin,k)$ or per $k$, per-site,
or per $q$ records, and closed with `finish_*`. Products must be written
before THC. Large arrays are written immediately. An extra site, $k$, or
$q$ record is a typed validation error before any further HDF5 child is
created. Semantic THC vertex `column` identity is checked locally in $q$
against the product $n_k/n_{\mathrm{orb}}$ pair layout while the
borrowed $q$ record is alive; the session retains only small counters,
$q$ bindings, auxiliary dimension/provenance, and pair-layout counts, not
vertex tables. Stream `finish` requires all four sections and runs the
shared cross-section alignment validator. Mixed partial sections, mixed
representation tags, and cross-section mismatches are typed validation
errors. A failed or interrupted write may leave an incomplete file.
`/mpb` stays `absent_not_computed`. `/exchange` stays present with three
absent children and no `total_relation`.

`read_mldump_v1(path)` returns
`MldumpFileV1 { header, payload, exchange }` with
`payload: MldumpPayloadV1`. All four groups absent yields
`HeaderOnly`. All four present with `/orbitals/@representation =
scalar_koelling_harmon` yields `Scalar(ScalarMldumpV1)`, including
earlier published files whose `/products`,`/thc`,`/coulomb`
representation attrs are all absent. All four present with
`spinor_full_first_variation` on orbitals and the three companion groups
yields `Spinor(SpinorMldumpV1)`. Mixed presence, mixed or partial
companion tags, a cross-section mismatch, or a local payload
violation is rejected.
Borrowed writer DTOs are public concrete structs with slice references
and explicit shapes, including `ScalarOrbitalsBeginV1`,
`SpinorOrbitalsBeginV1`, `ScalarProductsBeginV1`,
`SpinorProductsBeginV1`, `MldumpThcBeginV1`, `MldumpCoulombBeginV1`, and
the per-record refs. Neutral `MldumpThc*` / `MldumpCoulomb*` DTOs are
shared across representations.
Owned reader DTOs use `Vec`/`String` records with declared dimensions.

Validation is the trust boundary: schema name/version, exact numeric
HDF5 dtypes including attributes, finite numbers, nonnegative full-BZ
weights, positive volume/radii/mesh first radius/log increment/counts,
canonical $q$ in $[0,1)$, $q_{\mathrm{in}}=q_{\mathrm{canonical}}+G_{\mathrm{transfer}}$,
ordered $k-q$ wrap identities, $k+G$ Cartesian reconstruction, product
$q$ Cartesian/global binding to mesh canonical $q$, semantic vertex
column decode, THC/Coulomb auxiliary dimension and provenance identity,
Gamma metadata only at canonical $q=0$, Hermitian finite $V$, positive
selected parent-grid weights, auxiliary, $\zeta$, and vertex layout alignment,
required groups/datasets, and status-versus-payload consistency. These
payload validators apply on write and on read.

## 7. Exclusions

MLDUMP v1 does not serialize CoQui or SPEX native layouts, runtime
`SpinorProductInput` / `ScalarProductInput` objects, MPB payloads,
occupations, core-valence products, the singular Gamma head,
GW/RPA/self-energy, or MPI.
