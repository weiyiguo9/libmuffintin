# 19. Versioned MLDUMP HDF5 interchange

This note freezes MLDUMP v1, a libmuffintin-owned, inspectable HDF5 schema
for later runtime materialization. It is not CoQui-native and not
SPEX-native. `libmuffintin-io` owns the typed DTOs, schema constants, and
the reader/writer. Runtime, mixed-product, THC, and Coulomb types stay out
of `libmuffintin-io`. M-L6b and later stages write live objects into this
stable boundary; this stage serializes only the header, geometry, mesh, and
group-status table.

## 1. Identity and ownership

Every file carries root attributes

- `schema_name = "libmuffintin.mldump"`
- `schema_version = 1`

Indices are zero-based. Floating-point payloads are IEEE-754 `f64`; integer
indices/counts use `i64`, and reciprocal labels use `i32`. Complex arrays,
when a later stage writes them, use a final length-2 axis with index $0$
the real part and index $1$ the imaginary part. That convention is recorded
on `/meta` in v1 even though this stage writes no complex dataset.

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
are optional in later stages and are not fabricated here.

## 3. Status representation

Every reserved group carries a string attribute `status` whose only v1
values are `present` and `absent_not_computed`. A `present` group holds its
documented datasets. An `absent_not_computed` group holds no child member
and is never filled with zeros. Later stages may flip a reserved
group from `absent_not_computed` to `present` without changing v1 names or
the status encoding.

Exchange valence, core, and total are separate seams under `/exchange`.
`total_relation` is absent unless a real same-run valence+core payload
exists. M-L6a writes all three exchange seams as `absent_not_computed` and
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
/orbitals                     status=absent_not_computed
/products                     status=absent_not_computed
/mpb                          status=absent_not_computed
/thc                          status=absent_not_computed
/coulomb                      status=absent_not_computed
/exchange                     status=present   (reserved table; no numeric datasets)
  /valence                    status=absent_not_computed
  /core                       status=absent_not_computed
  /total                      status=absent_not_computed
```

Every multidimensional numeric dataset has an `axes` attribute. Row-major
storage matches HDF5 C order: for `direct_basis`, row $i$ is primitive
vector $\mathbf a_i$ and column $j$ is the Cartesian component. The
canonical transfer stored in `q_canonical_fractional` is the folded
$q$ used by later pair maps; `q_global_umklapp` is
$G_{\mathrm{transfer}}$ with
$q_{\mathrm{in}}=q_{\mathrm{canonical}}+G_{\mathrm{transfer}}$
within the documented scale-aware tolerance
$|a-b|\le 10^{-12}\max(|a|,|b|,1)$, the same $10^{-12}$ floor as the
M-L1/M-L5b mesh-coordinate gate. Every canonical $q$ component lies in
$[0,1)$. Each $q$ stores exactly $n_k$ $k-q$ records in canonical $k$
order: the record at position $i_k$ has $k_{\mathrm{index}}=i_k$. The
per-$k$ wrap satisfies
$k_{\mathrm{frac}}-q_{\mathrm{canonical}}=k_{\mathrm{frac}}^{\mathrm{mapped}}+G_{\mathrm{wrap}}$
with that same tolerance and does not reinsert the global transfer
Umklapp. Stored $k$ fractions are used as written; they are not
re-folded. Numeric datasets are read only when the on-disk element type
is the exact v1 dtype (`f64`, `i32`, or `i64`); HDF5 convertible
widening is rejected.

Radial meshes record site binding, first radius $r_0$, logarithmic
increment $h$ in $r_i=r_0 e^{i h}$, and point count. They do not embed
runtime mesh types.

## 5. Complex encoding for later stages

A later complex array with logical shape $(d_0,\ldots,d_N)$ is stored as
`f64` with shape $(d_0,\ldots,d_N,2)$ and a final axis named `re_im`.
Index $0$ is $\mathrm{Re}$ and index $1$ is $\mathrm{Im}$. v1 header files
contain no such dataset. Pair vertices and the finite Coulomb body are
permitted by the schema later; the Gamma singular head is never inserted
into $V$. This stage serializes neither quantity.

## 6. Public API

`write_mldump_v1(path, &MldumpV1)` and `read_mldump_v1(path)` are the
only I/O entry points. `MldumpV1` holds producer metadata, geometry, the
$k$/$q$ mesh, and reserved-group statuses. The M-L6a writer and reader
require orbitals, products, MPB, THC, Coulomb, and all three exchange
seams to be `absent_not_computed`. An absent group may not carry any
child member (dataset or subgroup); `/exchange` itself stays `present`
because it owns the three status children. Validation is the trust
boundary: schema name/version, exact numeric HDF5 dtypes, finite numbers,
nonnegative full-BZ weights, positive volume/radii/mesh first radius/log
increment/counts,
canonical $q$ in $[0,1)$, $q_{\mathrm{in}}=q_{\mathrm{canonical}}+G_{\mathrm{transfer}}$,
ordered $k-q$ wrap identities, $3$-vector and $k$-map shapes, required
groups/datasets, and status-versus-payload consistency.

## 7. Exclusions

MLDUMP v1 does not serialize CoQui or SPEX native layouts, runtime
`SpinorProductInput` / `ScalarProductInput` objects, MPB or THC results,
Coulomb operators, PairVertex coefficients, occupations, core-valence
products, Gamma singular heads, GW/RPA/self-energy, MPI, or material
acceptance. It does not complete M-L.
