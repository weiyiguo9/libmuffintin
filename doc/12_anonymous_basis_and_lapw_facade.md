# 12. Anonymous basis extraction and the LAPW facade

This note records the anonymous-basis facade contract. It does not add a mixed product
basis, THC, Coulomb operators, an umbrella `libmuffintin` crate, or a
distributed tensor runtime. Formulae for matching, the SPEX kinetic
convention, and APW+lo layout remain in [08](08_lapw_matching_and_overlap.md),
[09](09_hamiltonian_eigensolver_and_reference_reports.md), and
[10](10_apw_lo_and_collinear_spin.md). The tensor substrate remains
[11](11_tensorized_numerical_substrate.md).

## 1. Package, crate-target, and directory names

Workspace directories uniformly use the `crates/mt-*` prefix, while published
Cargo package names use `libmuffintin-*`. Explicit Rust library targets omit
the platform-supplied `lib` prefix and use `muffintin_*`.

| Directory | Cargo package | Rust crate target |
|---|---|---|
| `crates/mt-core` | `libmuffintin-core` | `muffintin_core` |
| `crates/mt-radial` | `libmuffintin-radial` | `muffintin_radial` |
| `crates/mt-sphere` | `libmuffintin-sphere` | `muffintin_sphere` |
| `crates/mt-grid` | `libmuffintin-grid` | `muffintin_grid` |
| `crates/mt-io` | `libmuffintin-io` | `muffintin_io` |
| `crates/mt-tensor` | `libmuffintin-tensor` | `muffintin_tensor` |
| `crates/mt-envelope` | `libmuffintin-envelope` | `muffintin_envelope` |
| `crates/mt-basis` | `libmuffintin-basis` | `muffintin_basis` |
| `crates/mt-operators` | `libmuffintin-operators` | `muffintin_operators` |
| `crates/mt-recipes` | `libmuffintin-recipes` | `muffintin_recipes` |
| `crates/mt-lapw` | `libmuffintin-lapw` | `muffintin_lapw` |
| `crates/mt-auxiliary-ir` | `libmuffintin-auxiliary-ir` | `muffintin_auxiliary_ir` |
| `crates/mt-mpb` | `libmuffintin-mpb` | `muffintin_mpb` |
| `crates/mt-thc` | `libmuffintin-thc` | `muffintin_thc` |
| `crates/mt-coulomb` | `libmuffintin-coulomb` | `muffintin_coulomb` |

There is no compatibility package named `mt-*`. Rust imports use underscores,
for example `muffintin_core`. Rust artifacts therefore start with a single
platform prefix, such as `libmuffintin_core.rlib`. A system-facing target named
`muffintin` maps conventionally to `libmuffintin.so` or `libmuffintin.a` on
Linux.

## 2. Dependency DAG

```text
core
tensor
radial            → core
sphere            → core, radial
grid              → core
io                → grid
envelope          → core
basis             → core, envelope, radial
operators         → core, tensor, basis, faer
recipes           → core, basis, envelope
lapw              → recipes, operators, basis, envelope, core, radial, tensor
auxiliary-ir      → core, radial, basis
mpb               → auxiliary-ir, operators, core, radial, basis, envelope
```

`libmuffintin-basis` stores host augmentation coefficients only. Backend
tensor handles stay inside `libmuffintin-tensor`. `recipes::lapw` never
depends on `libmuffintin-lapw`.

## 3. Envelope, spec, compile, assemble

`libmuffintin-envelope` owns `PlaneWave` and a concrete `PlaneWaveEnvelope`
that stores that plane-wave set. Rayleigh evaluation and the site-translation
phase stay in the same crate. There is no envelope trait family.

The specification is historical-method-name-free: block names do not encode
LAPW, APW+lo, or another method family. v0.2 typed variants implement only a
plane-wave envelope with APW site augmentation and confined site-local
overlays. This is not an arbitrary method-neutral payload and does not
introduce a trait hierarchy.

```text
BasisSpec
  └─ BasisBlock[]
       ├─ PlaneWaveEnvelope { envelope, sites: ApwSiteAugmentation[] }
       └─ ConfinedSite { site, local_orbitals }
```

`ApwSiteAugmentation` carries the required muffin-tin position, radius, and
`(u, udot)` boundary columns. `ConfinedSite` carries the required site index
and local-orbital layout. `compile` stores each APW site's position and
radius on `CompiledBasis` as `ApwSiteGeometry`. Unused optional fields such
as `l_max` and `interstitial` are not part of the IR.

Compilation and assembly are:

```text
PlaneWaveEnvelope + LapwSiteInput
        │
        ▼
recipes::lapw() ─→ BasisSpec ─compile→ CompiledBasis ─assemble_compiled→ OperatorSet(H, S)
                       ▲
                       │
              explicit BasisSpec
```

`compile` takes only a `BasisSpec`. Plane waves come from the envelope block;
an empty spec does not acquire waves from any other argument. Duplicate
confined-site blocks are an error. `compile` performs SPEX APW matching and
Rayleigh $\times$ site-phase augmentation, and keeps APW site geometry on
the compiled result. `assemble_compiled` checks that geometry against
`InterstitialGeometry` spheres on both the facade and explicit-spec routes,
fills the interstitial plane-wave block in `libmuffintin-lapw`, and then
calls `libmuffintin-operators::add_site_contributions` for every
$P^\dagger B P$ site congruence. The filtered generalized solver is
`libmuffintin-operators::solve_generalized_hermitian`.

Two construction routes must agree within the APW+LO tolerances:

- **Facade.** `assemble_eigenproblem` receives a `PlaneWaveEnvelope`,
  geometry, potential, `LapwSiteInput` recipe sites, and local
  `SiteOperatorBlocks`. Internally it runs
  `recipes::lapw` $\to$ `compile` $\to$ `assemble_compiled`. The collinear
  driver compiles the basis once and reuses it for both spins.
- **Explicit spec.** Callers fill `BasisSpec` by hand, including a real
  `PlaneWaveEnvelope` block when plane waves are required, then call
  `compile` and `assemble_compiled`. This route must not call
  `recipes::lapw()`.

`recipes::lapw()` is a provenance-bearing constructor of `BasisSpec`.

## 4. What stays in the LAPW facade

The facade is not an empty re-export crate. It keeps LAPW-specific physics:

- interstitial kinetic assembly with the explicit strategy
  `KineticOperatorConvention::SpexSymmetricLaplacian`;
- Hermitian interstitial potential lookup;
- the SPEX spherical radial identity
  `spex_spherical_radial_hamiltonian`;
- the collinear no-SOC driver;
- $(k,\mathrm{band})$ reference comparison at the default one-meV tolerance.

It re-exports envelope, layout, matching, `BasisSpec`, `compile`,
`OperatorSet` as `LapwEigenproblem`, `BasisLayout` as `LapwBasisLayout`,
`SiteOperatorBlocks` from the operator crate, and `recipes::lapw`. Local
site blocks store only $S$ and $H$; projection maps live on `CompiledBasis`.

The interstitial Hamiltonian element is

```math
H^I_{ij}
= \frac{|q_i|^2+|q_j|^2}{4}\,\Theta_I(G_i-G_j)
+ V^I(G_i-G_j),
```

with $\mathbf q = \mathbf k+\mathbf G$. The prefactor is
`KineticOperatorConvention::SpexSymmetricLaplacian.prefactor`, not a hidden
literal. The gradient form must not be mixed with the SPEX radial identity.

Site contributions remain the tensor substrate congruence

```math
M_a = P_a^\dagger B_a P_a
```

evaluated as `einsum("ci,cd,dj->ij", [P^*, B, P])` through
`hermitian_congruence`. Eigenvectors keep column-major `[GlobalBasis, Band]`
storage.

## 5. Acceptance

The facade requires the APW+LO $H$, $S$, retained overlap rank, filtered rank,
eigenvalues, absolute and relative residuals, and (up to a global phase on
non-degenerate columns) eigenvectors to match between the facade and
explicit-`BasisSpec` routes on the existing empty-lattice, translated-sphere,
APW+lo, and collinear fixtures. It does not claim the Cu/SPEX one-meV gate,
product bases, or a distributed runtime.
