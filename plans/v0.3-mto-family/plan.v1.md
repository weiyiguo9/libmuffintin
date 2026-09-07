# libmuffintin v0.3 plan — LMTO route: screened FP-TB-LMTO, smooth-Hankel presets, and basis optimization

- Workstream ID: `v0.3-mto-family`
- Plan version: 1
- Approval: draft (not authorized)
- Supersedes: none
- Imported: 2026-09-08 from `scratch/libmuffintin_v0.3_mto_family_plan.md` (last modified 2026-08-27), body unchanged
- Status source: false; state lives in `STATUS.md` and `ledger.md`
- Path map: `libmuffintin_v0.4_emto_nmto_plan.md` is `plans/v0.4-emto-nmto/plan.v1.md`; `design_decision_02_mto_poisson.md` is `decisions/ADR-0003-mto-density-poisson-kinetic.md`

Status: revised draft, 2026-08-27. **Split 2026-08-27**: NMTO, EMTO,
full-potential unification, and the adaptive-grid/DMK Coulomb route moved to
`libmuffintin_v0.4_emto_nmto_plan.md`; v0.4 inherits this plan's §0, §1, §2,
and §9 without restating them. `design_decision_02_mto_poisson.md` is merged
into both plans and retired to an archived rationale record with a mapping
table; the plans are normative.

Implementation baseline: repository `main` has completed the M-A through M-Kc
implementation contracts; their independent cross-code/material acceptance
remains a separate open gate.  Assumption for starting the full v0.3 roadmap:
v0.2 M-L is also complete, including the backend-neutral tensor substrate,
public-package rename, anonymous `BasisSpec/BasisBlock/CompiledBasis` core,
`libmuffintin-lapw` facade/reference preset, adaptive-grid THC, Coulomb metric,
the method-neutral product-space IR, the SPEX-compatible LAPW mixed product
basis, minimal DFT, and MPB/THC integration tests.

The v0.3 objective is not “implement another MTO code.” It extends the anonymous
basis IR with site-centered envelopes, scattering, screening, and linearized
energy representations, while shipping historical constructions only as
reference presets built from exactly those public components. The energy-mesh
transform (NMTO) and Green-function workflow (EMTO) consumers of these layers
are v0.4 deliverables.

v0.3 is library-first: it distributes no default executable and does not widen
the v0.2 `muffintin` TOML interface. `libmuffintin-basisopt` carries an opt-in
`[[bin]]` target that is not part of the default release artifacts (users
`cargo install` or build it themselves); it is the only executable anywhere in
v0.3, and its input/cache/provenance protocol is therefore the only file-level
contract v0.3 ships: independently versioned, marked experimental, and
explicitly non-normative for the future TOML interface — the later binary
interface designs its tokens fresh rather than inheriting the `basisopt` wire
format. Users compose the published Rust specs, recipes, algorithms, and
`libmuffintin-basisopt` crate in their own Rust application or an explicitly
experimental thin Rust/Python binding (no API-stability promise). File-level
input and stable token spelling are deferred until a later binary interface is
designed from demonstrated library use.

## 0. Frozen scope decisions

1. **The unit of abstraction is an anonymous basis specification, not a
   historical method name.**

   ```text
   BasisSpec = BasisBlock[]
   BasisBlock = support/envelope
              + site/channel selection
              + screening representation
              + energy representation
              + augmentation
   CompiledBasis + OperatorSet -> spectral consumer
   spectral result + closure   -> workflow
   ```

   There is no public `BasisFamily::{Lapw, Jpo, Nmto, Emto}` and no universal
   `MtoRecipe` type. These names do not describe the same architectural layer:

   - LAPW and particular FP-LMTO/JPO constructions are basis presets;
   - PMT is a heterogeneous union of basis blocks;
   - NMTO is an energy-mesh interpolation/downfolding transform;
   - EMTO is a kink-matrix and Green-function spectral workflow;
   - SCA, FCD, and FP are density/potential closures.

   Historical names appear only in `recipes`, `transforms`, `workflows`,
   regression tests, documentation, and provenance. They never define the
   generic type or crate boundary. Any FP-LMTO variant is therefore a preset
   assembled from public components; libmuffintin ships selected, validated
   combinations without restricting user-defined combinations.

2. **Andersen screening and smooth Hankels are orthogonal choices.**
   Andersen/Ke-style screened spherical waves belong to the screening and
   hard-sphere boundary layer. QuESTAAL's smooth Hankel belongs to the
   envelope-kernel layer. Neither is allowed to own augmentation, radial
   solutions, or the eigensolver.

3. **Energy treatment is a separate axis.**
   A two-energy JPO basis is not silently called NMTO. NMTO means an explicit
   energy mesh, divided-difference/interpolation construction, and optional
   downfolding. EMTO uses an energy-dependent kink/slope matrix and a contour
   Green-function solution; it is not a high-order LMTO enum value.

4. **Potential spheres and screening spheres are distinct objects.**
   Non-overlapping augmentation spheres, hard screening spheres, and EMTO's
   overlapping potential spheres may have different radii. No generic
   `sphere_radius` field is permitted.

5. **FCD/SCA/FP are density-potential closures, not basis names.**
   The canonical EMTO vertical slice is `exact-energy scattering + overlapping
   potential spheres + SCA + FCD`. A full-potential screened-wave workflow is
   also supported, but must not be labelled “FP-EMTO” until its variational
   relation to the canonical EMTO kink equation is derived and tested.

6. **Terminology:** this plan uses **SCA** (spherical-cell approximation), the
   name used by the EMTO literature and code. If “SCS” refers to a different
   intended construction, it must become a separately specified closure.

7. **`libmuffintin-lapw` remains a supported facade/reference preset.** Its
   industry-standard name and stable public entry points are retained, while
   plane-wave envelopes, augmentation, basis layout, operator assembly, and
   eigensolvers live in method-neutral crates. Its M-F results remain regression
   anchors throughout v0.3.

8. **Library-only public workflow path.** v0.3 ships no default executable or
   Python interface. Any downstream executable or experimental Rust/Python
   binding calls the same public components available to library users; no
   private workflow path is permitted.

9. **Relativity reuses v0.2.** Scalar-relativistic and 4c radial solutions are
   supplied through the existing `RadialSolution`/`RadialComponents` contract.
   The new scattering algebra must be scalar-generic and channel-generic so a
   later spin-angular `κ,m_j` structure matrix does not require a rewrite.

10. **CPA, transport, forces, and phonons are out of scope for v0.3.** The EMTO
   interfaces must not block CPA, but v0.3 ends at ordered-crystal SCF and
   Green functions.

11. **Automatic differentiation is out of scope.** v0.3 does not make radial
    solvers, structure constants, eigensolvers, or SCF differentiable. All
    basis search is derivative-free; local sensitivities use controlled
    parameter perturbations.

12. **Gaussian orbitals are out of scope.** The envelope interface must not
    prevent a future `GaussianEnvelope`, but v0.3 does not add GTO recipes,
    `libcint` integration, or X2C/DKH interstitial machinery.

13. **No conic-solver dependency.** Overlap health is handled by diagnostics,
    explicit rank reduction, feasible parameter domains, and derivative-free
    search. v0.3 does not introduce SCS, Clarabel, or an SDP/conic
    reformulation.

14. **The v0.2 product-space and MBP crates are foundational, not LAPW
    attachments.** `libmuffintin-product` owns the raw pair-space and common
    auxiliary-basis/vertex contracts; `libmuffintin-mbp` and
    `libmuffintin-thc` are sibling compression backends; and
    `libmuffintin-coulomb` consumes either. Every v0.3 one-particle preset must
    expose the same product-evaluation capabilities. The SPEX LAPW MPB remains
    the normative baseline, while MTO/PMT product spaces are tested through
    public adapters rather than private GW-oriented code.

    `libmuffintin-thc` does not own a Poisson or Coulomb-backend seam. Its
    Coulomb-aware selection and diagnostics consume backend-neutral injected
    pair-pair Grams, including their q/layout and Hermitian/PSD validation;
    production $V^q$ assembly remains in `libmuffintin-coulomb`. Any future
    FFT, DMK, or other Poisson choice therefore belongs behind the Coulomb
    assembly/recipe layer and must not add a backend enum or configuration to
    the THC public interface.

15. **All dense algebra reuses the M-Fb tensor substrate.** Structure matrices,
    screening transforms, cross-block assembly, kink matrices, contour Green
    functions, pair vertices, and Coulomb actions express their contractions
    through `libmuffintin-tensor`; they do not reopen backend selection inside
    method crates. RSTSR remains the default and tenferro-rs remains an optional
    parity backend. Backend-specific tensor types never enter `BasisSpec`,
    scattering conventions, workflow manifests, or serialized artifacts.

    v0.3 preserves CTF-like distributed extensibility through a global tensor
    model: execution world/context, global rank and shape, index expressions,
    symmetry, and automatic placement intent. Physics code does not
    prescribe local shards or manually combine partial reductions; a CTF
    implementation retains ownership of processor mapping, redistribution, and
    contraction planning. MPI and a production CTF backend remain outside v0.3
    unless a separate measured milestone is approved; “CTF-compatible” is not
    reported as “distributed.”

16. **Orbital-config V2 deferrals land here.** The doc/17 V2 channels/token
    input design (scratch/orbit_config.md, closed 2026-08-24) defers four
    items to v0.3: fill-mode channel generators (SPEX `fbas`/`l:+`, Elk
    `lorbcnd`) as recipe-generator features rather than token grammar; the
    `l`-first energy-generation mode (Elk/SPEX/Questaal convention — V2
    ships κ-first with degeneracy averaging only, FLEUR's j-resolved LO
    average being the κ-first precedent); and the explicit
    `fallback = "seed"` opt-in for energy-generator failures (V2 is
    hard-error only).  The fourth item is Rust-library execution support for
    `derivative-order >= 3`: the V2 IR already accepts every nonnegative order
    but returns a typed `NotImplemented` result above HDLO order 2. Higher
    orders remain an explicit v0.3 library feature, using shifted-energy radial
    solutions plus orthogonalization rather than widening the V2 runtime input;
    file-level token spelling is deferred with the future binary interface.

## 1. Central factorization

### 1.1 The eight independent layers

| Layer | Public choices in v0.3 | Must not know about |
|---|---|---|
| Geometry | non-overlap MT; hard-sphere screening; overlapping potential spheres | energy interpolation, SCF |
| Envelope kernel | Helmholtz Hankel/Neumann; smooth Hankel; plane wave from v0.2 | screening representation |
| Structure matrix | real-space/Ewald; reciprocal/Bloch; cached tabulation | augmentation and density |
| Screening | unscreened; Andersen representation transform; hard-sphere boundary screening | smoothness of envelope |
| Energy representation | exact `E`; linearized `(Eν, ∂E)`; independent multi-head; interpolation mesh | density closure |
| Augmentation | value/slope; value-and-derivative; Ke-style double augmentation; v0.2 full-potential sphere algebra | band solver |
| Spectral consumer | generalized eigenproblem; kink cancellation; contour Green function | SCF mixing |
| Density closure | v0.2 full potential; SCA+FCD; diagnostic ASA | screening convention |

The dependency direction is one-way. For example, an NMTO builder consumes
energy-dependent kink matrices; the kink-matrix implementation never imports
NMTO interpolation.

### 1.2 Validated presets, transforms, and workflows

| Public convenience | Kind | Envelope | Screening | Energy/spectral path | Reference closure |
|---|---|---|---|---|---|
| `recipes::fp_tb_lmto()` | basis preset | ordinary Helmholtz/Hankel SSW | Andersen/hard-sphere | linearized generalized eigenproblem | full potential + double augmentation |
| `recipes::questaal_jpo()` | basis preset | smooth Hankel | hard-core matrix screening | independent 1–2 energies + generalized eigenproblem | full potential |
| `recipes::questaal_pmt()` | composition preset | plane waves + smooth Hankels | screened site-centered block | independent 1–2 energies + mixed generalized eigenproblem | full potential |
| `transforms::nmto()` | basis transform | consumes screened spherical-wave data | preserves declared representation | mesh `{epsilon_0...epsilon_N}` + downfolding | inherited/frozen-potential first |
| `workflows::emto_sca_fcd()` | spectral/SCF workflow | screened spherical waves | hard spheres | exact `E`, kink equation + contour Green function | SCA + FCD |
| `workflows::ssw_fp_exact_e()` | experimental workflow | ordinary SSW | hard spheres | exact-energy kink/Green-function path | full potential |

For a basis preset, the last two columns describe its validated reference
workflow, not state owned by `BasisSpec`. The last row is intentionally not
named `emto_fp` in the public API during v0.3. The `transforms::nmto()`,
`workflows::emto_sca_fcd()`, and `ssw_fp_exact_e` rows are v0.4 deliverables,
kept here as the family map.

### 1.3 Orthogonal pair-space factorization

The eight layers above describe the one-particle construction. Product space is
an orthogonal pipeline with its own geometry, windows, cutoffs, and provenance:

```text
CompiledBasis + ProductWindow + ProductPartition
  -> RawProductSpace
  -> {MbpCompression | ThcCompression}
  -> CompiledAuxiliaryBasis + PairVertex
  -> CoulombOperator
  -> response/GW consumer
```

The explicit MBP is not merely a validator for THC. It is a supported auxiliary
basis with a different compression model: channel-resolved radial overlap
diagonalization plus interstitial plane waves, versus real-space interpolation
and per-q ζ factors. Both must retain the uncompressed product-space manifest
needed to diagnose missing radial, angular, finite-q, spinor, or core-valence
channels. No one-particle recipe is permitted to define a private product basis.

### 1.4 Orthogonal tensor-execution factorization

Tensor execution is orthogonal to both the one-particle and pair-space
factorizations. Physical code operates on global tensors and declares index
expressions directly; a backend chooses storage layout, einsum or GEMM lowering,
device placement, processor mapping, and redistribution. Rank-2 factorization,
SVD, and Hermitian/generalized eigensolution are a separate matrix-linear-
algebra capability: local providers may use faer, while distributed providers
may use CTF's ScaLAPACK-backed `Matrix` operations directly or bridge CTF-owned
matrices to SLATE when SLATE's solver coverage or accelerator path is selected.
The execution world/context owns backend and resource state explicitly, so no
global device or implicit host transfer is allowed.

The distributed extension point is the global tensor plus its world, symmetry,
and placement intent. A CTF-like backend may distribute any legal combination
of $q$, point, auxiliary, band, energy, or matrix indices according to its own
contraction planner. Domain objects continue to see the same global canonical
axes, gauge, units, and results; they do not receive local shards. v0.3 does
not choose a process-grid decomposition or promise distributed performance.

## 2. Mathematical contracts

### 2.1 Channels and boundary data

```rust
struct SiteChannel {
    site: SiteId,
    angular: AngularChannel, // (l,m) now; spin-angular later
    head: HeadId,
}

#[non_exhaustive]
pub struct SphereRadii {
    augmentation: Option<f64>,
    screening: Option<f64>,
    potential: Option<f64>,
    density: Option<f64>,
}

pub struct RadialJet<T> {
    /// radial[k] = d^k f / dr^k at the sphere radius; radial[0] is the value.
    radial: Vec<T>, // nonempty
    energy_derivative: Option<Box<RadialJet<T>>>,
}
```

The first public `SphereRadii` API is constructed through
`SphereRadii::builder()` and exposes getters rather than requiring downstream
struct literals. Its four roles are independent; in particular, the
non-overlapping density hard sphere is not inferred from the augmentation,
screening, or potential radius. The `density` role currently has no production
consumer (the v&d scheme that motivated it was demoted to a test-only oracle
on 2026-08-24), but the role is well defined and near-zero-cost, so it ships
in the first public version; the test oracle and any future density-side
geometry consume the same field.

`RadialJet` is a method-neutral value type in `libmuffintin-core`; radial,
density, and field producers construct it without importing a matching backend.
The existing `libmuffintin-radial::BoundaryData` remains unchanged and is only a
radius-aware first-order adapter from a `RadialJet<f64>` containing at least its
zeroth- and first-order entries for APW value/slope matching. It does not wrap or
own a jet, so the existing matching path is unchanged.
All logarithmic derivatives, Wronskians, slopes, and matching coefficients are
derived from `RadialJet`; no backend re-evaluates radial derivatives by
finite differences.

The runtime representation and artifact schema are separate contracts.
`RadialJet<T>::radial` is fixed before M-M; `libmuffintin-io` does not directly
serialize the runtime type or extend Snapshot V1/V2. The first real persisted
consumer receives an independently versioned DTO with a nonempty finite
`radial_derivatives` array and the same order-by-index convention. The containing
field/channel schema declares the quantity and length units, so entry $k$ has
quantity unit times length unit to the power $-k$.

### 2.2 Envelope kernels

```rust
trait EnvelopeKernel {
    fn value(&self, l: u32, energy: f64, r: f64) -> Complex64;
    fn radial_derivative(&self, l: u32, energy: f64, r: f64) -> Complex64;
    fn energy_derivative(&self, l: u32, energy: f64, r: f64) -> Complex64;
    fn source_term(&self, l: u32, energy: f64, r: f64) -> Complex64;
}
```

Implementors:

- `HelmholtzKernel`: regular/irregular Bessel, Hankel, and Neumann pairs;
- `SmoothHankelKernel`: Gaussian-smoothed Helmholtz solution with its source
  normalization and analytic energy derivative, cross-checked by complex-step
  or high-order finite differences;
- `PlaneWaveEnvelope`: the existing v0.2 implementation.

The smooth-Hankel Gaussian source factor is unit-tested directly against the
Helmholtz identity. This prevents normalization or `exp(E r_s²/4)` errors from
being absorbed into screening coefficients.

### 2.3 Structure matrices

`StructureMatrix<E>` maps outgoing heads to regular tails in a declared
normalization and harmonic convention:

```rust
struct StructureMatrix {
    energy: f64,
    kpoint: KPoint,
    channels: ChannelMap,
    values: HermitianMatrix<Complex64>,
    convention: StructureConvention,
}
```

Required operations:

- real-space lattice sum and reciprocal/Ewald reference implementations;
- Bloch phase and reciprocal-folding contract shared with v0.2 THC;
- analytic or complex-step `∂E S(E)`;
- channel restriction/prolongation for downfolding;
- serialization keyed by geometry, radii, energy, `k`, harmonic convention,
  and kernel parameters.

No raw matrix may cross the public boundary without `StructureConvention`.

### 2.4 Screening

```rust
trait ScreeningTransform {
    fn apply(&self, bare: &StructureMatrix, boundary: &FreeBoundaryData)
        -> Result<ScreenedStructureMatrix>;
    fn inverse_map(&self) -> RepresentationMap;
}
```

Two implementations are required:

1. `AndersenScreening`: exact representation transformation parameterized by
   channel-resolved screening constants. It follows the SSW/third-generation
   MTO convention and exposes both forward and inverse maps.
2. `HardSphereScreening`: constructs the boundary matrix from regular and
   irregular free solutions at the hard-sphere radii and solves the equivalent
   screened system, conventionally of the form
   `(α⁻¹ - S⁰)⁻¹` up to the declared signs and normalizations.

These two must agree after conversion on a common Helmholtz test problem. The
test compares physical boundary residuals and reconstructed waves, not raw
matrix entries.

### 2.5 Kink and slope matrices

```rust
struct KinkMatrix {
    energy: f64,
    values: Matrix<Complex64>,
    derivative: Option<Matrix<Complex64>>,
    channel_map: ChannelMap,
}
```

The builder combines screened-wave slopes with potential-function boundary
data. Its invariants are:

- Hermiticity on the real energy axis;
- exact channel ordering and dimensional units;
- zeros/eigenvalue crossings invariant under valid screening transformations;
- stable derivative `∂E K(E)` for normalization, LMTO linearization, NMTO, and
  contour integration.

### 2.6 Energy representations

```rust
enum EnergyRepresentation {
    Exact,
    Linearized { centers: ChannelValues<f64>, order: u8 },
    IndependentHeads { energies: Vec<ChannelValues<f64>> },
    EnergyMesh { energies: Vec<f64>, order: usize },
}
```

- `IndependentHeads` reproduces the two-energy JPO construction: each head is
  screened at its own energy and the heads coexist in one basis.
- `transforms::nmto()` consumes `EnergyMesh` plus the full set of kink matrices
  and constructs energy-independent orbitals using divided differences/Lagrange
  matrices. Increasing the order changes interpolation accuracy and orbital
  range, not the active-orbital count.
- Downfolding is an operation on active/passive channel blocks of `K(E)`, with
  an explicit Schur-complement residual and reconstruction map.

### 2.7 Product and auxiliary-basis contracts

```rust
struct ProductWindow {
    left: OrbitalWindow,
    right: OrbitalWindow,
    transfer: KTransfer,
    include_core_valence: bool,
}

struct ProductPartition {
    spheres: NonOverlappingSphereSet,
    interstitial: InterstitialGeometry,
}

trait ProductEvaluator {
    fn raw_products(
        &self,
        window: &ProductWindow,
        partition: &ProductPartition,
    ) -> Result<RawProductSpace>;
}

struct CompiledAuxiliaryBasis { /* functions, duals, conventions, provenance */ }
struct PairVertex { /* pair-to-auxiliary map for each canonical q */ }
```

`ProductPartition` is explicitly not an alias for augmentation, screening, or
potential spheres. `MbpCompression` and `ThcCompression` consume the same raw
pair manifest and emit the same auxiliary capability. Consumers request
`CompiledAuxiliaryBasis + PairVertex`; they do not switch on an MPB/THC enum.

### 2.8 Coulomb kernel

```rust
pub struct CoulombRecipe {
    pub sphere: SpherePoisson,     // Weinert (existing MT two-region route)
    pub kernel: CoulombKernel,
}

/// V(r) = ∫ v(r,r'; BC) ρ(r') dr'. Mathematically the Green function of
/// -Δ/4π; named "kernel" because "Green function" in this library is the
/// electron propagator (contour GF, g(z) = K(z)⁻¹).
pub struct CoulombKernel {
    pub bc: CoulombBoundaryCondition,
    pub backend: CoulombBackend,   // #[non_exhaustive] Spectral | Dmk | SpectralEwald | Fmm
}

/// Per-axis periodicity plus open-direction environment; a boundary
/// condition is not a single dim integer.
pub struct CoulombBoundaryCondition {
    pub periodicity: [Axis; 3],           // P | O -> PPP / PPO / POO / OOO
    pub environment: OpenEnvironment,     // #[non_exhaustive] Vacuum | Dirichlet | MetallicGate
                                          //   | DielectricHalfSpace | TwoDielectricHalfSpaces
    pub zero_mode: ZeroModePolicy,        // G∥=0: Neutral | Dipolar | Charged | GateCompensated
}
```

Backend support per boundary condition is a capability matrix validated at
compile time, not a type hierarchy. One calculation holds exactly one Coulomb
BC object; Hartree, exact exchange, and any later RPA/GW/BSE kernels derive
from it — a workflow must never fall back to the 3D-periodic $4\pi/G^2$ kernel
at $W = v + vPW$ after the Hartree term already dropped the images.

v0.3 implements PPP × Spectral only (the FFT + Gaussian-compensation route
used by M-O/M-P); all other variants are constructible-not-executable. The
`Dmk` backend and open boundary conditions land with the v0.4 adaptive-grid
route. The pair-space consumer is unchanged: `libmuffintin-thc` sees only the
injected Coulomb gram $V_{\mu\nu} = \iint \zeta_\mu\, v\, \zeta_\nu$, so
backend and BC stay invisible behind the existing seam.

## 3. Crate and module layout

```text
crates/
  libmuffintin-core       v0.2 conventions, lattice and harmonic algebra;
                          method-neutral RadialJet value type
  libmuffintin-radial     v0.2 SR/Dirac radial solvers, RadialJet producers,
                          and unchanged BoundaryData APW adapter
  libmuffintin-sphere     v0.2 augmentation and full-potential sphere algebra
  libmuffintin-grid       v0.2 composite/double grids
  libmuffintin-io         v0.2 shared input/output and provenance
  libmuffintin-tensor     v0.2 M-Fb tensor values/views, semantic contractions,
                          execution contexts and backend adapters
  libmuffintin-dft        v0.2 density and potential closures
  libmuffintin-product    v0.2 raw product-space IR, ProductPartition,
                          CompiledAuxiliaryBasis and PairVertex contracts
  libmuffintin-mbp        v0.2 radial-product + interstitial-PW reference path
  libmuffintin-thc        v0.2 adaptive-grid ISDF/THC product factorization
  libmuffintin-coulomb    v0.2 Coulomb operators over auxiliary bases

  libmuffintin-envelope   EXTEND
    plane_wave/  existing v0.2 plane-wave implementation
    helmholtz/   Bessel/Hankel/Neumann envelope kernels
    smooth/      smooth-Hankel kernels and source normalization

  libmuffintin-scattering NEW
    geometry/    augmentation, screening, and potential sphere sets
    free/        Bessel/Hankel/Neumann and boundary pairs
    structure/   bare structure constants, lattice sums, cache
    screening/   Andersen and hard-sphere transforms
    boundary/    convention-tagged free and potential boundary data

  libmuffintin-basis      EXTEND
    spec/        BasisSpec, BasisBlock and capability validation
    augment/     linearized, multi-head and double augmentation
    compile/     CompiledBasis and heterogeneous block layout
    assembly/    cross-block overlap and Hamiltonian projections

  libmuffintin-operators  EXTEND
    generalized eigenproblems, rank-revealing overlap handling,
    slope/kink matrices, and contour Green functions over libmuffintin-tensor

  libmuffintin-product    EXTEND
    site-centered and heterogeneous APW/MTO pair evaluators, exact-energy
    residue/wave evaluators, and product partitions independent of potential
    and screening spheres

  libmuffintin-mbp        EXTEND
    MTO-MTO and APW-MTO radial/interstitial product channels through the same
    CompiledAuxiliaryBasis and PairVertex contracts as the LAPW reference

  libmuffintin-thc        EXTEND
    collocation of site-centered and heterogeneous compiled bases with no
    MTO-specific selector path

  libmuffintin-transforms NEW
    mesh interpolation, divided differences, active/passive downfolding,
    Wannier-like export

  libmuffintin-recipes    NEW
    validated basis presets and public workflow assembly;
    historical defaults, regression fixtures and provenance labels

  libmuffintin-basisopt   NEW, PUBLISHED LIBRARY CRATE (opt-in [[bin]],
                          not default-distributed)
    black-box evaluator protocol, parameter domains, feasibility filters,
    multi-fidelity objectives, derivative-free optimizers, cache, provenance

  libmuffintin-lapw       RETAINED FACADE
    stable LAPW API and preset backed by the public components above
```

Crate boundaries follow mathematical responsibilities, not historical method
names. `libmuffintin-scattering` is the new one-particle foundational layer;
`libmuffintin-product` is the corresponding pair-space boundary.
`libmuffintin-mbp`, `libmuffintin-thc`, and `libmuffintin-coulomb` never depend
on historical basis names. `libmuffintin-transforms` and
`libmuffintin-recipes` consume the scattering and product capabilities through
public interfaces. NMTO and EMTO are deliberately not crates: they are
respectively a transform and a workflow. `libmuffintin-lapw` is the exception
only as a stable facade and reference preset, not as the owner of generic
implementation.

`libmuffintin-basisopt` is a published companion crate outside the physical
libraries' dependency tree: no basis, radial, DFT, or workflow crate depends on
it. v0.3 publishes the crate with an opt-in `[[bin]]` target that is not part
of the default release artifacts; a caller either builds/installs that binary
explicitly or uses the public library protocol from its own Rust application
or experimental Rust/Python binding. The crate does not import
private basis types, radial solvers, or SCF state; public `BasisSpec` schemas,
transforms, and workflows expose evaluators through a small serialization-safe
protocol. The physical library therefore remains architecturally independent of
optimizer internals.

## 4. Anonymous basis API and capability checking

```rust
let spec = BasisSpec::builder(cell)
    .add(
        BasisBlock::plane_waves(cutoff)
            .augment(FullPotentialAugmentation::from_radial(apw_radial)),
    )
    .add(
        BasisBlock::site_centered(channels)
            .envelope(SmoothHankel::new(rsmh))
            .screening(HardSphereScreening::new(hcr))
            .energy(IndependentHeads::new([e1, e2]))
            .augment(FullPotentialAugmentation::from_radial(mto_radial)),
    )
    .build()?;

let basis: CompiledBasis = spec.compile(kmesh)?;
let operators = OperatorSet::assemble(&basis, &potential)?;
let bands = GeneralizedEigenproblem.solve(&operators)?;
let report: BasisDiagnostics = basis.diagnose()?;
```

The builder performs capability validation before expensive work. Examples of
rejected combinations:

- overlapping potential spheres with the v0.2 interstitial step function;
- `EnergyMesh` passed to the NMTO transform without `K(E)` and `∂E K(E)` support;
- contour Green functions with a kernel that cannot evaluate complex energy;
- heterogeneous block assembly without all required cross-block overlaps;
- double augmentation without the full-space and radial-grid quadratures;
- FCD requested without multipole reconstruction.

Concrete Rust enums are preferred for public component choices. Traits remain
at the numerical-kernel boundary. A reserved enum variant may be constructed in
v0.3 without being executable: capability validation returns a typed
unsupported/unavailable error before expensive work. This is a Rust spec
contract, not a parser or token-spelling contract; serde/TOML vocabulary is
deferred with the future binary interface. Capability validation operates on
what a block provides and consumes, never on whether it carries a LAPW, JPO,
PMT, NMTO, or EMTO label.

Every compiled object records a provenance manifest containing an optional
`preset_origin`, conventions, radii, energy mesh, channel map, screening
parameters, kernel parameters, radial-potential hash, and tolerances.

## 5. Reference presets, transforms, and workflows

### 5.1 Andersen/Ke screened-wave FP-TB-LMTO

This is the first vertical slice because it validates ordinary Helmholtz SSW,
screening, full-potential matrix elements, and double augmentation without the
additional smooth-Hankel source.

Pipeline:

```text
radial solutions
  -> ordinary free-wave structure constants
  -> Andersen/hard-sphere screening
  -> linearized augmented MTO
  -> double-grid/double augmentation
  -> H,S
  -> v0.2 full-potential SCF
```

The published Co/Ni/Fe/Si tests are reproduced before broadening the materials
set. This path is shipped as `recipes::fp_tb_lmto()`, not represented by an
`FpLmtoBasis` core type. Alternative FP-LMTO constructions remain ordinary
`BasisSpec` combinations and may later receive their own validated presets.
The compiled preset must also generate raw product channels and feed both the
v0.2 MBP and THC backends. Agreement is tested on retained product spans,
Coulomb spectra, and random pair actions, not only on one-particle bands.

### 5.2 Smooth-Hankel JPO and PMT

Pipeline:

```text
SmoothHankel(Eν, rsmh)
  -> bare structure matrix S0(Eν)
  -> hard-core screening at HCR
  -> centered head + screened J tails
  -> sphere augmentation + interstitial H,S
  -> optional union with AugmentedPW
  -> rank-revealing generalized eigensolver
```

The first target is exact reproduction of QuESTAAL's screened/unscreened
representation invariance on small systems. QuESTAAL PMT is represented by the
validated `recipes::questaal_pmt()` combination of an augmented plane-wave
block and a smooth-Hankel site-centered block; PMT is not a new basis-family
type. The second target is a parameter-surface study for PuB6 and UGa2.
Its product evaluator must include APW-APW, APW-MTO, and MTO-MTO pairs and feed
the same MPB/THC contracts. Product completeness is diagnosed separately from
the one-particle overlap condition number.

### 5.3 NMTO, EMTO, and full-potential unification — moved to v0.4

`transforms::nmto()`, `workflows::emto_sca_fcd()`, and the `ssw_fp_exact_e`
experiment are specified in `libmuffintin_v0.4_emto_nmto_plan.md` §2, together
with the adaptive-grid/DMK Coulomb route (§3 there) and auxiliary-center
geometry (§4 there).

## 6. Published `libmuffintin-basisopt` library crate

Optimization is an independent consumer of the public `BasisSpec`, transform,
and workflow APIs, not logic embedded in `libmuffintin-basis` and not a default
executable. The crate is published with v0.3, while invocation remains opt-in.
Stable reference presets run directly when their diagnostics remain inside the
validated basin. A caller may explicitly use the Rust library for guarded
optimization or offline searches that construct new robust presets.

### 6.1 Black-box contract

```rust
trait BasisEvaluator {
    type Parameters;
    type Fidelity;

    fn evaluate(
        &self,
        parameters: &Self::Parameters,
        fidelity: &Self::Fidelity,
    ) -> Result<EvaluationRecord>;
}

struct EvaluationRecord {
    feasibility: FeasibilityReport,
    objectives: ObjectiveVector,
    diagnostics: DiagnosticMap,
    artifacts: ArtifactRefs,
    provenance: Provenance,
}
```

The core evaluator contract has no serde bound. The optional cache, provenance,
and manifest layers use dedicated versioned DTO adapters when persistence is
actually required; they do not turn evaluator parameter types into a file-level
input contract.

The same protocol can later optimize LAPW local orbitals, product bases, THC
selectors, or mixed APW/MTO rank allocation. `basisopt` knows
nothing about what `E1`, `HCR`, or `rsmh` mean; the adapter supplies domains,
constraints, and diagnostic names.

### 6.2 Search variables for the MTO adapter

```text
continuous: Eν, HCR, screening constants α, smooth radius rsmh,
            potential-sphere radii, contour/mesh placement
discrete:   active channels, lmax, number of heads, NMTO order,
            downfolded channels, APW/MTO rank split
```

Physical ordering, radius bounds, channel continuity, and core exclusion are
encoded as domains rather than penalty terms.

### 6.3 Multi-fidelity evaluation

1. **Feasibility:** boundary residuals, screening-matrix conditioning,
   structure-matrix decay, positive sphere radii, and channel completeness.
2. **Frozen potential:** minimum eigenvalue and condition number of `S`, ghost
   detection, band-window error against the converged v0.2 LAPW reference,
   representation invariance, and basis size/range.
3. **Short SCF:** total/free-energy drift, charge residual, moment stability,
   and iteration count.
4. **Production confirmation:** converged observable differences and transfer
   to neighboring volume, magnetic state, and k mesh.

Hard constraints are applied before scalar ranking:

```text
S positive definite
no ghost crossing in target window
boundary and Hermiticity residuals below tolerance
screened/unscreened observables invariant within tolerance
```

The default objective is Pareto-valued rather than one magic score:

```text
(band error, log condition number, orbital range, basis size, SCF cost)
```

Algorithms in v0.3 are derivative-free:

- deterministic Sobol or Latin-hypercube exploration;
- coordinate/pattern search for local basin mapping;
- derivative-free trust-region search;
- optional Bayesian optimization or CMA-ES adapters.

A deterministic Sobol scan plus local refinement remains the normative
reproducibility path. Gradient descent, differentiable eigensolvers, implicit
SCF differentiation, adjoints, and automatic differentiation are explicitly
deferred beyond v0.3.

### 6.4 Trigger policy

Optimization is not repeated every SCF iteration. A specification is optimized
on the initial or frozen converged potential and reused while diagnostics stay
inside an admissible basin. Re-optimization is triggered only by:

- loss of positive definiteness or a condition-number threshold crossing;
- a ghost or band-window residual crossing;
- a large change in radial logarithmic derivatives;
- a volume, chemical, or magnetic-state change outside the specification's
  validated transfer envelope.

Every accepted optimum includes a local sensitivity report, so a broad robust
basin is preferred over a marginally better sharp optimum.

### 6.5 Invocation policy

Three modes are public library policies selected by the caller:

1. `direct` (default): use a validated preset and only emit diagnostics;
2. `guarded`: invoke `basisopt` when a risk monitor trips;
3. `offline`: deliberately map a family of structures and publish a new preset.

No library workflow silently launches an optimization. Guarded mode has explicit
trial, wall-time, and fidelity budgets and falls back to the last validated
preset or user specification if no feasible improvement is found.

## 7. Milestones

Every milestone below uses the M-Fb tensor substrate for dense contractions.
RSTSR is the default acceptance path; the tenferro-rs parity suite is rerun when
a milestone adds a new semantic contraction. Backend parity is numerical and
shape/layout based, not bitwise. Distributed execution is deferred, but new
operations must preserve global tensor rank, index expressions, symmetry and
automatic placement intent so a CTF planner can select its own processor mapping instead
of inheriting a decomposition hard-coded by a method crate.

### M-M — conventions and free scattering

- write doc/18 (adopted axes, capability matrices, Coulomb kernel BC) before
  implementation starts; it is the normative reference for every reserved
  variant introduced here;
- introduce non-exhaustive, builder-constructed `SphereRadii` with its density
  role, method-neutral arbitrary-order `RadialJet`, `ChannelMap`, and
  `StructureConvention`;
- retain `BoundaryData` as the unchanged first-order APW matching adapter;
- implement ordinary free-wave boundary pairs at real and complex energy;
- two independent structure-constant evaluators;
- Green-identity acceptance $\langle\psi(\epsilon_1)|\psi(\epsilon_2)\rangle_I
  = a\,S_{12}$ as a free cross-check of structure constants plus screening,
  with only the minimal divided-difference helper it needs (the full NMTO
  divided-difference machinery stays in v0.4);
- free-electron and empty-lattice kink tests;
- convention round trips and dimensional-analysis tests.

### M-N — screening equivalence

- Andersen representation transformation;
- hard-sphere boundary screening;
- forward/inverse reconstruction of the physical wave;
- screening invariance of kink zeros and logarithmic derivatives;
- diagnostics group recorded in versioned artifacts (degradation explicit, as
  in the v0.2 mixing-degradation precedent): interstitial homogeneous
  eigenvalue estimate $\varepsilon_{\mathrm{hom}}$, screening/structure matrix
  condition numbers, real-space localization/decay report;
- optional and non-blocking: a fixed-parameter Nohara value-and-derivative
  oracle (constant density in the bcc/diamond interstitial at the published
  $N_R$, $l_{\max}$, and energy meshes) as an independent cross-check of
  multi-energy screened structure constants, hard-sphere screening, and
  `RadialJet` boundary data. It lives in the test layer, is not public API,
  and doubles as one of the few runnable public implementations of that
  algorithm.

### M-O — FP-TB-LMTO preset

- linearized energy representation;
- Ke-style double augmentation using v0.2 grids and sphere algebra;
- both one-center expansions of the smooth envelope inside spheres: numerical
  radial projection (v0.2 grids and sphere algebra) and Questaal-style
  $P_{kL}$ analytics, cross-checked against each other;
- smooth-part kinetic energy by spectral differentiation on the uniform grid
  (M-P validates the alternative source-identity route, so the two lines
  cross-check); sphere kinetic stays radial/augmented;
- smooth Poisson through the §2.8 kernel contract at PPP × Spectral with
  Gaussian compensation charges;
- compile the named preset and its equivalent anonymous `BasisSpec` to the same
  canonical form and provenance;
- full H,S and SCF through public basis/operator APIs;
- raw MTO product channels through `libmuffintin-product`, followed by MPB and
  THC construction with matched Coulomb/action diagnostics;
- Co, Ni, Fe, and Si regression against the published workflow and v0.2 LAPW.

### M-P — smooth-Hankel JPO and QuESTAAL PMT presets

- verified smooth-Hankel kernel and source identity;
- one- and two-energy independent heads;
- hard-core-screened structure matrices;
- interstitial and augmentation blocks, including F–F, F–J, and J–J terms;
- mixed APW+MTO assembly with rank-revealing overlap handling;
- APW-APW, APW-MTO, and MTO-MTO pair channels through the public product IR;
- MPB and THC auxiliary representations with separately reported one-particle
  and product-space conditioning;
- record ISDF/THC rank against the LAPW baseline: localized MTO pair densities
  are expected to compress at least as well as plane-wave products (a testable
  expectation, not an assertion);
- `recipes::questaal_jpo()` and `recipes::questaal_pmt()` compile to anonymous
  blocks with no privileged assembly path;
- QuESTAAL small-system representation-invariance regressions.

### M-Q — optimizer and difficult-material map

- published `libmuffintin-basisopt` protocol, cache, and provenance;
- public-basis adapter with component parameter schemas and feasibility filters;
- Sobol/local baseline plus optional Bayesian/trust-region adapters;
- sensitivity maps for ordinary metals and semiconductors;
- PuB6 and UGa2 robustness maps in `(E1,E2,HCR,rsmh)`;
- freeze default recipes only after basin-width, not single-point, validation.

### M-R, M-S, M-T — moved to v0.4

The NMTO transform, canonical EMTO workflow, and full-potential unification
milestones are specified in `libmuffintin_v0.4_emto_nmto_plan.md` §5, which
also adds M-U (adaptive-grid Coulomb route). v0.3 ends at M-Q; the v0.3 API is
frozen at the end of M-Q without waiting for the v0.4 unification verdict.

## 8. Validation matrix

| Invariant | Unit/synthetic test | Cross-code test |
|---|---|---|
| tensor semantics/backend parity | analytic/oracle fixtures vs RSTSR and tenferro global-tensor contractions | optional CTF world/contract probe |
| free-wave normalization and Wronskian | analytic Bessel identities | none |
| structure constants | real vs reciprocal lattice sums | QuESTAAL/EMTO tabulation where available |
| screening representation invariance | reconstructed boundary wave and kink zeros | QuESTAAL screened vs unscreened |
| smooth-Hankel source | direct Helmholtz residual on radial grids | QuESTAAL radial samples |
| augmentation | boundary continuity and independent composite-grid integration | FP-TB-LMTO paper/code; v0.2 LAPW |
| product generation | raw counts, Gaunt selection rules, radial overlap spectra | v0.2 SPEX-compatible LAPW MPB |
| auxiliary product span | MPB/THC principal angles, Coulomb spectra and random actions | LAPW baseline, then FP-TB-LMTO and PMT |
| overlap health | positive spectrum, rank, condition number | LAPW band-window comparison |

The NMTO-order, EMTO-Green-function, and FCD rows moved to the v0.4 plan §6.

All cross-family comparisons must use matched radial potentials, sphere radii,
angular cutoffs, relativity, XC, k mesh, and occupations. A band comparison
with unmatched potentials is a workflow smoke test, not a basis validation.

## 9. Numerical failure policy

Every workflow returns structured diagnostics rather than printing warnings:

```rust
struct BasisDiagnostics {
    boundary_residual: f64,
    hermiticity_residual: f64,
    overlap_min_eigenvalue: f64,
    overlap_condition: f64,
    screening_condition: f64,
    representation_delta: ObservableDelta,
    ghost_candidates: Vec<GhostCandidate>,
    real_space_decay: DecayProfile,
    sensitivity: Option<SensitivityReport>,
}
```

Default actions:

- loss of positive definiteness: fail, never silently discard vectors;
- near-linear dependence: report pivoted rank and offer an explicit reduction;
- screening solve near singular: fail with the offending channel/eigenvector;
- ghost candidate: bracket and identify its dominant channels;
- convention mismatch: type or validation error, never conversion by guess;
- optimizer failure: retain the last validated preset or user specification,
  not the last trial.

## 10. Main risks

1. **Convention disease.** MTO traditions differ in signs, radial scaling,
   spherical harmonics, and what is called a structure/slope/kink matrix.
   Tagged conventions and physical reconstruction tests are mandatory.
2. **False unification of geometries.** EMTO overlap, JPO hard cores, and LAPW
   augmentation spheres cannot share one spatial partition.
3. **Confusing energy heads with NMTO order.** Independent JPO heads increase
   basis rank; NMTO order does not. The API and docs must keep this visible.
4. **Full-potential naming inflation.** FCD accuracy and a full nonspherical
   variational Hamiltonian are not the same claim.
5. **Optimization overfitting.** A sharp optimum at one volume/k mesh is less
   useful than a slightly worse robust basin.
6. **Optimizer coupling.** If physical code starts depending on a particular
   search algorithm or optimizer state, the component boundary has failed.
7. **Relativistic channel growth.** Scalar `(l,m)` assumptions must stay out of
   the core scattering matrix layout.
8. **Product-basis mismatch.** MTO optimization for one-particle bands may be
   poor for GW products. v0.2 MPB and THC diagnostics must therefore be
   optionally included in the final objective through the common product-space
   interface, but not before the one-particle basis is stable. A THC residual
   alone is not allowed to replace the explicit MPB baseline.

## 11. Documentation and examples

Proposed derivation notes:

```text
doc/18_mto_density_poisson_kinetic_axes.md
doc/19_scattering_conventions_and_structure_matrices.md
doc/20_screening_representations.md
doc/21_smooth_hankel_jpo_and_pmt_presets.md
doc/22_basisopt_protocol_and_basis_adapters.md
```

doc/18 records the adopted representation/poisson/kinetic axes, the capability
matrices, and the Coulomb kernel BC contract; it is written before M-M starts.
v0.4 continues the numbered sequence with doc/23–26 (see the v0.4 plan §7).

Examples are expressed twice: a compact preset and an equivalent explicit,
anonymous specification. CI checks that both compile to the same canonical
specification and provenance manifest.

```rust
let a = recipes::questaal_jpo(system, JpoDefaults::robust())?;

let b = BasisSpec::builder(system)
    // all components shown explicitly in the example
    .build()?;

assert_eq!(a.canonical_spec(), b.canonical_spec());
```

This is the decisive architectural test: presets and workflows are conveniences
assembled from components, not privileged implementations. The same test covers
`recipes::fp_tb_lmto()` and `recipes::questaal_pmt()`.

## 12. Definition of done for v0.3

v0.3 is complete when:

1. ordinary, plane-wave, and smooth-Hankel blocks compile through one anonymous
   basis/augmentation API;
2. Andersen and hard-sphere screening reconstruct the same physical solution
   in their common domain;
3. `recipes::fp_tb_lmto()`, `recipes::questaal_jpo()`, and
   `recipes::questaal_pmt()` have explicit `BasisSpec` equivalents and agree
   with LAPW/reference results at frozen tolerances;
4. NMTO and EMTO acceptance criteria moved to the v0.4 plan §8;
6. the published `libmuffintin-basisopt` crate finds a robust admissible basin
   and reproduces it from a stored manifest for PuB6 and UGa2, while validated
   presets still execute directly without invoking an optimizer;
7. FP-TB-LMTO and PMT compiled bases feed `libmuffintin-product`,
   `libmuffintin-mbp`, `libmuffintin-thc`, and `libmuffintin-coulomb` through
   the same public contracts as LAPW; MPB/THC span and Coulomb/action
   comparisons pass at independently converged tolerances;
8. `libmuffintin-lapw` remains a supported facade and all M-F LAPW regressions
   pass after generic implementation is moved into method-neutral crates;
9. no workflow owns a private radial solver, screening routine, augmentation,
   product-generation, auxiliary-basis, or matrix-assembly path;
10. historical method names are absent from generic basis enums and crate
   boundaries, except for the explicitly retained LAPW facade;
11. every basis and auxiliary-basis result carries conventions, parameters,
   pair windows, diagnostics, and
   provenance sufficient for exact reconstruction;
12. every new dense contraction passes analytic/oracle fixtures and the
    RSTSR/tenferro semantic parity suite, while public schemas and workflow
    manifests remain independent of backend tensor types;
13. global tensor rank, index expressions, symmetry and automatic placement intent survive
    every basis/product/scattering path, allowing a future CTF adapter to retain
    CTF-owned processor mapping and redistribution; v0.3 makes no unverified
    distributed-performance claim.

## 13. Primary references

- A. Zhang et al., “Full-potential screened spherical wave based muffin-tin
  orbital method for density functional theory,” Phys. Rev. B 110, 155126
  (2024), https://doi.org/10.1103/PhysRevB.110.155126
- A. Zhang et al., “Implementation of full-potential screened spherical wave
  based muffin-tin orbital for all-electron density functional theory,”
  arXiv:2503.07524, https://arxiv.org/abs/2503.07524
- O. K. Andersen, T. Saha-Dasgupta, and S. Ezhov, “Third-generation
  muffin-tin orbitals,” arXiv:cond-mat/0203083,
  https://arxiv.org/abs/cond-mat/0203083
- D. Pashov et al., “Questaal: A package of electronic structure methods based
  on the linear muffin-tin orbital technique,” Comput. Phys. Commun. 249,
  107065 (2020), https://doi.org/10.1016/j.cpc.2019.107065
