# **libmuffintin v0.2 plan — adaptive-grid ISDF/THC route**

- Workstream ID: `v0.2-isdf-thc`
- Plan version: 1
- Approval: accepted (M-A to M-Kc closed on `main`; M-L open)
- Supersedes: none
- Imported: 2026-09-08 from `scratch/libmuffintin_v0.2_plan.md` (last modified 2026-08-27), body unchanged
- Status source: false; state lives in `STATUS.md` and `ledger.md`
- Path map: the companion evidence named below as sibling files is under `evidence/2026-08-20-thc-scratch-fixtures/`; `isdf_adaptive_2510.20826.pdf` is arXiv:2510.20826 and is not tracked

Status: revised implementation roadmap (2026-08-24).  M-K is
`IMPLEMENTATION_CLOSED`; independent cross-code/material acceptance remains
open and no production-validation or release claim follows from that closure.

Implementation baseline: repository `main` has completed the M-A through
M-Kc implementation contracts, including the tensor foundation, anonymous
basis/operator route, auxiliary/THC/Coulomb layers, SRA spinor substrate,
minimal DFT workflow, and orbital-configuration V2.  External acceptance is a
separate open gate: internal regressions do not replace frozen independent
radial/band fixtures or material comparisons.  M-L therefore starts from the
closed M-K implementation interfaces without inheriting a production claim.

Companion evidence in this directory: `thc_mt_kpoint_test.py`,
`thc_mt_kpoint_results.txt`, `thc_lapw_end_to_end_test.py`,
`thc_lapw_end_to_end_results.txt`, `thc_lapw_coqui_threshold_test.py`,
`thc_lapw_coqui_threshold_results.txt`, `grid_budget_sm_dy.py`,
`isdf_adaptive_2510.20826.pdf`.

## **0. Scope decisions (frozen for v0.2)**

1. **FLEUR converter: FROZEN.**  No FLEUR I/O work in v0.2.  Validation is
   done by comparing numbers directly against SPEX
   (`/Users/zerozaki07/Documents/dft_codes/spex06.00pre36`).  The SPEX LAPW
   basis is 1:1 compatible with FLEUR (same conventions, different parameter
   names), so SPEX is the single reference for radial functions, mixed
   product basis, and Coulomb matrix data.

2. **Minimal DFT goes in.**  A self-contained LAPW SCF is needed anyway to
   produce orbitals without an external code.  Write it against the SPEX
   implementation: the LDA parametrization SPEX uses internally (identify
   the exact routine in the SPEX source at implementation time) plus GGA
   (PBE, SPEX/FLEUR conventions).  Two deliberate additions beyond SPEX:

   Fermi–Dirac smearing for occupations (SPEX does not have it) and a
   selectable mixer family (linear / Broyden / Pulay–Anderson).
   xc-functional generality beyond LDA+GGA is out of scope.

3. **Two THC/ISDF grid paths, one algorithm.**

   - **adaptive path** (production): MT exponential-radial shells × angular
     points + coarse uniform interstitial grid;

   - **uniform FFT path** (debugging only): plain N³ grid, kept because (a) on
     smooth/pseudized toy functions both paths must agree to tolerance,
     (b) it is apples-to-apples with CoQuí's existing FFT-grid THC.

4. **No DMK.**  The Coulomb metric for the ISDF auxiliary functions goes
   through the muffin-tin analytic route (Y_LM projection + multipoles +

   Weinert pseudo-charge + interstitial plane waves).  DMK-class black-box
   solvers are only required for unstructured bases (GTO/NAO); the MT
   two-region geometry makes them unnecessary.  (Periodic k-point DMK does
   not exist yet anyway.)

5. **Relativity / SOC policy.**  Core states are always solved with the
   existing 4c Dirac solver (`mt-radial::solve_core_dirac`) on the
   spherical SCF potential.  **Default strategy: as soon as SOC and/or
   noncollinear magnetism enters, everything inside the MT spheres is
   4c Dirac** — M-Ka activates the currently reserved
   `solve_valence_dirac` path (κ-resolved, large+small components), adds
   its first and second energy derivatives, confined SRA HDLOs, and the
   missing typed spinor
   augmentation and (H,S) assembly, promoting
   valence 4c from "reserved interface" to the active path.

**The 4c treatment is MT-only**: the interstitial plane-wave part stays
   2-component Pauli spinors with no small component, exactly as in SPEX —
   the SOC operator and the small component are confined to the spheres,
   and SRA matching at R_MT uses the large-component value and radial
   derivative only.  Variational
   structure on top of that:

   - **no SOC, no magnetism / collinear magnetism**: scalar-relativistic

     Koelling–Harmon, 1-component per spin channel;

   - **nonmagnetic + SOC**: **second variation** in the reduced
     first-variation basis (the one permitted cheap path), with matrix
     elements built from the Dirac-derived radial functions — this is not
     an open question: SPEX does exactly this (see the source evidence in

     §1);

   - **magnetic + SOC / noncollinear**: **first variation only** — full 4c
     spinor Hamiltonian in a single variational step, following SPEX.  No
     second-variation shortcut here, ever.

6. **Crystal symmetry via** `moyo` **only** (pure-Rust symmetry finder from
   the spglib org).  No spglib C bindings, not even as a fallback.

   Details in §2.

7. **Architecture doctrine: LAPW facade, anonymous basis core.**
   `libmuffintin-lapw` remains because LAPW is an established method name and
   the current M-F implementation is its normative regression.  It becomes a
   facade/reference preset over a method-neutral `BasisSpec`, `BasisBlock`,
   augmentation, operator assembly, and eigensolver API.  Historical method
   names provide defaults, tests, and provenance; they are not type or crate
   boundaries.  No MTO basis is implemented in v0.2, but nothing in v0.2 may
   block heterogeneous APW/MTO blocks in one `BasisSpec`.  Details in §2b.

8. **No Gaussian basis in v0.2.**  A future `GaussianEnvelope` is not
   forbidden by the interfaces, but GTO/libcint/X2C/DKH work is outside the
   muffin-tin-focused v0.2 and v0.3 critical path.

9. **The LAPW mixed product basis is a first-class v0.2 reference path.**
   Product-space construction is not deferred to post-v0.2 and is not owned by
   `libmuffintin-lapw`.  v0.2 ships a method-neutral product-space IR plus a
   SPEX-compatible LAPW mixed product basis (MPB).  MPB and ISDF/THC are sibling
   factorizations of the same orbital-pair space and must expose common
   auxiliary-basis, pair-vertex, metric, convention, and provenance contracts.
   `libmuffintin-coulomb` consumes those contracts rather than a THC-specific
   representation.  The explicit MPB path is the reference needed to decide
   which radial products, angular channels, interstitial plane waves, and
   compression tolerances are physically adequate before relying on THC.

10. **Tensor execution is backend-neutral before M-G.**  M-Fb moves the
   current dense contractions behind a small, physics-bearing tensor contract
   before package extraction or new product-space work begins.  RSTSR is the
   default local backend; tenferro-rs is an optional parity backend.  Neither
   backend type may appear in basis, radial, snapshot, or exported-result
   schemas.  The contract fixes axis roles, conjugation, units, gauge,
   Hermiticity, explicit host/device movement, and deterministic ordering.

   M-Fb covers the existing LAPW $P^\dagger B P$, overlap whitening
   $X^\dagger H X$, and residual contractions, then supplies the same tensor
   expression model to MPB/THC/Coulomb work.  Its first-class objects are a
   global tensor, index expression, and execution world/context.  Global rank,
   shape, index labels and symmetry are preserved; `Placement::Auto` is the
   normative policy and the backend owns decomposition, redistribution and
   contraction planning.  No Rust/domain API describes local shards.

   RSTSR and tenferro-rs initially implement a one-process world.  A future CTF
   binding constructs a global `CTF::Tensor` in a `CTF::World` and delegates
   automatic rank mapping to CTF rather than receiving manually sharded local
   tiles from libmuffintin.  v0.2 does not require MPI or production CTF, but
   its tensor model must not erase the information CTF needs.

   Distributed tensor contraction and distributed rank-2 linear algebra are
   separate capabilities, but one provider may implement both.  CTF is the
   target for global tensor expressions and automatic processor mapping, and
   its `Matrix` layer may satisfy the matrix capability through CTF's existing
   ScaLAPACK-backed Cholesky, solve, QR, SVD, and Hermitian eigensolver
   interfaces.  SLATE is an optional modern alternative for dense distributed
   matrix factorization/eigensolution, not a mandatory CTF companion.  An
   explicit descriptor/redistribution bridge is required only when SLATE is
   selected.

## **1. Evidence base (what this plan is built on)**

- Paper: Zhu, Yeh, Morales, Greengard, Jiang, Kaye, *ISDF on adaptive real
  space grids*, arXiv:2510.20826 (JCTC 2026).  Key results used here:

  ISDF = ID of the pair-density collocation matrix (App. A, pivoted

  Cholesky of the Gram); Theorem 1 / Remark 3: a grid resolving the

**single-particle** functions with upsampling ×1.5–2 resolves all pair
  densities; α = N_μ/N_orb ≈ 8 (pseudo) → ≈ 16 (all-electron, full-ERI
  chemical accuracy), insensitive to basis locality; periodic k-point
  support is explicitly future work.

- k-point structure: Yeh & Morales (JCTC 19, 6197 (2023)) — one
  q-independent set of interpolation points, per-q interpolation vectors

  ζ^q_μ.  Upstream CoQuí implements this on FFT grids by selecting the shared
  points from the q=0 metric and then evaluating all q.  CoQuí's
  `interpolating_basis_nonuniform_rgrid` path exists but is **Γ-only**
  (`thc_aux.icc`: "non gamma-point is not implemented yet") — the k-point ×
  adaptive-grid combination is the open gap this plan targets.

- Experimental CoQuí fork `mmorale3/coqui:uspp-paw-isdf`, commit `caef773`,
  adds three selection-only controls: energy-pair weighting, a filtered-orbital
  surrogate, and a two-pass Coulomb-metric re-ranking.  The last first builds
  an overcomplete L2 candidate pool, then ranks the pool with a bare or
  attenuated Coulomb metric whose kernel is averaged over the full q mesh.

  It still calls the first pass with `iq=0`; it is therefore a q-aware metric
  re-ranking of a q=0-generated pool, not a full all-q pair-matrix ID.

- The separate `mmorale3/coqui:paw-ac-updated` line keeps the global smooth
  interpolation-point call at `iq=0`, but commit `44c79e9` replaces q=0-only

  PAW augmentation-channel truncation by
  `max_q sum_G 4pi/(Omega|q+G|^2)|eta_IJ(q+G)|^2`, normalized relative to the
  largest channel.  It documents and tests the failure mode relevant to LAPW:
  channels without L=0 weight can vanish at q=0 yet matter at finite q.  This
  is local augmentation-channel selection, not a replacement global real-space
  point algorithm, but its all-q criterion is adopted below for MT channels.

- Corrected scratch point-selection test (`thc_mt_kpoint_test.py`, 2×2×2
  mesh, Z=20 core-like orbital) includes the reciprocal-lattice phase when
  folding k-q.  In this deliberately non-overlapping atomic limit the pair
  space reaches machine precision by N_mu≈24; the earlier apparent high-q
  rank was an Umklapp-gauge artifact.  The valid result is narrower: adaptive
  quadrature is stable at 6×10⁻⁴–9×10⁻⁴ while uniform-grid quadrature changes
  catastrophically with origin/half/random shifts.  This supports the grid
  choice, not a periodic-LAPW compression factor.

- Synthetic two-region LAPW/APW+lo end-to-end test
  (`thc_lapw_end_to_end_test.py`, two atoms, s/p/d/f augmentation, 2×2×1
  k/q mesh) constructs shared points, per-q zeta and the finite-cutoff periodic

  Coulomb metric entirely on each candidate grid.  Against an independent
  composite-grid reference (2.5×10⁻² reference-grid change), the refined
  adaptive grid gives pair-Fourier 2.36×10⁻², ERI Frobenius 4.93×10⁻²,
  max-element 4.56×10⁻² and random-action 6.23×10⁻².  The 16³ uniform grid
  (4096 points, matched to the 4041-point coarse adaptive grid rather than the
  refined grid) misses the sharp LO and has O(1) ERI/action error.

- Grid budgets (`grid_budget_sm_dy.py`, PseudoDojo `pseudodojo_experiments`
  4f-in-valence hints): Sm-4f ONCV normal = 61 Ha (ρ=488 Ry) → 78k FFT
  pts/atom; Dy-4f = 68 Ha → 87k; MT-adaptive tight ≈ 56–57k pts/atom
  (0.65–0.73× the ONCV grid); AE-on-uniform-FFT would need ~10⁸–10⁹.

- CoQuí accuracy conventions: quickstart GW uses `thresh=1e-3`,
  `ecut=1.2×ecutwfc` (Si 5×5×5, 20 bands → N_μ=262 ≈ 13×nbnd); API default
  `thresh=1e-5`; guidance: loosest thresh that converges the observable.

- Direct CoQuí-semantics scan (`thc_lapw_coqui_threshold_test.py`) uses its
  q=0-only point selection, orbital normalization
  `(Na·Nb·Ns·Nk·Npol)^(-1/4)`, and absolute (not initial-relative) maximum

  Cholesky-residual threshold.  On the refined adaptive synthetic LAPW grid,
  `thresh=1e-3` selects only N_mu=11 (alpha=1.83) from the six-band window and
  leaves 91% valence-exchange / 73% occupied-action error; `1e-8` selects 76
  (alpha=12.67) and reduces those to 1.7% / 6.1%.  Restricting both the THC
  metric and ERI to the three smooth valence bands does not rescue the loose
  threshold (`1e-3`: 66% / 60%); `1e-8` gives N_mu=22 (alpha=7.33), 0.16% /
  1.7%.  This reproduces q=0 CoQuí semantics as a compatibility baseline; it
  does not justify making q=0 the LAPW production selector.  Tutorial threshold
  values are not portable from pseudopotential Si to a synthetic all-electron

  LAPW window: acceptance is observable convergence, not `thresh` itself.

- SPEX mixed product basis reference: `src/mixedbasis.f` — radial products
  on the exponential mesh, overlap diagonalization, eigenvalue cutoff `TOL`
  (default 1e-4, line 106; diagonalization around lines 447–463).

- SPEX relativity/SOC architecture (verified in source; this is the
  normative reference for §0.5 and `mt-dft`):

  - `src/dirac.f`: ONE radial solver family; `dirac_hom` solves the
    scalar-relativistic equations and, for l<0 (κ-encoded), the full
    relativistic Dirac equations — SR and 4c are the same backend;

  - `src/iterate.f`: SPEX's **internal DFT SCF** (ITERATE mode) — the
    line-by-line reference for our `mt-dft`.  Valence basis functions
    `bas1/bas2` (large/small components) come from `dirac_hom_x`
    (lines 1061, 2502); core states are always SOC-split Dirac
    (`lcore_soc` forced true, `nindxc` doubled for l≥1, lines 181–186);

    SOC in first variation is the default path
    (`iterate_inv_soc`, line 45);

  - `src/getinput.f:3053–3055`: `SECVAR` requires SOC on, and
    `nspin==2 → Error('Second variation (SECVAR) not implemented for
    magnetic systems.')` — the §0.5 policy is literally SPEX's input
    validation.

## **2. Crate layout**

Current M-F package names remain valid source references.  M-Fb first creates
`mt-tensor` without renaming the existing packages.  M-G then performs the
public-package rename and extracts generic responsibilities from `mt-lapw`:

```
crates/
  libmuffintin-core       renamed from mt-core; conventions and primitives
  libmuffintin-radial     renamed from mt-radial; SR/4c radial engines
  libmuffintin-sphere     renamed from mt-sphere; sphere fields/algebra
  libmuffintin-grid       renamed from mt-grid; canonical quadrature grids
  libmuffintin-io         renamed from mt-io; versioned artifacts
  libmuffintin-tensor     renamed from M-Fb mt-tensor; backend-neutral local
                          tensor contractions and execution contexts
  libmuffintin-envelope   NEW: PlaneWaveEnvelope in v0.2; future envelopes
  libmuffintin-basis      NEW: BasisSpec/BasisBlock, layouts, augmentation maps
  libmuffintin-operators  NEW: generic H/S containers, assembly and eigensolver
  libmuffintin-recipes    NEW: validated presets, defaults and provenance
  libmuffintin-lapw       retained facade/reference preset over those components
  libmuffintin-product    NEW: product-space IR, partitions, pair vertices,
                          and common auxiliary-basis contracts
  libmuffintin-mbp        NEW: radial-product + interstitial-PW mixed product
                          basis; SPEX-compatible reference construction
  libmuffintin-thc        NEW: k-point ISDF/THC alternative product factorization
  libmuffintin-coulomb    NEW: Weinert Coulomb operators/metrics over any
                          compiled auxiliary basis
  libmuffintin-dft        NEW: minimal LDA/GGA SCF, basis-agnostic driver
  libmuffintin-sym        NEW: thin moyo wrapper
```

The repository may keep the existing directory names during the mechanical
transition, but published Cargo package names use the `libmuffintin-*` prefix.
The umbrella `libmuffintin` crate re-exports the stable public surface.

### **libmuffintin-sphere (implemented by M-F; extended in v0.2)**

- Motivation: the universal radial primitive already exists
  (`mt-radial::radial_integral` over the `RadialComponents` trait, kernels
  overlap / r^n / sampled / `PotentialMultipole`), but the angular algebra
  is left to the caller — `PotentialMultipole` only **validates** (L,M).

  Promote the full composition into a library primitive:

  ⟨i l₁m₁ | V | j l₂m₂⟩_MT

    = Σ_LM  G^{LM}_{l₁m₁,l₂m₂} · ∫ dr [p_i p_j + Q_i Q_j] V_LM(r)

- Type sketch:

  ```text
  SphereField    channels[(L,M)] -> radial samples     (V, n, ζ-projections)
  SphereOrbital  (l,m) + RadialComponents              (u, u̇, LO, 4c)
  matrix_element(left: &SphereOrbital, field: &SphereField,
                 right: &SphereOrbital) -> f64/Complex
  ```

- Complex and real Gaunt paths both supported (already in `mt-core`) — no
  harmonic-convention lock-in.

- This is on the v0.2 critical path, not MTO groundwork done early:
  consumers are `mt-dft` (full-potential sphere (H,S) blocks, density
  synthesis, vxc — n(r) and V(r) are `SphereField`s), `mt-coulomb`
  (ζ Y_LM projection + multipole moments), and the M-L MPB/THC comparison.
  The same primitives feed the v0.2 `libmuffintin-product` and
  `libmuffintin-mbp` crates; they are not deferred MTO groundwork.

### **libmuffintin-product**

- Owns the method-neutral product-space IR.  Its input is a compiled
  one-particle basis capability, never a concrete `LapwBasis`; its public
  objects include `ProductPartition`, `PairChannel`, `RawProductSpace`,
  `CompiledAuxiliaryBasis`, and `PairVertex`.

- `ProductPartition` is independent of augmentation, screening, and potential
  sphere sets.  v0.2 implements the non-overlapping MT-sphere + interstitial
  partition required by LAPW MPB.  This separation later permits an MTO or EMTO
  one-particle workflow to project products onto a suitable auxiliary partition
  without pretending that overlapping potential spheres are LAPW spheres.

- Pair metadata includes `q`, the canonical Umklapp gauge, spinor/component
  structure, band/radial windows, and valence–valence, core–valence, or
  core–core provenance.  Raw sphere products retain their radial functions and
  coupled $(L,M)$ channels before any spectral cutoff; interstitial products
  retain their reciprocal support before any plane-wave cutoff.

- The crate defines representation-neutral evaluation and projection
  contracts used by both MPB and THC.  It does not own an MPB `TOL`, an ISDF
  `thresh`, a Coulomb cutoff, or a GW workflow.

### **libmuffintin-mbp**

- Builds the conventional LAPW mixed product basis from the product IR:
  radial products are coupled by Gaunt coefficients for each site and $L$,
  their radial overlap matrices are diagonalized, and retained eigenmodes are
  combined with interstitial plane waves in the analytic step-function
  geometry.  Thresholds are channel-resolved and recorded; no universal
  conversion to the THC `thresh` is asserted.

- SPEX `src/mixedbasis.f` is the normative implementation reference for
  radial-product enumeration, overlap diagonalization, normalization, and the
  default `TOL` semantics.  The first fixture stores the untruncated overlap
  spectra as well as the retained basis, so changing a cutoff cannot hide a
  product-generation error.

- Emits the common `CompiledAuxiliaryBasis` and `PairVertex` contracts.
  `libmuffintin-coulomb` therefore assembles the same $V^q$ interface for an
  MPB or THC auxiliary basis, and later GW code does not import either backend.

- Acceptance: radial-product counts and overlap spectra versus SPEX; principal
  angles between retained local spans; finite-$q$ interstitial completeness;
  and Coulomb-metric/pair-action agreement after independently converging MPB
  `TOL`, angular cutoffs, and interstitial $G$-cutoff.

### **libmuffintin-grid (implemented by M-F; production refinements in v0.2)**

- `AtomGrid`: `ExponentialMesh` radial shells × angular set (Lebedev
  preferred; Fibonacci fallback), per-point weights `r² Δr · w_ang`.

- `InterstitialGrid`: uniform cell grid, nearest-image folding, points
  inside any MT sphere dropped; weights `V/N³` (later: step-function
  corrected weights near sphere boundaries).

- `CompositeGrid = Σ AtomGrid + InterstitialGrid` with a stable point
  ordering and a serialization format (points, weights, region tags).

- Debug alternative: `UniformGrid(N)` with identical trait surface.

- Acceptance: quadrature of analytic densities (Gaussians/Slaters, on- and
  off-center) to 1e-10 (adaptive) and known O(Δ²) behavior (uniform);
  property tests against `mt-core` step-function normalization
  `Σ_G Θ_G` vs interstitial volume.

### **libmuffintin-thc**

- Implements an alternative compression of the same product-space contract as
  `libmuffintin-mbp`.  Point selection and ζ fitting are THC-specific, but
  pair windows, $q$/Umklapp gauge, auxiliary-basis metadata, pair vertices,
  and downstream Coulomb interfaces come from `libmuffintin-product`.

- Bloch collocation: `u_{ik}(r)` on any grid from a basis evaluator trait
  (`chi(displacement) -> [f64; NORB]`-style; later the real LAPW evaluator).

- Gauge contract (before any norm, sketch, or fit): `kminus(k,q)` returns both
  the folded k index and reciprocal shift `Gwrap`; every pair column is put in
  the same canonical-q gauge with `exp(+i Gwrap.r)`.  Gauge covariance is an
  algebraic regression, not an accuracy metric.  A fit residual is meaningful
  only after this contract holds.

- Selection: one point set **shared across all q**, with three explicit modes:
  1. `q0_l2`: upstream-CoQuí compatibility/debug baseline, including its
     orbital-count normalization and absolute residual semantics;
  2. `allq_l2`: production baseline, randomized sketch / pivoted QR or

     Cholesky of all canonical-q pair blocks, with adaptive quadrature weights;
  3. `allq_coulomb_pool`: production accuracy mode.  Build an oversampled
     all-q L2 pool (`Npool = pool_factor * Nmu`, initially factor 2), then
     re-rank that finite pool in a q-averaged or q-maximum Coulomb metric.

     This adapts `caef773` but deliberately removes its q=0-only first-pass
     blind spot.

- MT augmentation channels get an additional relative `max_q` Coulomb-norm
  retention gate, following `44c79e9`: a channel cannot be dropped merely
  because its q=0/L=0 component vanishes.  This gate selects angular/radial
  channels; it is separate from the global real-space interpolation points.

- Fit: per-q least squares for ζ^q on the grid; report weighted relative

  L2 residuals plus Coulomb-weighted pair/action residuals.  Report q=0 and
  worst finite-q values separately; never collapse them into one number before
  checking the worst q.

- Termination: `thresh` is supported for the L2 modes; Coulomb-pool re-ranking
  uses an explicit `n_ipts` because the first- and second-pass residuals have
  different meanings.  No fixed value is labelled "accurate": scan the
  requested band window and converge Hartree/exchange/GW observables.

- FFT path = same code with `UniformGrid`; CI cross-check: smooth toy basis
  (no cusp) → adaptive and FFT paths agree to fit-tolerance; sharp toy basis
  (Z≥20) → reproduce the shift-sensitive q=0 quadrature behavior in
  `thc_mt_kpoint_results.txt`.  A separate synthetic LAPW two-region test must
  pass candidate-only pair-Fourier, ERI and action thresholds from
  `thc_lapw_end_to_end_test.py`; density-fit residual alone is not acceptance.

- Export: HDF5 layout compatible with CoQuí `run_isdf` output (interpolating
  points + values; `write_zeta_on_fft_mesh` analog for the adaptive grid) so

  CoQuí can consume the result once its nonuniform k-point path lands.

### **libmuffintin-coulomb (can trail THC by one milestone)**

- Project ζ^q_μ on `CompositeGrid` → per-atom radial × Y_LM (angular sums
  over shells) + interstitial plane waves (ecut parameter).

- V^q_μν assembly: intra-sphere r_<^L/r_>^{L+1} radial integrals
  (`mt-radial` kernels), multipole moments, Weinert pseudo-charge for the
  interstitial/cross terms, q→0 head following SPEX (Friedrich et al.,
  arXiv:0811.2363).

- Acceptance: toy lattice where V_μν is computable by direct Ewald-summed
  quadrature; agreement to metric tolerance.

### **libmuffintin-dft**

- Reference implementation: SPEX `src/iterate.f` (ITERATE mode) — follow
  its structure and radial conventions (`dirac.f` backend) directly.

- Order of work mirrors SPEX/FLEUR structure: (1) LAPW matching
  coefficients (uses existing `BoundaryData`), (2) (H,S) assembly for a
  spherical MT potential + step-function interstitial, collinear spin
  channels from the start, (3) generalized eigensolver (first variation;
  scalar-relativistic Koelling–Harmon for the SOC-free cases),
  (4) SOC/noncollinear per the §0.5 4c-default policy: consume the M-Ka
  `solve_valence_dirac` and typed spinor substrate inside the spheres only —
  interstitial plane waves stay 2-component (SPEX convention, large-
  component matching at R_MT); spinor augmentation + assembly is the new
  work item; nonmagnetic+SOC → **second variation** in
  the reduced first-variation basis; magnetic+SOC/noncollinear → **full 4c
  spinor first variation only** (SPEX policy, no second variation);
  noncollinear vxc via the standard locally-collinear projection, (5) occupations/Fermi level via **Fermi–Dirac smearing**
  (our addition — SPEX lacks it; include the −TS entropy term so the free
  energy is variational), (6) **core states: 4c Dirac**
  (`solve_core_dirac`) on the spherical SCF potential every iteration;
  core density from P²+Q² including the small component, (7) density
  synthesis (MT radial×Y_LM + interstitial PW; valence + core),
  (8) Hartree via Weinert (shared with `mt-coulomb`),
  (9) vxc: **LDA (SPEX-internal parametrization) + GGA (PBE)** — GGA needs

  ∇n in both regions: radial derivatives on the exponential mesh + Y_LM
  gradient algebra in spheres, G-vector multiplication in the interstitial,
  (10) SCF mixing: **linear, Broyden, and Pulay–Anderson**, selectable,
  with the MT/interstitial density coefficients (both spins) concatenated
  into one mixing vector with a consistent metric.

- BZ integration: SCF occupations use the smearing path on the regular
  k-mesh only.  A **tetrahedron integrator is added for DOS only** in the
  initial version (linear tetrahedron, Blöchl corrections optional later);
  it is deliberately NOT wired into the regular SCF integration.

- Explicit non-goals: meta-GGA/hybrids, forces, IBZ reduction inside the

  SCF loop (full BZ first; moyo-based reduction is a later optimization),
  performance.

- Acceptance ladder: (a) free-electron empty-lattice bands; (b) Si/SrVO3
  eigenvalues + `u_l`, `u̇_l`, LO radial functions vs SPEX-side data;
  (c) later Sm/Dy fcc/bcc (the target physics case).

### **libmuffintin-sym (crystal symmetry via moyo)**

- Dependency: the [`moyo`](https://crates.io/crates/moyo) crate — pure-Rust
  crystal symmetry finder from the spglib GitHub org (K. Shinohara), faster
  than spglib, **no C FFI** (fits the `forbid(unsafe_code)` workspace
  ethos; dependencies with `unsafe` are allowed but a pure-Rust dep is
  preferred over spglib bindings).

- Magnetic support is built in: `MoyoMagneticDataset`, `MagneticCell`,
  `NonCollinear` moments, the 1651 UNI/BNS magnetic space groups (same
  database lineage as spglib) — exactly what the magnetic+SOC first-
  variation cases need.

- `mt-sym` stays thin: type conversion (Bohr cells, site labels), symmetry
  operations in libmuffintin conventions, k-star/q-star maps.  Consumers:
  density/potential symmetrization (lattice-harmonic Y_LM stars built on
  top — that part is ours), tetrahedron DOS mesh reduction, later SCF-IBZ
  and THC-IBZ (CoQuí exploits IBZ symmetry in its Cholesky metric too).

- No spglib fallback (frozen decision): if moyo turns out to have a gap,
  fix or contribute upstream — same maintainer group as spglib.

## **2b. Architecture doctrine (anonymous basis, named presets)**

The method-neutral object is a heterogeneous `BasisSpec`:

```text
BasisSpec
  └─ BasisBlock[]
       ├─ envelope / locality
       ├─ channel and radial span
       ├─ augmentation map
       ├─ interstitial representation
       └─ provenance tags
BasisSpec ─compile→ CompiledBasis ─assemble→ OperatorSet(H,S,...)
recipes::lapw() ────────────────┘
```

- **No historical basis enum.**  Do not introduce
  `BasisFamily::{Lapw,Jpo,Nmto,Emto}`.  These names do not denote equivalent
  layers: LAPW/JPO are construction presets, PMT is a heterogeneous block
  union, NMTO is an energy-mesh transform, EMTO is a kink/Green-function
  workflow, and FCD/SCA/FP are closures.  Historical names live only under
  `recipes`, `transforms`, `workflows`, regression fixtures, and
  provenance.

- **M-Fb makes tensor algebra the numerical model before basis extraction.**
  `mt-tensor` owns the backend-neutral global `Tensor`, `TensorView`,
  index-expression, symmetry/placement metadata, execution world/context, and
  explicit host transfer.  It replaces the hand-written LAPW loops for
  $P^\dagger B P$, $X^\dagger H X$, and generalized-eigen residuals; the
  overlap-spectrum filtering algorithm remains library-owned while the
  ordinary Hermitian eigendecomposition is a separate matrix-linear-algebra
  capability.  Local execution may use faer through RSTSR/tenferro; a future
  distributed execution may use CTF's ScaLAPACK-backed `Matrix` operations or
  pair CTF contractions with a separate SLATE provider.

  Physics modules express contractions directly as tensor/index statements
  with declared axes; stable wrappers are added only where they enforce units,
  gauge, Hermiticity or another physical invariant.  Implementations may lower
  the same expression to einsum, `dot_general`, GEMM, or a distributed
  contraction planner.  RSTSR is the canonical default frontend/backend;
  tenferro-rs is a feature-gated second implementation.  There is no production
  scalar backend and no duplicate loop implementation.  Tiny analytic arrays
  and stored oracle fixtures test the tensor results directly.

  Site-local dimensions remain segmented rather than padded.  Each segment is
  nevertheless a global tensor in its execution world.  $k$, spin, $q$, point,
  orbital and auxiliary indices remain visible to the contraction planner;
  libmuffintin does not preselect a $q$-slab or point-tile decomposition.  A
  CTF adapter may therefore retain CTF's own process-grid/rank-allocation and
  redistribution decisions.  M-Fb ships local RSTSR/tenferro execution and a
  CTF ABI design/probe, not a production distributed runtime.

- **M-G extraction from the existing M-F `mt-lapw`.**

  M-G consumes the already tensorized operator substrate; it does not redesign
  tensor storage or contraction semantics.  `PlaneWave` and Rayleigh
  evaluation move behind `Envelope` in
  `libmuffintin-envelope`; `LapwBasisLayout`, local-block ranges, and
  projection maps become generic `BasisSpec/BasisLayout` machinery in
  `libmuffintin-basis`; dense Hermitian operator containers and the
  overlap-filtered generalized eigensolver move to
  `libmuffintin-operators`.  LAPW-specific surface-discontinuity and kinetic
  conventions remain in the LAPW assembly strategy.  The public
  `libmuffintin-lapw` facade re-exports compatibility names and constructs
  exactly the same M-F matrices through `recipes::lapw()`.

- **Envelope/augmentation split.**  v0.2 implements
  `PlaneWaveEnvelope` plus confined site-local blocks.  Boundary projection,
  partial-wave expansion, Gaunt algebra, and sphere matrix elements are public
  augmentation components.  Smooth Hankel, screened spherical waves, and
  structure constants are v0.3 additions to the same interfaces, not new
  basis-class hierarchies.

- **Product, MBP, THC, and DFT consume compiled capabilities.**  None
  accepts a `LapwBasis` concrete type.  Product construction asks for
  pair/radial/region projections; MPB asks for raw product channels and an
  auxiliary partition; THC asks for Bloch collocation/evaluation; operator
  assembly asks for region projections and matrix-element capabilities.  This
  is the v0.2 evidence that later APW/MTO mixtures need no private path.

- ****`step_function` **stays non-overlapping-periodic scoped.**
  `InterstitialGeometry` is valid for the LAPW/full-potential partition but
  not for EMTO overlapping potential spheres or hard screening spheres.  A
  later scattering geometry remains separate.

- **PMT is a v0.3 composition test, not a type.**  It is simply a
  `BasisSpec` containing augmented plane-wave and augmented site-centered
  blocks.  Cross-block overlap is diagnosed through the Schur complement and
  rank-revealing decomposition; no `PMTBasis` implementation is created.
  QuESTAAL PMT may be shipped later as `recipes::questaal_pmt()`, meaning a
  validated combination with trusted defaults, regressions, and provenance—not
  a privileged assembly path.

- **FP-LMTO names are presets too.**  Andersen/Ke FP-TB-LMTO,
  smooth-Hankel FP-LMTO, and future variants are constructed from the same
  envelope, screening, energy, augmentation, and operator components.  v0.3
  may ship selected presets, but user-defined combinations remain anonymous
  `BasisSpec` values.

- **Product basis is orthogonal to one-particle recipe names, but is not
  deferred.**  v0.2 implements
  `raw radial/interstitial products → auxiliary partition → MPB or THC
  compression → Coulomb metric`.  `recipes::lapw()` supplies the normative
  first producer because SPEX already provides a trusted LAPW MPB, not because
  LAPW owns the product-space types.  M-L compares MPB and THC as two
  independently converged representations of the same pair space.

## **3. Milestones**

M-A through M-F are retained below as historical milestones.  Their status is
`IMPLEMENTED`, but their contracts and acceptance tests remain normative:
v0.2 may reorganize ownership without weakening or silently replacing them.
The corresponding derivations remain `doc/01` through `doc/10`.

- **M-A (core conventions and geometry primitives) — IMPLEMENTED.**

  - Hartree/Bohr typed units, explicit I/O conversion, energy-zero metadata,
    right-handed direct lattices and $A^T B=2\pi I$;

  - complex Condon–Shortley and real tesseral spherical harmonics, stable
    $(l,m)$ indexing, Wigner 3j and SPEX-convention Gaunt coefficients;

  - spherical Bessel functions, reciprocal-vector enumeration by Cartesian
    cutoff, Bloch/Fourier phases, and the analytic non-overlapping interstitial
    step function with complete $G-G'$ support;

  - exponential radial meshes and the SPEX-compatible high-order quadrature,
    including origin correction and cumulative primitives.

  Acceptance remains: explicit unit round trips, harmonic orthonormality and
  Gaunt selection rules, $A^T B=2\pi I$, analytic-versus-numerical sphere
  transforms, translated-cell step-function Hermiticity, exact free-electron
  kinetic convention, and polynomial/smooth-function quadrature tests.

- **M-B (radial solutions, local radial algebra and relativity) — IMPLEMENTED.**

  - nonrelativistic and Koelling–Harmon scalar-relativistic valence radial
    solvers in the declared $u_l=P_l/r$ and Hartree-potential convention;

  - normalized energy derivatives $\dot u_l$, stored boundary value/slope
    data, local-orbital construction, radial overlap/Hamiltonian blocks and
    nonspherical-potential radial integrals;

  - a separate spherical four-component Dirac bound-core solver with explicit
    $\kappa$, large/small components and inside/outside normalization;

  - the valence 4c API is deliberately reserved and must return unsupported;
    spinor augmentation and assembly are not retrospectively claimed by M-B.

  Acceptance remains: large-$c$ NR/SR agreement, energy finite-difference
  checks of $\dot u$, $\langle u|u\rangle=1$ and
  $\langle u|\dot u\rangle=0$, zero LO boundary value/slope, Hermitian radial
  blocks, hydrogenic/core matching tests, explicit Hartree/Rydberg adapter
  equivalence, and rejection of a false scalar fallback for valence 4c.

- **M-C (grids, sphere algebra and versioned artifacts) — IMPLEMENTED.**

  - typed atom-centred, uniform, interstitial and stable composite grids, with
    deterministic point ordering and an optional RSTSR tensor boundary;

  - $(L,M)$-resolved `SphereField`/`SphereOrbital` algebra and
    Gaunt-weighted radial matrix elements in complex or real harmonics;

  - independently versioned, human-diffable TOML physical snapshots and
    materialized grid artifacts; unknown versions, units, fields, invalid
    channels and non-finite data are rejected.

  Acceptance remains: normalized angular rules, off-centre Gaussian/Slater
  quadrature, periodic sphere removal and interstitial-volume convergence,
  stable composite ordering, low-order sphere selection/Hermiticity tests,
  artifact round trips and both default/RSTSR-feature builds.

- **M-D (LAPW boundary matching and overlap) — IMPLEMENTED.**

  - SPEX-convention APW matching of plane-wave value and radial slope to
    $u_l,\dot u_l$, with explicit Rayleigh factors and site translations;

  - dense complex overlap assembly from the analytic interstitial block plus
    muffin-tin augmentation corrections.

  Acceptance remains: value/slope residuals $\le 10^{-10}$, Hermitian $S$
  for translated spheres and nonzero $k$, exact $S=I$ for the empty-sphere
  plane-wave cell, and explicit rejection of mixed-$k$, missing-site and
  inconsistent-channel inputs.

- **M-E (LAPW Hamiltonian and generalized eigensolver) — IMPLEMENTED.**

  - the SPEX symmetric-Laplacian interstitial kinetic convention, interstitial
    potential convolution, and full Hermitian muffin-tin Hamiltonian blocks for
    spherical and warped potentials;

  - dense $H,S$ containers, overlap-spectrum diagnostics, filtered
    generalized Hermitian eigensolution, $C^HSC=I$ and residual reports;

  - versioned $(k,\mathrm{band})$ reference reports with explicit missing-data
    failure.  The real Cu/SPEX one-meV gate remains pending a matched frozen
    potential/basis fixture and must not be replaced by synthetic evidence.

  Acceptance remains: the full empty-lattice free-electron matrix, Hermiticity
  of interstitial+sphere assembly, the spherical radial identity, filtering of
  near-null overlap directions, rejection of significant negative overlap, and
  eigenvector normalization/residual checks.

- **M-F (APW+LO layout and collinear spin) — IMPLEMENTED.**

  - deterministic global `[APW][site LO]` layout and site-local
    $(l,m,n)$ ordering;

  - one projection rule $P^\dagger(S/H)P$ for APW–APW, APW–LO and LO–LO
    blocks, with local orbitals confined by zero value and slope at the sphere;

  - independent collinear up/down frozen-potential channels without SOC.

  Acceptance remains: deterministic indexing, LO boundary residuals, Hermitian
  $S,H$ from the same projection layout, and independent generalized-
  eigenproblem residuals for both spin channels.  M-F does not claim SCF,
  occupations, potential construction, SOC, noncollinearity or spinor assembly.

- **M-Fb (tensorized numerical substrate; pre-M-G)**: add
  `mt-tensor` and migrate the existing M-F dense numerical kernels before
  reorganizing package ownership.  Tensor expressions must make axis order,
  complex conjugation, batch dimensions, and reductions visible; RSTSR is the
  default backend and tenferro-rs is a feature-gated parity backend.  Backend
  tensors remain private, serialized artifacts remain host/backend-neutral,
  and the public result types preserve physical units and provenance.

  Acceptance requires RSTSR and enabled tenferro-rs paths to agree with direct
  analytic fixtures and the frozen M-F outputs for $P^\dagger B P$,
  $X^\dagger H X$, Hermitian eigendecomposition inputs/outputs, and residual
  contractions within declared tolerances.  The complete M-F $H,S$, retained
  overlap rank, eigenvalues,
  eigenvectors up to degenerate-subspace freedom, and residuals must remain
  tolerance-identical.  Tests cover complex conjugation, non-contiguous views,
  shape/axis errors, explicit host transfer, $k$/spin batches, and expression
  equivalence under legal reshapes.  The CTF probe specifies an opaque C++ ABI
  for `World`, global tensor construction, index-string contraction, and
  coordinate I/O with explicit `f64`/`Complex64` instantiations; it must leave
  tensor mapping and redistribution inside CTF.  No production MPI/GPU/AD or
  distributed-performance claim is part of this milestone.

  Binding details and the optional CTF-to-SLATE ownership bridge are specified
  in `scratch/ctf_slate_rust_binding_brief_plan.md`.

- **M-G (anonymous basis extraction + package rename)**: introduce
  `BasisSpec`, heterogeneous `BasisBlock`, `CompiledBasis`, generic
  operator containers and eigensolver; move generic plane-wave envelopes,
  layouts and projections out of `mt-lapw`; add `libmuffintin-recipes` for
  canonical presets, defaults and provenance; retain `libmuffintin-lapw` as
  the compatibility facade over the `recipes::lapw()` preset.  Rename public Cargo
  packages to `libmuffintin-*`.  Acceptance is tolerance-identical M-F H/S
  matrices, retained overlap rank, eigenvalues, eigenvectors and residuals
  through both the facade and explicit `BasisSpec` routes.

- **M-H (product-space IR + LAPW mixed product basis)**: implement
  `libmuffintin-product` and `libmuffintin-mbp`; enumerate untruncated
  radial products and coupled $(L,M)$ channels, construct the independent
  non-overlapping auxiliary partition, diagonalize channel-resolved radial
  overlap matrices, and add interstitial plane waves.  Freeze SPEX
  `mixedbasis.f` fixtures containing raw product counts, overlap spectra,
  retained local modes, pair vertices, and finite-$q$ Coulomb blocks.  Include
  valence–valence and selected core–valence products.  Acceptance requires
  matching SPEX conventions before applying either MPB `TOL` or THC
  compression.

- **M-I (libmuffintin-thc, toy basis)**: k-point ISDF on both paths with the toy
  evaluator; port the Umklapp regression and shift/seed/column diagnostics
  from `thc_mt_kpoint_test.py`, then port the candidate-only Coulomb/ERI/action
  smoke test from `thc_lapw_end_to_end_test.py`.  At identical `Nmu`, compare
  `q0_l2`, `allq_l2`, and `allq_coulomb_pool`; record per-q L2/Coulomb/action
  errors and fail if q=0 selection hides a finite-q channel.  CI compares
  against the recorded, explicitly finite-cutoff reference numbers only after
  this selector comparison freezes the production default.

- **M-J (libmuffintin-coulomb)**: assemble Weinert $V^q$ over the
  common auxiliary-basis contract.  Exercise both the explicit MPB and ζ/THC
  representations on the toy lattice; neither is a privileged input type.

- **M-Ka (4c relativistic substrate + SRA-LAPW/HDLO path)**:
  activate the reserved four-component valence path before SCF and before the
  magnetic + SOC integration claim in M-L.  This is a parallel typed spinor
  route; it does not reinterpret the implemented scalar $(l,m)$ LAPW and
  collinear APIs.

  **Implementation status (2026-08-24): `IMPLEMENTATION_CLOSED`.** Analytic
  second energy derivatives, confined SRA HDLO construction, the typed spinor
  substrate, and the repository-local non-empty-sphere large-$c$ reduction gate
  are implemented and exercised by focused regressions.  This status closes
  the implementation dependency for M-L; it does not claim cross-code
  acceptance.  The independent frozen FlapwMBPT-SRA or source-equivalent
  radial/augmentation/sphere/band fixture is still absent, so the external
  acceptance paragraph below remains binding.

  - `libmuffintin-core` owns the shared spin-angular contract: validated
    $\kappa$, exact integer $2\mu$, $\Lambda=(\kappa,\mu)$ channel ordering,
    the $\Omega_{\kappa\mu}$ phase convention, Clebsch--Gordan coefficients,
    and spinor-Gaunt reductions.  The existing $\kappa$ type moves out of the
    radial implementation rather than being duplicated or hidden behind an
    optional field on the scalar $(l,m)$ type.

  - `libmuffintin-radial` activates `solve_valence_dirac` for a central scalar
    potential and fixed real energy.  Its public solution stores the physical
    reduced components $(P_\kappa,Q_\kappa)$, while an internal $cQ$ scaling
    remains private.  Implement the analytic first energy derivative
    $(\dot P_\kappa,\dot Q_\kappa)$ and analytic second derivative
    $(\ddot P_\kappa,\ddot Q_\kappa)$.  With
    $A=2+(E-V)/c^2$, the scaled-component second-order equations are

    ```math
    \ddot P'=-\frac{\kappa}{r}\ddot P+A\ddot q
    +\frac{2\dot q}{c^2},\qquad
    \ddot q'=\frac{\kappa}{r}\ddot q+(V-E)\ddot P-2\dot P.
    ```

    The fixed normalization/phase gauge obeys
    $\langle R_\kappa|\dot R_\kappa\rangle=0$ and the exact identity
    $\langle R_\kappa|\ddot R_\kappa\rangle
    =-\langle\dot R_\kappa|\dot R_\kappa\rangle$.  The common Dirac
    boundary trace retains $(P,Q)$ and derives radial derivatives from the
    first-order equations; it is not the scalar LAPW `BoundaryData`.

  - The SRA-LAPW adapter derives $P'$ from the Dirac equation and then forms
    the existing large-component boundary pair

    ```math
    U=P(R)/R,\qquad U_r=P'(R)/R-P(R)/R^2.
    ```

    Only $(U,U_r)$ enters the existing value/slope $2\times2$ LAPW match.
    The interstitial is a two-component Pauli plane-wave space with no small
    component; the sphere overlap, Hamiltonian, and variational
    Schlosser--Marcus surface convention retain the declared SRA large/small
    physics.  A missing interstitial small component is part of the model, not
    an assembly fallback.

  - M-Ka constructs one confined SRA HDLO per requested spinor channel as
    $R_{\mathrm{HDLO}}=\ddot R+aR+b\dot R$, with

    ```math
    \begin{pmatrix}U&\dot U\\U_r&\dot U_r\end{pmatrix}
    \begin{pmatrix}a\\b\end{pmatrix}
    =-\begin{pmatrix}\ddot U\\\ddot U_r\end{pmatrix}.
    ```

    Both large-component boundary traces therefore vanish.  The full
    large/small spinor is normalized in the sphere and remains a site-local
    orbital with no interstitial tail; a singular construction is an explicit
    error rather than a fallback.

  - `libmuffintin-sphere` adds a parallel `SpinorSphereOrbital` path.  Scalar
    $v_{LM}$ matrix elements keep the $PP$ contribution in
    $\Omega_\kappa$ and the $QQ$ contribution in $\Omega_{-\kappa}$ as
    separate radial integrals with separate CG $\times$ Gaunt factors.  This
    permits non-diagonal $(\kappa,\mu)\leftrightarrow(\kappa',\mu')$ angular
    blocks without pretending that the central-potential radial solver itself
    is a coupled-$\kappa$ solver.

  - `libmuffintin-basis` adds parallel `SpinorBasisLayout` and compiled
    spinor-augmentation types.  The plane-wave order is
    `spin * n_g + g` (spin slow), all plane waves precede all site-local
    spinor orbitals, and ordinary LO/HELO channels are explicit
    $(\kappa,\mu,n)$ entries.  `libmuffintin-operators` and the LAPW facade
    add a spinor projection builder and lift the unchanged SRA interstitial
    kernel into equal-spin blocks.  The generic Hermitian congruence and
    generalized eigensolver remain shared; the existing scalar `BasisLayout`,
    `PlaneWaveAugmentation`, and independent `Collinear<T>` route keep their
    current meanings.

  - **KKR/LMTO boundary:** the common $\Lambda$ layout, Dirac radial
    solutions, energy derivatives, and full $(P,Q)$ trace contain no
    plane-wave, LAPW, step-function, screening, or SCF types.  SRA is one
    envelope adapter, not the radial representation.  Later fully relativistic
    KKR/LMTO code must be able to consume the unprojected trace to build
    convention-tagged regular/irregular Wronskians, complex-energy single-site
    $t_{\Lambda\Lambda'}(z)$, screened potential functions, and kink/slope
    matrices without reconstructing information discarded by $(U,U_r)$.

  - **Explicit non-goals:** FRA-LAPW and relativistic interstitial plane waves;
    complex-energy scattering, structure constants, screening, and Green
    functions; radial magnetic fields or general coupled-$\kappa$ Dirac
    equations; spinor product/MPB/THC identities; density/potential synthesis,
    occupations, XC, mixing, and SCF.
    Do not add `kappa` to the existing scalar `ProductRadialId`; M-L owns the
    relativistic product bridge once real spinor orbitals exist. M-Ka is
    SRA-only. Scattering and magnetic-radial request APIs are outside this
    milestone; none of these absent modes may be represented by a fallback to
    SRA or scalar-relativistic data.

  - **Far-future FRA research option:** FRA is retained only for
    methodological completeness as a non-production research direction.  It
    is outside v0.2, outside current implementation and acceptance, and does
    not create an FRA request API or negative-test surface.  A later research
    milestone must separately define relativistic interstitial functions,
    four-component matching, surface terms, and independent fixtures.

  Acceptance requires complete $(\kappa,\mu)$ enumeration and normalized
  spinor harmonics; Dirac ODE residuals for $(P,Q)$ and
  $(\dot P,\dot Q)$, and $(\ddot P,\ddot Q)$; phase-aligned
  centred-finite-difference agreement for both energy derivatives on the mesh
  and at the full boundary traces; unit norm, derivative orthogonality, and
  $\langle R|\ddot R\rangle=-\langle\dot R|\dot R\rangle$; separate $PP/QQ$
  spinor-Gaunt oracle values; SRA boundary conversion and confined HDLO value
  and slope residuals no larger than `1e-10`; deterministic spinor PW/LO
  indexing; Hermitian spinor $H,S$,
  positive retained overlap, and bounded generalized-eigen residuals; and the
  large-$c$ reduction in which one repository-local scalar LAPW frozen fixture is
  duplicated across the two Pauli-spin blocks. A frozen independent
  FlapwMBPT-SRA or source-equivalent fixture
  must cover selected radial traces, augmentation coefficients, sphere blocks,
  and frozen-potential bands before the full path is called cross-code
  validated. FRA is explicitly not part of v0.2 implementation or acceptance;
  it remains only a far-future non-production research option.

  Primary reference contracts are Kutepov's Dirac APW/LAPW formulation
  (arXiv:2012.04992) for the radial equations, energy derivatives, and
  FRA/SRA boundary distinction; Ebert et al. (arXiv:1512.04294) for the
  general-potential fully relativistic KKR channel/trace boundary; and the
  QuESTAAL KKR/LMTO Green-function documentation for the separation between
  exact-energy scattering, screened potential functions, and LMTO
  linearization.  Formula conventions are recorded with the fixture rather
  than inferred from method names.

- **M-Kb (libmuffintin-dft)**: consume the M-Ka typed relativistic
  core/spinor contracts and implement minimal LDA/GGA SCF (FD smearing,
  linear/Broyden/Pulay–Anderson mixers); SPEX cross-validation ladder:

  **Implementation status (2026-08-24): `IMPLEMENTATION_CLOSED`.** The
  library, ordered TOML workflow, executable, and versioned snapshot/restart
  boundaries are closed.  The following ladder remains external acceptance,
  not unfinished implementation:
  (a) Si/SrVO3 nonmagnetic scalar-relativistic, (b) nonmagnetic + SOC via
  second variation (fcc Pt or Au bands vs SPEX), (c) collinear magnetic
  without SOC (bcc Fe); tetrahedron DOS cross-checked against a SPEX/FLEUR
  reference.  M-Kb does not reimplement the radial or spinor substrate.

- **M-Kc (orbital-configuration V2) — IMPLEMENTATION_CLOSED.**:
  hard-replace the V1 orbital fields with `[basis.envelope]`, spectroscopic
  channel tokens, layered recipe artifacts, and current-potential generation.
  The implemented generators are `explicit`, `atomic`, `band-center`,
  `log-derivative`, `band-cog`, `fermi-offset`, and
  `frozen-snapshot`.  Materialized energies and provenance are retained in
  SCF state and reused by downstream bands/DOS tasks.  V2 accepts any
  nonnegative `derivative-order`, executes the current 0/1 and 2 contracts,
  and returns a typed `NotImplemented` result for order 3 or greater.  Fill
  generators, $l$-first generation, seed fallback, and executable higher-order
  derivatives are explicitly deferred to v0.3.  This closure covers the
  library/runtime/schema/documentation contract only; the M-Kb material
  cross-validation ladder remains open.

- **M-L (integration)**: after the M-Ka/M-Kb/M-Kc implementation
  contracts, construct the
  SPEX-compatible MPB and THC factors from real scalar and spinor orbitals of
  the anonymous compiled basis produced by `recipes::lapw()`.  M-L owns the
  explicit relativistic orbital/product bridge (including separate large-large
  and small-small angular factors) rather than extending the scalar
  `ProductRadialId` in M-Ka; optionally
  including the 4c core radial functions in the THC window (core–valence
  exchange / core excitations — the all-electron motivation of
  arXiv:2510.20826); compare ζ and MPB spans, pair vertices, Coulomb spectra,
  and random pair actions at independently converged tolerances (MPB `TOL`
  and THC `thresh` do
  not have a universal one-to-one mapping); CoQuí-compatible

  HDF5 export; Sm/Dy fcc/bcc demo — this is the magnetic + SOC **full 4c
  spinor first-variation** case — with the grid budget from
  `grid_budget_sm_dy.py`.

Each new milestone gets a numbered derivation note under `doc/` in the existing

style (continue after the implemented M-F documents; proposed additions:
`11_tensorized_numerical_substrate.md`,
`12_anonymous_basis_and_lapw_facade.md`,
`13_product_space_and_lapw_mbp.md`,
`14_kpoint_isdf_thc.md`, `15_weinert_coulomb_metric.md`,
`16_dirac_spinor_substrate_and_sra_lapw.md`,
`17_minimal_lda_scf.md`, and
`18_lapw_mbp_thc_integration.md`).  The v0.3 draft's proposed
documentation sequence must consequently begin at `doc/19`; that draft is
not renumbered by this v0.2-only edit.

## **4. Risks / open questions**

- RSTSR and tenferro-rs do not expose identical ownership, layout, einsum, or
  eigensolver APIs.  The adapter contract is therefore defined by the physical
  contractions and invariants we actually need, not by an attempted union of
  both libraries.  Backend-specific optimization stays private and exact
  dependency versions are pinned while either project is pre-1.0.

- CTF-like distribution is a future execution backend, not a v0.2 data-model
  feature.  Premature Rust-owned communicator or sharding types could freeze
  the wrong decomposition before MPB/THC memory profiles exist.  M-Fb instead
  preserves global tensor rank/shape, index expressions, symmetry, placement
  intent, and an opaque execution world.  The CTF backend remains responsible
  for choosing processor mappings and redistributions from the actual
  contraction graph.

- CTF's public API is C++ templates, so Rust must not attempt to mirror its
  template or expression types.  A small C++ shim explicitly instantiates
  `double` and `std::complex<double>`, owns `CTF::World`/`CTF::Tensor` behind
  opaque handles, and exposes global construction, index contraction, and
  coordinate I/O.  This ABI adaptation must preserve CTF's global tensor and
  rank-allocation mechanism rather than degrade it to a local-tile callback.

- CoQuí's nonuniform k-point path is upstream work we don't control; the

  HDF5 export keeps us decoupled (worst case: consume THC factors in our
  own downstream tooling).

- Angular quadrature for products needs Lebedev exactness ≥ 2·lmax of the
  augmentation (lmax=8 → order ≥17, 110–194 pts); Fibonacci is only a
  stopgap for tests.

- Interstitial weights near sphere boundaries: plain uniform weights were
  fine at toy tolerances; step-function-corrected weights may be needed at
  1e-5.  Measure before building.

- MPB/THC comparison in M-L must not assume matched numerical tolerances are
  meaningful across
  the two constructions (spectral truncation vs ID); define the comparison
  metric (principal angles between spans on the sphere) in doc 08.

- The two `mmorale3/coqui` branches are experimental and divergent: the
  q-averaged global re-ranker and the full-q PAW channel gate are evidence for
  the selector design, not an upstream API or a dependency.  Reimplement the
  small metric contracts locally and validate them on the gauge-correct LAPW
  toy before treating either as normative.

- moyo's magnetic dataset is newer than its space-group core; validate the
  magnetic operations against spglib's magnetic dataset on our magnetic
  test structures before relying on them for symmetrization.

- Magnetic + SOC first variation doubles the (H,S) dimension (full spinor
  basis) — the eigensolver cost ×8 relative to a collinear channel; fine
  for v0.2 system sizes, but flag it in doc 10 so nobody "optimizes" it
  back to second variation for magnetic cases.

## **5. Reference implementation (Python, working)**

Full version: `scratch/thc_mt_kpoint_test.py` (run with

`/Users/zerozaki07/tmp/xca/.triqs/venv/bin/python`).  The construction

pipeline to port is reproduced here as the normative reference:

```python
import numpy as np
from numpy.linalg import lstsq, norm
from scipy.linalg import qr
# ---- adaptive grid: exponential radial shells x angular + interstitial ----
def log_radial(r0, r1, n):
    """Exponential mesh r_j = r0 e^{jh}, trapezoid weights for int f r^2 dr."""
    h = np.log(r1 / r0) / (n - 1)
    r = r0 * np.exp(h * np.arange(n))
    w = r**3 * h                      # dr = r h on the log mesh
    w[0] *= 0.5; w[-1] *= 0.5
    return r, w
def fib_sphere(n):                    # stopgap; production uses Lebedev
    i = np.arange(n) + 0.5
    ct = 1.0 - 2.0 * i / n
    st = np.sqrt(1.0 - ct**2)
    th = np.pi * (1.0 + 5.0**0.5) * i
    return np.column_stack([st*np.cos(th), st*np.sin(th), ct])
def atom_grid(r0, r1, nrad, nang):
    r, wr = log_radial(r0, r1, nrad)
    ang = fib_sphere(nang)
    pts = (r[:, None, None] * ang[None, :, :]).reshape(-1, 3)
    w = np.repeat(wr * 4.0 * np.pi / nang, nang)
    return pts, w
def fold(pts, A):                     # nearest-image displacement from atom
    return pts - A * np.round(pts / A)
def uniform_grid(n, A):               # FFT/debug path AND interstitial donor
    t = (np.arange(n) + 0.5) / n * A
    X, Y, Z = np.meshgrid(t, t, t, indexing="ij")
    pts = fold(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]), A)
    return pts, np.full(len(pts), A**3 / n**3)
def adaptive_grid(A, RMT, RCUT, nrad, nang, ninter):
    """MT shells inside R_MT + coarse uniform shell out to the orbital cutoff."""
    mt_pts, mt_w = atom_grid(2e-3, RMT, nrad, nang)
    up, uw = uniform_grid(ninter, A)
    d = norm(up, axis=1)
    keep = (d > RMT) & (d <= RCUT)
    return np.vstack([mt_pts, up[keep]]), np.concatenate([mt_w, uw[keep]])
# ---- Bloch collocation on any grid (chi = localized basis evaluator) -----
def bloch_u(pts, A, KFRAC, chi, images):
    """Cell-periodic parts u_{ik}(r): (npts, nk, norb)."""
    U = np.zeros((len(pts), len(KFRAC), chi(pts).shape[1]), dtype=complex)
    for T in images:                              # e.g. [-1,0,1]^3
        c = chi(pts - A * T)
        ph = np.exp(2j * np.pi * (KFRAC @ T))     # e^{i k.T}
        U += ph[None, :, None] * c[:, None, :]
    ph_r = np.exp(-2j * np.pi / A * (pts @ KFRAC.T))
    return U * ph_r[:, :, None]
# ---- canonical-q pair blocks, including the BZ-folding gauge phase -------
def pair_matrix(U, pts, iq, kminus):
    """Columns conj(u_{i,k-q}) u_{j,k} for all (k,i,j): (npts, nk*norb^2)."""
    npts, nk, norb = U.shape
    cols = []
    for ik in range(nk):
        left, reciprocal_shift = kminus(ik, iq)
        umklapp = np.exp(2j*np.pi/A * (pts @ reciprocal_shift))
        cols.append(umklapp[:, None, None]
                    * np.conj(U[:, left, :, None]) * U[:, ik, None, :])
    return np.stack(cols, axis=1).reshape(npts, nk * norb * norb)
# ---- q-shared production pool: randomized all-q sketch + pivoted QR -------
def select_points(U, pts, w, nmax, kminus, sketch=160, rng=np.random.default_rng(7)):
    Zall = np.hstack([pair_matrix(U, pts, iq, kminus) for iq in range(U.shape[1])])
    Omega = rng.normal(size=(Zall.shape[1], sketch)) \
            + 1j*rng.normal(size=(Zall.shape[1], sketch))
    S = (np.sqrt(w)[:, None] * Zall) @ Omega
    _, _, piv = qr(S.T, mode="economic", pivoting=True)
    return piv[:nmax]
def fit_zeta(Z, rows, w):
    """Candidate-only zeta and its weighted L2 pair-density residual."""
    X, *_ = lstsq(rows.T, Z.T, rcond=None)
    zeta = X.T
    resid = norm(np.sqrt(w)[:, None] * (Z - zeta @ rows)) \
            / norm(np.sqrt(w)[:, None] * Z)
    return zeta, resid
```

Usage skeleton (see `thc_mt_kpoint_test.py::main` for the full driver with

reference-grid error measurement and the SVD lower bound):

```python
pts, w = adaptive_grid(A=6.0, RMT=2.0, RCUT=2.9, nrad=20, nang=26, ninter=12)
U      = bloch_u(pts, 6.0, KFRAC, chi, IMAGES)          # KFRAC: 2x2x2 mesh
piv    = select_points(U, pts, w, nmax=alpha * NORB, kminus=kminus)
for iq in range(len(KFRAC)):                             # per-q zeta fit
    Z = pair_matrix(U, pts, iq, kminus)
    zeta, err = fit_zeta(Z, pair_matrix(U[piv], pts[piv], iq, kminus), w)
```

This all-q selector is the production L2 baseline.  Keep a separate q=0-only

implementation solely for upstream-CoQuí compatibility, and add the finite-pool

Coulomb re-ranker after M-I establishes its q-averaged versus q-maximum metric.

Porting notes: `log_radial` ↔ `mt_core::ExponentialMesh` (reuse its

quadrature weights instead of the inline trapezoid); `fib_sphere` → Lebedev

tables in `libmuffintin-core`; `chi` → toy evaluator first, generic
`CompiledBasis` evaluator after M-G;

`select_points` sketch size ≥ 2×nmax; complex QRCP is requested through the
M-Fb tensor/backend capability (the RSTSR implementation may lower to faer or
LAPACK; no product-space crate imports a concrete solver); weights enter
selection as √w column scaling and

fitting via √w row scaling, pivot rows stay **unweighted**.

## Primary references

- J. Lu and L. Ying, "Compression of the electron repulsion integral tensor
  in tensor hypercontraction format with cubic scaling cost," J. Comput.
  Phys. 302, 329 (2015) — interpolative separable density fitting (ISDF)
- E. G. Hohenstein, R. M. Parrish, and T. J. Martínez, "Tensor
  hypercontraction density fitting," J. Chem. Phys. 137, 044103 (2012)
- H. Zhu et al., adaptive-grid ISDF with DMK, arXiv:2510.20826
  (companion PDF `isdf_adaptive_2510.20826.pdf` in this directory)
- C. Friedrich, S. Blügel, and A. Schindlmayr, "Efficient implementation of
  the GW approximation within the FLAPW method," Phys. Rev. B 81, 125102
  (2010) — SPEX mixed product basis, the normative MPB reference
- M. Weinert, "Solution of Poisson's equation: Beyond Ewald-type methods,"
  J. Math. Phys. 22, 2433 (1981) — pseudo-charge Coulomb metric
- SPEX reference source: `/Users/zerozaki07/Documents/dft_codes/spex06.00pre36`
  (`src/mixedbasis.f` for MPB semantics)
