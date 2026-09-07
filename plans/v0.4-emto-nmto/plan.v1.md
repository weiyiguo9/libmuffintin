# libmuffintin v0.4 plan — EMTO/NMTO route: energy-mesh transforms, Green-function workflows, and the adaptive-grid Coulomb route

- Workstream ID: `v0.4-emto-nmto`
- Plan version: 1
- Approval: draft (not authorized)
- Supersedes: none
- Imported: 2026-09-08 from `scratch/libmuffintin_v0.4_emto_nmto_plan.md` (last modified 2026-08-27), body unchanged
- Status source: false; state lives in `STATUS.md` and `ledger.md`
- Path map: `libmuffintin_v0.3_mto_family_plan.md` is `plans/v0.3-mto-family/plan.v1.md`; `design_decision_02_mto_poisson.md` is `decisions/ADR-0003-mto-density-poisson-kinetic.md`

Status: draft, 2026-08-27. Split out of the former v0.3 MTO-family plan
(2026-08-27); also absorbs the EMTO/NMTO- and DMK-relevant decisions of
`design_decision_02_mto_poisson.md` (merged and retired the same day).

Inheritance: this plan assumes a completed v0.3 (M-M through M-Q: scattering
conventions, `RadialJet`, screening equivalence, FP-TB-LMTO and smooth-Hankel
JPO/PMT presets, published `libmuffintin-basisopt`). It inherits the v0.3 plan's
§0 frozen scope decisions, §1 central factorization, §2 mathematical contracts,
and §9 numerical failure policy verbatim and does not restate them. Historical
names remain preset/transform/workflow labels only.

## 1. Scope

In scope:

- `transforms::nmto()` — energy-mesh interpolation and downfolding;
- `workflows::emto_sca_fcd()` — canonical EMTO kink/contour-Green-function SCF;
- the `ssw_fp_exact_e` experiment and the full-potential unification verdict;
- the adaptive-grid representation and DMK Coulomb backend (FFI over
  PeriodicDMK), as the second first-class representation/poisson route;
- optional auxiliary/empty centers with automated placement.

Out of scope (inherited exclusions): CPA, transport, forces, phonons, a
completed GW workflow, automatic differentiation, GTO recipes. Native PPO/open
boundary conditions are a contract-level deliverable only (see §4.4); surface
production work is deferred beyond v0.4.

## 2. Reference transforms and workflows

### 2.1 NMTO transform

NMTO is implemented only after the v0.3 kink-matrix and screening invariance
tests pass. The initial consumer is frozen-potential band interpolation and
downfolding, not a new SCF loop.

Outputs:

- Nth-order energy-independent orbitals;
- active/passive reconstruction map;
- real-space range/localization diagnostics;
- Hamiltonian and overlap in the nonorthogonal and Löwdin-orthogonalized bases;
- Wannier-like export with exact provenance of the selected energy window and
  channels.

NMTO consumes the kink matrices and divided-difference machinery of the v0.3
scattering layer; full-potential NMTO is the NMTO transform riding on a
full-potential representation (three-component in v0.3, adaptive-grid in §4),
never a private density path.

### 2.2 EMTO SCA+FCD

The canonical EMTO path keeps its separate geometry and spectral semantics:

```text
optimized overlapping potential spheres
  -> exact-energy potential functions
  -> screened spherical waves and slope/kink matrix K(z)
  -> contour Green function
  -> density of states and charge moments
  -> SCA electrostatics + FCD reconstruction
  -> SCF
```

The v0.2 full-potential density machinery is reused for FCD reconstruction and
cross-checks, but its non-overlapping interstitial partition is not imposed on
the EMTO scattering problem.

The EMTO interstitial density/Poisson closure stays SCA/FCD. The
Nohara–Andersen value-and-derivative interpolation is **not** a production
upgrade path: it was demoted (2026-08-24) to an optional fixed-parameter test
oracle living in the v0.3 M-N test layer, and is not consumed by this workflow.

Likewise, an eventual MBP analysis partition is an independent
`ProductPartition`, not the set of overlapping EMTO potential spheres. v0.4
only requires a frozen-potential residue/wave evaluator and product-partition
projection test; a complete GW polarization workflow remains out of scope.

### 2.3 Full-potential screened-wave experiment

An experimental workflow combines exact-energy SSW/kink data with v0.2
full-potential matrix elements. It is kept under `ssw_fp_exact_e` until one of
the following is established:

1. it is a controlled full-potential correction to canonical EMTO; or
2. it is better classified as the exact-energy limit of FP-TB-MTO.

The name follows the mathematics, not the desired feature checklist.

## 3. Adaptive-grid representation and DMK Coulomb backend

Both representation/poisson routes are first class (decision 2026-08-24):

- **three-component + FFT** (v0.3): anchors the library to published FP-TB-LMTO
  and QuESTAAL workflows; every regression acceptance runs through it;
- **adaptive-grid + DMK** (this plan): the forward path. Poisson is ultimately
  numerical; the smooth/onsite bookkeeping of three-component exists to make
  uniform-grid FFT applicable, not as physics. The historical analytic
  apparatus (USW screening, v&d interpolation, double augmentation) largely
  compensated for fast adaptive solvers that did not exist yet.

Epistemic split: the DMK route has no published MTO reference numbers; its
correctness is validated against the three-component route on identical
systems. Running the routes in this order is itself the validation plan.

### 3.1 Implementation: FFI over PeriodicDMK, no self-implementation

[PeriodicDMK](https://github.com/xuanzhaogao/PeriodicDMK) (Jiang–Gao, Flatiron,
2026-06, MIT) provides periodic Coulomb lattice sums in arbitrary triclinic
cells with an optional Bloch phase (`evaluate_complex`), a C ABI
(`pdmk_capi.h`), and a build-once/evaluate-many tree suited to SCF loops.

- Crate: `crates/mt-pdmk`, package `libmuffintin-pdmk`, target `muffintin_pdmk`;
  feature-gated, source-build only (upstream bakes `-march=native` and fetches
  dependencies at configure time), never a default workspace dependency.
- Work items:
  1. continuous density → quadrature-weighted point-charge adapter, shared with
     the adaptive-grid ISDF construction (Zhu et al., arXiv:2510.20826);
  2. finite-q quasi-periodic validation against the upstream DUCC Ewald
     reference and `libmuffintin-coulomb`'s own `ewald.rs`;
  3. weak-form kinetic (`weak_grid`): high-order local bases on the adaptive
     grid; no spectral differentiation and no differentiation of interpolated
     densities;
  4. capability-matrix entries for the Coulomb kernel contract (§3.2).

The real cost of this route is not the Poisson solve (FFI) but the
adaptive-grid representation/quadrature layer — which is the same layer the
adaptive-grid ISDF/THC target needs, so the investment is consumed twice.

### 3.2 Coulomb kernel backends

The `CoulombKernel` contract (BC × backend, defined in the v0.3 plan §2.8) is
extended here with the `Dmk` backend: PPP via PeriodicDMK's Bloch path, OOO via
free-space DMK. Backend support per boundary condition is a capability matrix,
not a type hierarchy. The analytic quasi-2D PPO kernel
$v_{G_\parallel}(z,z') = (2\pi/G_\parallel)\,e^{-G_\parallel|z-z'|}$ is the
designated first open-boundary implementation but is optional within v0.4.

Selection guidance (former "unfreeze triggers", demoted 2026-08-24 to
advisory): non-periodic/low-dimensional boundary conditions; bases without
two-region structure; open-structure interstitial PW blowup (weak — ISDF
$\zeta$ smoothness is set by grid density, decoupled from basis sharpness);
full-potential routes that bypass three-component (e.g. pure-MTO FP-NMTO).

## 4. Auxiliary and empty centers

With a global Coulomb solve $V = 4\pi(-\Delta)^{-1}\rho$, space-filling empty
spheres are no longer a geometric requirement; auxiliary centers are a
convergence knob for the local representation only. The FP-NMTO geometry is

```text
real atomic centers + optional auxiliary/empty centers
```

Placement can be automated: on the octree, evaluate the OMT reconstruction
residual

```math
\epsilon(\mathbf r) = \left|\rho_{\mathrm{tree}}(\mathbf r) - \rho_{\mathrm{OMT}}(\mathbf r)\right|
```

and insert auxiliary centers in the largest void/residual regions. This is a
`basisopt`-shaped discrete search (placement plus radius/channel parameters)
and reuses the M-Q protocol. Boundary: this changes the density/Poisson side
only; the kink/screening hard-sphere geometry is a separate object and is not
affected.

## 5. Milestones

### M-R — NMTO transform and downfolding

- energy mesh and stable divided differences;
- N=0/1 limits cross-checked against exact and LMTO results;
- active/passive Schur downfolding and reconstruction;
- convergence order on semiconductor and transition-metal band windows;
- localized/Wannier-like export.

### M-S — canonical EMTO workflow

- overlapping-potential-sphere geometry and optimization;
- complex-energy kink matrices and derivatives;
- contour Green function, electron count, and DOS sum rules;
- frozen-potential residue/wave projection onto an independent
  `ProductPartition`, without claiming a completed GW workflow;
- ordered-crystal SCA+FCD SCF;
- cross-validation against the public EMTO code on elemental benchmarks.

### M-U — adaptive-grid Coulomb route

- `libmuffintin-pdmk` FFI crate (feature-gated) with build isolation;
- continuous-density quadrature adapter shared with adaptive-grid ISDF;
- weak-form kinetic on the adaptive grid;
- finite-q validation against DUCC Ewald and `libmuffintin-coulomb` Ewald;
- DMK-versus-FFT agreement on the M-O/M-P regression systems;
- auxiliary-center auto-placement prototype on one open structure.

M-R and M-S do not depend on M-U and may proceed in parallel with it.

### M-T — full-potential unification

- `ssw_fp_exact_e` experiment;
- compare SCA+FCD, FP-TB-LMTO, JPO-FP, and exact-energy SSW at matched radial
  potentials and channel cutoffs; include the adaptive-grid + DMK
  representation where M-U has landed;
- compare their MPB/THC product spans and Coulomb/action convergence on the
  common auxiliary-basis interface;
- decide, from derivation and numerical evidence, whether a public
  `workflows::emto_fp()` label is legitimate;
- freeze the v0.4 API only after this comparison.

## 6. Validation matrix

| Invariant | Unit/synthetic test | Cross-code test |
|---|---|---|
| NMTO order | expected product-form interpolation error and N=0/1 limits | Stuttgart NMTO examples |
| EMTO Green function | contour independence, electron-count and moment sum rules | EMTO 5.8 |
| FCD | multipole and total-charge conservation | EMTO total energies |
| DMK Coulomb action | finite-q agreement with Ewald references | DMK vs FFT on M-O/M-P systems |
| adaptive-grid density | octree reconstruction residual convergence | three-component densities |

All cross-family comparisons use matched radial potentials, sphere radii,
angular cutoffs, relativity, XC, k mesh, and occupations, as in v0.3 §8.

## 7. Documentation

Numbered derivation notes (continuing the v0.3 sequence; see v0.3 plan §11):

```text
doc/23_linearized_and_nth_order_mtos.md
doc/24_kink_matrices_and_emto_green_functions.md
doc/25_fcd_sca_and_full_potential_closures.md
doc/26_adaptive_grid_coulomb_and_auxiliary_centers.md
```

## 8. Definition of done for v0.4

1. `transforms::nmto()` N=0/1 limits and downfolding reconstruction are
   verified;
2. ordered-crystal `workflows::emto_sca_fcd()` satisfies contour and charge sum
   rules and matches reference elemental results;
3. the DMK route reproduces three-component results on the M-O/M-P regression
   systems at matched inputs, with the disagreement budget recorded;
4. the auxiliary-center prototype demonstrates automated placement on one open
   structure with a documented residual reduction;
5. the M-T comparison is recorded and the `emto_fp` naming verdict is decided
   from evidence;
6. all inherited v0.3 invariants (conventions, provenance, tensor-substrate
   parity, no private method paths) continue to hold.

## 9. Primary references

- O. K. Andersen, T. Saha-Dasgupta, and S. Ezhov, "Third-generation
  muffin-tin orbitals," arXiv:cond-mat/0203083
- O. K. Andersen and T. Saha-Dasgupta, "Muffin-tin orbitals of arbitrary
  order," arXiv:cond-mat/0010454
- L. Vitos et al., "Application of the Exact Muffin-Tin Orbitals Theory,"
  arXiv:cond-mat/0005313
- EMTO 5.8 documentation, https://emto.gitlab.io/
- A. Zhang et al., Phys. Rev. B 110, 155126 (2024); arXiv:2503.07524
- S. Jiang and L. Greengard, "A dual-space multilevel kernel-splitting
  framework for discrete and continuous convolution," arXiv:2308.00292
- X. Gao, L. Greengard, and S. Jiang, "An Adaptive Fast Algorithm for
  Periodic Coulomb Lattice Sums in Arbitrary Unit Cells," arXiv:2606.28608;
  code: https://github.com/xuanzhaogao/PeriodicDMK
- H. Zhu et al., adaptive-grid ISDF with DMK, arXiv:2510.20826
