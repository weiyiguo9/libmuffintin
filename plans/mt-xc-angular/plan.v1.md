# Production MT XC angular quadrature correctness

- Workstream ID: `mt-xc-angular`
- Plan version: 1
- Approval: proposed; user authorized recording and publication on 2026-09-10, not implementation or numerical runs
- Supersedes: none
- Related: `diamond-omt`, ledger `evt-0063`, `evt-0064`

## 1. Problem and provenance

Treat the reported symmetry-breaking MT XC projection as a production
numerical-correctness bug, not merely a coarse-grid tuning issue. The supplied
report and repair proposal are the source for this plan; source locations and
numbers below were not independently rechecked during harness ingestion.
Context revisions are libmuffintin `5f5dc92c7bbc9d562bfc53e65e23cc9082a5f34e`
and pymuffintin `303beef281a03d264903436d36b4ce78b174a26e` from evt-0063,
not proven experiment revisions. Recover the actual checkpoint, commands,
logs, crystal orientation, and paired source revisions before validation.

Reported observations:

| Quantity | Supplied observation |
|---|---|
| Γ25′ in LAPW and NMTO | 13.571 / 13.631 / 13.788 eV rather than strict triple degeneracy; Γ15 splitting about 0.16 eV |
| Checkpoint density | MT l=1,2 about 1e-15; l=4 only m=0,±4 with ratio √(5/14); interstitial star spread 1e-15, forbidden ρ(002) 3e-10 |
| Interstitial potential | Cubic, star spread about 1e-15 |
| MT potential at r=1.42 | Forbidden l=1 about 1.7e-3 Ha, l=2 m=±1 about 1.4e-2 Ha, l=4 m=±1 about 2.8e-2 Ha; replacing PBE with LDA barely changes them |
| Reproduced LDA exchange projection on 50 Fibonacci points | (1,0) +0.0015; (2,1) +0.0090−0.0087i; (4,1) +0.0239−0.0217i Ha |
| Exported potential at the same channels | +0.0017; +0.0105−0.0099i; +0.0277−0.0247i Ha; accurate projection reportedly leaves only about 1e-15 |
| Allowed cubic component (4,0) | 0.0315 versus 0.0284 Ha, roughly 10% error |

Reported code pointers in libmuffintin: `crates/mt-dft/src/xc_field.rs:1926`
sets `angular_point_count = ((l_max + 1)^2 * 2).max(50)`; scalar-block and
full regional paths near lines 363 and 657 construct Fibonacci grids.
The `doc/07_grids…md` discussion reportedly describes Fibonacci as a test
fallback without a polynomial exactness degree and calls for Lebedev in
production. `build_regional_potential(density, xc, noncollinear_route)` is
shared by LAPW and NMTO; the Python interface reportedly exposes no angular
control, while `angular-points` exists only for molecular input.

## 2. Mathematical contract and limitations

Replace the production Fibonacci rule with a weighted Lebedev rule of
explicit algebraic exactness degree. Nodes and weights must change together:

```math
V_{\ell m}(r) \approx \sum_a w_a Y_{\ell m}^{*}(\hat r_a)
v_{\mathrm{xc}}(r,\hat r_a), \qquad \sum_a w_a = 4\pi.
```

For spherical harmonics truncated at $L$, $t \ge 2L$ is the degree floor
for exact harmonic inner products, not an XC accuracy guarantee. LDA's
nonlinearity and PBE's gradient dependence require a separate accuracy
contract for allowed coefficients. Quadrature precision must be independent
of the potential's `l_max`.

Lebedev's octahedral node/weight symmetry protects the reported cubic case
when grid and crystal axes align and pointwise XC evaluation preserves that
symmetry. It does not guarantee exact symmetry for arbitrary orientations
or site groups. Respect the actual tetrahedral site group; do not replace it
with the full octahedral group or delete all odd-l coefficients. Increasing
Fibonacci point counts is not the structural fix. Post-projection removal
of forbidden components cannot repair allowed-component integration errors.

## 3. Implementation scope, pending authorization

1. Provide an explicit-degree weighted Lebedev rule through the existing
   `AngularGrid` interface. The reported implementation has no built-in
   Lebedev table and requires positive weights; select compatible rules
   deliberately, since not all Lebedev tables have positive weights.
2. Apply one rule-selection contract to scalar-block and full regional XC
   paths. Use consistent weights for XC energy, potential projection, and
   the density–XC-potential integral. Cover the corresponding XC spherical
   average in `crates/mt-dft/src/core_potential.rs` (reported near line 90).
3. Expose an explicit rule or exactness degree through native/Python inputs
   and pymuffintin TOML, separately from potential `l_max`. Decide the exact
   interface and default rule before coding; choose the production default
   using the bounded validation below, not a claimed linear exactness bound.
4. Fix the shared implementation, not separate LAPW/NMTO patches. Do not
   replace unrelated THC grids or expand this work into empty-sphere or
   radial-basis implementation. Update affected quadrature documentation.

## 4. Acceptance design, not yet runnable

Before the first run, register concrete gate IDs, quantities, reference
artifacts, tolerances, commands, and finite run/diagnostic budgets in
`gates.md`. Numeric tolerances, reference rule, candidate default rules,
and run limits are unresolved; this plan does not invent acceptance values
or authorize exploratory sweeps.

| Stage | Fixed comparison | Required distinction |
|---|---|---|
| Quadrature core | Weight normalization and harmonic inner products through the selected degree | Core production rule contract, not proof of nonlinear XC accuracy |
| Fixed checkpoint density | Forbidden MT coefficients and allowed coefficients against a specified high-accuracy angular reference; LDA and PBE scope fixed before running | Symmetry residual and allowed-component integration error are separate quantities |
| Frozen-potential Γ spectrum | Build the corrected potential from the same density, then freeze it for the LAPW/NMTO eigensolves; compare target multiplet splittings | No SCF density relaxation mixed into the integration-fix test |

Run only focused affected-path checks. A passing contract is closed; a
failure gets the pre-agreed smallest discriminating diagnostics and then a
handoff, not extra precision or unbounded parameter scans. No NaN/Inf or
execution failure counts as a pass. Record paired revisions, exact commands,
and original logs in evidence entries when those runs actually happen.

## 5. Interpretation boundaries

The supplied reproduction is strong reported causal evidence, but the
fraction of the earlier native LAPW–SPEX 0.14 eV discrepancy attributable to
this bug is unknown until a controlled before/after comparison. Earlier
3×3×3 comparisons are affected in scope, not retroactively corrected here.
Likewise, both LAPW and NMTO using the same faulty path does not prove exact
error cancellation between methods. Preserve evt-0063's OMT observations,
but do not promote its expected cancellation or minimal-basis explanation
to established conclusions on the strength of this record.

No implementation, numerical rerun, gate pass, or quantitative SPEX
attribution is claimed by this plan.
