# H₂ Hartree–Fock acceptance plan (small box first)

- Workstream ID: `h2-hf`
- Plan version: 1
- Approval: accepted 2026-09-08 (immutable; direction changes make plan.v2)
- Supersedes: none
- Imported: 2026-09-08 from `scratch/h2_hf_plan.md` (last modified 2026-09-08), body unchanged
- Status source: false; state lives in `STATUS.md` and `ledger.md`
- Path map: `scratch/h2_hf_pyscf_refs.json` is `plans/h2-hf/pyscf-rhf-references.json`

Status: plan, 2026-09-08. Follows the LDA acceptance in `examples/h2_dft`
(gates HOMO 1 mHa, total 20 mHa, truncated-step XC, orbital cutoff 7 leaves
0.07 mHa). Nothing below touches XC, mixing, or the LDA example.

Every numerical step is written as a Stop That Digit contract
(`~/.codex/skills/stop-that-digit/SKILL.md`): quantity, reference,
tolerance, class, budget, and a stopping rule are fixed here, before the
first run. A step that passes is closed. A step that fails gets exactly the
diagnostics listed for it and then hands off with `DIGIT / HANDOFF`; nothing
in this plan authorizes changing a tolerance, adding a row, or trying a
different kernel, box, mixer, or cutoff to make a number move.

## 0. Why H₂ HF, and why now

The open Kr HF discrepancy (−2787.682 vs −2788.884 Ha) sits in the VV
exchange sector (VV −2.102 vs −2.926 Ha); CC is settled and CV is small.
H₂ isolates exactly that sector: no core, so no CV/CC; nonrelativistic
through `speed-of-light = 137035.9895`; one doubly occupied orbital, so the
exchange energy is not an unknown but an identity.

For a closed shell of one spatial orbital $\phi$ with $n = 2|\phi|^2$:

```math
E_H = \tfrac12 (n|n) = 2(\phi\phi|\phi\phi), \qquad
E_x = -\tfrac12 \sum_\sigma (\phi\phi|\phi\phi) = -(\phi\phi|\phi\phi)
\quad\Rightarrow\quad E_x = -\tfrac12 E_H .
```

The identity holds for any Coulomb kernel as long as $E_H$ and $E_x$ use the
same kernel and the same gauge. In a periodic box with the $G=0$ term dropped
on both sides it holds term by term over $G \neq 0$. PySCF reproduces the
ratio −0.5 to 1e-15 in the isolated molecule and in the periodic box with
`exxdiv=None` (table below). In libmuffintin the two sides come from two
independent code paths, Weinert electrostatics (`electron_hartree`) and the
mixed-product-basis exchange (`exchange_energy`), so the residual
$|E_x + E_H/2|$ measures product-basis completeness plus kernel consistency
and needs no external reference at all. Koopmans holds in HF, so the HOMO is
exact as well.

## 1. References (PySCF 2.14.0, aug-cc-pV5Z, RHF, `conv_tol` 1e-10)

Periodic rows are Gamma-only cubic cells with Gaussian density fitting
(`aug-cc-pv5z-jkfit`). Energies in Hartree; $E_H$ and $E_x$ from
`get_jk` with the converged density matrix. Raw values are in
`plans/h2-hf/pyscf-rhf-references.json`.

| Reference | $E$ | HOMO | $E_H$ | $E_x$ | $E_x/E_H$ |
|---|---:|---:|---:|---:|---:|
| isolated | −1.1336107 | −0.5946525 | 1.3171828 | −0.6585914 | −0.5000000 |
| box 8, `exxdiv=None` | −0.8124204 | −0.2640980 | 0.5951402 | −0.2975701 | −0.5 |
| box 10, `exxdiv=None` | −0.8624143 | −0.3183767 | 0.7534239 | −0.3767120 | −0.5 |
| box 8, `exxdiv=ewald` | −1.1670826 | −0.6187602 | 0.5951403 | −0.6522323 | −1.096 |
| box 10, `exxdiv=ewald` | −1.1461441 | −0.6021065 | 0.7534239 | −0.6604417 | −0.877 |

Which row matches which libmuffintin kernel:

- `periodic-finite-body` (`CoulombRequest::cubic`, Gamma treatment
  `FiniteBody`, "no divergent head is silently added") is the
  neutralizing-background convention. Its counterpart is the same-box
  `exxdiv=None` row. Total energies in this convention are far from the
  physical HF energy (−0.81 at box 8) and move by 50 mHa between boxes 8
  and 10, so they are only ever compared inside the same box.
- `spencer-alavi-sphere` and `smoothed-spencer-alavi-sphere` truncate the
  kernel at $R_c = (3\Omega/4\pi)^{1/3}$ (4.96, 6.20, 7.44 Bohr for boxes
  8, 10, 12). Their counterpart is the isolated row, up to the electrostatic
  finite-size error of the Hartree side, which the LDA study measured as
  −1.83 mHa at box 10 and −0.27 mHa at box 12.
- `exxdiv=ewald` is not a target. The Madelung correction is right for a
  uniform background, not for a localized molecule; it overshoots the
  isolated energy by 33 mHa at box 8 and 12 mHa at box 10.

The HF limit at $R = 1.4$ Bohr is −1.13363 Ha; aug-cc-pV5Z sits 0.02 mHa
above it, so the isolated row is exact for this purpose.

## 2. What already exists (verified against `main` at 9c870eb)

- Driver: `run_gamma_valence_hf(&mut CheckpointPhysics, &GammaValenceHfSpec)`
  in `crates/mt-runtime/src/hf_scf.rs`. `validate_spec` requires
  `relativity = ScfRelativity::SpinorFirstVariation`, a full Gamma-only
  k mesh, and every `core_sites` entry empty; the Gamma treatment is
  `FiniteBody`. Spec fields: `config`, `product_l_max`, `product_g_max`,
  `overlap_tolerance` (`muffintin_prodbasis::mpb::DEFAULT_TOLERANCE` = 1e-4),
  `coulomb: CoulombRequest`, `max_fock_iterations`,
  `fock_density_tolerance`, `fock_feedback_tolerance`, `fock_mixing`.
- Fixture: `crates/mt-runtime/tests/gamma_valence_hf.rs` runs one hydrogen
  atom through this driver and gates the three internal identities
  (`exchange_energy_identity_residual`, `eigenvalue_identity_residual`,
  `total_energy_identity_residual`) at 1e-8 Ha.
- Energy assembly: `energy_diagnostic` forms
  `total = h0_expectation − electron_hartree + nuclear_nuclear + exchange_energy + occupation.correction`.
  `electrostatic.electron_hartree` is computed every iteration
  (`RegionalElectrostaticResult` in `crates/mt-dft/src/hartree.rs`) but is
  not exposed in `GammaValenceHfIterationDiagnostic` or
  `GammaValenceHfResult`. That is the one missing number.
- Kernels: `CoulombRequest::cubic(box, lexp)` (periodic, `DEFAULT_LEXP` 14),
  `.with_spencer_alavi_sphere(1, InverseBohr(fock_fourier_g))`,
  `.with_smoothed_spencer_alavi_sphere(1, InverseBohr(fock_fourier_g), InverseBohr(omega))`.
  See `exchange_coulomb_request` in
  `crates/mt-runtime/examples/kr_relaxed_core_hf.rs`.
- Geometry: `h2_geometry(box, rmt, log_increment)` in
  `crates/mt-runtime/examples/h2_dft.rs` builds the two-site cell with
  `RadialEquationTag::ScalarKoellingHarmon`. The Kr example's spinor-first
  route uses `RadialBasisSpinV2::Scalar` with
  `RadialEquationTag::FullyRelativisticDirac`; with $c$ = 137035.9895 both are
  nonrelativistic to 1e-6 relative. Use the Dirac tag to stay on the path
  the driver is tested on.
- Cost: spinor-first doubles the plane-wave dimension. Plane-wave counts
  $N \approx \Omega G^3/6\pi^2$:

  | box | orbital $G$ | $N_{pw}$ | spinor dimension |
  |---:|---:|---:|---:|
  | 8 | 4 | 550 | 1100 |
  | 8 | 5 | 1080 | 2160 |
  | 8 | 6 | 1870 | 3740 |
  | 10 | 6 | 3650 | 7300 |

  The LDA orbital-cutoff-8 run (8600 dimensions) was killed for memory on the
  24 GB machine before its first iteration, while the dense matrices alone
  are 1.2 GB each. Box 10 at orbital 6 in spinor form is in that range. A
  memory kill is a `DIGIT / HANDOFF` with the dimension recorded; it is not
  fixed inside this plan.

## 3. Deliverables (code, in commit order)

1. `feat(runtime)`: add `h0_expectation`, `electron_hartree`,
   `electron_nuclear`, `nuclear_nuclear` (Hartree) to
   `GammaValenceHfIterationDiagnostic` and `GammaValenceHfResult`, filled
   from the values `energy_diagnostic` already receives. Print nothing new
   in the fixture beyond these fields; do not add a gate on them there.

   ```text
   digit: Q=exchange/eigenvalue/total identity residuals (Ha) ref=fixture tol=1e-8 class=R
   ```

   Verify with `cargo test -p libmuffintin-runtime --test gamma_valence_hf`,
   once. Pass closes the step.

2. `feat(examples)`: `crates/mt-runtime/examples/h2_hf.rs`, mirroring
   `h2_dft.rs`. Reuse `h2_geometry`. Arguments: output directory, box,
   orbital $G$, field $G$, product $G$, product $l_{max}$, overlap
   tolerance, `--exchange-coulomb` (same three names as the Kr example),
   `--fock-fourier-g`, `--fock-smoothing-omega`, `--lexp`, speed of light,
   RMT. Defaults: box 8, RMT 0.65, 401 radial points, $T$ = 1 mHa,
   $c$ = 137035.9895, `lexp` 14, `FockMixing` CDIIS as in the Kr example.
   Start from the neutral atomic superposition exactly as `h2_dft` does; do
   not add restart plumbing. Print two machine-readable lines at the end:

   ```text
   hf_energy_terms_ha h0=… electron_hartree=… nuclear_hartree=… exchange=… occupation_correction=… band=… total=…
   hf_identity_ha exchange=… eigenvalue=… total=… hartree_exchange=…
   ```

   where `hartree_exchange` is $|E_x + E_H/2|$. Print the HOMO from
   `orbital_energies` and the wall time as `h2_dft` does.
3. `feat(examples)`: `examples/h2_dft/hf_reference.py` (isolated RHF plus
   periodic RHF with `exxdiv=None` for a box given on the command line,
   JSON shaped like the LDA scripts plus `e_hartree`, `e_exchange`,
   `exchange_ratio`) and an HF mode in `examples/h2_dft/compare.py` reading
   the `hf_energy_terms_ha` line with the gates of section 5. The section 1
   values came from exactly this recipe:

   ```python
   mol = gto.M(atom="H 0 0 -0.7; H 0 0 0.7", unit="Bohr", basis="aug-cc-pv5z")
   mf = scf.RHF(mol).run(conv_tol=1e-11)
   dm = mf.make_rdm1(); vj, vk = mf.get_jk(mol, dm)
   e_h = 0.5 * einsum("ij,ji", vj, dm); e_x = -0.25 * einsum("ij,ji", vk, dm)
   # periodic: pbc.gto.Cell, a = L*eye(3), pbc.scf.RHF(cell, exxdiv=None).density_fit(auxbasis="aug-cc-pv5z-jkfit")
   ```

   Run with `PYTHONPATH=/tmp/libmuffintin-h2-python /opt/homebrew/bin/python3`
   (`unalias python python3` first). Box 8 takes 25 s, box 10 takes 15 s.
   The regenerated JSON must reproduce section 1 to 1e-6 Ha; if it does not,
   that is a `DIGIT / FAIL` on the reference script, not a reason to touch
   the LAPW side.
4. `docs(examples)`: a "Hartree–Fock" section in `examples/h2_dft/README.md`
   holding the tables of sections 4 and 6 and the stamps; logs under
   `examples/h2_dft/results/hf-*.log`, named like the LDA logs.

Conventional Commits with a body for every `feat`; no attribution trailers;
focused tests only; build the example with `--features fft-fftw` like the
LDA one.

## 4. Numerical steps

Common settings unless a row says otherwise: box 8, RMT 0.65,
$c$ = 137035.9895, $T$ = 1 mHa, `lexp` 14, `periodic-finite-body`. Every
run records $E$, HOMO, $E_H$, $E_x$, the three driver identities,
`hartree_exchange`, Fock iterations, and wall time, as one table row.

### Step A0, smoke (scope=verify)

Orbital 4, field 12, product 4, product $l$ 2, TOL 1e-4.

```text
digit: Q=driver identity residuals (Ha) ref=0 tol=1e-8 class=A
validity: SCF converged, finite energies, electron count 2 to 1e-8
budget: 1 run + at most 3 diagnostics
```

A pass closes A0 regardless of the `hartree_exchange` value; that number is
recorded, not judged here. On failure the permitted diagnostics, in order,
each answering one question:

1. Same run with `max_fock_iterations` doubled. Question: convergence limit
   or defect? A pass here is a pass of A0 with the larger limit recorded.
2. Same run at orbital 3 / product 3. Question: dimension-dependent defect
   or setup error? Only informative if it changes which identity fails.
3. The existing `gamma_valence_hf` hydrogen-atom fixture with
   `product_g_max` raised to 4. Question: does the driver itself hold its
   identities at this product cutoff, which separates a two-site setup
   error in `h2_hf.rs` from a driver defect?

After these, `DIGIT / HANDOFF` with the failing identity and its value.

### Step A1, identity floor (scope=study, explicitly requested here)

Base: orbital 5, field 12, product 6, product $l$ 4, TOL 1e-4. Rows are
the base plus one change each; the list is closed.

| row | change from base |
|---|---|
| 1 | base |
| 2 | product $G$ 4 |
| 3 | product $G$ 8 |
| 4 | product $G$ 10 |
| 5 | product $l$ 2 |
| 6 | product $l$ 6 |
| 7 | TOL 1e-5 |
| 8 | TOL 1e-6 |
| 9 | `lexp` 18 |
| 10 | field 18 |

```text
digit: Q=hartree_exchange (Ha) ref=0 tol=none (study) class=P
stopping rule: run the ten rows once each and stop; do not add rows.
```

Skip a remaining row on an axis only when the previous row on that axis
moved Q by less than 0.05 mHa, and say so in the table. The study output
is the table and one sentence naming the row with the smallest Q and the
axes that still moved it by more than 0.05 mHa on their last step. No
verdict is attached to the study itself.

### Step A1v, identity verdict (scope=verify)

Take the settings of the smallest-Q row of A1, with field 18 and the best
value of every axis that was still moving (that row may coincide with an
A1 row; then no new run is needed).

```text
digit: Q=hartree_exchange (Ha) ref=0 tol=5e-4 class=A
budget: 1 run (or reuse) + at most 3 diagnostics
```

On failure, diagnostics in order, one run each, each with its question:

1. Same settings at box 10. Question: does Q scale with the box (kernel or
   gauge mismatch between Weinert and MPB) or stay (product-basis
   completeness)?
2. Same settings with `lexp` 24. Question: is the Weinert multipole
   truncation on the $E_H$ side the missing piece?
3. Same settings with `--exchange-coulomb spencer-alavi-sphere` and
   `--fock-fourier-g` equal to product $G$. Question: does the residual
   change by more than its own size, which would place it in the periodic
   Gamma body treatment rather than in the product basis? This diagnostic
   changes the kernel on one side only, so its Q is not expected to be
   zero; only the change of Q is read.

After these, `DIGIT / HANDOFF` with the table of the three diagnostics
and the one competing explanation they did not separate. Do not try box
12, other omegas, other TOLs, other mixers, or a different radial mesh.

### Step A2, same-box external (scope=verify)

Settings of A1v (whether it passed or handed off; state which). Reference
is the box-8 `exxdiv=None` JSON produced by `hf_reference.py`.

```text
digit: Q=E_H, E, HOMO (Ha) ref=hf-periodic-reference-box8.json
       tol=E_H 5e-4, E 2e-3, HOMO 1e-3 (absolute) class=A
budget: 1 run (reuse A1v) + at most 2 diagnostics
```

All three must pass; do not report the one that passed. On failure:

1. Orbital 6 (spinor dimension 3740). Question: plane-wave basis, as in the
   LDA study where orbital 6 to 7 was worth 0.74 mHa?
2. Field 24. Question: electrostatics, as in the LDA study where field
   18 to 24 was worth 0.14 mHa?

Then `DIGIT / HANDOFF`. Orbital 7 is not authorized here (LDA orbital 7
cost 1314 s scalar; spinor is eight times that).

### Step B, kernel study (scope=study, explicitly requested here)

Settings of A1v, orbital 5. Rows are closed.

| row | kernel | box | `--fock-fourier-g` | omega |
|---|---|---:|---:|---:|
| 1 | spencer-alavi-sphere | 8 | product $G$ | |
| 2 | spencer-alavi-sphere | 8 | 2 × product $G$ | |
| 3 | spencer-alavi-sphere | 10 | product $G$ | |
| 4 | spencer-alavi-sphere | 12 | product $G$ | |
| 5 | smoothed-spencer-alavi-sphere | 8 | product $G$ | 0.8 |
| 6 | smoothed-spencer-alavi-sphere | 8 | product $G$ | 1.6 |
| 7 | smoothed-spencer-alavi-sphere | 8 | product $G$ | 3.2 |
| 8 | smoothed-spencer-alavi-sphere | 12 | product $G$ | 0.8 |

```text
digit: Q=E_x and E (Ha) ref=isolated (E_x −0.6585914, E −1.1336107)
       tol=none (study) class=P
stopping rule: run the eight rows once each and stop.
```

Row 4 at box 12 is 3200 plane waves scalar, 6400 spinor; if it is killed
for memory, record that and continue with the remaining rows.

### Step Bv, kernel verdict (scope=verify)

```text
digit: Q=E_x(row 4) − (−0.6585914) (Ha) ref=isolated tol=1e-3 class=A
finding: |E_x(row 8) − E_x(row 4)| compared with |Q|
budget: 0 additional runs
```

Pass or fail, the finding line is reported as is: if the omega 0.8 kernel
at box 12 differs from the sharp kernel at box 12 by more than the sharp
kernel's own residual against the isolated value, the smoothed kernel is
the first Kr suspect. That is a report, not a fix; no omega tuning, no
extra rows.

## 5. Gates, fixed now

| Quantity | Gate | Step |
|---|---|---|
| driver identities (exchange, eigenvalue, total) | ≤ 1e-8 Ha | A0, every run |
| $\lvert E_x + E_H/2\rvert$ | ≤ 5e-4 Ha | A1v |
| $E_H$ vs same-box `exxdiv=None` | ≤ 5e-4 Ha | A2 |
| $E$ vs same-box `exxdiv=None` | ≤ 2e-3 Ha | A2 |
| HOMO vs same-box `exxdiv=None` | ≤ 1e-3 Ha | A2 |
| $E_x$ (sharp Spencer–Alavi, box 12) vs isolated | ≤ 1e-3 Ha | Bv |

These are not revised after seeing results. 50 mHa is not a gate here: H₂
HF has no XC grid, the plane-wave basis was worth 0.8 mHa in LDA at
orbital 6, and the isolated reference is exact to 0.02 mHa. The Kr VV error
is 28 % of the sector; on H₂'s $E_x$ of −0.66 Ha even a 5 % product-basis
error is 33 mHa, which a 50 mHa gate would pass.

## 6. Report format

One stamp per step in the README section and in the final message, in the
skill's form, for example:

```text
DIGIT / PASS
Q: hartree_exchange (Ha); class: A; ref: 0
bound: 5e-4; Delta: 1.2e-4; d: 0.24
checks: driver identities ≤ 1e-8; runs: 1; numerical verification closed
```

A handoff names the one unresolved question and the runs that were made.
Total run count for the whole plan if nothing fails: 1 + 10 + 1 + 8 = 20
runs at box 8 to 12, orbital 4 to 5, all minutes each on the 10-thread
machine except the box-12 rows.

## 7. Boundaries

- Do not modify `xc_field.rs`, the mixers, `h2_dft.rs`, or the LDA gates.
- Do not adjust gates, add rows, or change kernels, boxes, mixers, radial
  meshes, or cutoffs outside the listed diagnostics.
- A memory kill records the dimension and hands off.
- Keep the LDA references and logs untouched; new JSON files are
  `hf-reference.json`, `hf-periodic-reference-box8.json`, and so on.
