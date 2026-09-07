# Ledger

Append-only, newest last. `evt` entries record state changes, `evd` entries
record gate runs in the form given in `gates.md`. A wrong entry is
superseded by a later entry that names it; nothing is edited or deleted.

## 2026-09-08 · evt-0001 · harness initialized · actor: claude

Orphan `harness` branch created from the `graft-rs` model with the reduced
record set of `README.md` and ADR-0001. No gate is claimed by this entry.

## 2026-09-08 · evt-0002 · scratch plans and decisions imported · actor: claude

Imported from the git-excluded `scratch/` of the `main` checkout, bodies
unchanged, each with a header naming its workstream, approval, and source:
`v0.1-lapw-foundation` (accepted, historical, closed), `v0.2-isdf-thc`
(accepted; M-A to M-Kc closed on `main`, M-L open), `v0.3-mto-family`
(draft), `v0.4-emto-nmto` (draft), `ctf-slate-binding` (draft), `h2-hf`
(proposed). Design notes imported as ADR-0002 (orbital configuration V2,
closed 2026-08-24), ADR-0003 (MTO density, Poisson, and kinetic axes,
superseded 2026-08-27 by the v0.3 and v0.4 plans), and ADR-0004
(relativistic core–valence exchange, recorded 2026-08-28). Scratch evidence
imported under `evidence/` with its original dates. Approval states are the
ones the source files carried; no plan was accepted by this import.

## 2026-09-07 · evd-0001 · h2-lda G-H2-LDA-1 G-H2-LDA-2 · main a1fdb87 (XC change), recorded through 9c870eb860f07e27abd34b761b90154450cd4adc

```text
DIGIT / PASS
Q: HOMO eigenvalue, total energy (Ha); class: A; ref: examples/h2_dft/periodic-reference.json (10 Bohr box)
bound: 1e-3 / 2e-2; Delta: 1.71e-4 / 8.12e-4; d: 0.17 / 0.04
command: cargo run --release -p libmuffintin-runtime --features fft-fftw --example h2_dft -- <out> 10 6 18 137035.9895 78
log: examples/h2_dft/results/box10-field18-grid78-step.log (main)
scope: box 10, orbital cutoff 6, field cutoff 18, truncated-step interstitial XC. The
0.8 mHa total residual was decomposed on main (examples/h2_dft/README.md): orbital
cutoff 6 to 7 removes 0.74 mHa, field cutoff is converged to 0.1 mHa, PySCF-side
errors are below 0.02 mHa. No HF or Kr claim.
```

## 2026-09-08 · evt-0003 · h2-hf plan.v1 proposed · actor: claude

`plans/h2-hf/plan.v1.md` written as Stop That Digit contracts (steps A0,
A1, A1v, A2, B, Bv; gates G-H2-HF-0 to G-H2-HF-5). Reference data in
`plans/h2-hf/pyscf-rhf-references.json`. Awaiting the user's acceptance
event before Codex starts; no run has been made.

## 2026-09-08 · evt-0004 · kr-hf registered as open · actor: claude

Workstream registered without a plan on record. State as reported on
2026-09-05 and not re-verified here: converged frozen-SRA Kr total
−2787.682 Ha against GTO 4c-DC-HF −2788.884 Ha; the VV exchange sector is
0.82 Ha too weak, CV 0.12 Ha too strong, CC matches after the smoothed
Spencer–Alavi kernel change; every run used box 8 and omega 0.8. The `h2-hf`
workstream is the sector-targeted test for this gap.

## 2026-09-08 · evt-0005 · ctf-slate-binding superseded · actor: user

The CTF/SLATE binding plan is retired. Its role is taken by the standalone
`ctf-rs` repository (`~/tmp/ctf-rs`, a Rust port of cc4s CTF pinned at
`f69cbb46e23bc2f39cda5722ce096f56301dab4f`) with its own coverage inventory
and validation record; nothing from it is tracked on this branch. The plan
file stays as history with a superseded header.

## 2026-09-08 · evt-0006 · ctf-rs plan.v1 proposed; ctf-slate-binding frozen · actor: user

The user asked for the dense-first order to live here as a plan rather than
as an edit of ctf-rs `docs/coverage.md`. `plans/ctf-rs/plan.v1.md` (B0, D1 to
D6, S1; class R contracts against the pinned upstream) awaits acceptance.
`plans/ctf-slate-binding/plan.v1.md` is frozen: superseded, no further edits.
This restates evt-0005 with the new plan path.

- state: ctf-rs = proposed
- note: ctf-rs = plan.v1 dense-first; per-run evidence stays in ctf-rs docs/validation.md
- state: ctf-slate-binding = frozen
- note: ctf-slate-binding = superseded by plans/ctf-rs/plan.v1.md

## 2026-09-08 · evt-0007 · STATUS.md generated from ledger state lines · actor: claude

`update-status.py` renders STATUS.md from `- state:` and `- note:` lines in
this ledger, the plan headers, and the main tip. The current states are
restated here so the generator has a complete input; nothing changes state.

- state: v0.1-lapw-foundation = closed
- note: v0.1-lapw-foundation = superseded by v0.2
- state: v0.2-isdf-thc = active (M-A to M-Kc closed, M-L open)
- note: v0.2-isdf-thc = first M-L probe in evidence/2026-08-28-ml0-adaptive-rank-probe/
- state: v0.3-mto-family = draft, not authorized
- state: v0.4-emto-nmto = draft, not authorized
- note: v0.4-emto-nmto = inherits v0.3
- state: h2-lda = closed 2026-09-07
- note: h2-lda = evd-0001; gates in examples/h2_dft/README.md on main
- state: h2-hf = proposed
- note: h2-hf = awaiting user acceptance before Codex starts
- state: kr-hf = open
- note: kr-hf = 1.2 Ha total-energy gap in the VV sector as reported 2026-09-05; box and omega axes untested

## 2026-09-08 · evt-0008 · ctf-rs plan.v1 accepted · actor: user

The user accepted `plans/ctf-rs/plan.v1.md` and handed it to Codex on MSI
(`D:/projects/ctf-rs`, harness mirror read-only). Authorized: B0, then D1 to
D6 in order; S1 stays deferred. Milestone closes and handoffs arrive as
ctf-rs commits plus `docs/validation.md` sections and are recorded here as
`evd` entries with the ctf-rs SHA.

- state: ctf-rs = active: B0 authorized, D1 to D6 in order; S1 deferred
- note: ctf-rs = plan.v1 accepted 2026-09-08; Codex on MSI executes; evidence in ctf-rs docs/validation.md

## 2026-09-08 · evt-0009 · two-writer protocol adopted · actor: user

Codex on MSI may now write harness records itself instead of handing lines
back: it commits on `harness-msi` in `D:/projects/libmuffintin-harness` with
ledger IDs from 1001, pushes that branch, and the Mac merges it into
`harness`. `ledger.md` and `STATUS.md` are union-merged and `STATUS.md` is
regenerated after each merge (README "Two writers"). The ctf-rs milestone
records of evt-0008 therefore arrive as `evd-1xxx` entries.

## 2026-09-08 · evt-0010 · h2-hf plan.v1 accepted · actor: user

`plans/h2-hf/plan.v1.md` is accepted and immutable. Codex on the Mac
executes it against `main` at 69dafb8: deliverables 1 to 4, then steps A0,
A1, A1v, A2, B, Bv as Stop That Digit contracts. Gates G-H2-HF-0 to 5 are
open with the bounds fixed in `gates.md`. Verify steps arrive as `evd`
entries from evd-0002; the closing `evt` carries `closed` or `handoff`.

- state: h2-hf = active: deliverables 1 to 4, then A0 to Bv in order
- note: h2-hf = plan.v1 accepted 2026-09-08; Codex on the Mac executes; logs in examples/h2_dft/results/hf-*.log
