# Acceptance gates

A gate is a quantity, a reference, and a bound fixed before the first run.
Bounds are never revised after a result is seen; a different bound is a new
gate ID recorded with an event in `ledger.md`. Unit-test tolerances inside
the crates on `main` are not gates and are not listed here.

| ID | Workstream | Quantity | Reference | Bound | Enforced by (on `main`) | State |
|---|---|---|---|---|---|---|
| G-H2-LDA-1 | `h2-lda` | HOMO eigenvalue, LDA-PW92, matched box | periodic PySCF, `examples/h2_dft/periodic-reference.json` | 1e-3 Ha | `examples/h2_dft/compare.py` | closed, evd-0001 |
| G-H2-LDA-2 | `h2-lda` | total energy, matched box | same | 2e-2 Ha | `examples/h2_dft/compare.py` | closed, evd-0001 |
| G-H2-HF-0 | `h2-hf` | exchange, eigenvalue, and total identity residuals | 0 | 1e-8 Ha | `crates/mt-runtime/tests/gamma_valence_hf.rs` | proposed |
| G-H2-HF-1 | `h2-hf` | $\lvert E_x + E_H/2 \rvert$ at the converged product basis | 0 | 5e-4 Ha | `compare.py` HF mode (planned) | proposed |
| G-H2-HF-2 | `h2-hf` | $E_H$, box 8, field 18 | periodic PySCF RHF `exxdiv=None` | 5e-4 Ha | same | proposed |
| G-H2-HF-3 | `h2-hf` | total energy, box 8 | same | 2e-3 Ha | same | proposed |
| G-H2-HF-4 | `h2-hf` | HOMO, box 8 | same | 1e-3 Ha | same | proposed |
| G-H2-HF-5 | `h2-hf` | $E_x$, sharp Spencer–Alavi kernel, box 12 | isolated PySCF RHF, −0.6585914 Ha | 1e-3 Ha | same | proposed |
| G-KR-HF | `kr-hf` | frozen-SRA total energy | GTO 4c-DC-HF, −2788.884 Ha | none set | `examples/relativistic_hf/` | open; 1.2 Ha gap as reported 2026-09-05 |

## Discipline

Every gate run follows Stop That Digit
(`~/.codex/skills/stop-that-digit/SKILL.md`):

- The contract is fixed first: quantity, units, reference, bound, class
  (R refactor, A algorithm, P parameter), budget.
- A difference inside the bound is closed even when unexplained. No
  roundoff proof, no extra precision, no sweep, no second backend.
- A failure gets at most the diagnostics its plan names, each stating the
  two explanations it separates, then `DIGIT / HANDOFF`.
- A study (parameter ladder) is a closed row list with a stopping rule and
  carries no verdict of its own; the verdict is a separate verify step.
- Bounds are not derived from noise, SCF residuals, or a wish for
  reassurance. `NaN`, non-convergence, and wrong electron counts fail
  regardless of the bound.

## Evidence entry

```text
## <date> · evd-NNNN · <workstream> <gate IDs> · main <sha>
DIGIT / PASS|FAIL|HANDOFF
Q: <quantity> (<unit>); class: <R|A|P>; ref: <reference>
bound: <b>; Delta: <d>; d: <Delta/bound>
command: <exact command>
log: <path on main or under evidence/>
scope: <what this run proves and what stays open>
```
