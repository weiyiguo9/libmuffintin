# Hartree–Fock is an `xc` kind of the `dft-scf` task

- Status: accepted 2026-09-08 (user decision)
- Date: 2026-09-08

## Decision

Hartree–Fock enters libmuffintin through the same versioned input file and
the same `dft-scf` task as density-functional SCF. It is selected by the
existing `xc` field, `kind = "exact-exchange"`, the way `lda-pw92` and `pbe`
are selected today. There is no separate task kind, no separate runner entry,
and no per-example construction of HF specs.

The driver family is a function of the input, not of the caller:

| `xc` | `relativity` | core channels in `basis.channels` | driver on `main` |
|---|---|---|---|
| `exact-exchange` | `spinor-first-variation` | none | `run_gamma_valence_hf` (Gamma) or `run_valence_hf` (k mesh) |
| `exact-exchange` | `spinor-first-variation` | present | `run_relaxed_core_hf` |
| `exact-exchange` | `soc-second-variation` | none or frozen | `run_kh_soc_valence_hf` |
| `exact-exchange` | `scalar` | any | rejected at input validation |

## Why

The DFT path reached the input schema first and its variants are one enum
field. HF grew as a sequence of research probes (Gamma valence, KH+SOC
valence, relaxed core) that each kept a spec struct and an example that
rebuilds geometry and kernel by hand; `h2_dft.rs`, `kr_relaxed_core_hf.rs`,
and `h2_hf.rs` carry three copies of the same molecule-in-box setup. With
exact exchange as an `xc` kind, a molecule in a box is one input file for
either method, and the inputs of the two methods differ in one table.

## Consequences

- The three spec structs stay as the lowering target; consolidating their
  four Fock loops is a later workstream, not this decision.
- Examples consume the input and print; they do not construct specs.
- The atomic-superposition start for an `exact-exchange` task uses the same
  LDA start as the DFT task; the choice is stated in the plan, not derived.
- Tracked as workstream `hf-input`, `plans/hf-input/plan.v1.md`, sequenced
  after `h2-hf` reaches `closed` or `handoff`.
