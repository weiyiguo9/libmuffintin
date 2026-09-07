# Exact exchange as an input kind (`hf-input`)

- Workstream ID: `hf-input`
- Plan version: 1
- Approval: proposed (awaiting user acceptance; starts after `h2-hf` is closed or handed off)
- Supersedes: none
- Decision: `decisions/ADR-0005-hf-is-an-xc-kind-of-dft-scf.md`

Status: plan, 2026-09-08, written against `main` at 3ef6bca. Pure
refactor: every number that exists today is reproduced exactly; no gate on
physics is added or moved.

## 1. Schema (input version stays 3; all additions are optional or new enum variants)

```toml
[task.scf.xc]
kind = "exact-exchange"
product-l-max = 4
product-g-max = 6.0            # Bohr^-1
overlap-tolerance = 1e-4       # muffintin_prodbasis::mpb::DEFAULT_TOLERANCE
gamma = "finite-body"          # GammaExchangeTreatment; "reject" is the other value

[task.scf.xc.kernel]
kind = "periodic-finite-body"  # | "spencer-alavi-sphere" | "smoothed-spencer-alavi-sphere"
lexp = 14
# fourier-g = 6.0              # required by the two sphere kinds (Bohr^-1)
# smoothing-omega = 0.8        # required by the smoothed kind (Bohr^-1)

[task.scf.fock]                # required when xc.kind = "exact-exchange", rejected otherwise
max-iterations = 40
density-tolerance = 1e-7
feedback-tolerance = 1e-8      # Ha
mixing = { kind = "commutator-diis", history = 6, startup-steps = 2, damping = 0.5 }
#        { kind = "linear", alpha = 0.3 } | { kind = "pulay-anderson", alpha = 0.3, history = 6 }

[task.scf.fock.second-variation]   # only with relativity = "soc-second-variation"
hartree-update = "outer-density"   # | "coupled-fock"
commutator-tolerance = 1e-8
scalar-mixing = { kind = "linear", alpha = 0.3 }
spinor-mixing = { kind = "linear", alpha = 0.3 }
virtual-level-shift = 0.0
core-treatment = "valence-only"    # | "frozen"

[task.scf.core]                    # only when basis.channels marks core states
max-iterations = 60
energy-tolerance = 1e-9
radial-tolerance = 1e-9
sector-numerical-tolerance = 1e-8
maximum-shell-spill = 1e-6
```

Field names map one to one onto `GammaValenceHfSpec`,
`KhSocValenceHfSpec`, and `RelaxedCoreHfSpec` in
`crates/mt-runtime/src/hf_scf.rs` and onto `CoulombRequest::cubic`,
`with_spencer_alavi_sphere`, `with_smoothed_spencer_alavi_sphere`. The
existing `mixing` and `convergence` tables keep their meaning (outer density
loop). `kernel.kind` values are the three names `kr_relaxed_core_hf.rs` and
`h2_hf.rs` already accept on the command line.

## 2. Deliverables (commit order)

1. `feat(runtime)`: `ExchangeCorrelation::ExactExchange` and the `fock`,
   `fock.second-variation`, `core` tables in `crates/mt-runtime/src/input.rs`
   with `deny_unknown_fields` validation: `fock` present iff exact exchange;
   `second-variation` iff `soc-second-variation`; `core` iff core channels;
   `relativity = "scalar"` with exact exchange is an `InputError`.
2. `feat(runtime)`: lowering in `crates/mt-runtime/src/runner.rs` from
   `Task::DftScf` to the three spec structs (`scf_config` is reused
   unchanged for the `config` field) and dispatch in `execute_prepared_with`
   to the driver named by the ADR table. The `molecule` start keeps
   `ScfExchangeCorrelation` LDA-PW92 for an exact-exchange task, exactly as
   `h2_hf.rs` does today.
3. `test(runtime)`: `crates/mt-runtime/tests/input_exact_exchange.rs`,
   one case per row of the table in section 3: parse the listed TOML, lower
   it, `assert_eq!` against the spec the listed fixture or example builds by
   hand today (the specs derive `PartialEq`). No driver runs in this test.
4. `refactor(examples)`: `crates/mt-runtime/examples/h2_hf.rs` loads
   `examples/h2_dft/molecule-hf.toml` through `prepare_input`, runs the
   dispatched driver, and keeps only its printing; `h2_geometry` and the
   spec construction are deleted. `kr_relaxed_core_hf.rs` likewise drops
   `exchange_coulomb_request`, the spec literals, and the geometry build in
   favor of `examples/relativistic_hf/kr-*.toml`; its diagnostics printing
   stays. Command-line flags that only duplicated schema fields are removed;
   the box, cutoffs, and kernel flags used by the `h2-hf` studies remain as
   overrides applied to the parsed `Input` before lowering.
5. `docs`: the `dft-scf` input description on `main` (README input section
   and the numbered document that already describes the task) gains the
   `exact-exchange` kind and the three tables; `examples/h2_dft/README.md`
   points at `molecule-hf.toml`.

Conventional Commits with a body for every `feat`/`refactor`; no
attribution trailers; focused tests only.

## 3. Acceptance (class R, no physics runs except G-HFI-2)

| row | input | hand-built reference | driver |
|---|---|---|---|
| 1 | `molecule-hf.toml`, box 8, orbital 4 | `h2_hf.rs` at the `h2-hf` A0 settings | `run_gamma_valence_hf` |
| 2 | H atom, Gamma | `tests/gamma_valence_hf.rs` | `run_gamma_valence_hf` |
| 3 | H atom, 2×2×2 mesh | `tests/valence_hf_mesh.rs` | `run_valence_hf` |
| 4 | KH+SOC valence | `tests/kh_soc_hf.rs` | `run_kh_soc_valence_hf` |
| 5 | relaxed core | `tests/relaxed_core_hf.rs` | `run_relaxed_core_hf` |
| 6 | Kr production settings | `kr_relaxed_core_hf.rs` default CLI | `run_relaxed_core_hf` |

```text
G-HFI-1  digit: Q=lowered spec vs hand-built spec (PartialEq) ref=rows 1 to 6 tol=exact class=R
         budget: cargo test -p libmuffintin-runtime --test input_exact_exchange, once
G-HFI-2  digit: Q=hf_energy_terms_ha total (Ha) ref=the h2-hf A0 log on main tol=1e-10 class=R
         budget: 1 run of the converted h2_hf example at the A0 settings
```

A failing row of G-HFI-1 is a lowering defect and is fixed in deliverable 2;
it never changes a spec struct or a driver. A G-HFI-2 failure with G-HFI-1
passing means the example's start or overrides differ from the A0 run; the
one permitted diagnostic is a diff of the two printed `route=` lines, then
`DIGIT / HANDOFF`.

## 4. Boundaries

- No change to any Fock loop, energy assembly, kernel, or tolerance in
  `hf_scf.rs`; the four loops stay (consolidation is a later workstream).
- No new gate on a physical quantity; `h2-hf` logs, references, README
  tables, and stamps are untouched.
- No input version bump; a version-3 DFT file parses and lowers unchanged.
- Starts only after `h2-hf` is `closed` or `handoff` in `STATUS.md`, so the
  digit test and the refactor never overlap in the working tree.
