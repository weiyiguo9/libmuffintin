# Repository guidance

## Local only reference
@AGENTS.local.md


## Rust MSRV

- Keep the workspace `rust-version` at 1.89 (the floor set by
  `libmuffintin-symmetry`'s moyo dependency, which pins `nalgebra` 0.35).
- Do not raise it for an optional backend.
- `tenferro-einsum` 0.3.0 needs rustc 1.96. Document that in `README.md`.
  Anyone enabling `backend-tenferro` raises `rust-version` locally to 1.96;
  the default RSTSR+TBLIS path stays 1.89.

## Test scope

- Default to focused tests for the affected crate, module, or execution path.
- Do not run `cargo test --workspace` after ordinary feature work, localized
  fixes, documentation changes, or routine API edits.
- Run the full workspace test suite only for a major cross-crate refactor or
  another change whose blast radius genuinely spans most of the workspace.

## Commit messages

Follow Conventional Commits 1.0.0.

Format:

    <type>[optional scope]: <description>

    <body: 1-3 sentences explaining what changed and why>

    [optional footer(s)]

Rules:

- type: feat | fix | refactor | perf | docs | test | build | ci | chore
- description: imperative mood, lowercase, no trailing period, ≤72 chars total
- Body is required for feat, fix, refactor, perf; optional for docs/chore.
  Explain motivation and effect, not a restatement of the diff.
- Breaking changes: append `!` after type/scope and add a
  `BREAKING CHANGE:` footer describing migration.
- scope: use module/crate/package name when the change is localized,
  e.g. `feat(solver): ...`
- One logical change per commit; do not mix refactor with behavior changes.
- Do not add "Co-Authored-By" or tool attribution lines.

## Crate naming

- Name workspace directories `crates/mt-<suffix>`.
- Name Cargo packages and dependency keys `libmuffintin-<suffix>`.
- Give every library target an explicit underscore name:

  ```toml
  [lib]
  name = "muffintin_<suffix>"
  ```

- Use `muffintin_<suffix>` in Rust imports and paths. Do not use
  `libmuffintin_<suffix>` as a Rust target or import name; POSIX toolchains add
  the filesystem `lib` prefix, and a target beginning with `lib` produces the
  unwanted `liblibmuffintin_*` artifact name.
- Keep Unix artifacts single-prefixed, for example `libmuffintin_core.rlib`.
  If a system-facing aggregate library is added, name its target `muffintin`
  so Linux produces `libmuffintin.so` or `libmuffintin.a`.
- The product-basis crate is `crates/mt-prodbasis`, Cargo package and
  dependency key `libmuffintin-prodbasis`, Rust library target and import
  `muffintin_prodbasis`. Its root holds the method-neutral product-space IR;
  the mixed-product and THC producers live under `mpb::` and `thc::`.
  Coulomb assembly consumes only the root IR types; `mpb::`/`thc::` types
  must not become `libmuffintin-coulomb` public inputs (a documented
  convention, no longer compiler-enforced).

## Repository documentation mathematics

- The rules in this section apply only when writing Markdown documentation in
  this repository, including `README.md`, `CONVENTIONS.md`, and files under
  `doc/`. They do not apply to an agent's user-facing conversation output.
- In conversation, write mathematics using Codex-native delimiters: `\(...\)`
  for inline mathematics and `\[...\]` for display mathematics.

- Write inline mathematics as `$...$`.
- Write display mathematics with a GitHub-compatible `math` fence:

  ````text
  ```math
  ...
  ```
  ````

- Do not use `\(...\)`, `\[...\]`, or `$$...$$` delimiters in Markdown.
- Do not use GitHub-disallowed MathJax macros such as `\operatorname`. Use
  supported upright text instead, for example `\mathrm{Re}`, `\mathrm{Im}`,
  and `\mathrm{sgn}`; add explicit `\,` spacing or parentheses where the
  operator needs separation from its operand.
- When Markdown-sensitive characters make ordinary `$...$` ambiguous, use
  GitHub's math-aware backtick delimiter, written as `` $`...`$ `` in source.
  This is a math delimiter, not ordinary inline code.
- Do not wrap mathematical variables or expressions in ordinary backticks.
  Reserve ordinary backticks and code fences for identifiers, paths, commands,
  source literals, and actual code.
- Keep a blank line before and after each display-math fence so both local
  MathJax-aware Markdown previews and GitHub parse it as a separate block.
- Do not attach prose hyphens directly to inline-math delimiters. Write
  `finite $q$` rather than `finite-$q$`, and rephrase suffix forms such as
  `$q$-dependent` as `dependent on $q$`.
- Do not use TeX-style `--` punctuation in prose; GitHub renders both hyphens
  literally. Use a Unicode en dash (`–`) for paired names, relationships, and
  numeric ranges. Preserve literal `--` only in code, command-line options,
  and Markdown table separators.

## Diagrams and tables

- Prefer a GitHub-rendered ` ```mermaid ` fence over prose for a branching
  control-flow pipeline or a crate dependency/data-flow DAG. Use
  `flowchart TD` for sequential procedures and `graph LR` for dependency
  DAGs.
- Prefer a plain fenced code block with Unicode box-drawing characters
  (`│ ├ └ ─ →`) over mermaid for a compact linear flow or a layout/ordering
  diagram: a small pipeline with no branching, memory or column ordering,
  or a tree-shaped file/module listing. Mermaid is heavyweight there and
  box-drawing keeps alignment exact.
- GitHub does not render `$...$` math inside mermaid node labels or inside a
  box-drawing fence. Label both with code identifiers (`ScalarProductInput`,
  `PairBlock`), not formulas; formulas stay in the formula layer as prose or
  math fences.
- Quote a mermaid node label that contains parentheses, commas, or other
  mermaid-significant characters, for example `A["fit zeta (weighted L2)"]`.
- Prefer a GitHub table over prose for an enumerable contract: inputs and
  outputs, engine or format comparisons, gate values, schema fields.
- A diagram or table must condense the prose it replaces, not sit alongside
  it unchanged. Skip a diagram where the prose it would replace is already
  tight; do not add decorative diagrams.

## Numbered documentation

- Keep the canonical derivations in `doc/` and name them
  `NN_lowercase_snake_case.md`, using a zero-padded two-digit sequence.
- Add a new document only for a distinct topic, using the next unused number.
  Extend or revise the existing numbered document when the topic already has
  one.
- Preserve the existing numbering and keep cross-document notation and
  conventions consistent; update earlier numbered documents when needed.
- Do not add unnumbered one-off files to `doc/`.
- Separate three layers instead of interleaving them: formulas (mathematical
  contracts, conventions, derivations — math fences), algorithms
  (method-neutral procedures — where mermaid flowcharts belong), and
  implementation (concrete crate/type/function bindings, gate values,
  fixture pointers — where tables and code identifiers belong).
  [18](doc/18_lapw_mpb_thc_integration.md) is the reference layout.
