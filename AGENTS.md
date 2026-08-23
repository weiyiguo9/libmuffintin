# Repository guidance

## Local only reference
@AGENTS.local.md


## Rust MSRV

- Keep the workspace `rust-version` at 1.85 (the edition 2024 floor).
- Do not raise it for an optional backend.
- `tenferro-einsum` 0.3.0 needs rustc 1.96. Document that in `README.md`.
  Anyone enabling `backend-tenferro` raises `rust-version` locally to 1.96;
  the default RSTSR+TBLIS path stays 1.85.

## Test scope

- Default to focused tests for the affected crate, module, or execution path.
- Do not run `cargo test --workspace` after ordinary feature work, localized
  fixes, documentation changes, or routine API edits.
- Run the full workspace test suite only for a major cross-crate refactor or
  another change whose blast radius genuinely spans most of the workspace.

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
- Name the method-neutral auxiliary/product-space IR crate
  `crates/mt-auxiliary-ir`, its Cargo package and dependency key
  `libmuffintin-auxiliary-ir`, and its Rust library target and import
  `muffintin_auxiliary_ir`. Do not reintroduce the former `mt-product` names.

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

## Numbered documentation

- Keep the canonical derivations in `doc/` and name them
  `NN_lowercase_snake_case.md`, using a zero-padded two-digit sequence.
- Add a new document only for a distinct topic, using the next unused number.
  Extend or revise the existing numbered document when the topic already has
  one.
- Preserve the existing numbering and keep cross-document notation and
  conventions consistent; update earlier numbered documents when needed.
- Do not add unnumbered one-off files to `doc/`.
