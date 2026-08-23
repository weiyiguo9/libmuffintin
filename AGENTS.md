# Repository guidance

## Rust MSRV

- Keep the workspace `rust-version` at 1.85 (the edition 2024 floor).
- Do not raise it for an optional backend.
- `tenferro-einsum` 0.3.0 needs rustc 1.96. Document that in `README.md`.
  Anyone enabling `backend-tenferro` raises `rust-version` locally to 1.96;
  the default RSTSR+TBLIS path stays 1.85.

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

## Numbered documentation

- Keep the canonical derivations in `doc/` and name them
  `NN_lowercase_snake_case.md`, using a zero-padded two-digit sequence.
- Add a new document only for a distinct topic, using the next unused number.
  Extend or revise the existing numbered document when the topic already has
  one.
- Preserve the existing numbering and keep cross-document notation and
  conventions consistent; update earlier numbered documents when needed.
- Do not add unnumbered one-off files to `doc/`.
