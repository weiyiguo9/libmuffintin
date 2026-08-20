# Repository guidance

## Documentation mathematics

- Write inline mathematics as `$...$`.
- Write display mathematics with a GitHub-compatible `math` fence:

  ````text
  ```math
  ...
  ```
  ````

- Do not use `\(...\)`, `\[...\]`, or `$$...$$` delimiters in Markdown.
- Do not wrap mathematical variables or expressions in backticks. Reserve
  backticks and ordinary code fences for identifiers, paths, commands, source
  literals, and actual code.
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
