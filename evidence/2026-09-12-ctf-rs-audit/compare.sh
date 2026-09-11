#!/bin/bash
# Compare the DIGIT lines of the baseline run (master e9b6bba, before the
# sparse refactor) with the final run (master 7ffd5f2), per target and rank,
# after stripping wall-clock fields that legitimately differ between runs.
S=/private/tmp/claude-501/-Users-zerozaki07-tmp-libmuffintin/e260fabd-b225-4ee1-80e9-75310e7132bc/scratchpad/audit
extract() {
  local dir="$1" out="$2"
  ( cd "$dir" && for f in *-[124].log *-7.log; do
      [ -f "$f" ] || continue
      t="${f%.log}"
      grep -h 'DIGIT / ' "$f" | sed "s/^/$t\t/"
    done ) \
  | sed -E 's/[a-z_]*(seconds|_s|time|elapsed|wall)[a-z_]*=[0-9.eE+-]+//g; s/[0-9.]+ ?(s|ms) elapsed//g' \
  | sort > "$out"
}
extract "$S/baseline" "$S/baseline-digits.norm"
extract "$S/final" "$S/final-digits.norm"
echo "baseline lines: $(wc -l < "$S/baseline-digits.norm"); final lines: $(wc -l < "$S/final-digits.norm")"
echo "--- lines only in baseline (target<TAB>line) ---"
comm -23 "$S/baseline-digits.norm" "$S/final-digits.norm" | cut -c1-200
echo "--- lines only in final ---"
comm -13 "$S/baseline-digits.norm" "$S/final-digits.norm" | cut -c1-200
