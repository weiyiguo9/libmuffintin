# CTF-R1-2 diagnostic 3

Class R; quantity: per-call, per-rank exchange counts and FNV-1a-64 bucket
hashes, plus the existing exact i8 cyclic_reshuffle assertions (bound 0).
Reference: scratch archive of f2039d3; candidate: scratch archive of 813d90a.
Budget: two numerical runs total, one two-rank run per revision. Diagnostic 1
was previously spent; diagnostic 2 is excluded. Identical traces require
HANDOFF with no fix or acceptance rerun. No delivery source is instrumented.

Scratch directories: `/home/xylxp/ctf-rs-r1-2-current` and
`/home/xylxp/ctf-rs-r1-2-baseline`, populated from `current.tar` and
`baseline.tar` produced by `git archive` of the revisions above. Separate
Linux target directories are set in `diagnostic-3.sh`. The existing
`libmuffintin-wsl-keepalive` PID 322 was reused; none started.

Instrumentation is identical in both `Comm::exchange` implementations: a
process-local AtomicUsize call index, rank, send_counts, recv_counts, and
per-bucket FNV-1a hashes (offset 0xcbf29ce484222325, wrapping multiplier
0x100000001b3). A complete line is written to stderr in one write. Only
the bool/complex/Word calls in `run` were removed to select i8; layouts,
assertions, initialization, communicator splits and cleanup were unchanged.

Exact commands, invoked once each from PowerShell:

```powershell
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1-2/diagnostic-3.sh current > D:/projects/runs/ctf-rs-r1-2/current.log 2>&1
wsl -d Ubuntu-26.04 -- bash /mnt/d/projects/runs/ctf-rs-r1-2/diagnostic-3.sh baseline > D:/projects/runs/ctf-rs-r1-2/baseline.log 2>&1
```

Each script compiles its single test without execution first, then uses a
60-second runtime supervisor (five-second kill grace) to bound MPI peers left
blocked by a rank-local assertion panic. Its exact nested commands are in the
script. Trace comparison sorts by (call index, rank), not scheduling order.

## Result

Both no-run builds completed. Current numerical run exited 124 at the
60-second supervisor; baseline numerical run exited 101 after MPI reported
the rank-local panic. Both failed rank 1, local offset 0, actual 0, expected
-5 (absolute difference 5, exact bound 0). No cyclic_reshuffle process remained.

Both traces have exactly two complete rows: call 0, ranks 0 and 1. Counts
and all sent/received hashes are identical, so no first divergent line exists.
PowerShell wrapped each raw stderr trace over several log lines; extraction
joined whitespace across the complete tuple before sorting/comparison. The
full counts and hashes appear in `current.trace` and `baseline.trace`.

BRIEF-2's identical-trace branch applies: DIGIT / HANDOFF, no wrapper fix,
no acceptance rerun, no further diagnostics. This reproduces a pre-R1 baseline
failure; it does not establish global R1 equivalence.
