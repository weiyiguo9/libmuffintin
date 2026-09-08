#!/usr/bin/env bash
set -euo pipefail
case "$1" in current|baseline) revision="$1" ;; *) exit 2 ;; esac
cd "/home/xylxp/ctf-rs-r1-2-$revision"
export CARGO_TARGET_DIR="/home/xylxp/.cache/ctf-rs-r1-2-$revision-target"
export OPENBLAS_NUM_THREADS=1
export CARGO_BUILD_JOBS=2
cargo test --locked --test cyclic_reshuffle --no-run
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER='mpirun --oversubscribe -n 2'
timeout --signal=TERM --kill-after=5s 60s cargo test --locked --test cyclic_reshuffle
