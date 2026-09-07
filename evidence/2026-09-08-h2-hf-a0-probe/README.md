# H₂ HF A0 probe (answers the evd-0002 handoff question)

Two runs of the exact A0 command of `plans/h2-hf/plan.v1.md` against `main`
f65c193, release build with `--features fft-fftw`, `RAYON_NUM_THREADS=10`.

- `probe.sh`, `rss.log`: A0 binary as committed, RSS sampled every 20 s,
  12 minute cap. Peak 2.06 GB at 40 s, then 0.1–0.3 GB; one core busy
  (CPU 90–99 %). No output before the cap, as the example prints only
  after the driver returns.
- `sample-top.txt`: `sample` of the main thread at 240 s (15 s window):
  all samples inside `spinor_mpb::build_spinor_mpb_from_basis`, 77 % in
  `contract_interstitial_selections` → `einsum("pr,ra->pa")` → TBLIS
  `tblis_tensor_mult`, most of it in `tci_barrier_wait`.
- `timing.log`: an untracked copy of `h2_hf.rs` with
  `set_hf_verbosity(Timings)`, 6 minute cap. First MPB rebuild:
  `vv.mt_contraction` 49.7 s, then `vv.interstitial` still running after
  300 s. The Fock loop allows 40 rebuilds per outer iteration.

Reading: the `fft-fftw` variant of `contract_interstitial_selections`
issues one TBLIS einsum per band pair with a `[1, n_raw]` left factor; the
non-fftw variant batches 64 selections per einsum. Time, not memory, killed
A0.

## After the batching commit (main 44fd188 / a5bd71d)

- `sample-batched-top.txt`: 20 s sample at 120 s. Inside
  `contract_interstitial_selections`: 43 % in the batched einsum (TBLIS →
  BLIS gemm, six of every ten samples in `bli_thrcomm_barrier_atomic`),
  46 % in the per-pair `PairFft::correlate` and amplitude copies, 9 % in
  element-wise `DenseEigenvectors::at` reads.
- `timing-sizes-1thread.log`: a scratch build printing the problem size at
  the start of `vv.interstitial`:

  ```text
  selections=1060900  n_raw=3809  n_pw=515  theta_shape=[3809, 515]
  pair fft grid dims=[21, 21, 21] len=9261
  ```

  `run_valence_hf` → `rebuild_exchange` (`crates/mt-runtime/src/hf_scf.rs`)
  selects every `(left_band, right_band)` pair of the 1030-band spinor
  window, 1030² = 1 060 900 vertices per MPB rebuild, six 21³ FFTs and one
  gather each, before the exchange assembly applies the occupations. The
  relaxed-core frame (`rebuild_core_feedback_frame`) already restricts
  `left_band` to bands with nonzero valence occupation. Batching the
  projection could not help: the pair count, not the per-pair cost, is the
  problem.
- Thread count is not the lever: with all thread counts forced to 1 the
  first `vv.interstitial` was still running at a 420 s cap; with ten
  threads at a 300 s cap (`vv.mt_contraction` 48.4 s and 37.1 s).
