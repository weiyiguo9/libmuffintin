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
