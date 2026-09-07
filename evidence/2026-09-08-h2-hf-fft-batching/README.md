# H2 HF FFTW interstitial batching

Main revision `44fd1886a60bc6ee2b1ecb8cf303ab6cc4abc644` batches the
`fft-fftw` interstitial projection by left band in chunks of 64 while retaining
one FFT correlation per band pair. The non-FFTW implementation is unchanged.

## Class-R checks

The focused fixture ran once in each configuration:

```sh
H2_HF_INTERSTITIAL_DUMP=/tmp/h2-hf-vertices-nonfftw.txt \
  HDF5_DIR="$(brew --prefix hdf5)" TBLIS_DIR="$(brew --prefix tblis)" \
  cargo test -p libmuffintin-runtime --test gamma_valence_hf

H2_HF_INTERSTITIAL_DUMP=/tmp/h2-hf-vertices-fftw.txt \
  RUSTFLAGS="-L native=$(brew --prefix fftw)/lib" \
  HDF5_DIR="$(brew --prefix hdf5)" TBLIS_DIR="$(brew --prefix tblis)" \
  cargo test -p libmuffintin-runtime --features fft-fftw \
  --test gamma_valence_hf
```

Both passed the unchanged exchange, eigenvalue, and total identity assertions
at `1e-8` Ha. For this verification only, an untracked hook dumped the
interstitial coefficient block of every vertex from the fixture's first
`build_spinor_mpb` call. The hook was removed before commit. The 432 complex
coefficients in [`vertices-nonfftw.txt`](vertices-nonfftw.txt) and
[`vertices-fftw.txt`](vertices-fftw.txt) were compared in the same key order
using

```sh
gawk 'BEGIN {max=0; n=0; bad=0} {if ($1!=$7 || $2!=$8 || $3!=$9 || $4!=$10) bad++; dr=$5-$11; di=$6-$12; d=sqrt(dr*dr+di*di); if (d>max) max=d; n++} END {printf "vertices=%d key_mismatches=%d max_abs_complex_delta=%.17e\n", n, bad, max; if (bad || max>1e-10) exit 1}' <(paste /tmp/h2-hf-vertices-nonfftw.txt /tmp/h2-hf-vertices-fftw.txt)
```

The result was `vertices=432 key_mismatches=0
max_abs_complex_delta=0.00000000000000000e+00`.

## A0 timing report

An untracked copy of `h2_hf.rs` called
`set_hf_verbosity(HfVerbosity::Timings)` and ran the A0 settings once. A monitor
terminated it at 600 s if the first `vv.mpb_rebuild` had not completed. The raw
output is [`timing-after.log`](timing-after.log).

| Revision | `vv.mt_contraction` (s) | `vv.interstitial` (s) | First rebuild |
|---|---:|---:|---|
| before, `f65c193` | 49.667629 | >300 | not completed at probe cap |
| after, `44fd188` | 35.201104 | >564 | not completed at 600 s process cap |

`vv.mt_contraction` spends its time in the already band-batched dense site
contraction `einsum("il,aij,jr->alr")` over auxiliary and site-coordinate
indices; it is not a per-pair contraction, so the same batching change does not
apply and it was left unchanged.

## A0 rerun

The committed example was rebuilt with `--features fft-fftw` and the exact
evd-0002 A0 command ran unchanged. It did not return before the new 1800 s wall
limit and was terminated with exit status 124. Its log is
`examples/h2_dft/results/hf-a0.log` on main revision `a5bd71d`, which records
the termination line; A1 through Bv were not run.
