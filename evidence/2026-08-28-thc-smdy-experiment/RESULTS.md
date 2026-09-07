# Sm all-electron exchange: SPEX MPB-LRI versus libmuffintin THC

## Headline result

The decisive issue was not NUFFT and not the scalar-relativistic small
component.  It was an unconverged angular cutoff in the initial SPEX mixed
product basis.

SPEX defaults to `MBASIS LCUT=5`, but a 4f x 4f product contains an `L=6`
channel.  Raising `LCUT` from 5 to 6 increases the MPB dimension from 168 to
256 and changes occupied up-spin 4f exchange by **2.67--4.37 eV**.  Raising it
again to 7 (286 functions) changes those rows by only 0.01--0.02 meV.  `GCUT`
3 to 4 changes the 20 targets by at most 3.01 meV, and `TOL` 1e-4 to 1e-5 by
at most 12.2 meV.  `LCUT=6` is therefore the converged SPEX reference used
below.

The existing, unchanged libmuffintin THC implementation compresses the sampled
pair space well.  At QR threshold `3e-3`, the selected rank is 113 on 3x3x3
and 116 on 4x4x4, or 7.06 and 7.25 times the 16-orbital selection window.  THC
differs from its uncompressed two-component 24^3 grid by only **0.0209 eV max /
0.0078 eV RMS** and **0.0108/0.0036 eV**, respectively.

The remaining 24^3-grid error against converged SPEX is 2.31--2.38 eV max,
but it is a sampling floor rather than a THC floor.  On a muffin-tin-adaptive
grid, a Rust NUFFT/Fourier calculation agrees with LCUT=6 SPEX to **0.255 eV
max / 0.131 eV RMS** at 24^3 reciprocal modes and **0.204/0.100 eV** with an
adaptive spherical reciprocal cutoff of 15.76 bohr^-1.

A direct real-Sm Weinert test also succeeds.  For the same finite-q up-4f pair,
the unchanged libmuffintin Weinert assembler gives **1.067694427 eV**
(`Lmax=6`, `GCUT=4`), while the independently evaluated Fourier sphere at
12 bohr^-1 gives **1.072652386 eV**, a 4.96 meV difference.  Thus Weinert is
the appropriate production path; NUFFT was useful as an independent diagnostic
but is not required by THC.

## Reference conventions

All main SPEX numbers are valence-only diagonal exchange at Gamma, both spins,
bands 5--14, from `JOB HF 1:(5-16)`.  The occupied sum still contains all
occupied valence states on the full k mesh.  SPEX's separately printed
frozen-core exchange is subtracted explicitly.

The Rust contraction literally omits `q=0,G=0`.  Standard SPEX adds an analytic
finite-k pole replacement through `divergence_x` in `selfenergy.f`: 4.31349 eV
on 3x3x3 and 3.23515 eV on 4x4x4 for affected Gamma channels.  A detached
build-worktree-only switch disables that term for the strict comparison.
Standard-pole logs are retained but never mixed into the tables below.

Both scalar-relativistic components are combined as

`rho_ij(r) = conj(L_i)L_j + conj(S_i)S_j`,

with `S=0` in the interstitial.  The largest observed L+S minus L-only exchange
change is only 0.002107 eV, but both components are included everywhere in the
reported experiment.

## libmuffintin THC rank/error curves

Errors are max/RMS over all 20 target rows, in eV, against strict LCUT=6 SPEX.
`THC-grid` compares with the same adapter's uncompressed L+S 24^3 Fourier grid,
and therefore isolates interpolation/fitting error.  Timings include shared
libmuffintin QR selection and all per-q fits for that row.

### 3x3x3 k mesh (27 full-BZ points)

| QR threshold | Nmu | Nmu/Norb | max / RMS vs SPEX (eV) | max / RMS THC-grid (eV) | wall (s) |
|---:|---:|---:|---:|---:|---:|
| direct grid, no THC | - | - | 2.3111 / 1.0620 | - | 0.27 |
| 1e-1 | 19 | 1.1875 | 10.0628 / 5.2916 | 9.0131 / 5.0934 | 8.26 |
| 3e-2 | 39 | 2.4375 | 6.6550 / 2.9026 | 4.3439 / 2.4076 | 14.16 |
| 1e-2 | 71 | 4.4375 | 4.3907 / 1.6273 | 4.4890 / 1.2443 | 26.57 |
| **3e-3** | **113** | **7.0625** | **2.3313 / 1.0671** | **0.0209 / 0.0078** | **46.26** |

The attached LCUT=6 rerun completed in 95.5 s.  Tightening THC converges to the
separate regular-grid limit; it cannot remove that grid's 2.31 eV error floor.

### 4x4x4 k mesh (64 full-BZ points)

| QR threshold | Nmu | Nmu/Norb | max / RMS vs SPEX (eV) | max / RMS THC-grid (eV) | wall (s) |
|---:|---:|---:|---:|---:|---:|
| direct grid, no THC | - | - | 2.3765 / 1.0987 | - | 0.65 |
| 1e-1 | 19 | 1.1875 | 9.4477 / 5.4341 | 8.3821 / 5.2463 | 20.12 |
| 3e-2 | 40 | 2.5000 | 6.9247 / 2.9541 | 4.5508 / 2.4538 | 34.80 |
| 1e-2 | 71 | 4.4375 | 5.4189 / 1.7677 | 5.5226 / 1.4437 | 61.58 |
| **3e-3** | **116** | **7.2500** | **2.3837 / 1.1008** | **0.0108 / 0.0036** | **111.14** |

The attached rerun completed in 225.7 s.  Increasing Nk from 27 to 64 changes
the tight rank by only 2.7%, even though work and memory grow with Nk.

## Tight-rank band-resolved comparison

These are the `3e-3` L+S THC values.  Positive error means THC is less negative
than converged strict SPEX.

| spin | band | SPEX 3^3 | THC 3^3 | error 3^3 | SPEX 4^3 | THC 4^3 | error 4^3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| up | 5 | -8.9368 | -9.0395 | -0.1027 | -10.0363 | -10.1426 | -0.1063 |
| up | 6 | -16.4329 | -17.3283 | -0.8954 | -17.8137 | -18.7903 | -0.9766 |
| up | 7 | -23.7672 | -25.1508 | -1.3836 | -24.6707 | -26.1098 | -1.4392 |
| up | 8 | -23.7672 | -25.1515 | -1.3843 | -24.6707 | -26.1113 | -1.4406 |
| up | 9 | -23.7672 | -25.1533 | -1.3861 | -24.6707 | -26.1060 | -1.4353 |
| up | 10 | -29.5057 | -27.1835 | 2.3222 | -30.4533 | -28.0696 | 2.3837 |
| up | 11 | -29.5057 | -27.1956 | 2.3101 | -30.4533 | -28.0729 | 2.3804 |
| up | 12 | -29.5058 | -27.1745 | 2.3313 | -30.4533 | -28.0729 | 2.3804 |
| up | 13 | -7.6716 | -7.6608 | 0.0108 | -7.6528 | -7.6408 | 0.0120 |
| up | 14 | -7.6716 | -7.6624 | 0.0092 | -7.6528 | -7.6411 | 0.0117 |
| down | 5 | -5.5071 | -5.5839 | -0.0768 | -6.3957 | -6.4756 | -0.0798 |
| down | 6 | -2.2177 | -2.1976 | 0.0200 | -2.2231 | -2.2028 | 0.0203 |
| down | 7 | -4.6830 | -4.6717 | 0.0113 | -4.7148 | -4.7035 | 0.0114 |
| down | 8 | -4.6830 | -4.6717 | 0.0113 | -4.7148 | -4.7034 | 0.0114 |
| down | 9 | -4.6830 | -4.6717 | 0.0113 | -4.7148 | -4.7034 | 0.0114 |
| down | 10 | -2.1694 | -2.2804 | -0.1111 | -2.1965 | -2.3066 | -0.1102 |
| down | 11 | -2.1694 | -2.2802 | -0.1109 | -2.1965 | -2.3067 | -0.1102 |
| down | 12 | -2.1694 | -2.2800 | -0.1107 | -2.1965 | -2.3081 | -0.1117 |
| down | 13 | -2.0663 | -2.1027 | -0.0364 | -2.0790 | -2.1160 | -0.0370 |
| down | 14 | -2.0663 | -2.1015 | -0.0352 | -2.0790 | -2.1136 | -0.0346 |

The structured up-4f errors change sign between bands 7--9 and 10--12, while
other rows are small.  That pattern is caused by the coarse uniform sampling;
the adaptive evaluation below removes it.

## Real-space and reciprocal-space grid diagnosis

The independent `nufft_crosscheck` binary uses Rust `oxifft` 0.4.2 type-1 3D
NUFFT.  SPEX evaluates L and S on 96 radial shells x 194 Fibonacci directions
inside `RMT=2.6 bohr` and a 24^3 interstitial grid (27,940 total points).  The
weight sum equals the cell volume and the 20 Gamma orbital norms are
0.998010--1.001864.  On identical samples, NUFFT agrees with explicit weighted
NDFT to `2.57e-6` relative error.

Against LCUT=6 SPEX, the adaptive 24^3-mode result is **0.2546 eV max /
0.1308 eV RMS**, compared with 2.3111/1.0620 eV on the uniform grid.  Selected
adaptive values are:

| spin/band | adaptive Fourier (eV) | SPEX (eV) | error (eV) |
|---|---:|---:|---:|
| up 5 | -8.9530 | -8.9368 | -0.0162 |
| up 7 | -23.5220 | -23.7672 | 0.2452 |
| up 10 | -29.2970 | -29.5057 | 0.2087 |
| up 12 | -29.3244 | -29.5058 | 0.1813 |
| down 5 | -5.5225 | -5.5071 | -0.0154 |

Reciprocal space is a discrete `q+G` lattice, so its adaptive form is a growing
physical sphere rather than arbitrary off-lattice frequency points.  One 40^3
NUFFT box was accumulated by complete spherical shells:

| spherical cutoff (bohr^-1) | max error (eV) | RMS error (eV) |
|---:|---:|---:|
| 3.502 | 5.038 | 2.749 |
| 6.129 | 2.344 | 1.287 |
| 8.756 | 1.159 | 0.627 |
| 11.382 | 0.545 | 0.277 |
| 14.009 | 0.248 | 0.123 |
| **15.760** | **0.204** | **0.100** |
| 17.511 | 0.264 | 0.122 |

The last shell is no longer monotone because the interstitial parent grid is
only 24^3 and is beyond its clean high-G sampling range; its worst row is the
neighbouring up band 14, not a 4f row.  This coupled real/reciprocal limitation
is why the 15.76 bohr^-1 row is the defensible best result rather than treating
the largest available mode box as automatically converged.

## Direct libmuffintin Weinert check on a real Sm pair

To test Weinert itself, not a toy fixture, a separate exporter sampled target
Gamma up band 7 and occupied up band 7 at full-BZ k point 2
(`q_frac=[0,0,1/3]`).  It uses all 997 exact SPEX exponential radial shells x
194 angles plus a 40^3 interstitial grid: 236,304 points, of which 193,418 are
muffin-tin points.  The quadrature weight sum is 224.044705351315 bohr^3 versus
cell volume 224.044705351558 bohr^3.

The external Rust binary calls the existing public
`libmuffintin_coulomb::assemble_sampled_coulomb` API without library changes:

| method | angular / reciprocal setting | pair Coulomb (eV) |
|---|---|---:|
| Weinert | Lmax=5, GCUT=3 | 1.027807792 |
| Weinert | Lmax=5, GCUT=4 | 1.027994902 |
| Weinert | Lmax=6, GCUT=3 | 1.067503423 |
| **Weinert** | **Lmax=6, GCUT=4** | **1.067694427** |
| Fourier sphere | 3 bohr^-1 | 0.439651559 |
| Fourier sphere | 4 bohr^-1 | 0.640856985 |
| Fourier sphere | 8 bohr^-1 | 0.965315005 |
| **Fourier sphere** | **12 bohr^-1** | **1.072652386** |
| Fourier sphere | 15.76 bohr^-1 | 1.110260181 |

The Weinert matrix element is real to about `3e-18 Ha`.  `GCUT=3→4` changes it
by only 0.191 meV; `Lmax=5→6` changes it by about 39.7 meV for this single 4f
pair.  Weinert L6/G4 and the independently converged Fourier region agree to
4.96 meV.  The 15.76-bohr^-1 Fourier excess is consistent with the known
high-G/interstitial quadrature limit, not a Weinert failure.

## Compact core--valence exchange

Deep-core exchange should not be forced into the global k-point THC columns.
In the scalar-collinear SPEX formulation every core orbital is confined to its
muffin tin, so its exchange pair with a valence state is also site-local.  The
sum over all occupied relativistic core shells and Coulomb multipoles can be
performed once, producing a small band- and k-independent radial kernel

`K_core(site,l,spin;n,n')`.

For each valence state the complete core contribution is then

`Sigma_core = - sum(site,l,m) c_lm^dagger K_core(site,l,spin) c_lm`.

This is exactly the block structure already present inside SPEX
`exchange_core`: large and small radial products are combined before the
intra-sphere Poisson integral; core SOC degeneracies are summed into the
kernel; and, without valence SOC, the same radial block applies to every `m`.
There is no q loop, FFT, NUFFT, interpolation-point fit, or occupied-k sum.

An environment-gated build-worktree exporter accumulated these blocks for the
same Sm orbitals.  The external Rust binary `core_compact_check` read only the
blocks and ordinary valence muffin-tin coefficients, reconstructed every SPEX
core-exchange row, and tested optional block eigenvalue truncation.  No
libmuffintin source was changed.

Sm has `nindx=[3,3,2,2,2,2,2,2,2,2]` for `l=0..9`.  Across both spins the exact
operator contains only 72 active real symmetric entries: **576 bytes**.  A
band evaluation touches 420 small dense-block entries; this cost is independent
of Nk.

| relative block-eigen threshold | total retained rank | factor storage (doubles) | max / RMS core error 3^3 (eV) | max / RMS core error 4^3 (eV) |
|---:|---:|---:|---:|---:|
| exact | 44 | 144 | 0 / 0 | 0 / 0 |
| 1e-1 | 20 | 64 | 0.2268 / 0.1649 | 0.2268 / 0.1649 |
| 1e-2 | 24 | 76 | 0.2268 / 0.1649 | 0.2268 / 0.1649 |
| **3e-3** | **37** | **119** | **2.69e-5 / 6.93e-6** | **2.69e-5 / 6.94e-6** |
| 1e-6 | 44 | 144 | 0 / 0 | 0 / 0 |

The exact block contraction reproduces the full-precision SPEX value to all 12
reported decimal places.  Eigenfactor truncation is not attractive as standalone
storage: rank 37 is extremely accurate but needs 119 doubles when its rotation
vectors are included, more than the 72-double exact symmetric blocks.  Rank 20
saves only eight doubles relative to that exact representation while adding
0.227 eV error.  If the eigen-rotations are permanently folded into the radial
basis, only the retained eigenvalues need be stored, but that complicates the
basis convention for negligible absolute memory savings.

Representative exact reconstructed core contributions on 3x3x3 are
`-22.362070 eV` (up band 7), `-23.605102 eV` (up band 10),
`-1.541320 eV` (down band 7), and `-22.824908 eV` (down band 13).
They are much larger than the desired valence error tolerance, so keeping core
and valence as separately testable terms is important.

The recommended total-exchange comparison is therefore

`Sigma_x = Sigma_valence^(THC+Weinert) + Sigma_core^(exact local blocks)`.

Compare both terms separately to SPEX before adding them.  libmuffintin's THC
`core_orbital` marker is useful for toy diagnostics and residual grouping, and
its product IR/MPB supports selected core radial functions, but treating all
46 Sm core electrons as ordinary Bloch THC orbitals would duplicate k columns
and discard the much stronger locality above.

For convenience, the exact compact core term has also been joined to the tight
valence THC rows in `core_valence_total_sm3.csv` and
`core_valence_total_sm4.csv`.  The combined error is intentionally identical
to the valence error because the core reconstruction is exact; the files keep
all three quantities (`valence`, `core`, and `total`) visible to prevent hidden
double counting.

## What was matched

- Same converged scalar-relativistic SPEX PBE Sm orbitals, including L and S.
- Same one-atom fcc cell: `a=5.101908807468821 A` (33.2 A^3/atom),
  `RMT=2.6 bohr`, no U and no SOC.
- Same Gamma target, both spins, bands 5--14; same full occupied valence sum and
  exact SPEX k/occupation weights (16 electrons total).
- Same canonical transfer/Umklapp convention and literal `q=0,G=0` omission.
- Converged SPEX product angular space (`MBASIS LCUT=6`), with GCUT/TOL checks.
- Existing libmuffintin THC selection/fitting and Weinert assembly APIs, called
  from external Rust experiment code; no libmuffintin source edits.

## What remains unmatched

- The rank curves use a regular 24^3 parent grid and external Fourier Coulomb,
  while the best absolute cross-check uses an adaptive regional grid.  The
  library THC fit has therefore been shown nearly exact on its regular sampled
  space, and the adaptive direct Coulomb has been shown close to SPEX, but a
  full all-q adaptive-THC plus Weinert contraction for all 20 rows was not run.
- The real-Sm Weinert call is a representative finite-q 4f pair, not a dump of
  SPEX's complete `V^q` matrix.  It validates the library assembler and the
  decisive L=6 channel directly, but is not by itself a full exchange sum.
- The pair check uses libmuffintin's supported `LEXP=12`; this SPEX build
  defaults to 14.  For the tested one-site L<=6 charge and converged GCUT this
  does not prevent 5 meV Fourier agreement, but it is not an identity claim for
  every block of a full SPEX `V^q` matrix.
- Core-inclusive THC was not attempted; all valence tables are explicit
  valence-only exchange.  Core-inclusive *total* exchange can instead use the
  separately validated exact compact blocks above without putting core states
  into the THC fit.
- The compact-core exporter presently covers one-site scalar-collinear valence
  states.  Multiple sites generalize to one local block set per site/type;
  valence SOC/noncollinearity requires coupled spin/m blocks rather than the
  repeated scalar `m` block tested here.
- Sm supplies both requested k meshes.  Dy was not needed to establish the
  rank scaling, grid floor, MPB cutoff failure, and Weinert result.

## Reproducibility

SPEX is installed at `tools/spex/bin/spex` and reports 06.00pre38
(`6a8249f/mod`) from checkout commit `b7778ba`.  Experimental exporters and
the strict-pole switch exist only in detached build worktree `build/spex-src`;
the original SPEX checkout is untouched.  The external Rust crate is
`experiment/rust-thc`; libmuffintin under `codes/libmuffintin` is unchanged.

Representative commands:

```bash
cd ~/projects/thc-smdy/experiment/rust-thc
cargo build --release
cargo test --release

cd ~/projects/thc-smdy
RAYON_NUM_THREADS=10 experiment/rust-thc/target/release/spex-thc-experiment \
  runs/sm_fcc_3x3_ref/thc_orbitals_grid24_ls.bin \
  runs/sm_fcc_3x3_ref/hf_q0_strict_lcut6.log runs/rust_thc/sm3_lcut6

RAYON_NUM_THREADS=10 experiment/rust-thc/target/release/nufft_crosscheck \
  runs/sm_fcc_3x3_ref/nufft_adaptive_tight_ls.bin \
  runs/sm_fcc_3x3_ref/hf_q0_strict_lcut6.log \
  runs/rust_thc/nufft_recip40_cube.csv 40 \
  runs/rust_thc/nufft_recip40_shells.csv

experiment/rust-thc/target/release/weinert_pair_check \
  runs/sm_fcc_3x3_ref/weinert_pair.bin

experiment/rust-thc/target/release/core_compact_check \
  runs/sm_fcc_3x3_ref/core_compact.bin \
  runs/rust_thc/core_compact_bands.csv \
  runs/rust_thc/core_compact_summary.csv
```

Canonical LCUT=6 outputs are `runs/rust_thc/sm3_summary.csv`,
`sm3_bands.csv`, `sm4_summary.csv`, and `sm4_bands.csv`.  Adaptive reciprocal
and Weinert data are in `nufft_recip40_{cube,shells}.csv` and
`weinert_pair_check.log`.  Compact-core curves are in
`core_compact_{summary,bands}.csv` and `core_compact_sm4_{summary,bands}.csv`.
Joined total-exchange tables are `core_valence_total_sm3.csv` and
`core_valence_total_sm4.csv`.
Default-LCUT5 and earlier large-only CSVs are
preserved with explicit names.  All failed attempts are in `notes/LOG.md`.

## Verdict

For this restricted Sm all-electron window, libmuffintin THC has modest,
k-insensitive rank scaling: about `7*Norb` makes the fit numerically exact on
the supplied pair space.  The initial apparent multi-eV failure was mainly a
bad reference comparison: default SPEX `LCUT=5` omitted the required 4f x 4f
`L=6` channel.  With `LCUT=6`, adaptive real-space sampling and reciprocal
shell convergence reduce the full 20-row direct error to about 0.2 eV max,
and a real-pair libmuffintin Weinert result agrees with independent Fourier to
5 meV.  Core--valence exchange has an even simpler exact representation: 576
bytes of local radial blocks reproduce SPEX to micro-eV printed precision, so
it should remain separate from THC.  The honest remaining task for a production
claim is to feed adaptive valence THC zeta functions for every q into the same
Weinert assembler; increasing THC rank, adding core Bloch columns, or requiring
NUFFT is not the next bottleneck.
