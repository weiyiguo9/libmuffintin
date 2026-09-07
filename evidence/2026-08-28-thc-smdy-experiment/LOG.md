# Experiment log

All timestamps are Asia/Shanghai unless noted. Commands were launched from Windows with `wsl -d Ubuntu-26.04 -- bash -lc "cd ~/projects/thc-smdy && ..."` unless an exact invocation is shown.

## 2026-08-27

- Read the host `AGENTS.md`, `BRIEF.md`, and `PROMPT.txt`; read `notes/ENV.md` plus SPEX `AGENTS.md`, `README.md`, and `INSTALL` before configuring.
- WSL transient failure during concurrent read-only probes: one `wsl ... cat` returned `Wsl/Service/E_UNEXPECTED`. `wsl --status` and `wsl -l -v` still showed `Ubuntu-26.04` running as WSL 2; a subsequent `wsl -d Ubuntu-26.04 -- true` and serial file read succeeded without restart.
- Existing SPEX checkout state before this task: branch `feat/wannier-spin-quant-axis`, commit `b7778ba`; only `.claude/` was untracked. The checked-in `Makefile`/`src/Makefile` point to an unavailable HPC Intel/MKL environment (`/home/wguo`, `/sw`, `/gpfs`), and no local SPEX executable was present.
- Failed documentation probe: `pdfinfo`, `pdftotext`, and `rg` were not installed inside WSL. This was diagnostic only; install the missing read-only tooling with the root WSL entry point before retrying.
- Installed only the missing documentation/search tools with `wsl -d Ubuntu-26.04 -u root -- apt-get install -y poppler-utils ripgrep` (14 new packages, no upgrades). Existing compiler and scientific-library packages were not reinstalled.
- Installed the missing MPI-build prerequisite `libscalapack-openmpi-dev` (plus its runtime and `mpi-default-bin`); SPEX's own manual states that MPI builds require ScaLAPACK.
- Failed configure attempt (transcript: `runs/build/configure.log`): an out-of-tree invocation in `build/spex-mpi` detected GNU Fortran 15.2, MPI-3, and the generic SPEX interface, then aborted with `Source file src/readwrite.f does not exist!`. This configure script assumes an in-source layout even when invoked by path. To preserve the original checkout/configuration, use a detached Git worktree under `build/spex-src` and configure/build in that source-layout copy.
- Created detached build worktree `build/spex-src` at commit `b7778ba`, leaving the original checkout untouched.
- Failed configure attempt in the worktree: HDF5 module discovery ignored the split Debian include/lib layout and aborted with `Could not compile with HDF5`; the failed transcript remains in `runs/build/configure.log`. Created `tools/HDF5-openmpi/{include,lib}` as symlinks to `/usr/include/hdf5/openmpi` and `/usr/lib/x86_64-linux-gnu/hdf5/openmpi`, respectively, to present the root layout that SPEX's `--with-hdf5` expects.
- Successful configure command: `./configure --enable-mpi=yes --with-fc=mpif90 --with-blas=-lopenblas --with-lapack=-llapack --with-hdf5=/home/xylxp/projects/thc-smdy/tools/HDF5-openmpi --with-libdir=/usr/lib/x86_64-linux-gnu --prefix=/home/xylxp/projects/thc-smdy/tools/spex FC=mpif90 MPIFC=mpif90 FCFLAGS='-O2 -fallow-argument-mismatch -fallow-invalid-boz'`. It found MPI-3, `-lscalapack-openmpi`, FFTW3, parallel HDF5 with `mpi_f08`, and the generic SPEX interface; ELPA was absent and optional. Transcript: `runs/build/configure.log`.
- Two `nohup make -j6` launch attempts exited with WSL and produced an empty log/PID file; no compiler process survived. Re-launched persistently as systemd user unit `thc-spex-build.service` with working directory `build/spex-src` and both output streams appended to `runs/build/make.log`: `systemd-run --user --unit=thc-spex-build --collect --property=WorkingDirectory=... --property=StandardOutput=append:... --property=StandardError=append:... /usr/bin/make -j6`.
- First real build failed in six multi-line `RWarn(...)` macro invocations in `src/selfenergy.f` (`runs/build/make.log`): GNU traditional preprocessing ended the macro argument at the Fortran continuation, yielding `Syntax error in expression`. Applied a minimal build-worktree-only portability patch that constructs each long message in a fixed-length character variable and calls `RWarn(trim(message))` on one line. No physics or libmuffintin source was changed. Rebuild output: `runs/build/make2.log`, systemd unit `thc-spex-build2.service`.
- Rebuild succeeded and produced `build/spex-src/src/spex.inv` and `spex.noinv` (6.8/7.0 MB). First `make install` failed because the configured prefix did not exist (`runs/build/install.log`); after creating `tools/spex`, installation succeeded (`runs/build/install2.log`). `/home/xylxp/projects/thc-smdy/tools/spex/bin/spex --version` reports SPEX `06.00pre38` (`6a8249f/mod`) despite the checkout directory name `spex06.00pre36`, GNU Fortran 15.2, OpenMPI 5.0.10, generic SPEX DFT interface, compile time `2026-08-28 00:05:14`.
- Prepared `runs/sm_fcc_3x3_dft/spex.inp`: one-atom fcc Sm, `a=5.101908807468821 A` (33.2 A^3/atom), `RADIUS 2.6`, `SPIN 2`, starting `m=6`, PBE without U or SOC, `BZ 3 3 3`, `NBAND 36`, Gaussian DFT/SENERGY integration with 0.002 Ha smearing, Pulay 0.05/20, `CONV 1e-7`, `ITER 120`. `spex -x` parsed it successfully and selected the inversion executable.
- Started the SCF as systemd user unit `thc-sm3-dft.service` in `runs/sm_fcc_3x3_dft`, six MPI ranks with one OpenBLAS thread each: `/usr/bin/time -v mpiexec -n 6 tools/spex/bin/spex.inv --inp=spex.inp`. Combined output: `runs/sm_fcc_3x3_dft/dft.log`.
- The first SCF launch completed iteration 1 (residual `7.626577e-2`, Sm spin moment `5.76 mu_B`) and wrote `spex.pot`, then OpenMPI repeatedly emitted `OSC UCX component priority set inside component query failed` and aborted during the next diagonalization. There was no SPEX error. Restarted from the saved potential with `OMPI_MCA_osc=pt2pt` (same six ranks/threading) as unit `thc-sm3-dft2.service`; output `runs/sm_fcc_3x3_dft/dft2.log`.
- The `pt2pt` restart failed in `MPI_Init` because this OpenMPI 5.0.10 package provides only OSC `sm`, `rdma`, `monitoring`, and `ucx` components. `ompi_info --param osc all` confirmed that inventory. Restarted with the installed intra-node `OMPI_MCA_osc=sm` backend as unit `thc-sm3-dft3.service`; output `runs/sm_fcc_3x3_dft/dft3.log`.
- Added an experiment-only, environment-gated (`SPEX_THC_DUMP`) full-BZ stream dump to the build worktree's `preparedata.f`. The normal generic `spex.dft` stores only the IBZ; this dump records the exact symmetry-expanded `cmt`/`cpw`, per-k G vectors, eigenvalues, and `wintgr` occupation weights used by HF. It does not alter orbitals or exchange. Rebuild unit `thc-spex-build3.service`, output `runs/build/make3.log`.
- Exporter rebuild 3 failed because the root-only `Rcall` macro cannot appear as the single statement of an inline `if`; GNU reported `Syntax error in IF-clause`. Expanded it to a normal `if ... then` block and queued rebuild 4 (`runs/build/make4.log`).
- Rebuild 4 and `make install` succeeded (`runs/build/make4.log`, `runs/build/install3.log`), installing the environment-gated full-BZ exporter.
- Sm 3x3x3 restart with OSC `sm` converged at SCF iteration 70: residual density `5e-8`, Fermi energy `0.34545406 Ha`, metallic moment `6.88887 mu_B`; timed restart run was 17.55 s wall and 82 MB maximum RSS (`runs/sm_fcc_3x3_dft/dft3.log`).
- Prepared `runs/sm_fcc_3x3_ref` with symlinks to the converged potential/mixing files and `JOB DFT0 KS 1:(1-36)` at otherwise identical parameters. Started six-rank OSC-`sm` unit `thc-sm3-ref.service` with `SPEX_THC_DUMP=thc_orbitals.bin`; output `runs/sm_fcc_3x3_ref/ref.log`.
- The first DFT0/KS pass wrote valid `spex.dft` and completed full-BZ preparation, then the experiment-only dump segfaulted in `get_gpt1` because that helper referenced released symmetry-expansion workspace. SPEX's native output is intact. Replaced it with the persistent production mapping `gpt(:,pgpt(ig,k))`; rebuild 5 output is `runs/build/make5.log`.
- Rebuild 5/install succeeded and the repeated 3x3x3 KS pass (`runs/sm_fcc_3x3_ref/ref2.log`) produced a valid 10 MB full-BZ orbital dump in 3.27 s. Header dimensions were `nk=27`, `nk_ibz=4`, `nspin=2`, `maxband=36`, `bando=13`, `maxgpt=259`, and `maxlmindx=204`.
- The user revised the test-side requirement: the experiment must be Rust and must call libmuffintin's existing THC implementation without changing any libmuffintin source. Replaced the planned Python implementation with the external crate `experiment/rust-thc`, depending by path on the unchanged `libmuffintin-thc`, `libmuffintin-auxiliary-ir`, and `libmuffintin-core` crates. The adapter uses the public `PairBlock`, `pivots_from_pair_blocks`, and `fit_per_q` APIs.
- Installed the previously absent Rust toolchain (`rustc`/`cargo` 1.93.1) through the root WSL entry point. This was required by the revised experiment; no libmuffintin files were edited.
- Chose Gamma target bands 5--14 for the reported ten-band window (the SPEX run also prints 15--16), both collinear spins. Bands 6--12 cover the occupied/up-spin and empty/down-spin 4f clusters, with bands 5, 13, and 14 supplying neighbouring sp/d character. The occupied sum remains all `bando=13` valence bands at every full-BZ k point, using SPEX's dumped `wintgr` weights.
- SPEX 3x3x3 MPB-LRI reference: `JOB HF 1:(5-16)`, six ranks, OSC `sm`; `runs/sm_fcc_3x3_ref/hf.log`. Wall 3.51 s, maximum RSS 65 MB. SPEX printed frozen-core exchange separately, so the Rust valence-only result is compared to `sigmax - core contribution` band by band rather than being mislabeled as an all-electron core-inclusive result.
- Added an environment-gated regular-grid trailer to the build-worktree-only exporter. It calls SPEX's production `wavefunction_r1` evaluator for bands 1--16, all full-BZ k points and both spins. The first 32^3 timing attempt was stopped after 117 s because repeated Fourier transforms ran at only about one effective core in this WSL instance. Changed the production sampling to 24^3 (about 0.28 bohr spacing along a primitive vector); this discretization is independently visible as the exact-grid-to-MPB error floor.
- Initial Rust numerical result was smaller than SPEX by almost exactly `(2*pi)^2`. Orbital quadrature norms were near unity, identifying the issue as a reciprocal-metric convention rather than wavefunction normalization. SPEX's dumped `rlat` already contains `2*pi` (`lat^T rlat = 2*pi I`); removed the duplicate factor in the external Rust adapter and reran. This failed attempt and correction are retained in `runs/rust_thc/sm3.log` history through the final CSVs.
- Final 3x3x3 Rust/libmuffintin scan: `runs/rust_thc/sm3_summary.csv` and `sm3_bands.csv`. The 24^3 direct-Fourier limit differs from SPEX valence MPB by max/RMS 4.2369/1.6253 eV. Library QRCP thresholds 1e-1, 3e-2, 1e-2, 3e-3 selected ranks 19, 39, 71, 113; at rank 113 the error is 4.2369/1.6280 eV, i.e. the THC fit has reached the separate Fourier-grid floor. Total Rust scan wall was 88.17 s, peak RSS 447 MB.
- Prepared 4x4x4 Sm from a copy (not a symlink) of the converged 3x3x3 potential. It converged in 21 iterations to residual density `6e-8`, Fermi energy `0.34472376 Ha`, and moment `6.78323164 mu_B`; `runs/sm_fcc_4x4_dft/dft.log`, 7.49 s wall, 81 MB RSS.
- First 4x4x4 KS/export attempt failed as expected because `JOB KS` requires an existing `spex.dft`; changed it to `JOB DFT0 KS 1:(1-36)`. The next exporter attempt segfaulted while writing raw `cmt(:,:,:,:nkpt,:)`: SPEX had allocated coefficients according to its IBZ storage policy, so `nkpt=64` was not a safe raw array bound. Exporter format version 2 omits these unused raw coefficient arrays and retains the production-evaluated grid trailer; rebuild 9/install 7. The repeated export succeeded (`runs/sm_fcc_4x4_ref/export3.log`), producing a 433 MB dump in 14.30 s.
- SPEX 4x4x4 MPB-LRI reference: same `JOB HF 1:(5-16)` target and parameters; `runs/sm_fcc_4x4_ref/hf.log`. Wall 3.41 s, maximum RSS 68 MB. Started the external Rust/libmuffintin scan as systemd user unit `thc-rust-sm4.service`, output `runs/rust_thc/sm4.log`.
- The first 4x4x4 Rust launch was stopped when WSL suspended the unmonitored user service. It also exposed `bando=14` (versus 13 on 3x3x3). Expanded the external packed-column stride from 13 to 14; ten targets times fourteen occupied channels still fit in the unchanged library's 12x12 `PairColumnLayout`. Restarted as `thc-rust-sm4b.service` and kept a foreground WSL poll attached until completion.
- Final 4x4x4 Rust/libmuffintin scan: `runs/rust_thc/sm4_summary.csv` and `sm4_bands.csv`. Thresholds 1e-1, 3e-2, 1e-2, 3e-3 selected ranks 19, 40, 71, 116. At rank 116 the THC-to-direct-grid error is max/RMS 0.0108/0.0036 eV, while error to SPEX valence MPB is 3.1555/1.3112 eV. Total scan wall 206.05 s, peak RSS 1.05 GB. Tight rank rose only 2.7% from the 27-k-point result.
- To audit the strict neutralizing-background convention, added a build-worktree-only exchange experiment setting SPEX's finite-mesh `gamma_divergence` replacement to zero and repeated both HF jobs (`hf_q0.log`). The printed diagonal exchange was byte-for-byte unchanged, showing that the relevant Gamma correction block is inactive in this compiled HF path; the existing MPB references already correspond to the numerical q=0/G=0 omission used by the Rust contraction.
- `cargo test --release` passed for the external experiment crate (no unit tests are defined; this verifies compilation/linkage against the unchanged path dependencies). Initial `cargo fmt --check` failed because `rustfmt` was not installed, so installed the matching Ubuntu `rustfmt` 1.93 package, ran `cargo fmt`, and then `cargo fmt --check` plus the release test successfully.
- Final report written to `notes/RESULTS.md`, including both rank curves, all 20 band-resolved tight-rank values per mesh, matched/unmatched conventions, rerun parameters, and the negative absolute-accuracy verdict.

## 2026-08-28: two-component and Rust NUFFT cross-check

- The user correctly required the scalar-relativistic large and small components
  to be combined.  Audited SPEX `wavefproducts.f`: the physical muffin-tin pair
  density uses the sum of `bas1*bas1` and `bas2*bas2`; the small component is
  absent in the interstitial.  Added `wavefunction_r1_small` only to the detached
  SPEX build worktree and changed both experiment dump formats/readers to form
  `conj(L_i)*L_j + conj(S_i)*S_j`.  No libmuffintin file was changed.
- Evaluated available Rust NUFFT choices.  `apollo-nufft 0.7.0` has a convenient
  reusable 3D interface but requires Rust 1.95, newer than the installed 1.93.1.
  Selected pure-Rust `oxifft` (resolved version 0.4.2), whose type-1 3D transform
  compiles on the installed toolchain.  FINUFFT was kept as an external C++
  reference option, not linked into this Rust-only experiment.
- Added external binary `experiment/rust-thc/src/bin/nufft_crosscheck.rs`.  It
  maps fractional samples to centred NUFFT angles, computes a 24^3 type-1 mode
  box, omits the `q=0,G=0` mode, contracts the exact occupied-state sum, and
  repeats a representative weighted transform by direct NDFT.  NUFFT versus
  NDFT relative error is `2.56501077e-6`.
- Added environment-gated `SPEX_THC_NUFFT_DUMP` sampling to the detached SPEX
  worktree.  The first 9,824-point grid (64 radial shells x 110 Fibonacci
  angles plus a 16^3 interstitial grid) had raw total quadrature weight
  225.90 bohr^3 instead of the 224.044 bohr^3 cell volume because midpoint
  exclusion only approximates the interstitial volume.  Renormalized only the
  interstitial weights so the total is exactly the cell volume before using
  the dump.  Output: `runs/sm_fcc_3x3_ref/nufft_adaptive_ls.bin` (260 MB).
- Medium-grid selected-row run:
  `runs/rust_thc/nufft_crosscheck_strict.csv`.  Large+small errors versus the
  strict SPEX reference were -0.0755 eV (up band 5), -3.6751 eV (up 7),
  -4.5677 eV (up 10), and -0.0471 eV (down 5).  The small component itself
  changed these values by only about 1--2 meV.
- Tightened the adaptive export to 96 radial shells x 194 angular points and a
  24^3 interstitial grid: 27,940 points, 738 MB dump at
  `runs/sm_fcc_3x3_ref/nufft_adaptive_tight_ls.bin`.  Full 20-row output is
  `runs/rust_thc/nufft_crosscheck_tight_full.csv`; wall 60.82 s and peak RSS
  857 MB.  Orbital norms are 0.998010--1.001864.  Max/RMS error versus strict
  SPEX is 4.190964/2.163892 eV.  Up band 5 improves to -0.016857 eV error and
  all down-spin rows are within 0.109 eV, but occupied up-spin 4f errors remain
  2.527--4.191 eV.  Maximum L+S minus L-only effect is 0.0021067 eV.
- Re-audited the Coulomb singularity after the first worktree experiment in
  `exchange.f` produced byte-identical HF numbers.  The active `JOB HF` pole
  replacement is `divergence_x` in `selfenergy.f`, not the earlier patched
  block.  Added a build-worktree-only strict switch and repeated both references:
  `runs/sm_fcc_3x3_ref/hf_q0_strict.log` and
  `runs/sm_fcc_4x4_ref/hf_q0_strict.log`.  Removing SPEX's analytic finite-mesh
  pole replacement shifts affected Gamma exchange by +4.31349 eV on 3x3x3 and
  +3.23515 eV on 4x4x4.  These strict logs now match the Rust experiment's
  literal `q=0,G=0` omission.
- Re-exported regular 24^3 two-component dumps:
  `runs/sm_fcc_3x3_ref/thc_orbitals_grid24_ls.bin` (365 MB) and
  `runs/sm_fcc_4x4_ref/thc_orbitals_grid24_ls.bin` (865 MB).  Reran the complete
  libmuffintin threshold scans against the strict SPEX references.  On 3x3x3,
  ranks are 19/39/71/113 and the tight THC-grid max/RMS is
  0.020881/0.007765 eV; exact-grid versus SPEX is 4.992819/2.240009 eV.  On
  4x4x4, ranks are 19/40/71/116, tight THC-grid is 0.010791/0.003632 eV, and
  exact-grid versus SPEX is 5.034132/2.254486 eV.  Full logs are
  `runs/rust_thc/sm3_ls.log` and `sm4_ls.log`.
- Preserved the previous large-only strict CSVs under
  `sm{3,4}_large_only_strict_{summary,bands}.csv`, then promoted the combined
  L+S results to the canonical `sm3_summary.csv`, `sm3_bands.csv`,
  `sm4_summary.csv`, and `sm4_bands.csv` paths.
- Updated `notes/RESULTS.md`: all main tables now use combined L+S orbitals and
  the strict SPEX convention; added the NUFFT/NDFT cross-check and corrected
  the conclusion.  The evidence rejects "FFT algorithm error" and "missing
  small component" as explanations of the remaining occupied-4f discrepancy.
- Follow-up reciprocal-space audit: a first build after making the NUFFT mode
  count dynamic failed because the new `modes` scalar was shadowed by the local
  five-mode validation array.  Renamed that array and the release build then
  succeeded.  Extended the external Rust binary to accumulate complete
  physical `|q+G|` spheres from a larger centred mode box.  This is the proper
  adaptive reciprocal cutoff for a periodic system, whose frequencies are
  discrete reciprocal-lattice vectors.  Started the 40^3-box / seven-shell run
  as user unit `thc-nufft-recip40.service`; output is redirected to
  `runs/rust_thc/nufft_recip40.log`, with cube results in
  `nufft_recip40_cube.csv` and adaptive spherical convergence in
  `nufft_recip40_shells.csv`.
- The 40^3 reciprocal run completed in 46.24 s wall (1.09 GB peak RSS).  The
  largest complete sphere is 17.511 bohr^-1.  Error versus SPEX is smallest at
  the first 3.502 bohr^-1 shell (1.274 eV max / 0.619 eV RMS) and grows
  monotonically as high-G shells are added, reaching 4.437/2.297 eV at
  17.511 bohr^-1.  For up band 12, successive values are -24.549, -27.251,
  -28.440, -29.055, -29.354, -29.475, -29.570 eV while strict SPEX is
  -25.133 eV.  Thus high-G Fourier convergence moves away from the default
  SPEX MPB value rather than toward it.
- This exposed a further representation mismatch: SPEX default `MBASIS LCUT=5`
  omits the L=6 angular channel present in a 4f x 4f product, while the sampled
  Rust pair density retains it.  Prepared diagnostic strict-HF inputs
  `spex_hf_lcut6.inp` and `spex_hf_lcut7.inp`; their redirected logs are
  `hf_q0_strict_lcut6.log` and `hf_q0_strict_lcut7.log`.
- Both angular-cutoff diagnostics completed successfully.  Adding L=6 changes
  occupied up-4f valence exchange by -2.674 to -4.372 eV; raising L=6 to L=7
  changes those values by only 0.00001--0.00002 eV.  The missing f x f L=6
  channel, not THC, caused the multi-eV default-reference discrepancy.
  Re-referencing the adaptive reciprocal shells to converged LCUT=6 gives
  max/RMS errors 0.2042/0.0996 eV at 15.760 bohr^-1 and 0.2637/0.1224 eV at
  17.511 bohr^-1; the largest latter error is neighbouring up band 14, not 4f.
- Prepared two further MPB diagnostics: `LCUT=6,GCUT=4` and
  `LCUT=6,TOL=1e-5`.  Logs are redirected to
  `hf_q0_strict_lcut6_gcut4.log` and `hf_q0_strict_lcut6_tol1e5.log`.
- The further 3x3 MPB checks completed: increasing GCUT 3 to 4 changes the 20
  valence targets by at most 0.00301 eV; tightening TOL from 1e-4 to 1e-5
  changes them by at most 0.01220 eV.  LCUT=6 is therefore the only
  multi-electronvolt convergence correction.  Prepared the matching 4x4x4
  LCUT=6 strict reference as `hf_q0_strict_lcut6.log`.
- The 4x4x4 LCUT=6 strict reference completed successfully.  To make the raw
  deliverable self-consistent rather than only algebraically rebasing old CSVs,
  launched both complete libmuffintin threshold scans again against the LCUT=6
  logs as `thc-rust-sm3-lcut6.service` and `thc-rust-sm4-lcut6.service`, five
  Rayon threads each.  Redirected outputs are `runs/rust_thc/sm3_lcut6.log`,
  `sm4_lcut6.log` and prefixes `sm3_lcut6`, `sm4_lcut6`.
- Both background THC reruns were stopped together by the WSL user-session
  lifecycle after about 45.5 s, before producing CSVs; neither log contains a
  Rust error and journal peaks were only 0.99/2.2 GB.  This repeats the earlier
  WSL background-service persistence failure.  Restarted them serially inside
  one attached WSL command, with all output still redirected to the same logs.
- Attached reruns completed: 3x3x3 in 95.5 s and 4x4x4 in 225.7 s.  Against
  LCUT=6, the regular 24^3 exact-grid max/RMS errors are 2.3111/1.0620 eV and
  2.3765/1.0987 eV; tight THC errors are 2.3313/1.0671 eV and
  2.3837/1.1008 eV.  The unchanged tight THC-to-grid errors remain
  0.0209/0.0078 and 0.0108/0.0036 eV, so this residual is again the regular
  sampling representation, not the fit.
- At the user's request, started a direct real-Sm Weinert check rather than
  inferring it from Fourier convergence.  Added an environment-gated,
  build-worktree-only SPEX exporter for one finite-q up-spin band-7 pair.  It
  samples both L/S components on all 997 exact SPEX exponential radial shells
  x 194 angles and a 40^3 interstitial grid.  Added external Rust binary
  `weinert_pair_check`, which calls the unchanged public
  `libmuffintin-coulomb::assemble_sampled_coulomb` API and independently
  evaluates the same pair with oxifft.  Build/export/run logs are assigned
  `runs/build/make_weinert.log`, `runs/sm_fcc_3x3_ref/weinert_export.log`, and
  `runs/rust_thc/weinert_pair_check.log`.
- The SPEX exporter rebuilt/installed successfully.  The first Rust Weinert
  build failed because Cargo package names (`libmuffintin-*`) differ from their
  Rust crate names (`muffintin_*`); corrected only the external adapter imports.
  The next release build succeeded.
- The real-pair export completed in 3.87 s SPEX time and produced a 20 MB dump:
  236,304 points, 193,418 in the muffin tin.  The reconstructed libmuffintin
  quadrature sum is 224.044705351315 bohr^3 versus 224.044705351558 bohr^3 cell
  volume.  Pair transfer is fractional `[0,0,1/3]`, Cartesian
  `[0.2172336,0.2172336,-0.2172336]` bohr^-1.
- Direct unchanged-libmuffintin Weinert results for the real Sm pair:
  L5/G3 1.027807792 eV, L5/G4 1.027994902 eV, L6/G3 1.067503423 eV,
  L6/G4 1.067694427 eV.  Imaginary residue is below 3e-18 Ha.  Independent
  Fourier spheres on the identical pair give 0.439651559, 0.640856985,
  0.965315005, 1.072652386, and 1.110260181 eV at cutoffs 3, 4, 8, 12,
  and 15.76 bohr^-1.  Weinert L6/G4 versus the clean Fourier-12 value differs
  by 4.96 meV.  Runtime was 262.4 s wall, peak RSS 194 MB.
- Preserved the combined-component default-LCUT5 CSVs as
  `sm{3,4}_lcut5_{summary,bands}.csv`, then promoted the completed LCUT=6 runs
  to canonical `sm3_summary.csv`, `sm3_bands.csv`, `sm4_summary.csv`, and
  `sm4_bands.csv`.  Rewrote `notes/RESULTS.md` around the corrected conclusion:
  THC compression succeeds; the original multi-eV discrepancy was primarily
  the missing SPEX L=6 product channel; NUFFT is diagnostic rather than
  required; and libmuffintin Weinert works on a real Sm finite-q pair.

## 2026-08-28: compact core--valence exchange

- Audited libmuffintin and SPEX core paths.  `libmuffintin-thc` can tag one
  fixture orbital as core for residual grouping, and auxiliary-IR/MPB supports
  selected core radial functions.  Real Sm has many relativistic core shells,
  however, and SPEX `exchange_core` already exposes a stronger simplification:
  the core-valence pair is confined to one muffin tin.  For collinear valence
  states all core shells/multipoles can be folded into k/band-independent
  radial blocks `K_core(site,l,spin;n,n')`, followed by
  `-sum_m c^dagger K c` for each band.
- Added an environment-gated `SPEX_CORE_COMPACT_DUMP` experiment only to the
  detached SPEX build worktree.  It accumulates the exact L+S intra-sphere
  core kernel and writes the small blocks plus target valence `cmt`
  coefficients.  Build log: `runs/build/make_core_compact.log`.
- First compact export failed intentionally at a guard which rejected
  `lcore_soc=true`.  In this calculation valence SOC/noncollinearity is off but
  SPEX retains relativistically split core shells; the scalar compact form
  remains valid because the split-shell degeneracy is already in `qfac0` and
  the m-dependent SOC terms require valence `l_soc`.  Narrowed the guard to
  reject only noncollinear/valence-SOC cases, rebuilt
  (`make_core_compact2.log`), and reran successfully.
- The 3x3x3 compact dump is 79 KB because it also carries 24 target-state cmt
  records.  The reusable core operator itself has l=0..9 radial sizes
  `[3,3,2,2,2,2,2,2,2,2]`: 72 active symmetric doubles, exactly 576 bytes
  across both spins.  External Rust `core_compact_check` reconstructs all 24
  printed SPEX core rows with max/RMS 2.070e-6/1.481e-6 eV, limited by the
  seven-decimal values stored in the dump.
- Added a block-eigen truncation scan.  The first Cargo build failed on a
  closure-lifetime error while collecting faer eigenvectors; replaced it with
  explicit row/column loops.  A transient Windows-to-WSL UNC semaphore timeout
  delayed that patch; waking Ubuntu and retrying succeeded.
- Core truncation results on 3x3x3: exact block rank 44; threshold 1e-1 rank
  20 gives max/RMS 0.22677/0.16490 eV; 1e-2 rank 24 is essentially identical;
  3e-3 rank 37 gives 2.677e-5/7.059e-6 eV; 1e-6 restores rank 44 and the
  printed-reference floor.  A rank-37 factor needs 119 doubles including
  rotations, so it is larger than the exact 72-double symmetric blocks.  The
  rank-20 factor needs 64 doubles but saves only 64 bytes while adding 0.227 eV;
  exact symmetric storage is the recommended representation.
- Repeated the compact export/reconstruction on 4x4x4.  It has the same block
  dimensions and 576-byte exact storage; exact max/RMS is
  2.070e-6/1.481e-6 eV and the rank-37 result is
  2.675e-5/7.068e-6 eV.  Outputs:
  `runs/rust_thc/core_compact_{summary,bands}.csv` and
  `core_compact_sm4_{summary,bands}.csv`.
- Updated `notes/RESULTS.md` with the representation, both error curves, exact
  storage/cost, limitations for valence SOC/multiple sites, and the recommended
  split `Sigma_x = Sigma_valence^(THC+Weinert) + Sigma_core^(local blocks)`.
  No libmuffintin source was modified.
- Traced the initial 2.07e-6 eV exact-block residual to a unit-conversion
  mismatch: the Rust adapter used a newer CODATA Hartree-to-eV value while the
  SPEX reference was multiplied by SPEX's compiled `hartree` constant.  Bumped
  the experiment dump to version 2 and exported that exact constant.  After
  rebuilding and repeating both meshes, the exact compact contraction agrees
  with SPEX to all 12 CSV decimal places (reported max/RMS 0/0).  Final rank-37
  max/RMS errors are 2.687e-5/6.932e-6 eV on 3x3x3 and
  2.685e-5/6.941e-6 eV on 4x4x4.
- Extended the Rust checker with an optional join against the canonical tight
  valence THC band tables.  It writes `core_valence_total_sm3.csv` and
  `core_valence_total_sm4.csv` with separate THC-valence, compact-core,
  combined, SPEX-valence, SPEX-core, and SPEX-total columns.  Since the core
  blocks are exact, the combined error equals the valence error by construction;
  the explicit columns make double counting auditable.

## 2026-08-28: relativistic core-valence representation note

- Started an idle Ubuntu keepalive (`sleep infinity`, WSL PID 414) so the
  distro remains awake during document/source inspection. Its redirected logs
  are `runs/wsl_keepalive.stdout.log` and `runs/wsl_keepalive.stderr.log`;
  both were empty after startup.
- The first shell-only keepalive launch failed before starting a process
  because nested PowerShell/WSL quotes truncated the Bash `if`. Replaced it
  with a hidden Windows `Start-Process` invoking
  `wsl -- env THC_SMDY_KEEPALIVE=1 sleep infinity`, then verified PID 414 in
  Ubuntu. A concurrent source-read batch also produced a transient WSL
  `E_UNEXPECTED`; waking the distro and reading serially succeeded.
- Added `notes/core-valence.md`. It fixes the production split as valence THC
  plus Weinert and exact local core exchange, then gives separate contractions
  for full-Dirac core with KH valence, second-variation SOC, 4c-in-MT
  SRA-LAPW, magnetic/noncollinear first variation, and 4c LMTO. It explicitly
  distinguishes a site-local operator from a center-diagonal MTO matrix and
  treats the current LMTO material as a plan rather than an implementation.
  No file under `codes/libmuffintin` was modified.
