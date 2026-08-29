# 17. Minimal full-potential DFT workflow

This note defines the closed DFT contract, the orbital-configuration V2 extension, and the public-library neutral atomic-start path. These contracts consume the anonymous-basis LAPW facade, the Weinert conventions, and the Dirac valence scalar and four-component radial substrate; they do not create a second basis, Coulomb, or Dirac convention. The implemented scope is regular-full-Brillouin-zone LDA/PBE self-consistency, current-potential radial-energy generation, Fermi–Dirac or Gaussian occupations, three density mixers, scalar Koelling–Harmon bands, optional SOC second variation, collinear and noncollinear four-component first variation, frozen-potential bands, versioned restart snapshots, tetrahedron DOS, and a public neutral-atom-superposition generator for a validated Snapshot V2 restart. Energies and temperatures are Hartree and lengths are Bohr.

## 1. References and fixed execution order

The SPEX anchors in this note were inspected in the local directory `spex06.00pre36` at commit `b7778ba9f15ea30274fa1f6962a1d531c5679e5d`; its `ChangeLog` identifies the live tree as 06.00pre38, and line numbers are not claimed for another release. The FLEUR Pauli-field and energy-parameter anchors were inspected in the local FLEUR checkout, while Elk energy search was inspected in `elk-10.4.9`. The all-electron FlapwMBPT radial/basis reference is the local `FlapwMBPT2106_type_B.tar.gz` archive, whose source root is `FlapwMBPT.2106/src`; the obsolete claim that only `ComDMFTv.2.0/src/FlapwMBPT` is relevant is not retained.

SPEX `src/iterate.f:841-1788` and FlapwMBPT `dft_loop.F:1-35` fix the iteration dependency order implemented by `ScfPhysics` and `run_scf`:

```text
input density
  -> electronic Hartree + periodic nuclei + XC potential
  -> four-component core solve at every site
  -> materialize the immutable channel recipe on the current potential
  -> radial basis, LAPW matching, H and S
  -> regular full-BZ eigensolutions, chemical potential, and occupations
  -> same-potential spectral refinement for band-cog or fermi-offset, if requested
  -> valence density + P^2 + Q^2 core density
  -> total energy, physical residual, convergence, and mixing
```

The driver owns this order, occupation counting, convergence, and mixer history. A material kernel owns density initialization, electrostatics and XC, radial construction, scalar/SOC routing, eigensolutions, orbital density synthesis, and frozen-potential spectra. The DFT layer consumes compiled capabilities and never depends on a concrete `LapwBasis` façade.

## 2. One runtime, ordered tasks, and the library seam

`libmuffintin-runtime` provides the single `muffintin` binary and an ordinary Rust library boundary suitable for a later Python binding. The binary is not named after DFT because the same versioned task registry can acquire THC or other workflows without creating another executable.

One TOML document has one explicit execution-order array and one block for each named task. The array is authoritative; TOML map order is not execution order. A consumer names an earlier typed output such as `scf.state`, so a later bands or DOS task reuses that exact converged potential, basis, relativity route, and state rather than mutable ambient solver state.

```toml
format = "libmuffintin-input"
version = 2
snapshot = "material.snapshot.toml"

[workflow]
tasks = ["scf", "bands", "dos"]

[task.scf]
kind = "dft-scf"
electron-count = 14.0

[task.scf.k-mesh]
mesh = [8, 8, 8]
shift = [0.5, 0.5, 0.5]

[task.scf.basis]
l-max = 8
energy-generator = "atomic"
recipe = "recipes/si.toml"

[task.scf.basis.envelope]
kind = "plane-wave"
cutoff = 4.0

[task.scf.basis.channels.Si]
lo = ["3s", "3p"]

[task.scf.basis.channels."Si-1"]
lo = ["+3d@-0.15"]
hdlo = ["+4f"]

[task.scf.occupations]
kind = "fermi-dirac"
temperature = 0.001

[task.scf.xc]
kind = "lda-pw92"
noncollinear-route = "local-spin-frame"

[task.scf.mixing]
kind = "pulay-anderson"
beta = 0.4
history = 6

[task.scf.relativity]
kind = "soc-second-variation"
band-window = [0, 24]

[task.scf.convergence]
energy-tolerance = 1.0e-8
density-tolerance = 1.0e-7
max-iterations = 80

[task.bands]
kind = "dft-bands"
source = "scf.state"
bands = 16

[[task.bands.path]]
label = "G"
k = [0.0, 0.0, 0.0]

[[task.bands.path]]
label = "X"
k = [0.5, 0.0, 0.0]

[task.dos]
kind = "dft-dos"
source = "scf.state"
points = 401
broadening = 0.005

[task.dos.k-mesh]
mesh = [16, 16, 16]
shift = [0.0, 0.0, 0.0]

[task.dos.energy-window]
minimum = -1.0
maximum = 1.0
```

Task child data use named subblocks plus typed arrays such as band-path points. Unknown fields, task kinds, orphan blocks, duplicate IDs, forward sources, and incompatible outputs are errors. Input V1 is rejected with a migration diagnostic: `[basis.envelope]` replaces the flat `plane-wave-cutoff`, and the channel table replaces `local-orbitals` and `state-overrides`. V2 has no compatibility aliases for those removed fields.

The channel table separates identity from treatment. A token is a spectroscopic state such as `5p`, `5p-`, `5p+`, `5p1/2`, or `5p3/2`; the two $j$ spellings normalize to the same signed $\kappa$. Treatments are `core`, `valence`, `lo`, and `hdlo`. Species rows are complete treatment replacements, while site rows contain only `+` additions and `-` removals. A suffix selects a low-frequency per-channel exception: `@atomic`, `@band-center`, `@log-derivative`, `@band-cog`, `@fermi-offset`, `@frozen`, or an explicit energy such as `@-0.15` or `@-4.08eV`. Bare numbers are Hartree and `eV` suffixes are converted once during normalization.

The runtime starts from the typed FLEUR `default.econfig` catalogue for $Z=1$ through $103$. The deterministic merge order is built-in defaults, external recipe artifact, task-level `energy-generator`, species replacement, site edit, then token suffix. An external recipe is the same normalized serde IR used for provenance; workflow TOML does not accept a second inline typed-record syntax. Built-in valence partners are lowered to one scalar $l$ recipe and expanded back to signed $\kappa$ only at the route boundary. Missing base channels through `l-max` are added with the FLEUR rule that selects the first unrepresented $n$, generalized to arbitrary $l$.

Every materialized record retains its site, $(n,l|\kappa)$ identity, treatment, derivative order, generator, immutable seed, realized energy, and recipe provenance. `derivative-order` is an arbitrary nonnegative integer in the IR. V2 executes the ordinary order-zero/order-one linearized pair and order two through `hdlo`; order three and above remains schema-valid but returns a typed execution-time `NotImplemented` error. Fill generators, generation starting from $l$, and seed fallback remain deferred; a failed bracket never silently keeps the seed.

The seven executable energy generators have one current-potential seam:

| Generator | Implemented value |
|---|---|
| `explicit` | The exact normalized Hartree seed |
| `atomic` | A node-selected bound Dirac energy for each signed $\kappa$ |
| `band-center` | The midpoint between the current-potential $u(R_{\mathrm{MT}})=0$ top and $u'(R_{\mathrm{MT}})=0$ bottom |
| `log-derivative` | The selected branch at $R_{\mathrm{MT}}u'/u=-(l+1)$ |
| `band-cog` | The occupied projected-DOS first moment divided by its physical weight |
| `fermi-offset` | The current chemical potential plus the recipe offset, with the Elk default $-0.1$ Ha when absent |
| `frozen-snapshot` | The exact snapshot anchor, retained as the FLEUR/SPEX regression route |

`atomic` expands a scalar $l$ recipe into its signed $\kappa$ partners and uses exact $2j+1$ degeneracies when folding back to scalar. Full first variation keeps the partner energies distinct. `band-cog` projects $d=P_{\mathrm{site}}C$ and evaluates the full selected muffin-tin quadratic form $d^\dagger S_{\mathrm{MT}}d$, including nonorthogonal radial cross terms; occupation, k weight, and state degeneracy are multiplied exactly once by the generator. `band-cog` and `fermi-offset` run in a bounded inner basis-refinement loop after occupations but before density synthesis, against the same outer-iteration potential. Every outer iteration starts again from the immutable recipe and snapshot provisional anchor, not the previous iteration's generated energy.

## 3. Regional density and the physical metric

The single density representation is charge plus Cartesian magnetization, with each of $n,m_x,m_y,m_z$ stored as a `RegionalScalarField` over identical muffin-tin meshes and an exact reciprocal layout. The density-matrix convention is

```math
\rho=\frac12\left(nI+\mathbf m\cdot\boldsymbol\sigma\right).
```

A collinear density is the exact subset $n=n_\uparrow+n_\downarrow$, $m_z=n_\uparrow-n_\downarrow$, and $m_x=m_y=0$; it is not a separate SCF state type. The interstitial density coefficients represent the periodic orbital extension, and physical integrals apply the analytic interstitial step function. A regional potential instead stores $V_0,B_x,B_y,B_z$ with matrix-element-ready masked interstitial coefficients and convention $V=V_0I+\mathbf B\cdot\boldsymbol\sigma$. This separation prevents a masked potential from being reused as an unmasked Poisson source.

For scalar LAPW eigenvectors, the muffin-tin coefficients are formed after the unique compiled site projection. Large and scalar-relativistic small radial products enter separately. The interstitial term uses the exact plane-wave difference $G_{\mathrm{right}}-G_{\mathrm{left}}$, the cell normalization $1/\Omega$, explicit k weights, and band occupations. Full-spinor synthesis retains all physical $P/Q$ pair products and both interstitial Pauli components, so transverse magnetization remains in the same `RegionalDensity` used by mixing, convergence, XC, restart, and the next Hamiltonian.

For regional scalar fields, let $\langle a,b\rangle_R$ denote the muffin-tin radial integral plus the interstitial step-function contraction. The density metric is the Pauli trace

```math
\langle \rho_a,\rho_b\rangle
=\frac12\left[\langle n_a,n_b\rangle_R+\sum_{j=x,y,z}\langle m_{a,j},m_{b,j}\rangle_R\right].
```

When $m_x=m_y=0$, this reduces exactly to the previous sum of explicit up/down metrics. Every mixer and the reported density RMS uses this metric; serialized coefficient order is never treated as a Euclidean physical norm.

## 4. Four-component core density

Each automatically selected bound-core channel is identified by principal quantum number, signed Dirac $\kappa$, and its FLEUR occupation not exceeding $2|\kappa|$. User overrides are applied before the core count is formed. The solver isolates a unique bracket below the outer continuum with the required large-component node count, performs the two-sided four-component Dirac solve on an extended physical radial potential, and reports explicit `NotFound` or `Ambiguous` failures rather than selecting a root heuristically.

The true muffin-tin density is always

```math
n_c(r)=\frac{f_c}{4\pi r^2}\left[P_c(r)^2+Q_c(r)^2\right].
```

The extended tail is not discarded. Inside the sphere, a SPEX-style smooth pseudocharge matches the boundary value and derivative; its spherical-Bessel transform supplies the interstitial Fourier contribution with the exact site phase. After assembly at finite $G$, each spin's constant coefficient is solved from the regional step-function integral so the represented charge equals the requested occupation exactly at any reciprocal cutoff; moments at finite $G$ and the true muffin-tin $P^2+Q^2$ density are unchanged, and the adjustment is reported explicitly. Closed shells split equally between the two collinear channels, while an explicit spin partition preserves the total occupation without double counting.

The frozen-snapshot bootstrap copies the physical muffin-tin monopole and obtains the outer shape from the snapshot's periodic interstitial field. A compact cubic Hermite representation bridge matches value and slope at the muffin-tin boundary and decays with zero value and slope correction at the outer mesh endpoint, where the original periodic field is recovered; the join parameters and residuals are diagnostics rather than a hidden atomic tail. The same explicit bridge reconciles the independently discretized muffin-tin XC projection and pointwise outer XC in later iterations, while raw unmasked electrostatics remains unchanged. Core bracket scans are measured relative to the outer continuum, so a constant potential-energy shift moves the bracket and eigenvalue together. The resulting initial $P^2+Q^2$ density makes the first Hartree input neutral, and every subsequent iteration rebuilds the physical continuation before solving the core again.

## 5. Weinert electrostatics and nuclei

Weinert is a general Coulomb/Poisson component in `libmuffintin-coulomb`, not a DFT-owned algorithm. Its public inputs are geometry, muffin-tin multipoles, a Hermitian Fourier charge field, and an explicit constant-mode treatment; it has no dependency on SCF, occupations, XC, or `libmuffintin-dft`. The DFT adapter only converts a regional electronic density and periodic nuclei to that reusable boundary, evaluates DFT energy contractions, and masks the returned operator potential. It does not route a charge density through `CompiledAuxiliaryBasis` and does not create a second Coulomb convention.

Positive electron number density first produces the repulsive electronic Hartree potential. Its $G=0$ source is removed by an explicit uniform-background treatment and its potential gauge is $V_{G=0}=0$. Periodic point nuclei are then evaluated separately with

```math
V_{\mathrm{nuc},G}=-\frac{4\pi}{\Omega G^2}\sum_s Z_s e^{-iG\cdot R_s},\qquad G\ne0,
```

and the own-site muffin-tin monopole restores $-Z_s/r$. The combined adapter requires physical electron and nuclear source charges to cancel. Raw electrostatic Fourier coefficients remain available for physical continuation and energy contractions; only the operator-facing `RegionalPotential` receives the step-function convolution.

The energy adapter evaluates $C=\int n(V_H+V_{\mathrm{nuc}})$, $M=E_{en}+2E_{II}$, $E_H=\tfrac12\int nV_H$, $E_{en}=\int nV_{\mathrm{nuc}}$, and $E_{II}$ in one common gauge. Discrete muffin-tin and Fourier boundary values are matched using the actual Poisson mesh endpoint, not a second independently rounded multipole quadrature.

## 6. LDA/PW92, PBE, and noncollinear XC

The implemented local-density choice follows the `xlda + cpw92` formulas, and the gradient choice follows `xpbe + cpbe` (`src/iterate.f:261-326`, `src/xcfunc.f:105-141,564-783`). The collinear point kernel accepts both spin densities, gradients, and Hessians. PBE includes the density gradient, Laplacian, and gradient–Hessian contraction required by the divergence term in the functional derivative.

The public noncollinear choices are method names, not producer names. `LocalSpinFrame` rotates the density locally onto $\hat{\mathbf m}$, independently projects the magnetization value, first derivatives, and second derivatives onto that direction, evaluates the two-channel point kernel, and rotates the splitting back to $\mathbf B_{xc}$. This matches the locally collinear construction in SPEX 06.00pre38 `potential.f:586-600,869-889,1001-1022`; the same Pauli matrix convention $V_0I+\mathbf B\cdot\boldsymbol\sigma$ is explicit in FLEUR `math/BfieldtoVmat.f90:9-124` and `vgen/xcBfield.f90:27-36`. At a magnetization node $|\mathbf m|<10^{-12}$, where a local spin direction is undefined, the polarization jet and $\mathbf B_{xc}$ are set to zero rather than selecting a global axis; this preserves global spin-rotation covariance even when the magnetization derivatives at the node are nonzero.

`MagnetizationField` instead treats $(n\pm|\mathbf m|)/2$ as scalar eigenvalue fields. Its PBE route differentiates the full magnetization magnitude, including

```math
\partial_i\partial_j|\mathbf m|
=\frac{\partial_i\mathbf m\cdot\partial_j\mathbf m+\mathbf m\cdot\partial_i\partial_j\mathbf m}{|\mathbf m|}
-\frac{(\mathbf m\cdot\partial_i\mathbf m)(\mathbf m\cdot\partial_j\mathbf m)}{|\mathbf m|^3}.
```

Both routes return $V_0=(v_\uparrow+v_\downarrow)/2$ and $\mathbf B_{xc}=(v_\uparrow-v_\downarrow)\hat{\mathbf m}/2$, without an extra factor of two or $\mu_B$, and report the total-energy contraction $\int(nV_0+\mathbf m\cdot\mathbf B_{xc})$. In the interstitial, analytic Fourier derivatives supply the density jet and a deterministic midpoint grid performs the nonlinear forward/inverse transform. Inside each sphere, a deterministic angular rule, quartic radial interpolation, and fourth-order Cartesian differences supply the same point-kernel contract.

## 7. Occupations and chemical potential

A state carries an eigenvalue $\epsilon_i$, regular-mesh weight $w_i$, and an explicit positive integer degeneracy $g_i$. Its capacity is $w_i g_i$. Nonmagnetic scalar bands may use $g_i=2$ only when spin is not explicitly enumerated; collinear and spinor routes use $g_i=1$. The concrete DFT kernel enumerates explicit spin states and normalizes full-BZ k weights to one.

For finite electronic temperature $T>0$,

```math
f_i(\mu,T)=\frac{1}{1+\exp[(\epsilon_i-\mu)/T]},
\qquad
N=\sum_i w_i g_i f_i.
```

The monotone count is dynamically bracketed and bisected. Separate logistic tails prevent overflow. Exactly empty and exactly full finite-temperature spectra require infinite chemical potentials, so the finite-Hartree API rejects those endpoints.

SPEX `src/gauss.f:17,57-62` uses standard deviation $\sigma$ and

```math
f_i^G(\mu,\sigma)=\frac12\,\mathrm{erfc}\left(\frac{\epsilon_i-\mu}{\sqrt{2}\,\sigma}\right).
```

The implementation evaluates the smaller `erfc` tail first through `libm`. The optional SPEX temperature-like input conversion remains an explicitly named adapter,

```math
\sigma=\frac{4}{\sqrt{2\pi}}T,
```

so the Gaussian and Fermi–Dirac slopes agree at the chemical potential.

The Fermi–Dirac entropy and variational correction are

```math
S=-\sum_i w_i g_i\left[f_i\ln f_i+(1-f_i)\ln(1-f_i)\right],
\qquad
-TS=-T S.
```

The Gaussian functional instead reports

```math
C_\sigma=-\frac{\sigma}{\sqrt{2\pi}}\sum_i w_i g_i\exp\left[-\frac{(\epsilon_i-\mu)^2}{2\sigma^2}\right].
```

$C_\sigma$ is a generalized broadening correction, not a thermodynamic entropy. SPEX applies Gaussian occupations but does not include this correction in `src/iterate.f:874-911`; it is an explicit variational addition.

The input `electron-count` is the total electron count. The driver subtracts the automatically selected, override-adjusted core occupations exactly once before solving valence occupations and rejects a configuration with no represented valence electron.

## 8. Scalar, second-variation SOC, and four-component routes

The scalar route constructs Schrödinger or Koelling–Harmon radial solutions, analytic first and second energy derivatives, distinct-energy local orbitals, and HDLOs from the materialized channel recipe. Scalar $l$ atomic records are generated $\kappa$ first and folded with exact partner degeneracies. A magnetic `frozen-snapshot` retains the original up/down energy anchors bitwise for the two scalar channels. LO and HDLO records with signed $\kappa$ are spinor-only and are omitted rather than silently reinterpreted as scalar $l$ orbitals. Scalar-relativistic overlap uses the physical $PP+QQ$ norm. Nonspherical sphere fields enter through Gaunt matrix elements, and all site blocks use the same compiled projection order as density synthesis.

The optional nonmagnetic SOC second-variation route first solves a scalar Koelling–Harmon problem, selects an explicit source-band window, projects the site $\xi(r)\,\mathbf L\cdot\mathbf S$ operator into that subspace, solves the ordinary Hermitian doubled-spin problem, and reconstructs global Pauli spinors. This route rejects magnetic, noncollinear, full-spinor first-variation inputs, and active local orbitals labeled by signed $\kappa$ rather than silently changing the approximation. A runtime window must begin at the lowest represented valence band because the concrete V2 kernel does not silently drop occupied scalar states below the window. A site edit that removes or reclassifies the automatic LO with signed $\kappa$ explicitly requests the conventional second-variation approximation for that channel.

The full first-variation route keeps physical four-component $P/Q$ orbitals in every muffin-tin sphere and a two-component SRA interstitial basis. A scalar $l$ recipe expands to the complete partner set with signed $\kappa$ through `l-max`; atomic and spinor `band-cog` generation retain distinct partner energies, while a generator resolved by $l$ supplies the same value to both partners. Explicit LO and HDLO requests remain labeled by exact signed $\kappa$. The default recipe also assigns the sixth-period $5p_{1/2}$ channel for $Z=55\ldots86$ and the supported seventh-period $6p_{1/2}$ channel for $Z=87\ldots103$ to ordinary `lo` treatment with the `atomic` generator. The same current-potential generator seam supplies these confined full $P/Q$ LOs; there is no ambient relativistic-LO energy map and no duplicate bound-state resolver. Multiple LOs sharing one signed $\kappa$ remain distinct. A site `-` edit removes an automatic LO or a site `+` edit reclassifies the channel without a separate state-override interface. Sphere and interstitial magnetic fields both enter as $V_0I+\mathbf B\cdot\boldsymbol\sigma$, including the exact off-diagonal factors $B_x\mp iB_y$, with no scalar fallback. The synthesized $m_x,m_y,m_z$ fields pass unchanged through mixing, XC, the next 4c Hamiltonian, and restart serialization.

Snapshot V2 stores shared geometry and radial-basis metadata plus either a frozen Pauli potential or a restart pair containing $n,m_x,m_y,m_z$ and $V_0,B_x,B_y,B_z$. Snapshot V1 remains readable and normalizes exactly to V2: scalar data maps to $V_0$ with zero $\mathbf B$, while up/down data maps to $V_0=(V_\uparrow+V_\downarrow)/2$, $B_z=(V_\uparrow-V_\downarrow)/2$, and zero transverse fields. The runtime uses header-based version dispatch and one normalized V2 path.

## 9. Neutral atomic-start Snapshot V2

`FreeAtomScfSpec` makes the exponential radial mesh, potential mixing, potential tolerance, tail tolerance, and maximum iteration count explicit. Public `run_free_atom_lda` starts from the bare nuclear potential and resolves the neutral FLEUR occupation catalogue into every occupied signed $\kappa$ channel. Each radial iteration solves those bound Dirac orbitals, forms the physical $P^2+Q^2$ density, evaluates the isolated spherical Hartree potential and unpolarized LDA/PW92 potential, and mixes the resulting effective potential. Success requires the potential residual, integrated-charge error, and outer logarithmic-shell charge to satisfy the caller's tolerances. Bound-state, quadrature, XC, convergence, and tail failures remain typed rather than producing a partial atomic state.

`build_atomic_superposition_density` places the converged neutral atoms on the supplied crystal and sums every periodic image whose verified radial tail intersects a target muffin-tin sphere. It evaluates the overlapped density on the target radial/angular grid and projects all requested nonspherical muffin-tin channels instead of retaining only a spherical on-site term. Its interstitial Fourier coefficients are neutral-atom form factors on the exact production density layout: the union of every $G_{\mathrm{right}}-G_{\mathrm{left}}$ difference from every plane-wave envelope on the regular k mesh. After muffin-tin projection and finite Fourier truncation, only the represented $G=0$ coefficient is corrected so the regional integral equals the requested charge; finite $G$ coefficients and the nonspherical muffin-tin density are unchanged. Interpolation beyond the verified atomic tail is a typed error.

The path is deliberately nonmagnetic and neutral-only. `electron_count` must satisfy

```math
N_e=\sum_s Z_s,
```

and the generated magnetization components are zero. Public `materialize_atomic_snapshot_v2` takes the structure, DFT task, and free-atom controls, constructs the exact production layout and atomic-superposition density, then uses the production Weinert electronic Hartree, periodic nuclei, and task-selected XC path to build the operator potential. It emits `InitialV2::Restart { density, potential }` inside a validated `SnapshotV2`. A pre-resolved basis is rejected, as are the snapshot-dependent `frozen-snapshot` generator and the spectral `band-cog` and `fermi-offset` generators, which require information unavailable before the first crystal solve.

This is a public library generator for a cold-start restart artifact. It is not a new one-run CLI input, does not guess local or total magnetic moments, and does not add a magnetic superposition-of-atoms policy.

## 10. Mixing, convergence, and total energy

Linear mixing, type-2 Broyden, and Pulay–Anderson all use residual $r=\rho_{\mathrm{in}}-\rho_{\mathrm{out}}$ and the same four-component regional physical inner product. Each nonlinear mixer owns bounded history and rejects a layout change instead of mixing unrelated coefficient vectors.

An iteration converges only after both a density RMS threshold and a total-energy-change threshold are available and satisfied. The retained state is the accepted fixed-point input density together with the potential it generated, the immutable normalized channel recipe, exact materialized channel energies, chemical potential, relativity route, total energy, and ordered diagnostics. Each iteration diagnostic carries the same realized channel records used for its final band and density pass, including generator, seed, realized energy, method diagnostic, and recipe provenance. Frozen-potential bands and DOS therefore consume the basis stored in their selected `ScfState`, not the kernel's most recently materialized basis.

The finite-temperature total-energy expression is assembled once as

```math
E=E_{\mathrm{band}}+E_{\mathrm{core}}+\frac{M-C}{2}+E_{xc}-\int\left(nV_0+\mathbf m\cdot\mathbf B_{xc}\right)+E_{\mathrm{occ}},
```

where $E_{\mathrm{occ}}$ is either $-TS$ or $C_\sigma$. Core eigenvalues, band energy, electrostatics, XC, and the occupation correction each enter exactly once.

## 11. Bands and DOS

A bands task consumes an earlier `ScfState` and solves its frozen potential on the requested ordered reciprocal path. A DOS task consumes the same typed state, solves a regular full-BZ mesh, and applies the linear tetrahedron method. The implementation uses periodic eight-corner cells, six tetrahedra per cell, the shortest body diagonal with deterministic tie averaging, and normalized integrated band capacity. Smearing is not used inside the tetrahedron integral; any requested broadening is downstream presentation metadata.

## 12. Acceptance status and evidence boundary

The implementation has focused gates for weighted occupations, Gaussian and logistic tails, scalar/4c radial identities, local orbitals and HDLOs labeled by signed $\kappa$, all seven radial-energy generators, band-center brackets and diagnostics, degeneracy averaging over signed $\kappa$, physical nonorthogonal `band-cog` projection, same-iteration spectral refinement, bitwise frozen anchors, recipe merge priority and editing, automatic rLO injection/removal, V1 migration failure, V2 round-trip, plane-wave and sphere density synthesis, core $P^2+Q^2$ and spill, Weinert electronic and nuclear potentials, electrostatic boundary matching, LDA/PW92 free-atom convergence, periodic-image superposition with nonspherical muffin-tin projection, finite-layout charge closure, production-potential restart materialization, LDA/PBE functional derivatives, both noncollinear XC reductions, spin-rotation covariance, transverse Pauli blocks, four-component mixing, Snapshot V1-to-V2 normalization and restart, SCF ordering and source reuse, scalar and SOC eigensolutions, and tetrahedron normalization. The unified CLI also executes a minimal one-site authored snapshot through the concrete kernel; the atomic-snapshot generator remains a library generator rather than another CLI input route.

These gates close the DFT implementation contract and the orbital-configuration V2 extension at the library, TOML workflow, snapshot/restart, and executable levels. They also close the public-library neutral atomic-start path and therefore the v0.2 implementation sequence. This closure does not claim a release tag, material acceptance, or cross-code validation. Si and SrVO3 scalar results, Pt or Au second-variation SOC, collinear and noncollinear magnetic fixtures, and a regular-mesh tetrahedron DOS still require frozen cross-code reference artifacts before the DFT route can be called cross-code accepted or production validated. Meta-GGA, hybrids, forces, SCF symmetry reduction, tetrahedron occupations, charged atomic starts, and magnetic moment initialization remain outside this contract.
