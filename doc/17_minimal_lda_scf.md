# 17. Minimal full-potential DFT workflow

This note defines the M-Kb implementation candidate. M-Kb consumes the M-G LAPW basis/operator boundary, the M-J Weinert conventions, and the M-Ka scalar and four-component radial substrate; it does not create a second basis, Coulomb, or Dirac convention. The implemented scope is regular-full-Brillouin-zone LDA/PBE self-consistency, Fermi--Dirac or Gaussian occupations, three density mixers, scalar Koelling--Harmon bands, optional SPEX-style second-variation SOC, a four-component first-variation library route, frozen-potential bands, and tetrahedron DOS. Energies and temperatures are Hartree and lengths are Bohr.

## 1. References and fixed execution order

The SPEX anchors in this note were inspected in `spex06.00pre36` at commit `b7778ba9f15ea30274fa1f6962a1d531c5679e5d`; line numbers are not claimed for another SPEX release. The relevant all-electron FlapwMBPT source is `ComDMFTv.2.0/src/FlapwMBPT`, not the local TRIQS directories named `dft_tools`.

SPEX `src/iterate.f:841-1788` and FlapwMBPT `dft_loop.F:1-35` fix the iteration dependency order implemented by `ScfPhysics` and `run_scf`:

```text
input density
  -> electronic Hartree + periodic nuclei + XC potential
  -> four-component core solve at every site
  -> radial basis, LAPW matching, H and S
  -> regular full-BZ eigensolutions
  -> chemical potential and occupations
  -> valence density + P^2 + Q^2 core density
  -> total energy, physical residual, convergence, and mixing
```

The driver owns this order, occupation counting, convergence, and mixer history. A material kernel owns density initialization, electrostatics and XC, radial construction, scalar/SOC routing, eigensolutions, orbital density synthesis, and frozen-potential spectra. The DFT layer consumes compiled capabilities and never depends on a concrete `LapwBasis` façade.

## 2. One runtime, ordered tasks, and the library seam

`libmuffintin-runtime` provides the single `muffintin` binary and an ordinary Rust library boundary suitable for a later Python binding. The binary is not named after DFT because the same versioned task registry can acquire THC or other workflows without creating another executable.

One TOML document has one explicit execution-order array and one block for each named task. The array is authoritative; TOML map order is not execution order. A consumer names an earlier typed output such as `scf.state`, so a later bands or DOS task reuses that exact converged potential, basis, relativity route, and state rather than mutable ambient solver state.

```toml
format = "libmuffintin-input"
version = 1
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
plane-wave-cutoff = 4.0
l-max = 8

[[task.scf.basis.local-orbitals]]
site = "Si-1"
kappa = 1
energy = -0.15
kind = "lo"

[task.scf.occupations]
kind = "fermi-dirac"
temperature = 0.001

[task.scf.xc]
kind = "lda-pw92"

[task.scf.mixing]
kind = "pulay-anderson"
beta = 0.4
history = 6

[task.scf.relativity]
kind = "spex-second-variation"
band-window = [0, 24]

[task.scf.convergence]
energy-tolerance = 1.0e-8
density-tolerance = 1.0e-7
max-iterations = 80

[[task.scf.core-states]]
site = "Si-1"
principal-quantum-number = 1
kappa = -1
occupation = 2.0

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

Task child data may therefore use either arrays of typed records, such as local orbitals, core states, and band-path points, or named subblocks, such as k meshes, mixing, XC, and convergence. Unknown fields, task kinds, orphan blocks, duplicate IDs, forward sources, and incompatible outputs are errors. This is the only V1 syntax; no second shorthand parser is retained.

## 3. Regional density and the physical metric

A collinear density has explicit up and down components. Each component contains one angularly resolved `SphereField` on every muffin-tin mesh and one Hermitian Fourier field on an exact reciprocal layout. The interstitial density coefficients represent the periodic orbital extension; physical integrals apply the analytic interstitial step function. A regional potential instead stores the matrix-element-ready masked interstitial coefficients used by LAPW assembly. The distinction prevents a masked potential from being reused as an unmasked Poisson source.

For scalar LAPW eigenvectors, the muffin-tin coefficients are formed after the unique compiled site projection. Large and scalar-relativistic small radial products enter separately. The interstitial term uses the exact plane-wave difference $G_{\mathrm{right}}-G_{\mathrm{left}}$, the cell normalization $1/\Omega$, explicit k weights, and band occupations.

The collinear physical inner product is

```math
\langle a,b\rangle
=\sum_{\sigma,s,L,M}\int_0^{R_s}r^2 a^*_{sLM\sigma}(r)b_{sLM\sigma}(r)\,dr
+\Omega\sum_{\sigma,G,G'}a^*_{G\sigma}\,\theta_{G-G'}\,b_{G'\sigma}.
```

Every mixer and the reported density RMS uses this metric. Serialized coefficient order is never treated as a Euclidean physical norm.

The full-spinor density boundary additionally exposes charge and Cartesian magnetization as four `RegionalScalarField` objects. Muffin-tin pair products retain every physical $P$ and $Q$ component, while the interstitial uses the two Pauli components of the SRA spinor basis. Converting this result to a collinear density is allowed only when transverse magnetization is absent within the declared tolerance.

## 4. Four-component core density

Each requested bound-core channel is identified by principal quantum number, signed Dirac $\kappa$, and an occupation not exceeding $2|\kappa|$. The solver isolates a unique bracket below the outer continuum with the required large-component node count, performs the two-sided four-component Dirac solve on an extended physical radial potential, and reports explicit `NotFound` or `Ambiguous` failures rather than selecting a root heuristically.

The true muffin-tin density is always

```math
n_c(r)=\frac{f_c}{4\pi r^2}\left[P_c(r)^2+Q_c(r)^2\right].
```

The extended tail is not discarded. Inside the sphere, a SPEX-style smooth pseudocharge matches the boundary value and derivative; its spherical-Bessel transform supplies the interstitial Fourier contribution with the exact site phase. After finite-$G$ assembly, each spin's constant coefficient is solved from the regional step-function integral so the represented charge equals the requested occupation exactly at any reciprocal cutoff; finite-$G$ moments and the true muffin-tin $P^2+Q^2$ density are unchanged, and the adjustment is reported explicitly. Closed shells split equally between the two collinear channels, while an explicit spin partition preserves the total occupation without double counting.

The frozen-snapshot bootstrap copies the physical muffin-tin monopole and obtains the outer shape from the snapshot's periodic interstitial field. A compact cubic Hermite representation bridge matches value and slope at the muffin-tin boundary and decays with zero value and slope correction at the outer mesh endpoint, where the original periodic field is recovered; the join parameters and residuals are diagnostics rather than a hidden atomic tail. The same explicit bridge reconciles the independently discretized muffin-tin XC projection and pointwise outer XC in later iterations, while raw unmasked electrostatics remains unchanged. Core bracket scans are measured relative to the outer continuum, so a constant potential-energy shift moves the bracket and eigenvalue together. The resulting initial $P^2+Q^2$ density makes the first Hartree input neutral, and every subsequent iteration rebuilds the physical continuation before solving the core again.

## 5. Weinert electrostatics and nuclei

Weinert is a general Coulomb/Poisson component in `libmuffintin-coulomb`, not a DFT-owned algorithm. Its public inputs are geometry, muffin-tin multipoles, a Hermitian Fourier charge field, and an explicit constant-mode treatment; it has no dependency on SCF, occupations, XC, or `libmuffintin-dft`. The DFT adapter only converts a regional electronic density and periodic nuclei to that reusable boundary, evaluates DFT energy contractions, and masks the returned operator potential. It does not route a charge density through `CompiledAuxiliaryBasis` and does not create a second Coulomb convention.

Positive electron number density first produces the repulsive electronic Hartree potential. Its $G=0$ source is removed by an explicit uniform-background treatment and its potential gauge is $V_{G=0}=0$. Periodic point nuclei are then evaluated separately with

```math
V_{\mathrm{nuc},G}=-\frac{4\pi}{\Omega G^2}\sum_s Z_s e^{-iG\cdot R_s},\qquad G\ne0,
```

and the own-site muffin-tin monopole restores $-Z_s/r$. The combined adapter requires physical electron and nuclear source charges to cancel. Raw electrostatic Fourier coefficients remain available for physical continuation and energy contractions; only the operator-facing `RegionalPotential` receives the step-function convolution.

The energy adapter evaluates $C=\int n(V_H+V_{\mathrm{nuc}})$, $M=E_{en}+2E_{II}$, $E_H=\tfrac12\int nV_H$, $E_{en}=\int nV_{\mathrm{nuc}}$, and $E_{II}$ in one common gauge. Discrete muffin-tin and Fourier boundary values are matched using the actual Poisson mesh endpoint, not a second independently rounded multipole quadrature.

## 6. LDA/PW92 and PBE

The implemented local-density choice follows SPEX `xlda + cpw92`; the gradient choice follows `xpbe + cpbe` (`src/iterate.f:261-326`, `src/xcfunc.f:105-141,564-783`). The point kernel accepts both spin densities, gradients, and Hessians. PBE includes the density gradient, Laplacian, and gradient--Hessian contraction required by the divergence term in the functional derivative.

In the interstitial, analytic Fourier derivatives supply the density jet and a deterministic midpoint grid performs the nonlinear forward/inverse transform. Inside each sphere, a deterministic angular rule, quartic radial interpolation, and fourth-order Cartesian differences supply the same point-kernel contract. The result reports both $E_{xc}$ and $\int\sum_\sigma n_\sigma v_{xc,\sigma}$ for total-energy bookkeeping.

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

so the Gaussian and Fermi--Dirac slopes agree at the chemical potential.

The Fermi--Dirac entropy and variational correction are

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

The input `electron-count` is the total electron count. The driver subtracts the requested core occupations exactly once before solving valence occupations and rejects a configuration with no represented valence electron.

## 8. Scalar, second-variation SOC, and four-component routes

The scalar route constructs Schrödinger or Koelling--Harmon radial solutions, analytic first and second energy derivatives, distinct-energy local orbitals, and HDLOs. Scalar-relativistic overlap uses the physical $PP+QQ$ norm. Nonspherical sphere fields enter through Gaunt matrix elements, and all site blocks use the same compiled projection order as density synthesis.

The optional nonmagnetic SPEX-style SOC route first solves a scalar Koelling--Harmon problem, selects an explicit source-band window, projects the site $\xi(r)\,\mathbf L\cdot\mathbf S$ operator into that subspace, solves the ordinary Hermitian doubled-spin problem, and reconstructs global Pauli spinors. This route rejects magnetic, noncollinear, and full-spinor first-variation inputs rather than silently changing the approximation. A runtime window must begin at the lowest represented valence band because the concrete V1 kernel does not silently drop occupied scalar states below the window.

The full first-variation library route keeps physical four-component $P/Q$ orbitals in every muffin-tin sphere and a two-component SRA interstitial basis. Each scalar $l$ linearization energy is inherited by the two signed-$\kappa$ partners with that large-component $l$; explicit signed-$\kappa$ LO and HDLO requests override the local-orbital content, including the $p_{1/2}$ channel $\kappa=+1$. Magnetic and SOC sphere fields enter as $V_0 I+\mathbf B\cdot\boldsymbol\sigma$ with no scalar fallback.

Snapshot V1 currently carries scalar or collinear diagonal spin potentials. The reusable first-variation library supports transverse fields and returns Cartesian spin density, while the concrete V1 SCF adapter accepts only the collinear fixed-point subset until a versioned snapshot schema can represent transverse input and restart fields.

## 9. Mixing, convergence, and total energy

Linear mixing, type-2 Broyden, and Pulay--Anderson all use residual $r=n_{\mathrm{in}}-n_{\mathrm{out}}$ and the same regional physical inner product. Each nonlinear mixer owns bounded history and rejects a layout change instead of mixing unrelated coefficient vectors.

An iteration converges only after both a density RMS threshold and a total-energy-change threshold are available and satisfied. The retained state is the accepted fixed-point input density together with the potential it generated, the exact basis controls, the chemical potential, relativity route, total energy, and ordered diagnostics.

The finite-temperature total-energy expression is assembled once as

```math
E=E_{\mathrm{band}}+E_{\mathrm{core}}+\frac{M-C}{2}+E_{xc}-\int n v_{xc}+E_{\mathrm{occ}},
```

where $E_{\mathrm{occ}}$ is either $-TS$ or $C_\sigma$. Core eigenvalues, band energy, electrostatics, XC, and the occupation correction each enter exactly once.

## 10. Bands and DOS

A bands task consumes an earlier `ScfState` and solves its frozen potential on the requested ordered reciprocal path. A DOS task consumes the same typed state, solves a regular full-BZ mesh, and applies the linear tetrahedron method. The implementation uses periodic eight-corner cells, six tetrahedra per cell, the shortest body diagonal with deterministic tie averaging, and normalized integrated band capacity. Smearing is not used inside the tetrahedron integral; any requested broadening is downstream presentation metadata.

## 11. Acceptance status and evidence boundary

The implementation has focused gates for weighted occupations, Gaussian and logistic tails, scalar/4c radial identities, signed-$\kappa$ local orbitals and HDLOs, plane-wave and sphere density synthesis, core $P^2+Q^2$ and spill, Weinert electronic and nuclear potentials, electrostatic boundary matching, LDA/PBE functional derivatives, three mixers, SCF ordering and source reuse, scalar and SOC eigensolutions, and tetrahedron normalization. The unified CLI also executes a minimal one-site snapshot through the concrete kernel.

These gates establish an executable M-Kb implementation candidate; they do not substitute for the planned external acceptance fixtures. Si and SrVO3 scalar results, Pt or Au second-variation SOC, collinear bcc Fe, and a regular-mesh tetrahedron DOS still require frozen SPEX/FLEUR reference artifacts before the milestone can be called cross-code accepted. Meta-GGA, hybrids, forces, SCF symmetry reduction, and tetrahedron occupations are outside M-Kb.
