# 17. Minimal DFT and finite-temperature occupations

This note defines the M-Kb boundary. M-Kb consumes the frozen-potential one-particle and M-Ka radial/spinor contracts; it does not redefine them. The eventual target is a minimal full-potential LDA/PBE self-consistent cycle. The implemented occupation slice is deliberately smaller: regular-full-BZ Fermi--Dirac and Gaussian occupations, chemical-potential solution, and their distinct variational energy corrections. Energies and temperatures are in Hartree and lengths are Bohr.

## 1. Reference implementations and execution order

The SPEX source anchors below were inspected in `spex06.00pre36` at commit `b7778ba9f15ea30274fa1f6962a1d531c5679e5d`; line numbers are not claimed for another SPEX release.

SPEX `src/iterate.f:841-1788` fixes the full-potential execution order:

```text
rho_in
  -> Hartree and exchange-correlation potentials
  -> radial/core basis and H,S
  -> generalized eigensolution
  -> chemical potential and occupations
  -> core plus valence density
  -> residual, mixing, and rho_in(next)
```

Its LDA choice is `xlda` plus `cpw92`, and its PBE choice is `xpbe` plus `cpbe` (`src/iterate.f:261-326`, `src/xcfunc.f:105-141,564-783`). SPEX uses Gaussian broadening rather than true Fermi--Dirac occupations, so it is a layout and SCF-order reference, not an oracle for the finite-temperature formula below.

The independent FlapwMBPT reference supplies a true bounded logistic in `ComDMFTv.2.0/src/FlapwMBPT/fermi_dirac.F:1-14`, chemical-potential search in `search_mu_0.F:1-193`, and the order `radial -> potential -> bands -> chemical potential` in `dft_loop.F:1-35`. The local directories named `dft_tools` contain TRIQS DFTTools rather than FlapwMBPT; the relevant all-electron source is the `ComDMFTv.2.0` tree.

## 2. Weighted state contract

A state carries an eigenvalue $\epsilon_i$, regular-mesh weight $w_i$, and an explicit positive integer degeneracy $g_i$. Its electron capacity is $w_i g_i$. The degeneracy is not inferred from a spin label:

- a nonmagnetic scalar, no-SOC band normally has $g_i=2$;
- an explicitly enumerated collinear-spin or spinor band has $g_i=1$.

This matches the effective SPEX counting in `src/iterate_subs.f:1545-1549` while keeping the Rust representation independent of SPEX's array dimensions. The weights need not be equal, but they must be finite and positive. A caller that normalizes the k weights to one obtains the usual number of electrons per cell.

## 3. Fermi--Dirac chemical potential

At a finite electronic temperature $T>0$, the fractional occupation is

```math
f_i(\mu,T)=\frac{1}{1+\exp[(\epsilon_i-\mu)/T]}.
```

The chemical potential is the unique finite solution of

```math
N=\sum_i w_i g_i f_i(\mu,T)
```

for $0<N<\sum_i w_i g_i$. The implementation first brackets the monotone electron-count function outside the eigenvalue range and then bisects it. The logistic tails are evaluated by separate positive- and negative-argument forms so an irrelevant exponential cannot overflow. Exactly empty and exactly full finite-temperature spectra require $\mu=-\infty$ and $\mu=+\infty$, respectively, so the finite-`Hartree` API rejects those endpoint electron counts instead of returning occupations inconsistent with its reported chemical potential.

The electron tolerance applies to the weighted count, not separately to each occupation. A requested electron count outside the open represented-capacity interval is a hard error; it is not repaired by changing a $G=0$ density coefficient later.

## 4. Gaussian broadening

SPEX `src/gauss.f:17,57-62` uses a normalized Gaussian with standard deviation $\sigma$ and the cumulative occupation

```math
f_i^{G}(\mu,\sigma)=\frac12\,\mathrm{erfc}\left(\frac{\epsilon_i-\mu}{\sqrt{2}\,\sigma}\right).
```

The public Gaussian occupation and solver accept $\sigma$ directly. They evaluate the smaller complementary-error-function tail first, using `libm::erfc`, so the empty-state tail does not lose precision through `1+\mathrm{erf}` cancellation. The same weighted-state validation, finite-domain endpoint rule, dynamic bracket, and electron-count bisection are shared privately with the Fermi--Dirac path.

The current SPEX `KSUM ... GAUSS T` interface treats its input as a temperature-like parameter and converts it in `src/getinput.f:1889-1899,1978-1986` according to

```math
\sigma=\frac{4}{\sqrt{2\pi}}T.
```

This makes the Gaussian and Fermi--Dirac slopes equal at $\epsilon=\mu$. M-Kb exposes that conversion as a separately named adapter; it is not hidden inside the standard-deviation API.

For a variational functional consistent with the broadened occupations, M-Kb also reports the generalized Gaussian correction

```math
C_\sigma=-\sigma\sum_i w_i g_i\frac{\exp[-(\epsilon_i-\mu)^2/(2\sigma^2)]}{\sqrt{2\pi}}.
```

This correction follows by integrating the broadened one-state grand potential. It is not a physical entropy or Fermi--Dirac `-TS`, and it is an explicit libmuffintin addition: SPEX applies the Gaussian occupations to its band sum but does not include $C_\sigma$ in `src/iterate.f:874-911`.

## 5. Entropy and the variational free-energy term

The dimensionless independent-electron entropy is

```math
S=-\sum_i w_i g_i
\left[f_i\ln f_i+(1-f_i)\ln(1-f_i)\right].
```

Terms at $f_i=0$ or $f_i=1$ are defined by continuity as zero. The band-energy sum and finite-temperature correction are

```math
E_{\mathrm{band}}=\sum_i w_i g_i f_i\epsilon_i,
\qquad
-TS=-T S.
```

M-Kb exposes $-TS$ explicitly because the eventual SCF result must report the variational free energy rather than silently treating fractional occupations as a zero-temperature energy. SPEX's current total-energy expression in `src/iterate.f:874-911` has no such term and is therefore not copied verbatim.

## 6. Full-potential closure still required

The remaining M-Kb production cycle must preserve these source-backed contracts:

- muffin-tin density synthesis includes separate large-large and small-small radial products; a Dirac core contributes $P^2+Q^2$ every iteration;
- interstitial density uses Fourier coefficients of $\rho(\mathbf r)=\sum_G\rho_G e^{i\mathbf G\cdot\mathbf r}$;
- the Hartree path reuses the M-J Weinert conventions, including the neutral $G=0$ treatment, instead of creating a second pseudocharge convention;
- LDA is SPEX `xlda + cpw92`; PBE additionally needs the density gradient, Laplacian, and the gradient--Hessian contraction needed by the divergence of the GGA potential;
- density residuals combine muffin-tin radial integrals and step-function weighted interstitial Fourier coefficients. Linear, type-2 Broyden, and Pulay--Anderson mixers must share that physical metric rather than take an unweighted Euclidean dot product of serialized coefficients.

Until this regional density representation and metric exist, M-Kb does not expose placeholder mixer or SCF-driver abstractions.

## 7. Implemented acceptance slice

The finite-temperature slice is accepted when it demonstrates:

1. finite, typed energies and explicit positive state weights and degeneracy;
2. overflow-safe Fermi--Dirac and Gaussian tails and half occupation at $\epsilon_i=\mu$;
3. SPEX Gaussian standard-deviation values and the slope-matched $T\mapsto\sigma$ conversion;
4. weighted electron conservation for unequal k weights and degeneracies;
5. covariance under a uniform shift of every eigenvalue and the chemical potential;
6. nonnegative Fermi--Dirac entropy, a nonpositive `-TS` term, and the independently derived Gaussian smearing correction;
7. explicit errors for invalid temperatures, Gaussian widths, tolerances, iteration limits, state metadata, endpoint electron counts, and electron counts outside the represented capacity.

This slice does not claim a density update, LDA/PBE evaluation, Poisson solve, core loop, mixer, self-consistent material result, SOC second variation, noncollinear first variation, or tetrahedron DOS. Those remain M-Kb work, and the relativistic orbital/product bridge remains owned by M-L.
