# 15. Weinert/SPEX finite $q$ Coulomb metric

This note records the Weinert Coulomb public contract. Product kinematics remain
[13](13_product_space_and_lapw_mpb.md). The k-point ISDF/THC kernels remain
[14](14_kpoint_isdf_thc.md). There is no DFT/SCF driver, no GW/ERI
production consumer, no live SPEX $V^q$ dump, and no umbrella crate.

## 1. Scope

`libmuffintin-coulomb` assembles the Hermitian Coulomb operator

```math
V^q_{IJ}=\int\mathrm{d}^3r\,\mathrm{d}^3r'\,
M_I^{q*}(r)\,\frac{1}{\lvert r-r'\rvert}\,M_J^q(r')
```

over the common product IR `CompiledAuxiliaryBasis`. $M_I$ is either a
SPEX mixed-product function or an interpolation-point function after
projection onto the same physical charge expansion. Neither mixed-product
nor interpolation-point types from `muffintin_prodbasis::mpb` / `muffintin_prodbasis::thc`
are public inputs.

Analytic-source tests lock SPEX `coulombmatrix.f` formulas. The direct
Ewald kernel is an independent toy oracle, not production assembly.

## 2. Package

| Directory | Package |
|---|---|
| `crates/mt-coulomb` | `libmuffintin-coulomb` |

Production dependencies: `libmuffintin-prodbasis`, `libmuffintin-core`,
and `libmuffintin-envelope`.

```text
coulomb → prodbasis, core, envelope
```

The DAG stays acyclic. With mixed-product and THC producers merged
into `libmuffintin-prodbasis` as its `mpb::` and `thc::` modules, the
rule that Coulomb assembly consumes only the root product-basis IR
types is a documented convention rather than a Cargo dependency
boundary; `mpb::`/`thc::` types must not become public inputs here.
The toy k-point THC path still injects toy Grams; its production L2
default remains `allq_l2`.

## 3. Public types

`CoulombRequest` owns a validated direct `Cell`, the matching reciprocal
lattice from $a_i\cdot b_j=2\pi\delta_{ij}$, Weinert `LEXP` (default 4,
hard cap 12), and an optional `InterpolationProjection` (`pw_cutoff`,
$l_{\max}$). Volume equality is not enough: mixed-product $G$ labels are
checked against `request.reciprocal()` (index, Cartesian, norm, and
$q+G$). A same-volume skew cell is `WaveLatticeMismatch`. The assembled
operator stores the cell and reciprocal used.

`AuxiliaryLayout` is the identity compared by pair vertices and
$V^q$: exact `TransferQ`, the exact `AuxiliaryRegion` sequence, and the
muffin-tin/interstitial split. Recipe strings are not identity.
`PairVertex::from_auxiliary` is the checked constructor;
`apply` / `quadratic_form` reject a same-dimension permuted region list
as `VertexLayout`.

`CoulombOperator` stores that layout, the cell/reciprocal,
`AuxiliaryKind::{MixedProduct, InterpolationPoints, PointChargeOracle}`,
a full $n\times n$ row-major Hermitian matrix (SPEX stores packed
$I\le J$; this type fills the lower triangle by conjugation), and
optional `GammaHead` when $\lvert q\rvert=0$.

## 4. One charge-expansion boundary

Both representations are reduced to muffin-tin radial $\times Y_{LM}$
pieces plus interstitial $|q+G|$ plane waves, then assembled with the same
Weinert kernels.

### Mixed product

The compiled payload **is** the expansion: retained per-site modes on the
stored meshes (SPEX `basm`, already including one factor of $r$) in
$site\to L\to M=-L,\ldots,L\to n$ order, then auxiliary $|q+G|$ waves.
This is not a fake interpolation-point payload. Use `assemble_coulomb`.

### Sampled interpolation functions

Production interpolation assembly is
$V^q_{\mu\nu}=\langle\zeta_\mu^q\lvert v\rvert\zeta_\nu^q\rangle$, not a
point-charge Gram on the interpolation nodes. `SampledAuxiliaryFunctions`
is representation-neutral and lives in `libmuffintin-coulomb`:

- exact $q$ and `AuxiliaryLayout` (checked $n_\mu=$ `auxiliary.dimension`);
- parent-grid points, strictly-positive-capable quadrature weights, and
  `SampledPointSupport` labels;
- one explicit `ExponentialMesh` per muffin-tin site; each muffin-tin sample
  names its exact radial shell index and its coordinate is checked against
  that shell;
- row-major $\zeta$ samples $n_{\mathrm{grid}}\times n_\mu$.

Each $\zeta$ column is projected by quadrature onto the common expansion:
muffin-tin samples accumulate $w_p\zeta_\mu(r_p)Y_{LM}^*(\hat r_p)$ on the
declared site radial shell, without nearest-node snapping; interstitial/uniform samples accumulate
$w_p\zeta_\mu(r_p)e^{-i(q+G)\cdot r_p}/\sqrt{\Omega}$. This is not a
delta at the interpolation *node*. Use `assemble_sampled_coulomb`.
$q$, point/order, and dimension mismatches are rejected.

Runtime THC integration feeds `ThcResult.fits[iq].zeta` on the matching parent
grid. Spans differ from mixed product; tests do **not** claim elementwise
$V^{\mathrm{MPB}}=V^{\mathrm{THC}}$.

### Toy point-charge oracle

`assemble_point_charge_oracle` maps interpolation *nodes* to weighted
deltas $w_\mu\delta(r-r_\mu)$. That path is an explicit Ewald/Weinert
cross-check, not production $\zeta$. Identity $\zeta$ on the node grid
reproduces this oracle; that equality is plumbing, not a THC claim.

## 5. SPEX block structure

Source: local SPEX trees A/B, `src/coulombmatrix.f`, SHA prefix
`6ea02fd7`. Line numbers below are that file.

Radial integrals use SPEX `intgrf` (`numerics.f:196`), which is
`ExponentialMesh::integrate` (weights include the exponential Jacobian
$r$). Primitives follow `primitive` (`numerics.f:267-345`).

### Multipole moments (`coulombmatrix.f:242-253`)

```math
q_{nL}=\int_0^R r^{L+1} b_n(r)\,\mathrm{d}r,\qquad
q_n^{(2)}=\int_0^R r^{3} b_n(r)\,\mathrm{d}r\quad(L=0).
```

Spherical-Bessel moments of a plane wave (`coulombmatrix.f:278-288`):

```math
\int_0^R r^{L+2} j_L(qr)\,\mathrm{d}r
=
\begin{cases}
R^3/3,& q=0,\,L=0,\\
0,& q=0,\,L>0,\\
R^{L+2} j_{L+1}(qR)/q,& q\neq 0.
\end{cases}
```

### $g_{L_1M_1,L_2M_2}$ (`coulombmatrix.f:223-240`)

With $s_n=\sqrt{n!}$ (`getinput.f:1348`):

```math
g_{L_1M_1,L_2M_2}
=\frac{s_{L_1+L_2+M_2-M_1}\,s_{L_1+L_2+M_1-M_2}}
{s_{L_1+M_1}\,s_{L_1-M_1}\,s_{L_2+M_2}\,s_{L_2-M_2}
\sqrt{(2L_1+1)(2L_2+1)(2L_1+2L_2+1)}}
(4\pi)^{3/2}.
```

$g_{00,00}=(4\pi)^{3/2}$. Independent tests lock this closed form.

### Intra-sphere MT-MT (`coulombmatrix.f:327-366`)

Same site, same $L,M$:

```math
v_{n_1 n_2}^{L}
=\frac{4\pi}{2L+1}
\int_0^R\mathrm{d}r\,
b_{n_1}\Bigl(
r^{-L}\int_0^r b_{n_2} r'^{L+1}\,\mathrm{d}r'
+ r^{L+1}\int_r^R b_{n_2} r'^{-L}\,\mathrm{d}r'
\Bigr).
```

For the normalized $L=0$ constant $b(r)=r/\sqrt{R^3/3}$, this equals
$6/5\,Q^2/R$ with $Q=\sqrt{4\pi R^3/3}$ (twice the electrostatic
self-energy, because $V_{IJ}$ has no $1/2$).

### Inter-sphere MT-MT (`coulombmatrix.f:368-410`)

```math
V_{a L_1 M_1 n_1,\,b L_2 M_2 n_2}
=(-1)^{L_2+M_2} q_{n_1 L_1} q_{n_2 L_2}
g_{L_1M_1,L_2M_2}
e^{i q\cdot(R_b-R_a)}
S_{L_1+L_2,\,M_1-M_2}(a,b;q).
```

### Structure constants (`coulombmatrix.f:2287-2583`)

Andersen/SPEX Ewald sum

```math
S_{LM}(a,b;q)
=\sum_{T}
e^{i q\cdot T}
\frac{Y_{LM}^*(T+R_b-R_a)}{\lvert T+R_b-R_a\rvert^{L+1}},
```

omitting $T+R_b-R_a=0$. Real-space $g_L$ polynomials follow SPEX
`coulombmatrix.f:2430-2442` exactly: $L=4$ skips $a/9$ and ends $a/10$;
$L=5,6,7$ use `HLP9` through $a/10$, $a/12$, $a/13$; $L\ge 8$ is
$a^{-(L+1)}$ in real space only (reciprocal $g_L$ stops at $L=7$).
The high $L$ real-space cutoff uses SPEX `CONVPARAM2` (`CONVTYPE=2`):
$(1/\mathrm{CONVPARAM2})^{1/7}\times\mathrm{latcon}$, so advertised
`LEXP`$\le 12$ ($2L_{\mathrm{exp}}\le 24$) is supported. `LEXP`$>12$ is
rejected. The $L=0$ on-site constant $-5/16\sqrt{4\pi}$ is applied before
the final $\mathrm{scale}^{L+1}$. Cartesian $q$ is used ($e^{i q\cdot T}$,
not crystal $2\pi k\cdot n$).

Independent oracle for $L\ge 1$: brute-force real-space sum of
$Y_{LM}^*/R^{L+1}$ (including $L=8$ and $L=12$).

### MT-PW (`coulombmatrix.f:438-545`)

Plane waves are $\Theta_I(r) e^{i(q+G)\cdot r}/\sqrt{\Omega}$. The
finite $q$ element is (2a) Bessel overlap divided by $|q+G|^2$ minus (2b) the
Weinert intra-sphere kernel plus (2c) inter-sphere structure-constant
coupling of Bessel moments, all divided by $\sqrt{\Omega}$, with
$Y_{LM}^*(q+G)$ and $i^L$, matching SPEX `harmonicsr` then `conjg`.
The (2c) parity factor is $(-1)^{L_1+M_1}$ with SPEX `idum=1` inside
every $L_1$ loop (`~484`).

At $q=0$, $G=0$, the $1/q^2$ term is replaced by the $L=0$ second-moment
formula `-cdum * moment2 / 6 / sqrt(Ω)`. Linear/quadratic Gamma body
terms at `~497-511` are included in $c_{\mathrm{sum}}$ before spherical
average: $L=0$ uses $R^5/30$ at $G=0$ and
$(j_0 R^2-2j_2/3)/10$ at finite $G$; $L=1$ uses the
$i\sqrt{4\pi}\,j_1 Y_{1M}^*/3$ term.

### PW-PW (`coulombmatrix.f:667-964`)

(3a) full-space $4\pi/|q+G|^2$ minus muffin-tin integrals of
$e^{i(G_2-G_1)\cdot r}$. Empty spheres make this diagonal
$4\pi/\lvert q+G\rvert^2$ with vanishing off-diagonals. (3b) inter-sphere
Bessel-moment coupling through $S_{LM}$, with `idum=1` inside every $L_2$
loop (`~775`). (3c) same-sphere `sphbessel_integral` (`coulombmatrix.f:1331`).
At Gamma the $G=0$ $4\pi/q^2$ diagonal is omitted, and the finite/finite,
zero/finite, and zero/zero Taylor corrections of `~830-915` are added
before spherical-average subtraction.

### Hermitian completion

SPEX stores $I\le J$. After assembly, $V_{JI}=V_{IJ}^*$.

## 6. Gamma body and head (`coulombmatrix.f:497-511`, `830-915`, `1466-1539`)

At $q=0$ the operator diverges. The stored matrix is the **finite body**.
Block-local Taylor terms are applied first (MT-PW `~497-511`, PW-PW
`~830-915`), then SPEX `coulomb_sphaverage` subtracts

```math
\frac{4\pi}{3}
\Bigl(
\partial_m c_I^*\partial_m c_J
+\tfrac12\bigl(c_I^*\Delta c_J+(\Delta c_I)^* c_J\bigr)
\Bigr)
```

from the packed matrix. `GammaHead` records:

- `spherical_average_subtracted = true`;
- `head_prefactor = 4π`;
- `constant_coefficients` = the monopole Fourier vector $\omega_I$
  (`coeff` in SPEX: $L=0$ $\sqrt{4\pi}\int r b/\sqrt{\Omega}$, plus
  $\Theta_I(-G)$ for plane waves).

The divergent head $4\pi/|q|^2\,|\omega\rangle\langle\omega|$ is **not**
written into the matrix and is **not** silently zeroed. Tests require a
finite body, a nonzero $\omega$, Hermiticity, and reconstruction of the
rank-one head from $\omega$. Friedrich et al. arXiv:0811.2363 names the
$q\to 0$ spherical-average convention; the implementation follows the
SPEX source, not a re-derivation of that paper.

### 6.1 Spencer–Alavi spherical truncation

`CoulombRequest::with_spencer_alavi_sphere` is the explicit molecule/insulator
alternative corresponding to VASP `HFRCUT=-1`. For cell volume $\Omega$ and
the number $N_k$ of points in the full Brillouin zone, its automatic radius is

```math
R_c=\left(\frac{3N_k\Omega}{4\pi}\right)^{1/3}.
```

The reciprocal kernel is

```math
v_{R_c}(Q)=\frac{4\pi}{Q^2}\left(1-\cos(QR_c)\right),
\qquad
v_{R_c}(0)=2\pi R_c^2.
```

This is a complete alternative Coulomb operator, not a replacement for only
the Gamma-head element. Every auxiliary function is projected analytically to
its Fourier coefficients, including radial MT harmonics and the interstitial
step-function PW coefficients, and the matrix is assembled as

```math
V_{IJ}^{q,R_c}
=\sum_G \rho_I(q+G)^*v_{R_c}(|q+G|)\rho_J(q+G).
```

The request carries an explicit reciprocal cutoff for this sum. It is a
caller-visible convergence parameter analogous to the finite FFT grid used by
a PW implementation. A truncated operator has no separated `GammaHead`; it
records `SpencerAlaviSphere` metadata containing $R_c$, $N_k$, and the Fourier
cutoff. Empty-sphere PW and isolated $G=0$ mixed MT/interstitial outer-product
oracles lock the finite limit and the shared-kernel normalization.

This direct Fourier realization is not a production all-electron shortcut.
Unlike the periodic Weinert path, it does not retain the analytic MT
short-range Coulomb integrals outside the selected Fourier space. A cutoff
adequate for interstitial valence products can therefore badly underestimate
deep-core CC and CV exchange. The Kr diagnostic in
[23](23_core_valence_exchange.md) separates SCF convergence from this
representation error. Increasing only the orbital basis or improving DIIS
cannot repair a truncated interaction metric; the all-electron cutoff must
be converged independently or a separately specified dual-space kernel must
retain its short-range part. Replacing only the CC energy by an isolated
radial value would instead mix different operators and is not a correction
to the existing HF Hamiltonian.

### 6.2 Dual-space smoothed spherical boundary

`with_smoothed_spencer_alavi_sphere(N_k, G_max, omega)` selects a distinct
kernel with an explicit positive smoothing parameter $\omega$ in inverse
bohr. It does not change the sharp-sphere path. Following
[Yang et al., Eq. 9](https://arxiv.org/html/2609.00203v1),

```math
v_{\mathrm{sTC}}(Q)
=\frac{4\pi}{Q^2}\left[1-\cos(QR_c)e^{-Q^2/(4\omega^2)}\right],
\qquad
v_{\mathrm{sTC}}(0)=2\pi R_c^2+\frac{\pi}{\omega^2}.
```

The short-distance Coulomb singularity is unchanged. Only the truncation
boundary is smoothed over a length of order $1/\omega$; the dimensionless
parameter $\eta=\omega R_c$ controls its sharpness. Increasing $\eta$ at
fixed $R_c$ approaches the sharp sphere, but requires a larger reciprocal
cutoff. Convergence in $\eta$, the Fourier cutoff, and the physical box size
are separate questions.

The implementation first assembles the periodic Weinert finite body,
including its analytic MT integrals, then adds

```math
\Delta V_{IJ}
=\sum_{|q+G|\le G_{\max}}
\rho_I(q+G)^*\Delta v(q+G)\rho_J(q+G),
```

where $\Delta v(Q)=-4\pi\cos(QR_c)e^{-Q^2/(4\omega^2)}/Q^2$ for nonzero
$Q$. At $Q=0$, $\Delta v$ is the complete finite $v_{\mathrm{sTC}}(0)$,
because the periodic finite body omits that Fourier component. There is no
remaining `GammaHead`. The correction is one weighted `gi,gj->ij` tensor
contraction on the shared TBLIS backend. The Fourier cutoff limits only
this exponentially damped correction, not the compact MT short-range
interaction. All VV, CV, VC, and CC sectors consume the resulting single
operator; no radial energy-only replacement is made.

The mixed-product Fourier construction shares each radial transform across
magnetic components and directions with exactly equal $|q+G|$. Shell keys
use the unmodified floating-point norm bits, not a rounded geometric
tolerance. Angular harmonics and site phases are then applied separately.
This avoids repeating the same radial Bessel integration for every matrix
column and reciprocal direction; sampled interpolation functions retain
their general charge-expansion transform.

`SpencerAlaviSphere::smoothing` distinguishes the sharp boundary (`None`)
from the smoothed boundary (`Some(omega)`). The Kr example selects the new
kernel with `--exchange-coulomb smoothed-spencer-alavi-sphere` and requires
`--fock-smoothing-omega` explicitly. Manifest version 5 records both $\omega$
and $\eta$, together with the radius and the Fourier cutoff. This option is
not claimed to be the exact VASP `HFRCUT=-1` kernel at finite $\omega$.

An independent compact-charge diagnostic used the normalized hydrogenic
1s density with $Z=20$ in an 8 bohr cell and an MT radius of 0.8 bohr. Its
isolated Coulomb self integral is $5Z/8=12.5$ Ha. The direct sharp-sphere
Fourier sum at cutoff 4.5 gave 2.776936 Ha. The dual-space result at
$\omega=1.2$ inverse bohr was 12.500000112 Ha at cutoff 10 and
12.500000110 Ha at cutoff 12. The separate onsite radial integral differed
from 12.5 Ha by less than $2\times10^{-11}$ Ha. The smoothed periodic
kernel still has image tails, so comparison with the isolated integral is
not a same-kernel equality gate. Empty-sphere PW checks at Gamma and finite
$q$ agreed with the full reciprocal formula within $3\times10^{-14}$.

## 7. Pair vertices

`PairVertex` stores `AuxiliaryLayout`, not a raw count split.
`PairVertex::from_auxiliary` copies layout and provenance from the
compiled auxiliary. Application is $Vc$ and $c_L^\dagger V c_R$. A
different `TransferQ`, a different MT/I count, or a same-count permuted
`AuxiliaryRegion` sequence is an error (`VertexTransferQ`,
`VertexDimension`, `VertexLayout`). Provenance/recipe strings are not
compared.

## 8. Independent oracles and recorded tolerances

| Quantity | Oracle | Recorded gate |
|---|---|---|
| $g_{00,00}$ | $(4\pi)^{3/2}$ | $10^{-12}$ |
| $g_{10,00}$ | $(4\pi)^{3/2}/3$ | $10^{-12}$ |
| $j_L$ moment $q=0$ | $R^3/3$ ($L=0$), else $0$ | $10^{-15}$ |
| $j_L$ moment finite $q$ | trapezoid $\int r^{L+2} j_L$ | $5\times 10^{-4}$ |
| Radial primitive $r^2$ | $r^{3}/3$ | $2\times 10^{-3}$ relative |
| $L=0$ constant intra-sphere | $6/5\,Q^2/R$ | $5\%$ |
| SPEX `real_g` $L=4..7$ at $a=2$ | source polynomials / HLP9 | $10^{-14}$ |
| $S_{10}$ on-site, finite $q$ | brute-force $Y_{10}^*/R^2$ | $5\times 10^{-3}$ |
| $S_{L=8}$, $S_{L=12}$ | brute-force $Y_{LM}^*/R^{L+1}$ | $5\times 10^{-2}$ relative |
| Empty-sphere PW-PW | diagonal $4\pi/\lvert q+G\rvert^2$, off-diagonal $0$ | $10^{-8}$ relative |
| MT-PW $2a+2b+2c$ | independent SPEX reconstruction | $10^{-8}$ relative |
| One-sphere PW-PW | independent (3a)+(3b)+(3c) | $10^{-8}$ relative |
| Gamma empty-sphere finite $G$ | $4\pi/\lvert G\rvert^2$; $G=0$ body omitted | $10^{-8}$ relative |
| Gamma PW-PW Taylor | independent `~830-915` on $G=0$ / finite pairs | $10^{-8}$ relative |
| Hermiticity | $\max\lvert V-V^\dagger\rvert$ | $10^{-10}$ mixed product / Gamma PW-PW, $10^{-8}$ interpolation |
| Finite $q$ min eigenvalue | faer self-adjoint EVD | $>-10^{-6}$ mixed product, $>-10^{-5}$ sampled $\zeta$, $>-10^{-4}$ THC $\zeta$ |
| Ewald successive residual | cutoff scan, not Abramowitz–Stegun `erfc` | $10^{-6}$; `erfc` $\sim 1.5\times 10^{-7}$ |
| Two-site monopoles vs Ewald | see below | $10^{-6}$ |

Two muffin-tin unit charges at $(2,2,2)$ and $(6,2,2)$ in cubic $a=8$,
$R=0.8$, $q=2\pi\hat y/a$. Direct Ewald $\eta=\pi/a^2$ with independently
increased real/reciprocal cutoffs (successive change $<10^{-6}$). A
second $\eta$ is compared to the same limit. Exhausted scans return
`EwaldNotConverged` rather than the last value. Weinert (1b) versus that
kernel: relative error $1.7\times 10^{-7}$. Gate $10^{-6}$. Absolute
`erfc` accuracy is not claimed at $10^{-8}$.

This is analytic-source plus toy-Ewald validation. It is not a SPEX
numeric dump comparison (no $V^q$ dump exists in the three local trees)
and not a real-material accuracy claim.

## 9. Both representations

Mixed-product auxiliaries use `assemble_coulomb`. Interpolation-point
auxiliaries use `assemble_sampled_coulomb` with parent-grid $\zeta$.
Dev-only THC integration uses `run_thc` (default `allq_l2`) and
`ThcResult.fits[iq].zeta`. Spans differ. Tests compare kinds, layouts,
PSD/actions, and exercise the shared expansion plumbing with identity $\zeta$
on the node grid versus the explicit point-charge route. That equality is
not an independent physics oracle. The tests do **not** claim
elementwise $V^{\mathrm{MPB}}=V^{\mathrm{THC}}$.

## 10. Limitations

- Convention and toy fixtures only. No live SPEX untruncated product dump
  and no SPEX $V^q$ dump.
- No Coulomb assembler consumption in GW/ERI, no SCF/Hartree driver (DFT),
  no MPI/CTF, no HDF5 export.
- Toy k-point THC Coulomb ranking still uses injected Grams. Production $V^q$ for
  interpolation points requires sampled $\zeta$, not the node list.
- `LEXP` is capped at 12 in this crate. Default toy `LEXP` is 4.
  `LEXP`$>12$ is rejected.
- Workspace `rust-version` remains 1.89. Optional tenferro remains 1.96.
