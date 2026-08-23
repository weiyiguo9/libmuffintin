# libmuffintin conventions

This file and `libmuffintin-core` are the single source of truth for conventions that
affect cross-code numerical comparisons. Internal floating-point values are
`f64`; no implicit unit or basis conversion is permitted.

## Units and radial-potential meaning

Internal units are Hartree atomic units:

- energy: Hartree ($\mathrm{Ha}$), with $1\,\mathrm{Ry} = 0.5\,\mathrm{Ha}$;
- length: Bohr ($a_0$);
- reciprocal length: $a_0^{-1}$;
- kinetic operator on a smooth function: $T = -\frac{1}{2}\nabla^2$.

The Rust API uses the `Hartree`, `Bohr`, `InverseBohr`, and `VolumeBohr3`
newtypes. File formats and user interfaces must label conversions explicitly.

The spherical radial potential is the **actual scalar value** $V(r)$ in
Hartree. It is not the coefficient $v_{00}(r)$ in an expansion
$V(r) = v_{00}(r)Y_{00}$. For an input that stores the latter coefficient, use
$V(r) = v_{00}(r)/\sqrt{4\pi}$ at the input boundary.

## Spherical harmonics and indexing

Complex harmonics are orthonormal Condon–Shortley harmonics

```math
Y_{lm}(\theta,\phi)
= \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}
  P_l^m(\cos\theta)\exp(im\phi),
```

where $P_l^m$ includes the Condon–Shortley phase. Therefore
$Y_{l,-m} = (-1)^m Y_{lm}^*$ and $Y_{11}$ has a minus sign.

The zero-based index is

```math
\mathrm{lm} = l(l+1) + m,
```

so each channel occupies $l^2,\ldots,(l+1)^2-1$ in increasing
$m=-l,\ldots,l$ order.
This is SPEX's one-based order shifted down by one.

Real tesseral harmonics use the same signed $m$ label:

```math
\begin{aligned}
R_{l0} &= Y_{l0}, \\
R_{lm} &= \sqrt{2}(-1)^m \mathrm{Re}\,Y_{lm}, && m>0, \\
R_{l,-m} &= -\sqrt{2}\mathrm{Im}\,Y_{lm}, && m>0.
\end{aligned}
```

Thus the $l=1$ order $m=-1,0,1$ is proportional to $y,z,x$.

## Wigner 3j and Gaunt convention

`wigner_3j` uses the standard Racah/Condon–Shortley convention for integer
angular momenta. `gaunt` exactly matches `src/numerics.f` in SPEX:

```math
G(1,2,3)
= \int Y_{l_1m_1}^*Y_{l_2m_2}Y_{l_3m_3}^*\,d\Omega.
```

In particular, the magnetic selection rule is $m_3 = m_2 - m_1$. Do not replace
this with an unconjugated triple-product helper. `real_gaunt` is the ordinary
triple product of the real harmonics defined above.

## Exponential mesh and radial quadrature

The radial mesh is

```math
r_i = r_0\exp(ih), \qquad i=0,\ldots,N-1.
```

`ExponentialMesh` reproduces SPEX `src/numerics.f:intgr_init`: a seventh-order
closed Newton–Cotes (Weddle-like) block with weights
$[41,216,27,272,27,216,41]hr/140$ on six-interval blocks, preceded when
needed by SPEX's tabulated seven-point Lagrange end rule. The radial Jacobian
$dr = r\,dx$ is already in the weights.

For an outward mesh, `intgr` also integrates $[0,r_0]$ by inferring
$f(r)=cr^x$ from the first two samples and adding $r_0f(r_0)/(x+1)$. As in
SPEX, sign-changing/tiny initial data receives no correction and an inferred
$x \leq -0.99$ receives the finite fallback $r_0f(r_0)/2$. Inward meshes
($h<0$)
never add an origin contribution. Mesh parameters and this quadrature identity
are serialized data, not adjustable implementation details.

## Composite-grid convention

Real-space grid positions are Cartesian Bohr and volume weights are Bohr
cubed. Atom grids use shell-major/angular-minor order; uniform grids use
lexicographic fractional $(i,j,k)$ order with $k$ fastest; composite grids use
stable atom-index order followed by the interstitial. This order is serialized
and must be preserved by tensor conversions and THC point indices.

For an exponential radial weight $w_i$ (which already integrates $dr$) and an
angular weight $w_a$ normalized to $\sum_a w_a = 4\pi$, the three-dimensional
weight is $w_i r_i^2 w_a$. Interstitial midpoint points are rejected using the
periodic nearest-image distance in the full direct-lattice metric.

## Sphere-field convention

`SphereField` always names either the complex Condon–Shortley or real tesseral
harmonic basis. Its samples are coefficients of normalized harmonics. Thus a
constant physical scalar $v$ is represented by the $(0,0)$ coefficient
$\sqrt{4\pi}v$. A sphere matrix element is the sum of the corresponding Gaunt
factor times

```math
\int (p_{\mathrm{left}}p_{\mathrm{right}}
+ Q_{\mathrm{left}}Q_{\mathrm{right}})V_{LM}\,dr.
```

## Versioned-artifact convention

The physical-input `SnapshotV1` and the derived `GridArtifactV1` have separate
format discriminators and version numbers. A grid-layout change does not bump
the snapshot schema. Both formats explicitly label units and harmonic/Fourier
conventions. RSTSR conversion is an optional grid consumer boundary; it is not
canonical storage and is not serialized.

## Reciprocal lattice and cutoff

Direct and reciprocal primitive vectors obey

```math
\mathbf a_i\cdot\mathbf b_j = 2\pi\delta_{ij}.
```

$\mathbf G = \sum_i n_i\mathbf b_i$. A set of $\mathbf G$ vectors includes every
integer vector satisfying the **Cartesian** norm test
$|\mathbf G| \leq G_{\max}$; it is not an integer cube or a
component-wise cutoff. Enumeration bounds use the reciprocal dual basis, so
skew-cell cancellations cannot omit vectors. Output order is deterministic:
increasing Cartesian norm, then lexicographic $(n_1,n_2,n_3)$.

## Interstitial step function and its double cutoff

The interstitial indicator is one outside all nonoverlapping muffin-tin spheres
and zero inside them. Its cell-normalized Fourier coefficient follows SPEX
`src/overlap.f:stepfunction`:

```math
\Theta_I(\mathbf G) = \delta_{\mathbf G,0}
- \frac{1}{\Omega}\sum_a \exp(-i\mathbf G\cdot\mathbf R_a)
  \frac{4\pi[\sin(|\mathbf G|R_a)
  - |\mathbf G|R_a\cos(|\mathbf G|R_a)]}{|\mathbf G|^3}.
```

At $\mathbf G=0$, the sphere term is its analytic volume limit $4\pi R_a^3/3$.
The implementation evaluates the equivalent stable form
$\frac{4\pi R^3}{3}\frac{3j_1(|\mathbf G|R)}{|\mathbf G|R}$ and uses a
small-argument series.

For a plane-wave set selected by $|\mathbf k+\mathbf G|$ or
$|\mathbf G| \leq G_{\max}$, overlap and interstitial matrix elements need
$\Theta_I(\mathbf G-\mathbf G')$. The coefficient table must therefore be
complete for all actual pair differences. A safe bound independent of $\mathbf k$
for an origin-centered $|\mathbf G| \leq G_{\max}$ set is **$2G_{\max}$**.
Truncating the step table at $G_{\max}$ is not allowed. Consumers should
enumerate the basis first, form its actual differences when possible, and
otherwise use the safe double cutoff.

## LAPW plane-wave and matching convention

The interstitial basis is
$\exp[i(\mathbf k+\mathbf G)\cdot\mathbf r]/\sqrt{\Omega}$. Its complex
Rayleigh coefficient is $4\pi i^lY_{lm}^*(\hat{\mathbf q})/\sqrt{\Omega}$,
with $\mathbf q=\mathbf k+\mathbf G$ and $q=|\mathbf q|$; expansion about site
$\mathbf R_a$ adds
$\exp(i\mathbf q\cdot\mathbf R_a)$.

For the physical, unnormalized energy derivative $\dot u$, APW matching solves

```math
\begin{bmatrix}
u(R) & \dot u(R) \\
u'(R) & \dot u'(R)
\end{bmatrix}
\begin{bmatrix}A\\B\end{bmatrix}
=
\begin{bmatrix}j_l(qR)\\qj_l'(qR)\end{bmatrix}.
```

SPEX normalizes its stored $\dot u$ column and compensates $B$ by `dotnorm`;
libmuffintin does neither, so both represent the same physical augmented wave.
The overlap is $\Theta_I(\mathbf G-\mathbf G')$ plus site-resolved
$c^\dagger O c$ radial blocks.

## Interstitial kinetic convention: explicit strategy choice

There is no hidden default. `KineticOperatorConvention` must be selected by an
assembly strategy because discontinuous augmented functions leave a boundary
term:

```math
\begin{aligned}
\text{Gradient form (v0.1 plan):}\qquad
&\frac{1}{2}\mathbf K\cdot\mathbf K'\,
 \Theta_I(\mathbf K-\mathbf K'), \\
\text{SPEX symmetric-Laplacian production form:}\qquad
&\frac{1}{4}(|\mathbf K|^2+|\mathbf K'|^2)
 \Theta_I(\mathbf K-\mathbf K').
\end{aligned}
```

Here $\mathbf K=\mathbf k+\mathbf G$ and
$\mathbf K'=\mathbf k+\mathbf G'$. The second minus the first is exactly

```math
\frac{1}{4}|\mathbf K-\mathbf K'|^2\Theta_I(\mathbf K-\mathbf K').
```

The forms coincide on the diagonal but are not interchangeable off diagonal.
The SPEX-named enum variant records the production reference; the gradient
variant records the formula written in the v0.1 plan. Surface-discontinuity
handling remains private to the LAPW assembly strategy as required by A7.
The `libmuffintin-lapw` M-E assembler explicitly selects the SPEX symmetric-Laplacian
form and combines it only with the matching SPEX radial Hamiltonian identity.

## LAPW radial Hamiltonian and overlap filtering

In the raw $(u,\dot u)$ basis at linearization energy $E$, the SPEX spherical
radial block is

```math
\begin{aligned}
h_{00} &= EO_{00}, \\
h_{01} &= EO_{01} + \frac{O_{00}}{2}, \\
h_{11} &= EO_{11}.
\end{aligned}
```

The spherical potential is already included in the radial equation and is not
added again as a sampled $(L,M)=(0,0)$ field. Non-spherical $v_{LM}$ terms enter
the full site Hermitian block separately.

The generalized problem is reduced with the eigensystem of $S$, retaining
positive directions above a declared relative threshold. A significantly
negative overlap eigenvalue is an error, not a filter candidate. Returned
vectors are normalized by $C^\dagger S C = I$, and each eigenpair carries an
$HC-SC\varepsilon$ residual.

## APW+LO basis and collinear-spin convention

The global one-spin basis is ordered as all plane waves followed by local
orbitals grouped in stored site order. Within a site, LO order is increasing
$l$, then $m=-l,\ldots,l$, then the radial LO number. The local operator block
is ordered as every $\mathrm{lm}$ channel's $(u,\dot u)$ pair followed by those
LOs.

Overlap and Hamiltonian use the same site projection $P^\dagger(\text{block})P$,
evaluated as `einsum("ci,cd,dj->ij", [P^*, B, P])` in `libmuffintin-tensor`. The
coefficient tensor $P$ has axes $[\text{site coordinates}][\text{site basis}]$.
Each site remains a separate tensor; sites are not padded into one rectangle.
The einsum layer may run on RSTSR+TBLIS or, later, tenferro-rs; the subscripts
are the contract. Interstitial terms occupy only the plane-wave corner. An APW
coefficient already contains $\exp(i\mathbf q\cdot\mathbf R_a)$; APW–LO terms
inherit its conjugate from $P^*$ and must not receive a second site phase.
LO–LO terms have no interstitial contribution. Host snapshots and `libmuffintin-io`
artifacts stay backend-neutral.

Collinear spin without SOC is two independent generalized eigenproblems that
share geometry and basis layout but have separate potentials, radial blocks,
$H$, $S$, and eigensolutions. It is not represented as a coupled $2N\times2N$
spinor matrix, and no cross-spin block exists.
