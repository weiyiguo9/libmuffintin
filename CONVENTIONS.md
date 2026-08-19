# libmuffintin conventions

This file and `mt-core` are the single source of truth for conventions that
affect cross-code numerical comparisons. Internal floating-point values are
`f64`; no implicit unit or basis conversion is permitted.

## Units and radial-potential meaning

Internal units are Hartree atomic units:

- energy: Hartree (`Ha`), with `1 Ry = 0.5 Ha`;
- length: Bohr (`a0`);
- reciprocal length: `a0^-1`;
- kinetic operator on a smooth function: `T = -1/2 nabla^2`.

The Rust API uses the `Hartree`, `Bohr`, `InverseBohr`, and `VolumeBohr3`
newtypes. File formats and user interfaces must label conversions explicitly.

The spherical radial potential is the **actual scalar value** `V(r)` in
Hartree. It is not the coefficient `v_00(r)` in an expansion
`V(r) = v_00(r) Y_00`. For an input that stores the latter coefficient, use
`V(r) = v_00(r) / sqrt(4 pi)` at the input boundary.

## Spherical harmonics and indexing

Complex harmonics are orthonormal Condon--Shortley harmonics

```text
Y_lm(theta,phi) = sqrt[(2l+1)/(4pi) (l-m)!/(l+m)!]
                   P_l^m(cos theta) exp(i m phi),
```

where `P_l^m` includes the Condon--Shortley phase. Therefore
`Y_l,-m = (-1)^m conj(Y_lm)` and `Y_11` has a minus sign.

The zero-based index is

```text
lm = l(l+1) + m,
```

so each channel occupies `l^2 .. (l+1)^2-1` in increasing `m=-l..l` order.
This is SPEX's one-based order shifted down by one.

Real tesseral harmonics use the same signed `m` label:

```text
R_l0  = Y_l0
R_lm  = sqrt(2) (-1)^m Re Y_lm,              m > 0
R_l,-m = -sqrt(2) Im Y_lm,                    m > 0.
```

Thus the `l=1` order `m=-1,0,1` is proportional to `y,z,x`.

## Wigner 3j and Gaunt convention

`wigner_3j` uses the standard Racah/Condon--Shortley convention for integer
angular momenta. `gaunt` exactly matches `src/numerics.f` in SPEX:

```text
G(1,2,3) = integral conj(Y_l1m1) Y_l2m2 conj(Y_l3m3) dOmega.
```

In particular, the magnetic selection rule is `m3 = m2 - m1`. Do not replace
this with an unconjugated triple-product helper. `real_gaunt` is the ordinary
triple product of the real harmonics defined above.

## Exponential mesh and radial quadrature

The radial mesh is

```text
r_i = r_0 exp(i h),   i=0,...,N-1.
```

`ExponentialMesh` reproduces SPEX `src/numerics.f:intgr_init`: a seventh-order
closed Newton--Cotes (Weddle-like) block with weights
`[41,216,27,272,27,216,41] h r / 140` on six-interval blocks, preceded when
needed by SPEX's tabulated seven-point Lagrange end rule. The radial Jacobian
`dr = r dx` is already in the weights.

For an outward mesh, `intgr` also integrates `0..r_0` by inferring
`f(r)=c r^x` from the first two samples and adding `r_0 f(r_0)/(x+1)`. As in
SPEX, sign-changing/tiny initial data receives no correction and an inferred
`x <= -0.99` receives the finite fallback `r_0 f(r_0)/2`. Inward meshes (`h<0`)
never add an origin contribution. Mesh parameters and this quadrature identity
are serialized data, not adjustable implementation details.

## Composite-grid convention

Real-space grid positions are Cartesian Bohr and volume weights are Bohr
cubed. Atom grids use shell-major/angular-minor order; uniform grids use
lexicographic fractional `(i,j,k)` order with `k` fastest; composite grids use
stable atom-index order followed by the interstitial. This order is serialized
and must be preserved by tensor conversions and THC point indices.

For an exponential radial weight `w_i` (which already integrates `dr`) and an
angular weight `w_a` normalized to `sum_a w_a = 4 pi`, the three-dimensional
weight is `w_i r_i^2 w_a`. Interstitial midpoint points are rejected using the
periodic nearest-image distance in the full direct-lattice metric.

## Sphere-field convention

`SphereField` always names either the complex Condon--Shortley or real tesseral
harmonic basis. Its samples are coefficients of normalized harmonics. Thus a
constant physical scalar `v` is represented by the `(0,0)` coefficient
`sqrt(4 pi) v`. A sphere matrix element is the sum of the corresponding Gaunt
factor times `integral (p_left p_right + Q_left Q_right) V_LM dr`.

## Versioned-artifact convention

The physical-input `SnapshotV1` and the derived `GridArtifactV1` have separate
format discriminators and version numbers. A grid-layout change does not bump
the snapshot schema. Both formats explicitly label units and harmonic/Fourier
conventions. RSTSR conversion is an optional grid consumer boundary; it is not
canonical storage and is not serialized.

## Reciprocal lattice and cutoff

Direct and reciprocal primitive vectors obey

```text
a_i dot b_j = 2 pi delta_ij.
```

`G = sum_i n_i b_i`. A G-vector set includes every integer vector satisfying
the **Cartesian** norm test `|G| <= G_max`; it is not an integer cube or a
component-wise cutoff. Enumeration bounds use the reciprocal dual basis, so
skew-cell cancellations cannot omit vectors. Output order is deterministic:
increasing Cartesian norm, then lexicographic `(n1,n2,n3)`.

## Interstitial step function and its double cutoff

The interstitial indicator is one outside all nonoverlapping muffin-tin spheres
and zero inside them. Its cell-normalized Fourier coefficient follows SPEX
`src/overlap.f:stepfunction`:

```text
Theta_I(G) = delta_G0
 - (1/Omega) sum_a exp(-i G dot R_a)
   4 pi [sin(|G|R_a) - |G|R_a cos(|G|R_a)] / |G|^3.
```

At `G=0`, the sphere term is its analytic volume limit `4 pi R_a^3/3`.
The implementation evaluates the equivalent stable form
`(4 pi R^3/3) 3 j_1(|G|R)/(|G|R)` and uses a small-argument series.

For a plane-wave set selected by `|k+G|` or `|G| <= G_max`, overlap and
interstitial matrix elements need `Theta_I(G-G')`. The coefficient table must
therefore be complete for all actual pair differences. A `k`-independent safe
bound for an origin-centered `|G| <= G_max` set is **`2 G_max`**. Truncating the
step table at `G_max` is not allowed. Consumers should enumerate the basis
first, form its actual differences when possible, and otherwise use the safe
double cutoff.

## LAPW plane-wave and matching convention

The interstitial basis is `exp(+i (k+G) dot r) / sqrt(Omega)`. Its complex
Rayleigh coefficient is `4 pi i^l conj(Y_lm(qhat)) / sqrt(Omega)`, with
`q=k+G`; expansion about site `R_a` adds `exp(+i q dot R_a)`.

For the physical, unnormalized energy derivative `udot`, APW matching solves

```text
[ u(R)     udot(R)  ] [A]   [j_l(qR)       ]
[ u'(R)    udot'(R) ] [B] = [q j_l'(qR)].
```

SPEX normalizes its stored `udot` column and compensates `B` by `dotnorm`;
libmuffintin does neither, so both represent the same physical augmented wave.
The overlap is `Theta_I(G-G')` plus site-resolved `c^H O c` radial blocks.

## Interstitial kinetic convention: explicit strategy choice

There is no hidden default. `KineticOperatorConvention` must be selected by an
assembly strategy because discontinuous augmented functions leave a boundary
term:

```text
Gradient form (v0.1 plan):
  1/2 K dot K' Theta_I(K-K')

SPEX symmetric-Laplacian production form:
  1/4 (|K|^2 + |K'|^2) Theta_I(K-K')
```

Here `K=k+G` and `K'=k+G'`. The second minus the first is exactly

```text
1/4 |K-K'|^2 Theta_I(K-K').
```

The forms coincide on the diagonal but are not interchangeable off diagonal.
The SPEX-named enum variant records the production reference; the gradient
variant records the formula written in the v0.1 plan. Surface-discontinuity
handling remains private to the LAPW assembly strategy as required by A7.
The `mt-lapw` M-E assembler explicitly selects the SPEX symmetric-Laplacian
form and combines it only with the matching SPEX radial Hamiltonian identity.

## LAPW radial Hamiltonian and overlap filtering

In the raw `(u, udot)` basis at linearization energy `E`, the SPEX spherical
radial block is

```text
h00 = E O00
h01 = E O01 + O00/2
h11 = E O11.
```

The spherical potential is already included in the radial equation and is not
added again as a sampled `(L,M)=(0,0)` field. Non-spherical `v_LM` terms enter
the full site Hermitian block separately.

The generalized problem is reduced with the eigensystem of `S`, retaining
positive directions above a declared relative threshold. A significantly
negative overlap eigenvalue is an error, not a filter candidate. Returned
vectors are normalized by `C^H S C = I`, and each eigenpair carries an
`H C - S C epsilon` residual.

## APW+LO basis and collinear-spin convention

The global one-spin basis is ordered as all plane waves followed by local
orbitals grouped in stored site order. Within a site, LO order is increasing
`l`, then `m=-l..l`, then the radial LO number. The local operator block is
ordered as every `lm` channel's `(u, udot)` pair followed by those LOs.

Overlap and Hamiltonian use the same site projection `P^H block P`.
Interstitial terms occupy only the plane-wave corner. An APW coefficient
already contains `exp(+i q dot R_a)`; APW--LO terms inherit its conjugate from
`P^H` and must not receive a second site phase. LO--LO terms have no
interstitial contribution.

Collinear spin without SOC is two independent generalized eigenproblems that
share geometry and basis layout but have separate potentials, radial blocks,
`H`, `S`, and eigensolutions. It is not represented as a coupled `2N x 2N`
spinor matrix, and no cross-spin block exists.
