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
