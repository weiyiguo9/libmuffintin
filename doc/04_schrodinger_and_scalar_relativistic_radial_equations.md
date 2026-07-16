# 04. Schrödinger and scalar-relativistic radial equations

This note defines the homogeneous radial solution, its energy derivative, and
the boundary data used by LAPW matching.  Energies and potentials are Hartree,
lengths are Bohr, and \(V(r)\) is the physical spherical potential described in
[01](01_units_and_global_conventions.md), not a normalized-harmonic coefficient.

## 1. Nonrelativistic radial equation

For

\[
\psi_{lm}(\mathbf r)=\frac{P_l(r,E)}rY_{lm}(\hat r),
\]

the radial Schrödinger equation is

\[
\boxed{\left[-\frac12\frac{d^2}{dr^2}
+\frac{l(l+1)}{2r^2}+V(r)-E\right]P_l(r,E)=0}.
\]

For a finite potential, the regular solution starts as
\(P_l\propto r^{l+1}\).  For a Coulomb singularity the leading power is still
the regular one, with the next series coefficient determined by the nuclear
charge.  Overall amplitude is arbitrary until normalization.

The radial inner product is

\[
\langle P_i|P_j\rangle_r=\int_0^R P_i^{*}(r)P_j(r)dr.
\]

The normalized homogeneous solution satisfies \(\langle P|P\rangle_r=1\).
Its physical value and slope at the muffin-tin radius are

\[
U(E)=u_l(R,E)=\frac{P(R,E)}R,
\qquad
U_r(E)=\left.\frac{du_l}{dr}\right|_R
=\frac{P'(R,E)}R-\frac{P(R,E)}{R^2}.
\]

The dimensionless logarithmic derivative is

\[
D(E)=R\frac{U_r(E)}{U(E)}
=R\frac{P'(R,E)}{P(R,E)}-1.
\]

The pair \((U,U_r)\), not \(D\) alone, is required for boundary matching.

## 2. Koelling--Harmon scalar-relativistic equation

The scalar-relativistic approximation keeps the relativistic radial mass but
averages away explicit spin--orbit coupling.  Define

\[
M(r,E)=1+\frac{E-V(r)}{2c^2},
\qquad
W(r,E)=\frac{l(l+1)}{2M(r,E)r^2}+V(r)-E.
\]

Using the large radial numerator \(P\) and an internally scaled auxiliary
component \(\widetilde Q=cQ\), the SPEX first-order system is

\[
\boxed{P'=\frac{P}{r}+2M\widetilde Q},
\qquad
\boxed{\widetilde Q'=-\frac{\widetilde Q}{r}+WP}.
\]

Eliminating \(\widetilde Q=(P'-P/r)/(2M)\) yields

\[
-\frac12\left(\frac{d}{dr}+\frac1r\right)
\frac1M\left(\frac{d}{dr}-\frac1r\right)P
+\frac{l(l+1)}{2Mr^2}P+(V-E)P=0.
\]

As \(c\to\infty\), \(M\to1\) and this reduces to the Schrödinger equation.
The returned small component is the physical \(Q=\widetilde Q/c\), and the
normalization is

\[
\int_0^R\left(P^2+Q^2\right)dr=1.
\]

This order of operations matters: normalize with \(Q\), not with the scaled
Runge--Kutta variable \(\widetilde Q\).  SPEX performs the division and norm at
`src/dirac.f:194-204`.

The scalar-relativistic angular function remains \(Y_{lm}\) times an
independent Pauli spin label.  Although two radial arrays \((P,Q)\) are
returned, this is a functional two-component scalar-relativistic valence
description, not the four-component \(\kappa\)-resolved Dirac spinor of [06].

## 3. Energy derivative

Differentiate a parameter-independent Hamiltonian equation
\((H-E)|u(E)\rangle=0\):

\[
\boxed{(H-E)|\dot u\rangle=|u\rangle},
\qquad \dot u=\frac{\partial u}{\partial E}.
\]

For the scalar-relativistic first-order system,

\[
M_E=\frac1{2c^2},
\qquad
W_E=-1-\frac{l(l+1)}{4M^2c^2r^2},
\]

and the derivative equations in the same scaled variable are

\[
\boxed{\dot P'=\frac{\dot P}{r}+2M\dot{\widetilde Q}
+2M_E\widetilde Q},
\]

\[
\boxed{\dot{\widetilde Q}'=-\frac{\dot{\widetilde Q}}{r}
+W\dot P+W_EP}.
\]

If the inhomogeneous solver accepts the already returned physical component
\(Q\), its first source term is
\(2M_E\widetilde Q=Q/c\).  SPEX uses exactly this representation in
`src/dirac.f:311-531`.

Differentiating a normalized family gives

\[
2\operatorname{Re}\langle u|\dot u\rangle=0,
\]

but a direct inhomogeneous integration retains a homogeneous gauge freedom
\(|\dot u\rangle\mapsto|\dot u\rangle+a|u\rangle\).  For real radial
solutions, impose the LAPW gauge by projection,

\[
|\dot u_\perp\rangle=|\dot u_{\rm raw}\rangle
-|u\rangle\langle u|\dot u_{\rm raw}\rangle,
\qquad
\langle u|\dot u_\perp\rangle=0.
\]

Both large and small components, and both boundary value and boundary slope,
must receive the same subtraction.  SPEX does this in
`src/iterate.f:2489-2525`.  The projected derivative is not separately
normalized; its squared norm

\[
\dot N=\langle\dot u_\perp|\dot u_\perp\rangle
\]

is a physical basis metric used in LAPW overlap blocks.

## 4. Boundary derivatives

For every homogeneous or energy-derivative solution, define

\[
U=\frac{P(R)}R,\qquad
U_r=\frac{P'(R)}R-\frac{P(R)}{R^2}.
\]

In the scalar-relativistic system,

\[
P'(R)=\frac{P(R)}R+2M(R)\widetilde Q(R),
\]

so

\[
U_r=\frac{2M(R)\widetilde Q(R)}R.
\]

SPEX's returned `dp` is this \(U_r\), not \(P'(R)\); the conversion appears in
`src/dirac.f:194-204`.  The same distinction applies to the energy derivative:
`dot_boundary_value` is \(\dot P(R)/R\), whereas `dot_boundary_slope` is
\(\dot P'(R)/R-\dot P(R)/R^2\).

The LAPW boundary matrix is therefore

\[
B_l(E)=
\begin{pmatrix}
U&\dot U\\
U_r&\dot U_r
\end{pmatrix}.
\]

Its determinant measures whether \(u\) and \(\dot u\) can span arbitrary
value/slope data at \(R\).  Solvers should report a singular or poorly
conditioned \(B_l\), not continue with unstable coefficients.

## 5. Test identities

1. At large \(c\), compare scalar-relativistic and Schrödinger solutions at
   the same \(V,l,E\).
2. Compare \(\dot P\) and its boundary data with centered finite differences
   in energy, after applying the same normalization and orthogonal gauge.
3. Verify \(\langle u|u\rangle=1\),
   \(\langle u|\dot u\rangle=0\), and the stored \(\dot N\).
4. Recompute \(U_r\) both from a numerical derivative of \(P/r\) and from the
   first-order equation.
5. Confirm that a SPEX `vmt(:,1)` snapshot was divided by
   \(\sqrt{4\pi}\) before it is used as \(V(r)\).

## Source anchors

The SPEX and FLEUR anchors below were inspected in the local trees
`/Users/zerozaki07/Documents/dft_codes/spex06.00pre36` and
`/Users/zerozaki07/Documents/dft_codes/fleur`.

- SPEX homogeneous scalar-relativistic equations: `spex06.00pre36/src/dirac.f:44-204`.
- SPEX inhomogeneous energy-derivative equations: `spex06.00pre36/src/dirac.f:311-531`.
- SPEX normalized and orthogonalized radial basis: `spex06.00pre36/src/iterate.f:2489-2525`.
- SPEX boundary matrix: `spex06.00pre36/src/hamilton.f:105-123`.
- FLEUR scalar-relativistic reference: `fleur/global/radsra.f:1-140`.
