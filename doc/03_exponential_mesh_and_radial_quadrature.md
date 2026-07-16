# 03. Exponential radial mesh and high-order quadrature

This note derives the radial mesh and integration scheme reproduced from SPEX.
The formulas apply to a scalar integrand \(f(r)\); products of radial functions
are formed first and passed through the same transformation.

## 1. Exponential mesh

Choose \(r_{\min}>0\), a logarithmic increment \(h>0\), and \(N\geq2\):

\[
r_i=r_{\min}e^{ih},\qquad i=0,\ldots,N-1.
\]

The muffin-tin radius is the final point,

\[
R=r_{N-1}=r_{\min}e^{(N-1)h},
\qquad h=\frac{\log(R/r_{\min})}{N-1}.
\]

Storing \((r_{\min},h,N,R)\) is deliberately redundant: a reader checks the
last equality and rejects a mesh whose metadata and samples disagree.  SPEX
constructs this mesh in `src/getinput.f:560-604,663-688`; its FLEUR reader
recovers the logarithmic increment in `src/readwrite_fleur.f:169-227`.

Set

\[
x=\log(r/r_{\min}),\qquad r=r_{\min}e^x,qquad dr=r\,dx.
\]

The physical radial integral becomes an equally spaced integral,

\[
I=\int_{r_{\min}}^R f(r)dr
=\int_0^{(N-1)h}F(x)dx,
\qquad F(x)=r(x)f(r(x)).
\]

Thus every quadrature weight derived below multiplies \(F_i=r_if_i\), not
\(f_i\) alone.

## 2. Origin correction

The logarithmic grid omits \(r=0\).  If the first points follow the regular
power law \(f(r)\simeq C r^p\), estimate

\[
p=\frac{\log|f_1/f_0|}{h}
\]

when the values have a consistent nonzero sign.  The missing interval is then

\[
I_{\rm origin}=\int_0^{r_{\min}}Cr^pdr
=\frac{r_{\min}f(r_{\min})}{p+1},\qquad p>-1.
\]

This term is optional only when the caller has analytically established that it
is negligible.  Sign changes, zeros, nonfinite \(p\), or \(p\leq-1\) must not
be hidden by the power-law extrapolation.  SPEX's corresponding correction is
at the start of `src/numerics.f:7-123`.

## 3. Seven-point closed Newton--Cotes panel

On a panel of six logarithmic steps, let \(t=(x-x_0)/h\) and interpolate the
seven samples \(F(x_0+jh)\), \(j=0,\ldots,6\), by

\[
p_6(t)=\sum_{j=0}^{6}F_jL_j(t),
\qquad
L_j(t)=\prod_{\substack{k=0\\k\ne j}}^6\frac{t-k}{j-k}.
\]

Integrating each cardinal polynomial over \([0,6]\) gives

\[
\int_0^6L_j(t)dt
=\frac1{140}(41,216,27,272,27,216,41)_j.
\]

Therefore a complete panel contributes

\[
\boxed{I_{0:6}=\frac{h}{140}
\left(41F_0+216F_1+27F_2+272F_3+27F_4+216F_5+41F_6\right)}.
\]

One way to verify the weights, without trusting a table, is to require

\[
\sum_{j=0}^{6}w_jj^n=\frac{6^{n+1}}{n+1},
\qquad n=0,\ldots,6.
\]

Symmetry supplies one additional cancellation, so the rule is exact through
degree seven.  The panel error is ninth order in its width and the composite
fixed-interval error is \(O(h^8)\) for a smooth transformed integrand \(F\).
Adjacent panels share their endpoint; it is included with the endpoint weight
from each neighboring panel.

## 4. Prefix remainder on an arbitrary mesh length

A mesh does not necessarily contain \(6q+1\) points.  Write

\[
q=\left\lfloor\frac{N-1}{6}\right\rfloor,
\qquad n_0=N-6q,
\]

so \(1\leq n_0\leq6\).  The suffix beginning at sample \(n_0-1\) consists of
\(q\) full six-step panels.  The prefix contains \(n_0-1\) intervals and is
integrated by the same degree-six interpolant through the first seven samples.

For subinterval \(s=1,\ldots,6\), define

\[
c_{js}=60480\int_{s-1}^{s}L_j(t)dt.
\]

The integer columns \((c_{0s},\ldots,c_{6s})^T\) are

\[
\begin{array}{c|rrrrrr}
j&s=1&s=2&s=3&s=4&s=5&s=6\\\hline
0&19087&-863&271&-191&271&-863\\
1&65112&25128&-2760&1608&-2088&6312\\
2&-46461&46989&30819&-6771&7299&-20211\\
3&37504&-16256&37504&37504&-16256&37504\\
4&-20211&7299&-6771&30819&46989&-46461\\
5&6312&-2088&1608&-2760&25128&65112\\
6&-863&271&-191&271&-863&19087
\end{array}.
\]

Thus the prefix contribution is

\[
I_{\rm prefix}=\frac{h}{60480}
\sum_{s=1}^{n_0-1}\sum_{j=0}^{6}c_{js}F_j.
\]

This construction explains the apparently opaque SPEX integer table: every
column is simply an exactly integrated Lagrange cardinal basis over one unit
subinterval.  It also permits a direct independent test by recomputing those
rational integrals.  SPEX stores this table and panel rule in
`src/numerics.f:7-123`; `src/numerics.f:127-205` precomputes the resulting
linear weights.

## 5. Final integration algorithm

For \(N\geq7\), the total integral is

\[
I=I_{\rm origin}+I_{\rm prefix}
+\sum_{p=0}^{q-1}I_{n_0-1+6p:n_0-1+6(p+1)}.
\]

Equivalently, precompute weights \(W_i\) such that

\[
I=I_{\rm origin}+\sum_{i=0}^{N-1}W_i\,r_if_i.
\]

Precomputation is preferred when many radial products share one mesh.  It
avoids rebuilding the same panel structure, and it makes the quadrature a
reproducible linear operation.  A short mesh needs an explicitly chosen lower
order fallback; it must not index seven samples that do not exist.

For a cumulative primitive \(J(r_i)=\int_0^{r_i}f(r)dr\), integrate complete
panels and then the required columns \(1,\ldots,s\) of the same prefix table.
SPEX implements the primitive in `src/numerics.f:250-345`.  Its logarithmic-grid
derivative routine is in `src/numerics.f:365-389`.

## 6. Radial products and dimensions

For scalar or Dirac radial numerators, representative products are

\[
S_{ij}=\int_0^R[P_i(r)P_j(r)+Q_i(r)Q_j(r)]dr,
\]

and

\[
V_{ij}^{LM}=\int_0^R V_{LM}(r)
[P_i(r)P_j(r)+Q_i(r)Q_j(r)]dr.
\]

They are evaluated by setting \(f\) to the bracketed expression, or to that
expression times \(V_{LM}\), then multiplying each sample by the logarithmic
Jacobian \(r_i\).  No extra \(r^2\) belongs here because \(P=ru\) already
contains the radial numerator.  If the arrays instead store \(u\), the caller
must form \(r^2u_iu_j\) before integration.

## 7. Validation

1. Check the mesh endpoint and monotonicity exactly from its metadata.
2. For \(F(x)=x^n\), verify one panel through \(n=7\) and observe eighth-order
   composite convergence on a smooth nonpolynomial function.
3. Recompute every integer table entry from \(60480\int L_j\) in a test.
4. Integrate \(f(r)=r^p\) on \([0,R]\) for several regular \(p\), including
   cases whose first interval matters.
5. Compare precomputed weights, direct panels, and cumulative primitives for
   random smooth data.

## Source anchors

The SPEX anchors below were inspected in the local tree
`/Users/zerozaki07/Documents/dft_codes/spex06.00pre36`.

- SPEX mesh construction: `spex06.00pre36/src/getinput.f:560-604,663-688`.
- SPEX mesh import: `spex06.00pre36/src/readwrite_fleur.f:169-227`.
- SPEX high-order integral and origin correction: `spex06.00pre36/src/numerics.f:7-123`.
- SPEX precomputed weights: `spex06.00pre36/src/numerics.f:127-205`.
- SPEX cumulative integral and derivatives: `spex06.00pre36/src/numerics.f:250-389`.
- SPEX mesh-length rule description: `spex06.00pre36/src/getinput.f:1013-1020`.
