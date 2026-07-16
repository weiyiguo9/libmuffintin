# 02. Angular functions, reciprocal vectors, and the analytic step function

This note fixes the angular and reciprocal-space formulas used by
`libmuffintin`.  All phases and indices are part of the public convention.

## 1. Complex spherical harmonics and storage

Let \(P_l^m(x)\) denote the associated Legendre function without an embedded
Condon--Shortley phase.  The normalized complex harmonic is

\[
Y_{lm}(\theta,\phi)=(-1)^m
\sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}
P_l^m(\cos\theta)e^{im\phi},\qquad m\geq0,
\]

and negative orders are defined by

\[
Y_{l,-m}=(-1)^mY_{lm}^{*}.
\]

Hence \(Y_{00}=1/\sqrt{4\pi}\),
\(\int Y_{lm}^{*}Y_{l'm'}d\Omega=\delta_{ll'}\delta_{mm'}\), and the
Condon--Shortley phase is explicit.  If a special-function library instead
builds \((-1)^m\) into its associated Legendre function, the prefactor above
must be omitted exactly once.

Channels are ordered by increasing \(l\), and within an \(l\) block by
\(m=-l,-l+1,\ldots,l\).  The public zero-based compound index is

\[
L(l,m)=l(l+1)+m,
\]

so the block for angular momentum \(l\) occupies \(l^2,\ldots,(l+1)^2-1\).
SPEX's one-based array in `src/numerics.f:482-600` occupies the same block as
`Y(l**2+1:(l+1)**2)`.

## 2. Real spherical harmonics

The real basis used for real-valued muffin-tin fields is the unitary transform

\[
R_{l0}=Y_{l0},
\]

\[
R_{lm}=\frac{Y_{l,-m}+(-1)^mY_{lm}}{\sqrt2},\qquad m>0,
\]

and, writing \(p=-m>0\),

\[
R_{l,-p}=\frac{i}{\sqrt2}
\left[Y_{lp}-(-1)^pY_{l,-p}\right].
\]

These functions are real on the sphere and orthonormal.  This is the transform
implemented by SPEX `src/vector.f:241-280`.  A snapshot must name its angular
basis; changing from real to complex harmonics is a matrix transform, not an
index relabeling.

## 3. Wigner \(3j\) symbols and Gaunt coefficients

The Wigner \(3j\) symbol is evaluated with the Racah formula.  For integral or
half-integral arguments satisfying \(m_1+m_2+m_3=0\), define

\[
\Delta(j_1j_2j_3)=
\frac{(j_1+j_2-j_3)!(j_1-j_2+j_3)!(-j_1+j_2+j_3)!}
{(j_1+j_2+j_3+1)!}.
\]

Then

\[
\begin{split}
\begin{pmatrix}j_1&j_2&j_3\\m_1&m_2&m_3\end{pmatrix}
={}&(-1)^{j_1-j_2-m_3}
\sqrt{\Delta(j_1j_2j_3)}
\sqrt{\prod_{i=1}^{3}(j_i+m_i)!(j_i-m_i)!}\\
&\times\sum_z\frac{(-1)^z}{z!
(j_1+j_2-j_3-z)!(j_1-m_1-z)!(j_2+m_2-z)!}\\
&\hspace{36mm}\times
\frac{1}{(j_3-j_2+m_1+z)!(j_3-j_1-m_2+z)!},
\end{split}
\]

where the sum contains precisely the integer \(z\) values for which every
factorial argument is nonnegative.  Triangle, magnetic-sum, and
\(|m_i|\leq j_i\) failures return zero before the sum.  This is the convention
in SPEX `src/numerics.f:604-667`.

For the usual one-conjugate Gaunt coefficient,

\[
\mathcal G^{LM}_{lm,l'm'}=
\int Y_{lm}^{*}(\hat r)Y_{LM}(\hat r)Y_{l'm'}(\hat r)d\Omega,
\]

the closed form is

\[
\mathcal G^{LM}_{lm,l'm'}=(-1)^m
\sqrt{\frac{(2l+1)(2L+1)(2l'+1)}{4\pi}}
\begin{pmatrix}l&L&l'\\0&0&0\end{pmatrix}
\begin{pmatrix}l&L&l'\\-m&M&m'\end{pmatrix}.
\]

It vanishes unless \(|l-l'|\leq L\leq l+l'\), \(l+L+l'\) is even, and
\(-m+M+m'=0\).  SPEX's routine calls its arguments in the two-conjugate form

\[
g=\int Y_{l_1m_1}^{*}Y_{l_2m_2}Y_{l_3m_3}^{*}d\Omega,
\]

which equals

\[
(-1)^{m_1+m_3}
\sqrt{\frac{(2l_1+1)(2l_2+1)(2l_3+1)}{4\pi}}
\begin{pmatrix}l_1&l_2&l_3\\0&0&0\end{pmatrix}
\begin{pmatrix}l_1&l_2&l_3\\-m_1&m_2&-m_3\end{pmatrix}.
\]

The two APIs must not share a table without making this conjugation difference
explicit.

## 4. Spherical Bessel functions

The regular spherical Bessel function satisfies

\[
x^2j_l''+2xj_l'+[x^2-l(l+1)]j_l=0,
\qquad j_l(x)\sim\frac{x^l}{(2l+1)!!}\quad(x\to0).
\]

Starting values and recurrences are

\[
j_0(x)=\frac{\sin x}{x},\qquad
j_1(x)=\frac{\sin x-x\cos x}{x^2},
\]

\[
j_{l+1}(x)=\frac{2l+1}{x}j_l(x)-j_{l-1}(x),
\qquad
j_l'(x)=\frac{l}{x}j_l(x)-j_{l+1}(x).
\]

Upward recurrence is adequate when \(l\) is not large relative to \(|x|\);
otherwise use normalized downward recurrence, as SPEX does in
`src/numerics.f:750-908`.  At \(x=0\), use the series rather than a divided
formula.  The parity rule is \(j_l(-x)=(-1)^lj_l(x)\).

The angular expansion connecting plane waves and radial solutions is

\[
e^{i\mathbf q\cdot\mathbf r}=4\pi\sum_{lm}i^l
j_l(qr)Y_{lm}^{*}(\hat q)Y_{lm}(\hat r).
\]

It supplies both the \(i^l\) phase and the APW boundary values
\(j_l(qR)\), \(qj_l'(qR)\); see SPEX `src/hamilton.f:248-264`.

## 5. Reciprocal vectors and phase factors

With direct columns \(A=(\mathbf a_1\ \mathbf a_2\ \mathbf a_3)\),

\[
B=2\pi A^{-T},\qquad \mathbf G=B\mathbf g,\qquad
\mathbf K_{\mathbf G}=B(\mathbf k+\mathbf g).
\]

For a site at fractional coordinate \(\mathbf s_a\),
\(\boldsymbol\tau_a=A\mathbf s_a\) and

\[
e^{-i\mathbf G\cdot\boldsymbol\tau_a}
=e^{-i2\pi\mathbf g\cdot\mathbf s_a}.
\]

This identity is useful for computing phases without mixing Cartesian and
fractional coordinates.  The reciprocal construction is identical to SPEX
`src/getinput.f:3017-3028`.

## 6. Analytic muffin-tin/interstitial step Fourier coefficient

Let nonoverlapping spheres \(S_a=\{|\mathbf r-\boldsymbol\tau_a|<R_a\}\)
be removed from the unit cell, and define the interstitial characteristic
function

\[
\chi_I(\mathbf r)=1-\sum_a\mathbf1_{S_a}(\mathbf r).
\]

Its Fourier coefficient is

\[
\Theta_{\mathbf G}=\frac1\Omega\int_\Omega
\chi_I(\mathbf r)e^{-i\mathbf G\cdot\mathbf r}d^3r.
\]

For \(\mathbf G=0\), direct volume subtraction gives

\[
\Theta_{0}=1-\frac1\Omega\sum_a\frac{4\pi R_a^3}{3}.
\]

For \(G>0\), translate each sphere and align the polar axis with
\(\mathbf G\):

\[
\begin{split}
\int_{S_a}e^{-i\mathbf G\cdot\mathbf r}d^3r
&=e^{-i\mathbf G\cdot\boldsymbol\tau_a}
4\pi\int_0^{R_a}r^2j_0(Gr)dr\\
&=e^{-i\mathbf G\cdot\boldsymbol\tau_a}
\frac{4\pi[\sin(GR_a)-GR_a\cos(GR_a)]}{G^3}\\
&=e^{-i\mathbf G\cdot\boldsymbol\tau_a}
4\pi R_a^3\frac{j_1(GR_a)}{GR_a}.
\end{split}
\]

Therefore

\[
\boxed{\Theta_{\mathbf G}=-\frac{4\pi}{\Omega}
\sum_a R_a^3\frac{j_1(GR_a)}{GR_a}
e^{-i\mathbf G\cdot\boldsymbol\tau_a}},\qquad G>0.
\]

The \(G\to0\) limit of a sphere contribution is its volume, because
\(j_1(x)/x\to1/3\).  For real \(\chi_I\),
\(\Theta_{-\mathbf G}=\Theta_{\mathbf G}^{*}\).  SPEX implements these
relations and the negative Fourier phase in `src/overlap.f:141-202`.

The matrix element between normalized plane waves is

\[
\langle\mathbf K_{\mathbf G}|\chi_I|
\mathbf K_{\mathbf G'}\rangle=\Theta_{\mathbf G-\mathbf G'}.
\]

Consequently, the difference set required for a basis cutoff is larger than
the basis itself.  It must cover every \(\mathbf G'-\mathbf G\) that can occur;
SPEX's full-plane-wave convolution comments in
`src/wavefproducts.f:556-567,708` make this cutoff distinction explicit.

## 7. Numerical checks

1. Verify orthonormality and \(Y_{l,-m}=(-1)^mY_{lm}^{*}\) on an angular grid.
2. Compare tabulated Gaunt coefficients with direct angular quadrature and
   reject selection-rule violations exactly.
3. Compare the analytic sphere transform with numerical integration for
   \(GR\ll1\), \(GR\sim1\), and \(GR\gg1\).
4. Check \(\Theta_0\), Hermitian symmetry, and translation phases for a
   nonsymmetric multi-atom cell.
5. Check that \(qj_l'(qR)\), not merely \(j_l'(qR)\), is used as the radial
   derivative in APW matching.

## Source anchors

The SPEX anchors below were inspected in the local tree
`/Users/zerozaki07/Documents/dft_codes/spex06.00pre36`.

- SPEX harmonics and indexing: `spex06.00pre36/src/numerics.f:482-600`.
- SPEX real/complex transform: `spex06.00pre36/src/vector.f:241-280`.
- SPEX Wigner \(3j\) and Gaunt routines: `spex06.00pre36/src/numerics.f:604-667`.
- SPEX spherical Bessel functions: `spex06.00pre36/src/numerics.f:750-908`.
- SPEX plane-wave/APW matching values: `spex06.00pre36/src/hamilton.f:248-264`.
- SPEX reciprocal lattice: `spex06.00pre36/src/getinput.f:3017-3028`.
- SPEX analytic step function: `spex06.00pre36/src/overlap.f:12-50,141-202`.
