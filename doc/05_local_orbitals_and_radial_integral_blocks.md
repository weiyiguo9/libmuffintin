# 05. Local-orbital construction and radial integral blocks

This note derives the \(2\times2\) construction used for LAPW local orbitals
and organizes the muffin-tin overlap and Hamiltonian into reusable radial
blocks.  The derivation is independent of the later plane-wave coefficients.

## 1. Primitive radial functions

For a fixed site, spin channel, and angular momentum \(l\), let

\[
\phi_1=u_l(r,E_l),\qquad
\phi_2=\dot u_l(r,E_l),\qquad
\phi_n=u_l(r,E_n^{\rm lo}),\quad n\geq3.
\]

Each symbol denotes all radial components returned by the chosen equation.  In
the scalar-relativistic case, for example,

\[
\phi_i\equiv(P_i,Q_i),\qquad
\langle\phi_i|\phi_j\rangle_r=
\int_0^R(P_iP_j+Q_iQ_j)dr.
\]

The homogeneous \(\phi_1\) and \(\phi_n\) are normalized.  The energy
derivative is projected to \(\langle\phi_1|\phi_2\rangle_r=0\), but its norm
\(\dot N_l=\langle\phi_2|\phi_2\rangle_r\) is generally not one.

Write the physical large-component boundary values as

\[
b_i=\begin{pmatrix}U_i\\U_i'\end{pmatrix},
\quad
U_i=\frac{P_i(R)}R,
\quad
U_i'=\frac{P_i'(R)}R-\frac{P_i(R)}{R^2}.
\]

## 2. The \(2\times2\) boundary construction

Define the LAPW boundary matrix

\[
B_l=\begin{pmatrix}U_1&U_2\\U_1'&U_2'\end{pmatrix}.
\]

For every additional homogeneous solution \(\phi_n\), choose coefficients
\((a_n,b_n)\) so

\[
\Phi_n^{\rm lo}=\phi_n+a_n\phi_1+b_n\phi_2
\]

has both value and slope zero at \(R\).  The two equations are

\[
B_l\begin{pmatrix}a_n\\b_n\end{pmatrix}=-
\begin{pmatrix}U_n\\U_n'\end{pmatrix},
\]

and hence

\[
\boxed{\begin{pmatrix}a_n\\b_n\end{pmatrix}=-B_l^{-1}b_n^{\rm boundary}}.
\]

Explicitly, with \(d=U_1U_2'-U_2U_1'\),

\[
a_n=\frac{-U_2'U_n+U_2U_n'}d,
\qquad
b_n=\frac{U_1'U_n-U_1U_n'}d.
\]

Substitution proves
\(\Phi_n^{\rm lo}(R)=(\Phi_n^{\rm lo})'(R)=0\).  Both large and small radial
components receive the same coefficients, even though only the large-component
value and slope define the scalar-relativistic LAPW match.  SPEX implements
this solve in `src/hamilton.f:314-392`.

The determinant and condition number of \(B_l\) are part of diagnostics.  A
near-singular matrix indicates a bad linearization energy or indistinguishable
boundary vectors; it must not be masked by changing the LO definition.

SPEX uses raw \(\dot u\) boundary data for this solve, then divides derivative
rows and columns by \(\sqrt{\dot N_l}\) in its assembled basis.  An
implementation may instead normalize \(\dot u\) first, but then it must rescale
the second column of \(B_l\), the matching coefficient, and every radial block
consistently.  Mixing the two representations changes the basis.

## 3. Transformation of radial blocks

Collect the primitive functions into a row vector
\(\boldsymbol\phi=(\phi_1,\phi_2,\phi_3,\ldots)\).  The transformed basis is
\(\boldsymbol\Phi=\boldsymbol\phi T\), where

\[
T_{:1}=e_1,\qquad T_{:2}=e_2,
\qquad T_{:n}=a_ne_1+b_ne_2+e_n\quad(n\geq3).
\]

For any sesquilinear radial operator block \(X^{\rm raw}\),

\[
\boxed{X=T^{\dagger}X^{\rm raw}T}.
\]

This single congruence generates APW--APW (indices 1--2), APW--LO (one index
1--2), and LO--LO (both indices \(\geq3\)) blocks.  It is less error-prone than
special-casing each expanded formula, and preserves Hermiticity by
construction.  SPEX applies the same row/column update and then routes the
three subblocks to `hmt1`, `hmt2`, and `hmt3` in
`src/hamilton.f:396-458,507-612`.

## 4. Overlap block

The primitive overlap is

\[
S_{ij}^{\rm raw}=\int_0^R
[P_i^{*}(r)P_j(r)+Q_i^{*}(r)Q_j(r)]dr.
\]

For a Schrödinger solution omit \(Q\).  The transformed metric is
\(S=T^\dagger S^{\rm raw}T\).  In the standard projected gauge,

\[
S_{11}=1,\qquad S_{12}=0,\qquad S_{22}=\dot N_l,
\]

before any optional normalization of \(\dot u\).  SPEX calculates the raw and
transformed overlaps in `src/hamilton.f:314-331,633-647`.

## 5. Spherical radial Hamiltonian block

Let the primitive index carry an energy \(E_i\).  For homogeneous functions,
\(H\phi_i=E_i\phi_i\).  For the energy derivative,

\[
H\phi_2=E_l\phi_2+\phi_1.
\]

A manifestly Hermitian primitive matrix is therefore

\[
\boxed{H_{ij}^{(0)}=
\frac{E_i+E_j}{2}S_{ij}
+\frac12\left(\delta_{i2}S_{1j}+\delta_{j2}S_{i1}\right)}.
\]

For \(i=j=2\), the final term is
\(\operatorname{Re}S_{12}=0\) in the projected gauge.  This identity avoids a
second numerical application of the radial differential operator and is the
formula in SPEX `src/hamilton.f:396-458`.  The transformed spherical block is
again \(H^{(0)}=T^\dagger H^{(0),\rm raw}T\).

The term called “kinetic plus spherical potential” here includes the physical
\(V_{00}^{\rm phys}(r)\) already used to generate the radial solutions.  A
SPEX adapter first converts
\(V_{00}^{\rm phys}=\mathtt{vmt}(:,1)/\sqrt{4\pi}\); the spherical coefficient
must not subsequently be added a second time.

## 6. Nonspherical potential blocks

Expand the additional physical potential as

\[
V_{\rm ns}(\mathbf r)=\sum_{LM\ne00}V_{LM}(r)Y_{LM}(\hat r).
\]

For radial indices \(i,j\) and angular channels \((l,m),(l',m')\), define

\[
I_{ij}^{LM}(l,l')=\int_0^R V_{LM}(r)
[P_{il}^{*}P_{jl'}+Q_{il}^{*}Q_{jl'}]dr.
\]

The matrix element factorizes as

\[
\boxed{\langle ilm|V_{LM}|jl'm'\rangle
=I_{ij}^{LM}(l,l')\,
\mathcal G^{LM}_{lm,l'm'}}.
\]

Triangle, parity, and magnetic selection rules should be applied before radial
integration.  Each radial matrix is transformed with the appropriate left and
right LO matrices,

\[
I^{LM}_{\rm new}=T_l^\dagger I^{LM}_{\rm raw}T_{l'}.
\]

This is SPEX's factorization in `src/hamilton.f:461-612`.  Its `vmt` angular
channels are normalized-harmonic expansion coefficients; snapshot import must
retain a declared angular basis and transform coefficients and Gaunt tensors
together.

## 7. Interstitial kinetic convention and sphere boundary terms

The radial blocks cannot be separated from the chosen interstitial kinetic
form.  The v0.1 plan specifies

\[
T^{\nabla}_{GG'}=\frac12\mathbf K_G\cdot\mathbf K_{G'}
\Theta_{G-G'},
\]

whereas SPEX assembles

\[
T^{\Delta}_{GG'}=\frac14(K_G^2+K_{G'}^2)\Theta_{G-G'}.
\]

Their difference is

\[
T^{\Delta}_{GG'}-T^{\nabla}_{GG'}
=\frac14|\mathbf G-\mathbf G'|^2\Theta_{G-G'}.
\]

It is the interface term generated by integration by parts through the
discontinuous step function.  Consequently the two expressions are not
drop-in replacements: the muffin-tin kinetic blocks and their surface terms
must be derived with the same convention.  SPEX symmetrizes the sphere
Laplacian as well; its nonrelativistic diagnostic form explicitly contains the
boundary value/slope term in `src/iterate_subs.f:11-53`, while the interstitial
form is at `src/iterate_subs.f:99-110` and `src/hamilton.f:935-953`.

The assembly strategy must therefore serialize or otherwise fix one convention
and validate the complete empty-lattice Hamiltonian.  Agreement of only the
diagonal \(G=G'\) elements cannot distinguish the two.

## 8. Validation

1. Check every LO's value and slope at \(R\) to the matching tolerance.
2. Compare explicit LO-expanded matrix elements with \(T^\dagger XT\).
3. Check Hermiticity of overlap, spherical Hamiltonian, and every Gaunt-weighted
   nonspherical block.
4. Confirm the derivative normalization convention by rescaling \(\dot u\) and
   demonstrating invariant physical matrix elements after all coefficients and
   blocks are transformed consistently.
5. Test both kinetic conventions only as complete assemblies, including their
   corresponding surface terms; never compare an interstitial term in
   isolation and declare the convention equivalent.

## Source anchors

The SPEX anchors below were inspected in the local tree
`/Users/zerozaki07/Documents/dft_codes/spex06.00pre36`.

- SPEX APW boundary matrix and derivative norm: `spex06.00pre36/src/hamilton.f:77-125`.
- SPEX LO boundary construction: `spex06.00pre36/src/hamilton.f:314-392`.
- SPEX spherical Hamiltonian identity and LO transforms: `spex06.00pre36/src/hamilton.f:396-458`.
- SPEX nonspherical radial integrals and block routing: `spex06.00pre36/src/hamilton.f:461-612`.
- SPEX transformed overlap: `spex06.00pre36/src/hamilton.f:633-647`.
- SPEX sphere/interstitial symmetrized kinetic terms: `spex06.00pre36/src/iterate_subs.f:11-110` and `spex06.00pre36/src/hamilton.f:935-953`.
