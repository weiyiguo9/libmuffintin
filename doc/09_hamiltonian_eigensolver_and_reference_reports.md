# 09. LAPW Hamiltonian, filtered eigensolver, and reference reports

This note fixes the M-E implementation against SPEX
`src/hamilton.f:396-488,938-980` and `src/wrapper.f:2433-2474`.  It covers the
Hamiltonian and numerical comparison machinery.  It does not claim a Cu
cross-code result unless a complete external reference fixture is supplied.

## 1. One kinetic convention

M-E uses the SPEX symmetric-Laplacian convention consistently:

\[
 H^I_{ij}=\frac{|\mathbf q_i|^2+|\mathbf q_j|^2}{4}
 \Theta_I(\mathbf G_i-\mathbf G_j)
 +V^I(\mathbf G_i-\mathbf G_j).
\]

Here \(V^I\) is already the cell-normalized warped interstitial coefficient
used by the LAPW assembly, including whatever step-function convolution the
producer applies.  The alternative gradient form
\(\frac12\mathbf q_i\cdot\mathbf q_j\Theta_I\) differs by a surface term and
must not be mixed with the SPEX radial Hamiltonian below.

`InterstitialPotential` is keyed by integer reciprocal coordinates and
enforces \(V(-G)=V(G)^*\), with a real zero coefficient.

## 2. Muffin-tin Hamiltonian

For APW channel coefficients \(c^a_{i,lm,\alpha}\), flattened in `lm` order
and then \(\alpha=(u,\dot u)\), the site contribution is

\[
 H^{MT,a}_{ij}=(c^a_i)^\dagger h^a c^a_j.
\]

The supplied site block \(h^a\) is a full complex Hermitian matrix.  It may
therefore contain both the spherical radial Hamiltonian and the non-spherical
`v_LM` matrix elements produced by `mt-sphere`; the assembly code does not
special-case a harmonic channel.

For the spherical part, SPEX uses the radial-equation identity

\[
 h_{n_1n_2}^{(0)}=
 \frac{(E_{n_1}+E_{n_2})O_{n_1n_2}+D_{n_1n_2}}{2},
\]

where, in the ordered \((u,\dot u)\) basis,

\[
 D_{uu}=D_{\dot u\dot u}=0,\qquad
 D_{u\dot u}=D_{\dot u u}=O_{uu}.
\]

Thus

\[
 h^{(0)}=
 \begin{pmatrix}
 E O_{uu} & E O_{u\dot u}+\frac12O_{uu}\\
 E O_{u\dot u}+\frac12O_{uu} & E O_{\dot u\dot u}
 \end{pmatrix}.
\]

The physical spherical potential is already included through the radial
equation and must not be added again as an `(L,M)=(0,0)` field block.

## 3. Filtered generalized Hermitian eigensolver

The LAPW problem is

\[
 HC=SC\varepsilon.
\]

Near-linear dependence is handled before solving.  First decompose

\[
 S=U\,\mathrm{diag}(s)\,U^\dagger.
\]

An eigenvalue more negative than an independent floating-point noise bound is
an invalid overlap and produces an error.  Positive directions satisfying
\(s_i>\tau\max_j|s_j|\) are retained, and

\[
 X=U_{\rm keep}\,\mathrm{diag}(s_{\rm keep}^{-1/2}),
 \qquad X^\dagger SX=I.
\]

`faer` then diagonalizes the ordinary Hermitian matrix
\(\widetilde H=X^\dagger HX\).  The original-basis vectors are \(C=XZ\).
The result reports retained and filtered dimensions and, for every band,

\[
 r_n=\|Hc_n-\varepsilon_nSc_n\|_2
\]

with a scale-relative companion.  Forming `inverse(S) * H` is forbidden: it
loses Hermiticity and hides the overlap conditioning that must remain a public
diagnostic.

## 4. Reference comparison contract

Reference energies and calculated energies are both Hartree and are indexed
explicitly by `(k_index, band_index)`.  The default tolerance is

\[
 1\ {\rm meV}=\frac{10^{-3}}{27.211386245988}\ {\rm Ha}.
\]

The report retains every signed residual, the maximum absolute residual, and
the aggregate pass/fail flag.  An empty reference set is an error rather than a
vacuous pass.  This interface is suitable for SPEX or FLEUR-compatible LAPW
eigenvalue fixtures, but no material-specific values are embedded in the
crate.

The available SPEX source tree contains the production formulas but no
complete Cu `spex.dft`/potential fixture.  A separate all-electron FLEUR Cu
eigenvalue file can test the report reader once converted, but eigenvalues
alone cannot validate the reconstructed H without the corresponding frozen
potential and basis parameters.  Therefore the real `Cu <= 1 meV` gate remains
pending external fixture data; synthetic tests must not be presented as that
cross-code result.

## 5. M-E focused acceptance

The in-repository tests require:

- the full empty-lattice matrix
  \(H_{GG'}=\delta_{GG'}|k+G|^2/2\), including zero off-diagonal elements;
- Hermiticity of interstitial plus dense muffin-tin assembly;
- the SPEX spherical `2 x 2` radial identity above;
- filtering of a near-null overlap direction and rejection of a significant
  negative overlap eigenvalue;
- `C^H S C = I` and small `HC-SC epsilon` residuals;
- one-meV reference-report pass/fail and missing-data behavior.
