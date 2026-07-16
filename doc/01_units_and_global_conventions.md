# 01. Hartree/Bohr units and global conventions

This note is normative for `libmuffintin`. It fixes the quantities that must not be inferred from a file name, a calling code, or a numerical value.

## 1. Atomic-unit system

All values inside the library use Hartree atomic units,

\[
\hbar=m_e=e=4\pi\epsilon_0=1.
\]

Consequently,

\[
[r]=a_0,\qquad [G]=a_0^{-1},\qquad [E]=E_h,
\qquad
\hat H=-\frac12\nabla^2+V,
\]

and a free plane wave has kinetic energy

\[
T(\mathbf k+\mathbf G)=\frac12|\mathbf k+\mathbf G|^2.
\]

**The internal energy unit is Hartree, never Rydberg.** In particular, an unlabelled scalar in a radial solver, an energy parameter, a potential channel, a Hamiltonian matrix, or an eigenvalue is in Hartree. One Rydberg is one half Hartree. The factor of two must appear at an I/O boundary, not inside the radial or assembly algebra.

The internal length unit is Bohr. Angstrom and electron-volt are accepted only by explicit conversion routines or a self-describing input field. Useful relations are

\[
1\ a_0\simeq0.5291772109\ \text{angstrom},\qquad
1\ E_h\simeq27.21138625\ \text{eV},\qquad
c=\alpha^{-1}\simeq137.036.
\]

Conversion constants are I/O metadata. They must not be used to distinguish units by approximate equality.

This agrees with the semantic use of `hartree` in SPEX: its arrays are internally in atomic units and multiplication by `hartree` converts an energy for eV output. SPEX declares the eV multiplier, Bohr-to-Angstrom multiplier, and atomic-unit speed of light in `src/global.f:319-326`; explicit eV and Angstrom inputs are divided by those multipliers in `src/getkey.inc:432-438` and `src/getkey.inc:482-488`.

## 2. Coordinates, lattice matrices, and reciprocal vectors

The direct lattice matrix has primitive vectors as columns,

\[
A=(\mathbf a_1\ \mathbf a_2\ \mathbf a_3),\qquad
\Omega=\det A>0.
\]

Fractional and Cartesian positions obey

\[
\mathbf r=A\mathbf s.
\]

The reciprocal lattice matrix is

\[
B=2\pi A^{-T}=(\mathbf b_1\ \mathbf b_2\ \mathbf b_3),
\qquad \mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij},
\]

so an integer triplet \(\mathbf g\in\mathbb Z^3\) denotes

\[
\mathbf G=B\mathbf g.
\]

Reduced \(\mathbf k\) coordinates follow the same rule. Thus `k + g` is dimensionless until multiplication by \(B\). SPEX constructs the three reciprocal columns by cross products and then multiplies by \(2\pi/\Omega\) in `src/getinput.f:3017-3028`. It also rejects a left-handed direct basis; `libmuffintin` adopts the same invariant.

## 3. Fourier and plane-wave normalization

For a periodic scalar field,

\[
f_{\mathbf G}=\frac1\Omega\int_\Omega
f(\mathbf r)e^{-i\mathbf G\cdot\mathbf r}\,d^3r,
\qquad
f(\mathbf r)=\sum_{\mathbf G}f_{\mathbf G}
e^{i\mathbf G\cdot\mathbf r}.
\]

The normalized plane wave is

\[
\langle\mathbf r|\mathbf k+\mathbf G\rangle
=\Omega^{-1/2}e^{i(\mathbf k+\mathbf G)\cdot\mathbf r}.
\]

With this convention, an interstitial overlap is the Fourier coefficient of the interstitial characteristic function at \(\mathbf G-\mathbf G'\). Translation of a sphere centered at \(\boldsymbol\tau\) contributes \(e^{-i\mathbf G\cdot\boldsymbol\tau}\). SPEX's analytic step function uses precisely this negative-sign Fourier phase in `src/overlap.f:141-172`; its APW phase construction uses the corresponding positive phase in `src/hamilton.f:213-232`.

There are two common interstitial kinetic-energy assemblies, and they must be
recorded as an assembly convention rather than silently interchanged.  With
\(\mathbf K_{\mathbf G}=\mathbf k+\mathbf G\) and
\(\Theta_{\mathbf G-\mathbf G'}\) the interstitial step coefficient, the
gradient form is

\[
T^{\nabla}_{\mathbf G\mathbf G'}=
\frac12\mathbf K_{\mathbf G}\!\cdot\!\mathbf K_{\mathbf G'}
\Theta_{\mathbf G-\mathbf G'},
\]

whereas SPEX uses the symmetrized-Laplacian form

\[
T^{\Delta}_{\mathbf G\mathbf G'}=
\frac14\left(|\mathbf K_{\mathbf G}|^2+|\mathbf K_{\mathbf G'}|^2\right)
\Theta_{\mathbf G-\mathbf G'}.
\]

They differ by

\[
T^{\Delta}-T^{\nabla}
=\frac14|\mathbf K_{\mathbf G}-\mathbf K_{\mathbf G'}|^2
 \Theta_{\mathbf G-\mathbf G'}
=\frac14|\mathbf G-\mathbf G'|^2
 \Theta_{\mathbf G-\mathbf G'}.
\]

This is the surface term produced when integration by parts acts on the
discontinuous interstitial characteristic function.  It is therefore tied to
how muffin-tin and interstitial pieces are assembled.  The plan's gradient
form and SPEX's symmetrized-Laplacian form are each valid only together with
their corresponding sphere/interface terms.  SPEX's convention is visible in
`src/hamilton.f:935-953` and `src/iterate_subs.f:99-110`.

## 4. Radial functions and potentials

The scalar radial wavefunction convention is

\[
\psi_{lm}(\mathbf r)=u_l(r)Y_{lm}(\hat{\mathbf r})
=\frac{P_l(r)}rY_{lm}(\hat{\mathbf r}),
\qquad P_l(r)=r u_l(r).
\]

Radial arrays must say whether they store \(u\), \(P=ru\), or a code-specific quantity such as \(rV\). The library's potential channel \(V_{LM}(r)\) is an energy in Hartree, not \(rV\). A converter must undo any producer-specific radial prefactor before constructing the snapshot.

In particular, the physical spherical potential stored by `libmuffintin` is
the actual function \(V_{00}^{\rm phys}(r)\) that enters the radial equation.
For normalized \(Y_{00}=1/\sqrt{4\pi}\), SPEX's raw first muffin-tin
coefficient obeys

\[
V_{00}^{\rm phys}(r)=\frac{\mathtt{vmt}(r,1)}{\sqrt{4\pi}}.
\]

Thus a SPEX snapshot adapter divides `vmt(:,1)` by \(\sqrt{4\pi}\); downstream
radial solvers must not divide it again.  This normalization is explicit in
`src/getinput.f:1551-1569`, and the same conversion appears at the radial-solver
call boundary in `src/iterate.f:1032` and in the FLEUR reader
`src/readwrite_fleur.f:796-800`.

For a two-component radial solution, `large` and `small` mean the radial numerators whose norm is

\[
\int_0^R\left(P^2+Q^2\right)dr,
\]

unless a source adapter explicitly documents a scaled auxiliary component. SPEX's scalar-relativistic solver internally rescales its second first-order variable by \(1/c\) before normalization and return (`src/dirac.f:194-204`). The adapter must therefore record the returned physical component, not the pre-rescaling Runge--Kutta variable.

## 5. Angular, index, and conjugation rules

- Complex spherical harmonics use the Condon--Shortley phase and unit-sphere normalization. Their exact definition and real-harmonic transform are derived in [02](02_angular_reciprocal_and_step_function.md).
- Public \((l,m)\) storage is in increasing \(l\), then increasing \(m=-l,\ldots,l\). The zero-based compound index is \(l(l+1)+m\).
- Mathematical inner products conjugate the left argument: \(\langle f|g\rangle=\int f^*g\). Matrix storage and transformations must preserve this rule even when a real-harmonic intermediate happens to be real.
- A spin index is not a Dirac component index. Collinear spin channels are independent scalar problems; a four-spinor is defined separately in [06](06_dirac_4c_core_and_valence.md).

## 6. Boundary and serialization rules

Every serialized dimensional field carries a unit tag. The canonical tags are `hartree`, `bohr`, and `bohr^-1`; a converter may accept `eV`, `rydberg`, or `angstrom`, but the in-memory result is canonical. Energy zero is separate metadata: conversion of units must never silently shift a potential or energy parameter.

For each radial mesh, serialize \((r_{\min},h,N,R)\) and require

\[
R=r_{\min}e^{(N-1)h}
\]

within the declared tolerance. For each radial solution, serialize enough boundary data to recompute and check the logarithmic derivative. Do not serialize only a logarithmic derivative when LAPW matching also needs the value and slope separately.

## 7. Minimum invariants

1. Converting Hartree to eV and back, or Bohr to Angstrom and back, is explicit and round-trips within floating-point error.
2. \(A^TB=2\pi I\), \(\det A>0\), and reciprocal norms are computed only after conversion to Cartesian `bohr^-1`.
3. The free-electron kinetic diagonal is \(|\mathbf k+\mathbf G|^2/2\), which detects an accidental Rydberg convention immediately.
4. Fourier reconstruction uses the same sign as the analytic step function.
5. Snapshot readers reject absent or unknown dimensional units rather than guessing.

## Source anchors

The SPEX anchors below were inspected in the local tree
`/Users/zerozaki07/Documents/dft_codes/spex06.00pre36`.

- SPEX constants and radial-grid type: `spex06.00pre36/src/global.f:302-326` (`gridtype`, `hartree`, `clight`, `bohr`).
- SPEX explicit input conversions: `spex06.00pre36/src/key.f:34-52` and `spex06.00pre36/src/getkey.inc:432-438,482-488`.
- SPEX reciprocal lattice: `spex06.00pre36/src/getinput.f:3017-3028` (`def_reciprocal`).
- SPEX APW Cartesian reciprocal vector and phases: `spex06.00pre36/src/hamilton.f:213-232` (`hamiltonian_kinit`).
- SPEX step-function Fourier phase: `spex06.00pre36/src/overlap.f:141-172` (`stepfunction`).
- SPEX symmetrized-Laplacian interstitial kinetic term: `spex06.00pre36/src/hamilton.f:935-953` and `spex06.00pre36/src/iterate_subs.f:99-110`.
- SPEX spherical-potential normalization: `spex06.00pre36/src/getinput.f:1551-1569`, `spex06.00pre36/src/iterate.f:1032`, and `spex06.00pre36/src/readwrite_fleur.f:796-800`.
- SPEX scalar-relativistic normalization and returned boundary derivative: `spex06.00pre36/src/dirac.f:194-204` (`dirac_hom_x`).
