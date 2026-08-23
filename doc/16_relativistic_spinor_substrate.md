# 16. Relativistic spinor substrate, SRA-LAPW, and confined HDLO

This note defines the M-Ka contract that activates a parallel
four-component valence route.  It extends the Dirac conventions in
[06](06_dirac_4c_core_and_valence.md) and adapts them to the value/slope LAPW
machinery in [08](08_lapw_matching_and_overlap.md).  It does not change the
meaning of the scalar $(l,m)$ basis, `BoundaryData`, `BasisLayout`, or the
independent `Collinear<T>` route.  Energies and potentials are Hartree and
lengths are Bohr.

## 1. Shared spin-angular channels

`libmuffintin-core` owns the spin-angular labels used by radial, sphere,
basis, and future scattering code.  A validated `Kappa` is a nonzero signed
integer with

```math
j=|\kappa|-\frac12,\qquad
l(\kappa)=
\begin{cases}
-\kappa-1,&\kappa<0,\\
\kappa,&\kappa>0,
\end{cases}
\qquad
l(-\kappa)=l(\kappa)-\mathrm{sgn}(\kappa).
```

Half-integer magnetic quantum numbers are never stored as floating-point
values.  `TwiceMu` stores the exact odd integer $2\mu$ and is valid for a
given $\kappa$ precisely when

```math
-2j\le 2\mu\le 2j,\qquad 2\mu=-2j,-2j+2,\ldots,2j.
```

A `RelativisticChannel` is the pair $\Lambda=(\kappa,2\mu)$.  The canonical
enumeration is deterministic: traverse the supplied validated `Kappa` list
in its stored order and, within each $\kappa$, traverse $2\mu$ from $-2j$ to
$2j$ in steps of two.  Code that chooses a cutoff must materialize and retain
that `Kappa` list; no consumer may infer an alternative ordering from a flat
offset.  Duplicate $\kappa$, $\kappa=0$, an even $2\mu$, and an out-of-range
$2\mu$ are rejected.

With normalized Condon–Shortley spherical harmonics, the real
Clebsch–Gordan phase is fixed by the standard Wigner $3j$ relation

```math
\left\langle lm,\frac12s\middle|j\mu\right\rangle
=(-1)^{l-1/2+\mu}\sqrt{2j+1}
\begin{pmatrix}l&\frac12&j\\m&s&-\mu\end{pmatrix}.
```

The spinor-harmonic convention is then

```math
\boxed{
\Omega_{\kappa\mu}(\hat r)
=\sum_{m,s}
\left\langle l(\kappa)m,\frac12s\middle|j\mu\right\rangle
Y_{l(\kappa)m}(\hat r)\chi_s },
\qquad m+s=\mu.
```

Here $s=\pm\tfrac12$ and the Pauli basis is ordered
$(\chi_{+1/2},\chi_{-1/2})$.  The Clebsch–Gordan table, spinor harmonic, and
channel enumerator are one shared implementation rather than separate
crate-local copies.  They must give
$\int\Omega_{\Lambda}^{\dagger}\Omega_{\Lambda'}d\hat r=\delta_{\Lambda\Lambda'}$.

## 2. Central-field Dirac solution and energy derivatives

For a real central scalar potential and a real, rest-energy-subtracted energy
$E$, the physical spinor is

```math
\Psi_{\kappa\mu}(\mathbf r)=\frac1r
\begin{pmatrix}
P_\kappa(r)\Omega_{\kappa\mu}(\hat r)\\
iQ_\kappa(r)\Omega_{-\kappa\mu}(\hat r)
\end{pmatrix}.
```

The public solution stores the physical reduced components $(P,Q)$, whose
norm is $\int_0^R(P^2+Q^2)dr$.  Integration may privately use
$q=cQ$, for which the first-order system is

```math
\boxed{
P'=-\frac{\kappa}{r}P+
\left(2+\frac{E-V}{c^2}\right)q },
\qquad
\boxed{
q'=\frac{\kappa}{r}q+(V-E)P }.
```

The $cQ$ array is an implementation variable and must not escape through a
public solution, trace, norm, or serialized fixture.  In physical variables,

```math
P'=-\frac{\kappa}{r}P+\frac{E-V+2c^2}{c}Q,
\qquad
Q'=\frac{\kappa}{r}Q-\frac{E-V}{c}P.
```

At fixed $V$, analytic differentiation with respect to $E$ gives the
inhomogeneous system

```math
\dot P'=-\frac{\kappa}{r}\dot P+
\left(2+\frac{E-V}{c^2}\right)\dot q+\frac{q}{c^2},
```

```math
\dot q'=\frac{\kappa}{r}\dot q+(V-E)\dot P-P,
\qquad \dot Q=\frac{\dot q}{c}.
```

Thus the M-Ka `solve_valence_dirac` contract returns a normalized physical
solution and its analytic energy derivatives; it does not estimate them by
finite differences.  The regular-origin phase is fixed by choosing the
leading nonzero coefficient of $P$ positive.  After applying the same
normalization scale to the raw solution and derivative, remove the remaining
homogeneous freedom with

```math
a=\int_0^R(P\dot P_{\rm raw}+Q\dot Q_{\rm raw})dr,
```

```math
(\dot P,\dot Q)=
(\dot P_{\rm raw},\dot Q_{\rm raw})-a(P,Q).
```

The resulting gauge obeys

```math
\int_0^R(P^2+Q^2)dr=1,
\qquad
\langle R_\kappa|\dot R_\kappa\rangle
=\int_0^R(P\dot P+Q\dot Q)dr=0.
```

The first derivative is not separately normalized.  Its norm is retained as a
basis metric.  M-Ka also requires the analytic second energy derivative.  At
fixed $V$, differentiating the inhomogeneous system once more gives

```math
\ddot P'=-\frac{\kappa}{r}\ddot P+
\left(2+\frac{E-V}{c^2}\right)\ddot q+\frac{2\dot q}{c^2},
```

```math
\ddot q'=\frac{\kappa}{r}\ddot q+(V-E)\ddot P-2\dot P,
\qquad \ddot Q=\frac{\ddot q}{c}.
```

After fixing the first-derivative gauge above, the second derivative must
obey the exact twice-differentiated normalization identity

```math
\boxed{
\langle R_\kappa|\ddot R_\kappa\rangle
=-\langle\dot R_\kappa|\dot R_\kappa\rangle }.
```

Equivalently, remove the homogeneous part of a raw second derivative with

```math
b=\langle R_\kappa|\ddot R_{\kappa,\mathrm{raw}}\rangle
+\langle\dot R_\kappa|\dot R_\kappa\rangle,
\qquad
\ddot R_\kappa=\ddot R_{\kappa,\mathrm{raw}}-bR_\kappa.
```

This is a normalization identity, not an optional post-hoc orthogonalization
of $\ddot R_\kappa$ to $R_\kappa$.

## 3. Full Dirac trace and the SRA adapter

The representation-neutral boundary object is

```text
DiracBoundaryTrace<T = f64> { p, q, p_prime, q_prime }
```

at the declared radius.  All four entries are physical $(P,Q,P',Q')$;
`p_prime` and `q_prime` are evaluated from the first-order Dirac equations,
not by differentiating sampled arrays.  The energy derivatives have analogous
traces $(\dot P,\dot Q,\dot P',\dot Q')$ and
$(\ddot P,\ddot Q,\ddot P',\ddot Q')$ evaluated from their inhomogeneous
equations.  A full trace is not scalar LAPW `BoundaryData` and must not be
lossily converted at the radial-solver boundary.

The current valence solver produces the default real trace. The scalar type
parameter permits a later complex-energy scattering solver to return
`DiracBoundaryTrace<Complex64>` without changing the channel or boundary
shape; M-Ka itself does not implement that solver.

SRA-LAPW is an explicit downstream adapter.  It discards the small-component
boundary data for envelope matching and maps the large component to the
existing scalar pair

```math
\boxed{U=\frac{P(R)}R},
\qquad
\boxed{U_r=\frac{P'(R)}R-\frac{P(R)}{R^2}}.
```

The derivative column is mapped by the same rule,

```math
\dot U=\frac{\dot P(R)}R,
\qquad
\dot U_r=\frac{\dot P'(R)}R-\frac{\dot P(R)}{R^2}.
```

The second-derivative column uses the identical linear adapter,

```math
\ddot U=\frac{\ddot P(R)}R,
\qquad
\ddot U_r=\frac{\ddot P'(R)}R-\frac{\ddot P(R)}{R^2}.
```

Only the base and first-derivative `BoundaryData` columns enter the unchanged
$2\times2$ augmented-plane-wave value/slope match.  The second-derivative
column is consumed by the confined HDLO construction below.  The interstitial
envelope is a two-component Pauli plane wave and has no small component.
Inside a sphere, the matched coefficients
multiply both physical radial components of each Dirac solution.  This is the
declared SRA approximation, including its Schlosser–Marcus surface
convention; absence of an interstitial small component is not permission to
substitute scalar-relativistic radial data. FRA is not exposed by this path.

For each requested $(\kappa,2\mu)$ channel, the confined SRA HDLO is the
physical spinor combination

```math
R_{\mathrm{HDLO}}=\ddot R_\kappa+aR_\kappa+b\dot R_\kappa,
```

where the two coefficients are determined by the large-component boundary
system

```math
\begin{pmatrix}
U&\dot U\\
U_r&\dot U_r
\end{pmatrix}
\begin{pmatrix}a\\b\end{pmatrix}
=-
\begin{pmatrix}\ddot U\\\ddot U_r\end{pmatrix}.
```

Thus $U_{\mathrm{HDLO}}=U_{r,\mathrm{HDLO}}=0$ exactly up to the solver
tolerance.  The full $(P_{\mathrm{HDLO}},Q_{\mathrm{HDLO}})$ pair is then
normalized with the physical radial norm; the small component is retained
inside the sphere even though SRA has no interstitial small component.  A
singular or ill-conditioned value/slope system is an explicit construction
failure, not permission to drop a constraint or substitute a scalar radial
function.  The HDLO is a site-local orbital and has no interstitial tail.

For one normalized angular channel, let $I$ and $t$ denote the interstitial
and muffin-tin large-component traces and let the prime denote the outward
normal derivative. In the library's Hartree convention, $T=-\nabla^2/2$, the
bilinear Schlosser–Marcus correction is

```math
\Delta T_{LR}=-\frac{R^2}{4}
\left[
t_L^* I_R'-(t_L')^*I_R-I_L^*t_R'+(I_L')^*t_R
\right].
```

The factor would be $-1/2$ in Kutepov's Rydberg convention. The explicit
`SraSurfaceTrace` helper keeps $I$ and $t$ separate, is Hermitian under
$L\leftrightarrow R$, and vanishes when both value and derivative are
continuous. This term belongs to LAPW weak-form assembly, not to the sphere
volume integral.

## 4. Separate large- and small-component sphere algebra

For a scalar potential channel $V_{LM}(r)Y_{LM}(\hat r)$, define two spinor
Gaunt factors,

```math
\mathcal G^{PP}_{LM}(\Lambda,\Lambda')
=\int d\hat r\,
\Omega_{\kappa\mu}^{\dagger}Y_{LM}
\Omega_{\kappa'\mu'},
```

```math
\mathcal G^{QQ}_{LM}(\Lambda,\Lambda')
=\int d\hat r\,
\Omega_{-\kappa\mu}^{\dagger}Y_{LM}
\Omega_{-\kappa'\mu'}.
```

Their Clebsch–Gordan reductions are evaluated independently:

```math
\mathcal G^{PP}_{LM}
=\sum_s C^{\Lambda *}_{m s}C^{\Lambda'}_{m' s}
G_{l m,LM,l'm'},
\qquad
m=\mu-s,\quad m'=\mu'-s,
```

where $l=l(\kappa)$ and $l'=l(\kappa')$.  The $QQ$ expression uses the
same reduction with $l(-\kappa)$ and $l(-\kappa')$, and
$G_{l m,LM,l'm'}=\int Y_{lm}^{*}Y_{LM}Y_{l'm'}d\hat r$ uses the scalar
Gaunt convention of [02](02_angular_reciprocal_and_step_function.md).
Consequently a scalar sphere matrix element has the form

```math
\sum_{LM}\int_0^R dr\,V_{LM}(r)
\left[
P_\kappa P_{\kappa'}\mathcal G^{PP}_{LM}
+Q_\kappa Q_{\kappa'}\mathcal G^{QQ}_{LM}
\right].
```

There is no scalar-potential $PQ$ cross term.  The $PP$ and $QQ$ radial
integrals and angular factors remain separate in `SpinorSphereOrbital` and
its operator blocks; collapsing them into one effective Gaunt factor loses
the distinct $l(\kappa)$ and $l(-\kappa)$ selection rules.  Non-spherical
$V_{LM}$ may couple different $\Lambda$ and $\Lambda'$, but that sphere
coupling does not turn the central-field radial solver into a radial Dirac
solver coupled in $\kappa$.

## 5. SRA plane-wave projection and basis order

At fixed $\mathbf k$, the interstitial Pauli plane waves are

```math
\chi_{s\mathbf G}(\mathbf r)=\Omega_{\rm cell}^{-1/2}
e^{i(\mathbf k+\mathbf G)\cdot\mathbf r}\chi_s.
```

Their global index is exactly

```text
pw_index(spin, g) = spin * n_g + g
```

with spin slow, spin order $(+\tfrac12,-\tfrac12)$, and the existing stored
$\mathbf G$ order unchanged.  Every Pauli plane wave precedes every
site-local spinor orbital.  The local suffix is deterministic in stored site
order, then stored $\kappa$ order, increasing $2\mu$, and radial index $n$
fastest for ordinary LO/HELO entries $(\kappa,2\mu,n)$.  A requested HDLO is
an explicit deterministic site-local entry in the same channel order.

Projecting a scalar plane wave of spin $s$ onto a large-component spinor
channel gives

```math
C_{\Lambda s}(\mathbf q)=\frac{4\pi}{\sqrt{\Omega_{\rm cell}}}
i^{l(\kappa)}
\left\langle l(\kappa)m,\frac12s\middle|j\mu\right\rangle
Y_{l(\kappa)m}^{*}(\hat{\mathbf q}),
\qquad m=\mu-s.
```

The coefficient is zero if $m$ is not an allowed integer magnetic quantum
number.  The site-centered coefficient also carries the existing separate
phase $e^{i\mathbf q\cdot\mathbf R_a}$ and the two real SRA value/slope match
coefficients.  This defines the compiled spinor augmentation without routing
through scalar `lm` storage.

The spinor projection matrix maps global `[Pauli PW][site spinor LO]` columns
to site-local $(\Lambda,\mathrm{radial})$ coordinates.  Both sphere overlap
and Hamiltonian are carried by `SpinorSiteOperatorBlocks`, whose explicit
ordered `RelativisticChannel` list must match every augmentation exactly;
equal matrix dimensions alone are not accepted. They use the shared
congruence $P_a^\dagger B_aP_a$. The
unchanged SRA interstitial overlap, kinetic, and scalar-potential kernels are
lifted into equal-spin blocks with a factor $\delta_{ss'}$; any spin mixing
comes from explicit spinor sphere blocks, not from reinterpreting the
collinear driver.

## 6. KKR/LMTO consumer boundary

`Kappa`, `TwiceMu`, `RelativisticChannel`, the Dirac solution and its energy
derivatives, and `DiracBoundaryTrace` are free of plane-wave, LAPW, step-function,
screening, and SCF types.  SRA `BoundaryData` is one lossy envelope adapter,
not the canonical radial representation.

This boundary preserves the information that later fully relativistic
KKR/LMTO consumers need to construct convention-tagged regular/irregular
Wronskians, complex-energy single-site $t_{\Lambda\Lambda'}(z)$, screened
potential functions, and kink or slope matrices.  M-Ka does not implement
those consumers, complex-energy radial integration, structure constants,
screening, or Green functions.  A future complex-energy generalization must
extend the representation rather than reconstructing $Q$ and $Q'$ from the
SRA pair $(U,U_r)$.

## 7. Explicit non-goals and failure boundaries

M-Ka is SRA-only and does not provide FRA-LAPW or relativistic interstitial
plane waves; radial magnetic fields or general Dirac equations coupled in
$\kappa$; spinor product, MPB, or THC identities; density and potential
synthesis; occupations, XC, mixing, or SCF.  It does not add
`kappa` to scalar `ProductRadialId`; the relativistic orbital/product bridge
belongs to M-L. Scattering and magnetic-radial request APIs are likewise not
part of this milestone; absence of those modes must not be represented by a
fallback to SRA, Koelling–Harmon, or scalar data.

### Far-future FRA research option

FRA is retained only as a far-future, non-production research option for
methodological completeness.  It is outside v0.2, outside the current M-Ka
implementation and acceptance boundary, and does not justify an FRA request
API or negative-test surface now.  Any later FRA investigation must first
define relativistic interstitial basis functions, four-component matching,
surface terms, and independent fixtures as a separate research milestone; it
must not widen or silently alter the present SRA contract.

## 8. Acceptance boundary

M-Ka is accepted only when all of the following are demonstrated:

1. Complete deterministic $(\kappa,2\mu)$ enumeration, rejection of invalid
   labels, normalized spinor harmonics, and a fixed Clebsch–Gordan phase.
2. Dirac ODE residuals for $(P,Q)$, $(\dot P,\dot Q)$, and
   $(\ddot P,\ddot Q)$ using physical public components, with internal $cQ$
   scaling invisible at the API boundary.
3. Phase-aligned centered-finite-difference agreement for the analytic first
   and second energy derivatives on the mesh and for every entry of the full
   Dirac boundary traces.
4. Unit radial norm, $\langle R_\kappa|\dot R_\kappa\rangle=0$, and the exact
   identity $\langle R_\kappa|\ddot R_\kappa\rangle=-\langle\dot R_\kappa|\dot R_\kappa\rangle$
   in the fixed gauge.
5. Independent oracle values for $PP$ and $QQ$ spinor Gaunt reductions,
   including non-diagonal channel blocks and Hermitian conjugation checks.
6. SRA $(U,U_r)$, $(\dot U,\dot U_r)$, and
   $(\ddot U,\ddot U_r)$ conversion from equation-derived traces; an HDLO
   constructed from $\ddot R+aR+b\dot R$ with both large-component boundary
   residuals no larger than `1e-10`; and unit physical HDLO norm.
7. Deterministic `spin * n_g + g` Pauli-PW order, deterministic spinor-LO
   suffixes, and projection coefficients consistent with direct spinor
   harmonic evaluation.
8. Hermitian spinor $H$ and $S$, positive retained overlap, and bounded
   generalized-eigen residuals from the common projection congruence.
9. A large $c$ reduction in which one repository-local scalar LAPW frozen fixture is
   duplicated across the two Pauli-spin blocks, after the unitary
   coupled/uncoupled angular transformation.

The second-derivative/HDLO gate and the item 9 repository-local large $c$
fixture are implemented and exercised by focused regression tests.  This does
not claim completion of the full M-Ka acceptance list or cross-code validation
against a FlapwMBPT/source-equivalent band fixture. FRA is deliberately excluded
from both the implementation and the executable acceptance gates; M-Ka exposes
only SRA rather than a relativistic-mode request enum.

An independent frozen FlapwMBPT-SRA or source-equivalent fixture must still
cover selected physical radial traces, augmentation coefficients, sphere
blocks, and frozen-potential bands before this path is described as
cross-code validated.  This fixture is an acceptance capability boundary,
not a validation result claimed by this document.

The primary convention sources for that fixture are Kutepov's Dirac
APW/LAPW formulation (arXiv:2012.04992), Ebert et al.'s fully relativistic
KKR formulation (arXiv:1512.04294), and the QuESTAAL KKR/LMTO
Green-function documentation.  Fixture metadata records the exact equation,
unit, phase, SRA, and boundary conventions instead of inferring them from
a method name.
