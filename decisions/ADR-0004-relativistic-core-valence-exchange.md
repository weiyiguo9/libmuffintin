# Relativistic core–valence exchange in LAPW and MTO bases

- Status: recorded 2026-08-28 during the Sm/Dy THC experiment; `doc/23_core_valence_exchange.md` on `main` is the normative form where they differ
- Date: 2026-08-28
- Imported: 2026-09-08 from `scratch/thc_smdy_experiment/core-valence.md`, body unchanged

## Decision

The core and valence spaces should use different representations.

1. Bound core states are always solved as four-component Dirac states on the
   spherical site potential and retain $\kappa$, large and small radial
   components, an explicit occupation model over $2m_j$, and the norm outside
   the muffin-tin sphere.
2. Deep core states are **not THC orbitals**. Static core–valence Fock
   exchange is stored as an exact, site-local Hermitian operator in the
   natural partial-wave/spinor basis.
3. Extended valence–valence exchange remains the target of THC plus the
   Weinert muffin-tin Coulomb metric.
4. LAPW, SRA-LAPW, LMTO, and mixed APW/MTO bases consume the same local core
   operator through their site-augmentation projections. The global basis
   changes; the physical core operator does not.
5. A state with an appreciable interstitial tail or chemically active
   occupation is semicore/valence for this purpose. It must be promoted into
   the active one-particle and product windows rather than hidden in the
   deep-core approximation.

Thus the production split is

```math
\boxed{
\Sigma_x = \Sigma_{x,\mathrm{val}}^{\mathrm{THC+Weinert}}
          + \Sigma_{x,\mathrm{core}}^{\mathrm{local\ exact}}
}.
```

The Sm experiment already validates the scalar-collinear instance of this
decision: 72 symmetric doubles (576 bytes) reproduce the SPEX core exchange
rows exactly at the exported precision. A low-rank factor is not smaller at
useful accuracy. Adding core Bloch columns to THC would therefore add rank and
duplicate equivalent translated core states without improving the result.

## 1. Common exact operator

For a core state $c$ on site $a$, define the core exchange matrix in any
valence basis $\phi_i$ as

```math
(\Sigma^x_{\mathrm{core}})_{ij}
=-\sum_a\sum_{c\in\mathrm{core}(a)} f_c
\iint_{\mathrm{MT}_a} d^3r\,d^3r'
[\phi_i^\dagger(\mathbf r)\psi_c(\mathbf r)]
\frac{1}{|\mathbf r-\mathbf r'|}
[\psi_c^\dagger(\mathbf r')\phi_j(\mathbf r')].
```

The integral is local because a deep core orbital is confined to one site,
not because the valence basis is local. Let the restriction of a global basis
function to sphere $a$ be

```math
\phi_i|_{\mathrm{MT}_a}=\sum_\alpha
A^a_{\alpha i}\Phi^a_\alpha.
```

After all occupied core states and allowed Coulomb multipoles have been folded
into a local Hermitian block $K_a$, every basis family uses the same
congruence:

```math
\boxed{
\Sigma^x_{\mathrm{core}}=-\sum_a A_a^\dagger K_a A_a }.
```

This is the useful compact representation. $K_a$ is independent of $k$ point,
band, and envelope choice. Only $A_a$ depends on whether the global basis is
LAPW, SRA-LAPW, LMTO, or a mixture.

The MT truncation must not be silent. The Dirac core record retains
`norm_mt` and `norm_outside`. A configured tolerance on `norm_outside` decides
whether the local operator is accepted; failure promotes the state to a
semicore/valence treatment.

## 2. Relativistic radial conventions

A central-field Dirac state is

```math
\Psi_{\kappa\mu}(\mathbf r)=\frac1r
\begin{pmatrix}
P_\kappa(r)\Omega_{\kappa\mu}(\hat r)\\
iQ_\kappa(r)\Omega_{-\kappa\mu}(\hat r)
\end{pmatrix},
```

with

```math
\kappa=-(l+1)\quad(j=l+1/2),\qquad
\kappa=+l\quad(j=l-1/2).
```

$P$ and $Q$ are the physical components, normalized with
$\int (P^2+Q^2)\,\mathrm{d}r$. An internal $cQ$ integration variable is never
an exchange input.

For a scalar Coulomb vertex, a valence–core transition density is the sum of
large–large and small–small parts,

```math
\rho_{vc}=\rho_{vc}^{PP}+\rho_{vc}^{QQ}.
```

There is no $PQ$ term in the Dirac inner product. However, the Coulomb action
on the total transition density contains all interactions among its $PP$ and
$QQ$ pieces; the two pieces must not be evaluated as unrelated exchange
energies. Their angular expansions use $\Omega_{\kappa}$ and
$\Omega_{-\kappa}$ separately because their orbital angular momenta and
selection rules differ.

This distinction is also why a Koelling–Harmon pair $(P,Q)$ cannot be
relabelled as a Dirac spinor. KH retains the angular object $Y_{lm}\chi_s$ and
has no $\kappa$ and no explicit $j$ splitting.

## 3. Configuration matrix

| Core | Valence route | Local index of $K_a$ | Required treatment |
|---|---|---|---|
| full Dirac | KH scalar relativistic, no valence SOC | $(l,\sigma,n)$ | spherical block repeated over $m$; current validated case |
| full Dirac | KH first variation plus nonmagnetic second-variation SOC | full first-variation $(l,m,\sigma,n)$ | build the coupled operator before SOC rotation, then apply $U_{\mathrm{SOC}}^\dagger K U_{\mathrm{SOC}}$ |
| full Dirac | 4c-in-MT SRA-LAPW | $(\kappa,2m_j,n)$ | keep separate $PP$ and $QQ$ spinor angular algebra; two-component interstitial |
| full Dirac | fully relativistic/4c LMTO | local relativistic channel $\Lambda=(\kappa,2m_j)$ plus radial/head index | use the complete Dirac boundary/scattering data and project every MTO tail entering the site |
| full Dirac | magnetic SOC or noncollinear 4c first variation | coupled spinor channels, possibly a core density matrix | no scalar spin blocks and no second-variation shortcut |

4c SRA-LAPW here means four components inside each MT sphere. The
interstitial envelope remains a two-component Pauli plane wave with zero small
component; matching uses only the large-component value and slope. A true
FRA-LAPW with relativistic four-component interstitial functions is a
different, currently unspecified research route.

## 4. Full-Dirac core plus KH valence

KH valence partial waves are labelled $(l,m,\sigma,n)$ and have physical
radial components $(P_{ln},Q_{ln})$ but scalar angular functions
$Y_{lm}\chi_\sigma$. For each core shell, valence radial pair, and allowed
Coulomb multipole $L$, form the radial transition product used by SPEX,

```math
f_{c,ln}(r)=
\frac{P_c(r)P_{ln}(r)+Q_c(r)Q_{ln}(r)}{r},
```

apply the two radial Poisson primitives, and sum the angular Gaunt factors and
core occupations. For a spherical closed core and scalar-collinear valence,
the result reduces to

```math
K_a = \bigoplus_{l,\sigma} K^a_{l\sigma},
```

where each radial matrix is repeated for every $m$. A band contraction is

```math
\Sigma^x_{b,\mathrm{core}}
=-\sum_{a,l,m,\sigma}
(c^a_{blm\sigma})^\dagger K^a_{l\sigma}c^a_{blm\sigma}.
```

SPEX keeps SOC-split Dirac core shells even when valence SOC is off. In this
inconsistent relativistic combination, SPEX deliberately removes the
core-SOC-induced terms that depend on $m$, so the exchange potential remains
spherical. The compact $(l,\sigma)$ block reproduces that convention; it should
be labelled **core-SOC spherical projection**, not advertised as a general
four-component identity.

If core occupations are unequal in $m_j$, if a magnetic core density is
retained, or if valence SOC is active, this reduction is invalid. Retain the
full spin-angular block or the appropriate core density matrix instead of
forcing an average over $m$.

## 5. Nonmagnetic second-variation SOC

Let $U_{\mathrm{SOC}}(k)$ mix first-variation scalar-relativistic basis states
into SOC eigenstates. The correct order is

```math
K_{\mathrm{SOC}}(k)
=U_{\mathrm{SOC}}^\dagger(k)K_{\mathrm{FV}}(k)U_{\mathrm{SOC}}(k).
```

$K_{\mathrm{FV}}$ must contain the diagonal spin terms, their dependence on
$m$, and all allowed spin-off-diagonal and orbital-off-diagonal core-SOC
terms. Rotating only the two scalar diagonal values discards precisely the
terms introduced by SOC. In SPEX language, the generalization must reproduce
the `qfac0`, `qfac1`, and `qfac2` structure of `exchange_core`, not only
`qfac0`.

Second variation is appropriate for nonmagnetic SOC in the adopted SPEX
policy. Magnetic SOC and noncollinear calculations instead use the full spinor
first-variation route below.

## 6. 4c-in-MT SRA-LAPW

The local valence basis is $\Phi_\alpha = \Phi_{(\kappa,2m_j,n)}$ with
physical $(P,Q)$ components. Build spinor transition densities by expanding

```math
\Omega_{\kappa_v\mu_v}^\dagger\Omega_{\kappa_c\mu_c}
\quad\text{and}\quad
\Omega_{-\kappa_v\mu_v}^\dagger\Omega_{-\kappa_c\mu_c}
```

into scalar $Y_{LM}$ channels using Clebsch–Gordan coefficients and scalar
Gaunts. Run the ordinary spherical multipole Poisson kernel on their summed
radial-angular transition density, then assemble

```math
K^a_{(\kappa\mu n),(\kappa'\mu'n')}.
```

The SRA plane-wave/spinor augmentation matrix is $A_a$. It maps global Pauli
plane waves and local orbitals to the site spinor channels, so the global
core exchange is again $-A_a^\dagger K_a A_a$. The small component is retained
in the sphere integral even though it is absent in the interstitial. Only the
large component participates in the SRA boundary match.

No Schlosser–Marcus surface term belongs to this Coulomb operator. That
surface correction is a kinetic weak-form term; boundary matching affects the
augmentation coefficients, not the definition of the local Coulomb kernel.

## 7. Fully relativistic 4c LMTO

The core decision does not change for LMTO. A relativistic LMTO/KKR consumer
must start from the unprojected Dirac boundary trace $(P,Q,P',Q')$, or its
complex-energy regular/irregular generalization, and channels
$\Lambda=(\kappa,2m_j)$. It must not reconstruct the missing small-component
information from the lossy SRA pair $(U,U_r)$.

For an LMTO basis function $\chi_I$, obtain its partial-wave coefficients
inside every physical sphere,

```math
\chi_I|_{\mathrm{MT}_a}=\sum_\alpha
A^a_{\alpha I}\Phi^a_\alpha.
```

Then use the same local spinor core block:

```math
(\Sigma^x_{\mathrm{core}})_{IJ}
=-\sum_a\sum_{\alpha\beta}
(A^a_{\alpha I})^*K^a_{\alpha\beta}A^a_{\beta J}.
```

“Site-local core operator” does **not** mean “diagonal in MTO centers.” An MTO
centred on site $b$ can have a screened or smooth-Hankel tail inside sphere
$a$; that tail contributes through $A^a$. Off-centre and off-diagonal global
matrix elements are therefore retained exactly by the projection.

For a genuinely four-component LMTO/interstitial formulation, the envelope
and scattering conventions determine $A_a$, while $K_a$ is unchanged. For an
SRA-like LMTO with Pauli interstitial envelopes, the same caveat as SRA-LAPW
applies: four components are retained in the sphere and the small component
is absent outside. The current libmuffintin v0.3 LMTO material is a design
plan, not a completed implementation, so these two capabilities must remain
explicit rather than being selected from a method name alone.

The valence products of an FP-LMTO or PMT basis may still feed the common
product IR and THC/MPB backends. This statement concerns **valence–valence**
products. It does not imply that deep core orbitals enter the THC selector.

## 8. Data model

The physical schemas should remain separate:

```text
CoreRadialShell {
  site, n, kappa, energy,
  p[], q[], norm_mt, norm_outside, dirac_convention
}

CoreOccupation =
  SphericalShell { shell_occupation }
  | SpinAngularDensity { channels: [(kappa, twice_mj)], density_matrix }

ValenceRelativity =
  KhScalar { spin_layout }
  | SecondVariationSoc { first_variation_layout, mixing }
  | SraLapw4c { relativistic_channels, pauli_interstitial }
  | Lmto4c { relativistic_channels, boundary_or_scattering_convention }
  | FirstVariation4c { relativistic_channels, magnetic_structure }

CoreExchangeBlock =
  ScalarRadial { site, l, spin, symmetric_matrix }
  | SpinAngular { site, channels, hermitian_matrix }
```

Do not infer the model from the number of radial arrays: KH and Dirac both
store $(P,Q)$ but have different angular objects. Every serialized block also
records units, spherical-harmonic phase, $\kappa$ ordering, radial mesh,
multipole cutoff, core occupation convention, and whether a spherical
core-SOC projection was applied.

## 9. When core products do belong in an auxiliary space

The “core is not THC” decision is specific to static deep-core Fock exchange.
Core-labelled product channels remain meaningful in two other cases:

- explicit core polarization or core spectroscopy in GW/RPA/BSE, where
  dynamic core transitions must enter the response product space; and
- semicore states promoted into the active valence window because their
  outside norm or chemical hybridization is non-negligible.

Those channels should carry core–valence provenance in the method-neutral
product IR and may be represented by an MPB or a separately converged THC
fit. They must not be conflated with the exact static local core-exchange
block, or the same core contribution will be counted twice.

## 10. Pitfalls that silently produce wrong numbers

Every item below has been observed, or is one step away from something that
was observed, in the Sm experiment. They share a failure mode: the code runs,
the residuals look healthy, and the number is wrong. Each entry names the
mistake, the damage, and the check that catches it.

### Reference-side traps

- **Comparing against an unconverged SPEX product angular cutoff.** SPEX
  defaults to `MBASIS LCUT=5`, but a $4f \times 4f$ product carries an $L=6$
  channel. Raising `LCUT` 5 to 6 moved occupied up-spin 4f exchange by 2.67 to
  4.37 eV; 6 to 7 moved it by 0.01 to 0.02 meV. Nothing in the SPEX output
  announces the omission. Converge `LCUT` before any comparison and report the
  `LCUT` to `LCUT+1` delta next to the result. The same physics appears in the
  Weinert check: `Lmax` 5 to 6 changed a single 4f pair by 39.7 meV.
- **Reading a sampling error as a factorization error.** THC converged cleanly
  to its parent grid and inherited that grid's own 2.31 eV error against SPEX.
  Tightening the rank cannot remove it. Always report two errors: against the
  external reference, and against the same adapter's uncompressed grid. A small
  second number with a large first number means the grid is at fault, not the
  factorization.
- **Absorbing SPEX's separately printed frozen-core exchange.** SPEX prints
  core exchange as its own quantity. A valence-only result compared against the
  unseparated total is mislabelled as all-electron, and adding
  $\Sigma_{x,\mathrm{core}}$ afterwards then double counts it.

### Relativistic traps

- **Treating the spherical block as a general four-component identity.** The
  $(l,\sigma)$ block repeated over $m$ is valid only for a spherical closed core
  contracted with scalar-collinear valence, and it matches SPEX only because
  SPEX deliberately removes the core-SOC-induced terms that depend on $m$ in
  that inconsistent combination. Assert the precondition explicitly and fail
  when valence SOC is active, when $m_j$ occupations are unequal, or when a
  magnetic core density is retained. Do not let a closed-shell sphericity
  argument carry over to a consistently SOC-split core.
- **Rotating only the two scalar diagonal values under second-variation SOC.**
  That discards precisely the terms SOC introduces. $K_{\mathrm{FV}}$ must
  already contain the dependence on $m$ and the spin-off-diagonal and
  orbital-off-diagonal core-SOC terms before $U_{\mathrm{SOC}}$ is applied. In
  SPEX terms, reproducing `qfac0` alone and calling it SOC is wrong; `qfac1`
  and `qfac2` carry the difference. Detection: the result must change when SOC
  is switched on, and must match a full first-variation core-exchange export.
- **Feeding the internal $cQ$ integration variable into exchange.** The
  physical small component is $Q$. The substitution is silent because the
  resulting state still looks normalizable. Check
  $\int (P^2 + Q^2)\,\mathrm{d}r = 1$ with the physical component before use.
- **Evaluating the $PP$ and $QQ$ pieces as two unrelated exchange energies.**
  There is no $PQ$ term in the Dirac inner product, which invites treating the
  two pieces as separable. The Coulomb action on the total transition density
  contains the interactions between them. Form
  $\rho_{vc} = \rho^{PP}_{vc} + \rho^{QQ}_{vc}$ first, then apply the kernel
  once. Their angular expansions still use $\Omega_{\kappa}$ and
  $\Omega_{-\kappa}$ separately.
- **Inferring the relativistic model from the number of radial arrays.** KH and
  Dirac both store $(P,Q)$; only their angular objects differ. A KH pair
  silently accepted as a Dirac spinor produces plausible numbers with no
  $\kappa$ and no $j$ splitting. Require an explicitly tagged model and refuse
  to infer one.

### Basis and bookkeeping traps

- **Implementing "site-local" as "diagonal in MTO centres".** An MTO centred on
  site $b$ has a screened or smooth-Hankel tail inside sphere $a$, and that tail
  contributes through $A^a$. Dropping it looks like a locality optimization and
  silently deletes real off-centre matrix elements. Detection: the off-centre
  global elements must be nonzero, and $A^\dagger K A$ built from two
  independently compiled bases must agree.
- **Silently truncating a state that is not deep core.** The local operator is
  only justified while the core orbital is confined. Record `norm_outside` for
  every shell and fail the local route on the configured tolerance rather than
  accepting a semicore state into the frozen block.
- **Counting a promoted semicore or dynamic core channel twice.** Once a
  channel enters the active product space it must leave the exact local block.
  Keep core–valence provenance on the product IR channel, and keep
  $\Sigma_{x,\mathrm{val}}$, $\Sigma_{x,\mathrm{core}}$, and
  $\Sigma_{x,\mathrm{total}}$ as separate output columns so the overlap is
  visible rather than summed away.

## 11. Required validation

1. Preserve the current scalar Sm regression: 3×3×3 and 4×4×4 results must
   remain identical to SPEX with the 576-byte exact blocks.
2. Compare the KH spherical projection against SPEX with core SOC averaged,
   core SOC split, and `scale_soc` set to zero.
3. For valence SOC, export the complete SPEX first-variation core-exchange
   matrix and verify both the `qfac1`/`qfac2` terms and
   $U_{\mathrm{SOC}}^\dagger K_{\mathrm{FV}} U_{\mathrm{SOC}}$.
4. Test independent $PP$ and $QQ$ spinor-Gaunt oracles, then the Coulomb action
   on their sum. Check Hermiticity and all angular selection rules.
5. In the nonmagnetic 4c route, verify rotational covariance, time reversal,
   and Kramers degeneracy. In magnetic and noncollinear cases, verify
   covariance under a common spin-and-space rotation.
6. Take the large $c$ limit and recover two Pauli copies of the scalar result
   after the coupled/uncoupled angular transformation.
7. For SRA-LAPW and LMTO, reconstruct the same in-sphere valence spinor from
   two independently compiled bases and check that $A^\dagger K A$ gives the
   same physical matrix element.
8. Record `norm_outside` for every core shell and fail the local-core route
   when the configured confinement tolerance is exceeded.
9. Keep $\Sigma_{x,\mathrm{val}}$, $\Sigma_{x,\mathrm{core}}$, and
   $\Sigma_{x,\mathrm{total}}$ as separate output columns so double counting
   remains visible.

## 12. Current implementation boundary

- Implemented and numerically validated here: full-Dirac core contracted with
  scalar-collinear KH valence through exact local radial blocks.
- Specified in libmuffintin documentation but not validated by this experiment:
  the 4c-in-MT SRA-LAPW spinor substrate and its full acceptance matrix.
- Planned, not currently implemented in libmuffintin: fully relativistic
  LMTO/KKR consumers and their complex-energy scattering machinery.
- Explicitly outside the current route: FRA-LAPW with relativistic
  interstitial functions.

No libmuffintin source change is required for this note or for the validated
scalar experiment. The next code extension, if requested, should implement
the method-neutral `SpinAngular` core block and basis projection outside the
existing crates first, then cross-check it against SPEX before proposing a
library API.

## Local source anchors

- `codes/libmuffintin/doc/04_schrodinger_and_scalar_relativistic_radial_equations.md`
- `codes/libmuffintin/doc/06_dirac_4c_core_and_valence.md`
- `codes/libmuffintin/doc/16_relativistic_spinor_substrate.md`
- `codes/libmuffintin/scratch/libmuffintin_v0.2_plan.md`
- `codes/libmuffintin/scratch/libmuffintin_v0.3_mto_family_plan.md`
- `codes/spex06.00pre36/src/exchange.f`, subroutine `exchange_core`
- `notes/RESULTS.md`, compact core–valence exchange section
