# 06. Four-component Dirac core states and the reserved valence interface

This note defines the relativistic objects that cannot be represented by the
scalar Koelling--Harmon label alone.  It also separates a production core-state
solver from a future four-component valence basis.  All equations written as
library equations use Hartree/Bohr units.

## 1. Shifted Dirac Hamiltonian

Subtract the electron rest energy so that the eigenvalue \(\epsilon\) is on the
same energy scale as a nonrelativistic band energy:

\[
\boxed{H_D=c\boldsymbol\alpha\cdot\mathbf p
+(\beta-I)c^2+V(r)},
\qquad H_D\Psi=\epsilon\Psi.
\]

The unshifted total energy is \(\epsilon+c^2\).  The shift changes no spinor or
energy difference but makes an accidental rest-energy offset easy to detect.

For a spherical potential, define the spin-angular quantum number

\[
\kappa=
\begin{cases}
-(l+1),&j=l+\tfrac12,\\
+l,&j=l-\tfrac12,
\end{cases}
\qquad \kappa\ne0.
\]

Conversely,

\[
j=|\kappa|-\frac12,\qquad
l(\kappa)=
\begin{cases}-\kappa-1,&\kappa<0,\\\kappa,&\kappa>0.\end{cases}
\]

The spinor spherical harmonic is

\[
\Omega_{\kappa m_j}(\hat r)=
\sum_{m_l,m_s}
\langle l m_l,\tfrac12m_s|jm_j\rangle
Y_{lm_l}(\hat r)\chi_{m_s}.
\]

The opposite component has orbital angular momentum
\(\bar l=l-\operatorname{sgn}\kappa\).  With a fixed phase convention, write

\[
\boxed{
\Psi_{\kappa m_j}(\mathbf r)=\frac1r
\begin{pmatrix}
P_\kappa(r)\Omega_{\kappa m_j}(\hat r)\\
iQ_\kappa(r)\Omega_{-\kappa m_j}(\hat r)
\end{pmatrix}}.
\]

Each \(\Omega\) is a two-entry Pauli spinor.  The displayed object therefore
has four complex spinor components even though spherical symmetry reduces its
radial dependence to two scalar functions \(P\) and \(Q\).  “Four-component”
does **not** mean four independent radial arrays.

## 2. Hartree radial Dirac equations

For the shifted Hamiltonian and the phase above,

\[
\boxed{P_\kappa'=-\frac{\kappa}{r}P_\kappa
+\frac{\epsilon-V+2c^2}{c}Q_\kappa},
\]

\[
\boxed{Q_\kappa'=\frac{\kappa}{r}Q_\kappa
-\frac{\epsilon-V}{c}P_\kappa}.
\]

SPEX integrates the scaled small component \(q=cQ\).  In that representation,

\[
P'=-\frac\kappa rP+\left[2+\frac{\epsilon-V}{c^2}\right]q,
\qquad
q'=\frac\kappa rq+(V-\epsilon)P.
\]

These are the equations described at `src/dirac.f:638-657` and used by its
core backends.  A public API should accept \(\kappa\) directly; SPEX's legacy
negative-`l` sequence that encodes
\(-1,+1,-2,+2,\ldots\) is an adapter detail, not a new quantum-number
convention.

These two radial equations assume a central scalar potential.  A radial
spin-polarizing field can couple the two allowed \(\kappa\) channels at fixed
\((l,m_j)\), producing a fourfold coupled radial system rather than two
independent central-field systems.  FLEUR's `core/spratm.f` and
`core/coredir.f` implement that extension.  It is a separate equation variant;
an adapter must not flatten its coupled coefficient matrices into one
\((P,Q)\) pair.

For a Coulombic origin \(V\sim-Z/r\), the regular leading power is

\[
P,Q\propto r^\gamma,\qquad
\gamma=\sqrt{\kappa^2-(Z/c)^2}.
\]

The leading component ratio follows from either first-order equation and must
be initialized with the same phase convention as the spinor.  FLEUR derives
this origin expansion in `core/inconz.f:1-138`.

## 3. Normalization and boundary data

The physical norm on a radial interval is

\[
N[a,b]=\int_a^b(P^2+Q^2)dr.
\]

If the stored auxiliary component is \(q=cQ\), the same norm is

\[
N[a,b]=\int_a^b(P^2+q^2/c^2)dr.
\]

For a bound core state on an extended domain,

\[
N_{\rm total}=N[0,\infty]=1,
\quad N_{\rm MT}=N[0,R_{\rm MT}],
\quad N_{\rm outside}=N[R_{\rm MT},\infty]=1-N_{\rm MT}.
\]

Production data should retain both \(N_{\rm MT}\) and
\(N_{\rm outside}\), because a deep core is nearly confined but not
mathematically truncated at the muffin-tin radius.  If only the in-sphere
arrays are serialized, the converged total normalization and outside norm must
be explicit metadata.

The first-order Dirac boundary vector is minimally

\[
b_\kappa(R)=\begin{pmatrix}P_\kappa(R)\\Q_\kappa(R)\end{pmatrix},
\]

or equivalently the physical radial amplitudes \((P/R,Q/R)\).  The derivatives
are not independent:

\[
P'(R),Q'(R)
\]

follow from the two radial equations.  Component logarithmic derivatives may
be stored as diagnostics where their denominators are nonzero, but they do not
replace the two-component boundary vector.  In particular, the scalar LAPW
pair \((u,u')\) cannot specify a four-component Dirac match.

## 4. Bound-core shooting and matching

A bound core solution requires the spherical potential beyond
\(R_{\rm MT}\), far enough into the decaying tail.  A robust shooting method is:

1. Extend the spherical potential on the logarithmic mesh and find a classical
   turning point \(r_t\).
2. Integrate the regular Coulomb solution outward from the origin to \(r_t\).
3. Integrate the decaying solution inward from a remote outer point to
   \(r_t\).
4. Rescale one branch so the large components \(P_{\rm out}(r_t)\) and
   \(P_{\rm in}(r_t)\) agree.
5. Use the remaining small-component mismatch
   \(Q_{\rm in}(r_t)-Q_{\rm out}(r_t)\) as the eigenvalue residual.
6. Refine \(\epsilon\), splice both components with one common amplitude, and
   normalize over the entire extended mesh.

Continuity of only \(P\) is a scale choice; the zero of the remaining
\(Q\) mismatch supplies the eigenvalue condition.  SPEX's `corestate` follows
this procedure in `src/dirac.f:1565-1677`.  The low-level outward and inward
backends are active, although its public `core_dirac_hom_out` wrapper is marked
disabled; new code should call a supported library-owned interface instead of
copying that wrapper state.

FLEUR uses the same physical separation: `core/coredir.f:3-154` enumerates the
\(\kappa=-(l+1)\) and \(\kappa=+l\) channels, `core/cfnorm.f:37-98` matches and
normalizes large and small components, and `core/spratm.f:4-23` identifies the
fully relativistic core path.

## 5. Scalar-relativistic valence versus four-component Dirac

The three radial models must not be inferred from the number of returned
arrays:

| model | radial unknowns | angular object | channel label | intended role |
|---|---|---|---|---|
| Schrödinger | \(P\) | \(Y_{lm}\chi_s\) | \(l,s\) | analytic/reference valence |
| scalar Koelling--Harmon | \(P,Q\) | \(Y_{lm}\chi_s\) | \(l,s\) | v0.1 functional 2c valence |
| Dirac 4c | \(P_\kappa,Q_\kappa\) | \((\Omega_\kappa,\Omega_{-\kappa})\) | \(\kappa,m_j\) | core; future 4c valence |

Koelling--Harmon returns a “small” radial correction, but it has no explicit
\(j=l\pm1/2\) splitting and no spin-angular entanglement.  It is therefore a
functional two-component scalar-relativistic valence formulation.  A true 4c
valence implementation needs \(\kappa\)-resolved homogeneous solutions,
energy derivatives, relativistic envelope expansions, two-component boundary
matching, and compatible Hamiltonian/overlap assembly.

The v0.1 `DiracKappa` valence variant is schema-reserved only.  Its solver,
energy derivative, envelope match, and assembly are intentionally empty.
Calling it must return an explicit unsupported-feature error; silently falling
back to Koelling--Harmon would change the physics while preserving a misleading
type label.

## 6. Core and valence are separate schema decisions

One per-sphere `radial_eq` field is insufficient.  Production LAPW calculations
commonly use scalar-relativistic valence functions and fully relativistic core
states in the same sphere.  The snapshot therefore needs separate contracts,
conceptually

```text
sphere.valence_equation
sphere.valence_energies[]
sphere.core.equation
sphere.core.states[] = { n, kappa, energy, occupancy,
                         large, small, norm_mt, norm_outside, ... }
```

This is a physical split, not duplicated metadata.  FlapwMBPT makes the same
choice through distinct `irel` and `irel_core` controls; `cor_new.F:35-104`
enumerates core channels according to the latter.  FLEUR similarly routes
valence through `global/radsra.f` and fully relativistic core states through
its `core/` solvers.

## 7. Cross-code unit dialects

The equations above are the library/SPEX Hartree dialect with
\(c\simeq137.0359895\).  SPEX declares this value in `src/global.f:319-326`.

FlapwMBPT's radial source is in a Rydberg dialect.  Its `units_mod.F:10-60`
defines the Rydberg-to-eV constant, comments a light velocity near
\(274.074\), and its `radsch.F:448-558` uses that convention for the
`irel0=2` Dirac equations and a norm containing `q**2/c**2`.  Energies and
potentials from that source must be divided by two on import to a Hartree API,
and its scaled small component must be decoded according to the source
equations.  Copying the numerical \(c\), energy factors, or norm expression
piecemeal creates a factor-of-two error.

FLEUR's core routines also contain explicit Hartree/Rydberg conversion
boundaries; `core/spratm.f:4-23` warns that its atomic core solver operates in
the Rydberg convention internally.  Adapters convert at that boundary.  No
Rydberg factor belongs in the library equations or stored canonical fields.

## 8. Validation

1. Verify the \(\kappa\leftrightarrow(l,j)\) map for
   \(s_{1/2},p_{1/2},p_{3/2},d_{3/2},d_{5/2}\) and reject \(\kappa=0\).
2. Check the Coulomb origin exponent and hydrogenic Dirac energies for modest
   \(Z\), below the point-nucleus critical regime.
3. At a converged core eigenvalue, verify both component matches at \(r_t\),
   total norm one, and `norm_mt + norm_outside = 1`.
4. Check derivatives reconstructed from the equations against numerical
   derivatives of both radial components.
5. Test Hartree/SPEX and Rydberg/FlapwMBPT adapters against the same physical
   model after explicit conversion.
6. Assert that the reserved 4c valence path returns “unsupported” and never a
   scalar-relativistic solution.

## Source anchors

These anchors were inspected in the local source trees
`/Users/zerozaki07/Documents/dft_codes/spex06.00pre36`,
`/Users/zerozaki07/Documents/dft_codes/fleur`, and
`/Users/zerozaki07/Documents/dft_codes/ComDMFTv.2.0/src/FlapwMBPT`.

- SPEX \(\kappa\) convention and full Dirac equations: `spex06.00pre36/src/dirac.f:638-657,678-1007`.
- SPEX outward/inward core matching: `spex06.00pre36/src/dirac.f:1565-1677`.
- FLEUR fully relativistic core driver and channels: `fleur/core/spratm.f:4-23` and `fleur/core/coredir.f:3-154`.
- FLEUR origin and normalization: `fleur/core/inconz.f:1-138` and `fleur/core/cfnorm.f:37-98`.
- FLEUR scalar-relativistic valence reference: `fleur/global/radsra.f:1-140`.
- FlapwMBPT equation/unit controls: `FlapwMBPT/units_mod.F:10-60`, `FlapwMBPT/radsch.F:1-24,448-558`, and `FlapwMBPT/cor_new.F:35-104`.
