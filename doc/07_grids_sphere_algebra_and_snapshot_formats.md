# 07. Composite grids, sphere algebra, and versioned artifacts

This note fixes the M-C contracts that sit between the radial primitives and
later LAPW/THC consumers.  It introduces no new electronic-structure theory:
it makes the existing radial and angular conventions composable and
serializable.

## 1. Units and point ordering

Every real-space point is stored in Cartesian Bohr and every quadrature weight
in Bohr cubed.  A grid is an ordered sequence

```math
  \mathcal G = \{(\mathbf r_p,w_p,\rho_p)\}_{p=0}^{N-1},
```

where $\rho_p$ is a region tag.  Ordering is part of the artifact contract:

1. an atom grid is radial-shell major and angular-point minor;
2. a uniform grid is lexicographic in fractional $(i,j,k)$, with $k$
   fastest;
3. a composite grid contains atom grids in stable atom-index order, followed
   by the interstitial grid.

The stable order lets a THC interpolation-point index and a serialized grid
refer to the same physical point without coordinate matching.

## 2. Atom and interstitial quadrature

For the exponential radial mesh $r_i=r_0e^{ih}$, `libmuffintin-core` supplies SPEX
`intgr_init` weights $w_i^{(r)}$ for integration over $dr$.  Crossing it
with an angular rule $(\hat{\mathbf r}_a,w_a^{(\Omega)})$, normalized by
$\sum_a w_a^{(\Omega)}=4\pi$, gives

```math
 \mathbf r_{ia}=\mathbf R+r_i\hat{\mathbf r}_a,
 \qquad
 w_{ia}=w_i^{(r)}r_i^2w_a^{(\Omega)}.
```

Production callers may supply a Lebedev rule.  The built-in Fibonacci rule is
a deterministic fallback for tests and exploratory grids; it is not assigned
a polynomial exactness order.

The uniform cell grid uses the midpoint rule in fractional coordinates.  Its
interstitial subset drops a point when its periodic nearest-image distance to
any muffin-tin centre is smaller than that sphere's radius.  The nearest-image
search is performed in the full direct-lattice metric, rather than by rounding
three Cartesian components, so skew cells remain valid.  The retained midpoint
weights converge to the analytic interstitial volume represented by
`muffintin_core::InterstitialGeometry`; boundary correction is a measured later
optimization, not part of the M-C contract.

## 3. Sphere fields and orbitals

A sphere field is an explicit harmonic expansion

```math
 F(\mathbf r)=\sum_{LM}F_{LM}(r)Y_{LM}(\hat{\mathbf r})
```

or its real-tesseral equivalent.  The harmonic convention is mandatory.  With
normalized harmonics, a physical constant $f$ is stored as
$F_{00}=\sqrt{4\pi}f$, not as $f$.

A sphere orbital carries an angular label $(l,m)$ and reduced radial
components $p=r u$ and optional physical small component $Q$.  Therefore
the field matrix element is

```math
 \langle l_1m_1|F|l_2m_2\rangle
 =\sum_{LM}\mathcal G^{LM}_{l_1m_1,l_2m_2}
 \int dr\,[p_1p_2+Q_1Q_2]F_{LM}(r).
```

`libmuffintin-sphere` evaluates the angular factor with the `libmuffintin-core` complex or real
Gaunt convention and the radial factor with `libmuffintin-radial::radial_integral`.
This is the common primitive for non-spherical muffin-tin potentials, density
synthesis, and later Coulomb multipoles.  SPEX performs the same separation in
`src/hamilton.f:461-488`.

## 4. Snapshot and grid artifacts are distinct

`SnapshotV1` is a human-diffable physical-input artifact.  It records:

- producer and energy-zero provenance;
- direct lattice, sites, nuclear charges, and muffin-tin radii;
- per-site/per-spin exponential meshes, radial-equation tags, potential
  channels, linearization energies, and local-orbital energies;
- cell-normalized interstitial Fourier coefficients and their phase contract.

Its TOML header is

```toml
format = "libmuffintin-snapshot"
version = 1
```

`GridArtifactV1` is a derived numerical artifact containing ordered Cartesian
points, weights, and region tags.  It has an independent header:

```toml
format = "libmuffintin-grid-artifact"
version = 1
```

The two version numbers must not be coupled.  Changing a point-selection or
quadrature representation is not a change to the physical snapshot schema.
Both readers reject unknown fields, unsupported versions, inconsistent array
lengths, invalid harmonic channels, and non-finite numerical samples before
the data enters numerical kernels.

The FLEUR converter remains frozen.  These formats describe what the library
consumes; no producer-specific parser is part of M-C.

## 5. Optional tensor boundary

The canonical grid storage is the typed Rust point sequence above.  With the
`libmuffintin-grid/rstsr` feature, callers may materialize positions as an $(N,3)$
RSTSR tensor and weights as a vector of length $N$.  This conversion is an optional
consumer boundary only: `libmuffintin-core`, `libmuffintin-radial`, `libmuffintin-sphere`, and `libmuffintin-io` do not
depend on a tensor implementation.  LAPW dense linear algebra uses `faer`
directly.  A later tensor backend can consequently replace RSTSR without
changing the physical or grid schemas.

## 6. M-C acceptance

The focused acceptance suite checks:

- normalized and deterministic angular rules;
- Gaussian and Slater radial integrals on an off-origin atom grid;
- periodic sphere removal and interstitial-volume convergence;
- stable composite ordering;
- complex/real low-order sphere matrix elements, selection rules, and
  Hermiticity;
- independent snapshot and grid-artifact TOML round trips and version errors;
- default builds without RSTSR and feature-enabled tensor conversion.
