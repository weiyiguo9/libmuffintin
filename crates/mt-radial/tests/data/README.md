# Future FLEUR radial golden fixture

No FLEUR numbers are committed in M-B because no provenance-complete printout
was supplied.  A future fixture belongs in this directory only together with:

- the FLEUR version/commit and input file;
- the physical spherical potential convention (`v_00 / sqrt(4 pi)`), in Ha;
- exact `(r0, h, N)`, `l`, equation, and energy parameters;
- printed `u(R)`, `du/dr(R)`, `du/dE(R)`, `d(du/dE)/dr(R)`, and the
  post-orthogonalization derivative norm;
- any LO energies plus value/slope residuals; and
- an explicit component convention (`p = r u`, physical small `Q`).

The eventual integration test must parse that fixture, run `RadialSolver`, and
compare each printed quantity independently.  It must not copy undocumented
values into Rust literals.  Until such a provenance bundle exists, the
analytic hydrogenic, square-well, derivative, and LO tests are the M-B gates.
