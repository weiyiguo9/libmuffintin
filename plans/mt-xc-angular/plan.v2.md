# Production MT XC Lebedev repair

- Workstream ID: `mt-xc-angular`
- Plan version: 2
- Approval: implementation and one focused verification authorized 2026-10-21
- Supersedes: plan.v1 implementation and acceptance scope; historical material validation remains deferred

Use positive-weight Lebedev rules through order 131. Preserve AngularGrid's
positive-weight contract; exclude orders 13, 25 and 27. The degree is distinct
from node count. Explicit unsupported orders fail. Automatic selection uses
the first supported degree at least max(29, 2L); 29 means 302 points. Degree
29 is a bounded-regression default, not a material-independent accuracy claim.
Regional full/block XC and core continuation share the selected degree.
Expose optional angular order through native SCF, Python and both TOML inputs.
Do not alter atomic-start, molecular sampling, or THC grids.

G-MT-XC-1 compares the default against fixed order 59 on a manufactured,
positive cubic density with monopole sqrt(4pi)*(0.3+0.02*r^2) and l=4
coefficients 0.01*r^4 times (1, sqrt(5/14), sqrt(5/14)) for m=(0,4,-4).
Nine radial nodes span 0.5 to 1.42 bohr. Check LDA-PW92 and PBE, all MT
coefficients, cubic forbidden coefficients and allowed-component relation,
both energy contractions, and complete versus blocked paths. The fixture is
not the missing historical diamond checkpoint. Core normalization and
continuation tests retain their existing bounds. Run the focused acceptance
set once after integration; at most three smallest diagnostic computations
after a failure, then rerun only affected failed checks after genuine fixes.
No production SCF, eigenvalue-degeneracy or SPEX acceptance is claimed.
