# Cu frozen-fixture contract

This directory is reserved for the future Cu cross-code acceptance fixture.
It is not currently a complete reference fixture and does not establish the
v0.1 tag gate.

## Required fixture contents

A valid fixture must contain, as one provenance-linked and immutable set:

- the frozen all-electron potential for every represented spin channel,
  including sphere meshes/channels and interstitial Fourier coefficients;
- the complete basis specification, including lattice and sites, muffin-tin
  radii, cutoffs, linearization energies, local-orbital definitions, angular
  cutoffs, units, and conventions;
- the exact k path and k-point ordering;
- reference eigenvalues indexed by k point and band, with the compared energy
  window and energy-zero convention;
- provenance identifying the producer code and version, input/calculation
  identity, export procedure, and any conversion steps;
- cryptographic hashes for every fixture file and, where applicable, the
  original source artifacts.

The acceptance report must compare matching `(k_index, band_index)` entries
and pass only when the maximum absolute residual in the declared window is no
larger than `1 meV`. Missing or empty reference data is an error: an empty
reference must never produce a vacuous pass.

## Current evidence and release status

Only a Cu eigenvalue artifact has been located so far. No corresponding SPEX
run input and no matching frozen potential/basis snapshot have been found.
Eigenvalues alone cannot validate reconstruction of the Hamiltonian, so they
must not be installed as a nominal reference fixture or used to claim the
`1 meV` gate.

The overlay band plot may be generated only after the complete fixture above
passes provenance and hash checks and the numerical gate. Until then, do not
create placeholder reference files, synthetic cross-code data, or a fake
overlay plot. The Cu acceptance condition for the v0.1 tag is therefore
currently unmet.
