# 10. APW+lo basis layout and collinear spin channels

This note fixes the M-F basis layout and assembly contract against SPEX
`spex06.00pre36/src/hamilton.f`. It extends the APW-only conventions in
[08](08_lapw_matching_and_overlap.md) and
[09](09_hamiltonian_eigensolver_and_reference_reports.md); it does not add
self-consistency or spin-orbit coupling.

## 1. Global basis order

At a fixed k point and in one spin channel, the global basis is

```math
 \mathcal B_\sigma=
 [\,\chi_{\mathbf G_1},\ldots,\chi_{\mathbf G_{N_{\rm PW}}}\,]
 [\,\Phi^{\rm lo}_{1},\ldots,\Phi^{\rm lo}_{N_{\rm LO}}\,].
```

Thus every plane wave precedes every local orbital: `[PW][site LO]`. The LO
suffix is deterministic. Sites are traversed in stored site order; within one
site, $l$ increases, then $m=-l,\ldots,l$, and finally the local radial
index $n=3,\ldots,n_l$ increases fastest. Equivalently, the site-local order
is $(l,m,n)$, with $n$ contiguous inside a fixed $(l,m)$ block. The APW
radial indices $n=1,2$ denote $u_l,\dot u_l$ and are augmentation
coordinates, not additional global basis columns.

SPEX states the global `[APW][LO]` order at `hamilton.f:801-804`. Its LO count
and site offsets are formed at `hamilton.f:1212-1223`; the explicit
$l$, $m$, $n$ indexing is visible in the APW--LO and LO--LO assembly at
`hamilton.f:1331-1393,1473-1510`.

## 2. Local-orbital boundary condition

For each site, spin, $l$, and $n\ge3$, construct

```math
 \Phi^{\rm lo}_{lmn}
 =Y_{lm}\bigl(\phi_{ln}+a_{ln}u_l+b_{ln}\dot u_l\bigr)
```

with

```math
 \Phi^{\rm lo}_{lmn}(R)=0,
 \qquad
 \partial_r\Phi^{\rm lo}_{lmn}(R)=0.
```

Both equations are required. They make an LO strictly confined to its
muffin-tin sphere, so it has no interstitial tail and needs no plane-wave
matching column. SPEX forms the coefficients from the APW boundary inverse and
applies the same combination to both scalar-relativistic radial components at
`hamilton.f:372-392`. The resulting APW--APW, APW--LO, and LO--LO radial blocks
are named `hmt1`, `hmt2`, and `hmt3`; their meanings and normalized-
$\dot u$ convention are recorded at `hamilton.f:286-295`, and their
spherical assembly is shown at `hamilton.f:396-459`.

## 3. One projection rule for overlap and Hamiltonian

For site $a$ and spin $\sigma$, collect its transformed radial-angular
coordinates in the order above and define a coefficient matrix
$P_{a\sigma}$ from global basis columns to those site coordinates:

- a PW column contains its two APW augmentation coefficients in every
  applicable $(l,m)$ block and zero in LO coordinates;
- an LO column belonging to site $a$ selects its corresponding transformed
  LO coordinate;
- an LO column belonging to another site is zero.

Let $`S^{a\sigma}_{\mathrm{MT}}`$ and $`H^{a\sigma}_{\mathrm{MT}}`$ be the full site
matrices in those same coordinates, including all APW--APW, APW--LO, and
LO--LO blocks. Then both operators obey the same congruence:

```math
 S_\sigma=S^I+\sum_a P_{a\sigma}^{\dagger}
 S^{a\sigma}_{\rm MT}P_{a\sigma},
 \qquad
 H_\sigma=H^I_\sigma+\sum_a P_{a\sigma}^{\dagger}
 H^{a\sigma}_{\rm MT}P_{a\sigma}.
```

The interstitial matrices $S^I$ and $H^I_\sigma$ occupy only the PW--PW
corner because every LO has zero value and slope at the sphere boundary. This
single $P^\dagger X P$ rule is normative; separately coded block formulas
must be equivalent to it and must not reorder one operator differently from
the other. The numerical evaluation of that congruence is the local tensor
substrate in [11](11_tensorized_numerical_substrate.md). SPEX builds the
interstitial PW corner at `hamilton.f:938-999`, the APW--APW projection at
`hamilton.f:1023-1207`, the APW--LO blocks at `hamilton.f:1212-1393`, and
the LO--LO blocks at `hamilton.f:1397-1510`.

## 4. Collinear spin without SOC

For collinear spin polarization with SOC disabled, spin is a channel label,
not a coupled basis index. Construct the radial functions, LO transforms,
augmentation coefficients, overlap, and Hamiltonian separately for
$\sigma=\uparrow,\downarrow$, using that channel's frozen potential:

```math
 H_\sigma C_\sigma=S_\sigma C_\sigma\varepsilon_\sigma,
 \qquad \sigma\in\{\uparrow,\downarrow\}.
```

There are no $H_{\uparrow\downarrow}$ or
$S_{\uparrow\downarrow}$ blocks and no SU(2) rotation. The two generalized
Hermitian problems are solved and reported independently; sharing geometry or
a PW enumeration does not permit reusing spin-dependent radial blocks. SPEX's
spin-index selection is explicit at `hamilton.f:345-356,899-910`; the
off-diagonal channels and doubled spin basis shown later in that source belong
to the SOC/noncollinear path and are outside this contract.

## 5. Scope and acceptance

M-F acceptance requires deterministic `[PW][site LO]` indexing, zero LO value
and slope at every sphere boundary, Hermitian $S$ and $H$ from the same
projection layout, and independent residual checks for both collinear spin
channels. This milestone consumes a frozen potential and fixed basis. It does
not perform SCF, density/potential updates, mixing, SOC, noncollinear spin, or
spinor assembly.
