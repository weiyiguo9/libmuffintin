# 13. Product-space IR and the SPEX mixed product basis

This note records the M-H public contract. Toy k-point ISDF/THC is [14](14_toy_kpoint_isdf_thc.md).
The finite-$q$ Weinert/SPEX Coulomb operator is [15](15_weinert_coulomb_metric.md).
It does not add SCF, an umbrella crate, or a
distributed tensor runtime. The one-particle facade remains
[12](12_anonymous_basis_and_lapw_facade.md).
Tensor axes remain [11](11_tensorized_numerical_substrate.md). SPEX Gaunt
and the exponential mesh remain [02](02_angular_reciprocal_and_step_function.md)
and [03](03_exponential_mesh_and_radial_quadrature.md).

## 1. Packages

| Directory | Package |
|---|---|
| `crates/libmuffintin-product` | `libmuffintin-product` |
| `crates/libmuffintin-mpb` | `libmuffintin-mpb` |
| `crates/libmuffintin-thc` | `libmuffintin-thc` (M-I; interpolation-point payload) |

There is no `libmuffintin-mbp` alias. `libmuffintin-product` and
`libmuffintin-mpb` do not depend on `libmuffintin-lapw`, THC, or Coulomb.
`libmuffintin-product` does not own `CompiledBasis`.

## 2. Dependency DAG

```text
product  → core, radial, basis
mpb      → product, operators, core, radial, basis, envelope
thc      → product, core, basis, faer
coulomb  → product, core, basis, grid
         (dev: mpb, thc)
```

`libmuffintin-product` owns no solver. Channel overlap diagonalization uses
`libmuffintin-operators::solve_real_symmetric`.

## 3. Product IR

The specification is historical-method-name-free. v0.2 implements one
non-overlapping muffin-tin plus interstitial partition.

```text
ProductSource
  partition, radials, q, interstitial_pair_support, provenance
  # no CompiledBasis / APW P / envelope G
  # interstitial_pair_support is finite raw orbital-pair G labels

RawProductSpace                         # before TOL
  radial_products / coupled (L,M,n)
  overlap spectra
  interstitial_pair_support: raw pair-G, copied from the source

CompiledAuxiliaryBasis
  AuxiliaryRepresentation::MixedProduct
    per-site mesh + retained MT modes
    AuxiliaryInterstitialSupport: MPB |q+G| ≤ g_cut
    optional CutoffRecord
  AuxiliaryRepresentation::InterpolationPoints
    real-space interpolation points (M-I THC)
    muffin-tin tagged points, then interstitial/uniform

PairVertex
  OrbitalPair identity (MT, interstitial G, or composite of both)
  combined coefficients: MT then interstitial
```

Muffin-tin radial-factor identifiers (`ProductRadialId`) enumerate raw MPB
products. [`PairVertex`] carries a representation-neutral [`OrbitalPair`].
A dual-arm [`PairVertexSpec`] keeps both identities; it does not drop the
interstitial label or invent a left/right plane-wave pair from one arm.

An analytic [`InterstitialPairSpec`] names a component of the raw pair
support and expands it as
`amplitude * Θ_I(G_{\mathrm{rel}}+G_{\mathrm{wrap}}-G_{\mathrm{aux}})`.
A G label absent from that support is an error, even if it happens to lie
in the MPB auxiliary $|q+G|$ set. A muffin-tin-only spec leaves interstitial
coefficients zero because no interstitial expansion was requested. A muffin-tin
pair absent from the raw products is an error, not an all-zero MT block.

`TransferQ::fold_by_reciprocal_vector` stores $q_{\mathrm{in}}-G$ and the
Umklapp vector $G$. Site phases use $\exp(+i q\cdot R_a)$.

Raw muffin-tin products keep radial samples

```math
b(r)=\frac{p_i(r)p_j(r)+Q_i(r)Q_j(r)}{r}
```

after one-particle-norm scaling. Coupled channels and the compiled muffin-tin
flatten are $site \to L \to M=-L,\ldots,L \to n$.

A consumer can integrate retained muffin-tin modes using only
`CompiledAuxiliaryBasis` site meshes.

### 3.1 Raw pair support versus MPB auxiliary $g_{\mathrm{cut}}$

These are distinct objects:

- **Raw interstitial orbital-pair reciprocal support** is a finite list of
  relative reciprocal labels supplied by the one-particle/pair capability
  (`ProductSource` / `RawProductSpace`). The current convention path uses a
  deterministic list of those labels. It is not an enumeration of every
  reciprocal vector, and it is not filtered by MPB `product_g_max`.
- **MPB auxiliary interstitial plane-wave support** is constructed in
  `libmuffintin-mpb` from the lattice, canonical $q$, and `product_g_max`
  by the SPEX membership test $|q+G|\le g_{\mathrm{cut}}$. That set is stored
  only on `CompiledAuxiliaryBasis`.

The $g_{\mathrm{cut}}$-limited auxiliary PW set is not “untruncated raw
products.” Pair-vertex context matching compares complete
[`ProductPartition`] objects, including [`InterstitialGeometry`] and cell
volume, exact raw pair-support identity and order, and exact auxiliary-wave
kinematics, labels, $|q+G|$ values, cutoff, and SPEX $|G|$ then $G$-index
order. Same counts with a different volume or permuted G labels are rejected.

This milestone uses convention fixtures only. There is no live SPEX
untruncated dump of orbital-pair reciprocal support.

## 4. SPEX mixed-product constructor

`libmuffintin-mpb` follows `mixedbasis.f` (SPEX 06.00pre36):

1. unordered valence–valence pairs and selected core–valence pairs;
2. triangle $|l_1-l_2|\le L\le l_1+l_2$ and parity $L+l_1+l_2$ even
   (`mixedbasis.f:335–337`);
3. product convention `/r` (`mixedbasis.f:344–346`);
4. $L=0$ orthogonalization to $r/\sqrt{R^3/3}$ (`mixedbasis.f:432–437`);
5. real-symmetric overlap diagonalization of the untruncated channel
   (`mixedbasis.f:446–455`);
6. optional `TOL` with default $10^{-4}$ (`mixedbasis.f:106`), applied only
   after spectra exist, retaining $\lambda \ge \mathrm{TOL}\,n_{\mathrm{spin}}$
   because SPEX drops `eig < tolerance*nspin` (`mixedbasis.f:463`);
7. Löwdin transform and $L=0$ constant prepend (`mixedbasis.f:469–477`);
8. interstitial auxiliary membership $|q+G|\le g_{\mathrm{cut}}$
   (`mixedbasis.f:247–249`), ordered by $|G|$ then integer index, constructed
   independently of the raw pair-G list.

`write_mixedbasis` is post-`TOL`. Untruncated overlap spectra are therefore
not recovered from a `spex.mb` file. This milestone does not check in a live
SPEX numerical dump; focused tests lock the encoded conventions. Eigenvector
signs are not a SPEX convention; retained spans are compared as projectors
(the $L=0$ constant is excluded from that comparison).

Finite-$q$ kinematics (canonical $q$, Umklapp, site phase, $|q+G|$
auxiliary completeness, $\Theta_I$ pair vertices) belong to M-H. The $1/r$
Coulomb kernel and Weinert $V^q$ belong to [15](15_weinert_coulomb_metric.md).

## 5. Acceptance

M-H requires:

- untruncated muffin-tin product counts and channel spectra independent of
  `TOL`;
- raw interstitial pair support intact when the MPB auxiliary $g_{\mathrm{cut}}$
  filter drops a label that the capability still supplied;
- retained $L=0$ constant function and an independently recomputed
  nonzero-cutoff projector on the retained (non-constant) span;
- valence–valence plus selected core–valence provenance;
- $|q+G|$ auxiliary completeness and a nonzero analytic interstitial
  pair vertex whose every coefficient matches
  $\Theta_I(G_{\mathrm{raw}}+G_{\mathrm{wrap}}-G_{\mathrm{aux}})$, including
  Umklapp;
- muffin-tin flatten $site\to L\to M\to n$ with $L>0$ and at least two $n$;
- complete context matching (partition including cell volume, exact wave
  identity/order, mesh/mode lengths, absent MT pair as an error);
- no `product`/`mpb` $\to$ `lapw` or Coulomb/THC edge.
