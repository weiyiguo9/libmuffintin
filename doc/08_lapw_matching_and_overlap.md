# 08. LAPW boundary matching and overlap assembly

This note fixes the LAPW matching and overlap conventions against SPEX
`src/hamilton.f:105-123,248-264,938-954`.  The matching contract stops at the overlap matrix;
Hamiltonian blocks and eigensolving are introduced separately in the Hamiltonian eigensolver note.
Crate ownership after the anonymous-basis facade split is recorded in
[12](12_anonymous_basis_and_lapw_facade.md).

## 1. Plane waves

The normalized interstitial basis function is

```math
 \chi_{\mathbf G\mathbf k}(\mathbf r)
 =\Omega^{-1/2}e^{i(\mathbf k+\mathbf G)\cdot\mathbf r},
 \qquad \mathbf q=\mathbf k+\mathbf G.
```

Direct and reciprocal vectors retain the `libmuffintin-core` convention
$\mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij}$.  `PlaneWave` stores `k`, the
integer-labelled reciprocal vector `G`, Cartesian `q`, and $|q|$; no code may
reconstruct a Cartesian cutoff from integer components.

## 2. Boundary solve

For each site type, spin channel, and angular momentum, define the unscaled
linearization pair $u_l,\dot u_l=\partial u_l/\partial E$ and its boundary
matrix

```math
 M_l=\begin{pmatrix}
 u_l(R)&\dot u_l(R)\\
 u_l'(R)&\dot u_l'(R)
 \end{pmatrix}.
```

The real APW coefficients are the direct $2 \times 2$ solve

```math
 \begin{pmatrix}A_l(q)\\B_l(q)\end{pmatrix}
 =M_l^{-1}
 \begin{pmatrix}j_l(qR)\\qj_l'(qR)\end{pmatrix}.
```

The implementation substitutes $(A,B)$ back into the original matrix and
reports separate value and slope residuals.  This follows SPEX exactly except
that libmuffintin stores the physical, unnormalized $\dot u$; SPEX normalizes
that column internally and compensates its stored `apw(2)` by `dotnorm`.

## 3. Angular coefficient and site phase

The Rayleigh identity is

```math
 e^{i\mathbf q\cdot\mathbf r}
 =4\pi\sum_{lm}i^l j_l(qr)
 Y_{lm}^{*}(\hat{\mathbf q})Y_{lm}(\hat{\mathbf r}).
```

Accordingly, the angular coefficient for the normalized plane wave is

```math
 C_{lm}(\mathbf q)=\frac{4\pi}{\sqrt\Omega}
 i^lY_{lm}^{*}(\hat{\mathbf q}).
```

Expansion about site $\mathbf R_a$ supplies the separate phase
$e^{i\mathbf q\cdot\mathbf R_a}$.  Keeping this phase separate makes the
Hermitian pair phase
$e^{i(\mathbf G_j-\mathbf G_i)\cdot\mathbf R_a}$ explicit after the common
Bloch vector cancels.  At $q=0$, the deterministic angular convention keeps
only $Y_{00}$; the Bessel boundary target eliminates every $l>0$ channel.

## 4. Overlap matrix

Let $c^a_{i,lm,\alpha}$ be the complete angular, phase, and radial matching
coefficient for plane wave $i$, with $\alpha\in\{u,\dot u\}$.  Let

```math
 O^a_{l,\alpha\beta}
 =\int_0^{R_a}dr\,[p_{l\alpha}p_{l\beta}
 +Q_{l\alpha}Q_{l\beta}].
```

The dense complex overlap is

```math
 S_{ij}=\Theta_I(\mathbf G_i-\mathbf G_j)
 +\sum_{a,l,m,\alpha,\beta}
 (c^a_{i,lm,\alpha})^{*}
 O^a_{l,\alpha\beta}
 c^a_{j,lm,\beta}.
```

$\Theta_I$ is the cell-normalized interstitial coefficient from `libmuffintin-core`, so
the first term is exactly SPEX `cstep`.  The upper triangle is evaluated and
the lower triangle filled by conjugation.  Local orbitals do not enter the matching contract.

## 5. Focused acceptance

The matching contract requires:

- value and slope matching residuals no larger than `1e-10`;
- Hermitian dense $S$ for translated spheres and nonzero $k$;
- an empty-sphere cell yielding the exact plane-wave identity overlap;
- explicit rejection of mixed-k matrices, missing site blocks, and inconsistent
  `lm` counts. `RadialOverlapBlock` stores only the three independent entries,
  so asymmetry is unrepresentable by construction.

The empty-sphere result is the overlap half of the empty-lattice regression.
Free-electron eigenvalues require the Hamiltonian eigensolver kinetic matrix and are therefore not
claimed by the matching contract alone.
