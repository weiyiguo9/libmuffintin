"""Per-atom real-space grid budgets for all-electron THC/ISDF:
LAPW-style MT-adaptive grid vs plane-wave FFT grids, for Sm/Dy fcc & bcc.

Assumptions (stated, order-of-magnitude honest):
- atomic volumes: Sm 33.2 A^3, Dy 31.5 A^3 (metallic RE)
- R_MT = 2.6 bohr (typical rare-earth), r0 = 1e-5 bohr (heavy-element mesh start)
- MT radial: exponential mesh, n = ln(R_MT/r0)/h; h = 0.10 (moderate ~1e-3),
  h = 0.05 (tight ~1e-5), h = 0.016 (solver-grade FLEUR/WIEN2k ~780 pts)
- angular: Lebedev 110 (exact to l=17) / 194 (exact to l=23); pair densities of
  lmax ~ 8 augmentation reach L ~ 16-20
- interstitial: FFT grid at LAPW density cutoff Gmax = 12 bohr^-1, keep points
  outside the sphere
- PP reference is ONCVPSP norm-conserving (no augmentation charge):
  Gmax_rho = 2*sqrt(ecutwfc[Ry]) i.e. ecutrho = 4*ecutwfc;
  f-in-valence lanthanide hints per PseudoDojo (Lu: 46/50/58 Ha ~ 92-116 Ry),
  4f-frozen "3+" potentials ~50 Ry for contrast
- AE-grade uniform FFT: spacing needed to resolve the 1s product e^{-2Zr}:
  dx = c/(2Z), c = 1 (loose) or 0.5 (strict)
"""
import numpy as np

BOHR3 = 1.0 / 0.529177**3          # A^3 -> bohr^3
# ONCV f-in-valence ecutwfc hints (Ry) from PseudoDojo/pseudodojo_experiments
# (Sm-4f.djrepo: 58/61/63 Ha; Dy-4f.djrepo: 66/68/71 Ha)
CASES = [("Sm", 62, 33.2, (116, 122, 126)), ("Dy", 66, 31.5, (132, 136, 142))]
RMT, R0, GMAX_I = 2.6, 1e-5, 12.0

def report():
    for el, Z, va, hints in CASES:
        V = va * BOHR3
        afcc, abcc = (4 * V) ** (1 / 3), (2 * V) ** (1 / 3)
        sph = 4 * np.pi / 3 * RMT**3
        inter = V * (GMAX_I / np.pi) ** 3 * (1 - sph / V)
        print(f"\n=== {el} (Z={Z})  V/atom = {V:.1f} bohr^3 "
              f"(a_fcc={afcc*0.529177:.2f} A, a_bcc={abcc*0.529177:.2f} A, "
              f"nn_fcc={afcc/np.sqrt(2):.2f} bohr) ===")
        print(f"  interstitial FFT part (Gmax={GMAX_I}/bohr): {inter:,.0f} pts/atom")
        for tag, h, nang in [("moderate ~1e-3", 0.10, 110),
                             ("tight    ~1e-5", 0.05, 194),
                             ("solver-grade  ", 0.016, 194)]:
            nrad = int(np.log(RMT / R0) / h) + 1
            tot = nrad * nang + inter
            print(f"  MT-adaptive [{tag}] nrad={nrad:4d} x nang={nang:3d} "
                  f"+ inter = {tot:9,.0f} pts/atom")
        for tag, ecw in [("4f-frozen 3+  ", 50), ("f-val low     ", hints[0]),
                         ("f-val normal  ", hints[1]), ("f-val high    ", hints[2])]:
            n = V * (2 * np.sqrt(ecw) / np.pi) ** 3
            print(f"  ONCV FFT [{tag}] ecutwfc={ecw:3d} Ry "
                  f"(rho={4*ecw} Ry): {n:10,.0f} pts/atom")
        for c in (1.0, 0.5):
            dx = c / (2 * Z)
            n = V / dx**3
            print(f"  AE-grade uniform FFT dx={dx*1000:.1f} mbohr (c={c}): "
                  f"{n:14,.2e} pts/atom")
        # headline ratios (tight adaptive as the reference)
        ad = (int(np.log(RMT / R0) / 0.05) + 1) * 194 + inter
        pp = V * (2 * np.sqrt(hints[1]) / np.pi) ** 3
        ae = V * (2 * Z) ** 3
        print(f"  -> adaptive(tight)/ONCV-FFT(f-val normal) = {ad/pp:.2f}   "
              f"AE-uniform(c=1)/adaptive(tight) = {ae/ad:,.0f}x")

report()
