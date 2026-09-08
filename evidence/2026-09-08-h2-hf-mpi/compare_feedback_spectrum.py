"""Gauge-invariant rank-independence check of a first-rebuild band-feedback dump.

The band-space feedback is defined only up to unitary rotations inside
degenerate band subspaces (Kramers pairs and cubic-box degeneracies), and the
eigensolver's gauge inside those subspaces is not reproducible across thread
or rank counts. The spectrum of the Hermitian feedback per k block is
gauge-invariant, so it is the quantity compared here.
"""

import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as handle:
        data = handle.read()
    (blocks,) = struct.unpack("<Q", data[:8])
    offset = 8
    matrices = []
    for _ in range(blocks):
        k, dimension = struct.unpack("<QQ", data[offset : offset + 16])
        offset += 16
        count = dimension * dimension
        values = np.frombuffer(data[offset : offset + 16 * count], dtype=np.complex128)
        offset += 16 * count
        matrices.append((k, values.reshape(dimension, dimension)))
    assert offset == len(data), "trailing bytes"
    return matrices


reference_path, candidate_path = sys.argv[1:]
bound = 1e-12
maximum = 0.0
for (k_old, old), (k_new, new) in zip(load(reference_path), load(candidate_path), strict=True):
    assert k_old == k_new and old.shape == new.shape, "block index or dimension differs"
    assert np.isfinite(old).all() and np.isfinite(new).all()
    assert abs(old - old.conj().T).max() < 1e-10 and abs(new - new.conj().T).max() < 1e-10
    delta = abs(np.linalg.eigvalsh(old) - np.linalg.eigvalsh(new)).max()
    maximum = max(maximum, delta)
    print(f"k={k_old} dimension={old.shape[0]} max_eigenvalue_delta={delta:.17e}")
print(f"maximum={maximum:.17e} class=R ref=n1 bound={bound:.17e}")
print("DIGIT / PASS" if maximum <= bound else "DIGIT / HANDOFF")
sys.exit(0 if maximum <= bound else 1)
