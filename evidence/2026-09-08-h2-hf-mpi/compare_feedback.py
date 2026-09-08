"""Compare all ordered band-feedback entries against the rank-one dump."""

import math
import struct
import sys


reference_path, candidate_path = sys.argv[1:]
bound = 1e-12
maximum = 0.0
location = None
entries = 0
with open(reference_path, "rb") as reference, open(candidate_path, "rb") as candidate:
    header = reference.read(8)
    assert header == candidate.read(8), "block counts differ"
    blocks, = struct.unpack("<Q", header)
    for block in range(blocks):
        shape = reference.read(16)
        assert shape == candidate.read(16), "block index or dimension differs"
        k, dimension = struct.unpack("<QQ", shape)
        for entry in range(dimension * dimension):
            old = complex(*struct.unpack("<dd", reference.read(16)))
            new = complex(*struct.unpack("<dd", candidate.read(16)))
            assert all(math.isfinite(x) for x in (old.real, old.imag, new.real, new.imag))
            delta = abs(new - old)
            if delta > maximum:
                maximum = delta
                location = (k, entry // dimension, entry % dimension)
            entries += 1
    assert reference.read(1) == candidate.read(1) == b"", "trailing bytes"
print(f"blocks={blocks} entries={entries} maximum={maximum:.17e} location={location}")
print(f"class=R ref=n1 bound={bound:.17e}")
print("DIGIT / PASS" if maximum <= bound else "DIGIT / HANDOFF")
sys.exit(0 if maximum <= bound else 1)
