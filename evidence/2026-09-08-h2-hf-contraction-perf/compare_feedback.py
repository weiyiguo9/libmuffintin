"""Compare every entry of ordered first-rebuild band-space feedback blocks."""

import math
import struct
import sys


reference_path, candidate_path = sys.argv[1:]
bound = 1e-10
maximum = 0.0
location = None
entries = 0
with open(reference_path, "rb") as reference, open(candidate_path, "rb") as candidate:
    header = reference.read(8)
    assert header == candidate.read(8), "k-block counts differ"
    blocks, = struct.unpack("<Q", header)
    for block in range(blocks):
        shape = reference.read(16)
        assert shape == candidate.read(16), "k-block index or dimension differs"
        k, dimension = struct.unpack("<QQ", shape)
        for entry in range(dimension * dimension):
            old = complex(*struct.unpack("<dd", reference.read(16)))
            new = complex(*struct.unpack("<dd", candidate.read(16)))
            assert all(math.isfinite(x) for x in (old.real, old.imag, new.real, new.imag))
            difference = abs(new - old)
            if difference > maximum:
                maximum = difference
                location = (k, entry // dimension, entry % dimension)
            entries += 1
    assert reference.read(1) == candidate.read(1) == b"", "trailing dump bytes"
print(f"blocks={blocks} entries={entries} maximum={maximum:.17e} location={location}")
print(f"class=R ref=10f6b39 bound={bound:.17e}")
print("DIGIT / PASS" if maximum <= bound else "DIGIT / HANDOFF")
sys.exit(0 if maximum <= bound else 1)
