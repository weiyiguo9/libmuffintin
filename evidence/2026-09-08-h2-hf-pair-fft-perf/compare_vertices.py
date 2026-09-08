"""Compare every complex vertex and its selection label in two scratch dumps."""

import math
import struct
import sys


reference_path, candidate_path, bound_text = sys.argv[1:]
bound = float(bound_text)
maximum = 0.0
maximum_location = None
with open(reference_path, "rb") as reference, open(candidate_path, "rb") as candidate:
    header = reference.read(16)
    assert header == candidate.read(16), "vertex dimensions differ"
    selections, width = struct.unpack("<QQ", header)
    for selection in range(selections):
        label = reference.read(24)
        assert label == candidate.read(24), f"selection order differs at {selection}"
        for column in range(width):
            old = complex(*struct.unpack("<dd", reference.read(16)))
            new = complex(*struct.unpack("<dd", candidate.read(16)))
            assert all(math.isfinite(x) for x in (old.real, old.imag, new.real, new.imag))
            delta = abs(new - old)
            if delta > maximum:
                maximum = delta
                maximum_location = (selection, column)
    assert reference.read(1) == candidate.read(1) == b"", "trailing dump data"
print(f"selections={selections} n_pw={width} entries={selections * width}")
print(f"maximum={maximum:.17e} bound={bound:.17e} location={maximum_location}")
print("DIGIT / PASS" if maximum <= bound else "DIGIT / HANDOFF")
sys.exit(0 if maximum <= bound else 1)
