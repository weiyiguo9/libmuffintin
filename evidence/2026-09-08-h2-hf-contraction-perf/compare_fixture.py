"""Report the fixed fixture energy and identity checks for one implementation stage."""

import math
from pathlib import Path
import re
import sys


directory = Path(__file__).parent
stage, = sys.argv[1:]
passed = True
for feature in ("fftw", "direct"):
    old = (directory / f"fixture-before-{feature}.log").read_text()
    new = (directory / f"fixture-{stage}-{feature}.log").read_text()
    pattern = r"CONTRACT_FIXTURE total=(\S+) exchange=(\S+)"
    reference = [float(x) for x in re.search(pattern, old).groups()]
    candidate = [float(x) for x in re.search(pattern, new).groups()]
    identities = [float(value) for row in re.findall(
        r"CONTRACT_ID exchange=(\S+) eigenvalue=(\S+) total=(\S+)", new
    ) for value in row]
    assert identities and all(math.isfinite(x) for x in reference + candidate + identities)
    differences = [abs(a - b) for a, b in zip(candidate, reference)]
    success = max(differences) <= 1e-10 and max(identities) <= 1e-8
    passed &= success
    print(f"stage={stage} feature={feature} class=R ref=10f6b39 "
          f"total_delta={differences[0]:.17e} exchange_delta={differences[1]:.17e} bound=1e-10 Ha")
    print(f"identity_max={max(identities):.17e} ref=fixture bound=1e-8 Ha")
    print("DIGIT / PASS" if success else "DIGIT / HANDOFF")
sys.exit(0 if passed else 1)
