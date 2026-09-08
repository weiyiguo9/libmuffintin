"""Extract the first complete Gamma Fock iteration from a timing log."""

from pathlib import Path
import re
import sys


lines = Path(sys.argv[1]).read_text().splitlines()
phases = {}
active = False
for line in lines:
    if line == "[hf timing] begin gamma.fock.iteration":
        active = True
    match = re.search(r"\[hf timing\] end (\S+) elapsed_s=(\S+)", line)
    if active and match:
        phase, seconds = match.groups()
        phases[phase] = phases.get(phase, 0.0) + float(seconds)
        if phase == "gamma.fock.iteration":
            break
for phase in ("gamma.rebuild.contraction", "gamma.rebuild.mpb", "gamma.fock.iteration"):
    print(f"{phase}={phases[phase]:.6f}")
print(f"first_iteration_wall={phases['gamma.fock.iteration']:.6f}")
