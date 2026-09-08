"""Extract completed Gamma Fock phase timings, without rerunning any calculation."""

import collections
import re
import sys


path, count = sys.argv[1], int(sys.argv[2])
iterations = []
current = None
with open(path) as source:
    for line in source:
        if line.strip() == "[hf timing] begin gamma.fock.iteration":
            current = collections.defaultdict(float)
        match = re.search(r"\[hf timing\] end (\S+) elapsed_s=(\S+)", line)
        if match and current is not None:
            phase, seconds = match.groups()
            current[phase] += float(seconds)
            if phase == "gamma.fock.iteration":
                iterations.append(current)
                current = None

for index, phases in enumerate(iterations[:count], 1):
    total = phases["gamma.fock.iteration"]
    print(f"### Fock iteration {index}\n")
    print("| Phase | Seconds | Share of iteration |")
    print("|---|---:|---:|")
    for phase, seconds in sorted(phases.items(), key=lambda pair: -pair[1]):
        print(f"| `{phase}` | {seconds:.6f} | {100 * seconds / total:.3f}% |")
    compile_seconds = phases["gamma.rebuild.mpb"] - phases["vv.mpb_rebuild"]
    cache_seconds = compile_seconds + phases["gamma.rebuild.coulomb_assembly"]
    print(f"\nBasis compile by subtraction: {compile_seconds:.6f} s; "
          f"compile + Coulomb: {cache_seconds:.6f} s ({100 * cache_seconds / total:.3f}%).\n")
print(f"Completed iterations in log: {len(iterations)}; requested: {count}.")
if current is not None:
    print("Final iteration incomplete; its completed subphases have no full-iteration denominator.")
    for phase, seconds in current.items():
        print(f"{phase}: {seconds:.6f} s")
