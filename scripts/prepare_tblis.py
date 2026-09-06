"""Fetch the pinned TBLIS wrapper and apply the checked-in soft patch."""

import json
from pathlib import Path
import subprocess


def git(*args, cwd):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main():
    root = Path(__file__).resolve().parents[1]
    pin = json.loads((root / "dependencies/tblis.json").read_text())
    patch = root / pin["patch"]
    checkout = root / ".local-deps/tblis-rs"
    exclude = Path(git("rev-parse", "--git-path", "info/exclude", cwd=root))
    if not exclude.is_absolute():
        exclude = root / exclude
    with exclude.open("a+", encoding="utf-8") as stream:
        stream.seek(0)
        if "/.local-deps/" not in stream.read().splitlines():
            stream.write("\n/.local-deps/\n")

    if checkout.exists():
        if git("rev-parse", "HEAD", cwd=checkout) != pin["commit"]:
            raise RuntimeError("TBLIS pin changed; move the old checkout aside first")
        git("apply", "--reverse", "--check", str(patch), cwd=checkout)
    else:
        checkout.mkdir(parents=True)
        git("init", cwd=checkout)
        git("fetch", "--depth=1", pin["repository"], pin["commit"], cwd=checkout)
        git("checkout", "--detach", pin["commit"], cwd=checkout)
        git("apply", "--check", str(patch), cwd=checkout)
        git("apply", str(patch), cwd=checkout)
    print(f"Prepared tblis at {pin['commit']} with {pin['patch']}")


if __name__ == "__main__":
    main()
