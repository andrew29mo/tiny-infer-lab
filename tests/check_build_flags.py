"""Regression for same-timestamp stale objects after a flag change."""
import argparse
import hashlib
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--cc", default="cc")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
base = "-std=c11 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off"


def build(flags):
    subprocess.run(["make", "-s", "build/tiny_infer.o", "CC=" + args.cc, "CFLAGS=" + flags],
                   cwd=root, check=True)
    return hashlib.sha256((root / "build/tiny_infer.o").read_bytes()).hexdigest()


try:
    fast = build(base + " -O3")
    slow = build(base + " -O0")
    restored = build(base + " -O3")
    if fast == slow or fast != restored:
        raise AssertionError("Changing compiler flags did not rebuild the correct engine object")
finally:
    # Restore the Makefile's default flags, even if the regression fails.
    subprocess.run(["make", "-s", "build/tiny_infer.o", "CC=" + args.cc], cwd=root, check=True)
print("PASS: rapid -O3/-O0/-O3 flag changes rebuild the production object")
