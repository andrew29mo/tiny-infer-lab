"""Prove the oracle rejects two deliberately broken production engines."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--cc", default="cc")
parser.add_argument("--blas", type=int, default=0)
args = parser.parse_args()
sys.path.insert(0, str(ROOT / "tools"))
from workloads import native_flags
defines, libs = native_flags(args.blas)
source = (ROOT / "src/tiny_infer.c").read_text()
mutations = {
    "remove_ReLU": ("if (h[i] < 0) h[i] = 0;", "if (h[i] < 0) h[i] = h[i];"),
    "wrong_bias": ("float sum = b[j];", "float sum = b[j] + 1.0f;"),
    "wrong_input_first_bias": ("y[j] = b[j];", "y[j] = b[j] + 1.0f;"),
    "skip_second_layer": ("dense(h, m->w2, m->b2, y, hid, out, kernel);",
                          "for (size_t i = 0; i < out; ++i) y[i] = 0;"),
}
for name, (before, after) in mutations.items():
    if source.count(before) != 1:
        raise SystemExit("Mutation target changed: " + name)
    with tempfile.TemporaryDirectory(prefix="mutation-", dir=ROOT / "build") as temp:
        path = Path(temp)
        broken, lib = path / "broken.c", path / "broken.so"
        broken.write_text(source.replace(before, after))
        subprocess.run(shlex.split(args.cc) + ["-std=c11", "-O3", "-ffp-contract=off", "-Iinclude",
                        "-Isrc", "-fPIC", "-shared"] + defines + [str(broken), "src/blas_backend.c"] + libs + ["-o", str(lib)], cwd=ROOT, check=True)
        for mode in ([], ["-O"]):
            result = subprocess.run([sys.executable] + mode + ["tests/test_reference.py", str(lib)],
                                    cwd=ROOT, capture_output=True, text=True)
            # A crash/import failure is not evidence of a numerical rejection.
            if result.returncode == 0 or "AssertionError" not in result.stderr:
                raise SystemExit("FAIL: oracle did not numerically reject " + name + "\n" + result.stderr)
        print("PASS: normal and -O oracle reject mutation " + name)

if args.blas:
    with tempfile.TemporaryDirectory(prefix="blas-mutation-", dir=ROOT / "build") as temp:
        path = Path(temp)
        source = (ROOT / "src/blas_backend.c").read_text()
        target = "1.0f, w, (int)out, x, 1, 1.0f, y, 1"
        if source.count(target) != 1:
            raise SystemExit("BLAS mutation target changed")
        broken, lib = path / "blas.c", path / "blas.so"
        broken.write_text(source.replace(target, "0.5f, w, (int)out, x, 1, 1.0f, y, 1"))
        subprocess.run(shlex.split(args.cc) + ["-std=c11", "-O3", "-Iinclude", "-Isrc", "-fPIC", "-shared"]
                       + defines + ["src/tiny_infer.c", str(broken)] + libs + ["-o", str(lib)], cwd=ROOT, check=True)
        for mode in ([], ["-O"]):
            result = subprocess.run([sys.executable] + mode + ["tests/test_reference.py", str(lib)], cwd=ROOT, capture_output=True, text=True)
            if result.returncode == 0 or "AssertionError" not in result.stderr:
                raise SystemExit("BLAS numerical fault was not detected")
        print("PASS: normal and -O oracle reject wrong BLAS alpha")
