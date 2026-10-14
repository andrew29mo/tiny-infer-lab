"""Check the production engine object's unresolved symbols, not a test copy."""
import re
import subprocess
import sys

symbols = subprocess.check_output(["nm", "-u", sys.argv[1]], text=True)
for name in ("malloc", "calloc", "realloc", "free", "aligned_alloc", "posix_memalign"):
    if re.search(r"\b_?" + name + r"\b", symbols):
        raise SystemExit("FAIL: engine refers to heap allocator " + name)
print("PASS: production engine object has no heap allocator references")
