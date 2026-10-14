"""Exercise the real shared library against an independent float64 oracle.

Python uses math.fsum and column dot products. C inputs are rounded to float32
before reference evaluation. No NumPy, model downloads or third-party packages.
"""
import ctypes as C
import math
import random
import sys
from pathlib import Path

F = C.c_float
P = C.POINTER(F)
S = C.c_size_t


class Model(C.Structure):
    _fields_ = [("input_dim", S), ("hidden_dim", S), ("output_dim", S),
                ("w1", P), ("b1", P), ("w2", P), ("b2", P),
                ("initialized", C.c_uint32)]


def load(path):
    lib = C.CDLL(str(Path(path).resolve()))
    lib.ti_model_init.argtypes = [C.POINTER(Model), S, S, S, P, S, P, S, P, S, P, S]
    lib.ti_model_init.restype = C.c_int
    lib.ti_forward.argtypes = [C.POINTER(Model), P, S, P, S, P, S, C.c_int]
    lib.ti_forward.restype = C.c_int
    lib.ti_blas_available.restype = C.c_int
    lib.ti_blas_configure.restype = C.c_int
    lib.ti_blas_is_single_threaded.restype = C.c_int
    lib.ti_blas_name.restype = C.c_char_p
    if lib.ti_blas_available():
        if lib.ti_blas_configure() != 0 or not lib.ti_blas_is_single_threaded():
            raise RuntimeError("Cannot confirm single-thread BLAS")
    return lib


def oracle(x, w1, b1, w2, b2, hidden, output):
    h = [max(0.0, math.fsum([b1[j]] + [x[i] * w1[i * hidden + j]
                                     for i in range(len(x))])) for j in range(hidden)]
    return [math.fsum([b2[j]] + [h[i] * w2[i * output + j]
                                for i in range(hidden)]) for j in range(output)]


def check(condition, detail):
    if not condition:
        raise AssertionError(detail)


def run(path):
    lib = load(path)
    rng = random.Random(20261014)
    shapes = [(1, 1, 1), (2, 3, 2), (7, 5, 3), (31, 17, 9),
              (128, 256, 32), (129, 7, 65)]
    shapes += [(rng.randint(1, 40), rng.randint(1, 40), rng.randint(1, 25))
               for _ in range(100)]
    kernels = range(3 if lib.ti_blas_available() else 2)
    worst = 0.0
    for case, (ni, nh, no) in enumerate(shapes):
        def array(n):
            return (F * n)(*(rng.uniform(-0.5, 0.5) for _ in range(n)))
        x, w1, b1, w2, b2 = array(ni), array(ni * nh), array(nh), array(nh * no), array(no)
        expected = oracle(x, w1, b1, w2, b2, nh, no)
        m = Model()
        check(lib.ti_model_init(C.byref(m), ni, nh, no, w1, len(w1), b1, nh,
                               w2, len(w2), b2, no) == 0, "model initialization failed")
        for kernel in kernels:
            hidden, actual = (F * nh)(), (F * no)()
            check(lib.ti_forward(C.byref(m), x, ni, hidden, nh, actual, no, kernel) == 0,
                  "forward failed")
            for j, (a, e) in enumerate(zip(actual, expected)):
                error = abs(a - e)
                worst = max(worst, error)
                check(math.isfinite(a) and error <= 1e-5 + 3e-5 * abs(e),
                      (case, kernel, j, a, e, error))
    print("PASS: %d shapes x %d kernels against float64 oracle; worst absolute error %.9g"
          % (len(shapes), len(kernels), worst))


if __name__ == "__main__":
    run(sys.argv[1])
