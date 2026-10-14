"""Deterministic benchmark inputs and native BLAS build configuration."""
import ctypes as C
import itertools
import shlex
import sys

KERNELS = ("output_first", "input_first", "system_blas")
ORDERS = tuple(itertools.permutations(KERNELS))
DEFAULT_SHAPES = ((8, 16, 4), (31, 17, 9), (128, 256, 32),
                  (256, 512, 64), (1024, 1024, 256), (1024, 64, 1024))


def shape_name(shape):
    return "x".join(map(str, shape))


def parse_shapes(text):
    shapes = []
    for item in text.split(","):
        dims = tuple(map(int, item.split("x")))
        if len(dims) != 3 or any(d < 1 or d > 4096 for d in dims):
            raise ValueError("Each shape must be INPUTxHIDDENxOUTPUT; dimensions 1..4096")
        if dims in shapes:
            raise ValueError("Duplicate shape")
        shapes.append(dims)
    return shapes


def arrays(shape):
    state = 20261014
    def array(n):
        nonlocal state
        values = []
        for _ in range(n):
            state = (state * 1664525 + 1013904223) & 0xffffffff
            values.append(((state >> 8) / 16777216.0 - 0.5) * 0.25)
        return (C.c_float * n)(*values)
    ni, nh, no = shape
    return array(ni), array(ni * nh), array(nh), array(nh * no), array(no)


def native_flags(blas):
    if not blas:
        return [], []
    if sys.platform == "darwin":
        return ["-DTI_ENABLE_BLAS", "-DACCELERATE_NEW_LAPACK"], ["-framework", "Accelerate"]
    if sys.platform.startswith("linux"):
        return ["-DTI_ENABLE_BLAS", "-DTI_BLAS_OPENBLAS"], ["-lopenblas"]
    raise RuntimeError("System BLAS is supported on macOS (Accelerate) and Linux (OpenBLAS)")


def compile_commands(cc, cflags, blas):
    cc, flags = shlex.split(cc), ["-Iinclude"] + shlex.split(cflags)
    defines, libs = native_flags(blas)
    flags += defines
    objects = ["build/measured_engine.o", "build/measured_blas_backend.o"]
    commands = [cc + flags + ["-fPIC", "-c", source, "-o", obj]
                for source, obj in zip(("src/tiny_infer.c", "src/blas_backend.c"), objects)]
    commands += [cc + flags + ["tools/benchmark.c"] + objects + libs + ["-lm", "-o", "build/measured_benchmark"],
                 cc + ["-shared"] + objects + libs + ["-lm", "-o", "build/measured_engine.so"],
                 cc + flags + ["tests/test_engine.c"] + objects + libs + ["-lm", "-o", "build/measured_test"]]
    return commands, objects
