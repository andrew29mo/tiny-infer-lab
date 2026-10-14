# Tiny Infer Lab

A small C inference engine and a reproducible CPU benchmark suite. It compares
loop layouts and native system BLAS across network sizes, with independent
numerical checks before accepting measurements.

The workload is a synthetic two-layer float32 network:

```text
hidden = ReLU(input × W1 + bias1)
output = hidden × W2 + bias2
```

This is an engineering experiment, not a trained AI model. Performance conclusions
are scoped to the measured shapes, hardware, provider and threading policy.

## Quick start

Requires a C11 compiler, Make and Python 3.9+. The default build has no external
numerical dependency. Build tools support macOS and Linux.

```sh
make demo
make test sanitize mutation build-check
make bench
```

The demo accepts two optional finite input values:

```sh
./build/demo -2 3
```

For a three-backend comparison on macOS 15+ using Apple Accelerate:

```sh
make BLAS=1 test sanitize mutation
make BLAS=1 bench
```

On Debian/Ubuntu, install `libopenblas-dev` before the same BLAS commands. Linux
builds explicitly link system OpenBLAS; they do not silently select an arbitrary
BLAS provider. No Python numerical package or model download is required.

## What is compared

| Backend | Implementation |
| --- | --- |
| `output_first` | Accumulate each output column through the inputs |
| `input_first` | Visit each input row and update contiguous outputs |
| `system_blas` | Native single-thread CBLAS `sgemv`, enabled with `BLAS=1` |

Every backend uses the same input validation, bias, ReLU and finite-result checks.
BLAS sees the same row-major weight buffers without packing or conversion.
Thread configuration is set and queried before measurement. Vendor BLAS can use
FMA or a different accumulation order; results must meet the same elementwise
numerical tolerance, not bitwise equality.

The default suite covers **8→16→4, 31→17→9, 128→256→32, 256→512→64,
1024→1024→256, and 1024→64→1024**. Results are reported separately per shape;
a speedup on one shape is not advertised as a general engine speedup.

## Measured result

On the Apple M5 Pro baseline, the loop-order gain ranges from **1.64× to 17.42×**
across six shapes; it is **5.50×** on the original 128→256→32 case. System BLAS
is faster than input_first in **five of six** cases, including **1.84×** on the
original case. The tiny 8→16→4 case slightly favors input_first.

The approximately 5.5× gain is a comparison between two handwritten loop layouts,
not evidence of outperforming optimized BLAS. See the [per-shape result table,
process ranges and raw evidence](docs/RESULTS.md).

## Evidence and reproducibility

Selected, publication-ready measurements live under [`results/published/`](results/published/).
Local experiments and historical review logs are excluded from Git. To run a suite:

```sh
python3 tools/run_benchmark.py --blas 1 --runs 6 --samples 30
```

For a shorter smoke test or custom dimensions:

```sh
python3 tools/run_benchmark.py --blas 1 --runs 1 --samples 6 \
  --repetitions 1024 --shapes 8x16x4,31x17x9
```

Each run retains source/object/binary hashes, compiler/link commands, OS/CPU,
queried BLAS thread policy, exact reference outputs, raw CSV and per-process
statistics. The C tests, reference-test shared library and benchmark link the
**same production objects**. The runner rejects source changes during a run.

Six process rounds and all six backend permutations reduce fixed order bias;
network order rotates too. All backends perform the same repetition count per
shape/process. No CSV output occurs between measured batches. The runner checks
every batch's full final output and accumulated checksum against the independent
Python float64 oracle.

Median/p95/p99 describe **batch-average latency**, not individual-call tail latency.
Warm repeated data, scheduler/frequency variation and the selected shapes limit
generalization. Process ranges are not confidence intervals. CI timings are smoke
checks, never speed gates. See the [complete protocol](docs/BENCHMARKS.md).

## Correctness checks

- Hand-derived answers, non-square matrices, ReLU boundaries and buffer guards.
- Invalid arguments, capacities, overlapping buffers, dimension overflow,
  NaN/Inf and non-finite arithmetic results.
- 106 deterministic/random shapes per enabled backend against an independent
  float64 `math.fsum` oracle, under both normal Python and `python -O`.
- A separate oracle check for every selected benchmark shape, including the
  measured executable's outputs. Acceptance: `abs(error) <= 1e-5 + 3e-5*abs(reference)`.
- ASan/UBSan on production C sources; vendor BLAS internals are not instrumented.
- Numerical mutation tests: missing ReLU, wrong bias in either C kernel, missing
  second layer, and wrong BLAS scaling when enabled.
- Report corruption tests, including errors that cancel in the checksum, all
  three-backend permutations, and compiler-option rebuild regression.

The handwritten engine performs no heap allocation; caller-owned weights and
workspace must remain valid. We check known allocator references in its object.
Vendor BLAS may allocate or cache internally; its first use is not presented as
allocation-free. The library's public buffer and concurrency contracts are in
[`include/tiny_infer.h`](include/tiny_infer.h).

## Layout

```text
include/          Public C API
src/              Engine and optional BLAS adapter
examples/         Runnable inference demo
tests/            Production-object, reference, mutation and reporting checks
tools/            Deterministic workloads, timed executable and suite runner
docs/             Measurement protocol
results/published/ Selected raw measurements and provenance
```

The [CI matrix](https://github.com/andrew29mo/tiny-infer-lab/actions/workflows/ci.yml)
checks macOS and Linux with BLAS enabled and disabled.

## Project status

This repository is being prepared for public review and currently has no selected
open-source license. Do not assume a license grant. See [contribution guidance](CONTRIBUTING.md).
