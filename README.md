# Tiny Infer Lab

A small C inference engine and a reproducible CPU benchmark suite. It compares
loop layouts and native system BLAS across network sizes, with independent
numerical checks before accepting measurements.

The synthetic benchmark workload is a two-layer float32 network:

```text
hidden = ReLU(input × W1 + bias1)
output = hidden × W2 + bias2
```

The synthetic suite uses generated weights. A separate [trained digits classifier](docs/REAL_MODEL.md)
validates held-out accuracy and inference against scikit-learn. Performance conclusions
are scoped to the measured workload, hardware, provider, API and threading policy.

## Quick start

Requires a C11 compiler, Make, Python 3.9+ and Git for the test/publication tools. The default build has no external
numerical dependency. Build tools support macOS and Linux. Optional trained-model
tools use pinned numerical packages on Python 3.9–3.12; the optional provider
comparison uses Python 3.12. See their setup guides.

```sh
make demo
make test sanitize mutation build-check
make publication-check
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
| `system_blas` | Native single-thread CBLAS: `sgemv` for vectors, `sgemm` for batches; `BLAS=1` |

Every backend uses the same input validation, bias, ReLU and finite-result checks.
BLAS sees the same row-major buffers without application-side packing or conversion;
vendor internals remain opaque.
Thread configuration is set and queried before measurement. Vendor BLAS can use
FMA or a different accumulation order; results must meet the same elementwise
numerical tolerance, not bitwise equality.

The default suite covers **8→16→4, 31→17→9, 128→256→32, 256→512→64,
1024→1024→256, and 1024→64→1024**. Results are reported separately per shape;
a speedup on one shape is not advertised as a general engine speedup.

## Real-model validation

An optional 64→32→10 handwritten-digit classifier uses a fixed held-out split,
trained weights and scikit-learn as a reference, plus an independent float64 oracle.
It compares all enabled C backends on every test image and measures single-image
array-input APIs separately from sklearn batch throughput and a prepared NumPy
diagnostic. Training-isolation and deliberately broken C implementations challenge
the validation pipeline. See [setup, protocol and results](docs/REAL_MODEL.md).
The default engine build and synthetic tests still require no numerical Python packages.

## Batch inference

The C API also supports contiguous batches, with caller-owned workspace and a
system-BLAS SGEMM path. [Matched-batch validation and real measurements](docs/BATCH_INFERENCE.md)
compare batch sizes 1, 8, 32, 128 and 450 with NumPy and sklearn using the same
450 held-out images and tail partitions. Simultaneously measured direct and wrapped single-image C controls show how
call/validation and wrapper overhead affect batching. The [batch review](docs/BATCH_REVIEW.md)
adds anti-stale tail checks and untimed observations of actual native inputs/calls.

## Provider comparison

An optional [same-model comparison](docs/PROVIDER_COMPARISON.md) adds emlearn and
ONNX Runtime CPU alongside production C, NumPy and sklearn. It checks exact model
parameter bits and all held-out outputs, and measures matched batches, isolated
process RSS and deployment payloads separately. emlearn uses an explicit C-loop
batch adapter; ONNX Runtime uses native dynamic batches. Their API/allocation
differences and dependency-size exclusions are part of the reported scope.
On the measured host, all six backends retain **438/450 (97.33%)** accuracy;
emlearn is fastest at batch 1 and ONNX Runtime at batches 8, 32, 128 and 450.
At batch 450, ONNX Runtime measures **0.059 µs/prediction** versus C BLAS
**0.142 µs/prediction**, with median warm Python-process RSS **60.422 MiB** versus
**38.938 MiB**. These are Python-facing warm throughput and process-RSS figures,
not bare-kernel timing or model-only RAM. Reviewed RSS snapshots precede
validation, and deployment totals retain per-file inventories. The linked report shows every backend,
all batches, process ranges and deployment dependencies; no universal win is claimed.

## Reviewed batch-control measurements

The held-out digits model scores **438/450 (97.33%)** with complete class agreement
across the tested C backends and sklearn. On Apple M5 Pro, the batch-450 comparison
(source `1743574`) measures **11.90×** versus the direct single-image control,
with process ratios 11.53–12.27×. This is an end-to-end wrapper comparison that
includes validation, collection and softmax; it does not isolate the Python/C
crossing or native-kernel speed. NumPy remains faster than C BLAS at the larger
tested batch sizes. See [all reviewed and historical tables](docs/BATCH_INFERENCE.md).
The separate [single-image comparison](docs/REAL_MODEL.md) has its own API scope.

## Archived synthetic measurements

On the archived Apple M5 Pro baseline (source revision `2d95f10`), the loop-order gain ranges from **1.64× to 17.42×**
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
Archived reports are verified against their recorded Git source revision, not
silently relabeled as a measurement of the current checkout. Use a full Git clone
for `make publication-check`; a shallow clone must fetch the measured history.
The verifier rejects incomplete suites/manifests and regenerates reference
answers before checking all saved measurements and statistics.

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
- Publication regressions reject empty reports, omitted process runs, forged
  reference outputs, inconsistent metadata and missing raw-data hashes.

The handwritten engine performs no heap allocation; caller-owned weights and
workspace must remain valid and supplied counts must be truthful. Raw pointer
validity cannot be established by this C API. We check known allocator references
in the handwritten engine object.
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
docs/             Measurement protocol and review evidence
models/           Trained JSON weights, split provenance and dataset attribution
results/published/ Selected raw measurements and provenance
```

The [CI matrix](https://github.com/andrew29mo/tiny-infer-lab/actions/workflows/ci.yml)
checks macOS and Linux with BLAS enabled and disabled, plus optional trained-model
and batch validation with BLAS, plus the optional six-provider comparison. The documented tables and README numeric claims
are checked against archived reports; numerical/provenance verifiers validate
those reports separately. CI timing is not a performance gate.

## Project status

See the [pre-publication review and corrected findings](docs/PREPUBLICATION_REVIEW.md)
and the [real-model](docs/REAL_MODEL_REVIEW.md) and [batch](docs/BATCH_REVIEW.md) reviews.
The [provider-comparison review](docs/PROVIDER_REVIEW.md) documents graph identity,
actual input observations, phase-separated RSS and file-level deployment evidence.

This is an experimental engineering project for public review. See
[contribution guidance](CONTRIBUTING.md).
The publication guard scans tracked files and reachable Git history for known
credential/key patterns, personal home paths, local-only files and binaries;
it is a targeted check, not an exhaustive security or secret audit.

## License

The project code and documentation are licensed under the [MIT License](LICENSE).
Optional system BLAS libraries retain their own licenses; the MIT License does
not relicense those dependencies. The real-model dataset retains its
[CC BY 4.0 terms and attribution](models/digits/README.md). Selecting MIT does not change repository visibility.
