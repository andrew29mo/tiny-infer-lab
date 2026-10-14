# Benchmark protocol

This suite compares `output_first`, `input_first`, and optional system CBLAS for
one vector through two dense layers with ReLU. It is an educational CPU
experiment, not a trained model or a comparison with complete ML frameworks.

## Matched work

Every backend calls the same `ti_forward` validation, bias handling, ReLU and
finite-result checks. BLAS uses `cblas_sgemv` with a transpose view of the same
row-major weights: `y = W^T x + bias`, with `beta=1` after copying the bias. It
uses no application-level preprocessing, weight packing or alternate weight layout;
vendor BLAS internals remain opaque. Matrix/vector
multiplication is appropriate because batch size is one.

All C kernels use the same compilation settings. Vendor BLAS is a precompiled
system library and can use FMA, reassociation and hardware-specific instructions;
this is a numerical-tolerance comparison, not bit-for-bit equivalence. Do not
attribute the entire difference to a particular cache or SIMD mechanism without
additional profiling.

The library configures and queries single-thread BLAS before timing:

- macOS 15+: Apple Accelerate `BLASSetThreading` / `BLASGetThreading`.
- Linux: system OpenBLAS `openblas_set_num_threads` / `openblas_get_num_threads`.

The run fails if the requested BLAS or single-thread setting is unavailable.
There is no silent fallback to the handwritten kernels. A confirmed threading
policy is not a trace of every internal vendor instruction or helper activity.
The handwritten engine allocates no heap; no equivalent claim is made about
vendor BLAS internals, especially during first use.

## Workloads

The predefined shapes are:

| Input | Hidden | Output | Purpose |
| ---: | ---: | ---: | --- |
| 8 | 16 | 4 | Tiny workload / call overhead |
| 31 | 17 | 9 | Non-square, non-aligned dimensions |
| 128 | 256 | 32 | Original MVP comparison |
| 256 | 512 | 64 | Medium dense workload |
| 1024 | 1024 | 256 | Larger weight footprint |
| 1024 | 64 | 1024 | Narrow hidden layer, wide output |

No cross-shape aggregate is used to hide shape-dependent results. All are batch
size 1 with fixed synthetic float32 data and seed 20261014. The independent
Python oracle receives the exact float32 inputs and accumulates with float64
`math.fsum`. Acceptance is elementwise `abs(error) <= 1e-5 + 3e-5*abs(reference)`.
This bound is a tested acceptance criterion, not a proof for all possible inputs.

## Measurement

1. Compile the production engine and BLAS adapter once. Link those exact objects
   into C tests, the shared library, and the measured executable. Run C tests,
   normal/optimized-Python reference tests and the allocator-reference check.
2. Validate each selected shape and backend against the independent oracle.
3. Execute six process rounds by default. Rotate shape order between rounds.
   Within each shape/process, rotate all backend permutations between samples;
   also rotate the initial permutation between processes. For three backends,
   use a sample count divisible by six for a complete order balance.
4. Warm each backend 32 times before and after calibration. Use the maximum
   calibrated repetition count for every backend in that matched group. The
   fastest of three trial batches must reach 2 ms, capped at 65,536 repetitions.
   Explicit repetition counts bypass calibration; too-short batches can fail
   because the clock cannot resolve their duration.
5. Time full inference calls using `CLOCK_MONOTONIC`. Consume a rotating output
   element in a checksum after each call. Record the complete final output of
   every batch outside timing. Defer CSV output until all timing is complete.
6. Validate every recorded final output and checksum from the measured executable
   against the independent oracle. This does not inspect every element of every
   internal call. Reject missing/reordered samples, unequal counts, invalid
   durations, inconsistent reported timings and numerical mismatches.
7. Retain per-process CSV, combined per-shape CSV, source/object/executable hashes,
   compiler commands, provider linkage, queried thread policy, reference outputs,
   per-process statistics and order-specific matched ratios.

Median/p95/p99 summarize **batch-average latency**, not individual-call tail
latency. Ratios of pooled medians differ from medians of matched ratios. A ratio
`input_first / system_blas > 1` means BLAS is faster. Process min/max ranges are
not confidence intervals. Samples within a process are not independent machines.

Repeated warm data, uncontrolled frequency/scheduling, shared-host noise, selected
shapes and the fixed input distribution limit generalization. Cache-cold models,
training, multi-vector batches, quantization, GPU/NPU and real-time deadlines are
outside this suite. CI runs correctness and a smoke test; its timings are not
used as performance acceptance thresholds.

## Primary references

- [Apple BLAS documentation](https://developer.apple.com/documentation/accelerate/blas-library)
- [Apple BLAS single-thread policy](https://developer.apple.com/documentation/accelerate/blas_threading_single_threaded)
- [OpenBLAS project](https://github.com/OpenMathLib/OpenBLAS)
