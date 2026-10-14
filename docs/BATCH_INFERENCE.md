# Matched-batch C inference

`ti_forward_batch` computes `ReLU(X W1 + b1) W2 + b2` for a contiguous row-major
matrix of images. The model layout and single-image API are unchanged. Counts
are total float elements. A batch must be positive. The caller supplies
`batch * hidden_dim` workspace floats and `batch * output_dim` output floats.
There is no implicit padding, stride support, softmax or allocation of C scratch.
Workspace/output must be disjoint from the full input and all weights; shared
models require separate workspaces per concurrent call. Pointer validity and
model immutability remain caller obligations. Failure may partially write buffers.

The handwritten backends execute row loops inside one C call and validate full
buffer ranges once. The system BLAS backend uses two row-major `cblas_sgemm`
operations, including bias broadcast. All batch/dimension products and BLAS
integer limits are checked before arithmetic. Inputs and layer outputs must be
finite; hidden values are checked before ReLU. Vendor BLAS internals may allocate.

## Reproduce

The optional digits-model environment is described in [real-model setup](REAL_MODEL.md).
From the repository root:

```sh
build/real-model-venv/bin/python tools/batch_model.py build --blas 1
build/real-model-venv/bin/python tests/test_batch_model.py build/real_engine.so
build/real-model-venv/bin/python -O tests/test_batch_model.py build/real_engine.so
build/real-model-venv/bin/python tests/test_batch_mutations.py --blas 1
build/real-model-venv/bin/python tools/batch_model.py benchmark --runs 6 --samples 20
build/real-model-venv/bin/python tools/verify_batch.py results/published/digits-batch-reviewed-20261014
```

Use a new output directory (`--output`) for each benchmark. Measurements require
committed source files, matching build metadata and a full Git history. Set
`build --blas 0` to exercise only the two handwritten C backends. macOS BLAS needs
macOS 15+ and Command Line Tools; Linux uses system OpenBLAS development files.
Default dependency-free C checks include `make test sanitize`, with BLAS optional.
No retraining or new train/test split is introduced.

## Equal workloads and timing scope

Every backend processes all the same 450 held-out float32 digit images in the
same order, using requested batch sizes **1, 8, 32, 128 and 450**. Every backend
uses identical partitions, including the last partial batch: for size 128 this
is 128 + 128 + 128 + 66. Size 8 ends with 2 images; size 32 ends with 2 images.
The last batch is neither dropped nor padded. Reported latency is the full-sweep
elapsed time divided by 450, including tails, rather than a full-batch-only rate.

C, NumPy and sklearn receive the same pre-created contiguous array views. Array
conversion/validation, input pointer creation, inference, row-wise stable softmax,
result collection and NaN-initialized result-array allocation are inside timing.
Before each timed sweep, C/NumPy scratch and C single-image output state are
poisoned with NaNs outside timing, so a missing tail cannot reuse warm answers. C and NumPy reuse
caller-owned scratch allocated before timing; sklearn's public estimator allocates
its own intermediates. C and NumPy share the same softmax function. NumPy uses
matrix-matrix dot with preallocated outputs, bias addition and in-place ReLU;
its input finite check is timed. C validates input/layer finiteness and aliasing
natively. sklearn retains its normal public API checks. These policies are made
explicit; equal input batches do not make the APIs' validation semantics identical.
Training, normalization, model setup, warm-up and reference calculation are excluded.

Two single-image controls are measured in the same processes:
`c_single_direct_input_first` writes each row directly into the shared sweep result
using pre-created `(1,64)` input views. Its normal array validation, ctypes pointer
creation, C call and softmax remain timed. It avoids the extra block wrapper.
`c_single_loop_input_first` retains the original block wrapper, including per-row
slicing, temporary block allocation and extra copying. The direct control is the
primary numerator for improvement claims; the wrapped control quantifies wrapper
cost. Both execute the same handwritten `input_first` kernel once per image.
Neither comparison isolates the ctypes crossing: repeated validation, Python
collection and matrix-shaped versus vector-shaped softmax differ too. SGEMM
additionally changes matrix-vector loops to matrix-matrix operations.

A baseline uses six fresh processes, 20 measured sweeps per API/batch size per
process. Five primary backends (three C, NumPy, sklearn) cycle through all 120
backend permutations across the six processes; both single-image diagnostics are
inserted with rotating positions. Batch-size order rotates too. Timing uses
`perf_counter_ns`. No files are written between timed blocks. A separate untimed audit intercepts calls to the loaded C bindings and verifies
actual row counts, full ordered inputs and input hashes through the same sweep
path. Its recorded observations are distinct from expected call counts in timed
rows; timing itself is not instrumented. Every timed output
is checked elementwise and for exact class agreement after timing. Numerical
threadpools and native C BLAS are set and queried at one thread. Providers are
recorded because NumPy/sklearn OpenBLAS and native Accelerate differ on this host.

Reported values are medians of process medians, with process ranges. They are not
individual-call tail latencies or confidence intervals. One small model, warm
inputs and one host do not establish universal superiority or cold-start latency.

## Correctness and evidence

- Native hand-derived multi-row results, bias/ReLU, guard regions, all capacities,
  invalid arguments, aliasing, overflow and nonfinite paths; ASan/UBSan exercise
  those same production sources with and without BLAS in CI.
- 45 independent random/non-square batch/shape cases compare the formal shared
  library against float64 `math.fsum` and the single-vector C API, including batch 1.
- All 450 real images at each selected size are checked against sklearn and the
  independent float64 logits, with `1e-5 + 3e-5*abs(reference)` tolerance and exact
  class agreement. Native C executable, Python tests and measurement share the
  same production object files. Native headers document the buffer contract.
- Seven compiled faults (stale unwritten row, batch ReLU, last row, row stride,
  second layer, GEMM bias and GEMM row count) must be numerically rejected by both synthetic and real-model
  checks under normal Python and `-O`; a crash/import failure does not count.
- Reports retain every saved test output, numerical errors, raw sweep timings,
  actual partitions and C call counts, providers, compiler commands and source/
  object/library hashes. Sources/build outputs are checked for changes during
  measurement. Verification regenerates references, checks historical source
  blobs and recomputes accuracy and timing summaries. Corruption fixtures alter
  last rows, tails, call counts, sample order, references and metadata even after
  file hashes are updated.

## Reviewed results

See [the batch review findings](BATCH_REVIEW.md).

Measured on Apple M5 Pro, source `1743574`, six processes and 20
sweeps per API/batch size/process. Values are microseconds/image, the median
of process medians; complete process ranges are retained in the manifest.

| Batch | C input_first | C BLAS | NumPy | sklearn | C direct single | C wrapped single |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.825 | 5.692 | 5.943 | 32.362 | 4.376 | 4.860 |
| 8 | 1.074 | 0.980 | 0.889 | 4.376 | 4.304 | 4.501 |
| 32 | 0.537 | 0.435 | 0.335 | 1.316 | 4.332 | 4.404 |
| 128 | 0.400 | 0.227 | 0.192 | 0.487 | 4.331 | 4.417 |
| 450 | 0.366 | 0.147 | 0.133 | 0.223 | 4.343 | 4.412 |

Batch 450 improves over the **direct** single-image control by **11.90×**
(median of six within-process ratios; process range 11.53–12.27×).
Using the wrapped control on this new run gives 12.11×. Ratios are computed
from paired process medians, rather than dividing rounded table values.
The historical 12.24× remains a measured wrapper comparison, not a pure
ctypes or native-kernel speedup. Batch 1 remains slower than direct single
calls; NumPy remains faster than native BLAS at the larger tested sizes.

All 450 images, including every 2-row and 66-row tail, still score 438/450
(97.33%) with exact class agreement across all backends. Native engine
object hashes match the original run byte-for-byte: the review changes the
validation/measurement harness, not the C numerical implementation.

[Reviewed manifest](../results/published/digits-batch-reviewed-20261014/manifest.json),
[all outputs](../results/published/digits-batch-reviewed-20261014/correctness.json)
and six worker reports retain observations, raw times and paired ratios.

## Original results (historical wrapper comparison)

Measured on Apple M5 Pro, source `32921e6`, with six fresh processes
and 20 sweeps per backend/batch size/process. Values below are microseconds
per image (median of process medians), including the common tail partitions.

| Requested batch | C input_first | C system BLAS | NumPy reused buffers | sklearn | C single loop |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.666 | 5.515 | 5.806 | 31.775 | 4.843 |
| 8 | 1.051 | 0.948 | 0.869 | 4.257 | 4.470 |
| 32 | 0.531 | 0.430 | 0.324 | 1.272 | 4.383 |
| 128 | 0.396 | 0.219 | 0.186 | 0.472 | 4.348 |
| 450 | 0.355 | 0.144 | 0.126 | 0.215 | 4.341 |

Under this historical protocol, the handwritten `input_first` calculation measured **12.24×**
at batch 450 relative to the block-wrapped single-image loop. This is not the
reviewed direct-control ratio above. These original buffers were not poisoned
between sweeps; new anti-stale verification must not be attributed to this run.
C forward calls fall from 450 per sweep to one. The improvement includes
amortized ctypes calls, slicing/collection and repeated validation; it is
not a measurement of ctypes cost alone. Batch 1 is slower than the single
adapter due to its matrix-shaped wrapper/softmax overhead. Batching is not
automatically beneficial for an isolated image.

Native SGEMM gains further matrix-matrix efficiency. NumPy remains faster
than C system BLAS at the larger tested sizes on this host. This does not
establish universal backend superiority; providers and validation differ.

All selected batch sizes and backends score **438/450 = 97.33%**, with 100%
class agreement with sklearn. Worst saved C/NumPy logit absolute error is
`1.32437828e-05`; worst saved probability absolute error is `1.01327896e-06`.
Elementwise combined tolerance passes; absolute maximum alone is not the acceptance rule.

[Manifest, full process ranges and summaries](../results/published/digits-batch-20261014/manifest.json),
[all numerical outputs](../results/published/digits-batch-20261014/correctness.json)
and six `run-*.json` files preserve the evidence. The strided `output_first`
backend is also recorded there, including regressions; it is not omitted from testing.
