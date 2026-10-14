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
build/real-model-venv/bin/python tools/verify_batch.py results/published/digits-batch-20261014
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
result collection and result-array allocation are inside timing. C and NumPy reuse
caller-owned scratch allocated before timing; sklearn's public estimator allocates
its own intermediates. C and NumPy share the same softmax function. NumPy uses
matrix-matrix dot with preallocated outputs, bias addition and in-place ReLU;
its input finite check is timed. C validates input/layer finiteness and aliasing
natively. sklearn retains its normal public API checks. These policies are made
explicit; equal input batches do not make the APIs' validation semantics identical.
Training, normalization, model setup, warm-up and reference calculation are excluded.

The `c_single_loop_input_first` diagnostic receives the same blocks but calls the
reviewed single-image C adapter once per row. Its Python slicing and temporary
block results are timed. Compare it with batch `input_first` to observe the cost
of Python calls and repeated per-row validation being amortized while retaining
the same handwritten dense calculation. This is an operational wrapper comparison,
not a pure measurement of the ctypes crossing alone. SGEMM additionally changes
the computation from matrix-vector loops to matrix-matrix operations.

A baseline uses six fresh processes, 20 measured sweeps per API/batch size per
process. Five primary backends (three C, NumPy, sklearn) cycle through all 120
backend permutations across the six processes; the single-image diagnostic is
inserted with a rotating position. Batch-size order rotates too. Timing uses
`perf_counter_ns`. No files are written between timed blocks. Every timed output
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
- Six compiled faults (batch ReLU, last row, row stride, second layer, GEMM bias
  and GEMM row count) must be numerically rejected by both synthetic and real-model
  checks under normal Python and `-O`; a crash/import failure does not count.
- Reports retain every saved test output, numerical errors, raw sweep timings,
  actual partitions and C call counts, providers, compiler commands and source/
  object/library hashes. Sources/build outputs are checked for changes during
  measurement. Verification regenerates references, checks historical source
  blobs and recomputes accuracy and timing summaries. Corruption fixtures alter
  last rows, tails, call counts, sample order, references and metadata even after
  file hashes are updated.

## Results

The first complete matched-batch baseline will be recorded here after validation.
