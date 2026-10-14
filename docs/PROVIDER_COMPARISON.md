# Same-model provider comparison

This optional experiment compares the **same frozen float32 64→32→10 ReLU MLP**
on the same 450 held-out digits images, in the same order, at requested batch
sizes **1, 8, 32, 128 and 450**. No retraining, quantization or test-set selection
is performed. The [original split and limitations](REAL_MODEL.md) still apply.

## Reproduce

Use Python 3.12 and a C compiler. Linux additionally requires OpenBLAS development
headers/libraries; macOS uses system Accelerate.

```sh
python3.12 -m venv build/comparison-venv
build/comparison-venv/bin/python -m pip install -r requirements-comparison.txt
build/comparison-venv/bin/python tools/compare_providers.py prepare
build/comparison-venv/bin/python tests/test_providers.py
build/comparison-venv/bin/python -O tests/test_providers.py
# Measurement sources must be committed; output directory must not already exist.
build/comparison-venv/bin/python tools/compare_providers.py benchmark
build/comparison-venv/bin/python tools/verify_providers.py results/local/providers
```

Core builds do not require these packages. emlearn/ONNX are export-time tools;
the emlearn C runtime is compiled from installed upstream headers without vendoring
them. ONNX Runtime uses its installed CPU runtime. Primary versions are pinned;
recorded package versions and installed-payload inventories describe the measured
environment. This does not promise identical transitive wheels on every platform.

## Model identity and correctness

- Original model JSON, dataset hash, test IDs and labels are retained. The fixed
  pixel `/16` preprocessing uses no fitted test statistics.
- emlearn's default exporter rounds literals to six decimal places. This tool
  replaces only generated parameter literals with exact C99 float32 hex literals.
  It reads back all 2,410 **compiled** parameters and checks their bytes. Upstream
  emlearn dense/activation/softmax routines perform the computation.
- ONNX contains MatMul, Add, ReLU, MatMul, Add and Softmax, with a dynamic batch
  dimension. Serialized initializers must have the same float32 bits as the C,
  NumPy and sklearn parameters. No converter retrains or approximates the model.
- Every provider/batch returns all 450 probability vectors. They are checked
  against a separate float64 `math.fsum` oracle (`atol=1e-5`, `rtol=3e-5`) and must
  agree on every class. Every timed sweep's output is checked outside timing.
- Tests observe actual C/session input blocks, including tails of 2 and 66 rows.
  Warm stale tails, wrong probabilities, changed ONNX weights, and a compiled
  emlearn adapter that skips each final row must be rejected. Normal and optimized
  Python both exercise the checks.

## Timing contract

All rows compare **probability-producing Python-facing adapters**, not bare kernels.
Each timed sample processes all 450 images four times, including conversion/shape
and finite-input checks, adapter calls, output handling and the same NaN-initialized
collector. Model loading, export, session creation, reference calculation, warmup
and numerical checking are excluded. Input scaling is prepared before timing for
every provider. Tiny C additionally has its own native input/output guards; library
validation and allocation behavior are intrinsic differences and are not removed.

- Tiny C: production batch API, either handwritten input-first or system BLAS;
  NumPy softmax is included.
- emlearn: **one Python→C call per requested block**, then a C loop calls upstream
  single-image inference including its softmax. This is an explicitly added batch
  adapter, not an upstream vectorized batch kernel or emlearn's stock Python API.
  Its upstream static scratch is single-threaded and non-reentrant.
- ONNX Runtime: `CPUExecutionProvider`, sequential execution, one intra-op and one
  inter-op thread, full graph optimization. Standard `session.run` output allocation
  and Python API cost are included; no I/O binding or custom/reduced runtime build.
- NumPy reuses dense buffers; sklearn uses public `predict_proba`.

Numerical BLAS pools are limited to one thread, as is tiny's native BLAS. Six fresh
timing processes each record 12 sample groups. Cyclic/reversed backend orders balance
positions, and batch order rotates. Tables use the median of process medians, with
process ranges retained in raw evidence. These are warm batch-mean costs, not
per-image latency percentiles, confidence intervals, or cold startup measurements.
The new common adapter checks differ from the previous batch experiment, so old
11.90×/12.24× ratios must not be substituted into this experiment.

## Memory contract

RSS is measured separately from timing, in **a fresh subprocess for every provider,
batch size and repetition**, with three repetitions. Common Python, NumPy, monitoring
and the input array are loaded before the baseline. The probe records baseline RSS,
RSS after provider initialization, RSS after five complete sweeps, the signed warm
RSS difference, and process peak RSS (`getrusage`, platform units normalized to bytes).
Memory probes also verify predictions. Scratch capacity matches the requested batch.

Warm RSS is live process residency; peak RSS is the entire process high-water mark,
including imports and temporary loading allocations. The difference is not pure
model RAM or peak inference workspace; allocator behavior, shared pages and sampling
noise matter. Common NumPy overhead must be added when considering a full Python
deployment. These measurements do not establish microcontroller RAM usage.

## Deployment-size contract

Report uncompressed artifact bytes separately from installed dependency-payload
bytes. Tiny uses its actual BLAS-enabled shared library plus float32 `weights.npz`;
emlearn's shared library contains the weights and upstream runtime; ONNX needs its
model plus the installed ONNX Runtime package and declared runtime dependencies.
All measured Python adapters share NumPy, reported separately. sklearn includes its
additional reference stack. emlearn export tools and the ONNX exporter are excluded
from runtime requirements.

The installed payload counts distribution-listed package files, including shipped
libraries/data/tests, but excludes metadata, scripts and generated bytecode. Per-
package file counts and inventory hashes are archived. This is a reproducible wheel
payload estimate, **not** a minimal executable, compressed download, Python virtualenv
size, or complete deployment closure. Python/stdlib, adapter source, OS libraries
and system BLAS are excluded explicitly; native link names are recorded. Tiny's
handwritten row shares the BLAS-enabled library, so it is not a minimal no-BLAS build.
Dataset and third-party package licenses remain their own; project MIT does not
relicense them. See [dataset attribution](../models/digits/README.md).

## Measured results

Results are added only after source-committed, verified runs. No universal speed,
memory or size advantage is assumed.
