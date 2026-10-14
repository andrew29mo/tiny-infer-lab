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

Measured source `fc480ed`, Python 3.12.14, emlearn 0.23.2, ONNX Runtime 1.31.0,
NumPy 1.26.4 and sklearn 1.6.1. The machine is Apple M5 Pro, macOS/Darwin 27,
arm64. The sandbox prevented the runner's CPU-name query (`cpu: null` in the
manifest); the chip name was separately confirmed by a read-only system query.
No timings were rerun, replaced or selected based on that missing metadata.

All six backends and all five batches scored **438/450 (97.33%)**, with complete
class agreement. The largest probability error against the independent float64
oracle was **1.416e-6** (rounded), below the declared elementwise tolerance.

Timing: median of six process medians, **microseconds per prediction**, four full
450-image sweeps per sample and 12 samples per process. Divide neither by the
requested batch nor by 450 again. At batch 128 the final call still has 66 rows.

| Batch size | Tiny input | Tiny BLAS | emlearn | ORT CPU | NumPy | sklearn |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.271 | 6.078 | 3.374 | 4.963 | 5.395 | 23.145 |
| 8 | 1.115 | 1.006 | 1.031 | 0.750 | 0.809 | 3.116 |
| 32 | 0.546 | 0.449 | 0.783 | 0.236 | 0.305 | 0.933 |
| 128 | 0.400 | 0.219 | 0.718 | 0.101 | 0.167 | 0.362 |
| 450 | 0.365 | 0.140 | 0.697 | 0.059 | 0.122 | 0.183 |

emlearn is fastest at batch 1; ORT CPU is fastest at every larger tested batch.
The emlearn C loop amortizes the Python boundary but does not gain matrix-batch
efficiency. ORT's graph/kernel optimizations and native softmax are enabled, while
tiny and NumPy include Python/NumPy softmax. These results establish performance
for the documented interfaces; they do not isolate the dense kernels or prove a
universal provider ranking. Tiny C is not presented as faster than these runtimes.

Memory: median **warm process RSS in MiB**, three independent processes per cell.
These totals include the approximately 38 MiB common Python/NumPy baseline.

| RSS batch | Tiny input | Tiny BLAS | emlearn | ORT CPU | NumPy | sklearn |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 38.906 | 38.875 | 38.484 | 60.328 | 38.672 | 100.719 |
| 8 | 38.922 | 38.984 | 38.469 | 60.312 | 38.719 | 100.812 |
| 32 | 38.828 | 38.828 | 38.328 | 60.312 | 38.672 | 100.734 |
| 128 | 38.672 | 38.812 | 38.297 | 60.344 | 38.719 | 100.609 |
| 450 | 38.734 | 38.812 | 38.375 | 60.375 | 38.750 | 100.625 |

Batch-450 probe details, median MiB. Increment is computed within each process,
then summarized; subtracting independently summarized columns may differ slightly.
Sub-MiB differences are sensitive to page residency and should not be treated as
precise model/workspace savings or evidence that larger batches use less memory.

| Memory backend | Baseline RSS | Warm increment | Peak RSS |
| --- | --- | --- | --- |
| tiny_input | 38.016 | 0.703 | 38.781 |
| tiny_blas | 38.109 | 0.734 | 38.812 |
| emlearn | 38.062 | 0.328 | 38.422 |
| onnxruntime | 38.078 | 22.297 | 60.375 |
| numpy | 38.078 | 0.672 | 38.750 |
| sklearn | 38.062 | 62.641 | 101.172 |

Deployment: **bytes**, uncompressed artifacts and additional installed packages.
Every row additionally uses the common NumPy package payload of **45,209,250 bytes**
and the separately excluded Python/OS components. An ONNX model alone is not a
complete ONNX Runtime deployment. Package inventory digests cover installed payloads,
not a stripped embedded build; no claim is made that tiny is the smallest possible
implementation or that wheel payload bytes equal resident memory.

| Deployment backend | Artifact bytes | Extra package bytes |
| --- | --- | --- |
| tiny_input | 44766 | 0 |
| tiny_blas | 44766 | 0 |
| emlearn | 67448 | 0 |
| onnxruntime | 9944 | 80192288 |
| numpy | 10606 | 0 |
| sklearn | 10606 | 130195580 |

[Manifest and per-process timing ranges](../results/published/providers-20261014/manifest.json),
[full probability vectors and model identity](../results/published/providers-20261014/correctness.json),
[all 90 RSS probes](../results/published/providers-20261014/memory.json), and
[deployment inventory](../results/published/providers-20261014/deployment.json) retain
the evidence. A verifier regenerates references and recomputes statistics; corruption
tests change raw data and recompute hashes to challenge semantic validation. CI runs
production correctness and archived evidence checks on macOS/Linux, plus timing/RSS
smokes that are not speed gates. These local results remain one model, split, host,
float precision and warm execution policy; cold startup, power and MCU hardware are
outside the measurement scope.
