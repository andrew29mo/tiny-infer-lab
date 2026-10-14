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
  The serialized graph's edges, parameter shapes/dtypes, Softmax axis, opset and
  input/output dimensions are also checked, not merely operator names.
- Every provider/batch returns all 450 probability vectors. They are checked
  against a separate float64 `math.fsum` oracle (`atol=1e-5`, `rtol=3e-5`) and must
  agree on every class. Every timed sweep's output is checked outside timing.
- Tests observe actual C/session input blocks, including tails of 2 and 66 rows.
  Warm stale tails, wrong probabilities, changed ONNX weights, and a compiled
  emlearn adapter that skips each final row must be rejected. Normal and optimized
  Python both exercise the checks.
  Reviewed runs additionally archive untimed actual input observations for every
  provider/batch, at C, ORT, NumPy dot and sklearn public API boundaries. Hooks are
  removed before timing; ordered input hashes and call/row counts must match the
  canonical test split. The same sweep implementation is then timed.

## Timing contract

All rows compare **probability-producing Python-facing adapters**, not bare kernels.
Each timed sample processes all 450 images four times, including conversion/shape
and finite-input checks, adapter calls, output handling and the same NaN-initialized
collector. Model loading, export, session creation, reference calculation, warmup
and numerical checking are excluded. Input scaling is prepared before timing for
every provider. Tiny C additionally has its own native input/output guards; library
validation and allocation behavior are intrinsic differences and are not removed.
Previous sample outputs are released outside the next timer, so their deallocation
is not charged to the next backend. Every newly timed output is still retained
until it has been checked outside the timed interval.

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
and the input/reference arrays are loaded before the baseline. The probe records baseline RSS,
RSS after provider initialization, RSS after five complete sweeps, the signed warm
RSS difference, and process peak RSS (`getrusage`, platform units normalized to bytes).
The warm and peak snapshots are taken **before** reference validation and post-run
threadpool inspection. Each probe then verifies its actual last warm sweep and
records post-validation RSS/peak separately, as well as actual input hashes and
scratch capacity. A regression test drives real NumPy inference while making the
monitor report different values before/after validation to enforce this ordering.
Scratch capacity matches the requested batch.

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
package file sizes and SHA-256 hashes are archived, as are artifact sizes/hashes.
Counts, totals and inventory digests are recomputed from those raw records; missing
installed files fail collection. This is a reproducible wheel
payload estimate, **not** a minimal executable, compressed download, Python virtualenv
size, or complete deployment closure. Python/stdlib, adapter source, OS libraries
and system BLAS are excluded explicitly; native link names are recorded. Tiny's
handwritten row shares the BLAS-enabled library, so it is not a minimal no-BLAS build.
Dataset and third-party package licenses remain their own; project MIT does not
relicense them. See [dataset attribution](../models/digits/README.md).

## Measured results

Reviewed v2 source `c7ff4ea`, Python 3.12.14, emlearn 0.23.2, ONNX Runtime 1.31.0,
NumPy 1.26.4 and sklearn 1.6.1, on Apple M5 Pro, Darwin 27, arm64. CPU identity
is captured directly in this manifest. The previous v1 data and its different RSS
phase are preserved in the [original report](PROVIDER_COMPARISON_ORIGINAL.md).
See [review findings and negative experiments](PROVIDER_REVIEW.md) for why this
run was repeated. No fastest-run selection is performed.

Every generated model/input artifact and both native libraries are byte-identical
to v1. The model and held-out split are unchanged. All six providers at all five
batches score **438/450 (97.33%)**, with complete class agreement. Maximum
probability error against the independent float64 oracle is **1.416e-6** (rounded),
below the declared tolerance.

Timing: median of six process medians, **microseconds per prediction**, four full
450-image sweeps per sample and 12 samples per process. These are amortized costs,
not the response latency of an individual image. The final batch-128 call has 66
rows; batch 8 ends with two. Actual observed inputs cover all 450 rows in order.

| Batch size | Tiny input | Tiny BLAS | emlearn | ORT CPU | NumPy | sklearn |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.356 | 6.225 | 3.440 | 4.996 | 5.501 | 23.393 |
| 8 | 1.143 | 1.032 | 1.046 | 0.755 | 0.821 | 3.162 |
| 32 | 0.556 | 0.456 | 0.792 | 0.240 | 0.310 | 0.950 |
| 128 | 0.399 | 0.221 | 0.725 | 0.101 | 0.169 | 0.376 |
| 450 | 0.369 | 0.141 | 0.710 | 0.058 | 0.122 | 0.185 |

emlearn is fastest at batch 1; ORT CPU is fastest at every larger tested batch.
The emlearn C loop amortizes the Python boundary but does not gain matrix-batch
efficiency. ORT's graph/kernel optimizations and native softmax are enabled, while
tiny and NumPy include Python/NumPy softmax. Tiny C also includes native guards.
The comparison is fair for these documented probability-producing interfaces;
it does not isolate dense kernels or prove a universal provider ranking.

Memory: median **warm process RSS in MiB**, three fresh subprocesses per cell,
captured before numerical validation. Totals include a roughly 38 MiB common
Python/NumPy/reference-array baseline. Each probe's actual last warm output is
then validated; post-validation metrics are retained separately.

| RSS batch | Tiny input | Tiny BLAS | emlearn | ORT CPU | NumPy | sklearn |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 38.906 | 38.875 | 38.625 | 60.312 | 38.766 | 94.172 |
| 8 | 38.906 | 38.984 | 38.484 | 60.500 | 38.766 | 94.250 |
| 32 | 38.828 | 38.969 | 38.469 | 60.453 | 38.750 | 94.141 |
| 128 | 38.844 | 38.875 | 38.406 | 60.422 | 38.844 | 94.281 |
| 450 | 38.797 | 38.844 | 38.453 | 60.344 | 38.781 | 94.172 |

Batch-450 probe details, median MiB. Increment is computed within each process,
then summarized. These are process residency and process-lifetime peak figures,
not model RAM or peak kernel workspace. Changes relative to v1 are not attributed
solely to the phase fix: allocator, shared-page and import behavior also vary.
Sub-MiB differences do not establish meaningful engine RAM advantages.

| Memory backend | Baseline RSS | Warm increment | Peak RSS |
| --- | --- | --- | --- |
| tiny_input | 38.172 | 0.625 | 38.891 |
| tiny_blas | 38.156 | 0.672 | 38.906 |
| emlearn | 38.281 | 0.188 | 38.641 |
| onnxruntime | 38.281 | 22.109 | 60.344 |
| numpy | 38.219 | 0.594 | 38.781 |
| sklearn | 38.203 | 55.969 | 94.734 |

Deployment: **bytes**, uncompressed artifacts and additional installed packages.
Every row additionally uses the common NumPy package payload of **45,209,250 bytes**
and the separately excluded Python/OS components. The 3,548 raw package-file records
allow recomputing package sizes/counts/digests, rather than trusting an aggregate.
An ONNX model alone is not a complete ONNX Runtime deployment. Package payload is
not a stripped embedded build or resident memory. Counts exclude build/export
and monitoring tools, bytecode and wheel metadata; interpreter, adapter-source,
OS and system-BLAS exclusions still prevent this from being a complete closure.
The actual BLAS-enabled tiny library and emlearn audit-enabled library are used;
these are not independently minimized production binaries.

| Deployment backend | Artifact bytes | Extra package bytes |
| --- | --- | --- |
| tiny_input | 44766 | 0 |
| tiny_blas | 44766 | 0 |
| emlearn | 67448 | 0 |
| onnxruntime | 9944 | 80192288 |
| numpy | 10606 | 0 |
| sklearn | 10606 | 130195580 |

[Manifest and timing process ranges](../results/published/providers-reviewed-20261014/manifest.json),
[full probability vectors and actual call observations](../results/published/providers-reviewed-20261014/correctness.json),
[all 90 phased RSS probes](../results/published/providers-reviewed-20261014/memory.json), and
[per-file deployment inventory](../results/published/providers-reviewed-20261014/deployment.json)
retain the evidence. Verifiers recompute references, observations and totals;
rehashed corruption tests challenge their semantic checks. File inventories and
hashes are consistency/provenance evidence, not cryptographic attestation against
an author fabricating all records. These are one model, split, host, float precision
and warm execution policy. CI checks correctness/provenance on macOS/Linux and
runs timing/RSS smokes without performance gates. Cold startup, power and MCU
hardware validation are outside scope.
