# emlearn / ONNX Runtime comparison review — 2026-10-14

Scope started at `3dc5929`: same frozen model, canonical test split, actual
production adapters, original timing/RSS/deployment records, tests and claims.
The repository remains Private. This review found validation/measurement gaps,
not evidence of wrong original weights, test-set tuning or fabricated timings.

## Findings and corrections

1. **The export identity guard did not fully check the architecture.** Changing
   ONNX Softmax axis from 1 to 0 still passed the bitwise-weight/operator-name
   audit. The full numerical comparison was a separate safeguard and was not
   shown to accept that wrong model. The guard now also checks graph edges,
   attributes, opset, tensor shapes/dtypes and dynamic input/output contracts.
   A valid copied model must pass before the changed-axis fixture is rejected.
2. **Internally balanced deployment-size substitutions could pass.** Setting
   recorded ORT package bytes to 1 and updating provider totals still passed the
   old verifier: the underlying per-file inventory was represented only by a
   digest, so its sum could not be independently recomputed. V2 retains each of
   3,548 eligible installed files' relative path, bytes and SHA-256, plus artifact
   bytes/hashes tied to build hashes. Missing installed files fail collection;
   totals, counts and digests must agree with raw records. Rehashed corruption
   tests cover balanced totals, changed raw files, missing inventories, artifact
   totals and omission of runtime dependencies. This still cannot attest against
   fabricating every record together; installed payload is not a deployment closure.
3. **RSS snapshots followed validation allocations.** Original warm/peak RSS
   included loading the reference and checking probabilities. V2 loads the common
   input/reference before baseline, snapshots loaded/warm/peak phases before
   numerical validation and post-run threadpool inspection, then validates the
   actual last warm output and records post-validation RSS/peak separately. A
   phase-sensitive monitor fixture runs real NumPy inference and fails if a
   validation-time RSS increase is charged to the warm snapshot. Peak remains
   whole-process lifetime peak through the sampled phase, including imports and
   setup; it is not a peak inference workspace estimate.
4. **Actual input observations were limited to tests.** All reviewed timing
   processes now archive untimed observations at real C, ORT, NumPy dot and
   sklearn boundaries using the same sweep path. Ordered input hashes, observed
   calls, total rows and tails must match the canonical 450 rows. Hooks are removed
   before warmup/timing and restored on errors. Restoration preserves whether the
   original attribute belonged to the class or instance; assigning a bound method
   back to an instance incorrectly bypassed class-descriptor lookup in a first
   v2 candidate. That defect was reproduced on actual sklearn dispatch, then
   fixed and tested on both sklearn/ORT success paths and error restoration.
   Test hooks do not contribute to
   measured times. Sample outputs are released outside the next timer rather
   than charging the prior sample's deallocation to the next backend.

## Verified identity and production coverage

The final reviewed source is `95d831d`. The first v2 candidate at `c7ff4ea` was
superseded because its method restoration changed dispatch; its records remain
in [Git history](https://github.com/andrew29mo/tiny-infer-lab/tree/abcb4ea/results/published/providers-reviewed-20261014).
The current v2 report was remeasured after the fix, not selected by speed or
relabeled as a run of the fixed code. All seven generated model/input/build artifacts
and both tiny production objects are byte-identical to v1: model, inputs and
native implementations did not change. All 2,410 compiled emlearn parameters
and all serialized ONNX initializers match the frozen float32 weights. Canonical
dataset hash, test IDs, input hashes and output order agree. Requested batches
1/8/32/128/450 include the actual 2-row and 66-row tails.

Normal Python and `python -O` exercise the actual loaded libraries/session,
independent float64 oracle, call observation/restoration, bad-input guards,
changed weights/axis and a genuinely compiled emlearn adapter that skips rows.
Warm stale tails and false probabilities are also rejected. Report tests cover
both archived protocols and corrupt data after recomputing file hashes.

## Measurement judgment

Six fresh timing processes retain 12 samples per provider/batch, four complete
450-image sweeps per sample, balanced backend positions and rotating batch order.
Every timed sweep's full output is checked outside timing. One-thread BLAS and
actual ORT CPU/session policies are checked. All six providers retain
**438/450 (97.33%)** accuracy with full class agreement. The unchanged model is
not retrained; original training-isolation and split limitations still apply.

The comparison is credible for the documented **Python-facing, probability-
producing interfaces**. It intentionally includes library allocation/validation
differences. emlearn has an added C-loop batch adapter; ORT has optimized native
batch/softmax; tiny has native safety checks and NumPy softmax. It does not prove
equal kernel work or rank bare kernels. emlearn remains fastest at batch 1 and
ORT at the larger tested batches; tiny does not claim a universal speed win.

Ninety fresh subprocesses cover every provider/batch three times. Phase-separated
RSS supports process-footprint comparisons with common Python/NumPy overhead,
not MCU RAM or model-only claims. Page residency/import variability remains;
differences between v1/v2 are not attributed entirely to the phase correction.
Deployment quantities remain uncompressed artifacts/package payloads, excluding
interpreter/stdlib, adapter source, OS/system BLAS and monitoring/export/build tools.
Tiny uses its BLAS-enabled library even on the handwritten row; neither C library
is an independently minimized executable. Runtime dependency names were checked
against the installed distributions' active, non-extra requirements.

See [reviewed measurements and raw evidence](PROVIDER_COMPARISON.md) and
[original protocol/results](PROVIDER_COMPARISON_ORIGINAL.md). V1 data remains
at its original source revision; it does not gain V2 observations or file-level
evidence retroactively. Documentation checks validate both sets of tables against
their own reports. CI checks correctness and provenance on macOS/Linux; its smoke
timings are not performance gates or proof of universal performance. Remote CI
completion is verified separately after the final commit.
