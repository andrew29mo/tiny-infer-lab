# Batch inference review — 2026-10-14

Scope: production-object coverage, completeness/freshness of all tail predictions,
equal-workload comparisons and the interpretation of the original 12.24× result.
Repository visibility remains Private. Historical measurements are preserved.

## Findings and fixes

1. **Warm output state could conceal skipped tail computation.** A controlled
   reproduction warmed the final two-image output buffer, then skipped the C
   binding call for that tail while returning success. Original numerical/class
   checks still passed with only 56 native calls instead of 57 for size 8.
   This demonstrates a harness weakness, not evidence that production C omitted
   rows. C source loops cover every row and no numerical implementation change
   was needed. The revised harness poisons C/NumPy scratch and single-image C
   outputs before each sweep outside timing. The shared result collector is NaN
   initialized inside timing for every backend. Omitted tail computation or
   omitted result collection can no longer inherit previously correct answers.
2. **Call counts were expected counts, not observed counts.** A separate untimed
   audit now intercepts the actual loaded single/batch C bindings through the
   same sweep path, recording each batch size, row total and hash of ordered
   input floats. It confirms 450 native rows exactly once, including 2-row and
   66-row tails. Timed rows explicitly label expected C calls. No interception
   or input copying is included in timing. These observations do not claim
   that every timed call was instrumented or cryptographically attested.
3. **The 12.24× numerator included a block wrapper.** The original loop allocated
   temporary blocks, sliced individual rows and copied block results into the
   common output collector. That was disclosed, but it is not a minimal direct
   single-image API baseline. The new run measures both controls in the same
   processes. The direct control uses pre-created single-image array views and
   writes straight into the common collector. It still times ordinary input
   handling, ctypes calls, validation and softmax. Within-process median ratios
   are retained; the reviewed direct comparison is approximately 11.90× at 450.
   The new wrapped comparison is approximately 12.11×. Original 12.24× remains
   valid only for its historical wrapper protocol, not as a pure call/kernel gain.
4. **Build evidence required stronger structural checks.** The verifier now
   validates complete compile/link commands, including numerical flags and the
   exact production objects used by native tests. Build tests also guard source,
   object and library stability across both the single-image and batch phases.
   Corruption fixtures alter flags, test links, observed tails/inputs and paired
   ratios after updating file hashes; they must still fail semantic validation.

## Observed evidence

- Native batch executable, Python oracle tests and measurement load/link the same
  production objects. Current native object hashes exactly match the original
  batch run; all numerical C code remains unchanged by this review.
- Native guard/capacity/alias/overflow/nonfinite checks and 45 independent batch/
  shape cases exercise the real shared-library symbol. Seven compiled production
  faults are rejected numerically by synthetic and learned-model checks, both
  normally and under `python -O`; build/import errors and crashes are not counted.
- New fixtures warm and then skip native tails of size 2/66, drop a 66-row result
  slice, inspect exact ordered native inputs, and ensure the direct baseline
  cannot enter the nested block wrapper. Explicit partition checks cover indices
  0–449 exactly once for every selected size.
- A fresh six-process, 20-sweep run checks all 450 outputs for every API and size,
  including both single-image controls. Accuracy remains 438/450 (97.33%), with
  complete sklearn class agreement and elementwise float64-oracle checks.
- [Results and raw evidence](BATCH_INFERENCE.md) include both historical and
  reviewed protocols. Only primary backends participate in the balanced 120
  permutations; both diagnostics have rotating insertion positions.

## Interpretation and limits

The measurements support a substantial reduction in the full inference-wrapper
cost through batching. They include repeated validation, Python collection,
softmax layout and ctypes overhead, so neither 11.90× nor 12.24× isolates the
cost of crossing Python/C. The SGEMM backend also changes the arithmetic layout.
NumPy remains faster at larger tested sizes; no universal C superiority follows.

C/NumPy reuse caller-owned scratch, whereas sklearn's public estimator allocates
intermediates and performs broader validation. All receive the same float32
images and partitions, but their APIs are not semantically identical. Scratch
poisoning controls freshness/cache state and is disclosed as outside timing;
collector initialization is timed for everyone. The fresh protocol is compared
internally, not used to reinterpret historical timings. One host, one small
model, warm inputs and process ranges limit generalization. The unchanged data
split is not writer-independent and was not used for new training or tuning.
