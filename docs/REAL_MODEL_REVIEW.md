# Real-model review — 2026-10-14

Scope: sample-level train/test isolation, provenance of learned weights,
production-code coverage, numerical acceptance and sklearn timing fairness.
The repository remains Private. Original timing evidence is preserved.

## Findings and corrections

1. **The timing input contracts differed.** Original C calls indexed a cached
   input pointer and reused scratch space, while sklearn received an array and
   ran its public estimator checks each time. The original report disclosed an
   API comparison but did not clearly identify the cached-pointer advantage.
   The reviewed C adapter accepts the same `(1, 64)` float32 array shape on each
   call. Conversion, shape checking, ctypes pointer creation and native finite
   checks are timed. Single-image result-array allocation is now timed too.
   Scratch-buffer allocation differences remain and are stated explicitly.
   A prepared NumPy dot/ReLU/softmax diagnostic and separate sklearn batch timing
   make the general estimator overhead visible. Neither is mixed into equal-API
   speed ratios. Fresh schema-2 evidence replaces the original figures for the
   array-API comparison; historical measurements are not rewritten.
2. **Metadata omission could weaken provenance checks.** Empty source/build maps
   and threadpool lists could pass vacuous checks. Verification now requires the
   exact expected source/object inventories, compiler commands, valid hashes,
   nonempty BLAS threadpool observations, provider/version metadata and the
   recorded backend/sample schedule. Building and measuring both guard source,
   object and library stability. Measured sources must match the recorded commit.
   Corruption tests remove these fields or alter timing order and update file
   hashes, so merely detecting a checksum change cannot satisfy the regression.
3. **Training policy and isolation needed direct tests.** Restoration now checks
   seed, preprocessing and complete training parameters. Split checks reject
   index overlap and exact duplicate images. Actual fitting is observed in a
   regression: it receives exactly the 1,347 training rows/labels. Two fits with
   fixed indices, after changing all 450 held-out images and labels, must produce
   identical weights, biases, iteration count and training loss.
4. **Numerical tests needed to challenge faulty production libraries directly.**
   Export corruption tests alone did not demonstrate that learned-model checks
   reject broken C arithmetic. Five compiled mutations (missing ReLU, wrong bias
   in each handwritten kernel, omitted second layer and incorrect BLAS scale)
   are now evaluated on the actual trained model. Both normal Python and `-O`
   must reject them numerically; a build error, import error or crash is not proof.
   The array-input adapter is also exercised on all 450 images and invalid inputs.
   Numerical output shapes must match exactly, preventing broadcast acceptance.

## Evidence observed

- No overlapping train/test indices or exact duplicate images across the split.
  Preprocessing is fixed division by 16, with no learned statistics.
- In the review environment (Python 3.9.6, NumPy 1.26.4, scipy 1.13.1,
  sklearn 1.6.1), a fresh training run reproduced all committed weights, biases,
  iteration count and training loss exactly. Original sklearn export
  probabilities also matched exactly. This is observed same-host provenance
  evidence; exact cross-platform training reproducibility is not promised.
- Training-isolation regression passes with both Python modes. All five compiled
  faulty production libraries are numerically rejected in both modes.
- All 450 held-out examples still score 438/450 (97.33%); all three C backends
  agree with sklearn on every class and pass elementwise logit/probability checks.
- [Reviewed results](REAL_MODEL.md) retain six fresh processes, 12 sweeps per
  backend/process, original nanosecond timings, hashes and numerical references.
  Old and new reports undergo the strengthened verifier and corruption fixtures.

## Limits of the conclusion

The evidence supports no observed sample-level fitting leakage in this pipeline;
stratified splitting itself legitimately uses class labels. Writer identities are
not available in this packaged subset, so related writing styles can occur on
both sides. This is not writer-independent evaluation or the original UCI split.
The test set is now a published regression set; repeated future tuning against
its score would turn it into development data and require a new untouched test.

Public API timings include differing validation, allocation and Python overhead;
they do not isolate kernel speed. Prepared NumPy and batched sklearn have distinct
contracts. One host/model, warm data and process ranges limit generalization.
Source hashes and verifier checks provide consistency, not cryptographic proof
that arbitrary timing reports cannot be fabricated.
