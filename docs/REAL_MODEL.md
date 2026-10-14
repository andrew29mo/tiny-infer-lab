# Real-model validation: handwritten digits

The trained 64→32→10 float32 ReLU MLP classifies real 8×8 handwritten digits.
The engine computes logits; a Python adapter applies stable softmax. Class labels
are digits 0–9. This adds no model parser or softmax operation to the C API.

## Reproduce

From the repository root (Python 3.9–3.12):

```sh
python3 -m venv build/real-model-venv
build/real-model-venv/bin/python -m pip install -r requirements-real-model.txt
build/real-model-venv/bin/python tools/real_model.py build --blas 1
build/real-model-venv/bin/python tests/test_real_model.py build/real_engine.so
build/real-model-venv/bin/python -O tests/test_real_model.py build/real_engine.so
build/real-model-venv/bin/python tools/real_model.py benchmark --runs 6 --samples 12
build/real-model-venv/bin/python tools/verify_real_model.py results/published/digits-20261014
```

macOS requires macOS 15+ and Command Line Tools. Linux needs system OpenBLAS
headers/library (`libopenblas-dev` on Debian/Ubuntu). `build --blas 0` checks just
the two handwritten backends. Existing default build/tests stay dependency-free.
Use a new output directory for each benchmark (`--output`); existing directories
are rejected. Shared CI only runs correctness and timing smoke checks.

Retraining is optional and explicit:

```sh
build/real-model-venv/bin/python tools/real_model.py train --model build/retrained-digits.json
build/real-model-venv/bin/python tools/real_model.py check --model build/retrained-digits.json
```

The committed artifact is used for regression checks; retraining never silently
replaces it. See [dataset provenance, licensing and split limitations](../models/digits/README.md).

## Accuracy and numerical correctness

- A fixed stratified split uses 1,347 training and 450 held-out examples. Only
  training data enters fitting; preprocessing divides pixels by 16 without fitting.
- Acceptance was set to at least 94% held-out accuracy; no test-driven tuning.
- The stored original sklearn probabilities check the float32 export round trip.
  Reconstructed sklearn `MLPClassifier.predict_proba` checks the committed weights.
- Every test image is also recomputed using independent float64 `math.fsum` logits.
  All C backends must match logits and probabilities elementwise within
  `1e-5 + 3e-5*abs(reference)` and agree on every predicted class.
- C objects linked into the measured shared library also run the C regression
  executable and 106-shape reference test. The real-model tests call that actual
  shared library, under normal Python and `python -O`.
- Regression fixtures corrupt splits, shapes, class mappings, weights, dataset
  identity and output values; class agreement alone cannot pass numerical checks.

The committed model scores **438/450 = 97.33%**. All three C backends agree with
sklearn on all 450 classes. This is one held-out split, not a population accuracy
guarantee, writer-independent result, original UCI benchmark or production OCR.
Confusion matrices and all test logits/probabilities are retained with raw evidence.

## Timing contract

Each sample times one complete sweep of the same 450 preprocessed float32 images.
The single-image comparison calls each public prediction adapter once per image:
C through ctypes with softmax versus sklearn `predict_proba` with softmax. Both
return ten probabilities, and output collection is included. All timed outputs
are verified after timing. Data loading, normalization, training, model setup,
reference evaluation and warm-up are excluded. Python/ctypes overhead, validation,
allocation differences and softmax are included. This is an API comparison, not
an isolated native kernel speed comparison or a claim about other runtimes.

A separate `sklearn_batch` measurement sends all 450 images in one call, allowing
optimized matrix-matrix computation. Its amortized time is reported as context,
not mixed into single-image speedup ratios; the C API supports one image per call.

The baseline uses six fresh worker processes and 12 timed sweeps per backend per
process. The four single-image backend orders cycle through all 24 permutations,
with batch position rotated. All numerical threadpools are limited and queried at
one thread, and native C BLAS configuration is queried separately. Each backend
gets the same images/counts. Results summarize per-process medians, then their
median and range. Ranges are not confidence intervals or individual-call tails.
Warm data and one small model/host limit generalization.

Reports retain source hashes/revision, compiler commands, object/library hashes,
package versions, queried numerical providers, model/data hashes, exact split,
raw nanosecond sweep timings and numerical evidence. The verifier checks recorded
source blobs in full Git history and regenerates numerical references. This is
consistency/provenance checking, not cryptographic proof of honest timing.

## Measured baseline

Raw evidence and the measured per-API table are added after the first complete run.
