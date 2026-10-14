# Selected benchmark evidence

- [Reviewed same-model six-provider comparison, 2026-10-14](providers-reviewed-20261014/manifest.json):
  exact graph/weights, actual input observations in six timing processes,
  90 phased RSS probes and per-file deployment inventory;
  [results](../../docs/PROVIDER_COMPARISON.md) and [review](../../docs/PROVIDER_REVIEW.md).
- [Original same-model six-provider comparison, 2026-10-14](providers-20261014/manifest.json):
  production C, emlearn C-loop batch adapter, ONNX Runtime CPU, NumPy and sklearn;
  five batch sizes, six timing processes and 90 isolated RSS probes. See the
  [historical protocol/results and limitations](../../docs/PROVIDER_COMPARISON_ORIGINAL.md).

- [Apple M5 Pro, 2026-10-14](apple-m5-pro-20261014/report.json): six shapes, three
  backends, six process rounds, 30 matched groups per process.
- [Reviewed matched-batch measurement, 2026-10-14](digits-batch-reviewed-20261014/manifest.json):
  anti-stale tail checks, observed native calls/inputs, direct and wrapped controls,
  six processes and paired improvement ratios; [review](../../docs/BATCH_REVIEW.md).
- [Original matched-batch digits measurement, 2026-10-14](digits-batch-20261014/manifest.json):
  batch sizes 1/8/32/128/450, matched tails, C/NumPy/sklearn, six processes and
  20 sweeps per API/batch/process; [protocol and results](../../docs/BATCH_INFERENCE.md).
- [Reviewed digits array-API measurement, 2026-10-14](digits-reviewed-20261014/manifest.json):
  matched single-image array inputs, prepared NumPy diagnostic, and separate sklearn
  batch context; six processes and 12 sweeps per backend/process.
- [Original trained digits classifier, 2026-10-14](digits-20261014/manifest.json):
  450 held-out images, three C backends and sklearn, six process rounds,
  12 sweeps per backend/process, historical prepared-pointer contract; [protocol and results](../../docs/REAL_MODEL.md).
- [Interpretation and limitations](../../docs/RESULTS.md)

Verify raw data, outputs, timings and recorded source hashes with:

```sh
python3 tools/verify_report.py results/published/apple-m5-pro-20261014
```

Historical local experiments and personal review notes are excluded from Git.

The optional real-model evidence uses the pinned reference dependencies:

```sh
build/real-model-venv/bin/python tools/verify_real_model.py results/published/digits-20261014
build/real-model-venv/bin/python tools/verify_real_model.py results/published/digits-reviewed-20261014
```

Matched-batch evidence uses the same optional environment:

```sh
build/real-model-venv/bin/python tools/verify_batch.py results/published/digits-batch-20261014
build/real-model-venv/bin/python tools/verify_batch.py results/published/digits-batch-reviewed-20261014
```
