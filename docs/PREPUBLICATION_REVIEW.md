# Pre-publication review — 2026-10-14

This review covers production code, tests, documentation, archived performance
claims, tracked files and reachable Git history. At the time of the initial
review, the repository was Private and no open-source license had been selected.

Licensing update (2026-10-14): the owner subsequently approved the
[MIT License](../LICENSE) for project code and documentation. README and
contribution guidance now reflect that decision. Optional system BLAS libraries
retain their own licenses. The repository remains Private.

## Final review of the expanded project — 2026-10-14

The final pass started from `c2938f3` and includes the synthetic suite, original
and reviewed single-image model reports, original and reviewed batch reports,
all public documentation, tracked content and reachable Git history. The earlier
findings/validation below describe the initial synthetic-only review, not a claim
that newer engine objects are identical to the initial synthetic objects.

Corrections in this pass:

- README now separates latest reviewed digits/batch results from the archived
  synthetic measurements, identifies SGEMV versus SGEMM and documents the optional
  Python dependency range. No timing data or measured source revision is changed.
- `tools/check_documentation.py` checks all six published numerical tables,
  per-process ranges, headline accuracy and ratios, source attribution and local
  documentation links against archived reports. Separate numerical/provenance
  verifiers regenerate their references and recompute raw statistics.
- Regression fixtures reject stale table values, a false 100% accuracy claim,
  the historical wrapper ratio substituted for the direct-control ratio, and a
  wrong measured revision. The temporary fixture must pass unchanged before
  corruptions are tested, preventing missing link targets from masking a false
  pass; a broken-link corruption is also rejected. Normal Python and `python -O`
  are both exercised.
- Publication rules and ignore patterns additionally cover credential/config
  filenames, credential-bearing URLs, compiled objects/libraries, common archive
  snapshots and editor backups. These rules were tested with synthetic fixtures;
  no real credential or unwanted artifact was found requiring removal or history
  rewriting. Generated numerical JSON is marked as generated for readable source
  review, while the raw evidence remains available in Git.

Reviewed evidence and boundaries:

- Native C, batch/reference, allocation-symbol, sanitizer and compiled-fault tests
  exercise production sources/objects. Optional model checks cover training
  isolation, 450 held-out predictions, all configured batch tails, stale-output
  rejection, observed native call inputs and independent float64 references.
- All five archived measurement families are verified: one synthetic report,
  two single-image model reports and two batch reports. Historical timings retain
  their original source hashes and API contracts. No new speed measurement is
  needed for this documentation/publication-only change.
- Current primary batch evidence remains a paired within-process **11.90×**
  direct-control comparison, with **11.53–12.27×** observed process range. Original
  **12.24×** is kept as a historical block-wrapper comparison. Neither isolates
  the ctypes boundary or numerical kernel. NumPy's larger-batch advantage and
  the batch-1 regression remain visible. Accuracy is **438/450 (97.33%)** on one
  published split, with no writer-independent or production-OCR claim.
- Tracked names/content, reachable historical blobs and commit metadata were
  inspected. Supplemental scans found no credential assignments or temporary
  host paths. Git identities consistently use the owner's GitHub noreply address.
  GitHub owner/copyright identity and benchmark CPU/OS/timestamps are deliberate
  attribution/provenance, not scrubbed evidence. Local build environments, logs,
  personal agreements and trial measurements remain ignored.
- MIT is present and linked consistently. Dataset attribution retains CC BY 4.0;
  system BLAS and optional reference packages retain their own licenses. No
  third-party package source or local environment is vendored into the repository.

This is a bounded publication review, not a guarantee against unknown secret
formats or fabricated measurements. Timing evidence covers one host/model suite
and warm workloads; CI correctness checks do not certify universal performance.
The repository remains **Private**, and no publication/access-grant operation is
part of this review. The post-push CI run is checked separately at completion;
a document entry is not itself proof that remote checks passed.

## Initial findings corrected

1. **An empty evidence report could pass verification.** The verifier now rejects
   missing schemas/manifests/workloads, omitted or duplicate runs, inconsistent
   shapes/backends, incorrect command/timer/memory metadata and incomplete raw
   data. Stored reference outputs are regenerated from the deterministic inputs
   instead of being trusted as authoritative answers.
2. **Archived source checks depended on the current working tree.** Historical
   clean reports are now checked against the recorded Git source blobs. Full
   history is fetched in CI. Uncommitted reports can only validate their matching
   current files. Archived timings remain attributed to their measured revision.
3. **The public allocation contract was too broad.** The header now limits the
   no-heap claim to handwritten kernels and explicitly excludes vendor BLAS
   internals. It documents valid-pointer/count obligations, immutable model state,
   buffer separation and the scope of BLAS threading configuration.
4. **Optional metadata tools could unexpectedly abort a benchmark.** A missing
   optional executable now produces unavailable metadata. Reports describe actual
   C flags rather than unconditionally claiming contraction is off for every
   user-supplied flag set. Disabled BLAS is labeled as disabled.
5. **Publication checks were manual and future local files lacked ignore rules.**
   A repeatable guard now scans tracked files and every historical file path in
   reachable commits, including deleted files whose blobs were reused elsewhere.
   Known credential/key files and local working agreements/account settings are
   ignored. The guard does not print matched secret values.

## Validation

- C hand-derived/non-square results, boundary/error paths and buffer guards.
- 106 shapes per enabled backend, normal Python and `python -O`.
- ASan/UBSan on production C; vendor BLAS internals are not instrumented.
- Four handwritten-kernel mutations and the optional BLAS scaling mutation are
  rejected numerically under both Python modes.
- Synthetic publication regression fixtures reject empty/incomplete reports,
  forged or non-finite reference outputs, missing processes, incorrect metadata
  and source mismatches. A deleted local-only path is detected even when its
  content is reused by a harmless file in a later commit.
- New two-backend and three-backend smoke runs verify the updated runner and
  persisted-report checks. These smoke timings are not added to the baseline.
- The archived six-shape data was checked against its original source snapshot,
  regenerated references, all saved final-output vectors and recomputed per-run
  and pooled statistics. Maximum saved-output absolute error is approximately
  `7.5314e-7`. Shape-dependent speedup claims and the ratio direction were checked.
- Reviewed C engine and BLAS-adapter objects, built with the archived flags, are
  byte-identical to the archived object hashes. This review does not relabel or
  replace the historical performance measurements.
- Relative documentation links resolve. Git author identities use GitHub noreply
  addresses. The reviewed tree/history contained no matching personal home paths,
  known token/private-key formats, local agreements/review logs or compiled binaries.

The [CI matrix](https://github.com/andrew29mo/tiny-infer-lab/actions/workflows/ci.yml)
checks macOS/Linux with BLAS enabled and disabled, including the historical
measurement and publication guard. Its timings are not performance gates.

## Scope and remaining decisions

The privacy guard covers known patterns and tracked/reachable repository content;
it is not an exhaustive secret detector, an audit of unrelated accounts, or proof
that no unknown secret format exists. GitHub identity plus benchmark hardware,
OS and timestamps are intentional provenance. Local experiments remain excluded
from Git. No private data was found requiring a history rewrite.

The initial review covered one warm synthetic suite on one host, with batch-mean
latencies and process ranges rather than per-call tails or confidence intervals.
A subsequent [real-model extension](REAL_MODEL.md) separately validates a trained
digits classifier. Neither suite claims hardware-board validation or universal
superiority over BLAS. The evidence verifier checks consistency and
provenance; it is not cryptographic attestation that timings cannot be fabricated.

Public visibility remains an explicit owner decision. The subsequent MIT
licensing update documented above does not publish the repository.
