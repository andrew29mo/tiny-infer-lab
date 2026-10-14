# Measured results: Apple M5 Pro

Measured on 2026-10-14 with Apple Accelerate CBLAS in queried single-thread mode.
Source revision: [`2d95f10`](https://github.com/andrew29mo/tiny-infer-lab/commit/2d95f1095771d9f72981684ab558cc4933deb57b).
The report also records source/object/executable hashes and exact build commands.

Six independent processes per shape, 30 matched groups per process, three backends.
That is 3,240 steady measurement batches. Each matched group uses identical
repetition counts. All six execution permutations and all six shape positions
are balanced. We retain individual process statistics rather than treating all
batches as independent machines.

## Pooled median latency

Times are microseconds per inference, computed from batch averages.

| Input→hidden→output | output_first | input_first | system BLAS | Loop speedup | BLAS over input_first |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8→16→4 | 0.080 | 0.039 | 0.042 | 2.03× | 0.93× |
| 31→17→9 | 0.219 | 0.133 | 0.072 | 1.64× | 1.84× |
| 128→256→32 | 15.990 | 2.908 | 1.580 | 5.50× | 1.84× |
| 256→512→64 | 104.973 | 10.314 | 3.188 | 10.18× | 3.24× |
| 1024→1024→256 | 1286.855 | 73.873 | 12.172 | 17.42× | 6.07× |
| 1024→64→1024 | 72.188 | 9.175 | 2.481 | 7.87× | 3.70× |

The ratio is `input_first latency / BLAS latency`; below one means input_first
was faster. The smallest
case is only about 7% apart in this run; this is a local overhead-sensitive
comparison, not a universal crossover threshold.

## What the evidence supports

- Loop ordering helps these six workloads, but its measured gain varies from
  **1.64× to 17.42×**. The original **approximately 5.5×** is specific to the
  128→256→32 case on this host, not a general engine speedup.
- Accelerate is faster than input_first in **five of six** cases. It is about
  **1.84× faster on the original case**, and **6.07×** on 1024→1024→256.
- input_first is slightly faster on 8→16→4. API overhead, checks, memory access
  and vendor implementation choices all contribute. These measurements do not
  isolate which mechanism causes each difference.
- Correctness is checked against Python float64 math.fsum for each backend and
  workload, plus each batch's complete final output and accumulated checksum.
  The largest absolute error among these six workload checks is below 1e-6; the
  separate 106-shape regression suite has maximum error about 2.33e-6.

## Process-to-process ratio ranges

These observed min/max ranges are not confidence intervals.

| Shape | output_first / input_first | input_first / system BLAS |
| --- | ---: | ---: |
| 8x16x4 | 1.945–2.043 | 0.918–0.964 |
| 31x17x9 | 1.610–1.717 | 1.807–1.861 |
| 128x256x32 | 5.371–5.508 | 1.810–1.851 |
| 256x512x64 | 9.981–10.365 | 3.106–3.249 |
| 1024x1024x256 | 17.384–17.450 | 6.035–6.099 |
| 1024x64x1024 | 7.682–7.997 | 3.687–3.725 |

## Reproduce and inspect

Use a full Git clone so the measured source revision is available. The stored
results remain an immutable measurement of `2d95f10`; later validation or
documentation changes do not turn them into a new benchmark of a newer revision.

```sh
make BLAS=1 test sanitize mutation
python3 tools/run_benchmark.py --blas 1 --runs 6 --samples 30
python3 tools/verify_report.py results/published/apple-m5-pro-20261014
```

- [Full report](../results/published/apple-m5-pro-20261014/report.json)
- [Readable summary](../results/published/apple-m5-pro-20261014/summary.txt)
- Each shape directory contains `run-*.csv`, timer-resolution stderr and `samples.csv`.
- [Measurement protocol and limitations](BENCHMARKS.md)

These are warm, repeated synthetic inputs on one shared macOS arm64 host, not
model accuracy, cold-cache behavior, per-call tail latency, GPU/NPU performance
or a general comparison with full inference frameworks. macOS/Linux CI checks
correctness and a small smoke suite; CI timing is not merged into this baseline.
