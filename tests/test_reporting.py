"""Known statistics and adversarial records, including checksum cancellation."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_stats import percentile, summarize, validate_samples


def check(condition, detail):
    if not condition:
        raise AssertionError(detail)


def row(phase, sample, kernel, reps, elapsed):
    return {"phase": phase, "sample": str(sample), "kernel": kernel,
            "repetitions": str(reps), "elapsed_ns": str(elapsed),
            "ns_per_inference": str(elapsed / reps), "checksum": "1" if reps == 1 else "0",
            "output_0": "1", "output_1": "-1"}


check(percentile([40, 10, 20, 30], .95) == 40, "p95")
check(percentile([40, 10, 20, 30], .5) == 20, "p50 nearest-rank")
rows = [row("model_init", 0, "shared", 1, 1),
        row("first_call", 0, "output_first", 1, 2),
        row("first_call", 0, "input_first", 1, 1)]
for sample, ns in enumerate([100, 200, 300, 400]):
    pair = [row("steady", sample, "output_first", 10, ns * 2),
            row("steady", sample, "input_first", 10, ns)]
    rows.extend(pair if sample % 2 == 0 else list(reversed(pair)))
validate_samples(rows, 4, [1.0, -1.0])
stats = summarize(rows)
check(stats["input_first"]["median_ns"] == 25, "median")
check(stats["input_first"]["p95_ns"] == 40, "p95")
check(stats["ratios"]["output_over_input_first"]["ratio_of_pooled_medians"] == 2, "speedup")
check(stats["ratios"]["output_over_input_first"]["matched_ratio_by_order"] == {"output_first,input_first": 2, "input_first,output_first": 2}, "order-specific ratios")
corruptions = [rows[:-1], rows + [rows[3]], rows[1:]]
for updates in ({"checksum": "999"}, {"elapsed_ns": "0"},
                {"output_0": "2", "output_1": "-2"},
                {"output_0": "nan"}, {"repetitions": "20"},
                {"ns_per_inference": "999"}, {"phase": "invented"}):
    broken = copy.deepcopy(rows)
    broken[3].update(updates)
    corruptions.append(broken)
# Equal corruption on both kernels still cannot pass the independent reference.
broken = copy.deepcopy(rows)
for r in broken[1:]:
    r.update(output_0="2", output_1="-2")
corruptions.append(broken)
for broken in corruptions:
    try:
        validate_samples(broken, 4, [1.0, -1.0])
    except (ValueError, KeyError):
        pass
    else:
        raise AssertionError("Corrupted benchmark data was accepted")
print("PASS: known statistics, paired order, completeness, equal work, timings and full outputs; checksum cancellation rejected")

# Three-backend records must cover all six permutations and the BLAS outputs.
import itertools
kernels = ("output_first", "input_first", "system_blas")
for start in range(6):
    orders = list(itertools.permutations(kernels))
    triples = [row("model_init", 0, "shared", 1, 1)]
    triples += [row("first_call", 0, k, 1, 4) for k in orders[start]]
    for sample in range(6):
        triples += [row("steady", sample, k, 10, {"output_first": 400, "input_first": 200, "system_blas": 100}[k])
                    for k in orders[(sample + start) % 6]]
    validate_samples(triples, 6, [1.0, -1.0], kernels, start)
    summary = summarize(triples, kernels)
    check(summary["ratios"]["input_over_system_blas"]["ratio_of_pooled_medians"] == 2, "BLAS ratio")
    triples[-1]["output_0"] = "9"
    try:
        validate_samples(triples, 6, [1.0, -1.0], kernels, start)
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong three-backend output accepted")
print("PASS: all six three-backend permutations, BLAS ratios and full-output validation")
