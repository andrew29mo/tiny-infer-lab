"""Validate complete measured outputs and summarize matched groups."""
import itertools
import math
import statistics


def percentile(values, q):
    if not values or not 0 < q <= 1:
        raise ValueError("Invalid percentile request")
    return sorted(values)[math.ceil(q * len(values)) - 1]


def validate_samples(rows, count, expected, kernels=("output_first", "input_first"), start=0):
    nk = len(kernels)
    orders = list(itertools.permutations(kernels))
    if len(rows) != nk * (count + 1) + 1 or rows[0]["phase"] != "model_init":
        raise ValueError("Missing or extra measurement records")
    first = rows[1:nk + 1]
    if any(r["phase"] != "first_call" or r["sample"] != "0" or int(r["repetitions"]) != 1 for r in first):
        raise ValueError("Invalid first-call records")
    if tuple(r["kernel"] for r in first) != orders[start % len(orders)]:
        raise ValueError("Wrong first-call order")
    steady = rows[nk + 1:]
    if len({int(r["repetitions"]) for r in steady}) != 1:
        raise ValueError("Unequal work in matched measurements")
    for sample in range(count):
        group = steady[nk * sample:nk * (sample + 1)]
        if any(r["phase"] != "steady" or r["sample"] != str(sample) for r in group):
            raise ValueError("Missing or duplicate samples")
        if tuple(r["kernel"] for r in group) != orders[(sample + start) % len(orders)]:
            raise ValueError("Incorrect execution order")
    for row in rows:
        reps, elapsed = int(row["repetitions"]), int(row["elapsed_ns"])
        reported = float(row["ns_per_inference"])
        if reps <= 0 or elapsed < 0 or (row["phase"] == "steady" and elapsed == 0):
            raise ValueError("Invalid count or zero steady duration; increase repetitions above timer resolution")
        if not math.isfinite(reported) or abs(reported - elapsed / reps) > .00051:
            raise ValueError("Reported timing disagrees with raw duration")
        if row["phase"] == "model_init":
            if row["kernel"] != "shared" or reps != 1:
                raise ValueError("Invalid initialization record")
            continue
        for j, target in enumerate(expected):
            actual = float(row["output_%d" % j])
            if not math.isfinite(actual) or abs(actual - target) > 1e-5 + 3e-5 * abs(target):
                raise ValueError("Measured executable full output disagrees with independent oracle")
        quotient, remainder = divmod(reps, len(expected))
        gold = quotient * math.fsum(expected) + math.fsum(expected[:remainder])
        checksum = float(row["checksum"])
        if not math.isfinite(checksum) or abs(checksum - gold) > 1e-5 * reps + 3e-5 * abs(gold):
            raise ValueError("Measured executable checksum disagrees with independent oracle")


def summarize(rows, kernels=("output_first", "input_first")):
    result = {}
    groups = {}
    for name in kernels:
        selected = [r for r in rows if r["phase"] == "steady" and r["kernel"] == name]
        values = [int(r["elapsed_ns"]) / int(r["repetitions"]) for r in selected]
        if not values or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Missing or invalid timing samples")
        result[name] = {"samples": len(values), "repetitions": sorted({int(r["repetitions"]) for r in selected}),
                        "median_ns": statistics.median(values), "p95_ns": percentile(values, .95),
                        "p99_ns": percentile(values, .99), "min_ns": min(values), "max_ns": max(values)}
    for row in rows:
        if row["phase"] == "steady":
            groups.setdefault((row.get("run", "0"), row["sample"]), []).append(row)
    result["ratios"] = {}
    for fast in kernels[1:]:
        ratios, by_order = [], {}
        for group in groups.values():
            if len(group) != len(kernels) or {r["kernel"] for r in group} != set(kernels):
                raise ValueError("Incomplete matched group")
            times = {r["kernel"]: int(r["elapsed_ns"]) / int(r["repetitions"]) for r in group}
            value = times[kernels[0]] / times[fast]
            ratios.append(value)
            order = ",".join(r["kernel"] for r in group)
            by_order.setdefault(order, []).append(value)
        result["ratios"]["output_over_" + fast] = {
            "ratio_of_pooled_medians": result[kernels[0]]["median_ns"] / result[fast]["median_ns"],
            "median_matched_ratio": statistics.median(ratios),
            "matched_ratio_by_order": {k: statistics.median(v) for k, v in by_order.items()}}
    if "system_blas" in kernels:
        result["ratios"]["input_over_system_blas"] = {
            "ratio_of_pooled_medians": result["input_first"]["median_ns"] / result["system_blas"]["median_ns"]}
    return result
