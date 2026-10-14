"""Build once, verify production objects, and measure a multi-shape suite."""
import argparse
import csv
import ctypes as C
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_reference import Model, load, oracle
from benchmark_stats import percentile, summarize, validate_samples
from workloads import DEFAULT_SHAPES, KERNELS, arrays, compile_commands, parse_shapes, shape_name


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optional(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def exact_workload_check(lib, shape, kernels):
    ni, nh, no = shape
    x, w1, b1, w2, b2 = arrays(shape)
    expected = oracle(x, w1, b1, w2, b2, nh, no)
    model = Model()
    if lib.ti_model_init(C.byref(model), ni, nh, no, w1, len(w1), b1, nh, w2, len(w2), b2, no):
        raise RuntimeError("Model initialization failed")
    errors = {}
    for kernel in range(len(kernels)):
        h, y = (C.c_float * nh)(), (C.c_float * no)()
        if lib.ti_forward(C.byref(model), x, ni, h, nh, y, no, kernel):
            raise RuntimeError("Inference failed")
        worst = 0
        for actual, target in zip(y, expected):
            error = abs(actual - target)
            worst = max(worst, error)
            if not math.isfinite(actual) or error > 1e-5 + 3e-5 * abs(target):
                raise RuntimeError("Independent oracle mismatch: " + kernels[kernel])
        errors[kernels[kernel]] = worst
    return {"passed": True, "oracle": "Python float64 math.fsum", "absolute_tolerance": 1e-5,
            "relative_tolerance": 3e-5, "worst_absolute_error_by_kernel": errors,
            "expected_output_float64": expected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--cflags", default="-std=c11 -O3 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off")
    parser.add_argument("--blas", type=int, choices=(0, 1), default=0)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--shapes", default=",".join(shape_name(s) for s in DEFAULT_SHAPES))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        shapes = parse_shapes(args.shapes)
    except ValueError as error:
        parser.error(str(error))
    if not 2 <= args.samples <= 10000 or not 1 <= args.runs <= 100 or (args.repetitions is not None and not 1 <= args.repetitions <= 1000000):
        parser.error("samples: 2..10000; runs: 1..100; repetitions: 1..1000000")
    os.chdir(ROOT)
    Path("build").mkdir(exist_ok=True)
    sources = sorted([*ROOT.glob("src/*.[ch]"), *ROOT.glob("include/*.h"), *ROOT.glob("examples/*.c"),
                      *ROOT.glob("tools/*.py"), *ROOT.glob("tools/*.c"), *ROOT.glob("tests/*.py"),
                      *ROOT.glob("tests/*.c"), ROOT / "Makefile"])
    before = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    commands, objects = compile_commands(args.cc, args.cflags, args.blas)
    for command in commands:
        subprocess.run(command, check=True)
    checks = [["build/measured_test"], [sys.executable, "tests/test_reference.py", "build/measured_engine.so"],
              [sys.executable, "-O", "tests/test_reference.py", "build/measured_engine.so"],
              [sys.executable, "tests/check_no_heap.py", objects[0]]]
    logs = []
    for command in checks:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # Publish portable provenance without the caller's home/interpreter path.
        public_command = ["python3" if x == sys.executable else x for x in command]
        logs.append({"command": public_command, "stdout": result.stdout, "stderr": result.stderr})
    lib = load("build/measured_engine.so")
    if bool(lib.ti_blas_available()) != bool(args.blas):
        raise RuntimeError("Requested BLAS unavailable; refusing a silent fallback")
    kernels = KERNELS if args.blas else KERNELS[:2]
    timestamp = dt.datetime.now(dt.timezone.utc)
    out = args.output_dir or Path("results/local") / timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    out.mkdir(parents=True, exist_ok=False)
    linked = optional(["otool", "-L", "build/measured_benchmark"]) if sys.platform == "darwin" else optional(["ldd", "build/measured_benchmark"])
    metadata = {
        "schema_version": 3, "timestamp_utc": timestamp.isoformat(),
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "cpu": optional(["sysctl", "-n", "machdep.cpu.brand_string"]) if sys.platform == "darwin" else platform.processor(),
                 "logical_cpus": os.cpu_count()},
        "compiler_version": subprocess.check_output(shlex.split(args.cc) + ["--version"], text=True).strip(),
        "compile_commands": commands, "source_sha256": before,
        "artifact_sha256": {p: sha(Path(p)) for p in objects + ["build/measured_benchmark", "build/measured_engine.so", "build/measured_test"]},
        "source_git_commit": optional(["git", "rev-parse", "HEAD"]),
        "source_git_status": optional(["git", "status", "--porcelain", "--untracked-files=no"]),
        "preflight_checks": logs,
        "blas": {"requested": bool(args.blas), "provider": lib.ti_blas_name().decode(),
                 "single_thread_confirmed": bool(lib.ti_blas_is_single_threaded()),
                 "control": "BLASSetThreading/BLASGetThreading" if sys.platform == "darwin" else "OpenBLAS set/get_num_threads",
                 "linked_libraries": linked,
                 "internal_allocations": "not asserted; vendor BLAS may initialize/cache internally"},
        "method": {"runs_per_shape": args.runs, "samples_per_kernel_per_run": args.samples,
                   "clock": "CLOCK_MONOTONIC", "dtype": "float32", "batch_size": 1, "seed": 20261014,
                   "orders": "all permutations, rotating per sample and per process",
                   "shape_order": "rotated between process rounds; not all rounds of one shape first",
                   "warmup": "32 calls per kernel before calibration and 32 after",
                   "repetitions": "identical for all kernels per shape/run; maximum of all calibrations",
                   "calibration": "fastest of 3 trials reaches 2 ms, cap 65536; bypassed with explicit repetitions",
                   "timed": "same ti_forward checks, bias, two dense layers, ReLU, finite checks, checksum/loop overhead",
                   "not_timed": "allocation, generation, model init, oracle, result copies, all CSV output",
                   "validation": "all final output elements per batch plus accumulated checksum against independent oracle",
                   "percentiles": "nearest-rank distribution of batch-average latency; not per-call tail latency",
                   "weights": "same row-major buffers; BLAS sgemv transpose view, no prepacking or format conversion",
                   "floating_point": "C contraction off; vendor BLAS may use FMA/reassociation; numerical tolerance, not bitwise equality",
                   "limitations": "warm fixed synthetic data; no affinity/frequency control; process ranges are not confidence intervals"},
        "workloads": [],
    }
    workloads = {}
    for shape in shapes:
        name = shape_name(shape)
        (out / name).mkdir()
        gold = exact_workload_check(lib, shape, kernels)
        ni, nh, no = shape
        workloads[name] = {"shape": list(shape), "name": name, "kernels": list(kernels),
                           "weights_bytes": (ni * nh + nh + nh * no + no) * 4,
                           "workspace_bytes": nh * 4, "correctness": gold, "runs": [], "rows": []}
    for run in range(args.runs):
        # Rotate workload order too, to spread slow thermal/frequency drift.
        order = shapes[run % len(shapes):] + shapes[:run % len(shapes)]
        for shape in order:
            name = shape_name(shape)
            workload = workloads[name]
            command = ["build/measured_benchmark", str(args.samples),
                       str(args.repetitions) if args.repetitions else "auto", str(run % 6)] + list(map(str, shape))
            measured = subprocess.run(command, check=True, capture_output=True, text=True)
            prefix = out / name / ("run-%02d" % run)
            prefix.with_suffix(".csv").write_text(measured.stdout)
            prefix.with_suffix(".stderr.txt").write_text(measured.stderr)
            rows = list(csv.DictReader(io.StringIO(measured.stdout)))
            validate_samples(rows, args.samples, workload["correctness"]["expected_output_float64"], kernels, run)
            for row in rows:
                row["run"] = str(run)
            workload["rows"].extend(rows)
            resolution = int(measured.stderr.strip().split("clock_resolution_ns=")[-1])
            workload["runs"].append({"run": run, "order_index": run % 6, "command": command,
                                     "clock_reported_resolution_ns": resolution, "summary": summarize(rows, kernels)})
        print("Completed process round %d/%d" % (run + 1, args.runs), flush=True)
    for name, workload in workloads.items():
        rows = workload.pop("rows")
        with (out / name / "samples.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        workload["summary"] = summarize(rows, kernels)
        for key, ratio in workload["summary"]["ratios"].items():
            values = [r["summary"]["ratios"][key]["ratio_of_pooled_medians"] for r in workload["runs"]]
            ratio["process_ratio_min"] = min(values)
            ratio["process_ratio_max"] = max(values)
        metadata["workloads"].append(workload)
    if before != {str(p.relative_to(ROOT)): sha(p) for p in sources}:
        raise RuntimeError("Source changed during measurement; results rejected")
    metadata["data_sha256"] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "report.json").write_text(json.dumps(metadata, indent=2) + "\n")
    subprocess.run([sys.executable, "tools/verify_report.py", str(out)], check=True)
    lines = ["Tiny Infer Lab — multi-shape CPU comparison", "UTC: " + timestamp.isoformat(),
             "Provider: " + metadata["blas"]["provider"],
             "All production objects and measured final outputs verified against independent oracle.",
             "Shape | output_first us | input_first us | BLAS us | output/input | input/BLAS"]
    for w in metadata["workloads"]:
        stats = w["summary"]
        blas_ns = stats["system_blas"]["median_ns"] if args.blas else None
        lines.append("%s | %.3f | %.3f | %s | %.3fx | %s" % (
            w["name"], stats["output_first"]["median_ns"] / 1000, stats["input_first"]["median_ns"] / 1000,
            "%.3f" % (blas_ns / 1000) if blas_ns else "disabled",
            stats["ratios"]["output_over_input_first"]["ratio_of_pooled_medians"],
            "%.3fx" % (stats["input_first"]["median_ns"] / blas_ns) if blas_ns else "n/a"))
    lines += ["Ratios > 1 mean the denominator is faster. Six shapes are not a universal benchmark.",
              "Percentiles describe batch means, not individual-call tail latency."]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("Saved: " + str(out))


if __name__ == "__main__":
    main()
