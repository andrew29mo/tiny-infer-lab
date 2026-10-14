"""Verify a complete schema-3 report against its source snapshot and raw CSV."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from benchmark_stats import summarize, validate_samples
from workloads import KERNELS, arrays, parse_shapes, shape_name

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_reference import oracle


def source_names(root, revision=None):
    extensions = {"src": {".c", ".h"}, "include": {".h"}, "examples": {".c"},
                  "tools": {".py", ".c"}, "tests": {".py", ".c"}}
    if revision:
        names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision], cwd=root, text=True).splitlines()
    else:
        names = [str(p.relative_to(root)) for folder in extensions for p in (root / folder).glob("*") if p.is_file()]
        names.append("Makefile")
    return {name for name in names if name == "Makefile" or
            (len(PurePosixPath(name).parts) == 2 and PurePosixPath(name).parts[0] in extensions
             and PurePosixPath(name).suffix in extensions[PurePosixPath(name).parts[0]])}


def local_path(root, relative):
    if not isinstance(relative, str):
        raise ValueError("Manifest path must be a string")
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative or str(path) != relative:
        raise ValueError("Unsafe manifest path")
    result = root / relative
    try:
        result.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Manifest path escapes its directory") from error
    return result


def verify_sources(report, root):
    manifest = report.get("source_sha256")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Missing source manifest")
    revision = report.get("source_git_commit")
    use_revision = report.get("source_git_status") == "" and bool(revision)
    if use_revision:
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Invalid measured source revision")
        result = subprocess.run(["git", "cat-file", "-e", revision + "^{commit}"], cwd=root,
                                capture_output=True)
        if result.returncode:
            raise ValueError("Measured revision unavailable; use a full Git clone or fetch its history")
    else:
        revision = None
    if set(manifest) != source_names(root, revision):
        raise ValueError("Incomplete source manifest")
    for relative, digest in manifest.items():
        path = local_path(root, relative)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid source digest")
        if revision:
            data = subprocess.check_output(["git", "show", revision + ":" + relative], cwd=root)
        else:
            data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Source snapshot hash mismatch: " + relative)
    return revision or "current files (uncommitted/no Git snapshot)"


def verify(directory, root=ROOT):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    if report.get("schema_version") != 3:
        raise ValueError("Unsupported or missing report schema")
    source = verify_sources(report, root)
    method = report.get("method", {})
    runs, samples = method.get("runs_per_shape"), method.get("samples_per_kernel_per_run")
    if type(runs) is not int or not 1 <= runs <= 100 or type(samples) is not int or not 2 <= samples <= 10000:
        raise ValueError("Missing or invalid run/sample counts")
    if method.get("seed") != 20261014 or method.get("dtype") != "float32" or method.get("batch_size") != 1:
        raise ValueError("Unsupported deterministic workload definition")
    blas = report.get("blas", {})
    if type(blas.get("requested")) is not bool:
        raise ValueError("Missing BLAS selection")
    if blas["requested"] and (blas.get("single_thread_confirmed") is not True or blas.get("provider") in (None, "disabled")):
        raise ValueError("BLAS thread/provider evidence missing")
    kernels = KERNELS if blas["requested"] else KERNELS[:2]
    workloads = report.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("Empty workload suite")
    names, expected_files = set(), set()
    for workload in workloads:
        shape = workload.get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or any(type(x) is not int for x in shape):
            raise ValueError("Invalid workload shape")
        name = shape_name(parse_shapes(shape_name(shape))[0])
        if workload.get("name") != name or name in names or tuple(workload.get("kernels", ())) != kernels:
            raise ValueError("Duplicate shape or inconsistent workload/backends")
        names.add(name)
        run_records = workload.get("runs", [])
        if [r.get("run") for r in run_records] != list(range(runs)):
            raise ValueError("Missing, duplicate or out-of-order process runs")
        ni, nh, no = shape
        # Regenerate answers, rather than trusting the report's stored oracle.
        x, w1, b1, w2, b2 = arrays(shape)
        expected = oracle(x, w1, b1, w2, b2, nh, no)
        correctness = workload.get("correctness", {})
        recorded = correctness.get("expected_output_float64", [])
        if (correctness.get("passed") is not True or correctness.get("absolute_tolerance") != 1e-5
                or correctness.get("relative_tolerance") != 3e-5 or len(recorded) != no
                or any(not math.isfinite(a) or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                       for a, b in zip(recorded, expected))):
            raise ValueError("Stored oracle does not match regenerated reference")
        if workload.get("weights_bytes") != (ni * nh + nh + nh * no + no) * 4 or workload.get("workspace_bytes") != nh * 4:
            raise ValueError("Incorrect memory footprint")
        expected_files.add(name + "/samples.csv")
        for run in range(runs):
            expected_files.update({name + "/run-%02d.csv" % run, name + "/run-%02d.stderr.txt" % run})
    manifest = report.get("data_sha256", {})
    if set(manifest) != expected_files:
        raise ValueError("Incomplete raw-data manifest")
    for relative, digest in manifest.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid data digest")
        if hashlib.sha256(local_path(directory, relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Raw data hash mismatch: " + relative)
    for workload in workloads:
        folder = local_path(directory, workload["name"])
        expected = workload["correctness"]["expected_output_float64"]
        combined = []
        for run in workload["runs"]:
            if run.get("order_index") != run["run"] % 6:
                raise ValueError("Invalid process ordering")
            resolution = (folder / ("run-%02d.stderr.txt" % run["run"])).read_text().strip()
            if not re.fullmatch(r"clock_resolution_ns=[1-9][0-9]*", resolution):
                raise ValueError("Missing timer resolution")
            if int(resolution.split("=")[1]) != run.get("clock_reported_resolution_ns"):
                raise ValueError("Timer resolution mismatch")
            command = run.get("command", [])
            if len(command) != 7 or command[1] != str(samples) or command[3] != str(run["order_index"]) or command[4:] != list(map(str, workload["shape"])):
                raise ValueError("Benchmark command/shape/count mismatch")
            with (folder / ("run-%02d.csv" % run["run"])).open() as file:
                rows = list(csv.DictReader(file))
            validate_samples(rows, samples, expected, kernels, run["order_index"])
            if command[2] != "auto" and any(int(r["repetitions"]) != int(command[2]) for r in rows if r["phase"] == "steady"):
                raise ValueError("Explicit repetition count mismatch")
            if summarize(rows, kernels) != run["summary"]:
                raise ValueError("Per-process statistics mismatch")
            combined += [dict(row, run=str(run["run"])) for row in rows]
        with (folder / "samples.csv").open() as file:
            if list(csv.DictReader(file)) != combined:
                raise ValueError("Combined CSV differs from process CSVs")
        actual = summarize(combined, kernels)
        for kernel in kernels:
            if actual[kernel] != workload["summary"][kernel]:
                raise ValueError("Pooled timing statistics mismatch")
        if set(actual["ratios"]) != set(workload["summary"]["ratios"]):
            raise ValueError("Missing or extra ratio statistics")
        for name, ratio in actual["ratios"].items():
            for key, value in ratio.items():
                if workload["summary"]["ratios"][name][key] != value:
                    raise ValueError("Pooled ratio mismatch")
            process = [r["summary"]["ratios"][name]["ratio_of_pooled_medians"] for r in workload["runs"]]
            recorded = workload["summary"]["ratios"][name]
            if recorded["process_ratio_min"] != min(process) or recorded["process_ratio_max"] != max(process):
                raise ValueError("Process ratio range mismatch")
    print("PASS: complete report, regenerated oracle, raw-data hashes, outputs and recomputed statistics; source=" + source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
