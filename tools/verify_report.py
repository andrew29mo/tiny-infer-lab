"""Independently re-read persisted CSV and verify published report consistency."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from benchmark_stats import summarize, validate_samples

ROOT = Path(__file__).resolve().parents[1]


def verify(directory):
    report = json.loads((directory / "report.json").read_text())
    for relative, digest in report["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Source hash differs (check out the measured source revision): " + relative)
    for relative, digest in report["data_sha256"].items():
        if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Raw data hash mismatch: " + relative)
    for workload in report["workloads"]:
        folder = directory / workload["name"]
        combined = []
        for run in workload["runs"]:
            with (folder / ("run-%02d.csv" % run["run"])).open() as f:
                rows = list(csv.DictReader(f))
            validate_samples(rows, report["method"]["samples_per_kernel_per_run"],
                             workload["correctness"]["expected_output_float64"],
                             tuple(workload["kernels"]), run["order_index"])
            if summarize(rows, tuple(workload["kernels"])) != run["summary"]:
                raise ValueError("Per-process statistics mismatch")
            combined += [dict(r, run=str(run["run"])) for r in rows]
        with (folder / "samples.csv").open() as f:
            if list(csv.DictReader(f)) != combined:
                raise ValueError("Combined CSV differs from process CSVs")
        actual = summarize(combined, tuple(workload["kernels"]))
        for kernel in workload["kernels"]:
            if actual[kernel] != workload["summary"][kernel]:
                raise ValueError("Pooled timing statistics mismatch")
        for name, ratio in actual["ratios"].items():
            for key, value in ratio.items():
                if workload["summary"]["ratios"][name][key] != value:
                    raise ValueError("Pooled ratio mismatch")
            process = [r["summary"]["ratios"][name]["ratio_of_pooled_medians"] for r in workload["runs"]]
            recorded = workload["summary"]["ratios"][name]
            if recorded["process_ratio_min"] != min(process) or recorded["process_ratio_max"] != max(process):
                raise ValueError("Process ratio range mismatch")
    print("PASS: persisted source/data hashes, full outputs, matched work and recomputed per-process/pooled statistics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
