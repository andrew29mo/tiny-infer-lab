"""Adversarial synthetic fixtures for publication checks, not timing evidence."""
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from benchmark_stats import summarize
from check_publication import check as scan_repository, findings
from run_benchmark import optional
from verify_report import source_names, verify
from workloads import arrays
sys.path.insert(0, str(ROOT / "tests"))
from test_reference import oracle


class PublicationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        shape, kernels = [2, 3, 2], ["output_first", "input_first"]
        expected = oracle(*arrays(shape), 3, 2)
        self.folder = self.directory / "2x3x2"
        self.folder.mkdir()

        def row(phase, sample, kernel, reps, elapsed):
            quotient, remainder = divmod(reps, 2)
            result = {"phase": phase, "sample": str(sample), "kernel": kernel,
                      "repetitions": str(reps), "elapsed_ns": str(elapsed),
                      "ns_per_inference": str(elapsed / reps),
                      "checksum": str(quotient * math.fsum(expected) + math.fsum(expected[:remainder]))}
            result.update({"output_%d" % i: str(value) for i, value in enumerate(expected)})
            return result
        rows = [row("model_init", 0, "shared", 1, 10), row("first_call", 0, kernels[0], 1, 20),
                row("first_call", 0, kernels[1], 1, 10)]
        for sample in range(2):
            pair = [row("steady", sample, kernels[0], 10, 200), row("steady", sample, kernels[1], 10, 100)]
            rows += pair if sample == 0 else list(reversed(pair))
        self.write_csv("run-00.csv", rows)
        self.write_csv("samples.csv", [dict(r, run="0") for r in rows])
        (self.folder / "run-00.stderr.txt").write_text("clock_resolution_ns=1\n")
        stats = summarize(rows)
        pooled = copy.deepcopy(stats)
        for ratio in pooled["ratios"].values():
            ratio["process_ratio_min"] = ratio["process_ratio_max"] = ratio["ratio_of_pooled_medians"]
        self.report = {
            "schema_version": 3, "source_git_commit": None, "source_git_status": None,
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_names(ROOT)},
            "method": {"runs_per_shape": 1, "samples_per_kernel_per_run": 2,
                       "seed": 20261014, "dtype": "float32", "batch_size": 1},
            "blas": {"requested": False},
            "data_sha256": {"2x3x2/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.folder.iterdir()},
            "workloads": [{"name": "2x3x2", "shape": shape, "kernels": kernels,
                           "weights_bytes": 68, "workspace_bytes": 12,
                           "correctness": {"passed": True, "absolute_tolerance": 1e-5,
                                           "relative_tolerance": 3e-5, "expected_output_float64": expected},
                           "runs": [{"run": 0, "order_index": 0, "clock_reported_resolution_ns": 1,
                                     "command": ["build/measured_benchmark", "2", "10", "0", "2", "3", "2"],
                                     "summary": stats}], "summary": pooled}]}

    def write_csv(self, name, rows):
        with (self.folder / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def verify(self, report):
        (self.directory / "report.json").write_text(json.dumps(report))
        with contextlib.redirect_stdout(io.StringIO()):
            verify(self.directory)

    def test_valid_fixture(self):
        self.verify(self.report)

    def test_empty_incomplete_or_inconsistent_reports_rejected(self):
        invalid = [{"schema_version": 3, "source_sha256": {}, "data_sha256": {}, "workloads": []}]
        for field, value in (("source_sha256", {}), ("data_sha256", {}), ("workloads", [])):
            bad = copy.deepcopy(self.report)
            bad[field] = value
            invalid.append(bad)
        for change in ("runs", "samples", "oracle", "nan", "shape", "duplicate", "threads", "timer", "command", "memory", "source"):
            bad = copy.deepcopy(self.report)
            w = bad["workloads"][0]
            if change == "runs": bad["method"]["runs_per_shape"] = 2
            elif change == "samples": bad["method"]["samples_per_kernel_per_run"] = 3
            elif change == "oracle": w["correctness"]["expected_output_float64"][0] += 1
            elif change == "nan": w["correctness"]["expected_output_float64"][0] = float("nan")
            elif change == "shape": w["name"] = "../outside"
            elif change == "duplicate": bad["workloads"].append(copy.deepcopy(w))
            elif change == "threads": bad["blas"] = {"requested": True, "single_thread_confirmed": False}
            elif change == "timer": w["runs"][0]["clock_reported_resolution_ns"] = 2
            elif change == "command": w["runs"][0]["command"][4] = "99"
            elif change == "memory": w["weights_bytes"] = 0
            elif change == "source": bad["source_sha256"]["src/tiny_infer.c"] = "0" * 64
            invalid.append(bad)
        for bad in invalid:
            with self.subTest(case=invalid.index(bad)), self.assertRaises(ValueError):
                self.verify(bad)

    def test_optional_missing_metadata_command(self):
        self.assertIsNone(optional([str(self.directory / "missing-executable")]))

    def test_targeted_private_content_rules(self):
        examples = [(".env", b"value"), ("AGENTS.md", b"local"), ("build/a", b"binary"),
                    ("results/local/log.txt", b"local"), ("src/a.c", ("/" + "Users/example/file").encode()),
                    ("notes.txt", ("gh" + "p_" + "x" * 36).encode()),
                    ("id.pem", ("-----BEGIN " + "PRIVATE KEY-----").encode()),
                    ("a", b"\x7fELFcontent")]
        for name, data in examples:
            self.assertTrue(findings(name, data), name)
        self.assertFalse(findings("src/a.c", b"int value = 0;"))

    def test_deleted_local_path_detected_even_when_blob_is_reused(self):
        repo = self.directory / "repository"
        repo.mkdir()
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=repo, stderr=subprocess.DEVNULL)
        git("init", "-b", "main")
        git("config", "user.name", "Fixture")
        git("config", "user.email", "fixture@example.invalid")
        (repo / ".env").write_text("same harmless contents\n")
        git("add", ".env")
        git("commit", "-m", "fixture one")
        git("mv", ".env", "safe.txt")
        git("commit", "-m", "fixture two")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            scan_repository(repo)


if __name__ == "__main__":
    unittest.main()
