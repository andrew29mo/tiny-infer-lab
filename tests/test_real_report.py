"""Challenge the numerical/timing verifier after bypassing simple file hashes."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import real_model as rm
from verify_real_model import verify

BASELINE = rm.ROOT / 'results/published/digits-20261014'


class ReportTests(unittest.TestCase):
    def test_committed_report(self):
        with threadpool_limits(limits=1):
            verify(BASELINE)

    def test_corrupted_evidence_rejected(self):
        for defect in ['empty', 'missing_sample', 'timing', 'summary', 'reference', 'accuracy']:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp) / 'report'
                shutil.copytree(BASELINE, directory)
                manifest = json.loads((directory / 'manifest.json').read_text())
                run = json.loads((directory / 'run-00.json').read_text())
                checks = json.loads((directory / 'correctness.json').read_text())
                if defect == 'empty':
                    manifest['runs'] = 0
                elif defect == 'missing_sample':
                    run['raw'].pop()
                elif defect == 'timing':
                    run['raw'][0]['elapsed_ns'] *= 2
                elif defect == 'summary':
                    manifest['summaries']['input_first']['median_ns'] += 10
                elif defect == 'reference':
                    checks['reference_probabilities'][0][0] += 0.1
                else:
                    checks['metrics']['input_first']['correct'] = 450
                rm.save(directory / 'run-00.json', run)
                rm.save(directory / 'correctness.json', checks)
                for name in manifest['files']:
                    manifest['files'][name] = rm.digest(directory / name)
                rm.save(directory / 'manifest.json', manifest)
                with self.assertRaises(ValueError), threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()):
                    verify(directory)


if __name__ == '__main__':
    unittest.main()
