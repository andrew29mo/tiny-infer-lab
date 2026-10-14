"""Corrupt evidence AFTER rehashing it: semantic checks must still reject it."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import real_model as rm
from verify_providers import verify

PUBLISHED=ROOT/'results/published/providers-20261014'


class ReportTests(unittest.TestCase):
    def test_archived_report(self):
        with contextlib.redirect_stdout(io.StringIO()):verify(PUBLISHED)

    def test_rehashed_corruptions_rejected(self):
        cases=[('run-00.json',lambda r:r['raw'].pop()),
               ('run-00.json',lambda r:r['raw'][0].update(predictions=449)),
               ('correctness.json',lambda r:r['outputs']['128']['emlearn'].update(correct=450)),
               ('correctness.json',lambda r:r['outputs']['128']['emlearn']['probabilities'][-1].__setitem__(0,.5)),
               ('memory.json',lambda r:r[0].update(incremental_warm_rss_bytes=999)),
               ('deployment.json',lambda r:r['providers']['onnxruntime'].update(artifact_bytes=1))]
        for filename,change in cases:
            with self.subTest(file=filename),tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp)/'report';shutil.copytree(PUBLISHED,directory)
                with contextlib.redirect_stdout(io.StringIO()):verify(directory)
                value=json.loads((directory/filename).read_text());change(value)
                rm.save(directory/filename,value)
                m=json.loads((directory/'manifest.json').read_text())
                m['files'][filename]=rm.digest(directory/filename);rm.save(directory/'manifest.json',m)
                with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):verify(directory)


if __name__=='__main__':unittest.main()
