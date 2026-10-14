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
REVIEWED=ROOT/'results/published/providers-reviewed-20261014'


class ReportTests(unittest.TestCase):
    def test_archived_report(self):
        for directory in (PUBLISHED,REVIEWED):
            with self.subTest(directory=directory.name),contextlib.redirect_stdout(io.StringIO()):verify(directory)

    def test_rehashed_corruptions_rejected(self):
        cases=[('run-00.json',lambda r:r['raw'].pop()),
               ('run-00.json',lambda r:r['raw'][0].update(predictions=449)),
               ('correctness.json',lambda r:r['outputs']['128']['emlearn'].update(correct=450)),
               ('correctness.json',lambda r:r['outputs']['128']['emlearn']['probabilities'][-1].__setitem__(0,.5)),
               ('memory.json',lambda r:r[0].update(incremental_warm_rss_bytes=999)),
               ('deployment.json',lambda r:r['providers']['onnxruntime'].update(artifact_bytes=1)),
               ('deployment.json',lambda r:r['providers']['onnxruntime'].update(
                    additional_packages=[],additional_package_bytes=0))]
        for filename,change in cases:
            with self.subTest(file=filename),tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp)/'report';shutil.copytree(PUBLISHED,directory)
                with contextlib.redirect_stdout(io.StringIO()):verify(directory)
                value=json.loads((directory/filename).read_text());change(value)
                rm.save(directory/filename,value)
                m=json.loads((directory/'manifest.json').read_text())
                m['files'][filename]=rm.digest(directory/filename);rm.save(directory/'manifest.json',m)
                with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):verify(directory)

    def test_reviewed_observations_and_raw_inventories(self):
        def balanced_size(r):
            r['packages']['onnxruntime']['bytes']=1
            for row in r['providers'].values():
                row['additional_package_bytes']=sum(r['packages'][n]['bytes'] for n in row['additional_packages'])
        def raw_file(r):
            files=r['package_files']['onnxruntime'];files[next(iter(files))]['bytes']+=1
        def wrong_input(r):
            r['observations']['128']['emlearn']['input_sha256']='0'*64
        def artifact_size(r):
            r['artifact_bytes']['tiny.so']=1
            for row in r['providers'].values():
                row['artifact_bytes']=sum(r['artifact_bytes'][n] for n in row['artifacts'])
        cases=[('deployment.json',balanced_size),('deployment.json',raw_file),
               ('deployment.json',artifact_size),('deployment.json',lambda r:r.pop('package_files')),
               ('run-00.json',wrong_input),('correctness.json',wrong_input),
               ('memory.json',lambda r:r[0].update(rss_phase='after_validation')),
               ('memory.json',lambda r:r[0].update(input_sha256='0'*64))]
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'report';shutil.copytree(REVIEWED,directory)
            with contextlib.redirect_stdout(io.StringIO()):verify(directory)
            manifest=json.loads((directory/'manifest.json').read_text())
            for filename,change in cases:
                with self.subTest(file=filename,case=change.__name__):
                    value=json.loads((REVIEWED/filename).read_text());change(value)
                    rm.save(directory/filename,value)
                    m=json.loads(json.dumps(manifest));m['files'][filename]=rm.digest(directory/filename)
                    rm.save(directory/'manifest.json',m)
                    try:
                        with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):verify(directory)
                    finally:
                        shutil.copyfile(REVIEWED/filename,directory/filename)
                        rm.save(directory/'manifest.json',manifest)


if __name__=='__main__':unittest.main()
