"""Documentation regressions must reject stale numbers and source attribution."""
import contextlib
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check_documentation import check


class DocumentationTests(unittest.TestCase):
    def test_current_documentation(self):
        with contextlib.redirect_stdout(io.StringIO()):check(ROOT)

    def test_modified_claims_are_rejected(self):
        cases=[('README.md','**11.90×**','**12.24×**'),
               ('README.md','438/450 (97.33%)','450/450 (100.00%)'),
               ('README.md','**5.50×**','**5.49×**'),
               ('docs/RESULTS.md','| 128→256→32 | 15.990','| 128→256→32 | 1.000'),
               ('docs/REAL_MODEL.md','| 4.746 |','| 0.001 |'),
               ('docs/BATCH_INFERENCE.md','| 450 | 0.366 |','| 450 | 0.001 |'),
               ('docs/BATCH_INFERENCE.md','source `1743574`','source `0000000`'),
               ('README.md','](LICENSE)','](MISSING_LICENSE)')]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ['README.md','CONTRIBUTING.md','LICENSE']:
                shutil.copyfile(ROOT/name,root/name)
            for folder in ['docs','models','.github','include']:
                shutil.copytree(ROOT/folder,root/folder)
            # Reuse immutable evidence without repeatedly copying large JSON outputs.
            (root/'results').mkdir()
            (root/'results/published').symlink_to(ROOT/'results/published',target_is_directory=True)
            # A valid copied fixture must pass first: otherwise any unrelated
            # missing link could make every corrupt-claim case falsely pass.
            with contextlib.redirect_stdout(io.StringIO()):check(root)
            for name,before,after in cases:
                with self.subTest(name=name,claim=before):
                    path=root/name;original=path.read_text()
                    self.assertIn(before,original)
                    path.write_text(original.replace(before,after,1))
                    try:
                        with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):check(root)
                    finally:path.write_text(original)


if __name__=='__main__':unittest.main()
