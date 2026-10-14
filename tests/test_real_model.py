"""Optional real-model regression: use actual C and reject corrupt exports/outputs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import real_model as rm

LIBRARY = sys.argv.pop(1) if len(sys.argv) > 1 else 'build/real_engine.so'


class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with threadpool_limits(limits=1):
            cls.engine, cls.clf, cls.x, cls.checks = rm.evaluate(LIBRARY)

    def test_accuracy_and_agreement(self):
        for metrics in self.checks['metrics'].values():
            self.assertGreaterEqual(metrics['accuracy'], 0.94)
            self.assertEqual(metrics['agreement'], 1.0)
            self.assertEqual(metrics['total'], 450)
            self.assertEqual(sum(map(sum, metrics['confusion_matrix'])), 450)

    def test_corrupted_logits_rejected_even_with_same_classes(self):
        p = np.array(self.checks['reference_probabilities'])
        ref = np.array(self.checks['oracle_logits'])
        for logits, probs in [(ref + 0.1, p), (ref, p[:, ::-1]),
                              (ref * np.nan, p)]:
            with self.assertRaises(ValueError):
                rm.validate_outputs(logits, probs, ref, p)

    def test_bad_export_or_split_rejected(self):
        original = json.loads(rm.MODEL.read_text())
        for defect in ['overlap', 'shape', 'nan', 'dataset', 'class_mapping', 'weight']:
            data = copy.deepcopy(original)
            if defect == 'overlap':
                data['test_ids'][0] = data['train_ids'][0]
            elif defect == 'shape':
                data['weights'][0] = data['weights'][0][:-1]
            elif defect == 'nan':
                data['weights'][0][0][0] = float('nan')
            elif defect == 'dataset':
                data['dataset_sha256'] = 'bad'
            elif defect == 'class_mapping':
                data['classes'].reverse()
            else:
                data['biases'][1][0] += 1
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.json'
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError), threadpool_limits(limits=1):
                    rm.evaluate(LIBRARY, path)


if __name__ == '__main__':
    unittest.main()
