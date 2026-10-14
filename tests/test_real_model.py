"""Optional real-model regression: use actual C and reject corrupt exports/outputs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

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
        for defect in ['overlap', 'shape', 'nan', 'dataset', 'class_mapping', 'weight', 'seed', 'preprocessing', 'training_parameters']:
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
            elif defect == 'weight':
                data['biases'][1][0] += 1
            elif defect == 'seed':
                data['seed'] += 1
            elif defect == 'preprocessing':
                data['preprocessing'] = 'fit scaling on all images'
            else:
                data['training_parameters']['early_stopping'] = True
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.json'
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError), threadpool_limits(limits=1):
                    rm.evaluate(LIBRARY, path)

    def test_public_adapter_inputs(self):
        with self.assertRaises(ValueError):
            self.engine.predict_proba(np.zeros((2, 64), np.float32), 0)
        bad = np.zeros((1, 64), np.float32)
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.engine.predict_proba(bad, 0)
        with self.assertRaises(ValueError):
            self.engine.predict_proba(np.zeros((1, 63), np.float32), 0)

    def test_duplicate_image_guard(self):
        x, _, _ = rm.dataset()
        model = json.loads(rm.MODEL.read_text())
        x[model['test_ids'][0]] = x[model['train_ids'][0]]
        with self.assertRaisesRegex(ValueError, 'duplicate image'):
            rm.split_audit(x, model['train_ids'], model['test_ids'])

    def test_training_isolated_from_heldout_features_and_labels(self):
        x, y, sha = rm.dataset()
        artifact = json.loads(rm.MODEL.read_text())
        tr, te = artifact['train_ids'], artifact['test_ids']
        actual_fit = rm.MLPClassifier.fit
        seen = []
        def fit_spy(estimator, features, labels, **kwargs):
            np.testing.assert_array_equal(features, x[tr])
            np.testing.assert_array_equal(labels, y[tr])
            seen.append(len(labels))
            return actual_fit(estimator, features, labels, **kwargs)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[te] += 10
        changed_y[te] = (changed_y[te] + 1) % 10
        with tempfile.TemporaryDirectory() as temp:
            artifacts = []
            for i, (features, labels) in enumerate([(x, y), (changed_x, changed_y)]):
                path = Path(temp) / ('trained-%d.json' % i)
                # Hold the split fixed: stratified splitting legitimately uses labels.
                with mock.patch.object(rm, 'dataset', return_value=(features, labels, sha)), \
                     mock.patch.object(rm, 'train_test_split', return_value=(np.array(tr), np.array(te))), \
                     mock.patch.object(rm.MLPClassifier, 'fit', new=fit_spy):
                    rm.train(path)
                artifacts.append(json.loads(path.read_text()))
            for key in ['weights', 'biases', 'iterations', 'training_loss']:
                self.assertEqual(artifacts[0][key], artifacts[1][key])
        self.assertEqual(seen, [1347, 1347])


if __name__ == '__main__':
    unittest.main()
