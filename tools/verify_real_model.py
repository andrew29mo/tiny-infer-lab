"""Recompute archived digits correctness, raw timing summaries and provenance."""
import json
import itertools
import re
from pathlib import Path
import subprocess
import sys
import numpy as np
import real_model as rm


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    rm.require(manifest['schema'] in (1, 2) and type(manifest['runs']) is int
               and 1 <= manifest['runs'] <= 24 and type(manifest['samples']) is int
               and 1 <= manifest['samples'] <= 100,
               'empty/invalid manifest')
    rm.require(set(manifest['sources']) == set(rm.LEGACY_SOURCES if manifest['schema'] == 1 else rm.SOURCES), 'incomplete source evidence')
    rm.require(re.fullmatch('[0-9a-f]{40}', manifest['source_revision']) is not None,
               'invalid source revision')
    for sha in list(manifest['sources'].values()) + [manifest['library_sha256']]:
        rm.require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha) is not None, 'invalid source/library hash')
    if manifest['schema'] == 2:
        rm.require(manifest['protocol'] == 'single-image-array-api-v2', 'timing protocol mismatch')
    expected_files = {'correctness.json'} | {'run-%02d.json' % i for i in range(manifest['runs'])}
    rm.require(set(manifest['files']) == expected_files, 'missing/extra evidence files')
    for name, sha in manifest['files'].items():
        rm.require(Path(name).name == name and rm.digest(directory / name) == sha, 'raw evidence hash')
    for name, sha in manifest['sources'].items():
        blob = subprocess.check_output(['git', 'show', manifest['source_revision'] + ':' + name], cwd=rm.ROOT)
        rm.require(rm.hashlib.sha256(blob).hexdigest() == sha, 'historical source mismatch: ' + name)
    build = manifest['build_metadata']
    rm.require(set(build['sources']) == set(rm.BUILD_SOURCES) and
               set(build['objects']) == set(rm.OBJECTS), 'incomplete build evidence')
    rm.require(isinstance(build['compiler'], str) and bool(build['compiler']) and
               len(build['commands']) == 4 and all(isinstance(c, list) and c for c in build['commands']),
               'missing build commands/compiler')
    rm.require(all(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha) is not None
                   for sha in build['objects'].values()), 'invalid object hashes')
    rm.require(build['library_sha256'] == manifest['library_sha256'], 'library/build mismatch')
    rm.require(all(manifest['sources'][p] == sha for p, sha in build['sources'].items()), 'build source mismatch')
    data, x, y, clf = rm.restore(rm.MODEL)
    checks = json.loads((directory / 'correctness.json').read_text())
    rm.require(checks['model_sha256'] == rm.digest(rm.MODEL) and
               checks['dataset_sha256'] == data['dataset_sha256'] and
               checks['test_ids'] == data['test_ids'] and checks['labels'] == y.tolist(), 'model/dataset mismatch')
    if manifest['schema'] == 2:
        rm.require(checks['split_audit'] == rm.split_audit(rm.dataset()[0], data['train_ids'], data['test_ids']),
                   'split audit mismatch')
    reference = clf.predict_proba(x)
    rm.require(np.allclose(reference, data['export_reference_probabilities'], atol=rm.ATOL, rtol=rm.RTOL),
               'original model export mismatch')
    rm.require(np.allclose(reference, checks['reference_probabilities'], atol=rm.ATOL, rtol=rm.RTOL),
               'stored sklearn reference mismatch')
    # Historical error magnitudes use the recorded reference; regenerated vendor
    # float32 results may differ by a few ULPs across platforms/providers.
    reference = np.array(checks['reference_probabilities'])
    w1, w2 = [a.ravel().tolist() for a in clf.coefs_]
    b1, b2 = [a.tolist() for a in clf.intercepts_]
    logits = np.array([rm.oracle(row.tolist(), w1, b1, w2, b2, 32, 10) for row in x])
    rm.require(np.array_equal(logits, checks['oracle_logits']), 'stored float64 oracle mismatch')
    names = set(checks['metrics'])
    rm.require(names in ({'output_first', 'input_first'}, {'output_first', 'input_first', 'system_blas'}),
               'missing backends')
    rm.require(set(checks['outputs']) == names, 'output backend mismatch')
    for name in names:
        out = checks['outputs'][name]
        probs = np.array(out['probabilities'])
        actual = np.array(out['logits'])
        rm.validate_outputs(actual, probs, logits, reference)
        pred = probs.argmax(1)
        metrics = checks['metrics'][name]
        confusion = np.zeros((10, 10), dtype=int)
        np.add.at(confusion, (y, pred), 1)
        rm.require(metrics['correct'] == int((pred == y).sum()) and metrics['total'] == len(y)
                   and metrics['accuracy'] == float(np.mean(pred == y)) >= 0.94
                   and metrics['agreement'] == 1.0 and metrics['confusion_matrix'] == confusion.tolist()
                   and metrics['max_logit_error'] == float(np.max(abs(actual - logits)))
                   and metrics['max_probability_error'] == float(np.max(abs(probs - reference))),
                   'incorrect stored accuracy/error statistics')
    rm.require(checks['sklearn_accuracy'] == float(np.mean(reference.argmax(1) == y)), 'reference accuracy')
    primary = [n for n in ['output_first', 'input_first', 'system_blas'] if n in names] + ['sklearn_single']
    names |= {'sklearn_single', 'sklearn_batch'}
    if manifest['schema'] == 2:
        names.add('numpy_prepared_single')
    rm.require(set(manifest['summaries']) == names, 'summary backend mismatch')
    medians = {name: [] for name in names}
    for i in range(manifest['runs']):
        run = json.loads((directory / ('run-%02d.json' % i)).read_text())
        rm.require(run['run'] == i and run['schema'] == manifest['schema'], 'run identity mismatch')
        rm.require(bool(run['threadpools']) and
                   any(p['user_api'] == 'blas' for p in run['threadpools']) and
                   all(type(p['num_threads']) is int and p['num_threads'] == 1 for p in run['threadpools']),
                   'missing/uncontrolled thread policy')
        rm.require(set(run['versions']) == {'python', 'numpy', 'scipy', 'sklearn'} and
                   all(isinstance(v, str) and bool(v) for v in run['versions'].values()), 'missing package versions')
        rm.require(isinstance(run['c_blas'], str) and bool(run['c_blas']), 'missing C provider')
        if 'system_blas' in names:
            rm.require(run['c_single_thread'], 'C BLAS threads')
        rows = run['raw']
        rm.require(len(rows) == len(names) * manifest['samples'], 'missing samples')
        rm.require({(s['backend'], s['sample']) for s in rows} ==
                   {(n, j) for n in names for j in range(manifest['samples'])}, 'duplicate/missing sample')
        orders = list(itertools.permutations(primary))
        expected_order = []
        for j in range(manifest['samples']):
            order = list(orders[(i * manifest['samples'] + j) % len(orders)])
            order.insert((j + i) % (len(order) + 1), 'sklearn_batch')
            if manifest['schema'] == 2:
                order.insert((j * 2 + i) % (len(order) + 1), 'numpy_prepared_single')
            expected_order.extend((n, j) for n in order)
        rm.require([(s['backend'], s['sample']) for s in rows] == expected_order, 'timing order mismatch')
        for row in rows:
            rm.require(type(row['elapsed_ns']) is int and type(row['sample']) is int, 'invalid timing types')
            rm.require(row['predictions'] == len(y) and row['elapsed_ns'] > 0 and
                       row['ns_per_prediction'] == row['elapsed_ns'] / len(y), 'invalid timing')
        for name in names:
            medians[name].append(float(np.median([s['ns_per_prediction'] for s in rows if s['backend'] == name])))
    for name, values in medians.items():
        rm.require(manifest['summaries'][name] == dict(median_ns=float(np.median(values)),
                   process_medians_ns=values, min_process_median_ns=min(values),
                   max_process_median_ns=max(values)), 'summary mismatch')
    print('PASS: digits accuracy, outputs, raw timings, summaries and historical source hashes')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        verify(sys.argv[1])
