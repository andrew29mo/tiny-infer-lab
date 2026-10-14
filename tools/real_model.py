"""Train/export a real digits MLP; check production C against sklearn, then time APIs.

No pickle loading, dataset download, or test-set tuning. Optional dependencies only.
"""
import argparse
import ctypes as C
import datetime as dt
import hashlib
import itertools
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_reference import Model, load, oracle

MODEL = ROOT / 'models/digits/model.json'
SEED = 20261014
ATOL, RTOL = 1e-5, 3e-5


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def dataset():
    data = load_digits()
    # Canonical little-endian raw bytes before preprocessing, including labels.
    sha = hashlib.sha256(data.data.astype('<f4').tobytes() +
                         data.target.astype('<i4').tobytes()).hexdigest()
    return np.ascontiguousarray(data.data / 16, dtype=np.float32), data.target, sha


def train(path):
    x, y, sha = dataset()
    train_ids, test_ids = train_test_split(np.arange(len(y)), test_size=0.25,
                                         random_state=SEED, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam',
                        max_iter=1500, random_state=SEED, early_stopping=False,
                        tol=1e-5, n_iter_no_change=30)
    # Configuration is fixed in advance. No test input/label is used in fit or selection.
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        clf.fit(x[train_ids], y[train_ids])
    save(path, {'schema': 1, 'shape': [64, 32, 10], 'seed': SEED,
                'dataset_sha256': sha, 'train_ids': train_ids.tolist(),
                'test_ids': test_ids.tolist(), 'preprocessing': 'float32(pixel / 16)',
                'classes': clf.classes_.tolist(), 'training_parameters': clf.get_params(),
                'iterations': int(clf.n_iter_), 'training_loss': float(clf.loss_),
                'versions': versions(),
                'export_reference_probabilities': clf.predict_proba(x[test_ids]).tolist(),
                'weights': [a.tolist() for a in clf.coefs_],
                'biases': [a.tolist() for a in clf.intercepts_]})
    print('Exported trained model:', Path(path).name)


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__)


def restore(path):
    data = json.loads(Path(path).read_text())
    x, y, sha = dataset()
    require(data['schema'] == 1 and data['shape'] == [64, 32, 10], 'model schema/shape')
    require(data['dataset_sha256'] == sha, 'dataset hash mismatch')
    tr, te = data['train_ids'], data['test_ids']
    require(len(tr) == 1347 and len(te) == 450 and
            sorted(tr + te) == list(range(len(y))), 'split overlap/missing samples')
    expected_tr, expected_te = train_test_split(np.arange(len(y)), test_size=0.25,
                                               random_state=SEED, stratify=y)
    require(tr == expected_tr.tolist() and te == expected_te.tolist(), 'split differs from fixed policy')
    require(data['classes'] == list(range(10)), 'unexpected class mapping')
    weights = [np.ascontiguousarray(a, dtype=np.float32) for a in data['weights']]
    biases = [np.ascontiguousarray(a, dtype=np.float32) for a in data['biases']]
    require([a.shape for a in weights + biases] == [(64, 32), (32, 10), (32,), (10,)],
            'bad weight shapes')
    require(all(np.isfinite(a).all() for a in weights + biases), 'nonfinite model')
    # Restore only documented learned attributes, never execute serialized Python.
    clf = MLPClassifier(hidden_layer_sizes=(32,), activation='relu')
    clf.coefs_, clf.intercepts_ = weights, biases
    clf.classes_, clf.n_features_in_ = np.arange(10), 64
    clf.n_layers_, clf.n_outputs_, clf.out_activation_ = 3, 10, 'softmax'
    return data, x[te], y[te], clf


def pointer(a):
    return a.ctypes.data_as(C.POINTER(C.c_float))


class Engine:
    def __init__(self, library, clf, x):
        self.lib = load(library)
        self.weights = clf.coefs_ + clf.intercepts_  # retain borrowed buffers
        w1, w2, b1, b2 = self.weights
        self.model = Model()
        require(self.lib.ti_model_init(C.byref(self.model), 64, 32, 10,
                    pointer(w1), w1.size, pointer(b1), b1.size,
                    pointer(w2), w2.size, pointer(b2), b2.size) == 0, 'C model init')
        self.h, self.out = np.empty(32, np.float32), np.empty(10, np.float32)
        self.hp, self.op = pointer(self.h), pointer(self.out)
        self.inputs = [pointer(row) for row in x]
        self.x = x
        self.names = ['output_first', 'input_first']
        if self.lib.ti_blas_available():
            self.names.append('system_blas')

    def logits(self, i, k):
        require(self.lib.ti_forward(C.byref(self.model), self.inputs[i], 64,
                    self.hp, 32, self.op, 10, k) == 0, 'C forward')
        return self.out

    def probabilities(self, i, k):
        a = self.logits(i, k)
        e = np.exp(a - a.max())
        return e / e.sum()


def validate_outputs(logits, probabilities, reference_logits, reference_probs):
    require(np.isfinite(logits).all() and np.isfinite(probabilities).all(), 'nonfinite outputs')
    require(np.allclose(logits, reference_logits, atol=ATOL, rtol=RTOL), 'logit mismatch')
    require(np.allclose(probabilities, reference_probs, atol=ATOL, rtol=RTOL), 'probability mismatch')
    require(np.array_equal(probabilities.argmax(1), reference_probs.argmax(1)), 'class disagreement')


def evaluate(library, model=MODEL):
    data, x, y, clf = restore(model)
    engine = Engine(library, clf, x)
    reference_probs = clf.predict_proba(x)
    require(np.allclose(reference_probs, data['export_reference_probabilities'],
                        atol=ATOL, rtol=RTOL), 'export roundtrip mismatch')
    # Independent float64 fsum checks every test image, including near-zero logits.
    w1, w2 = [a.ravel().tolist() for a in clf.coefs_]
    b1, b2 = [a.tolist() for a in clf.intercepts_]
    reference_logits = np.array([oracle(row.tolist(), w1, b1, w2, b2, 32, 10) for row in x])
    expected_probs = np.exp(reference_logits - reference_logits.max(axis=1, keepdims=True))
    expected_probs /= expected_probs.sum(axis=1, keepdims=True)
    require(np.allclose(reference_probs, expected_probs, atol=ATOL, rtol=RTOL), 'sklearn/oracle mismatch')
    accuracy = float(np.mean(reference_probs.argmax(1) == y))
    require(accuracy >= 0.94, 'accuracy below predeclared 94% floor')
    metrics, outputs = {}, {}
    for k, name in enumerate(engine.names):
        logits = np.array([engine.logits(i, k).copy() for i in range(len(x))])
        probabilities = np.array([engine.probabilities(i, k) for i in range(len(x))])
        validate_outputs(logits, probabilities, reference_logits, reference_probs)
        pred = probabilities.argmax(1)
        confusion = np.zeros((10, 10), dtype=int)
        np.add.at(confusion, (y, pred), 1)
        metrics[name] = dict(correct=int((pred == y).sum()), total=len(y),
                             accuracy=float(np.mean(pred == y)), agreement=1.0,
                             max_logit_error=float(np.max(abs(logits - reference_logits))),
                             max_probability_error=float(np.max(abs(probabilities - reference_probs))),
                             confusion_matrix=confusion.tolist())
        outputs[name] = {'logits': logits.tolist(), 'probabilities': probabilities.tolist()}
    return engine, clf, x, dict(metrics=metrics, sklearn_accuracy=accuracy,
             model_sha256=digest(model), dataset_sha256=data['dataset_sha256'],
             test_ids=data['test_ids'], labels=y.tolist(),
             reference_probabilities=reference_probs.tolist(),
             oracle_logits=reference_logits.tolist(), outputs=outputs)


def worker(args):
    with threadpool_limits(limits=1):
        engine, clf, x, checks = evaluate(args.library, args.model)
        rows = [row.reshape(1, -1) for row in x]
        calls = {name: (lambda i, k=k: engine.probabilities(i, k))
                 for k, name in enumerate(engine.names)}
        calls['sklearn_single'] = lambda i: clf.predict_proba(rows[i])[0]
        calls['sklearn_batch'] = lambda i: clf.predict_proba(x)
        # Batch is a context measurement, not part of single-image speed ratios.
        orders = list(itertools.permutations(engine.names + ['sklearn_single']))
        raw = []
        reference = np.array(checks['reference_probabilities'])
        for name in calls:
            for i in range(16):
                calls[name](i)
        for sample in range(args.samples):
            order = list(orders[(args.run * args.samples + sample) % len(orders)])
            order.insert((sample + args.run) % (len(order) + 1), 'sklearn_batch')
            for name in order:
                if name == 'sklearn_batch':
                    start = time.perf_counter_ns()
                    answer = calls[name](0)
                    elapsed = time.perf_counter_ns() - start
                else:
                    # Keep every full prediction; verify all of them outside timing.
                    answer = np.empty((len(x), 10), dtype=np.float32)
                    start = time.perf_counter_ns()
                    for i in range(len(x)):
                        answer[i] = calls[name](i)
                    elapsed = time.perf_counter_ns() - start
                require(np.allclose(answer, reference, atol=ATOL, rtol=RTOL), 'timed output mismatch')
                raw.append(dict(sample=sample, backend=name, elapsed_ns=elapsed,
                                predictions=len(x), ns_per_prediction=elapsed / len(x)))
        pools = [{k: v for k, v in pool.items() if k != 'filepath'} for pool in threadpool_info()]
        require(all(p['num_threads'] == 1 for p in pools), 'uncontrolled numerical threads')
        save(args.output, dict(schema=1, run=args.run, checks=checks, raw=raw,
                              versions=versions(), threadpools=pools,
                              c_blas=engine.lib.ti_blas_name().decode(),
                              c_single_thread=bool(engine.lib.ti_blas_is_single_threaded())))
        print('PASS: all 450 held-out images, C backends vs sklearn and float64 oracle')


def cpu_name():
    if sys.platform == 'darwin':
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return platform.processor() or None


def benchmark(args):
    require(1 <= args.runs <= 24 and 1 <= args.samples <= 100, 'invalid run/sample count')
    sources = ['tools/real_model.py', 'tests/test_reference.py', 'src/tiny_infer.c',
               'src/blas_backend.c', 'src/blas_backend.h', 'include/tiny_infer.h',
               'Makefile', 'requirements-real-model.txt', 'models/digits/model.json',
               'tools/verify_real_model.py', 'tests/test_real_model.py', 'tests/test_real_report.py']
    before = {p: digest(ROOT / p) for p in sources}
    build_info = json.loads((ROOT / 'build/real-model-build.json').read_text())
    require(build_info['library_sha256'] == digest(args.library), 'stale build library')
    require(all(digest(ROOT / p) == sha for p, sha in build_info['sources'].items()), 'stale build source')
    require(digest(args.model) == before['models/digits/model.json'], 'benchmark requires tracked model')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for run in range(args.runs):
        path = args.output / ('run-%02d.json' % run)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker',
                        '--library', str(args.library.resolve()), '--model', str(args.model.resolve()),
                        '--output', str(path.resolve()), '--run', str(run),
                        '--samples', str(args.samples)], check=True, cwd=ROOT)
        records.append(json.loads(path.read_text()))
    require(before == {p: digest(ROOT / p) for p in sources}, 'source changed during run')
    require(all(r['checks'] == records[0]['checks'] for r in records), 'process check mismatch')
    save(args.output / 'correctness.json', records[0]['checks'])
    for run, record in enumerate(records):
        record.pop('checks')
        save(args.output / ('run-%02d.json' % run), record)
    summaries = {}
    for name in sorted({s['backend'] for s in records[0]['raw']}):
        process_medians = [float(np.median([s['ns_per_prediction'] for s in r['raw']
                                          if s['backend'] == name])) for r in records]
        summaries[name] = dict(median_ns=float(np.median(process_medians)),
                               process_medians_ns=process_medians,
                               min_process_median_ns=min(process_medians),
                               max_process_median_ns=max(process_medians))
    save(args.output / 'manifest.json', dict(schema=1, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
         source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
         sources=before, library_sha256=digest(args.library),
         build_metadata=json.loads((ROOT / 'build/real-model-build.json').read_text()),
         platform=dict(system=platform.system(), release=platform.release(), machine=platform.machine(),
                       cpu=cpu_name()),
         runs=args.runs, samples=args.samples, summaries=summaries,
         files={p.name: digest(p) for p in args.output.glob('*.json')}))
    print(json.dumps(summaries, indent=2))


def build(args):
    from workloads import native_flags
    flags, libs = native_flags(args.blas)
    objects = ['build/real_tiny_infer.o', 'build/real_blas_backend.o']
    commands = []
    for source, obj in zip(['src/tiny_infer.c', 'src/blas_backend.c'], objects):
        commands.append(['cc', '-Iinclude', '-std=c11', '-O3', '-Wall', '-Wextra', '-Wpedantic',
                         '-Werror', '-ffp-contract=off', '-fPIC'] + flags + ['-c', source, '-o', obj])
    commands += [['cc', '-shared'] + objects + libs + ['-lm', '-o', str(args.library)],
                 ['cc', '-Iinclude', '-std=c11', '-O3'] + flags + ['tests/test_engine.c'] +
                 objects + libs + ['-lm', '-o', 'build/real_test_engine']]
    (ROOT / 'build').mkdir(exist_ok=True)
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run(['build/real_test_engine'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'tests/test_reference.py', str(args.library)], cwd=ROOT, check=True)
    save(ROOT / 'build/real-model-build.json', dict(commands=commands,
         compiler=subprocess.check_output(['cc', '--version'], text=True).splitlines()[0],
         objects={p: digest(ROOT / p) for p in objects}, library_sha256=digest(args.library),
         sources={p: digest(ROOT / p) for p in ['src/tiny_infer.c', 'src/blas_backend.c',
                   'src/blas_backend.h', 'include/tiny_infer.h', 'tools/real_model.py']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'build', 'check', 'benchmark', 'worker'])
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--library', type=Path, default=Path('build/real_engine.so'))
    parser.add_argument('--output', type=Path, default=Path('results/local/digits'))
    parser.add_argument('--blas', type=int, choices=[0, 1], default=1)
    parser.add_argument('--runs', type=int, default=6)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--run', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'train':
        train(args.model)
    elif args.command == 'build':
        build(args)
    elif args.command == 'benchmark':
        benchmark(args)
    elif args.command == 'worker':
        worker(args)
    else:
        with threadpool_limits(limits=1):
            _, _, _, checks = evaluate(args.library, args.model)
            print(json.dumps(checks['metrics'], indent=2))


if __name__ == '__main__':
    main()
