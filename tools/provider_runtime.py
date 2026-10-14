"""Inference-only adapters and isolated RSS probes; no training imports for C/ORT.

All adapters accept a contiguous float32 (N,64) block and return (N,10)
probabilities. Conversion, finite-input checks and probability output are timed.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
from test_reference import Model, load

NAMES = ('tiny_input', 'tiny_blas', 'emlearn', 'onnxruntime', 'numpy', 'sklearn')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pointer(a):
    return a.ctypes.data_as(C.POINTER(C.c_float))


def partitions(batch):
    require(type(batch) is int and 1 <= batch <= 450, 'invalid batch')
    return [(i, min(i+batch, 450)) for i in range(0, 450, batch)]


def softmax(a):
    e = np.exp(a-a.max(axis=1, keepdims=True))
    return e/e.sum(axis=1, keepdims=True)


class Provider:
    def __init__(self, name, artifacts, capacity=450):
        require(name in NAMES, 'unknown provider')
        self.name, self.artifacts = name, Path(artifacts)
        self.capacity = capacity
        if name.startswith('tiny_') or name=='numpy':
            self.h = np.empty((capacity,32), np.float32)
        if name.startswith('tiny_') or name in ('numpy','emlearn'):
            self.y = np.empty((capacity,10), np.float32)
        if name.startswith('tiny_') or name in ('numpy', 'sklearn'):
            with np.load(self.artifacts/'weights.npz', allow_pickle=False) as data:
                self.weights = [np.ascontiguousarray(data[k]) for k in ('w1','b1','w2','b2')]
            w1,b1,w2,b2 = self.weights
        if name.startswith('tiny_'):
            self.lib = load(self.artifacts/'tiny.so')
            self.model = Model()
            require(self.lib.ti_model_init(C.byref(self.model),64,32,10,
                    pointer(w1),w1.size,pointer(b1),b1.size,pointer(w2),w2.size,
                    pointer(b2),b2.size)==0, 'tiny model init')
            self.kernel = 1 if name=='tiny_input' else 2
            require(name!='tiny_blas' or self.lib.ti_blas_is_single_threaded()==1,
                    'BLAS must be available and single-threaded')
        elif name=='emlearn':
            self.lib = C.CDLL(str((self.artifacts/'emlearn.so').resolve()))
            self.lib.comparison_emlearn.argtypes = [C.POINTER(C.c_float),C.c_size_t,C.POINTER(C.c_float)]
            self.lib.comparison_emlearn.restype = C.c_int
        elif name=='onnxruntime':
            import onnxruntime as ort
            ort.disable_telemetry_events()
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(self.artifacts/'digits.onnx'), options,
                                                providers=['CPUExecutionProvider'])
            require(self.session.get_providers()==['CPUExecutionProvider'], 'unexpected execution provider')
            actual=self.session.get_session_options()
            require(actual.intra_op_num_threads==actual.inter_op_num_threads==1 and
                    actual.execution_mode==ort.ExecutionMode.ORT_SEQUENTIAL and
                    actual.graph_optimization_level==ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                    'ORT session policy changed')
        elif name=='sklearn':
            from sklearn.neural_network import MLPClassifier
            self.clf = MLPClassifier(hidden_layer_sizes=(32,),activation='relu')
            self.clf.coefs_,self.clf.intercepts_ = [w1,w2],[b1,b2]
            self.clf.classes_,self.clf.n_features_in_ = np.arange(10),64
            self.clf.n_layers_,self.clf.n_outputs_,self.clf.out_activation_ = 3,10,'softmax'

    def poison(self):
        for name in ('h','y'):
            if hasattr(self,name):getattr(self,name).fill(np.nan)

    def __call__(self, block):
        x = np.asarray(block, dtype=np.float32, order='C')
        require(x.ndim==2 and x.shape[1]==64 and 1<=len(x)<=self.capacity and np.isfinite(x).all(),
                'expected finite 1..450 by 64 input')
        n = len(x)
        if self.name.startswith('tiny_'):
            status = self.lib.ti_forward_batch(C.byref(self.model),pointer(x),n,x.size,
                        pointer(self.h),n*32,pointer(self.y),n*10,self.kernel)
            require(status==0, 'tiny forward')
            return softmax(self.y[:n])
        if self.name=='emlearn':
            require(self.lib.comparison_emlearn(pointer(x),n,pointer(self.y))==0, 'emlearn forward')
            return self.y[:n]
        if self.name=='onnxruntime':
            return self.session.run(['probabilities'],{'input':x})[0]
        if self.name=='sklearn':
            return self.clf.predict_proba(x)
        w1,b1,w2,b2 = self.weights
        np.dot(x,w1,out=self.h[:n]); np.add(self.h[:n],b1,out=self.h[:n])
        np.maximum(self.h[:n],0,out=self.h[:n])
        np.dot(self.h[:n],w2,out=self.y[:n]); np.add(self.y[:n],b2,out=self.y[:n])
        return softmax(self.y[:n])


def sweep(provider, x, batch):
    # Collection cost is identical and timed for all backends; omitted tails stay NaN.
    out = np.full((450,10),np.nan,np.float32)
    for a,b in partitions(batch):
        out[a:b] = provider(x[a:b])
    return out


def validate(answer, gold):
    require(answer.shape==gold.shape==(450,10) and np.isfinite(answer).all() and
            np.allclose(answer,gold,atol=1e-5,rtol=3e-5) and
            np.array_equal(answer.argmax(1),gold.argmax(1)), 'provider probabilities/classes mismatch')


def memory(args):
    # Fresh subprocess per provider/batch/repetition. Common NumPy/input are baseline.
    import psutil
    from threadpoolctl import threadpool_info, threadpool_limits
    x = np.load(args.artifacts/'input.npy',allow_pickle=False)
    with threadpool_limits(limits=1):
        process = psutil.Process()
        baseline = process.memory_info().rss
        start = time.perf_counter_ns()
        provider = Provider(args.backend,args.artifacts,capacity=args.batch)
        load_ns = time.perf_counter_ns()-start
        loaded = process.memory_info().rss
        with threadpool_limits(limits=1):
            for _ in range(5):
                answer = sweep(provider,x,args.batch)
            pools=[{k:v for k,v in p.items() if k!='filepath'} for p in threadpool_info()]
            require(pools and all(p['num_threads']==1 for p in pools),'uncontrolled probe threads')
        gold = np.load(args.artifacts/'gold.npy',allow_pickle=False)
        validate(answer,gold)
        warm = process.memory_info().rss
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak if sys.platform=='darwin' else peak*1024)
        print(json.dumps(dict(backend=args.backend,batch=args.batch,baseline_rss_bytes=baseline,
            loaded_rss_bytes=loaded,warm_rss_bytes=warm,incremental_warm_rss_bytes=warm-baseline,
            process_peak_rss_bytes=peak_bytes,load_ns=load_ns,correctness_checked=True,threadpools=pools)))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--backend',choices=NAMES,required=True)
    p.add_argument('--batch',type=int,required=True)
    memory(p.parse_args())
