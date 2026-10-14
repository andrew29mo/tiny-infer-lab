"""Same frozen digits model: tiny C, emlearn, ORT CPU, NumPy and sklearn.

Optional Python 3.12 toolchain. No retraining or automatic best-result selection.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
import real_model as rm
import provider_runtime as pr

ROOT = rm.ROOT
BATCHES = [1,8,32,128,450]
ARTIFACTS = Path('build/comparison')
SOURCES = sorted(set(rm.SOURCES + ['tools/provider_emlearn.c','tools/provider_runtime.py',
    'tools/compare_providers.py','tools/verify_providers.py','tests/test_providers.py',
    'tests/test_provider_report.py','requirements-comparison.txt']))
PACKAGES = ['numpy','scipy','scikit-learn','joblib','threadpoolctl','emlearn','onnx',
            'onnxruntime','psutil','setuptools','flatbuffers','packaging','protobuf']


def versions():
    return dict(python=platform.python_version(), **{p:metadata.version(p) for p in PACKAGES})


def parameter_bytes(clf):
    w1,w2 = clf.coefs_; b1,b2 = clf.intercepts_
    return np.concatenate([a.ravel() for a in (w1,b1,w2,b2)]).astype('<f4').tobytes()


def export_emlearn(clf, directory):
    # Importing emlearn.bayes changes NumPy's global seterr policy. Keep exporter
    # imports from changing validation or timing behavior elsewhere in this process.
    with np.errstate():
        import emlearn.net
    # Upstream exporter prints six decimals. Replace ONLY parameter literals with
    # C99 hex float32 literals; compiled bytes must match the frozen model exactly.
    text = emlearn.net.c_generate_net_loadable(['relu','softmax'],clf.coefs_,clf.intercepts_,'digits')
    for layer in range(2):
        for kind,array in [('weights',clf.coefs_[layer]),('biases',clf.intercepts_[layer])]:
            name = 'digits_layer_%d_%s'%(layer,kind)
            pattern = r'('+name+r'\[\d+\]\s*=\s*\{)[^}]+(\})'
            literals = ', '.join(float(a).hex()+'f' for a in array.ravel())
            text,count = re.subn(pattern,lambda m:m[1]+literals+m[2],text)
            rm.require(count==1, 'emlearn exporter parameter layout changed')
    (directory/'digits_emlearn.h').write_text(text)
    include = Path(emlearn.net.__file__).parent
    command = ['cc','-std=c11','-O3','-ffp-contract=off','-fPIC','-shared',
               '-I'+str(directory),'-I'+str(include),'tools/provider_emlearn.c','-lm',
               '-o',str(directory/'emlearn.so')]
    subprocess.run(command,check=True,cwd=ROOT)
    # Do not save the absolute local package include path into published metadata.
    command[command.index('-I'+str(include))] = '-I<installed-emlearn-headers>'
    return dict(command=command,headers={p.name:rm.digest(p) for p in sorted(include.glob('*.h'))},
                exact_float32_literals=True)


def export_onnx(clf, directory):
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    w1,w2 = clf.coefs_; b1,b2 = clf.intercepts_
    params = [('w1',w1),('b1',b1),('w2',w2),('b2',b2)]
    nodes = [h.make_node('MatMul',['input','w1'],['a']),h.make_node('Add',['a','b1'],['b']),
             h.make_node('Relu',['b'],['h']),h.make_node('MatMul',['h','w2'],['c']),
             h.make_node('Add',['c','b2'],['logits']),
             h.make_node('Softmax',['logits'],['probabilities'],axis=1)]
    graph = h.make_graph(nodes,'frozen-digits',[h.make_tensor_value_info('input',T.FLOAT,['N',64])],
                [h.make_tensor_value_info('probabilities',T.FLOAT,['N',10])],
                [nh.from_array(a,name=n) for n,a in params])
    model = h.make_model(graph,opset_imports=[h.make_opsetid('',17)],ir_version=9,
                        producer_name='tiny-infer-lab')
    onnx.checker.check_model(model,full_check=True)
    onnx.save(model,directory/'digits.onnx')
    rm.require(all(np.array_equal(nh.to_array(t).view(np.uint32),a.view(np.uint32))
               for t,(_,a) in zip(model.graph.initializer,params)), 'ONNX parameter bits changed')


def audit_exports(directory, clf):
    import ctypes as C
    import onnx
    from onnx import numpy_helper as nh
    p = pr.Provider('emlearn',directory)
    actual = np.empty(2410,np.float32)
    p.lib.comparison_weights.argtypes = [C.POINTER(C.c_float)]
    p.lib.comparison_weights.restype = None
    p.lib.comparison_weights(pr.pointer(actual))
    expected = parameter_bytes(clf)
    rm.require(actual.astype('<f4').tobytes()==expected, 'compiled emlearn parameters differ')
    model = onnx.load(directory/'digits.onnx')
    rm.require([n.op_type for n in model.graph.node]==['MatMul','Add','Relu','MatMul','Add','Softmax']
               and [n.name for n in model.graph.initializer]==['w1','b1','w2','b2'], 'ONNX graph differs')
    arrays = [nh.to_array(t) for t in model.graph.initializer]
    rm.require(np.concatenate([a.ravel() for a in arrays]).astype('<f4').tobytes()==expected,
               'serialized ONNX parameters differ')
    with np.load(directory/'weights.npz',allow_pickle=False) as data:
        rm.require(b''.join(data[k].astype('<f4').tobytes() for k in ('w1','b1','w2','b2'))==expected,
                   'tiny/NumPy/sklearn parameter file differs')
    return dict(parameter_sha256=hashlib.sha256(expected).hexdigest(),parameter_count=2410,
                compiled_emlearn_bitwise_equal=True,onnx_initializers_bitwise_equal=True,
                numpy_archive_bitwise_equal=True)


def prepare(args):
    directory = args.artifacts
    directory.mkdir(parents=True,exist_ok=True)
    model,x,y,clf = rm.restore(rm.MODEL)
    rm.build(argparse.Namespace(blas=1,library=Path('build/real_engine.so')))
    shutil.copyfile('build/real_engine.so',directory/'tiny.so')
    w1,w2=clf.coefs_;b1,b2=clf.intercepts_
    np.savez(directory/'weights.npz',w1=w1,b1=b1,w2=w2,b2=b2)
    np.save(directory/'input.npy',x,allow_pickle=False)
    np.save(directory/'gold.npy',clf.predict_proba(x),allow_pickle=False)
    emlearn_build = export_emlearn(clf,directory)
    export_onnx(clf,directory)
    info = dict(schema=1,versions=versions(),sources={p:rm.digest(p) for p in SOURCES},
        tiny_build=json.loads(Path('build/real-model-build.json').read_text()),
        emlearn_build=emlearn_build,export_audit=audit_exports(directory,clf),
        artifacts={p.name:rm.digest(p) for p in directory.iterdir() if p.is_file() and p.name!='build.json'})
    rm.save(directory/'build.json',info)
    print('PASS: unchanged model exported; compiled emlearn/ONNX parameter bits verified')


def check(args):
    model,x,y,clf = rm.restore(rm.MODEL)
    # Full float64 fsum oracle, not just stored sklearn predictions.
    w1,w2 = [a.ravel().tolist() for a in clf.coefs_]; b1,b2 = [a.tolist() for a in clf.intercepts_]
    logits=np.array([rm.oracle(row.tolist(),w1,b1,w2,b2,32,10) for row in x])
    gold=pr.softmax(logits)
    reference=clf.predict_proba(x)
    pr.validate(reference,gold)
    rm.require(np.array_equal(np.load(args.artifacts/'input.npy',allow_pickle=False),x) and
               np.array_equal(np.load(args.artifacts/'gold.npy',allow_pickle=False),reference),
               'worker input/reference artifacts differ from frozen data')
    rm.require(np.allclose(reference,model['export_reference_probabilities'],atol=rm.ATOL,rtol=rm.RTOL),
               'frozen export reference mismatch')
    providers = {name:pr.Provider(name,args.artifacts) for name in pr.NAMES}
    outputs = {}
    with threadpool_limits(limits=1):
        for batch in args.batches:
            outputs[str(batch)] = {}
            for name,provider in providers.items():
                provider.poison()
                answer = pr.sweep(provider,x,batch)
                pr.validate(answer,gold)
                outputs[str(batch)][name] = dict(probabilities=answer.tolist(),
                    correct=int((answer.argmax(1)==y).sum()),max_abs_error=float(abs(answer-gold).max()))
    return dict(model_sha256=rm.digest(rm.MODEL),dataset_sha256=model['dataset_sha256'],
                test_ids=model['test_ids'],labels=y.tolist(),reference_probabilities=gold.tolist(),
                export_audit=audit_exports(args.artifacts,clf),outputs=outputs)


def order(run,sample,batches):
    # Cyclic Latin orders plus reversal: each backend occupies every slot in 6 samples.
    names=list(pr.NAMES); shift=(sample+run)%len(names)
    names=names[shift:]+names[:shift]
    if (sample//len(names)+run)%2:names.reverse()
    shift=(sample+run)%len(batches)
    return [(b,n) for b in batches[shift:]+batches[:shift] for n in names]


def worker(args):
    x = np.load(args.artifacts/'input.npy',allow_pickle=False)
    gold = np.load(args.artifacts/'gold.npy',allow_pickle=False)
    providers={n:pr.Provider(n,args.artifacts) for n in pr.NAMES}
    raw=[]
    with threadpool_limits(limits=1):
        for b in args.batches:
            for p in providers.values():
                for _ in range(2):pr.validate(pr.sweep(p,x,b),gold)
        for sample in range(args.samples):
            for b,name in order(args.run,sample,args.batches):
                p=providers[name];p.poison()
                start=time.perf_counter_ns()
                answers=[pr.sweep(p,x,b) for _ in range(args.repeats)]
                elapsed=time.perf_counter_ns()-start
                for answer in answers:pr.validate(answer,gold)
                raw.append(dict(sample=sample,batch=b,backend=name,elapsed_ns=elapsed,
                    predictions=450*args.repeats,ns_per_prediction=elapsed/(450*args.repeats),
                    adapter_calls=len(pr.partitions(b))*args.repeats,
                    emlearn_single_calls=450*args.repeats if name=='emlearn' else None))
        pools=[{k:v for k,v in p.items() if k!='filepath'} for p in threadpool_info()]
        rm.require(pools and all(p['num_threads']==1 for p in pools), 'uncontrolled BLAS threads')
    rm.save(args.output,dict(run=args.run,versions=versions(),threadpools=pools,raw=raw,
        ort=dict(providers=providers['onnxruntime'].session.get_providers(),intra_threads=1,
                 inter_threads=1,execution='sequential',optimization='all'),all_outputs_checked=True))


def summarize(records,batches):
    out={}
    for b in batches:
        out[str(b)]={}
        for n in pr.NAMES:
            medians=[float(np.median([s['ns_per_prediction'] for s in r['raw']
                        if s['batch']==b and s['backend']==n])) for r in records]
            out[str(b)][n]=dict(median_ns=float(np.median(medians)),process_medians_ns=medians,
                               minimum_ns=min(medians),maximum_ns=max(medians))
    return out


def deployment(directory):
    # Installed wheel payload, excluding bytecode/cache/metadata/scripts. This is
    # not a minimal linker closure or compressed wheel size; common NumPy separate.
    packages={}
    for name in ['numpy','onnxruntime','flatbuffers','packaging','protobuf','scipy',
                 'scikit-learn','joblib','threadpoolctl']:
        dist=metadata.distribution(name)
        files={}
        for p in dist.files or []:
            path=Path(str(p))
            if '..' in path.parts or '__pycache__' in path.parts or path.suffix=='.pyc' or any(
                    q.endswith(('.dist-info','.data')) for q in path.parts):continue
            actual=Path(dist.locate_file(p))
            if actual.is_file():files[str(p)]=dict(bytes=actual.stat().st_size,sha256=rm.digest(actual))
        rm.require(files, 'empty distribution inventory')
        packages[name]=dict(version=dist.version,bytes=sum(v['bytes'] for v in files.values()),
            file_count=len(files),inventory_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
    artifact_sizes={p.name:p.stat().st_size for p in directory.iterdir() if p.name in
                    {'tiny.so','emlearn.so','digits.onnx','weights.npz'}}
    model_files={'tiny_input':['tiny.so','weights.npz'],'tiny_blas':['tiny.so','weights.npz'],
                 'emlearn':['emlearn.so'],'onnxruntime':['digits.onnx'],
                 'numpy':['weights.npz'],'sklearn':['weights.npz']}
    extra={'onnxruntime':['onnxruntime','flatbuffers','packaging','protobuf'],
           'sklearn':['scikit-learn','scipy','joblib','threadpoolctl']}
    providers={n:dict(artifact_bytes=sum(artifact_sizes[p] for p in model_files[n]),
        artifacts=model_files[n],additional_packages=extra.get(n,[]),
        additional_package_bytes=sum(packages[p]['bytes'] for p in extra.get(n,[]))) for n in pr.NAMES}
    return dict(artifact_bytes=artifact_sizes,packages=packages,providers=providers,
        common_packages=['numpy'],common_package_bytes=packages['numpy']['bytes'],
        exclusions=['Python interpreter/stdlib','OS libraries and system BLAS','adapter source',
                    'build tools, emlearn generator and ONNX exporter','wheel metadata, scripts and bytecode'],
        native_links={n:subprocess.check_output(['otool','-L',str(directory/n)] if sys.platform=='darwin'
                    else ['ldd',str(directory/n)],text=True).replace(str(directory),'build/comparison')
                      for n in ['tiny.so','emlearn.so']})


def benchmark(args):
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    sources={p:rm.digest(p) for p in SOURCES}
    for p,sha in sources.items():
        rm.require(hashlib.sha256(subprocess.check_output(['git','show',revision+':'+p])).hexdigest()==sha,
                   'commit measurement sources before benchmarking')
    info=json.loads((args.artifacts/'build.json').read_text())
    rm.require(info['sources']==sources and info['versions']==versions(), 'stale build/version')
    before={p.name:rm.digest(p) for p in args.artifacts.iterdir() if p.is_file()}
    rm.require(all(before[p]==sha for p,sha in info['artifacts'].items()),'stale artifacts')
    args.output.mkdir(parents=True,exist_ok=False)
    checks=check(args);rm.save(args.output/'correctness.json',checks)
    records=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             VECLIB_MAXIMUM_THREADS='1',ORT_DISABLE_TELEMETRY='1')
    for run in range(args.runs):
        path=args.output/('run-%02d.json'%run)
        subprocess.run([sys.executable,'tools/compare_providers.py','worker','--artifacts',str(args.artifacts),
            '--output',str(path),'--run',str(run),'--samples',str(args.samples),'--repeats',str(args.repeats),
            '--batches',','.join(map(str,args.batches))],check=True,env=env)
        records.append(json.loads(path.read_text()))
        print('Verified timing process',run+1,flush=True)
    memory=[]
    for repeat in range(args.memory_runs):
        for b in args.batches:
            for n in pr.NAMES:
                output=subprocess.check_output([sys.executable,'tools/provider_runtime.py',
                    '--artifacts',str(args.artifacts),'--backend',n,'--batch',str(b)],text=True,env=env)
                memory.append(dict(repeat=repeat,**json.loads(output)))
        print('Verified isolated memory process group',repeat+1,flush=True)
    rm.save(args.output/'memory.json',memory)
    rm.save(args.output/'deployment.json',deployment(args.artifacts))
    rm.require(sources=={p:rm.digest(p) for p in SOURCES} and before=={
        p.name:rm.digest(p) for p in args.artifacts.iterdir() if p.is_file()},'source/artifact changed during run')
    rm.save(args.output/'manifest.json',dict(schema=1,protocol='same-model-providers-v1',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_revision=revision,sources=sources,
        versions=versions(),platform=dict(system=platform.system(),release=platform.release(),
        machine=platform.machine(),cpu=rm.cpu_name()),build=info,runs=args.runs,samples=args.samples,
        repeats=args.repeats,memory_runs=args.memory_runs,batches=args.batches,backends=list(pr.NAMES),
        partitions={str(b):[z-a for a,z in pr.partitions(b)] for b in args.batches},
        summaries=summarize(records,args.batches),
        files={p.name:rm.digest(p) for p in args.output.glob('*.json')}))
    print(json.dumps({b:{n:round(v['median_ns']/1000,3) for n,v in rows.items()}
                     for b,rows in summarize(records,args.batches).items()},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','check','benchmark','worker'])
    p.add_argument('--artifacts',type=Path,default=ARTIFACTS)
    p.add_argument('--output',type=Path,default=Path('results/local/providers'))
    p.add_argument('--runs',type=int,default=6)
    p.add_argument('--samples',type=int,default=12)
    p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--memory-runs',type=int,default=3)
    p.add_argument('--run',type=int,default=0)
    p.add_argument('--batches',default=','.join(map(str,BATCHES)))
    args=p.parse_args();args.batches=list(map(int,args.batches.split(',')))
    rm.require(args.batches and len(set(args.batches))==len(args.batches) and
        all(1<=b<=450 for b in args.batches) and 1<=args.runs<=24 and 1<=args.samples<=100
        and 1<=args.repeats<=32 and 1<=args.memory_runs<=5,'invalid measurement configuration')
    if args.command=='prepare':prepare(args)
    elif args.command=='check':
        check(args);print('PASS: all same-model provider outputs vs independent oracle')
    elif args.command=='worker':worker(args)
    else:benchmark(args)


if __name__=='__main__':main()
