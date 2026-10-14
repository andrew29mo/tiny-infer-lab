"""Matched-batch digits inference: production C, buffer-reusing NumPy and sklearn."""
import argparse
import ctypes as C
import datetime as dt
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
import real_model as rm
from workloads import native_flags

BATCHES = (1, 8, 32, 128, 450)
SOURCES = sorted(set(rm.SOURCES + ['tools/batch_model.py', 'tools/verify_batch.py',
    'tests/test_batch.c', 'tests/test_batch_reference.py', 'tests/test_batch_model.py',
    'tests/test_batch_report.py', 'tests/test_batch_mutations.py']))
DIAGNOSTIC = 'c_single_loop_input_first'
DIRECT = 'c_single_direct_input_first'
DIAGNOSTICS = {DIAGNOSTIC,DIRECT}


def partitions(size):
    return [(i, min(i + size, 450)) for i in range(0, 450, size)]


def prepared(block):
    block = np.asarray(block, dtype=np.float32, order='C')
    rm.require(block.ndim == 2 and block.shape[1] == 64 and 1 <= len(block) <= 450,
               'expected 1..450 images of 64 features')
    return block


def softmax(logits):
    rm.require(np.isfinite(logits).all(), 'nonfinite logits')
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class CBatch:
    def __init__(self, engine, sizes):
        self.engine = engine
        self.buffers = {}
        for n in sizes:
            h,y = np.empty((n,32),np.float32),np.empty((n,10),np.float32)
            self.buffers[n] = h,y,rm.pointer(h),rm.pointer(y)

    def poison(self):
        for h,y,_,_ in self.buffers.values():
            h.fill(np.nan);y.fill(np.nan)

    def logits(self, block, kernel):
        block = prepared(block)
        rm.require(len(block) in self.buffers, 'batch size was not configured')
        h,y,hp,yp = self.buffers[len(block)]
        status = self.engine.lib.ti_forward_batch(C.byref(self.engine.model), rm.pointer(block),
                   len(block), block.size, hp, h.size, yp, y.size, kernel)
        rm.require(status != 4, 'nonfinite batch C input/output')
        rm.require(status == 0, 'batch C forward')
        return y

    def probabilities(self, block, kernel):
        return softmax(self.logits(block, kernel))


class NumpyBatch:
    def __init__(self, clf, sizes):
        self.clf = clf
        self.buffers = {n: (np.empty((n,32),np.float32),np.empty((n,10),np.float32))
                        for n in sizes}

    def poison(self):
        for h,y in self.buffers.values():
            h.fill(np.nan);y.fill(np.nan)

    def logits(self, block):
        block = prepared(block)
        rm.require(np.isfinite(block).all(), 'nonfinite input')
        h,y = self.buffers[len(block)]
        np.dot(block, self.clf.coefs_[0], out=h)
        np.add(h, self.clf.intercepts_[0], out=h)
        np.maximum(h, 0, out=h)
        np.dot(h, self.clf.coefs_[1], out=y)
        np.add(y, self.clf.intercepts_[1], out=y)
        return y

    def probabilities(self, block):
        return softmax(self.logits(block))


def setup(library, sizes=BATCHES):
    model, x, y, clf = rm.restore(rm.MODEL)
    engine = rm.Engine(library, clf, x)
    actual_sizes = {b-a for size in sizes for a,b in partitions(size)}
    c, numpy = CBatch(engine,actual_sizes), NumpyBatch(clf,actual_sizes)
    calls = {name: (lambda block,k=k: c.probabilities(block,k)) for k,name in enumerate(engine.names)}
    calls['numpy_batch'] = numpy.probabilities
    calls['sklearn_batch'] = clf.predict_proba
    def single_loop(block):
        answer = np.empty((len(block),10),np.float32)
        for i in range(len(block)):
            answer[i] = engine.predict_proba(block[i:i+1],1)[0]
        return answer
    calls[DIAGNOSTIC] = single_loop
    # DIRECT is dispatched by sweep(); the callback remains useful for block checks.
    calls[DIRECT] = single_loop
    return model,x,y,clf,engine,c,numpy,calls


def reference(model,x,clf):
    probs = clf.predict_proba(x)
    rm.require(np.allclose(probs,model['export_reference_probabilities'],atol=rm.ATOL,rtol=rm.RTOL),
               'original sklearn export mismatch')
    w1,w2 = [w.ravel().tolist() for w in clf.coefs_]
    b1,b2 = [b.tolist() for b in clf.intercepts_]
    logits = np.array([rm.oracle(row.tolist(),w1,b1,w2,b2,32,10) for row in x])
    rm.require(np.allclose(softmax(logits),probs,atol=rm.ATOL,rtol=rm.RTOL),'sklearn/oracle mismatch')
    return logits,probs


def correctness(library, sizes=BATCHES):
    setup_values = setup(library,sizes)
    model,x,y,clf,engine,c,numpy,calls = setup_values
    gold_logits,gold_probs = reference(model,x,clf)
    outputs = {}
    for size in sizes:
        result = {}
        for name,call in calls.items():
            poison(setup_values)
            probs = (sweep(name,[],calls,engine,[row.reshape(1,64) for row in x]) if name==DIRECT else
                     np.concatenate([call(x[a:b]) for a,b in partitions(size)]))
            logits = None
            if name in engine.names:
                k = engine.names.index(name)
                c.poison()
                logits = np.concatenate([c.logits(x[a:b],k).copy() for a,b in partitions(size)])
            elif name == 'numpy_batch':
                numpy.poison()
                logits = np.concatenate([numpy.logits(x[a:b]).copy() for a,b in partitions(size)])
            rm.require(probs.shape == (450,10) and np.isfinite(probs).all() and
                       np.allclose(probs,gold_probs,atol=rm.ATOL,rtol=rm.RTOL) and
                       np.array_equal(probs.argmax(1),gold_probs.argmax(1)), 'batch probability mismatch')
            if logits is not None:
                rm.validate_outputs(logits,probs,gold_logits,gold_probs)
            accuracy = float(np.mean(probs.argmax(1)==y))
            rm.require(accuracy >= .94, 'batch accuracy below floor')
            result[name] = dict(probabilities=probs.tolist(),
                 logits=logits.tolist() if logits is not None else None,
                 correct=int((probs.argmax(1)==y).sum()), accuracy=accuracy,
                 max_probability_error=float(np.max(abs(probs-gold_probs))),
                 max_logit_error=float(np.max(abs(logits-gold_logits))) if logits is not None else None)
        outputs[str(size)] = result
    checks = dict(model_sha256=rm.digest(rm.MODEL),dataset_sha256=model['dataset_sha256'],
                  test_ids=model['test_ids'],labels=y.tolist(),oracle_logits=gold_logits.tolist(),
                  reference_probabilities=gold_probs.tolist(),outputs=outputs,
                  split_audit=rm.split_audit(rm.dataset()[0],model['train_ids'],model['test_ids']))
    return setup_values,checks


def schedule(names,batches,run,sample,samples):
    primary = sorted(n for n in names if n not in DIAGNOSTICS)
    orders = list(itertools.permutations(primary))
    rotated = batches[(run+sample)%len(batches):] + batches[:(run+sample)%len(batches)]
    out = []
    for size in rotated:
        offset = batches.index(size)
        order = list(orders[(run*samples+sample+offset)%len(orders)])
        order.insert((run+sample+offset)%(len(order)+1),DIAGNOSTIC)
        if DIRECT in names:
            order.insert((run+sample*2+offset)%(len(order)+1),DIRECT)
        out.extend((size,name) for name in order)
    return out


def poison(values):
    values[5].poison();values[6].poison()
    values[4].h.fill(np.nan);values[4].out.fill(np.nan)


def sweep(name,pieces,calls,engine,single_rows):
    # Filling the collector is timed for every API. A skipped tail stays NaN.
    answer=np.full((450,10),np.nan,np.float32)
    if name==DIRECT:
        for i,row in enumerate(single_rows):
            answer[i]=engine.predict_proba(row,1)[0]
    else:
        for slot,block in pieces:
            answer[slot]=calls[name](block)
    return answer


def validate_sweep(answer,reference):
    rm.require(answer.shape==(450,10) and np.isfinite(answer).all() and
               np.allclose(answer,reference,atol=rm.ATOL,rtol=rm.RTOL) and
               np.array_equal(answer.argmax(1),reference.argmax(1)), 'timed batch mismatch')


def audit_native_calls(values,batches,reference):
    # Untimed audit of the very same sweep path, not instrumentation in timings.
    _,x,_,_,engine,_,_,calls=values
    lib=engine.lib
    original_batch,original_single=lib.ti_forward_batch,lib.ti_forward
    single_rows=[row.reshape(1,64) for row in x]
    observed=[]
    def wrap(original,is_batch):
        def record(*args):
            status=original(*args)
            if status==0:
                rows=args[2] if is_batch else 1
                count=args[3] if is_batch else args[2]
                rm.require(count==rows*64, 'observed native input count mismatch')
                actual=np.ctypeslib.as_array(args[1],shape=(count,)).reshape(rows,64).copy()
                observed.append(actual)
            return status
        return record
    results={}
    lib.ti_forward_batch=wrap(original_batch,True)
    lib.ti_forward=wrap(original_single,False)
    try:
        for size in batches:
            results[str(size)]={}
            pieces=[(slice(a,b),x[a:b]) for a,b in partitions(size)]
            for name in list(engine.names)+[DIAGNOSTIC,DIRECT]:
                observed.clear();poison(values)
                answer=sweep(name,pieces,calls,engine,single_rows)
                validate_sweep(answer,reference)
                expected=[1]*450 if name in DIAGNOSTICS else [b-a for a,b in partitions(size)]
                rm.require([len(a) for a in observed]==expected and
                           np.array_equal(np.concatenate(observed),x), 'native row coverage mismatch')
                results[str(size)][name]=dict(observed_calls=len(observed),
                    observed_rows=sum(map(len,observed)),
                    batch_rows=expected,
                    input_sha256=rm.hashlib.sha256(np.concatenate(observed).astype('<f4').tobytes()).hexdigest())
    finally:
        lib.ti_forward_batch,lib.ti_forward=original_batch,original_single
    return results


def worker(args):
    with threadpool_limits(limits=1):
        values,checks = correctness(args.library,args.batches)
        _,x,_,_,engine,_,_,calls = values
        single_rows=[row.reshape(1,64) for row in x]
        pieces = {size:[(slice(a,b),x[a:b]) for a,b in partitions(size)] for size in args.batches}
        for size in args.batches:
            for call in calls.values():
                for _,block in pieces[size]:
                    call(block)
        gold = np.array(checks['reference_probabilities'])
        observations=audit_native_calls(values,args.batches,gold)
        raw = []
        for sample in range(args.samples):
            for size,name in schedule(list(calls),args.batches,args.run,sample,args.samples):
                poison(values)  # outside timing: invalidate all reused scratch/output state
                start = time.perf_counter_ns()
                answer = sweep(name,pieces[size],calls,engine,single_rows)
                elapsed = time.perf_counter_ns()-start
                validate_sweep(answer,gold)
                raw.append(dict(sample=sample,batch_size=size,backend=name,predictions=450,
                    adapter_calls=(450 if name==DIRECT else len(pieces[size])), expected_c_forward_calls=(450 if name in DIAGNOSTICS else
                    len(pieces[size]) if name in engine.names else None), elapsed_ns=elapsed,
                    ns_per_prediction=elapsed/450))
        pools = [{k:v for k,v in p.items() if k!='filepath'} for p in threadpool_info()]
        rm.require(bool(pools) and all(p['num_threads']==1 for p in pools),'uncontrolled threads')
        rm.save(args.output,dict(schema=2,run=args.run,checks=checks,raw=raw,observations=observations,versions=rm.versions(),
                    threadpools=pools,c_blas=engine.lib.ti_blas_name().decode(),
                    c_single_thread=bool(engine.lib.ti_blas_is_single_threaded())))
        print('PASS: matched batch sizes',args.batches,'all 450 outputs/classes verified')


def build(args):
    before={p:rm.digest(p) for p in SOURCES}
    rm.build(args)
    production=json.loads((rm.ROOT/'build/real-model-build.json').read_text())
    _,libs = native_flags(args.blas)
    command = ['cc','-Iinclude','-std=c11','-O3','-Wall','-Wextra','-Wpedantic','-Werror',
               'tests/test_batch.c']+rm.OBJECTS+libs+['-lm','-o','build/real_test_batch']
    subprocess.run(command,cwd=rm.ROOT,check=True)
    subprocess.run(['build/real_test_batch'],cwd=rm.ROOT,check=True)
    for mode in ([],['-O']):
        subprocess.run([sys.executable]+mode+['tests/test_batch_reference.py',str(args.library)],
                       cwd=rm.ROOT,check=True)
    rm.require(before=={p:rm.digest(p) for p in SOURCES} and
               production['library_sha256']==rm.digest(args.library) and
               all(rm.digest(p)==v for p,v in production['objects'].items()),
               'source/production objects changed during batch build tests')
    rm.save(rm.ROOT/'build/batch-model-build.json',dict(
        production=production,
        batch_test_command=command, batch_test_sha256=rm.digest('build/real_test_batch')))


def summaries(records,batches,names):
    out = {}
    for size in batches:
        out[str(size)]={}
        for name in names:
            medians = [float(np.median([s['ns_per_prediction'] for s in r['raw']
                         if s['batch_size']==size and s['backend']==name])) for r in records]
            out[str(size)][name]=dict(median_ns=float(np.median(medians)),process_medians_ns=medians,
                         min_process_median_ns=min(medians),max_process_median_ns=max(medians))
    return out


def improvements(stats):
    result={}
    for batch,rows in stats.items():
        result[batch]={}
        for title,numerator,denominator in [
            ('direct_over_batch_input_first',DIRECT,'input_first'),
            ('wrapped_over_batch_input_first',DIAGNOSTIC,'input_first'),
            ('wrapper_over_direct',DIAGNOSTIC,DIRECT)]:
            pairs=[a/b for a,b in zip(rows[numerator]['process_medians_ns'],
                                     rows[denominator]['process_medians_ns'])]
            result[batch][title]=dict(median=float(np.median(pairs)),
                 process_ratios=pairs,minimum=min(pairs),maximum=max(pairs))
    return result


def benchmark(args):
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    before={p:rm.digest(p) for p in SOURCES}
    for name,sha in before.items():
        blob=subprocess.check_output(['git','show',revision+':'+name])
        rm.require(rm.hashlib.sha256(blob).hexdigest()==sha,'commit measured sources before benchmarking')
    info=json.loads(Path('build/batch-model-build.json').read_text())
    prod=info['production']
    libhash=rm.digest(args.library)
    rm.require(prod['library_sha256']==libhash and set(prod['sources'])==set(rm.BUILD_SOURCES)
               and all(rm.digest(p)==sha for p,sha in prod['sources'].items()),'stale library/source')
    rm.require(set(prod['objects'])==set(rm.OBJECTS) and
               all(rm.digest(p)==sha for p,sha in prod['objects'].items()),'stale objects')
    buildhash=rm.digest('build/batch-model-build.json')
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for run in range(args.runs):
        path=args.output/('run-%02d.json'%run)
        subprocess.run([sys.executable,'tools/batch_model.py','worker','--library',str(args.library),
             '--batches',','.join(map(str,args.batches)),'--samples',str(args.samples),'--run',str(run),
             '--output',str(path)],check=True)
        records.append(json.loads(path.read_text()))
    rm.require(before=={p:rm.digest(p) for p in SOURCES} and rm.digest(args.library)==libhash
               and buildhash==rm.digest('build/batch-model-build.json') and
               all(rm.digest(p)==sha for p,sha in prod['objects'].items()),'build/source changed during run')
    rm.require(all(r['checks']==records[0]['checks'] for r in records),'process correctness disagreement')
    rm.save(args.output/'correctness.json',records[0]['checks'])
    names=list(records[0]['checks']['outputs'][str(args.batches[0])])
    for run,record in enumerate(records):
        record.pop('checks')
        rm.save(args.output/('run-%02d.json'%run),record)
    stats=summaries(records,args.batches,names)
    rm.save(args.output/'manifest.json',dict(schema=2,protocol='matched-batch-poisoned-direct-v2',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_revision=revision,sources=before,
        library_sha256=libhash,build_metadata=info,runs=args.runs,samples=args.samples,batches=args.batches,
        partitions={str(b):[end-start for start,end in partitions(b)] for b in args.batches},
        summaries=stats,improvements=improvements(stats),platform=dict(system=platform.system(),release=platform.release(),
        machine=platform.machine(),cpu=rm.cpu_name()),
        files={p.name:rm.digest(p) for p in args.output.glob('*.json')}))
    print(json.dumps({b:{n:round(s['median_ns']/1000,3) for n,s in rows.items()}
                      for b,rows in stats.items()},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','check','benchmark','worker'])
    parser.add_argument('--library',type=Path,default=Path('build/real_engine.so'))
    parser.add_argument('--output',type=Path,default=Path('results/local/batch-digits'))
    parser.add_argument('--blas',type=int,choices=[0,1],default=1)
    parser.add_argument('--batches',default=','.join(map(str,BATCHES)))
    parser.add_argument('--runs',type=int,default=6)
    parser.add_argument('--samples',type=int,default=20)
    parser.add_argument('--run',type=int,default=0)
    args=parser.parse_args()
    args.batches=list(map(int,args.batches.split(',')))
    rm.require(len(args.batches)>0 and len(set(args.batches))==len(args.batches) and
               all(1<=b<=450 for b in args.batches) and 1<=args.runs<=24 and 1<=args.samples<=100,
               'invalid batch/run/sample counts')
    if args.command=='build': build(args)
    elif args.command=='benchmark': benchmark(args)
    elif args.command=='worker': worker(args)
    else:
        with threadpool_limits(limits=1):
            _,checks=correctness(args.library,args.batches)
            print('PASS: matched-batch numerical checks; accuracy',
                  checks['outputs'][str(args.batches[0])]['input_first']['accuracy'])


if __name__=='__main__': main()
