"""Recompute matched-batch accuracy, full outputs, source provenance and timings."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from threadpoolctl import threadpool_limits
import batch_model as bm
import real_model as rm


def sha(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def verify(directory):
    directory=Path(directory)
    m=json.loads((directory/'manifest.json').read_text())
    rm.require(type(m['schema']) is int and m['schema']==1 and m['protocol']=='matched-batch-array-api-v1',
               'invalid batch schema/protocol')
    batches=m['batches']
    rm.require(bool(batches) and len(set(batches))==len(batches) and
               all(type(b) is int and 1<=b<=450 for b in batches),'invalid batch sizes')
    rm.require(type(m['runs']) is int and 1<=m['runs']<=24 and type(m['samples']) is int
               and 1<=m['samples']<=100,'invalid run/sample counts')
    rm.require(m['partitions']=={str(b):[end-start for start,end in bm.partitions(b)] for b in batches},
               'tail partition mismatch')
    expected_files={'correctness.json'}|{'run-%02d.json'%i for i in range(m['runs'])}
    rm.require(set(m['files'])==expected_files,'missing/extra evidence')
    for name,digest in m['files'].items():
        rm.require(Path(name).name==name and sha(digest) and rm.digest(directory/name)==digest,'evidence hash mismatch')
    rm.require(set(m['sources'])==set(bm.SOURCES) and re.fullmatch('[0-9a-f]{40}',m['source_revision']),
               'source inventory/revision mismatch')
    for name,digest in m['sources'].items():
        blob=subprocess.check_output(['git','show',m['source_revision']+':'+name],cwd=rm.ROOT)
        rm.require(sha(digest) and rm.hashlib.sha256(blob).hexdigest()==digest,'historical source mismatch')
    info=m['build_metadata'];p=info['production']
    rm.require(set(p['sources'])==set(rm.BUILD_SOURCES) and set(p['objects'])==set(rm.OBJECTS)
               and all(sha(v) for v in p['objects'].values()),'incomplete build inventory')
    rm.require(all(m['sources'][name]==digest for name,digest in p['sources'].items()),'build/source mismatch')
    rm.require(sha(m['library_sha256']) and p['library_sha256']==m['library_sha256']
               and len(p['commands'])==4 and bool(p['compiler']) and sha(info['batch_test_sha256'])
               and 'tests/test_batch.c' in info['batch_test_command']
               and all(obj in info['batch_test_command'] for obj in rm.OBJECTS),'incomplete build/test metadata')
    model,x,y,clf=rm.restore(rm.MODEL)
    gold_logits,gold_probs=bm.reference(model,x,clf)
    checks=json.loads((directory/'correctness.json').read_text())
    rm.require(checks['model_sha256']==rm.digest(rm.MODEL)==m['sources']['models/digits/model.json']
               and checks['dataset_sha256']==model['dataset_sha256'] and checks['test_ids']==model['test_ids']
               and checks['labels']==y.tolist(),'model/data mismatch')
    rm.require(checks['split_audit']==rm.split_audit(rm.dataset()[0],model['train_ids'],model['test_ids']),
               'split audit mismatch')
    rm.require(np.array_equal(gold_logits,checks['oracle_logits']) and
               np.shape(checks['reference_probabilities'])==(450,10) and
               np.allclose(gold_probs,checks['reference_probabilities'],atol=rm.ATOL,rtol=rm.RTOL),
               'stored reference mismatch')
    # Recompute historical error magnitudes against the archived provider outputs.
    gold_probs=np.array(checks['reference_probabilities'])
    rm.require(set(checks['outputs'])==set(map(str,batches)),'missing batch outputs')
    names=set(checks['outputs'][str(batches[0])])
    core={'output_first','input_first','numpy_batch','sklearn_batch',bm.DIAGNOSTIC}
    rm.require(names in (core,core|{'system_blas'}),'backend inventory mismatch')
    for size in batches:
        rm.require(set(checks['outputs'][str(size)])==names,'batch backend mismatch')
        for name,result in checks['outputs'][str(size)].items():
            probs=np.array(result['probabilities']);logits=result['logits']
            rm.require(probs.shape==(450,10) and np.isfinite(probs).all() and
                       np.allclose(probs,gold_probs,atol=rm.ATOL,rtol=rm.RTOL) and
                       np.array_equal(probs.argmax(1),gold_probs.argmax(1)),'stored batch probability mismatch')
            rm.require(result['correct']==int((probs.argmax(1)==y).sum()) and
                       result['accuracy']==float(np.mean(probs.argmax(1)==y))>=.94 and
                       result['max_probability_error']==float(np.max(abs(probs-gold_probs))), 'accuracy/probability statistics mismatch')
            if name not in {'sklearn_batch',bm.DIAGNOSTIC}:
                logits=np.array(logits)
                rm.validate_outputs(logits,probs,gold_logits,gold_probs)
                rm.require(result['max_logit_error']==float(np.max(abs(logits-gold_logits))),'logit error mismatch')
            else:
                rm.require(logits is None and result['max_logit_error'] is None,'unexpected logit evidence')
    records=[]
    for i in range(m['runs']):
        r=json.loads((directory/('run-%02d.json'%i)).read_text())
        rm.require(r['schema']==1 and r['run']==i and r['threadpools'] and
                   any(t['user_api']=='blas' for t in r['threadpools']) and
                   all(type(t['num_threads']) is int and t['num_threads']==1 for t in r['threadpools']),
                   'run/thread policy mismatch')
        rm.require(set(r['versions'])=={'python','numpy','scipy','sklearn'} and
                   all(isinstance(v,str) and v for v in r['versions'].values()) and bool(r['c_blas']),
                   'missing provider/version metadata')
        if 'system_blas' in names:rm.require(r['c_single_thread'] is True,'native BLAS threads')
        expected=[]
        for sample in range(m['samples']):
            expected.extend((sample,b,n) for b,n in bm.schedule(sorted(names),batches,i,sample,m['samples']))
        rm.require([(row['sample'],row['batch_size'],row['backend']) for row in r['raw']]==expected,
                   'sample/backend/order mismatch')
        for row in r['raw']:
            calls=len(bm.partitions(row['batch_size']))
            expected_native=(450 if row['backend']==bm.DIAGNOSTIC else calls
                              if row['backend'] in {'output_first','input_first','system_blas'} else None)
            rm.require(row['predictions']==450 and row['adapter_calls']==calls and
                       row['c_forward_calls']==expected_native and type(row['elapsed_ns']) is int
                       and row['elapsed_ns']>0 and row['ns_per_prediction']==row['elapsed_ns']/450,
                       'call counts/timing mismatch')
        records.append(r)
    rm.require(m['summaries']==bm.summaries(records,batches,sorted(names)),'summary mismatch')
    print('PASS: matched-batch references, every saved row, accuracy, tails, raw timings and historical provenance')


if __name__=='__main__':
    with threadpool_limits(limits=1):verify(sys.argv[1])
