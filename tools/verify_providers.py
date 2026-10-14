"""Verify archived provider evidence without trusting headline statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import compare_providers as cp
import provider_runtime as pr
import real_model as rm


def verify(directory):
    directory=Path(directory)
    m=json.loads((directory/'manifest.json').read_text())
    rm.require(m['schema']==1 and m['protocol']=='same-model-providers-v1','schema/protocol mismatch')
    batches=m['batches']
    rm.require(batches and len(batches)==len(set(batches)) and all(type(b) is int and 1<=b<=450 for b in batches),
               'invalid batches')
    for name,maximum in [('runs',24),('samples',100),('repeats',32),('memory_runs',5)]:
        rm.require(type(m[name]) is int and 1<=m[name]<=maximum,'invalid count: '+name)
    expected={'correctness.json','memory.json','deployment.json'}|{'run-%02d.json'%i for i in range(m['runs'])}
    rm.require(set(m['files'])==expected,'incomplete evidence inventory')
    for name,sha in m['files'].items():
        rm.require(rm.digest(directory/name)==sha,'evidence hash mismatch')
    rm.require(m['backends']==list(pr.NAMES) and m['partitions']=={
        str(b):[z-a for a,z in pr.partitions(b)] for b in batches},'backends/partition mismatch')
    rm.require(set(m['sources'])==set(cp.SOURCES),'source inventory mismatch')
    for name,sha in m['sources'].items():
        blob=subprocess.check_output(['git','show',m['source_revision']+':'+name],cwd=rm.ROOT)
        rm.require(hashlib.sha256(blob).hexdigest()==sha,'historical source mismatch')
    build=m['build']
    rm.require(build['sources']==m['sources'] and build['versions']==m['versions'], 'stale build sources/versions')
    rm.require(set(build['artifacts'])=={'tiny.so','emlearn.so','weights.npz','input.npy','gold.npy',
                                       'digits_emlearn.h','digits.onnx'},'incomplete artifact inventory')
    rm.require(build['tiny_build']['library_sha256']==build['artifacts']['tiny.so'] and
        all(build['tiny_build']['sources'][n]==m['sources'][n] for n in rm.BUILD_SOURCES), 'tiny build mismatch')
    eml=build['emlearn_build']
    rm.require(eml['exact_float32_literals'] is True and eml['headers'] and
        eml['command']==['cc','-std=c11','-O3','-ffp-contract=off','-fPIC','-shared',
            '-Ibuild/comparison','-I<installed-emlearn-headers>','tools/provider_emlearn.c',
            '-lm','-o','build/comparison/emlearn.so'],'emlearn build contract mismatch')
    model,x,y,clf=rm.restore(rm.MODEL)
    w1,w2=[a.ravel().tolist() for a in clf.coefs_];b1,b2=[a.tolist() for a in clf.intercepts_]
    gold=pr.softmax(np.array([rm.oracle(row.tolist(),w1,b1,w2,b2,32,10) for row in x]))
    audit=dict(parameter_sha256=hashlib.sha256(cp.parameter_bytes(clf)).hexdigest(),parameter_count=2410,
        compiled_emlearn_bitwise_equal=True,onnx_initializers_bitwise_equal=True,numpy_archive_bitwise_equal=True)
    c=json.loads((directory/'correctness.json').read_text())
    rm.require(c['model_sha256']==rm.digest(rm.MODEL)==m['sources']['models/digits/model.json'] and
        c['dataset_sha256']==model['dataset_sha256'] and c['test_ids']==model['test_ids'] and
        c['labels']==y.tolist(),'model/split mismatch')
    rm.require(c['export_audit']==build['export_audit']==audit,'parameter export mismatch')
    stored_gold=np.asarray(c['reference_probabilities'])
    rm.require(stored_gold.shape==(450,10) and np.isfinite(stored_gold).all() and
        np.allclose(stored_gold,gold,atol=1e-14,rtol=1e-12),'stored oracle mismatch')
    rm.require(set(c['outputs'])==set(map(str,batches)),'missing batch correctness')
    for b in batches:
        rm.require(set(c['outputs'][str(b)])==set(pr.NAMES),'missing provider correctness')
        for n,result in c['outputs'][str(b)].items():
            answer=np.array(result['probabilities'],dtype=np.float32)
            pr.validate(answer,gold)
            rm.require(result['correct']==int((answer.argmax(1)==y).sum()) and
                result['max_abs_error']==float(abs(answer-stored_gold).max()),'incorrect accuracy/error statistics')
    records=[]
    for i in range(m['runs']):
        r=json.loads((directory/('run-%02d.json'%i)).read_text())
        rm.require(r['run']==i and r['versions']==m['versions'] and r['all_outputs_checked'] is True,
                   'run/version/check mismatch')
        rm.require(r['threadpools'] and all(p['num_threads']==1 for p in r['threadpools']), 'uncontrolled threadpool')
        rm.require(r['ort']==dict(providers=['CPUExecutionProvider'],intra_threads=1,inter_threads=1,
            execution='sequential',optimization='all'),'ORT execution policy mismatch')
        expected=[(s,b,n) for s in range(m['samples']) for b,n in cp.order(i,s,batches)]
        rm.require([(r['sample'],r['batch'],r['backend']) for r in r['raw']]==expected,'timing schedule mismatch')
        for row in r['raw']:
            rm.require(type(row['elapsed_ns']) is int and row['elapsed_ns']>0 and
                row['predictions']==450*m['repeats'] and
                row['ns_per_prediction']==row['elapsed_ns']/row['predictions'] and
                row['adapter_calls']==len(pr.partitions(row['batch']))*m['repeats'] and
                row['emlearn_single_calls']==(450*m['repeats'] if row['backend']=='emlearn' else None),
                'timing denominator/call mismatch')
        records.append(r)
    rm.require(m['summaries']==cp.summarize(records,batches),'summary/raw mismatch')
    memory=json.loads((directory/'memory.json').read_text())
    rm.require([(r['repeat'],r['batch'],r['backend']) for r in memory]==[
        (i,b,n) for i in range(m['memory_runs']) for b in batches for n in pr.NAMES],'memory coverage mismatch')
    for r in memory:
        rm.require(r['correctness_checked'] is True and all(type(r[k]) is int and r[k]>0 for k in
            ['baseline_rss_bytes','loaded_rss_bytes','warm_rss_bytes','process_peak_rss_bytes','load_ns']) and
            r['incremental_warm_rss_bytes']==r['warm_rss_bytes']-r['baseline_rss_bytes'],'invalid memory metrics')
        rm.require(r['threadpools'] and all(p['num_threads']==1 for p in r['threadpools']),
                   'uncontrolled memory-probe threadpool')
    size=json.loads((directory/'deployment.json').read_text())
    rm.require(set(size['providers'])==set(pr.NAMES) and set(size['artifact_bytes'])==
        {'tiny.so','emlearn.so','digits.onnx','weights.npz'},'deployment inventory mismatch')
    for n,r in size['providers'].items():
        rm.require(r['artifact_bytes']==sum(size['artifact_bytes'][p] for p in r['artifacts']) and
            r['additional_package_bytes']==sum(size['packages'][p]['bytes'] for p in r['additional_packages']),
            'deployment total mismatch')
    rm.require(size['common_packages']==['numpy'] and
        size['common_package_bytes']==size['packages']['numpy']['bytes'] and size['exclusions'],
        'deployment common/excluded footprint mismatch')
    print('PASS: six providers, frozen parameters/split, full outputs, raw timing/RSS/size and source provenance')
    return m


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    verify(p.parse_args().directory)
