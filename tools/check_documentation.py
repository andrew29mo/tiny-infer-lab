"""Check published Markdown claims against the archived, separately verified reports.

Standard library only; never regenerate or relabel historical measurements.
"""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def require(condition,message):
    if not condition:raise ValueError(message)


def read_json(root,name):
    return json.loads((root/'results/published'/name).read_text())


def table(text,first_header):
    found=[];active=False
    for line in text.splitlines():
        if not line.startswith('|'):
            if active:break
            continue
        cells=[s.strip() for s in line.strip('|').split('|')]
        if cells[0]==first_header:
            require(not active,'duplicate table header')
            active=True;continue
        if active and not all(re.fullmatch(r':?-+:?',cell) for cell in cells):found.append(cells)
    require(active,'missing table: '+first_header)
    return found


def expect_table(root,path,header,expected):
    require(table((root/path).read_text(),header)==expected,'table disagrees with evidence: '+path+' / '+header)


def section(text,heading):
    marker='## '+heading+'\n'
    require(text.count(marker)==1,'missing/duplicate section: '+heading)
    return text.split(marker,1)[1].split('\n## ',1)[0]


def check(root=ROOT):
    report=read_json(root,'apple-m5-pro-20261014/report.json')
    rows=[];ranges=[]
    gains=[];blas_wins=0
    for w in report['workloads']:
        s=w['summary'];ratios=s['ratios'];shape=w['shape']
        rows.append(['→'.join(map(str,shape))]+['%.3f'%(s[k]['median_ns']/1000)
                    for k in ['output_first','input_first','system_blas']]+[
                    '%.2f×'%ratios[k]['ratio_of_pooled_medians']
                    for k in ['output_over_input_first','input_over_system_blas']])
        ranges.append([w['name']]+['%.3f–%.3f'%(ratios[k]['process_ratio_min'],ratios[k]['process_ratio_max'])
                     for k in ['output_over_input_first','input_over_system_blas']])
        gains.append(ratios['output_over_input_first']['ratio_of_pooled_medians'])
        blas_wins+=s['system_blas']['median_ns']<s['input_first']['median_ns']
    expect_table(root,'docs/RESULTS.md','Input→hidden→output',rows)
    expect_table(root,'docs/RESULTS.md','Shape',ranges)
    real_specs=[('digits-reviewed-20261014','API/contract',[
        ('C array API: output_first','output_first'),('C array API: input_first','input_first'),
        ('C array API: system BLAS','system_blas'),('sklearn public API: single image','sklearn_single'),
        ('NumPy prepared input/reused buffers (diagnostic)','numpy_prepared_single'),
        ('sklearn 450-image batch (amortized)','sklearn_batch')]),
        ('digits-20261014','API',[
        ('C `output_first` + ctypes + softmax','output_first'),('C `input_first` + ctypes + softmax','input_first'),
        ('C system BLAS + ctypes + softmax','system_blas'),
        ('sklearn `predict_proba`, one image per call','sklearn_single'),
        ('sklearn `predict_proba`, 450 images per call (amortized)','sklearn_batch')])]
    for folder,header,labels in real_specs:
        m=read_json(root,folder+'/manifest.json');expected=[]
        for label,name in labels:
            r=m['summaries'][name]
            expected.append([label,'%.3f'%(r['median_ns']/1000),
                    '%.3f–%.3f'%(r['min_process_median_ns']/1000,r['max_process_median_ns']/1000)])
        expect_table(root,'docs/REAL_MODEL.md',header,expected)
        require(m['source_revision'][:7] in (root/'docs/REAL_MODEL.md').read_text(),'missing real-model source attribution')
    for folder,header,names in [
        ('digits-batch-reviewed-20261014','Batch',['input_first','system_blas','numpy_batch','sklearn_batch',
                                             'c_single_direct_input_first','c_single_loop_input_first']),
        ('digits-batch-20261014','Requested batch',['input_first','system_blas','numpy_batch','sklearn_batch',
                                                'c_single_loop_input_first'])]:
        m=read_json(root,folder+'/manifest.json')
        rows=[[str(b)]+['%.3f'%(m['summaries'][str(b)][name]['median_ns']/1000) for name in names] for b in m['batches']]
        expect_table(root,'docs/BATCH_INFERENCE.md',header,rows)
        require(m['source_revision'][:7] in (root/'docs/BATCH_INFERENCE.md').read_text(),'missing batch source attribution')
    m=read_json(root,'digits-batch-reviewed-20261014/manifest.json')
    paired=m['improvements']['450']['direct_over_batch_input_first']
    batch_text=section((root/'docs/BATCH_INFERENCE.md').read_text(),'Reviewed results')
    require('**%.2f×**'%paired['median'] in batch_text and
            '%.2f–%.2f×'%(paired['minimum'],paired['maximum']) in batch_text,'reviewed ratio/range mismatch')
    old=read_json(root,'digits-batch-20261014/manifest.json')['summaries']['450']
    ratio=old['c_single_loop_input_first']['median_ns']/old['input_first']['median_ns']
    require('**%.2f×**'%ratio in section((root/'docs/BATCH_INFERENCE.md').read_text(),
            'Original results (historical wrapper comparison)'),'historical wrapper ratio mismatch')
    checks=read_json(root,'digits-batch-reviewed-20261014/correctness.json')
    labels=checks['labels'];prob=checks['reference_probabilities']
    correct=sum(max(range(10),key=row.__getitem__)==label for row,label in zip(prob,labels))
    require(len(labels)==len(prob)==450 and all(v['correct']==correct for rows in checks['outputs'].values()
            for v in rows.values()),'accuracy evidence mismatch')
    readme=(root/'README.md').read_text()
    current=section(readme,'Reviewed batch-control measurements')
    for claim in ['%d/450 (%.2f%%)'%(correct,100*correct/450),
                  '**%.2f×**'%paired['median'],m['source_revision'][:7],
                  '%.2f–%.2f×'%(paired['minimum'],paired['maximum'])]:
        require(claim in current,'README reviewed claim mismatch: '+claim)
    archived=section(readme,'Archived synthetic measurements')
    original=next(w for w in report['workloads'] if w['shape']==[128,256,32])['summary']['ratios']
    for claim in ['**%.2f× to %.2f×**'%(min(gains),max(gains)),
                  '**five of six**' if blas_wins==5 else '**%d of six**'%blas_wins,
                  report['source_git_commit'][:7],
                  '**%.2f×**'%original['output_over_input_first']['ratio_of_pooled_medians'],
                  '**%.2f×**'%original['input_over_system_blas']['ratio_of_pooled_medians']]:
        require(claim in archived,'README synthetic claim mismatch: '+claim)
    # Relative documentation links, including issue/PR templates; external links are not fetched.
    documents=[root/'README.md',root/'CONTRIBUTING.md',root/'results/published/README.md']
    documents += list((root/'docs').rglob('*.md')) + list((root/'models').rglob('*.md'))
    documents += list((root/'.github').rglob('*.md'))
    for path in sorted(documents):
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in link or link.startswith('#'):continue
            require((path.parent/link.split('#')[0]).exists(),'broken relative documentation link: '+str(path.relative_to(root)))
    print('PASS: all six published result tables, scoped README accuracy/ratios and relative documentation links')


if __name__=='__main__':check()
