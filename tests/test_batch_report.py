"""Report adversarial fixtures bypass hashes to challenge actual semantics."""
import json
from collections import Counter
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import real_model as rm
import batch_model as bm
from verify_batch import verify
BASELINE=rm.ROOT/'results/published/digits-batch-reviewed-20261014'
LEGACY=rm.ROOT/'results/published/digits-batch-20261014'


class ReportTests(unittest.TestCase):
    def test_baseline(self):
        with threadpool_limits(limits=1):
            verify(BASELINE)
            verify(LEGACY)

    def test_primary_orders_are_balanced_in_actual_samples(self):
        m=json.loads((BASELINE/'manifest.json').read_text())
        for batch in m['batches']:
            counts=Counter()
            for i in range(m['runs']):
                rows=json.loads((BASELINE/('run-%02d.json'%i)).read_text())['raw']
                for sample in range(m['samples']):
                    order=tuple(row['backend'] for row in rows if row['sample']==sample and
                                row['batch_size']==batch and row['backend'] not in bm.DIAGNOSTICS)
                    counts[order]+=1
            self.assertEqual(len(counts),math.factorial(5))
            self.assertEqual(set(counts.values()),{1})

    def test_corruption_rejected(self):
        defects=['tail','missing_batch','last_row','calls','timing','order','summary','source','threads','reference','observed_tail','observed_input','paired_ratio','link','flags']
        for defect in defects:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as temp:
                d=Path(temp)/'report';shutil.copytree(BASELINE,d)
                m=json.loads((d/'manifest.json').read_text())
                r=json.loads((d/'run-00.json').read_text())
                c=json.loads((d/'correctness.json').read_text())
                if defect=='tail':m['partitions']['128'][-1]-=1
                elif defect=='missing_batch':c['outputs'].pop('8')
                elif defect=='last_row':c['outputs']['128']['input_first']['probabilities'][-1]=[0.1]*10
                elif defect=='calls':r['raw'][0]['adapter_calls']+=1
                elif defect=='timing':r['raw'][0]['elapsed_ns']*=2
                elif defect=='order':r['raw'][0],r['raw'][1]=r['raw'][1],r['raw'][0]
                elif defect=='summary':m['summaries']['32']['input_first']['median_ns']+=1
                elif defect=='source':m['sources']={}
                elif defect=='threads':r['threadpools']=[]
                elif defect=='reference':c['oracle_logits'][-1][0]+=1
                elif defect=='observed_tail':r['observations']['128']['input_first']['batch_rows'][-1]-=1
                elif defect=='observed_input':r['observations']['8']['input_first']['input_sha256']='0'*64
                elif defect=='paired_ratio':m['improvements']['450']['direct_over_batch_input_first']['median']+=1
                elif defect=='link':m['build_metadata']['batch_test_command'].remove('build/real_tiny_infer.o')
                else:m['build_metadata']['production']['commands'][0].remove('-ffp-contract=off')
                rm.save(d/'correctness.json',c);rm.save(d/'run-00.json',r)
                m['files']={name:rm.digest(d/name) for name in m['files']}
                rm.save(d/'manifest.json',m)
                with self.assertRaises(ValueError),threadpool_limits(limits=1):verify(d)


if __name__=='__main__':unittest.main()
