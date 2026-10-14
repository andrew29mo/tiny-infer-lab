"""Actual trained-model batches, every row and tail, and adapter boundaries."""
import ctypes as C
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import batch_model as bm
import real_model as rm

LIBRARY=sys.argv.pop(1) if len(sys.argv)>1 else 'build/real_engine.so'


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with threadpool_limits(limits=1):
            cls.values,cls.checks=bm.correctness(LIBRARY)

    def test_all_heldout_examples_and_tails(self):
        for size,backends in self.checks['outputs'].items():
            self.assertEqual(sum(b-a for a,b in bm.partitions(int(size))),450)
            for result in backends.values():
                self.assertEqual(result['correct'],438)
                self.assertEqual(np.shape(result['probabilities']),(450,10))
        self.assertEqual([b-a for a,b in bm.partitions(128)],[128,128,128,66])

    def test_batch_one_matches_single(self):
        _,x,_,_,engine,c,_,_=self.values
        for k in range(len(engine.names)):
            np.testing.assert_allclose(c.probabilities(x[:1],k),engine.predict_proba(x[:1],k),atol=rm.ATOL,rtol=rm.RTOL)

    def test_invalid_adapter_inputs(self):
        c,numpy=self.values[5:7]
        for fn in [lambda block:c.probabilities(block,0),numpy.probabilities]:
            for bad in [np.zeros((0,64),np.float32),np.zeros((2,63),np.float32),
                        np.zeros(64,np.float32),np.full((8,64),np.nan,np.float32)]:
                with self.assertRaises(ValueError):fn(bad)

    def test_last_row_is_not_ignored(self):
        _,x,_,_,engine,c,_,_=self.values
        for k in range(len(engine.names)):
            changed=x[:8].copy()
            initial=c.logits(changed,k).copy()
            changed[-1]=1-changed[-1]
            actual=c.logits(changed,k).copy()
            np.testing.assert_array_equal(initial[:-1],actual[:-1])
            self.assertGreater(float(np.max(abs(initial[-1]-actual[-1]))),0.01)

    def test_tail_partitions_cover_each_index_once(self):
        expected={1:[1]*450,8:[8]*56+[2],32:[32]*14+[2],128:[128]*3+[66],450:[450]}
        for batch,lengths in expected.items():
            pieces=bm.partitions(batch)
            self.assertEqual([b-a for a,b in pieces],lengths)
            self.assertEqual([i for a,b in pieces for i in range(a,b)],list(range(450)))

    def test_stale_tail_cache_is_rejected(self):
        _,x,_,_,engine,c,_,calls=self.values
        # Warm each exact tail first, reproducing the old false-pass condition.
        actual=engine.lib.ti_forward_batch
        rows=[row.reshape(1,64) for row in x]
        for batch,tail in [(8,2),(32,2),(128,66)]:
            with self.subTest(batch=batch):
                c.probabilities(x[-tail:],1)
                def skip_tail(*args):
                    return 0 if args[2]==tail else actual(*args)
                pieces=[(slice(a,b),x[a:b]) for a,b in bm.partitions(batch)]
                with mock.patch.object(engine.lib,'ti_forward_batch',side_effect=skip_tail):
                    bm.poison(self.values)
                    with self.assertRaisesRegex(ValueError,'nonfinite logits'):
                        bm.sweep('input_first',pieces,calls,engine,rows)

    def test_dropped_tail_collection_is_rejected(self):
        _,x,_,_,engine,_,_,calls=self.values
        pieces=[(slice(a,b),x[a:b]) for a,b in bm.partitions(128)]
        rows=[row.reshape(1,64) for row in x]
        gold=np.array(self.checks['reference_probabilities'])
        good=bm.sweep('input_first',pieces,calls,engine,rows)
        bm.validate_sweep(good,gold)
        bad=bm.sweep('input_first',pieces[:-1],calls,engine,rows)
        with self.assertRaisesRegex(ValueError,'timed batch mismatch'):
            bm.validate_sweep(bad,gold)

    def test_observed_native_calls_and_inputs_cover_all_rows(self):
        gold=np.array(self.checks['reference_probabilities'])
        observations=bm.audit_native_calls(self.values,list(bm.BATCHES),gold)
        for batch,expected in [(8,[8]*56+[2]),(32,[32]*14+[2]),(128,[128]*3+[66]),(450,[450])]:
            seen=observations[str(batch)]
            self.assertEqual(seen['input_first']['batch_rows'],expected)
            self.assertEqual(seen[bm.DIRECT]['batch_rows'],[1]*450)
            self.assertEqual(seen['input_first']['input_sha256'],seen[bm.DIRECT]['input_sha256'])

    def test_direct_path_has_no_nested_batch_wrapper(self):
        _,x,_,_,engine,_,_,calls=self.values
        rows=[row.reshape(1,64) for row in x]
        direct=dict(calls)
        direct[bm.DIRECT]=mock.Mock(side_effect=AssertionError('nested wrapper used'))
        result=bm.sweep(bm.DIRECT,[],direct,engine,rows)
        bm.validate_sweep(result,np.array(self.checks['reference_probabilities']))


if __name__=='__main__':unittest.main()
