"""Actual trained-model batches, every row and tail, and adapter boundaries."""
import ctypes as C
from pathlib import Path
import sys
import unittest
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


if __name__=='__main__':unittest.main()
