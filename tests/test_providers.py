"""Challenge the actual production adapters/exports and benchmark collector."""
import argparse
import contextlib
import ctypes as C
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import compare_providers as cp
import provider_runtime as pr
import real_model as rm


class ProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args=argparse.Namespace(artifacts=cp.ARTIFACTS,batches=cp.BATCHES)
        cls.model,cls.x,cls.y,cls.clf=rm.restore(rm.MODEL)
        cls.gold=cls.clf.predict_proba(cls.x)

    def test_all_production_outputs(self):
        checks=cp.check(self.args)
        for rows in checks['outputs'].values():
            self.assertEqual(set(rows),set(pr.NAMES))
            self.assertTrue(all(v['correct']==438 for v in rows.values()))

    def test_actual_native_and_ort_inputs_cover_tail(self):
        # Interpose only at the actual native/session boundary, outside measurement.
        for name in ('tiny_input','tiny_blas','emlearn','onnxruntime'):
            with self.subTest(backend=name):
                provider=pr.Provider(name,cp.ARTIFACTS);observed=[]
                if name=='onnxruntime':
                    original=provider.session.run
                    def wrapped(outputs,inputs):
                        observed.append(inputs['input'].copy());return original(outputs,inputs)
                    provider.session.run=wrapped
                else:
                    symbol='comparison_emlearn' if name=='emlearn' else 'ti_forward_batch'
                    original=getattr(provider.lib,symbol)
                    def wrapped(*args):
                        ptr,n=(args[0],args[1]) if name=='emlearn' else (args[1],args[2])
                        observed.append(np.ctypeslib.as_array(ptr,shape=(n*64,)).reshape(n,64).copy())
                        return original(*args)
                    setattr(provider.lib,symbol,wrapped)
                for batch in (8,128,450):
                    observed.clear();provider.poison()
                    pr.validate(pr.sweep(provider,self.x,batch),self.gold)
                    self.assertEqual([len(a) for a in observed],[b-a for a,b in pr.partitions(batch)])
                    np.testing.assert_array_equal(np.concatenate(observed),self.x)

    def test_warm_wrong_or_missing_tail_rejected(self):
        p=pr.Provider('emlearn',cp.ARTIFACTS)
        pr.validate(pr.sweep(p,self.x,128),self.gold)
        original=p.lib.comparison_emlearn
        def drop_tail(x,n,out):
            return 0 if n==66 else original(x,n,out)
        p.lib.comparison_emlearn=drop_tail
        with self.assertRaises(ValueError):pr.validate(pr.sweep(p,self.x,128),self.gold)
        def wrong(x):return np.full((len(x),10),.1,np.float32)
        with self.assertRaises(ValueError):pr.validate(pr.sweep(wrong,self.x,128),self.gold)

    def test_export_bits_and_changed_onnx_rejected(self):
        cp.audit_exports(cp.ARTIFACTS,self.clf)
        import onnx
        from onnx import numpy_helper as nh
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            for name in ('emlearn.so','digits.onnx','weights.npz'):
                shutil.copyfile(cp.ARTIFACTS/name,path/name)
            cp.audit_exports(path,self.clf)  # valid fixture before corruption
            model=onnx.load(path/'digits.onnx')
            arr=nh.to_array(model.graph.initializer[1]).copy();arr[0]+=1
            model.graph.initializer[1].CopyFrom(nh.from_array(arr,'b1'))
            onnx.save(model,path/'digits.onnx')
            with self.assertRaises(ValueError):cp.audit_exports(path,self.clf)

    def test_compiled_emlearn_tail_mutation_rejected(self):
        # Compile a real faulty adapter against the real upstream implementation.
        with np.errstate():
            import emlearn
        include=Path(emlearn.__file__).parent
        source=(ROOT/'tools/provider_emlearn.c').read_text()
        self.assertIn('i < rows;',source)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);shutil.copyfile(cp.ARTIFACTS/'digits_emlearn.h',path/'digits_emlearn.h')
            (path/'bad.c').write_text(source.replace('i < rows;','i + 1 < rows;'))
            subprocess.run(['cc','-std=c11','-O3','-ffp-contract=off','-shared','-fPIC',
                '-I'+str(path),'-I'+str(include),str(path/'bad.c'),'-lm','-o',str(path/'emlearn.so')],check=True)
            provider=pr.Provider('emlearn',path)
            for batch in (1,8,128,450):
                provider.poison()
                with self.assertRaises(ValueError):pr.validate(pr.sweep(provider,self.x,batch),self.gold)

    def test_schedule_and_partitions(self):
        for batch in cp.BATCHES:
            self.assertEqual(sum(b-a for a,b in pr.partitions(batch)),450)
        for run in range(6):
            positions={n:[] for n in pr.NAMES}
            for sample in range(6):
                names=[n for b,n in cp.order(run,sample,[128])]
                self.assertEqual(set(names),set(pr.NAMES))
                for i,n in enumerate(names):positions[n].append(i)
            self.assertTrue(all(sorted(v)==list(range(6)) for v in positions.values()))

    def test_bad_input_rejected_by_all_adapters(self):
        with threadpool_limits(limits=1):
            for n in pr.NAMES:
                p=pr.Provider(n,cp.ARTIFACTS)
                for x in [np.zeros((0,64),np.float32),np.zeros((1,63),np.float32),np.full((1,64),np.nan)]:
                    with self.assertRaises(ValueError):p(x)


if __name__=='__main__':unittest.main()
