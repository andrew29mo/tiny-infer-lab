"""Compiled production batch faults must fail numerically, not crash/import-fail."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from workloads import native_flags
parser=argparse.ArgumentParser()
parser.add_argument('--blas',type=int,choices=[0,1],default=0)
args=parser.parse_args()
defines,libs=native_flags(args.blas)
engine=(ROOT/'src/tiny_infer.c').read_text()
blas=(ROOT/'src/blas_backend.c').read_text()
mutations=[
 ('batch_relu','engine','if (h[index] < 0) h[index] = 0;','if (h[index] < 0) h[index] = h[index];'),
 ('batch_last_row','engine','for (size_t row = 0; row < batch; ++row)\n            dense',
  'for (size_t z = 0; z < batch * out; ++z) y[z] = 0;\n        for (size_t row = 0; row + 1 < batch; ++row)\n            dense'),
 ('batch_row_stride','engine','dense(x + row * in, w, b, y + row * out','dense(x, w, b, y + row * out'),
 ('batch_second_layer','engine','batch_dense(h, m->w2, m->b2, y, batch, m->hidden_dim, m->output_dim, kernel);',
  'for (size_t j = 0; j < by; ++j) y[j] = 0;')]
if args.blas:
    mutations += [('gemm_bias','blas','w, (int)out, 1.0f, y, (int)out','w, (int)out, 0.0f, y, (int)out'),
                  ('gemm_batch_rows','blas','(int)batch, (int)out, (int)in','(int)(batch - 1), (int)out, (int)in')]
for name,where,target,replacement in mutations:
    source=engine if where=='engine' else blas
    if source.count(target)!=1:raise SystemExit('mutation target changed: '+name)
    with tempfile.TemporaryDirectory(dir=ROOT/'build') as temp:
        path=Path(temp);e=path/'engine.c';b=path/'blas.c';library=path/'fault.so'
        e.write_text(engine.replace(target,replacement) if where=='engine' else engine)
        b.write_text(blas.replace(target,replacement) if where=='blas' else blas)
        subprocess.run(['cc','-Iinclude','-Isrc','-std=c11','-O3','-ffp-contract=off','-fPIC','-shared']+
                       defines+[str(e),str(b)]+libs+['-lm','-o',str(library)],cwd=ROOT,check=True)
        # Both independent synthetic oracle and the learned-model pipeline must reject.
        commands=[['tests/test_batch_reference.py',str(library)],
                  ['-c',"import sys;sys.path.insert(0,'tools');import batch_model as b;b.correctness(sys.argv[1])",str(library)]]
        for mode in ([],['-O']):
            for command in commands:
                result=subprocess.run([sys.executable]+mode+command,cwd=ROOT,capture_output=True,text=True)
                if result.returncode==0 or not any(s in result.stderr for s in
                    ['AssertionError: batch oracle mismatch','AssertionError: batch/single mismatch',
                     'ValueError: batch probability mismatch','ValueError: logit mismatch']):
                    raise SystemExit('fault not numerically rejected: '+name+'\n'+result.stderr)
        print('PASS: synthetic/trained batch checks reject '+name+' under normal Python and -O')
