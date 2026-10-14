"""Independent float64 checks of the production batch symbol, including last rows."""
import ctypes as C
import random
import sys
from test_reference import Model, F, load, oracle, check


def run(path):
    lib = load(path)
    rng = random.Random(20261014)
    cases = [(1,1,1,1), (3,2,2,1), (7,31,17,9), (16,7,5,3), (33,64,32,10)]
    cases += [(rng.randint(1,17),rng.randint(1,25),rng.randint(1,25),rng.randint(1,15)) for _ in range(40)]
    worst = 0
    for batch, ni, nh, no in cases:
        def array(n):
            return (F * n)(*(rng.uniform(-0.5, 0.5) for _ in range(n)))
        x,w1,b1,w2,b2 = array(batch*ni),array(ni*nh),array(nh),array(nh*no),array(no)
        model = Model()
        check(lib.ti_model_init(C.byref(model),ni,nh,no,w1,len(w1),b1,nh,w2,len(w2),b2,no)==0,'init')
        reference = [oracle(list(x)[r*ni:(r+1)*ni],w1,b1,w2,b2,nh,no) for r in range(batch)]
        for k in range(3 if lib.ti_blas_available() else 2):
            h,y = (F*(batch*nh))(),(F*(batch*no))()
            check(lib.ti_forward_batch(C.byref(model),x,batch,len(x),h,len(h),y,len(y),k)==0,'batch status')
            for r in range(batch):
                for j in range(no):
                    err=abs(y[r*no+j]-reference[r][j]);worst=max(worst,err)
                    check(err <= 1e-5+3e-5*abs(reference[r][j]), 'batch oracle mismatch')
                single_h,single_y=(F*nh)(),(F*no)()
                row=(F*ni)(*list(x)[r*ni:(r+1)*ni])
                check(lib.ti_forward(C.byref(model),row,ni,single_h,nh,single_y,no,k)==0,'single status')
                for j in range(no):
                    check(abs(y[r*no+j]-single_y[j])<=1e-5+3e-5*abs(single_y[j]),'batch/single mismatch')
    print('PASS: 45 batch/shape cases against float64 oracle and production single API; max error %.9g'%worst)


if __name__ == '__main__':
    run(sys.argv[1])
