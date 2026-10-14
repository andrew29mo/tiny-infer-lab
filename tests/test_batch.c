#include "tiny_infer.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); exit(1); \
} } while (0)

int main(void) {
    if (ti_blas_available()) CHECK(ti_blas_configure() == TI_OK);
    float w1[] = {1,0,0,1}, b1[] = {-1,2}, w2[] = {3,-2}, b2[] = {0.5f};
    float x[] = {2,-1,-2,-3,4,1}, h[8] = {0}, y[5] = {0};
    float gold[] = {1.5f,0.5f,3.5f};
    ti_model m;
    CHECK(ti_model_init(&m,2,2,1,w1,4,b1,2,w2,2,b2,1) == TI_OK);
    for (int k = 0; k < (ti_blas_available() ? 3 : 2); ++k) {
        h[0]=91; h[7]=92; y[0]=93; y[4]=94;
        CHECK(ti_forward_batch(&m,x,3,6,h+1,6,y+1,3,(ti_kernel)k) == TI_OK);
        for (size_t row=0; row<3; ++row) CHECK(y[row+1] == gold[row]);
        CHECK(h[0]==91 && h[7]==92 && y[0]==93 && y[4]==94);
        CHECK(ti_forward_batch(&m,x,3,5,h+1,6,y+1,3,(ti_kernel)k)==TI_CAPACITY);
        CHECK(ti_forward_batch(&m,x,3,6,h+1,5,y+1,3,(ti_kernel)k)==TI_CAPACITY);
        CHECK(ti_forward_batch(&m,x,3,6,h+1,6,y+1,2,(ti_kernel)k)==TI_CAPACITY);
        CHECK(ti_forward_batch(&m,x,0,6,h+1,6,y+1,3,(ti_kernel)k)==TI_INVALID);
        CHECK(ti_forward_batch(&m,NULL,3,6,h+1,6,y+1,3,(ti_kernel)k)==TI_INVALID);
        CHECK(ti_forward_batch(&m,x,3,6,NULL,6,y+1,3,(ti_kernel)k)==TI_INVALID);
        CHECK(ti_forward_batch(&m,x,3,6,h+1,6,NULL,3,(ti_kernel)k)==TI_INVALID);
        CHECK(ti_forward_batch(&m,x,3,6,x,6,y+1,3,(ti_kernel)k)==TI_ALIAS);
        CHECK(ti_forward_batch(&m,x,3,6,h+1,6,h+3,3,(ti_kernel)k)==TI_ALIAS);
        CHECK(ti_forward_batch(&m,x,3,6,h+1,6,x+2,3,(ti_kernel)k)==TI_ALIAS);
        CHECK(ti_forward_batch(&m,x,1,2,w1,2,y+1,1,(ti_kernel)k)==TI_ALIAS);
        CHECK(ti_forward_batch(&m,x,1,2,h+1,2,w2,1,(ti_kernel)k)==TI_ALIAS);
        CHECK(ti_forward_batch(&m,x,SIZE_MAX,6,h+1,6,y+1,3,(ti_kernel)k)==TI_OVERFLOW);
        x[5]=NAN;
        CHECK(ti_forward_batch(&m,x,3,6,h+1,6,y+1,3,(ti_kernel)k)==TI_NONFINITE);
        x[5]=1;
    }
    CHECK(ti_forward_batch(&m,x,3,6,h+1,6,y+1,3,(ti_kernel)99)==TI_INVALID);
    CHECK(ti_forward_batch(NULL,x,3,6,h+1,6,y+1,3,TI_INPUT_FIRST)==TI_INVALID);
    if (!ti_blas_available())
        CHECK(ti_forward_batch(&m,x,3,6,h+1,6,y+1,3,TI_SYSTEM_BLAS)==TI_UNSUPPORTED);
    w1[0]=-FLT_MAX; x[4]=FLT_MAX;
    CHECK(ti_model_init(&m,2,2,1,w1,4,b1,2,w2,2,b2,1)==TI_OK);
    CHECK(ti_forward_batch(&m,x,3,6,h+1,6,y+1,3,TI_INPUT_FIRST)==TI_NONFINITE);
    w1[0]=1; x[4]=FLT_MAX; w2[0]=FLT_MAX;
    CHECK(ti_model_init(&m,2,2,1,w1,4,b1,2,w2,2,b2,1)==TI_OK);
    CHECK(ti_forward_batch(&m,x,3,6,h+1,6,y+1,3,TI_OUTPUT_FIRST)==TI_NONFINITE);
    m.initialized=0;
    CHECK(ti_forward_batch(&m,x,3,6,h+1,6,y+1,3,TI_INPUT_FIRST)==TI_INVALID);
    puts("PASS: batch hand-derived rows, bias/ReLU, guards, sizes, alias, overflow and nonfinite checks");
    return 0;
}
