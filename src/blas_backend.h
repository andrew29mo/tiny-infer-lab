#ifndef TI_BLAS_BACKEND_H
#define TI_BLAS_BACKEND_H
#include <stddef.h>
/* Internal: dimensions must fit INT_MAX; buffers already validated. */
void ti_blas_dense(const float *x, const float *w, const float *b, float *y,
                   size_t in, size_t out);
void ti_blas_batch_dense(const float *x, const float *w, const float *b, float *y,
                         size_t batch, size_t in, size_t out);
#endif
