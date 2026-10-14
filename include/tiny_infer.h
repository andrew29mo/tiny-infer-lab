#ifndef TINY_INFER_H
#define TINY_INFER_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    TI_OK = 0, TI_INVALID, TI_OVERFLOW, TI_CAPACITY, TI_NONFINITE, TI_ALIAS,
    TI_UNSUPPORTED
} ti_status;
typedef enum { TI_OUTPUT_FIRST = 0, TI_INPUT_FIRST = 1, TI_SYSTEM_BLAS = 2 } ti_kernel;

/* Configure BLAS before inference, outside timing. Accelerate policy applies
 * to the calling thread; OpenBLAS thread configuration is process-wide. */
int ti_blas_available(void);
ti_status ti_blas_configure(void);
int ti_blas_is_single_threaded(void);
const char *ti_blas_name(void);

/* Row-major W1[input][hidden], W2[hidden][output]. Borrowed, immutable
 * weights must remain alive and unchanged after initialization. */
typedef struct {
    size_t input_dim, hidden_dim, output_dim;
    const float *w1, *b1, *w2, *b2;
    uint32_t initialized;
} ti_model;

ti_status ti_model_init(ti_model *model, size_t input_dim, size_t hidden_dim,
                       size_t output_dim, const float *w1, size_t w1_count,
                       const float *b1, size_t b1_count,
                       const float *w2, size_t w2_count,
                       const float *b2, size_t b2_count);

/* Single vector: y = ReLU(x W1 + b1) W2 + b2.
 * Model must be initialized by ti_model_init and not modified by the caller.
 * Counts must describe valid live float arrays; raw pointer validity cannot
 * be established by this C API. Buffers must not overlap the model object.
 * The handwritten kernels allocate no heap; vendor BLAS internals may allocate.
 * Workspace and output must be disjoint from one another,
 * input, and all weights. On failure workspace/output can be partially written.
 * Concurrent calls require separate workspace/output; weights can be shared. */
ti_status ti_forward(const ti_model *model, const float *input, size_t input_count,
                     float *workspace, size_t workspace_count,
                     float *output, size_t output_count, ti_kernel kernel);
/* Contiguous row-major batch: input[batch][input_dim],
 * workspace[batch][hidden_dim], output[batch][output_dim]. Batch must be > 0.
 * Counts are total float elements, not bytes or row counts. The single-vector
 * pointer, immutability, aliasing, failure and concurrency obligations above
 * also apply to the full batch ranges. No padding/strides are supported.
 * Handwritten backends run row loops inside C; system BLAS uses two SGEMMs.
 * The caller owns all workspace/output; vendor BLAS may allocate internally. */
ti_status ti_forward_batch(const ti_model *model, const float *input, size_t batch,
                           size_t input_count, float *workspace, size_t workspace_count,
                           float *output, size_t output_count, ti_kernel kernel);
const char *ti_status_string(ti_status status);
#endif
