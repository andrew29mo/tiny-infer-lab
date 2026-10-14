#include "tiny_infer.h"
#include "blas_backend.h"

#if defined(TI_ENABLE_BLAS) && defined(__APPLE__)
#include <Accelerate/Accelerate.h>
int ti_blas_available(void) {
    if (__builtin_available(macOS 15.0, *)) return 1;
    return 0;
}
ti_status ti_blas_configure(void) {
    if (__builtin_available(macOS 15.0, *)) {
        if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED) != 0) return TI_UNSUPPORTED;
        return BLASGetThreading() == BLAS_THREADING_SINGLE_THREADED ? TI_OK : TI_UNSUPPORTED;
    }
    return TI_UNSUPPORTED;
}
int ti_blas_is_single_threaded(void) {
    if (__builtin_available(macOS 15.0, *))
        return BLASGetThreading() == BLAS_THREADING_SINGLE_THREADED;
    return 0;
}
const char *ti_blas_name(void) { return "Apple Accelerate CBLAS"; }
#elif defined(TI_ENABLE_BLAS) && defined(TI_BLAS_OPENBLAS)
#include <cblas.h>
int ti_blas_available(void) { return 1; }
ti_status ti_blas_configure(void) {
    openblas_set_num_threads(1);
    return openblas_get_num_threads() == 1 ? TI_OK : TI_UNSUPPORTED;
}
int ti_blas_is_single_threaded(void) { return openblas_get_num_threads() == 1; }
const char *ti_blas_name(void) { return openblas_get_config(); }
#else
int ti_blas_available(void) { return 0; }
ti_status ti_blas_configure(void) { return TI_UNSUPPORTED; }
int ti_blas_is_single_threaded(void) { return 0; }
const char *ti_blas_name(void) { return "disabled"; }
#endif

void ti_blas_dense(const float *x, const float *w, const float *b, float *y,
                   size_t in, size_t out) {
#if defined(TI_ENABLE_BLAS)
    /* Same row-major weights as the C kernels: y = W^T x + bias.
     * beta=1 includes the bias copy in timing. No application-side packing. */
    for (size_t j = 0; j < out; ++j) y[j] = b[j];
    cblas_sgemv(CblasRowMajor, CblasTrans, (int)in, (int)out,
                1.0f, w, (int)out, x, 1, 1.0f, y, 1);
#else
    (void)x; (void)w; (void)b; (void)y; (void)in; (void)out;
#endif
}

void ti_blas_batch_dense(const float *x, const float *w, const float *b, float *y,
                         size_t batch, size_t in, size_t out) {
#if defined(TI_ENABLE_BLAS)
    for (size_t row = 0; row < batch; ++row)
        for (size_t col = 0; col < out; ++col) y[row * out + col] = b[col];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)batch, (int)out, (int)in, 1.0f, x, (int)in,
                w, (int)out, 1.0f, y, (int)out);
#else
    (void)x; (void)w; (void)b; (void)y; (void)batch; (void)in; (void)out;
#endif
}
