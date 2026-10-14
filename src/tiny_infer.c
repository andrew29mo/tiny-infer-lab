#include "tiny_infer.h"
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include "blas_backend.h"

#define TI_MAGIC UINT32_C(0x54494e59)

static int safe_elements(size_t a, size_t b, size_t *count) {
    if (a == 0 || b == 0 || a > SIZE_MAX / b) return 0;
    *count = a * b;
    return *count <= SIZE_MAX / sizeof(float);
}
static int finite_array(const float *a, size_t n) {
    for (size_t i = 0; i < n; ++i) if (!isfinite(a[i])) return 0;
    return 1;
}
static int overlap(const float *a, size_t na, const float *b, size_t nb) {
    uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    size_t ba = na * sizeof(float), bb = nb * sizeof(float);
    /* Invalid wrapping address ranges are rejected conservatively. */
    if (pa > UINTPTR_MAX - ba || pb > UINTPTR_MAX - bb) return 1;
    return pa < pb + bb && pb < pa + ba;
}

ti_status ti_model_init(ti_model *m, size_t in, size_t hid, size_t out,
                       const float *w1, size_t n1, const float *b1, size_t nb1,
                       const float *w2, size_t n2, const float *b2, size_t nb2) {
    size_t need1, need2;
    if (!m) return TI_INVALID;
    *m = (ti_model){0};
    if (!in || !hid || !out || !w1 || !b1 || !w2 || !b2) return TI_INVALID;
    if (!safe_elements(in, hid, &need1) || !safe_elements(hid, out, &need2))
        return TI_OVERFLOW;
    if (n1 < need1 || n2 < need2 || nb1 < hid || nb2 < out) return TI_CAPACITY;
    if (!finite_array(w1, need1) || !finite_array(b1, hid) ||
        !finite_array(w2, need2) || !finite_array(b2, out)) return TI_NONFINITE;
    *m = (ti_model){in, hid, out, w1, b1, w2, b2, TI_MAGIC};
    return TI_OK;
}

static void dense(const float *x, const float *w, const float *b, float *y,
                  size_t in, size_t out, ti_kernel kernel) {
    if (kernel == TI_SYSTEM_BLAS) {
        ti_blas_dense(x, w, b, y, in, out);
    } else if (kernel == TI_OUTPUT_FIRST) {
        for (size_t j = 0; j < out; ++j) {
            float sum = b[j];
            for (size_t i = 0; i < in; ++i) sum += x[i] * w[i * out + j];
            y[j] = sum;
        }
    } else {
        for (size_t j = 0; j < out; ++j) y[j] = b[j];
        for (size_t i = 0; i < in; ++i)
            for (size_t j = 0; j < out; ++j) y[j] += x[i] * w[i * out + j];
    }
}

ti_status ti_forward(const ti_model *m, const float *x, size_t nx,
                     float *h, size_t nh, float *y, size_t ny, ti_kernel kernel) {
    size_t n1, n2;
    if (!m || m->initialized != TI_MAGIC || !x || !h || !y ||
        (kernel != TI_OUTPUT_FIRST && kernel != TI_INPUT_FIRST && kernel != TI_SYSTEM_BLAS)) return TI_INVALID;
    if (kernel == TI_SYSTEM_BLAS) {
        if (!ti_blas_available()) return TI_UNSUPPORTED;
        if (m->input_dim > INT_MAX || m->hidden_dim > INT_MAX || m->output_dim > INT_MAX)
            return TI_OVERFLOW;
    }
    if (!safe_elements(m->input_dim, m->hidden_dim, &n1) ||
        !safe_elements(m->hidden_dim, m->output_dim, &n2)) return TI_OVERFLOW;
    size_t in = m->input_dim, hid = m->hidden_dim, out = m->output_dim;
    if (nx < in || nh < hid || ny < out) return TI_CAPACITY;
    const float *reads[] = {x, m->w1, m->b1, m->w2, m->b2};
    size_t lengths[] = {in, n1, hid, n2, out};
    if (overlap(h, hid, y, out)) return TI_ALIAS;
    for (size_t i = 0; i < 5; ++i)
        if (overlap(h, hid, reads[i], lengths[i]) ||
            overlap(y, out, reads[i], lengths[i])) return TI_ALIAS;
    if (!finite_array(x, in)) return TI_NONFINITE;
    dense(x, m->w1, m->b1, h, in, hid, kernel);
    /* Check before ReLU, so negative infinity cannot be silently hidden. */
    if (!finite_array(h, hid)) return TI_NONFINITE;
    for (size_t i = 0; i < hid; ++i) if (h[i] < 0) h[i] = 0;
    dense(h, m->w2, m->b2, y, hid, out, kernel);
    return finite_array(y, out) ? TI_OK : TI_NONFINITE;
}

const char *ti_status_string(ti_status s) {
    switch (s) {
        case TI_OK: return "ok";
        case TI_INVALID: return "invalid argument or uninitialized model";
        case TI_OVERFLOW: return "dimension overflow";
        case TI_CAPACITY: return "insufficient buffer capacity";
        case TI_NONFINITE: return "non-finite input, weight or result";
        case TI_ALIAS: return "overlapping buffers";
        case TI_UNSUPPORTED: return "BLAS unavailable or thread configuration unsupported";
    }
    return "unknown status";
}
