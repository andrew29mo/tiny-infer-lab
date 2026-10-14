#include "tiny_infer.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); exit(1); \
} } while (0)

static void nonsquare_reference(void) {
    enum { I = 31, H = 17, O = 9 };
    float x[I], w1[I * H], b1[H], w2[H * O], b2[O], h[H + 2], y[O + 2];
    double gold_h[H], gold[O];
    for (size_t i = 0; i < I; ++i) x[i] = ((int)(i % 7) - 3) * 0.125f;
    for (size_t i = 0; i < I * H; ++i) w1[i] = ((int)(i % 13) - 6) * 0.0625f;
    for (size_t i = 0; i < H * O; ++i) w2[i] = ((int)(i % 11) - 5) * 0.0625f;
    for (size_t i = 0; i < H; ++i) b1[i] = ((int)(i % 3) - 1) * 0.25f;
    for (size_t i = 0; i < O; ++i) b2[i] = ((int)(i % 5) - 2) * 0.125f;
    for (size_t j = 0; j < H; ++j) {
        double sum = b1[j];
        for (size_t i = 0; i < I; ++i) sum += (double)x[i] * w1[i * H + j];
        gold_h[j] = fmax(0, sum);
    }
    for (size_t j = 0; j < O; ++j) {
        gold[j] = b2[j];
        for (size_t i = 0; i < H; ++i) gold[j] += gold_h[i] * w2[i * O + j];
    }
    ti_model m;
    CHECK(ti_model_init(&m, I, H, O, w1, I * H, b1, H, w2, H * O, b2, O) == TI_OK);
    for (int k = 0; k < (ti_blas_available() ? 3 : 2); ++k) {
        h[0] = 51; h[H + 1] = 52; y[0] = 53; y[O + 1] = 54;
        CHECK(ti_forward(&m, x, I, h + 1, H, y + 1, O, (ti_kernel)k) == TI_OK);
        for (size_t j = 0; j < O; ++j) CHECK(fabs(y[j + 1] - gold[j]) < 1e-5);
        CHECK(h[0] == 51 && h[H + 1] == 52 && y[0] == 53 && y[O + 1] == 54);
    }
}

int main(void) {
    if (ti_blas_available()) {
        CHECK(ti_blas_configure() == TI_OK);
        CHECK(ti_blas_is_single_threaded());
    }
    nonsquare_reference();
    /* Hand-derived: x=[2,-1], h=ReLU([2,-1]+[-1,2])=[1,1],
     * y=1*3+1*(-2)+0.5=1.5. Guards catch writes outside active ranges. */
    float w1[] = {1, 0, 0, 1}, b1[] = {-1, 2}, w2[] = {3, -2}, b2[] = {0.5f};
    float x[] = {2, -1}, h[] = {91, 0, 0, 92}, y[] = {93, 0, 94};
    ti_model m;
    CHECK(ti_model_init(&m, 2, 2, 1, w1, 4, b1, 2, w2, 2, b2, 1) == TI_OK);
    if (!ti_blas_available()) {
        CHECK(ti_blas_configure() == TI_UNSUPPORTED);
        CHECK(ti_forward(&m, x, 2, h + 1, 2, y + 1, 1, TI_SYSTEM_BLAS) == TI_UNSUPPORTED);
    }
    for (int kernel = 0; kernel < (ti_blas_available() ? 3 : 2); ++kernel) {
        CHECK(ti_forward(&m, x, 2, h + 1, 2, y + 1, 1, (ti_kernel)kernel) == TI_OK);
        CHECK(y[1] == 1.5f && h[1] == 1 && h[2] == 1);
        CHECK(h[0] == 91 && h[3] == 92 && y[0] == 93 && y[2] == 94);
    }
    x[0] = -2; x[1] = -3;
    CHECK(ti_forward(&m, x, 2, h + 1, 2, y + 1, 1, TI_INPUT_FIRST) == TI_OK);
    CHECK(h[1] == 0 && h[2] == 0 && y[1] == 0.5f);
    x[0] = 2; x[1] = -1;
    CHECK(ti_forward(&m, x, 1, h + 1, 2, y + 1, 1, TI_INPUT_FIRST) == TI_CAPACITY);
    CHECK(ti_forward(&m, x, 2, h + 1, 1, y + 1, 1, TI_INPUT_FIRST) == TI_CAPACITY);
    CHECK(ti_forward(&m, x, 2, h + 1, 2, y + 1, 0, TI_INPUT_FIRST) == TI_CAPACITY);
    CHECK(ti_forward(&m, x, 2, h + 1, 2, y + 1, 1, (ti_kernel)99) == TI_INVALID);
    CHECK(ti_forward(&m, NULL, 2, h + 1, 2, y + 1, 1, TI_INPUT_FIRST) == TI_INVALID);
    CHECK(ti_forward(&m, x, 2, x, 2, y + 1, 1, TI_INPUT_FIRST) == TI_ALIAS);
    CHECK(ti_forward(&m, x, 2, h + 1, 2, h + 2, 1, TI_INPUT_FIRST) == TI_ALIAS);
    CHECK(ti_forward(&m, x, 2, h + 1, 2, w2, 1, TI_INPUT_FIRST) == TI_ALIAS);
    x[0] = NAN;
    CHECK(ti_forward(&m, x, 2, h + 1, 2, y + 1, 1, TI_INPUT_FIRST) == TI_NONFINITE);
    w1[0] = 1; x[0] = 2; w2[0] = FLT_MAX;
    CHECK(ti_model_init(&m, 2, 2, 1, w1, 4, b1, 2, w2, 2, b2, 1) == TI_OK);
    x[0] = 3;
    CHECK(ti_forward(&m, x, 2, h + 1, 2, y + 1, 1, TI_OUTPUT_FIRST) == TI_NONFINITE);
    x[0] = INFINITY;
    CHECK(ti_forward(&m, x, 2, h + 1, 2, y + 1, 1, TI_INPUT_FIRST) == TI_NONFINITE);
    CHECK(ti_model_init(&m, 0, 2, 1, w1, 4, b1, 2, w2, 2, b2, 1) == TI_INVALID);
    CHECK(ti_forward(&m, x, 2, h + 1, 2, y + 1, 1, TI_INPUT_FIRST) == TI_INVALID);
    CHECK(ti_model_init(&m, SIZE_MAX, 2, 1, w1, 4, b1, 2, w2, 2, b2, 1) == TI_OVERFLOW);
    CHECK(ti_model_init(&m, SIZE_MAX / sizeof(float) + 1, 1, 1,
                        w1, 4, b1, 2, w2, 2, b2, 1) == TI_OVERFLOW);
    CHECK(ti_model_init(&m, SIZE_MAX / sizeof(float), 1, 2, w1, 4, b1, 2, w2, 2, b2, 1) == TI_CAPACITY);
    CHECK(ti_model_init(&m, 2, 2, 1, w1, 3, b1, 2, w2, 2, b2, 1) == TI_CAPACITY);
    w1[0] = NAN;
    CHECK(ti_model_init(&m, 2, 2, 1, w1, 4, b1, 2, w2, 2, b2, 1) == TI_NONFINITE);
    w1[0] = FLT_MAX; x[0] = FLT_MAX;
    CHECK(ti_model_init(&m, 2, 2, 1, w1, 4, b1, 2, w2, 2, b2, 1) == TI_OK);
    CHECK(ti_forward(&m, x, 2, h + 1, 2, y + 1, 1, TI_INPUT_FIRST) == TI_NONFINITE);
    puts("PASS: hand-derived output, ReLU, guards, capacity, alias, invalid input, overflow and non-finite checks");
    return 0;
}
