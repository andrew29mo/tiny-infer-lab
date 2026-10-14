/* Batch adapter only: arithmetic and softmax are upstream emlearn functions.
 * The generated model retains the original float32 bits, not six-digit rounding.
 * Static upstream scratch makes this single-thread adapter non-reentrant. */
#include "digits_emlearn.h"
#include <stddef.h>
#include <string.h>

int comparison_emlearn(const float *x, size_t rows, float *out) {
    if (!x || !out || rows < 1 || rows > 450) return -1;
    for (size_t i = 0; i < rows; ++i) {
        int status = eml_net_predict_proba(&digits, x + 64*i, 64, out + 10*i, 10);
        if (status != EmlOk) return status;
    }
    return 0;
}

/* Read actual compiled arrays back for bitwise export verification. */
void comparison_weights(float *out) {
    memcpy(out, digits_layer_0_weights, 2048*sizeof(float));
    memcpy(out+2048, digits_layer_0_biases, 32*sizeof(float));
    memcpy(out+2080, digits_layer_1_weights, 320*sizeof(float));
    memcpy(out+2400, digits_layer_1_biases, 10*sizeof(float));
}
