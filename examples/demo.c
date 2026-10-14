#include "tiny_infer.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    float input[] = {2, -1};
    if (argc != 1 && argc != 3) {
        fprintf(stderr, "Usage: %s [input0 input1]\n", argv[0]); return 1;
    }
    if (argc == 3) {
        for (size_t i = 0; i < 2; ++i) {
            char *end;
            errno = 0;
            input[i] = strtof(argv[i + 1], &end);
            if (errno || !*argv[i + 1] || *end || !isfinite(input[i])) {
                fprintf(stderr, "Input must be a finite float\n"); return 1;
            }
        }
    }
    const float w1[] = {1, 0, -1, 0, 1, 1}, b1[] = {-1, 2, 0};
    const float w2[] = {3, 1, -2, 2, 1, -1}, b2[] = {0.5f, -0.5f};
    float hidden[3], output[2];
    ti_model model;
    ti_status status = ti_model_init(&model, 2, 3, 2, w1, 6, b1, 3, w2, 6, b2, 2);
    if (status != TI_OK) { fprintf(stderr, "%s\n", ti_status_string(status)); return 1; }
    for (int k = 0; k < 2; ++k) {
        status = ti_forward(&model, input, 2, hidden, 3, output, 2, (ti_kernel)k);
        if (status != TI_OK) { fprintf(stderr, "%s\n", ti_status_string(status)); return 1; }
        printf("%s: input=[%g, %g], hidden=[%g, %g, %g], output=[%g, %g]\n",
               k == 0 ? "output_first" : "input_first", input[0], input[1],
               hidden[0], hidden[1], hidden[2], output[0], output[1]);
    }
    return 0;
}
