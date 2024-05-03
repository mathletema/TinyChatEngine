#include "lib.h"
#include <cstdio>
#include "matmul.h"

matmul::MatmulOperator op;

void init() {
    printf("SIMD init!!\n");
    op = matmul::MatmulOperator();
}

void static_matmul(float* __restrict__ x, float* __restrict__ y, float* __restrict__ out, int m, int n, int k) {
    struct matmul_params params;

    params.A.row = m;
    params.A.column = k;
    params.A.data_ptr = x;

    params.B.row = k;
    params.B.column = n;
    params.B.data_ptr = y;

    params.C.row = m;
    params.C.column = n;
    params.C.data_ptr = out;

    params.opt_params.num_thread = 32;

    op.mat_mul_accelerator_transposed_fastover_column(&params);
}