#ifndef MATMUL_COREML_FP32_H
#define MATMUL_COREML_FP32_H

namespace coreml {

/*
 * instantiates coreml model
 * does nothing if model already instantiated
 * raises error if model unable to start
 */

void init_batched_matmul_transpose();

 
/*
 * Computes c = a @ b.T in a batched fashion
 *
 * assumes dimensions
 *   a.shape = bz x m x k
 *   b.shape = bz x n x k
 *   c.shape = bz x m x n
 */

void batched_matmul_transposed(float * a, float * b, float * c, int bz, int m, int n, int k);

}

#endif