#ifndef MATMUL_COREML_FP32_H
#define MATMUL_COREML_FP32_H

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

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
 * 
 * unbatched version assumes arrays are 2d instead of 3d
 */

void matmul_transposed(float * a, float * b, float * c, int m, int n, int k);
void batched_matmul_transposed(float * a, float * b, float * c, int bz, int m, int n, int k);

}

#endif