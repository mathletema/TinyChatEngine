#ifndef BATCH_ADD_COREML_FP32_H
#define BATCH_ADD_COREML_FP32_H

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

namespace coreml {

/*
 * instantiates coreml model
 * does nothing if model already instantiated
 * raises error if model unable to start
 */

void init_batch_add();

 
/*
 * Computes z = x + y
 *
 * assumes dimensions
 *   x.shape = bz x n x k
 *   y.shape = 1 x n x k
 */

void batch_add(float * x, float * y, float * z, int bz, int n, int k);

}

#endif