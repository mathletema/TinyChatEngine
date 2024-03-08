#ifndef SOFTMAX_COREML_FP32_H
#define SOFTMAX_COREML_FP32_H

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

namespace coreml {

/*
 * instantiates coreml model
 * does nothing if model already instantiated
 * raises error if model unable to start
 */

void init_softmax();

 
/*
 * Computes y = softmax(x, axis=-1)
 *
 * assumes dimensions
 *   x.shape = bz x n x k
 *   y.shape = bz x n x k
 */

void softmax(float * x, float * y, int bz, int n, int k);

}

#endif