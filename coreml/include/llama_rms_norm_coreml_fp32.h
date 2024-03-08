#ifndef LLAMA_RMS_NORM_COREML_FP32_H
#define LLAMA_RMS_NORM_COREML_FP32_H

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

namespace coreml {

/*
 * instantiates coreml model
 * does nothing if model already instantiated
 * raises error if model unable to start
 */

void init_llama_rms_norm();

 
/*
 * TODO: FILL OUT
 */

void llama_rms_norm(float * x, float * y, float * weight, int m, int n, int k, float eps);

}

#endif