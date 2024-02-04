#ifndef COREML_ENGINE_H
#define COREML_ENGINE_H

#include <CoreML/CoreML.h>
#include "common.h"

void CoreML_init();
void CoreML_exit();
void CoreML_log(const char * message_format, ...);
void CoreML_handle_errors (NSError *error);

MLMultiArray * CoreML_arr_to_MLMultiArray(float * data, int dim1, int dim2, int s1, int s2);
MLMultiArray * CoreML_arr_to_MLMultiArray(float * data, int dim1, int dim2, int dim3, int s1, int s2, int s3);
MLMultiArray *CoreML_Matrix3D_to_MLMultiArray(Matrix3D<float> input);

void CoremL_matmul_128_MLMultiArray (MLMultiArray * a, MLMultiArray * b, MLMultiArray * c);
void CoreML_matmul_128(float * a, float * b, float * c, int m, int n, int k);

#endif