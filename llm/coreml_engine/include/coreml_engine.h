#ifndef COREML_ENGINE_H
#define COREML_ENGINE_H

#include <CoreML/CoreML.h>
#include "common.h"

void CoreML_init();
void CoreML_exit();
void CoreML_log(const char * message_format, ...);
void CoreML_handle_errors (NSError *error);
MLMultiArray * Matrix3D_to_MLMultiArray(Matrix3D<float> input);

#endif