// include kernels
#include "coreml_utils.h"
#include "matmul_coreml_fp32.h"
#include "rope_coreml_fp32.h"
#include "llama_rms_norm_coreml_fp32.h"
#include "softmax_coreml_fp32.h"
#include "batch_add_coreml_fp32.h"

// include ops
#include "ops/coreml/BMM_F32T.h"
#include "ops/coreml/RotaryPosEmb.h"
#include "ops/coreml/LlamaRMSNorm.h"
void softmax_coreml(const Matrix3D<float> &input, Matrix3D<float> &output, int dim);
void batch_Add_coreml(const Matrix3D<float> &input, const Matrix3D<float> &input2, Matrix3D<float> &output);