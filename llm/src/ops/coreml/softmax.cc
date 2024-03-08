#include <cmath>

#include "operators.h"
#include "softmax_coreml_fp32.h"


void softmax_coreml(const Matrix3D<float> &input, Matrix3D<float> &output, const int dim) {
    PROFILE_START("softmax_coreml");

    coreml::softmax(input.m_data, output.m_data, input.m_dim_x, input.m_dim_y, input.m_dim_z);

    PROFILE_END("softmax_coreml");
}
