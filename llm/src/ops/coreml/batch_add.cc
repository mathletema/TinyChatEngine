#include "operators.h"

void batch_Add_coreml(const Matrix3D<float> &input, const Matrix3D<float> &input2, Matrix3D<float> &output) {
    PROFILE_START("batch_Add");
    assert(input.m_dim_y == input2.m_dim_y);
    assert(input.m_dim_z == input2.m_dim_z);
    assert(input.m_dim_x == output.m_dim_x);
    assert(input.m_dim_y == output.m_dim_y);
    assert(input.m_dim_z == output.m_dim_z);

    if (input.m_dim_x != input2.m_dim_x && input2.m_dim_x == 1) {
        // Find the maximum value in the input array
        coreml::batch_add(input.m_data, input2.m_data, output.m_data, input.m_dim_x, input.m_dim_y, input.m_dim_z);
    } else {
        throw("Unsupported dimension for softmax");
    }
    PROFILE_END("batch_Add");
}
