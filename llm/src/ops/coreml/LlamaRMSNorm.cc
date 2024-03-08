#include <cmath>
#include <iomanip>

#include "operators.h"
#include "utils.h"

void LlamaRMSNorm_coreml::forward(const Matrix3D<float> &x, Matrix3D<float> &output, float eps) {
    PROFILE_START(profile_name);

    const int last_dims = 2;

    assert(last_dims == 2);  // support the last dim for now
    assert(output.m_dim_x == x.m_dim_x);
    assert(output.m_dim_y == x.m_dim_y);
    assert(output.m_dim_z == x.m_dim_z);
    assert(x.m_dim_z == weight.m_dim_z);

    coreml::llama_rms_norm(x.m_data, output.m_data, weight.m_data, x.m_dim_x, x.m_dim_y, x.m_dim_z, eps);

    PROFILE_END(profile_name);
}
