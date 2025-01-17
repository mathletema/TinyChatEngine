#include "operators.h"
#include "utils.h"

void load_BMM_F32T_coreml(BMM_F32T_coreml &op, std::string prefix) {
    read_to_array((prefix + "/alpha.bin").c_str(), &op.alpha, 1);
}

BMM_F32T_coreml::BMM_F32T_coreml(float _alpha) {
    this->alpha = _alpha;
    coreml::init_batched_matmul_transpose();
}

void BMM_F32T_coreml::forward(const Matrix3D<float> &a, const Matrix3D<float> &weight, Matrix3D<float> &c) {
    const Matrix3D<float> b = weight;
    const int m = a.m_dim_y, n = b.m_dim_y, k = a.m_dim_z, b_size = b.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START_FLOPS(profile_name, ops);

    // a: m x k   b: n x k   c: m x n
    assert(a.m_dim_x == b.m_dim_x);  // batch dim
    assert(a.m_dim_z == b.m_dim_z);  // k
    assert(a.m_dim_y == c.m_dim_y);  // m
    assert(b.m_dim_y == c.m_dim_z);  // n

    coreml::batched_matmul_transposed(a.m_data, b.m_data, c.m_data, b_size, m, n, k);

    for (int i = 0; i <= b_size * m * n; i++) {
        c.m_data[i] *= alpha;
    }

    PROFILE_END(profile_name);
}

void BMM_F32T_coreml::forward_weight_untransposed(const Matrix3D<float> &a, const Matrix3D<float> &weight,
                                           Matrix3D<float> &c) {
    const Matrix3D<float> b = weight;
    const int m = a.m_dim_y, n = c.m_dim_z, k = a.m_dim_z, b_size = b.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START_FLOPS(profile_name, ops);

    // a: m x k   b: n x k   c: m x n
    assert(a.m_dim_x == b.m_dim_x);  // batch dim
    assert(a.m_dim_z == b.m_dim_y);  // k
    assert(a.m_dim_y == c.m_dim_y);  // m
    assert(b.m_dim_z == c.m_dim_z);  // n

    struct matmul_params params;
    params.A.row = a.m_dim_y;
    params.A.column = a.m_dim_z;
    params.A.data_ptr = a.m_data;
    params.B.row = b.m_dim_y;
    params.B.column = b.m_dim_z;
    params.B.data_ptr = b.m_data;
    params.C.row = c.m_dim_y;
    params.C.column = c.m_dim_z;
    params.C.data_ptr = c.m_data;
    params.opt_params.blk_size = BLK_SIZE;
    params.opt_params.num_thread = NUM_THREAD;
    params.alpha = alpha;

    matmul::MatmulOperator op = matmul::MatmulOperator();

    for (int i = 0; i < m * n * a.m_dim_x; i++) {
        params.C.data_ptr[i] = 0;
    }

    for (int bz = 0; bz < a.m_dim_x; bz++) {
        float *data_A = params.A.data_ptr + bz * m * k, *data_B = params.B.data_ptr + bz * k * n,
              *data_C = params.C.data_ptr + bz * m * n;
        for (int i = 0; i < m; i++)
            for (int kk = 0; kk < k; kk++) {
                float Aikk0 = data_A[i * k + kk];
                for (int j = 0; j < n; j++) {
                    float Bjk0 = data_B[kk * n + j];
                    data_C[i * n + j] += Aikk0 * Bjk0;
                }
            }
    }

    PROFILE_END(profile_name);
}
