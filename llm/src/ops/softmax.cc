#include <cmath>

#include "operators.h"

void softmax(const Matrix3D<float> &input, Matrix3D<float> &output, const int dim) {
    PROFILE_START("softmax");
    int len = input.length();

    assert(dim == 2 && "Unsupported dimension for softmax");

    // Find the maximum value in the input array
    for (int i = 0; i < input.m_dim_x; i++) {
        for (int j = 0; j < input.m_dim_y; j++) {
            float max_value = input.m_data[0];
            float sum = 0;
            // Find the maximum value in the input array
            for (int k = 0; k < input.m_dim_z; k++) {
                float value = input(i, j, k);
                if (value > max_value) {
                    max_value = value;
                }
            }

            // Compute the softmax values
            for (int k = 0; k < input.m_dim_z; k++) {
                float value = input(i, j, k);
                sum += std::exp(value - max_value);
            }

            // Normalize the softmax values and store them in the output array
            for (int k = 0; k < input.m_dim_z; k++) {
                float value = input(i, j, k);
                float final_v = (std::exp(value - max_value) / (sum + 1e-10));
                output(i, j, k) = final_v;
            }
        }
    }
    PROFILE_END("softmax");
}
