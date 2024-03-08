#include "batch_add_coreml_fp32.h"

#include <iostream>

#include "coreml_utils.h"

namespace coreml {

void batch_add (float * x, float * y, float * z, int bz, int n, int k) {
    for (int i = 0; i < bz; i++) {
        for (int j = 0; j < n; j++) {
            for (int k_ = 0; k_ < k; k_++) {
                z[n*k*i + k*j + k_] = x[n*k*i + k*j + k_] + y[k*j + k_];
            }
        }
    }
}

}