#include "matmul_coreml_fp32.h"

#include <iostream>

#include "coreml_utils.h"

namespace coreml {

bool running = false;
MLModel *model;
NSError *error;

void init_batched_matmul_transpose() {
    if (running) return;
    running = true;

    error = nil;
    
    NSURL *module_path =
        [NSURL URLWithString:@(PROJECT_PATH "/coreml/modules/matmul_transpose.mlpackage")];
    NSURL *compiled_path = [MLModel compileModelAtURL:module_path
                                                error:&error];
    handle_errors(error);

    MLModelConfiguration *config = [MLModelConfiguration new];
    config.computeUnits = MLComputeUnitsAll;

    model = [MLModel modelWithContentsOfURL:compiled_path
                              configuration:config
                                      error:&error];
    handle_errors(error);
};

void batched_matmul_transposed(float * a, float * b, float * c, int bz, int m, int n, int k) {
    std::cout << "Hey what's up!\n";
}

}
