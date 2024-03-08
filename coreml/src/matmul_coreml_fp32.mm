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
        [NSURL URLWithString:@(PROJECT_PATH "/coreml/modules/matmul_transpose_128.mlpackage")];
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

// TODO: use asynchronous ANE prediction to avoid blocking pipeline

void matmul_transposed(float * a, float * b, float * c, int m, int n, int k) {
    assert(k == 128 && "only k = 128 supported!");
    assert(m >= 1 && m <= 256 && "only 1 <= m <= 256 supported!");
    assert(n >= 1 && n <= 256 && "only 1 <= n <= 256 supported!");
    
    MLMultiArray * a__ = float_to_MLMultiArray(a, m, k, error);
    MLMultiArray * b__ = float_to_MLMultiArray(b, n, k, error);
    MLMultiArray * c__ = float_to_MLMultiArray(c, m, n, error);
    handle_errors(error);

    MLDictionaryFeatureProvider *inFeatures = [
        [MLDictionaryFeatureProvider alloc]
        initWithDictionary: @{
                @"x" : a__,
                @"y" : b__,
            }
        error:&error
    ];
    handle_errors(error);

    MLPredictionOptions * opts = [MLPredictionOptions alloc];
    opts.outputBackings = @{
        @"output" : c__,
    };

    [model predictionFromFeatures:inFeatures options:opts error:&error];
    handle_errors(error);
}

void batched_matmul_transposed(float * a, float * b, float * c, int bz, int m, int n, int k) {
    for (int bz_ = 0; bz_ < bz; bz_++) {
        coreml::matmul_transposed(a, b, c, m, n, k);
        a += m * k;
        b += k * n;
        c += m * n;
    }
}

}
