#include "softmax_coreml_fp32.h"

#include <iostream>

#include "coreml_utils.h"

namespace coreml {

bool softmax_running = false;
MLModel *softmax_model;

void init_softmax() {
    if (softmax_running) return;
    softmax_running = true;

    NSError * error = nil;
    
    NSURL *module_path =
        [NSURL URLWithString:@(PROJECT_PATH "/coreml/modules/softmax_4096.mlpackage")];
    NSURL *compiled_path = [MLModel compileModelAtURL:module_path
                                                error:&error];
    handle_errors(error);

    MLModelConfiguration *config = [MLModelConfiguration new];
    config.computeUnits = MLComputeUnitsAll;

    softmax_model = [MLModel modelWithContentsOfURL:compiled_path
                              configuration:config
                                      error:&error];
    handle_errors(error);
};

void softmax(float * x, float * y, int bz, int n, int k) {
    // TODO: move this to some init inside LLaMA_Attention.cc
    init_softmax();

    NSError * error = nil;

    assert(bz == 32 && "softmax: configuration not supported");
    assert(1 <= n && n <= 4096 && "softmax: configuration not supported");
    assert(1 <= k && k <= 4096 && "softmax: configuration not supported");

    MLMultiArray * x__ = float_to_MLMultiArray_3D(x, bz, n, k, error);
    MLMultiArray * y__ = float_to_MLMultiArray_3D(y, bz, n, k, error);
    handle_errors(error);

    MLDictionaryFeatureProvider *inFeatures = [
        [MLDictionaryFeatureProvider alloc]
        initWithDictionary: @{
                @"x" : x__,
            }
        error:&error
    ];
    handle_errors(error);

    MLPredictionOptions * opts = [MLPredictionOptions alloc];
    opts.outputBackings = @{
        @"output" : y__,
    };

    [softmax_model predictionFromFeatures:inFeatures options:opts error:&error];
    handle_errors(error);
}

}