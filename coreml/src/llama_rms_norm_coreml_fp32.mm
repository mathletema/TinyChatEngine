#include "llama_rms_norm_coreml_fp32.h"
#include "coreml_utils.h"

namespace coreml {

bool rms_norm_running = false;
MLModel *rms_norm_model;

void init_llama_rms_norm() {
    if (rms_norm_running) return;
    rms_norm_running = true;

    NSError * error = nil;
    
    NSURL *module_path =
        [NSURL URLWithString:@(PROJECT_PATH "/coreml/modules/llama_rmsnorm_4096.mlpackage")];
    NSURL *compiled_path = [MLModel compileModelAtURL:module_path
                                                error:&error];
    handle_errors(error);

    MLModelConfiguration *config = [MLModelConfiguration new];
    config.computeUnits = MLComputeUnitsAll;

    rms_norm_model = [MLModel modelWithContentsOfURL:compiled_path
                              configuration:config
                                      error:&error];
    handle_errors(error);
};


void llama_rms_norm(float * x, float * y, float * weight, int bz, int n, int k, float eps) {
    // TODO: move all coreml inits together
    init_llama_rms_norm();

    NSError * error = nil;

    assert(bz == 1 && "llama_rms_norm: configuration not supported");
    assert(1 <= n && n <= 4096 && "llama_rms_norm: configuration not supported");
    assert(k == 4096 && "llama_rms_norm: configuration not supported");

    MLMultiArray * x__ = float_to_MLMultiArray_3D(x, bz, n, k, error);
    MLMultiArray * y__ = float_to_MLMultiArray_3D(y, bz, n, k, error);
    MLMultiArray * weight__ = float_to_MLMultiArray_3D(weight, 1, 1, k, error);
    handle_errors(error);

    MLDictionaryFeatureProvider *inFeatures = [
        [MLDictionaryFeatureProvider alloc]
        initWithDictionary: @{
                @"x" : x__,
                @"weight" : weight__,
            }
        error:&error
    ];
    handle_errors(error);

    MLPredictionOptions * opts = [MLPredictionOptions alloc];
    opts.outputBackings = @{
        @"output" : y__,
    };

    [rms_norm_model predictionFromFeatures:inFeatures options:opts error:&error];
    handle_errors(error);

    // for (int i = 0; i < bz; i++) {      // batches
    //     for (int j = 0; j < n; j++) {  // samples
    //         float var = 0;

    //         for (int k_ = 0; k_ < k; k_++) {  // hideden states
    //             var += x[n*k*i + k*j + k_] * x[n*k*i + k*j + k_];
    //         }
    //         var /= static_cast<float>(k);
    //         float variance = 1.0 / sqrt(var + eps);

    //         for (int k_ = 0; k_ < k; k_++) {
    //             float value = static_cast<float>(x[n*k*i + k*j + k_]);
    //             float fp_out = (value * variance);
    //             y[n*k*i + k*j + k_] = fp_out;
    //         }
    //     }
    // }

    // for (int i = 0; i < bz; i++) {
    //     for (int j = 0; j < n; j++) {
    //         for (int k_ = 0; k_ < k; k_++) {
    //             y[n*k*i + k*j + k_] *= weight[k_];
    //         }
    //     }
    // }

}

}
