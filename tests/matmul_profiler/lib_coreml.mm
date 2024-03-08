#include "lib.h"

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include <cstdio>
#include <iostream>


MLModel *lm_head_model;
MLModel *qkv_out_proj_model;
MLModel *down_proj_model;
MLModel *gate_proj_model;
NSError *error;

void handle_errors(NSError * error) {
    if (error != nil) {
        const char *error_str = [[NSString stringWithFormat:@"%@", [error userInfo]] UTF8String];
        std::cout << error_str << std::endl;
        throw std::runtime_error(error_str);
    }
}

MLMultiArray * float_to_MLMultiArray(float * data, int m, int n, NSError * error) {
    MLMultiArray * result = [[MLMultiArray alloc] initWithDataPointer:((void *) data)
                                                                shape:@[ @(m), @(n) ]
                                                            dataType:MLMultiArrayDataTypeFloat32
                                                             strides:@[ @(n), @(1) ]
                                                         deallocator:nil
                                                               error:&error];
    return result;
}

void init() {
    NSURL * module_path;
    NSURL * compiled_path;
    MLModelConfiguration * config;

    config = [MLModelConfiguration new];
    config.computeUnits = MLComputeUnitsAll;

    // init lm_head
    module_path =
        [NSURL URLWithString:@(PROJECT_PATH "/../coreml/modules/lm_head.mlpackage")];
    compiled_path = [MLModel compileModelAtURL:module_path
                                                error:&error];
    handle_errors(error);
    lm_head_model = [MLModel modelWithContentsOfURL:compiled_path
                              configuration:config
                                      error:&error];
    // init qkv_out_proj
    module_path =
        [NSURL URLWithString:@(PROJECT_PATH "/../coreml/modules/QKV_out_proj.mlpackage")];
    compiled_path = [MLModel compileModelAtURL:module_path
                                                error:&error];
    handle_errors(error);
    qkv_out_proj_model = [MLModel modelWithContentsOfURL:compiled_path
                              configuration:config
                                      error:&error];
    // init down_proj
    module_path =
        [NSURL URLWithString:@(PROJECT_PATH "/../coreml/modules/down_proj.mlpackage")];
    compiled_path = [MLModel compileModelAtURL:module_path
                                                error:&error];
    handle_errors(error);
    down_proj_model = [MLModel modelWithContentsOfURL:compiled_path
                              configuration:config
                                      error:&error];
    // init gate_proj
    module_path =
        [NSURL URLWithString:@(PROJECT_PATH "/../coreml/modules/gate_proj.mlpackage")];
    compiled_path = [MLModel compileModelAtURL:module_path
                                                error:&error];
    handle_errors(error);
    gate_proj_model = [MLModel modelWithContentsOfURL:compiled_path
                              configuration:config
                                      error:&error];
}

void qkv_out_proj(float* x, float* y, float* out) {printf("I'm qkv_proj\n"); }
void down_proj(float* x, float* y, float* out){ printf("I'm down proj\n"); }
void gate_proj(float * x, float * y, float* out) { printf("I'm gate_proj\n"); }

void static_matmul(float* x, float* y, float* out, int m, int n, int k) {
    MLMultiArray * x__ = float_to_MLMultiArray(x, m, k, error);
    MLMultiArray * y__ = float_to_MLMultiArray(y, n, k, error);
    MLMultiArray * out__ = float_to_MLMultiArray(out, m, n, error);
    handle_errors(error);

    MLDictionaryFeatureProvider *inFeatures = [
        [MLDictionaryFeatureProvider alloc]
        initWithDictionary: @{
                @"x" : x__,
                @"y" : y__,
            }
        error:&error
    ];
    handle_errors(error);

    MLPredictionOptions * opts = [MLPredictionOptions alloc];
    opts.outputBackings = @{
        @"matmul" : out__,
    };

    if (m == 1 && n == 32000 && k == 4096) {
        [lm_head_model predictionFromFeatures:inFeatures options:opts error:&error];
        handle_errors(error);
    }
    else if (m == 1 && n == 4096 && k == 4096) {
        [qkv_out_proj_model predictionFromFeatures:inFeatures options:opts error:&error];
        handle_errors(error);
    }
    else if (m == 1 && n == 4096 && k == 11008) {
        [down_proj_model predictionFromFeatures:inFeatures options:opts error:&error];
        handle_errors(error);
    }
    else if (m == 1 && n == 11008 && k == 4096) {
        [gate_proj_model predictionFromFeatures:inFeatures options:opts error:&error];
        handle_errors(error);
    }
    else {
        printf("unsupported dimensions!\n");
    }
}