#include "coreml_engine.h"
#include "CoreML/CoreML.h"

#include <stdexcept>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

bool __COREML_RUNNING = false;
FILE *__COREML_LOG_FP;
MLModel *model;

MLMultiArray *create_array(int dim) {
    NSError *error = nil;

    MLMultiArray *multiarray = [[MLMultiArray alloc] initWithShape:@[ @(dim), @128 ]
                                                          dataType:MLMultiArrayDataTypeFloat
                                                             error:&error];

    CoreML_handle_errors(error);

    return multiarray;
}

void test_conversion() {
    auto x = create_array(40);
    CoreML_log("40x128 stride: [%d, %d]\n", x.strides[0].intValue, x.strides[1].intValue);

    int a = 5;
    int b = 4;
    int c = 3;
    float *arr = (float *)malloc(a * b * c * sizeof(float));
    CoreML_log("arr located at %p\n", (void *)arr);

    Matrix3D<float> matrix = Matrix3D<float>(arr, a, b, c);

    // populate randomly
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            for (int k = 0; k < c; k++) {
                matrix(i, j, k) = (float)(((double)rand()) / ((double)RAND_MAX));
            }
        }
    }

    // print current matrix
    CoreML_log("Inital matrix: \n");
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            CoreML_log(">  ");
            for (int k = 0; k < c; k++) {
                CoreML_log("%.2f ", matrix(i, j, k));
            }
            CoreML_log("\n");
        }
        CoreML_log("> \n");
    }

    // convert
    MLMultiArray *matrix_objc = Matrix3D_to_MLMultiArray(matrix);

    // print converted matrix
    CoreML_log("Converted matrix: \n");
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            CoreML_log("> ");
            for (int k = 0; k < c; k++) {
                CoreML_log("%.2f ", [[matrix_objc objectForKeyedSubscript:@[ @(i), @(j), @(k) ]] floatValue]);
            }
            CoreML_log("\n");
        }
        CoreML_log("> \n");
    }

    // free!
    free(arr);
}

void test_inference() {
    NSError *error = nil;

    // initialize
    CoreML_log("creating arrays!\n");
    MLMultiArray *arr1 = [[MLMultiArray alloc]
        initWithShape:@[ @(1), @(128) ]
        dataType:MLMultiArrayDataTypeFloat
        error:&error];
    MLMultiArray *arr2 = [[MLMultiArray alloc]
        initWithShape:@[ @(1), @(128) ]
        dataType:MLMultiArrayDataTypeFloat
        error:&error];
    
    // populate
    CoreML_log("populating arrays!\n");
    for (int i = 0; i < 128; i++) {
        [arr1 setObject:@(i) forKeyedSubscript:@[ @(0), @(i) ]];
    }
    for (int i = 0; i < 128; i++) {
        [arr2 setObject:@(i % 3) forKeyedSubscript:@[ @(0), @(i) ]];
    }

    // print
    CoreML_log("arr1: ");
    for (int i = 0; i < 128; i++) {
        CoreML_log("%.2f ", [[arr1 objectForKeyedSubscript:@[ @(0), @(i) ]] floatValue]);
    }
    CoreML_log("\n");

    CoreML_log("arr2: ");
    for (int i = 0; i < 128; i++) {
        CoreML_log("%.2f ", [[arr2 objectForKeyedSubscript:@[ @(0), @(i) ]] floatValue]);
    }
    CoreML_log("\n");

    NSDictionary<NSString *, id> *featureDictionary = @{
        @"A" : arr1,
        @"B" : arr2,
    };

    MLDictionaryFeatureProvider *inFeatures = [[MLDictionaryFeatureProvider alloc] initWithDictionary:featureDictionary
                                                                                                error:&error];
    CoreML_handle_errors(error);

    CoreML_log("running model!\n");
    id<MLFeatureProvider> outFeatures =
        [model predictionFromFeatures:inFeatures error:&error];
    CoreML_handle_errors(error);
    CoreML_log("done running model!\n");

    MLMultiArray * output_arr = [[outFeatures featureValueForName:@"output"] multiArrayValue];
    CoreML_log("output arr num dims: [%d] ", output_arr.shape.count);
    for (int i = 0; i < output_arr.shape.count; i++) {
        CoreML_log("%d ", output_arr.shape[i].intValue);
    }
    CoreML_log("\n");

    CoreML_log("output[0, 0] = %.2f\n", [[output_arr objectForKeyedSubscript:@[@0, @0]] floatValue]);
    CoreML_log("should be 8086!\n");

}

void run_tests() {
    CoreML_log("Starting conversion test!\n");
    test_conversion();
    CoreML_log("Conversion test done!\n");

    CoreML_log("Starting inference test!\n");
    test_inference();
    CoreML_log("Inference test done!\n");
}

void CoreML_load_model() {
    CoreML_log("loading model!\n");

    NSError *error = nil;

    NSURL *matmul_transpose_128_module_path =
        [NSURL URLWithString:@(PROJECT_PATH "/coreml/modules/matmul_transpose_128.mlpackage")];
    CoreML_log("CoreML module path: %s\n", [[matmul_transpose_128_module_path path] UTF8String]);

    NSURL *matmul_transpose_128_compiled_path = [MLModel compileModelAtURL:matmul_transpose_128_module_path
                                                                     error:&error];
    CoreML_handle_errors(error);

    MLModelConfiguration *coreml_config = [MLModelConfiguration new];
    coreml_config.computeUnits = MLComputeUnitsAll;

    model = [MLModel modelWithContentsOfURL:matmul_transpose_128_compiled_path
                              configuration:coreml_config
                                      error:&error];
    CoreML_handle_errors(error);

    CoreML_log("model loaded!\n");
}

void CoreML_init() {
    if (__COREML_RUNNING) {
        CoreML_log("CoreML Engine is already running!\n");
        return;
    }
    __COREML_RUNNING = true;

    // initialize rand
    srand(0);

    const char *log_path = PROJECT_PATH "/llm/logs/coreml.out";
    __COREML_LOG_FP = fopen(log_path, "w");
    if (__COREML_LOG_FP == NULL) {
        throw std::runtime_error("Core ML log Could not be opened!\n");
        return;
    }
    CoreML_log("CoreML log opened at %s\n", log_path);

    CoreML_load_model();

    run_tests();

    CoreML_log("CoreML Engine succesfully initiated!\n");
};

bool CoreML_is_running() { return __COREML_RUNNING; }

void CoreML_exit() {
    if (!__COREML_RUNNING) {
        CoreML_log("CoreML Engine is not currently running!\n");
        return;
    }
    __COREML_RUNNING = false;
    CoreML_log("CoremL Engine successfully killed!\n");
}

void CoreML_log(const char *message, ...) {
    va_list args;
    va_start(args, message);

    vfprintf(__COREML_LOG_FP, message, args);
    fflush(__COREML_LOG_FP);

    va_end(args);
}

void CoreML_handle_errors(NSError *error) {
    if (error != nil) {
        const char *error_str = [[NSString stringWithFormat:@"%@", [error userInfo]] UTF8String];
        CoreML_log(error_str);
        throw std::runtime_error(error_str);
    }
}

MLMultiArray *Matrix3D_to_MLMultiArray(Matrix3D<float> input) {
    CoreML_log("Attempting to create an MLMultiArray!\n");
    NSError *error = nil;

    NSArray *strides = @[ @(input.m_dim_y * input.m_dim_z), @(input.m_dim_z), @1 ];
    CoreML_log("Input strides %d, %d, %d\n", [strides[0] intValue], [strides[1] intValue], [strides[2] intValue]);

    MLMultiArray *multiarray =
        [[MLMultiArray alloc] initWithDataPointer:((void *)input.m_data)
                                            shape:@[ @(input.m_dim_x), @(input.m_dim_y), @(input.m_dim_z) ]
                                         dataType:MLMultiArrayDataTypeFloat32
                                          strides:@[ @(input.m_dim_y * input.m_dim_z), @(input.m_dim_z), @(1) ]
                                      deallocator:nil
                                            error:&error];

    CoreML_handle_errors(error);

    CoreML_log("Output strides %d, %d, %d\n", multiarray.strides[0].intValue, multiarray.strides[1].intValue,
               multiarray.strides[2].intValue);

    return multiarray;
}