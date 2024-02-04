//  ██████╗ ██████╗ ██████╗ ███████╗███╗   ███╗██╗         ███████╗███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗
// ██╔════╝██╔═══██╗██╔══██╗██╔════╝████╗ ████║██║         ██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝
// ██║     ██║   ██║██████╔╝█████╗  ██╔████╔██║██║         █████╗  ██╔██╗ ██║██║  ███╗██║██╔██╗ ██║█████╗  
// ██║     ██║   ██║██╔══██╗██╔══╝  ██║╚██╔╝██║██║         ██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║██╔══╝  
// ╚██████╗╚██████╔╝██║  ██║███████╗██║ ╚═╝ ██║███████╗    ███████╗██║ ╚████║╚██████╔╝██║██║ ╚████║███████╗
//  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝    ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝╚══════╝
                                                                                                        

#include "coreml_engine.h"
#include "CoreML/CoreML.h"

#include <stdexcept>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

bool __COREML_RUNNING = false;
FILE *__COREML_LOG_FP;
MLModel *model;

/***************
 * START TESTS *
 ***************/

void test_conversion() {
    NSError *error = nil;

    int a = 5;
    int b = 4;
    int c = 3;

    float *arr = (float *) malloc(a * b * c * sizeof(float));
    Matrix3D<float> matrix = Matrix3D<float>(arr, a, b, c);

    for (int i = 0; i < a; i++)
        for (int j = 0; j < b; j++) 
            for (int k = 0; k < c; k++)
                matrix(i, j, k) = (float)(((double)rand()) / ((double)RAND_MAX));

    CoreML_log("Inital matrix: \n");
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            CoreML_log("   ");
            for (int k = 0; k < c; k++) {
                CoreML_log("%.2f ", matrix(i, j, k));
            }
            CoreML_log("\n");
        }
        CoreML_log("  \n");
    }

    MLMultiArray *matrix_objc = CoreML_Matrix3D_to_MLMultiArray(matrix);

    CoreML_log("Converted matrix: \n");
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            CoreML_log("  ");
            for (int k = 0; k < c; k++) {
                CoreML_log("%.2f ", [[matrix_objc objectForKeyedSubscript:@[ @(i), @(j), @(k) ]] floatValue]);
            }
            CoreML_log("\n");
        }
        CoreML_log("  \n");
    }

    MLMultiArray *matrix_objc_2 = CoreML_arr_to_MLMultiArray(arr, a, b, c, b*c, c, 1);

    CoreML_log("Converted matrix: \n");
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            CoreML_log("  ");
            for (int k = 0; k < c; k++) {
                CoreML_log("%.2f ", [[matrix_objc_2 objectForKeyedSubscript:@[ @(i), @(j), @(k) ]] floatValue]);
            }
            CoreML_log("\n");
        }
        CoreML_log("  \n");
    }

    free(arr);
}

void test_inference() {
    NSError *error = nil;

    CoreML_log("creating arrays!\n");
    MLMultiArray *arr1 = [[MLMultiArray alloc]
        initWithShape:@[ @(1), @(128) ]
        dataType:MLMultiArrayDataTypeFloat
        error:&error];
    MLMultiArray *arr2 = [[MLMultiArray alloc]
        initWithShape:@[ @(1), @(128) ]
        dataType:MLMultiArrayDataTypeFloat
        error:&error];
    MLMultiArray *arr_out = [[MLMultiArray alloc]
        initWithShape:@[ @(1), @(1) ]
        dataType:MLMultiArrayDataTypeFloat
        error:&error];
    
    // populate
    CoreML_log("populating arrays!\n");
    for (int i = 0; i < 128; i++) {
        [arr1 setObject:@(i) forKeyedSubscript:@[ @(0), @(i) ]];
    }
    for (int i = 0; i < 128; i++) {
        [arr2 setObject:@(1) forKeyedSubscript:@[ @(0), @(i) ]];
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

    CoremL_matmul_128_MLMultiArray(arr1, arr2, arr_out);

    CoreML_log("output[0, 0] = %.2f\n", [[arr_out objectForKeyedSubscript:@[@0, @0]] floatValue]);
    CoreML_log("should be 8128!\n");
}

void test_matmul_func() {
    float * a = (float*) malloc(  1 * 128 * sizeof(float));
    float * b = (float*) malloc(  1 * 128 * sizeof(float));
    float * c = (float*) malloc(  1 *   1 * sizeof(float));
    
    for (int i = 0; i < 128; i++) {  a[i] = i; b[i] = 1;  }
    c[0] = 0;

    CoreML_matmul_128(a, b, c, 1, 1, 128);

    CoreML_log("Output %f, should be 8128\n", c[0]);

    free(a);
    free(b);
    free(c);
}

void run_tests() {
    CoreML_log("Starting conversion test!\n");
    test_conversion();
    CoreML_log("Conversion test done!\n");

    CoreML_log("Starting inference test!\n");
    test_inference();
    CoreML_log("Inference test done!\n");

    CoreML_log("Starting matmul func test!\n");
    test_matmul_func();
    CoreML_log("Matmul func test done!\n");
}

/*************
 * END TESTS *
 *************/


/*******************
 * MAIN CODE START *
 *******************/

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

/**************************
 * ARRAY CONVERSION UTILS *
 **************************/

MLMultiArray * CoreML_arr_to_MLMultiArray(float * data, int dim1, int dim2, int s1, int s2) {
    NSError * error = nil;
    MLMultiArray * result = [[MLMultiArray alloc] initWithDataPointer:((void *) data)
                                                shape:@[ @(dim1), @(dim2) ]
                                             dataType:MLMultiArrayDataTypeFloat32
                                              strides:@[ @(s1), @(s2) ]
                                          deallocator:nil
                                                error:&error];
    CoreML_handle_errors(error);
    return result;
}

MLMultiArray * CoreML_arr_to_MLMultiArray(float * data, int dim1, int dim2, int dim3, int s1, int s2, int s3) {
    NSError * error = nil;
    MLMultiArray * result = [[MLMultiArray alloc] initWithDataPointer:((void *) data)
                                                shape:@[ @(dim1), @(dim2), @(dim3)]
                                             dataType:MLMultiArrayDataTypeFloat32
                                              strides:@[ @(s1), @(s2), @(s3) ]
                                          deallocator:nil
                                                error:&error];
    CoreML_handle_errors(error);
    return result;
}

MLMultiArray *CoreML_Matrix3D_to_MLMultiArray(Matrix3D<float> input) {
    return CoreML_arr_to_MLMultiArray(
        input.m_data,
        input.m_dim_x, input.m_dim_y, input.m_dim_z,
        input.m_dim_y * input.m_dim_z, input.m_dim_z, 1
    );
}


/**********************
 * MATMUL 128 WRAPPER *
 **********************/

void CoremL_matmul_128_MLMultiArray (MLMultiArray * a, MLMultiArray * b, MLMultiArray * c) {
    NSError * error = nil;
    // create input object
    MLDictionaryFeatureProvider *inFeatures = [
        [MLDictionaryFeatureProvider alloc]
        initWithDictionary: @{
                @"A" : a,
                @"B" : b,
            }
        error:&error
    ];
    CoreML_handle_errors(error);

    // create output object
    MLPredictionOptions * opts = [MLPredictionOptions alloc];
    opts.outputBackings = @{
        @"output" : c,
    };

    // run model
    CoreML_log("running model!\n");
    [model predictionFromFeatures:inFeatures options:opts error:&error];
    CoreML_handle_errors(error);
    CoreML_log("done running model!\n");
}

void CoreML_matmul_128(float * a, float * b, float * c, int m, int n, int k) {
    CoreML_log("entry into matmul func!\n");

    assert(k == 128 && "only k = 128 supported!");
    assert(m >= 1 && m <= 256 && "only 1 <= m <= 256 supported!");
    assert(n >= 1 && n <= 256 && "only 1 <= n <= 256 supported!");

    NSError * error = nil;
    
    MLMultiArray * a__ = CoreML_arr_to_MLMultiArray(a, m, k, k, 1);
    MLMultiArray * b__ = CoreML_arr_to_MLMultiArray(b, n, k, k, 1);
    MLMultiArray * c__ = CoreML_arr_to_MLMultiArray(c, m, n, n, 1);

    CoremL_matmul_128_MLMultiArray(a__, b__, c__);
}