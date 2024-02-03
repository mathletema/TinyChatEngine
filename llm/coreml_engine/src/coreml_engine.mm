#include "coreml_engine.h"
#include "CoreML/CoreML.h"

#include <stdio.h>
#include <stdarg.h>

bool __COREML_RUNNING = false;

FILE *__COREML_LOG_FP;


void CoreML_init() {
    if (__COREML_RUNNING) {
        CoreML_log("CoreML Engine is already running!\n");
        return;
    }
    __COREML_RUNNING = true;


    const char * __COREML_LOG_PATH = PROJECT_PATH "/llm/logs/coreml.out";
    __COREML_LOG_FP = fopen(__COREML_LOG_PATH, "w");
    if (__COREML_LOG_FP == NULL) {
        assert(0 && "Core ML log Could not be opened!\n");
        return;
    }

    CoreML_log("CoreML log opened at %s\n", __COREML_LOG_PATH);

    const char * __COREML_MATMUL_128_PATH = PROJECT_PATH "/coreml/modules/matmul_transpose_128.mlpackage";
    CoreML_log("CoreML module path: %s\n", __COREML_LOG_PATH);

    CoreML_log("CoreML Engine succesfully initiated!\n");
};

bool CoreML_is_running() {
    return __COREML_RUNNING;
}

void CoreML_exit() {
    if (!__COREML_RUNNING) {
        CoreML_log("CoreML Engine is not currently running!\n");
        return;
    }
    __COREML_RUNNING = false;
    CoreML_log("CoremL Engine successfully killed!\n");
}

void CoreML_log (const char * message, ...) {
    va_list args;
    va_start (args, message);

    vfprintf(__COREML_LOG_FP, message, args);
    fflush(__COREML_LOG_FP);

    va_end(args);
}
