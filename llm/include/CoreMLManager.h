#include <CoreML/CoreML.h>

struct CoreML_Manager {

    struct MatMulT_128_Manager {
        MLModel *model;
        NSError *error;
        NSURL *specUrl;
        NSURL *compiledUrl;
        (nullable id<MLBatchProvider>) predict(
            (id<MLBatchProvider>) inputBatch,
            (NSError **) error;
        )
    };

};

(nullable id<MLBatchProvider>) CoreML_Manager.MatMulT_128_Manager::predict