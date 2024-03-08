#include "coreml_utils.h"

#include <iostream>
#include <stdexcept>

namespace coreml {

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

MLMultiArray * float_to_MLMultiArray_3D(float * data, int m, int n, int k, NSError * error) {
    MLMultiArray * result = [[MLMultiArray alloc] initWithDataPointer:((void *) data)
                                                                shape:@[ @(m), @(n), @(k) ]
                                                            dataType:MLMultiArrayDataTypeFloat32
                                                             strides:@[ @(n*k), @(k), @(1) ]
                                                         deallocator:nil
                                                               error:&error];
    return result;
}

}