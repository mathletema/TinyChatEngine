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

}