#ifndef COREML_UTILS_H
#define COREML_UTILS_H

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

namespace coreml {

/*
 * error handling utility
 * if error != nil, prints error and throws runtime_error
*/

void handle_errors(NSError * error);

}
#endif