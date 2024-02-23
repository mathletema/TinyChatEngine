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

/*
 * wraps a static array of floats as a MLMultiArray to use as input for a
 * CoreML model.
 *
 * input
 *   data: m x n array of floats stored in row major order
 */

MLMultiArray * float_to_MLMultiArray(float * data, int m, int n, NSError * error);

}
#endif