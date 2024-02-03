#ifndef COREML_ENGINE_H
#define COREML_ENGINE_H

#include <iostream>
#include <fstream>

void CoreML_init();

void CoreML_exit();

void CoreML_log(const char * message_format, ...);

#endif