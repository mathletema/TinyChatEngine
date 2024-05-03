CC = clang++
CFLAGS = -std=c++11 -pthread -Ofast -mavx2 -mfma -ffast-math -DUSE_INT8_INT4_PRODUCT -fpermissive -DQM_x86
CFLAGS += -ObjC++
LDFLAGS = -framework Foundation -framework CoreML

PROJECT_PATH = "$(shell realpath ..)"
CFLAGS += -DPROJECT_PATH=\"$(PROJECT_PATH)\"
INCLUDE_DIRS = -I../../kernels/ -I../../llm/half-2.2.0/include/

# CFLAGS += -g -S -emit-llvm

SRCS = main.cpp 
OBJS = main.o
ifdef COREML
	SRCS += lib_coreml.mm
	OBJS += lib_coreml.o
else
	SRCS += lib_simd.cpp
	OBJS += lib_simd.o matmul_avx_fp32.o
endif

HDRS = lib.h

TARGET = matmul_profile

# Build target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDE_DIRS) -o $@ $(OBJS)

# Rule for compiling .cpp files
main.o: main.cpp $(HDRS)
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@

lib_simd.o: lib_simd.cpp $(HDRS)
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@

lib_coreml.o: lib_coreml.mm $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

matmul_avx_fp32.o: ../../kernels/avx/matmul_avx_fp32.cc
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Clean rule
clean:
	rm -f *.o
	rm -f $(TARGET)
