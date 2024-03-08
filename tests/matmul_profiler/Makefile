CC = clang++
CFLAGS = -Wall -g
CFLAGS += -ObjC++ -DQM_COREML 
LDFLAGS = -framework Foundation -framework CoreML

PROJECT_PATH = "$(shell realpath ..)"
CFLAGS += -DPROJECT_PATH=\"$(PROJECT_PATH)\"

SRCS = main.cpp 
OBJS = main.o
ifdef COREML
	SRCS += lib_coreml.mm
	OBJS += lib_coreml.o
else
	SRCS += lib_simd.cpp
	OBJS += lib_simd.o
endif

HDRS = lib.h

TARGET = matmul_profile

# Build target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS)

# Rule for compiling .cpp files
main.o: main.cpp $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

lib_simd.o: lib_simd.cpp $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

lib_coreml.o: lib_coreml.mm $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f *.o
	rm -f $(TARGET)
