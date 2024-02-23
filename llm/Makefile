##########################
# Ishank's test makefile #
##########################

# just use coreml dependencies. make sure that no function is
# left unoptimized.

# TODO: merge this with original makefile, adding suitable flags

# Compiler and flags
CXX = clang++
CXXFLAGS = -std=c++11 -pthread -Ofast -mavx2 -mfma -ffast-math -DUSE_INT8_INT4_PRODUCT -fpermissive -DQM_x86

# customize define
DEFINE =
CHATNAME = chat

# Executable and source files
CHAT_TARGET = chat
TARGET = $(CHAT_TARGET)

BUILDDIR := build/transformer
LIB_DIR = ../kernels
INCLUDE_DIRS = -I$(LIB_DIR) -I./include -I./include/nn_modules -I./json/single_include/ -I./half-2.2.0/include/
LIB =
LDFLAGS = -framework Foundation -framework CoreML

SRC = 
SRC += $(wildcard $(LIB_DIR)/*.cc)
SRC += $(wildcard $(LIB_DIR)/avx/*.cc)
SRC += $(wildcard src/nn_modules/non_cuda/*.cc)
SRC += $(wildcard src/*.cc)
SRC += $(wildcard src/nn_modules/*.cc)
SRC +=  $(wildcard src/ops/*.cc)

OBJS = $(addprefix $(BUILDDIR)/,$(SRC:.cc=.o))

# Default target
all: $(TARGET)

# Phony targets
.PHONY: all clean

# Pattern rules for non-CUDA
$(BUILDDIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Rule for CHAT_TARGET
$(CHAT_TARGET): %: application/%.cc $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -o $(CHATNAME) $^ $(LIB) $(LDFLAGS)

# Clean up
clean:
	rm -f $(TARGET)
	rm -rf *.dSYM
	rm -rf $(BUILDDIR)/../matmul_optimization
	rm -rf $(BUILDDIR)/$(LIB_DIR)
	rm -rf $(BUILDDIR)
	rm -rf $(PROFILEDIR)/$(LIB_DIR)
	rm -rf $(PROFILEDIR)
