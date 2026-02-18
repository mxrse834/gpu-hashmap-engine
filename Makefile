# CUDA Hashmap Engine Makefile

# Compiler and flags
NVCC := nvcc
CFLAGS := -Xptxas -v -g -G -O0 -arch=sm_75 -rdc=true

# Directories
SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build

# Source files
SOURCES := main.cu $(SRC_DIR)/hashmap.cu $(SRC_DIR)/hash.cu
OBJECTS := $(BUILD_DIR)/main.o $(BUILD_DIR)/hashmap.o $(BUILD_DIR)/hash.o
EXECUTABLE := gpu-hashmap

# Default target
all: $(EXECUTABLE)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile main.cu
$(BUILD_DIR)/main.o: main.cu $(INCLUDE_DIR)/hash.cuh $(INCLUDE_DIR)/hashmap.cuh $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c main.cu -o $@

# Compile src/hashmap.cu
$(BUILD_DIR)/hashmap.o: $(SRC_DIR)/hashmap.cu $(INCLUDE_DIR)/hash.cuh $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c $(SRC_DIR)/hashmap.cu -o $@

# Compile src/hash.cu
$(BUILD_DIR)/hash.o: $(SRC_DIR)/hash.cu $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c $(SRC_DIR)/hash.cu -o $@

# Link all object files
$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(CFLAGS) -o $@ $^ 

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE) 

###must run
.PHONY: all clean 
