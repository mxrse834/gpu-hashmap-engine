# CUDA Hashmap Engine Makefile

# Compiler and flags
NVCC := nvcc
CFLAGS := -G -g -rdc=true

# Directories
SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build

# Source files
SOURCES := main.cu src/hashmap.cu src/hash.cu
OBJECTS := $(BUILD_DIR)/main.o $(BUILD_DIR)/hashmap.o $(BUILD_DIR)/hash.o
EXECUTABLE := gpu-hashmap

# Default target
all: $(EXECUTABLE)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile main.cu
$(BUILD_DIR)/main.o: main.cu $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c main.cu -o $@

# Compile src/hashmap.cu
$(BUILD_DIR)/hashmap.o: src/hashmap.cu $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c src/hashmap.cu -o $@

# Compile src/hash.cu
$(BUILD_DIR)/hash.o: src/hash.cu $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c src/hash.cu -o $@

# Link all object files
$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(CFLAGS) -o $@ $^ 
	compute-sanitizer --tool memcheck ./$(EXECUTABLE)

# Test targets
test_simple: test_simple.cu
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -o test_simple test_simple.cu
	./test_simple

test_ops: test_hashmap_ops.cu
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -o test_ops test_hashmap_ops.cu
	./test_ops

test_hash: test_hash_func.cu src/hash.cu
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -o test_hash test_hash_func.cu src/hash.cu
	./test_hash

test_insert: test_insert.cu src/hash.cu src/hashmap.cu
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -o test_insert test_insert.cu src/hash.cu src/hashmap.cu
	./test_insert

test_lookup: test_lookup.cu src/hash.cu src/hashmap.cu
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -o test_lookup test_lookup.cu src/hash.cu src/hashmap.cu
	./test_lookup

test_integration: test_integration.cu src/hash.cu src/hashmap.cu
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -o test_integration test_integration.cu src/hash.cu src/hashmap.cu
	compute-sanitizer --tool memcheck ./test_integration

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE) test_simple test_ops test_hash test_insert test_lookup test_integration

# Rebuild from scratch
rebuild: clean all

.PHONY: all clean rebuild test_simple test_ops test_hash test_insert test_lookup test_integration
