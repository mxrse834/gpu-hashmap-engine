NVCC ?= nvcc
PYTHON ?= python3
CUDA_ARCH ?= sm_75
BUILD_TYPE ?= release

TARGET := gpu-hashmap
BUILD_DIR := build
HASH_TEST := $(BUILD_DIR)/hash-test
HASHMAP_TEST := $(BUILD_DIR)/hashmap-test
INCLUDE_DIR := include
SOURCES := main.cu src/hashmap.cu src/hash.cu
OBJECTS := $(BUILD_DIR)/main.o $(BUILD_DIR)/hashmap.o $(BUILD_DIR)/hash.o

# GCC's -Wpedantic diagnoses #line directives emitted by nvcc and used inside
# CUDA's Cooperative Groups headers, drowning project diagnostics in warnings.
# Keep the useful host warnings without applying that incompatible policy flag.
COMMON_FLAGS := -std=c++17 -arch=$(CUDA_ARCH) -rdc=true -I$(INCLUDE_DIR) \
	-Xcompiler=-Wall,-Wextra

ifeq ($(BUILD_TYPE),debug)
OPT_FLAGS := -O0 -g -G
else
OPT_FLAGS := -O3 -lineinfo
endif

NVCCFLAGS := $(COMMON_FLAGS) $(OPT_FLAGS)

.PHONY: all debug clean fixtures test sanitize profile

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/main.o: main.cu $(INCLUDE_DIR)/hashmap.cuh $(INCLUDE_DIR)/cuda_check.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/hashmap.o: src/hashmap.cu $(INCLUDE_DIR)/hashmap.cuh $(INCLUDE_DIR)/hash.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/hash.o: src/hash.cu $(INCLUDE_DIR)/hash.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD_DIR)/hash_test.o: test/hash_test.cu $(INCLUDE_DIR)/hash.cuh $(INCLUDE_DIR)/cuda_check.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(HASH_TEST): $(BUILD_DIR)/hash_test.o $(BUILD_DIR)/hash.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD_DIR)/hashmap_test.o: test/hashmap_test.cu $(INCLUDE_DIR)/hashmap.cuh $(INCLUDE_DIR)/cuda_check.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(HASHMAP_TEST): $(BUILD_DIR)/hashmap_test.o $(BUILD_DIR)/hashmap.o $(BUILD_DIR)/hash.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

debug:
	$(MAKE) clean
	$(MAKE) BUILD_TYPE=debug all $(HASH_TEST) $(HASHMAP_TEST)

fixtures:
	$(PYTHON) test/test_generator.py

test: all $(HASH_TEST) $(HASHMAP_TEST) fixtures
	./$(HASH_TEST)
	./$(HASHMAP_TEST)
	./$(TARGET) test/file_a.bin test/file_b.bin
	@set +e; ./$(TARGET) test/file_a.bin test/file_c.bin; status=$$?; \
		test $$status -eq 1
	@set +e; ./$(TARGET) test/file_a.bin test/file_reordered.bin; status=$$?; \
		test $$status -eq 1
	@set +e; ./$(TARGET) test/file_a.bin test/file_d.bin; status=$$?; \
		test $$status -eq 1
	./$(TARGET) test/empty_a.bin test/empty_b.bin

sanitize: debug fixtures
	compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
		./$(HASH_TEST)
	compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
		./$(HASHMAP_TEST)
	compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
		./$(TARGET) test/file_a.bin test/file_b.bin

profile: all fixtures
	ncu --set full ./$(TARGET) test/file_a.bin test/file_b.bin

clean:
	rm -rf $(BUILD_DIR) $(TARGET) *.ncu-rep
