#pragma once

#include <cuda_runtime.h>

#include <stdint.h>

#define HASHMAP_THREADS_PER_BLOCK 256u
#define HASHMAP_EMPTY_SLOT UINT32_C(0xFFFFFFFF)
#define HASHMAP_TOMBSTONE_SLOT UINT32_C(0xFFFFFFFE)

// Device-resident state. The host owns an identical descriptor whose pointers
// refer to device allocations, then copies the descriptor itself to the GPU.
typedef struct HashmapEngine
{
    uint32_t primary_capacity;
    uint32_t overflow_capacity;
    uint32_t byte_capacity;
    uint32_t master_byte_current;
    uint32_t batch_base;

    uint8_t *master_bytes;

    uint32_t *key_offsets;
    uint32_t *key_lengths;
    uint32_t *values;

    uint32_t *overflow_offsets;
    uint32_t *overflow_lengths;
    uint32_t *overflow_values;

    uint32_t *failed_inserts;
} HashmapEngine;

// Batch reservation is deliberately separate from insertion. Kernel launches
// in one CUDA stream are ordered, so every insertion block observes the same
// batch_base without relying on an invalid grid-wide barrier.
__global__ void reserve_batch_kernel(HashmapEngine *engine,
                                     uint32_t total_bytes,
                                     uint32_t key_count);

__global__ void insert_kernel(HashmapEngine *engine,
                              const uint8_t *bytes,
                              const uint32_t *offsets,
                              const uint32_t *lengths,
                              const uint32_t *input_values,
                              uint32_t key_count,
                              uint32_t total_bytes);

__global__ void lookup_kernel(const HashmapEngine *engine,
                              const uint8_t *query_bytes,
                              const uint32_t *query_offsets,
                              const uint32_t *query_lengths,
                              uint32_t query_count,
                              uint32_t total_query_bytes,
                              uint32_t *results,
                              uint8_t *found);

__global__ void delete_kernel(HashmapEngine *engine,
                              const uint8_t *query_bytes,
                              const uint32_t *query_offsets,
                              const uint32_t *query_lengths,
                              uint32_t query_count,
                              uint32_t total_query_bytes,
                              uint8_t *deleted);
