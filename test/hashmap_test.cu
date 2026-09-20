#include "cuda_check.cuh"
#include "hashmap.cuh"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_KEY_COUNT 4u
#define TEST_OVERFLOW_CAPACITY 8u

typedef struct
{
    HashmapEngine host;
    HashmapEngine *device;
} TestMap;

static void free_test_map(TestMap *map)
{
    if (map->host.master_bytes != NULL)
        cudaFree(map->host.master_bytes);
    if (map->host.key_offsets != NULL)
        cudaFree(map->host.key_offsets);
    if (map->host.key_lengths != NULL)
        cudaFree(map->host.key_lengths);
    if (map->host.values != NULL)
        cudaFree(map->host.values);
    if (map->host.overflow_offsets != NULL)
        cudaFree(map->host.overflow_offsets);
    if (map->host.overflow_lengths != NULL)
        cudaFree(map->host.overflow_lengths);
    if (map->host.overflow_values != NULL)
        cudaFree(map->host.overflow_values);
    if (map->host.failed_inserts != NULL)
        cudaFree(map->host.failed_inserts);
    if (map->device != NULL)
        cudaFree(map->device);
    memset(map, 0, sizeof(*map));
}

static void create_test_map(TestMap *map, uint32_t byte_capacity)
{
    HashmapEngine *engine;

    memset(map, 0, sizeof(*map));
    engine = &map->host;
    engine->primary_capacity = 1u;
    engine->overflow_capacity = TEST_OVERFLOW_CAPACITY;
    engine->byte_capacity = byte_capacity;
    engine->master_byte_current = 0u;
    engine->batch_base = HASHMAP_EMPTY_SLOT;

    CUDA_CHECK(cudaMalloc((void **)&engine->master_bytes, byte_capacity));
    CUDA_CHECK(cudaMalloc((void **)&engine->key_offsets, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&engine->key_lengths, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&engine->values, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&engine->overflow_offsets,
                          TEST_OVERFLOW_CAPACITY * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&engine->overflow_lengths,
                          TEST_OVERFLOW_CAPACITY * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&engine->overflow_values,
                          TEST_OVERFLOW_CAPACITY * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&engine->failed_inserts, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemset(engine->key_offsets, 0xFF, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(engine->overflow_offsets, 0xFF,
                          TEST_OVERFLOW_CAPACITY * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(engine->failed_inserts, 0, sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void **)&map->device, sizeof(HashmapEngine)));
    CUDA_CHECK(cudaMemcpy(map->device, engine, sizeof(HashmapEngine),
                          cudaMemcpyHostToDevice));
}

static void insert_one(TestMap *map, const char *key, uint32_t value)
{
    uint8_t *device_bytes = NULL;
    uint32_t *device_offset = NULL;
    uint32_t *device_length = NULL;
    uint32_t *device_value = NULL;
    uint32_t offset = 0u;
    uint32_t length = (uint32_t)strlen(key);

    CUDA_CHECK(cudaMalloc((void **)&device_bytes, length));
    CUDA_CHECK(cudaMalloc((void **)&device_offset, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&device_length, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&device_value, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(device_bytes, key, length, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_offset, &offset, sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_length, &length, sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_value, &value, sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    reserve_batch_kernel<<<1, 1>>>(map->device, length, 1u);
    check_kernel("reserve_batch_kernel");
    insert_kernel<<<1, 4>>>(map->device, device_bytes, device_offset,
                            device_length, device_value, 1u, length);
    check_kernel("insert_kernel");

    CUDA_CHECK(cudaFree(device_value));
    CUDA_CHECK(cudaFree(device_length));
    CUDA_CHECK(cudaFree(device_offset));
    CUDA_CHECK(cudaFree(device_bytes));
}

static uint32_t failed_insert_count(const TestMap *map)
{
    uint32_t failures = 0u;
    CUDA_CHECK(cudaMemcpy(&failures, map->host.failed_inserts, sizeof(failures),
                          cudaMemcpyDeviceToHost));
    return failures;
}

static void read_overflow_offsets(const TestMap *map,
                                  uint32_t offsets[TEST_OVERFLOW_CAPACITY])
{
    CUDA_CHECK(cudaMemcpy(offsets, map->host.overflow_offsets,
                          TEST_OVERFLOW_CAPACITY * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
}

static int verify_lookup(const uint8_t found[TEST_KEY_COUNT],
                         const uint32_t actual[TEST_KEY_COUNT],
                         const uint32_t expected[TEST_KEY_COUNT],
                         int missing_index)
{
    uint32_t i;
    int passed = 1;

    for (i = 0u; i < TEST_KEY_COUNT; ++i)
    {
        int should_exist = (int)i != missing_index;
        if ((found[i] != 0u) != should_exist ||
            (should_exist && actual[i] != expected[i]))
        {
            fprintf(stderr, "lookup mismatch for key %u: found=%u, value=%u\n",
                    i, (unsigned int)found[i], actual[i]);
            passed = 0;
        }
    }
    return passed;
}

int main(void)
{
    /*
     * With overflow capacity 8 and the three current XXH32 seeds, keys 1-3
     * all start at overflow slot 1. Sequential insertion therefore creates a
     * deterministic linear-probe cluster in slots 1, 2, and 3.
     */
    const char *keys[TEST_KEY_COUNT] = {
        "primary-filler",
        "overflow-key-7",
        "overflow-key-16",
        "overflow-key-40",
    };
    const uint32_t expected_values[TEST_KEY_COUNT] = {
        17u,
        UINT32_MAX,
        0u,
        0x12345678u,
    };
    uint32_t offsets[TEST_KEY_COUNT];
    uint32_t lengths[TEST_KEY_COUNT];
    uint32_t expected_overflow_offsets[3];
    uint32_t overflow_offsets[TEST_OVERFLOW_CAPACITY];
    uint32_t results[TEST_KEY_COUNT];
    uint8_t found[TEST_KEY_COUNT];
    uint8_t deleted = 0u;
    uint8_t *query_bytes = NULL;
    uint8_t *device_query_bytes = NULL;
    uint32_t *device_offsets = NULL;
    uint32_t *device_lengths = NULL;
    uint32_t *device_results = NULL;
    uint8_t *device_found = NULL;
    uint8_t *device_deleted = NULL;
    uint32_t total_bytes = 0u;
    uint32_t i;
    int passed = 1;
    TestMap map;

    for (i = 0u; i < TEST_KEY_COUNT; ++i)
    {
        offsets[i] = total_bytes;
        lengths[i] = (uint32_t)strlen(keys[i]);
        total_bytes += lengths[i];
    }

    expected_overflow_offsets[0] = offsets[1];
    expected_overflow_offsets[1] = offsets[2];
    expected_overflow_offsets[2] = offsets[3];

    create_test_map(&map, total_bytes);
    for (i = 0u; i < TEST_KEY_COUNT; ++i)
        insert_one(&map, keys[i], expected_values[i]);

    if (failed_insert_count(&map) != 0u)
    {
        fprintf(stderr, "forced-overflow insertion reported a failure\n");
        passed = 0;
    }

    read_overflow_offsets(&map, overflow_offsets);
    if (overflow_offsets[0] != HASHMAP_EMPTY_SLOT)
        passed = 0;
    for (i = 0u; i < 3u; ++i)
    {
        if (overflow_offsets[i + 1u] != expected_overflow_offsets[i])
            passed = 0;
    }
    if (!passed)
        fprintf(stderr, "test keys did not form the expected overflow cluster\n");

    query_bytes = (uint8_t *)malloc(total_bytes);
    if (query_bytes == NULL)
    {
        fprintf(stderr, "host allocation failed in hashmap test\n");
        free_test_map(&map);
        return 1;
    }
    for (i = 0u; i < TEST_KEY_COUNT; ++i)
        memcpy(query_bytes + offsets[i], keys[i], lengths[i]);

    CUDA_CHECK(cudaMalloc((void **)&device_query_bytes, total_bytes));
    CUDA_CHECK(cudaMalloc((void **)&device_offsets, sizeof(offsets)));
    CUDA_CHECK(cudaMalloc((void **)&device_lengths, sizeof(lengths)));
    CUDA_CHECK(cudaMalloc((void **)&device_results, sizeof(results)));
    CUDA_CHECK(cudaMalloc((void **)&device_found, sizeof(found)));
    CUDA_CHECK(cudaMalloc((void **)&device_deleted, sizeof(deleted)));
    CUDA_CHECK(cudaMemcpy(device_query_bytes, query_bytes, total_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_offsets, offsets, sizeof(offsets),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_lengths, lengths, sizeof(lengths),
                          cudaMemcpyHostToDevice));

    lookup_kernel<<<1, 128>>>(map.device, device_query_bytes, device_offsets,
                              device_lengths, TEST_KEY_COUNT, total_bytes,
                              device_results, device_found);
    check_kernel("lookup_kernel");
    CUDA_CHECK(cudaMemcpy(results, device_results, sizeof(results),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(found, device_found, sizeof(found),
                          cudaMemcpyDeviceToHost));
    if (!verify_lookup(found, results, expected_values, -1))
        passed = 0;

    delete_kernel<<<1, 128>>>(map.device, device_query_bytes,
                              device_offsets + 1u, device_lengths + 1u, 1u,
                              total_bytes, device_deleted);
    check_kernel("delete_kernel");
    CUDA_CHECK(cudaMemcpy(&deleted, device_deleted, sizeof(deleted),
                          cudaMemcpyDeviceToHost));
    read_overflow_offsets(&map, overflow_offsets);
    if (deleted != 1u || overflow_offsets[1] != HASHMAP_TOMBSTONE_SLOT)
    {
        fprintf(stderr, "failed to tombstone the first overflow slot\n");
        passed = 0;
    }

    lookup_kernel<<<1, 128>>>(map.device, device_query_bytes, device_offsets,
                              device_lengths, TEST_KEY_COUNT, total_bytes,
                              device_results, device_found);
    check_kernel("lookup_kernel after delete");
    CUDA_CHECK(cudaMemcpy(results, device_results, sizeof(results),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(found, device_found, sizeof(found),
                          cudaMemcpyDeviceToHost));
    if (!verify_lookup(found, results, expected_values, 1))
        passed = 0;

    CUDA_CHECK(cudaFree(device_deleted));
    CUDA_CHECK(cudaFree(device_found));
    CUDA_CHECK(cudaFree(device_results));
    CUDA_CHECK(cudaFree(device_lengths));
    CUDA_CHECK(cudaFree(device_offsets));
    CUDA_CHECK(cudaFree(device_query_bytes));
    free(query_bytes);
    free_test_map(&map);

    printf("forced-overflow lookup/delete: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
