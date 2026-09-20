#include "cuda_check.cuh"
#include "hash.cuh"

#include <cooperative_groups.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_KEY_COUNT 7u
#define TEST_HASH_COUNT (TEST_KEY_COUNT * 3u)

#define XXH_PRIME1 UINT32_C(0x9E3779B1)
#define XXH_PRIME2 UINT32_C(0x85EBCA77)
#define XXH_PRIME3 UINT32_C(0xC2B2AE3D)
#define XXH_PRIME4 UINT32_C(0x27D4EB2F)
#define XXH_PRIME5 UINT32_C(0x165667B1)

namespace cg = cooperative_groups;

typedef struct
{
    const uint8_t *data;
    uint32_t length;
} TestKey;

static const uint32_t test_seeds[3] = {
    0x9E3779B1u,
    0x517CC1B7u,
    0x85EBCA6Bu,
};

static uint32_t rotate_left_cpu(uint32_t value, uint32_t amount)
{
    return (value << amount) | (value >> (32u - amount));
}

static uint32_t load_u32_le_cpu(const uint8_t *data)
{
    return (uint32_t)data[0] |
           ((uint32_t)data[1] << 8u) |
           ((uint32_t)data[2] << 16u) |
           ((uint32_t)data[3] << 24u);
}

static uint32_t xxh32_cpu(const uint8_t *data, uint32_t length, uint32_t seed)
{
    uint32_t cursor = 0u;
    uint32_t hash;

    if (length >= 16u)
    {
        uint32_t accumulator1 = seed + XXH_PRIME1 + XXH_PRIME2;
        uint32_t accumulator2 = seed + XXH_PRIME2;
        uint32_t accumulator3 = seed;
        uint32_t accumulator4 = seed - XXH_PRIME1;

        for (; cursor + 16u <= length; cursor += 16u)
        {
            accumulator1 = rotate_left_cpu(
                               accumulator1 +
                                   load_u32_le_cpu(data + cursor) * XXH_PRIME2,
                               13u) *
                           XXH_PRIME1;
            accumulator2 = rotate_left_cpu(
                               accumulator2 +
                                   load_u32_le_cpu(data + cursor + 4u) * XXH_PRIME2,
                               13u) *
                           XXH_PRIME1;
            accumulator3 = rotate_left_cpu(
                               accumulator3 +
                                   load_u32_le_cpu(data + cursor + 8u) * XXH_PRIME2,
                               13u) *
                           XXH_PRIME1;
            accumulator4 = rotate_left_cpu(
                               accumulator4 +
                                   load_u32_le_cpu(data + cursor + 12u) * XXH_PRIME2,
                               13u) *
                           XXH_PRIME1;
        }

        hash = rotate_left_cpu(accumulator1, 1u) +
               rotate_left_cpu(accumulator2, 7u) +
               rotate_left_cpu(accumulator3, 12u) +
               rotate_left_cpu(accumulator4, 18u);
    }
    else
    {
        hash = seed + XXH_PRIME5;
    }

    hash += length;
    while (cursor + 4u <= length)
    {
        hash += load_u32_le_cpu(data + cursor) * XXH_PRIME3;
        hash = rotate_left_cpu(hash, 17u) * XXH_PRIME4;
        cursor += 4u;
    }
    while (cursor < length)
    {
        hash += (uint32_t)data[cursor] * XXH_PRIME5;
        hash = rotate_left_cpu(hash, 11u) * XXH_PRIME1;
        ++cursor;
    }

    hash ^= hash >> 15u;
    hash *= XXH_PRIME2;
    hash ^= hash >> 13u;
    hash *= XXH_PRIME3;
    hash ^= hash >> 16u;
    return hash;
}

__global__ void hash_batch_kernel(const uint8_t *bytes,
                                  const uint32_t *offsets,
                                  const uint32_t *lengths,
                                  uint32_t key_count,
                                  uint32_t *hashes)
{
    const cg::thread_block_tile<4> tile =
        cg::tiled_partition<4>(cg::this_thread_block());
    const uint32_t key =
        (blockIdx.x * blockDim.x + threadIdx.x) / tile.size();

    if (key >= key_count)
        return;

    uint32_t hash1 = 0u;
    uint32_t hash2 = 0u;
    uint32_t hash3 = 0u;
    hash3_xxh32(bytes, offsets[key], lengths[key], &hash1, &hash2, &hash3);

    if (tile.thread_rank() == 0u)
    {
        hashes[key * 3u + 0u] = hash1;
        hashes[key * 3u + 1u] = hash2;
        hashes[key * 3u + 2u] = hash3;
    }
}

int main(void)
{
    const uint8_t key_a[1] = {'a'};
    const uint8_t key_abc[3] = {'a', 'b', 'c'};
    uint8_t key_15[15];
    uint8_t key_16[16];
    uint8_t key_17[17];
    uint8_t key_65[65];
    TestKey keys[TEST_KEY_COUNT];
    uint32_t offsets[TEST_KEY_COUNT];
    uint32_t lengths[TEST_KEY_COUNT];
    uint32_t gpu_hashes[TEST_HASH_COUNT];
    uint32_t null_empty_hashes[3];
    uint8_t *bytes = NULL;
    uint8_t *device_bytes = NULL;
    uint32_t *device_offsets = NULL;
    uint32_t *device_lengths = NULL;
    uint32_t *device_hashes = NULL;
    uint32_t total_bytes = 0u;
    uint32_t key;
    uint32_t seed;
    uint32_t i;
    uint32_t blocks;
    int passed = 1;

    if (xxh32_cpu(NULL, 0u, 0u) != 0x02CC5D05u ||
        xxh32_cpu(key_abc, 3u, 0u) != 0x32D153FFu)
    {
        fprintf(stderr, "CPU XXH32 reference failed known vectors\n");
        return 1;
    }

    for (i = 0u; i < sizeof(key_15); ++i)
        key_15[i] = (uint8_t)((i * 37u + 3u * 11u) & 0xFFu);
    for (i = 0u; i < sizeof(key_16); ++i)
        key_16[i] = (uint8_t)((i * 37u + 4u * 11u) & 0xFFu);
    for (i = 0u; i < sizeof(key_17); ++i)
        key_17[i] = (uint8_t)((i * 37u + 5u * 11u) & 0xFFu);
    for (i = 0u; i < sizeof(key_65); ++i)
        key_65[i] = (uint8_t)((i * 37u + 6u * 11u) & 0xFFu);

    keys[0].data = NULL;
    keys[0].length = 0u;
    keys[1].data = key_a;
    keys[1].length = sizeof(key_a);
    keys[2].data = key_abc;
    keys[2].length = sizeof(key_abc);
    keys[3].data = key_15;
    keys[3].length = sizeof(key_15);
    keys[4].data = key_16;
    keys[4].length = sizeof(key_16);
    keys[5].data = key_17;
    keys[5].length = sizeof(key_17);
    keys[6].data = key_65;
    keys[6].length = sizeof(key_65);

    for (key = 0u; key < TEST_KEY_COUNT; ++key)
        total_bytes += keys[key].length;

    bytes = (uint8_t *)malloc(total_bytes);
    if (bytes == NULL)
    {
        fprintf(stderr, "host allocation failed in hash test\n");
        return 1;
    }

    total_bytes = 0u;
    for (key = 0u; key < TEST_KEY_COUNT; ++key)
    {
        offsets[key] = total_bytes;
        lengths[key] = keys[key].length;
        if (keys[key].length != 0u)
            memcpy(bytes + total_bytes, keys[key].data, keys[key].length);
        total_bytes += keys[key].length;
    }

    CUDA_CHECK(cudaMalloc((void **)&device_bytes, total_bytes));
    CUDA_CHECK(cudaMalloc((void **)&device_offsets, sizeof(offsets)));
    CUDA_CHECK(cudaMalloc((void **)&device_lengths, sizeof(lengths)));
    CUDA_CHECK(cudaMalloc((void **)&device_hashes, sizeof(gpu_hashes)));
    CUDA_CHECK(cudaMemcpy(device_bytes, bytes, total_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_offsets, offsets, sizeof(offsets),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_lengths, lengths, sizeof(lengths),
                          cudaMemcpyHostToDevice));

    blocks = (TEST_KEY_COUNT * 4u + 127u) / 128u;
    hash_batch_kernel<<<blocks, 128>>>(device_bytes, device_offsets,
                                       device_lengths, TEST_KEY_COUNT,
                                       device_hashes);
    check_kernel("hash_batch_kernel");
    CUDA_CHECK(cudaMemcpy(gpu_hashes, device_hashes, sizeof(gpu_hashes),
                          cudaMemcpyDeviceToHost));

    for (key = 0u; key < TEST_KEY_COUNT; ++key)
    {
        for (seed = 0u; seed < 3u; ++seed)
        {
            uint32_t expected =
                xxh32_cpu(keys[key].data, keys[key].length, test_seeds[seed]);
            uint32_t actual = gpu_hashes[key * 3u + seed];
            if (actual != expected)
            {
                fprintf(stderr,
                        "hash mismatch for key %u, seed %u: expected %u, got %u\n",
                        key, seed, expected, actual);
                passed = 0;
            }
        }
    }

    /* Also verify that a zero-length key accepts a null byte pointer. */
    hash_batch_kernel<<<1, 4>>>(NULL, device_offsets, device_lengths, 1u,
                                device_hashes);
    check_kernel("hash_batch_kernel null empty key");
    CUDA_CHECK(cudaMemcpy(null_empty_hashes, device_hashes,
                          sizeof(null_empty_hashes), cudaMemcpyDeviceToHost));
    for (seed = 0u; seed < 3u; ++seed)
    {
        uint32_t expected = xxh32_cpu(NULL, 0u, test_seeds[seed]);
        if (null_empty_hashes[seed] != expected)
        {
            fprintf(stderr,
                    "null empty-key mismatch for seed %u: expected %u, got %u\n",
                    seed, expected, null_empty_hashes[seed]);
            passed = 0;
        }
    }

    CUDA_CHECK(cudaFree(device_hashes));
    CUDA_CHECK(cudaFree(device_lengths));
    CUDA_CHECK(cudaFree(device_offsets));
    CUDA_CHECK(cudaFree(device_bytes));
    free(bytes);

    printf("GPU/CPU XXH32 vectors: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
