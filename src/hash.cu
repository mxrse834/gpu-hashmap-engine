#include "hash.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define XXH_PRIME1 UINT32_C(0x9E3779B1)
#define XXH_PRIME2 UINT32_C(0x85EBCA77)
#define XXH_PRIME3 UINT32_C(0xC2B2AE3D)
#define XXH_PRIME4 UINT32_C(0x27D4EB2F)
#define XXH_PRIME5 UINT32_C(0x165667B1)

static __device__ __forceinline__ uint32_t rotate_left(uint32_t value,
                                                        uint32_t amount)
{
    return (value << amount) | (value >> (32u - amount));
}

// Byte assembly keeps the hash correct for arbitrary key alignment. A future
// optimized path can use aligned vector loads after preserving this fallback.
static __device__ __forceinline__ uint32_t load_u32_le(const uint8_t *data)
{
    return (uint32_t)data[0] |
           ((uint32_t)data[1] << 8u) |
           ((uint32_t)data[2] << 16u) |
           ((uint32_t)data[3] << 24u);
}

static __device__ __forceinline__ uint32_t xxh_round(uint32_t accumulator,
                                                     uint32_t input)
{
    accumulator += input * XXH_PRIME2;
    accumulator = rotate_left(accumulator, 13u);
    accumulator *= XXH_PRIME1;
    return accumulator;
}

static __device__ __forceinline__ uint32_t finish_hash(const uint8_t *data,
                                                        uint32_t cursor,
                                                        uint32_t length,
                                                        uint32_t hash)
{
    hash += length;

    while (cursor + 4u <= length)
    {
        hash += load_u32_le(data + cursor) * XXH_PRIME3;
        hash = rotate_left(hash, 17u) * XXH_PRIME4;
        cursor += 4u;
    }

    while (cursor < length)
    {
        hash += (uint32_t)data[cursor] * XXH_PRIME5;
        hash = rotate_left(hash, 11u) * XXH_PRIME1;
        ++cursor;
    }

    hash ^= hash >> 15u;
    hash *= XXH_PRIME2;
    hash ^= hash >> 13u;
    hash *= XXH_PRIME3;
    hash ^= hash >> 16u;
    return hash;
}

__device__ void hash3_xxh32(const uint8_t *bytes,
                            uint32_t start,
                            uint32_t length,
                            uint32_t *hash1,
                            uint32_t *hash2,
                            uint32_t *hash3)
{
    const cg::thread_block_tile<4> tile = cg::tiled_partition<4>(cg::this_thread_block());
    const uint32_t lane = tile.thread_rank();
    const uint32_t seeds[3] = {0x9E3779B1u, 0x517CC1B7u, 0x85EBCA6Bu};

    uint32_t lane_base;
    if (lane == 0u)
        lane_base = XXH_PRIME1 + XXH_PRIME2;
    else if (lane == 1u)
        lane_base = XXH_PRIME2;
    else if (lane == 2u)
        lane_base = 0u;
    else
        lane_base = 0u - XXH_PRIME1;

    uint32_t accumulator1 = seeds[0] + lane_base;
    uint32_t accumulator2 = seeds[1] + lane_base;
    uint32_t accumulator3 = seeds[2] + lane_base;
    uint32_t cursor = 0u;

    if (length >= 16u)
    {
        for (; cursor + 16u <= length; cursor += 16u)
        {
            const uint32_t word = load_u32_le(bytes + start + cursor + lane * 4u);
            accumulator1 = xxh_round(accumulator1, word);
            accumulator2 = xxh_round(accumulator2, word);
            accumulator3 = xxh_round(accumulator3, word);
        }

        // Cooperative-group shuffles are collectives, so every tile lane must
        // execute them even though only lane 0 consumes the merged value.
        const uint32_t merged1 =
            rotate_left(tile.shfl(accumulator1, 0), 1u) +
            rotate_left(tile.shfl(accumulator1, 1), 7u) +
            rotate_left(tile.shfl(accumulator1, 2), 12u) +
            rotate_left(tile.shfl(accumulator1, 3), 18u);
        const uint32_t merged2 =
            rotate_left(tile.shfl(accumulator2, 0), 1u) +
            rotate_left(tile.shfl(accumulator2, 1), 7u) +
            rotate_left(tile.shfl(accumulator2, 2), 12u) +
            rotate_left(tile.shfl(accumulator2, 3), 18u);
        const uint32_t merged3 =
            rotate_left(tile.shfl(accumulator3, 0), 1u) +
            rotate_left(tile.shfl(accumulator3, 1), 7u) +
            rotate_left(tile.shfl(accumulator3, 2), 12u) +
            rotate_left(tile.shfl(accumulator3, 3), 18u);

        if (lane == 0u)
        {
            *hash1 = merged1;
            *hash2 = merged2;
            *hash3 = merged3;
        }
    }
    else if (lane == 0u)
    {
        *hash1 = seeds[0] + XXH_PRIME5;
        *hash2 = seeds[1] + XXH_PRIME5;
        *hash3 = seeds[2] + XXH_PRIME5;
    }

    if (lane == 0u)
    {
        // Avoid pointer arithmetic on a null input for a zero-length key.
        // finish_hash does not dereference key when length is zero.
        const uint8_t *key = length == 0u ? bytes : bytes + start;
        *hash1 = finish_hash(key, cursor, length, *hash1);
        *hash2 = finish_hash(key, cursor, length, *hash2);
        *hash3 = finish_hash(key, cursor, length, *hash3);
    }
}
