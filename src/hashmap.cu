#include "hashmap.cuh"

#include "hash.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

typedef cg::thread_block_tile<4> Tile;

static __device__ __forceinline__ bool range_is_valid(uint32_t start,
                                                       uint32_t length,
                                                       uint32_t total_bytes)
{
    return start <= total_bytes && length <= total_bytes - start;
}

static __device__ __forceinline__ uint32_t overflow_start(uint32_t hash1,
                                                           uint32_t hash2,
                                                           uint32_t hash3,
                                                           uint32_t capacity)
{
    return ((hash1 ^ hash2 ^ hash3) * 0x9E3779B9u) % capacity;
}

static __device__ bool key_equals(Tile tile,
                                  const HashmapEngine *engine,
                                  uint32_t stored_offset,
                                  uint32_t stored_length,
                                  const uint8_t *query_bytes,
                                  uint32_t query_offset,
                                  uint32_t query_length)
{
    bool different = stored_length != query_length;

    if (!different)
    {
        for (uint32_t i = tile.thread_rank(); i < query_length; i += tile.size())
        {
            if (engine->master_bytes[stored_offset + i] != query_bytes[query_offset + i])
                different = true;
        }
    }

    return !tile.any(different);
}

static __device__ bool slot_matches(Tile tile,
                                    const HashmapEngine *engine,
                                    const uint32_t *slot_offsets,
                                    const uint32_t *slot_lengths,
                                    const uint32_t *slot_values,
                                    uint32_t slot,
                                    const uint8_t *query_bytes,
                                    uint32_t query_offset,
                                    uint32_t query_length,
                                    uint32_t *matched_value,
                                    bool *is_empty)
{
    uint32_t stored_offset = HASHMAP_EMPTY_SLOT;
    uint32_t stored_length = 0u;
    uint32_t stored_value = 0u;

    if (tile.thread_rank() == 0u)
    {
        stored_offset = slot_offsets[slot];
        if (stored_offset != HASHMAP_EMPTY_SLOT &&
            stored_offset != HASHMAP_TOMBSTONE_SLOT)
        {
            stored_length = slot_lengths[slot];
            stored_value = slot_values[slot];
        }
    }

    stored_offset = tile.shfl(stored_offset, 0);
    stored_length = tile.shfl(stored_length, 0);
    stored_value = tile.shfl(stored_value, 0);
    *is_empty = stored_offset == HASHMAP_EMPTY_SLOT;

    if (*is_empty || stored_offset == HASHMAP_TOMBSTONE_SLOT)
        return false;

    if (!range_is_valid(stored_offset, stored_length, engine->master_byte_current))
        return false;

    const bool matches = key_equals(tile, engine, stored_offset, stored_length,
                                    query_bytes, query_offset, query_length);
    if (matches)
        *matched_value = stored_value;
    return matches;
}

static __device__ bool claim_slot(uint32_t *slot_offsets,
                                  uint32_t *slot_lengths,
                                  uint32_t *slot_values,
                                  uint32_t slot,
                                  uint32_t key_offset,
                                  uint32_t key_length,
                                  uint32_t value)
{
    uint32_t previous = atomicCAS(slot_offsets + slot, HASHMAP_EMPTY_SLOT, key_offset);
    if (previous != HASHMAP_EMPTY_SLOT)
    {
        previous = atomicCAS(slot_offsets + slot, HASHMAP_TOMBSTONE_SLOT, key_offset);
        if (previous != HASHMAP_TOMBSTONE_SLOT)
            return false;
    }

    slot_lengths[slot] = key_length;
    slot_values[slot] = value;
    return true;
}

static __device__ bool delete_from_slot(Tile tile,
                                        HashmapEngine *engine,
                                        uint32_t *slot_offsets,
                                        const uint32_t *slot_lengths,
                                        uint32_t slot,
                                        const uint8_t *query_bytes,
                                        uint32_t query_offset,
                                        uint32_t query_length,
                                        bool *is_empty)
{
    uint32_t stored_offset = HASHMAP_EMPTY_SLOT;
    uint32_t stored_length = 0u;

    if (tile.thread_rank() == 0u)
    {
        stored_offset = slot_offsets[slot];
        if (stored_offset != HASHMAP_EMPTY_SLOT &&
            stored_offset != HASHMAP_TOMBSTONE_SLOT)
            stored_length = slot_lengths[slot];
    }

    stored_offset = tile.shfl(stored_offset, 0);
    stored_length = tile.shfl(stored_length, 0);
    *is_empty = stored_offset == HASHMAP_EMPTY_SLOT;

    if (*is_empty || stored_offset == HASHMAP_TOMBSTONE_SLOT ||
        !range_is_valid(stored_offset, stored_length, engine->master_byte_current))
        return false;

    if (!key_equals(tile, engine, stored_offset, stored_length,
                    query_bytes, query_offset, query_length))
        return false;

    uint32_t deleted = 0u;
    if (tile.thread_rank() == 0u)
        deleted = atomicCAS(slot_offsets + slot, stored_offset,
                            HASHMAP_TOMBSTONE_SLOT) == stored_offset;
    return tile.shfl(deleted, 0) != 0u;
}

__global__ void reserve_batch_kernel(HashmapEngine *engine,
                                     uint32_t total_bytes,
                                     uint32_t key_count)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u)
        return;

    const uint32_t current = engine->master_byte_current;
    if (total_bytes > engine->byte_capacity || current > engine->byte_capacity - total_bytes)
    {
        engine->batch_base = HASHMAP_EMPTY_SLOT;
        *engine->failed_inserts += key_count;
        return;
    }

    engine->batch_base = current;
    engine->master_byte_current = current + total_bytes;
}

__global__ void insert_kernel(HashmapEngine *engine,
                              const uint8_t *bytes,
                              const uint32_t *offsets,
                              const uint32_t *lengths,
                              const uint32_t *input_values,
                              uint32_t key_count,
                              uint32_t total_bytes)
{
    ///////EMPTY OR NOT DIVISIBLE BY 4 then RETURN
    if ((blockDim.x % 4u) != 0u || engine->batch_base == HASHMAP_EMPTY_SLOT)
        return;

    const Tile tile = cg::tiled_partition<4>(cg::this_thread_block());
    const uint64_t global_thread =(uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t first_key = global_thread / 4u;
    const uint64_t key_stride =((uint64_t)gridDim.x * blockDim.x) / 4u;

    ///////GRID STRIDE LOOP : Thread requirement supersedes the available total threads
    for (uint64_t key_index64 = first_key; key_index64 < key_count;key_index64 += key_stride)
    {
        const uint32_t key_index = (uint32_t)key_index64;
        const uint32_t start = offsets[key_index];
        const uint32_t length = lengths[key_index];

        /////////EXTRA PRECAUTION TO PREVENT ILLEGEAL MEMORY ACCCESS
        if (!range_is_valid(start, length, total_bytes))
        {
            if (tile.thread_rank() == 0u)
                atomicAdd(engine->failed_inserts, 1u);
            continue;
        }

        ////////// COPY TO MASTER_BYTES FROM BYTES
        const uint32_t stored_offset = engine->batch_base + start;
        uint32_t num_words = length / 4;

        for (uint32_t word = tile.thread_rank();word < num_words;word += tile.size())
        {
            uint32_t src_offset = start + word * 4;
            uint32_t dst_offset = stored_offset + word * 4;

            *reinterpret_cast<uint32_t *>(engine->master_bytes + dst_offset) = *reinterpret_cast<const uint32_t *>(bytes + src_offset);
        }

        uint32_t remaining = length % 4;

        for (uint32_t i = tile.thread_rank();i < remaining;i += tile.size())
        {
        engine->master_bytes[stored_offset + num_words * 4 + i] =
        bytes[start + num_words * 4 + i];
        }

        ////////////USING XXHASH32 to generate 3 hashes
        uint32_t hash1 = 0u;
        uint32_t hash2 = 0u;
        uint32_t hash3 = 0u;
        hash3_xxh32(bytes, start, length, &hash1, &hash2, &hash3);

        //////
        if (tile.thread_rank() == 0u)
        {
            const uint32_t slots[3] = {
                hash1 % engine->primary_capacity,
                hash2 % engine->primary_capacity,
                hash3 % engine->primary_capacity,
            };

            bool inserted = false;
            for (uint32_t candidate = 0u; candidate < 3u && !inserted; ++candidate)
            {
                inserted = claim_slot(engine->key_offsets, engine->key_lengths,
                                      engine->values, slots[candidate], stored_offset,
                                      length, input_values[key_index]);
            }

            if (!inserted)
            {
                const uint32_t first_overflow_slot =
                    overflow_start(hash1, hash2, hash3, engine->overflow_capacity);
                for (uint32_t probe = 0u; probe < engine->overflow_capacity; ++probe)
                {
                    const uint32_t slot = (first_overflow_slot + probe) % engine->overflow_capacity;
                    if (claim_slot(engine->overflow_offsets, engine->overflow_lengths,
                                   engine->overflow_values, slot, stored_offset, length,
                                   input_values[key_index]))
                    {
                        inserted = true;
                        break;
                    }
                }
            }

            if (!inserted)
                atomicAdd(engine->failed_inserts, 1u);
        }
    }
}

__global__ void lookup_kernel(const HashmapEngine *engine,
                              const uint8_t *query_bytes,
                              const uint32_t *query_offsets,
                              const uint32_t *query_lengths,
                              uint32_t query_count,
                              uint32_t total_query_bytes,
                              uint32_t *results,
                              uint8_t *found)
{
    if ((blockDim.x % 4u) != 0u)
        return;

    const Tile tile = cg::tiled_partition<4>(cg::this_thread_block());
    const uint64_t global_thread = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t first_query = global_thread / 4u;
    const uint64_t query_stride =((uint64_t)gridDim.x * blockDim.x) / 4u;

    for (uint64_t query_index64 = first_query; query_index64 < query_count;query_index64 += query_stride)
    {
        const uint32_t query_index = (uint32_t)query_index64;
        const uint32_t start = query_offsets[query_index];
        const uint32_t length = query_lengths[query_index];

        if (tile.thread_rank() == 0u)
        {
            found[query_index] = 0u;
            results[query_index] = 0u;
        }

        if (!range_is_valid(start, length, total_query_bytes))
            continue;

        uint32_t hash1 = 0u;
        uint32_t hash2 = 0u;
        uint32_t hash3 = 0u;
        hash3_xxh32(query_bytes, start, length, &hash1, &hash2, &hash3);

        const uint32_t raw_hash1 = tile.shfl(hash1, 0);
        const uint32_t raw_hash2 = tile.shfl(hash2, 0);
        const uint32_t raw_hash3 = tile.shfl(hash3, 0);
        const uint32_t slots[3] = {
            raw_hash1 % engine->primary_capacity,
            raw_hash2 % engine->primary_capacity,
            raw_hash3 % engine->primary_capacity,
        };

        bool matched = false;
        uint32_t matched_value = 0u;
        for (uint32_t candidate = 0u; candidate < 3u && !matched; ++candidate)
        {
            bool is_empty = false;
            matched = slot_matches(tile, engine, engine->key_offsets, engine->key_lengths,
                                   engine->values, slots[candidate], query_bytes, start,
                                   length, &matched_value, &is_empty);
        }

        if (matched)
        {
            if (tile.thread_rank() == 0u)
            {
                results[query_index] = matched_value;
                found[query_index] = 1u;
            }
            continue;
        }

        const uint32_t first_overflow_slot =
            overflow_start(raw_hash1, raw_hash2, raw_hash3,
                           engine->overflow_capacity);
        for (uint32_t probe = 0u; probe < engine->overflow_capacity; ++probe)
        {
            const uint32_t slot =
                (first_overflow_slot + probe) % engine->overflow_capacity;
            bool is_empty = false;
            matched = slot_matches(tile, engine, engine->overflow_offsets,
                                   engine->overflow_lengths, engine->overflow_values,
                                   slot, query_bytes, start, length, &matched_value,
                                   &is_empty);
            if (matched || is_empty)
                break;
        }

        if (matched && tile.thread_rank() == 0u)
        {
            results[query_index] = matched_value;
            found[query_index] = 1u;
        }
    }
}

__global__ void delete_kernel(HashmapEngine *engine,
                              const uint8_t *query_bytes,
                              const uint32_t *query_offsets,
                              const uint32_t *query_lengths,
                              uint32_t query_count,
                              uint32_t total_query_bytes,
                              uint8_t *deleted)
{
    if ((blockDim.x % 4u) != 0u)
        return;

    const Tile tile = cg::tiled_partition<4>(cg::this_thread_block());
    const uint64_t global_thread = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t first_query = global_thread / 4u;
    const uint64_t query_stride = ((uint64_t)gridDim.x * blockDim.x) / 4u;

    for (uint64_t query_index64 = first_query; query_index64 < query_count;query_index64 += query_stride)
    {
        const uint32_t query_index = (uint32_t)query_index64;
        const uint32_t start = query_offsets[query_index];
        const uint32_t length = query_lengths[query_index];

        if (tile.thread_rank() == 0u)
            deleted[query_index] = 0u;

        if (!range_is_valid(start, length, total_query_bytes))
            continue;

        uint32_t hash1 = 0u;
        uint32_t hash2 = 0u;
        uint32_t hash3 = 0u;
        hash3_xxh32(query_bytes, start, length, &hash1, &hash2, &hash3);

        const uint32_t raw_hash1 = tile.shfl(hash1, 0);
        const uint32_t raw_hash2 = tile.shfl(hash2, 0);
        const uint32_t raw_hash3 = tile.shfl(hash3, 0);
        const uint32_t slots[3] = {
            raw_hash1 % engine->primary_capacity,
            raw_hash2 % engine->primary_capacity,
            raw_hash3 % engine->primary_capacity,
        };

        bool was_deleted = false;
        for (uint32_t candidate = 0u; candidate < 3u && !was_deleted; ++candidate)
        {
            bool is_empty = false;
            was_deleted = delete_from_slot(tile, engine, engine->key_offsets,
                                           engine->key_lengths, slots[candidate],
                                           query_bytes, start, length, &is_empty);
        }

        if (!was_deleted)
        {
            const uint32_t first_overflow_slot =
                overflow_start(raw_hash1, raw_hash2, raw_hash3,
                               engine->overflow_capacity);
            for (uint32_t probe = 0u; probe < engine->overflow_capacity; ++probe)
            {
                const uint32_t slot =
                    (first_overflow_slot + probe) % engine->overflow_capacity;
                bool is_empty = false;
                was_deleted = delete_from_slot(tile, engine, engine->overflow_offsets,
                                               engine->overflow_lengths, slot,
                                               query_bytes, start, length, &is_empty);
                if (was_deleted || is_empty)
                    break;
            }
        }

        if (was_deleted && tile.thread_rank() == 0u)
            deleted[query_index] = 1u;
    }
}
