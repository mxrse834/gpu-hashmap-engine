#define _FILE_OFFSET_BITS 64

#include "cuda_check.cuh"
#include "hashmap.cuh"

#include <cuda_runtime.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define FILE_CHUNK_BYTES 16u
#define FILE_INDEX_BYTES 4u
#define FILE_KEY_BYTES (FILE_INDEX_BYTES + FILE_CHUNK_BYTES)

typedef struct
{
    uint8_t *data;
    size_t size;
} FileBuffer;

typedef struct
{
    uint8_t *bytes;
    uint32_t *offsets;
    uint32_t *lengths;
    uint32_t *values;
    uint32_t key_count;
    uint32_t total_bytes;
} Batch;

typedef struct
{
    uint8_t *bytes;
    uint32_t *offsets;
    uint32_t *lengths;
    uint32_t *values;
} DeviceBatch;

typedef struct
{
    HashmapEngine host;
    HashmapEngine *device;
} DeviceHashmap;

static void free_file(FileBuffer *file)
{
    free(file->data);
    file->data = NULL;
    file->size = 0u;
}

static int read_file(const char *path, FileBuffer *result)
{
    FILE *file;
    off_t end;
    size_t bytes_read;

    memset(result, 0, sizeof(*result));
    file = fopen(path, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "error: cannot open %s: %s\n", path, strerror(errno));
        return 0;
    }

    if (fseeko(file, 0, SEEK_END) != 0 || (end = ftello(file)) < 0)
    {
        fprintf(stderr, "error: cannot determine file size: %s\n", path);
        fclose(file);
        return 0;
    }
    if ((uint64_t)end > (uint64_t)SIZE_MAX)
    {
        fprintf(stderr, "error: file is too large for this host: %s\n", path);
        fclose(file);
        return 0;
    }
    if (fseeko(file, 0, SEEK_SET) != 0)
    {
        fprintf(stderr, "error: cannot rewind file: %s\n", path);
        fclose(file);
        return 0;
    }

    result->size = (size_t)end;
    if (result->size != 0u)
    {
        result->data = (uint8_t *)malloc(result->size);
        if (result->data == NULL)
        {
            fprintf(stderr, "error: host allocation failed for %s\n", path);
            fclose(file);
            return 0;
        }

        bytes_read = fread(result->data, 1u, result->size, file);
        if (bytes_read != result->size)
        {
            fprintf(stderr, "error: incomplete read from %s\n", path);
            fclose(file);
            free_file(result);
            return 0;
        }
    }

    if (fclose(file) != 0)
    {
        fprintf(stderr, "error: failed to close %s\n", path);
        free_file(result);
        return 0;
    }
    return 1;
}

static void free_batch(Batch *batch)
{
    free(batch->bytes);
    free(batch->offsets);
    free(batch->lengths);
    free(batch->values);
    memset(batch, 0, sizeof(*batch));
}

static int make_positioned_chunk_batch(const FileBuffer *file, Batch *batch)
{
    uint64_t chunk_count64;
    uint64_t encoded_bytes64;
    uint32_t chunk;

    memset(batch, 0, sizeof(*batch));
    chunk_count64 = (uint64_t)(file->size / FILE_CHUNK_BYTES);
    if ((file->size % FILE_CHUNK_BYTES) != 0u)
        ++chunk_count64;

    if (chunk_count64 > UINT32_MAX ||
        chunk_count64 > SIZE_MAX / sizeof(uint32_t))
    {
        fprintf(stderr, "error: file exceeds the engine's 32-bit indexing limit\n");
        return 0;
    }

    encoded_bytes64 = chunk_count64 * FILE_KEY_BYTES;
    if (encoded_bytes64 >= HASHMAP_TOMBSTONE_SLOT)
    {
        fprintf(stderr, "error: encoded keys exceed the byte-arena limit\n");
        return 0;
    }

    batch->key_count = (uint32_t)chunk_count64;
    batch->total_bytes = (uint32_t)encoded_bytes64;
    batch->bytes = (uint8_t *)calloc((size_t)batch->total_bytes, 1u);
    batch->offsets = (uint32_t *)malloc((size_t)batch->key_count * sizeof(uint32_t));
    batch->lengths = (uint32_t *)malloc((size_t)batch->key_count * sizeof(uint32_t));
    batch->values = (uint32_t *)malloc((size_t)batch->key_count * sizeof(uint32_t));

    if (batch->bytes == NULL || batch->offsets == NULL || batch->lengths == NULL ||
        batch->values == NULL)
    {
        fprintf(stderr, "error: host allocation failed while encoding file chunks\n");
        free_batch(batch);
        return 0;
    }

    for (chunk = 0u; chunk < batch->key_count; ++chunk)
    {
        uint32_t destination = chunk * FILE_KEY_BYTES;
        size_t source = (size_t)chunk * FILE_CHUNK_BYTES;
        size_t remaining = file->size - source;
        size_t copy_size = remaining < FILE_CHUNK_BYTES ? remaining : FILE_CHUNK_BYTES;

        batch->offsets[chunk] = destination;
        batch->lengths[chunk] = FILE_KEY_BYTES;
        batch->values[chunk] = chunk;

        batch->bytes[destination + 0u] = (uint8_t)chunk;
        batch->bytes[destination + 1u] = (uint8_t)(chunk >> 8u);
        batch->bytes[destination + 2u] = (uint8_t)(chunk >> 16u);
        batch->bytes[destination + 3u] = (uint8_t)(chunk >> 24u);
        memcpy(batch->bytes + destination + FILE_INDEX_BYTES,
               file->data + source, copy_size);
    }
    return 1;
}

static void free_device_batch(DeviceBatch *batch)
{
    if (batch->bytes != NULL)
        cudaFree(batch->bytes);
    if (batch->offsets != NULL)
        cudaFree(batch->offsets);
    if (batch->lengths != NULL)
        cudaFree(batch->lengths);
    if (batch->values != NULL)
        cudaFree(batch->values);
    memset(batch, 0, sizeof(*batch));
}

static void upload_batch(const Batch *host, DeviceBatch *device, int upload_values)
{
    memset(device, 0, sizeof(*device));

    CUDA_CHECK(cudaMalloc((void **)&device->bytes, host->total_bytes));
    CUDA_CHECK(cudaMalloc((void **)&device->offsets,
                          (size_t)host->key_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&device->lengths,
                          (size_t)host->key_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(device->bytes, host->bytes, host->total_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device->offsets, host->offsets,
                          (size_t)host->key_count * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device->lengths, host->lengths,
                          (size_t)host->key_count * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    if (upload_values)
    {
        CUDA_CHECK(cudaMalloc((void **)&device->values,
                              (size_t)host->key_count * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(device->values, host->values,
                              (size_t)host->key_count * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    }
}

static void free_device_hashmap(DeviceHashmap *map)
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

static int create_device_hashmap(DeviceHashmap *map,
                                 uint32_t primary_capacity,
                                 uint32_t overflow_capacity,
                                 uint32_t byte_capacity)
{
    HashmapEngine *engine;

    memset(map, 0, sizeof(*map));
    if (primary_capacity == 0u || overflow_capacity == 0u || byte_capacity == 0u ||
        byte_capacity >= HASHMAP_TOMBSTONE_SLOT)
    {
        fprintf(stderr, "error: invalid hashmap capacity\n");
        return 0;
    }

    engine = &map->host;
    engine->primary_capacity = primary_capacity;
    engine->overflow_capacity = overflow_capacity;
    engine->byte_capacity = byte_capacity;
    engine->master_byte_current = 0u;
    engine->batch_base = HASHMAP_EMPTY_SLOT;

    CUDA_CHECK(cudaMalloc((void **)&engine->master_bytes, byte_capacity));
    CUDA_CHECK(cudaMalloc((void **)&engine->key_offsets,
                          (size_t)primary_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&engine->key_lengths,
                          (size_t)primary_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&engine->values,
                          (size_t)primary_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&engine->overflow_offsets,
                          (size_t)overflow_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&engine->overflow_lengths,
                          (size_t)overflow_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&engine->overflow_values,
                          (size_t)overflow_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&engine->failed_inserts, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemset(engine->key_offsets, 0xFF,
                          (size_t)primary_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(engine->overflow_offsets, 0xFF,
                          (size_t)overflow_capacity * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(engine->failed_inserts, 0, sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void **)&map->device, sizeof(HashmapEngine)));
    CUDA_CHECK(cudaMemcpy(map->device, engine, sizeof(HashmapEngine),
                          cudaMemcpyHostToDevice));
    return 1;
}

static uint32_t failed_insert_count(const DeviceHashmap *map)
{
    uint32_t failures;
    CUDA_CHECK(cudaMemcpy(&failures, map->host.failed_inserts, sizeof(failures),
                          cudaMemcpyDeviceToHost));
    return failures;
}

static int next_power_of_two(uint64_t value, uint32_t *result)
{
    if (value <= 1u)
    {
        *result = 1u;
        return 1;
    }
    if (value > (UINT64_C(1) << 31u))
        return 0;

    --value;
    value |= value >> 1u;
    value |= value >> 2u;
    value |= value >> 4u;
    value |= value >> 8u;
    value |= value >> 16u;
    value |= value >> 32u;
    *result = (uint32_t)(value + 1u);
    return 1;
}

static uint32_t launch_blocks(uint32_t key_count, const cudaDeviceProp *properties)
{
    uint64_t threads_needed = (uint64_t)key_count * 4u;
    uint64_t blocks_needed =
        (threads_needed + HASHMAP_THREADS_PER_BLOCK - 1u) / HASHMAP_THREADS_PER_BLOCK;
    uint64_t block_cap = (uint64_t)properties->multiProcessorCount * 4u;

    if (block_cap == 0u)
        block_cap = 1u;
    if (blocks_needed == 0u)
        blocks_needed = 1u;
    if (blocks_needed > block_cap)
        blocks_needed = block_cap;
    return (uint32_t)blocks_needed;
}

static void print_usage(const char *program)
{
    printf("Usage: %s <file-a> <file-b>\n", program);
}

int main(int argc, char **argv)
{
    FileBuffer file_a;
    FileBuffer file_b;
    Batch insert_batch;
    Batch query_batch;
    DeviceBatch device_insert;
    DeviceBatch device_query;
    DeviceHashmap map;
    uint32_t *device_results = NULL;
    uint8_t *device_found = NULL;
    uint32_t *host_results = NULL;
    uint8_t *host_found = NULL;
    uint32_t primary_capacity;
    uint32_t overflow_capacity;
    uint32_t blocks;
    uint32_t failures;
    uint32_t i;
    uint64_t primary_request;
    uint64_t overflow_request;
    int device_id;
    int identical;
    int exit_code = 2;
    cudaDeviceProp properties;

    memset(&file_a, 0, sizeof(file_a));
    memset(&file_b, 0, sizeof(file_b));
    memset(&insert_batch, 0, sizeof(insert_batch));
    memset(&query_batch, 0, sizeof(query_batch));
    memset(&device_insert, 0, sizeof(device_insert));
    memset(&device_query, 0, sizeof(device_query));
    memset(&map, 0, sizeof(map));

    if (argc == 2 && strcmp(argv[1], "--help") == 0)
    {
        print_usage(argv[0]);
        return 0;
    }
    if (argc != 3)
    {
        print_usage(argv[0]);
        return 2;
    }

    if (!read_file(argv[1], &file_a) || !read_file(argv[2], &file_b))
        goto cleanup;

    printf("file-a bytes: %zu\n", file_a.size);
    printf("file-b bytes: %zu\n", file_b.size);

    if (file_a.size != file_b.size)
    {
        printf("result: DIFFERENT (size mismatch)\n");
        exit_code = 1;
        goto cleanup;
    }
    if (file_a.size == 0u)
    {
        printf("result: IDENTICAL\n");
        exit_code = 0;
        goto cleanup;
    }

    if (!make_positioned_chunk_batch(&file_a, &insert_batch) ||
        !make_positioned_chunk_batch(&file_b, &query_batch))
        goto cleanup;

    primary_request = (uint64_t)insert_batch.key_count * 2u;
    overflow_request = (uint64_t)insert_batch.key_count / 4u + 1u;
    if (primary_request < 64u)
        primary_request = 64u;
    if (overflow_request < 64u)
        overflow_request = 64u;

    if (!next_power_of_two(primary_request, &primary_capacity) ||
        !next_power_of_two(overflow_request, &overflow_capacity))
    {
        fprintf(stderr, "error: requested table capacity exceeds 32-bit limits\n");
        goto cleanup;
    }

    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id));
    blocks = launch_blocks(insert_batch.key_count, &properties);

    printf("device: %s (sm_%d%d)\n", properties.name, properties.major,
           properties.minor);
    printf("keys: %u x %u bytes, launch: %u blocks x %u threads\n",
           insert_batch.key_count, FILE_KEY_BYTES, blocks,
           HASHMAP_THREADS_PER_BLOCK);

    if (!create_device_hashmap(&map, primary_capacity, overflow_capacity,
                               insert_batch.total_bytes))
        goto cleanup;

    upload_batch(&insert_batch, &device_insert, 1);
    reserve_batch_kernel<<<1, 1>>>(map.device, insert_batch.total_bytes,
                                   insert_batch.key_count);
    check_kernel("reserve_batch_kernel");
    insert_kernel<<<blocks, HASHMAP_THREADS_PER_BLOCK>>>(
        map.device, device_insert.bytes, device_insert.offsets,
        device_insert.lengths, device_insert.values, insert_batch.key_count,
        insert_batch.total_bytes);
    check_kernel("insert_kernel");

    failures = failed_insert_count(&map);
    if (failures != 0u)
    {
        fprintf(stderr, "error: hashmap insertion failed for %u keys\n", failures);
        goto cleanup;
    }

    upload_batch(&query_batch, &device_query, 0);
    CUDA_CHECK(cudaMalloc((void **)&device_results,
                          (size_t)query_batch.key_count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&device_found,
                          (size_t)query_batch.key_count * sizeof(uint8_t)));

    lookup_kernel<<<blocks, HASHMAP_THREADS_PER_BLOCK>>>(
        map.device, device_query.bytes, device_query.offsets, device_query.lengths,
        query_batch.key_count, query_batch.total_bytes, device_results, device_found);
    check_kernel("lookup_kernel");

    host_results = (uint32_t *)malloc((size_t)query_batch.key_count * sizeof(uint32_t));
    host_found = (uint8_t *)malloc((size_t)query_batch.key_count * sizeof(uint8_t));
    if (host_results == NULL || host_found == NULL)
    {
        fprintf(stderr, "error: host allocation failed for lookup results\n");
        goto cleanup;
    }

    CUDA_CHECK(cudaMemcpy(host_results, device_results,
                          (size_t)query_batch.key_count * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_found, device_found,
                          (size_t)query_batch.key_count * sizeof(uint8_t),
                          cudaMemcpyDeviceToHost));

    identical = 1;
    for (i = 0u; i < query_batch.key_count; ++i)
    {
        if (host_found[i] == 0u || host_results[i] != i)
        {
            identical = 0;
            break;
        }
    }

    printf("result: %s\n", identical ? "IDENTICAL" : "DIFFERENT");
    exit_code = identical ? 0 : 1;

cleanup:
    free(host_results);
    free(host_found);
    if (device_results != NULL)
        cudaFree(device_results);
    if (device_found != NULL)
        cudaFree(device_found);
    free_device_batch(&device_insert);
    free_device_batch(&device_query);
    free_device_hashmap(&map);
    free_batch(&insert_batch);
    free_batch(&query_batch);
    free_file(&file_a);
    free_file(&file_b);
    return exit_code;
}
