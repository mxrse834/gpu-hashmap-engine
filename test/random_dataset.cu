#include<iostream>
#include<cstdint>
#include<cooperative_groups.h>
#include<curand_kernel.h>
#include<errno.h>
#include<cuda_runtime.h>
#include<chrono>
#include"hash.cuh"

#define XXH_PRIME1 UINT32_C(0x9E3779B1)
#define XXH_PRIME2 UINT32_C(0x85EBCA77)
#define XXH_PRIME3 UINT32_C(0xC2B2AE3D)
#define XXH_PRIME4 UINT32_C(0x27D4EB2F)
#define XXH_PRIME5 UINT32_C(0x165667B1)

namespace cg = cooperative_groups;
typedef uint32_t u32;
typedef uint8_t u8;

typedef struct
{
    const uint8_t *data;
    uint32_t length;
} TestKey;

__global__ void pretty_random_bol_gen(uint32_t *b , uint32_t *o ,uint32_t* l, uint32_t n , uint64_t seed = 3000)
{
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < n/4)
    {
        if(tid<(n/16))
        {
            o[tid] = tid*16; 
        }
        if(tid == (n/16)+1)
        {
            o[tid] = n-1;
        }
        curandStateXORWOW_t state;
        curand_init(seed ,tid , 0 , &state);
        b[tid] = curand(&state);
    }

}

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
/*
TEST 1
known XXH32 vectors
CPU == expected
GPU == expected

TEST 2
random 1-byte keys
CPU == GPU

TEST 3
random 15/16/17-byte keys
CPU == GPU

TEST 4
random 65-byte keys
CPU == GPU

TEST 5
large random dataset
CPU == GPU

TEST 6
empty key
CPU == GPU


    batch->key_count = (uint32_t)chunk_count64;
    batch->total_bytes = (uint32_t)encoded_bytes64;
    batch->bytes = (uint8_t *)malloc((size_t)batch->total_bytes);
    batch->offsets = (uint32_t *)malloc((size_t)batch->key_count * sizeof(uint32_t));
    batch->lengths = (uint32_t *)malloc((size_t)batch->key_count * sizeof(uint32_t));
    batch->values = (uint32_t *)malloc((size_t)batch->key_count * sizeof(uint32_t));

    if (batch->bytes == NULL || batch->offsets == NULL || batch->lengths == NULL || batch->values == NULL)
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

*/

int main(int argc , char* argv[])
{
if(argc != 1)
{fprintf(stderr,"Incorrect number of arguments");return -1;}
u32* gpu_a;
u32* cpu_a;
char* end;
u32 n = (u8)(strtoull(argv[1],&end,10));
if(end != "\0" || errno == EINVAL || errno == ERANGE)
{fprintf(stderr,"Parsing error");return -1;}    

cudaMalloc((void**)&gpu_a,sizeof(u8)*n); // atleast 256byte alignment is guaranteed so neednt worry about u8 -> u32 from base address
pretty_random_bol_gen(gpu_a,n); // n is strictly thr number of bytes in the array
cudaMemcpy(cpu_a,gpu_a,sizeof(u32)*n,cudaMemcpyDeviceToHost);

hash_batch_kernel();
xxh32_cpu()




u32* 


}