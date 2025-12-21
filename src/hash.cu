// attempt to port a modified 32 bit xxhash on gpu
// coop grps , shf_sync , ballot_sync
// NOTES -
// 1) bit associativity calc was a massive fail
// 2) later we wanna tweak the program to launch 2 diff kernels for for lebght greater than 16 and one for lesser than 16

// We have 3 options for calculations -
// 1)
//  ============================================================================
//  STRATEGY 1: MAXIMUM BATCH PARALLELISM (BEST FOR MANY STRINGS)
//  Each thread hashes ONE complete string independently
//  Use when: Hashing millions of passwords, URLs, database keys
//  Parallelism: N threads = N strings simultaneously
//  ============================================================================

// 2)
//  ============================================================================
//  STRATEGY 2: WARP-COOPERATIVE (BEST FOR MEDIUM STRINGS)
//  4 threads per string, process v1/v2/v3/v4 in parallel
//  Use when: Strings are 100-10000 bytes, moderate batch size
//  Parallelism: N/4 threads = N strings, with 4x internal parallelism
//  ============================================================================

// 3)
//  ============================================================================
//  STRATEGY 3: ULTRA-WIDE CHUNK PARALLELISM (BEST FOR HUGE STRINGS)
//  Process multiple chunks in parallel, then combine
//  Use when: Single massive string (GB+), like streaming file hash
//  Parallelism: Thousands of threads work on same string
//  ============================================================================

/*CPU PSEUDOCODE

Step 1:initialize counter
if length >= 16:
    v1 = seed + PRIME1 + PRIME2
    v2 = seed + PRIME2
    v3 = seed
    v4 = seed - PRIME1
else:
    acc = seed + PRIME5



Step 2: Process input in 16-byte chunks
for each 16-byte block in data:
    v1 = round(v1, word0)
    v2 = round(v2, word1)
    v3 = round(v3, word2)
    v4 = round(v4, word3)

where round(acc, input) =
    acc = acc + (input * PRIME2)
    acc = rotate_left(acc, 13)
    acc = acc * PRIME1



Step 3: Merge accumulators
if length >= 16:
    acc = rotate_left(v1, 1) +
          rotate_left(v2, 7) +
          rotate_left(v3, 12) +
          rotate_left(v4, 18)
else:
    acc = seed + PRIME5



Step 4: Process remaining bytes (<16)
acc = acc + length

while 4 bytes remain:
    k1 = word
    k1 = k1 * PRIME3
    k1 = rotate_left(k1, 17)
    k1 = k1 * PRIME4
    acc = acc ^ k1
    acc = rotate_left(acc, 17) * PRIME1 + PRIME4

while 1 byte remains:
    k1 = byte * PRIME5
    acc = acc ^ k1
    acc = rotate_left(acc, 11) * PRIME1




Step 5: Final avalanche (mixing)
acc = acc ^ (acc >> 15)
acc = acc * PRIME2
acc = acc ^ (acc >> 13)
acc = acc * PRIME3
acc = acc ^ (acc >> 16)


*/
#include<hash.cuh>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <cooperative_groups.h>
using namespace std;
#define TPB 1024
#define SEED 0
namespace cg = cooperative_groups;
#define P1 2246822519;
#define P2 3266489917;
#define P3 4294967296;
#define P4 668265263;
#define P5 374761393;

/*__device__ __forceinline__ uint32_t inst(uint32_t x,int s)
{
    return __funnelshift_l(x,x,s);
}*/
__device__ __constant__ uint32_t g[7] = {(0x9E3779B1 + 0x85EBCA77), 0x85EBCA77, 0, -0x9E3779B1, 0xC2B2AE3D, 0x27D4EB2F, 0x165667B1};
__device__ __constant__ uint32_t g1[4] = {1, 7, 12, 18};
__device__ __forceinline__ uint32_t inst(uint32_t x, int r)
{
    return (x << r) | (x >> (32 - r));
}

__device__ __forceinline__ uint32_t round(uint32_t r, uint32_t w)
{
    // if we have say a 33 byte string to hash
    // we know each v(1,2,3,4) will handle 4 bytes
    // we need to form a grid stride loop to continue processing consecutive words
    r += w * g[1];
    r = inst(r, 13);
    r *= (-g[3]);
    return r;
}

__device__ void xh332(
    uint8_t *bytes,
    uint32_t tid, 
    uint32_t wid,
    uint32_t *offset,
    uint32_t *words, 
    uint32_t &res1,
    uint32_t &res2,
    uint32_t &res3,
    uint32_t length_bytes,
    uint32_t length_offset
) {
    //const uint32_t *words = reinterpret_cast<const uint32_t *>(bytes);
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(cg::this_thread_block());
    //uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;  
    //uint32_t local_wid=(i+threadIdx.x)/4;
    uint32_t start = offset[wid];
    uint32_t len = offset[wid + 1] - start;  
    uint32_t posn = (start / 4) + tile.thread_rank();
    
    // Three sets of accumulators - one per hash function
    uint32_t v1[4], v2[4], v3[4];
    
    // Three different seeds for three hash functions
    const uint32_t seeds[3] = {0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D};
    
    // Initialize all three sets of accumulators
    // Each uses g[tile.thread_rank()] + their respective seed
    v1[tile.thread_rank()] = g[tile.thread_rank()] + seeds[0];
    v2[tile.thread_rank()] = g[tile.thread_rank()] + seeds[1];
    v3[tile.thread_rank()] = g[tile.thread_rank()] + seeds[2];
    
    uint32_t mask = (len >= 16);
    
    // Initialize results for all three hash functions
    // Uses same logic as your: res = (1 - mask) * (SEED + g[6])
    res1 = (1 - mask) * (seeds[0] + g[6]);
    res2 = (1 - mask) * (seeds[1] + g[6]);
    res3 = (1 - mask) * (seeds[2] + g[6]);
    
    uint32_t j = 0;
    
    // Process 16-byte chunks 
    // Load word once, process through all three hash functions
    for (; j + 16 <= len; j += 16) {
        uint32_t word = words[posn];  // Single load from memory
        v1[tile.thread_rank()] = round(v1[tile.thread_rank()], word);
        v2[tile.thread_rank()] = round(v2[tile.thread_rank()], word);
        v3[tile.thread_rank()] = round(v3[tile.thread_rank()], word);
        posn += 4;
    }
    
    // Merge accumulators - your exact logic, repeated for each hash
    if (tile.any(mask) && tile.thread_rank() == 0) {
        // Hash function 1 merge
        res1 = inst(v1[0], 1);
        res1 += tile.shfl(inst(v1[1], 7), 1);
        res1 += tile.shfl(inst(v1[2], 12), 2);
        res1 += tile.shfl(inst(v1[3], 18), 3);
        
        // Hash function 2 merge
        res2 = inst(v2[0], 1);
        res2 += tile.shfl(inst(v2[1], 7), 1);
        res2 += tile.shfl(inst(v2[2], 12), 2);
        res2 += tile.shfl(inst(v2[3], 18), 3);
        
        // Hash function 3 merge
        res3 = inst(v3[0], 1);
        res3 += tile.shfl(inst(v3[1], 7), 1);
        res3 += tile.shfl(inst(v3[2], 12), 2);
        res3 += tile.shfl(inst(v3[3], 18), 3);
    }
    // Process remaining bytes - YOUR EXACT LOGIC, just for three results
    if (tile.thread_rank() == 0) {
        // Add length to all three results
        res1 += len;
        res2 += len;
        res3 += len;
        
        uint32_t processed = (len / 16) * 16;
        uint32_t k1;
        uint32_t i = processed;  // Using your variable name and initialization
        
        // Process remaining 4-byte chunks - your exact for loop structure
        for (i; (i + 4) <= len; i += 4) {
            k1 = words[(start + i)/4];
            k1 *= g[4];
            k1 = inst(k1, 17);
            k1 *= g[5];
            
            // Apply k1 to all three hash results
            res1 ^= k1;
            res1 = inst(res1, 17) * -g[3] + g[5];
            
            res2 ^= k1;
            res2 = inst(res2, 17) * -g[3] + g[5];
            
            res3 ^= k1;
            res3 = inst(res3, 17) * -g[3] + g[5];
        }
        
        // Process remaining individual bytes
        while (i < len) {
            k1 = bytes[start + i] * g[6];
            
            // Apply to all three results
            res1 ^= k1;
            res1 = inst(res1, 11) * -g[3];
            
            res2 ^= k1;
            res2 = inst(res2, 11) * -g[3];
            
            res3 ^= k1;
            res3 = inst(res3, 11) * -g[3];
            
            i++;
        }
        
        // Final avalanche
        res1 ^= res1 >> 15;
        res1 *= g[1];
        res1 ^= res1 >> 13;
        res1 *= g[4];
        res1 ^= res1 >> 16;
        
        res2 ^= res2 >> 15;
        res2 *= g[1];
        res2 ^= res2 >> 13;
        res2 *= g[4];
        res2 ^= res2 >> 16;
        
        res3 ^= res3 >> 15;
        res3 *= g[1];
        res3 ^= res3 >> 13;
        res3 *= g[4];
        res3 ^= res3 ;
    }
}

/*
#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>

namespace cg = cooperative_groups;

#define SEED 0
#define PRIME1 0x9E3779B1U
#define PRIME2 0x85EBCA77U
#define PRIME3 0xC2B2AE3DU
#define PRIME4 0x27D4EB2FU
#define PRIME5 0x165667B1U

__device__ __forceinline__ uint32_t rotl(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

__device__ __forceinline__ uint32_t round(uint32_t acc, uint32_t input) {
    acc += input * PRIME2;
    acc = rotl(acc, 13);
    acc *= PRIME1;
    return acc;
}

// CPU reference
uint32_t xxhash32_cpu(const uint8_t* data, size_t len, uint32_t seed) {
    const uint8_t* p = data;
    const uint8_t* end = data + len;
    uint32_t h32;

    if (len >= 16) {
        const uint8_t* limit = end - 16;
        uint32_t v1 = seed + PRIME1 + PRIME2;
        uint32_t v2 = seed + PRIME2;
        uint32_t v3 = seed;
        uint32_t v4 = seed - PRIME1;

        do {
            v1 = round(v1, *((const uint32_t*)p)); p += 4;
            v2 = round(v2, *((const uint32_t*)p)); p += 4;
            v3 = round(v3, *((const uint32_t*)p)); p += 4;
            v4 = round(v4, *((const uint32_t*)p)); p += 4;
        } while (p <= limit);

        h32 = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    } else {
        h32 = seed + PRIME5;
    }

    h32 += (uint32_t)len;

    while (p + 4 <= end) {
        h32 += (*((const uint32_t*)p)) * PRIME3;
        h32 = rotl(h32, 17) * PRIME4;
        p += 4;
    }

    while (p < end) {
        h32 += (*p) * PRIME5;
        h32 = rotl(h32, 11) * PRIME1;
        p++;
    }

    h32 ^= h32 >> 15;
    h32 *= PRIME2;
    h32 ^= h32 >> 13;
    h32 *= PRIME3;
    h32 ^= h32 >> 16;

    return h32;
}

// ============================================================================
// STRATEGY 1: MAXIMUM BATCH PARALLELISM (BEST FOR MANY STRINGS)
// Each thread hashes ONE complete string independently
// Use when: Hashing millions of passwords, URLs, database keys
// Parallelism: N threads = N strings simultaneously
// ============================================================================
__global__ void xxhash32_max_throughput(
    const uint8_t* data,         // Concatenated string data
    const uint32_t* offsets,     // Start offset for each string
    const uint32_t* lengths,     // Length of each string
    uint32_t* results,
    int num_strings,
    uint32_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_strings) return;

    const uint8_t* str = data + offsets[idx];
    uint32_t len = lengths[idx];
    const uint8_t* p = str;
    const uint8_t* end = str + len;
    uint32_t h32;

    // Each thread does complete xxHash independently
    if (len >= 16) {
        const uint8_t* limit = end - 16;
        uint32_t v1 = seed + PRIME1 + PRIME2;
        uint32_t v2 = seed + PRIME2;
        uint32_t v3 = seed;
        uint32_t v4 = seed - PRIME1;

        do {
            v1 = round(v1, *((const uint32_t*)p)); p += 4;
            v2 = round(v2, *((const uint32_t*)p)); p += 4;
            v3 = round(v3, *((const uint32_t*)p)); p += 4;
            v4 = round(v4, *((const uint32_t*)p)); p += 4;
        } while (p <= limit);

        h32 = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    } else {
        h32 = seed + PRIME5;
    }

    h32 += len;

    while (p + 4 <= end) {
        h32 += (*((const uint32_t*)p)) * PRIME3;
        h32 = rotl(h32, 17) * PRIME4;
        p += 4;
    }

    while (p < end) {
        h32 += (*p) * PRIME5;
        h32 = rotl(h32, 11) * PRIME1;
        p++;
    }

    h32 ^= h32 >> 15;
    h32 *= PRIME2;
    h32 ^= h32 >> 13;
    h32 *= PRIME3;
    h32 ^= h32 >> 16;

    results[idx] = h32;
}

// ============================================================================
// STRATEGY 2: WARP-COOPERATIVE (BEST FOR MEDIUM STRINGS)
// 4 threads per string, process v1/v2/v3/v4 in parallel
// Use when: Strings are 100-10000 bytes, moderate batch size
// Parallelism: N/4 threads = N strings, with 4x internal parallelism
// ============================================================================
__global__ void xxhash32_warp_coop(
    const uint8_t* data,
    const uint32_t* offsets,
    const uint32_t* lengths,
    uint32_t* results,
    int num_strings,
    uint32_t seed
) {
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(cg::this_thread_block());

    int string_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 4;
    int acc_id = threadIdx.x % 4;

    if (string_idx >= num_strings) return;

    const uint8_t* str = data + offsets[string_idx];
    uint32_t len = lengths[string_idx];
    uint32_t acc, h32;

    // Initialize accumulators
    if (acc_id == 0)      acc = seed + PRIME1 + PRIME2;
    else if (acc_id == 1) acc = seed + PRIME2;
    else if (acc_id == 2) acc = seed;
    else                  acc = seed - PRIME1;

    // Process 16-byte chunks
    if (len >= 16) {
        int num_chunks = len / 16;
        const uint32_t* words = (const uint32_t*)str;

        for (int chunk = 0; chunk < num_chunks; chunk++) {
            uint32_t word = words[chunk * 4 + acc_id];
            acc = round(acc, word);
        }

        if (acc_id == 0) {
            uint32_t v1 = acc;
            uint32_t v2 = tile.shfl(acc, 1);
            uint32_t v3 = tile.shfl(acc, 2);
            uint32_t v4 = tile.shfl(acc, 3);
            h32 = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        }
    } else {
        if (acc_id == 0) h32 = seed + PRIME5;
    }

    // Tail processing (thread 0 only)
    if (acc_id == 0) {
        h32 += len;

        int tail_start = (len / 16) * 16;
        const uint8_t* p = str + tail_start;
        const uint8_t* end = str + len;

        while (p + 4 <= end) {
            h32 += (*((const uint32_t*)p)) * PRIME3;
            h32 = rotl(h32, 17) * PRIME4;
            p += 4;
        }

        while (p < end) {
            h32 += (*p) * PRIME5;
            h32 = rotl(h32, 11) * PRIME1;
            p++;
        }

        h32 ^= h32 >> 15;
        h32 *= PRIME2;
        h32 ^= h32 >> 13;
        h32 *= PRIME3;
        h32 ^= h32 >> 16;

        results[string_idx] = h32;
    }
}

// ============================================================================
// STRATEGY 3: ULTRA-WIDE CHUNK PARALLELISM (BEST FOR HUGE STRINGS)
// Process multiple chunks in parallel, then combine
// Use when: Single massive string (GB+), like streaming file hash
// Parallelism: Thousands of threads work on same string
// ============================================================================
__global__ void xxhash32_chunk_parallel(
    const uint8_t* data,
    uint32_t len,
    uint32_t* chunk_results,  // Intermediate results per chunk
    int num_chunks,
    uint32_t seed
) {
    int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int acc_id = chunk_idx % 4;
    int chunk_set = chunk_idx / 4;

    if (chunk_set >= num_chunks) return;

    uint32_t acc;
    if (acc_id == 0)      acc = seed + PRIME1 + PRIME2;
    else if (acc_id == 1) acc = seed + PRIME2;
    else if (acc_id == 2) acc = seed;
    else                  acc = seed - PRIME1;

    // Each group of 4 threads processes one 16-byte chunk
    int offset = chunk_set * 16;
    const uint32_t* words = (const uint32_t*)(data + offset);
    uint32_t word = words[acc_id];
    acc = round(acc, word);

    // Store intermediate result
    chunk_results[chunk_idx] = acc;
}

// Reduction kernel to combine chunk results
__global__ void xxhash32_chunk_reduce(
    const uint32_t* chunk_results,
    int num_chunks,
    const uint8_t* data,
    uint32_t len,
    uint32_t* final_result,
    uint32_t seed
) {
    if (blockIdx.x > 0 || threadIdx.x > 0) return;

    uint32_t h32;

    if (num_chunks > 0) {
        // Combine all v1/v2/v3/v4 across chunks
        uint32_t v1 = seed + PRIME1 + PRIME2;
        uint32_t v2 = seed + PRIME2;
        uint32_t v3 = seed;
        uint32_t v4 = seed - PRIME1;

        for (int c = 0; c < num_chunks; c++) {
            v1 = chunk_results[c * 4 + 0];
            v2 = chunk_results[c * 4 + 1];
            v3 = chunk_results[c * 4 + 2];
            v4 = chunk_results[c * 4 + 3];
        }

        h32 = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
    } else {
        h32 = seed + PRIME5;
    }

    h32 += len;

    // Process tail
    int tail_start = (num_chunks * 16);
    const uint8_t* p = data + tail_start;
    const uint8_t* end = data + len;

    while (p + 4 <= end) {
        h32 += (*((const uint32_t*)p)) * PRIME3;
        h32 = rotl(h32, 17) * PRIME4;
        p += 4;
    }

    while (p < end) {
        h32 += (*p) * PRIME5;
        h32 = rotl(h32, 11) * PRIME1;
        p++;
    }

    h32 ^= h32 >> 15;
    h32 *= PRIME2;
    h32 ^= h32 >> 13;
    h32 *= PRIME3;
    h32 ^= h32 >> 16;

    *final_result = h32;
}

// ============================================================================
// STRATEGY 4: PERSISTENT THREADS (MAXIMUM GPU UTILIZATION)
// Threads keep pulling work from queue until done
// Use when: Variable-length strings, want 100% GPU utilization
// ============================================================================
__global__ void xxhash32_persistent(
    const uint8_t* data,
    const uint32_t* offsets,
    const uint32_t* lengths,
    uint32_t* results,
    int num_strings,
    uint32_t seed,
    int* work_counter
) {
    while (true) {
        // Atomically grab next work item
        int idx = atomicAdd(work_counter, 1);
        if (idx >= num_strings) break;

        const uint8_t* str = data + offsets[idx];
        uint32_t len = lengths[idx];
        const uint8_t* p = str;
        const uint8_t* end = str + len;
        uint32_t h32;

        if (len >= 16) {
            const uint8_t* limit = end - 16;
            uint32_t v1 = seed + PRIME1 + PRIME2;
            uint32_t v2 = seed + PRIME2;
            uint32_t v3 = seed;
            uint32_t v4 = seed - PRIME1;

            do {
                v1 = round(v1, *((const uint32_t*)p)); p += 4;
                v2 = round(v2, *((const uint32_t*)p)); p += 4;
                v3 = round(v3, *((const uint32_t*)p)); p += 4;
                v4 = round(v4, *((const uint32_t*)p)); p += 4;
            } while (p <= limit);

            h32 = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        } else {
            h32 = seed + PRIME5;
        }

        h32 += len;

        while (p + 4 <= end) {
            h32 += (*((const uint32_t*)p)) * PRIME3;
            h32 = rotl(h32, 17) * PRIME4;
            p += 4;
        }

        while (p < end) {
            h32 += (*p) * PRIME5;
            h32 = rotl(h32, 11) * PRIME1;
            p++;
        }

        h32 ^= h32 >> 15;
        h32 *= PRIME2;
        h32 ^= h32 >> 13;
        h32 *= PRIME3;
        h32 ^= h32 >> 16;

        results[idx] = h32;
    }
}

int main() {
    const int NUM_STRINGS = 1000000;  // 1 MILLION strings
    const int AVG_LEN = 64;

    std::cout << "=== MAXIMUM PARALLELISM XXHASH32 BENCHMARK ===\n\n";

    // Generate test data
    std::vector<std::string> strings(NUM_STRINGS);
    std::vector<uint32_t> offsets(NUM_STRINGS);
    std::vector<uint32_t> lengths(NUM_STRINGS);
    std::vector<uint32_t> cpu_hashes(NUM_STRINGS);

    size_t total_bytes = 0;
    for (int i = 0; i < NUM_STRINGS; i++) {
        strings[i] = std::string(AVG_LEN, 'a' + (i % 26));
        offsets[i] = total_bytes;
        lengths[i] = strings[i].length();
        total_bytes += lengths[i];
        cpu_hashes[i] = xxhash32_cpu((const uint8_t*)strings[i].c_str(), lengths[i], SEED);
    }

    std::cout << "Dataset: " << NUM_STRINGS << " strings, "
              << total_bytes / 1024.0 / 1024.0 << " MB\n\n";

    // Allocate GPU memory
    uint8_t* d_data;
    uint32_t *d_offsets, *d_lengths, *d_results;
    cudaMalloc(&d_data, total_bytes);
    cudaMalloc(&d_offsets, NUM_STRINGS * sizeof(uint32_t));
    cudaMalloc(&d_lengths, NUM_STRINGS * sizeof(uint32_t));
    cudaMalloc(&d_results, NUM_STRINGS * sizeof(uint32_t));

    // Copy concatenated data
    uint8_t* h_data = new uint8_t[total_bytes];
    for (int i = 0; i < NUM_STRINGS; i++) {
        memcpy(h_data + offsets[i], strings[i].c_str(), lengths[i]);
    }
    cudaMemcpy(d_data, h_data, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), NUM_STRINGS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths.data(), NUM_STRINGS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Benchmark all strategies
    std::vector<uint32_t> results(NUM_STRINGS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // STRATEGY 1: Max Throughput (1 thread = 1 string)
    {
        int threads = 256;
        int blocks = (NUM_STRINGS + threads - 1) / threads;

        cudaEventRecord(start);
        xxhash32_max_throughput<<<blocks, threads>>>(d_data, d_offsets, d_lengths, d_results, NUM_STRINGS, SEED);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        cudaMemcpy(results.data(), d_results, NUM_STRINGS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        int errors = 0;
        for (int i = 0; i < NUM_STRINGS; i++) {
            if (results[i] != cpu_hashes[i]) errors++;
        }

        std::cout << "STRATEGY 1 - Max Throughput (1 thread/string):\n";
        std::cout << "  Time: " << ms << " ms\n";
        std::cout << "  Throughput: " << (total_bytes / 1e6) / (ms / 1000.0) << " MB/s\n";
        std::cout << "  Hashes/sec: " << (NUM_STRINGS / 1e6) / (ms / 1000.0) << " M/s\n";
        std::cout << "  Errors: " << errors << "\n\n";
    }

    // STRATEGY 2: Warp Cooperative (4 threads = 1 string)
    {
        int threads = 256;
        int blocks = ((NUM_STRINGS * 4) + threads - 1) / threads;

        cudaEventRecord(start);
        xxhash32_warp_coop<<<blocks, threads>>>(d_data, d_offsets, d_lengths, d_results, NUM_STRINGS, SEED);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        cudaMemcpy(results.data(), d_results, NUM_STRINGS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        int errors = 0;
        for (int i = 0; i < NUM_STRINGS; i++) {
            if (results[i] != cpu_hashes[i]) errors++;
        }

        std::cout << "STRATEGY 2 - Warp Cooperative (4 threads/string):\n";
        std::cout << "  Time: " << ms << " ms\n";
        std::cout << "  Throughput: " << (total_bytes / 1e6) / (ms / 1000.0) << " MB/s\n";
        std::cout << "  Hashes/sec: " << (NUM_STRINGS / 1e6) / (ms / 1000.0) << " M/s\n";
        std::cout << "  Errors: " << errors << "\n\n";
    }

    // STRATEGY 4: Persistent Threads
    {
        int* d_counter;
        cudaMalloc(&d_counter, sizeof(int));
        cudaMemset(d_counter, 0, sizeof(int));

        int threads = 256;
        int blocks = 108;  // Tune to GPU (e.g., 108 SMs on A100)

        cudaEventRecord(start);
        xxhash32_persistent<<<blocks, threads>>>(d_data, d_offsets, d_lengths, d_results, NUM_STRINGS, SEED, d_counter);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        cudaMemcpy(results.data(), d_results, NUM_STRINGS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        int errors = 0;
        for (int i = 0; i < NUM_STRINGS; i++) {
            if (results[i] != cpu_hashes[i]) errors++;
        }

        std::cout << "STRATEGY 4 - Persistent Threads:\n";
        std::cout << "  Time: " << ms << " ms\n";
        std::cout << "  Throughput: " << (total_bytes / 1e6) / (ms / 1000.0) << " MB/s\n";
        std::cout << "  Hashes/sec: " << (NUM_STRINGS / 1e6) / (ms / 1000.0) << " M/s\n";
        std::cout << "  Errors: " << errors << "\n\n";

        cudaFree(d_counter);
    }

    // Cleanup
    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_offsets);
    cudaFree(d_lengths);
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
    */