# GPU-Accelerated High-Performance Key-Value Store (GPU HashMap Engine)
//consider a future possibility of dual capability of or benchmark cpu vs gpu by running GXhash on cpu and custom gpu or cucollection on gpu
## Project Overview

This is my first CUDA C++ application project and also my first project on GitHub! 🚀 

This project aims to achieve a highly efficient hashing engine using GPU parallelization. The system is designed as a core GPU data-structure engine for real-time analytics, caching layers, and infrastructure-scale workloads.

## Problem Statement

Building a high-performance GPU-accelerated in-memory Key-Value Store using CUDA C++ that supports:
- **Batch Insertions & Lookups** of key-value pairs in parallel
- **Aggregation Queries** (SUM, MIN, MAX) across key sets
- **Prefix Sum Range Queries**
- **Divergence-efficient filtering & Stream Compaction**
- **Warp & Shared Memory optimizations** for high throughput

## Repository Structure

```
src/           # Core CUDA code (kernels, device functions)
data/          # Example input datasets  
benchmarks/    # Nsight Compute results, profiling logs
docs/          # Design documents, architecture decisions
README.md      # Project overview (this file)
```

## Build Instructions

```bash
nvcc hashmap.cu -o hm
./hm
```


## Hardware Constraints

Current development is optimized for:
- **RTX 4070** and **RTX 2070 SUPER** GPUs
- Solutions must respect the capabilities and limits of these GPUs



## Implementation Journey

To maximize efficiency, I'm focusing on optimizing **4 core metrics**:

1. **Hashing Function** - Generate IDs as unique as possible with minimal collisions
2. **Index Calculation/Hash Table** - Efficient parallel data structure for storage and retrieval
3. **Collision Handling** - System to resolve hash conflicts efficiently
4. **Dynamic Resizing** - Automatically scale storage as key-value pairs increase

### Evolution Through Versions

The project follows an iterative approach from naive (Version I) to highly optimized (Version N).


## Version 1: Baseline Implementation

### Design Choices

**Hash Function:**
- Simple modulus: `value % table_size`

**Hash Table Structure:**  
- 1D array

**Collision Handling:**
- Linear probing

### Problems Identified

1. **Race Conditions**: Multiple threads accessing same hash buckets simultaneously cause data corruption
2. **SIMT Limitations**: Atomic operations lead to serialization and performance degradation  
3. **Poor Hash Distribution**: Modulo operator creates clustering with non-uniform data
4. **Inefficient Initialization**: Setting `h_keys[tid] = -1` inside kernels wastes SM resources

### Potential Solutions Explored

- **Thread Grouping**: Divide threads into groups working on separate hash buckets
- **Atomic Operations**: Use `atomicCAS` for exclusive access to hash table slots
- **Cooperative Groups**: Fine-grained synchronization within thread blocks and warps
- **Warp-level Communication**: Improve efficiency and reduce contention

---

## Version 2: Complete Working Implementation ✅

### Architecture Overview

**Core Data Structure:**
```cpp
int h[] = {10, 20, 30, 40, 50, 60, 70, 80};  // Hash table values
int n = 8;                                   // Fixed hash table size
```

**Memory Layout:**
- **Hash Keys Array** (`g_k`): Device array storing the actual hash table
- **Hash Values Array** (`g_v`): Device array with input values to be hashed
- **Search Array** (`g_temp`): Device array with user input search queries
- **Location Results** (`locn`): Device array storing found indices

### Key Improvements Implemented

#### 1. **Optimized Memory Initialization**
```cpp
cudaMemset(g_k, 0xFF, size);     // Hash table initialized to -1
cudaMemset(locn, 0xFF, size1);   // Results initialized to -1 (not found)
```
**Advantage**: Leverages GPU's DMA engines instead of wasting SM cycles

**Key Features:**
- **Linear Probing**: `(base + i) % n` for collision resolution
- **Atomic Operations**: `atomicCAS` prevents race conditions
- **Early Exit**: Return immediately on successful insertion
- **Not Found Handling**: Relies on pre-initialized -1 values
- **Dynamic sizing**: Vector grows as user adds elements
- **Error handling**: Catches invalid input gracefully
- **Flexible termination**: "eol" to end input



### CUDA Memory Operations Deep Dive

**cudaMemset Analysis:**
```cpp
cudaMemset(g_k, 0xFF, size);  // Sets each byte to 0xFF
```
- **Byte-wise limitation**: Can only store values 0-255
- **For -1 initialization**: `0xFF = 11111111` in binary = -1 for signed integers
- **DMA advantage**: Uses dedicated memory controllers, not SMs


### Debugging Journey

#### Error 1: SIGSEGV (Address Boundary Error)
**Cause**: 
int *output;  // Uninitialized pointer
cudaMemcpy(output, locn, size1, cudaMemcpyDeviceToHost); 

**Solution**: 
int *output = new int[i];  // Proper allocation


#### Error 2: Wrong Memory Transfer Pattern
**Cause**: Using `&temp` instead of `temp.data()` for vector
**Solution**: Understanding vector object vs vector data distinction

### Performance Test Results

**Test Input:**
20, 70, 90, 0


**Expected Hash Table State After Insertion:**
Index: 0  1  2  3  4  5  6  7
Value: 80 - 10 30 20 50 60 70
       ^              ^     ^
      (80%8=0)    (20%8=4) (70%8=6→7)


**Actual Output:**
20 was found at 4
70 was found at 7  
The element 90 is not a part of the hash table
The element 0 is not a part of the hash table


**Analysis**: 

### Debugging Tools Integration

**compute-sanitizer Integration:**
/opt/cuda/bin/compute-sanitizer ./hm
========= COMPUTE-SANITIZER
GPU HashMap Engine - CUDA Kernel Skeleton Initialized
...
========= ERROR SUMMARY: 0 errors


**What compute-sanitizer detects:**
1. **Memory access errors** → Out-of-bounds access
2. **Race conditions** → Simultaneous memory modifications
3. **API errors** → CUDA runtime/driver misuse

## Current Status: Version 2 Complete 

### Working Features
- **Batch Insertion**: Parallel hash table population using linear probing
- **Collision Resolution**: Atomic `atomicCAS` operations prevent race conditions  
- **Parallel Lookup**: Multi-threaded search with proper bounds checking
- **Dynamic Input**: User can search for any number of elements
- **Memory Efficiency**: DMA-based initialization, proper cleanup
- **Error Handling**: Robust input validation and memory management
- **Not Found Detection**: Proper handling of missing elements

### Performance Characteristics
- **Hash Function**: `value % table_size` (simple but collision-prone)
- **Collision Resolution**: Linear probing with atomic synchronization
- **Memory Access**: Coalesced for optimal bandwidth utilization  
- **Thread Efficiency**: Minimal warp divergence in insertion/lookup loops
- **Space Complexity**: O(n) hash table + O(search_count) auxiliary arrays

### Current Limitations
1. **Fixed Hash Table Size**: Currently hardcoded to 8 elements
2. **Simple Hash Function**: Modulo operation creates clustering  
3. **Linear Probing Overhead**: Performance degrades with high collision rates
4. **No Dynamic Resizing**: Cannot grow hash table at runtime

## Next Phase: Version 3 - Warp Optimizations 🔄

### Research Topics

#### 1. **Warp Ballot Synchronization**
```cpp
unsigned int active = __ballot_sync(0xffffffff, stillSearching);
if (!(active & (1 << lane_id))) break;
```
**Current Understanding:**
- **Advantage**: Saves compute power by early-exiting threads that found their elements
- **Limitation**: Freed threads in a warp cannot be redirected to other work
- **Use Case**: Efficient divergent search termination

#### 2. **Shuffle Synchronization** 
```cpp
int peer_value = __shfl_sync(0xffffffff, value, source_lane);
```
**Potential Applications:**
- **Intra-warp communication**: Share hash values without global memory
- **Cooperative collision resolution**: Threads can help each other find empty slots
- **Load balancing**: Redistribute work within warps dynamically

#### 3. **Cooperative Groups**
- **Fine-grained synchronization**: Beyond block-level barriers
- **Dynamic thread regrouping**: Adapt group size based on workload
- **Cross-warp cooperation**: Coordinate between warps for complex operations

IMPLEMENTATION PLAN:
1) a new hashing algorithm which is far more efficient 
2) trying to use shared memory 
3) dynamic resixzing of hash table 
4) check possibilities of splitting the hash table into parts

envisioning
>when the user enters a new hash element into the table via the insert kernel, an efficient alg shud create a hash for it that must be unique,
>this hash must be converting into a location as unique as possible.insert here.when looking up elements in the table or inserting, warps wll be scheduled ,ideally if we can achive perfect conditions the 32 threads in a warp finish simultaneously with no waiting for a single thread to find a new spot due to collision(which is possible only with all unique mapping)

>TRAin of THOUGHT
1)why note skip the conversion step of the hashed value to memory location by directly calculating a hash that will itself be small (not 64 bit something that will be within array size)
issues identified - everytime the hash table size chnages (dynamic resizing based on number of elements)
inefficieny- less collisions are experrienced due to a large 64 bit hash(the number of hash combinations are far more than array size)
2)maybe try multiple different hash buckets in a hash which will greatly reduce search search complexity or maybe 2d storage so we put in similar elements in a single location but increasing its depth from 1 to n colliding elements




PROBLEMS TO ANSWER
1) What is the sweet spot for division into number of buckets like how how many parts shud i divide it into(RESEARCH)
2) (we r using 64 bit hashes) selecting a hash that strikes a balance between calculation time collisions required(that is all the performance shud not be sacrificed just to give uniquiness)
3) 





WHAT WE WANT TO ACHIVE AT THE END:

some things identified 
unlike cpus its very expensive to linearly probe on a gpu cuz the entire warp ahs to wait owing to SIMT

What the “ideal” GPU Hashmap might look like

Hash function: Multiply-shift, cheap & well-mixed.

Memory: Keys[] + Values[] arrays in global memory, aligned & padded.

Probe strategy: Warp-cooperative linear probing with block size = 32 slots (fits warp).

Atomics: Warp-wide atomicCAS batches.

Shared memory caching for hot regions.

Overflow stash for extreme collisions.

Bulk operations: Designed to insert/look up batches of keys at once (not single-key API).




1) for now we have decided on 3 diff strategies - namely one byte one string (CPU type comp), 2nd one is 4 bytes one string , third is is for very long strings  
  alternate thought - tweak this idea and launch only 2 threads (or add 2 kernel launch options in each one for greater than 16 byyte and for lesser this would avoid excessive branching and wasteful checks ?)


  1. Hash Function (you did this ✔)
2. Hash Table Layout (you are starting this) - cuckoo hashing
3. Collision Strategy - cuckoo hashing
4. Insertion Kernel 
5. Lookup Kernel
6. Deletion Strategy (tombstones etc.)
7. Memory Management + load factors
8. Parallel correctness (no races)
9. Performance optimizations



// STRATEGY 2
// 1 block 4096 bytes can be handled considerign att 1024 threads in a lbock are launched(we have 4 bytes being dealt with per thread)
// total bytes dealt with are 4096*n where n is no of blocks
// we have to employ grid stride loops
/*__global__ void xxhash(uint8_t *bytes, uint32_t *offset, uint32_t *cr)
{
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(cg::this_thread_block());
    const uint32_t *words = reinterpret_cast<const uint32_t *>(bytes);

    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x; // llest assume the launch odf 1024 threads per block
    // uint32_t warp_id = tid / warpSize;
    // uint32_t lane_id = tid % warpSize;
    //uint32_t lid = tid % 4;
    uint32_t wid = tid / 4;
    // NOTE:while shared memory may be fast it still is slower than the L1 cache and since data is consecutive it will most definitely hit L1,
    // i.e. we can conclude that shared mem only proveides a real advantage when teh same daata is used a high number of times

    if (wid >= 630)
        return; // hard coded boundary check defined by us to deal with no of strings = to no of offsets

    uint32_t start = offset[wid];
    uint32_t len = offset[wid + 1] - start;
    uint32_t posn = (start / 4) + tile.thread_rank();

    // int stride = blockDim.x * blockIdx.x;
    uint32_t mask = (len >= 16);
    uint32_t v[4];
    v[tile.thread_rank()] = g[tile.thread_rank()]; // initializing all 4 accumulators with their respective constants
    uint32_t res = (1 - mask) * (SEED + g[6]);
    uint32_t fin;
    uint32_t i = 0;
    // while(tile.all(i<len))
    for (; i + 16 <= len; i += 16)                                         // 0,16,32....so on
    {                                                                      // we are processign in grps of 16 so this is a check to make sure its within sentence range
        v[tile.thread_rank()] = round(v[tile.thread_rank()], words[posn]); /// given that a sentence(string) has more than 16 bytes this part will run and load in grps of 4
        posn += 4;
    }


    /*Step 3: Merge accumulators
    if length >= 16:
        acc = rotate_left(v1, 1) +
              rotate_left(v2, 7) +
              rotate_left(v3, 12) +
              rotate_left(v4, 18)
    */
    if (tile.any(mask) && tile.thread_rank() == 0)
    {
        res = inst(v[0], 1);
        res += tile.shfl(inst(v[1], 7), 1);
        res += tile.shfl(inst(v[2], 12), 2);
        res += tile.shfl(inst(v[3], 18), 3);
    }
    // now we can safely say we have all tiles having a particular value of res in their first thread which will hold
    // either the PRIME5*SEED value (if its length is less than 16 bytes long) or the accumulated value we have until step 3

    // this will read consecutive 4 bytes together that is its been converted from a char type array to a int type one
    // include all threads upto the closest multiple to 4 ( here all tid upto 16 )
    /*if(lane_id<8)
    {   v1=SEED + g[0] + g[1];
        v1+=words[(lane_id)*4]*g[1];
        v1+=inst(v1,13);
        v1*=g[0];
    }
    else if(lane_id<16)
    {
        v2=SEED + g[1];
        v2+=words[((lane_id-8)*4)+1]*g[1];
        v2+=inst(v2,13);
        v2*=g[0];
    }
    else if(lane_id<24)
    {
        v3=SEED;
        v3+=words[((lane_id-16)*4)+2]*g[1];
        v3+=inst(v3,13);
        v3*=g[0];
    }
    else if (lane_id<32)
    {
        v4=SEED-g[0];
        v4+=words[((lane_id-8)*4)+3]*g[1];
        v4+=inst(v4,13);
        v4*=g[0];
    }*/
    /*int n=length/16;
    if(length>16)
    {if(wid==0)
    {v[lid]=g[lid]; //were assuming seed=0 in the constants only
    for(int i=0;i<n;i+=4)
     {
     v[lid]+=words[lid+i]*g[1];    //{(0x9E3779B1+0x85EBCA77),0x85EBCA77,0,-0x9E3779B1,0xC2B2AE3D,0x27D4EB2F,0x165667B1};
     v[lid]+=inst(v[lid],13);
     v[lid]*=g[0];
     }
    }
    }
    if(lane_id==0)
    {res = inst(v[0],1);
     res+= __shfl_down_sync(0xffffffff,inst(v[1],7),1);
     res+= __shfl_down_sync(0Xffffffff,inst(v[2],12),2);
     res+= __shfl_down_sync(0Xffffffff,inst(v[3],18),3);
    }
    //now v1 of all first 8 threads stores the final accumulated values of all 8 words
    */

    /*Step 4: Process remaining bytes (<16)
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
        */
    if (tile.thread_rank() == 0)
    {
        res += len;
        uint32_t processed = (len / 16) * 16; // Bytes already processed
        uint32_t k1;
        uint32_t i = processed;
        for (i; (i + 4) <= len; i += 4)
        {
            k1 = words[(start + i)/4]; // will load consecutive bytes into con threads
            k1 *= g[4];
            k1 = inst(k1, 17);
            k1 *= g[5];
            res ^= k1;
            res = inst(res, 17) * -g[3] + g[5];
        }
        // i = (4 * i) - 4;
        while (i < len)
        {
            /*for(int offset= (length/4)*4;offset>0;offset>>=1)
            {
            k1 ^= __shfl_down_sync(0xffffffff,k1,offset);
            }*/
            k1 = bytes[start + i] * g[6];
            res ^= k1;
            res = inst(res, 11) * -g[3];
            i++;
        }

        /*
        Step 5: Final avalanche (mixing)
        acc = acc ^ (acc >> 15)
        acc = acc * PRIME2
        acc = acc ^ (acc >> 13)
        acc = acc * PRIME3
        acc = acc ^ (acc >> 16)
        */
        res ^= res >> 15;
        res *= g[1];
        res ^= res >> 13;
        res *= g[4];
        res ^= res >> 16;
        cr[wid] = res;
    }*/
}

///
PROBLEMS KNOWN
1) 75% thread underultilixzation in hashmap.cu
2) methods to append the new offsets into the MASTEROFFSET ARRAY AND MASTERBYTE ARRAY.vector is a host size  function 