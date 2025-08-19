# GPU-Accelerated High-Performance Key-Value Store (GPU HashMap Engine)

## Problem Statement
Building a high-performance GPU-accelerated in-memory Key-Value Store using CUDA C++.  
The system will support:
- Batch Insertions & Lookups of key-value pairs in parallel.
- Aggregation Queries (SUM, MIN, MAX) across key sets.
- Prefix Sum Range Queries.
- Divergence-efficient filtering & Stream Compaction.
- Warp & Shared Memory optimizations for high throughput.

This will serve as a core GPU data-structure engine, designed for real-time analytics, caching layers, and infra-scale workloads.

## Project Goals
1. Implement GPU-optimized batch insert and lookup operations.
2. Design collision handling using shared memory and warp-level strategies.
3. Support aggregation queries using warp-level reductions.
4. Efficiently handle divergent queries using stream compaction.
5. Structure code in a clean, modular, GitHub-ready format.

---

## Repo Structure
src/ # Core CUDA code (kernels, device functions)
data/ # Example input datasets
benchmarks/ # Nsight Compute results, profiling logs
docs/ # Design documents, architecture decisions
README.md # Project overview


## Roadmap
- Phase 1: Batch Insertions & Lookups (hashmap.cu)
- Phase 2: Shared Memory & Warp Optimizations
- Phase 3: Aggregation Queries
- Phase 4: Stream Compaction & Divergence Handling
- Phase 5: API Wrapping & GitHub Finalization



This is my first CUDA C++ application based project and also my first project on git :) this project aims to achieve a highly efficient hashing engine using a GPU. The program structure is as mentioned-
## Repo Structure
src/           # Core CUDA code (kernels, device functions)
data/          # Example input datasets
benchmarks/    # Nsight Compute results, profiling logs
docs/          # Design documents, architecture decisions
README.md      # Project overview

let us drive straight into the process-
At this point i somewhat feel that we need to narrow down on maximizing 3 metrics to make the code as efficient as possible- 
for my main file ( src/hashmap.cu). these 3 requirements are as follows - 
1)A hashing function-Hashing function is responsible(performs the work of reducing a large input element into an efficient "hash" ultimately to calculate the index to store this value in , it is a general algorithm that will generate as less collision as possible and be efficiently able to arrange a large dataset into hash table with minimal overlap of elements , hence we have a final aim of lets say "GENERATING IDs AS UNIQUE AS POSSIBLE". this function should be highly efficient and optimized for parallel computation, so our main task is to select a function that is most suitable to our usecase on a GPU

2)index calculation/hash table - after a efficient hashing function we must be able to index this hash effectively such that it will be easier to pull later on as well as be easier to store . as efficient of a data structure the table has for parallel draw , more efficient our kernel will be as indexing will be made simpler.

3)collision handling - finally the system we use to handle collisions is very important as this will determine how easy it will be to retrive an item later on 

4)dynamic resizing - we dont want to unnecessarily allocate a very big chunk of storage to it , it should dynamically increase as the number of key value pairs increases


Now for actual benchmarking purposes lets start with the simplest approach(the below will be represented in roman numerals starting from the most naive i) to the the most efficient n)-

Version1 begins:
Hash Table Design 

  Hash Function:

    Simple modulus: value % table_size

  Hash Table Structure:

    1D array

  Collision Handling:

    Linear probing

Problems Faced in this Approach

    Since multiple elements can have the same modulus (where n is the table size), simultaneous read/write operations from different threads may cause race conditions. This can result in overwriting or corruption of data in the hash table.

    Any solution to handle concurrent insertions reliably will likely involve trade-offs in either additional memory usage, increased compute, or higher time complexity.

    Our current hardware constraints are a mix of RTX 4070 and RTX 2070 SUPER GPUs, so solutions must respect the capabilities and limits of these GPUs.

Potential Ideas to Address These Issues

    Divide threads into groups that work on different hash buckets or even separate hash tables entirely. Parameters like group size or division strategy would need to be tuned.

    Use atomicCAS (compare-and-swap) operations to guarantee exclusive access to hash table slots.

    Explore cooperative groups, a CUDA programming model that allows fine-grained synchronization and cooperation inside thread blocks and across warps.

    Employ warp-level scheduling and communication to improve efficiency and reduce contention.

Known Problems in the Current Implementation

    1) Currently, you are initializing h_keys[tid] = -1 inside the kernel, which is inefficient.
    Improvement: Use cudaMemset to initialize the entire table to -1 before launching kernels.
    This leverages the GPU's DMA engines, which are dedicated hardware components optimized for fast bulk memory operations, offloading this work from the SMs (Streaming Multiprocessors).

    2) The use of atomicCAS can lead to serialization and slowdowns when many threads contend for the same hash bucket during collisions(limitations due to SIMT).

    3) Linear probing suffers performance degradation when there are many collisions because threads will have to probe many slots sequentially, increasing latency.

    4) The modulo operator (%) as a hash function tends to create many collisions, particularly when the data has non-uniform distributions or patterns that cluster in certain buckets.


>now in the version 2 - lest work on the the above problem 2

changes made - 
1) for loop used in place of do while to instantly exit the program when ptc ==-1 also code size is reduced
2) lookup kernel written based on same insertion logic 
3) addition of main
4) cudaMemset is being used inplace of in kernel initialization and wasting SM's utilization were using DMA's instead
5) 









///edit this out -
///1) were considering 2 ways to do this _shfl_sync and _ballot_sync
weve studied ballot_sync until now - learning 
>) limitation - even tho were able to save unnecessarily wasted compute power using this method were not able tyo redirect the threads that are freed in a warp to do other work .
>) how to perform it : 






















"unsigned int active = __ballot_sync(0xffffffff, stillSearching);
if (!(active & (1 << lane_id))) break;
"
next topic :
shl_sync
also check if therea are any other methods and then write a lookup proegram ( thats phase 2)
and _fss